"""Benchmark SuffixGPU drafter vs arctic_inference suffix-cpu.

Two parts:
1. accuracy: simulated spec-decode over shared synthetic repetitive
   streams (two waves; wave 2 reuses the same phrase distributions so
   cross-request/global memory pays off). The "target model" replays a
   fixed reference stream, so accepted length = longest prefix of the
   draft matching the reference continuation (+1 bonus token). Reports
   tokens/step, acceptance rate, and per-step propose latency.
2. latency: standalone propose() latency sweep over batch size /
   context length for the GPU drafter (local-only and local+global),
   eager and CUDA-graph replay.

Run inside the vllm venv:
  LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.580.105.08 \
  ../vllm/.venv/bin/python benchmarks/bench_vs_cpu.py --mode both
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suffix_gpu.proposer import SuffixGPUDrafter  # noqa: E402

VOCAB = 32000


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def make_libs(rng: np.random.Generator, n_global: int = 24,
              n_priv: int = 6):
    """Zipf-weighted phrase libraries (RAG / code-edit style reuse)."""
    global_lib = [rng.integers(0, VOCAB, size=rng.integers(8, 33))
                  for _ in range(n_global)]
    gw = 1.0 / np.arange(1, n_global + 1)
    gw /= gw.sum()
    return global_lib, gw, n_priv


def make_streams(
    rng: np.random.Generator,
    libs,
    num_requests: int,
    prompt_len: int,
    gen_len: int,
    p_global: float,
    p_private: float,
) -> list[np.ndarray]:
    global_lib, gw, n_priv = libs
    n_global = len(global_lib)
    streams = []
    total = prompt_len + gen_len
    for _ in range(num_requests):
        private_lib = [rng.integers(0, VOCAB, size=rng.integers(8, 33))
                       for _ in range(n_priv)]
        pw = 1.0 / np.arange(1, n_priv + 1)
        pw /= pw.sum()
        toks: list[int] = []
        while len(toks) < total:
            u = rng.random()
            if u < p_global:
                seg = global_lib[rng.choice(n_global, p=gw)]
            elif u < p_global + p_private:
                seg = private_lib[rng.choice(n_priv, p=pw)]
            else:
                seg = rng.integers(0, VOCAB, size=rng.integers(4, 13))
            toks.extend(int(t) for t in seg)
        streams.append(np.asarray(toks[:total], dtype=np.int64))
    return streams


# ----------------------------------------------------------------------
# accuracy simulation
# ----------------------------------------------------------------------
@dataclass
class SimResult:
    name: str
    steps: int = 0
    req_steps: int = 0
    committed: int = 0
    drafted: int = 0
    accepted: int = 0
    step_ms: list[float] = field(default_factory=list)

    def report(self) -> str:
        tps = self.committed / max(1, self.req_steps)
        acc = self.accepted / max(1, self.drafted)
        lat = statistics.mean(self.step_ms) if self.step_ms else 0.0
        p50 = statistics.median(self.step_ms) if self.step_ms else 0.0
        return (f"{self.name:12s} steps={self.steps:5d} "
                f"tokens/step={tps:5.2f} "
                f"drafted/step={self.drafted / max(1, self.req_steps):5.2f} "
                f"accept_rate={acc:5.1%} "
                f"propose_ms mean={lat:7.2f} p50={p50:7.2f}")


def accept_len(draft: list[int], ref: np.ndarray) -> int:
    n = 0
    for i, t in enumerate(draft):
        if i < len(ref) and t == ref[i]:
            n += 1
        else:
            break
    return n


def sim_gpu(
    drafter: SuffixGPUDrafter,
    streams: list[np.ndarray],
    prompt_len: int,
    device: torch.device,
    name: str,
    ingest_chunk: int = 64,
    use_graph: bool = False,
) -> SimResult:
    b = len(streams)
    total = max(len(s) for s in streams)
    k = drafter.k
    s_buf = total + k + 8
    buf = torch.zeros((b, s_buf), dtype=torch.int32, device=device)
    for i, st in enumerate(streams):
        buf[i, :prompt_len] = torch.from_numpy(st[:prompt_len]).to(
            device, torch.int32)
    lens = np.full(b, prompt_len, dtype=np.int64)
    res = SimResult(name)

    # Prefill's sampled token primes the first decode step.
    pending: list[list[int]] = [[int(st[prompt_len])] for st in streams]
    finished = np.zeros(b, dtype=bool)
    flushed = np.zeros(b, dtype=bool)
    num_tok_t = torch.from_numpy(lens.astype(np.int32)).to(device)
    sampled_buf = torch.full((b, k + 1), -1, dtype=torch.int32,
                             device=device)

    graph = None
    if use_graph and device.type == "cuda":
        # Warm up (Triton JIT, allocator) on scratch state, then
        # capture the full update+propose chain; the graph feeds the
        # new counts back into its own input buffer.
        wb, wn, ws = buf.clone(), num_tok_t.clone(), sampled_buf.clone()
        for _ in range(3):
            drafter.propose_with_update(wn, wb, ws)
        torch.cuda.synchronize(device)
        del wb, wn, ws
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            draft_t, nv_t, nc_t = drafter.propose_with_update(
                num_tok_t, buf, sampled_buf)
            num_tok_t.copy_(nc_t)

    while not (finished & (np.array([len(p) for p in pending]) == 0)).all():
        sampled_np = np.full((b, k + 1), -1, dtype=np.int32)
        for i in range(b):
            if pending[i]:
                sampled_np[i, :len(pending[i])] = pending[i]
        sampled_buf.copy_(torch.from_numpy(sampled_np))
        drafter.poll()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if graph is not None:
            graph.replay()
        else:
            draft_t, nv_t, nc_t = drafter.propose_with_update(
                num_tok_t, buf, sampled_buf)
            num_tok_t.copy_(nc_t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        res.step_ms.append((time.perf_counter() - t0) * 1e3)

        draft = draft_t.cpu().numpy()
        nv = nv_t.cpu().numpy()
        res.steps += 1
        for i in range(b):
            committed_now = len(pending[i])
            if committed_now == 0:
                continue
            res.req_steps += 1
            lens[i] += committed_now
            res.committed += committed_now
            pending[i] = []
            ref = streams[i]
            pos = int(lens[i])
            remain = len(ref) - pos
            if remain <= 0:
                finished[i] = True
                continue
            d = [int(t) for t in draft[i, :nv[i]]]
            a = accept_len(d, ref[pos:])
            res.drafted += len(d)
            res.accepted += min(a, max(0, remain - 1))
            commit = min(a + 1, remain)
            pending[i] = [int(t) for t in ref[pos:pos + commit]]
        # In-flight ingestion (host-side, off the propose path).
        act = [i for i in range(b) if not flushed[i]]
        keys = [f"{name}-{i}" for i in act]
        rows = [buf[i, prompt_len:int(lens[i])] for i in act]
        lengths = [int(lens[i]) - prompt_len for i in act]
        fin = [i for i in act if finished[i] and not pending[i]]
        if act:
            drafter.ingest_active(keys, rows, lengths, chunk=ingest_chunk)
        if fin:
            drafter.ingest_active(
                [f"{name}-{i}" for i in fin],
                [buf[i, prompt_len:int(lens[i])] for i in fin],
                [int(lens[i]) - prompt_len for i in fin],
                final=True, chunk=ingest_chunk)
            for i in fin:
                flushed[i] = True
    return res


def sim_cpu(
    cache,
    streams: list[np.ndarray],
    prompt_len: int,
    k: int,
    max_tree_depth: int,
    max_spec_factor: float,
    min_token_prob: float,
    wave: int,
    name: str,
) -> SimResult:
    b = len(streams)
    lens = np.full(b, prompt_len, dtype=np.int64)
    finished = np.zeros(b, dtype=bool)
    started = np.zeros(b, dtype=bool)
    pending: list[list[int]] = [[] for _ in range(b)]
    res = SimResult(name)

    while not finished.all():
        t0 = time.perf_counter()
        drafts: list[list[int]] = []
        for i in range(b):
            if finished[i]:
                drafts.append([])
                continue
            rid = f"w{wave}r{i}"
            if not started[i]:
                cache.start_request(
                    rid, streams[i][:prompt_len].astype(np.int32))
                started[i] = True
            if pending[i]:
                cache.add_active_response(rid, pending[i])
                pending[i] = []
            pos = int(lens[i])
            start = max(0, pos - max_tree_depth)
            pattern = streams[i][start:pos].astype(np.int32)
            d = cache.speculate(
                rid, pattern, max_spec_tokens=k,
                max_spec_factor=max_spec_factor,
                min_token_prob=min_token_prob)
            drafts.append(list(d.token_ids))
        res.step_ms.append((time.perf_counter() - t0) * 1e3)

        res.steps += 1
        for i in range(b):
            if finished[i]:
                continue
            res.req_steps += 1
            ref = streams[i]
            pos = int(lens[i])
            remain = len(ref) - pos
            d = drafts[i]
            a = accept_len(d, ref[pos:])
            commit = min(a + 1, remain)
            res.drafted += len(d)
            res.accepted += min(a, max(0, remain - 1))
            pending[i].extend(int(t) for t in ref[pos:pos + commit])
            lens[i] += commit
            res.committed += commit
            if lens[i] >= len(ref):
                finished[i] = True
                cache.add_active_response(f"w{wave}r{i}", pending[i])
                pending[i] = []
                cache.stop_request(f"w{wave}r{i}")
    return res


def run_accuracy(args, device: torch.device) -> None:
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    for label, pg, pp in (("high-repetition", 0.35, 0.35),
                          ("low-repetition ", 0.15, 0.15)):
        rng = np.random.default_rng(args.seed)
        libs = make_libs(rng)
        waves = [make_streams(rng, libs, args.batch, args.prompt_len,
                              args.gen_len, pg, pp) for _ in range(2)]
        print(f"\n== accuracy [{label}] B={args.batch} "
              f"prompt={args.prompt_len} gen={args.gen_len} "
              f"k={args.k} depth={args.depth} "
              f"factor={args.spec_factor} minp={args.min_token_prob}")

        cache = SuffixDecodingCache(max_tree_depth=args.depth,
                                    max_cached_requests=1000)
        for w, streams in enumerate(waves):
            r = sim_cpu(cache, streams, args.prompt_len, args.k,
                        args.depth, args.spec_factor, args.min_token_prob,
                        w, f"cpu-wave{w}")
            print("  " + r.report())

        variants = [(True, False, "gpu-eager"), (True, True, "gpu-graph"),
                    (False, False, "gpu-local-eager"),
                    (False, True, "gpu-local-graph")]
        for enable_global, use_graph, tag in variants:
            if use_graph and device.type != "cuda":
                continue
            drafter = SuffixGPUDrafter(
                k=args.k, device=device, max_pattern_len=args.depth,
                min_match_len=1,
                enable_global=enable_global,
                global_capacity=1 << 20, delta_capacity=1 << 15,
                max_spec_factor=args.spec_factor,
                min_token_prob=args.min_token_prob,
                rebuild_stream=torch.cuda.Stream(device)
                if device.type == "cuda" else None,
            )
            for w, streams in enumerate(waves):
                r = sim_gpu(drafter, streams, args.prompt_len, device,
                            f"{tag}-wave{w}", use_graph=use_graph)
                print("  " + r.report())
            del drafter


# ----------------------------------------------------------------------
# latency sweep
# ----------------------------------------------------------------------
def sweep_gpu_latency(device: torch.device, k: int, max_pattern_len: int,
                      batches: list[int], seqs: list[int]) -> None:
    rng = np.random.default_rng(0)
    for enable_global in (False, True):
        tag = "local+global" if enable_global else "local-only  "
        for s in seqs:
            for b in batches:
                drafter = SuffixGPUDrafter(
                    k=k, device=device, max_pattern_len=max_pattern_len,
                    enable_global=enable_global,
                    global_capacity=1 << 20, delta_capacity=1 << 15,
                    max_spec_factor=2.0, min_token_prob=0.1,
                    rebuild_stream=torch.cuda.Stream(device)
                    if device.type == "cuda" else None,
                )
                if enable_global:
                    docs = [torch.from_numpy(
                        rng.integers(0, VOCAB, size=4096)).to(
                            device, torch.int32) for _ in range(96)]
                    drafter.harvest_rows(docs, [4096] * len(docs))
                    for _ in range(500):
                        drafter.poll()
                        if drafter.global_index._rebuild_event is None:
                            break
                        time.sleep(0.01)
                buf = torch.from_numpy(
                    rng.integers(0, VOCAB, size=(b, s))).to(
                        device, torch.int32)
                lens_np = rng.integers(s // 2, s, size=b).astype(np.int32)
                num_tok = torch.from_numpy(lens_np).to(device)
                for _ in range(3):
                    drafter.propose(num_tok, buf)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                iters = 10
                t0 = time.perf_counter()
                for _ in range(iters):
                    drafter.propose(num_tok, buf)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ms = (time.perf_counter() - t0) * 1e3 / iters
                graph_ms = float("nan")
                if device.type == "cuda":
                    try:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            drafter.propose(num_tok, buf)
                        for _ in range(3):
                            g.replay()
                        torch.cuda.synchronize(device)
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            g.replay()
                        torch.cuda.synchronize(device)
                        graph_ms = (time.perf_counter() - t0) * 1e3 / iters
                    except Exception as e:
                        print(f"    (graph capture failed: {e})")
                print(f"  gpu {tag} B={b:4d} S={s:6d}: eager {ms:8.2f} "
                      f"ms/step | graph {graph_ms:7.2f} ms/step")
                del drafter, buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["accuracy", "latency", "both"],
                    default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--depth", type=int, default=24,
                    help="max_tree_depth / max_pattern_len")
    ap.add_argument("--spec-factor", type=float, default=2.0)
    ap.add_argument("--min-token-prob", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"torch {torch.__version__} device={device} "
          f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")

    if args.mode in ("accuracy", "both"):
        run_accuracy(args, device)

    if args.mode in ("latency", "both"):
        print(f"\n== latency sweep k={args.k} depth={args.depth}")
        sweep_gpu_latency(device, args.k, args.depth,
                          batches=[1, 8, 32, 128], seqs=[4096, 16384])


if __name__ == "__main__":
    main()
