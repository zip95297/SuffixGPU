"""Spec-Bench replay benchmark: SuffixGPU drafter vs arctic suffix-cpu.

Drafter-level only -- no LLM engine. The "target model" replays each
question's reference answer, so the token streams are fully determined
by (dataset, tokenizer) and results are reproducible bit-for-bit.

Data: Spec-Bench question.jsonl (hemingkx/Spec-Bench, md5
0c39ae23e6f213549c66d6d691c99034). Categories with full references
(translation / summarization / math_reasoning, 80 questions each) are
replayed: prompt = turns[0], generation = reference[0].

Tokenizer: any HF tokenizer path/name (--tokenizer); token ids are
produced with `encode(text, add_special_tokens=False)`.

Modes:
  replay     simulated spec-decode (accept = longest draft prefix that
             matches the reference continuation, +1 bonus token, greedy
             target semantics). Two waves per category; wave 2 re-sends
             the same questions so cross-request/global memory pays off.
             Reports tokens/step, acceptance rate, propose latency for
             arctic-cpu and SuffixGPU (eager / CUDA graph / local-only).
  agreement  teacher-forced lockstep: both drafters see the identical
             committed history every step; their drafts are compared
             token-by-token (consistency check of the draft function
             itself). Residual divergence is majority-vote tie-breaking,
             which is unspecified across implementations.
  scale      warm-state batch scaling: both drafters warmed with wave-0
             responses, wave-1 requests parked mid-generation, then the
             per-step draft cost for the whole batch is timed at
             B = --scale-batches (CPU speculate loop vs one batched
             GPU propose, eager and CUDA graph).

Run (this machine):
  LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.580.105.08 \
  python benchmarks/bench_specbench.py --mode both --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_vs_cpu import SimResult, accept_len  # noqa: E402
from suffix_gpu.proposer import SuffixGPUDrafter  # noqa: E402

DEFAULT_DATA = str(Path.home() / "question.jsonl")
DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CATEGORIES = ("translation", "summarization", "math_reasoning")


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_specbench(path: str, categories: list[str], tokenizer,
                   max_prompt_tokens: int, min_gen_tokens: int,
                   limit: int) -> dict[str, list[tuple[list[int], list[int]]]]:
    per_cat: dict[str, list[tuple[list[int], list[int]]]] = {
        c: [] for c in categories}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cat = d.get("category")
            if cat not in per_cat or len(per_cat[cat]) >= limit:
                continue
            refs = d.get("reference") or []
            if not refs or not refs[0]:
                continue
            prompt = tokenizer.encode(d["turns"][0],
                                      add_special_tokens=False)
            gen = tokenizer.encode(refs[0], add_special_tokens=False)
            if len(gen) < min_gen_tokens:
                continue
            per_cat[cat].append((prompt[-max_prompt_tokens:], gen))
    return per_cat


# ----------------------------------------------------------------------
# replay simulation (per-request prompt lengths)
# ----------------------------------------------------------------------
def replay_gpu(drafter: SuffixGPUDrafter, items, device: torch.device,
               name: str, use_graph: bool = False,
               ingest_chunk: int = 64) -> SimResult:
    b = len(items)
    k = drafter.k
    streams = [np.asarray(p + g, dtype=np.int64) for p, g in items]
    prompt_lens = np.asarray([len(p) for p, _ in items], dtype=np.int64)
    s_buf = max(len(s) for s in streams) + k + 8
    buf = torch.zeros((b, s_buf), dtype=torch.int32, device=device)
    for i, (p, _) in enumerate(items):
        buf[i, :len(p)] = torch.tensor(p, dtype=torch.int32)
    lens = prompt_lens.copy()
    res = SimResult(name)

    pending = [[int(streams[i][prompt_lens[i]])] for i in range(b)]
    finished = np.zeros(b, dtype=bool)
    flushed = np.zeros(b, dtype=bool)
    num_tok_t = torch.from_numpy(lens.astype(np.int32)).to(device)
    sampled_buf = torch.full((b, k + 1), -1, dtype=torch.int32,
                             device=device)

    graph = None
    if use_graph and device.type == "cuda":
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
        act = [i for i in range(b) if not flushed[i]]
        if drafter.global_index is not None and act:
            keys = [f"{name}-{i}" for i in act]
            rows = [buf[i, int(prompt_lens[i]):int(lens[i])] for i in act]
            lengths = [int(lens[i]) - int(prompt_lens[i]) for i in act]
            drafter.ingest_active(keys, rows, lengths, chunk=ingest_chunk)
            fin = [i for i in act if finished[i] and not pending[i]]
            if fin:
                drafter.ingest_active(
                    [f"{name}-{i}" for i in fin],
                    [buf[i, int(prompt_lens[i]):int(lens[i])] for i in fin],
                    [int(lens[i]) - int(prompt_lens[i]) for i in fin],
                    final=True, chunk=ingest_chunk)
                for i in fin:
                    flushed[i] = True
    return res


def replay_cpu(cache, items, k: int, depth: int, spec_factor: float,
               min_token_prob: float, wave: int, name: str) -> SimResult:
    b = len(items)
    streams = [np.asarray(p + g, dtype=np.int64) for p, g in items]
    prompt_lens = [len(p) for p, _ in items]
    lens = np.asarray(prompt_lens, dtype=np.int64)
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
            rid = f"{name}-w{wave}r{i}"
            if not started[i]:
                cache.start_request(
                    rid, streams[i][:prompt_lens[i]].astype(np.int32))
                started[i] = True
            if pending[i]:
                cache.add_active_response(rid, pending[i])
                pending[i] = []
            pos = int(lens[i])
            start = max(0, pos - depth)
            pattern = streams[i][start:pos].astype(np.int32)
            d = cache.speculate(
                rid, pattern, max_spec_tokens=k,
                max_spec_factor=spec_factor,
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
                cache.add_active_response(f"{name}-w{wave}r{i}", pending[i])
                pending[i] = []
                cache.stop_request(f"{name}-w{wave}r{i}")
    return res


def run_replay(args, device: torch.device, data) -> None:
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    for cat, items in data.items():
        toks = sum(len(p) + len(g) for p, g in items)
        gen = sum(len(g) for _, g in items)
        print(f"\n== replay [{cat}] n={len(items)} total_tokens={toks} "
              f"gen_tokens={gen} k={args.k} depth={args.depth} "
              f"factor={args.spec_factor} minp={args.min_token_prob}")

        cache = SuffixDecodingCache(max_tree_depth=args.depth,
                                    max_cached_requests=1000)
        for w in range(args.waves):
            r = replay_cpu(cache, items, args.k, args.depth,
                           args.spec_factor, args.min_token_prob, w,
                           f"cpu-wave{w}")
            print("  " + r.report())
        del cache

        variants = [(True, False, "gpu-eager"), (True, True, "gpu-graph"),
                    (False, False, "gpu-local-eager")]
        for enable_global, use_graph, tag in variants:
            if use_graph and device.type != "cuda":
                continue
            extra = {}
            if args.max_occurrences is not None:
                extra["max_occurrences"] = args.max_occurrences
            if args.num_backoff is not None:
                extra["num_backoff"] = args.num_backoff
            drafter = SuffixGPUDrafter(
                k=args.k, device=device, max_pattern_len=args.depth,
                min_match_len=1,
                enable_global=enable_global,
                global_capacity=1 << 20, delta_capacity=1 << 15,
                max_spec_factor=args.spec_factor,
                min_token_prob=args.min_token_prob,
                rebuild_stream=torch.cuda.Stream(device)
                if device.type == "cuda" else None,
                **extra,
            )
            for w in range(args.waves):
                r = replay_gpu(drafter, items, device,
                               f"{tag}-wave{w}", use_graph=use_graph)
                print("  " + r.report())
            del drafter


# ----------------------------------------------------------------------
# lockstep agreement (teacher forcing)
# ----------------------------------------------------------------------
def run_agreement(args, device: torch.device, data) -> None:
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    print(f"\n== agreement (teacher-forced lockstep) k={args.k} "
          f"depth={args.depth} factor={args.spec_factor} "
          f"minp={args.min_token_prob}")
    for cat, items in data.items():
        b = len(items)
        streams = [np.asarray(p + g, dtype=np.int64) for p, g in items]
        prompt_lens = [len(p) for p, _ in items]
        s_buf = max(len(s) for s in streams) + args.k + 8

        cache = SuffixDecodingCache(max_tree_depth=args.depth,
                                    max_cached_requests=1000)
        extra = {}
        if args.max_occurrences is not None:
            extra["max_occurrences"] = args.max_occurrences
        if args.num_backoff is not None:
            extra["num_backoff"] = args.num_backoff
        drafter = SuffixGPUDrafter(
            k=args.k, device=device, max_pattern_len=args.depth,
            min_match_len=1, enable_global=False,
            max_spec_factor=args.spec_factor,
            min_token_prob=args.min_token_prob,
            **extra)

        buf = torch.zeros((b, s_buf), dtype=torch.int32, device=device)
        for i, (p, _) in enumerate(items):
            buf[i, :len(p)] = torch.tensor(p, dtype=torch.int32)
            cache.start_request(
                f"{cat}-r{i}", streams[i][:prompt_lens[i]].astype(np.int32))
        lens = np.asarray(prompt_lens, dtype=np.int64)

        steps = exact = first = 0
        cpu_alen: list[int] = []
        gpu_alen: list[int] = []
        max_gen = max(len(g) for _, g in items)
        for t in range(max_gen - 1):
            active = [i for i in range(b)
                      if t + 1 < len(streams[i]) - prompt_lens[i]]
            if not active:
                break
            for i in active:
                nxt = int(streams[i][lens[i]])
                cache.add_active_response(f"{cat}-r{i}", [nxt])
                buf[i, int(lens[i])] = nxt
                lens[i] += 1
            num_tok = torch.from_numpy(lens.astype(np.int32)).to(device)
            draft_t, nv_t = drafter.propose(num_tok, buf)
            draft = draft_t.cpu().numpy()
            nv = nv_t.cpu().numpy()
            for i in active:
                pos = int(lens[i])
                start = max(0, pos - args.depth)
                pattern = streams[i][start:pos].astype(np.int32)
                d = cache.speculate(
                    f"{cat}-r{i}", pattern, max_spec_tokens=args.k,
                    max_spec_factor=args.spec_factor,
                    min_token_prob=args.min_token_prob)
                cd = [int(x) for x in d.token_ids]
                gd = [int(x) for x in draft[i, :nv[i]]]
                steps += 1
                if cd == gd:
                    exact += 1
                if (cd[:1] or [-1]) == (gd[:1] or [-1]):
                    first += 1
                ref = streams[i][pos:]
                cpu_alen.append(accept_len(cd, ref))
                gpu_alen.append(accept_len(gd, ref))
        print(f"  [{cat}] steps={steps} exact-draft={exact / steps:6.1%} "
              f"first-token={first / steps:6.1%} "
              f"accept_len cpu={statistics.mean(cpu_alen):5.2f} "
              f"gpu={statistics.mean(gpu_alen):5.2f}")
        del drafter, cache


# ----------------------------------------------------------------------
# batch-scaling latency (warm state)
# ----------------------------------------------------------------------
def run_scale(args, device: torch.device, data) -> None:
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    pool = [it for items in data.values() for it in items]
    iters = args.scale_iters
    s_buf = 4096
    print(f"\n== scale (warm, per-step draft cost for the whole batch) "
          f"k={args.k} depth={args.depth} iters={iters}")
    print("  B | cpu ms/step | gpu eager ms | gpu graph ms | "
          "cpu/graph speedup")
    for b in args.scale_batches:
        items = [pool[i % len(pool)] for i in range(b)]
        streams = [np.asarray(p + g, dtype=np.int64) for p, g in items]
        prompt_lens = [len(p) for p, _ in items]
        # park every request mid-generation
        pos = [min(pl + max(1, len(g) // 2), len(s) - 1)
               for (p, g), pl, s in zip(items, prompt_lens, streams)]

        # --- CPU: warm global tree with wave-0 full responses ---
        cache = SuffixDecodingCache(max_tree_depth=args.depth,
                                    max_cached_requests=max(1000, 4 * b))
        for i in range(b):
            rid = f"s-w0-{i}"
            cache.start_request(
                rid, streams[i][:prompt_lens[i]].astype(np.int32))
            cache.add_active_response(
                rid, streams[i][prompt_lens[i]:].astype(np.int32))
            cache.stop_request(rid)
        pats = []
        for i in range(b):
            rid = f"s-w1-{i}"
            cache.start_request(
                rid, streams[i][:prompt_lens[i]].astype(np.int32))
            cache.add_active_response(
                rid, streams[i][prompt_lens[i]:pos[i]].astype(np.int32))
            start = max(0, pos[i] - args.depth)
            pats.append(streams[i][start:pos[i]].astype(np.int32))
        for i in range(b):  # warm-up pass
            cache.speculate(f"s-w1-{i}", pats[i], max_spec_tokens=args.k,
                            max_spec_factor=args.spec_factor,
                            min_token_prob=args.min_token_prob)
        t0 = time.perf_counter()
        for _ in range(iters):
            for i in range(b):
                cache.speculate(f"s-w1-{i}", pats[i],
                                max_spec_tokens=args.k,
                                max_spec_factor=args.spec_factor,
                                min_token_prob=args.min_token_prob)
        cpu_ms = (time.perf_counter() - t0) * 1e3 / iters
        del cache

        # --- GPU: same warm state ---
        extra = {}
        if args.max_occurrences is not None:
            extra["max_occurrences"] = args.max_occurrences
        if args.num_backoff is not None:
            extra["num_backoff"] = args.num_backoff
        drafter = SuffixGPUDrafter(
            k=args.k, device=device, max_pattern_len=args.depth,
            min_match_len=1, enable_global=True,
            global_capacity=1 << 20, delta_capacity=1 << 16,
            max_spec_factor=args.spec_factor,
            min_token_prob=args.min_token_prob,
            rebuild_stream=torch.cuda.Stream(device)
            if device.type == "cuda" else None,
            **extra)
        rows = [torch.tensor(streams[i][prompt_lens[i]:],
                             dtype=torch.int32, device=device)
                for i in range(b)]
        drafter.harvest_rows(rows, [len(r) for r in rows])
        for _ in range(1000):
            drafter.poll()
            if drafter.global_index._rebuild_event is None:
                break
            time.sleep(0.005)
        buf = torch.zeros((b, s_buf), dtype=torch.int32, device=device)
        for i in range(b):
            n = min(pos[i], s_buf)
            buf[i, :n] = torch.tensor(streams[i][:n], dtype=torch.int32)
        num_tok = torch.tensor([min(p, s_buf) for p in pos],
                               dtype=torch.int32, device=device)
        for _ in range(3):
            drafter.propose(num_tok, buf)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            drafter.propose(num_tok, buf)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eager_ms = (time.perf_counter() - t0) * 1e3 / iters
        graph_ms = float("nan")
        if device.type == "cuda":
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
        print(f"  {b:4d} | {cpu_ms:11.2f} | {eager_ms:12.2f} | "
              f"{graph_ms:12.2f} | {cpu_ms / graph_ms:6.1f}x")
        del drafter, buf, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["replay", "agreement", "scale", "both", "all"],
                    default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    ap.add_argument("--limit", type=int, default=80,
                    help="max questions per category")
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--min-gen-tokens", type=int, default=16)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--spec-factor", type=float, default=2.0)
    ap.add_argument("--min-token-prob", type=float, default=0.1)
    ap.add_argument("--scale-batches", type=lambda s: [int(x) for x in
                    s.split(",")], default=[8, 32, 128, 256, 512])
    ap.add_argument("--scale-iters", type=int, default=20)
    ap.add_argument("--max-occurrences", type=int, default=None,
                    help="override drafter max_occurrences (default: class default)")
    ap.add_argument("--num-backoff", type=int, default=None,
                    help="override drafter num_backoff (default: class default)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    md5 = hashlib.md5(Path(args.data).read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"torch {torch.__version__} device={device} "
          f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")
    print(f"data={args.data} md5={md5}")
    print(f"tokenizer={tokenizer.name_or_path} vocab={len(tokenizer)}")

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    data = load_specbench(args.data, cats, tokenizer,
                          args.max_prompt_tokens, args.min_gen_tokens,
                          args.limit)
    for c, items in data.items():
        print(f"  {c}: {len(items)} questions")

    if args.mode in ("replay", "both", "all"):
        run_replay(args, device, data)
    if args.mode in ("agreement", "both", "all"):
        run_agreement(args, device, data)
    if args.mode in ("scale", "all"):
        run_scale(args, device, data)


if __name__ == "__main__":
    main()
