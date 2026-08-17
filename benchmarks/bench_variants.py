"""Variant-corpus bench: expose drafter accuracy gaps hidden by replay.

Motivation
----------
`bench_specbench.py --mode replay` uses each question's reference answer
as both the "target model" stream *and* the sequence that gets ingested
into the global index. After wave 1 warms the index, the longest match
by construction has a 100%-correct continuation of the target, so
mode-vote / arctic-score / multi-length-backoff never divide.

The vLLM engine test `test_suffix_gpu_acceptance` catches a different
regime: the same prompt is served 10 times, each output drifts by a
few tokens (non-batch-invariant math), and by wave 10 the tree holds
~9 near-duplicate variants of the target being drafted. That is
exactly where SuffixGPU's window-mode vote across occurrences and
arctic's suffix-tree exact per-child counts split, so wave-10 GPU
acceptance sits noticeably below CPU. This bench reproduces that
regime offline (drafter-level, no LLM forward pass).

Design
------
For each Spec-Bench question q we pre-generate `W` noisy variants of
the reference answer (each token replaced iid with probability `eps`
by a random vocab token). On wave `w`:

1. Draft-replay against `target = variants[q][w]`. Neither drafter
   ingests the target row (arctic: `start_request` without
   `add_active_response`; GPU: no `ingest_active` for target). So the
   target is held out from the corpus.
2. Ingest `variants[q][w]` into both drafters (arctic:
   `start_request` + `add_active_response` + `stop_request` per q;
   GPU: `ingest_active(chunk)` with the engine's chunked write path,
   final flush at wave-end).

By wave ~10 the arctic global tree and the GPU global index each hold
`w` near-duplicate documents of the target being drafted, and the
GPU/CPU accept-rate gap opens up.

Metrics reported per wave, side-by-side and as (cpu - gpu) deltas:
  - `tokens/step`         committed / req_steps           (higher = better)
  - `accept_rate`         accepted / drafted (all steps, engine metric-compatible)
  - `real_rate`           accepted / drafted on steps that actually drafted
                          (mirrors `test_suffix_gpu_acceptance`'s
                           `num_accept / num_draft` where padded slots
                           are excluded)
  - `mean_al`             mean accepted-token length over steps that drafted

Run:
  LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.580.105.08 \
  python benchmarks/bench_variants.py --device cuda \
    --data ~/question.jsonl --waves 10 --eps 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_specbench import load_specbench  # noqa: E402
from benchmarks.bench_vs_cpu import accept_len  # noqa: E402
from suffix_gpu.proposer import SuffixGPUDrafter  # noqa: E402

DEFAULT_DATA = str(Path.home() / "question.jsonl")

_LEGACY_PRESET = dict(vote_smoothing_alpha=0.0, local_mode="backoff",
                      merge_paths=False, dynamic_k=False,
                      eviction="fifo")


def _preset_extra(args) -> dict:
    extra = {}
    if args.preset == "legacy":
        extra.update(_LEGACY_PRESET)
    if args.max_occurrences is not None:
        extra["max_occurrences"] = args.max_occurrences
    if args.alpha is not None:
        extra["vote_smoothing_alpha"] = args.alpha
    if args.eviction is not None:
        extra["eviction"] = args.eviction
    return extra

DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CATEGORIES = ("translation", "summarization", "math_reasoning")


# ----------------------------------------------------------------------
# per-wave metrics
# ----------------------------------------------------------------------
@dataclass
class WaveResult:
    name: str
    steps: int = 0
    req_steps: int = 0
    committed: int = 0
    drafted: int = 0          # over all req_steps (incl. zero-draft ones)
    accepted: int = 0
    real_drafted: int = 0     # sum over req_steps where nv > 0
    drafted_req_steps: int = 0
    accepted_on_drafted: int = 0
    step_ms: list[float] = field(default_factory=list)

    @property
    def tokens_per_step(self) -> float:
        return self.committed / max(1, self.req_steps)

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.drafted)

    @property
    def real_rate(self) -> float:
        return self.accepted_on_drafted / max(1, self.real_drafted)

    @property
    def mean_al(self) -> float:
        return self.accepted_on_drafted / max(1, self.drafted_req_steps)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.step_ms) if self.step_ms else 0.0

    def row(self) -> str:
        return (f"tok/step={self.tokens_per_step:5.2f}  "
                f"accept={self.accept_rate:6.1%}  "
                f"real={self.real_rate:6.1%}  "
                f"mean_al={self.mean_al:4.2f}  "
                f"drafted/step={self.drafted / max(1, self.req_steps):5.2f}  "
                f"ms={self.mean_ms:6.2f}")


# ----------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------
def make_variants(ref: list[int], n_variants: int, eps: float,
                  vocab_size: int, seed: int) -> list[np.ndarray]:
    """Return `n_variants` noisy copies of `ref`.

    Each position is replaced with a random token id (uniform over the
    tokenizer's vocab) with probability `eps`; other positions are kept
    verbatim. Length is preserved so `accept_len` against the target
    stays well-defined.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(ref, dtype=np.int64)
    out: list[np.ndarray] = []
    for _ in range(n_variants):
        v = base.copy()
        if eps > 0 and vocab_size > 1 and v.size > 0:
            mask = rng.random(v.size) < eps
            n = int(mask.sum())
            if n:
                v[mask] = rng.integers(0, vocab_size, size=n,
                                       dtype=np.int64)
        out.append(v)
    return out


# ----------------------------------------------------------------------
# target replay (does NOT ingest target into either drafter)
# ----------------------------------------------------------------------
def replay_target_cpu(cache, items, k: int, depth: int,
                      spec_factor: float, min_token_prob: float,
                      name: str) -> WaveResult:
    """Draft-only replay against a held-out target using arctic.

    The prompt is registered via `start_request` so `speculate` has the
    local tree context (mirrors the engine, which sees the prompt as
    part of the request state). `add_active_response` is deliberately
    *not* called during the draft: the target tokens must never enter
    the global tree, or wave w+1's "held-out" property is destroyed.
    """
    b = len(items)
    streams = [np.asarray(p + list(g), dtype=np.int64) for p, g in items]
    prompt_lens = [len(p) for p, _ in items]
    lens = np.asarray(prompt_lens, dtype=np.int64)
    finished = np.zeros(b, dtype=bool)
    res = WaveResult(name)

    rids = [f"{name}-r{i}" for i in range(b)]
    for i in range(b):
        cache.start_request(rids[i],
                            streams[i][:prompt_lens[i]].astype(np.int32))
    try:
        while not finished.all():
            t0 = time.perf_counter()
            drafts: list[list[int]] = []
            for i in range(b):
                if finished[i]:
                    drafts.append([])
                    continue
                pos = int(lens[i])
                start = max(0, pos - depth)
                pattern = streams[i][start:pos].astype(np.int32)
                d = cache.speculate(rids[i], pattern, max_spec_tokens=k,
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
                n_real = len(d)
                n_acc = min(a, max(0, remain - 1))
                commit = min(a + 1, remain)
                res.drafted += n_real
                res.accepted += n_acc
                if n_real > 0:
                    res.real_drafted += n_real
                    res.drafted_req_steps += 1
                    res.accepted_on_drafted += n_acc
                lens[i] += commit
                res.committed += commit
                if lens[i] >= len(ref):
                    finished[i] = True
    finally:
        for rid in rids:
            try:
                cache.stop_request(rid)
            except ValueError:
                pass
    return res


def replay_target_gpu(drafter: SuffixGPUDrafter, items, device: torch.device,
                      name: str) -> WaveResult:
    """Draft-only replay against a held-out target using SuffixGPUDrafter.

    Mirrors `bench_specbench.replay_gpu`'s `propose_with_update` loop
    (async-scheduling-glue, updates the resident token buffer from
    sampled ids on device) but *never* calls `ingest_active`, so the
    target tokens do not enter the global index.
    """
    b = len(items)
    k = drafter.k
    streams = [np.asarray(p + list(g), dtype=np.int64) for p, g in items]
    prompt_lens = np.asarray([len(p) for p, _ in items], dtype=np.int64)
    s_buf = max(len(s) for s in streams) + k + 8
    buf = torch.zeros((b, s_buf), dtype=torch.int32, device=device)
    for i, (p, _) in enumerate(items):
        buf[i, :len(p)] = torch.tensor(p, dtype=torch.int32)
    lens = prompt_lens.copy()
    res = WaveResult(name)

    pending = [[int(streams[i][prompt_lens[i]])] for i in range(b)]
    finished = np.zeros(b, dtype=bool)
    num_tok_t = torch.from_numpy(lens.astype(np.int32)).to(device)
    sampled_buf = torch.full((b, k + 1), -1, dtype=torch.int32,
                             device=device)

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
            n_real = len(d)
            n_acc = min(a, max(0, remain - 1))
            res.drafted += n_real
            res.accepted += n_acc
            if n_real > 0:
                res.real_drafted += n_real
                res.drafted_req_steps += 1
                res.accepted_on_drafted += n_acc
            commit = min(a + 1, remain)
            pending[i] = [int(t) for t in ref[pos:pos + commit]]
    return res


# ----------------------------------------------------------------------
# wave-level corpus ingestion (near-duplicates accumulate here)
# ----------------------------------------------------------------------
def ingest_wave_cpu(cache, items, variants: list[list[np.ndarray]],
                    wave: int, tag: str) -> None:
    """Append `variants[q][wave]` for every question to the arctic cache.

    Each variant becomes one cached response under a fresh request id,
    so on subsequent waves the global suffix tree matches identically
    to the engine that has served the same prompt `wave + 1` times.
    """
    for i, (p, _) in enumerate(items):
        rid = f"{tag}-q{i}-w{wave}"
        cache.start_request(rid, np.asarray(p, dtype=np.int32))
        cache.add_active_response(
            rid, variants[i][wave].astype(np.int32))
        cache.stop_request(rid)


def ingest_wave_gpu(drafter: SuffixGPUDrafter, variants: list[list[np.ndarray]],
                    wave: int, tag: str, chunk: int,
                    device: torch.device) -> None:
    """Chunked ingestion of `variants[q][wave]` into the GPU global index.

    Feeds each row through `ingest_active` with a growing valid length
    (chunk-by-chunk), mirroring the engine's in-flight write path
    (`docs.append(row[start:ln])` with overlap on every chunk-sized
    step), then a `final=True` flush. Blocks briefly until any pending
    background SA rebuild finishes so wave w+1's draft sees the new
    index snapshot in place.
    """
    if drafter.global_index is None:
        return
    b = len(variants)
    rows = [torch.from_numpy(variants[i][wave].astype(np.int32)).to(device)
            for i in range(b)]
    keys = [f"{tag}-q{i}-w{wave}" for i in range(b)]
    total_lens = [int(r.shape[0]) for r in rows]
    max_len = max(total_lens) if total_lens else 0

    stop = chunk
    while stop < max_len:
        lens_now = [min(stop, tl) for tl in total_lens]
        drafter.ingest_active(keys, rows, lens_now, chunk=chunk)
        stop += chunk
    drafter.ingest_active(keys, rows, total_lens,
                          final=True, chunk=chunk)

    for _ in range(4000):
        drafter.poll()
        if drafter.global_index._rebuild_event is None:
            return
        time.sleep(0.002)


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def _delta(cpu: WaveResult, gpu: WaveResult) -> str:
    d_tps = cpu.tokens_per_step - gpu.tokens_per_step
    d_acc = (cpu.accept_rate - gpu.accept_rate) * 100.0
    d_real = (cpu.real_rate - gpu.real_rate) * 100.0
    d_al = cpu.mean_al - gpu.mean_al
    return (f"Δ(cpu-gpu) tok/step={d_tps:+5.2f}  "
            f"accept={d_acc:+5.1f}pp  "
            f"real={d_real:+5.1f}pp  "
            f"mean_al={d_al:+5.2f}")


def run_variants(args, device: torch.device, data, vocab_size: int) -> int:
    """Return worst (cpu - gpu) real_rate delta across every wave, in pp."""
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    worst_real_delta_pp = -1e9

    for cat, items in data.items():
        # variants[q][w] used for wave w's target and, after wave w's
        # draft-replay, for wave w's ingestion.
        variants: list[list[np.ndarray]] = []
        for i, (_, gen) in enumerate(items):
            variants.append(make_variants(
                list(gen), n_variants=args.waves, eps=args.eps,
                vocab_size=vocab_size,
                seed=args.seed + 1000 * (hash(cat) & 0xffff) + i))

        print(f"\n== variants [{cat}] n={len(items)} W={args.waves} "
              f"eps={args.eps} k={args.k} depth={args.depth} "
              f"factor={args.spec_factor} minp={args.min_token_prob} "
              f"chunk={args.chunk}")

        cache = SuffixDecodingCache(
            max_tree_depth=args.depth,
            max_cached_requests=-1)  # unlimited: keep every wave's variant
        drafter = SuffixGPUDrafter(
            k=args.k, device=device, max_pattern_len=args.depth,
            min_match_len=1,
            enable_global=True,
            global_capacity=args.global_capacity,
            delta_capacity=args.delta_capacity,
            max_spec_factor=args.spec_factor,
            min_token_prob=args.min_token_prob,
            rebuild_stream=torch.cuda.Stream(device)
            if device.type == "cuda" else None,
            **_preset_extra(args),
        )

        for w in range(args.waves):
            # Target for this wave = variants[q][w]; corpus at this
            # point holds variants[0..w-1] for every q (ingested at
            # previous wave-ends). Wave 0 is a cold-start draft.
            target_items = [
                (list(items[i][0]), variants[i][w].tolist())
                for i in range(len(items))]

            cpu_res = replay_target_cpu(
                cache, target_items, args.k, args.depth,
                args.spec_factor, args.min_token_prob,
                name=f"cpu-w{w}")
            gpu_res = replay_target_gpu(
                drafter, target_items, device, name=f"gpu-w{w}")

            print(f"  wave={w:2d}  cpu {cpu_res.row()}")
            print(f"           gpu {gpu_res.row()}")
            print(f"           {_delta(cpu_res, gpu_res)}")
            worst_real_delta_pp = max(
                worst_real_delta_pp,
                (cpu_res.real_rate - gpu_res.real_rate) * 100.0)

            # Ingest wave w's variant into both drafters so wave w+1
            # sees an extra near-duplicate of its target.
            ingest_wave_cpu(cache, items, variants, w, tag=f"{cat}")
            ingest_wave_gpu(drafter, variants, w, tag=f"{cat}",
                            chunk=args.chunk, device=device)

        del cache, drafter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return worst_real_delta_pp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    ap.add_argument("--limit", type=int, default=80,
                    help="max questions per category")
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--min-gen-tokens", type=int, default=16)
    ap.add_argument("--waves", type=int, default=10,
                    help="matches test_suffix_gpu_acceptance's 10 chats")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="per-token replacement probability for variants")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--spec-factor", type=float, default=2.0)
    ap.add_argument("--min-token-prob", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=64,
                    help="ingest_active chunk size (engine writes at 64)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--max-occurrences", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--eviction", choices=["fifo", "lfu"], default=None)
    ap.add_argument("--global-capacity", type=int, default=1 << 21)
    ap.add_argument("--delta-capacity", type=int, default=1 << 16)
    ap.add_argument("--fail-if-real-rate-delta", type=float, default=None,
                    help="exit nonzero if any wave's (cpu-gpu) real_rate "
                         "exceeds this many percentage points")
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

    worst_pp = run_variants(args, device, data, vocab_size=len(tokenizer))
    print(f"\nworst (cpu-gpu) real_rate delta across all waves: "
          f"{worst_pp:+5.2f}pp")

    if (args.fail_if_real_rate_delta is not None
            and worst_pp > args.fail_if_real_rate_delta):
        print(f"FAIL: exceeded threshold "
              f"{args.fail_if_real_rate_delta:+5.2f}pp")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
