# SuffixGPU

GPU-resident suffix decoding drafter for LLM speculative decoding.

SuffixGPU moves the [SuffixDecoding](https://arxiv.org/abs/2411.04975) draft
path fully onto the accelerator: variable-length suffix matching with
frequency-greedy expansion, plus a cross-request global memory backed by a
GPU suffix array. It mirrors the vLLM `ngram_gpu` proposer contract — device
tensors in, device tensors out, **no per-step host synchronization** — so it
composes with async scheduling and CUDA-graph capture.

## Highlights

- **Fully device-resident draft path.** `propose` / `update_state` /
  `propose_with_update` never sync to the host; all shapes are static and
  loop bounds are fixed, keeping the hot path `torch.compile`- and
  CUDA-graph-friendly.
- **Multi-length backoff matching.** Per-request self-match considers the
  longest suffix plus shorter, better-supported lengths (`num_backoff`
  candidates from one match_back pass); the global path probes the same
  ladder of capped lengths from a single SA walk.
- **Score-ranked expansion.** Depth-wise majority vote over matched
  continuations with fused arctic-style scoring (sum of per-depth chain
  probabilities) and an adaptive stop rule (`max_spec_factor`,
  `max_spec_offset`, `min_token_prob`); the best-scored candidate wins.
- **Cross-request global memory.** Finished (or in-flight) sequences are
  ingested into a corpus ring + append-only delta buffer. The suffix array is
  rebuilt on a **side CUDA stream, double-buffered**, and swapped in-place so
  tensor identity is preserved for captured CUDA graphs.
- **Fused Triton kernels** for the hot path (`match_back`, `sa_search`,
  `expand_chain`, `first_occurrences`, `scatter_append`) with pure-PyTorch
  fallbacks — the package runs on CPU / CUDA / MPS without Triton.
- **int32 intermediates** on the dominant `[B, S]` buffers to cut memory
  traffic.

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"        # + pytest
# optional: CPU oracle for the equivalence tests / accuracy benchmark
uv pip install -e ".[vllm]"        # + arctic-inference
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.5. Triton is optional and picked up
automatically on CUDA.

## Quickstart

```python
import torch
from suffix_gpu import SuffixGPUDrafter

drafter = SuffixGPUDrafter(
    k=16,                  # max draft tokens per request
    device="cuda",
    enable_global=True,    # cross-request suffix-array memory
    max_spec_factor=2.0,   # adaptive cap: factor * match_len + offset
    min_token_prob=0.1,    # stop expanding when chain prob drops below
)

# Resident decode state (all on device, int32):
#   token_ids_gpu      [B, S]  token buffer per request
#   num_tokens_no_spec [B]     valid length per request
#   sampled_token_ids  [B, T]  last verifier output, -1 padded

draft, num_valid, num_tokens = drafter.propose_with_update(
    num_tokens_no_spec, token_ids_gpu, sampled_token_ids,
)
# draft:     [B, k] int32, -1 padded
# num_valid: [B]    int32, number of proposed tokens per request

# feed finished requests into the global memory
drafter.harvest_finished(row_indices, lengths, token_ids_gpu)
drafter.poll()   # swap in a completed background SA rebuild, if any
```

`propose(num_tokens_no_spec, token_ids_gpu, combined_mask)` is also available
when you manage state updates yourself.

## Configuration

| Argument | Default | Meaning |
| --- | --- | --- |
| `k` | — | Maximum draft length per request |
| `device` | `"cpu"` | Torch device for all buffers |
| `max_pattern_len` | `32` | Longest suffix (pattern) considered for matching |
| `min_match_len` | `1` | Minimum suffix match length to draft from |
| `max_occurrences` | `32` | Occurrences kept per match for expansion/voting |
| `enable_global` | `False` | Enable the cross-request suffix-array memory |
| `global_capacity` | `1<<22` | Corpus ring capacity (tokens) |
| `delta_capacity` | `1<<16` | Append-only delta buffer capacity |
| `rebuild_threshold` | `delta_capacity // 2` | Delta fill level that triggers a background SA rebuild |
| `rebuild_stream` | `None` | CUDA stream for background rebuilds |
| `max_spec_factor` / `max_spec_offset` | `None` / `0.0` | Adaptive draft-length cap: `factor * match_len + offset` |
| `min_token_prob` | `0.0` | Cumulative-probability cutoff during expansion |
| `num_backoff` | `4` | Candidate match lengths per path: local support thresholds `2^0..2^(C-1)`, global capped lengths halving from `max_pattern_len` (plus a final 2; distinct caps saturate at ~log2 of the pattern length). `1` = longest match only; larger values probe more lengths at small marginal cost |

## How it works

```
                    ┌────────────────────────────────────────────┐
 token_ids [B,S] ──►│ 1. local match      backoff candidate      │
 num_tokens [B]     │    (local_matcher)  lengths in own ctx     │
                    │ 2. global match     SA interval search +   │
                    │    (sa_search /     delta brute-force scan │
                    │     global_index)   at capped lengths      │
                    │ 3. expand + score   majority vote, fused   │
                    │    (expand)         score, best wins       │
                    └────────────────┬───────────────────────────┘
                                     ▼
                        draft [B,k], num_valid [B]
```

1. **Local matching** — for each request, one match_back pass over its own
   context yields `num_backoff` candidate suffix lengths (for support
   threshold t, the largest length occurring ≥ t times — the t-th largest
   match_back value), each with up to `max_occurrences` earliest
   continuation sites.
2. **Global matching** — the same tail is searched in the shared corpus via
   fixed-iteration binary search on the suffix array (all pattern lengths
   `1..max_pattern_len` as one flattened batch, each backoff cap selecting
   its longest hit from the same walk), plus one brute-force scan of the
   not-yet-indexed delta buffer serving every cap.
3. **Expansion & selection** — all candidates share one fused expand+score
   kernel: continuations are extended depth by depth by majority vote,
   tracking the empirical chain probability for the adaptive stop rule and
   summing it into the arctic-style score (expected accepted tokens). The
   best-scored candidate wins per path, and the local/global winners
   compete by the same score.
4. **Global index maintenance** — new documents append to the delta; when it
   fills past `rebuild_threshold`, a fresh suffix array is built on a side
   stream over a staging corpus and event-poll swapped in without touching
   captured graphs. Oldest documents are evicted when the ring is full.

## Layout

| Path | Role |
| --- | --- |
| `suffix_gpu/proposer.py` | `SuffixGPUDrafter` — orchestrates match → expand → draft |
| `suffix_gpu/local_matcher.py` | Per-request variable-length suffix self-matching |
| `suffix_gpu/suffix_array.py` | Suffix array construction (prefix doubling, pure torch) |
| `suffix_gpu/sa_search.py` | Fixed-iteration batched binary search over the SA |
| `suffix_gpu/expand.py` | Frequency-ranked chain expansion + adaptive stop |
| `suffix_gpu/global_index.py` | Corpus ring, delta buffer, double-buffered SA rebuild |
| `suffix_gpu/triton_kernels.py` | Fused Triton kernels (optional, auto-detected) |
| `suffix_gpu/reference.py` | Naive host-side oracles used by the tests |

## Benchmarks

Full results, environment details, and all tables: **[RESULTS.md](RESULTS.md)**.

Drafter-level comparison vs the `arctic_inference.SuffixDecodingCache` CPU
suffix tree on [Spec-Bench](https://github.com/hemingkx/Spec-Bench) reference
replay (`question.jsonl`, md5 `0c39ae23e6f213549c66d6d691c99034`, tokenized
with `NousResearch/Meta-Llama-3.1-8B-Instruct` rev `d10aef79`) — NVIDIA L20,
torch 2.13, batch = whole category (58–80 requests), k=16:

| | tokens/step (cold / warm global) | propose per step, B=80 warm |
| --- | --- | --- |
| arctic suffix-cpu | 1.10–1.57 / 6.3–11.1 | 1.0–3.4 ms (loop over batch) |
| **SuffixGPU (CUDA graph)** | 1.09–1.56 / 6.2–11.2 | **0.5 ms**, flat, no host sync |

- **Correctness**: 221 tests pass, including fuzz equivalence vs arctic on
  unambiguous corpora; teacher-forced lockstep on Spec-Bench gives 54–77%
  token-identical drafts (residual = majority-vote tie ordering) with
  comparable useful-draft length against the reference continuation.
- **Batch scaling (warm)**: CPU wins small batches (B≤32); crossover at
  B≈64–128; at B=128/256/512 the GPU graph path is **2.2× / 2.7× / 2.1×
  faster** than the CPU speculate loop — and it never syncs to the host,
  so it composes with async scheduling and CUDA graphs.
- **Memory**: ≤ 532 MB reserved VRAM at B=256, S=16384, 4M-token corpus
  (persistent drafter state 96 MB; no extra CUDA-graph pool retention).

```bash
# Spec-Bench replay + lockstep agreement vs the Arctic CPU implementation
python benchmarks/bench_specbench.py --mode both --device cuda \
    --data question.jsonl --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct

# synthetic accuracy + latency sweep (eager & CUDA graph)
python benchmarks/bench_vs_cpu.py --mode both --device cuda \
    --batch 32 --k 16 --spec-factor 2.0 --min-token-prob 0.1

# persistent buffer / peak / CUDA-graph pool memory accounting
python benchmarks/bench_memory.py --device cuda
```

## Testing

```bash
pytest -v
```

Every module is tested against the naive references in
`suffix_gpu/reference.py`; the suite is parametrized over `cpu` / `cuda` /
`mps`. `tests/test_verification.py` additionally fuzzes end-to-end
draft/verify equivalence against `arctic-inference` (skipped automatically if
the package is not installed).

## TODO

- **Global tree save & load interface** — persist the cross-request global
  memory (corpus ring + suffix array + delta buffer) to disk and restore it
  across sessions, so the accumulated suffix statistics survive restarts
  instead of being rebuilt from scratch.

## References

- Oliaro et al., *SuffixDecoding: A Model-Free Approach to Speeding Up Large
  Language Model Inference* — [arXiv:2411.04975](https://arxiv.org/abs/2411.04975).
  The draft semantics reproduced on device: longest-suffix matching,
  frequency-greedy expansion, adaptive speculation length.
- [Snowflake ArcticInference](https://github.com/snowflakedb/ArcticInference) —
  CPU suffix-tree reference (`arctic_inference.suffix_decoding`), used as the
  equivalence oracle in tests and benchmarks.
- [vLLM](https://github.com/vllm-project/vllm)
  `v1/spec_decode/ngram_proposer_gpu.py` — the GPU proposer contract this
  drafter mirrors: device tensors in/out, no per-step host sync,
  async-scheduler compatible.
- Manber & Myers, *Suffix Arrays: A New Method for On-Line String Searches*
  (SIAM J. Comput., 1993) — the prefix-doubling construction in
  `suffix_gpu/suffix_array.py`.
