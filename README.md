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
pip install suffix-gpu                 # library only (torch >= 2.5)
pip install "suffix-gpu[test]"         # + pytest, numpy
pip install "suffix-gpu[vllm]"         # + arctic-inference (CPU oracle)
pip install "suffix-gpu[bench]"        # + benchmark deps (transformers, ...)
```

From source:

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
| `max_occurrences` | `128` | Occurrences kept per match for expansion/voting |
| `enable_global` | `False` | Enable the cross-request suffix-array memory |
| `global_capacity` | `1<<22` | Corpus ring capacity (tokens) |
| `delta_capacity` | `1<<16` | Append-only delta buffer capacity |
| `rebuild_threshold` | `delta_capacity // 2` | Delta fill level that triggers a background SA rebuild |
| `rebuild_stream` | `None` | CUDA stream for background rebuilds |
| `max_spec_factor` / `max_spec_offset` | `None` / `0.0` | Adaptive draft-length cap: `factor * match_len + offset` |
| `min_token_prob` | `0.0` | Cumulative-probability cutoff during expansion |
| `num_backoff` | `8` | Candidate match lengths per path: local support thresholds `2^0..2^(C-1)`, global capped lengths halving from `max_pattern_len` (plus a final 2; distinct caps saturate at ~log2 of the pattern length). `1` = longest match only; larger values probe more lengths at small marginal cost |

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

- **Drafter-level knob sweep:** full B/occ/backoff CPU/eager/graph timing tables are moved to the end of this README to keep the main benchmark section compact.

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

## Drafter-level knob sweep details

- **Drafter-level knob sweep (L20, k=16, depth=24, synthetic `bench_vs_cpu` workload, 10 iters):**

  CPU time depends on batch size and workload, but not on GPU-only `max_occurrences` / `num_backoff` knobs.

  High-repetition workload.

  | B | max_occurrences | num_backoff | CPU ms | GPU eager ms | GPU graph ms |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 32 | 32 | 1 | 0.481 | 2.585 | 0.414 |
  | 32 | 32 | 4 | 0.481 | 2.655 | 0.433 |
  | 32 | 32 | 8 | 0.481 | 2.830 | 0.441 |
  | 32 | 32 | 16 | 0.481 | 2.579 | 0.463 |
  | 32 | 64 | 1 | 0.481 | 2.662 | 0.449 |
  | 32 | 64 | 4 | 0.481 | 3.032 | 0.474 |
  | 32 | 64 | 8 | 0.481 | 2.421 | 0.489 |
  | 32 | 64 | 16 | 0.481 | 2.713 | 0.532 |
  | 32 | 128 | 1 | 0.481 | 2.798 | 0.510 |
  | 32 | 128 | 4 | 0.481 | 2.765 | 0.559 |
  | 32 | 128 | 8 | 0.481 | 2.637 | 0.593 |
  | 32 | 128 | 16 | 0.481 | 2.561 | 0.698 |
  | 32 | 256 | 1 | 0.481 | 2.583 | 0.692 |
  | 32 | 256 | 4 | 0.481 | 2.612 | 0.789 |
  | 32 | 256 | 8 | 0.481 | 2.493 | 0.959 |
  | 32 | 256 | 16 | 0.481 | 2.539 | 1.150 |
  | 64 | 32 | 1 | 0.762 | 2.537 | 0.404 |
  | 64 | 32 | 4 | 0.762 | 2.890 | 0.428 |
  | 64 | 32 | 8 | 0.762 | 2.576 | 0.441 |
  | 64 | 32 | 16 | 0.762 | 2.447 | 0.468 |
  | 64 | 64 | 1 | 0.762 | 2.484 | 0.435 |
  | 64 | 64 | 4 | 0.762 | 2.928 | 0.486 |
  | 64 | 64 | 8 | 0.762 | 2.623 | 0.528 |
  | 64 | 64 | 16 | 0.762 | 3.418 | 0.598 |
  | 64 | 128 | 1 | 0.762 | 2.570 | 0.504 |
  | 64 | 128 | 4 | 0.762 | 2.558 | 0.609 |
  | 64 | 128 | 8 | 0.762 | 2.634 | 0.749 |
  | 64 | 128 | 16 | 0.762 | 2.674 | 0.935 |
  | 64 | 256 | 1 | 0.762 | 2.503 | 0.693 |
  | 64 | 256 | 4 | 0.762 | 2.653 | 1.108 |
  | 64 | 256 | 8 | 0.762 | 2.717 | 1.307 |
  | 64 | 256 | 16 | 0.762 | 2.794 | 2.005 |
  | 128 | 32 | 1 | 2.226 | 2.611 | 0.599 |
  | 128 | 32 | 4 | 2.226 | 2.687 | 0.706 |
  | 128 | 32 | 8 | 2.226 | 2.524 | 0.737 |
  | 128 | 32 | 16 | 2.226 | 2.476 | 0.788 |
  | 128 | 64 | 1 | 2.226 | 2.427 | 0.646 |
  | 128 | 64 | 4 | 2.226 | 2.615 | 0.817 |
  | 128 | 64 | 8 | 2.226 | 2.707 | 0.902 |
  | 128 | 64 | 16 | 2.226 | 2.743 | 1.026 |
  | 128 | 128 | 1 | 2.226 | 2.535 | 0.750 |
  | 128 | 128 | 4 | 2.226 | 2.570 | 1.098 |
  | 128 | 128 | 8 | 2.226 | 2.557 | 1.325 |
  | 128 | 128 | 16 | 2.226 | 2.512 | 1.713 |
  | 128 | 256 | 1 | 2.226 | 2.380 | 1.074 |
  | 128 | 256 | 4 | 2.226 | 2.728 | 1.907 |
  | 128 | 256 | 8 | 2.226 | 2.967 | 2.733 |
  | 128 | 256 | 16 | 2.226 | 4.183 | 3.951 |
  | 256 | 32 | 1 | 4.067 | 5.592 | 0.887 |
  | 256 | 32 | 4 | 4.067 | 3.054 | 1.057 |
  | 256 | 32 | 8 | 4.067 | 2.626 | 1.115 |
  | 256 | 32 | 16 | 4.067 | 2.885 | 1.196 |
  | 256 | 64 | 1 | 4.067 | 2.614 | 0.946 |
  | 256 | 64 | 4 | 4.067 | 2.935 | 1.264 |
  | 256 | 64 | 8 | 4.067 | 2.971 | 1.415 |
  | 256 | 64 | 16 | 4.067 | 3.249 | 1.674 |
  | 256 | 128 | 1 | 4.067 | 2.724 | 1.076 |
  | 256 | 128 | 4 | 4.067 | 2.742 | 1.801 |
  | 256 | 128 | 8 | 4.067 | 2.755 | 2.256 |
  | 256 | 128 | 16 | 4.067 | 3.552 | 3.136 |
  | 256 | 256 | 1 | 4.067 | 2.565 | 1.697 |
  | 256 | 256 | 4 | 4.067 | 3.770 | 3.561 |
  | 256 | 256 | 8 | 4.067 | 5.315 | 5.095 |
  | 256 | 256 | 16 | 4.067 | 7.846 | 7.626 |

  Low-repetition workload.

  | B | max_occurrences | num_backoff | CPU ms | GPU eager ms | GPU graph ms |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 32 | 32 | 1 | 0.214 | 2.512 | 0.413 |
  | 32 | 32 | 4 | 0.214 | 2.485 | 0.434 |
  | 32 | 32 | 8 | 0.214 | 2.592 | 0.439 |
  | 32 | 32 | 16 | 0.214 | 2.516 | 0.462 |
  | 32 | 64 | 1 | 0.214 | 2.578 | 0.447 |
  | 32 | 64 | 4 | 0.214 | 2.520 | 0.473 |
  | 32 | 64 | 8 | 0.214 | 2.845 | 0.487 |
  | 32 | 64 | 16 | 0.214 | 2.607 | 0.532 |
  | 32 | 128 | 1 | 0.214 | 2.597 | 0.509 |
  | 32 | 128 | 4 | 0.214 | 2.637 | 0.558 |
  | 32 | 128 | 8 | 0.214 | 3.020 | 0.590 |
  | 32 | 128 | 16 | 0.214 | 2.778 | 0.696 |
  | 32 | 256 | 1 | 0.214 | 2.776 | 0.691 |
  | 32 | 256 | 4 | 0.214 | 2.715 | 0.787 |
  | 32 | 256 | 8 | 0.214 | 2.604 | 0.954 |
  | 32 | 256 | 16 | 0.214 | 2.725 | 1.149 |
  | 64 | 32 | 1 | 0.829 | 2.389 | 0.404 |
  | 64 | 32 | 4 | 0.829 | 2.681 | 0.428 |
  | 64 | 32 | 8 | 0.829 | 2.534 | 0.442 |
  | 64 | 32 | 16 | 0.829 | 2.567 | 0.468 |
  | 64 | 64 | 1 | 0.829 | 2.411 | 0.435 |
  | 64 | 64 | 4 | 0.829 | 2.622 | 0.484 |
  | 64 | 64 | 8 | 0.829 | 2.553 | 0.529 |
  | 64 | 64 | 16 | 0.829 | 2.603 | 0.599 |
  | 64 | 128 | 1 | 0.829 | 2.577 | 0.504 |
  | 64 | 128 | 4 | 0.829 | 2.541 | 0.610 |
  | 64 | 128 | 8 | 0.829 | 2.753 | 0.752 |
  | 64 | 128 | 16 | 0.829 | 2.743 | 0.936 |
  | 64 | 256 | 1 | 0.829 | 2.404 | 0.694 |
  | 64 | 256 | 4 | 0.829 | 2.498 | 1.106 |
  | 64 | 256 | 8 | 0.829 | 2.562 | 1.306 |
  | 64 | 256 | 16 | 0.829 | 2.749 | 1.991 |
  | 128 | 32 | 1 | 1.325 | 2.542 | 0.608 |
  | 128 | 32 | 4 | 1.325 | 2.559 | 0.671 |
  | 128 | 32 | 8 | 1.325 | 2.610 | 0.688 |
  | 128 | 32 | 16 | 1.325 | 2.619 | 0.736 |
  | 128 | 64 | 1 | 1.325 | 2.688 | 0.656 |
  | 128 | 64 | 4 | 1.325 | 2.469 | 0.775 |
  | 128 | 64 | 8 | 1.325 | 2.660 | 0.851 |
  | 128 | 64 | 16 | 1.325 | 2.868 | 0.975 |
  | 128 | 128 | 1 | 1.325 | 2.650 | 0.760 |
  | 128 | 128 | 4 | 1.325 | 2.581 | 1.051 |
  | 128 | 128 | 8 | 1.325 | 2.615 | 1.278 |
  | 128 | 128 | 16 | 1.325 | 2.771 | 1.648 |
  | 128 | 256 | 1 | 1.325 | 2.894 | 1.073 |
  | 128 | 256 | 4 | 1.325 | 2.778 | 1.870 |
  | 128 | 256 | 8 | 1.325 | 2.954 | 2.671 |
  | 128 | 256 | 16 | 1.325 | 4.100 | 3.897 |
  | 256 | 32 | 1 | 3.216 | 2.956 | 0.886 |
  | 256 | 32 | 4 | 3.216 | 2.622 | 1.012 |
  | 256 | 32 | 8 | 3.216 | 2.570 | 1.094 |
  | 256 | 32 | 16 | 3.216 | 2.592 | 1.176 |
  | 256 | 64 | 1 | 3.216 | 2.666 | 0.946 |
  | 256 | 64 | 4 | 3.216 | 2.674 | 1.210 |
  | 256 | 64 | 8 | 3.216 | 2.781 | 1.399 |
  | 256 | 64 | 16 | 3.216 | 3.032 | 1.658 |
  | 256 | 128 | 1 | 3.216 | 2.679 | 1.075 |
  | 256 | 128 | 4 | 3.216 | 3.709 | 1.751 |
  | 256 | 128 | 8 | 3.216 | 2.625 | 2.235 |
  | 256 | 128 | 16 | 3.216 | 3.327 | 3.119 |
  | 256 | 256 | 1 | 3.216 | 2.648 | 1.690 |
  | 256 | 256 | 4 | 3.216 | 3.724 | 3.505 |
  | 256 | 256 | 8 | 3.216 | 5.310 | 5.072 |
  | 256 | 256 | 16 | 3.216 | 7.832 | 7.606 |
