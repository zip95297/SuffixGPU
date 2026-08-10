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
- **Variable-length suffix matching.** Per-request longest-suffix self-match
  (single-pass rolling AND + one `topk`), following vLLM `NgramGPUKernel`
  occurrence semantics.
- **Frequency-ranked expansion.** Depth-wise majority vote over matched
  continuations with an adaptive stop rule (`max_spec_factor`,
  `max_spec_offset`, `min_token_prob`) matching Arctic/SuffixDecoding
  semantics.
- **Cross-request global memory.** Finished (or in-flight) sequences are
  ingested into a corpus ring + append-only delta buffer. The suffix array is
  rebuilt on a **side CUDA stream, double-buffered**, and swapped in-place so
  tensor identity is preserved for captured CUDA graphs.
- **Fused Triton kernels** for the hot path (`match_back`, `sa_search`,
  `expand_chain`, `scatter_append`) with pure-PyTorch fallbacks — the package
  runs on CPU / CUDA / MPS without Triton.
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

## How it works

```
                    ┌────────────────────────────────────────────┐
 token_ids [B,S] ──►│ 1. local match      longest suffix of each │
 num_tokens [B]     │    (local_matcher)  request in its own ctx │
                    │ 2. global match     SA interval search +   │
                    │    (sa_search /     delta brute-force scan │
                    │     global_index)                          │
                    │ 3. expand           depth-wise majority    │
                    │    (expand)         vote + adaptive stop   │
                    └────────────────┬───────────────────────────┘
                                     ▼
                        draft [B,k], num_valid [B]
```

1. **Local matching** — for each request, find the longest suffix of its
   generated tokens that reoccurs earlier in the same context, collecting up
   to `max_occurrences` continuation sites in one pass.
2. **Global matching** — the same tail is searched in the shared corpus via
   fixed-iteration binary search on the suffix array (all pattern lengths
   `1..max_pattern_len` as one flattened batch), plus a brute-force scan of
   the not-yet-indexed delta buffer.
3. **Expansion** — continuations are extended depth by depth; each step takes
   the majority token across occurrences and tracks an empirical chain
   probability used for the adaptive stop rule.
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

```bash
# accuracy vs the Arctic CPU implementation + latency sweep (eager & CUDA graph)
python benchmarks/bench_vs_cpu.py --mode both --device cuda \
    --batch 32 --k 16 --spec-factor 2.0 --min-token-prob 0.1

# persistent buffer / peak / CUDA-graph pool memory accounting
python benchmarks/bench_memory.py --device cuda
```

`bench_vs_cpu.py` simulates speculative decoding and reports tokens/step and
acceptance rate against `arctic_inference.SuffixDecodingCache` as the oracle.

## Testing

```bash
pytest -v
```

Every module is tested against the naive references in
`suffix_gpu/reference.py`; the suite is parametrized over `cpu` / `cuda` /
`mps`. `tests/test_verification.py` additionally fuzzes end-to-end
draft/verify equivalence against `arctic-inference` (skipped automatically if
the package is not installed).

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
