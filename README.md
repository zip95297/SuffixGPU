# SuffixGPU

GPU-resident suffix decoding drafter for LLM speculative decoding.

Moves the SuffixDecoding (arXiv:2411.04975) draft path fully onto the
accelerator device: variable-length suffix matching with frequency ranking,
plus a cross-request global memory backed by a suffix array. Mirrors the
vLLM `ngram_gpu` proposer contract (device tensors in/out, no per-step host
sync) so it composes with async scheduling.

## Layout

- `suffix_gpu/suffix_array.py` — suffix array construction (prefix doubling)
- `suffix_gpu/sa_search.py` — fixed-iteration binary search over the SA
- `suffix_gpu/local_matcher.py` — per-request brute-force variable-length matcher
- `suffix_gpu/expand.py` — frequency-ranked chain expansion
- `suffix_gpu/global_index.py` — corpus ring + delta buffer + double-buffered SA
- `suffix_gpu/proposer.py` — `SuffixGPUDrafter` orchestrator
- `suffix_gpu/reference.py` — naive references used by tests

## Development

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
pytest -v
```
