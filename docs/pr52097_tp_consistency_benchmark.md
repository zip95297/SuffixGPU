# PR #52097 multi-TP consistency and throughput report

Date: 2026-09-14

## Scope

This report validates the SuffixGPU multi-TP rebuild consistency fix against
vLLM PR [#52097](https://github.com/vllm-project/vllm/pull/52097), compares
Qwen3-8B throughput at TP1/TP2/TP4, and profiles the remaining rank skew without
adding a GPU synchronization.

Tested revisions:

- vLLM PR head: `4157d67257066a4677cd36344884236a16fa132e`
- vLLM consistency commit: `b078ac1142`
- SuffixGPU consistency commit: `020dd34`
- SuffixGPU version: editable `0.1.2`

## Consistency failure and fix

The first CPU-collective implementation reproduced a mismatch during the
second rebuild:

```text
rank0: active_epoch=1 pending_epoch=2 active_len=65642 docs=791
       fingerprint=8311849676532684854
rank1: active_epoch=1 pending_epoch=2 active_len=65642 docs=791
       fingerprint=8043133984068228136
```

Epoch, corpus length, and document count matched, but ordered document layout
did not. The run completed 197 requests before the mismatch and failed the
remaining 59. The cause was relying on Python dictionary insertion order for
request ingestion; equal request sets do not imply equal insertion histories
after request removal, slot swapping, and batch compaction.

The implemented protocol has two independent requirements:

1. vLLM sorts active request-ID/index pairs and finished IDs before ingestion,
   ensuring every TP rank builds the same ordered corpus.
2. While a rebuild is pending, ranks all-gather six `int64` metadata fields on
   the TP Gloo CPU group. They validate active epoch, pending epoch, active
   length, document count, layout fingerprint, and local readiness. All ranks
   commit only when all are ready; otherwise they keep matching against the old
   suffix array and try again at a later proposal.

The implementation does not call `torch.cuda.synchronize()`, wait on the
rebuild CUDA event, or communicate readiness with NCCL. The layout fingerprint
also does not copy GPU token content to the host.

## Benchmark configuration

- Model: Qwen3-8B exact local snapshot
- Spec-Bench requests: 256 per formal cell
- Concurrency: 1, 4, 8, 32, 64, 128, 256
- Tensor parallelism: 1, 2, 4
- Methods: NgramGPU, SuffixCPU, AsyncNoSpec, SuffixGPU
- `max_model_len=16384`
- `max_num_batched_tokens=8192`
- `max_num_seqs=320`
- GPU memory utilization: 0.9
- Prefix cache: disabled
- SuffixGPU: async scheduling, `k=5`, factor 0.5, minimum probability 0.1,
  maximum occurrences 128, backoff 8, cached requests 1000, tree depth 24
- NgramGPU and AsyncNoSpec: async scheduling
- SuffixCPU: synchronous scheduling

The four services were run concurrently on disjoint GPU sets to use all eight
RTX 4090 GPUs. Results therefore represent the requested practical deployment
layout and may include shared CPU/host contention. They are not a strict
single-service, low-noise A/B measurement.

## Output throughput

All values are output tokens/s.

| TP | Method | c1 | c4 | c8 | c32 | c64 | c128 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | NgramGPU | 60.0 | 224.6 | 416.1 | 1174.7 | 1488.6 | 1430.2 |
| 1 | SuffixCPU | 79.7 | 501.2 | 985.4 | 2375.8 | 2268.1 | 2258.8 |
| 1 | AsyncNoSpec | 59.5 | 223.3 | 428.4 | 1295.8 | 1446.8 | 1467.6 |
| 1 | SuffixGPU | 75.6 | 436.6 | 834.8 | 2182.9 | 2509.5 | 2416.0 |
| 2 | NgramGPU | 85.6 | 299.0 | 480.4 | 1083.4 | 1428.4 | 1661.4 |
| 2 | SuffixCPU | 111.9 | 574.2 | 971.6 | 1725.3 | 1848.2 | 1975.1 |
| 2 | AsyncNoSpec | 99.7 | 333.7 | 580.1 | 1246.4 | 1689.2 | 2007.2 |
| 2 | SuffixGPU | 106.3 | 503.5 | 854.1 | 1625.3 | 1849.7 | 2005.3 |
| 4 | NgramGPU | 99.5 | 338.0 | 576.1 | 1081.8 | 1397.0 | 1545.7 |
| 4 | SuffixCPU | 145.7 | 663.2 | 954.1 | 1591.0 | 1690.8 | 1731.6 |
| 4 | AsyncNoSpec | 145.2 | 464.9 | 756.6 | 1341.5 | 1678.7 | 1931.6 |
| 4 | SuffixGPU | 134.5 | 602.8 | 891.2 | 1534.0 | 1704.3 | 1771.3 |

SuffixGPU exceeded NgramGPU in every valid matrix cell. At c=64 the margins
were 68.6% for TP1, 29.5% for TP2, and 22.0% for TP4. At c=128 they were 68.9%,
20.7%, and 14.6%, respectively. The shrinking margin with higher TP confirms
that TP/host overhead and target-model scaling reduce the relative value of
draft acceleration, but the fixed implementation still remains ahead of
NgramGPU through c=128.

## Failures and correctness result

82 of the 84 formal cells completed successfully. TP2 and TP4 SuffixGPU at
c=256 failed when the rejection sampler cloned target logits and requested an
additional 656 MiB. The clients returned only partial output, so their reported
throughputs are invalid and excluded from the table. This is a peak-memory
failure after drafting, not a suffix-array rebuild mismatch.

No valid SuffixGPU run reported an epoch/layout mismatch, commit error,
collective hang, or engine failure attributable to the consistency protocol.
TP1 c=256 completed successfully. A valid TP2/TP4 c=256 measurement will require
additional rejection-sampler/KV-cache headroom, which would change the launch
configuration and should therefore be reported as a separate follow-up run.

## Nsight Systems profile

### Complete vLLM TP2 serving capture

The profiled service used the same SuffixGPU settings at c=64. It completed all
256 requests, produced 65,482 output tokens in 45.3098 s, and measured 1445.21
output tok/s. Two natural rebuilds were captured on the dedicated side stream:

| Rebuild | Rank 0 activity | Rank 1 activity | Start skew | Completion skew |
| ---: | --- | --- | ---: | ---: |
| 1 | 125.904248-125.928861 s | 125.904473-125.929679 s | 0.225 ms | 0.818 ms |
| 2 | 147.468685-147.479845 s | 147.468442-147.478094 s | 0.243 ms | 1.751 ms |

The completion skew is large enough for one rank to observe ready while another
is pending. The protocol handles this by using the old suffix array for that
proposal, with no GPU wait. No mismatch or hang occurred.

The profiled throughput must not be compared directly with the unprofiled
matrix as a consistency-overhead measurement: Nsight CUDA tracing perturbs the
process, and the matrix services ran concurrently. A clean overhead A/B would
alternate the original PR and the fixed commit on the same GPUs and request
order for at least five repetitions without profiler instrumentation.

### CPU collective capture

The real six-`int64` Gloo all-gather was profiled separately. Excluding the
initialization call, observed host durations were 0.34-1.35 ms under Nsight.
The older unprofiled one-value all-reduce reference measured approximately
196.5 us p50, 227.4 us p90, and 278.4 us p99, but is not an exact substitute for
the operation now used. Since the collective occurs only while a rebuild is
pending and the rebuild threshold is roughly 32K new tokens, its amortized cost
is small relative to the 20-second-scale rebuild interval.

## Artifacts

```text
/data/zip/AIInfra/benchmarks/results/pr52097_exact_tp_matrix_clean_20260913/
  THROUGHPUT_COMPLETE
  PROFILE_COMPLETE
  tp2_rebuild_collective.nsys-rep
  actual_vllm_profile/vllm_tp2_suffixgpu_c64.nsys-rep
  actual_vllm_profile/vllm_tp2_suffixgpu_c64.sqlite
  actual_vllm_profile/vllm_tp2_suffixgpu_c64.stats.txt
```

## Conclusion

The observed TP inconsistency required both canonical ingestion and coordinated
commit; either measure alone is incomplete. The fixed branch passed all
non-memory-limited TP1/TP2/TP4 tests, and the serving profile confirmed that
rank rebuild completion can differ by up to 1.75 ms under instrumentation. The
current old-SA plus CPU-metadata-poll design preserves correctness across that
window without introducing a GPU synchronization.
