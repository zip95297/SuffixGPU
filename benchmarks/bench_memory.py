"""Measure GPU memory overhead of the SuffixGPU drafter.

Reports, for a given config: persistent buffers (drafter state +
resident token buffer), transient propose peak (eager), and CUDA-graph
pool retention.

Run:
  LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.580.105.08 \
  ../vllm/.venv/bin/python benchmarks/bench_memory.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suffix_gpu.proposer import SuffixGPUDrafter  # noqa: E402

MB = 1024 * 1024


def bucket_sizes(max_batch: int) -> list[int]:
    sizes = []
    batch = 1
    while batch < max_batch:
        sizes.append(batch)
        batch *= 2
    sizes.append(max_batch)
    return sizes


def capture_retained_memory(
    drafter: SuffixGPUDrafter,
    num_tok: torch.Tensor,
    buf: torch.Tensor,
    sampled: torch.Tensor,
    buckets: list[int],
    *,
    shared_pool: bool,
    largest_first: bool,
    device: torch.device,
) -> float:
    for batch in buckets:
        drafter.propose_with_update(num_tok[:batch], buf[:batch], sampled[:batch])
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info(device)[0]

    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    graphs: list[torch.cuda.CUDAGraph] = []
    outputs: list[tuple[torch.Tensor, ...]] = []
    capture_order = reversed(buckets) if largest_first else buckets
    for batch in capture_order:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outputs.append(
                drafter.propose_with_update(
                    num_tok[:batch], buf[:batch], sampled[:batch]
                )
            )
        graphs.append(graph)

    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    retained = free_before - torch.cuda.mem_get_info(device)[0]
    for graph in graphs:
        graph.reset()
    graphs.clear()
    outputs.clear()
    del pool
    gc.collect()
    torch.cuda.empty_cache()
    return retained / MB


def measure(
    b: int,
    s: int,
    k: int,
    depth: int,
    cap_pow: int,
    delta_pow: int,
    device: torch.device,
) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    drafter = SuffixGPUDrafter(
        k=k,
        device=device,
        max_pattern_len=depth,
        enable_global=True,
        global_capacity=1 << cap_pow,
        delta_capacity=1 << delta_pow,
        max_spec_factor=2.0,
        min_token_prob=0.1,
        rebuild_stream=torch.cuda.Stream(device),
    )
    after_drafter = torch.cuda.memory_allocated(device)

    buf = torch.randint(0, 32000, (b, s), dtype=torch.int32, device=device)
    num_tok = torch.randint(s // 2, s, (b,), dtype=torch.int32, device=device)
    sampled = torch.full((b, k + 1), 3, dtype=torch.int32, device=device)
    after_state = torch.cuda.memory_allocated(device)

    for _ in range(3):
        drafter.propose_with_update(num_tok.clone(), buf, sampled)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    drafter.propose_with_update(num_tok.clone(), buf, sampled)
    torch.cuda.synchronize(device)
    eager_peak = torch.cuda.max_memory_allocated(device)

    buckets = bucket_sizes(b)
    single_graph = capture_retained_memory(
        drafter,
        num_tok,
        buf,
        sampled,
        [b],
        shared_pool=False,
        largest_first=False,
        device=device,
    )
    independent_ascending = capture_retained_memory(
        drafter,
        num_tok,
        buf,
        sampled,
        buckets,
        shared_pool=False,
        largest_first=False,
        device=device,
    )
    shared_ascending = capture_retained_memory(
        drafter,
        num_tok,
        buf,
        sampled,
        buckets,
        shared_pool=True,
        largest_first=False,
        device=device,
    )
    shared_descending = capture_retained_memory(
        drafter,
        num_tok,
        buf,
        sampled,
        buckets,
        shared_pool=True,
        largest_first=True,
        device=device,
    )
    reserved = torch.cuda.memory_reserved(device)

    print(f"B={b:4d} S={s:6d} k={k} depth={depth} cap=2^{cap_pow} delta=2^{delta_pow}")
    print(
        "  drafter persistent (corpus+SA+staging+delta): "
        f"{(after_drafter - base) / MB:8.1f} MB"
    )
    print(
        "  resident token buffer [B,S] int32:            "
        f"{(after_state - after_drafter) / MB:8.1f} MB"
    )
    print(
        "  propose transient peak (eager):               "
        f"{(eager_peak - after_state) / MB:8.1f} MB"
    )
    print(f"  CUDA graph retained (single max bucket):       {single_graph:8.1f} MB")
    print(
        "  CUDA graph retained (independent, ascending): "
        f"{independent_ascending:8.1f} MB"
    )
    print(f"  CUDA graph retained (shared, ascending):      {shared_ascending:8.1f} MB")
    print(
        f"  CUDA graph retained (shared, descending):     {shared_descending:8.1f} MB"
    )
    print(f"  total reserved by allocator after cleanup:     {reserved / MB:8.1f} MB")
    del drafter, buf, num_tok, sampled
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"{torch.cuda.get_device_name(device)}")
    for b, s, cap_pow, delta_pow in (
        (32, 4096, 20, 15),
        (128, 16384, 20, 15),
        (256, 16384, 22, 16),
    ):
        measure(b, s, 16, 24, cap_pow, delta_pow, device)


if __name__ == "__main__":
    main()
