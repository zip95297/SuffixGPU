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
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suffix_gpu.proposer import SuffixGPUDrafter  # noqa: E402

MB = 1024 * 1024


def measure(b: int, s: int, k: int, depth: int, cap_pow: int,
            delta_pow: int, device: torch.device) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    drafter = SuffixGPUDrafter(
        k=k, device=device, max_pattern_len=depth, enable_global=True,
        global_capacity=1 << cap_pow, delta_capacity=1 << delta_pow,
        max_spec_factor=2.0, min_token_prob=0.1,
        rebuild_stream=torch.cuda.Stream(device))
    after_drafter = torch.cuda.memory_allocated(device)

    buf = torch.randint(0, 32000, (b, s), dtype=torch.int32,
                        device=device)
    num_tok = torch.randint(s // 2, s, (b,), dtype=torch.int32,
                            device=device)
    sampled = torch.full((b, k + 1), 3, dtype=torch.int32, device=device)
    after_state = torch.cuda.memory_allocated(device)

    for _ in range(3):
        drafter.propose_with_update(num_tok.clone(), buf, sampled)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    drafter.propose_with_update(num_tok.clone(), buf, sampled)
    torch.cuda.synchronize(device)
    eager_peak = torch.cuda.max_memory_allocated(device)

    before_graph = torch.cuda.memory_allocated(device)
    g = torch.cuda.CUDAGraph()
    nt = num_tok.clone()
    with torch.cuda.graph(g):
        drafter.propose_with_update(nt, buf, sampled)
    g.replay()
    torch.cuda.synchronize(device)
    after_graph = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    print(f"B={b:4d} S={s:6d} k={k} depth={depth} "
          f"cap=2^{cap_pow} delta=2^{delta_pow}")
    print(f"  drafter persistent (corpus+SA+staging+delta): "
          f"{(after_drafter - base) / MB:8.1f} MB")
    print(f"  resident token buffer [B,S] int32:            "
          f"{(after_state - after_drafter) / MB:8.1f} MB")
    print(f"  propose transient peak (eager):               "
          f"{(eager_peak - after_state) / MB:8.1f} MB")
    print(f"  CUDA graph pool retained:                     "
          f"{(after_graph - before_graph) / MB:8.1f} MB")
    print(f"  total reserved by allocator:                  "
          f"{reserved / MB:8.1f} MB")
    del drafter, buf, num_tok, sampled, g
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"{torch.cuda.get_device_name(device)}")
    for b, s, cap_pow, delta_pow in ((32, 4096, 20, 15),
                                     (128, 16384, 20, 15),
                                     (256, 16384, 22, 16)):
        measure(b, s, 16, 24, cap_pow, delta_pow, device)


if __name__ == "__main__":
    main()
