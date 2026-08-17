"""Dynamic-k: the accept-length EMA modulates the emission cap."""

from __future__ import annotations

import torch

from suffix_gpu.proposer import SuffixGPUDrafter

K = 8


def _drafter(device):
    return SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=8, max_occurrences=8,
        dynamic_k=True, ema_decay=0.5, dyn_k_scale=1.0,
        dyn_k_offset=0.0, dyn_k_min=1)


def _step(drafter, buf, nts, accepted_plus_one):
    sampled = torch.full((1, K + 1), -1, dtype=torch.int32,
                         device=buf.device)
    period = [1, 2, 3, 4]
    for j in range(accepted_plus_one):
        sampled[0, j] = period[(nts + j) % 4]
    counts = torch.tensor([nts], dtype=torch.int32, device=buf.device)
    draft, nv, new_counts = drafter.propose_with_update(
        counts, buf, sampled)
    return int(nv[0].item()), int(new_counts[0].item())


def test_dynamic_k_cap_follows_accept_ema(device):
    d = _drafter(device)
    buf = torch.zeros(1, 64, dtype=torch.int32, device=device)
    pattern = ([1, 2, 3, 4] * 5)[:20]
    buf[0, :20] = torch.tensor(pattern, dtype=torch.int32)

    # Step 1: 4 valid sampled ids => accepted 3 => EMA jumps to 3
    # (asymmetric: up is instant) => cap ceil(1.5*3) = 5, capped by
    # chain validity at 3.
    nv, nts = _step(d, buf, 20, 4)
    assert nts == 24
    assert nv == 3

    # Step 2: EMA stays 3 => cap 3.
    nv, nts = _step(d, buf, nts, 4)
    assert nts == 28
    assert nv == 3

    # Row recycled: EMA reset => standalone propose sees cap dyn_k_min.
    d.reset_rows([0])
    lens = torch.tensor([nts], dtype=torch.int32, device=device)
    _, nv_t = d.propose(lens, buf)
    assert nv_t[0].item() == 1
    # The next update_state relearns instantly (EMA jumps to 3).
    nv, nts = _step(d, buf, nts, 4)
    assert nv == 3


def test_dynamic_k_uncapped_before_history(device):
    # No update_state yet: EMA buffer absent, no cap applied.
    d = _drafter(device)
    buf = torch.zeros(1, 64, dtype=torch.int32, device=device)
    buf[0, :20] = torch.tensor(([1, 2, 3, 4] * 5)[:20],
                               dtype=torch.int32)
    lens = torch.tensor([20], dtype=torch.int32, device=device)
    draft, nv = d.propose(lens, buf)
    assert nv[0].item() > 3


def test_dynamic_k_off_is_uncapped(device):
    d = SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=8, max_occurrences=8,
        dynamic_k=False)
    buf = torch.zeros(1, 64, dtype=torch.int32, device=device)
    buf[0, :20] = torch.tensor(([1, 2, 3, 4] * 5)[:20],
                               dtype=torch.int32)
    nv, _ = _step_like(d, buf)
    assert nv > 3


def _step_like(drafter, buf):
    sampled = torch.full((1, K + 1), -1, dtype=torch.int32,
                         device=buf.device)
    sampled[0, 0] = 1
    counts = torch.tensor([20], dtype=torch.int32, device=buf.device)
    draft, nv, new_counts = drafter.propose_with_update(
        counts, buf, sampled)
    return int(nv[0].item()), int(new_counts[0].item())


def test_env_preset_legacy(device, monkeypatch):
    monkeypatch.setenv("SUFFIX_GPU_PRESET", "legacy")
    d = SuffixGPUDrafter(k=4, device=device, max_pattern_len=8,
                         max_occurrences=8, enable_global=True,
                         global_capacity=256, delta_capacity=64)
    assert d.vote_smoothing_alpha == 0.0
    assert d.local_mode == "backoff"
    assert d.merge_paths is False
    assert d.dynamic_k is False
    assert d.global_index.eviction == "fifo"
