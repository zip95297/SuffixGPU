"""Soft weighted-backoff local matcher vs the naive host oracle."""

from __future__ import annotations

import random

import pytest
import torch

from suffix_gpu.local_matcher import LocalMatchKernel
from suffix_gpu.proposer import SuffixGPUDrafter
from suffix_gpu.reference import naive_soft_local_match

K = 6
P = 8
R = 16


def _kernel(device, **kw):
    args = dict(k=K, max_pattern_len=P, max_occurrences=R,
                local_mode="soft", soft_lambda=0.5,
                vote_smoothing_alpha=0.0)
    args.update(kw)
    return LocalMatchKernel(**args).to(device)


def _run(kernel, seq, device):
    buf = torch.zeros(1, max(len(seq) + 2, 16), dtype=torch.int32,
                      device=device)
    buf[0, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
    lens = torch.tensor([len(seq)], dtype=torch.int32, device=device)
    mask = torch.ones(1, dtype=torch.bool, device=device)
    draft, nv, mlen, occ, score = kernel(lens, buf, mask)
    return draft[0].tolist()[: nv[0].item()], mlen[0].item()


@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("minp", [0.0, 0.2])
def test_soft_matches_naive_fuzz(device, alpha, minp):
    g = random.Random(1000)
    kernel = _kernel(device, vote_smoothing_alpha=alpha,
                     min_token_prob=minp)
    for trial in range(30):
        n = g.randint(2, 40)
        seq = [g.randint(0, 5) for _ in range(n)]
        got_chain, got_len = _run(kernel, seq, device)
        exp_chain, exp_len = naive_soft_local_match(
            seq, K, P, max_occurrences=R, min_token_prob=minp,
            soft_lambda=0.5, alpha=alpha)
        assert got_len == exp_len, f"trial {trial}: {seq}"
        assert got_chain == exp_chain, (
            f"trial {trial}: {seq} -> {got_chain} != {exp_chain}")


def test_soft_spec_cap(device):
    kernel = _kernel(device, max_spec_factor=1.0)
    seq = [1, 2, 7, 8, 9, 1, 2]
    chain, mlen = _run(kernel, seq, device)
    exp_chain, exp_len = naive_soft_local_match(
        seq, K, P, max_occurrences=R, max_spec_factor=1.0,
        soft_lambda=0.5)
    assert mlen == exp_len
    assert chain == exp_chain


def test_soft_weights_prefer_long_match(device):
    # Sequence where the longest match continues with 9 once, while
    # many short (length-1) matches continue with 3. With lambda=0.5
    # and 4 short sites the short votes win (4 * 0.5 = 2 > 1); with a
    # sharp lambda the long match dominates.
    seq = [7, 5, 3, 1, 5, 3, 2, 5, 3, 4, 5, 3, 8, 2, 9, 8, 2]
    sharp = _kernel(device, soft_lambda=0.01)
    chain_sharp, _ = _run(sharp, seq, device)
    flat = _kernel(device, soft_lambda=1.0)
    chain_flat, _ = _run(flat, seq, device)
    exp_sharp, _ = naive_soft_local_match(seq, K, P, max_occurrences=R,
                                          soft_lambda=0.01)
    exp_flat, _ = naive_soft_local_match(seq, K, P, max_occurrences=R,
                                         soft_lambda=1.0)
    assert chain_sharp == exp_sharp
    assert chain_flat == exp_flat


def test_soft_drafter_with_global_smoke(device):
    drafter = SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=P, max_occurrences=R,
        enable_global=True, global_capacity=1 << 12,
        delta_capacity=1 << 8, rebuild_threshold=1 << 6,
        local_mode="soft")
    doc = [4, 5, 6, 7, 4, 5, 6, 7, 4, 5]
    buf = torch.zeros(2, 32, dtype=torch.int32, device=device)
    buf[0, :len(doc)] = torch.tensor(doc, dtype=torch.int32)
    drafter.harvest_finished([0], [len(doc)], buf)
    cur = [9, 9, 4, 5, 6]
    buf[1, :len(cur)] = torch.tensor(cur, dtype=torch.int32)
    lens = torch.tensor([len(doc), len(cur)], dtype=torch.int32,
                        device=device)
    draft, nv = drafter.propose(lens, buf)
    assert draft.shape == (2, K)
    assert nv.shape == (2,)
    got = draft[1].tolist()[: nv[1].item()]
    assert got[:1] == [7]  # continuation of "4 5 6" from the doc
