"""Joint local+global voting (merge_paths): the union of both
occurrence sets can extend the chain past where either path alone is
cut by min_token_prob."""

from __future__ import annotations

import torch

from suffix_gpu.proposer import SuffixGPUDrafter

K = 3
P = 8
R = 8


def _drafter(device, merge_paths):
    return SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=P, max_occurrences=R,
        enable_global=True, global_capacity=1 << 10,
        delta_capacity=1 << 8, rebuild_threshold=1 << 9,
        min_token_prob=0.5, vote_smoothing_alpha=1.0,
        local_mode="soft", merge_paths=merge_paths)


def _setup(drafter, device):
    # Global doc: two occurrences of "5 -> 7 1".
    doc = [5, 7, 1, 0, 5, 7, 1]
    buf = torch.zeros(2, 24, dtype=torch.int32, device=device)
    buf[0, :len(doc)] = torch.tensor(doc, dtype=torch.int32)
    drafter.harvest_finished([0], [len(doc)], buf)
    # Local ctx: two more occurrences of "5 -> 7 1", tail "5".
    ctx = [5, 7, 1, 0, 5, 7, 1, 0, 9, 5]
    buf[1, :len(ctx)] = torch.tensor(ctx, dtype=torch.int32)
    lens = torch.tensor([len(doc), len(ctx)], dtype=torch.int32,
                        device=device)
    return buf, lens


def test_joint_vote_extends_chain(device):
    # With alpha=1 and min_token_prob=0.5: each path alone has two
    # occurrences (p0 = 2/3, p1 = 4/9 < 0.5 -> chain cut after one
    # token), while the union has four (p0 = 4/5, p1 = 16/25 >= 0.5).
    on = _drafter(device, merge_paths=True)
    buf, lens = _setup(on, device)
    draft, nv = on.propose(lens, buf)
    assert nv[1].item() == 2
    assert draft[1].tolist()[:2] == [7, 1]

    off = _drafter(device, merge_paths=False)
    buf, lens = _setup(off, device)
    draft, nv = off.propose(lens, buf)
    assert nv[1].item() == 1
    assert draft[1].tolist()[:1] == [7]


def test_joint_never_engages_on_length_mismatch(device):
    # Local match is length 2 ("8 5"), global only knows length-1 "5":
    # outputs must be identical with and without merge_paths.
    doc = [5, 3, 5, 3]
    ctx = [8, 5, 6, 1, 8, 5, 6, 2, 8, 5]
    outs = []
    for merge in (True, False):
        d = _drafter(device, merge_paths=merge)
        buf = torch.zeros(2, 24, dtype=torch.int32, device=device)
        buf[0, :len(doc)] = torch.tensor(doc, dtype=torch.int32)
        d.harvest_finished([0], [len(doc)], buf)
        buf[1, :len(ctx)] = torch.tensor(ctx, dtype=torch.int32)
        lens = torch.tensor([len(doc), len(ctx)], dtype=torch.int32,
                            device=device)
        draft, nv = d.propose(lens, buf)
        outs.append((draft[1].tolist(), nv[1].item()))
    assert outs[0] == outs[1]
