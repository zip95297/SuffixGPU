from __future__ import annotations

import torch

from suffix_gpu.expand import _majority_token, expand_chain


def test_majority_basic(device):
    values = torch.tensor([[5, 5, 7, 7, 7],
                           [1, 2, 1, -1, -1]], dtype=torch.int32,
                          device=device)
    active = torch.tensor([[True] * 5, [True, True, True, False, False]],
                          device=device)
    out = _majority_token(values, active, -1)
    assert out.tolist() == [7, 1]


def test_majority_run_lengths(device):
    # Run of three 5s beats a single 7; guards the run-length math.
    values = torch.tensor([[5, 7, 5, 5, -1]], dtype=torch.int32,
                          device=device)
    active = torch.tensor([[True] * 5], device=device)
    assert _majority_token(values, active, -1).tolist() == [5]


def test_majority_tie_smallest_token(device):
    values = torch.tensor([[9, 9, 3, 3, -1]], dtype=torch.int32,
                          device=device)
    active = torch.tensor([[True] * 5], device=device)
    assert _majority_token(values, active, -1).tolist() == [3]


def test_majority_empty_row(device):
    values = torch.zeros(1, 4, dtype=torch.int32, device=device)
    active = torch.zeros(1, 4, dtype=torch.bool, device=device)
    assert _majority_token(values, active, -1).tolist() == [-1]


def test_expand_chain_exact(device):
    # occ0: [4, 5, 6], occ1: [4, 7, 8]. Depth 0: unanimous 4. Depth 1:
    # tie 5 vs 7 -> smallest (5), filtering out occ1. Depth 2: only
    # occ0 remains -> 6.
    cont = torch.tensor([[[4, 5, 6], [4, 7, 8]]], dtype=torch.int32,
                        device=device)
    num_occ = torch.tensor([2], dtype=torch.int64, device=device)
    chain, nv = expand_chain(cont, num_occ, 3)
    assert chain.tolist() == [[4, 5, 6]]
    assert nv.tolist() == [3]


def test_expand_chain_prefix_filtering(device):
    # occ0: [4, 5], occ1: [3, 9] -> depth0 vote: 3 vs 4 tie -> 3,
    # occ0 filtered out; depth1 only occ1 -> 9.
    cont = torch.tensor([[[4, 5, -1], [3, 9, 2]]], dtype=torch.int32,
                        device=device)
    num_occ = torch.tensor([2], dtype=torch.int64, device=device)
    chain, nv = expand_chain(cont, num_occ, 3)
    assert chain.tolist() == [[3, 9, 2]]
    assert nv.tolist() == [3]


def test_expand_chain_truncation(device):
    cont = torch.tensor([[[4, -1, -1], [4, 1, -1]]], dtype=torch.int32,
                        device=device)
    num_occ = torch.tensor([2], dtype=torch.int64, device=device)
    chain, nv = expand_chain(cont, num_occ, 3)
    assert chain.tolist() == [[4, 1, -1]]
    assert nv.tolist() == [2]


def test_expand_chain_zero_occ(device):
    cont = torch.full((2, 3, 4), -1, dtype=torch.int32, device=device)
    num_occ = torch.tensor([0, 0], dtype=torch.int64, device=device)
    chain, nv = expand_chain(cont, num_occ, 4)
    assert (chain == -1).all()
    assert nv.tolist() == [0, 0]
