"""Equivalence and semantics tests for the v2 (sort + run-length)
expand kernel: unweighted results must match the legacy pairwise
kernel bit-for-bit; weighted / smoothed votes must match the torch
reference semantics."""

from __future__ import annotations

import pytest
import torch

from suffix_gpu import triton_kernels
from suffix_gpu.expand import expand_chain


def _rand_case(rng, b, r, k, device):
    cont = torch.randint(0, 9, (b, r, k), dtype=torch.int32,
                         generator=rng, device="cpu")
    # Random -1 tails (monotone padding like real gathers) plus some
    # fully padded rows.
    tail = torch.randint(0, k + 1, (b, r), generator=rng)
    idx = torch.arange(k).view(1, 1, k)
    cont = torch.where(idx < tail.unsqueeze(2), cont, -1)
    occ = torch.randint(0, r + 1, (b,), dtype=torch.int64, generator=rng)
    cap = torch.randint(0, k + 1, (b,), dtype=torch.int64, generator=rng)
    return cont.to(device), occ.to(device), cap.to(device)


@pytest.mark.parametrize("r", [4, 32, 128, 256])
@pytest.mark.parametrize("minp", [0.0, 0.1])
def test_v2_matches_pairwise_kernel(r, minp):
    if not torch.cuda.is_available() or not triton_kernels.HAS_TRITON:
        pytest.skip("CUDA + Triton required")
    rng = torch.Generator().manual_seed(1234 + r)
    for trial in range(8):
        cont, occ, cap = _rand_case(rng, 33, r, 16, "cuda")
        c1, nv1, ne1, s1 = triton_kernels.expand_chain_pairwise(
            cont, occ, 16, minp, cap)
        c2, nv2, ne2, s2 = triton_kernels.expand_chain(
            cont, occ, 16, minp, cap)
        assert torch.equal(c1, c2), f"chain mismatch trial {trial}"
        assert torch.equal(nv1, nv2)
        assert torch.equal(ne1, ne2)
        assert torch.equal(s1, s2), f"score mismatch trial {trial}"


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_v2_alpha_matches_torch(alpha):
    if not torch.cuda.is_available() or not triton_kernels.HAS_TRITON:
        pytest.skip("CUDA + Triton required")
    rng = torch.Generator().manual_seed(99)
    for _ in range(4):
        cont, occ, cap = _rand_case(rng, 17, 32, 12, "cuda")
        cg, nvg, neg, sg = expand_chain(
            cont, occ, 12, min_token_prob=0.05, cap=cap, alpha=alpha)
        ct, nvt, net, st = expand_chain(
            cont.cpu(), occ.cpu(), 12, min_token_prob=0.05,
            cap=cap.cpu(), alpha=alpha)
        assert torch.equal(cg.cpu(), ct)
        assert torch.equal(nvg.cpu(), nvt)
        assert torch.equal(neg.cpu(), net)
        assert torch.allclose(sg.cpu(), st, rtol=1e-5, atol=1e-6)


def test_v2_weighted_matches_torch():
    if not torch.cuda.is_available() or not triton_kernels.HAS_TRITON:
        pytest.skip("CUDA + Triton required")
    rng = torch.Generator().manual_seed(7)
    for _ in range(4):
        cont, occ, cap = _rand_case(rng, 17, 64, 12, "cuda")
        w = (torch.rand(17, 64, generator=rng) + 0.05).to("cuda")
        cg, nvg, neg, sg = expand_chain(
            cont, occ, 12, min_token_prob=0.05, cap=cap, weights=w)
        ct, nvt, net, st = expand_chain(
            cont.cpu(), occ.cpu(), 12, min_token_prob=0.05,
            cap=cap.cpu(), weights=w.cpu())
        assert torch.equal(cg.cpu(), ct)
        assert torch.equal(nvg.cpu(), nvt)
        assert torch.equal(neg.cpu(), net)
        assert torch.allclose(sg.cpu(), st, rtol=1e-4, atol=1e-5)


def test_weighted_vote_semantics_cpu():
    # Two occurrences say token 7 with weight 1 each; one says token 3
    # with weight 5: the heavy occurrence must win depth 0.
    cont = torch.tensor([[[7, 1], [7, 1], [3, 2]]], dtype=torch.int32)
    occ = torch.tensor([3], dtype=torch.int64)
    w = torch.tensor([[1.0, 1.0, 5.0]])
    chain, nv, _, _ = expand_chain(cont, occ, 2, weights=w)
    assert chain[0, 0].item() == 3
    assert chain[0, 1].item() == 2
    assert nv.item() == 2


def test_alpha_damps_single_occurrence_cpu():
    # One occurrence, alpha=1: p = (1/2)^d, so with min_token_prob=0.2
    # the chain stops after two tokens instead of running to k.
    cont = torch.tensor([[[5, 6, 7, 8]]], dtype=torch.int32)
    occ = torch.tensor([1], dtype=torch.int64)
    chain, nv, _, _ = expand_chain(cont, occ, 4, min_token_prob=0.2,
                                   alpha=1.0)
    assert nv.item() == 2
    chain0, nv0, _, _ = expand_chain(cont, occ, 4, min_token_prob=0.2,
                                     alpha=0.0)
    assert nv0.item() == 4
