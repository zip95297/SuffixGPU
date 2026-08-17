from __future__ import annotations

import pytest
import torch

from suffix_gpu.local_matcher import LocalMatchKernel
from suffix_gpu.proposer import SuffixGPUDrafter
from suffix_gpu.reference import naive_local_match

K = 4
MAX_P = 8
R = 16


def _run(kernel, seqs, device, mask=None):
    width = max(len(s) for s in seqs)
    b = len(seqs)
    buf = torch.zeros(b, width, dtype=torch.int32, device=device)
    lens = torch.zeros(b, dtype=torch.int32, device=device)
    for i, s in enumerate(seqs):
        buf[i, :len(s)] = torch.tensor(s, dtype=torch.int32)
        lens[i] = len(s)
    if mask is None:
        mask = torch.ones(b, dtype=torch.bool, device=device)
    draft, nv, mlen, occ, score = kernel(lens, buf, mask)
    return draft.cpu(), nv.cpu(), mlen.cpu()


CASES = [
    [1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3],
    [7, 7, 7, 7, 7, 7],
    [1, 2, 1, 2, 1, 2, 9, 1, 2],
    [5, 6, 7, 8, 5, 6, 7, 8, 5, 6],
    [1],
    [1, 2],
    [3, 1, 4, 1, 5, 9, 2, 6],
    [2, 2, 3, 2, 2, 4, 2, 2, 3],
]


@pytest.mark.parametrize("seq", CASES)
def test_local_match_vs_reference(seq, device):
    kernel = LocalMatchKernel(k=K, max_pattern_len=MAX_P, max_occurrences=R)
    draft, nv, mlen = _run(kernel, [seq], device)
    exp_chain, exp_len = naive_local_match(seq, K, MAX_P, 1, R)
    assert mlen.item() == exp_len
    got = draft[0].tolist()[:nv[0].item()]
    assert got == exp_chain[:len(got)]
    assert nv.item() == len(exp_chain)


def test_local_match_batch(device):
    kernel = LocalMatchKernel(k=K, max_pattern_len=MAX_P, max_occurrences=R)
    draft, nv, mlen = _run(kernel, CASES, device)
    for i, seq in enumerate(CASES):
        exp_chain, exp_len = naive_local_match(seq, K, MAX_P, 1, R)
        assert mlen[i].item() == exp_len, f"row {i}"
        got = draft[i].tolist()[:nv[i].item()]
        assert got == exp_chain[:len(got)], f"row {i}"


def test_mask_disables_rows(device):
    kernel = LocalMatchKernel(k=K, max_pattern_len=MAX_P, max_occurrences=R)
    seq = [1, 2, 3, 1, 2, 3, 4, 1, 2, 3]
    mask = torch.tensor([False, True], dtype=torch.bool, device=device)
    draft, nv, _ = _run(kernel, [seq, seq], device, mask=mask)
    exp_chain, _ = naive_local_match(seq, K, MAX_P, 1, R)
    assert nv[0].item() == 0
    assert (draft[0] == -1).all()
    assert draft[1].tolist()[:nv[1].item()] == exp_chain


def test_random_vs_reference(device):
    kernel = LocalMatchKernel(k=K, max_pattern_len=6, max_occurrences=8)
    g = torch.Generator().manual_seed(0)
    seqs = []
    for _ in range(16):
        n = int(torch.randint(8, 48, (1,), generator=g).item())
        seq = torch.randint(0, 4, (n,), generator=g).tolist()
        seqs.append(seq)
    draft, nv, mlen = _run(kernel, seqs, device)
    for i, seq in enumerate(seqs):
        exp_chain, exp_len = naive_local_match(seq, K, 6, 1, 8)
        assert mlen[i].item() == exp_len, f"row {i}: {seq}"
        got = draft[i].tolist()[:nv[i].item()]
        assert got == exp_chain[:len(got)], f"row {i}: {seq}"


def test_proposer_contract(device):
    drafter = SuffixGPUDrafter(k=K, device=device, max_pattern_len=MAX_P,
                               max_occurrences=R,
                               vote_smoothing_alpha=0.0,
                               local_mode="backoff",
                               merge_paths=False)
    seq = [1, 2, 3, 4, 1, 2, 3, 4, 9, 1, 2, 3, 4]
    buf = torch.tensor([seq], dtype=torch.int32, device=device)
    lens = torch.tensor([len(seq)], dtype=torch.int32, device=device)
    draft, nv = drafter.propose(lens, buf)
    exp_chain, _ = naive_local_match(seq, K, MAX_P, 1, R)
    assert draft.shape == (1, K)
    assert draft.dtype == torch.int32
    assert draft[0].tolist()[:nv[0].item()] == exp_chain


@pytest.mark.parametrize("compile_fn", [torch.compile], ids=["compiled"])
def test_local_match_torch_compile_cpu(compile_fn):
    kernel = LocalMatchKernel(k=K, max_pattern_len=4, max_occurrences=8)
    compiled = compile_fn(kernel, dynamic=False)
    seq = [1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2]
    draft, nv, mlen = _run(compiled, [seq], torch.device("cpu"))
    exp_chain, exp_len = naive_local_match(seq, K, 4, 1, 8)
    assert mlen.item() == exp_len
    assert draft[0].tolist()[:nv[0].item()] == exp_chain
