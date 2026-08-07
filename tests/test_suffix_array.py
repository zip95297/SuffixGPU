from __future__ import annotations

import pytest
import torch

from suffix_gpu.reference import naive_suffix_array
from suffix_gpu.suffix_array import build_suffix_array


def _assert_valid_sa(tokens: list[int], sa: torch.Tensor) -> None:
    n = len(tokens)
    assert sa.shape == (n,)
    assert sorted(sa.tolist()) == list(range(n))
    # Suffixes must be lexicographically non-decreasing.
    for a, b in zip(sa.tolist(), sa.tolist()[1:]):
        assert tuple(tokens[a:]) < tuple(tokens[b:])


@pytest.mark.parametrize(
    "tokens",
    [
        [1, 2, 3, 1, 2, 3, 1, 2],
        [5, 5, 5, 5, 5],
        [0],
        [2, 1],
        [1, 2, 1, 2, 1, 2, 1],
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    ],
)
def test_build_suffix_array_structured(tokens, device):
    sa = build_suffix_array(torch.tensor(tokens, dtype=torch.int32,
                                         device=device))
    _assert_valid_sa(tokens, sa.cpu())
    assert sa.tolist() == naive_suffix_array(tokens)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("n,vocab", [(31, 3), (64, 8), (200, 16), (7, 2)])
def test_build_suffix_array_random(seed, n, vocab, device):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (n,), generator=g)
    sa = build_suffix_array(tokens.to(device).to(torch.int32))
    _assert_valid_sa(tokens.tolist(), sa.cpu())


def test_empty_and_single(device):
    assert build_suffix_array(
        torch.empty(0, dtype=torch.int32, device=device)).shape == (0,)
    assert build_suffix_array(
        torch.tensor([7], dtype=torch.int32, device=device)).tolist() == [0]
