from __future__ import annotations

import pytest
import torch

from suffix_gpu.reference import (naive_longest_suffix_match,
                                  naive_occurrences, naive_search_interval,
                                  naive_suffix_array)
from suffix_gpu.sa_search import longest_suffix_match, search_interval
from suffix_gpu.suffix_array import build_suffix_array


def _make(tokens: list[int], device):
    t = torch.tensor(tokens, dtype=torch.int32, device=device)
    return t, build_suffix_array(t)


def _search(corpus, sa, pattern: list[int], device):
    pat = torch.tensor([pattern], dtype=torch.int32, device=device)
    plen = torch.tensor([len(pattern)], dtype=torch.int64, device=device)
    s, e = search_interval(sa, corpus, pat, plen)
    return s.item(), e.item()


CORPUS = [1, 2, 3, 1, 2, 4, 1, 2, 3, 9, 1, 2, 3, 1, 5]


@pytest.mark.parametrize(
    "pattern", [[1], [1, 2], [1, 2, 3], [1, 2, 4], [9], [5], [1, 2, 3, 1],
                [7], [1, 2, 3, 1, 2, 3], [3, 1], [2, 4, 1]])
def test_search_interval(pattern, device):
    corpus, sa = _make(CORPUS, device)
    s, e = _search(corpus, sa, pattern, device)
    exp = naive_search_interval(naive_suffix_array(CORPUS), CORPUS, pattern)
    assert (s, e) == exp
    assert sorted(sa[s:e].tolist()) == naive_occurrences(CORPUS, pattern)


@pytest.mark.parametrize("seed", range(6))
def test_search_interval_random(seed, device):
    g = torch.Generator().manual_seed(seed)
    n = 128
    tokens = torch.randint(0, 4, (n,), generator=g).tolist()
    corpus, sa = _make(tokens, device)
    ref_sa = naive_suffix_array(tokens)
    for _ in range(20):
        m = int(torch.randint(1, 8, (1,), generator=g).item())
        start = int(torch.randint(0, n - m + 1, (1,), generator=g).item())
        pattern = tokens[start:start + m]
        s, e = _search(corpus, sa, pattern, device)
        assert (s, e) == naive_search_interval(ref_sa, tokens, pattern)


@pytest.mark.parametrize(
    "query,max_len",
    [
        ([1, 2, 3, 1, 2], 8),
        ([9, 9, 1, 2, 3], 8),
        ([4, 4, 4], 4),
        ([1, 2, 3, 1, 5, 1, 2, 3], 10),
    ],
)
def test_longest_suffix_match(query, max_len, device):
    corpus, sa = _make(CORPUS, device)
    q = torch.tensor([query], dtype=torch.int32, device=device)
    qlen = torch.tensor([len(query)], dtype=torch.int64, device=device)
    match_len, s, e = longest_suffix_match(sa, corpus, q, qlen, max_len)
    exp = naive_longest_suffix_match(CORPUS, query, max_len)
    assert match_len.item() == exp
    if exp > 0:
        assert sorted(sa[s:e].tolist()) == naive_occurrences(
            CORPUS, query[-exp:])


def test_longest_match_batch(device):
    corpus, sa = _make(CORPUS, device)
    queries = [[1, 2, 3], [5, 5, 5], [9, 1, 2]]
    width = 4
    q = torch.zeros(len(queries), width, dtype=torch.int32, device=device)
    qlen = torch.zeros(len(queries), dtype=torch.int64, device=device)
    for i, seq in enumerate(queries):
        q[i, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
        qlen[i] = len(seq)
    match_len, _, _ = longest_suffix_match(sa, corpus, q, qlen, width)
    exp = [naive_longest_suffix_match(CORPUS, seq, width) for seq in queries]
    assert match_len.tolist() == exp
