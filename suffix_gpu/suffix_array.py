"""Suffix array construction via prefix doubling, in pure torch.

Runs eagerly (not under torch.compile): rebuilds are rare, background
operations, so plain host control flow is fine here. The per-step hot
paths (search, matching) live in sa_search/local_matcher instead.
"""

from __future__ import annotations

import math

import torch


def build_suffix_array(tokens: torch.Tensor) -> torch.Tensor:
    """Build the suffix array of a 1-D token sequence.

    Args:
        tokens: 1-D tensor of token ids (any int dtype, values >= 0).

    Returns:
        1-D int64 tensor ``sa`` where ``sa[i]`` is the start position of
        the i-th lexicographically smallest suffix. Ties among equal
        suffixes are impossible (distinct start positions imply distinct
        suffixes), so the result is unique up to correct ordering.
    """
    tokens = tokens.reshape(-1)
    n = tokens.shape[0]
    device = tokens.device
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if n == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)

    # rank[i] = rank of the length-1 prefix (the token itself).
    rank = tokens.to(torch.int64)
    base = int(n) + 1
    sa = torch.argsort(rank)
    k = 1
    rounds = max(1, math.ceil(math.log2(n)))
    for _ in range(rounds):
        # Packed key: (rank[i], rank[i + k] or sentinel). Ranks are >= 0;
        # the out-of-range second half maps to 0 via the +1 offset, which
        # sorts before every real rank value (>= 1). Requires
        # (n + 1)^2 < 2^63, i.e. n < ~3e9.
        second = torch.zeros(n, dtype=torch.int64, device=device)
        if k < n:
            second[:-k] = rank[k:] + 1
        key = rank * base + second
        sa = torch.argsort(key)
        sorted_key = key[sa]
        distinct = torch.ones(n, dtype=torch.int64, device=device)
        distinct[1:] = (sorted_key[1:] != sorted_key[:-1]).to(torch.int64)
        new_rank_sorted = torch.cumsum(distinct, 0) - 1
        new_rank = torch.empty_like(rank)
        new_rank[sa] = new_rank_sorted
        rank = new_rank
        k *= 2
    return sa
