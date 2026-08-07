"""Fixed-iteration binary search over a suffix array.

All functions are pure tensor ops with static loop bounds, so they are
torch.compile friendly. Index tensors are int64 throughout.

Pattern convention: patterns are passed padded to a fixed width ``M``
together with per-row valid lengths, since different requests match
different-length suffixes.
"""

from __future__ import annotations

import math

import torch


def _cmp_pattern(
    sa: torch.Tensor,
    corpus: torch.Tensor,
    mid: torch.Tensor,
    pattern: torch.Tensor,
    pattern_len: torch.Tensor,
) -> torch.Tensor:
    """Lexicographic comparison of SA suffixes against padded patterns.

    Args:
        sa: suffix array [n] i64.
        corpus: token corpus [n] int.
        mid: SA indices to compare [B] i64.
        pattern: padded patterns [B, M] int.
        pattern_len: valid pattern lengths [B] i64.

    Returns:
        [B] int64 tensor with -1 if suffix < pattern, 0 if the suffix
        starts with the pattern, 1 if suffix > pattern.
    """
    n = sa.shape[0]
    m = pattern.shape[1]
    device = sa.device
    pos = sa[mid]
    offs = torch.arange(m, dtype=torch.int64, device=device)
    idx = pos.unsqueeze(1) + offs
    valid_pos = idx < n
    tok = torch.where(valid_pos, corpus[idx.clamp(0, n - 1)], -1)
    pat_valid = offs.unsqueeze(0) < pattern_len.unsqueeze(1)
    eq_ok = (tok == pattern) | ~pat_valid
    lcp_raw = torch.cumprod(eq_ok.to(torch.int64), dim=1).sum(dim=1)
    lcp = torch.minimum(lcp_raw, pattern_len)
    full = lcp >= pattern_len
    spos = (pos + lcp).clamp(0, n - 1)
    schar = torch.where(pos + lcp < n, corpus[spos], -1)
    pchar = pattern.gather(1, lcp.clamp(max=m - 1).unsqueeze(1)).squeeze(1)
    cmp_ = torch.where(
        schar < pchar, -1, torch.where(schar > pchar, 1, 0))
    return torch.where(full, 0, cmp_)


def search_interval(
    sa: torch.Tensor,
    corpus: torch.Tensor,
    pattern: torch.Tensor,
    pattern_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SA interval [start, end) of suffixes starting with each pattern.

    Args:
        sa: suffix array [n] i64.
        corpus: token corpus [n] int.
        pattern: padded patterns [B, M] int.
        pattern_len: valid pattern lengths [B] i64.

    Returns:
        (start [B] i64, end [B] i64); empty interval when start == end.
    """
    n = sa.shape[0]
    # Bounds may take n + 1 distinct values; resolve a range of size n + 1.
    iters = max(1, math.ceil(math.log2(n + 1)))
    b = pattern.shape[0]
    device = sa.device
    zero = torch.zeros(b, dtype=torch.int64, device=device)
    full = torch.full((b,), n, dtype=torch.int64, device=device)

    # Lower bound: first SA index with cmp >= 0.
    lo, hi = zero.clone(), full.clone()
    for _ in range(iters):
        # Converged rows can produce mid == n; clamp, the cmp is a no-op
        # there since where() leaves lo/hi unchanged.
        mid = ((lo + hi) // 2).clamp(max=n - 1)
        c = _cmp_pattern(sa, corpus, mid, pattern, pattern_len)
        lo = torch.where(c < 0, mid + 1, lo)
        hi = torch.where(c < 0, hi, mid)
    start = lo

    # Upper bound: first SA index with cmp > 0.
    lo, hi = zero.clone(), full.clone()
    for _ in range(iters):
        mid = ((lo + hi) // 2).clamp(max=n - 1)
        c = _cmp_pattern(sa, corpus, mid, pattern, pattern_len)
        lo = torch.where(c <= 0, mid + 1, lo)
        hi = torch.where(c <= 0, hi, mid)
    return start, lo


def _tail_pattern(
    query: torch.Tensor,
    query_len: torch.Tensor,
    length: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Padded pattern [B, width] holding query[-length:] per row."""
    q = query.shape[1]
    device = query.device
    b = query.shape[0]
    offs = torch.arange(width, dtype=torch.int64, device=device)
    jidx = query_len.unsqueeze(1) - length.unsqueeze(1) + offs.unsqueeze(0)
    valid = (offs.unsqueeze(0) < length.unsqueeze(1)) & (jidx >= 0)
    tok = torch.where(
        valid, query[:, :q].gather(1, jidx.clamp(0, q - 1)), 0)
    return tok


def longest_suffix_match(
    sa: torch.Tensor,
    corpus: torch.Tensor,
    query: torch.Tensor,
    query_len: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Longest suffix of each query that occurs in the corpus.

    The predicate "suffix of length L occurs" is monotone in L, so the
    length is found by a fixed-iteration binary search, each step of
    which is a full SA interval search.

    Args:
        sa: suffix array [n] i64.
        corpus: token corpus [n] int.
        query: padded query rows [B, Q] int.
        query_len: valid query lengths [B] i64.
        max_len: maximum match length to consider.

    Returns:
        (match_len [B] i64, start [B] i64, end [B] i64) where
        [start, end) is the SA interval at the matched length.
    """
    b = query.shape[0]
    device = sa.device
    zero = torch.zeros(b, dtype=torch.int64, device=device)
    hi = torch.full((b,), max_len, dtype=torch.int64, device=device)
    lo = zero.clone()
    iters = max(1, math.ceil(math.log2(max_len + 1)))
    for _ in range(iters):
        mid = (lo + hi + 1) // 2
        pattern = _tail_pattern(query, query_len, mid, max_len)
        plen = mid.expand(b).contiguous()
        s, e = search_interval(sa, corpus, pattern, plen)
        pred = (e > s) & (query_len >= mid)
        lo = torch.where(pred, mid, lo)
        hi = torch.where(pred, hi, mid - 1)
    pattern = _tail_pattern(query, query_len, lo, max_len)
    start, end = search_interval(sa, corpus, pattern, lo)
    return lo, start, end
