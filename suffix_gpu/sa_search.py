"""Fixed-iteration binary search over a suffix array.

All functions are pure tensor ops with static loop bounds, so they are
torch.compile friendly. Index tensors are int64 throughout.

Pattern convention: patterns are passed padded to a fixed width ``M``
together with per-row valid lengths, since different requests match
different-length suffixes.

Hot-path structure: search_interval resolves the lower and upper bound
in one merged binary-search loop (2B rows), and longest_suffix_match
searches all candidate lengths 1..max_len as one batch instead of an
outer per-length binary search, so the SA is walked exactly once.
"""

from __future__ import annotations

import math

import torch

from suffix_gpu import triton_kernels


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

    Lower bound (first cmp >= 0) and upper bound (first cmp > 0) are
    resolved together in a single fixed-iteration loop over 2B rows.

    Args:
        sa: suffix array [n] i64.
        corpus: token corpus [n] int.
        pattern: padded patterns [B, M] int.
        pattern_len: valid pattern lengths [B] i64.

    Returns:
        (start [B] i64, end [B] i64); empty interval when start == end.
    """
    if triton_kernels.available(sa, corpus, pattern):
        return triton_kernels.sa_search(sa, corpus, pattern, pattern_len)
    n = sa.shape[0]
    # Bounds may take n + 1 distinct values; resolve a range of size n + 1.
    iters = max(1, math.ceil(math.log2(n + 1)))
    b = pattern.shape[0]
    device = sa.device
    pat2 = torch.cat([pattern, pattern], dim=0)
    plen2 = torch.cat([pattern_len, pattern_len], dim=0)
    is_upper = torch.zeros(2 * b, dtype=torch.bool, device=device)
    is_upper[b:] = True
    lo = torch.zeros(2 * b, dtype=torch.int64, device=device)
    hi = torch.full((2 * b,), n, dtype=torch.int64, device=device)
    for _ in range(iters):
        # Converged rows can produce mid == n; clamp, the cmp is a no-op
        # there since where() leaves lo/hi unchanged.
        mid = ((lo + hi) // 2).clamp(max=n - 1)
        c = _cmp_pattern(sa, corpus, mid, pat2, plen2)
        go_right = (c < 0) | (is_upper & (c == 0))
        lo = torch.where(go_right, mid + 1, lo)
        hi = torch.where(go_right, hi, mid)
    return lo[:b], lo[b:]


def longest_suffix_match(
    sa: torch.Tensor,
    corpus: torch.Tensor,
    query: torch.Tensor,
    query_len: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Longest suffix of each query that occurs in the corpus.

    All candidate lengths 1..max_len are searched as one flattened
    batch (B * max_len rows) in a single interval search, then the
    longest non-empty interval is selected per query.

    Args:
        sa: suffix array [n] i64.
        corpus: token corpus [n] int.
        query: padded query rows [B, Q] int.
        query_len: valid query lengths [B] i64.
        max_len: maximum match length to consider.

    Returns:
        (match_len [B] i64, start [B] i64, end [B] i64) where
        [start, end) is the SA interval at the matched length; (0, 0)
        when there is no match.
    """
    b, q = query.shape
    m = max_len
    device = sa.device
    lengths = torch.arange(1, m + 1, dtype=torch.int64, device=device)
    offs = torch.arange(m, dtype=torch.int64, device=device)
    # pattern[b, l, j] = query[query_len - (l+1) + j], zero-padded.
    idx = (query_len.view(b, 1, 1) - lengths.view(1, m, 1)
           + offs.view(1, 1, m))
    valid = (offs.view(1, 1, m) < lengths.view(1, m, 1)) & (idx >= 0)
    tok = query.gather(1, idx.clamp(0, q - 1).reshape(b, m * m)
                       ).reshape(b, m, m)
    pat = torch.where(valid, tok, torch.zeros_like(tok))
    start, end = search_interval(
        sa, corpus, pat.reshape(b * m, m),
        lengths.view(1, m).expand(b, m).reshape(-1))
    start = start.view(b, m)
    end = end.view(b, m)
    found = (end > start) & (lengths.view(1, m) <= query_len.view(b, 1))
    best = (found.to(torch.int64) * lengths.view(1, m)).max(dim=1).values
    pick = (best - 1).clamp(min=0).unsqueeze(1)
    s_best = start.gather(1, pick).squeeze(1)
    e_best = end.gather(1, pick).squeeze(1)
    zero = torch.zeros_like(s_best)
    return (best, torch.where(best > 0, s_best, zero),
            torch.where(best > 0, e_best, zero))
