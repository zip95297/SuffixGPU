"""Naive reference implementations used as test oracles.

All functions operate on plain Python lists / host values. They are
intentionally simple and unoptimized.
"""

from __future__ import annotations

from collections.abc import Sequence


def suffix_key(tokens: Sequence[int], i: int) -> tuple[int, ...]:
    return tuple(tokens[i:])


def naive_suffix_array(tokens: Sequence[int]) -> list[int]:
    """Suffix array with index as tie-breaker (canonical order)."""
    n = len(tokens)
    return sorted(range(n), key=lambda i: suffix_key(tokens, i))


def _suffix_starts_with(tokens: Sequence[int], pos: int,
                        pattern: Sequence[int]) -> bool:
    if pos + len(pattern) > len(tokens):
        return False
    return all(tokens[pos + j] == t for j, t in enumerate(pattern))


def _suffix_cmp(tokens: Sequence[int], pos: int,
                pattern: Sequence[int]) -> int:
    """-1 if suffix < pattern, 0 if suffix starts with pattern, else 1."""
    n = len(tokens)
    for j, t in enumerate(pattern):
        if pos + j >= n:
            return -1
        if tokens[pos + j] < t:
            return -1
        if tokens[pos + j] > t:
            return 1
    return 0


def naive_search_interval(sa: Sequence[int], tokens: Sequence[int],
                          pattern: Sequence[int]) -> tuple[int, int]:
    """Interval [lo, hi) of SA entries whose suffix starts with pattern."""
    lo = 0
    hi = len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        if _suffix_cmp(tokens, sa[mid], pattern) < 0:
            lo = mid + 1
        else:
            hi = mid
    start = lo
    hi = len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        if _suffix_cmp(tokens, sa[mid], pattern) <= 0:
            lo = mid + 1
        else:
            hi = mid
    return start, lo


def naive_occurrences(tokens: Sequence[int],
                      pattern: Sequence[int]) -> list[int]:
    m = len(pattern)
    return [i for i in range(len(tokens) - m + 1)
            if _suffix_starts_with(tokens, i, pattern)]


def naive_longest_suffix_match(tokens: Sequence[int], query: Sequence[int],
                               max_len: int) -> int:
    """Largest L <= max_len such that query[-L:] occurs in tokens."""
    best = 0
    for length in range(1, min(max_len, len(query)) + 1):
        if naive_occurrences(tokens, query[-length:]):
            best = length
    return best


def naive_local_match(tokens: Sequence[int], k: int, max_pattern_len: int,
                      min_match_len: int = 1,
                      max_occurrences: int = 32) -> tuple[list[int], int]:
    """Reference for LocalMatchKernel.

    Finds the longest suffix of `tokens` (length in
    [min_match_len, max_pattern_len]) that occurred earlier in the same
    sequence (occurrences may extend into the tail), then drafts k
    tokens by depth-wise majority vote over the earliest
    `max_occurrences` occurrences (ties resolve to the smallest token
    id).

    Returns:
        (draft chain, match length); chain empty when no match.
    """
    from collections import Counter

    q = len(tokens)
    best_len = 0
    occ: list[int] = []
    for length in range(min_match_len, max_pattern_len + 1):
        if length > q:
            continue
        pattern = tokens[q - length:]
        positions = [s for s in range(q - length)
                     if list(tokens[s:s + length]) == list(pattern)]
        if positions:
            best_len = length
            occ = positions
    if best_len == 0:
        return [], 0
    active = occ[:max_occurrences]
    chain: list[int] = []
    for _ in range(k):
        toks = [tokens[s + best_len + len(chain)] for s in active
                if s + best_len + len(chain) < q]
        if not toks:
            break
        cnt = Counter(toks)
        top = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        chain.append(top)
        active = [s for s in active
                  if s + best_len + len(chain) - 1 < q
                  and tokens[s + best_len + len(chain) - 1] == top]
    return chain, best_len
