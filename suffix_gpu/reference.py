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
