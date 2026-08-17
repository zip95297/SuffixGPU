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
                      max_occurrences: int = 32,
                      min_token_prob: float = 0.0,
                      max_spec_factor: float | None = None,
                      max_spec_offset: float = 0.0,
                      support_thresholds: Sequence[int] = (1, 2, 4, 8),
                      ) -> tuple[list[int], int]:
    """Reference for LocalMatchKernel (multi-length scored backoff).

    Candidate lengths are, for each support threshold t, the largest
    suffix length whose occurrence count is >= t. Each candidate drafts
    by depth-wise majority vote (ties to the smallest token id) with the
    cumulative-probability cutoff, is truncated to the adaptive cap
    floor(max_spec_factor * L + max_spec_offset), and is scored by the
    sum of per-depth chain probabilities. The best score wins; exact
    ties prefer the longer match.

    Returns:
        (draft chain, match length); chain empty when no match.
    """
    from collections import Counter

    q = len(tokens)
    occ_by_len: dict[int, list[int]] = {}
    for length in range(1, min(max_pattern_len, q - 1) + 1 if q > 1 else 0):
        pattern = list(tokens[q - length:])
        ends = [i for i in range(length, q)
                if list(tokens[i - length:i]) == pattern]
        if ends:
            occ_by_len[length] = ends

    candidates: list[int] = []
    for t_sup in support_thresholds:
        best = 0
        for length, ends in occ_by_len.items():
            if len(ends) >= t_sup and length > best:
                best = length
        if best >= min_match_len:
            candidates.append(best)

    best_chain: list[int] = []
    best_len = 0
    best_score = -1.0
    for lc in candidates:
        ends = occ_by_len[lc][:max_occurrences]
        chain: list[int] = []
        cum = 1.0
        cums: list[float] = []
        active = list(ends)
        for d in range(k):
            live = [i for i in active if i + d < q]
            if not live:
                break
            cnt = Counter(tokens[i + d] for i in live)
            top = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            cum_next = cum * cnt[top] / len(live)
            if min_token_prob > 0 and cum_next < min_token_prob:
                break
            cum = cum_next
            chain.append(top)
            cums.append(cum)
            active = [i for i in live if tokens[i + d] == top]
        if max_spec_factor is None:
            cap = k
        else:
            cap = max(0, int(max_spec_factor * lc + max_spec_offset + 1e-9))
        num_emit = min(len(chain), cap)
        score = sum(cums[:num_emit])
        if score > best_score or (score == best_score and lc > best_len):
            best_chain = chain[:num_emit]
            best_len = lc
            best_score = score
    if best_len == 0:
        return [], 0
    return best_chain, best_len


def naive_soft_local_match(tokens: Sequence[int], k: int,
                           max_pattern_len: int,
                           min_match_len: int = 1,
                           max_occurrences: int = 32,
                           min_token_prob: float = 0.0,
                           max_spec_factor: float | None = None,
                           max_spec_offset: float = 0.0,
                           soft_lambda: float = 0.5,
                           alpha: float = 0.0,
                           ) -> tuple[list[int], int]:
    """Reference for the soft weighted-backoff local matcher.

    Every occurrence site (end position i with match length
    mb[i] >= min_match_len) votes with weight
    ``soft_lambda ** (L_max - mb[i])``; the R sites kept are the best
    by (length desc, position asc). Draft by weighted majority vote
    (ties to smallest token), chain probability
    p_d = prod(v_w / (a_w + alpha)), emission capped by
    floor(max_spec_factor * L_max + max_spec_offset).

    Returns:
        (draft chain, longest match length); empty when no site.
    """
    q = len(tokens)
    sites: list[tuple[int, int]] = []  # (mb, end pos)
    for i in range(q):
        best = 0
        for length in range(1, min(max_pattern_len, i, q - 1) + 1):
            if i - length < 0:
                break
            if list(tokens[i - length:i]) == list(tokens[q - length:]):
                best = length
        if best >= max(1, min_match_len):
            sites.append((best, i))
    if not sites:
        return [], 0
    sites.sort(key=lambda t: (-t[0], t[1]))
    sites = sites[:max_occurrences]
    l_max = sites[0][0]
    occ = [(pos, soft_lambda ** (l_max - mb)) for mb, pos in sites]

    chain: list[int] = []
    cum = 1.0
    active = list(occ)
    for d in range(k):
        live = [(i, w) for i, w in active if i + d < q]
        if not live:
            break
        votes: dict[int, float] = {}
        for i, w in live:
            votes[tokens[i + d]] = votes.get(tokens[i + d], 0.0) + w
        top = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        a_w = sum(w for _, w in live)
        cum_next = cum * votes[top] / (a_w + alpha)
        if min_token_prob > 0 and cum_next < min_token_prob:
            break
        cum = cum_next
        chain.append(top)
        active = [(i, w) for i, w in live if tokens[i + d] == top]
    if max_spec_factor is None:
        cap = k
    else:
        cap = max(0, int(max_spec_factor * l_max + max_spec_offset
                         + 1e-9))
    return chain[:cap], l_max
