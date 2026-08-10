from __future__ import annotations

from collections import Counter

import torch

from suffix_gpu.global_index import GlobalIndex
from suffix_gpu.proposer import SuffixGPUDrafter
from suffix_gpu.reference import (naive_longest_suffix_match,
                                  naive_occurrences)

K = 4
R = 8
P = 8


def _naive_chain(corpus: list[int], query: list[int], k: int,
                 max_len: int) -> tuple[list[int], int]:
    length = naive_longest_suffix_match(corpus, query, max_len)
    if length == 0:
        return [], 0
    active = naive_occurrences(corpus, query[-length:])[:R]
    chain: list[int] = []
    for _ in range(k):
        toks = [corpus[s + length + len(chain)] for s in active
                if s + length + len(chain) < len(corpus)]
        if not toks:
            break
        cnt = Counter(toks)
        top = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        chain.append(top)
        active = [s for s in active
                  if s + length + len(chain) - 1 < len(corpus)
                  and corpus[s + length + len(chain) - 1] == top]
    return chain, length


def _query(index: GlobalIndex, query: list[int], device) -> tuple[
        list[int], int, int]:
    q = torch.tensor([query], dtype=torch.int32, device=device)
    qlen = torch.tensor([len(query)], dtype=torch.int64, device=device)
    match_len, cont, occ = index.query(q, qlen, P)
    chain, nv = index.expand(cont, occ)
    return (chain[0].tolist()[:nv[0].item()], match_len[0].item(),
            occ[0].item())


def _append(index: GlobalIndex, docs: list[list[int]], device) -> None:
    index.append_documents(
        [torch.tensor(d, dtype=torch.int32, device=device) for d in docs])


def test_delta_path(device):
    idx = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                      max_occurrences=R, rebuild_threshold=100000,
                      device=device)
    doc = [5, 6, 7, 8, 9, 5, 6, 7, 8, 10]
    _append(idx, [doc], device)
    chain, mlen, occ = _query(idx, [5, 6, 7, 8], device)
    exp_chain, exp_len = _naive_chain(doc, [5, 6, 7, 8], K, P)
    assert mlen == exp_len
    assert chain == exp_chain
    assert occ == 2


def test_sa_path_after_rebuild(device):
    idx = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                      max_occurrences=R, rebuild_threshold=1,
                      device=device)
    doc = [5, 6, 7, 8, 9, 5, 6, 7, 8, 10]
    _append(idx, [doc], device)
    assert idx.active_len == len(doc)
    assert idx.delta_len == 0
    chain, mlen, occ = _query(idx, [5, 6, 7, 8], device)
    exp_chain, exp_len = _naive_chain(doc, [5, 6, 7, 8], K, P)
    assert mlen == exp_len
    assert chain == exp_chain
    assert occ == 2


def test_delta_sa_consistency(device):
    doc1 = [1, 2, 3, 4, 1, 2, 3, 9]
    doc2 = [7, 1, 2, 3, 4, 8, 1, 2]
    corpus = doc1 + doc2
    q_before = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                           max_occurrences=R, rebuild_threshold=100000,
                           device=device)
    _append(q_before, [doc1, doc2], device)
    r_before = _query(q_before, [1, 2, 3, 4], device)
    q_after = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                          max_occurrences=R, rebuild_threshold=1,
                          device=device)
    _append(q_after, [doc1, doc2], device)
    r_after = _query(q_after, [1, 2, 3, 4], device)
    assert r_before[1] == r_after[1]
    assert r_before[0] == r_after[0]
    exp_chain, exp_len = _naive_chain(corpus, [1, 2, 3, 4], K, P)
    assert r_after[1] == exp_len
    assert r_after[0] == exp_chain


def test_multiple_rebuilds(device):
    idx = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                      max_occurrences=R, rebuild_threshold=1,
                      device=device)
    docs = [[1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5]]
    for d in docs:
        _append(idx, [d], device)
    corpus = [t for d in docs for t in d]
    assert idx.active_len == len(corpus)
    chain, mlen, occ = _query(idx, [1, 1], device)
    exp_chain, exp_len = _naive_chain(corpus, [1, 1], K, P)
    assert mlen == exp_len
    assert chain == exp_chain


def test_eviction(device):
    idx = GlobalIndex(capacity=12, delta_capacity=64, k=K,
                      max_occurrences=R, rebuild_threshold=1,
                      device=device)
    _append(idx, [[1, 1, 1, 1]], device)
    _append(idx, [[2, 2, 2, 2]], device)
    _append(idx, [[3, 3, 3, 3]], device)
    _append(idx, [[4, 4, 4, 4]], device)
    assert idx.active_len <= 12
    _, mlen_new, _ = _query(idx, [4, 4], device)
    assert mlen_new == 2
    _, mlen_old, _ = _query(idx, [1, 1, 1, 1], device)
    assert mlen_old == 0


def test_empty_index(device):
    idx = GlobalIndex(capacity=64, delta_capacity=16, k=K,
                      max_occurrences=R, device=device)
    chain, mlen, occ = _query(idx, [1, 2, 3], device)
    assert (mlen, occ, chain) == (0, 0, [])


def test_proposer_global_harvest(device):
    drafter = SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=P, max_occurrences=R,
        enable_global=True, global_capacity=1024, delta_capacity=256,
        rebuild_threshold=1)
    # Request 0 finished with a distinctive repeating response.
    buf = torch.zeros(2, 32, dtype=torch.int32, device=device)
    done = [9, 8, 7, 6, 5, 9, 8, 7, 6, 4]
    buf[0, :len(done)] = torch.tensor(done, dtype=torch.int32)
    drafter.harvest_finished([0], [len(done)], buf)
    # Request 1 is generating the same prefix.
    cur = [9, 8, 7, 6]
    buf[1, :len(cur)] = torch.tensor(cur, dtype=torch.int32)
    lens = torch.tensor([0, len(cur)], dtype=torch.int32, device=device)
    mask = torch.tensor([False, True], dtype=torch.bool, device=device)
    draft, nv = drafter.propose(lens, buf, mask)
    exp_chain, _ = _naive_chain(done, cur, K, P)
    assert nv[1].item() == len(exp_chain)
    assert draft[1].tolist()[:nv[1].item()] == exp_chain
    assert nv[0].item() == 0


def test_delta_occurrence_positions_compacted(device):
    """Regression (P0-1): occurrence positions must be compacted left.

    The buggy path emitted fake "position 0" rows for unmatched slots,
    which joined the majority vote and could flip the drafted token.
    """
    idx = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                      max_occurrences=R, rebuild_threshold=100000,
                      device=device)
    # [7, 7] occurs at positions 1, 10 and 19, all continuing with 101;
    # position 0 starts with a different token on purpose.
    doc = [9, 7, 7, 101, 5, 5, 5, 5, 5, 3,
           7, 7, 101, 6, 6, 6, 6, 6, 6,
           7, 7, 101, 4]
    _append(idx, [doc], device)

    q = torch.tensor([[7, 7] + [0] * (P - 2)], dtype=torch.int32,
                     device=device)
    qlen = torch.tensor([2], dtype=torch.int64, device=device)
    d_len, d_pos, d_cnt = idx._delta_match(q, qlen, P)
    assert d_len[0].item() == 2
    assert d_cnt[0].item() == 3
    assert d_pos[0, :3].tolist() == [1, 10, 19]

    chain, mlen, occ = _query(idx, [7, 7], device)
    exp_chain, exp_len = _naive_chain(doc, [7, 7], K, P)
    assert mlen == exp_len
    assert occ == 3
    assert chain == exp_chain
    assert chain[0] == 101


def test_delta_match_len_full_verification(device):
    """Regression (P0-2): a common first token must not hide a longer
    match sitting past the first `max_occurrences` seed positions."""
    idx = GlobalIndex(capacity=1024, delta_capacity=256, k=K,
                      max_occurrences=4, rebuild_threshold=100000,
                      device=device)
    # 10 early positions start with token 1 but do not match; the only
    # occurrence of [1, 2, 3, 4] sits at the end of the delta.
    doc = [1, 9] * 10 + [1, 2, 3, 4, 6]
    _append(idx, [doc], device)
    chain, mlen, occ = _query(idx, [1, 2, 3, 4], device)
    assert mlen == naive_longest_suffix_match(doc, [1, 2, 3, 4], P) == 4
    assert occ == 1
    assert chain == [6]


class _NeverDoneEvent:
    """Stands in for an in-flight CUDA rebuild event."""

    def query(self) -> bool:
        return False


def test_delta_overflow_with_rebuild_in_flight(device):
    """Regression (P0-3): appends that overflow the delta while a
    rebuild is in flight must drop oldest docs, not crash copy_."""
    idx = GlobalIndex(capacity=64, delta_capacity=16, k=K,
                      max_occurrences=R, rebuild_threshold=100000,
                      device=device)
    _append(idx, [[1] * 6, [2] * 6], device)
    idx._rebuild_event = _NeverDoneEvent()
    idx._pending = (idx.active_len, idx.active_doc_lens)
    _append(idx, [[3] * 10], device)  # 12 + 10 > 16: must make room
    assert idx.delta_len <= idx.delta_capacity
    _, mlen_new, _ = _query(idx, [3, 3], device)
    assert mlen_new == 2
    _, mlen_kept, _ = _query(idx, [2, 2], device)
    assert mlen_kept == 2
    _, mlen_dropped, _ = _query(idx, [1, 1], device)
    assert mlen_dropped == 0
    idx._rebuild_event = None
    idx._pending = None


def test_delta_overflow_sync_rebuild(device):
    """Overflow with no rebuild in flight absorbs the delta into the SA."""
    idx = GlobalIndex(capacity=64, delta_capacity=16, k=K,
                      max_occurrences=R, rebuild_threshold=100000,
                      device=device)
    _append(idx, [[1] * 12], device)
    _append(idx, [[2] * 12], device)
    assert idx.active_len == 12
    assert idx.delta_len == 12
    _, mlen_sa, _ = _query(idx, [1, 1], device)
    assert mlen_sa == 2
    _, mlen_delta, _ = _query(idx, [2, 2], device)
    assert mlen_delta == 2
