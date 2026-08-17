from __future__ import annotations

import torch

from suffix_gpu.proposer import SuffixGPUDrafter

K = 4
P = 8


LEGACY = dict(vote_smoothing_alpha=0.0, local_mode="backoff",
              merge_paths=False, dynamic_k=False, eviction="fifo")


def _drafter(device, enable_global=False, **kw):
    return SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=P, max_occurrences=8,
        enable_global=enable_global, global_capacity=1024,
        delta_capacity=256, rebuild_threshold=100000, **LEGACY, **kw)


def test_update_state_scatter(device):
    d = _drafter(device)
    buf = torch.zeros(2, 10, dtype=torch.int32, device=device)
    buf[0, :3] = torch.tensor([1, 2, 3], dtype=torch.int32)
    buf[1, :5] = torch.tensor([9, 9, 9, 9, 9], dtype=torch.int32)
    counts = torch.tensor([3, 5], dtype=torch.int32, device=device)
    sampled = torch.tensor([[7, 8, -1], [4, -1, -1]], dtype=torch.int32,
                           device=device)
    new_counts = d.update_state(counts, buf, sampled)
    assert new_counts.tolist() == [5, 6]
    assert buf[0, :5].tolist() == [1, 2, 3, 7, 8]
    assert buf[1, :6].tolist() == [9, 9, 9, 9, 9, 4]


def test_update_state_clips_at_buffer_end(device):
    d = _drafter(device)
    buf = torch.zeros(1, 4, dtype=torch.int32, device=device)
    counts = torch.tensor([3], dtype=torch.int32, device=device)
    sampled = torch.tensor([[5, 6]], dtype=torch.int32, device=device)
    new_counts = d.update_state(counts, buf, sampled)
    assert buf[0].tolist()[3] == 5  # first fits
    assert new_counts.tolist() == [5]  # count still advances


def test_propose_with_update_drafts_repetition(device):
    d = _drafter(device)
    buf = torch.zeros(1, 32, dtype=torch.int32, device=device)
    hist = [5, 6, 7, 8, 5, 6, 7]
    buf[0, :len(hist)] = torch.tensor(hist, dtype=torch.int32)
    counts = torch.tensor([len(hist)], dtype=torch.int32, device=device)
    sampled = torch.tensor([[8, 5, -1, -1, -1]], dtype=torch.int32,
                           device=device)
    draft, nv, new_counts = d.propose_with_update(counts, buf, sampled)
    assert new_counts.tolist() == [len(hist) + 2]
    # History ...5,6,7,8,5 -> longest suffix match continues 6,7,8,5.
    n = nv[0].item()
    assert n >= 1
    assert draft[0, :n].tolist() == [6, 7, 8, 5][:n]


def test_propose_with_update_masks_empty_rows(device):
    d = _drafter(device)
    buf = torch.zeros(2, 16, dtype=torch.int32, device=device)
    rep = [3, 4, 3, 4, 3]
    for i in range(2):
        buf[i, :len(rep)] = torch.tensor(rep, dtype=torch.int32)
    counts = torch.tensor([len(rep), len(rep)], dtype=torch.int32,
                          device=device)
    sampled = torch.tensor([[4, -1], [-1, -1]], dtype=torch.int32,
                           device=device)
    draft, nv, _ = d.propose_with_update(counts, buf, sampled)
    assert nv[0].item() > 0        # row 0 drafted
    assert nv[1].item() == 0       # row 1 had no sampled tokens
    assert (draft[1] == -1).all()


def test_propose_with_update_respects_max_model_len(device):
    d = _drafter(device)
    buf = torch.zeros(1, 16, dtype=torch.int32, device=device)
    rep = [3, 4, 3, 4, 3, 4]
    buf[0, :len(rep)] = torch.tensor(rep, dtype=torch.int32)
    counts = torch.tensor([len(rep)], dtype=torch.int32, device=device)
    sampled = torch.tensor([[3, -1]], dtype=torch.int32, device=device)
    _, nv, new_counts = d.propose_with_update(
        counts, buf, sampled, max_model_len=len(rep) + 1)
    assert new_counts.tolist() == [len(rep) + 1]
    assert nv[0].item() == 0  # at the length cap: no draft


def test_ingest_active_chunks_and_final(device):
    d = _drafter(device, enable_global=True)
    gi = d.global_index
    row = torch.arange(100, 200, dtype=torch.int32, device=device)
    # Below chunk threshold: nothing ingested.
    d.ingest_active(["r0"], [row], [10], chunk=16)
    assert gi.delta_len == 0
    # Crossing the threshold emits one chunk.
    d.ingest_active(["r0"], [row], [20], chunk=16)
    assert gi.delta_len == 21  # 20 tokens + SEP
    # Growth below threshold waits...
    d.ingest_active(["r0"], [row], [30], chunk=16)
    assert gi.delta_len == 21
    # ...but final flushes the tail (with overlap) and forgets the key.
    d.ingest_active(["r0"], [row], [30], final=True, chunk=16)
    assert gi.delta_len > 21
    assert "r0" not in d._ingested


def test_ingest_active_overlap_keeps_boundary_patterns(device):
    d = _drafter(device, enable_global=True)
    row = torch.arange(500, 564, dtype=torch.int32, device=device)
    d.ingest_active(["r"], [row], [32], chunk=32)
    d.ingest_active(["r"], [row], [64], chunk=32)
    # Pattern spanning the first chunk boundary (tokens 28..36).
    q = row[28:36].unsqueeze(0)
    qlen = torch.tensor([8], dtype=torch.int64, device=device)
    mlen, cont, occ = d.global_index.query(q.to(torch.int32), qlen, P)
    assert mlen[0].item() == 8
    chain, nv = d.global_index.expand(cont, occ)
    n = nv[0].item()
    assert n > 0
    assert chain[0, :n].tolist() == row[36:36 + n].tolist()


def test_ingest_then_cross_request_draft(device):
    d = _drafter(device, enable_global=True)
    phrase = [11, 12, 13, 14, 15, 16, 17, 18]
    resp = torch.tensor(phrase * 8, dtype=torch.int32, device=device)
    d.ingest_active(["a"], [resp], [len(resp)], chunk=16)
    # A different request whose tail matches the shared phrase.
    buf = torch.zeros(1, 32, dtype=torch.int32, device=device)
    cur = [99, 98] + phrase[:6]
    buf[0, :len(cur)] = torch.tensor(cur, dtype=torch.int32)
    counts = torch.tensor([len(cur)], dtype=torch.int32, device=device)
    mask = torch.ones(1, dtype=torch.bool, device=device)
    draft, nv = d.propose(counts, buf, mask)
    n = nv[0].item()
    assert n > 0
    assert draft[0, :n].tolist() == (phrase[6:] + phrase)[:n]
