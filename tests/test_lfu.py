"""LFU global-memory tests: credit attribution, utility-ranked
eviction, decay/aging, and stale-position invalidation."""

from __future__ import annotations

import torch

from suffix_gpu.global_index import GlobalIndex
from suffix_gpu.proposer import SuffixGPUDrafter


def _doc(tok, n):
    return torch.full((n,), tok, dtype=torch.int32)


def _mk_index(device, eviction, capacity=64, protect=0):
    return GlobalIndex(capacity=capacity, delta_capacity=64, k=4,
                       max_occurrences=8, rebuild_threshold=1,
                       device=device, eviction=eviction,
                       lfu_protect_rebuilds=protect)


def test_lfu_keeps_credited_doc_under_pressure(device):
    # doc1 and doc2 fill the ring; doc1 gets accepted credit, doc2
    # none. Appending doc3 forces eviction: LFU must drop doc2, FIFO
    # would drop doc1.
    gi = _mk_index(device, "lfu")
    gi.append_documents([_doc(11, 24).to(device)])   # doc1 @ [0, 25)
    gi.append_documents([_doc(22, 24).to(device)])   # doc2 @ [25, 50)
    assert gi.active_len == 50
    pos = torch.arange(4, dtype=torch.int64, device=device).view(1, 4)
    w = torch.ones(1, 4, dtype=torch.float32, device=device)
    tier = torch.ones(1, dtype=torch.int8, device=device)
    gi.credit_accepted(pos, w, tier)                 # 4 credits -> doc1
    gi.append_documents([_doc(33, 24).to(device)])   # forces eviction
    kept_tokens = set(gi.corpus[:gi.active_len].tolist())
    assert 11 in kept_tokens        # credited doc survives
    assert 22 not in kept_tokens    # zero-utility doc evicted
    assert 33 in kept_tokens

    fifo = _mk_index(device, "fifo")
    fifo.append_documents([_doc(11, 24).to(device)])
    fifo.append_documents([_doc(22, 24).to(device)])
    fifo.append_documents([_doc(33, 24).to(device)])
    kept_tokens = set(fifo.corpus[:fifo.active_len].tolist())
    assert 11 not in kept_tokens    # FIFO drops the oldest
    assert 22 in kept_tokens


def test_lfu_decay_and_age(device):
    gi = _mk_index(device, "lfu", capacity=256)
    gi.append_documents([_doc(7, 10).to(device)])
    assert list(gi.active_doc_ages) == [0]
    pos = torch.zeros(1, 2, dtype=torch.int64, device=device)
    gi.credit_accepted(pos + torch.tensor([1, 2], device=device),
                       torch.ones(1, 2, device=device),
                       torch.ones(1, dtype=torch.int8, device=device))
    before = float(gi.hit[:gi.active_len].sum())
    assert before == 2.0
    gi.append_documents([_doc(8, 10).to(device)])    # rebuild
    assert list(gi.active_doc_ages) == [1, 0]
    after = float(gi.hit[:11].sum())
    assert abs(after - before * gi.lfu_decay) < 1e-6


def test_lfu_protection_shields_young_docs(device):
    # Equal (zero) utility: the old doc is unprotected and evicted,
    # the young one survives on age protection alone.
    from collections import deque
    gi = _mk_index(device, "lfu", protect=1)
    gi.append_documents([_doc(11, 24).to(device)])
    gi.append_documents([_doc(22, 24).to(device)])
    gi.active_doc_ages = deque([5, 0])   # doc1 old, doc2 young
    gi.append_documents([_doc(33, 24).to(device)])
    kept = set(gi.corpus[:gi.active_len].tolist())
    assert 22 in kept and 33 in kept and 11 not in kept


def test_proposer_credit_flow(device):
    d = SuffixGPUDrafter(
        k=4, device=device, max_pattern_len=8, max_occurrences=8,
        enable_global=True, global_capacity=1 << 10,
        delta_capacity=1 << 8, rebuild_threshold=1,
        vote_smoothing_alpha=0.0, local_mode="backoff",
        merge_paths=False, dynamic_k=False, eviction="lfu")
    gi = d.global_index
    doc = [5, 6, 7, 8, 5, 6, 7, 8]
    buf = torch.zeros(2, 32, dtype=torch.int32, device=device)
    buf[0, :len(doc)] = torch.tensor(doc, dtype=torch.int32)
    d.harvest_finished([0], [len(doc)], buf)   # rebuild -> corpus
    d.poll()                                   # sync credit epoch
    # Request 1 ends with "5 6": the global index drafts "7 8 ...".
    cur = [1, 2, 5, 6]
    buf[1, :len(cur)] = torch.tensor(cur, dtype=torch.int32)
    lens = torch.tensor([len(doc), len(cur)], dtype=torch.int32,
                        device=device)
    draft, nv = d.propose(lens, buf)
    assert draft[1, 0].item() == 7
    assert float(gi.hit.sum()) == 0.0
    # Verifier accepts 2 drafts + 1 bonus token => credit 2.
    sampled = torch.full((2, 5), -1, dtype=torch.int32, device=device)
    sampled[1, :3] = torch.tensor([7, 8, 5], dtype=torch.int32)
    d.update_state(lens, buf, sampled)
    assert abs(float(gi.hit.sum()) - 2.0) < 1e-5
    # All credit landed inside the harvested document's extent.
    assert abs(float(gi.hit[:gi.active_len].sum()) - 2.0) < 1e-5


def test_epoch_invalidation_drops_stale_positions(device):
    d = SuffixGPUDrafter(
        k=4, device=device, max_pattern_len=8, max_occurrences=8,
        enable_global=True, global_capacity=1 << 10,
        delta_capacity=1 << 8, rebuild_threshold=1,
        vote_smoothing_alpha=0.0, local_mode="backoff",
        merge_paths=False, dynamic_k=False, eviction="lfu")
    gi = d.global_index
    doc = [5, 6, 7, 8, 5, 6, 7, 8]
    buf = torch.zeros(2, 32, dtype=torch.int32, device=device)
    buf[0, :len(doc)] = torch.tensor(doc, dtype=torch.int32)
    d.harvest_finished([0], [len(doc)], buf)
    d.poll()
    cur = [1, 2, 5, 6]
    buf[1, :len(cur)] = torch.tensor(cur, dtype=torch.int32)
    lens = torch.tensor([len(doc), len(cur)], dtype=torch.int32,
                        device=device)
    d.propose(lens, buf)
    # Coordinates move (another harvest triggers rebuild + compaction)
    # before the verifier result lands: poll() must drop the cached
    # positions so nothing is credited.
    d.harvest_finished([0], [len(doc)], buf)
    d.poll()
    sampled = torch.full((2, 5), -1, dtype=torch.int32, device=device)
    sampled[1, :3] = torch.tensor([7, 8, 5], dtype=torch.int32)
    d.update_state(lens, buf, sampled)
    assert float(gi.hit.sum()) == 0.0
