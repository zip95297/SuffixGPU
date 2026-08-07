"""Cross-request global memory: suffix array over finished responses.

Design (phase 1):
- The active (corpus, SA) pair is immutable between rebuilds, so the
  query path never races with writers.
- New documents land in an append-only delta buffer and are matched by
  seed-and-verify brute force.
- Rebuild copies (eviction-trimmed active corpus + delta snapshot) into
  the staging buffer, builds a fresh SA (on a side stream under CUDA),
  and swaps the active pair once the build event completes. Tokens
  appended after the snapshot stay in the delta until the next rebuild.
"""

from __future__ import annotations

import math
from collections import deque

import torch

from suffix_gpu.expand import expand_chain
from suffix_gpu.sa_search import longest_suffix_match
from suffix_gpu.suffix_array import build_suffix_array


class GlobalIndex:
    """GPU-resident global suffix index with periodic rebuilds."""

    def __init__(
        self,
        capacity: int,
        delta_capacity: int,
        k: int,
        max_occurrences: int = 32,
        rebuild_threshold: int | None = None,
        device: torch.device | str = "cpu",
        rebuild_stream: torch.cuda.Stream | None = None,
    ):
        self.capacity = capacity
        self.delta_capacity = delta_capacity
        self.k = k
        self.max_occurrences = max_occurrences
        self.rebuild_threshold = rebuild_threshold or max(
            1, delta_capacity // 2)
        self.device = torch.device(device)
        self.rebuild_stream = rebuild_stream

        self.corpus = torch.zeros(capacity, dtype=torch.int32,
                                  device=self.device)
        self.sa = torch.zeros(capacity, dtype=torch.int64,
                              device=self.device)
        self.staging_corpus = torch.zeros(capacity, dtype=torch.int32,
                                          device=self.device)
        self.staging_sa = torch.zeros(capacity, dtype=torch.int64,
                                      device=self.device)
        self.delta = torch.zeros(delta_capacity, dtype=torch.int32,
                                 device=self.device)

        self.active_len = 0
        self.delta_len = 0
        self.delta_snap = 0
        self.active_doc_lens: deque[int] = deque()
        self.delta_doc_lens: list[int] = []
        self._rebuild_event: torch.cuda.Event | None = None
        self._pending: tuple[int, deque[int], int] | None = None

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def append_documents(self, docs: list[torch.Tensor]) -> None:
        """Append finished-response token tensors to the delta."""
        for doc in docs:
            doc = doc.to(self.device).to(torch.int32).reshape(-1)
            n = doc.shape[0]
            if n == 0:
                continue
            if n > self.delta_capacity:
                doc = doc[-self.delta_capacity:]
                n = self.delta_capacity
            if self.delta_len + n > self.delta_capacity:
                self._make_room(n)
            self.delta[self.delta_len:self.delta_len + n].copy_(doc)
            self.delta_len += n
            self.delta_doc_lens.append(n)
        self.maybe_rebuild()

    def _make_room(self, needed: int) -> None:
        if self._rebuild_event is None:
            self._launch_rebuild(force=True)
        if self.delta_len + needed <= self.delta_capacity:
            return
        # Background rebuild pending: drop oldest pending docs.
        pending_tokens = self.delta_len - self.delta_snap
        keep = max(0, self.delta_capacity - needed - self.delta_snap)
        drop = pending_tokens - keep
        if drop > 0:
            self._drop_pending_tokens(drop)

    def _drop_pending_tokens(self, drop: int) -> None:
        acc = 0
        docs_dropped = 0
        for ln in self.delta_doc_lens[self._count_docs_within(
                self.delta_snap):]:
            if acc >= drop:
                break
            acc += ln
            docs_dropped += 1
        if acc < drop:
            return
        start = self.delta_snap + acc
        remaining = self.delta_len - start
        if remaining > 0:
            self.delta[self.delta_snap:self.delta_snap + remaining] = \
                self.delta[start:self.delta_len]
        self.delta_len = self.delta_snap + remaining
        absorbed_docs = self._count_docs_within(self.delta_snap)
        self.delta_doc_lens = (self.delta_doc_lens[:absorbed_docs]
                               + self.delta_doc_lens[
                                   absorbed_docs + docs_dropped:])

    # ------------------------------------------------------------------
    # rebuild
    # ------------------------------------------------------------------
    def maybe_rebuild(self) -> None:
        """Kick off a rebuild when pending delta grew past threshold."""
        if self._rebuild_event is not None or self.delta_len == 0:
            return
        pending = self.delta_len - self.delta_snap
        if pending < self.rebuild_threshold:
            return
        self._launch_rebuild()

    def _launch_rebuild(self, force: bool = False) -> None:
        # Evict oldest whole docs until the snapshot fits into capacity.
        keep_start = 0
        for ln in self.active_doc_lens:
            if self.active_len - keep_start + self.delta_len \
                    <= self.capacity:
                break
            keep_start += ln
        n_active_keep = self.active_len - keep_start
        snap = min(self.delta_len, self.capacity - n_active_keep)
        # Clamp the snapshot to a whole-doc boundary in the delta.
        acc = 0
        for ln in self.delta_doc_lens:
            if acc + ln <= snap:
                acc += ln
            else:
                break
        snap = acc
        n_new = n_active_keep + snap
        if n_new == 0:
            return

        dst = self.staging_corpus
        if n_active_keep > 0:
            dst[:n_active_keep].copy_(
                self.corpus[keep_start:self.active_len])
        if snap > 0:
            dst[n_active_keep:n_new].copy_(self.delta[:snap])

        new_doc_lens: deque[int] = deque()
        acc = 0
        for ln in self.active_doc_lens:
            acc += ln
            if acc > keep_start:
                new_doc_lens.append(ln)
        acc = 0
        for ln in self.delta_doc_lens:
            acc += ln
            if acc <= snap:
                new_doc_lens.append(ln)

        if (self.device.type == "cuda" and self.rebuild_stream is not None
                and not force):
            self.rebuild_stream.wait_stream(
                torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.rebuild_stream):
                self.staging_sa[:n_new] = build_suffix_array(dst[:n_new])
            event = torch.cuda.Event()
            event.record(self.rebuild_stream)
            self._pending = (n_new, new_doc_lens, snap)
            self._rebuild_event = event
            self.delta_snap = snap
        else:
            self.staging_sa[:n_new] = build_suffix_array(dst[:n_new])
            self._finish_swap(n_new, new_doc_lens, snap)

    def _count_docs_within(self, tokens: int) -> int:
        acc = 0
        count = 0
        for ln in self.delta_doc_lens:
            acc += ln
            if acc > tokens:
                break
            count += 1
        return count

    def _finish_swap(self, n_new: int, new_doc_lens: deque[int],
                     snap: int) -> None:
        (self.corpus, self.staging_corpus) = (self.staging_corpus,
                                              self.corpus)
        (self.sa, self.staging_sa) = (self.staging_sa, self.sa)
        self.active_len = n_new
        self.active_doc_lens = new_doc_lens
        remaining = self.delta_len - snap
        if remaining > 0:
            self.delta[:remaining] = self.delta[snap:self.delta_len]
        self.delta_len = remaining
        docs_absorbed = self._count_docs_within(snap)
        self.delta_doc_lens = self.delta_doc_lens[docs_absorbed:]
        self.delta_snap = 0
        self._rebuild_event = None
        self._pending = None

    def poll_rebuild(self) -> None:
        """Swap in a finished background rebuild (non-blocking)."""
        if self._rebuild_event is None:
            return
        if self._rebuild_event.query():
            n_new, new_doc_lens, snap = self._pending
            self._finish_swap(n_new, new_doc_lens, snap)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def query(
        self,
        query: torch.Tensor,
        query_len: torch.Tensor,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Longest-suffix match against corpus + delta.

        Args:
            query: [B, P] int32 padded query tails.
            query_len: [B] int64 valid lengths.
            max_len: maximum match length.

        Returns:
            (match_len [B] i64, cont [B, R, k] int32 padded with -1,
             occ_count [B] i64). Occurrences come from either the SA or
            the delta, whichever matched longer (ties prefer the SA).
        """
        self.poll_rebuild()
        b = query.shape[0]
        r = self.max_occurrences
        device = self.device
        match_len = torch.zeros(b, dtype=torch.int64, device=device)
        cont = torch.full((b, r, self.k), -1, dtype=torch.int32,
                          device=device)
        occ_count = torch.zeros(b, dtype=torch.int64, device=device)

        if self.active_len > 0:
            sa_len, sa_start, sa_end = longest_suffix_match(
                self.sa[:self.active_len], self.corpus[:self.active_len],
                query, query_len, max_len)
        else:
            sa_len = torch.zeros_like(match_len)
            sa_start = sa_end = sa_len
        d_len, d_pos, d_cnt = self._delta_match(query, query_len, max_len)

        use_sa = (sa_len >= d_len) & (sa_len > 0)
        use_delta = (d_len > sa_len) & (d_len > 0)
        match_len = torch.where(use_sa, sa_len,
                                torch.where(use_delta, d_len, match_len))

        offs = torch.arange(r, dtype=torch.int64, device=device)
        if self.active_len > 0:
            take = (sa_start.unsqueeze(1) + offs) < sa_end.unsqueeze(1)
            sa_idx = (sa_start.unsqueeze(1) + offs).clamp(
                min=0, max=self.active_len - 1)
            pos = self.sa[:self.active_len][sa_idx.reshape(-1)].reshape(b, r)
            sa_occ = torch.where(take, pos, 0)
            sa_cnt = torch.minimum(sa_end - sa_start,
                                   torch.full_like(sa_end, r))
            sa_cont = self._gather_cont(self.corpus, self.active_len,
                                        sa_occ, sa_cnt, match_len, use_sa)
            cont = torch.where(use_sa.unsqueeze(1).unsqueeze(2), sa_cont,
                               cont)
            occ_count = torch.where(use_sa, sa_cnt, occ_count)

        if self.delta_len > 0:
            d_cont = self._gather_cont(self.delta, self.delta_len, d_pos,
                                       d_cnt, match_len, use_delta)
            cont = torch.where(use_delta.unsqueeze(1).unsqueeze(2), d_cont,
                               cont)
            occ_count = torch.where(use_delta, d_cnt, occ_count)
        return match_len, cont, occ_count

    def _gather_cont(
        self,
        src: torch.Tensor,
        src_len: int,
        occ_pos: torch.Tensor,
        occ_cnt: torch.Tensor,
        match_len: torch.Tensor,
        sel: torch.Tensor,
    ) -> torch.Tensor:
        """Gather k-token continuations after occurrences in src."""
        b, r = occ_pos.shape
        row = torch.arange(r, device=self.device).unsqueeze(0)
        row_active = (row < occ_cnt.unsqueeze(1)) & sel.unsqueeze(1)
        idx = (occ_pos.unsqueeze(2) + match_len.unsqueeze(1).unsqueeze(2)
               + torch.arange(self.k, device=self.device))
        valid = row_active.unsqueeze(2) & (idx < src_len) & (idx >= 0)
        vals = src[idx.clamp(0, src_len - 1).reshape(b, r * self.k)
                   ].reshape(b, r, self.k)
        return torch.where(valid, vals, -1)

    def _delta_match(
        self,
        query: torch.Tensor,
        query_len: torch.Tensor,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Seed-and-verify longest match within the delta buffer.

        Returns (match_len [B], positions [B, C], count [B]).
        """
        b = query.shape[0]
        c = self.max_occurrences
        device = self.device
        dlen = self.delta_len
        zero = torch.zeros(b, dtype=torch.int64, device=device)
        if dlen == 0:
            return (zero, zero.unsqueeze(1).expand(b, c).contiguous(),
                    zero.clone())
        pos_all = torch.arange(dlen, dtype=torch.int64, device=device)
        offs = torch.arange(max_len, dtype=torch.int64, device=device)
        delta_active = self.delta[:dlen]

        def verify(length: torch.Tensor) -> tuple[torch.Tensor,
                                                  torch.Tensor]:
            pat_idx = (query_len.unsqueeze(1) - length.unsqueeze(1)
                       + offs.unsqueeze(0))
            pat_valid = offs.unsqueeze(0) < length.unsqueeze(1)
            pat = torch.where(
                pat_valid,
                query.gather(1, pat_idx.clamp(0, query.shape[1] - 1)), 0)
            seed = delta_active.unsqueeze(0) == pat[:, :1]
            seed &= ((pos_all.unsqueeze(0) + length.unsqueeze(1) - 1 < dlen)
                     & (query_len >= length).unsqueeze(1))
            key = pos_all.unsqueeze(0) + (~seed).to(torch.int64) * (dlen + 1)
            order = torch.argsort(key, dim=1)
            width = min(dlen, c)
            cand = pos_all.unsqueeze(0).expand(b, dlen).gather(
                1, order[:, :width])
            if width < c:
                cand = torch.cat([
                    cand, torch.zeros(b, c - width, dtype=torch.int64,
                                      device=device)], dim=1)
            cand_active = seed.gather(1, cand.clamp(0, dlen - 1))
            idx = cand.unsqueeze(2) + offs.unsqueeze(0).unsqueeze(0)
            valid = (idx < dlen) & pat_valid.unsqueeze(1)
            toks = delta_active[idx.clamp(0, dlen - 1)]
            full = ((toks == pat.unsqueeze(1)) | ~valid).all(dim=2)
            ok = cand_active & full
            cnt = ok.sum(dim=1)
            pos_out = torch.where(ok, cand, 0)
            return cnt, pos_out

        hi0 = min(max_len, dlen)
        lo, hi = zero.clone(), torch.full(
            (b,), hi0, dtype=torch.int64, device=device)
        iters = max(1, math.ceil(math.log2(hi0 + 1)))
        pos_keep = torch.zeros(b, c, dtype=torch.int64, device=device)
        cnt_keep = zero.clone()
        for _ in range(iters):
            mid = (lo + hi + 1) // 2
            cnt, pos_out = verify(mid)
            pred = (cnt > 0) & (mid > 0)
            lo = torch.where(pred, mid, lo)
            hi = torch.where(pred, hi, mid - 1)
            pos_keep = torch.where(pred.unsqueeze(1), pos_out, pos_keep)
            cnt_keep = torch.where(pred, cnt, cnt_keep)
        cnt, pos_out = verify(lo)
        final_ok = lo > 0
        pos_out = torch.where(final_ok.unsqueeze(1), pos_out, pos_keep)
        cnt = torch.where(final_ok, cnt, cnt_keep)
        return lo, pos_out, torch.minimum(cnt, torch.full_like(cnt, c))

    def expand(self, cont: torch.Tensor,
               occ_count: torch.Tensor) -> tuple[torch.Tensor,
                                                  torch.Tensor]:
        return expand_chain(cont, occ_count, self.k)
