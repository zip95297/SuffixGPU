"""Cross-request global memory: suffix array over finished responses.

Design (phase 2):
- The active (corpus, SA) pair is immutable between rebuilds, so the
  query path never races with writers.
- Query shapes are static: the corpus buffer is padded with a sentinel
  token (int32 max) that sorts after every real token, and searches
  always run over the full capacity. The SA is built over the real
  prefix plus one sentinel column (build and query then agree on suffix
  ordering); padding entries are appended in position order, which is
  valid because every all-sentinel suffix compares greater than any
  real-token pattern. The delta path masks by a device-side length
  scalar instead of slicing.
- New documents land in an append-only delta buffer and are matched by
  full brute-force verification (rolling AND over pattern offsets).
- Rebuild copies (eviction-trimmed active corpus + delta snapshot) into
  the staging buffer and immediately compacts the snapshot out of the
  delta (staging owns a copy), so appends always have the full delta
  capacity available while the SA builds on a side stream. The active
  pair swaps once the build event completes.
"""

from __future__ import annotations

from collections import deque

import torch

from suffix_gpu import triton_kernels
from suffix_gpu.expand import expand_chain, score_chain
from suffix_gpu.sa_search import longest_suffix_match
from suffix_gpu.suffix_array import build_suffix_array

PAD_TOKEN = torch.iinfo(torch.int32).max
# Written between documents. Negative, so it can never equal a query
# token (real ids are >= 0): matches and continuations cannot cross
# document boundaries.
SEP_TOKEN = -2


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

        self.corpus = torch.full((capacity,), PAD_TOKEN, dtype=torch.int32,
                                 device=self.device)
        self.staging_corpus = self.corpus.clone()
        self.sa = torch.arange(capacity, dtype=torch.int64,
                               device=self.device)
        self.staging_sa = self.sa.clone()
        self.delta = torch.zeros(delta_capacity, dtype=torch.int32,
                                 device=self.device)
        self.delta_len_t = torch.zeros((), dtype=torch.int64,
                                       device=self.device)

        self.active_len = 0
        self.delta_len = 0
        self.active_doc_lens: deque[int] = deque()
        self.delta_doc_lens: list[int] = []
        self._rebuild_event: torch.cuda.Event | None = None
        self._pending: tuple[int, deque[int]] | None = None

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def append_documents(self, docs: list[torch.Tensor]) -> None:
        """Append finished-response token tensors to the delta.

        Each document is followed by one SEP_TOKEN, so matches and
        continuations never span document boundaries. Recorded doc
        lengths include the separator.
        """
        for doc in docs:
            doc = doc.to(self.device).to(torch.int32).reshape(-1)
            n = doc.shape[0]
            if n == 0:
                continue
            if n + 1 > self.delta_capacity:
                doc = doc[-(self.delta_capacity - 1):]
                n = doc.shape[0]
            if self.delta_len + n + 1 > self.delta_capacity:
                self._make_room(n + 1)
            self.delta[self.delta_len:self.delta_len + n].copy_(doc)
            self.delta[self.delta_len + n] = SEP_TOKEN
            self.delta_len += n + 1
            self.delta_len_t.fill_(self.delta_len)
            self.delta_doc_lens.append(n + 1)
        self.maybe_rebuild()

    def _make_room(self, needed: int) -> None:
        self.poll_rebuild()
        if self._rebuild_event is None:
            self._launch_rebuild()
        if self.delta_len + needed <= self.delta_capacity:
            return
        # A rebuild is in flight (staging buffers busy): drop oldest docs.
        self._drop_oldest_docs(
            self.delta_len + needed - self.delta_capacity)

    def _drop_oldest_docs(self, drop: int) -> None:
        acc = 0
        docs_dropped = 0
        for ln in self.delta_doc_lens:
            if acc >= drop:
                break
            acc += ln
            docs_dropped += 1
        remaining = self.delta_len - acc
        if remaining > 0:
            self.delta[:remaining] = self.delta[acc:self.delta_len].clone()
        self.delta_len = remaining
        self.delta_len_t.fill_(remaining)
        self.delta_doc_lens = self.delta_doc_lens[docs_dropped:]

    # ------------------------------------------------------------------
    # rebuild
    # ------------------------------------------------------------------
    def maybe_rebuild(self) -> None:
        """Kick off a rebuild when the delta grew past the threshold."""
        self.poll_rebuild()
        if self._rebuild_event is not None or self.delta_len == 0:
            return
        if self.delta_len < self.rebuild_threshold:
            return
        self._launch_rebuild()

    def _launch_rebuild(self) -> None:
        if self._rebuild_event is not None:
            return
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
        dst.fill_(PAD_TOKEN)
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

        # Staging owns a copy of the snapshot; compact it out of the
        # delta right away so appends always see the full capacity.
        docs_absorbed = self._count_docs_within(snap)
        remaining = self.delta_len - snap
        if remaining > 0:
            self.delta[:remaining] = self.delta[snap:self.delta_len].clone()
        self.delta_len = remaining
        self.delta_len_t.fill_(remaining)
        self.delta_doc_lens = self.delta_doc_lens[docs_absorbed:]

        # Build over the real prefix plus one sentinel column so the
        # build-time suffix order matches query-time comparisons.
        m = min(n_new + 1, self.capacity)
        if self.device.type == "cuda" and self.rebuild_stream is not None:
            self.rebuild_stream.wait_stream(
                torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.rebuild_stream):
                self._build_staging_sa(dst, m)
            event = torch.cuda.Event()
            event.record(self.rebuild_stream)
            self._pending = (n_new, new_doc_lens)
            self._rebuild_event = event
        else:
            self._build_staging_sa(dst, m)
            self._finish_swap(n_new, new_doc_lens)

    def _build_staging_sa(self, dst: torch.Tensor, m: int) -> None:
        self.staging_sa[:m] = build_suffix_array(dst[:m])
        if m < self.capacity:
            # All-sentinel suffixes compare greater than any real
            # pattern, so any internal order is valid for search.
            self.staging_sa[m:] = torch.arange(
                m, self.capacity, dtype=torch.int64, device=self.device)

    def _count_docs_within(self, tokens: int) -> int:
        acc = 0
        count = 0
        for ln in self.delta_doc_lens:
            acc += ln
            if acc > tokens:
                break
            count += 1
        return count

    def _finish_swap(self, n_new: int, new_doc_lens: deque[int]) -> None:
        # In-place copy instead of a reference swap: captured CUDA
        # graphs (and compiled propose paths) bind the active tensors'
        # storage, so the active buffers must keep their identity.
        self.corpus.copy_(self.staging_corpus)
        self.sa.copy_(self.staging_sa)
        self.active_len = n_new
        self.active_doc_lens = new_doc_lens
        self._rebuild_event = None
        self._pending = None

    def poll_rebuild(self) -> None:
        """Swap in a finished background rebuild (non-blocking)."""
        if self._rebuild_event is None:
            return
        if self._rebuild_event.query():
            n_new, new_doc_lens = self._pending
            self._finish_swap(n_new, new_doc_lens)

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

        Pure tensor ops (compile-safe): callers on the host side are
        responsible for calling poll_rebuild() to swap in finished
        rebuilds; the write paths (append_documents/maybe_rebuild)
        already do.

        Args:
            query: [B, P] int32 padded query tails.
            query_len: [B] int64 valid lengths.
            max_len: maximum match length.

        Returns:
            (match_len [B] i64, cont [B, R, k] int32 padded with -1,
             occ_count [B] i64). Occurrences come from either the SA or
            the delta, whichever matched longer (ties prefer the SA).
        """
        b = query.shape[0]
        r = self.max_occurrences
        device = self.device

        sa_len, sa_start, sa_end = longest_suffix_match(
            self.sa, self.corpus, query, query_len, max_len)
        d_len, d_pos, d_cnt = self._delta_match(query, query_len, max_len)

        use_sa = (sa_len >= d_len) & (sa_len > 0)
        use_delta = (d_len > sa_len) & (d_len > 0)
        zero = torch.zeros(b, dtype=torch.int64, device=device)
        match_len = torch.where(use_sa, sa_len,
                                torch.where(use_delta, d_len, zero))

        # Sample the SA interval: sequential when it fits, strided
        # otherwise (the interval is ordered by continuation, so taking
        # a prefix would bias the vote toward small token ids).
        offs = torch.arange(r, dtype=torch.int64, device=device)
        ilen = sa_end - sa_start
        strided = sa_start.unsqueeze(1) + (offs.unsqueeze(0)
                                           * ilen.unsqueeze(1)) // r
        seq = sa_start.unsqueeze(1) + offs.unsqueeze(0)
        sa_idx = torch.where((ilen > r).unsqueeze(1), strided, seq).clamp(
            0, self.capacity - 1)
        sa_cnt = torch.minimum(ilen, torch.full_like(ilen, r))
        take = offs.unsqueeze(0) < sa_cnt.unsqueeze(1)
        pos = self.sa[sa_idx.reshape(-1)].reshape(b, r)
        sa_occ = torch.where(take, pos, 0)
        sa_cont = self._gather_cont(self.corpus, self.capacity, sa_occ,
                                    sa_cnt, match_len, use_sa)
        d_cont = self._gather_cont(self.delta, self.delta_len_t, d_pos,
                                   d_cnt, match_len, use_delta)

        cont = torch.full((b, r, self.k), -1, dtype=torch.int32,
                          device=device)
        cont = torch.where(use_sa.unsqueeze(1).unsqueeze(2), sa_cont, cont)
        cont = torch.where(use_delta.unsqueeze(1).unsqueeze(2), d_cont,
                           cont)
        occ_count = torch.where(use_sa, sa_cnt,
                                torch.where(use_delta, d_cnt, zero))
        return match_len, cont, occ_count

    def draft(
        self,
        query: torch.Tensor,
        query_len: torch.Tensor,
        max_len: int,
        k: int,
        min_token_prob: float = 0.0,
        max_spec_factor: float | None = None,
        max_spec_offset: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Multi-length scored draft over corpus + delta.

        Queries at several capped match lengths (a shorter cap widens
        the occurrence set), expands each candidate chain, scores it
        arctic-style (sum of per-depth chain probabilities, capped by
        floor(max_spec_factor * match_len + max_spec_offset)) and keeps
        the best-scored candidate; exact ties prefer the longer match.

        Returns:
            (chain [B, k] int32, num_valid [B] int64, match_len [B] i64,
             occ_count [B] i64, score [B] f32)
        """
        b = query.shape[0]
        device = self.device
        caps = sorted({max_len, max(2, max_len // 2), max(2, max_len // 4), 2},
                      reverse=True)

        best_chain = torch.full((b, k), -1, dtype=torch.int32, device=device)
        best_emit = torch.zeros(b, dtype=torch.int64, device=device)
        best_len = torch.zeros(b, dtype=torch.int64, device=device)
        best_occ = torch.zeros(b, dtype=torch.int64, device=device)
        best_score = torch.full((b,), -1.0, dtype=torch.float32,
                                device=device)
        for cap_len in caps:
            match_len, cont, occ_count = self.query(query, query_len, cap_len)
            chain, num_valid = expand_chain(
                cont, occ_count, k, min_token_prob=min_token_prob)
            chain = torch.where(match_len.unsqueeze(1) > 0, chain, -1)
            num_valid = torch.where(match_len > 0, num_valid,
                                    torch.zeros_like(num_valid))
            if max_spec_factor is None:
                cap = torch.full_like(match_len, k)
            else:
                cap = (max_spec_factor * match_len.to(torch.float32)
                       + max_spec_offset).floor().to(torch.int64).clamp(min=0)
            num_emit, score = score_chain(cont, occ_count, chain, num_valid,
                                          cap=cap)
            score = torch.where(match_len > 0, score, torch.zeros_like(score))
            # Longer caps are evaluated first, so a strict > keeps the
            # longer match on exact ties (arctic semantics).
            take = score > best_score
            best_chain = torch.where(take.unsqueeze(1),
                                     chain.to(torch.int32), best_chain)
            best_emit = torch.where(take, num_emit, best_emit)
            best_len = torch.where(take, match_len, best_len)
            best_occ = torch.where(take, occ_count, best_occ)
            best_score = torch.where(take, score, best_score)

        slot = torch.arange(k, device=device).unsqueeze(0)
        best_chain = torch.where(slot < best_emit.unsqueeze(1), best_chain,
                                 -1)
        best_score = best_score.clamp(min=0.0)
        return best_chain, best_emit, best_len, best_occ, best_score

    def _gather_cont(
        self,
        src: torch.Tensor,
        src_len: int | torch.Tensor,
        occ_pos: torch.Tensor,
        occ_cnt: torch.Tensor,
        match_len: torch.Tensor,
        sel: torch.Tensor,
    ) -> torch.Tensor:
        """Gather k-token continuations after occurrences in src."""
        b, r = occ_pos.shape
        n = src.shape[0]
        row = torch.arange(r, device=self.device).unsqueeze(0)
        row_active = (row < occ_cnt.unsqueeze(1)) & sel.unsqueeze(1)
        idx = (occ_pos.unsqueeze(2) + match_len.unsqueeze(1).unsqueeze(2)
               + torch.arange(self.k, device=self.device))
        valid = row_active.unsqueeze(2) & (idx < src_len) & (idx >= 0)
        vals = src[idx.clamp(0, n - 1).reshape(b, r * self.k)
                   ].reshape(b, r, self.k)
        ok = valid & (vals != PAD_TOKEN) & (vals >= 0)
        return torch.where(ok, vals, -1)

    def _delta_match(
        self,
        query: torch.Tensor,
        query_len: torch.Tensor,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Longest match within the delta buffer, fully verified.

        One pass computes match_back[i]: the longest common suffix of
        the delta tokens ending at i (exclusive) and each query tail.
        A window of length L starting at pos matches iff
        match_back[pos + L] >= L, so the longest match is a single max.
        Returns (match_len [B], positions [B, C] compacted left in
        position order, count [B] clamped to C).
        """
        b = query.shape[0]
        c = self.max_occurrences
        device = self.device
        cap = self.delta_capacity
        q = query.shape[1]
        n_i = cap + 1

        # Query tail, newest-first; slots past the query length get -3,
        # which never equals delta content (tokens >= 0, SEP -2, left
        # pad -1), so match lengths are capped at query_len.
        offs = torch.arange(max_len, dtype=torch.int64, device=device)
        pat_idx = query_len.unsqueeze(1) - 1 - offs.unsqueeze(0)
        pat = torch.where(
            pat_idx >= 0,
            query.gather(1, pat_idx.clamp(min=0, max=q - 1)),
            torch.full((1, 1), -3, dtype=query.dtype, device=device))

        if triton_kernels.available(self.delta, pat):
            mb = triton_kernels.match_back(self.delta, pat, n_i)
        else:
            lp = torch.cat([
                torch.full((max_len,), -1, dtype=self.delta.dtype,
                           device=device), self.delta])
            acc = torch.ones(b, n_i, dtype=torch.bool, device=device)
            mb = torch.zeros(b, n_i, dtype=torch.int32, device=device)
            one = torch.ones((), dtype=torch.int32, device=device)
            for t in range(max_len):
                seg = lp[max_len - 1 - t:max_len - 1 - t + n_i]
                acc = acc & (seg.unsqueeze(0) == pat[:, t:t + 1])
                mb = mb + acc * one
        pos_i = torch.arange(n_i, dtype=torch.int32, device=device)
        # Window [i - L, i) must lie inside the committed delta region.
        valid_i = pos_i.unsqueeze(0) <= self.delta_len_t.to(torch.int32)
        mb = torch.where(valid_i, mb, torch.zeros_like(mb))

        lo = mb.max(dim=1).values.to(torch.int64)
        mask = valid_i & (mb >= lo.unsqueeze(1).to(torch.int32)) \
            & (lo > 0).unsqueeze(1)
        cnt = mask.sum(dim=1)
        key = pos_i.unsqueeze(0) + (~mask).to(torch.int32) * (n_i + 1)
        width = min(c, n_i)
        top = torch.topk(key, width, dim=1, largest=False).values.to(
            torch.int64)
        occ_end = torch.where(top <= n_i - 1, top, torch.zeros_like(top))
        occ = (occ_end - lo.unsqueeze(1)).clamp(min=0)
        if width < c:
            occ = torch.cat([
                occ, torch.zeros(b, c - width, dtype=torch.int64,
                                 device=device)], dim=1)
        return lo, occ, torch.minimum(cnt, torch.full_like(cnt, c))

    def expand(self, cont: torch.Tensor, occ_count: torch.Tensor,
               min_token_prob: float = 0.0) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
        return expand_chain(cont, occ_count, self.k,
                            min_token_prob=min_token_prob)
