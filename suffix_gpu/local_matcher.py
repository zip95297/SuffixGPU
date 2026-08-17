"""Per-request variable-length suffix matcher with multi-length backoff.

Finds, for each request, candidate suffix-match lengths of its own token
history (the longest match plus shorter, better-supported fallbacks),
drafts a continuation chain for each candidate by frequency-ranked
majority vote, and keeps the candidate with the best arctic-style score
(sum of per-depth chain probabilities = expected accepted tokens).

Occurrence semantics mirror vLLM's NgramGPUKernel: an occurrence must
start before the tail and leave at least one committed continuation token
(pos + L < q_len); overlapping the tail is allowed, which is what makes
periodic repetition draftable.

Why backoff: the longest match is often rare (one occurrence with an
idiosyncratic continuation), while a shorter suffix has many occurrences
whose continuations agree. Arctic walks every match length and keeps the
best-scored draft; matching only the longest length systematically
under-drafts on counter/templated traffic.

Implementation notes (all pure torch ops, static loop bounds; every
candidate rides the same batched kernels, so the marginal cost of an
extra support threshold is small):
- One pass computes match_back[i]: the length of the longest common
  suffix between the tokens ending at i (exclusive) and the request
  tail. A window of length L starting at pos matches the length-L tail
  iff match_back[pos + L] >= L.
- The candidate for support threshold t is the largest L with
  cnt(L) = #{i : match_back[i] >= L} >= t, which is exactly the t-th
  largest match_back value: one topk over match_back yields every
  candidate at once.
- All candidates extract their earliest occurrences together with one
  batched smallest-k topk over position keys, gather continuations in
  one shot, and share a single fused expand+score call capped by
  floor(max_spec_factor * L + max_spec_offset).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from suffix_gpu import triton_kernels
from suffix_gpu.expand import expand_chain


class LocalMatchKernel(nn.Module):
    """Multi-length local suffix matcher with score-ranked expansion."""

    def __init__(self, k: int, max_pattern_len: int = 32,
                 min_match_len: int = 1, max_occurrences: int = 32,
                 min_token_prob: float = 0.0,
                 max_spec_factor: float | None = None,
                 max_spec_offset: float = 0.0,
                 support_thresholds: Sequence[int] = (1, 2, 4, 8),
                 vote_smoothing_alpha: float = 0.0,
                 local_mode: str = "backoff",
                 soft_lambda: float = 0.5):
        super().__init__()
        self.k = k
        self.max_pattern_len = max_pattern_len
        self.min_match_len = max(1, min_match_len)
        self.max_occurrences = max_occurrences
        self.min_token_prob = min_token_prob
        self.max_spec_factor = max_spec_factor
        self.max_spec_offset = max_spec_offset
        self.vote_smoothing_alpha = vote_smoothing_alpha
        if local_mode not in ("backoff", "soft"):
            raise ValueError(f"unknown local_mode {local_mode!r}")
        self.local_mode = local_mode
        self.soft_lambda = float(soft_lambda)
        self.support_thresholds = tuple(int(t) for t in support_thresholds)
        if not self.support_thresholds:
            raise ValueError("support_thresholds must be non-empty")
        self.register_buffer(
            "_thresholds",
            torch.tensor(self.support_thresholds, dtype=torch.int64),
            persistent=False)

    def _spec_cap(self, match_len: torch.Tensor) -> torch.Tensor:
        if self.max_spec_factor is None:
            return torch.full_like(match_len, self.k)
        cap = (self.max_spec_factor * match_len.to(torch.float32)
               + self.max_spec_offset).floor().to(torch.int64)
        return cap.clamp(min=0)

    def _compute_match_back(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared match-back scan: (mb, q_len, valid_i, pos)."""
        b, s = token_ids.shape
        p = self.max_pattern_len
        device = token_ids.device
        q_len = num_tokens_no_spec.to(torch.int64)

        # Tail pattern, newest-first: pat[b, t] = token[q_len - 1 - t].
        # Slots past the row's history get -2, which matches nothing
        # (buffer tokens are >= 0, left padding is -1), so match_back
        # is automatically capped at q_len.
        offs_p = torch.arange(p, dtype=torch.int64, device=device)
        pat_idx = q_len.unsqueeze(1) - 1 - offs_p.unsqueeze(0)
        pat = torch.where(
            pat_idx >= 0,
            token_ids.gather(1, pat_idx.clamp(min=0, max=s - 1)),
            torch.full((1, 1), -2, dtype=token_ids.dtype, device=device))

        # match_back[b, i] = max L such that the L tokens ending at i
        # (exclusive) equal the length-L tail.
        if triton_kernels.available(token_ids, pat):
            mb = triton_kernels.match_back(token_ids, pat, s)
        else:
            lp = F.pad(token_ids, (p, 0), value=-1)
            acc = torch.ones(b, s, dtype=torch.bool, device=device)
            mb = torch.zeros(b, s, dtype=torch.int32, device=device)
            one = torch.ones((), dtype=torch.int32, device=device)
            for t in range(p):
                seg = lp[:, p - 1 - t:p - 1 - t + s]
                acc = acc & (seg == pat[:, t:t + 1])
                mb = mb + acc * one
        pos = torch.arange(s, dtype=torch.int32, device=device)
        # End position i = pos + L must satisfy i < q_len: the
        # occurrence starts before the tail and leaves at least one
        # committed continuation token.
        valid_i = pos.unsqueeze(0) < q_len.unsqueeze(1).to(torch.int32)
        mb = torch.where(valid_i, mb, torch.zeros_like(mb))
        return mb, q_len, valid_i, pos

    def _gather_cont_at(self, token_ids: torch.Tensor,
                        q_len: torch.Tensor,
                        occ_end: torch.Tensor,
                        row_active: torch.Tensor) -> torch.Tensor:
        """Gather k continuation tokens after flat occurrence ends.

        occ_end/row_active: [B, N]; returns [B, N, K] int32, -1 padded.
        """
        b, s = token_ids.shape
        k = self.k
        n = occ_end.shape[1]
        offs_k = torch.arange(k, dtype=torch.int64, device=occ_end.device)
        idx = occ_end.unsqueeze(2) + offs_k  # [B, N, K]
        valid_idx = (idx < q_len.view(b, 1, 1)) & row_active.unsqueeze(2)
        vals = token_ids.gather(
            1, idx.clamp(0, s - 1).reshape(b, n * k)).reshape(b, n, k)
        return torch.where(valid_idx, vals, -1)

    def gather_soft(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft backoff: one weighted ensemble over all match lengths.

        Instead of C hard (length, support) candidates, take the R
        best occurrence sites ranked by (match length desc, position
        asc) and let every site vote with weight
        ``soft_lambda ** (L_max - mb[site])`` — an interpolated
        backoff over all suffix orders at once (single expand, C=1).

        Returns:
            (match_len [B, 1] i64 = longest match,
             cont [B, 1, R, K] int32, occ_count [B, 1] i64,
             weights [B, R] f32).
        """
        b, s = token_ids.shape
        r = self.max_occurrences
        device = token_ids.device
        mb, q_len, valid_i, pos = self._compute_match_back(
            num_tokens_no_spec, token_ids)

        eligible = (mb >= self.min_match_len) & valid_i \
            & combined_mask.unsqueeze(1)
        mb_e = torch.where(eligible, mb, torch.zeros_like(mb))
        # Rank: longer match first, earlier position first within a
        # length. All eligible keys exceed every ineligible key.
        key = (mb_e.to(torch.int64) * (s + 1)
               + (s - 1 - pos.to(torch.int64)).unsqueeze(0))
        width = min(r, s)
        top_key = torch.topk(key, width, dim=1).values  # [B, width]
        if width < r:
            top_key = torch.cat(
                [top_key, torch.zeros(b, r - width, dtype=torch.int64,
                                      device=device)], dim=1)
        top_mb = top_key // (s + 1)                    # [B, R]
        top_pos = (s - 1) - (top_key % (s + 1))
        n_occ = eligible.sum(dim=1).clamp(max=r)       # [B]
        row_active = (torch.arange(r, device=device).view(1, r)
                      < n_occ.unsqueeze(1))
        occ_end = torch.where(row_active, top_pos,
                              torch.zeros_like(top_pos))

        match_len = top_mb[:, 0].clamp(min=0)          # longest match
        weights = torch.pow(
            torch.full((), self.soft_lambda, dtype=torch.float32,
                       device=device),
            (match_len.unsqueeze(1) - top_mb).to(torch.float32).clamp(
                min=0))
        weights = torch.where(row_active, weights,
                              torch.zeros_like(weights))

        cont = self._gather_cont_at(token_ids, q_len, occ_end,
                                    row_active)
        return (match_len.unsqueeze(1), cont.unsqueeze(1),
                n_occ.unsqueeze(1), weights)

    def gather_candidates(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backoff candidate lengths with continuation blocks.

        Returns:
            (cand [B, C] i64 candidate match lengths,
             cont [B, C, R, K] int32 continuations (-1 padded),
             occ_count [B, C] i64 occurrence counts).
        """
        b, s = token_ids.shape
        k = self.k
        r = self.max_occurrences
        device = token_ids.device
        mb, q_len, valid_i, pos = self._compute_match_back(
            num_tokens_no_spec, token_ids)

        # Candidate lengths: the largest L with cnt(L) >= t occurrences
        # is exactly the t-th largest match_back value (cnt(L) =
        # #{i : mb[i] >= L} is non-increasing in L), so one topk over
        # mb yields every support threshold's candidate at once.
        if self._thresholds.device != device:
            self._thresholds = self._thresholds.to(device)
        w_thr = min(max(self.support_thresholds), s)
        top_mb = torch.topk(mb, w_thr, dim=1).values  # [B, w] descending
        tidx = (self._thresholds - 1).clamp(min=0, max=w_thr - 1)
        cand = top_mb.gather(1, tidx.view(1, -1).expand(b, -1)).to(
            torch.int64)
        # Thresholds beyond the buffer width can never be met.
        cand = torch.where(self._thresholds.view(1, -1) > w_thr,
                           torch.zeros_like(cand), cand)
        cand = torch.where(
            (cand >= self.min_match_len) & combined_mask.unsqueeze(1),
            cand, torch.zeros_like(cand))
        # Adjacent duplicates draft identical chains (same length =>
        # same occurrence set); zeroing them skips their occurrence
        # scans while argmax still picks the surviving first copy.
        if cand.shape[1] > 1:
            dup = torch.zeros_like(cand, dtype=torch.bool)
            dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
            cand = torch.where(dup, torch.zeros_like(cand), cand)
        c = cand.shape[1]

        # Earliest occurrence end positions for all candidates: one
        # early-exit scan kernel on CUDA, else one batched smallest-k
        # topk over position keys (matching positions keep their
        # position, others sort past the end).
        if triton_kernels.available(mb, cand):
            occ_end, occ_count = triton_kernels.first_occurrences(
                mb, cand.reshape(b * c), c, r)
            occ_end = occ_end.reshape(b, c, r)
            occ_count = occ_count.reshape(b, c)
        else:
            mask = (valid_i.unsqueeze(1)
                    & (mb.unsqueeze(1) >= cand.unsqueeze(2).to(torch.int32))
                    & (cand > 0).unsqueeze(2))  # [B, C, S]
            occ_count = mask.sum(dim=2).clamp(max=r)  # [B, C]
            key = pos.view(1, 1, s) + (~mask).to(torch.int32) * (s + 1)
            width = min(r, s)
            top = torch.topk(key.reshape(b * c, s), width, dim=1,
                             largest=False).values.to(torch.int64)
            occ_end = torch.where(top <= s - 1, top, torch.zeros_like(top))
            if width < r:
                occ_end = torch.cat(
                    [occ_end, torch.zeros(b * c, r - width,
                                          dtype=torch.int64,
                                          device=device)], dim=1)
            occ_end = occ_end.reshape(b, c, r)

        # Continuation gather for all candidates in one shot.
        offs_k = torch.arange(k, dtype=torch.int64, device=device)
        idx = occ_end.unsqueeze(3) + offs_k  # [B, C, R, K]
        row_active = (torch.arange(r, device=device).view(1, 1, r)
                      < occ_count.unsqueeze(2)).unsqueeze(3)
        valid_idx = (idx < q_len.view(b, 1, 1, 1)) & row_active
        vals = token_ids.gather(
            1, idx.clamp(0, s - 1).reshape(b, c * r * k)).reshape(
                b, c, r, k)
        cont = torch.where(valid_idx, vals, -1)
        return cand, cont, occ_count

    def gather(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor | None]:
        """Mode dispatch: (len [B,C], cont [B,C,R,K], occ [B,C],
        weights [B, C*R] f32 or None)."""
        if self.local_mode == "soft":
            cand, cont, occ, w = self.gather_soft(
                num_tokens_no_spec, token_ids, combined_mask)
            return cand, cont, occ, w
        cand, cont, occ = self.gather_candidates(
            num_tokens_no_spec, token_ids, combined_mask)
        return cand, cont, occ, None

    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Match and draft on per-request token buffers.

        Args:
            num_tokens_no_spec: [B] int32 token counts.
            token_ids: [B, S] int32 token buffer.
            combined_mask: [B] bool rows allowed to draft.

        Returns:
            (draft_tokens [B, k] int32, num_valid [B] int32,
             match_len [B] i64, occ_count [B] i64, score [B] f32)
        """
        b, _ = token_ids.shape
        k = self.k
        r = self.max_occurrences
        device = token_ids.device
        cand, cont, occ_count, weights = self.gather(
            num_tokens_no_spec, token_ids, combined_mask)
        c = cand.shape[1]
        cont_all = cont.reshape(b * c, r, k)
        occ_all = occ_count.reshape(b * c)
        w_all = None if weights is None else weights.reshape(b * c, r)

        cap = self._spec_cap(cand.reshape(b * c))
        chain, _, num_emit, score = expand_chain(
            cont_all, occ_all, k, min_token_prob=self.min_token_prob,
            cap=cap, weights=w_all, alpha=self.vote_smoothing_alpha)

        score = score.reshape(b, c)
        num_emit = num_emit.reshape(b, c)
        chain = chain.reshape(b, c, k)
        occ_all = occ_all.reshape(b, c)
        sel_chain, sel_emit, sel_len, sel_occ, sel_score = \
            select_local_best(cand, chain, num_emit, occ_all, score)

        slot_k = torch.arange(k, device=device).unsqueeze(0)
        sel_chain = torch.where(slot_k < sel_emit.unsqueeze(1), sel_chain,
                                -1)
        return (sel_chain.to(torch.int32), sel_emit.to(torch.int32),
                sel_len, sel_occ, sel_score)


def select_local_best(
    cand: torch.Tensor,
    chain: torch.Tensor,
    num_emit: torch.Tensor,
    occ: torch.Tensor,
    score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Pick the best-scored local candidate per request.

    Prefer longer matches on exact score ties (arctic iterates lengths
    ascending with a >= comparison, so longer wins).
    """
    b = cand.shape[0]
    tie = cand.to(torch.float32) * 1e-6
    best = (score + tie).argmax(dim=1)
    bidx = torch.arange(b, device=cand.device)
    return (chain[bidx, best], num_emit[bidx, best], cand[bidx, best],
            occ[bidx, best], score[bidx, best])
