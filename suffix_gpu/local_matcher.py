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
                 support_thresholds: Sequence[int] = (1, 2, 4, 8)):
        super().__init__()
        self.k = k
        self.max_pattern_len = max_pattern_len
        self.min_match_len = max(1, min_match_len)
        self.max_occurrences = max_occurrences
        self.min_token_prob = min_token_prob
        self.max_spec_factor = max_spec_factor
        self.max_spec_offset = max_spec_offset
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
        b, s = token_ids.shape
        k = self.k
        r = self.max_occurrences
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
        cont_all = torch.where(valid_idx, vals, -1).reshape(b * c, r, k)
        occ_all = occ_count.reshape(b * c)

        cap = self._spec_cap(cand.reshape(b * c))
        chain, _, num_emit, score = expand_chain(
            cont_all, occ_all, k, min_token_prob=self.min_token_prob,
            cap=cap)

        score = score.reshape(b, c)
        num_emit = num_emit.reshape(b, c)
        chain = chain.reshape(b, c, k)
        occ_all = occ_all.reshape(b, c)
        # Prefer longer matches on exact score ties (arctic iterates
        # lengths ascending with a >= comparison, so longer wins).
        tie = cand.to(torch.float32) * 1e-6
        best = (score + tie).argmax(dim=1)

        bidx = torch.arange(b, device=device)
        sel_chain = chain[bidx, best]
        sel_emit = num_emit[bidx, best]
        sel_len = cand[bidx, best]
        sel_occ = occ_all[bidx, best]
        sel_score = score[bidx, best]

        slot_k = torch.arange(k, device=device).unsqueeze(0)
        sel_chain = torch.where(slot_k < sel_emit.unsqueeze(1), sel_chain,
                                -1)
        return (sel_chain.to(torch.int32), sel_emit.to(torch.int32),
                sel_len, sel_occ, sel_score)
