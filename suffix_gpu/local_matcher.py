"""Per-request variable-length suffix matcher.

Finds, for each request, the longest suffix of its own token history that
already occurred earlier in the same request, then drafts the continuation
by frequency-ranked majority vote over the earliest occurrences.

Occurrence semantics mirror vLLM's NgramGPUKernel: an occurrence must
start before the tail and leave at least one committed continuation token
(pos + L < q_len); overlapping the tail is allowed, which is what makes
periodic repetition draftable.

Implementation notes (all pure torch ops, static loop bounds):
- Match masks are computed by a rolling AND over pattern offsets on a
  padded buffer, so no [B, W, L] window tensor is ever materialized.
- The best match length is found by binary search: the predicate "the
  length-L tail occurs earlier" is monotone in L (any occurrence of the
  length-L tail contains an occurrence of the length-(L-1) tail).
- Occurrence positions are extracted once, at the final length, with a
  single topk (earliest positions first, compacted left).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from suffix_gpu.expand import expand_chain


class LocalMatchKernel(nn.Module):
    """Longest-suffix local matcher with frequency-ranked expansion."""

    def __init__(self, k: int, max_pattern_len: int = 32,
                 min_match_len: int = 1, max_occurrences: int = 32,
                 min_token_prob: float = 0.0):
        super().__init__()
        self.k = k
        self.max_pattern_len = max_pattern_len
        self.min_match_len = max(1, min_match_len)
        self.max_occurrences = max_occurrences
        self.min_token_prob = min_token_prob

    def _match_mask(
        self,
        padded: torch.Tensor,
        q_len: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        """Positions whose window equals the per-row trailing suffix.

        Args:
            padded: [B, S + P] token buffer right-padded with -1.
            q_len: [B] i64 committed token counts.
            length: [B] i64 per-row pattern lengths (0 disables the row).

        Returns:
            [B, S] bool mask over window start positions.
        """
        b = padded.shape[0]
        s = padded.shape[1] - self.max_pattern_len
        device = padded.device
        pos = torch.arange(s, dtype=torch.int64, device=device)
        m = (pos.unsqueeze(0) + length.unsqueeze(1)) < q_len.unsqueeze(1)
        m &= ((length > 0) & (length <= q_len)).unsqueeze(1)
        for j in range(self.max_pattern_len):
            need = length > j
            pat_idx = (q_len - length + j).clamp(0, s - 1)
            pat_j = padded.gather(1, pat_idx.unsqueeze(1))
            m &= (padded[:, j:j + s] == pat_j) | ~need.unsqueeze(1)
        return m

    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match and draft on per-request token buffers.

        Args:
            num_tokens_no_spec: [B] int32 token counts.
            token_ids: [B, S] int32 token buffer.
            combined_mask: [B] bool rows allowed to draft.

        Returns:
            (draft_tokens [B, k] int32, num_valid [B] int32,
             match_len [B] i64, occ_count [B] i64)
        """
        b, s = token_ids.shape
        k = self.k
        r = self.max_occurrences
        p = self.max_pattern_len
        device = token_ids.device
        q_len = num_tokens_no_spec.to(torch.int64)
        padded = F.pad(token_ids, (0, p), value=-1)

        lo = torch.zeros(b, dtype=torch.int64, device=device)
        hi = torch.full((b,), p, dtype=torch.int64, device=device)
        iters = max(1, math.ceil(math.log2(p + 1)))
        for _ in range(iters):
            mid = (lo + hi + 1) // 2
            found = self._match_mask(padded, q_len, mid).any(dim=1)
            lo = torch.where(found, mid, lo)
            hi = torch.where(found, hi, mid - 1)
        best_len = torch.where(lo >= self.min_match_len, lo,
                               torch.zeros_like(lo))
        best_len = torch.where(combined_mask, best_len,
                               torch.zeros_like(best_len))

        mask = self._match_mask(padded, q_len, best_len)
        occ_count = torch.minimum(
            mask.sum(dim=1),
            torch.full((b,), r, dtype=torch.int64, device=device))
        pos = torch.arange(s, dtype=torch.int64, device=device)
        key = pos.unsqueeze(0) + (~mask).to(torch.int64) * (s + 1)
        width = min(r, s)
        top = torch.topk(key, width, dim=1, largest=False).values
        occ_pos = torch.where(top <= s - 1, top, torch.zeros_like(top))
        if width < r:
            occ_pos = torch.cat(
                [occ_pos, torch.zeros(b, r - width, dtype=torch.int64,
                                      device=device)], dim=1)

        offs_k = torch.arange(k, dtype=torch.int64, device=device)
        idx = occ_pos.unsqueeze(2) + best_len.unsqueeze(1).unsqueeze(2) \
            + offs_k
        row_active = (
            torch.arange(r, device=device).unsqueeze(0) <
            occ_count.unsqueeze(1)).unsqueeze(2)
        valid_idx = (idx < q_len.unsqueeze(1).unsqueeze(2)) & row_active
        vals = token_ids.gather(
            1, idx.clamp(0, s - 1).reshape(b, r * k)).reshape(b, r, k)
        cont = torch.where(valid_idx, vals, -1)

        chain, num_valid = expand_chain(
            cont, occ_count, k, min_token_prob=self.min_token_prob)
        return (chain.to(torch.int32), num_valid.to(torch.int32),
                best_len, occ_count)
