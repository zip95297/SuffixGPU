"""Per-request variable-length suffix matcher.

Finds, for each request, the longest suffix of its own token history that
already occurred earlier in the same request, then drafts the continuation
by frequency-ranked majority vote over all occurrences.

Pure torch ops with static loop bounds (torch.compile friendly), mirroring
the vLLM NgramGPUKernel contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from suffix_gpu.expand import expand_chain


class LocalMatchKernel(nn.Module):
    """Longest-suffix local matcher with frequency-ranked expansion."""

    def __init__(self, k: int, max_pattern_len: int = 32,
                 min_match_len: int = 1, max_occurrences: int = 32):
        super().__init__()
        self.k = k
        self.max_pattern_len = max_pattern_len
        self.min_match_len = max(1, min_match_len)
        self.max_occurrences = max_occurrences

    def _matches(
        self, token_ids: torch.Tensor, q_len: torch.Tensor, length: int,
    ) -> torch.Tensor:
        """Match mask [B, S-L+1] for windows equal to the trailing suffix."""
        b, s = token_ids.shape
        device = token_ids.device
        if s - length + 1 <= 0:
            return torch.zeros(b, 0, dtype=torch.bool, device=device)
        pos = torch.arange(s - length + 1, dtype=torch.int64, device=device)
        offs = torch.arange(length, dtype=torch.int64, device=device)
        tail_idx = q_len.unsqueeze(1) - length + offs.unsqueeze(0)
        tail_valid = tail_idx >= 0
        pattern = token_ids.gather(1, tail_idx.clamp(0, s - 1))
        win_idx = (pos.unsqueeze(1) + offs.unsqueeze(0)).clamp(0, s - 1)
        win = token_ids[:, win_idx]
        eq = (win == pattern.unsqueeze(1)) & tail_valid.unsqueeze(1)
        match = eq.all(dim=2)
        within = (pos + length).unsqueeze(0) <= q_len.unsqueeze(1)
        not_self = pos.unsqueeze(0) != (q_len - length).unsqueeze(1)
        return match & within & not_self

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
             match_len [B] int64, occ_count [B] int64)
        """
        b, s = token_ids.shape
        k = self.k
        r = self.max_occurrences
        device = token_ids.device
        q_len = num_tokens_no_spec.to(torch.int64)
        best_len = torch.zeros(b, dtype=torch.int64, device=device)
        counts = {
            length: self._matches(token_ids, q_len, length).sum(dim=1)
            for length in range(self.min_match_len, self.max_pattern_len + 1)
        }
        for length in range(self.min_match_len, self.max_pattern_len + 1):
            cand = combined_mask & (counts[length] > 0)
            best_len = torch.where(cand, length, best_len)

        occ_pos = torch.zeros(b, r, dtype=torch.int64, device=device)
        occ_count = torch.zeros(b, dtype=torch.int64, device=device)
        cont = torch.full((b, r, k), -1, dtype=token_ids.dtype, device=device)
        offs_k = torch.arange(k, dtype=torch.int64, device=device)
        big = s + 1
        for length in range(self.min_match_len, self.max_pattern_len + 1):
            w = s - length + 1
            if w <= 0:
                continue
            sel = (best_len == length) & combined_mask
            match = self._matches(token_ids, q_len, length)
            pos_l = torch.arange(w, dtype=torch.int64, device=device)
            # Earliest occurrences first: sort by (non-match, position).
            key = pos_l.unsqueeze(0) + (~match).to(torch.int64) * big
            order = torch.argsort(key, dim=1)
            width = min(w, r)
            top_pos = pos_l.unsqueeze(0).expand(b, w).gather(
                1, order[:, :width])
            if width < r:
                pad = torch.zeros(b, r - width, dtype=torch.int64,
                                  device=device)
                top_pos = torch.cat([top_pos, pad], dim=1)
            occ_pos = torch.where(sel.unsqueeze(1), top_pos, occ_pos)
            occ_count = torch.where(
                sel, torch.minimum(counts[length],
                                   torch.full_like(occ_count, r)), occ_count)
            idx = occ_pos.unsqueeze(2) + length + offs_k
            row_active = (
                torch.arange(r, device=device).unsqueeze(0) <
                occ_count.unsqueeze(1)).unsqueeze(2)
            valid_idx = (idx < q_len.unsqueeze(1).unsqueeze(2)) & row_active
            vals = token_ids.gather(
                1, idx.clamp(0, s - 1).reshape(b, r * k)).reshape(b, r, k)
            cont_cur = torch.where(valid_idx, vals, -1)
            cont = torch.where(sel.unsqueeze(1).unsqueeze(2), cont_cur, cont)

        chain, num_valid = expand_chain(cont, occ_count, k)
        chain = torch.where(best_len.unsqueeze(1) > 0, chain, -1)
        num_valid = torch.where(best_len > 0, num_valid,
                                torch.zeros_like(num_valid))
        num_valid = torch.where(
            combined_mask, num_valid, torch.zeros_like(num_valid))
        return (chain.to(torch.int32), num_valid.to(torch.int32),
                best_len, occ_count)
