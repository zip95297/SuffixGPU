"""Per-request variable-length suffix matcher.

Finds, for each request, the longest suffix of its own token history that
already occurred earlier in the same request, then drafts the continuation
by frequency-ranked majority vote over the earliest occurrences.

Occurrence semantics mirror vLLM's NgramGPUKernel: an occurrence must
start before the tail and leave at least one committed continuation token
(pos + L < q_len); overlapping the tail is allowed, which is what makes
periodic repetition draftable.

Implementation notes (all pure torch ops, static loop bounds):
- One pass computes match_back[i]: the length of the longest common
  suffix between the tokens ending at i (exclusive) and the request
  tail. A window of length L starting at pos matches the length-L tail
  iff match_back[pos + L] >= L, so the longest drafteable length is
  simply max(match_back[i]) over valid end positions - no per-length
  binary search.
- Occurrence positions are extracted once, at the final length, with a
  single topk (earliest positions first, compacted left).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from suffix_gpu import triton_kernels
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
        # (exclusive) equal the length-L tail. Rolling AND over pattern
        # offsets on a left-padded buffer; the cumulative sum of the
        # running AND is exactly the run length.
        # int32 intermediates: [B, S] buffers dominate transient
        # memory, and lengths/positions always fit in 32 bits.
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

        best_len = mb.max(dim=1).values.to(torch.int64)
        best_len = torch.where(
            (best_len >= self.min_match_len) & combined_mask,
            best_len, torch.zeros_like(best_len))

        # All windows matching at the final length (mb >= L contains an
        # occurrence of the length-L tail ending at i).
        mask = valid_i & (mb >= best_len.unsqueeze(1).to(torch.int32)) \
            & (best_len > 0).unsqueeze(1)
        occ_count = torch.minimum(
            mask.sum(dim=1),
            torch.full((b,), r, dtype=torch.int64, device=device))
        key = pos.unsqueeze(0) + (~mask).to(torch.int32) * (s + 1)
        width = min(r, s)
        top = torch.topk(key, width, dim=1, largest=False).values.to(
            torch.int64)
        occ_end = torch.where(top <= s - 1, top, torch.zeros_like(top))
        if width < r:
            occ_end = torch.cat(
                [occ_end, torch.zeros(b, r - width, dtype=torch.int64,
                                      device=device)], dim=1)

        # Continuations start at the match end position i.
        offs_k = torch.arange(k, dtype=torch.int64, device=device)
        idx = occ_end.unsqueeze(2) + offs_k
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
