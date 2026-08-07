"""SuffixGPUDrafter: orchestrates local + global drafting.

The local path matches each request's own history; the global path
matches a cross-request suffix index over finished responses. The two
candidates are scored by (match_len, occurrence_count) per request.
"""

from __future__ import annotations

import torch

from suffix_gpu.expand import expand_chain
from suffix_gpu.global_index import GlobalIndex
from suffix_gpu.local_matcher import LocalMatchKernel


class SuffixGPUDrafter:
    """Device-resident suffix-decoding drafter.

    Mirrors the vLLM NgramProposerGPU contract: device tensors in,
    device tensors out, no host synchronization on the propose path.
    """

    def __init__(
        self,
        k: int,
        device: torch.device | str = "cpu",
        max_pattern_len: int = 32,
        min_match_len: int = 1,
        max_occurrences: int = 32,
        enable_global: bool = False,
        global_capacity: int = 1 << 22,
        delta_capacity: int = 1 << 16,
        rebuild_threshold: int | None = None,
        rebuild_stream: torch.cuda.Stream | None = None,
    ):
        self.k = k
        self.device = torch.device(device)
        self.max_pattern_len = max_pattern_len
        self.local_kernel = LocalMatchKernel(
            k=k,
            max_pattern_len=max_pattern_len,
            min_match_len=min_match_len,
            max_occurrences=max_occurrences,
        ).to(self.device)
        self.global_index: GlobalIndex | None = None
        if enable_global:
            self.global_index = GlobalIndex(
                capacity=global_capacity,
                delta_capacity=delta_capacity,
                k=k,
                max_occurrences=max_occurrences,
                rebuild_threshold=rebuild_threshold,
                device=self.device,
                rebuild_stream=rebuild_stream,
            )

    def _gather_tails(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded [B, P] tails (last P tokens) and their lengths."""
        b, s = token_ids_gpu.shape
        p = self.max_pattern_len
        q_len = num_tokens_no_spec.to(torch.int64)
        tail_len = torch.minimum(q_len, torch.full_like(q_len, p))
        offs = torch.arange(p, dtype=torch.int64, device=self.device)
        idx = q_len.unsqueeze(1) - tail_len.unsqueeze(1) + offs.unsqueeze(0)
        valid = offs.unsqueeze(0) < tail_len.unsqueeze(1)
        tails = torch.where(
            valid, token_ids_gpu.gather(1, idx.clamp(0, s - 1)), 0)
        return tails, tail_len

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        combined_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft tokens from local history and the global index.

        Args:
            num_tokens_no_spec: [B] int32 token counts.
            token_ids_gpu: [B, S] int32 token buffer.
            combined_mask: optional [B] bool rows allowed to draft.

        Returns:
            (draft_tokens [B, k] int32, num_valid_draft_tokens [B] int32)
        """
        b = num_tokens_no_spec.shape[0]
        if combined_mask is None:
            combined_mask = torch.ones(
                b, dtype=torch.bool, device=self.device)
        local_draft, local_nv, local_len, local_occ = self.local_kernel(
            num_tokens_no_spec, token_ids_gpu, combined_mask)
        if self.global_index is None:
            return local_draft, local_nv

        tails, tail_len = self._gather_tails(num_tokens_no_spec,
                                             token_ids_gpu)
        g_len, cont, occ_cnt = self.global_index.query(
            tails.to(torch.int32), tail_len, self.max_pattern_len)
        g_chain, g_nv = expand_chain(cont, occ_cnt, self.k)
        g_chain = torch.where(g_len.unsqueeze(1) > 0, g_chain, -1)
        g_nv = torch.where(g_len > 0, g_nv, torch.zeros_like(g_nv))

        pick_global = ((g_len > local_len.to(torch.int64))
                       | ((g_len == local_len.to(torch.int64))
                          & (occ_cnt > local_occ)
                          & (g_len > 0))) & combined_mask
        draft = torch.where(pick_global.unsqueeze(1),
                            g_chain.to(torch.int32), local_draft)
        num_valid = torch.where(pick_global, g_nv.to(torch.int32),
                                local_nv)
        num_valid = torch.where(
            combined_mask, num_valid, torch.zeros_like(num_valid))
        return draft, num_valid

    def harvest_finished(
        self,
        row_indices: list[int],
        lengths: list[int],
        token_ids_gpu: torch.Tensor,
    ) -> None:
        """Ingest finished requests' tokens into the global index.

        Args:
            row_indices: rows of `token_ids_gpu` holding finished reqs.
            lengths: host-side token counts for those rows.
            token_ids_gpu: [M, S] int32 resident token buffer.
        """
        rows = [token_ids_gpu[r] for r in row_indices]
        self.harvest_rows(rows, lengths)

    def harvest_rows(self, rows: list[torch.Tensor],
                      lengths: list[int]) -> None:
        """Ingest pre-sliced token rows into the global index."""
        if self.global_index is None or not rows:
            return
        docs = [row[:ln] for row, ln in zip(rows, lengths) if ln > 0]
        if docs:
            self.global_index.append_documents(docs)

    def load_model(self, *args, **kwargs) -> None:
        pass
