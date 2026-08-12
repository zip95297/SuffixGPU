"""SuffixGPUDrafter: orchestrates local + global drafting.

The local path matches each request's own history; the global path
matches a cross-request suffix index over finished responses. Both
paths draft with multi-length backoff (``num_backoff`` candidate
lengths each) and the two winners compete by arctic-style score
(sum of per-depth chain probabilities).

Draft length is adaptive, mirroring arctic_inference SuffixDecoding:
each candidate chain is truncated to
``max_spec_factor * match_len + max_spec_offset`` tokens, and chain
expansion stops once the estimated chain probability falls below
``min_token_prob``.
"""

from __future__ import annotations

import torch

from suffix_gpu import triton_kernels
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
        max_occurrences: int = 128,
        enable_global: bool = False,
        global_capacity: int = 1 << 22,
        delta_capacity: int = 1 << 16,
        rebuild_threshold: int | None = None,
        rebuild_stream: torch.cuda.Stream | None = None,
        max_spec_factor: float | None = None,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.0,
        num_backoff: int = 8,
    ):
        self.k = k
        self.device = torch.device(device)
        self.max_pattern_len = max_pattern_len
        self.max_spec_factor = max_spec_factor
        self.max_spec_offset = max_spec_offset
        self.min_token_prob = min_token_prob
        self.num_backoff = max(1, int(num_backoff))
        # Local candidates: support thresholds 2^0 .. 2^(C-1) (arctic's
        # length/support Pareto frontier gets denser as C grows).
        support_thresholds = tuple(
            2 ** i for i in range(self.num_backoff))
        # Global candidates: capped lengths halving from the full
        # pattern length, plus a final 2. Distinct values saturate at
        # ~log2(max_pattern_len), so very large C stops adding caps.
        if self.num_backoff == 1:
            caps = [max_pattern_len]
        else:
            caps = [max(2, max_pattern_len >> i)
                    for i in range(self.num_backoff - 1)] + [2]
        caps = sorted(set(caps), reverse=True)
        self._global_caps = torch.tensor(caps, dtype=torch.int64,
                                         device=self.device)
        self.local_kernel = LocalMatchKernel(
            k=k,
            max_pattern_len=max_pattern_len,
            min_match_len=min_match_len,
            max_occurrences=max_occurrences,
            min_token_prob=min_token_prob,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            support_thresholds=support_thresholds,
        ).to(self.device)
        self.global_index: GlobalIndex | None = None
        self._ingested: dict = {}
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

    def poll(self) -> None:
        """Host-side: swap in a finished background rebuild, if any.

        Call once per step outside the (compile-safe) propose path.
        """
        if self.global_index is not None:
            self.global_index.poll_rebuild()

    def update_state(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Device-side batch state update (async-scheduling glue).

        Scatters the previous step's sampled token ids (still on
        device; -1 padded) into the resident token buffer and returns
        the updated per-request token counts. No host synchronization.

        Args:
            num_tokens_no_spec: [B] int32/int64 committed counts
                (pre-update).
            token_ids_gpu: [B, S] int32 resident buffer, updated
                in place.
            sampled_token_ids: [B, T] int32 last-step sampled ids,
                -1 beyond each row's accepted count.
            valid_sampled_tokens_count: optional [B] number of valid
                ids per row; derived from ``!= -1`` when omitted.

        Returns:
            [B] int32 updated token counts (also usable as the next
            ``num_tokens_no_spec``).
        """
        b, s = token_ids_gpu.shape
        t = sampled_token_ids.shape[1]
        base = num_tokens_no_spec.to(torch.int64)
        if valid_sampled_tokens_count is None:
            valid_sampled_tokens_count = (
                sampled_token_ids != -1).sum(dim=1)
        cnt = valid_sampled_tokens_count.to(torch.int64)
        if triton_kernels.available(token_ids_gpu, sampled_token_ids):
            triton_kernels.scatter_append(
                token_ids_gpu, base, cnt, sampled_token_ids)
            return (base + cnt).to(torch.int32)
        # One column per iteration: clamped out-of-range positions would
        # collide across columns in a single scatter and race.
        for j in range(t):
            pos_j = (base + j).clamp(max=s - 1).unsqueeze(1)
            ok_j = ((j < cnt) & (sampled_token_ids[:, j] != -1)
                    & (base + j < s)).unsqueeze(1)
            old_j = token_ids_gpu.gather(1, pos_j)
            val_j = torch.where(
                ok_j, sampled_token_ids[:, j:j + 1].to(
                    token_ids_gpu.dtype), old_j)
            token_ids_gpu.scatter_(1, pos_j, val_j)
        return (base + cnt).to(torch.int32)

    def propose_with_update(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor | None = None,
        max_model_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """update_state + eligibility mask + propose, all on device.

        Rows draft only when they accepted at least one token last step
        (mirrors the partial-prefill skip) and still have room.

        Returns:
            (draft_tokens [B, k] int32, num_valid [B] int32,
             new_num_tokens [B] int32)
        """
        if valid_sampled_tokens_count is None:
            valid_sampled_tokens_count = (
                sampled_token_ids != -1).sum(dim=1)
        new_counts = self.update_state(
            num_tokens_no_spec, token_ids_gpu, sampled_token_ids,
            valid_sampled_tokens_count)
        mask = valid_sampled_tokens_count.to(torch.int64) > 0
        limit = token_ids_gpu.shape[1] if max_model_len is None else \
            min(max_model_len, token_ids_gpu.shape[1])
        mask &= new_counts.to(torch.int64) < limit
        draft, num_valid = self.propose(new_counts, token_ids_gpu, mask)
        return draft, num_valid, new_counts

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
        (local_draft, local_nv, local_len, local_occ,
         local_score) = self.local_kernel(
            num_tokens_no_spec, token_ids_gpu, combined_mask)

        if self.global_index is None:
            draft, num_valid = local_draft, local_nv
        else:
            tails, tail_len = self._gather_tails(num_tokens_no_spec,
                                                 token_ids_gpu)
            g_chain, g_nv, g_len, g_occ, g_score = self.global_index.draft(
                tails.to(torch.int32), tail_len, self.max_pattern_len,
                self.k, min_token_prob=self.min_token_prob,
                max_spec_factor=self.max_spec_factor,
                max_spec_offset=self.max_spec_offset,
                caps=self._global_caps)

            pick_global = (g_score > local_score) & combined_mask
            draft = torch.where(pick_global.unsqueeze(1),
                                g_chain.to(torch.int32), local_draft)
            num_valid = torch.where(pick_global, g_nv.to(torch.int64),
                                    local_nv.to(torch.int64))

        num_valid = torch.where(
            combined_mask, num_valid.to(torch.int64),
            torch.zeros(b, dtype=torch.int64, device=self.device))
        slot = torch.arange(self.k, device=self.device).unsqueeze(0)
        draft = torch.where(slot < num_valid.unsqueeze(1), draft, -1)
        return draft.to(torch.int32), num_valid.to(torch.int32)

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

    def ingest_active(
        self,
        keys: list,
        rows: list[torch.Tensor],
        lengths: list[int],
        final: bool = False,
        chunk: int = 64,
    ) -> None:
        """Incrementally ingest in-flight responses (host-side).

        Cross-request sharing should not wait for request finish (the
        CPU suffix cache ingests responses immediately). Responses are
        fed to the global index in chunks; consecutive chunks overlap
        by max_pattern_len + k tokens so patterns and continuations
        spanning a chunk boundary stay findable.

        Args:
            keys: stable per-request identifiers.
            rows: response-only token rows (device tensors).
            lengths: current response lengths (host ints).
            final: flush remaining tail and forget the request.
            chunk: minimum new tokens before a chunk is emitted.
        """
        if self.global_index is None:
            return
        overlap = self.max_pattern_len + self.k
        docs = []
        for key, row, ln in zip(keys, rows, lengths):
            done = self._ingested.get(key, 0)
            if ln - done >= chunk or (final and ln > done):
                start = max(0, done - overlap)
                docs.append(row[start:ln])
                self._ingested[key] = ln
            if final:
                self._ingested.pop(key, None)
        if docs:
            self.global_index.append_documents(docs)

    def load_model(self, *args, **kwargs) -> None:
        pass
