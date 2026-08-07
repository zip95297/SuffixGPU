"""SuffixGPUDrafter: orchestrates local + global drafting.

Phase-1 drafter. The local path (per-request history) is always active;
the global suffix-array path is added in M3 via `enable_global`.
"""

from __future__ import annotations

import torch

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
    ):
        self.k = k
        self.device = torch.device(device)
        self.local_kernel = LocalMatchKernel(
            k=k,
            max_pattern_len=max_pattern_len,
            min_match_len=min_match_len,
            max_occurrences=max_occurrences,
        ).to(self.device)

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        combined_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft tokens from the requests' own histories.

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
        draft, num_valid, _, _ = self.local_kernel(
            num_tokens_no_spec, token_ids_gpu, combined_mask)
        return draft, num_valid

    def load_model(self, *args, **kwargs) -> None:
        pass
