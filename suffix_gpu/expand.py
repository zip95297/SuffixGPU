"""Frequency-ranked chain expansion from occurrence continuations.

Shared by the local matcher and the global index: given a per-request
matrix of continuation sequences (one row per occurrence of the matched
pattern), build the draft chain by depth-wise majority vote.
"""

from __future__ import annotations

import torch


def _majority_token(values: torch.Tensor, active: torch.Tensor,
                     sentinel: int) -> torch.Tensor:
    """Per-row majority token among active entries.

    Args:
        values: [B, R] token values (garbage allowed in inactive slots).
        active: [B, R] bool mask of valid entries.
        sentinel: value marking inactive slots; must sort below every
            real token id (use -1 for token ids >= 0).

    Returns:
        [B] int tensor; rows with no active entry get `sentinel`.
    """
    b, r = values.shape
    masked = torch.where(active, values, sentinel)
    sorted_v, _ = torch.sort(masked, dim=1)
    real = sorted_v != sentinel
    prev_v = torch.cat(
        [torch.full((b, 1), sentinel, dtype=sorted_v.dtype,
                    device=values.device), sorted_v[:, :-1]], dim=1)
    boundary = real & ((sorted_v != prev_v) |
                       ~torch.cat([torch.ones(b, 1, dtype=torch.bool,
                                              device=values.device),
                                   real[:, :-1]], dim=1))
    pos_idx = torch.arange(r, dtype=torch.int64, device=values.device)
    run_start = torch.cummax(
        torch.where(boundary, pos_idx.unsqueeze(0),
                    torch.zeros(b, r, dtype=torch.int64,
                                device=values.device)), dim=1).values
    run_len = pos_idx.unsqueeze(0) - run_start + 1
    run_end = real & torch.cat(
        [sorted_v[:, 1:] != sorted_v[:, :-1],
         torch.ones(b, 1, dtype=torch.bool, device=values.device)], dim=1)
    run_len = torch.where(run_end, run_len, torch.zeros_like(run_len))
    max_cnt = run_len.max(dim=1, keepdim=True).values
    winner = run_end & (run_len == max_cnt) & (max_cnt > 0)
    # Ties resolve to the smallest token id (rows are sorted ascending).
    cand = torch.where(
        winner, sorted_v.to(torch.int64),
        torch.full_like(sorted_v, torch.iinfo(torch.int64).max,
                        dtype=torch.int64))
    first_win = cand.argmin(dim=1)
    token = sorted_v.gather(1, first_win.unsqueeze(1)).squeeze(1)
    any_active = active.any(dim=1)
    return torch.where(any_active, token,
                       torch.full_like(token, sentinel))


def expand_chain(
    cont: torch.Tensor,
    num_occ: torch.Tensor,
    k: int,
    sentinel: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a draft chain by depth-wise majority vote over continuations.

    Args:
        cont: [B, R, k] int tensor of continuation tokens; row r of
            request b holds the k tokens following occurrence r, padded
            with `sentinel` past the sequence end.
        num_occ: [B] int tensor, number of valid occurrence rows per
            request (rows beyond it are ignored).
        k: draft length.
        sentinel: padding value sorting below all real tokens.

    Returns:
        (chain [B, k] int, num_valid [B] int64): the majority-vote
        chain and the count of leading valid tokens per request.
    """
    b, r, _ = cont.shape
    device = cont.device
    offs = torch.arange(r, device=device)
    active = offs.unsqueeze(0) < num_occ.unsqueeze(1)
    chain = torch.full((b, k), sentinel, dtype=cont.dtype, device=device)
    prefix_ok = active.clone()
    for d in range(k):
        active_d = prefix_ok & (cont[:, :, d] != sentinel)
        tok = _majority_token(cont[:, :, d], active_d, sentinel)
        valid = tok != sentinel
        chain[:, d] = torch.where(valid, tok, chain[:, d])
        prefix_ok = (
            prefix_ok
            & (cont[:, :, d] == tok.unsqueeze(1))
            & valid.unsqueeze(1)
        )
    # Chain validity is monotone: once a slot is sentinel, the rest are.
    filled = (chain != sentinel).to(torch.int64)
    num_valid = torch.cumprod(filled, dim=1).sum(dim=1)
    return chain, num_valid
