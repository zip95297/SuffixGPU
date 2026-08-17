"""Frequency-ranked chain expansion from occurrence continuations.

Shared by the local matcher and the global index: given a per-request
matrix of continuation sequences (one row per occurrence of the matched
pattern), build the draft chain by depth-wise majority vote and score
it arctic-style in the same pass.
"""

from __future__ import annotations

import torch

from suffix_gpu import triton_kernels


def _majority_token(values: torch.Tensor, active: torch.Tensor,
                     sentinel: int,
                     weights: torch.Tensor | None = None,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row majority token among active entries, with its vote mass.

    Pairwise-equality vote count: R (max_occurrences) is small, so the
    [B, R, R] comparison is cheaper than a sort + run-length scan.

    Args:
        values: [B, R] token values (garbage allowed in inactive slots).
        active: [B, R] bool mask of valid entries.
        sentinel: value returned for rows with no active entry.
        weights: optional [B, R] float vote mass per entry; None means
            unit votes (integer counts).

    Returns:
        (token [B], count [B]): rows with no active entry get
        (`sentinel`, 0). Ties resolve to the smallest token id.
    """
    eq = values.unsqueeze(2) == values.unsqueeze(1)
    if weights is None:
        votes = (eq & active.unsqueeze(1)).sum(dim=2)
    else:
        w = torch.where(active, weights.to(torch.float32),
                        torch.zeros_like(weights, dtype=torch.float32))
        votes = (eq.to(torch.float32) * w.unsqueeze(1)).sum(dim=2)
    votes = torch.where(active, votes, torch.zeros_like(votes))
    max_cnt = votes.max(dim=1, keepdim=True).values
    winner = active & (votes == max_cnt) & (max_cnt > 0)
    cand = torch.where(
        winner, values.to(torch.int64),
        torch.full_like(values, torch.iinfo(torch.int64).max,
                        dtype=torch.int64))
    tok = cand.min(dim=1).values
    any_active = active.any(dim=1)
    token = torch.where(any_active, tok,
                        torch.full_like(tok, sentinel)).to(values.dtype)
    count = torch.where(any_active, max_cnt.squeeze(1),
                        torch.zeros_like(max_cnt.squeeze(1)))
    return token, count


def expand_chain(
    cont: torch.Tensor,
    num_occ: torch.Tensor,
    k: int,
    sentinel: int = -1,
    min_token_prob: float = 0.0,
    cap: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Majority-vote chain expansion with fused arctic-style scoring.

    The expansion already tracks the cumulative chain probability
    (product of per-depth vote fractions) for the ``min_token_prob``
    cutoff; the arctic draft score is the sum of exactly those values
    over emitted depths (``score += prob``), so it is accumulated here
    instead of in a separate replay pass. Chain validity is monotone,
    hence depth d is emitted iff it is valid and ``d < cap``.

    Args:
        cont: [B, R, k] int tensor of continuation tokens; row r of
            request b holds the k tokens following occurrence r, padded
            with `sentinel` past the sequence end.
        num_occ: [B] int tensor, number of valid occurrence rows per
            request (rows beyond it are ignored).
        k: draft length.
        sentinel: padding value sorting below all real tokens.
        min_token_prob: stop the chain once its estimated probability
            (product of per-depth vote fractions, mirroring
            child.count / node.count on a suffix tree) drops below this
            threshold. 0 disables the cutoff.
        cap: optional [B] max tokens to emit (adaptive spec cap);
            emission stops at min(num_valid, cap). None means k.
        weights: optional [B, R] float per-occurrence vote mass;
            None means unit votes (legacy counting semantics).
        alpha: Laplace smoothing added to the vote denominator,
            p = v / (a + alpha); a single unanimous occurrence then
            yields p = 1/(1+alpha) < 1 instead of a fake 1.0, so
            unsupported long-match chains decay and better-supported
            candidates can win. 0 keeps legacy behavior exactly.

    Returns:
        (chain [B, k] int, num_valid [B] i64, num_emit [B] i64,
         score [B] f32): the majority-vote chain (uncapped), its count
        of leading valid tokens, the capped emission count, and the
        summed chain probability over emitted depths.
    """
    b, r, _ = cont.shape
    device = cont.device
    if cap is None:
        cap = torch.full((b,), k, dtype=torch.int64, device=device)
    cap = cap.to(torch.int64)
    if sentinel == -1 and triton_kernels.available(cont, num_occ) and (
            weights is None or weights.is_cuda):
        return triton_kernels.expand_chain(cont, num_occ, k,
                                           min_token_prob, cap,
                                           weights=weights, alpha=alpha)
    offs = torch.arange(r, device=device)
    active = offs.unsqueeze(0) < num_occ.unsqueeze(1)
    chain = torch.full((b, k), sentinel, dtype=cont.dtype, device=device)
    prefix_ok = active.clone()
    cum_prob = torch.ones(b, dtype=torch.float32, device=device)
    score = torch.zeros(b, dtype=torch.float32, device=device)
    for d in range(k):
        active_d = prefix_ok & (cont[:, :, d] != sentinel)
        tok, cnt = _majority_token(cont[:, :, d], active_d, sentinel,
                                   weights=weights)
        if weights is None:
            denom = active_d.sum(dim=1).clamp(min=1).to(torch.float32)
        else:
            denom = torch.where(
                active_d, weights.to(torch.float32),
                torch.zeros(1, dtype=torch.float32, device=device)
            ).sum(dim=1)
        cum_prob = cum_prob * (cnt.to(torch.float32) / (denom + alpha))
        valid = (tok != sentinel) & (cnt > 0) & (cum_prob >= min_token_prob)
        chain[:, d] = torch.where(valid, tok, chain[:, d])
        score = score + torch.where(valid & (cap > d), cum_prob,
                                    torch.zeros_like(cum_prob))
        prefix_ok = (
            prefix_ok
            & (cont[:, :, d] == tok.unsqueeze(1))
            & valid.unsqueeze(1)
        )
    # Chain validity is monotone: once a slot is sentinel, the rest are.
    filled = (chain != sentinel).to(torch.int64)
    num_valid = torch.cumprod(filled, dim=1).sum(dim=1)
    num_emit = torch.minimum(num_valid, cap.clamp(min=0))
    return chain, num_valid, num_emit, score
