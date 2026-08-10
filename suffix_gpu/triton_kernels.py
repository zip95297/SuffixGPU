"""Fused Triton kernels for the draft hot path (CUDA only).

Each kernel replaces a long chain of small torch ops with one launch:

- match_back: the rolling-AND suffix comparison over pattern offsets
  (local matcher rows and the shared delta buffer).
- sa_search: the full merged lower/upper-bound binary search over the
  suffix array, one thread-row per (bound, pattern) pair.
- expand_chain: the depth-wise majority-vote chain expansion.

All entry points have pure-torch fallbacks in their call sites; these
paths require CUDA tensors and are exercised by the same oracle tests
via the device fixture.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


def available(*tensors: torch.Tensor) -> bool:
    return HAS_TRITON and all(t.is_cuda for t in tensors)


if HAS_TRITON:

    @triton.jit
    def _match_back_kernel(src_ptr, pat_ptr, out_ptr, row_stride,
                           n_out, P: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m_i = i < n_out
        base = src_ptr + b * row_stride
        acc = tl.full((BLOCK,), True, tl.int1)
        mb = tl.zeros((BLOCK,), dtype=tl.int32)
        for t in tl.static_range(P):
            pat_t = tl.load(pat_ptr + b * P + t)
            idx = i - 1 - t
            tok = tl.load(base + idx, mask=m_i & (idx >= 0), other=-1)
            acc = acc & (tok == pat_t)
            mb += acc.to(tl.int32)
        tl.store(out_ptr + b * n_out + i, mb, mask=m_i)

    @triton.jit
    def _sa_search_kernel(sa_ptr, corpus_ptr, pat_ptr, plen_ptr, out_ptr,
                          n_corpus, n_rows, M: tl.constexpr,
                          ITERS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m_row = row < 2 * n_rows
        prow = tl.where(m_row, row % n_rows, 0)
        is_upper = row >= n_rows
        plen = tl.load(plen_ptr + prow, mask=m_row, other=0)
        j = tl.arange(0, M)
        pat = tl.load(pat_ptr + prow[:, None] * M + j[None, :],
                      mask=m_row[:, None], other=0)
        pat_valid = j[None, :] < plen[:, None]
        lo = tl.zeros((BLOCK,), dtype=tl.int64)
        hi = tl.zeros((BLOCK,), dtype=tl.int64) + n_corpus
        for _ in tl.static_range(ITERS):
            mid = tl.minimum((lo + hi) // 2, n_corpus - 1)
            pos = tl.load(sa_ptr + mid, mask=m_row, other=0)
            cidx = pos[:, None] + j[None, :]
            tok = tl.load(corpus_ptr + cidx,
                          mask=m_row[:, None] & (cidx < n_corpus),
                          other=-1)
            eq_ok = (tok == pat) | ~pat_valid
            first_bad = tl.min(
                tl.where(eq_ok, M, j[None, :]).to(tl.int64), axis=1)
            lcp = tl.minimum(first_bad, plen)
            full = lcp >= plen
            spos = pos + lcp
            schar = tl.load(corpus_ptr + tl.minimum(spos, n_corpus - 1),
                            mask=m_row & (spos < n_corpus), other=-1)
            pchar = tl.load(pat_ptr + prow * M + tl.minimum(lcp, M - 1),
                            mask=m_row, other=0)
            cmp = tl.where(schar < pchar, -1,
                           tl.where(schar > pchar, 1, 0))
            cmp = tl.where(full, 0, cmp)
            go = (cmp < 0) | (is_upper & (cmp == 0))
            lo = tl.where(go, mid + 1, lo)
            hi = tl.where(go, hi, mid)
        tl.store(out_ptr + row, lo, mask=m_row)

    @triton.jit
    def _expand_chain_kernel(cont_ptr, nocc_ptr, chain_ptr, nv_ptr,
                             min_prob, R: tl.constexpr, K: tl.constexpr,
                             RP: tl.constexpr):
        b = tl.program_id(0)
        r = tl.arange(0, RP)
        occ = tl.load(nocc_ptr + b)
        ok = (r < R) & (r < occ)
        nv = tl.zeros((), dtype=tl.int32)
        prob = tl.zeros((), dtype=tl.float32) + 1.0
        for d in tl.static_range(K):
            c = tl.load(cont_ptr + b * R * K + r * K + d,
                        mask=r < R, other=-1)
            act = ok & (c != -1)
            eq = c[:, None] == c[None, :]
            votes = tl.sum((eq & act[None, :]).to(tl.int32), axis=1)
            votes = tl.where(act, votes, 0)
            mx = tl.max(votes, axis=0)
            winner = act & (votes == mx) & (mx > 0)
            tok = tl.min(tl.where(winner, c, 2147483647), axis=0)
            nact = tl.maximum(tl.sum(act.to(tl.int32), axis=0), 1)
            prob = prob * (mx.to(tl.float32) / nact.to(tl.float32))
            valid = (mx > 0) & (prob >= min_prob)
            tl.store(chain_ptr + b * K + d, tl.where(valid, tok, -1))
            nv += valid.to(tl.int32)
            ok = ok & (c == tok) & valid
        tl.store(nv_ptr + b, nv)

    @triton.jit
    def _scatter_append_kernel(tok_ptr, base_ptr, cnt_ptr, samp_ptr,
                               s_len, T: tl.constexpr,
                               TP: tl.constexpr):
        b = tl.program_id(0)
        j = tl.arange(0, TP)
        base = tl.load(base_ptr + b)
        cnt = tl.load(cnt_ptr + b)
        val = tl.load(samp_ptr + b * T + j, mask=j < T, other=-1)
        pos = base + j
        ok = (j < T) & (j < cnt) & (val != -1) & (pos < s_len)
        tl.store(tok_ptr + b * s_len + pos, val, mask=ok)


def match_back(src: torch.Tensor, pat: torch.Tensor,
               n_out: int) -> torch.Tensor:
    """match_back[b, i] = longest L with src[.., i-L:i] == tail_L.

    Args:
        src: [B, S] per-row buffer or [N] shared buffer (delta).
        pat: [B, P] newest-first patterns; fill values must never
            match buffer content.
        n_out: window end positions per row.

    Returns:
        [B, n_out] int32 match lengths (<= P).
    """
    b, p = pat.shape
    src = src.contiguous()
    pat = pat.contiguous()
    out = torch.empty(b, n_out, dtype=torch.int32, device=src.device)
    row_stride = src.stride(0) if src.dim() == 2 else 0
    block = 256
    grid = (b, triton.cdiv(n_out, block))
    _match_back_kernel[grid](src, pat, out, row_stride, n_out,
                             P=p, BLOCK=block)
    return out


def sa_search(sa: torch.Tensor, corpus: torch.Tensor,
              pattern: torch.Tensor,
              pattern_len: torch.Tensor) -> tuple[torch.Tensor,
                                                  torch.Tensor]:
    """Merged lower/upper-bound SA interval search (one launch)."""
    n = sa.shape[0]
    b, m = pattern.shape
    m_pad = max(triton.next_power_of_2(m), 2)
    if m_pad != m:
        pattern = torch.nn.functional.pad(pattern, (0, m_pad - m))
    pattern = pattern.to(torch.int32).contiguous()
    plen = pattern_len.to(torch.int64).contiguous()
    out = torch.empty(2 * b, dtype=torch.int64, device=sa.device)
    iters = max(1, math.ceil(math.log2(n + 1)))
    block = 64
    grid = (triton.cdiv(2 * b, block),)
    _sa_search_kernel[grid](sa, corpus, pattern, plen, out, n, b,
                            M=m_pad, ITERS=iters, BLOCK=block)
    return out[:b], out[b:]


def expand_chain(cont: torch.Tensor, num_occ: torch.Tensor, k: int,
                 min_token_prob: float) -> tuple[torch.Tensor,
                                                 torch.Tensor]:
    """Depth-wise majority-vote chain expansion (one launch)."""
    b, r, _ = cont.shape
    cont = cont.to(torch.int32).contiguous()
    nocc = num_occ.to(torch.int64).contiguous()
    chain = torch.empty(b, k, dtype=torch.int32, device=cont.device)
    nv = torch.empty(b, dtype=torch.int32, device=cont.device)
    rp = max(triton.next_power_of_2(r), 2)
    _expand_chain_kernel[(b,)](cont, nocc, chain, nv,
                               float(min_token_prob), R=r, K=k, RP=rp)
    return chain, nv.to(torch.int64)


def scatter_append(token_ids_gpu: torch.Tensor, base: torch.Tensor,
                   cnt: torch.Tensor,
                   sampled: torch.Tensor) -> None:
    """Append each row's valid sampled tokens at its base offset."""
    b, s = token_ids_gpu.shape
    t = sampled.shape[1]
    tp = max(triton.next_power_of_2(t), 2)
    _scatter_append_kernel[(b,)](
        token_ids_gpu, base.to(torch.int64).contiguous(),
        cnt.to(torch.int64).contiguous(),
        sampled.to(token_ids_gpu.dtype).contiguous(), s, T=t, TP=tp)
