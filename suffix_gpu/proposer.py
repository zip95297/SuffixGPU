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

import os

import torch

from suffix_gpu import triton_kernels
from suffix_gpu.expand import expand_chain
from suffix_gpu.global_index import GlobalIndex
from suffix_gpu.local_matcher import LocalMatchKernel, select_local_best


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
        vote_smoothing_alpha: float = 1.0,
        local_mode: str = "soft",
        soft_lambda: float = 0.5,
        merge_paths: bool = True,
        dynamic_k: bool = True,
        ema_decay: float = 0.8,
        dyn_k_scale: float = 1.5,
        dyn_k_offset: float = 1.0,
        dyn_k_min: int = 2,
        eviction: str = "lfu",
        lfu_decay: float = 0.5,
        lfu_protect_rebuilds: int = 1,
        parallel_paths: bool = True,
    ):
        # Env preset for embedders whose config surface predates the
        # v2 knobs (e.g. vLLM A/B runs): SUFFIX_GPU_PRESET=legacy
        # restores the pre-v2 drafting semantics.
        if os.environ.get("SUFFIX_GPU_PRESET", "").lower() == "legacy":
            vote_smoothing_alpha = 0.0
            local_mode = "backoff"
            merge_paths = False
            dynamic_k = False
            eviction = "fifo"
        self.k = k
        self.device = torch.device(device)
        self.max_pattern_len = max_pattern_len
        self.max_spec_factor = max_spec_factor
        self.max_spec_offset = max_spec_offset
        self.min_token_prob = min_token_prob
        self.num_backoff = max(1, int(num_backoff))
        # Laplace smoothing of the vote fractions: p = v / (a + alpha).
        # A single-occurrence "unanimous" chain then decays instead of
        # riding a fake probability of 1.0. 0 restores legacy scoring.
        self.vote_smoothing_alpha = float(vote_smoothing_alpha)
        # local_mode "soft": one weighted ensemble over all match
        # lengths (occurrences vote with weight lambda^(L_max - L))
        # instead of the C-candidate hard backoff ladder.
        self.local_mode = local_mode
        self.soft_lambda = float(soft_lambda)
        # merge_paths: when the longest local and global matches have
        # equal length, additionally vote over the union of both
        # occurrence sets; the joint draft wins only on a strictly
        # greater score than either path alone.
        self.merge_paths = bool(merge_paths)
        # dynamic_k: per-row EMA of accepted draft lengths modulates
        # the emission cap (fewer wasted verification slots on rows
        # that rarely accept). All device-side, CUDA-graph safe.
        self.dynamic_k = bool(dynamic_k)
        self.ema_decay = float(ema_decay)
        self.dyn_k_scale = float(dyn_k_scale)
        self.dyn_k_offset = float(dyn_k_offset)
        self.dyn_k_min = int(dyn_k_min)
        self._accept_ema: torch.Tensor | None = None
        # LFU credit attribution: winning global occurrences of the
        # previous step, credited with this step's accept count in
        # update_state. Invalidated (tier=0) via poll() whenever the
        # index's position epoch moves.
        self.eviction = eviction
        self._last_pos: torch.Tensor | None = None
        self._last_w: torch.Tensor | None = None
        self._last_tier: torch.Tensor | None = None
        self._last_epoch = -1
        # parallel_paths: run the global match phase (SA walk + delta
        # scan) on a side stream concurrently with the local match
        # phase; fork/join via events, capturable into CUDA graphs.
        self.parallel_paths = bool(parallel_paths)
        self._ps_stream: torch.cuda.Stream | None = None
        self._ps_fork: torch.cuda.Event | None = None
        self._ps_join: torch.cuda.Event | None = None
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
            vote_smoothing_alpha=self.vote_smoothing_alpha,
            local_mode=self.local_mode,
            soft_lambda=self.soft_lambda,
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
                eviction=eviction,
                lfu_decay=lfu_decay,
                lfu_protect_rebuilds=lfu_protect_rebuilds,
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
        Also drops cached credit positions whenever stored coordinates
        moved (delta compaction / active swap), so stale positions are
        never credited.
        """
        if self.global_index is not None:
            self.global_index.poll_rebuild()
            if (self._last_tier is not None
                    and self.global_index.position_epoch
                    != self._last_epoch):
                self._last_tier.zero_()
                self._last_epoch = self.global_index.position_epoch

    def _ensure_credit(self, b: int, r: int) -> None:
        if self._last_pos is not None and self._last_pos.shape[0] >= b:
            return
        pos = torch.zeros(b, r, dtype=torch.int64, device=self.device)
        w = torch.zeros(b, r, dtype=torch.float32, device=self.device)
        tier = torch.zeros(b, dtype=torch.int8, device=self.device)
        if self._last_pos is not None:
            n0 = self._last_pos.shape[0]
            pos[:n0] = self._last_pos
            w[:n0] = self._last_w
            tier[:n0] = self._last_tier
        self._last_pos, self._last_w, self._last_tier = pos, w, tier

    # ------------------------------------------------------------------
    # dynamic-k accept EMA
    # ------------------------------------------------------------------
    def _ensure_ema(self, b: int) -> torch.Tensor:
        if self._accept_ema is None or self._accept_ema.shape[0] < b:
            fresh = torch.zeros(b, dtype=torch.float32,
                                device=self.device)
            if self._accept_ema is not None:
                fresh[: self._accept_ema.shape[0]] = self._accept_ema
            self._accept_ema = fresh
        return self._accept_ema

    def reset_rows(self, row_indices: torch.Tensor | list[int]) -> None:
        """Clear per-row drafter state when batch rows are recycled."""
        if not torch.is_tensor(row_indices):
            if not row_indices:
                return
            row_indices = torch.tensor(row_indices, dtype=torch.int64,
                                       device=self.device)
        if self._accept_ema is not None:
            self._accept_ema[row_indices] = 0.0
        if self._last_tier is not None:
            self._last_tier[row_indices] = 0

    def _dyn_cap(self, b: int) -> torch.Tensor | None:
        """Per-row emission limit from the accept EMA, or None."""
        if not self.dynamic_k or self._accept_ema is None \
                or self._accept_ema.shape[0] < b:
            return None
        kd = torch.ceil(self.dyn_k_scale * self._accept_ema[:b]
                        + self.dyn_k_offset)
        return kd.to(torch.int64).clamp(self.dyn_k_min, self.k)

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
        if self.dynamic_k:
            ema = self._ensure_ema(b)
            acc = (cnt - 1).clamp(min=0).to(torch.float32)
            upd = (self.ema_decay * ema[:b]
                   + (1.0 - self.ema_decay) * acc)
            ema[:b] = torch.where(cnt > 0, upd, ema[:b])
        if (self.global_index is not None
                and self.global_index.hit is not None
                and self._last_tier is not None
                and self._last_tier.shape[0] >= b):
            credit = (cnt - 1).clamp(min=0).to(torch.float32)
            self.global_index.credit_accepted(
                self._last_pos[:b],
                self._last_w[:b] * credit.unsqueeze(1),
                self._last_tier[:b])
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

        Local and global candidate blocks share one fused expand+score
        launch; selection keeps the hierarchical rules (within-local
        ties prefer the longer match, within-global the longer cap,
        global beats local only on a strictly greater score).

        Args:
            num_tokens_no_spec: [B] int32 token counts.
            token_ids_gpu: [B, S] int32 token buffer.
            combined_mask: optional [B] bool rows allowed to draft.

        Returns:
            (draft_tokens [B, k] int32, num_valid_draft_tokens [B] int32)
        """
        b = num_tokens_no_spec.shape[0]
        k = self.k
        if combined_mask is None:
            combined_mask = torch.ones(
                b, dtype=torch.bool, device=self.device)

        if self.global_index is None:
            (draft, num_valid, _, _, _) = self.local_kernel(
                num_tokens_no_spec, token_ids_gpu, combined_mask,
                cap_limit=self._dyn_cap(b))
        else:
            kd = self._dyn_cap(b)
            r = self.local_kernel.max_occurrences
            tails, tail_len = self._gather_tails(num_tokens_no_spec,
                                                 token_ids_gpu)
            use_ps = (self.parallel_paths
                      and self.device.type == "cuda")
            if use_ps:
                if self._ps_stream is None:
                    self._ps_stream = torch.cuda.Stream(self.device)
                    self._ps_fork = torch.cuda.Event()
                    self._ps_join = torch.cuda.Event()
                cur = torch.cuda.current_stream(self.device)
                self._ps_fork.record(cur)
                capturing = torch.cuda.is_current_stream_capturing()
                with torch.cuda.stream(self._ps_stream):
                    self._ps_stream.wait_event(self._ps_fork)
                    if not capturing:
                        # Allocator hint: tails is main-stream memory
                        # read on the side stream.
                        tails.record_stream(self._ps_stream)
                        tail_len.record_stream(self._ps_stream)
                    g_len, g_cont, g_occ, g_pos, g_tier = \
                        self.global_index._query_backoff(
                            tails.to(torch.int32), tail_len,
                            self.max_pattern_len, self._global_caps)
                    self._ps_join.record(self._ps_stream)
                # Local match runs concurrently on the main stream.
                cand_l, cont_l, occ_l, w_l = self.local_kernel.gather(
                    num_tokens_no_spec, token_ids_gpu, combined_mask)
                cur.wait_event(self._ps_join)
                if not capturing:
                    for t in (g_len, g_cont, g_occ, g_pos, g_tier):
                        t.record_stream(cur)
            else:
                cand_l, cont_l, occ_l, w_l = self.local_kernel.gather(
                    num_tokens_no_spec, token_ids_gpu, combined_mask)
                g_len, g_cont, g_occ, g_pos, g_tier = \
                    self.global_index._query_backoff(
                        tails.to(torch.int32), tail_len,
                        self.max_pattern_len, self._global_caps)

            cl = cand_l.shape[1]
            cg = g_len.shape[1]
            len_all = torch.cat([cand_l, g_len], dim=1)
            cont_all = torch.cat([cont_l, g_cont], dim=1)
            occ_all = torch.cat([occ_l, g_occ], dim=1)
            if w_l is None:
                w_all = None
            else:
                # Global rows vote with unit mass (fp32-exact counts).
                w_all = torch.cat(
                    [w_l.reshape(b, cl, r),
                     torch.ones(b, cg, r, dtype=torch.float32,
                                device=self.device)],
                    dim=1).reshape(b * (cl + cg), r)
            cap_all = self.local_kernel._spec_cap(len_all.reshape(-1))
            if kd is not None:
                cap_all = torch.minimum(
                    cap_all.view(b, cl + cg),
                    kd.unsqueeze(1)).reshape(-1)
            chain, _, emit, score = expand_chain(
                cont_all.reshape(b * (cl + cg), r, k),
                occ_all.reshape(-1), k,
                min_token_prob=self.min_token_prob, cap=cap_all,
                weights=w_all, alpha=self.vote_smoothing_alpha)
            chain = chain.view(b, cl + cg, k)
            emit = emit.view(b, cl + cg)
            score = score.view(b, cl + cg)

            (l_chain, l_emit, _, _, l_score) = select_local_best(
                cand_l, chain[:, :cl], emit[:, :cl], occ_all[:, :cl],
                score[:, :cl])
            # Global: first max = longest cap (caps are descending).
            g_score_all = score[:, cl:]
            best_g = g_score_all.argmax(dim=1)
            bidx = torch.arange(b, device=self.device)
            g_chain = chain[:, cl:][bidx, best_g]
            g_emit = emit[:, cl:][bidx, best_g]
            g_score = g_score_all[bidx, best_g]

            pick_global = (g_score > l_score) & combined_mask
            draft = torch.where(pick_global.unsqueeze(1), g_chain,
                                l_chain)
            num_valid = torch.where(pick_global, g_emit,
                                    l_emit.to(torch.int64))

            pick_joint = torch.zeros_like(pick_global)
            if self.merge_paths:
                # Joint candidate: union of the longest local and
                # longest global occurrence sets when both matched the
                # same length. Rows are gated by weight 0 (and -1
                # continuations) elsewhere, so shapes stay static.
                l_len0 = cand_l[:, 0]
                g_len0 = g_len[:, 0]
                joint_ok = (l_len0 > 0) & (l_len0 == g_len0) \
                    & combined_mask
                j_cont = torch.cat([cont_l[:, 0], g_cont[:, 0]], dim=1)
                if w_l is None:
                    w_loc = torch.ones(b, r, dtype=torch.float32,
                                       device=self.device)
                else:
                    w_loc = w_l.reshape(b, cl, r)[:, 0]
                j_w = torch.cat(
                    [w_loc, torch.ones(b, r, dtype=torch.float32,
                                       device=self.device)], dim=1)
                j_w = j_w * joint_ok.unsqueeze(1).to(torch.float32)
                j_occ = torch.full((b,), 2 * r, dtype=torch.int64,
                                   device=self.device)
                j_cap = self.local_kernel._spec_cap(l_len0)
                if kd is not None:
                    j_cap = torch.minimum(j_cap, kd)
                j_chain, _, j_emit, j_score = expand_chain(
                    j_cont, j_occ, k,
                    min_token_prob=self.min_token_prob, cap=j_cap,
                    weights=j_w, alpha=self.vote_smoothing_alpha)
                pick_joint = ((j_score > l_score) & (j_score > g_score)
                              & joint_ok)
                draft = torch.where(pick_joint.unsqueeze(1), j_chain,
                                    draft)
                num_valid = torch.where(pick_joint, j_emit, num_valid)

            if self.global_index.hit is not None:
                # Record the winning global occurrences; update_state
                # credits them with the next step's accept count.
                self._ensure_credit(b, r)
                sel_pos = torch.where(
                    pick_joint.unsqueeze(1), g_pos[:, 0],
                    g_pos[bidx, best_g])
                sel_occ = torch.where(pick_joint, g_occ[:, 0],
                                      g_occ[bidx, best_g])
                sel_tier = torch.where(pick_joint, g_tier[:, 0],
                                       g_tier[bidx, best_g])
                used = pick_global | pick_joint
                sel_tier = torch.where(used, sel_tier,
                                       torch.zeros_like(sel_tier))
                rowa = (torch.arange(r, device=self.device).view(1, r)
                        < sel_occ.unsqueeze(1))
                w = rowa.to(torch.float32) / sel_occ.clamp(
                    min=1).to(torch.float32).unsqueeze(1)
                self._last_pos[:b].copy_(sel_pos)
                self._last_w[:b].copy_(w)
                self._last_tier[:b].copy_(sel_tier)

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
