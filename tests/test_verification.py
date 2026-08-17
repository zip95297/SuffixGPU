"""End-to-end correctness verification of the GPU suffix draft path.

These tests exercise the full draft/verify semantics on device tensors
(CPU in CI): the speculative loop must reproduce the target model's
output exactly, global memory must warm up across requests, and results
must agree with the arctic_inference reference implementation.

Note: SuffixGPUDrafter.propose expects the token buffer to already
contain the latest sampled tokens; the scatter of newly sampled tokens
is the vLLM adapter's job (SuffixProposerGPU.propose). The helper below
mimics that scatter so the loop mirrors the real integration.
"""

from __future__ import annotations

import random

import pytest
import torch

from suffix_gpu.proposer import SuffixGPUDrafter

K = 4
P = 8
R = 16

# Pin pre-v2 drafter semantics so the arctic-equivalence oracles stay
# valid: no vote smoothing.
LEGACY = dict(vote_smoothing_alpha=0.0, local_mode="backoff",
              merge_paths=False, dynamic_k=False)


def _make_drafter(device, enable_global: bool = True) -> SuffixGPUDrafter:
    return SuffixGPUDrafter(
        k=K, device=device, max_pattern_len=P, max_occurrences=R,
        enable_global=enable_global, global_capacity=1 << 16,
        delta_capacity=1 << 12, rebuild_threshold=1 << 10, **LEGACY)


def _adapter_propose(drafter: SuffixGPUDrafter, buf: torch.Tensor,
                     nts_cpu: int, pending: list[int]):
    """Mimic SuffixProposerGPU.propose: scatter sampled tokens into the
    resident buffer, then draft with the updated length."""
    n = len(pending)
    buf[0, nts_cpu:nts_cpu + n] = torch.tensor(
        pending, dtype=torch.int32, device=buf.device)
    nts = torch.tensor([nts_cpu + n], dtype=torch.int32,
                       device=buf.device)
    return drafter.propose(nts, buf)


def _build_repetitive_doc(seed: int, n_blocks: int = 12) -> list[int]:
    """Document made of shuffled repeated blocks: highly compressible."""
    g = random.Random(seed)
    blocks = [[g.randint(0, 5) for _ in range(6)] for _ in range(4)]
    doc: list[int] = []
    for _ in range(n_blocks):
        doc.extend(g.choice(blocks))
    return doc


def _build_nonrepeating_doc(seed: int, n: int = 30) -> list[int]:
    """Random document with no repeated 3-gram (weak local matches)."""
    g = random.Random(seed)
    doc: list[int] = []
    while len(doc) < n:
        t = g.randint(0, 15)
        if len(doc) >= 2 and any(
                doc[i] == doc[-2] and doc[i + 1] == doc[-1]
                and doc[i + 2] == t
                for i in range(len(doc) - 2)):
            continue
        doc.append(t)
    return doc


def _verify_draft(draft: torch.Tensor, nv: int, doc: list[int],
                  base: int) -> int:
    """Accepted count: longest draft prefix matching the target doc."""
    a = 0
    while a < nv and base + a < len(doc) \
            and int(draft[0, a].item()) == doc[base + a]:
        a += 1
    return a


def test_spec_loop_exact_output_local(device):
    """Drafts must never corrupt the output: generated tokens equal the
    target model's greedy sequence exactly, and speculation must help."""
    drafter = _make_drafter(device, enable_global=False)
    doc = _build_repetitive_doc(7)
    prompt = doc[:4]
    target_len = len(doc) - len(prompt)
    width = len(doc) + K + 8
    buf = torch.zeros(1, width, dtype=torch.int32, device=device)
    buf[0, :len(prompt)] = torch.tensor(prompt, dtype=torch.int32)

    gen: list[int] = []
    pending = [doc[len(prompt)]]
    steps = 0
    accepted_total = 0
    while len(gen) < target_len:
        steps += 1
        assert steps < 10000, "loop did not converge"
        draft, nv = _adapter_propose(
            drafter, buf, len(prompt) + len(gen), pending)
        # Buffer must equal prompt + committed + just-sampled tokens.
        expect = prompt + gen + pending
        assert buf[0, :len(expect)].tolist() == expect, "scatter broken"
        gen.extend(pending)
        if len(gen) >= target_len:
            break
        base = len(prompt) + len(gen)
        a = _verify_draft(draft, int(nv[0].item()), doc, base)
        accepted_total += a
        pending = [int(draft[0, i].item()) for i in range(a)]
        if base + a < len(doc):
            pending.append(doc[base + a])
        else:
            gen.extend(int(draft[0, i].item()) for i in range(a))
            break

    gen = gen[:target_len]
    assert gen == doc[len(prompt):len(prompt) + target_len]
    assert steps < target_len, "no speculation gain on repetitive doc"
    assert accepted_total > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spec_loop_exact_output_random(seed, device):
    """Even on incompressible docs the loop must stay exact."""
    drafter = _make_drafter(device, enable_global=False)
    doc = _build_nonrepeating_doc(seed, 48)
    prompt = doc[:6]
    target_len = len(doc) - len(prompt)
    buf = torch.zeros(1, len(doc) + K + 8, dtype=torch.int32, device=device)
    buf[0, :len(prompt)] = torch.tensor(prompt, dtype=torch.int32)

    gen: list[int] = []
    pending = [doc[len(prompt)]]
    for _ in range(10000):
        draft, nv = _adapter_propose(
            drafter, buf, len(prompt) + len(gen), pending)
        gen.extend(pending)
        if len(gen) >= target_len:
            break
        base = len(prompt) + len(gen)
        a = _verify_draft(draft, int(nv[0].item()), doc, base)
        pending = [int(draft[0, i].item()) for i in range(a)]
        if base + a < len(doc):
            pending.append(doc[base + a])
        else:
            gen.extend(int(draft[0, i].item()) for i in range(a))
            break
    gen = gen[:target_len]
    assert gen == doc[len(prompt):len(prompt) + target_len]


def test_global_warmup_improves_acceptance(device):
    """On a doc with no internal repetition, round 1 cannot match locally;
    after harvesting, round 2 must draft from the global index."""
    drafter = _make_drafter(device)
    doc = _build_nonrepeating_doc(11, 32)
    prompt = doc[:4]
    target_len = len(doc) - len(prompt)

    def one_round() -> int:
        buf = torch.zeros(1, len(doc) + K + 8, dtype=torch.int32,
                          device=device)
        buf[0, :len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
        gen: list[int] = []
        pending = [doc[len(prompt)]]
        accepted_total = 0
        for _ in range(10000):
            draft, nv = _adapter_propose(
                drafter, buf, len(prompt) + len(gen), pending)
            gen.extend(pending)
            if len(gen) >= target_len:
                break
            base = len(prompt) + len(gen)
            a = _verify_draft(draft, int(nv[0].item()), doc, base)
            accepted_total += a
            pending = [int(draft[0, i].item()) for i in range(a)]
            if base + a < len(doc):
                pending.append(doc[base + a])
            else:
                gen.extend(int(draft[0, i].item()) for i in range(a))
                break
        assert gen[:target_len] == doc[len(prompt):len(prompt) + target_len]
        # Request finished: harvest the full response.
        drafter.harvest_rows([buf[0]], [len(prompt) + len(gen)])
        return accepted_total

    round1 = one_round()
    round2 = one_round()
    assert round2 > round1, "global index must warm up after harvest"
    assert round2 >= 8, "global matches should accept long drafts"


def test_query_consistency_across_rebuilds(device):
    """Interleaved appends and queries: match lengths must equal a naive
    oracle over the union of all docs, before and after rebuild swaps."""
    from suffix_gpu.reference import (naive_longest_suffix_match,
                                      naive_occurrences)

    drafter = _make_drafter(device)
    idx = drafter.global_index
    assert idx is not None
    idx.rebuild_threshold = 16  # force frequent rebuilds
    g = random.Random(3)
    corpus: list[int] = []
    for d in range(8):
        doc = [g.randint(0, 4) for _ in range(24)]
        idx.append_documents(
            [torch.tensor(doc, dtype=torch.int32, device=device)])
        corpus.extend(doc)
        query = doc[-6:]
        q = torch.tensor([query + [0] * (P - 6)], dtype=torch.int32,
                         device=device)
        qlen = torch.tensor([6], dtype=torch.int64, device=device)
        mlen, _, occ = idx.query(q, qlen, P)
        exp_len = naive_longest_suffix_match(corpus, query, P)
        assert int(mlen[0].item()) == exp_len, f"doc {d}"
        if exp_len > 0:
            exp_occ = naive_occurrences(corpus, query[-exp_len:])
            got = int(occ[0].item())
            assert 1 <= got <= min(len(exp_occ), R), f"doc {d}"


def test_arctic_fuzz_equivalence(device):
    """On unambiguous corpora, drafts must match arctic_inference."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("arctic_inference")
    from collections import Counter

    from arctic_inference.suffix_decoding import SuffixDecodingCache

    g = random.Random(5)
    checked = 0
    for trial in range(20):
        drafter = _make_drafter(device)
        core = [g.randint(0, 5) for _ in range(5)]
        tag = 100 + trial
        docs = [core + [tag, tag + 1, tag + 2, tag + 3]
                for _ in range(3)]

        cache = SuffixDecodingCache(max_tree_depth=24,
                                    max_cached_requests=100)
        cache.start_request("hist", np.array([], dtype=np.int32))
        buf = torch.zeros(len(docs), 64, dtype=torch.int32, device=device)
        lens = []
        for i, doc in enumerate(docs):
            cache.add_active_response(
                "hist", np.array(doc, dtype=np.int32))
            buf[i, :len(doc)] = torch.tensor(doc, dtype=torch.int32)
            lens.append(len(doc))
        drafter.harvest_finished(list(range(len(docs))), lens, buf)

        query = core[-4:]
        corpus = [t for doc in docs for t in doc]
        occ = [s for s in range(len(corpus) - len(query) + 1)
               if corpus[s:s + len(query)] == query]
        toks = [corpus[s + len(query)] for s in occ
                if s + len(query) < len(corpus)]
        cnt = Counter(toks)
        top2 = sorted(cnt.values(), reverse=True)[:2]
        if not toks or (len(top2) == 2 and top2[0] == top2[1]):
            continue  # ambiguous majority vote; skip
        expected0 = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        cache.start_request("new", np.array([], dtype=np.int32))
        arctic_draft = cache.speculate(
            "new", np.array(query, dtype=np.int32), max_spec_tokens=K,
            max_spec_factor=10.0, min_token_prob=0.0)
        if len(arctic_draft.token_ids) == 0:
            continue

        qbuf = torch.zeros(1, 64, dtype=torch.int32, device=device)
        qbuf[0, :len(query) - 1] = torch.tensor(query[:-1],
                                                dtype=torch.int32)
        draft, nv = _adapter_propose(drafter, qbuf, len(query) - 1,
                                     [query[-1]])
        assert int(nv[0].item()) > 0, f"trial {trial}"
        assert int(draft[0, 0].item()) == expected0, f"trial {trial}"
        assert int(arctic_draft.token_ids[0]) == expected0, f"trial {trial}"
        checked += 1
    assert checked >= 10


def test_adaptive_max_spec_factor(device):
    """num_valid must be clamped to factor * match_len + offset, with
    clamped slots padded to -1 (arctic adaptive-length semantics)."""
    # Tail [1, 2] matched once (pos 0, match_len 2), continuation
    # [7, 8, 9, 1] fills all k = 4 slots when unclamped.
    seq = [1, 2, 7, 8, 9, 1, 2]
    buf = torch.zeros(1, 16, dtype=torch.int32, device=device)
    buf[0, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
    nts = torch.tensor([len(seq)], dtype=torch.int32, device=device)

    base = SuffixGPUDrafter(k=K, device=device, max_pattern_len=P,
                            max_occurrences=R, enable_global=False,
                            **LEGACY)
    draft, nv = base.propose(nts, buf)
    assert nv[0].item() == 4
    assert draft[0].tolist() == [7, 8, 9, 1]

    clamped = SuffixGPUDrafter(k=K, device=device, max_pattern_len=P,
                               max_occurrences=R, enable_global=False,
                               max_spec_factor=1.0, **LEGACY)
    draft, nv = clamped.propose(nts, buf)
    assert nv[0].item() == 2  # 1.0 * match_len(2) + 0
    assert draft[0].tolist() == [7, 8, -1, -1]


def test_adaptive_min_token_prob(device):
    """Chain expansion must stop once the estimated chain probability
    (product of per-depth vote fractions) falls below the threshold."""
    # Tail [5] occurs at 0, 3 and 6; continuations vote 6, 6, 7 at
    # depth 0 (prob 2/3), then split 0 vs 1 at depth 1 (prob 1/3).
    seq = [5, 6, 0, 5, 6, 1, 5, 7, 2, 5]
    buf = torch.zeros(1, 16, dtype=torch.int32, device=device)
    buf[0, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
    nts = torch.tensor([len(seq)], dtype=torch.int32, device=device)

    mid = SuffixGPUDrafter(k=K, device=device, max_pattern_len=P,
                           max_occurrences=R, enable_global=False,
                           min_token_prob=0.5, **LEGACY)
    draft, nv = mid.propose(nts, buf)
    assert nv[0].item() == 1  # depth 0 passes (2/3), depth 1 cut (1/3)
    assert draft[0].tolist() == [6, -1, -1, -1]

    strict = SuffixGPUDrafter(k=K, device=device, max_pattern_len=P,
                              max_occurrences=R, enable_global=False,
                              min_token_prob=0.7, **LEGACY)
    draft, nv = strict.propose(nts, buf)
    assert nv[0].item() == 0  # 2/3 < 0.7: nothing drafted
    assert draft[0].tolist() == [-1, -1, -1, -1]
