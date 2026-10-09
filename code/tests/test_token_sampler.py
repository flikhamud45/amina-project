"""Sampled generation with a client-chosen seed (``pvi.fullcheck.token_sampler``, plan E2): the public Gumbel table,
the sampler's choices, and sampled responses proved and checked end to end."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from pvi.fullcheck import cut_protocol as cp
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import token_sampler as ts
from pvi.fullcheck.transformer import build_decoder, greedy_tokens, with_lm_positions
from test_plans import _TINY

SEED = bytes(range(32))


def test_the_gumbel_table_is_the_published_one():
    tab = ts.gumbel_table()
    assert hashlib.sha256(tab.numpy().astype("<i4").tobytes()).hexdigest() == ts._TABLE_SHA256
    assert bool((tab[1:] >= tab[:-1]).all()) and int(tab[0]) < 0 < int(tab[-1]) and int(tab.abs().max()) < 1 << 20


def test_the_sampler_choices():
    g = torch.Generator().manual_seed(0)
    z = torch.randint(-(1 << 20), 1 << 20, (97,), generator=g)
    s = ts.TokenSampler(SEED, 1 << 10)
    assert s.choose(z, 5) == s.choose(z, 5)                                   # deterministic
    assert ts.TokenSampler(SEED, 1, top_k=1).choose(z, 5) == int(z.argmax())   # top-1 is greedy
    assert ts.TokenSampler(SEED, (1 << 24) - 1).choose(z, 5) == int(z.argmax())  # a very low temperature too
    picks = {ts.TokenSampler(bytes([i]) * 32, 1).choose(torch.zeros(97, dtype=torch.int64), 0) for i in range(40)}
    assert len(picks) > 20                                                    # flat logits: spread choices
    top = torch.sort(-z, stable=True).indices[:5].tolist()
    assert all(ts.TokenSampler(bytes([i]) * 32, 1, top_k=5).choose(z, 3) in top for i in range(20))
    with pytest.raises(ValueError):
        ts.TokenSampler(b"short", 3)
    # uniform logits: each index about equally often over many positions
    counts = np.bincount([s.choose(torch.zeros(8, dtype=torch.int64), p) for p in range(4000)], minlength=8)
    assert counts.min() > 380 and counts.max() < 620


def _response(sampler, n=4, prompt_len=6):
    cfg = _TINY["llama"]
    graph = build_decoder(cfg, calib_tokens=8, seed=1, prune_last=True)
    prompt = torch.randint(0, cfg.vocab, (1, prompt_len), generator=torch.Generator().manual_seed(3))
    return with_lm_positions(graph, n), ts.sample_tokens(graph, prompt, n - 1, sampler)


@pytest.mark.parametrize("mode,stream", [("C", False), ("C", True), ("Kpre", False)])
def test_a_sampled_response_is_accepted_and_others_rejected(mode, stream):
    s = ts.TokenSampler(SEED, 1 << 12, top_k=20)
    gen, x = _response(s)
    if mode == "C":
        coms = proto.commit_graph(gen, 4, policy="auto")
        params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=True, plan=coms.plan)
        v = proto.Verifier(gen.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                           tables=coms.table_publics, stream=stream)
        prover = proto.Prover(gen, commitments=coms)
    else:
        params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=True)
        v = proto.Verifier(gen.public(), params, mode, weights={op.name: (op.weight, op.bias) for op in gen.mat_ops})
        v.precompute(proto.Challenger())
        prover = proto.Prover(gen)
    out = proto.run_query(prover, v, x, wire=True, sampler=s)
    assert out["accepted"], out["rejected_at"]
    assert v.sampler is None
    # another seed: accepted only if its choices happen to be these tokens, else rejected at the token rule
    other = ts.TokenSampler(bytes(32), 1 << 12, top_k=20)
    out2 = proto.run_query(prover, v, x, wire=True, sampler=other)
    want = [other.choose(c, x.shape[1] - 4 + j) for j, c in enumerate(_head_logits(gen, x).T[:3])]
    assert out2["accepted"] == (want == x[0, -3:].tolist())
    assert out2["accepted"] or out2["rejected_at"] == "token"


def _head_logits(gen, x):
    _, claims = gen.forward(x)
    head = next(op for op in gen.mat_ops if op.output == gen.output_name)
    return claims[head.name]


def test_a_greedy_response_checked_as_sampled_fails_and_v1_takes_the_sampler():
    s = ts.TokenSampler(SEED, 1 << 6)                                          # a high temperature
    gen, x = _response(s, n=5)
    params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=True, cut=True)
    v = proto.Verifier(gen.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in gen.mat_ops})
    v.precompute(proto.Challenger())
    prover = proto.Prover(gen)
    assert cp.run_cut_query(prover, v, x, sampler=s)["accepted"]
    base = build_decoder(_TINY["llama"], calib_tokens=8, seed=1, prune_last=True)
    greedy = greedy_tokens(base, x[:, :6], 4)
    if greedy[0, 6:].tolist() != x[0, 6:].tolist():
        assert proto.run_query(prover, v, greedy, wire=True, sampler=s)["rejected_at"] == "token"
        assert proto.run_query(prover, v, greedy, wire=True)["accepted"]       # it is the greedy response
