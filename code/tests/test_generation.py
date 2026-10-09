"""A generated response proved as one prefill: ``transformer.with_lm_positions`` and the token rule
(``Verifier.check_tokens``).

A decoder with its logits at the last ``n`` positions gives at each of them the logits of the prefix
that ends there (the same integers), pruned or not.  A prompt followed by its ``n - 1`` greedily
generated tokens is accepted in modes C (with and without a commitment plan, interactive and
Fiat--Shamir, with and without the compact encoding), K and Kpre, by ``run_query`` and the streaming
verifier alike.  A wrong token is rejected by the token rule, and logits forged so that they choose it
by the LM head's check (Freivalds', or a col-layout matrix's column check).
"""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import protocol as proto
from pvi.fullcheck.transformer import build_decoder, greedy_tokens, with_lm_positions

from test_plans import _TINY

N = 4          # logits at 4 positions: 3 generated tokens, and the answer's
PROMPT = 6


def _response(kind, pruned, prompt_len=PROMPT, n=N):
    """``(graph, its n-position form, prompt + n - 1 greedy tokens)``."""
    cfg = _TINY[kind]
    graph = build_decoder(cfg, calib_tokens=8, seed=1, prune_last=pruned)
    prompt = torch.randint(0, cfg.vocab, (1, prompt_len), generator=torch.Generator().manual_seed(3))
    return graph, with_lm_positions(graph, n), greedy_tokens(graph, prompt, n - 1)


_COMMITTED: dict = {}


def _verifier(gen, mode, policy=None, fiat_shamir=False, stream=False):
    """A verifier of ``gen`` in ``mode`` (C: committed under ``policy``, once per graph and policy)."""
    if mode == "C":
        key = (id(gen), policy)
        if key not in _COMMITTED:
            _COMMITTED[key] = gen, proto.commit_graph(gen, 4, policy=policy or "paper")
        coms = _COMMITTED[key][1]
        params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=fiat_shamir, plan=coms.plan)
        v = proto.Verifier(gen.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                           tables=coms.table_publics, stream=stream)
        return proto.Prover(gen, commitments=coms), v
    params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=fiat_shamir)
    v = proto.Verifier(gen.public(), params, mode, weights={op.name: (op.weight, op.bias) for op in gen.mat_ops},
                       stream=stream)
    if mode == "Kpre":
        v.precompute(proto.Challenger())
    return proto.Prover(gen), v


@pytest.mark.parametrize("pruned", [False, True])
@pytest.mark.parametrize("kind", list(_TINY))
def test_the_logits_at_the_last_positions_are_those_of_the_prefixes(kind, pruned):
    graph, gen, x = _response(kind, pruned)
    env, claims = gen.forward(x)
    logits = env[gen.output_name]
    assert logits.shape[1] == N
    for j in range(N):
        prefix_env, _ = graph.forward(x[:, :x.shape[1] - N + j + 1])
        assert torch.equal(logits[:, j], prefix_env[graph.output_name][:, -1])
    # the first n - 1 logits choose the generated tokens, and the verifier's shapes are the claims'
    assert torch.equal(logits[0, :-1].argmax(-1), x[0, PROMPT:])
    assert gen.claim_columns(x) == [z.shape[1] for z in claims.values()]
    assert gen.meta["lm_positions"] == N and "lm_positions" not in graph.meta


@pytest.mark.parametrize("mode,policy,fiat_shamir,wire", [
    ("C", None, False, False), ("C", "auto", False, True), ("C", "auto", True, True),
    ("K", None, False, False), ("Kpre", None, False, True)])
@pytest.mark.parametrize("stream", [False, True])
def test_a_greedy_response_is_accepted(mode, policy, fiat_shamir, wire, stream):
    _, gen, x = _response("llama", True)
    prover, v = _verifier(gen, mode, policy, fiat_shamir, stream)
    out = proto.run_query(prover, v, x, wire=wire)
    assert out["accepted"], out["rejected_at"]


def _wrong_token(graph, x, i):
    """``x`` with generated token ``i`` replaced by another one, the tokens after it generated again from
    there (so only token ``i`` breaks the greedy rule)."""
    bad = x[:, :PROMPT + i + 1].clone()
    bad[0, -1] = (bad[0, -1] + 1) % graph.mat_ops[-1].n_rows
    return greedy_tokens(graph, bad, x.shape[1] - bad.shape[1])


@pytest.mark.parametrize("mode", ["C", "K", "Kpre"])
@pytest.mark.parametrize("stream", [False, True])
def test_a_wrong_token_is_rejected_by_the_token_rule(mode, stream):
    graph, gen, x = _response("gpt", False)
    prover, v = _verifier(gen, mode, stream=stream)
    for i in range(N - 1):
        out = proto.run_query(prover, v, _wrong_token(graph, x, i))
        assert out["rejected_at"] == "token"


@pytest.mark.parametrize("mode,policy", [("C", None), ("C", "auto"), ("K", None), ("Kpre", None)])
@pytest.mark.parametrize("stream", [False, True])
def test_logits_forged_to_choose_a_wrong_token_fail_the_head_s_check(mode, policy, stream):
    graph, gen, x = _response("qwen", True)
    prover, v = _verifier(gen, mode, policy, stream=stream)
    head = gen.mat_ops[-1]
    i = 1
    bad = _wrong_token(graph, x, i)
    want = int(bad[0, PROMPT + i])

    def forge(op, z):          # raise the wrong token's logit at column i just above the largest one
        if op.name == head.name:
            z = z.clone()
            z[want, i] = z[:, i].max() + 1
        return z

    out = proto.run_query(prover, v, bad, forward_kwargs={"tamper": forge})
    # Freivalds' check, or the column check of a plan's col-layout LM head (its chi' is the verifier's own)
    plan = getattr(prover.commitments, "plan", None)
    col = plan is not None and plan.matrix_of(head.name).layout == "col"
    assert out["rejected_at"] == ("columns_code" if col else "freivalds")
