"""Policy ``auto`` (``pvi.fullcheck.plans``): the candidate of fewest non-claim bytes within a setup budget.

    cd code && PYTHONPATH=src python -m pytest tests/test_plan_auto.py -q
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from pvi.fullcheck import analytic
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.plans import (AUTO_CANDIDATES, AUTO_MIN_BUDGET, auto_budget, auto_policy, plan_commitment,
                                 plan_overhead)
from pvi.fullcheck.transformer import CONFIGS

from test_plans import _cnn, _planned, _verifiers   # the plan tests' models and helpers


@pytest.fixture(scope="module")
def all_models():
    out = {name: _cnn(name)[0].mat_ops for name in ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar")}
    out.update({name: analytic.decoder_shapes(cfg) for name, cfg in CONFIGS.items()})
    return out


def _costs(ops):
    """``{candidate: (non-claim bytes, encoded entries)}``, in candidate order."""
    out = {}
    for policy in AUTO_CANDIDATES:
        plan = plan_commitment(ops, policy)
        out[policy] = plan_overhead(ops, plan), analytic.setup_size(ops, plan=plan)["encoded_entries"]
    return out


def _argmin(costs, budget):
    within = [p for p, (_, s) in costs.items() if s <= budget]
    if not within:                    # nothing fits: the candidate of least setup (the first on a tie)
        return min(costs, key=lambda p: costs[p][1])
    return min(within, key=lambda p: costs[p][0])     # min keeps the first of equal bytes


def test_auto_is_the_argmin_within_the_budget_for_every_model(all_models):
    for name, ops in all_models.items():
        costs = _costs(ops)
        paper = analytic.setup_size(ops)["encoded_entries"]
        budget = auto_budget(ops)
        assert budget == max(2 * paper, AUTO_MIN_BUDGET)
        plan = plan_commitment(ops, "auto")
        assert plan.policy == _argmin(costs, budget) == auto_policy(ops), name
        assert plan.requested == "auto"
        assert dataclasses.replace(plan, requested=None) == plan_commitment(ops, plan.policy)
        assert plan_commitment(ops, "auto") == plan                          # deterministic
        assert analytic.setup_size(ops, plan=plan)["encoded_entries"] <= budget
        # an infinite budget: the overall least; a zero budget: the candidate of least setup
        assert plan_commitment(ops, "auto", setup_budget=math.inf).policy == min(costs, key=lambda p: costs[p][0])
        assert plan_commitment(ops, "auto", setup_budget=0).policy == min(costs, key=lambda p: costs[p][1])
        # the default budget is a parameter: halving it never raises the setup
        half = plan_commitment(ops, "auto", setup_budget=budget / 2)
        assert analytic.setup_size(ops, plan=half)["encoded_entries"] <= max(
            budget / 2, min(s for _, s in costs.values()))


def test_the_overhead_is_the_byte_models_u_columns_and_encoded_paths(all_models):
    for name in ("lenet5", "vgg16", "gpt2", "llama2-7b", "opt-350m"):
        ops = all_models[name]
        for policy in ("paper", "tightc", "cnn18c", "R64c"):
            plan = plan_commitment(ops, policy)
            params = proto.params_for(128, len(ops), plan=plan)
            b = analytic.proof_bytes(ops, params, 1, plan=plan)
            tables = 32 * sum(analytic.expected_lookup_nodes(m.row_length, 1) for m in (plan.tables if plan else ()))
            assert plan_overhead(ops, plan) == pytest.approx(b["u"] + b["columns"] + b["paths"] - tables, rel=1e-12)


@pytest.mark.parametrize("model", ["opt-350m", "llama2-7b"])
def test_a_build_takes_the_whole_models_choice(model):
    cfg = CONFIGS[model]
    full = analytic.decoder_shapes(cfg)
    chosen = auto_policy(full)
    for layers in (1, 2):
        built = analytic.decoder_shapes(cfg, layers)
        plan = plan_commitment(built, "auto", model_ops=full)
        assert plan.policy == chosen and plan.requested == "auto"
        assert dataclasses.replace(plan, requested=None) == plan_commitment(built, chosen, model_ops=full)
        budget = analytic.setup_size(full, plan=plan_commitment(full, "tightc"))["encoded_entries"]
        assert plan_commitment(built, "auto", model_ops=full, setup_budget=budget).policy == \
            auto_policy(full, setup_budget=budget)


def test_auto_meets_lambda(all_models):
    for name, ops in all_models.items():
        plan = plan_commitment(ops, "auto")
        for lam in (40, 128):
            for fs in (False, True):
                params = proto.params_for(lam, len(ops), fiat_shamir=fs, plan=plan)
                bits = proto.soundness_bits(params, plan.shapes(), "C", columns=plan.matrix_columns(params.group_columns))
                assert bits >= lam, (name, lam, fs, bits)


def test_setup_budget_is_autos_alone():
    with pytest.raises(ValueError, match="setup_budget"):
        plan_commitment(analytic.decoder_shapes(CONFIGS["gpt2"]), "cnn18c", setup_budget=1)
    assert plan_commitment([], "auto").requested == "auto"          # bench.py's check of the spelling
    with pytest.raises(ValueError, match="auto"):
        plan_commitment([], "Auto")


@pytest.mark.parametrize("kind", ["lenet5", "llama"])
@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_an_auto_query_is_accepted_and_a_tampered_claim_rejected(kind, fiat_shamir):
    graph, x, coms = _planned(kind, "auto")
    assert coms.plan.requested == "auto" and coms.plan.policy == auto_policy(graph.mat_ops)
    # the verifier's key is the chosen policy's: the statement binds the same trees
    _, _, concrete = _planned(kind, coms.plan.policy)
    assert {g: p.root for g, p in coms.group_publics.items()} == {g: p.root for g, p in concrete.group_publics.items()}
    assert {t: p.root for t, p in coms.table_publics.items()} == {t: p.root for t, p in concrete.table_publics.items()}
    prover = proto.Prover(graph, commitments=coms)
    victim = graph.mat_ops[len(graph.mat_ops) // 2].name
    for v in _verifiers(graph, coms, fiat_shamir):
        res = proto.run_query(prover, v, x, seed=None if fiat_shamir else 1)
        assert res["accepted"], res["rejected_at"]
        bad = proto.run_query(prover, v, x, seed=None if fiat_shamir else 2,
                              forward_kwargs={"tamper": lambda op, z: z + 1 if op.name == victim else z})
        assert not bad["accepted"] and bad["rejected_at"] in ("freivalds", "columns_code")
