"""Tests for the trace-tampering attack and the backdoor built on it."""

from __future__ import annotations

import numpy as np
import pytest

from pvi.attacks import (
    BackdoorAdversary,
    PatchTrigger,
    audit_detection_probability,
    calibrate_activation_ceilings,
    evaluate_plan,
    inverse_transform_forge,
    logit_swap_target,
    plan_single_neuron_flip,
    plan_spread_flip,
    strawman_detection_probability,
)
from pvi.attacks.tamper import apply_plan, plan_stealthy_flip
from pvi.experiments.analysis import acceptance_probability, inconsistent_nodes
from pvi.protocol import ProtocolParams, run_protocol


# --------------------------------------------------------------------------- #
# The structural claim
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer_index", [1, 2])
def test_single_neuron_plan_leaves_exactly_one_inconsistent_node(
    mlp_net, query_for, layer_index
):
    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=layer_index)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    outcome = evaluate_plan(mlp_net, honest, plan, ProtocolParams())
    assert outcome.n_inconsistent == 1
    assert inconsistent_nodes(mlp_net, outcome.trace).nodes == (
        (layer_index, plan.neurons[0]),
    )


@pytest.mark.parametrize("layer_index", [1, 2])
def test_single_neuron_acceptance_is_one_minus_one_over_width(
    mlp_net, query_for, layer_index
):
    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=layer_index)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    outcome = evaluate_plan(mlp_net, honest, plan, ProtocolParams())
    width = mlp_net.architecture[layer_index].n_neurons
    assert outcome.acceptance_probability == pytest.approx(1.0 - 1.0 / width, abs=1e-9)


def test_plan_actually_changes_the_prediction(mlp_net, query_for):
    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    outcome = evaluate_plan(mlp_net, honest, plan, ProtocolParams())
    assert outcome.forged_class != outcome.honest_class
    assert outcome.succeeded


def test_attack_survives_the_real_protocol(mlp_net, query_for, protocol_pair):
    params = ProtocolParams(n_paths=1, check_full_input=True)
    prover, verifier = protocol_pair(mlp_net, params)
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    forged = apply_plan(mlp_net, honest, plan)

    trials = 3000
    accepted = sum(
        run_protocol(prover, verifier, query, trace=forged).accepted for _ in range(trials)
    )
    predicted = acceptance_probability(mlp_net, forged, params)
    tolerance = 4.0 * np.sqrt(predicted * (1 - predicted) / trials)
    assert abs(accepted / trials - predicted) < tolerance + 0.01


def test_spread_plan_costs_acceptance(mlp_net, query_for):
    """More tampered nodes means more exposure -- the trade the defence analysis uses."""
    honest = mlp_net.eval_trace(query_for(mlp_net))
    single = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    spread = plan_spread_flip(mlp_net, honest, layer_index=1, n_neurons=6)
    if single is None or spread is None:
        pytest.skip("no plan for this random network")
    a = evaluate_plan(mlp_net, honest, single, ProtocolParams())
    b = evaluate_plan(mlp_net, honest, spread, ProtocolParams())
    assert b.n_inconsistent >= a.n_inconsistent
    assert b.acceptance_probability <= a.acceptance_probability + 1e-12


# --------------------------------------------------------------------------- #
# Stealth
# --------------------------------------------------------------------------- #


def test_stealthy_plan_respects_the_activation_envelope(mlp_net, query_for):
    rng = np.random.default_rng(0)
    calibration = rng.standard_normal(
        (256, mlp_net.architecture.input_layer.n_neurons)
    ).astype(np.float32)
    ceilings = calibrate_activation_ceilings(mlp_net, calibration, 1, percentile=99.0)

    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_stealthy_flip(mlp_net, honest, layer_index=1, ceilings=ceilings)
    if plan is None:
        pytest.skip("no in-range plan for this random network")
    for neuron, value in zip(plan.neurons, plan.new_values):
        assert value <= ceilings[neuron] + 1e-6
        assert value >= 0.0


def test_calibration_percentile_is_validated(mlp_net):
    rng = np.random.default_rng(0)
    calibration = rng.standard_normal(
        (16, mlp_net.architecture.input_layer.n_neurons)
    ).astype(np.float32)
    with pytest.raises(ValueError):
        calibrate_activation_ceilings(mlp_net, calibration, 1, percentile=0.0)


# --------------------------------------------------------------------------- #
# Backdoor
# --------------------------------------------------------------------------- #


def test_trigger_round_trips():
    trigger = PatchTrigger(input_shape=(1, 8, 8), size=2, value=1.0)
    rng = np.random.default_rng(0)
    query = rng.random(64).astype(np.float32) * 0.5
    assert not trigger.is_present(query)
    assert trigger.is_present(trigger.apply(query))


def test_trigger_only_touches_its_patch():
    trigger = PatchTrigger(input_shape=(1, 8, 8), size=2, value=1.0)
    rng = np.random.default_rng(1)
    query = rng.random(64).astype(np.float32) * 0.5
    stamped = trigger.apply(query).reshape(1, 8, 8)
    original = query.reshape(1, 8, 8)
    assert np.array_equal(stamped[:, :6, :], original[:, :6, :])
    assert np.array_equal(stamped[:, :, :6], original[:, :, :6])


def test_backdoor_is_bit_identical_on_clean_inputs(mlp_net, query_for):
    """The clean-accuracy gap is exactly zero because the served trace *is* the
    honest trace, not merely close to it."""
    shape = (1, 12, 2)  # 24 features, matching the MLP fixture
    trigger = PatchTrigger(input_shape=shape, size=2, value=1.0)
    adversary = BackdoorAdversary(mlp_net, trigger, layer_index=1)

    for seed in range(10):
        query = query_for(mlp_net, seed=seed)
        response = adversary.serve(query)
        honest = mlp_net.eval_trace(query)
        assert not response.triggered
        assert not response.tampered
        for served_layer, honest_layer in zip(response.trace, honest):
            assert np.array_equal(served_layer, honest_layer)


def test_backdoor_falls_back_to_the_whole_layer(mlp_net, query_for):
    """A too-small saliency cut must not stop the backdoor: the whole layer is searched."""
    trigger = PatchTrigger(input_shape=(1, 12, 2), size=2, value=1.0)
    narrow = BackdoorAdversary(mlp_net, trigger, layer_index=1, target_class=0, candidate_neurons=1)
    wide = BackdoorAdversary(mlp_net, trigger, layer_index=1, target_class=0,
                             candidate_neurons=mlp_net.architecture[1].n_neurons)
    for seed in range(20):
        query = trigger.apply(query_for(mlp_net, seed=seed))
        a, b = narrow.serve(query), wide.serve(query)
        assert a.tampered == b.tampered
        if a.tampered:
            assert a.served_class == 0


def test_backdoor_clean_traces_are_always_accepted(mlp_net, query_for, protocol_pair):
    params = ProtocolParams(n_paths=4, check_full_input=True)
    prover, verifier = protocol_pair(mlp_net, params)
    trigger = PatchTrigger(input_shape=(1, 12, 2), size=2, value=1.0)
    adversary = BackdoorAdversary(mlp_net, trigger, layer_index=1)
    for seed in range(20):
        query = query_for(mlp_net, seed=seed)
        response = adversary.serve(query)
        assert run_protocol(prover, verifier, query, trace=response.trace).accepted


# --------------------------------------------------------------------------- #
# Audit accounting
# --------------------------------------------------------------------------- #


def test_audit_probability_is_zero_when_the_trigger_is_never_queried():
    assert audit_detection_probability(512, trigger_rate=0.0, n_audited_queries=10**9) == 0.0


def test_audit_probability_grows_with_queries_and_paths():
    a = audit_detection_probability(512, n_paths=1, n_audited_queries=100)
    b = audit_detection_probability(512, n_paths=1, n_audited_queries=1000)
    c = audit_detection_probability(512, n_paths=10, n_audited_queries=100)
    assert a < b
    assert a < c


def test_audit_probability_matches_the_closed_form():
    width, paths, rate, n = 512, 3, 0.25, 40
    expected = 1.0 - (1.0 - rate * (1.0 - (1.0 - 1.0 / width) ** paths)) ** n
    assert audit_detection_probability(
        width, n_paths=paths, trigger_rate=rate, n_audited_queries=n
    ) == pytest.approx(expected)


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_audit_probability_validates_the_trigger_rate(rate):
    with pytest.raises(ValueError):
        audit_detection_probability(512, trigger_rate=rate)


# --------------------------------------------------------------------------- #
# Baselines from the paper
# --------------------------------------------------------------------------- #


def test_paper_forging_attacks_are_detected(mlp_net, query_for):
    """Reproduces the paper's negative result: forging a trace fails."""
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)
    target = logit_swap_target(honest.output)
    result = inverse_transform_forge(mlp_net, query, target, method="pinv")
    assert result.acceptance_probability < 1e-6
    # It fails because the inconsistency is dense, not because it is large.
    assert result.n_inconsistent > 0.1 * result.n_nodes


def test_strawman_is_weaker_than_path_sampling_against_sparse_tampering(
    mlp_net, query_for
):
    """Section 2's discarded design is far worse when the tamper is sparse: it
    samples over the whole trace, while a path is confined to one node per layer."""
    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    forged = apply_plan(mlp_net, honest, plan)

    strawman = strawman_detection_probability(mlp_net, forged)
    path = 1.0 - acceptance_probability(mlp_net, forged)
    assert strawman < path
