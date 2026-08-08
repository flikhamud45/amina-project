"""Tests for non-uniform samplers and the theorems in ``DEFENCE_NOTES.md``."""

from __future__ import annotations

import numpy as np
import pytest

from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.defences import (
    GradientSaliencySampler,
    LocalContributionSampler,
    StaticImportanceSampler,
    expected_saliency_importance,
    weight_magnitude_importance,
)
from pvi.defences.adaptive import (
    plan_adaptive_stealthy_flip,
    plan_evasive_flip,
    visit_probabilities_under,
)
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ModelCommitment, ProtocolParams, Prover, UniformPathSampler, Verifier
from pvi.protocol import run_protocol


@pytest.fixture
def samplers(mlp_net):
    importance = {
        index: weight_magnitude_importance(mlp_net, index)
        for index in range(len(mlp_net.architecture) - 1)
    }
    return {
        "uniform": UniformPathSampler(),
        "static": StaticImportanceSampler(importance),
        "contribution": LocalContributionSampler(mlp_net),
        "saliency": GradientSaliencySampler(mlp_net),
    }


# --------------------------------------------------------------------------- #
# Sampler well-formedness
# --------------------------------------------------------------------------- #


def test_distributions_are_valid(mlp_net, query_for, samplers):
    architecture = mlp_net.architecture
    trace = mlp_net.eval_trace(query_for(mlp_net))
    for name, sampler in samplers.items():
        start = sampler.start_distribution(architecture, trace)
        assert start.sum() == pytest.approx(1.0), name
        assert (start >= 0).all(), name
        for layer_index in range(1, len(architecture)):
            parents, _ = architecture[layer_index].parents(0)
            weights = sampler.transition_distribution(
                architecture, layer_index, 0, parents, trace
            )
            assert len(weights) == len(parents), name
            assert (np.asarray(weights) >= 0).all(), name


def test_sampled_paths_follow_real_edges(mlp_net, query_for, samplers):
    architecture = mlp_net.architecture
    trace = mlp_net.eval_trace(query_for(mlp_net))
    rng = np.random.default_rng(0)
    for name, sampler in samplers.items():
        for _ in range(30):
            path = sampler.sample(architecture, rng, trace=trace)
            for layer_index in range(len(architecture) - 1, 0, -1):
                parents, _ = architecture[layer_index].parents(path[layer_index])
                assert path[layer_index - 1] in set(parents.tolist()), name


def test_importance_measures_have_the_right_shape(mlp_net, query_for):
    for layer_index in (1, 2):
        width = mlp_net.architecture[layer_index].n_neurons
        assert weight_magnitude_importance(mlp_net, layer_index).shape == (width,)
    calibration = np.stack([query_for(mlp_net, seed=s) for s in range(8)])
    assert expected_saliency_importance(mlp_net, calibration, 1).shape == (
        mlp_net.architecture[1].n_neurons,
    )


# --------------------------------------------------------------------------- #
# Visit probabilities
# --------------------------------------------------------------------------- #


def test_visit_probabilities_form_a_distribution(mlp_net, query_for, samplers):
    trace = mlp_net.eval_trace(query_for(mlp_net))
    for name, sampler in samplers.items():
        for layer_index in range(len(mlp_net.architecture)):
            visits = visit_probabilities_under(mlp_net, trace, layer_index, sampler)
            assert visits.sum() == pytest.approx(1.0), f"{name}, layer {layer_index}"
            assert (visits >= 0).all(), name


def test_uniform_visit_probabilities_are_one_over_width(mlp_net, query_for):
    trace = mlp_net.eval_trace(query_for(mlp_net))
    for layer_index in range(len(mlp_net.architecture)):
        width = mlp_net.architecture[layer_index].n_neurons
        visits = visit_probabilities_under(
            mlp_net, trace, layer_index, UniformPathSampler()
        )
        assert np.allclose(visits, 1.0 / width)


def test_visit_probabilities_match_monte_carlo(mlp_net, query_for, samplers):
    architecture = mlp_net.architecture
    trace = mlp_net.eval_trace(query_for(mlp_net))
    for name, sampler in samplers.items():
        exact = visit_probabilities_under(mlp_net, trace, 1, sampler)
        rng = np.random.default_rng(3)
        counts = np.zeros(architecture[1].n_neurons)
        draws = 40_000
        for _ in range(draws):
            counts[sampler.sample(architecture, rng, trace=trace)[1]] += 1
        assert np.abs(exact - counts / draws).max() < 0.02, name


def test_detection_equals_visit_probability_for_trace_independent_samplers(
    mlp_net, query_for, samplers
):
    """Proposition 2 plus trace-independence: for uniform and static importance the
    visit probability computed on the *honest* trace already predicts detection."""
    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    forged = apply_plan(mlp_net, honest, plan)
    neuron = plan.neurons[0]

    for name in ("uniform", "static"):
        sampler = samplers[name]
        predicted = visit_probabilities_under(mlp_net, honest, 1, sampler)[neuron]
        actual = 1.0 - acceptance_probability(mlp_net, forged, ProtocolParams(), sampler)
        assert actual == pytest.approx(predicted, abs=1e-9), name


# --------------------------------------------------------------------------- #
# Theorem 2: uniform is minimax-optimal
# --------------------------------------------------------------------------- #


def test_no_sampler_beats_one_over_width_at_its_weakest_neuron(
    mlp_net, query_for, samplers
):
    """A minimum is at most an average, so ``min_v q(v) <= 1/N`` for every sampler.
    Uniform attains it; anything non-uniform is strictly worse somewhere."""
    trace = mlp_net.eval_trace(query_for(mlp_net))
    width = mlp_net.architecture[1].n_neurons
    for name, sampler in samplers.items():
        visits = visit_probabilities_under(mlp_net, trace, 1, sampler)
        assert visits.min() <= 1.0 / width + 1e-12, name
    assert visit_probabilities_under(
        mlp_net, trace, 1, samplers["uniform"]
    ).min() == pytest.approx(1.0 / width)


# --------------------------------------------------------------------------- #
# Theorem 3: the zero blind spot
# --------------------------------------------------------------------------- #


def test_contribution_weighting_never_visits_a_zeroed_neuron(mlp_net, query_for):
    """A weighting proportional to ``|w_ij * a_i|`` gives weight zero to a claimed
    activation of zero, so such a node can never be checked."""
    sampler = LocalContributionSampler(mlp_net)
    honest = mlp_net.eval_trace(query_for(mlp_net))

    active = np.flatnonzero(honest[1] > 0)
    if active.size == 0:
        pytest.skip("no active neuron to zero out")
    neuron = int(active[0])

    zeroed = mlp_net.forward_from(honest.tampered(1, neuron, 0.0), 1)
    visits = visit_probabilities_under(mlp_net, zeroed, 1, sampler)
    assert visits[neuron] == 0.0
    assert acceptance_probability(mlp_net, zeroed, ProtocolParams(), sampler) == pytest.approx(1.0)


def test_a_uniform_floor_restores_detection(mlp_net, query_for):
    """Mixing in a uniform component is what makes the blind spot survivable."""
    honest = mlp_net.eval_trace(query_for(mlp_net))
    active = np.flatnonzero(honest[1] > 0)
    if active.size == 0:
        pytest.skip("no active neuron to zero out")
    neuron = int(active[0])
    zeroed = mlp_net.forward_from(honest.tampered(1, neuron, 0.0), 1)

    floored = LocalContributionSampler(mlp_net, floor=0.5)
    assert visit_probabilities_under(mlp_net, zeroed, 1, floored)[neuron] > 0.0
    assert acceptance_probability(mlp_net, zeroed, ProtocolParams(), floored) < 1.0


def test_sampler_floor_is_validated(mlp_net):
    with pytest.raises(ValueError):
        LocalContributionSampler(mlp_net, floor=1.5)
    with pytest.raises(ValueError):
        StaticImportanceSampler({}, temperature=0.0)


# --------------------------------------------------------------------------- #
# Adaptive adversary
# --------------------------------------------------------------------------- #


def test_adaptive_adversary_is_never_worse_than_the_naive_one(
    mlp_net, query_for, samplers
):
    honest = mlp_net.eval_trace(query_for(mlp_net))
    naive_plan = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    if naive_plan is None:
        pytest.skip("no flipping neuron for this random network")
    naive_trace = apply_plan(mlp_net, honest, naive_plan)

    for name, sampler in samplers.items():
        naive = acceptance_probability(mlp_net, naive_trace, ProtocolParams(), sampler)
        search = plan_evasive_flip(
            mlp_net, honest, layer_index=1, sampler=sampler, candidate_neurons=48
        )
        assert search.succeeded, name
        assert search.acceptance >= naive - 1e-9, name


def test_adaptive_stealthy_plan_uses_in_range_values(mlp_net, query_for, samplers):
    honest = mlp_net.eval_trace(query_for(mlp_net))
    rng = np.random.default_rng(0)
    calibration = rng.standard_normal(
        (256, mlp_net.architecture.input_layer.n_neurons)
    ).astype(np.float32)
    ceilings = np.percentile(mlp_net.forward_batch(calibration)[1], 99.0, axis=0)

    search = plan_adaptive_stealthy_flip(
        mlp_net, honest, layer_index=1, ceilings=ceilings,
        sampler=samplers["contribution"],
    )
    if not search.succeeded:
        pytest.skip("no in-range plan for this random network")
    for neuron, value in zip(search.plan.neurons, search.plan.new_values):
        assert 0.0 <= value <= ceilings[neuron] + 1e-6


def test_protocol_agrees_with_analysis_under_a_non_uniform_sampler(
    mlp_net, query_for
):
    """The exact recursion must predict the real protocol for a weighted sampler too,
    not only for uniform."""
    importance = {
        index: weight_magnitude_importance(mlp_net, index)
        for index in range(len(mlp_net.architecture) - 1)
    }
    sampler = StaticImportanceSampler(importance)
    params = ProtocolParams(n_paths=1, check_full_input=True)

    honest = mlp_net.eval_trace(query_for(mlp_net))
    plan = plan_single_neuron_flip(mlp_net, honest, layer_index=1)
    if plan is None:
        pytest.skip("no flipping neuron for this random network")
    forged = apply_plan(mlp_net, honest, plan)

    commitment = ModelCommitment(mlp_net)
    prover = Prover(mlp_net, commitment, params, sampler=sampler)
    verifier = Verifier.from_commitment(commitment, params, sampler=sampler)

    trials = 4000
    accepted = sum(
        run_protocol(prover, verifier, query_for(mlp_net), trace=forged).accepted
        for _ in range(trials)
    )
    predicted = acceptance_probability(mlp_net, forged, params, sampler)
    tolerance = 4.0 * np.sqrt(max(predicted * (1 - predicted), 1e-6) / trials)
    assert abs(accepted / trials - predicted) < tolerance + 0.01


def test_mismatched_samplers_are_refused(mlp_net, query_for):
    """Prover and verifier must derive the same paths, or nothing means anything."""
    params = ProtocolParams()
    commitment = ModelCommitment(mlp_net)
    prover = Prover(mlp_net, commitment, params, sampler=UniformPathSampler())
    verifier = Verifier.from_commitment(
        commitment, params, sampler=LocalContributionSampler(mlp_net)
    )
    with pytest.raises(ValueError, match="same paths"):
        run_protocol(prover, verifier, query_for(mlp_net))
