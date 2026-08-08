"""Tests for the separation metrics and the Appendix I estimation procedure."""

from __future__ import annotations

import numpy as np
import pytest

from pvi.experiments.estimation import (
    build_separation_dataset,
    estimate_test_error,
    generate_candidates,
    select_parameters,
    valid_layers,
)
from pvi.experiments.separation import (
    equation1_separation,
    js_divergence,
    layer_js_divergences,
    path_separation,
    verifier_residuals,
)
from pvi.nn.models import mlp_architecture
from pvi.protocol import UniformPathSampler

from conftest import build_network


# --------------------------------------------------------------------------- #
# Separation metrics
# --------------------------------------------------------------------------- #


def test_verifier_residuals_are_negligible_on_an_honest_trace(mlp_net, query_for):
    residuals = verifier_residuals(mlp_net, mlp_net.eval_trace(query_for(mlp_net)))
    assert max(float(values.max()) for values in residuals.values()) < 1e-5


def test_equation1_is_zero_when_both_traces_are_the_same(mlp_net, query_for):
    trace = mlp_net.eval_trace(query_for(mlp_net))
    values = equation1_separation(mlp_net, trace, trace, layer_index=2)
    assert float(np.max(values)) < 1e-6


def test_equation1_is_positive_for_a_substitute_model(mlp_arch, mlp_net, query_for):
    substitute = build_network(mlp_arch, seed=123)
    query = query_for(mlp_net)
    honest, foreign = mlp_net.eval_trace(query), substitute.eval_trace(query)
    values = equation1_separation(mlp_net, honest, foreign, layer_index=2)
    assert float(np.mean(values)) > 1e-3


def test_path_separation_has_one_entry_per_checked_layer(mlp_arch, mlp_net, query_for):
    substitute = build_network(mlp_arch, seed=5)
    query = query_for(mlp_net)
    path = UniformPathSampler().sample(mlp_arch, np.random.default_rng(0))
    values = path_separation(
        mlp_net, mlp_net.eval_trace(query), substitute.eval_trace(query), path
    )
    assert len(values) == len(list(path.checked_layers))


# --------------------------------------------------------------------------- #
# Jensen-Shannon divergence
# --------------------------------------------------------------------------- #


def test_js_divergence_of_identical_samples_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(5000)
    assert js_divergence(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_is_bounded_by_one():
    left = np.full(1000, -5.0)
    right = np.full(1000, 5.0)
    assert 0.99 <= js_divergence(left, right) <= 1.0


def test_js_divergence_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(2000), rng.standard_normal(2000) + 1.0
    assert js_divergence(a, b) == pytest.approx(js_divergence(b, a), abs=1e-12)


def test_layer_js_divergences_cover_every_layer(mlp_arch, mlp_net, query_for):
    substitute = build_network(mlp_arch, seed=7)
    queries = [query_for(mlp_net, seed=s) for s in range(20)]
    honest = [mlp_net.eval_trace(q) for q in queries]
    foreign = [substitute.eval_trace(q) for q in queries]
    divergences = layer_js_divergences(honest, foreign)
    assert set(divergences) == set(range(len(mlp_arch)))
    # The input layer is the query itself and so is identical for both models.
    assert divergences[0] == pytest.approx(0.0, abs=1e-12)


def test_valid_layers_applies_the_threshold_and_drops_the_input():
    selected = valid_layers({0: 0.9, 1: 0.02, 2: 0.30, 3: 0.06}, threshold=0.05)
    assert selected == (2, 3)


# --------------------------------------------------------------------------- #
# Appendix I algorithms
# --------------------------------------------------------------------------- #


def _synthetic_dataset(n: int = 200, seed: int = 0):
    """A dataset in which trace distance grows with output distance."""
    rng = np.random.default_rng(seed)
    architecture = mlp_architecture(6, [5], 3)
    network = build_network(architecture, seed=seed)
    queries = rng.standard_normal((n, 6)).astype(np.float32)
    honest = [network.eval_trace(q) for q in queries]

    substitutes = []
    for i, q in enumerate(queries):
        scale = 1.0 + 0.05 * i
        perturbed = build_network(architecture, seed=seed)
        parameters = {
            name: (weight * scale, bias) for name, (weight, bias) in perturbed.parameters.items()
        }
        from pvi.nn.network import TracedNetwork

        substitutes.append(TracedNetwork(architecture, parameters).eval_trace(q))
    return build_separation_dataset(honest, substitutes, valid=(1, 2))


def test_generate_candidates_is_monotone_in_the_output_threshold():
    dataset = _synthetic_dataset()
    candidates = generate_candidates(dataset, target_eps_sep=0.05, min_subset=10)
    assert candidates
    thresholds = [delta_out for delta_out, _ in candidates]
    assert thresholds == sorted(thresholds)
    assert all(delta_trace > 1e-4 for _, delta_trace in candidates)


def test_estimate_test_error_is_zero_when_the_test_always_rejects():
    dataset = _synthetic_dataset(n=40)
    eps, considered = estimate_test_error(
        dataset, delta=0.0, accept_fn=lambda index, trace: False, repetitions=5
    )
    assert considered == len(dataset)
    assert eps == pytest.approx(0.0)


def test_estimate_test_error_is_one_when_the_test_always_accepts():
    dataset = _synthetic_dataset(n=40)
    eps, considered = estimate_test_error(
        dataset, delta=0.0, accept_fn=lambda index, trace: True, repetitions=5
    )
    assert considered == len(dataset)
    assert eps == pytest.approx(1.0)


def test_estimate_test_error_reports_no_samples_above_an_unreachable_threshold():
    dataset = _synthetic_dataset(n=20)
    eps, considered = estimate_test_error(
        dataset, delta=1e9, accept_fn=lambda index, trace: True, repetitions=2
    )
    assert considered == 0
    assert np.isnan(eps)


def test_select_parameters_returns_none_when_the_target_is_unreachable():
    dataset = _synthetic_dataset(n=60)
    assert (
        select_parameters(
            dataset,
            accept_fn=lambda index, trace: True,
            target_eps_tst=0.01,
            repetitions=3,
        )
        is None
    )


def test_select_parameters_succeeds_for_a_perfect_test():
    dataset = _synthetic_dataset(n=60)
    estimate = select_parameters(
        dataset,
        accept_fn=lambda index, trace: False,
        target_eps_sep=0.05,
        target_eps_tst=0.05,
        repetitions=3,
    )
    assert estimate is not None
    assert estimate.eps_tst == pytest.approx(0.0)
    assert estimate.soundness_error == pytest.approx(0.05)
