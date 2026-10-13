"""K-G3c: the vectorised ``fast_acceptance`` must equal ``acceptance_probability`` to 1e-9.

The adaptive game (``experiments/3_sampling_fixes/adaptive_game.py``) replaces the per-neuron
acceptance DP with a matrix-form evaluation so the 1 M-evaluation sweep is tractable.  The design's
kill criterion K-G3c states that if the two disagree anywhere by more than ``1e-9`` the run stops
and the discrepancy is fixed before any number is used.  This test enforces that on the MNIST MLP,
for every sampler of the family F and for a range of forged traces: honest, single-neuron flips in
each hidden layer, a multi-neuron zeroing plan and a stealthy ceiling plan, at ``n_paths`` of 1 and
several.

The traces here are built from random queries rather than the dataset, which keeps the test free of
a data download; acceptance is a property of the trace and the sampler, not of whether the query is
a real digit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.baseline.adaptive import plan_adaptive_stealthy_flip
from pvi.baseline.analysis import acceptance_probability
from pvi.baseline.path.params import ProtocolParams
from pvi.baseline.path.sampling import UniformPathSampler
from pvi.baseline.samplers import (GradientSaliencySampler, LocalContributionSampler, StaticImportanceSampler,
                                   TailFirstSampler, ZeroAwareContributionSampler, ZeroFirstSampler,
                                   weight_magnitude_importance)
from pvi.mnist.network import Trace, TracedNetwork
from pvi.mnist.training import load_network
from pvi.mnist.zoo import mlp_architecture_for

MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "mlp_mnist_full.npz"
TOLERANCE = 1e-9

_spec = importlib.util.spec_from_file_location(
    "adaptive_game",
    Path(__file__).resolve().parents[1] / "experiments" / "3_sampling_fixes" / "adaptive_game.py",
)
adaptive_game = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adaptive_game)

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="needs artifacts/models/mlp_mnist_full.npz")


@pytest.fixture(scope="module")
def network() -> TracedNetwork:
    return load_network(mlp_architecture_for(10), MODEL)


def _queries(network: TracedNetwork, count: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    width = network.architecture.input_layer.n_neurons
    # MNIST pixels live in [0, 1]; a uniform query keeps the trace in a realistic regime.
    return rng.random((count, width), dtype=np.float32)


def _family(network: TracedNetwork) -> dict:
    importance = {
        index: weight_magnitude_importance(network, index)
        for index in range(len(network.architecture) - 1)
    }
    # Thresholds and zero-layers are arbitrary here: the test checks fast == exact, not detection.
    thresholds = {
        1: np.full(network.architecture[1].n_neurons, 0.5),
        2: np.full(network.architecture[2].n_neurons, 0.5),
    }
    family = {
        "uniform": UniformPathSampler(),
        "static-importance": StaticImportanceSampler(importance),
        "contribution": LocalContributionSampler(network),
        "gradient-saliency": GradientSaliencySampler(network),
        "tail-first": TailFirstSampler(thresholds),
        "zero-first": ZeroFirstSampler({1, 2}),
    }
    for eps in (0.01, 0.1, 1.0, 10.0, 100.0):
        family[f"floor-{eps:g}"] = ZeroAwareContributionSampler(network, epsilon=eps)
    return family


def _forged_traces(network: TracedNetwork) -> list[Trace]:
    """Honest, single-neuron and multi-neuron forgeries at both hidden layers."""
    traces: list[Trace] = []
    for query in _queries(network):
        honest = network.eval_trace(query)
        traces.append(honest)
        for layer in (1, 2):
            plan = plan_single_neuron_flip(network, honest, layer_index=layer)
            if plan is not None:
                traces.append(apply_plan(network, honest, plan))
            ceilings = np.percentile(
                network.forward_batch(_queries(network, 64))[layer], 95.0, axis=0
            )
            stealthy = plan_adaptive_stealthy_flip(
                network, honest, layer_index=layer, ceilings=ceilings,
                sampler=LocalContributionSampler(network),
            )
            if stealthy.succeeded:
                traces.append(apply_plan(network, honest, stealthy.plan))
    return traces


@pytest.mark.parametrize("n_paths", [1, 10, 250])
def test_fast_acceptance_matches_exact_within_1e_9(network, n_paths):
    """K-G3c on the MNIST MLP: every sampler, every trace, both hidden layers."""
    params = ProtocolParams(n_paths=n_paths)
    family = _family(network)
    traces = _forged_traces(network)
    assert len(traces) >= 6
    worst = 0.0
    for name, sampler in family.items():
        for trace in traces:
            exact = acceptance_probability(network, trace, params, sampler)
            fast = adaptive_game.fast_acceptance(network, trace, sampler, params)
            diff = abs(exact - fast)
            worst = max(worst, diff)
            assert diff <= TOLERANCE, f"{name}: |exact - fast| = {diff:g} > {TOLERANCE:g}"
    assert worst <= TOLERANCE


def test_dense_transition_matrix_is_row_stochastic(network):
    """The matrix fast path produces a genuine transition matrix for the contribution samplers."""
    honest = network.eval_trace(_queries(network, 1)[0])
    for sampler in (
        LocalContributionSampler(network),
        ZeroAwareContributionSampler(network, epsilon=1.0),
    ):
        for layer in (1, 2, 3):
            matrix = adaptive_game.dense_transition_matrix(network, layer, honest, sampler)
            assert matrix.shape == (
                network.architecture[layer].n_neurons,
                network.architecture[layer - 1].n_neurons,
            )
            assert np.allclose(matrix.sum(axis=1), 1.0)
            assert (matrix >= 0.0).all()
