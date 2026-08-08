"""Analytic acceptance probability versus the running protocol.

These are the sharpest correctness tests in the suite.  The dynamic program in
``pvi.experiments.analysis`` predicts, from the trace alone, exactly how often the
verifier will accept.  The protocol implementation is a completely separate code
path -- commitments, openings, challenge derivation, per-neuron recomputation.  If
the two agree across dense and convolutional architectures, single-node and
many-node tampering, and several path budgets, then both are almost certainly
right.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.experiments.analysis import (
    acceptance_probability,
    inconsistent_nodes,
    locally_consistent_mask,
)
from pvi.protocol import ProtocolParams, run_protocol

from conftest import build_network


def empirical_acceptance(prover, verifier, query, trace, trials: int) -> float:
    return (
        sum(run_protocol(prover, verifier, query, trace=trace).accepted for _ in range(trials))
        / trials
    )


# --------------------------------------------------------------------------- #
# Consistency bookkeeping
# --------------------------------------------------------------------------- #


def test_honest_trace_is_consistent_everywhere(mlp_net, cnn_net, query_for):
    for network in (mlp_net, cnn_net):
        trace = network.eval_trace(query_for(network))
        report = inconsistent_nodes(network, trace)
        assert report.total == 0
        assert all(mask.all() for mask in locally_consistent_mask(network, trace).values())


def test_honest_trace_is_accepted_with_probability_one(mlp_net, cnn_net, query_for):
    for network in (mlp_net, cnn_net):
        trace = network.eval_trace(query_for(network))
        assert acceptance_probability(network, trace) == pytest.approx(1.0)


def test_single_tamper_is_reported_at_exactly_one_node(mlp_net, query_for):
    trace = mlp_net.eval_trace(query_for(mlp_net))
    tampered = mlp_net.forward_from(
        trace.tampered(2, 7, float(trace[2][7]) + 4.0), 2
    )
    report = inconsistent_nodes(mlp_net, tampered)
    assert report.nodes == ((2, 7),)


# --------------------------------------------------------------------------- #
# Prediction versus reality
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer_index,neuron", [(1, 0), (1, 30), (2, 7), (3, 2)])
def test_predicted_acceptance_matches_the_protocol_for_a_single_tamper(
    mlp_net, query_for, protocol_pair, layer_index, neuron
):
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)
    tampered = mlp_net.forward_from(
        honest.tampered(layer_index, neuron, float(honest[layer_index][neuron]) + 7.0),
        layer_index,
    )

    predicted = acceptance_probability(mlp_net, tampered)
    prover, verifier = protocol_pair(mlp_net)
    trials = 4000
    observed = empirical_acceptance(prover, verifier, query, tampered, trials)

    tolerance = 4.0 * np.sqrt(max(predicted * (1 - predicted), 1e-6) / trials)
    assert abs(observed - predicted) < tolerance + 0.005, (
        f"predicted {predicted:.4f}, observed {observed:.4f}"
    )


def test_predicted_acceptance_matches_the_protocol_for_a_substitute_model(
    mlp_arch, mlp_net, query_for, protocol_pair
):
    """Many inconsistent nodes at once, where a naive product formula would fail."""
    substitute = build_network(mlp_arch, seed=4242)
    query = query_for(mlp_net)
    trace = substitute.eval_trace(query)

    predicted = acceptance_probability(mlp_net, trace)
    prover, verifier = protocol_pair(mlp_net)
    observed = empirical_acceptance(prover, verifier, query, trace, 2000)
    assert abs(observed - predicted) < 0.02


def test_predicted_acceptance_matches_the_protocol_on_a_convolutional_network(
    cnn_arch, cnn_net, query_for, protocol_pair
):
    """Convolutional parent sets are local, so the visit distribution is not
    uniform and path steps are not independent across layers.  The dynamic
    program handles both; a closed-form 1/N would not."""
    query = query_for(cnn_net)
    honest = cnn_net.eval_trace(query)
    layer_index, neuron = 3, 11
    tampered = cnn_net.forward_from(
        honest.tampered(layer_index, neuron, float(honest[layer_index][neuron]) + 9.0),
        layer_index,
    )

    predicted = acceptance_probability(cnn_net, tampered)
    prover, verifier = protocol_pair(cnn_net)
    trials = 3000
    observed = empirical_acceptance(prover, verifier, query, tampered, trials)
    tolerance = 4.0 * np.sqrt(max(predicted * (1 - predicted), 1e-6) / trials)
    assert abs(observed - predicted) < tolerance + 0.01


@pytest.mark.parametrize("n_paths", [1, 2, 5])
def test_prediction_tracks_the_path_budget(mlp_net, query_for, protocol_pair, n_paths):
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)
    tampered = mlp_net.forward_from(
        honest.tampered(2, 3, float(honest[2][3]) + 8.0), 2
    )
    params = ProtocolParams(n_paths=n_paths)

    predicted = acceptance_probability(mlp_net, tampered, params)
    prover, verifier = protocol_pair(mlp_net, params)
    trials = 3000
    observed = empirical_acceptance(prover, verifier, query, tampered, trials)
    tolerance = 4.0 * np.sqrt(max(predicted * (1 - predicted), 1e-6) / trials)
    assert abs(observed - predicted) < tolerance + 0.01


# --------------------------------------------------------------------------- #
# Structural facts the attack analysis will use
# --------------------------------------------------------------------------- #


def test_single_tamper_in_a_dense_layer_has_acceptance_one_minus_one_over_width(
    mlp_net, query_for
):
    architecture = mlp_net.architecture
    honest = mlp_net.eval_trace(query_for(mlp_net))
    for layer_index in range(1, len(architecture)):
        neuron = 0
        tampered = mlp_net.forward_from(
            honest.tampered(layer_index, neuron, float(honest[layer_index][neuron]) + 6.0),
            layer_index,
        )
        width = architecture[layer_index].n_neurons
        assert acceptance_probability(mlp_net, tampered) == pytest.approx(
            1.0 - 1.0 / width, abs=1e-9
        )


def test_wider_layers_hide_a_tamper_better(mlp_net, query_for):
    """Acceptance rises with layer width -- the reason a tamper belongs in the
    widest layer that can still move the output."""
    architecture = mlp_net.architecture
    honest = mlp_net.eval_trace(query_for(mlp_net))
    probabilities = []
    for layer_index in (1, 2, 3):
        tampered = mlp_net.forward_from(
            honest.tampered(layer_index, 0, float(honest[layer_index][0]) + 6.0),
            layer_index,
        )
        probabilities.append(
            (architecture[layer_index].n_neurons, acceptance_probability(mlp_net, tampered))
        )
    by_width = sorted(probabilities)
    assert [p for _, p in by_width] == sorted(p for _, p in by_width)
