"""Tests for the layered-DAG model.

The single most important invariant in the whole codebase is checked here: for
every neuron of every layer, the *vectorised* forward pass and the *per-neuron*
local relation must agree.  The prover uses the former to build the trace and the
verifier uses the latter to check it, so any disagreement would show up as a
completeness failure -- or, worse, as a soundness gap that lets a tamper hide in
the discrepancy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.nn.architecture import (
    Architecture,
    Conv2dLayer,
    DenseLayer,
    InputLayer,
    MaxPool2dLayer,
)
from pvi.nn.models import mlp_architecture, small_cnn_architecture
from pvi.nn.network import Trace, TraceLayout, TracedNetwork

from conftest import build_network


# --------------------------------------------------------------------------- #
# The forward / local-relation agreement invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "architecture",
    [
        mlp_architecture(24, [32, 16], 5),
        small_cnn_architecture((1, 12, 12), 4, channels=(4, 6), hidden_width=16),
        small_cnn_architecture((3, 8, 8), 3, channels=(5,), hidden_width=8),
    ],
    ids=["mlp", "cnn-1ch", "cnn-3ch"],
)
def test_local_relation_matches_forward_pass_for_every_neuron(architecture):
    network = build_network(architecture, seed=3)
    rng = np.random.default_rng(11)
    query = rng.standard_normal(architecture.input_layer.n_neurons).astype(np.float32)
    trace = network.eval_trace(query)

    worst = 0.0
    for layer_index in range(1, len(architecture)):
        for neuron in range(architecture[layer_index].n_neurons):
            recomputed = network.recompute(trace, layer_index, neuron)
            worst = max(worst, abs(float(trace[layer_index][neuron]) - recomputed))
    # Both sides are float32 arithmetic in different summation orders; the gap is
    # rounding noise only.  The protocol's default tolerance is 1e-4.
    assert worst < 1e-5, f"largest local residual {worst:.3e}"


def test_convolution_padding_shrinks_border_parent_sets():
    layer = Conv2dLayer(
        name="conv",
        in_shape=(2, 6, 6),
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    corner_parents, corner_weights = layer.parents(0)  # channel 0, position (0, 0)
    centre = 3 * 6 + 3
    centre_parents, centre_weights = layer.parents(centre)

    assert len(corner_parents) == len(corner_weights)
    assert len(centre_parents) == len(centre_weights)
    # A 3x3 kernel over 2 input channels sees 18 inputs in the interior and only
    # the 2x2 in-bounds sub-window (times 2 channels) at a corner.
    assert len(centre_parents) == 18
    assert len(corner_parents) == 8


def test_parent_indices_are_within_the_previous_layer():
    architecture = small_cnn_architecture((1, 10, 10), 3, channels=(4, 4), hidden_width=8)
    for layer_index in range(1, len(architecture)):
        layer = architecture[layer_index]
        below = architecture[layer_index - 1].n_neurons
        for neuron in range(layer.n_neurons):
            parents, weights = layer.parents(neuron)
            assert parents.min() >= 0 and parents.max() < below
            assert len(np.unique(parents)) == len(parents)
            if layer.n_weight_groups:
                assert len(weights) == len(parents)
                assert weights.max() < layer.weight_group_size - 1


def test_max_pool_local_relation():
    layer = MaxPool2dLayer(name="pool", in_shape=(2, 4, 4), kernel_size=2)
    rng = np.random.default_rng(0)
    values = rng.standard_normal(2 * 4 * 4).astype(np.float32)
    out = layer.forward(values, None, None)
    for neuron in range(layer.n_neurons):
        parents, _ = layer.parents(neuron)
        assert out[neuron] == pytest.approx(layer.local_value(neuron, values[parents], None, None))


def test_split_row_keeps_the_bias():
    layer = DenseLayer(name="fc", in_features=4, out_features=2)
    weight = np.arange(8, dtype=np.float32).reshape(2, 4)
    bias = np.array([10.0, 20.0], dtype=np.float32)
    rows = layer.weight_rows(weight, bias)
    parents, weight_idx = layer.parents(1)
    weights, row_bias = layer.split_row(rows[1], weight_idx)
    assert np.array_equal(weights, weight[1])
    assert row_bias == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Architecture validation
# --------------------------------------------------------------------------- #


def test_architecture_requires_an_input_layer_first():
    with pytest.raises(TypeError):
        Architecture((DenseLayer(name="fc", in_features=2, out_features=2),))


def test_architecture_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        Architecture(
            (
                InputLayer(name="input", shape=(10,)),
                DenseLayer(name="fc", in_features=9, out_features=4),
            )
        )


def test_architecture_rejects_a_second_input_layer():
    with pytest.raises(TypeError):
        Architecture(
            (
                InputLayer(name="input", shape=(4,)),
                InputLayer(name="input2", shape=(4,)),
            )
        )


def test_layer_widths_and_trace_size_agree():
    architecture = mlp_architecture(8, [6, 4], 3)
    assert architecture.layer_widths == (8, 6, 4, 3)
    assert architecture.n_trace_entries == 21
    assert architecture.depth == 3


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #


def test_trace_layout_round_trip():
    layout = TraceLayout((5, 3, 2))
    seen = set()
    for layer in range(3):
        for neuron in range(layout.widths[layer]):
            index = layout.to_global(layer, neuron)
            assert layout.to_local(index) == (layer, neuron)
            seen.add(index)
    assert seen == set(range(layout.total))
    assert list(layout.output_indices) == [8, 9]


def test_trace_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        Trace((np.zeros(3, dtype=np.float64),))


def test_tampering_a_trace_leaves_the_original_untouched(mlp_net, query_for):
    trace = mlp_net.eval_trace(query_for(mlp_net))
    original = trace[1].copy()
    modified = trace.tampered(1, 0, 99.0)
    assert np.array_equal(trace[1], original)
    assert modified[1][0] == pytest.approx(99.0)


def test_forward_from_reproduces_the_honest_trace(mlp_net, query_for):
    """Re-propagating from an untouched layer must be a no-op."""
    trace = mlp_net.eval_trace(query_for(mlp_net))
    for layer_index in range(len(mlp_net.architecture)):
        rebuilt = mlp_net.forward_from(trace, layer_index)
        for i in range(len(trace)):
            assert np.allclose(rebuilt[i], trace[i], atol=1e-6)


def test_single_tamper_plus_propagation_leaves_exactly_one_inconsistency(
    mlp_net, query_for
):
    """The structural fact the attack rests on.

    Perturb one activation, then let the honest model propagate that perturbation
    forward.  Every local relation above the tampered node is satisfied by
    construction, and every one below is untouched -- so the trace is locally
    inconsistent at exactly one node.
    """
    architecture = mlp_net.architecture
    trace = mlp_net.eval_trace(query_for(mlp_net))
    layer_index, neuron = 2, 3

    tampered = mlp_net.forward_from(
        trace.tampered(layer_index, neuron, float(trace[layer_index][neuron]) + 5.0),
        layer_index,
    )

    inconsistent = [
        (l, j)
        for l in range(1, len(architecture))
        for j in range(architecture[l].n_neurons)
        if abs(float(tampered[l][j]) - mlp_net.recompute(tampered, l, j)) > 1e-4
    ]
    assert inconsistent == [(layer_index, neuron)]


# --------------------------------------------------------------------------- #
# Parameter handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
def test_chunked_evaluation_agrees_with_a_single_batch(cnn_net, chunk_size):
    """Chunking exists to bound memory and must not change the result.

    Agreement is numerical, not bit-exact: BLAS is free to block a matrix product
    differently for different batch sizes, so the summation order -- and hence the
    last bit or two -- can differ.  The gap is ~1e-7, five orders of magnitude
    below the protocol's tolerance, and it never touches the protocol anyway, which
    evaluates one query at a time (see the determinism test below).
    """
    rng = np.random.default_rng(0)
    batch = rng.standard_normal(
        (20, cnn_net.architecture.input_layer.n_neurons)
    ).astype(np.float32)
    chunked = cnn_net.outputs(batch, chunk_size=chunk_size)
    single = cnn_net.forward_batch(batch)[-1]
    assert np.abs(chunked - single).max() < 1e-5
    assert np.array_equal(chunked.argmax(axis=1), single.argmax(axis=1))


@pytest.mark.parametrize("network_name", ["mlp_net", "cnn_net"])
def test_eval_trace_is_bit_exactly_reproducible(request, network_name, query_for):
    """The property the commitment actually needs.

    A prover commits to ``EvalTrace(M, qry)``.  If re-evaluating the same query
    produced even a one-bit-different trace, the commitment would not reproduce and
    the protocol would be unusable.  ``eval_trace`` always runs a single query, so
    the shapes -- and therefore the summation order -- are fixed.
    """
    network = request.getfixturevalue(network_name)
    query = query_for(network)
    first = network.eval_trace(query)
    for _ in range(5):
        again = network.eval_trace(query)
        for layer_a, layer_b in zip(first, again):
            assert np.array_equal(layer_a, layer_b)


def test_chunk_size_must_be_positive(mlp_net):
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((4, mlp_net.architecture.input_layer.n_neurons)).astype(
        np.float32
    )
    with pytest.raises(ValueError):
        mlp_net.outputs(batch, chunk_size=0)


def test_missing_parameters_are_refused():
    architecture = mlp_architecture(4, [3], 2)
    with pytest.raises(ValueError):
        TracedNetwork(architecture, {})


def test_wrong_shaped_parameters_are_refused():
    architecture = mlp_architecture(4, [3], 2)
    bad = {
        "fc1": (np.zeros((3, 4), dtype=np.float32), np.zeros(3, dtype=np.float32)),
        "logits": (np.zeros((2, 99), dtype=np.float32), np.zeros(2, dtype=np.float32)),
    }
    with pytest.raises(ValueError):
        TracedNetwork(architecture, bad)
