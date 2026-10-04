"""End-to-end tests of the three-round protocol (Figure 4).

Covers the two properties Definition 4 demands -- correctness and soundness --
plus the forgery attempts a real adversary would try against the commitment
layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.commitments.merkle import MerkleOpening
from pvi.protocol import (
    ModelCommitment,
    ProtocolParams,
    Round2,
    UniformPathSampler,
    challenge_rng,
    run_protocol,
    sample_challenge,
)

from conftest import build_network


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_paths", [1, 3, 8])
def test_honest_prover_is_always_accepted_mlp(mlp_net, query_for, protocol_pair, n_paths):
    prover, verifier = protocol_pair(mlp_net, ProtocolParams(n_paths=n_paths))
    query = query_for(mlp_net)
    for _ in range(200):
        assert run_protocol(prover, verifier, query).accepted


def test_honest_prover_is_always_accepted_cnn(cnn_net, query_for, protocol_pair):
    prover, verifier = protocol_pair(cnn_net, ProtocolParams(n_paths=2))
    for seed in range(25):
        query = query_for(cnn_net, seed=seed)
        for _ in range(8):
            assert run_protocol(prover, verifier, query).accepted


def test_verifier_arithmetic_stays_in_the_provers_precision(mlp_net, query_for):
    """Regression test for a precision bug that cost us a wrong result.

    The prover evaluates layers in float32.  If the verifier adds a float64 bias to
    a float32 dot product, the sum is promoted to float64 -- arithmetic the prover
    never performed -- and the two disagree far more often than rounding requires.
    Every recomputed value must therefore be exactly representable in float32.
    """
    trace = mlp_net.eval_trace(query_for(mlp_net))
    architecture = mlp_net.architecture
    for layer_index in range(1, len(architecture)):
        for neuron in range(min(architecture[layer_index].n_neurons, 40)):
            recomputed = mlp_net.recompute(trace, layer_index, neuron)
            assert np.float32(recomputed) == recomputed, (
                f"layer {layer_index}, neuron {neuron}: verifier produced a value "
                f"that is not a float32, so it widened somewhere"
            )


def test_zero_tolerance_completeness_is_not_degenerate(mlp_net, query_for, protocol_pair):
    """The exact-equality test of Figure 3 should fail *sometimes*, not always.

    An honest prover and a careful verifier agree bit-for-bit on most nodes; they
    disagree only where BLAS blocked the matrix product differently.  If this rate
    collapses to zero, the verifier has stopped matching the prover's arithmetic --
    which is a bug in us, not a finding about the protocol.
    """
    params = ProtocolParams(n_paths=1, abs_tolerance=0.0)
    prover, verifier = protocol_pair(mlp_net, params)
    accepted = sum(
        run_protocol(prover, verifier, query_for(mlp_net, seed=seed)).accepted
        for seed in range(200)
    )
    assert accepted > 0, "no honest run passed exact equality: verifier arithmetic drifted"


# --------------------------------------------------------------------------- #
# Challenge handling
# --------------------------------------------------------------------------- #


def test_challenge_derivation_is_deterministic():
    challenge = sample_challenge()
    a = challenge_rng(challenge).integers(0, 1 << 30, size=16)
    b = challenge_rng(challenge).integers(0, 1 << 30, size=16)
    assert np.array_equal(a, b)


def test_distinct_challenges_give_distinct_paths(mlp_net):
    sampler = UniformPathSampler()
    architecture = mlp_net.architecture
    paths = {
        sampler.sample(architecture, challenge_rng(bytes([i]))).nodes for i in range(64)
    }
    assert len(paths) > 1


def test_prover_and_verifier_derive_the_same_paths(mlp_net, protocol_pair, query_for):
    prover, verifier = protocol_pair(mlp_net, ProtocolParams(n_paths=3))
    query = query_for(mlp_net)
    challenge = sample_challenge()
    _, state = prover.prove1(query)
    prover_paths = prover.paths_for(state, challenge)

    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)
    result = verifier.verify(query, round1, round2, challenge)
    assert result.paths == prover_paths


def test_sampled_paths_follow_real_parent_edges(cnn_net):
    """Each step must land inside the previous node's parent set."""
    architecture = cnn_net.architecture
    sampler = UniformPathSampler()
    rng = np.random.default_rng(0)
    for _ in range(200):
        path = sampler.sample(architecture, rng)
        assert len(path) == len(architecture)
        for layer_index in range(len(architecture) - 1, 0, -1):
            parents, _ = architecture[layer_index].parents(path[layer_index])
            assert path[layer_index - 1] in set(parents.tolist())


# --------------------------------------------------------------------------- #
# Other-model soundness
# --------------------------------------------------------------------------- #


def test_substitute_model_trace_is_rejected(mlp_arch, mlp_net, query_for, protocol_pair):
    """Definition 1's other-model soundness: an honest trace of a different model."""
    prover, verifier = protocol_pair(mlp_net)
    substitute = build_network(mlp_arch, seed=99)
    query = query_for(mlp_net)
    forged = substitute.eval_trace(query)

    accepted = sum(
        run_protocol(prover, verifier, query, trace=forged).accepted for _ in range(500)
    )
    assert accepted == 0


def test_substitute_differing_in_one_layer_is_still_rejected(
    mlp_arch, mlp_net, query_for, protocol_pair
):
    """Even a substitute that differs only in the last layer is caught, because
    every path passes through the output layer."""
    prover, verifier = protocol_pair(mlp_net)
    parameters = dict(mlp_net.parameters)
    weight, bias = parameters["logits"]
    parameters["logits"] = (weight * 1.5, bias)

    from pvi.nn.network import TracedNetwork

    substitute = TracedNetwork(mlp_arch, parameters)
    query = query_for(mlp_net)
    forged = substitute.eval_trace(query)
    accepted = sum(
        run_protocol(prover, verifier, query, trace=forged).accepted for _ in range(300)
    )
    assert accepted == 0


# --------------------------------------------------------------------------- #
# Single-node tampering: the 1/N bound of Section 5.2
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer_index", [1, 2])
def test_single_node_tamper_is_detected_with_probability_one_over_width(
    mlp_net, query_for, protocol_pair, layer_index
):
    prover, verifier = protocol_pair(mlp_net)
    architecture = mlp_net.architecture
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)

    neuron = 4
    tampered = mlp_net.forward_from(
        honest.tampered(layer_index, neuron, float(honest[layer_index][neuron]) + 6.0),
        layer_index,
    )

    trials = 4000
    accepted = sum(
        run_protocol(prover, verifier, query, trace=tampered).accepted
        for _ in range(trials)
    )
    width = architecture[layer_index].n_neurons
    observed = accepted / trials
    predicted = 1.0 - 1.0 / width
    tolerance = 4.0 * np.sqrt(predicted * (1 - predicted) / trials)
    assert abs(observed - predicted) < tolerance + 0.005, (
        f"layer {layer_index}: observed {observed:.4f}, predicted {predicted:.4f}"
    )


def test_more_paths_detect_a_single_node_tamper_more_often(
    mlp_net, query_for, protocol_pair
):
    architecture = mlp_net.architecture
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)
    layer_index, neuron = 2, 1
    tampered = mlp_net.forward_from(
        honest.tampered(layer_index, neuron, float(honest[layer_index][neuron]) + 6.0),
        layer_index,
    )
    width = architecture[layer_index].n_neurons

    for n_paths in (1, 4, 16):
        prover, verifier = protocol_pair(mlp_net, ProtocolParams(n_paths=n_paths))
        trials = 1500
        detected = sum(
            not run_protocol(prover, verifier, query, trace=tampered).accepted
            for _ in range(trials)
        )
        predicted = 1.0 - (1.0 - 1.0 / width) ** n_paths
        assert abs(detected / trials - predicted) < 0.05


# --------------------------------------------------------------------------- #
# Commitment-layer forgeries
# --------------------------------------------------------------------------- #


def test_claimed_output_must_match_the_committed_trace(mlp_net, query_for, protocol_pair):
    prover, verifier = protocol_pair(mlp_net)
    query = query_for(mlp_net)
    challenge = sample_challenge()
    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)

    from dataclasses import replace

    lying = replace(round1, claimed_output=round1.claimed_output + 1.0)
    result = verifier.verify(query, lying, round2, challenge)
    assert not result.accepted
    assert result.failures[0].kind == "output"


def test_missing_weight_opening_is_rejected(mlp_net, query_for, protocol_pair):
    prover, verifier = protocol_pair(mlp_net)
    query = query_for(mlp_net)
    challenge = sample_challenge()
    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)

    stripped = Round2(layer_openings=round2.layer_openings, weight_openings={})
    result = verifier.verify(query, round1, stripped, challenge)
    assert not result.accepted
    assert any(f.kind == "opening" for f in result.failures)


def test_weight_opening_from_a_different_model_is_rejected(
    mlp_arch, mlp_net, query_for, protocol_pair
):
    """Position binding: the prover cannot substitute another model's weights."""
    prover, verifier = protocol_pair(mlp_net)
    other = build_network(mlp_arch, seed=1234)
    other_commitment = ModelCommitment(other)

    query = query_for(mlp_net)
    challenge = sample_challenge()
    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)

    swapped = dict(round2.weight_openings)
    key = next(iter(swapped))
    swapped[key] = other_commitment.open_row(*key)
    result = verifier.verify(
        query, round1, Round2(round2.layer_openings, swapped), challenge
    )
    assert not result.accepted
    assert any(f.kind == "opening" for f in result.failures)


def test_weight_opening_for_the_wrong_row_is_rejected(mlp_net, query_for, protocol_pair):
    prover, verifier = protocol_pair(mlp_net)
    query = query_for(mlp_net)
    challenge = sample_challenge()
    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)

    swapped = dict(round2.weight_openings)
    layer_index, group = next(iter(swapped))
    other_group = (group + 1) % mlp_net.architecture[layer_index].n_weight_groups
    swapped[(layer_index, group)] = prover.model_commitment.open_row(
        layer_index, other_group
    )
    result = verifier.verify(
        query, round1, Round2(round2.layer_openings, swapped), challenge
    )
    assert not result.accepted


def test_tampered_trace_opening_is_rejected(mlp_net, query_for, protocol_pair):
    prover, verifier = protocol_pair(mlp_net)
    query = query_for(mlp_net)
    challenge = sample_challenge()
    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)

    from pvi.commitments.merkle import encode_f32_vector

    genuine = round2.layer_openings[1]
    forged_value = encode_f32_vector(
        np.zeros(mlp_net.architecture[1].n_neurons, dtype=np.float32)
    )
    openings = list(round2.layer_openings)
    openings[1] = MerkleOpening(
        index=genuine.index, value=forged_value, siblings=genuine.siblings
    )
    result = verifier.verify(
        query, round1, Round2(tuple(openings), round2.weight_openings), challenge
    )
    assert not result.accepted
    assert result.failures[0].kind == "opening"


def test_wrong_number_of_layer_openings_is_rejected(mlp_net, query_for, protocol_pair):
    prover, verifier = protocol_pair(mlp_net)
    query = query_for(mlp_net)
    challenge = sample_challenge()
    round1, state = prover.prove1(query)
    round2 = prover.prove2(state, challenge)
    result = verifier.verify(
        query, round1, Round2(round2.layer_openings[:-1], round2.weight_openings), challenge
    )
    assert not result.accepted


def test_input_layer_tampering_is_caught_by_the_full_input_check(
    mlp_net, query_for, protocol_pair
):
    prover, verifier = protocol_pair(mlp_net)
    query = query_for(mlp_net)
    honest = mlp_net.eval_trace(query)
    forged = mlp_net.forward_from(honest.tampered(0, 0, 5.0), 0)

    result = run_protocol(prover, verifier, query, trace=forged)
    assert not result.accepted
    assert result.failures[0].kind == "input_anchor"


def test_verifier_never_receives_the_model(mlp_net, protocol_pair):
    """The verifier holds a digest, not weights -- the property that makes its
    cost sublinear in the model and the reason it cannot simply recompute."""
    _, verifier = protocol_pair(mlp_net)
    from pvi.nn.network import TracedNetwork

    assert not any(
        isinstance(value, TracedNetwork) for value in vars(verifier).values()
    )
