"""The batched per-layer check must accept honest traces and reject every tamper.

This is the Revision-1 candidate defence (``protocol/batched.py``).  The
properties that matter, and that Sections 3-8 of ``DEFENCE_NOTES.md`` showed no
sampling rule can have:

* completeness is *exact* -- no tolerance, because the arithmetic is exact;
* detection does not depend on *where* in the layer the tamper sits;
* detection does not depend on the tampered value, in particular it does not
  collapse when the adversary claims zero (Theorem 3's exploit).
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.protocol.batched import (
    DEFAULT_PRIME,
    BatchedWeightCommitment,
    fixed_point_layer,
    prove_layer,
    verify_layer,
)


def _layer(rng, width=48, fan_in=64, scale_bits=8):
    weight = rng.normal(0.0, 0.2, size=(width, fan_in))
    bias = rng.normal(0.0, 0.05, size=width)
    activations_in = np.abs(rng.normal(0.0, 0.5, size=fan_in))
    return weight, bias, activations_in, scale_bits


def _commit(layer):
    return BatchedWeightCommitment(layer.matrix, prime=layer.prime)


def _run(layer, commitment, n_queries=16):
    proof = prove_layer(layer, commitment, n_queries=n_queries)
    accepted = verify_layer(
        proof, commitment.digest, commitment.params, commitment,
        layer.inputs, layer.outputs, prime=layer.prime, n_queries=n_queries,
    )
    return proof, accepted


def test_honest_layer_is_accepted_exactly():
    rng = np.random.default_rng(0)
    weight, bias, activations_in, scale_bits = _layer(rng)
    layer = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    commitment = _commit(layer)
    _, accepted = _run(layer, commitment)
    assert accepted


@pytest.mark.parametrize("seed", range(8))
def test_honest_completeness_has_no_tolerance(seed):
    """Exactly 1, not 1-up-to-tau: the whole point of exact field arithmetic."""
    rng = np.random.default_rng(seed)
    weight, bias, activations_in, scale_bits = _layer(rng)
    layer = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    commitment = _commit(layer)
    _, accepted = _run(layer, commitment)
    assert accepted


def _tampered(weight, bias, activations_in, scale_bits, mutate):
    honest = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    outputs = np.array(honest.pre_activations(), dtype=np.int64)
    outputs = np.maximum(outputs, 0)
    mutate(outputs)
    return fixed_point_layer(
        weight, bias, activations_in, activations_out=outputs, scale_bits=scale_bits
    )


@pytest.mark.parametrize("neuron", [0, 1, 7, 23, 47])
def test_single_neuron_tamper_is_rejected_wherever_it_sits(neuron):
    """Detection must not depend on *where* the tamper is -- no 1/N ceiling."""
    rng = np.random.default_rng(1)
    weight, bias, activations_in, scale_bits = _layer(rng)

    def mutate(outputs):
        outputs[neuron] = outputs[neuron] + 5000

    layer = _tampered(weight, bias, activations_in, scale_bits, mutate)
    commitment = _commit(layer)
    _, accepted = _run(layer, commitment)
    assert not accepted


def test_zeroing_tamper_is_rejected():
    """Theorem 3's exploit: claim zero where the truth is positive.

    Contribution weighting gives this detection exactly 0.  Here it must be
    caught -- this is the case the naive 'batch the layer' check misses, and the
    reason the sign witness exists.
    """
    rng = np.random.default_rng(2)
    weight, bias, activations_in, scale_bits = _layer(rng)
    honest = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    positive = np.flatnonzero(np.maximum(honest.pre_activations(), 0) > 0)
    assert len(positive) >= 3, "need some live neurons to zero out"

    def mutate(outputs):
        outputs[positive[:3]] = 0

    layer = _tampered(weight, bias, activations_in, scale_bits, mutate)
    commitment = _commit(layer)
    _, accepted = _run(layer, commitment)
    assert not accepted


def test_many_zeroed_neurons_are_rejected():
    rng = np.random.default_rng(3)
    weight, bias, activations_in, scale_bits = _layer(rng)
    honest = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    positive = np.flatnonzero(np.maximum(honest.pre_activations(), 0) > 0)

    def mutate(outputs):
        outputs[positive] = 0

    layer = _tampered(weight, bias, activations_in, scale_bits, mutate)
    commitment = _commit(layer)
    _, accepted = _run(layer, commitment)
    assert not accepted


def test_negative_activation_is_rejected_for_free():
    rng = np.random.default_rng(4)
    weight, bias, activations_in, scale_bits = _layer(rng)

    def mutate(outputs):
        outputs[5] = -17

    layer = _tampered(weight, bias, activations_in, scale_bits, mutate)
    commitment = _commit(layer)
    _, accepted = _run(layer, commitment)
    assert not accepted


def test_lying_about_the_sign_witness_is_rejected():
    """The adversary's best move against the zero-set check: forge ``y``."""
    rng = np.random.default_rng(5)
    weight, bias, activations_in, scale_bits = _layer(rng)
    honest = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    positive = np.flatnonzero(np.maximum(honest.pre_activations(), 0) > 0)

    def mutate(outputs):
        outputs[positive[:2]] = 0

    layer = _tampered(weight, bias, activations_in, scale_bits, mutate)
    commitment = _commit(layer)
    proof = prove_layer(layer, commitment, n_queries=16)

    # Replace the (negative, and therefore rejected) witness with a plausible
    # non-negative one: now the range check passes and only the algebra can catch it.
    forged = np.where(_as_signed(proof.sign_witness) < 0, 1, proof.sign_witness)
    forged_proof = type(proof)(
        combination=proof.combination,
        zero_indices=proof.zero_indices,
        sign_witness=np.asarray(forged, dtype=np.int64),
        column_openings=proof.column_openings,
    )
    accepted = verify_layer(
        forged_proof, commitment.digest, commitment.params, commitment,
        layer.inputs, layer.outputs, prime=layer.prime, n_queries=16,
    )
    assert not accepted


def _as_signed(values):
    values = np.asarray(values, dtype=np.int64)
    return np.where(values > DEFAULT_PRIME // 2, values - DEFAULT_PRIME, values)


def test_forged_combination_is_rejected():
    """The prover must send the true ``chi^T A``; the code check enforces it."""
    rng = np.random.default_rng(6)
    weight, bias, activations_in, scale_bits = _layer(rng)
    layer = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    commitment = _commit(layer)
    proof = prove_layer(layer, commitment, n_queries=16)

    tampered = np.array(proof.combination, dtype=np.int64)
    tampered[0] = (tampered[0] + 1) % DEFAULT_PRIME
    forged = type(proof)(
        combination=tampered,
        zero_indices=proof.zero_indices,
        sign_witness=proof.sign_witness,
        column_openings=proof.column_openings,
    )
    accepted = verify_layer(
        forged, commitment.digest, commitment.params, commitment,
        layer.inputs, layer.outputs, prime=layer.prime, n_queries=16,
    )
    assert not accepted


def test_openings_from_a_different_model_are_rejected():
    rng = np.random.default_rng(7)
    weight, bias, activations_in, scale_bits = _layer(rng)
    layer = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    commitment = _commit(layer)
    proof = prove_layer(layer, commitment, n_queries=16)

    other_weight = rng.normal(0.0, 0.2, size=weight.shape)
    other_layer = fixed_point_layer(other_weight, bias, activations_in, scale_bits=scale_bits)
    other_commitment = _commit(other_layer)

    accepted = verify_layer(
        proof, other_commitment.digest, other_commitment.params, other_commitment,
        layer.inputs, layer.outputs, prime=layer.prime, n_queries=16,
    )
    assert not accepted
