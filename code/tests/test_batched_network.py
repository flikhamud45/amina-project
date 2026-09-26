"""Tests for the public whole-network API.

``tests/test_batched.py`` re-implements the chaining inside its own helper, so
``fixed_point_network`` / ``prove_network`` / ``verify_network`` -- the three
functions a caller would actually use -- were never exercised.  Added after
review.

The broken-chain test here is the one that matters.  An earlier attempt
perturbed a layer's *reported* output, which changes that layer's own
transcript, so it was rejected by that layer's own check and the chaining was
never tested at all.  The real case is subtler: every per-layer proof is
individually valid, but one layer's proof was built on an input that is not
``rescale`` of the previous layer's output.  Only the verifier recomputing the
input catches it, and a positive control proves that is what rejects.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import build_network
from pvi.nn.models import mlp_architecture
from pvi.protocol.batched import (
    DEFAULT_PRIME,
    BatchedWeightCommitment,
    _augment,
    _extend_running,
    _layer_context,
    _signed,
    fixed_point_layer,
    fixed_point_network,
    prove_network,
    rescale,
    verify_layer,
    verify_network,
)

N_QUERIES = 12
SCALE_BITS = 8


def _setup(seed: int = 0):
    network = build_network(mlp_architecture(24, [32, 16], 5), seed=seed)
    query = np.abs(np.random.default_rng(seed + 100).normal(0, 0.5, size=24))
    commitments = [
        BatchedWeightCommitment(view.matrix, prime=DEFAULT_PRIME)
        for view in fixed_point_network(network, query)
    ]
    return network, query, commitments


def _prove_and_verify(network, query, views, commitments):
    proofs = prove_network(views, commitments, n_queries=N_QUERIES)
    accepted = verify_network(
        network.architecture, query, [v.outputs for v in views], proofs,
        commitments, n_queries=N_QUERIES,
    )
    return accepted, proofs


@pytest.mark.parametrize("seed", range(3))
def test_honest_network_is_accepted(seed):
    network, query, commitments = _setup(seed)
    views = fixed_point_network(network, query)
    accepted, _ = _prove_and_verify(network, query, views, commitments)
    assert accepted


@pytest.mark.parametrize("layer_index", [1, 2, 3])
def test_tamper_at_every_layer_is_rejected(layer_index):
    """Including the identity/logit layer, which the ReLU checks cannot handle."""
    network, query, commitments = _setup()
    honest = fixed_point_network(network, query)
    bad = np.array(_signed(honest[layer_index - 1].outputs, DEFAULT_PRIME), dtype=np.int64)
    if network.architecture[layer_index].activation == "relu":
        live = np.flatnonzero(bad > 0)
        assert len(live), "need a live neuron to zero"
        bad[live[0]] = 0                                    # the zero-hiding tamper
    else:
        bad[int(bad.argmax())] = int(bad.min()) - 1000      # flip the winning logit
    views = fixed_point_network(network, query, tampered={layer_index: bad})
    accepted, _ = _prove_and_verify(network, query, views, commitments)
    assert not accepted


def test_broken_chain_is_rejected():
    """Every per-layer proof is valid; layer 2's input is not the rescaled layer-1 output."""
    network, query, commitments = _setup()
    views = fixed_point_network(network, query)

    forged_input = rescale(views[0].outputs, SCALE_BITS, DEFAULT_PRIME).copy()
    forged_input[int(np.argmax(forged_input))] += 50

    weight, bias = network.parameters[network.architecture[2].name]
    views[1] = fixed_point_layer(weight, bias, forged_input, scale_bits=SCALE_BITS,
                                 prime=DEFAULT_PRIME, inputs_are_integers=True)
    weight, bias = network.parameters[network.architecture[3].name]
    views[2] = fixed_point_layer(
        weight, bias, rescale(views[1].outputs, SCALE_BITS, DEFAULT_PRIME),
        scale_bits=SCALE_BITS, prime=DEFAULT_PRIME, activation="identity",
        inputs_are_integers=True,
    )

    accepted, proofs = _prove_and_verify(network, query, views, commitments)
    assert not accepted, "the verifier must recompute layer 2's input and reject"

    # Positive control: layer 2's proof DOES verify against the input it was
    # built on.  Without this the test could be passing for the wrong reason.
    context = _layer_context(
        1, _extend_running(b"", views[0].outputs, proofs[0].combination)
    )
    commitment = commitments[1]
    assert verify_layer(
        proofs[1], commitment.public_params,
        _augment(forged_input, DEFAULT_PRIME), views[1].outputs,
        n_queries=N_QUERIES, context=context,
    ), "layer 2's own proof should be valid for its own input"


def test_running_transcript_is_cumulative():
    """Each layer's challenges must depend on *every* earlier layer, not just the last."""
    first = _extend_running(b"", np.array([1, 2, 3]), np.array([4, 5]))
    second = _extend_running(first, np.array([6]), np.array([7]))
    other_first = _extend_running(b"", np.array([9, 9, 9]), np.array([4, 5]))
    other_second = _extend_running(other_first, np.array([6]), np.array([7]))
    assert second != other_second, "layer 1's messages must still influence layer 2"


def test_verifier_never_reads_a_prover_supplied_input():
    """A prover cannot smuggle in its own layer input: the verifier recomputes it."""
    network, query, commitments = _setup()
    views = fixed_point_network(network, query)
    proofs = prove_network(views, commitments, n_queries=N_QUERIES)
    # Same proofs, but the verifier is given a different query: it must recompute
    # the input layer from that query and reject.
    other = np.abs(np.random.default_rng(999).normal(0, 0.5, size=24))
    assert not verify_network(
        network.architecture, other, [v.outputs for v in views], proofs,
        commitments, n_queries=N_QUERIES,
    )


def test_public_params_do_not_carry_the_model():
    """The verifier's view must not contain the weights, even reachable indirectly.

    Passing the prover's commitment object to the verifier worked -- it only
    read public fields -- but it left a verifier one attribute away from the
    model.  This pins the separation.
    """
    import dataclasses

    from pvi.protocol.batched import BatchedPublicParams

    network, query, commitments = _setup()
    views = fixed_point_network(network, query)
    params = commitments[0].public_params
    assert isinstance(params, BatchedPublicParams)

    secret = np.asarray(views[0].matrix, dtype=np.int64)
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if isinstance(value, np.ndarray) and value.shape == secret.shape:
            assert not np.array_equal(value, secret), (
                f"public params field {field.name!r} exposes the weight matrix"
            )
    # and the encoded matrix must not be reachable either
    assert not hasattr(params, "_encoded")
    assert not hasattr(params, "_matrix")


# --------------------------------------------------------------------------- #
# Verifier-drawn challenges (closing the Fiat-Shamir grinding hole)
# --------------------------------------------------------------------------- #


def test_verifier_randomness_accepts_an_honest_network():
    from pvi.protocol.batched import VerifierRandomness

    network, query, commitments = _setup()
    views = fixed_point_network(network, query)
    randomness = VerifierRandomness(seed=7)
    proofs = prove_network(views, commitments, n_queries=N_QUERIES, randomness=randomness)
    assert verify_network(
        network.architecture, query, [v.outputs for v in views], proofs, commitments,
        n_queries=N_QUERIES, randomness=randomness,
    )


def test_verifier_randomness_rejects_a_tamper():
    from pvi.protocol.batched import VerifierRandomness

    network, query, commitments = _setup()
    honest = fixed_point_network(network, query)
    bad = np.array(_signed(honest[0].outputs, DEFAULT_PRIME), dtype=np.int64)
    bad[int(np.flatnonzero(bad > 0)[0])] = 0
    views = fixed_point_network(network, query, tampered={1: bad})
    randomness = VerifierRandomness(seed=7)
    proofs = prove_network(views, commitments, n_queries=N_QUERIES, randomness=randomness)
    assert not verify_network(
        network.architecture, query, [v.outputs for v in views], proofs, commitments,
        n_queries=N_QUERIES, randomness=randomness,
    )


def test_challenges_are_unpredictable_across_runs():
    """The point of verifier randomness: the prover cannot know them in advance.

    Under Fiat-Shamir the challenges are a deterministic function of the
    prover's own message, which is what makes grinding possible.  Here two
    verifiers challenge the same message differently.
    """
    from pvi.protocol.batched import VerifierRandomness

    context = b"ctx"
    first = VerifierRandomness()
    second = VerifierRandomness()
    a = first.folding(context, 16, 4, DEFAULT_PRIME)
    b = second.folding(context, 16, 4, DEFAULT_PRIME)
    assert not np.array_equal(a.s, b.s), "two verifiers must not issue the same challenge"
    assert not np.array_equal(
        first.columns(context, 128, 8), second.columns(context, 128, 8)
    )


def test_verifier_randomness_is_stable_within_one_interaction():
    """Prover and verifier must see the same issued values for a given layer."""
    from pvi.protocol.batched import VerifierRandomness

    randomness = VerifierRandomness(seed=3)
    first = randomness.folding(b"layer-0", 16, 4, DEFAULT_PRIME)
    again = randomness.folding(b"layer-0", 16, 4, DEFAULT_PRIME)
    assert np.array_equal(first.s, again.s) and first.alpha == again.alpha
    assert np.array_equal(
        randomness.columns(b"layer-0", 128, 8), randomness.columns(b"layer-0", 128, 8)
    )
    # different layers get independent challenges
    other = randomness.folding(b"layer-1", 16, 4, DEFAULT_PRIME)
    assert not np.array_equal(first.s, other.s)


def test_columns_are_drawn_without_replacement():
    """A repeated column adds no soundness, so the t columns must be distinct."""
    from pvi.protocol.batched import VerifierRandomness

    randomness = VerifierRandomness(seed=11)
    columns = randomness.columns(b"c", 64, 24)
    assert len(columns) == 24
    assert len(set(int(c) for c in columns)) == 24


def test_verifier_randomness_removes_the_grinding_advantage():
    """Measure what grinding actually buys, under each challenge source.

    A cheating prover grinds by re-randomising some free part of its message
    (the sign witness is free: any non-negative vector passes the range check)
    and re-hashing until the challenge is favourable.  What matters is how many
    *independent* challenge draws that search yields.

    Under Fiat-Shamir the challenge is a function of the message, so every
    variant is a fresh draw -- 200 variants, 200 shots at a ~2**-24 event, and
    the whole search is offline and invisible.  Under verifier-drawn randomness
    the challenge is fixed before the prover's variants exist, so all 200
    variants face the *same* challenge: one shot, and a failure is a visible
    rejection.
    """
    from pvi.protocol.batched import VerifierRandomness, _derive_columns

    rng = np.random.default_rng(5)
    width, n_zero, n_columns, n_queries = 64, 12, 256, 8
    digest = b"\x00" * 32
    inputs = rng.integers(0, DEFAULT_PRIME, size=width, dtype=np.int64)
    outputs = rng.integers(0, DEFAULT_PRIME, size=width, dtype=np.int64)
    zero_indices = np.arange(n_zero, dtype=np.int64)
    combination = rng.integers(0, DEFAULT_PRIME, size=width, dtype=np.int64)

    fiat_shamir, verifier_drawn = set(), set()
    randomness = VerifierRandomness(seed=1)
    for _ in range(200):
        witness = rng.integers(0, 1000, size=n_zero, dtype=np.int64)   # free choice
        fiat_shamir.add(tuple(int(c) for c in _derive_columns(
            digest, inputs, outputs, zero_indices, witness, combination,
            n_columns, n_queries, DEFAULT_PRIME, b"ctx")))
        verifier_drawn.add(tuple(int(c) for c in randomness.columns(
            b"ctx", n_columns, n_queries)))

    assert len(fiat_shamir) > 100, "grinding should yield many independent draws"
    assert len(verifier_drawn) == 1, (
        "verifier-drawn challenges must not move when the prover re-randomises: "
        f"got {len(verifier_drawn)} distinct challenge sets"
    )
