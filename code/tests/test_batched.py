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
    BatchedLayerProof,
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


def _solve_mod(matrix, rhs, prime):
    """One solution of ``matrix @ u == rhs`` over F_p (free variables set to 0)."""
    a = np.mod(np.array(matrix, dtype=object), prime)
    b = np.mod(np.array(rhs, dtype=object), prime)
    rows, cols = a.shape
    pivots, r = [], 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if a[i, c] % prime), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        b[[r, piv]] = b[[piv, r]]
        inv = pow(int(a[r, c]), prime - 2, prime)
        a[r] = np.mod(a[r] * inv, prime)
        b[r] = (int(b[r]) * inv) % prime
        for i in range(rows):
            if i != r and a[i, c] % prime:
                f = int(a[i, c])
                a[i] = np.mod(a[i] - f * a[r], prime)
                b[i] = (int(b[i]) - f * int(b[r])) % prime
        pivots.append(c)
        r += 1
        if r == rows:
            break
    if any(int(b[i]) % prime for i in range(r, rows)):
        return None
    sol = np.zeros(cols, dtype=object)
    for i, c in enumerate(pivots):
        sol[c] = b[i]
    return np.array([int(v) % prime for v in sol], dtype=np.int64)


def test_adaptive_transcript_forgery_is_rejected():
    """The Fiat-Shamir ordering has to put ``u`` *before* the column challenge.

    If the sampled columns were fixed by ``(digest, x, a, y)`` alone, a cheating
    prover could read them off and solve for a ``u`` that matches the true
    ``chi^T A`` on exactly those columns while still hitting the final scalar
    identity -- ``t+1`` linear constraints on ``M+1`` unknowns.  That forgery was
    implemented against an earlier version of this module and it *was accepted*.
    Deriving the columns from ``u`` as well makes the attack circular: committing
    to the forged ``u`` re-randomises the very columns it was solved against.
    """
    from pvi.protocol import batched as B

    rng = np.random.default_rng(11)
    weight, bias, activations_in, scale_bits = _layer(rng)
    honest = fixed_point_layer(weight, bias, activations_in, scale_bits=scale_bits)
    z = honest.pre_activations()
    live = np.flatnonzero(z > 0)
    outputs = np.maximum(np.array(z, dtype=np.int64), 0)
    outputs[live[:2]] = 0  # zero-hiding tamper
    layer = fixed_point_layer(
        weight, bias, activations_in, activations_out=outputs, scale_bits=scale_bits
    )
    commitment = _commit(layer)
    n_queries = 16

    # The cheating prover forges a non-negative sign witness so the free range
    # check passes, then solves for a u that satisfies every algebraic check.
    zero_idx = np.flatnonzero(_as_signed(layer.outputs) == 0).astype(np.int64)
    true_y = B._as_field(-z[zero_idx], DEFAULT_PRIME)
    forged_y = np.where(_as_signed(true_y) < 0, 1, true_y).astype(np.int64)

    s, t, alpha, beta = B._derive_folding(
        commitment.digest, layer.inputs, layer.outputs, zero_idx, forged_y,
        layer.width, DEFAULT_PRIME,
    )
    chi = B._challenge_vector(
        layer.outputs, s, t, alpha, beta, zero_idx, DEFAULT_PRIME
    )
    quadratic = int(np.mod(np.sum(np.mod(
        np.mod(s * layer.outputs, DEFAULT_PRIME) * layer.outputs, DEFAULT_PRIME)), DEFAULT_PRIME))
    witness = int(np.mod(np.sum(np.mod(t * forged_y, DEFAULT_PRIME)), DEFAULT_PRIME))
    target = int(np.mod(alpha * quadratic - beta * witness, DEFAULT_PRIME))

    # Its best guess at the columns: those implied by the honest u.
    u_true = commitment.combination(chi)
    columns = B._derive_columns(
        commitment.digest, layer.inputs, layer.outputs, zero_idx, forged_y,
        u_true, commitment.n_columns, n_queries, DEFAULT_PRIME,
    )
    encoded = np.array([
        int(np.mod(np.sum(np.mod(chi * commitment._encoded[:, int(c)], DEFAULT_PRIME)),
                   DEFAULT_PRIME))
        for c in columns
    ])
    rows = [commitment._vandermonde[int(c), :] for c in columns]
    rows.append(B._as_field(layer.inputs, DEFAULT_PRIME))
    forged_u = _solve_mod(
        np.array(rows, dtype=object),
        np.array(list(encoded) + [target], dtype=object),
        DEFAULT_PRIME,
    )
    assert forged_u is not None, "expected the underdetermined system to be solvable"
    assert not np.array_equal(forged_u, u_true), "forgery must differ from the honest opening"

    forged_proof = BatchedLayerProof(
        combination=forged_u,
        zero_indices=zero_idx,
        sign_witness=forged_y,
        column_openings=commitment.open_columns(columns),
    )
    accepted = verify_layer(
        forged_proof, commitment.digest, commitment.params, commitment,
        layer.inputs, layer.outputs, prime=layer.prime, n_queries=n_queries,
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


# --------------------------------------------------------------------------- #
# Identity (logit) layers and whole-network chaining
# --------------------------------------------------------------------------- #
#
# Checking one layer in isolation proves nothing about the network, and the
# ReLU constraint system rejects an honest output layer outright (most logits
# are negative).  Both were real gaps found in review.


def _small_net(rng, widths=(20, 16, 12), n_classes=5):
    """A tiny dense net as (list of (W, b, activation))."""
    layers, fan_in = [], widths[0]
    for width in widths[1:]:
        layers.append((rng.normal(0, 0.25, size=(width, fan_in)),
                       rng.normal(0, 0.05, size=width), "relu"))
        fan_in = width
    layers.append((rng.normal(0, 0.25, size=(n_classes, fan_in)),
                   rng.normal(0, 0.05, size=n_classes), "identity"))
    return layers


def _run_chain(layers, query, scale_bits=8, tampered=None, n_queries=12):
    """Execute, prove and verify a whole chained network, verifier-side inputs."""
    from pvi.protocol.batched import (
        _augment, _layer_context, _transcript_bytes, rescale,
    )
    tampered = tampered or {}
    prime = DEFAULT_PRIME
    views, x = [], np.rint(np.asarray(query, float) * (1 << scale_bits)).astype(np.int64)
    for index, (w, b, act) in enumerate(layers):
        view = fixed_point_layer(w, b, x, activations_out=tampered.get(index),
                                 scale_bits=scale_bits, prime=prime,
                                 activation=act, inputs_are_integers=True)
        views.append(view)
        x = rescale(view.outputs, scale_bits, prime)
    commitments = [BatchedWeightCommitment(v.matrix, prime=prime) for v in views]

    proofs, running = [], b""
    for index, (view, cm) in enumerate(zip(views, commitments)):
        proof = prove_layer(view, cm, n_queries=n_queries,
                            context=_layer_context(index, running))
        proofs.append(proof)
        running = _transcript_bytes(view.outputs, proof.combination)

    # verifier: recompute every layer's input, never take it from the prover
    x = np.rint(np.asarray(query, float) * (1 << scale_bits)).astype(np.int64)
    running = b""
    for index, ((w, b, act), view, proof, cm) in enumerate(
        zip(layers, views, proofs, commitments)
    ):
        if not verify_layer(proof, cm.digest, cm.params, cm, _augment(x, prime),
                            view.outputs, prime=prime, n_queries=n_queries,
                            activation=act, context=_layer_context(index, running)):
            return False
        running = _transcript_bytes(view.outputs, proof.combination)
        x = rescale(view.outputs, scale_bits, prime)
    return True


def test_identity_layer_accepts_honest_negative_logits():
    """The ReLU constraint system rejects an honest logit layer 0/N; identity must not."""
    rng = np.random.default_rng(20)
    w = rng.normal(0, 0.3, size=(10, 32))
    b = rng.normal(0, 0.05, size=10)
    a_in = np.abs(rng.normal(0, 0.5, size=32))
    view = fixed_point_layer(w, b, a_in, scale_bits=8, activation="identity")
    assert (view.pre_activations() < 0).any(), "expected some negative logits"
    cm = _commit(view)
    proof = prove_layer(view, cm, n_queries=12)
    assert verify_layer(proof, cm.digest, cm.params, cm, view.inputs, view.outputs,
                        prime=view.prime, n_queries=12, activation="identity")


def test_identity_layer_rejects_a_tampered_logit():
    rng = np.random.default_rng(21)
    w = rng.normal(0, 0.3, size=(10, 32))
    b = rng.normal(0, 0.05, size=10)
    a_in = np.abs(rng.normal(0, 0.5, size=32))
    honest = fixed_point_layer(w, b, a_in, scale_bits=8, activation="identity")
    z = np.array(honest.pre_activations(), dtype=np.int64)
    z[int(z.argmax())] = int(z.min()) - 1000          # flip the argmax
    view = fixed_point_layer(w, b, a_in, activations_out=z, scale_bits=8,
                             activation="identity")
    cm = _commit(view)
    proof = prove_layer(view, cm, n_queries=12)
    assert not verify_layer(proof, cm.digest, cm.params, cm, view.inputs, view.outputs,
                            prime=view.prime, n_queries=12, activation="identity")


def test_whole_network_honest_is_accepted():
    rng = np.random.default_rng(22)
    layers = _small_net(rng)
    for seed in range(3):
        q = np.abs(np.random.default_rng(100 + seed).normal(0, 0.5, size=20))
        assert _run_chain(layers, q)


@pytest.mark.parametrize("target_layer", [0, 1, 2])
def test_whole_network_rejects_a_tamper_at_any_layer(target_layer):
    """Including the last (identity) layer -- previously unsupported entirely."""
    from pvi.protocol.batched import rescale
    rng = np.random.default_rng(23)
    layers = _small_net(rng)
    q = np.abs(np.random.default_rng(7).normal(0, 0.5, size=20))

    # honest outputs for the targeted layer, then tamper one coordinate
    x = np.rint(q * (1 << 8)).astype(np.int64)
    for index, (w, b, act) in enumerate(layers):
        view = fixed_point_layer(w, b, x, scale_bits=8, activation=act,
                                 inputs_are_integers=True)
        if index == target_layer:
            bad = np.array(_as_signed(view.outputs), dtype=np.int64)
            if act == "relu":
                live = np.flatnonzero(bad > 0)
                assert len(live), "need a live neuron to zero out"
                bad[live[0]] = 0                       # the zero-hiding tamper
            else:
                bad[0] = int(bad.min()) - 1000         # logits may all be negative
            assert not _run_chain(layers, q, tampered={index: bad})
            return
        x = rescale(view.outputs, 8, DEFAULT_PRIME)


def test_whole_network_rejects_an_inconsistent_interlayer_value():
    """The verifier recomputes each layer's input, so a broken chain is caught."""
    from pvi.protocol.batched import (
        _augment, _layer_context, _transcript_bytes, rescale,
    )
    rng = np.random.default_rng(24)
    layers = _small_net(rng)
    q = np.abs(np.random.default_rng(8).normal(0, 0.5, size=20))
    prime = DEFAULT_PRIME

    views, x = [], np.rint(q * (1 << 8)).astype(np.int64)
    for w, b, act in layers:
        v = fixed_point_layer(w, b, x, scale_bits=8, prime=prime, activation=act,
                              inputs_are_integers=True)
        views.append(v)
        x = rescale(v.outputs, 8, prime)
    cms = [BatchedWeightCommitment(v.matrix, prime=prime) for v in views]

    proofs, running = [], b""
    for i, (v, cm) in enumerate(zip(views, cms)):
        p = prove_layer(v, cm, n_queries=12, context=_layer_context(i, running))
        proofs.append(p)
        running = _transcript_bytes(v.outputs, p.combination)

    # the prover reports a layer-0 output that isn't what it proved
    claimed = [np.array(v.outputs, dtype=np.int64).copy() for v in views]
    claimed[0][0] = (int(claimed[0][0]) + 1000) % prime

    x = np.rint(q * (1 << 8)).astype(np.int64)
    running, accepted_all = b"", True
    for i, ((w, b, act), out, p, cm) in enumerate(zip(layers, claimed, proofs, cms)):
        if not verify_layer(p, cm.digest, cm.params, cm, _augment(x, prime), out,
                            prime=prime, n_queries=12, activation=act,
                            context=_layer_context(i, running)):
            accepted_all = False
            break
        running = _transcript_bytes(out, p.combination)
        x = rescale(out, 8, prime)
    assert not accepted_all
