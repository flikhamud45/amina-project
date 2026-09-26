"""A per-layer batched consistency check: the Revision-1 candidate defence.

Everything in ``DEFENCE_NOTES.md`` Sections 3-8 says the same thing: no rule for
choosing *which* nodes to check escapes the ``k/|U|`` ceiling once the adversary
knows the rule.  The only lever left is to change *what one check verifies*.
This module does that, and unlike Section 9's prototype it is an actual
protocol -- the verifier never holds the weight matrix.

The shape of the argument
-------------------------
Write the checked layer as an augmented matrix ``A = [W | b]`` of shape
``(N, M+1)`` and an augmented input ``x = [a_in ; 1]``, so the layer's
pre-activation is ``z = A x`` and its claimed output is ``a``.  The layer is
correct exactly when

1. ``a_j >= 0``                                for every ``j``     (ReLU range)
2. ``a_j * (a_j - z_j) == 0``                  for every ``j``     (complementarity)
3. ``z_j <= 0``  whenever ``a_j == 0``                             (correct branch)

Together these are equivalent to ``a == relu(A x)``: (2) forces ``a_j == z_j``
wherever ``a_j != 0``, (1) then makes that branch the right one, and (3) covers
the coordinates (2) says nothing about.

**(3) is where the zero blind spot reappears.**  A naive "batch the whole layer
into one random linear combination" check verifies (2) and stops, which is
vacuous exactly where ``a_j == 0`` -- precisely the attack of Section 4, in
arithmetic dress.  We close it by making the prover commit to a witness
``y_j := -z_j >= 0`` on the zero set, in the clear, which the verifier
range-checks for free and then ties back to the committed weights.

What the verifier actually does
-------------------------------
It checks ``a >= 0`` and ``y >= 0`` directly (both are in the clear), then folds
(2) and (3) into a *single* linear functional of ``A x``:

    ``<alpha * (s . a) + beta * t_Z, A x> == alpha * sum_j s_j a_j^2 - beta * <t_Z, y>``

with ``s``, ``t``, ``alpha``, ``beta`` Fiat-Shamir challenges.  Note the verifier
never learns ``z``; it only needs *one* evaluation of ``chi^T A`` for a challenge
vector ``chi`` it chose after the prover was committed.  That evaluation is what
the commitment below provides.

The weight commitment (Ligero-style, Merkle only)
-------------------------------------------------
``BatchedWeightCommitment`` Reed-Solomon-encodes each *row* of ``A`` to length
``rate * (M+1)`` and Merkle-commits the *columns* of the encoded matrix.  To open
``u = chi^T A`` the prover simply sends ``u``; the verifier re-encodes ``u``
itself and spot-checks ``t`` random columns, each of which must satisfy
``<chi, E[:, c]> == Enc(u)[c]``.  Because encoding is linear, an honest ``u``
passes every column; a dishonest one disagrees with ``Enc(chi^T A)`` on at least
``M' - M`` positions (Reed-Solomon distance), so each sampled column catches it
with probability ``>= 1 - 1/rate``.

This needs no proximity test, because in this protocol's threat model ``C_M`` is
an *honest* commitment to the intended model -- the whole premise is a prover
that commits a benign model and then lies about the trace (Section 1).  A
malicious *committer* would additionally need the standard Ligero proximity
argument; that is a real extension, not something this module claims.

Arithmetic
----------
Everything is exact integer arithmetic in a prime field, which also disposes of
the floating-point tolerance that ``SPEC_NOTES.md`` Section 2 had to introduce
(and that handed the adversary a free perturbation budget below ``1e-4``):
honest completeness here is exactly 1, not 1-up-to-tau.  The network is read
through a fixed-point view -- the committed model *is* the fixed-point model, in
the same spirit as the paper's own 8-bit quantisation setting (Section A).

Scope, stated plainly: the field is small enough that a whole layer's
pre-activations fit without rescaling, which holds comfortably for the first
layer of the MNIST MLPs used throughout this study.  Chaining the check through
*every* layer of a deep network needs either a larger prime with a wider
multiplication routine, or per-layer rescaling with a range argument -- standard
zkML engineering, and out of scope here.  See ``revision1_notes/phase4_batched.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pvi.commitments.merkle import MerkleOpening, MerkleVectorCommitment

__all__ = [
    "BatchedLayerProof",
    "BatchedWeightCommitment",
    "FixedPointLayer",
    "DEFAULT_PRIME",
    "fixed_point_layer",
    "prove_layer",
    "verify_layer",
]

# Largest prime below 2**26.  Chosen so that a full int64 matmul over the field
# cannot overflow: products are < 2**52, and summing up to ~2**11 of them stays
# under 2**63.  That keeps the encoder a single vectorised numpy matmul.
DEFAULT_PRIME = 67108859


# --------------------------------------------------------------------------- #
# Field helpers
# --------------------------------------------------------------------------- #


def _as_field(values: np.ndarray, prime: int) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=np.int64), prime)


def _signed(values: np.ndarray, prime: int) -> np.ndarray:
    """Interpret field elements as integers in ``(-prime/2, prime/2)``."""
    values = np.asarray(values, dtype=np.int64)
    return np.where(values > prime // 2, values - prime, values)


def _chunk_size(prime: int) -> int:
    """How many products of field elements can be summed before int64 overflows."""
    limit = ((1 << 62) - 1) // max((prime - 1) ** 2, 1)
    return max(int(limit), 1)


def _matmul(left: np.ndarray, right: np.ndarray, prime: int) -> np.ndarray:
    """Field matrix product, chunked so partial sums never overflow int64.

    Any layer width is fine: the contraction is split into blocks small enough
    that a block's products sum below ``2**63``, with a reduction between blocks.
    """
    left = left.astype(np.int64, copy=False)
    right = right.astype(np.int64, copy=False)
    contraction = left.shape[-1]
    step = _chunk_size(prime)
    if contraction <= step:
        return np.mod(left @ right, prime)

    acc = np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
    for start in range(0, contraction, step):
        stop = min(start + step, contraction)
        acc = np.mod(acc + (left[:, start:stop] @ right[start:stop, :]), prime)
    return acc


def _challenges(transcript: bytes, count: int, prime: int, tag: bytes) -> np.ndarray:
    """Fiat-Shamir: derive ``count`` field elements from a transcript."""
    out = np.empty(count, dtype=np.int64)
    produced = 0
    counter = 0
    while produced < count:
        block = hashlib.sha256(tag + counter.to_bytes(8, "big") + transcript).digest()
        for offset in range(0, len(block), 8):
            if produced >= count:
                break
            word = int.from_bytes(block[offset : offset + 8], "big")
            out[produced] = word % prime
            produced += 1
        counter += 1
    return out


def _transcript_bytes(*parts: np.ndarray) -> bytes:
    hasher = hashlib.sha256()
    for part in parts:
        array = np.ascontiguousarray(np.asarray(part, dtype=np.int64))
        hasher.update(len(array).to_bytes(8, "big"))
        hasher.update(array.tobytes())
    return hasher.digest()


# --------------------------------------------------------------------------- #
# Fixed-point view of a layer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FixedPointLayer:
    """A layer's weights and activations as exact field elements.

    ``matrix`` is the augmented ``[W | b]`` of shape ``(N, M+1)``; ``inputs`` is
    the augmented ``[a_in ; 1]``.  Pre-activations ``z = matrix @ inputs`` and the
    honest output is ``relu(z)``, both at scale ``input_scale * weight_scale``.
    """

    matrix: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    prime: int
    scale_bits: int

    @property
    def width(self) -> int:
        return self.matrix.shape[0]

    def pre_activations(self) -> np.ndarray:
        """The true ``z``, as signed integers.  Prover-side only.

        The stored matrix and inputs are field *representatives* in ``[0, p)``,
        so a negative weight is a large residue: the product has to be reduced
        modulo ``p`` and mapped back to a signed representative, not read off
        directly.
        """
        z = _matmul(self.matrix, self.inputs[:, None], self.prime)[:, 0]
        return _signed(z, self.prime)


def fixed_point_layer(
    weight: np.ndarray,
    bias: np.ndarray,
    activations_in: np.ndarray,
    activations_out: np.ndarray | None = None,
    *,
    scale_bits: int = 8,
    prime: int = DEFAULT_PRIME,
) -> FixedPointLayer:
    """Quantise one dense ReLU layer and its activations into the field.

    ``activations_out`` defaults to the honest ``relu(z)``.  Pass a tampered
    vector (already quantised at the layer's output scale) to model an
    adversarial trace.
    """
    scale = 1 << scale_bits
    weight_int = np.rint(np.asarray(weight, dtype=np.float64) * scale).astype(np.int64)
    bias_int = np.rint(np.asarray(bias, dtype=np.float64) * scale * scale).astype(np.int64)
    inputs_int = np.rint(np.asarray(activations_in, dtype=np.float64) * scale).astype(np.int64)

    matrix = np.concatenate([weight_int, bias_int[:, None]], axis=1)
    inputs = np.concatenate([inputs_int, np.ones(1, dtype=np.int64)])

    z = matrix @ inputs
    if np.max(np.abs(z)) >= prime // 2:
        raise OverflowError(
            f"pre-activations reach {int(np.max(np.abs(z)))}, which does not fit "
            f"signed in the field of size {prime}; lower scale_bits"
        )
    outputs = np.maximum(z, 0) if activations_out is None else np.asarray(
        activations_out, dtype=np.int64
    )
    return FixedPointLayer(
        matrix=_as_field(matrix, prime),
        inputs=_as_field(inputs, prime),
        outputs=_as_field(outputs, prime),
        prime=prime,
        scale_bits=scale_bits,
    )


# --------------------------------------------------------------------------- #
# Weight commitment
# --------------------------------------------------------------------------- #


class BatchedWeightCommitment:
    """Reed-Solomon encode the rows of ``A``, Merkle-commit the columns.

    Built once, at commit time, from the honest model -- exactly where the
    paper's own ``CommitToModel`` sits.  Supports opening ``chi^T A`` for a
    challenge vector the verifier picks afterwards.
    """

    def __init__(self, matrix: np.ndarray, *, prime: int = DEFAULT_PRIME, rate: int = 2) -> None:
        if rate < 2:
            raise ValueError(f"rate must be at least 2, got {rate}")
        self._prime = prime
        self._rate = rate
        self._matrix = _as_field(matrix, prime)
        self._n_rows, self._row_length = self._matrix.shape
        self._n_columns = self._row_length * rate
        if self._n_columns >= prime:
            raise ValueError("field too small for the requested code length")

        self._vandermonde = self._build_vandermonde(self._row_length, self._n_columns, prime)
        self._encoded = _matmul(self._matrix, self._vandermonde.T, prime)
        self._commitment = MerkleVectorCommitment.commit(
            [self._encoded[:, c].tobytes() for c in range(self._n_columns)]
        )

    @staticmethod
    def _build_vandermonde(length: int, n_points: int, prime: int) -> np.ndarray:
        """``V[k][i] = (k+1)^i`` -- evaluation of a degree-<length polynomial."""
        points = np.arange(1, n_points + 1, dtype=np.int64) % prime
        vandermonde = np.empty((n_points, length), dtype=np.int64)
        current = np.ones(n_points, dtype=np.int64)
        for i in range(length):
            vandermonde[:, i] = current
            current = np.mod(current * points, prime)
        return vandermonde

    @property
    def digest(self) -> bytes:
        return self._commitment.digest

    @property
    def params(self):
        return self._commitment.params

    @property
    def row_length(self) -> int:
        return self._row_length

    @property
    def n_columns(self) -> int:
        return self._n_columns

    def encode(self, row: np.ndarray) -> np.ndarray:
        """Reed-Solomon encode a single length-``row_length`` vector."""
        return _matmul(_as_field(row, self._prime)[None, :], self._vandermonde.T, self._prime)[0]

    def combination(self, challenge: np.ndarray) -> np.ndarray:
        """``chi^T A``, the value the prover must send."""
        return _matmul(_as_field(challenge, self._prime)[None, :], self._matrix, self._prime)[0]

    def open_columns(self, indices: np.ndarray) -> tuple[MerkleOpening, ...]:
        return self._commitment.open_many(int(i) for i in indices)


# --------------------------------------------------------------------------- #
# Proof
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BatchedLayerProof:
    """What the prover sends for one layer."""

    combination: np.ndarray
    """``u = chi^T A`` -- length ``M+1``."""

    zero_indices: np.ndarray
    """Coordinates where the claimed output is zero."""

    sign_witness: np.ndarray
    """``y_j = -z_j >= 0`` on ``zero_indices`` -- closes the zero blind spot."""

    column_openings: tuple[MerkleOpening, ...]

    def size_bytes(self) -> int:
        scalars = len(self.combination) + len(self.zero_indices) + len(self.sign_witness)
        return 8 * scalars + sum(int(o.size_bytes) for o in self.column_openings)


def _derive(
    digest: bytes,
    inputs: np.ndarray,
    outputs: np.ndarray,
    zero_indices: np.ndarray,
    sign_witness: np.ndarray,
    width: int,
    n_columns: int,
    n_queries: int,
    prime: int,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Fiat-Shamir challenges, derived after the prover is committed to y."""
    transcript = digest + _transcript_bytes(inputs, outputs, zero_indices, sign_witness)
    s = _challenges(transcript, width, prime, b"pvi/batched/s")
    t = _challenges(transcript, len(zero_indices), prime, b"pvi/batched/t")
    mixers = _challenges(transcript, 2, prime, b"pvi/batched/mix")
    columns = _challenges(transcript, n_queries, prime, b"pvi/batched/col") % n_columns
    return s, t, int(mixers[0]), int(mixers[1]), columns


def _challenge_vector(
    outputs: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    alpha: int,
    beta: int,
    zero_indices: np.ndarray,
    prime: int,
) -> np.ndarray:
    """``chi = alpha * (s . a) + beta * t_Z``, the single functional to open."""
    chi = np.mod(alpha * np.mod(s * outputs, prime), prime)
    if len(zero_indices):
        chi[zero_indices] = np.mod(chi[zero_indices] + beta * t, prime)
    return chi


def prove_layer(
    layer: FixedPointLayer,
    commitment: BatchedWeightCommitment,
    *,
    n_queries: int = 24,
) -> BatchedLayerProof:
    """Produce the proof for one layer of a (possibly adversarial) trace.

    The prover is honest about the *weights* (they are committed) but may have
    claimed any ``layer.outputs`` it likes; ``verify_layer`` is what decides.
    """
    prime = layer.prime
    outputs_signed = _signed(layer.outputs, prime)
    zero_indices = np.flatnonzero(outputs_signed == 0).astype(np.int64)

    z = _signed(_matmul(layer.matrix, layer.inputs[:, None], prime)[:, 0], prime)
    sign_witness = _as_field(-z[zero_indices], prime) if len(zero_indices) else np.empty(0, np.int64)

    s, t, alpha, beta, columns = _derive(
        commitment.digest, layer.inputs, layer.outputs, zero_indices, sign_witness,
        layer.width, commitment.n_columns, n_queries, prime,
    )
    chi = _challenge_vector(layer.outputs, s, t, alpha, beta, zero_indices, prime)
    return BatchedLayerProof(
        combination=commitment.combination(chi),
        zero_indices=zero_indices,
        sign_witness=sign_witness,
        column_openings=commitment.open_columns(columns),
    )


def verify_layer(
    proof: BatchedLayerProof,
    commitment_digest: bytes,
    commitment_params,
    encoder: BatchedWeightCommitment,
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    prime: int = DEFAULT_PRIME,
    n_queries: int = 24,
) -> bool:
    """Accept iff the claimed ``outputs`` really are ``relu(A @ inputs)``.

    ``encoder`` is used only for the *public* parameters of the code (the
    Vandermonde matrix and its dimensions).  The verifier never reads its
    matrix -- that is the point of the commitment.
    """
    width = len(outputs)
    outputs_signed = _signed(outputs, prime)

    # (1) ReLU range, free: the claimed activations are in the clear.
    if np.any(outputs_signed < 0):
        return False

    # The zero set is determined by the claimed output; the prover cannot choose it.
    expected_zero = np.flatnonzero(outputs_signed == 0).astype(np.int64)
    if not np.array_equal(expected_zero, np.asarray(proof.zero_indices, dtype=np.int64)):
        return False

    # (3a) the sign witness must itself be non-negative -- also free.
    witness_signed = _signed(proof.sign_witness, prime)
    if len(witness_signed) != len(expected_zero) or np.any(witness_signed < 0):
        return False

    s, t, alpha, beta, columns = _derive(
        commitment_digest, inputs, outputs, proof.zero_indices, proof.sign_witness,
        width, encoder.n_columns, n_queries, prime,
    )
    chi = _challenge_vector(outputs, s, t, alpha, beta, proof.zero_indices, prime)

    # The commitment opening: u must really be chi^T A.
    if len(proof.column_openings) != len(columns):
        return False
    encoded_u = encoder.encode(proof.combination)
    for opening, column in zip(proof.column_openings, columns):
        if opening.index != int(column):
            return False
        if not MerkleVectorCommitment.verify(commitment_params, commitment_digest, opening):
            return False
        column_values = np.frombuffer(opening.value, dtype=np.int64)
        if len(column_values) != width:
            return False
        lhs = int(np.mod(np.sum(np.mod(chi * column_values, prime)), prime))
        if lhs != int(encoded_u[int(column)]):
            return False

    # (2) + (3b), folded into one scalar identity over the committed weights.
    lhs = int(np.mod(np.sum(np.mod(proof.combination * _as_field(inputs, prime), prime)), prime))
    quadratic = int(np.mod(np.sum(np.mod(np.mod(s * outputs, prime) * outputs, prime)), prime))
    witness_term = (
        int(np.mod(np.sum(np.mod(t * proof.sign_witness, prime)), prime))
        if len(proof.zero_indices)
        else 0
    )
    rhs = int(np.mod(alpha * quadratic - beta * witness_term, prime))
    return lhs == rhs
