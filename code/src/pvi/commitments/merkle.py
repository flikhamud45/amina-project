"""Merkle-tree vector commitments.

This module implements the vector-commitment primitive specified in Appendix B of
Anchuri et al., instantiated -- as the authors do -- with a Merkle tree over
SHA-256 (their Section 8.1: *"a Merkle tree built on row-wise hashes (computed
using SHA-256)"*).

A vector commitment over an alphabet ``Sigma`` is a tuple of algorithms

    ``GenParams(1^lambda, n) -> pp``
    ``CommitVec(pp, v_1, ..., v_n) -> cm``
    ``Open(pp, v, i) -> pi_i``
    ``Verify(pp, cm, i, v_i, pi_i) -> {0, 1}``

satisfying *binding* and *position binding*.  Those two properties are exactly
what the protocol leans on: position binding is what stops a cheating prover from
answering the same trace index with different values on different challenges.

Design notes
------------

*Domain separation.*  Leaf, internal-node, padding and root hashes are computed
under four distinct one-byte tags.  Without this, a Merkle tree is vulnerable to
the classical second-preimage attack in which an internal node is passed off as a
leaf.  The paper does not discuss the point; we take the standard precaution.

*Length binding.*  The published commitment is ``H(ROOT_TAG || n || root)`` rather
than the bare root, so a commitment to a vector of length ``n`` cannot be
reinterpreted as a commitment to a vector of a different length.

*Padding.*  The leaf count is rounded up to a power of two using padding leaves
``H(PAD_TAG || index)``.  These are distinct from every data leaf (different tag),
so padding can never be opened as data.

*Canonical encoding.*  Committed values are byte strings.  For the floating-point
vectors used by the protocol (weights and activations) the caller is expected to
use :func:`encode_f32_vector`, which fixes IEEE-754 binary32 in big-endian byte
order.  Fixing the encoding matters: binding is a statement about byte strings, so
two encodings of "the same" float would otherwise be two different commitments.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "HASH_BYTES",
    "MerkleOpening",
    "MerkleParams",
    "MerkleVectorCommitment",
    "decode_f32_vector",
    "encode_f32_vector",
]

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

HASH_BYTES = 32
"""Digest length of SHA-256, in bytes."""

_LEAF_TAG = b"\x00"
_NODE_TAG = b"\x01"
_PAD_TAG = b"\x02"
_ROOT_TAG = b"\x03"


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def _u64(value: int) -> bytes:
    """Big-endian 8-byte encoding of a non-negative integer."""
    if value < 0:
        raise ValueError(f"cannot encode negative integer {value}")
    return struct.pack(">Q", value)


def _leaf_hash(index: int, value: bytes) -> bytes:
    return _sha256(_LEAF_TAG, _u64(index), _u64(len(value)), value)


def _pad_hash(index: int) -> bytes:
    return _sha256(_PAD_TAG, _u64(index))


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _sha256(_NODE_TAG, left, right)


# --------------------------------------------------------------------------- #
# Canonical encoding of floating-point data
# --------------------------------------------------------------------------- #


def encode_f32_vector(values: np.ndarray) -> bytes:
    """Encode an array as big-endian IEEE-754 binary32.

    The protocol commits to real-valued weights and activations.  Binding is a
    property of the committed *bytes*, so the float-to-bytes map has to be fixed
    once and for all; we fix it here.

    NaNs are rejected: IEEE-754 admits many NaN bit patterns, so allowing them
    would silently weaken binding (two "equal" values with different encodings).
    Honest traces of the networks we study never contain NaNs, and a trace that
    does is not one the verifier should accept.
    """
    array = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("refusing to commit to a vector containing NaN or infinity")
    return array.astype(">f4", copy=False).tobytes()


def decode_f32_vector(blob: bytes) -> np.ndarray:
    """Inverse of :func:`encode_f32_vector`."""
    if len(blob) % 4 != 0:
        raise ValueError(f"blob length {len(blob)} is not a multiple of 4")
    return np.frombuffer(blob, dtype=">f4").astype(np.float32)


# --------------------------------------------------------------------------- #
# Public parameters, openings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MerkleParams:
    """Public parameters ``pp`` output by ``GenParams(1^lambda, n)``.

    SHA-256 is a fixed-output hash, so the security parameter is not a free knob:
    we record the requested ``lambda`` and refuse values the instantiation cannot
    deliver (collision resistance caps out at 128 bits).
    """

    security_bits: int
    length: int

    def __post_init__(self) -> None:
        if self.security_bits > 128:
            raise ValueError(
                f"SHA-256 provides at most 128 bits of collision resistance, "
                f"{self.security_bits} requested"
            )
        if self.length <= 0:
            raise ValueError(f"vector length must be positive, got {self.length}")

    @property
    def tree_height(self) -> int:
        """Number of levels above the leaves once the leaf count is padded."""
        return max(1, self.length - 1).bit_length()


@dataclass(frozen=True)
class MerkleOpening:
    """An opening ``pi_i`` proving that position ``index`` holds ``value``."""

    index: int
    value: bytes
    siblings: tuple[bytes, ...]

    @property
    def size_bytes(self) -> int:
        """Wire size of the opening: the value plus the authentication path."""
        return len(self.value) + len(self.siblings) * HASH_BYTES


# --------------------------------------------------------------------------- #
# The scheme
# --------------------------------------------------------------------------- #


class MerkleVectorCommitment:
    """Merkle-tree instantiation of the vector-commitment syntax of Appendix B.

    ``CommitVec`` is deterministic, as required by the paper's
    ``CommitToModel`` (Definition 4).  The scheme is therefore *not* hiding --
    which is exactly the paper's position: it explicitly notes (footnote 2) that
    "cleartext opening" reveals information about the model and that
    zero-knowledge is left to future work.

    Instances hold the committed vector so that :meth:`open` can be called
    repeatedly; the tree is built once at construction time.
    """

    def __init__(self, params: MerkleParams, values: Sequence[bytes]) -> None:
        if len(values) != params.length:
            raise ValueError(
                f"parameters fix length {params.length}, got {len(values)} values"
            )
        self._params = params
        self._values: tuple[bytes, ...] = tuple(values)
        self._levels: list[list[bytes]] = self._build(self._values)

    # -- construction ------------------------------------------------------- #

    @staticmethod
    def gen_params(security_bits: int, length: int) -> MerkleParams:
        """``GenParams(1^lambda, n)``."""
        return MerkleParams(security_bits=security_bits, length=length)

    @classmethod
    def commit(
        cls,
        values: Sequence[bytes],
        *,
        security_bits: int = 128,
    ) -> "MerkleVectorCommitment":
        """``CommitVec(pp, v)`` -- build the tree over ``values``."""
        params = cls.gen_params(security_bits, len(values))
        return cls(params, values)

    @staticmethod
    def _build(values: tuple[bytes, ...]) -> list[list[bytes]]:
        """Return the tree bottom-up; ``levels[0]`` are leaves, ``levels[-1]`` the root."""
        padded_width = 1 << max(1, len(values) - 1).bit_length()
        level: list[bytes] = [_leaf_hash(i, v) for i, v in enumerate(values)]
        level.extend(_pad_hash(i) for i in range(len(values), padded_width))

        levels = [level]
        while len(level) > 1:
            level = [
                _node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)
            ]
            levels.append(level)
        return levels

    # -- accessors ---------------------------------------------------------- #

    @property
    def params(self) -> MerkleParams:
        return self._params

    @property
    def digest(self) -> bytes:
        """The published commitment ``cm``, binding both the root and the length."""
        return _sha256(_ROOT_TAG, _u64(self._params.length), self._levels[-1][0])

    def __len__(self) -> int:
        return self._params.length

    # -- opening ------------------------------------------------------------ #

    def open(self, index: int) -> MerkleOpening:
        """``Open(pp, v, i)`` -- produce the authentication path for position ``i``."""
        if not 0 <= index < self._params.length:
            raise IndexError(
                f"index {index} out of range for vector of length {self._params.length}"
            )
        siblings: list[bytes] = []
        node = index
        for level in self._levels[:-1]:
            siblings.append(level[node ^ 1])
            node >>= 1
        return MerkleOpening(
            index=index, value=self._values[index], siblings=tuple(siblings)
        )

    def open_many(self, indices: Iterable[int]) -> tuple[MerkleOpening, ...]:
        """Open several positions.  Duplicate indices are opened once each."""
        return tuple(self.open(i) for i in indices)

    # -- verification ------------------------------------------------------- #

    @staticmethod
    def verify(
        params: MerkleParams,
        digest: bytes,
        opening: MerkleOpening,
    ) -> bool:
        """``Verify(pp, cm, i, v_i, pi_i)``.

        Returns ``True`` iff ``opening`` is a valid authentication path for its
        own ``index``/``value`` against ``digest``.  The caller is responsible for
        checking that ``opening.index`` is the position it actually wanted -- an
        opening is only ever evidence about the position it names.
        """
        if not 0 <= opening.index < params.length:
            return False
        if len(opening.siblings) != params.tree_height:
            # A short path would let a prover present an internal node as a root.
            return False

        node = _leaf_hash(opening.index, opening.value)
        position = opening.index
        for sibling in opening.siblings:
            if position & 1:
                node = _node_hash(sibling, node)
            else:
                node = _node_hash(node, sibling)
            position >>= 1

        expected = _sha256(_ROOT_TAG, _u64(params.length), node)
        # Constant-time comparison is not security-relevant here (both sides are
        # public), but costs nothing and keeps the primitive above reproach.
        return hmac.compare_digest(expected, digest)
