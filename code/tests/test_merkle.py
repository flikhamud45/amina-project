"""Tests for the Merkle vector commitment (Appendix B).

Binding and position binding are the only properties the protocol relies on, so
they get the most attention here -- including the failure modes that a naive
Merkle implementation gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.commitments.merkle import (
    MerkleOpening,
    MerkleVectorCommitment,
    decode_f32_vector,
    encode_f32_vector,
)


def values(n: int, seed: int = 0) -> list[bytes]:
    rng = np.random.default_rng(seed)
    return [encode_f32_vector(rng.standard_normal(3).astype(np.float32)) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def test_encoding_round_trip():
    array = np.array([0.0, -1.5, 3.25, 1e-8], dtype=np.float32)
    assert np.array_equal(decode_f32_vector(encode_f32_vector(array)), array)


def test_encoding_is_canonical_across_input_dtypes():
    as_f64 = np.array([1.0, 2.0], dtype=np.float64)
    as_f32 = np.array([1.0, 2.0], dtype=np.float32)
    assert encode_f32_vector(as_f64) == encode_f32_vector(as_f32)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_encoding_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        encode_f32_vector(np.array([1.0, bad], dtype=np.float32))


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 16, 17, 100])
def test_open_and_verify_every_position(n):
    vc = MerkleVectorCommitment.commit(values(n))
    for index in range(n):
        opening = vc.open(index)
        assert MerkleVectorCommitment.verify(vc.params, vc.digest, opening)


def test_commitment_is_deterministic():
    data = values(10)
    assert (
        MerkleVectorCommitment.commit(data).digest
        == MerkleVectorCommitment.commit(data).digest
    )


def test_opening_out_of_range_raises():
    vc = MerkleVectorCommitment.commit(values(5))
    with pytest.raises(IndexError):
        vc.open(5)


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #


def test_changing_any_value_changes_the_digest():
    data = values(12)
    baseline = MerkleVectorCommitment.commit(data).digest
    for index in range(len(data)):
        mutated = list(data)
        mutated[index] = encode_f32_vector(np.array([9.0, 9.0, 9.0], dtype=np.float32))
        assert MerkleVectorCommitment.commit(mutated).digest != baseline


def test_length_is_bound():
    """A prefix of a vector must not verify against the longer vector's digest."""
    data = values(8)
    short = MerkleVectorCommitment.commit(data[:4])
    long = MerkleVectorCommitment.commit(data)
    assert short.digest != long.digest


def test_reordering_changes_the_digest():
    data = values(8)
    swapped = list(data)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert (
        MerkleVectorCommitment.commit(swapped).digest
        != MerkleVectorCommitment.commit(data).digest
    )


# --------------------------------------------------------------------------- #
# Position binding and forgery resistance
# --------------------------------------------------------------------------- #


def test_forged_value_is_rejected():
    vc = MerkleVectorCommitment.commit(values(16))
    genuine = vc.open(3)
    forged = MerkleOpening(
        index=genuine.index,
        value=encode_f32_vector(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        siblings=genuine.siblings,
    )
    assert not MerkleVectorCommitment.verify(vc.params, vc.digest, forged)


def test_opening_replayed_at_another_index_is_rejected():
    """The value at position i must not verify as the value at position j."""
    data = values(16)
    data[9] = data[3]  # identical payloads, different positions
    vc = MerkleVectorCommitment.commit(data)
    genuine = vc.open(3)
    replayed = MerkleOpening(index=9, value=genuine.value, siblings=genuine.siblings)
    assert not MerkleVectorCommitment.verify(vc.params, vc.digest, replayed)


def test_truncated_authentication_path_is_rejected():
    """Rejecting short paths stops an internal node being passed off as the root."""
    vc = MerkleVectorCommitment.commit(values(16))
    genuine = vc.open(5)
    truncated = MerkleOpening(
        index=genuine.index, value=genuine.value, siblings=genuine.siblings[:-1]
    )
    assert not MerkleVectorCommitment.verify(vc.params, vc.digest, truncated)


def test_corrupted_sibling_is_rejected():
    vc = MerkleVectorCommitment.commit(values(16))
    genuine = vc.open(5)
    siblings = list(genuine.siblings)
    siblings[0] = bytes(32)
    corrupted = MerkleOpening(
        index=genuine.index, value=genuine.value, siblings=tuple(siblings)
    )
    assert not MerkleVectorCommitment.verify(vc.params, vc.digest, corrupted)


def test_padding_cannot_be_opened_as_data():
    """A padded leaf is domain-separated from data, so index 5 of a length-5
    vector has no valid opening even though the tree has 8 leaves."""
    vc = MerkleVectorCommitment.commit(values(5))
    forged = MerkleOpening(index=5, value=b"", siblings=tuple(bytes(32) for _ in range(3)))
    assert not MerkleVectorCommitment.verify(vc.params, vc.digest, forged)


def test_opening_does_not_verify_against_a_different_commitment():
    a = MerkleVectorCommitment.commit(values(16, seed=1))
    b = MerkleVectorCommitment.commit(values(16, seed=2))
    assert not MerkleVectorCommitment.verify(b.params, b.digest, a.open(0))


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def test_opening_size_is_logarithmic():
    small = MerkleVectorCommitment.commit(values(16))
    large = MerkleVectorCommitment.commit(values(1024))
    assert len(large.open(0).siblings) == 10
    assert len(small.open(0).siblings) == 4
