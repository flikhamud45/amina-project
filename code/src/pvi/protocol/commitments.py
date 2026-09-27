"""The two commitments the protocol uses: ``C_M`` to the model, ``C_trc`` to the trace.

Figure 4 of Anchuri et al. specifies both.  ``C_M`` is computed once, at the end of
training, and published; the verifier holds only its digest.  ``C_trc`` is produced
by the prover for every query and is therefore *untrusted* -- the protocol's
guarantee is that whatever the prover committed to, it is stuck with it, which is
exactly the position-binding property of the vector commitment.

Commitment granularity
----------------------
The authors do not commit one leaf per scalar.  Section 8.1 describes "a Merkle
tree built on row-wise hashes": for a matrix-multiplication layer they commit the
input activation *matrix* and open "the relevant row and the corresponding Merkle
path for each layer".  We follow that:

* ``C_M`` has one leaf per **weight row** -- a dense neuron's incoming weights and
  bias, or a convolution's output-channel kernel and bias.  Checking a node
  therefore costs exactly one weight opening.
* ``C_trc`` has one leaf per **layer activation vector**.  In a transformer a layer
  holds many token rows and only one is opened; in the feed-forward networks
  studied here a layer is a single row, so the opened material is the layer
  itself.

Granularity affects proof size, not soundness.  What the security analysis turns
on is the set of nodes the verifier *checks*, and a node cannot be checked without
its weight row: the verifier does not hold ``M``, only ``C_M``.  The number of
weight-row openings is therefore the verifier's real budget, and it is what the
attack analysis is stated in terms of.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvi.commitments.merkle import (
    MerkleOpening,
    MerkleParams,
    MerkleVectorCommitment,
    decode_f32_vector,
    encode_f32_vector,
)
from pvi.nn.architecture import Architecture
from pvi.nn.network import Trace, TracedNetwork

__all__ = ["ModelCommitment", "TraceCommitment", "WeightRowIndex"]


# --------------------------------------------------------------------------- #
# Model commitment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WeightRowIndex:
    """Maps ``(layer, weight group)`` to a position in the committed weight vector.

    Both parties can compute this from the public architecture alone, so the
    verifier can tell which position an opening ought to be for.
    """

    architecture: Architecture

    @property
    def _offsets(self) -> dict[int, int]:
        offsets: dict[int, int] = {}
        running = 0
        for layer_index, layer in self.architecture.parameterised_layers():
            offsets[layer_index] = running
            running += layer.n_weight_groups
        return offsets

    def position(self, layer_index: int, group: int) -> int:
        offsets = self._offsets
        if layer_index not in offsets:
            raise KeyError(f"layer {layer_index} carries no weights")
        layer = self.architecture[layer_index]
        if not 0 <= group < layer.n_weight_groups:
            raise IndexError(f"weight group {group} out of range for {layer.name}")
        return offsets[layer_index] + group


class ModelCommitment:
    """``C_M`` -- a vector commitment to the model's weight rows.

    Constructed once from the trained model.  The prover keeps the instance (it
    holds the decommitment information); the verifier is given only
    :attr:`digest` and :attr:`params`.
    """

    def __init__(self, network: TracedNetwork) -> None:
        self._architecture = network.architecture
        self._index = WeightRowIndex(network.architecture)

        rows: list[bytes] = []
        for layer_index, layer in network.architecture.parameterised_layers():
            weight_rows = network.weight_rows(layer_index)
            if weight_rows.shape != (layer.n_weight_groups, layer.weight_group_size):
                raise ValueError(f"{layer.name}: unexpected weight-row shape")
            rows.extend(encode_f32_vector(row) for row in weight_rows)

        self._vc = MerkleVectorCommitment.commit(rows)

    @property
    def architecture(self) -> Architecture:
        return self._architecture

    @property
    def digest(self) -> bytes:
        return self._vc.digest

    @property
    def params(self) -> MerkleParams:
        return self._vc.params

    def open_row(self, layer_index: int, group: int) -> MerkleOpening:
        return self._vc.open(self._index.position(layer_index, group))

    @staticmethod
    def verify_row(
        params: MerkleParams,
        digest: bytes,
        index: WeightRowIndex,
        layer_index: int,
        group: int,
        opening: MerkleOpening,
    ) -> np.ndarray | None:
        """Check an opening and return the weight row it reveals, or ``None``.

        Returning the decoded row rather than a bare boolean keeps the verifier
        from ever touching a value it has not first authenticated.
        """
        expected_position = index.position(layer_index, group)
        if opening.index != expected_position:
            return None
        if not MerkleVectorCommitment.verify(params, digest, opening):
            return None
        row = decode_f32_vector(opening.value)
        if row.shape != (index.architecture[layer_index].weight_group_size,):
            return None
        return row


# --------------------------------------------------------------------------- #
# Trace commitment
# --------------------------------------------------------------------------- #


class TraceCommitment:
    """``C_trc`` -- a vector commitment to an execution trace, one leaf per layer.

    The trace passed in is whatever the prover chose to commit to.  Nothing here
    checks that it is the honest evaluation of any particular model; that is the
    verifier's job, and it is precisely what the protocol can only do
    approximately.
    """

    def __init__(self, trace: Trace) -> None:
        leaves = [encode_f32_vector(layer) for layer in trace]
        self._vc = MerkleVectorCommitment.commit(leaves)

    @property
    def digest(self) -> bytes:
        return self._vc.digest

    @property
    def params(self) -> MerkleParams:
        return self._vc.params

    def open_layer(self, layer_index: int) -> MerkleOpening:
        return self._vc.open(layer_index)

    @staticmethod
    def verify_layer(
        params: MerkleParams,
        digest: bytes,
        layer_index: int,
        expected_width: int,
        opening: MerkleOpening,
    ) -> np.ndarray | None:
        """Check an opening and return the layer activations it reveals, or ``None``."""
        if opening.index != layer_index:
            return None
        if not MerkleVectorCommitment.verify(params, digest, opening):
            return None
        values = decode_f32_vector(opening.value)
        if values.shape != (expected_width,):
            return None
        return values
