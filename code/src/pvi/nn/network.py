"""Execution traces and the networks that produce them.

Section 2 of Anchuri et al. defines the execution trace as "the list
``(a_1, a_2, ...)`` of activation values", one entry per neuron, "organized so
that each entry is associated with a specific layer and has a well-defined set of
parents".  :class:`Trace` is that object; :class:`TracedNetwork` is the
``EvalTrace`` function of Definition 5.

Neurons are addressed layer-locally, as ``(layer, neuron)``: ``parents()`` returns
positions inside the previous layer's activation vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np

from pvi.nn.architecture import Architecture, Layer, NO_WEIGHT_GROUP

__all__ = ["ModelParameters", "Trace", "TracedNetwork"]


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Trace:
    """One execution trace: activations grouped by layer, input layer first.

    Instances are treated as immutable; :meth:`tampered` returns a modified copy
    rather than mutating in place, which keeps honest and adversarial traces
    clearly separated in the experiments.
    """

    activations: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        for i, layer in enumerate(self.activations):
            if layer.ndim != 1:
                raise ValueError(f"layer {i} activations must be 1-D, got {layer.shape}")
            if layer.dtype != np.float32:
                raise ValueError(f"layer {i} activations must be float32, got {layer.dtype}")

    # -- structure ---------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, layer: int) -> np.ndarray:
        return self.activations[layer]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.activations)

    @property
    def output(self) -> np.ndarray:
        """``out(trc)`` -- the final layer's activations."""
        return self.activations[-1]

    # -- derived traces ----------------------------------------------------- #

    def tampered(self, layer: int, neuron: int, value: float) -> "Trace":
        """Return a copy with a single activation overwritten.

        Used by the attack: replacing one entry and leaving everything else alone
        creates a trace that is locally inconsistent at exactly one node.
        """
        activations = list(self.activations)
        modified = activations[layer].copy()
        modified[neuron] = np.float32(value)
        activations[layer] = modified
        return Trace(tuple(activations))

    def replace_layer(self, layer: int, values: np.ndarray) -> "Trace":
        """Return a copy with one whole layer replaced."""
        activations = list(self.activations)
        activations[layer] = np.ascontiguousarray(values, dtype=np.float32)
        return Trace(tuple(activations))


# --------------------------------------------------------------------------- #
# Parameters and network
# --------------------------------------------------------------------------- #

ModelParameters = Mapping[str, tuple[np.ndarray, np.ndarray]]
"""Layer name -> ``(weight, bias)``.  Parameter-free layers are simply absent."""


class TracedNetwork:
    """``EvalTrace(M, qry)`` together with the per-neuron accessors the protocol needs.

    This class is the single source of truth for what the model computes.  The
    same :meth:`forward_batch` code path produces both the accuracy numbers we
    report and the activations we commit to, so there is no possibility of the
    "evaluated model" and the "committed model" drifting apart.
    """

    def __init__(self, architecture: Architecture, parameters: ModelParameters) -> None:
        self._arch = architecture
        self._params: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for layer in architecture.layers:
            if layer.n_weight_groups == 0:
                if layer.name in parameters:
                    raise ValueError(f"{layer.name} takes no parameters but some were given")
                continue
            if layer.name not in parameters:
                raise ValueError(f"missing parameters for layer {layer.name}")
            weight, bias = parameters[layer.name]
            # weight_rows validates the shapes and gives us the committed layout.
            layer.weight_rows(weight, bias)
            self._params[layer.name] = (
                np.ascontiguousarray(weight, dtype=np.float32),
                np.ascontiguousarray(bias, dtype=np.float32),
            )

        self._weight_rows: dict[str, np.ndarray] = {
            layer.name: layer.weight_rows(*self._params[layer.name])
            for layer in architecture.layers
            if layer.n_weight_groups > 0
        }

    # -- accessors ---------------------------------------------------------- #

    @property
    def architecture(self) -> Architecture:
        return self._arch

    @property
    def parameters(self) -> ModelParameters:
        return dict(self._params)

    def weight_rows(self, layer_index: int) -> np.ndarray:
        """All committed weight rows of one layer, shape ``(groups, group_size)``."""
        layer = self._arch[layer_index]
        if layer.n_weight_groups == 0:
            raise TypeError(f"{layer.name} has no weight rows")
        return self._weight_rows[layer.name]

    def weight_row(self, layer_index: int, neuron: int) -> np.ndarray | None:
        """The single weight row neuron ``neuron`` of layer ``layer_index`` needs."""
        layer = self._arch[layer_index]
        group = layer.weight_group_of(neuron)
        if group == NO_WEIGHT_GROUP:
            return None
        return self._weight_rows[layer.name][group]

    # -- evaluation --------------------------------------------------------- #

    def forward_batch(self, queries: np.ndarray) -> list[np.ndarray]:
        """Run a batch and return every layer's activations, input layer first.

        Memory cost is ``batch x (total trace entries)`` floats, plus the
        im2col workspace convolutions need.  That is fine for the small batches
        the protocol works with, but it is emphatically not the way to evaluate a
        60 000-image test set -- use :meth:`outputs`, :meth:`predict` or
        :meth:`accuracy`, which stream in chunks and keep only the final layer.
        """
        flat = np.ascontiguousarray(queries, dtype=np.float32).reshape(
            queries.shape[0], -1
        )
        expected = self._arch.input_layer.n_neurons
        if flat.shape[1] != expected:
            raise ValueError(
                f"expected {expected} input features, got {flat.shape[1]}"
            )

        activations = [flat]
        for layer in self._arch.layers[1:]:
            weight, bias = self._params.get(layer.name, (None, None))
            activations.append(layer.forward_batch(activations[-1], weight, bias))
        return activations

    def outputs(self, queries: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
        """Final-layer activations for a batch, evaluated in chunks.

        Only the output layer is retained, so peak memory stays proportional to
        ``chunk_size`` rather than to the size of the dataset.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(
            len(queries), -1
        )
        chunks = [
            self.forward_batch(queries[start : start + chunk_size])[-1]
            for start in range(0, len(queries), chunk_size)
        ]
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), np.float32)

    def eval_trace(self, query: np.ndarray) -> Trace:
        """``EvalTrace(M, qry)`` for a single query."""
        batched = self.forward_batch(np.asarray(query, dtype=np.float32)[None, ...])
        return Trace(tuple(np.ascontiguousarray(a[0], dtype=np.float32) for a in batched))

    def predict(self, queries: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
        """Argmax of the output layer for a batch of queries."""
        return self.outputs(queries, chunk_size=chunk_size).argmax(axis=1)

    def accuracy(
        self, queries: np.ndarray, labels: np.ndarray, *, chunk_size: int = 512
    ) -> float:
        predicted = self.predict(queries, chunk_size=chunk_size)
        return float((predicted == np.asarray(labels)).mean())

    # -- local consistency -------------------------------------------------- #

    def recompute(self, trace: Trace, layer_index: int, neuron: int) -> float:
        """Recompute one activation from the trace's *claimed* parent values.

        This is the honest-verifier computation behind check ``(*)``:
        ``phi(sum_{i in G_j} w_ij a~_i)``.  Note that it reads the parents from
        ``trace`` -- which may be adversarial -- but the weights from this
        network, which is the committed ground-truth model ``M``.
        """
        layer = self._arch[layer_index]
        parent_idx, weight_idx = layer.parents(neuron)
        parent_values = trace[layer_index - 1][parent_idx]
        row = self.weight_row(layer_index, neuron)
        if row is None:
            return layer.local_value(neuron, parent_values, None, None)
        weights, bias = layer.split_row(row, weight_idx)
        return layer.local_value(neuron, parent_values, weights, bias)

    def forward_from(self, trace: Trace, layer_index: int) -> Trace:
        """Recompute every layer above ``layer_index`` from that layer's values.

        The adversary uses this to make a tampered trace *locally consistent
        everywhere above the tamper*: it perturbs one activation and then lets the
        honest model propagate the perturbation forward.
        """
        activations = list(trace.activations[: layer_index + 1])
        for layer in self._arch.layers[layer_index + 1 :]:
            weight, bias = self._params.get(layer.name, (None, None))
            nxt = layer.forward_batch(activations[-1][None, :], weight, bias)[0]
            activations.append(np.ascontiguousarray(nxt, dtype=np.float32))
        return Trace(tuple(activations))
