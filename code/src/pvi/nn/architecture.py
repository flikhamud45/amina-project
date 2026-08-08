"""The layered-DAG view of a network that the protocol operates on.

Anchuri et al. model a network as a DAG in which every neuron ``j`` has a parent
set ``G_j`` and an activation

    ``a_j = phi( sum_{i in G_j} w_ij a_i )``                     (Section 2)

and define the execution trace as the vector of all activations.  ``RandPathTest``
(Figure 3) walks a random path from an output neuron back to the input layer and
re-evaluates that one relation at each step.

To run the protocol we therefore need three things from every layer, for every
individual neuron:

1. its parent set ``G_j``, as indices into the previous layer's activation vector;
2. the weights ``(w_ij)_{i in G_j}`` that go with those parents;
3. the local relation itself, so the verifier can recompute ``a_j``.

That is what the classes here provide.  Everything is expressed per-neuron, which
is what the protocol needs and what makes the trace-tampering analysis in the
companion attack module exact.

Fused activations
-----------------
The paper's relation folds the non-linearity into the neuron: ``a_j`` is the
*post*-activation value, and there is one trace entry per neuron per layer.  We
follow that convention, so ``Dense`` and ``Conv2d`` carry their own ``activation``
rather than appearing as separate ReLU layers.  The final classification layer
uses ``identity`` so that the trace's output entries are the logits, which is what
``out(trc)`` reads (Remark 3).

Weight groups
-------------
Committing one Merkle leaf per scalar weight would make an opening cost
``|G_j|`` authentication paths.  The authors instead build the tree "on row-wise
hashes" (Section 8.1) so that a single opening yields every weight a node needs.
We model this with *weight groups*: each neuron maps to exactly one group, and the
group's committed value is the concatenation of that neuron's incoming weights
with its bias.  For ``Dense`` a group is a row of ``W``; for ``Conv2d``, where
weights are shared across spatial positions, a group is an output channel's
kernel.

Layers without parameters
-------------------------
``MaxPool2d`` has no weights, and its local relation is ``a_j = max_{i in G_j} a_i``
rather than an affine map followed by ``phi``.  The paper's formalism only spells
out the affine case, but the protocol needs no more than "a node's value is a
fixed, publicly known function of its parents", which pooling satisfies.  We note
this as a (conservative) extension of the paper's presentation; it is required to
handle the convolutional classifiers the authors themselves evaluate.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "Activation",
    "Architecture",
    "Conv2dLayer",
    "DenseLayer",
    "InputLayer",
    "Layer",
    "MaxPool2dLayer",
    "NO_WEIGHT_GROUP",
    "apply_activation",
]

Activation = Literal["relu", "identity"]

NO_WEIGHT_GROUP = -1
"""Weight-group index used by layers that carry no parameters."""


def apply_activation(z: np.ndarray | float, activation: Activation) -> np.ndarray | float:
    """Apply ``phi``."""
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "identity":
        return z
    raise ValueError(f"unknown activation {activation!r}")


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class Layer(abc.ABC):
    """One layer of the DAG.

    A layer knows its own output shape and, for each of its neurons, which
    neurons of the *previous* layer feed it and with which weights.
    """

    name: str

    # -- shape -------------------------------------------------------------- #

    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int, ...]:
        """Shape of this layer's activation tensor, excluding the batch axis."""

    @property
    def n_neurons(self) -> int:
        """Number of neurons, i.e. the length of this layer's slice of the trace."""
        return int(np.prod(self.out_shape))

    # -- parameters --------------------------------------------------------- #

    @property
    @abc.abstractmethod
    def n_weight_groups(self) -> int:
        """Number of committed weight rows.  Zero for parameter-free layers."""

    @property
    @abc.abstractmethod
    def weight_group_size(self) -> int:
        """Length of one weight row, *including* the trailing bias entry."""

    @abc.abstractmethod
    def weight_group_of(self, neuron: int) -> int:
        """Index of the weight row that neuron ``neuron`` needs.

        Returns :data:`NO_WEIGHT_GROUP` for parameter-free layers.
        """

    @abc.abstractmethod
    def weight_rows(self, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Lay the layer's parameters out as a ``(n_weight_groups, weight_group_size)`` matrix.

        The last column holds the bias.  This is the exact object committed to in
        ``C_M``, so its layout is part of the protocol specification.
        """

    # -- local structure ---------------------------------------------------- #

    @abc.abstractmethod
    def parents(self, neuron: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(parent_indices, weight_indices)`` for one neuron.

        ``parent_indices`` are positions in the previous layer's flat activation
        vector -- this is ``G_j``.  ``weight_indices`` are positions within the
        neuron's weight row, aligned element-wise with ``parent_indices``.

        Positions masked out by zero padding are simply absent from both arrays:
        they contribute nothing to the sum and have no trace entry to open.
        """

    @abc.abstractmethod
    def local_value(
        self,
        neuron: int,
        parent_values: np.ndarray,
        weights: np.ndarray | None,
        bias: float | None,
    ) -> float:
        """Recompute one neuron's activation from its parents.

        This is the verifier's side of the local consistency check ``(*)``.  It is
        deliberately written as an explicit per-neuron dot product rather than a
        vectorised call: the verifier only ever holds the values along the sampled
        path, never a whole layer's worth of weights.

        ``weights`` is aligned element-wise with ``parent_values``; ``bias`` is
        separate.  Use :meth:`split_row` to obtain them from a committed weight
        row -- passing the row directly would silently drop the bias, since the
        row is longer than the parent set whenever padding masks out parents.
        """

    def split_row(
        self, full_row: np.ndarray, weight_idx: np.ndarray
    ) -> tuple[np.ndarray | None, np.floating | None]:
        """Select the weights a neuron needs out of its committed row.

        A committed row is ``[w_0, ..., w_{K-1}, bias]``.  ``weight_idx`` picks the
        entries that correspond to the neuron's actual parents -- which is a strict
        subset whenever zero padding masks part of a receptive field.

        The bias is returned as a NumPy scalar, deliberately *not* as a Python
        ``float``.  A Python float is float64, and adding one to a float32 dot
        product promotes the whole expression to float64 -- arithmetic the prover
        never performed.  That mismatch would make the verifier disagree with an
        honest prover far more often than the underlying rounding requires, which
        matters directly for how tight the tolerance can be set.
        """
        if self.n_weight_groups == 0:
            return None, None
        return full_row[weight_idx], full_row[-1]

    # -- prover side -------------------------------------------------------- #

    @abc.abstractmethod
    def forward_batch(
        self,
        inputs: np.ndarray,
        weight: np.ndarray | None,
        bias: np.ndarray | None,
    ) -> np.ndarray:
        """Vectorised forward pass over a batch.

        ``inputs`` has shape ``(batch, previous_layer_neurons)``; the result has
        shape ``(batch, self.n_neurons)``.  This is the *only* forward
        implementation each layer provides, so the activations used for accuracy
        measurement and the activations committed in a trace are produced by
        exactly the same code path.
        """

    def forward(
        self,
        inputs: np.ndarray,
        weight: np.ndarray | None,
        bias: np.ndarray | None,
    ) -> np.ndarray:
        """Single-sample convenience wrapper around :meth:`forward_batch`."""
        return self.forward_batch(inputs[None, :], weight, bias)[0]


@dataclass(frozen=True)
class InputLayer(Layer):
    """The source layer.  Its activations are the query itself."""

    name: str
    shape: tuple[int, ...]

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self.shape

    @property
    def n_weight_groups(self) -> int:
        return 0

    @property
    def weight_group_size(self) -> int:
        return 0

    def weight_group_of(self, neuron: int) -> int:
        return NO_WEIGHT_GROUP

    def weight_rows(self, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        raise TypeError("the input layer has no parameters")

    def parents(self, neuron: int) -> tuple[np.ndarray, np.ndarray]:
        raise TypeError("the input layer has no parents")

    def local_value(
        self,
        neuron: int,
        parent_values: np.ndarray,
        weights: np.ndarray | None,
        bias: float | None,
    ) -> float:
        raise TypeError("the input layer has no local relation; it is anchored to qry")

    def forward_batch(
        self, inputs: np.ndarray, weight: np.ndarray | None, bias: np.ndarray | None
    ) -> np.ndarray:
        raise TypeError("the input layer has no forward pass; its activations are qry")


@dataclass(frozen=True)
class DenseLayer(Layer):
    """Fully connected layer with a fused activation.

    Every neuron's parent set is the whole previous layer, so a path step from
    this layer picks a uniformly random neuron of the layer below.
    """

    name: str
    in_features: int
    out_features: int
    activation: Activation = "relu"

    @property
    def out_shape(self) -> tuple[int, ...]:
        return (self.out_features,)

    @property
    def n_weight_groups(self) -> int:
        return self.out_features

    @property
    def weight_group_size(self) -> int:
        return self.in_features + 1

    def weight_group_of(self, neuron: int) -> int:
        return neuron

    def weight_rows(self, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"{self.name}: expected weight {(self.out_features, self.in_features)}, "
                f"got {weight.shape}"
            )
        if bias.shape != (self.out_features,):
            raise ValueError(
                f"{self.name}: expected bias {(self.out_features,)}, got {bias.shape}"
            )
        return np.concatenate(
            [weight.astype(np.float32), bias.astype(np.float32)[:, None]], axis=1
        )

    def parents(self, neuron: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(self.in_features, dtype=np.int64)
        return idx, idx

    def local_value(
        self,
        neuron: int,
        parent_values: np.ndarray,
        weights: np.ndarray | None,
        bias: float | None,
    ) -> float:
        assert weights is not None and bias is not None
        z = float(np.dot(weights, parent_values) + bias)
        return float(apply_activation(z, self.activation))

    def forward_batch(
        self, inputs: np.ndarray, weight: np.ndarray | None, bias: np.ndarray | None
    ) -> np.ndarray:
        assert weight is not None and bias is not None
        z = inputs @ weight.T + bias
        return np.asarray(apply_activation(z, self.activation), dtype=np.float32)


@dataclass(frozen=True)
class Conv2dLayer(Layer):
    """2-D convolution with a fused activation.

    Parent sets are receptive fields, so they are small and *local*: a path step
    from this layer stays inside one ``kernel x kernel`` window.  This locality is
    what makes convolutional layers a poor place to hide a trace tamper and dense
    layers a good one -- see the attack analysis.
    """

    name: str
    in_shape: tuple[int, int, int]  # (channels, height, width)
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    activation: Activation = "relu"

    @property
    def out_shape(self) -> tuple[int, int, int]:
        c_in, h_in, w_in = self.in_shape
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        if h_out <= 0 or w_out <= 0:
            raise ValueError(f"{self.name}: degenerate output shape")
        return (self.out_channels, h_out, w_out)

    @property
    def n_weight_groups(self) -> int:
        return self.out_channels

    @property
    def weight_group_size(self) -> int:
        c_in = self.in_shape[0]
        return c_in * self.kernel_size * self.kernel_size + 1

    def weight_group_of(self, neuron: int) -> int:
        _, h_out, w_out = self.out_shape
        return neuron // (h_out * w_out)

    def weight_rows(self, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        c_in = self.in_shape[0]
        expected = (self.out_channels, c_in, self.kernel_size, self.kernel_size)
        if weight.shape != expected:
            raise ValueError(f"{self.name}: expected weight {expected}, got {weight.shape}")
        flat = weight.astype(np.float32).reshape(self.out_channels, -1)
        return np.concatenate([flat, bias.astype(np.float32)[:, None]], axis=1)

    def parents(self, neuron: int) -> tuple[np.ndarray, np.ndarray]:
        c_in, h_in, w_in = self.in_shape
        _, h_out, w_out = self.out_shape
        oc, rem = divmod(neuron, h_out * w_out)
        oh, ow = divmod(rem, w_out)
        del oc  # the channel selects the weight row, not the parents

        k = self.kernel_size
        # Spatial offsets of the receptive field, shared across input channels.
        rows = oh * self.stride - self.padding + np.arange(k)
        cols = ow * self.stride - self.padding + np.arange(k)
        row_ok = (rows >= 0) & (rows < h_in)
        col_ok = (cols >= 0) & (cols < w_in)

        kh_idx, kw_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        valid = row_ok[:, None] & col_ok[None, :]
        kh_idx, kw_idx = kh_idx[valid], kw_idx[valid]
        r_idx, c_idx = rows[kh_idx], cols[kw_idx]

        channels = np.arange(c_in, dtype=np.int64)[:, None]
        parent = (
            channels * (h_in * w_in) + (r_idx[None, :] * w_in + c_idx[None, :])
        ).ravel()
        weight_idx = (
            channels * (k * k) + (kh_idx[None, :] * k + kw_idx[None, :])
        ).ravel()
        return parent.astype(np.int64), weight_idx.astype(np.int64)

    def local_value(
        self,
        neuron: int,
        parent_values: np.ndarray,
        weights: np.ndarray | None,
        bias: float | None,
    ) -> float:
        # The caller has already selected this neuron's output-channel row, so the
        # computation is identical to the dense case once parents are gathered.
        assert weights is not None and bias is not None
        z = float(np.dot(weights, parent_values) + bias)
        return float(apply_activation(z, self.activation))

    def forward_batch(
        self, inputs: np.ndarray, weight: np.ndarray | None, bias: np.ndarray | None
    ) -> np.ndarray:
        assert weight is not None and bias is not None
        batch = inputs.shape[0]
        c_in, h_in, w_in = self.in_shape
        _, h_out, w_out = self.out_shape
        k, s, p = self.kernel_size, self.stride, self.padding

        x = inputs.reshape(batch, c_in, h_in, w_in)
        if p:
            x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

        # im2col: (batch, c_in * k * k, h_out * w_out), laid out so that row
        # ``ic * k * k + kh * k + kw`` matches the flattened kernel ordering.
        cols = np.empty((batch, c_in * k * k, h_out * w_out), dtype=np.float32)
        channel_base = np.arange(c_in) * k * k
        for kh in range(k):
            for kw in range(k):
                patch = x[:, :, kh : kh + s * h_out : s, kw : kw + s * w_out : s]
                cols[:, channel_base + kh * k + kw, :] = patch.reshape(batch, c_in, -1)

        z = np.einsum(
            "oi,bip->bop", weight.reshape(self.out_channels, -1), cols, optimize=True
        )
        z = z + bias[None, :, None]
        activated = np.asarray(apply_activation(z, self.activation), dtype=np.float32)
        return activated.reshape(batch, -1)


@dataclass(frozen=True)
class MaxPool2dLayer(Layer):
    """Max pooling.  Local relation: ``a_j = max_{i in G_j} a_i`` (no weights)."""

    name: str
    in_shape: tuple[int, int, int]
    kernel_size: int
    stride: int | None = None

    @property
    def _stride(self) -> int:
        return self.kernel_size if self.stride is None else self.stride

    @property
    def out_shape(self) -> tuple[int, int, int]:
        c_in, h_in, w_in = self.in_shape
        h_out = (h_in - self.kernel_size) // self._stride + 1
        w_out = (w_in - self.kernel_size) // self._stride + 1
        return (c_in, h_out, w_out)

    @property
    def n_weight_groups(self) -> int:
        return 0

    @property
    def weight_group_size(self) -> int:
        return 0

    def weight_group_of(self, neuron: int) -> int:
        return NO_WEIGHT_GROUP

    def weight_rows(self, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        raise TypeError(f"{self.name}: max pooling has no parameters")

    def parents(self, neuron: int) -> tuple[np.ndarray, np.ndarray]:
        _, h_in, w_in = self.in_shape
        _, h_out, w_out = self.out_shape
        oc, rem = divmod(neuron, h_out * w_out)
        oh, ow = divmod(rem, w_out)
        k, s = self.kernel_size, self._stride

        kh_idx, kw_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        rows = oh * s + kh_idx.ravel()
        cols = ow * s + kw_idx.ravel()
        parent = oc * (h_in * w_in) + rows * w_in + cols
        # No weights: the companion array is empty but kept for a uniform signature.
        return parent.astype(np.int64), np.empty(0, dtype=np.int64)

    def local_value(
        self,
        neuron: int,
        parent_values: np.ndarray,
        weights: np.ndarray | None,
        bias: float | None,
    ) -> float:
        return float(np.max(parent_values))

    def forward_batch(
        self, inputs: np.ndarray, weight: np.ndarray | None, bias: np.ndarray | None
    ) -> np.ndarray:
        batch = inputs.shape[0]
        c_in, h_in, w_in = self.in_shape
        _, h_out, w_out = self.out_shape
        k, s = self.kernel_size, self._stride
        x = inputs.reshape(batch, c_in, h_in, w_in)
        windows = np.stack(
            [
                x[:, :, kh : kh + s * h_out : s, kw : kw + s * w_out : s]
                for kh in range(k)
                for kw in range(k)
            ],
            axis=0,
        )
        return np.asarray(windows.max(axis=0), dtype=np.float32).reshape(batch, -1)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Architecture:
    """An ordered stack of layers, starting with an :class:`InputLayer`.

    The architecture is *public*: the paper is explicit that the wiring, the layer
    count and the layer each neuron belongs to are known to both parties, and that
    only the weights and activations are committed (Section 5.1).  Consequently
    both sides agree on what "sample a random parent of neuron ``j``" means, which
    is what lets the verifier reconstruct the prover's path from the challenge
    alone.
    """

    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("an architecture needs at least an input layer")
        if not isinstance(self.layers[0], InputLayer):
            raise TypeError("the first layer must be an InputLayer")
        if any(isinstance(layer, InputLayer) for layer in self.layers[1:]):
            raise TypeError("only the first layer may be an InputLayer")

        for previous, layer in zip(self.layers, self.layers[1:]):
            declared = getattr(layer, "in_features", None)
            if declared is not None and declared != previous.n_neurons:
                raise ValueError(
                    f"{layer.name}: expects {declared} inputs but "
                    f"{previous.name} produces {previous.n_neurons}"
                )
            declared_shape = getattr(layer, "in_shape", None)
            if declared_shape is not None and tuple(declared_shape) != previous.out_shape:
                raise ValueError(
                    f"{layer.name}: expects input shape {tuple(declared_shape)} but "
                    f"{previous.name} produces {previous.out_shape}"
                )

    # -- convenience -------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    @property
    def depth(self) -> int:
        """Number of *checkable* layers, i.e. everything above the input layer."""
        return len(self.layers) - 1

    @property
    def input_layer(self) -> InputLayer:
        first = self.layers[0]
        assert isinstance(first, InputLayer)
        return first

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    @property
    def n_outputs(self) -> int:
        return self.output_layer.n_neurons

    @property
    def layer_widths(self) -> tuple[int, ...]:
        """Neuron count per layer, input layer first."""
        return tuple(layer.n_neurons for layer in self.layers)

    @property
    def n_trace_entries(self) -> int:
        """Total number of activations in one execution trace."""
        return sum(self.layer_widths)

    def parameterised_layers(self) -> tuple[tuple[int, Layer], ...]:
        """``(index, layer)`` for every layer that owns weights."""
        return tuple(
            (i, layer)
            for i, layer in enumerate(self.layers)
            if layer.n_weight_groups > 0
        )

    def describe(self) -> str:
        lines = [f"Architecture: {self.depth} checkable layers, "
                 f"{self.n_trace_entries} trace entries"]
        for i, layer in enumerate(self.layers):
            lines.append(
                f"  [{i}] {layer.name:<12} {type(layer).__name__:<16} "
                f"shape={layer.out_shape} neurons={layer.n_neurons}"
            )
        return "\n".join(lines)
