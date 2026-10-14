"""Non-uniform path sampling: the mitigation Anchuri et al. leave open.

Their Section 5.2 suggests that "adaptive sampling strategies that prioritize layers
or nodes where activations are statistically more sensitive to tampering" might
tighten the ``1/N`` bound, and reports "some partial results from this approach".
This module builds concrete readings of that suggestion so they can be measured
rather than speculated about (the paper's Sections ``sec:sampling`` and
``sec:break``, "Other samplers").

The first four differ in what information the sampling weights are allowed to
depend on, and that decides how they fare:

:class:`StaticImportanceSampler`
    Weights published alongside ``C_M``, derived from the model only.  Implementable,
    and verifiable by anyone holding the model.  But *public and fixed*, so an
    adversary reads them off and tampers wherever the weight is lowest.

:class:`LocalContributionSampler`
    Weight each parent ``i`` of node ``j`` by ``|w_ij * a~_i|`` -- its actual
    contribution to ``j``'s value.  This is the only proposal here a real verifier
    can compute: at each step it already holds the opened weight row and the parent
    activations, and needs nothing else.  It is very effective against a tamper that
    sets a large value, because a large value *is* a large contribution.

:class:`ZeroAwareContributionSampler`
    The same with an additive floor, ``|w_ij| * (|a~_i| + eps)``, so that a claimed
    zero is still visited.

:class:`GradientSaliencySampler`
    Weight neuron ``i`` by ``|d y / d a_i|`` at the claimed trace -- the most direct
    reading of "how much it influences the final value".

A caveat that decides the matter for the last one: **a real verifier cannot compute
it.**  Obtaining ``d y / d a_i`` for a whole layer requires every weight of every
layer above.  The verifier holds a digest of the model and opens a handful of rows;
if it could form layer-wide gradients it would already hold the model, and could
simply run the inference itself.  It is implemented anyway, as an upper bound on
what influence-weighting could achieve even given information no verifier has.

Two more samplers read the claimed values against their natural range:
:class:`TailFirstSampler` (values in the upper tail of the neuron's held-out range)
and :class:`ZeroFirstSampler` (claimed zeros).

The implementable samplers read the claimed trace, which is authored by the
prover; :mod:`pvi.baseline.adaptive` builds the adversary's reply that exploits it.
"""

from __future__ import annotations

import numpy as np

from pvi.baseline.path.sampling import PathSampler
from pvi.mnist.architecture import Architecture, DenseLayer
from pvi.mnist.gradients import output_direction_gradient
from pvi.mnist.network import Trace, TracedNetwork

__all__ = [
    "GradientSaliencySampler",
    "LocalContributionSampler",
    "StaticImportanceSampler",
    "TailFirstSampler",
    "ZeroAwareContributionSampler",
    "ZeroFirstSampler",
    "weight_magnitude_importance",
]


# --------------------------------------------------------------------------- #
# Published importance measures
# --------------------------------------------------------------------------- #


def weight_magnitude_importance(
    network: TracedNetwork, layer_index: int
) -> np.ndarray:
    """Importance of each neuron as the total magnitude of its outgoing weights.

    Depends only on the committed model, so it can be published with ``C_M`` and
    recomputed by anyone.  It ignores the input entirely, which is what makes it
    implementable -- and also what makes it predictable to an adversary.
    """
    above = network.architecture[layer_index + 1]
    if not isinstance(above, DenseLayer):
        raise TypeError(
            f"weight-magnitude importance needs a dense layer above; "
            f"{above.name} is a {type(above).__name__}"
        )
    weight, _ = network.parameters[above.name]
    return np.abs(weight.astype(np.float64)).sum(axis=0)


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #


class StaticImportanceSampler(PathSampler):
    """Sample proportionally to a fixed, published per-neuron importance.

    Weights are supplied per layer and never depend on the trace, so this sampler is
    in the same class as uniform: the set of checked nodes is independent of what the
    prover submitted.  In that class an adversary who knows the weights edits the
    least-visited neuron, and since a path's visit probabilities over a dense layer
    sum to one, no member of the class beats uniform's ``1/N`` against it.
    """

    transition_depends_on_source = False

    def __init__(self, importance: dict[int, np.ndarray]) -> None:
        self._importance = {
            layer: np.asarray(values, dtype=np.float64) for layer, values in importance.items()
        }

    def _weights(self, layer_index: int, size: int) -> np.ndarray:
        raw = self._importance.get(layer_index)
        if raw is None:
            return np.full(size, 1.0 / size)
        scores = np.maximum(raw, 0.0)
        total = scores.sum()
        return np.full(size, 1.0 / size) if total <= 0 else scores / total

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        return self._weights(len(architecture) - 1, architecture.output_layer.n_neurons)

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        below = layer_index - 1
        full = self._weights(below, architecture[below].n_neurons)
        return full[parent_indices]


class LocalContributionSampler(PathSampler):
    """Step to a parent with probability proportional to ``|w_ij * a~_i|``.

    The one proposal here that a real verifier can actually run: at the step from
    node ``j``, it already holds ``j``'s opened weight row and the claimed parent
    activations, and needs nothing further.

    It is strong against a tamper that writes a conspicuously large activation --
    such a value dominates the contribution sum and pulls the walk straight onto it.
    It is blind to a claimed activation of exactly zero, which gets transition weight
    zero; :func:`pvi.baseline.adaptive.plan_adaptive_stealthy_flip` exploits this
    and reaches detection 0.
    """

    def __init__(self, network: TracedNetwork) -> None:
        self._network = network

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        n = architecture.output_layer.n_neurons
        if trace is None:
            return np.full(n, 1.0 / n)
        # Start where the claimed output is most extreme: the winning logit is the
        # claim the verifier most wants to hold the prover to.
        scores = np.abs(trace.output.astype(np.float64))
        total = scores.sum()
        return np.full(n, 1.0 / n) if total <= 0 else scores / total

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        size = len(parent_indices)
        if trace is None:
            return np.full(size, 1.0 / size)
        layer = architecture[layer_index]
        row = self._network.weight_row(layer_index, neuron)
        _, weight_idx = layer.parents(neuron)
        parent_values = trace[layer_index - 1][parent_indices].astype(np.float64)
        if row is None:
            scores = np.abs(parent_values)
        else:
            scores = np.abs(row[weight_idx].astype(np.float64) * parent_values)
        total = scores.sum()
        return np.full(size, 1.0 / size) if total <= 0 else scores / total


class ZeroAwareContributionSampler(PathSampler):
    """``LocalContributionSampler`` with an additive floor on the activation term.

    The natural patch for the zero blind spot: score a parent ``i`` by
    ``|w_ij| * (|a~_i| + epsilon)`` instead of ``|w_ij * a~_i|``, so a claimed value
    of exactly zero no longer sends the transition weight to exactly zero.

    Measured on tampered traces of the MNIST MLP (``experiments/attack/floor_sampler.py``;
    the paper's Section ``sec:break``), ``epsilon = 1`` detects about 10x as often as
    uniform sampling -- but detection stays at about 2%, far from what a proof needs.
    """

    def __init__(self, network: TracedNetwork, *, epsilon: float) -> None:
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        self._network = network
        self._epsilon = epsilon

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        n = architecture.output_layer.n_neurons
        if trace is None:
            return np.full(n, 1.0 / n)
        scores = np.abs(trace.output.astype(np.float64)) + self._epsilon
        total = scores.sum()
        return np.full(n, 1.0 / n) if total <= 0 else scores / total

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        size = len(parent_indices)
        if trace is None:
            return np.full(size, 1.0 / size)
        layer = architecture[layer_index]
        row = self._network.weight_row(layer_index, neuron)
        _, weight_idx = layer.parents(neuron)
        parent_values = trace[layer_index - 1][parent_indices].astype(np.float64)
        if row is None:
            scores = np.abs(parent_values) + self._epsilon
        else:
            scores = np.abs(row[weight_idx].astype(np.float64)) * (
                np.abs(parent_values) + self._epsilon
            )
        total = scores.sum()
        return np.full(size, 1.0 / size) if total <= 0 else scores / total


class GradientSaliencySampler(PathSampler):
    """Sample proportionally to ``|d y / d a_i|`` at the *claimed* trace.

    The most literal reading of "prioritise nodes the output is most sensitive to".
    Not implementable by a real verifier -- forming these gradients needs the whole
    model -- so treat its numbers as an optimistic bound on the whole family.

    Gradients are cached per trace: the walk revisits the same layer many times
    across repeated challenges, and a backward pass per step would dominate runtime.
    The cache keeps a reference to each trace and checks it on lookup: keying on
    ``id(trace)`` alone would return another trace's gradients once a trace is
    freed and its id reused.
    """

    transition_depends_on_source = False

    def __init__(self, network: TracedNetwork) -> None:
        self._network = network
        self._cache: dict[tuple[int, int], tuple[Trace, np.ndarray]] = {}

    def _saliency(self, trace: Trace, layer_index: int) -> np.ndarray:
        key = (id(trace), layer_index)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is trace:
            return cached[1]

        logits = trace.output.astype(np.float64)
        winner = int(np.argmax(logits))
        rival = int(np.argsort(logits)[::-1][1]) if len(logits) > 1 else winner
        direction = np.zeros_like(logits)
        direction[winner] = 1.0
        direction[rival] -= 1.0
        scores = np.abs(
            output_direction_gradient(self._network, trace, layer_index, direction)
        )
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = (trace, scores)
        return scores

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        n = architecture.output_layer.n_neurons
        if trace is None:
            return np.full(n, 1.0 / n)
        scores = np.abs(trace.output.astype(np.float64))
        total = scores.sum()
        return np.full(n, 1.0 / n) if total <= 0 else scores / total

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        size = len(parent_indices)
        if trace is None:
            return np.full(size, 1.0 / size)
        scores = self._saliency(trace, layer_index - 1)[parent_indices]
        total = scores.sum()
        return np.full(size, 1.0 / size) if total <= 0 else scores / total


class TailFirstSampler(PathSampler):
    """Step uniformly among parents whose claimed value lies in the upper tail of its natural range.

    ``thresholds[l]`` holds, per neuron of layer ``l``, a percentile of that neuron's activation on
    held-out natural inputs.  A step into layer ``l`` picks uniformly among the parents ``i`` with
    ``a~_i >= thresholds[l][i]`` and ``a~_i > 0``, and uniformly among all parents when there is
    none; layers without thresholds (the input, which is anchored to the query) are stepped into
    uniformly, and the walk starts uniformly.  The condition ``a~_i > 0`` keeps a claimed zero of a
    neuron whose threshold is zero (a dead neuron) out of the tail, where it would only dilute it.

    The rule targets an edit that pushes values to the top of their range (the stealthy planner of
    :mod:`pvi.attacks.tamper`); an edit that lowers values towards their mode is invisible to it.
    The choice depends only on the claimed values of the layer below, not on the source node.
    """

    transition_depends_on_source = False

    def __init__(self, thresholds: dict[int, np.ndarray]) -> None:
        self._thresholds = {
            layer: np.asarray(values, dtype=np.float64) for layer, values in thresholds.items()
        }

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        n = architecture.output_layer.n_neurons
        return np.full(n, 1.0 / n)

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        size = len(parent_indices)
        threshold = self._thresholds.get(layer_index - 1)
        if trace is None or threshold is None:
            return np.full(size, 1.0 / size)
        values = trace[layer_index - 1][parent_indices].astype(np.float64)
        tail = (values >= threshold[parent_indices]) & (values > 0.0)
        count = int(tail.sum())
        return np.full(size, 1.0 / size) if count == 0 else tail / count


class ZeroFirstSampler(PathSampler):
    """Step uniformly among parents whose claimed value is exactly zero.

    The counterpart of :class:`TailFirstSampler` for the edit that lowers activations to their
    mode, the generic reply to contribution weighting (a zeroed neuron has contribution weight
    zero).  A step into one of ``layers`` picks uniformly among the parents with ``a~_i == 0``, and
    uniformly among all parents when there is none; other layers are stepped into uniformly, and
    the walk starts uniformly.  It never opens a non-zero edit preferentially.
    """

    transition_depends_on_source = False

    def __init__(self, layers: set[int] | frozenset[int]) -> None:
        self._layers = frozenset(int(layer) for layer in layers)

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        n = architecture.output_layer.n_neurons
        return np.full(n, 1.0 / n)

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        size = len(parent_indices)
        if trace is None or (layer_index - 1) not in self._layers:
            return np.full(size, 1.0 / size)
        zeros = trace[layer_index - 1][parent_indices] == 0.0
        count = int(zeros.sum())
        return np.full(size, 1.0 / size) if count == 0 else zeros / count
