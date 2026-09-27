"""Non-uniform path sampling: the defence the paper leaves open.

Section 5.2 suggests that "adaptive sampling strategies that prioritize layers or
nodes where activations are statistically more sensitive to tampering" might tighten
the ``1/N`` bound, and reports "some partial results from this approach".  This
module builds three concrete readings of that suggestion so they can be measured
rather than speculated about.

They differ in what information the sampling weights are allowed to depend on, and
that is the whole story:

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

:class:`GradientSaliencySampler`
    Weight neuron ``i`` by ``|d y / d a_i|`` at the claimed trace -- the most direct
    reading of "how much it influences the final value".

A caveat that decides the matter for the third one: **a real verifier cannot compute
it.**  Obtaining ``d y / d a_i`` for a whole layer requires every weight of every
layer above.  The verifier holds a digest of the model and opens a handful of rows;
if it could form layer-wide gradients it would already hold the model, and could
simply run the inference itself.  We implement it anyway, as an upper bound on what
influence-weighting could achieve even given information no verifier has.

The two samplers that *are* implementable both read the claimed trace, which is
authored by the prover; Section 3.2 of the report shows what that costs.
"""

from __future__ import annotations

import numpy as np

from pvi.nn.architecture import Architecture, DenseLayer
from pvi.nn.gradients import output_direction_gradient
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.sampling import PathSampler

__all__ = [
    "GradientSaliencySampler",
    "LocalContributionSampler",
    "StaticImportanceSampler",
    "ZeroAwareContributionSampler",
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
    prover submitted.  That is the class for which the minimax bound (report,
    Theorem 3.1) applies -- and the bound says no member of it beats uniform.
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
    It is weak against a tamper whose values look ordinary, which is exactly what
    ``plan_stealthy_flip`` produces.
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

    Measured on tampered traces (``experiments/3_sampling_fixes/floor_sampler.py``),
    ``epsilon`` around 1 beats uniform sampling by about 10x on average -- but
    detection stays at a few percent, far from what a proof needs.
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
