"""Exact analysis of what ``RandPathTest`` will do to a given trace.

The verifier's accept predicate is the conjunction of local consistency checks at
the nodes a path visits.  So for a *fixed* trace, acceptance is decided entirely by
two things: which nodes are locally inconsistent, and how likely a path is to visit
one of them.  Both are computable exactly, without sampling.

That makes this module the main validation instrument of the study:

* it turns "the protocol accepted 2 of 3000 runs" into a prediction that can be
  checked against theory, and
* it lets the attack be reasoned about analytically -- for a trace inconsistent at
  a single node ``v``, the acceptance probability is exactly ``1 - Pr[path visits v]``,
  with no Monte-Carlo error in the way.

The dynamic program below is exact for any layered architecture, including the
convolutional case where path steps are confined to receptive fields and the visit
distribution is not uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pvi.nn.architecture import DenseLayer
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.params import ProtocolParams

if TYPE_CHECKING:  # pragma: no cover
    from pvi.protocol.sampling import PathSampler

__all__ = [
    "InconsistencyReport",
    "acceptance_probability",
    "inconsistent_nodes",
    "locally_consistent_mask",
]


def locally_consistent_mask(
    network: TracedNetwork,
    trace: Trace,
    params: ProtocolParams | None = None,
) -> dict[int, np.ndarray]:
    """Boolean mask per layer: does each neuron satisfy its local relation?

    Computed with the layer's vectorised forward pass rather than neuron by
    neuron, which is what makes it affordable to run over a full convolutional
    trace.  The verifier's per-neuron recomputation agrees with this to within
    float noise, several orders of magnitude below the tolerance.
    """
    params = params or ProtocolParams()
    architecture = network.architecture
    masks: dict[int, np.ndarray] = {}
    for layer_index in range(1, len(architecture)):
        layer = architecture[layer_index]
        weight, bias = network.parameters.get(layer.name, (None, None))
        recomputed = layer.forward_batch(trace[layer_index - 1][None, :], weight, bias)[0]
        claimed = trace[layer_index]
        masks[layer_index] = np.abs(claimed - recomputed) <= params.abs_tolerance
    return masks


@dataclass(frozen=True)
class InconsistencyReport:
    """Where a trace violates the local relation, and how much it matters."""

    nodes: tuple[tuple[int, int], ...]
    per_layer_widths: dict[int, int]

    @property
    def total(self) -> int:
        return len(self.nodes)


def inconsistent_nodes(
    network: TracedNetwork,
    trace: Trace,
    params: ProtocolParams | None = None,
) -> InconsistencyReport:
    """Every ``(layer, neuron)`` at which ``trace`` violates its local relation."""
    masks = locally_consistent_mask(network, trace, params)
    nodes: list[tuple[int, int]] = []
    widths: dict[int, int] = {}
    for layer_index, mask in masks.items():
        bad = np.flatnonzero(~mask)
        widths[layer_index] = int(mask.size)
        nodes.extend((layer_index, int(neuron)) for neuron in bad)
    return InconsistencyReport(nodes=tuple(nodes), per_layer_widths=widths)


def acceptance_probability(
    network: TracedNetwork,
    trace: Trace,
    params: ProtocolParams | None = None,
    sampler: "PathSampler | None" = None,
) -> float:
    """Exact probability that the verifier accepts ``trace``.

    Let ``c_l(j)`` be 1 when neuron ``j`` of layer ``l`` satisfies its local
    relation, and let ``P_l(j -> i)`` be the sampler's probability of stepping from
    node ``j`` to parent ``i``.  Write ``g_l(j)`` for the probability that a path
    sitting at node ``j`` in layer ``l`` passes every remaining check on its way down
    to the input.  Then

        ``g_0(i) = 1``                              (the input layer is anchored)
        ``g_l(j) = c_l(j) * sum_{i in G_j} P_l(j -> i) g_{l-1}(i)``

    and, writing ``pi`` for the sampler's distribution over starting nodes,

        ``Pr[accept] = sum_j pi(j) g_L(j)``.

    With ``n_paths > 1`` the paths are drawn independently, so the result is raised
    to that power.

    The recursion is exact for any layered architecture and any sampler.  It does
    *not* assume the per-layer visit distribution is uniform -- false for
    convolutions and deliberately false for the importance-weighted samplers in
    ``pvi.defences`` -- and it does not assume checks at different layers are
    independent, which they are not once parent sets are local.
    """
    from pvi.protocol.sampling import UniformPathSampler, _as_distribution

    params = params or ProtocolParams()
    sampler = sampler or UniformPathSampler()
    architecture = network.architecture
    masks = locally_consistent_mask(network, trace, params)

    # g for the input layer: anchored to the query, so nothing left to check.
    g = np.ones(architecture[0].n_neurons, dtype=np.float64)

    for layer_index in range(1, len(architecture)):
        layer = architecture[layer_index]
        consistent = masks[layer_index]
        below = g

        # Fast path: for a dense layer every neuron shares the same parent set, and
        # for samplers that score the layer below without reference to which node is
        # asking, every neuron also shares the same transition distribution.  The
        # inner sum is then one dot product for the whole layer rather than one per
        # neuron -- which is what makes the adaptive-adversary search tractable.
        dense = isinstance(layer, DenseLayer)
        if dense and not sampler.transition_depends_on_source:
            parent_idx = np.arange(architecture[layer_index - 1].n_neurons)
            weights = _as_distribution(
                sampler.transition_distribution(
                    architecture, layer_index, 0, parent_idx, trace
                )
            )
            g = consistent.astype(np.float64) * float(np.dot(weights, below))
            continue

        current = np.zeros(layer.n_neurons, dtype=np.float64)
        for neuron in range(layer.n_neurons):
            if not consistent[neuron]:
                continue
            parent_idx, _ = layer.parents(neuron)
            weights = _as_distribution(
                sampler.transition_distribution(
                    architecture, layer_index, neuron, parent_idx, trace
                )
            )
            current[neuron] = float(np.dot(weights, below[parent_idx]))
        g = current

    start = _as_distribution(sampler.start_distribution(architecture, trace))
    single_path = float(np.dot(start, g))
    return single_path**params.n_paths
