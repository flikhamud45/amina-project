"""Sensitivity of the output to an internal activation.

Two very different parts of this study need the same quantity ``d y / d a_v``:

* the **attack**, to decide which neuron is cheapest to tamper with, and
* the **defence**, because the natural reading of the paper's suggestion to
  "prioritize layers or nodes where activations are statistically more sensitive to
  tampering" (Section 5.2) is to sample neurons in proportion to their influence.

That the attacker and the defender compute the *same* quantity is not a coincidence
in the implementation -- it is the crux of the defence analysis, and putting both on
one code path makes the symmetry impossible to lose sight of.

Gradients are evaluated at a given trace, so the ReLU derivative mask comes from
that trace's activations. When the trace is adversarial, so is the mask. That is
the opening the defence analysis exploits.

Scope: fully connected networks. Both of the paper's own gradient-based attacks
(Section 7.2 and Appendix E) are likewise run on a "simple ANN", and every
gradient-based experiment here uses the MLP setups.
"""

from __future__ import annotations

import numpy as np

from pvi.nn.architecture import DenseLayer
from pvi.nn.network import Trace, TracedNetwork

__all__ = ["output_direction_gradient", "requires_dense_network", "saliency"]


def requires_dense_network(network: TracedNetwork, what: str) -> None:
    """Raise unless every non-input layer is fully connected."""
    for layer in network.architecture.layers[1:]:
        if not isinstance(layer, DenseLayer):
            raise TypeError(
                f"{what} is implemented for fully connected networks; "
                f"layer {layer.name!r} is a {type(layer).__name__}"
            )


def output_direction_gradient(
    network: TracedNetwork,
    trace: Trace,
    layer_index: int,
    direction: np.ndarray,
) -> np.ndarray:
    """Gradient of ``direction . output`` with respect to layer ``layer_index``.

    Back-propagates ``direction`` from the output layer down to ``layer_index``,
    using the ReLU derivative mask implied by ``trace``.  With
    ``direction = e_target - e_current`` the result says, per neuron, how quickly
    raising that neuron's activation closes the margin between the two classes --
    which is exactly the ranking an attacker wants.

    The mask is read from ``trace``, not recomputed from the query.  For an honest
    trace those agree; for a forged one they need not, and a defender that trusts
    the claimed trace inherits whatever mask the prover chose to present.
    """
    requires_dense_network(network, "output_direction_gradient")
    architecture = network.architecture
    if not 0 <= layer_index < len(architecture):
        raise IndexError(f"layer {layer_index} out of range")

    grad = np.asarray(direction, dtype=np.float64)
    if grad.shape != (architecture.output_layer.n_neurons,):
        raise ValueError(
            f"direction must have shape {(architecture.output_layer.n_neurons,)}, "
            f"got {grad.shape}"
        )

    for current in range(len(architecture) - 1, layer_index, -1):
        layer = architecture[current]
        assert isinstance(layer, DenseLayer)
        if layer.activation == "relu":
            # d/dz max(0, z) = 1[z > 0], and a = max(0, z), so a > 0 iff z > 0.
            grad = grad * (trace[current] > 0.0)
        weight, _ = network.parameters[layer.name]
        grad = grad @ weight.astype(np.float64)
    return grad


def saliency(
    network: TracedNetwork,
    trace: Trace,
    layer_index: int,
    *,
    target_class: int | None = None,
) -> np.ndarray:
    """Per-neuron influence on the decision, as a non-negative score.

    With no ``target_class`` this is the gradient of the winning logit's margin over
    the runner-up; with one, the margin of that class over the current winner.  Used
    by the attacker to rank candidate neurons and by the influence-weighted sampler
    to decide where to look.
    """
    logits = trace.output
    winner = int(np.argmax(logits))
    if target_class is None:
        ordered = np.argsort(logits)[::-1]
        rival = int(ordered[1]) if len(ordered) > 1 else winner
        direction = np.zeros_like(logits, dtype=np.float64)
        direction[winner] = 1.0
        direction[rival] = -1.0
    else:
        direction = np.zeros_like(logits, dtype=np.float64)
        direction[target_class] = 1.0
        direction[winner] = direction[winner] - 1.0
    return np.abs(output_direction_gradient(network, trace, layer_index, direction))
