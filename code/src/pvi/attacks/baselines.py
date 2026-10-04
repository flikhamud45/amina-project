"""The paper's own adversaries, reproduced.

Anchuri et al. test their protocol against four cheating strategies and report that
all of them fail:

* **Gradient-descent reconstruction** (Section 7.2) -- optimise activations so that
  a chosen wrong output is backed by a trace consistent with ``M``'s weights.
* **Inverse transform** (Appendix E) -- back-solve the network layer by layer with a
  pseudo-inverse.
* **Logit swap** (Appendix F) -- swap the highest- and lowest-probability outputs,
  then optimise intermediate activations to match.

Reproducing them serves two purposes.  It confirms our implementation reproduces the
paper's *negative* results and not merely its positive ones.  And it establishes the
baseline against which the attack in :mod:`pvi.attacks.tamper` is measured.

A structural remark that the reproduction makes concrete
--------------------------------------------------------
With the input layer anchored to ``qry`` and every local relation satisfied, the
trace is *uniquely determined*: it is ``EvalTrace(M, qry)``, whose output is
``M(qry)``.  So a trace claiming any other output **must** be locally inconsistent
somewhere.  No amount of optimisation removes that; a forging adversary cannot win
by driving the inconsistency to zero.

What an adversary can control is *where* and *how many* nodes are inconsistent.
``RandPathTest`` does not threshold total separation -- it thresholds the per-node
residual at the handful of nodes a path happens to visit.  The right objective is
therefore to minimise the **support** of the inconsistency, not its magnitude.

Every attack in this module minimises magnitude, which is what the paper measures.
Each therefore ends up with inconsistency spread across most of the trace, and is
detected essentially always.  We record both quantities -- the paper's separation
value and the support -- so the two objectives can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pvi.experiments.analysis import acceptance_probability, inconsistent_nodes
from pvi.nn.architecture import DenseLayer
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.params import ProtocolParams

__all__ = [
    "ForgeryResult",
    "gradient_forge",
    "injection_forge",
    "inverse_transform_forge",
    "logit_swap_target",
]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForgeryResult:
    """Outcome of one forging attempt: how many nodes it leaves inconsistent, and
    how likely the verifier is to accept it."""

    name: str
    n_inconsistent: int
    n_nodes: int
    acceptance_probability: float


def _summarise(
    name: str,
    network: TracedNetwork,
    trace: Trace,
    params: ProtocolParams,
) -> ForgeryResult:
    report = inconsistent_nodes(network, trace, params)
    return ForgeryResult(
        name=name,
        n_inconsistent=report.total,
        n_nodes=int(sum(report.per_layer_widths.values())),
        acceptance_probability=acceptance_probability(network, trace, params),
    )


# --------------------------------------------------------------------------- #
# Choosing a malicious target output
# --------------------------------------------------------------------------- #


def logit_swap_target(honest_output: np.ndarray) -> np.ndarray:
    """Appendix F: swap the highest- and lowest-scoring outputs.

    The adversary's goal here is not a *particular* wrong answer but the maximally
    wrong one, which is the harshest case for the protocol.
    """
    target = np.array(honest_output, dtype=np.float32, copy=True)
    hi, lo = int(target.argmax()), int(target.argmin())
    target[hi], target[lo] = target[lo], target[hi]
    return target


# --------------------------------------------------------------------------- #
# Torch mirror for dense networks
# --------------------------------------------------------------------------- #


def _require_dense(network: TracedNetwork) -> None:
    for layer in network.architecture.layers[1:]:
        if not isinstance(layer, DenseLayer):
            raise TypeError(
                "the gradient-based attacks are implemented for fully connected "
                f"networks, as in the paper's Sections 7.2 and E; got {type(layer).__name__}"
            )


def _torch_params(network: TracedNetwork) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for layer in network.architecture.layers[1:]:
        weight, bias = network.parameters[layer.name]
        out.append(
            (
                torch.from_numpy(np.ascontiguousarray(weight)),
                torch.from_numpy(np.ascontiguousarray(bias)),
            )
        )
    return out


def _apply(layer: DenseLayer, z: torch.Tensor) -> torch.Tensor:
    return torch.relu(z) if layer.activation == "relu" else z


# --------------------------------------------------------------------------- #
# Section 7.2: gradient-descent reconstruction
# --------------------------------------------------------------------------- #


def gradient_forge(
    network: TracedNetwork,
    query: np.ndarray,
    target_output: np.ndarray,
    *,
    steps: int = 4000,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-3,
    params: ProtocolParams | None = None,
) -> ForgeryResult:
    """Optimise every interior activation to make a wrong output look consistent.

    This is deliberately *more* generous to the adversary than the paper's own
    version, which optimises only the input (Section 7.2) or a single injection
    point (Appendix F).  Here the entire interior of the trace is free; the input is
    pinned to ``qry`` because the verifier anchors it, and the output is pinned to
    the adversary's target because that is the point of the attack.

    The objective is the total squared local-consistency violation -- exactly what
    the paper's separation value measures.  If gradient descent cannot drive it
    below the verifier's tolerance, the paper's conclusion is reproduced.
    """
    _require_dense(network)
    protocol = params or ProtocolParams()
    architecture = network.architecture
    layers = architecture.layers[1:]
    torch_params = _torch_params(network)

    honest = network.eval_trace(query)
    fixed_input = torch.from_numpy(np.ascontiguousarray(honest[0]))
    fixed_output = torch.from_numpy(np.ascontiguousarray(target_output, dtype=np.float32))

    # Initialise the free interior activations at the honest trace: the best
    # starting point the adversary has, and the one the paper's setup implies.
    free = [
        torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(honest[i])).clone())
        for i in range(1, len(architecture) - 1)
    ]
    optimiser = torch.optim.Adam(free, lr=learning_rate, weight_decay=weight_decay)

    for _ in range(steps):
        optimiser.zero_grad()
        activations = [fixed_input, *free, fixed_output]
        loss = torch.zeros(())
        for index, layer in enumerate(layers):
            expected = _apply(layer, torch.nn.functional.linear(
                activations[index], *torch_params[index]
            ))
            loss = loss + torch.mean((activations[index + 1] - expected) ** 2)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        forged = Trace(
            tuple(
                [np.ascontiguousarray(honest[0], dtype=np.float32)]
                + [t.detach().numpy().astype(np.float32) for t in free]
                + [np.ascontiguousarray(target_output, dtype=np.float32)]
            )
        )
    return _summarise("gradient reconstruction (Sec. 7.2)", network, forged, protocol)


# --------------------------------------------------------------------------- #
# Appendix F: perturb one layer, then propagate
# --------------------------------------------------------------------------- #


def injection_forge(
    network: TracedNetwork,
    query: np.ndarray,
    target_output: np.ndarray,
    *,
    injection_layer: int = 1,
    steps: int = 4000,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-3,
    params: ProtocolParams | None = None,
) -> ForgeryResult:
    """Appendix F's structure: optimise one layer's activations, propagate honestly.

    The adversary replaces the activations of ``injection_layer`` and then lets the
    real model carry them forward, so every layer *above* the injection point is
    consistent by construction.  All the inconsistency lands in the injected layer.

    This is worth isolating because it is structurally the same manoeuvre as the
    attack in :mod:`pvi.attacks.tamper` -- the difference is only how many neurons
    of the injected layer get moved.  Here gradient descent moves as many as it
    likes; there, exactly one does.
    """
    _require_dense(network)
    protocol = params or ProtocolParams()
    architecture = network.architecture
    if not 1 <= injection_layer < len(architecture) - 1:
        raise ValueError(f"injection layer {injection_layer} out of range")

    layers = architecture.layers[1:]
    torch_params = _torch_params(network)
    honest = network.eval_trace(query)

    injected = torch.nn.Parameter(
        torch.from_numpy(np.ascontiguousarray(honest[injection_layer])).clone()
    )
    optimiser = torch.optim.Adam([injected], lr=learning_rate, weight_decay=weight_decay)
    goal = torch.from_numpy(np.ascontiguousarray(target_output, dtype=np.float32))

    for _ in range(steps):
        optimiser.zero_grad()
        activation = injected
        for index in range(injection_layer, len(layers)):
            activation = _apply(
                layers[index],
                torch.nn.functional.linear(activation, *torch_params[index]),
            )
        loss = torch.mean((activation - goal) ** 2)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        values = injected.detach().numpy().astype(np.float32)
    forged = network.forward_from(
        honest.replace_layer(injection_layer, values), injection_layer
    )
    label = f"logit-swap injection at layer {injection_layer} (App. F)"
    return _summarise(label, network, forged, protocol)


# --------------------------------------------------------------------------- #
# Appendix E: inverse transform
# --------------------------------------------------------------------------- #


def inverse_transform_forge(
    network: TracedNetwork,
    query: np.ndarray,
    target_output: np.ndarray,
    *,
    params: ProtocolParams | None = None,
) -> ForgeryResult:
    """Appendix E: back-solve the network from the target output down to the input.

    Starting at the desired output, invert each layer's affine map to obtain the
    activations below it.  As the paper notes, ReLU destroys information at negative
    pre-activations, so the inversion is only a pseudo-inverse and error accumulates
    downwards.  The input layer is then pinned back to ``qry``, since the verifier
    anchors it -- which is where the accumulated error surfaces.
    """
    _require_dense(network)
    protocol = params or ProtocolParams()
    architecture = network.architecture
    honest = network.eval_trace(query)

    activations: list[np.ndarray] = [np.asarray(target_output, dtype=np.float64)]
    for layer_index in range(len(architecture) - 1, 0, -1):
        layer = architecture[layer_index]
        assert isinstance(layer, DenseLayer)
        weight, bias = network.parameters[layer.name]
        above = activations[0]
        # Undo the activation function.  ReLU is not injective at zero; the
        # standard choice -- and the one the paper's setup implies -- is to take the
        # boundary value, which is the least-norm pre-image.
        pre_activation = above if layer.activation == "identity" else np.maximum(above, 0.0)
        below = np.linalg.pinv(weight.astype(np.float64)) @ (pre_activation - bias.astype(np.float64))
        if layer_index > 1:
            below = np.maximum(below, 0.0)  # the layer below is a ReLU output
        activations.insert(0, below)

    # The input layer is not the adversary's to choose: the verifier anchors it.
    activations[0] = np.asarray(honest[0], dtype=np.float64)
    forged = Trace(tuple(np.ascontiguousarray(a, dtype=np.float32) for a in activations))
    return _summarise("inverse transform (App. E)", network, forged, protocol)
