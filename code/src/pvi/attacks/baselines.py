"""The paper's own adversaries, reproduced.

Anchuri et al. test their protocol against four cheating strategies and report that
all of them fail:

* **RandTestStrawman** (Section 2) -- check one uniformly random node instead of a
  whole path.  The authors reject this design themselves, on the grounds that
  "discrepancies in the activations typically manifest only in late layers", which
  makes a random early-layer check pass even when the output is wrong.
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
from typing import Literal

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
    "strawman_detection_probability",
    "substitute_model_target",
]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForgeryResult:
    """Outcome of one forging attempt.

    ``mean_separation`` and ``max_residual`` are the paper's view: how far the
    forged trace is from satisfying the local relations.  ``n_inconsistent`` is the
    view that actually predicts detection: how many nodes violate them at all.
    """

    name: str
    trace: Trace
    target_output: np.ndarray
    achieved_output: np.ndarray
    mean_separation: float
    max_residual: float
    n_inconsistent: int
    n_nodes: int
    acceptance_probability: float
    output_matches_target: bool
    output_differs_from_honest: bool

    @property
    def support_fraction(self) -> float:
        return self.n_inconsistent / self.n_nodes

    def __str__(self) -> str:
        return (
            f"{self.name}: acceptance {self.acceptance_probability:.2e}, "
            f"{self.n_inconsistent}/{self.n_nodes} nodes inconsistent "
            f"({self.support_fraction:.1%}), mean separation "
            f"{self.mean_separation:.4f}, target hit={self.output_matches_target}"
        )


def _summarise(
    name: str,
    network: TracedNetwork,
    trace: Trace,
    target_output: np.ndarray,
    honest_output: np.ndarray,
    params: ProtocolParams,
) -> ForgeryResult:
    report = inconsistent_nodes(network, trace, params)
    residuals = []
    architecture = network.architecture
    for layer_index in range(1, len(architecture)):
        layer = architecture[layer_index]
        weight, bias = network.parameters.get(layer.name, (None, None))
        recomputed = layer.forward_batch(trace[layer_index - 1][None, :], weight, bias)[0]
        residuals.append(np.abs(trace[layer_index] - recomputed))
    pooled = np.concatenate(residuals)

    achieved = trace.output
    return ForgeryResult(
        name=name,
        trace=trace,
        target_output=np.asarray(target_output, dtype=np.float32),
        achieved_output=achieved,
        mean_separation=float(pooled.mean()),
        max_residual=float(pooled.max()),
        n_inconsistent=report.total,
        n_nodes=int(sum(report.per_layer_widths.values())),
        acceptance_probability=acceptance_probability(network, trace, params),
        output_matches_target=bool(
            int(achieved.argmax()) == int(np.asarray(target_output).argmax())
        ),
        output_differs_from_honest=bool(
            int(achieved.argmax()) != int(np.asarray(honest_output).argmax())
        ),
    )


# --------------------------------------------------------------------------- #
# Section 2: the strawman test
# --------------------------------------------------------------------------- #


def strawman_detection_probability(
    network: TracedNetwork,
    trace: Trace,
    params: ProtocolParams | None = None,
) -> float:
    """Detection probability of ``RandTestStrawman``: one uniformly random node.

    The paper discards this design because "discrepancies in the activations
    typically manifest only in late layers".  The claim is exactly quantifiable:
    the strawman detects with probability ``(inconsistent nodes) / (total nodes)``,
    and since late layers are also the *narrowest* layers, that ratio is tiny even
    when every late-layer node is wrong.
    """
    params = params or ProtocolParams()
    report = inconsistent_nodes(network, trace, params)
    total = sum(report.per_layer_widths.values())
    return report.total / total if total else 0.0


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


def substitute_model_target(
    substitute: TracedNetwork, query: np.ndarray
) -> np.ndarray:
    """Strong other-model soundness (Appendix H): the output of a substitute model."""
    return substitute.eval_trace(query).output


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
    name: str = "gradient reconstruction (Sec. 7.2)",
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
    return _summarise(name, network, forged, target_output, honest.output, protocol)


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
    name: str | None = None,
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
    label = name or f"logit-swap injection at layer {injection_layer} (App. F)"
    return _summarise(label, network, forged, target_output, honest.output, protocol)


# --------------------------------------------------------------------------- #
# Appendix E: inverse transform
# --------------------------------------------------------------------------- #


def inverse_transform_forge(
    network: TracedNetwork,
    query: np.ndarray,
    target_output: np.ndarray,
    *,
    method: Literal["pinv", "svd", "regularised"] = "pinv",
    ridge: float = 1e-4,
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

    def invert(weight: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        weight = weight.astype(np.float64)
        if method == "pinv":
            return np.linalg.pinv(weight) @ rhs
        if method == "svd":
            u, s, vt = np.linalg.svd(weight, full_matrices=False)
            keep = s > (s.max() * 1e-10 if s.size else 0.0)
            inverse = vt.T[:, keep] @ np.diag(1.0 / s[keep]) @ u.T[keep, :]
            return inverse @ rhs
        if method == "regularised":
            gram = weight.T @ weight + ridge * np.eye(weight.shape[1])
            return np.linalg.solve(gram, weight.T @ rhs)
        raise ValueError(f"unknown inversion method {method!r}")

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
        below = invert(weight, pre_activation - bias.astype(np.float64))
        if layer_index > 1:
            below = np.maximum(below, 0.0)  # the layer below is a ReLU output
        activations.insert(0, below)

    # The input layer is not the adversary's to choose: the verifier anchors it.
    activations[0] = np.asarray(honest[0], dtype=np.float64)
    forged = Trace(tuple(np.ascontiguousarray(a, dtype=np.float32) for a in activations))
    return _summarise(
        f"inverse transform, {method} (App. E)",
        network,
        forged,
        target_output,
        honest.output,
        protocol,
    )
