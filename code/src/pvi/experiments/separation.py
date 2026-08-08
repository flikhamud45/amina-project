"""Trace-separation metrics (Section 6 and Appendix I of Anchuri et al.).

Three related quantities appear in the paper, and it is worth keeping them apart
because they answer different questions.

**Separation value, Equation (1).**

    ``| phi( (a^{l-1}_M)^T W^l_M ) - phi( (a^{l-1}_Mtilde)^T W^l_M ) |``

Both terms use the *verified* model's weights ``W^l_M``; what differs is whose
layer-``l-1`` activations are fed in.  It measures how far layer ``l`` moves when
the substitute's activations are spliced into the honest model -- the paper's
"replace the activations of ``M`` along a random path with those from ``M~``"
experiment (Section 6.2.2).  This is the quantity the paper reports.

**Verifier residual.**

    ``| a~_j - phi( sum_{i in G_j} w_ij a~_i ) |``

This is what ``RandPathTest`` actually computes, and it is what decides
acceptance.  For a substitute model's honest trace the two quantities are closely
related but not identical, because the claimed ``a~_j`` was produced by ``M~``'s
weights rather than ``M``'s.  We report it alongside Equation (1) since detection
-- not separation -- is the property that matters.

**Jensen-Shannon divergence.**  Per-layer divergence between the two models'
activation distributions, as used for the layer filtering of Appendix I.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.sampling import Path

__all__ = [
    "LayerSeparation",
    "equation1_separation",
    "js_divergence",
    "layer_js_divergences",
    "path_separation",
    "verifier_residuals",
]


@dataclass(frozen=True)
class LayerSeparation:
    """Per-layer summary of one honest/substitute trace pair."""

    layer_index: int
    layer_name: str
    mean: float
    minimum: float
    maximum: float
    median: float

    @classmethod
    def summarise(
        cls, layer_index: int, layer_name: str, values: np.ndarray
    ) -> "LayerSeparation":
        return cls(
            layer_index=layer_index,
            layer_name=layer_name,
            mean=float(np.mean(values)),
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            median=float(np.median(values)),
        )


# --------------------------------------------------------------------------- #
# Equation (1)
# --------------------------------------------------------------------------- #


def equation1_separation(
    network: TracedNetwork,
    honest: Trace,
    substitute: Trace,
    *,
    layer_index: int,
    neurons: np.ndarray | None = None,
) -> np.ndarray:
    """Equation (1) for one layer, per neuron.

    ``honest`` is ``EvalTrace(M, qry)`` and ``substitute`` is the trace whose
    layer-``l-1`` activations get spliced in.  Both forward passes use ``M``'s
    weights, so the first term is simply ``honest[layer_index]``.
    """
    architecture = network.architecture
    layer = architecture[layer_index]
    weight, bias = network.parameters.get(layer.name, (None, None))

    spliced = layer.forward_batch(substitute[layer_index - 1][None, :], weight, bias)[0]
    values = np.abs(honest[layer_index] - spliced)
    return values if neurons is None else values[neurons]


def path_separation(
    network: TracedNetwork,
    honest: Trace,
    substitute: Trace,
    path: Path,
) -> np.ndarray:
    """Equation (1) restricted to the neurons a single path visits.

    This is the paper's per-path separation value: one number per checked layer,
    aggregated by mean in their figures.
    """
    return np.asarray(
        [
            float(
                equation1_separation(
                    network,
                    honest,
                    substitute,
                    layer_index=layer_index,
                    neurons=np.asarray([path[layer_index]]),
                )[0]
            )
            for layer_index in path.checked_layers
        ]
    )


# --------------------------------------------------------------------------- #
# Verifier residuals
# --------------------------------------------------------------------------- #


def verifier_residuals(
    network: TracedNetwork,
    trace: Trace,
    *,
    layer_index: int | None = None,
) -> dict[int, np.ndarray]:
    """``|a~_j - phi(sum w_ij a~_i)|`` for every neuron, per layer.

    With ``trace`` the honest trace of ``network`` this quantifies pure
    floating-point noise, which is what justifies the verifier's tolerance.  With a
    foreign trace it is the signal ``RandPathTest`` is looking for.
    """
    architecture = network.architecture
    targets = (
        range(1, len(architecture)) if layer_index is None else [layer_index]
    )

    out: dict[int, np.ndarray] = {}
    for index in targets:
        layer = architecture[index]
        weight, bias = network.parameters.get(layer.name, (None, None))
        recomputed = layer.forward_batch(trace[index - 1][None, :], weight, bias)[0]
        out[index] = np.abs(trace[index] - recomputed)
    return out


# --------------------------------------------------------------------------- #
# Jensen-Shannon divergence
# --------------------------------------------------------------------------- #


def js_divergence(
    left: np.ndarray, right: np.ndarray, *, bins: int = 64
) -> float:
    """Jensen-Shannon divergence between two empirical activation distributions.

    Computed on a shared histogram support, base-2, so the result lies in
    ``[0, 1]`` -- the convention the paper's reported values (0.0 to 0.719) follow.
    """
    left, right = np.asarray(left).ravel(), np.asarray(right).ravel()
    if left.size == 0 or right.size == 0:
        return 0.0

    low = float(min(left.min(), right.min()))
    high = float(max(left.max(), right.max()))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0

    edges = np.linspace(low, high, bins + 1)
    p = np.histogram(left, bins=edges)[0].astype(np.float64)
    q = np.histogram(right, bins=edges)[0].astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def layer_js_divergences(
    honest_traces: list[Trace],
    substitute_traces: list[Trace],
    *,
    bins: int = 64,
) -> dict[int, float]:
    """Per-layer JS divergence, pooled over many queries.

    Appendix I.1 uses this to select the "valid" layers ``L_valid`` -- those with
    ``JS > 0.05`` -- before estimating separation thresholds.
    """
    if not honest_traces:
        return {}
    n_layers = len(honest_traces[0])
    out: dict[int, float] = {}
    for layer_index in range(n_layers):
        left = np.concatenate([t[layer_index] for t in honest_traces])
        right = np.concatenate([t[layer_index] for t in substitute_traces])
        out[layer_index] = js_divergence(left, right, bins=bins)
    return out
