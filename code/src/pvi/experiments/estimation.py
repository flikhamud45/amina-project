"""Appendix I: estimating the soundness parameters of Theorem 1.

Theorem 1 gives other-model soundness ``eps <= eps_tst + eps_sep`` provided

* the model/query distributions satisfy ``(delta_out, delta_trace, eps_sep)``
  trace separation (Definition 2), and
* ``RandPathTest`` is an ``eps_tst``-good separation test at threshold
  ``delta_trace`` (Definition 3).

Neither parameter is derived analytically in the paper; Appendix I gives an
empirical procedure, reproduced here as :func:`generate_candidates`
(Algorithm 1), :func:`estimate_test_error` (Algorithm 2) and
:func:`select_parameters` (Section I.4).

Reproducing this is what turns "the protocol seems to work" into a quantitative
claim, and it is also what the attack later has to be measured against: the
attack's whole point is that these parameters are estimated on *substitute
models*, and a trace-tampering adversary is not one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pvi.nn.network import Trace, TracedNetwork

__all__ = [
    "ParameterEstimate",
    "SeparationDataset",
    "build_separation_dataset",
    "estimate_test_error",
    "generate_candidates",
    "select_parameters",
    "valid_layers",
]


# --------------------------------------------------------------------------- #
# Section I.1 -- the two metrics
# --------------------------------------------------------------------------- #


def valid_layers(
    js_by_layer: dict[int, float], *, threshold: float = 0.05
) -> tuple[int, ...]:
    """``L_valid`` of Appendix I.1: layers whose JS divergence exceeds ``threshold``.

    The input layer is always excluded -- it is identical for both models by
    construction, being the query itself.
    """
    return tuple(
        layer
        for layer, divergence in sorted(js_by_layer.items())
        if layer > 0 and divergence > threshold
    )


@dataclass(frozen=True)
class SeparationDataset:
    """``D = {(dOut(j), dTrc(j))}`` together with the traces that produced it."""

    d_out: np.ndarray
    d_trace: np.ndarray
    honest_traces: tuple[Trace, ...]
    substitute_traces: tuple[Trace, ...]
    valid_layers: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.d_out)


def build_separation_dataset(
    honest_traces: Sequence[Trace],
    substitute_traces: Sequence[Trace],
    valid: Sequence[int],
) -> SeparationDataset:
    """Compute the two Appendix I.1 metrics for a set of queries.

    ``dOut`` is the Euclidean distance between output (logit) vectors; ``dTrc`` is
    the mean over ``L_valid`` of the per-layer mean absolute activation difference.
    """
    if len(honest_traces) != len(substitute_traces):
        raise ValueError("trace lists must have equal length")
    if not valid:
        raise ValueError("L_valid is empty; no layer passed the JS filter")

    d_out = np.asarray(
        [
            float(np.linalg.norm(h.output.astype(np.float64) - s.output.astype(np.float64)))
            for h, s in zip(honest_traces, substitute_traces)
        ]
    )
    d_trace = np.asarray(
        [
            float(
                np.mean(
                    [
                        np.mean(np.abs(h[layer].astype(np.float64) - s[layer].astype(np.float64)))
                        for layer in valid
                    ]
                )
            )
            for h, s in zip(honest_traces, substitute_traces)
        ]
    )
    return SeparationDataset(
        d_out=d_out,
        d_trace=d_trace,
        honest_traces=tuple(honest_traces),
        substitute_traces=tuple(substitute_traces),
        valid_layers=tuple(valid),
    )


# --------------------------------------------------------------------------- #
# Algorithm 1 -- candidate thresholds
# --------------------------------------------------------------------------- #


def generate_candidates(
    dataset: SeparationDataset,
    *,
    target_eps_sep: float = 0.01,
    noise_floor: float = 1e-4,
    min_subset: int = 20,
) -> tuple[tuple[float, float], ...]:
    """Algorithm 1: candidate ``(delta_out, delta_trace)`` pairs.

    For each threshold ``x`` on the output distance, look at the queries whose
    output distance is at least ``x`` and take the ``eps_sep``-quantile of their
    trace distances.  That quantile is the largest ``delta_trace`` for which
    "output far implies trace far" holds with probability ``1 - eps_sep``.
    """
    order = np.argsort(dataset.d_out)
    d_out_sorted = dataset.d_out[order]

    candidates: list[tuple[float, float]] = []
    for x in np.unique(d_out_sorted):
        subset = dataset.d_trace[dataset.d_out >= x]
        if len(subset) < min_subset:
            break
        quantile = float(np.quantile(subset, target_eps_sep))
        if quantile > noise_floor:
            candidates.append((float(x), quantile))
    return tuple(candidates)


# --------------------------------------------------------------------------- #
# Algorithm 2 -- test error
# --------------------------------------------------------------------------- #


def estimate_test_error(
    dataset: SeparationDataset,
    delta: float,
    accept_fn: Callable[[int, Trace], bool],
    *,
    repetitions: int = 50,
) -> tuple[float, int]:
    """Algorithm 2: estimate ``eps_tst`` at trace threshold ``delta``.

    ``accept_fn(query_index, trace)`` runs one execution of the protocol and
    returns whether the verifier accepted.  Averaging the per-query false-negative
    rate over all queries whose trace distance is at least ``delta`` gives the
    estimate.

    Returns ``(eps_hat, n_considered)``.  A count of zero means no query in the
    dataset separated by ``delta``, and the estimate is undefined rather than zero.
    """
    total, considered = 0.0, 0
    for index, distance in enumerate(dataset.d_trace):
        if distance < delta:
            continue
        considered += 1
        failures = sum(
            1
            for _ in range(repetitions)
            if accept_fn(index, dataset.substitute_traces[index])
        )
        total += failures / repetitions
    if considered == 0:
        return float("nan"), 0
    return total / considered, considered


# --------------------------------------------------------------------------- #
# Section I.4 -- final selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParameterEstimate:
    """The parameters Theorem 1 needs, plus the resulting soundness bound."""

    delta_out: float
    delta_trace: float
    eps_sep: float
    eps_tst: float
    n_considered: int

    @property
    def soundness_error(self) -> float:
        """``eps <= eps_tst + eps_sep`` (Theorem 1)."""
        return self.eps_tst + self.eps_sep

    def __str__(self) -> str:
        return (
            f"delta_out={self.delta_out:.4f} delta_trace={self.delta_trace:.6f} "
            f"eps_sep={self.eps_sep:.4f} eps_tst={self.eps_tst:.4f} "
            f"=> soundness error <= {self.soundness_error:.4f} "
            f"(over {self.n_considered} queries)"
        )


def select_parameters(
    dataset: SeparationDataset,
    accept_fn: Callable[[int, Trace], bool],
    *,
    target_eps_sep: float = 0.01,
    target_eps_tst: float = 0.05,
    noise_floor: float = 1e-4,
    repetitions: int = 50,
) -> ParameterEstimate | None:
    """Section I.4: walk the candidate list and take the first that meets the target.

    Returns ``None`` when no candidate achieves ``eps_tst <= target_eps_tst``,
    which is itself an informative outcome -- it says the test cannot certify the
    model family at the requested error.
    """
    candidates = generate_candidates(
        dataset, target_eps_sep=target_eps_sep, noise_floor=noise_floor
    )
    for delta_out, delta_trace in candidates:
        eps_tst, considered = estimate_test_error(
            dataset, delta_trace, accept_fn, repetitions=repetitions
        )
        if considered and eps_tst <= target_eps_tst:
            return ParameterEstimate(
                delta_out=delta_out,
                delta_trace=delta_trace,
                eps_sep=target_eps_sep,
                eps_tst=eps_tst,
                n_considered=considered,
            )
    return None
