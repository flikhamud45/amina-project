"""The adaptive game: the best value-aware sampler against range-respecting edits (plan G3-min).

This is the minimal instantiation of the game of ``attack_theory/experiments.md`` Section 1 on the
MNIST MLP.  The defender publishes a sampler, or a mixture of samplers, from a fixed family F; the
adversary sees the model, the query, the sampler and the held-out natural ranges, and returns a
forged trace with a wrong output whose edited values lie in range.  Detection per path is the exact
``1 - acceptance`` of :mod:`pvi.baseline.analysis`; with ``s`` independent paths it is
``1 - (1 - d)^s``.

The reported value for a sampler is the *minimum* over the adversary's found attacks of ``d``.  A
stronger adversary can only lower it, so each number is an **upper bound** on the sampler's true
worst-case detection -- the direction that supports the claim "even the best sampler detects at most
X".

Two things in the earlier scripts are fixed here, as the design requires:

* **Held-out ranges.**  ``2_attack/run.py`` and ``3_sampling_fixes/run.py`` set ceilings from
  ``test_x[:3000]`` and attack ``test_x[0:60]``, which overlap.  Here ranges are calibrated on the
  *train* split and attacks run on disjoint *test* queries.
* **Two range levels.**  R1 reports that every *edited* value lies in range; R2 reports that every
  value in the whole recomputed trace lies in range (values pushed downstream can leave their own
  range).

The acceptance DP is evaluated by :func:`fast_acceptance`, a vectorised, matrix-form equivalent of
``acceptance_probability`` validated against it to within ``1e-9`` (kill criterion K-G3c) by
``tests/test_adaptive_game.py`` before any number here is used.

Efficiency.  The single-neuron pool (A1) and the zeroing plan (A3) do not depend on the cap, and
their acceptance vectors over the family do not depend on the cap either, so both are computed once
per query and reused across every cap; A1 feasibility at a cap is then a ceiling filter on the
already-found flipping value.  The ceiling plan (A2) and the exact-detection greedy (A4) are the
only cap- or sampler-dependent searches.  That keeps the full MNIST sweep inside the time budget.

Usage
-----
    cd code && PYTHONPATH=src python experiments/3_sampling_fixes/adaptive_game.py \
        --model mlp_mnist --layers 1 2 --queries 40 --calib train \
        --out artifacts/results/adaptive_game_mlp_mnist.json
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from pvi.attacks.tamper import smallest_flipping_value
from pvi.baseline.samplers import (GradientSaliencySampler, LocalContributionSampler, StaticImportanceSampler,
                                   TailFirstSampler, ZeroAwareContributionSampler, ZeroFirstSampler,
                                   weight_magnitude_importance)
from pvi.baseline.analysis import locally_consistent_mask
from pvi.mnist.architecture import DenseLayer
from pvi.mnist.gradients import output_direction_gradient, saliency
from pvi.mnist.network import Trace, TracedNetwork
from pvi.baseline.path.params import ProtocolParams
from pvi.baseline.path.sampling import UniformPathSampler
from pvi.baseline.path.sampling import PathSampler, _as_distribution
from pvi.results import write_json
from pvi.mnist.training import load_network
from pvi.mnist.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

#: Caps as percentiles of the held-out natural range; ``None`` is the unconstrained cap.
CAPS: tuple[float | None, ...] = (50.0, 75.0, 90.0, 95.0, 99.0, 100.0, None)
#: Floor magnitudes of the zero-aware contribution sampler (the family of ``floor_sampler.py``).
FLOOR_EPSILONS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
#: Support cap K for the multi-neuron adversaries.
KMAX = 48
#: Support limits K at which the adversary's best is reported (``k <= K``).
K_LIMITS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48)
#: Path counts at which detection is reported in the figures.
PATH_COUNTS: tuple[int, ...] = (1, 10, 100, 1000)

#: Cache of per-layer absolute weight magnitudes (float64), which are constant for a run; keyed by
#: ``(id(network), layer_index)``.  The contribution-type transition matrices are a broadcast product
#: of these with the claimed activations, so recomputing ``abs`` on every evaluation was the hot spot.
_ABS_WEIGHTS: dict[tuple[int, int], np.ndarray] = {}


def _abs_weight(network: TracedNetwork, layer_index: int) -> np.ndarray:
    key = (id(network), layer_index)
    cached = _ABS_WEIGHTS.get(key)
    if cached is None:
        layer = network.architecture[layer_index]
        cached = np.abs(network.parameters[layer.name][0].astype(np.float64))
        _ABS_WEIGHTS[key] = cached
    return cached


# --------------------------------------------------------------------------- #
# Vectorised exact acceptance (K-G3c)
# --------------------------------------------------------------------------- #


def dense_transition_matrix(
    network: TracedNetwork, layer_index: int, trace: Trace, sampler: PathSampler
) -> np.ndarray:
    """Row-stochastic ``(out, in)`` transition matrix of a source-dependent dense sampler.

    Entry ``[j, i]`` is the probability that a step from neuron ``j`` of ``layer_index`` lands on
    parent ``i`` of ``layer_index - 1``.  The two source-dependent samplers of the family score a
    parent by its claimed contribution, so the whole matrix is one broadcast product of the layer's
    weight magnitudes with the claimed parent magnitudes -- which is what turns the per-neuron DP of
    ``acceptance_probability`` into a single matrix multiply.

    The construction mirrors each sampler's own ``transition_distribution`` exactly (absolute value,
    optional additive floor, row normalisation, uniform fallback on an all-zero row), so the result
    agrees with the per-neuron computation to floating-point round-off.
    """
    layer = network.architecture[layer_index]
    assert isinstance(layer, DenseLayer)
    weight = _abs_weight(network, layer_index)  # (out, in), cached
    parent_values = trace[layer_index - 1].astype(np.float64)
    if isinstance(sampler, ZeroAwareContributionSampler):
        scores = weight * (np.abs(parent_values) + sampler._epsilon)[None, :]
    elif isinstance(sampler, LocalContributionSampler):
        scores = weight * np.abs(parent_values)[None, :]
    else:  # pragma: no cover - only the two contribution samplers reach the matrix fast path
        raise TypeError(
            f"no vectorised transition matrix for {type(sampler).__name__}; "
            "use the per-neuron fallback of fast_acceptance"
        )
    totals = scores.sum(axis=1)
    rows = np.empty_like(scores)
    nonzero = totals > 0.0
    rows[nonzero] = scores[nonzero] / totals[nonzero, None]
    rows[~nonzero] = 1.0 / scores.shape[1]
    return rows


def _accept_given_masks(
    network: TracedNetwork,
    trace: Trace,
    masks: dict[int, np.ndarray],
    sampler: PathSampler,
    n_paths: int,
) -> float:
    """The acceptance recursion given precomputed per-layer consistency masks.

    Separated from :func:`fast_acceptance` so that ``accept_vector`` can compute the (expensive)
    masks -- a full forward pass -- once per trace and reuse them for every sampler of the family.
    """
    architecture = network.architecture
    g = np.ones(architecture[0].n_neurons, dtype=np.float64)
    for layer_index in range(1, len(architecture)):
        layer = architecture[layer_index]
        consistent = masks[layer_index].astype(np.float64)
        dense = isinstance(layer, DenseLayer)
        if dense and not sampler.transition_depends_on_source:
            parent_idx = np.arange(architecture[layer_index - 1].n_neurons)
            weights = _as_distribution(
                sampler.transition_distribution(architecture, layer_index, 0, parent_idx, trace)
            )
            g = consistent * float(np.dot(weights, g))
        elif dense and isinstance(sampler, (LocalContributionSampler, ZeroAwareContributionSampler)):
            # Factored form of ``c * (P g)``, where ``P[j,i] = |W[j,i]| u_i / sum_i |W[j,i]| u_i`` and
            # ``u = |a| + eps``: ``(P g)[j] = (|W| (u*g))[j] / (|W| u)[j]``, two matrix-vector products
            # instead of building the dense matrix.  A zero denominator falls back to the uniform mean,
            # exactly as ``_as_distribution`` does per neuron -- so this matches the DP to round-off.
            eps = sampler._epsilon if isinstance(sampler, ZeroAwareContributionSampler) else 0.0
            absW = _abs_weight(network, layer_index)
            u = np.abs(trace[layer_index - 1].astype(np.float64)) + eps
            denom = absW @ u
            numer = absW @ (u * g)
            value = np.where(denom > 0.0, numer / np.where(denom > 0.0, denom, 1.0), g.mean())
            g = consistent * value
        else:  # exact per-neuron fallback (non-dense, or any other source-dependent sampler)
            below = g
            g = np.zeros(layer.n_neurons, dtype=np.float64)
            for neuron in range(layer.n_neurons):
                if not consistent[neuron]:
                    continue
                parent_idx, _ = layer.parents(neuron)
                weights = _as_distribution(
                    sampler.transition_distribution(
                        architecture, layer_index, neuron, parent_idx, trace
                    )
                )
                g[neuron] = float(np.dot(weights, below[parent_idx]))
    start = _as_distribution(sampler.start_distribution(architecture, trace))
    return float(np.dot(start, g)) ** n_paths


def fast_acceptance(
    network: TracedNetwork,
    trace: Trace,
    sampler: PathSampler,
    params: ProtocolParams | None = None,
) -> float:
    """Vectorised equivalent of ``acceptance_probability``, exact to float round-off.

    The recursion is the one proved in :mod:`pvi.baseline.analysis`: with ``c_l(j)`` the local
    consistency indicator and ``g_l(j)`` the probability that a path at node ``j`` of layer ``l``
    passes every remaining check,

        ``g_0 = 1``,    ``g_l = c_l * (P_l g_{l-1})``,    ``Pr[accept] = pi . g_L``,

    raised to ``n_paths``.  For a dense layer whose sampler ignores the source node ``P_l`` has
    identical rows, so ``P_l g_{l-1}`` is one dot product; for a source-dependent dense sampler
    ``P_l`` is built in one broadcast by :func:`dense_transition_matrix` and applied as a single
    matrix multiply.  A non-dense layer or any other source-dependent sampler falls back to the exact
    per-neuron sum, so the function is correct for any architecture and sampler while being fast for
    the dense MLP of this study.
    """
    params = params or ProtocolParams()
    masks = locally_consistent_mask(network, trace, params)
    return _accept_given_masks(network, trace, masks, sampler, params.n_paths)


def accept_vector(
    network: TracedNetwork,
    trace: Trace,
    samplers: list[PathSampler],
    params: ProtocolParams,
) -> np.ndarray:
    """Acceptance of ``trace`` under every sampler, computing the consistency masks only once.

    The masks are a full forward pass and dominate a single acceptance evaluation; sharing them
    across the family turns the ``O(|F|)`` forward passes per trace into one.
    """
    masks = locally_consistent_mask(network, trace, params)
    return np.array(
        [_accept_given_masks(network, trace, masks, s, params.n_paths) for s in samplers]
    )


# --------------------------------------------------------------------------- #
# Sampler family F
# --------------------------------------------------------------------------- #


def build_sampler_family(
    network: TracedNetwork,
    thresholds: dict[int, np.ndarray],
    zero_layers: set[int],
) -> dict[str, PathSampler]:
    """Family F of Section 1.1: uniform, static, contribution, floors, saliency, tail-, zero-first.

    ``thresholds`` holds the tail-first cut per hidden layer (a percentile of the held-out
    activations) and ``zero_layers`` the layers whose claimed zeros the zero-first sampler targets.
    Gradient saliency is included although a real verifier cannot compute it; it is an upper bound on
    what influence weighting could achieve, and is marked with ``*`` in the output.
    """
    importance = {
        index: weight_magnitude_importance(network, index)
        for index in range(len(network.architecture) - 1)
    }
    family: dict[str, PathSampler] = {
        "uniform": UniformPathSampler(),
        "static-importance": StaticImportanceSampler(importance),
        "contribution": LocalContributionSampler(network),
    }
    for eps in FLOOR_EPSILONS:
        family[f"floor-{eps:g}"] = ZeroAwareContributionSampler(network, epsilon=eps)
    family["gradient-saliency*"] = GradientSaliencySampler(network)
    family["tail-first"] = TailFirstSampler(thresholds)
    family["zero-first"] = ZeroFirstSampler(zero_layers)
    return family


# --------------------------------------------------------------------------- #
# Ranges and the two range levels R1, R2
# --------------------------------------------------------------------------- #


def calibrate_ranges(
    network: TracedNetwork, calib_x: np.ndarray, layers: list[int]
) -> dict[int, np.ndarray]:
    """Per-neuron activation samples for each tampered layer, from the held-out calibration set."""
    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    chunk = 512
    for start in range(0, len(calib_x), chunk):
        batch = network.forward_batch(calib_x[start : start + chunk])
        for layer in layers:
            acts[layer].append(batch[layer])
    return {layer: np.concatenate(acts[layer], axis=0) for layer in layers}


def ceilings_at(samples: np.ndarray, cap: float | None) -> np.ndarray:
    """Per-neuron ceiling at a cap percentile; a very large ceiling for the unconstrained cap."""
    if cap is None:
        return np.full(samples.shape[1], 1e4, dtype=np.float64)
    return np.percentile(samples.astype(np.float64), cap, axis=0)


def trace_in_range(
    forged: Trace,
    tamper_layer: int,
    ranges: dict[int, np.ndarray],
    cap: float | None,
    edited: tuple[int, ...],
) -> tuple[bool, bool]:
    """``(R1, R2)`` for a forged trace.

    The natural range of a neuron at cap ``p`` is ``[lo, hi_p]``: ``lo`` its minimum over the held-out
    calibration set (0 for every hidden neuron of the MNIST MLP, negative for the logits) and ``hi_p``
    its ``p``-th percentile.  R1: every *edited* value of the tampered layer lies in its range.  R2:
    additionally every value of every recomputed layer (the tampered layer and above, up to the
    logits) lies in its own range.  The unconstrained cap makes both vacuously true.
    """
    def floor(samples: np.ndarray) -> np.ndarray:
        if cap is None:
            return np.full(samples.shape[1], -1e4)
        return samples.astype(np.float64).min(axis=0)

    edited_idx = list(edited)
    lo_tamper = floor(ranges[tamper_layer])
    hi_tamper = ceilings_at(ranges[tamper_layer], cap)
    vals = forged[tamper_layer][edited_idx].astype(np.float64)
    r1 = bool(np.all((vals >= lo_tamper[edited_idx] - 1e-6) & (vals <= hi_tamper[edited_idx] + 1e-6)))
    r2 = r1
    for layer, samples in ranges.items():
        if layer < tamper_layer:
            continue
        lo, hi = floor(samples), ceilings_at(samples, cap)
        v = forged[layer].astype(np.float64)
        if not np.all((v >= lo - 1e-6) & (v <= hi + 1e-6)):
            r2 = False
            break
    return r1, r2


# --------------------------------------------------------------------------- #
# Adversaries A1-A4 (built on the existing components; selection calls the exact oracle)
# --------------------------------------------------------------------------- #


def single_neuron_pool(
    network: TracedNetwork, honest: Trace, layer: int, pool: int
) -> list[tuple[int, float, Trace]]:
    """A1 pool: the unconstrained smallest-flipping single-neuron edits over a candidate set.

    Candidates are the most and least salient neurons and an even spread across the layer (the three
    buckets of ``plan_evasive_flip``).  The flip value and forged trace do not depend on the cap, so
    they are found once; a cap selects the feasible subset by a ceiling filter.  Restricting the
    pool only weakens the adversary, keeping the reported detection an upper bound.
    """
    honest_class = int(honest.output.argmax())
    scores = saliency(network, honest, layer)
    ordered = np.argsort(scores)
    width = network.architecture[layer].n_neurons
    per = max(1, pool // 3)
    candidates = np.unique(
        np.concatenate(
            [ordered[::-1][:per], ordered[:per], np.linspace(0, width - 1, per).astype(int)]
        )
    )
    out: list[tuple[int, float, Trace]] = []
    for neuron in candidates:
        value = smallest_flipping_value(
            network, honest, layer, int(neuron), honest_class, None,
            max_value=1e4, require_nonnegative=True,
        )
        if value is None or value < 0.0:
            continue
        forged = network.forward_from(honest.tampered(layer, int(neuron), float(value)), layer)
        if int(forged.output.argmax()) == honest_class:
            continue
        out.append((int(neuron), float(value), forged))
    return out


def _greedy_inrange(
    network: TracedNetwork,
    honest: Trace,
    layer: int,
    proposed: np.ndarray,
    order: list[int],
    kmax: int,
) -> tuple[Trace, tuple[int, ...]] | None:
    """Add neurons in ``order`` (each set to its in-range ``proposed`` value) until the output flips."""
    logits = honest.output
    target = int(np.argsort(logits)[::-1][1])
    original = honest[layer].astype(np.float64)
    trace = honest
    chosen: list[int] = []
    for neuron in order[:kmax]:
        if proposed[neuron] == original[neuron]:
            continue
        trace = trace.tampered(layer, neuron, float(proposed[neuron]))
        chosen.append(neuron)
        forged = network.forward_from(trace, layer)
        if int(forged.output.argmax()) == target:
            return forged, tuple(chosen)
    return None


def _direction_gradient(network: TracedNetwork, honest: Trace, layer: int) -> np.ndarray:
    logits = honest.output
    honest_class = int(logits.argmax())
    target = int(np.argsort(logits)[::-1][1])
    direction = np.zeros_like(logits, dtype=np.float64)
    direction[target] = 1.0
    direction[honest_class] -= 1.0
    return output_direction_gradient(network, honest, layer, direction)


#: Value written by the near-zero lowering variant (A3b), clipped to each neuron's ceiling.
NEAR_ZERO = 1e-3


def zeroing_attack(
    network: TracedNetwork,
    honest: Trace,
    layer: int,
    gradient: np.ndarray,
    hi: np.ndarray | None = None,
) -> tuple[Trace, tuple[int, ...]] | None:
    """A3: the lowering family of Lemma M, ranked by the benefit of lowering.

    With ``hi`` omitted every move is to exactly zero (A3, the mode of a ReLU neuron).  With ``hi``
    given every move is to ``min(NEAR_ZERO, hi_i)`` (A3b): a value just above zero, still in range,
    which a sampler that targets claimed zeros does not see while a contribution-type sampler still
    gives it almost no weight.  The design notes name this as the adversary's reply to zero-first
    sampling; omitting it would overstate the zero-first sampler and any mixture that leans on it.
    """
    original = honest[layer].astype(np.float64)
    if hi is None:
        proposed = np.zeros_like(original)
    else:
        proposed = np.minimum(NEAR_ZERO, np.asarray(hi, dtype=np.float64))
    benefit = np.where(gradient < 0, np.abs(gradient) * (original - proposed), 0.0)
    order = [int(n) for n in np.argsort(benefit)[::-1] if benefit[n] > 0]
    return _greedy_inrange(network, honest, layer, proposed, order, KMAX)


def ceiling_attacks(
    network: TracedNetwork,
    honest: Trace,
    layer: int,
    hi: np.ndarray,
    gradient: np.ndarray,
) -> list[tuple[Trace, tuple[int, ...]]]:
    """A2: push each neuron to its in-range ceiling where that helps, else to zero; three rankings."""
    original = honest[layer].astype(np.float64)
    above = network.architecture[layer + 1]
    outgoing = (
        np.abs(network.parameters[above.name][0].astype(np.float64)).sum(axis=0)
        if isinstance(above, DenseLayer)
        else np.ones(network.architecture[layer].n_neurons)
    )
    proposed = np.where(gradient >= 0, hi, 0.0)
    benefit = np.abs(gradient) * np.abs(proposed - original)
    exposure = np.abs(proposed) * outgoing + 1e-12
    rankings = {
        "benefit": np.argsort(benefit)[::-1],
        "benefit_per_exposure": np.argsort(benefit / exposure)[::-1],
        "least_exposure": np.argsort(np.where(benefit > 0, exposure, np.inf)),
    }
    out: list[tuple[Trace, tuple[int, ...]]] = []
    for order in rankings.values():
        usable = [int(n) for n in order if benefit[n] > 0]
        found = _greedy_inrange(network, honest, layer, proposed, usable, KMAX)
        if found is not None:
            out.append(found)
    return out


def a4_attack(
    network: TracedNetwork,
    honest: Trace,
    layer: int,
    hi: np.ndarray,
    gradient: np.ndarray,
    detect,
    *,
    kmax: int = 32,
    pool: int = 8,
) -> tuple[Trace, tuple[int, ...]] | None:
    """A4: a greedy that, at each step, commits the in-range neuron of lowest exact detection under S.

    Each neuron is proposed at its helpful in-range bound (a ceiling, or zero).  At each step the
    ``pool`` most useful unused neurons (by first-order margin benefit) are tried, each resulting
    trace is detected exactly by ``detect`` (a single sampler or the mixture), and the lowest-detection
    neuron is committed; the search stops as soon as the output flips.  This is the adversary whose
    choice is guided by the sampler in force rather than by a proxy.
    """
    logits = honest.output
    target = int(np.argsort(logits)[::-1][1])
    original = honest[layer].astype(np.float64)
    proposed = np.where(gradient >= 0, hi, 0.0)
    benefit = np.abs(gradient) * np.abs(proposed - original)
    remaining = [int(n) for n in np.argsort(benefit)[::-1] if benefit[n] > 0]

    trace = honest
    chosen: list[int] = []
    for _ in range(min(kmax, len(remaining))):
        best_d, best_neuron, best_trace, best_flipped = math.inf, None, None, False
        for neuron in remaining[:pool]:
            candidate = network.forward_from(
                trace.tampered(layer, neuron, float(proposed[neuron])), layer
            )
            d = detect(candidate)
            if d < best_d:
                best_d = d
                best_neuron = neuron
                best_trace = candidate
                best_flipped = int(candidate.output.argmax()) == target
        if best_neuron is None:
            return None
        trace = trace.tampered(layer, best_neuron, float(proposed[best_neuron]))
        chosen.append(best_neuron)
        remaining.remove(best_neuron)
        if best_flipped:
            return best_trace, tuple(chosen)
    return None


# --------------------------------------------------------------------------- #
# LP mixture
# --------------------------------------------------------------------------- #


def solve_mixture(detection: list[np.ndarray], cols: list[int]) -> tuple[np.ndarray, float]:
    """Defender's optimal per-path mixture over the samplers ``cols``, and the game value.

    ``detection[q]`` is a ``(n_attacks_q, n_samplers)`` matrix of exact single-path detections for
    query ``q``; the mixture is formed only over the sampler columns ``cols`` (so a deployable
    mixture can exclude the non-implementable gradient-saliency sampler).  The adversary picks, per
    query, the attack minimising the mixture's detection; the defender maximises the mean over
    queries:

        ``max_alpha  (1/Q) sum_q min_a  alpha . D_{q,a}``,

    an LP in ``alpha`` (a distribution over ``cols``) and one slack ``t_q`` per query
    (``t_q <= alpha . D_{q,a}`` for every attack ``a`` of ``q``), solved with ``scipy.linprog``.  The
    returned ``alpha`` is full length over all samplers, zero outside ``cols``.
    """
    from scipy.optimize import linprog

    detection = [m for m in detection if m.shape[0] > 0]
    n_total = detection[0].shape[1]
    k = len(cols)
    n_queries = len(detection)
    n_vars = k + n_queries

    c = np.zeros(n_vars)
    c[k:] = -1.0 / n_queries  # maximise mean t_q

    rows: list[np.ndarray] = []
    for q, mat in enumerate(detection):
        sub = mat[:, cols]
        for a in range(sub.shape[0]):
            row = np.zeros(n_vars)
            row[:k] = -sub[a]  # t_q - alpha . D_{q,a} <= 0
            row[k + q] = 1.0
            rows.append(row)
    a_ub = np.vstack(rows)
    b_ub = np.zeros(len(rows))
    a_eq = np.zeros((1, n_vars))
    a_eq[0, :k] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * k + [(0.0, 1.0)] * n_queries
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    sub_alpha = np.clip(res.x[:k], 0.0, None)
    sub_alpha = sub_alpha / sub_alpha.sum()
    alpha = np.zeros(n_total)
    alpha[cols] = sub_alpha
    return alpha, float(-res.fun)


# --------------------------------------------------------------------------- #
# G1e: the empirical guessing probability (tamper locator)
# --------------------------------------------------------------------------- #


def surprise_scores(samples: np.ndarray, claimed: np.ndarray) -> np.ndarray:
    """Per-neuron surprise from the held-out marginal: ``|claimed - mean| / std`` (higher = odder)."""
    mean = samples.mean(axis=0)
    std = samples.std(axis=0) + 1e-6
    return np.abs(claimed.astype(np.float64) - mean) / std


def empirical_guessing(
    network: TracedNetwork,
    honest: Trace,
    layer: int,
    feasible: list[tuple[int, Trace]],
    samples: np.ndarray,
    q_grid: list[int],
) -> dict[str, dict[int, int]]:
    """G1e: top-q hit rate of the surprise locator for the single-neuron edit, from the forged trace.

    ``feasible`` are the in-range single-neuron flips at this cap (neuron, forged trace), with the
    smallest-value one taken as the greedy attack and a pseudo-random one as the randomised attack.
    The edited neuron is ranked among the whole layer by the surprise locator; the top-q hit rate is
    the empirical ``G_q``.  Theorem ``thm:guess`` says this number is the security of every
    spot-checker against the attack, so a greedy ``G_q`` well above ``q / N`` means the headline
    attack should be the randomised one.
    """
    if not feasible:
        return {}
    greedy_neuron, greedy_forged = min(feasible, key=lambda nv: float(nv[1][layer][nv[0]]))
    rng = np.random.default_rng(int(honest.output.argmax()) + len(feasible))
    randomised = feasible[int(rng.integers(len(feasible)))]
    out: dict[str, dict[int, int]] = {}
    for label, (neuron, forged) in (("greedy", (greedy_neuron, greedy_forged)), ("randomised", randomised)):
        scores = surprise_scores(samples, forged[layer])
        rank = list(np.argsort(scores)[::-1]).index(neuron)
        out[label] = {q: int(rank < q) for q in q_grid}
    return out


# --------------------------------------------------------------------------- #
# Per-layer game over all caps
# --------------------------------------------------------------------------- #


def s_for_target(d: float, target_bits: int = 40) -> int | None:
    """Paths for ``1 - 2^-target`` at single-path detection ``d``: ``ceil(target ln2 / -ln(1-d))``."""
    if d <= 0.0:
        return None
    if d >= 1.0:
        return 1
    return math.ceil(target_bits * math.log(2.0) / -math.log1p(-d))


def play_layer(
    network: TracedNetwork,
    queries: np.ndarray,
    ranges: dict[int, np.ndarray],
    layer: int,
    caps: tuple[float | None, ...],
    family: dict[str, PathSampler],
    params: ProtocolParams,
    pool: int,
    double_oracle_rounds: int,
    q_grid: list[int],
    log,
) -> dict[str, dict]:
    """Run the game at ``layer`` for every cap, reusing cap-independent work across caps."""
    names = list(family)
    samplers = [family[name] for name in names]
    width = network.architecture[layer].n_neurons
    n_samplers = len(names)

    def accepts(trace: Trace) -> np.ndarray:
        return accept_vector(network, trace, samplers, params)

    a4_archetypes = ["uniform", "contribution", "tail-first", "zero-first"]
    a4_detectors = [family[n] for n in a4_archetypes if n in family]
    sigma = ranges[layer].astype(np.float64).std(axis=0) + 1e-6  # held-out std per neuron
    # Ascending caps, unconstrained last, so that plans can be carried to looser caps.
    caps = tuple(sorted((c for c in caps if c is not None))) + ((None,) if None in caps else ())

    cap_keys = ["inf" if c is None else f"{c:g}" for c in caps]
    acc = {
        key: {
            "detection": [],  # per successful query: (n_attacks, n_samplers) detection matrix
            "qidx": [],       # the query index of each detection matrix
            "support_rows": [],  # per successful query: edited-neuron count of each attack row
            "success": [], "supports": [], "magnitude": [], "r1": [], "r2": [], "r2_max": [],
            "honest_r2": [], "honest_r2_max": [],
            "g1e": {k: {q: [] for q in q_grid} for k in ("greedy", "randomised")},
        }
        for key in cap_keys
    }

    for query_index, query in enumerate(queries):
        honest = network.eval_trace(query)
        gradient = _direction_gradient(network, honest, layer)

        # Cap-independent attacks and their family-wide acceptance vectors (found once).
        pool_edits = single_neuron_pool(network, honest, layer, pool)
        pool_accept = {neuron: accepts(forged) for neuron, _v, forged in pool_edits}
        zero = zeroing_attack(network, honest, layer, gradient)
        zero_accept = accepts(zero[0]) if zero is not None else None

        # Plans found at a tighter cap stay in range at every looser one (ceilings grow with the
        # cap and the floor is 0), so the adversary at cap p may use all of them: ``carried`` holds
        # the attacks of the caps already processed.  This makes detection non-increasing in p, as
        # it must be for a rational adversary.  ``caps`` is in ascending order, unconstrained last.
        carried: list[tuple[Trace, tuple[int, ...], np.ndarray]] = []

        for cap, key in zip(caps, cap_keys):
            store = acc[key]
            hi = ceilings_at(ranges[layer], cap)
            # Honest baselines for R2: does the unedited trace itself keep every value of the
            # tampered layer and above within the cap's range, and within the natural maximum?
            store["honest_r2"].append(trace_in_range(honest, layer, ranges, cap, ())[1])
            store["honest_r2_max"].append(trace_in_range(honest, layer, ranges, 100.0, ())[1])

            # A1 feasible at this cap: ceiling filter on the pool's flip values.
            attacks: list[tuple[Trace, tuple[int, ...], np.ndarray]] = list(carried)
            feasible_single: list[tuple[int, Trace]] = []
            for neuron, value, forged in pool_edits:
                if value <= hi[neuron] + 1e-9:
                    attacks.append((forged, (neuron,), pool_accept[neuron]))
                    feasible_single.append((neuron, forged))
            # A3 zeroing (in range at every cap) and A3b near-zero lowering (clipped to the cap).
            if zero is not None:
                attacks.append((zero[0], zero[1], zero_accept))
            near = zeroing_attack(network, honest, layer, gradient, hi=hi)
            if near is not None:
                attacks.append((near[0], near[1], accepts(near[0])))
            # A2 ceiling plans (cap-dependent).
            for forged, edited in ceiling_attacks(network, honest, layer, hi, gradient):
                attacks.append((forged, edited, accepts(forged)))
            # A4 greedy against a few representative detectors (cap- and sampler-dependent).  Every
            # other sampler's own evasion reduces to one of these archetypes -- uniform (least
            # support), contribution (zeroing), tail-first (avoid the tail), zero-first (raise off
            # zero) -- and each A4 trace is scored against the whole family, so the matrix still
            # bounds every sampler; the mixture-adaptive A4 of the double oracle covers the rest.
            for s in a4_detectors:
                res = a4_attack(
                    network, honest, layer, hi, gradient,
                    detect=lambda t, s=s: 1.0 - fast_acceptance(network, t, s, params),
                )
                if res is not None:
                    attacks.append((res[0], res[1], accepts(res[0])))

            if not attacks:
                store["success"].append(False)
                continue
            store["success"].append(True)

            # Deduplicate identical (support, rounded values) to keep the LP small.
            unique: dict[tuple, tuple[Trace, tuple[int, ...], np.ndarray]] = {}
            for forged, edited, vec in attacks:
                dkey = (edited, tuple(np.round(forged[layer][list(edited)], 6)))
                unique.setdefault(dkey, (forged, edited, vec))
            attacks = list(unique.values())
            carried = attacks

            detection = np.vstack([1.0 - vec for _f, _e, vec in attacks])
            store["detection"].append(detection)
            store["qidx"].append(query_index)
            store["support_rows"].append(np.array([len(e) for _f, e, _v in attacks]))

            # Support / R1 / R2 at the adversary's choice against the deployed uniform sampler.
            unif = names.index("uniform")
            chosen = int(detection[:, unif].argmin())
            forged, edited, _vec = attacks[chosen]
            store["supports"].append(len(edited))
            delta = np.abs(
                forged[layer][list(edited)].astype(np.float64)
                - honest[layer][list(edited)].astype(np.float64)
            )
            store["magnitude"].append(float(np.max(delta / sigma[list(edited)])))
            r1, r2 = trace_in_range(forged, layer, ranges, cap, edited)
            store["r1"].append(r1)
            store["r2"].append(r2)
            store["r2_max"].append(trace_in_range(forged, layer, ranges, 100.0, edited)[1])

            g1e = empirical_guessing(network, honest, layer, feasible_single, ranges[layer], q_grid)
            for kind in ("greedy", "randomised"):
                if kind in g1e:
                    for q in q_grid:
                        store["g1e"][kind][q].append(g1e[kind][q])

    # Samplers a real verifier can compute (gradient saliency, marked ``*``, cannot).
    impl_cols = [i for i, n in enumerate(names) if not n.endswith("*")]

    # Finalise each cap, in ascending order: deployable LP mixture (with double oracle) and the
    # saliency upper-bound mixture.  Attacks the double oracle finds at a tighter cap are in range at
    # every looser cap, so they are carried forward (``found``, keyed by query index).
    results: dict[str, dict] = {}
    found: dict[int, list[tuple[np.ndarray, int]]] = {}
    for cap, key in zip(caps, cap_keys):
        store = acc[key]
        detection = store["detection"]
        supports_rows = store["support_rows"]
        for pos, qi in enumerate(store["qidx"]):
            for row, k in found.get(qi, []):
                detection[pos] = np.vstack([detection[pos], row[None, :]])
                supports_rows[pos] = np.append(supports_rows[pos], k)
        mixture = None
        mixture_ub = None
        if detection:
            before = [m.shape[0] for m in detection]
            # Primary: the deployable mixture over implementable samplers, with the double oracle
            # adapting the adversary to it.
            alpha, value = solve_mixture(detection, impl_cols)
            rounds = [value]
            for _ in range(double_oracle_rounds):
                added = _double_oracle_round(
                    network, queries, store["qidx"], ranges, layer, cap, family, names, params,
                    alpha, detection, supports_rows,
                )
                if not added:
                    break
                alpha, value = solve_mixture(detection, impl_cols)
                rounds.append(value)
                if abs(rounds[-1] - rounds[-2]) < 1e-4:
                    break
            for pos, qi in enumerate(store["qidx"]):
                for r in range(before[pos], detection[pos].shape[0]):
                    found.setdefault(qi, []).append((detection[pos][r], int(supports_rows[pos][r])))
            mixture = {
                "weights": {names[i]: float(alpha[i]) for i in range(n_samplers) if alpha[i] > 1e-6},
                "detection_per_path": float(value),
                "s_40": s_for_target(float(value)),
                "double_oracle_values": [float(v) for v in rounds],
                "rounds": len(rounds) - 1,
                "implementable_only": True,
            }
            # Upper bound: the whole family including gradient saliency, on the same attack set.
            alpha_ub, value_ub = solve_mixture(detection, list(range(n_samplers)))
            mixture_ub = {
                "weights": {names[i]: float(alpha_ub[i]) for i in range(n_samplers) if alpha_ub[i] > 1e-6},
                "detection_per_path": float(value_ub),
                "s_40": s_for_target(float(value_ub)),
                "note": "upper bound: includes the non-implementable gradient-saliency sampler",
            }

        def summarise(values: list[float]) -> dict:
            mean = float(np.mean(values)) if values else 0.0
            return {
                "detection_per_path": mean,
                "median": float(np.median(values)) if values else 0.0,
                "s_40": s_for_target(mean),
            }

        # Each sampler's value: mean over successful queries of the adversary's best (minimum)
        # detection over the final attack set (base attacks plus every double-oracle addition).
        sampler_rows = {
            names[i]: summarise([float(m[:, i].min()) for m in detection]) for i in range(n_samplers)
        }

        # Support limit K: the adversary may only use plans editing at most K neurons.
        k_axis: dict[str, dict] = {}
        n_q = len(store["success"])
        for kmax in K_LIMITS:
            per_sampler: dict[str, float | None] = {}
            usable = [(m, s) for m, s in zip(detection, supports_rows) if np.any(s <= kmax)]
            for i, name in enumerate(names):
                vals = [float(m[s <= kmax, i].min()) for m, s in usable]
                per_sampler[name] = float(np.mean(vals)) if vals else None
            k_axis[str(kmax)] = {
                "success_rate": (len(usable) / n_q) if n_q else 0.0,
                "detection_per_path": per_sampler,
            }
        non_uniform = [n for n in names if n != "uniform"]
        best_name = max(non_uniform, key=lambda n: sampler_rows[n]["detection_per_path"])
        impl_non_uniform = [n for n in non_uniform if not n.endswith("*")]
        best_impl = max(impl_non_uniform, key=lambda n: sampler_rows[n]["detection_per_path"])
        results[key] = {
            "cap_percentile": ("inf" if cap is None else cap),
            "width": width,
            "n_queries": len(store["success"]),
            "success_rate": float(np.mean(store["success"])) if store["success"] else 0.0,
            "mean_support": float(np.mean(store["supports"])) if store["supports"] else None,
            "mean_magnitude_sigma": float(np.mean(store["magnitude"])) if store["magnitude"] else None,
            "R1_pass_rate": float(np.mean(store["r1"])) if store["r1"] else None,
            "R2_pass_rate": float(np.mean(store["r2"])) if store["r2"] else None,
            "R2_honest_baseline": float(np.mean(store["honest_r2"])) if store["honest_r2"] else None,
            "R2_max_pass_rate": float(np.mean(store["r2_max"])) if store["r2_max"] else None,
            "R2_max_honest_baseline": (
                float(np.mean(store["honest_r2_max"])) if store["honest_r2_max"] else None
            ),
            "samplers": sampler_rows,
            "support_limit": k_axis,
            "best_single_sampler": best_name,
            "best_single_implementable": best_impl,
            "mixture": mixture,
            "mixture_upper_bound_with_saliency": mixture_ub,
            "G1e": {
                kind: {
                    "n_queries": len(store["g1e"][kind][q_grid[0]]),
                    "q_over_N": {str(q): q / width for q in q_grid},
                    "top_q_hit_rate": {
                        str(q): (float(np.mean(store["g1e"][kind][q])) if store["g1e"][kind][q] else None)
                        for q in q_grid
                    },
                }
                for kind in ("greedy", "randomised")
            },
        }
        m = results[key]["mixture"]
        log(
            f"layer {layer} (N={width}) cap {key:>4}: "
            f"unif {sampler_rows['uniform']['detection_per_path']:.5f}  "
            f"best_impl[{best_impl}] {sampler_rows[best_impl]['detection_per_path']:.5f}  "
            f"mix_impl {m['detection_per_path']:.5f} (r{m['rounds']})  "
            f"mix_ub {mixture_ub['detection_per_path']:.5f}  "
            f"succ {results[key]['success_rate']:.2f}  k {results[key]['mean_support']}  "
            f"R1 {results[key]['R1_pass_rate']} R2 {results[key]['R2_pass_rate']} "
            f"(honest {results[key]['R2_honest_baseline']}) R2max {results[key]['R2_max_pass_rate']} "
            f"(honest {results[key]['R2_max_honest_baseline']})"
        )
    return results


def _double_oracle_round(
    network: TracedNetwork,
    queries: np.ndarray,
    qidx: list[int],
    ranges: dict[int, np.ndarray],
    layer: int,
    cap: float | None,
    family: dict[str, PathSampler],
    names: list[str],
    params: ProtocolParams,
    alpha: np.ndarray,
    detection: list[np.ndarray],
    supports_rows: list[np.ndarray],
) -> bool:
    """Re-run A4 against the current mixture; append any strictly better attack. Returns whether added.

    Only A4 is run here: A1, A2 and A3 do not depend on the detector, so their rows are already in
    the matrices, while A4's choice is adaptive to the mixture oracle, which is the per-path
    acceptance ``sum_k alpha_k accept_k`` (linear in ``alpha``).  ``detection[pos]`` belongs to query
    ``qidx[pos]``; iterating over that pairing keeps each new attack with its own query.
    """
    samplers = [family[name] for name in names]
    hi = ceilings_at(ranges[layer], cap)

    def detect_mix(trace: Trace) -> float:
        return float(np.dot(alpha, 1.0 - accept_vector(network, trace, samplers, params)))

    added = False
    for pos, qi in enumerate(qidx):
        honest = network.eval_trace(queries[qi])
        gradient = _direction_gradient(network, honest, layer)
        res = a4_attack(network, honest, layer, hi, gradient, detect=detect_mix)
        if res is None:
            continue
        forged, edited = res
        row = 1.0 - accept_vector(network, forged, samplers, params)
        if float(np.dot(alpha, row)) < float(np.min(detection[pos] @ alpha)) - 1e-9:
            detection[pos] = np.vstack([detection[pos], row[None, :]])
            supports_rows[pos] = np.append(supports_rows[pos], len(edited))
            added = True
    return added


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlp_mnist", choices=["mlp_mnist"])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--queries", type=int, default=40, help="disjoint test queries attacked")
    parser.add_argument("--calib", choices=["train", "test-tail"], default="train")
    parser.add_argument("--calib-size", type=int, default=5000)
    parser.add_argument("--caps", type=str, nargs="+", default=None,
                        help="override caps, e.g. 50 75 90 95 99 100 inf")
    parser.add_argument("--pool", type=int, default=96, help="A1 candidate neurons")
    parser.add_argument("--double-oracle", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tail-percentile", type=float, default=95.0, help="tail-first sampler cut")
    parser.add_argument("--out", default=str(RESULTS / "adaptive_game_mlp_mnist.json"))
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    from pvi.mnist.data import load_classification

    if args.caps is None:
        caps: tuple[float | None, ...] = CAPS
    else:
        caps = tuple(None if c.lower() in ("inf", "none") else float(c) for c in args.caps)

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification("mnist", seed=0)
    train_x = np.ascontiguousarray(dataset.train_x.reshape(len(dataset.train_x), -1), np.float32)
    test_x = np.ascontiguousarray(dataset.test_x.reshape(len(dataset.test_x), -1), np.float32)

    if args.calib == "train":
        calib_x = train_x[: args.calib_size]      # train and test are disjoint datasets
        attack_x = test_x[: args.queries]
    else:                                         # held-out test tail 5000-9999, disjoint head attacked
        calib_x = test_x[5000:10000]
        attack_x = test_x[: args.queries]

    layers = list(args.layers)
    # Ranges for every non-input layer: the tampered layers' ceilings, and R2 over the recomputed
    # layers up to and including the output logits.
    ranges = calibrate_ranges(network, calib_x, list(range(1, len(network.architecture))))
    params = ProtocolParams(n_paths=1)
    q_grid = [1, 2, 4, 8, 16, 32, 64]
    started = time.perf_counter()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"model {network.architecture.layer_widths}  calib={args.calib}({len(calib_x)})  "
        f"queries={len(attack_x)}  threads={args.threads}")

    out: dict = {
        "config": {
            "model": args.model, "layers": layers, "queries": int(len(attack_x)),
            "calib": args.calib, "calib_size": int(len(calib_x)),
            "caps": [("inf" if c is None else c) for c in caps],
            "pool": args.pool, "double_oracle_rounds": args.double_oracle,
            "tail_percentile": args.tail_percentile, "threads": args.threads,
        },
        "note": (
            "Detection is the exact single-path 1 - acceptance (fast_acceptance, validated to 1e-9 "
            "against acceptance_probability). The per-sampler value is the adversary's best (minimum) "
            "over A1-A4; an UPPER BOUND on the sampler's true worst-case detection -- a stronger "
            "adversary can only lower it."
        ),
        "27.7_over_N": {
            str(network.architecture[l].n_neurons): 27.7 / network.architecture[l].n_neurons
            for l in layers
        },
        "layers": {},
    }

    for layer in layers:
        thresholds = {
            l: np.percentile(ranges[l].astype(np.float64), args.tail_percentile, axis=0)
            for l in layers
        }
        family = build_sampler_family(network, thresholds, set(layers))
        layer_caps = play_layer(
            network, attack_x, ranges, layer, caps, family, params,
            args.pool, args.double_oracle, q_grid, log,
        )
        out["layers"][str(layer)] = {"width": network.architecture[layer].n_neurons, "caps": layer_caps}

    out["seconds"] = time.perf_counter() - started
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out)
    log(f"\nwrote {args.out} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
