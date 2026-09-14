"""Revision-1, Phase 0: does smoothing the zero blind spot actually help?

``ZeroAwareContributionSampler`` (``defences/sampling.py``) is the natural first
patch for Theorem 3: score a parent ``i`` by ``|w_ij| * (|a~_i| + epsilon)``
instead of ``|w_ij * a~_i|``, so a claimed value of exactly zero no longer sends
the transition weight to exactly zero.

The claim under test (``DEFENCE_NOTES.md`` Theorem 4, draft): this does not
escape the dichotomy of Theorems 2/3, it only chooses which side of it applies.
As the adversary drives its tampered activations to (or near) zero -- which costs
it nothing, since zero is what it wants to claim anyway -- the sampler's weight at
those neurons converges to ``epsilon * |w_ij|``, a trace-independent,
locally-computable rule. Theorem 2 already shows such rules are minimax-capped at
``<= 1/|U|`` and, per the measured ``static-importance`` numbers in
``DEFENCE_NOTES.md``, typically *worse* than uniform against an adversary that
knows the rule. So sweeping ``epsilon`` should trade "detection of the exact-zero
attack" against "how close the worst case gets to uniform" -- never past it.

Usage
-----
    python scripts/run_theorem4_check.py [--queries N] [--layer L]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import _smallest_flipping_value, apply_plan, plan_single_neuron_flip
from pvi.data import load_classification
from pvi.defences import LocalContributionSampler, ZeroAwareContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip, visit_probabilities_under
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ProtocolParams, UniformPathSampler
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

EPSILONS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    queries = dataset.test_x.reshape(len(dataset.test_x), -1)[: args.queries]
    params = ProtocolParams()
    ceilings = calibrate_activation_ceilings(network, queries, layer_index=args.layer, percentile=100.0)

    print(f"layer {args.layer} (width {network.architecture[args.layer].n_neurons}), "
          f"{len(queries)} queries\n")

    # --- naive/loud attack: does the epsilon floor cost us the easy case? ---
    naive_forged = []
    for q in queries:
        honest = network.eval_trace(q)
        plan = plan_single_neuron_flip(network, honest, layer_index=args.layer)
        if plan is not None:
            naive_forged.append(apply_plan(network, honest, plan))

    print(f"naive-attack forged traces: {len(naive_forged)}/{len(queries)}\n")

    uniform_naive = float(np.mean(
        [acceptance_probability(network, t, params) for t in naive_forged]
    ))

    # Fixed-plan check: what a *non-adaptive* zero-hiding plan (crafted once, against
    # plain contribution weighting) looks like as epsilon grows -- informative, but
    # NOT the worst case, since a real adversary would re-plan against each epsilon.
    fixed_zero_plans = []
    for q in queries:
        honest = network.eval_trace(q)
        search = plan_adaptive_stealthy_flip(
            network, honest, layer_index=args.layer, ceilings=ceilings,
            sampler=LocalContributionSampler(network), params=params,
        )
        if search.succeeded:
            fixed_zero_plans.append(apply_plan(network, honest, search.plan))
    uniform_zero_fixed = float(np.mean(
        [acceptance_probability(network, t, params) for t in fixed_zero_plans]
    ))

    rows = []
    print(f"{'epsilon':>10} | {'det. naive':>10} | {'det. fixed-plan':>15} | "
          f"{'det. ADAPTIVE (re-planned per eps)':>36} | {'vs uniform on that plan':>24}")
    print("-" * 115)
    for epsilon in EPSILONS:
        sampler = ZeroAwareContributionSampler(network, epsilon=epsilon)

        det_naive = 1.0 - float(np.mean(
            [acceptance_probability(network, t, params, sampler) for t in naive_forged]
        ))
        det_fixed = 1.0 - float(np.mean(
            [acceptance_probability(network, t, params, sampler) for t in fixed_zero_plans]
        ))

        # The real worst case: let the adversary re-plan its evasion *against this
        # specific epsilon*, exactly as EvasionSearch already does for every other
        # sampler in this codebase (Kerckhoffs: the sampler is public).
        accs, uniform_accs, n_succeeded = [], [], 0
        for q in queries:
            honest = network.eval_trace(q)
            search = plan_adaptive_stealthy_flip(
                network, honest, layer_index=args.layer, ceilings=ceilings,
                sampler=sampler, params=params,
            )
            if search.succeeded:
                n_succeeded += 1
                accs.append(search.acceptance)
                uniform_accs.append(search.best_uniform_acceptance)
        det_adaptive = 1.0 - float(np.mean(accs)) if accs else float("nan")
        det_uniform_same_plan = 1.0 - float(np.mean(uniform_accs)) if uniform_accs else float("nan")

        rows.append({
            "epsilon": epsilon,
            "detection_naive": det_naive,
            "detection_fixed_zero_plan": det_fixed,
            "detection_adaptive_replanned": det_adaptive,
            "detection_uniform_on_same_adaptive_plan": det_uniform_same_plan,
            "n_succeeded": n_succeeded,
        })
        gap = det_adaptive - det_uniform_same_plan
        print(f"{epsilon:>10.3g} | {det_naive:>10.5f} | {det_fixed:>15.6f} | "
              f"{det_adaptive:>36.6f} | {gap:+.6f}")

    print(f"\nuniform baseline -> detection naive {1.0 - uniform_naive:.5f}, "
          f"detection fixed zero-plan {1.0 - uniform_zero_fixed:.5f}")

    # --- exact Theorem 1/2 minimax check, not the heuristic multi-node search ---
    # This is the definitive test for the trace-independent limit (epsilon -> large):
    # the *exact* single-neuron minimax bound min_{v in U} q(v), computed the same
    # way DEFENCE_NOTES.md's Theorem 2 table was computed, with no heuristic-search
    # uncertainty about whether the adversary's true optimum was found.
    print("\n=== exact minimax check (Theorem 1/2), single query ===")
    honest0 = network.eval_trace(queries[0])
    usable = [
        n for n in range(network.architecture[args.layer].n_neurons)
        if _smallest_flipping_value(
            network, honest0, args.layer, n, int(honest0.output.argmax()), None,
            max_value=1e4, require_nonnegative=True,
        ) is not None
    ]
    usable_array = np.asarray(usable)
    print(f"|U| = {len(usable)} / {network.architecture[args.layer].n_neurons}, "
          f"uniform bound 1/|U| = {1.0 / max(len(usable), 1):.6f}")

    minimax_rows = []
    for epsilon in EPSILONS:
        sampler = ZeroAwareContributionSampler(network, epsilon=epsilon)
        visits = visit_probabilities_under(network, honest0, args.layer, sampler)
        restricted = visits[usable_array] if len(usable_array) else visits
        row = {
            "epsilon": epsilon,
            "min_visit_probability_over_U": float(restricted.min()),
            "worst_case_beats_uniform": bool(restricted.min() > 1.0 / max(len(usable), 1)),
        }
        minimax_rows.append(row)
        print(f"  epsilon={epsilon:>8.3g}  min_v q(v) over U = {row['min_visit_probability_over_U']:.6f}  "
              f"{'BEATS' if row['worst_case_beats_uniform'] else 'does not beat'} uniform's "
              f"{1.0 / max(len(usable), 1):.6f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "theorem4_check.json", "w") as fh:
        json.dump({
            "layer": args.layer,
            "n_queries": len(queries),
            "uniform_detection_naive": 1.0 - uniform_naive,
            "uniform_detection_zero_hiding_fixed_plan": 1.0 - uniform_zero_fixed,
            "sweep": rows,
            "usable_neurons": len(usable),
            "bound_one_over_u": 1.0 / max(len(usable), 1),
            "minimax_sweep": minimax_rows,
        }, fh, indent=2)
    print(f"\nwrote {RESULTS / 'theorem4_check.json'}")


if __name__ == "__main__":
    main()
