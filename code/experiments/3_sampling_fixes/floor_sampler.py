"""Does an additive floor on contribution weighting beat uniform sampling? (report, Sec. 3.2)

``ZeroAwareContributionSampler`` scores a parent by ``|w_ij| * (|a~_i| + eps)``.  At
``eps = 0`` it is ``LocalContributionSampler``, whose zero blind spot gives detection 0
against a zeroing attack.  The question is whether some ``eps > 0`` does better than
plain uniform sampling against an adversary that knows the rule.

The sampler reads the claimed activations, so detection must be measured on the
*tampered* trace (on the honest trace the answer is wrong).  For each sampler the
adversary picks the better of two attack families, and we report the adversary's best,
i.e. the sampler's worst case:

1. **single-neuron flip** -- for every ``v`` in ``U`` (neurons where a single-node
   tamper flips the output) build the forged trace and compute acceptance exactly;
2. **multi-neuron adaptive/zeroing** -- ``plan_adaptive_stealthy_flip`` searched
   against the sampler in force, which is what finds the zero-hiding attack.

Usage
-----
    python experiments/3_sampling_fixes/floor_sampler.py [--queries N] [--layer L]

Writes ``artifacts/results/floor_sampler.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import _smallest_flipping_value
from pvi.data import load_classification
from pvi.defences import ZeroAwareContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ProtocolParams, UniformPathSampler
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

EPSILONS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]


def single_neuron_worst_case(network, honest, layer, sampler, params, forged_cache):
    """Exact: the adversary's best single-neuron flip against this sampler."""
    best = None
    for forged in forged_cache.values():
        detection = 1.0 - acceptance_probability(network, forged, params, sampler)
        best = detection if best is None else min(best, detection)
    return best


def multi_neuron_worst_case(network, honest, layer, sampler, params, ceilings):
    """The adaptive multi-neuron search, run against this specific sampler."""
    search = plan_adaptive_stealthy_flip(
        network, honest, layer_index=layer, ceilings=ceilings,
        sampler=sampler, params=params,
    )
    if not search.succeeded:
        return None
    return 1.0 - search.acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    params = ProtocolParams()
    width = network.architecture[args.layer].n_neurons
    ceilings = calibrate_activation_ceilings(
        network, train_x[:3000], layer_index=args.layer, percentile=100.0
    )

    samplers = {"uniform": lambda: UniformPathSampler()}
    for eps in EPSILONS:
        samplers[f"eps={eps:g}"] = (
            lambda e=eps: ZeroAwareContributionSampler(network, epsilon=e)
        )

    per_sampler = {name: {"single": [], "multi": [], "best": []} for name in samplers}
    n_used = 0

    for query in test_x[: args.queries]:
        honest = network.eval_trace(query)
        winner = int(honest.output.argmax())

        # U and the forged trace for every usable neuron, computed once per query.
        forged_cache = {}
        for v in range(width):
            value = _smallest_flipping_value(
                network, honest, args.layer, v, winner, None,
                max_value=1e4, require_nonnegative=True,
            )
            if value is not None:
                forged_cache[v] = network.forward_from(
                    honest.tampered(args.layer, v, value), args.layer
                )
        if not forged_cache:
            continue
        n_used += 1

        for name, make in samplers.items():
            sampler = make()
            single = single_neuron_worst_case(
                network, honest, args.layer, sampler, params, forged_cache
            )
            multi = multi_neuron_worst_case(
                network, honest, args.layer, sampler, params, ceilings
            )
            candidates = [c for c in (single, multi) if c is not None]
            per_sampler[name]["single"].append(single)
            if multi is not None:
                per_sampler[name]["multi"].append(multi)
            per_sampler[name]["best"].append(min(candidates))

    print(f"layer {args.layer} (width {width}), {n_used} queries, "
          f"|U| computed exactly per query\n")
    print("Detection against the ADVERSARY'S BEST of two attack families.")
    print("Higher is better for the defender. All on the tampered trace.\n")
    header = f"{'sampler':>12} | {'single-neuron':>14} | {'multi-neuron':>13} | {'ADVERSARY BEST':>15}"
    print(header)
    print("-" * len(header))
    rows = {}
    for name in samplers:
        d = per_sampler[name]
        single = float(np.mean(d["single"])) if d["single"] else float("nan")
        multi = float(np.mean(d["multi"])) if d["multi"] else float("nan")
        best = float(np.mean(d["best"])) if d["best"] else float("nan")
        rows[name] = {"single_neuron": single, "multi_neuron": multi, "adversary_best": best}
        print(f"{name:>12} | {single:>14.6f} | {multi:>13.6f} | {best:>15.6f}")

    baseline = rows["uniform"]["adversary_best"]
    print(f"\nuniform's worst case = {baseline:.6f}")
    winners = [
        (n, r["adversary_best"] / baseline)
        for n, r in rows.items()
        if n != "uniform" and r["adversary_best"] > baseline
    ]
    if winners:
        print("samplers that BEAT uniform against the adversary's best attack:")
        for n, ratio in sorted(winners, key=lambda kv: -kv[1]):
            print(f"  {n:>10}: {rows[n]['adversary_best']:.6f}  ({ratio:.1f}x uniform)")
    else:
        print("no epsilon beats uniform against the adversary's best attack.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "floor_sampler.json",
               {"layer": args.layer, "width": width, "n_queries": n_used,
                "note": "detection measured on tampered traces; adversary picks "
                        "the better of single-neuron and multi-neuron attacks",
                "samplers": rows})
    print(f"\nwrote {RESULTS / 'floor_sampler.json'}")


if __name__ == "__main__":
    main()
