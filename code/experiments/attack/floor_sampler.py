"""Contribution weighting with an additive floor against an adversary who knows the rule (Sec. 3.5, ``sec:break``).

``ZeroAwareContributionSampler`` steps to parent ``i`` with probability proportional to ``|w_ij| (|a~_i| + delta)``
(the script's ``eps`` is the paper's floor ``delta``).  At ``delta = 0`` it is the plain contribution-weighted sampler,
whose zero blind spot gives detection 0 against the zeroing edit.  The question is whether some ``delta > 0`` beats
uniform sampling.  The sampler reads the claimed activations, so detection is measured on the *tampered* trace.  For
each sampler the adversary takes the better of two attack families, and the reported value is the adversary's best,
i.e. an upper bound on the sampler's worst case:

1. **single-neuron flip**: for every neuron of ``--layer`` where one edit (non-negative, at most ``--max-value``)
   flips the output, the forged trace's exact acceptance;
2. **multi-neuron lowering**: ``plan_adaptive_stealthy_flip`` searched against the sampler in force, within the
   ``--calib-percentile`` ceilings of the first ``--calib-images`` training images; this finds the zeroing edit.

Paper: "A floor delta, sampling by |w_ij|(|a_i|+delta), reaches 2% at delta=1" (Sec. 3.5) and "The floor delta=1
detects 10x as often as uniform sampling; other floors give 0.9--6x" (``ext:attack-measure``).

Usage (from ``code/``)::

    python -m experiments.attack.floor_sampler
    python -m experiments.attack.floor_sampler --queries 2 --epsilons 0 1 --out /tmp/floor_sampler.json

Writes ``--out`` (default ``artifacts/results/floor_sampler.json``): ``config``, ``env``, ``layer``, ``width``,
``n_queries`` (queries with at least one flipping neuron), ``note`` and ``samplers.<name>.adversary_best`` (mean over
queries of the adversary's best detection; names ``uniform`` and ``eps=<delta>``).  Uses
``artifacts/models/mlp_mnist_full.npz`` and MNIST; CPU only, about 40 min for the 20 queries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common.paths import MNIST_MODELS, RESULTS
from experiments.common.records import environment, write_json
from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import smallest_flipping_value
from pvi.baseline.adaptive import plan_adaptive_stealthy_flip
from pvi.baseline.analysis import acceptance_probability
from pvi.baseline.path.params import ProtocolParams
from pvi.baseline.path.sampling import UniformPathSampler
from pvi.baseline.samplers import ZeroAwareContributionSampler
from pvi.mnist.data import load_classification
from pvi.mnist.training import load_network
from pvi.mnist.zoo import mlp_architecture_for

MODEL = "mlp_mnist_full"


def single_neuron_worst_case(network, sampler, params, forged_cache):
    """Exact: the adversary's best single-neuron flip against this sampler."""
    best = None
    for forged in forged_cache.values():
        detection = 1.0 - acceptance_probability(network, forged, params, sampler)
        best = detection if best is None else min(best, detection)
    return best


def multi_neuron_worst_case(network, honest, layer, sampler, params, ceilings):
    """The adaptive multi-neuron search, run against this specific sampler."""
    search = plan_adaptive_stealthy_flip(network, honest, layer_index=layer, ceilings=ceilings,
                                         sampler=sampler, params=params)
    if not search.succeeded:
        return None
    return 1.0 - search.acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=20, help="test queries")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
                        help="floors delta of the zero-aware contribution sampler")
    parser.add_argument("--layer", type=int, default=1, help="the attacked hidden layer")
    parser.add_argument("--calib-images", type=int, default=3000,
                        help="training images whose activations set the multi-neuron adversary's ceilings")
    parser.add_argument("--calib-percentile", type=float, default=100.0,
                        help="the ceilings' percentile of each neuron's natural range")
    parser.add_argument("--max-value", type=float, default=1e4, help="largest value of a single-neuron edit")
    parser.add_argument("--models", default=str(MNIST_MODELS), help="folder of the trained models")
    parser.add_argument("--out", default=str(RESULTS / "floor_sampler.json"))
    args = parser.parse_args()

    layer = args.layer
    network = load_network(mlp_architecture_for(10), Path(args.models) / f"{MODEL}.npz")
    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    params = ProtocolParams()
    width = network.architecture[layer].n_neurons
    ceilings = calibrate_activation_ceilings(network, train_x[:args.calib_images], layer_index=layer,
                                             percentile=args.calib_percentile)

    samplers = {"uniform": lambda: UniformPathSampler()}
    for eps in args.epsilons:
        samplers[f"eps={eps:g}"] = lambda e=eps: ZeroAwareContributionSampler(network, epsilon=e)

    per_sampler = {name: [] for name in samplers}
    n_used = 0
    for query in test_x[:args.queries]:
        honest = network.eval_trace(query)
        winner = int(honest.output.argmax())

        # The neurons where one edit flips the output, and each one's forged trace, computed once per query.
        forged_cache = {}
        for v in range(width):
            value = smallest_flipping_value(network, honest, layer, v, winner, None, max_value=args.max_value,
                                            require_nonnegative=True)
            if value is not None:
                forged_cache[v] = network.forward_from(honest.tampered(layer, v, value), layer)
        if not forged_cache:
            continue
        n_used += 1
        for name, make in samplers.items():
            sampler = make()
            single = single_neuron_worst_case(network, sampler, params, forged_cache)
            multi = multi_neuron_worst_case(network, honest, layer, sampler, params, ceilings)
            per_sampler[name].append(min(c for c in (single, multi) if c is not None))

    print(f"layer {layer} (width {width}), {n_used} queries, exact over every flipping neuron per query\n")
    print("Detection against the adversary's best of two attack families, on the tampered trace.\n")
    header = f"{'sampler':>12} | {'adversary best':>15}"
    print(header)
    print("-" * len(header))
    rows = {}
    for name in samplers:
        d = per_sampler[name]
        best = float(np.mean(d)) if d else float("nan")
        rows[name] = {"adversary_best": best}
        print(f"{name:>12} | {best:>15.6f}")

    baseline = rows["uniform"]["adversary_best"]
    print(f"\nuniform's worst case = {baseline:.6f}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["adversary_best"]):
        if name != "uniform":
            print(f"  {name:>10}: {r['adversary_best']:.6f}  ({r['adversary_best'] / baseline:.1f}x uniform)")

    write_json(args.out, {"config": {k: v for k, v in vars(args).items() if k not in ("models", "out")},
                          "env": environment("cpu"), "layer": layer, "width": width, "n_queries": n_used,
                          "note": "detection measured on tampered traces; adversary picks "
                                  "the better of single-neuron and multi-neuron attacks",
                          "samplers": rows})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
