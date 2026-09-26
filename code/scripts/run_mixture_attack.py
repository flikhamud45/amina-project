"""Does a MIXTURE attack break the epsilon-floor sampler around eps = 1?

`DEFENCE_NOTES.md` Section 6 reports that a floor `eps` on contribution
weighting beats uniform around `eps = 1`, on every query tested, against two
attack families:

* **zeroing** many neurons -- each costs the attacker exposure `eps * |w|`, so
  a large `eps` punishes it;
* **raising** one neuron -- exposure `|w| * (value + eps)`, so a small `eps`
  punishes it.

Those pull in opposite directions, which is exactly why a middle `eps` looked
good.  The obvious attack nobody had run is the **mixture**: zero the neurons
with the smallest outgoing weights (cheap under any `eps`) while raising one
neuron by the least amount that still flips the output.  If the middle-`eps`
result is an artefact of only testing the two pure strategies, this is what
should expose it.

Usage
-----
    python scripts/run_mixture_attack.py [--queries N] [--layer L]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import TamperPlan, _smallest_flipping_value, apply_plan
from pvi.data import load_classification
from pvi.defences import ZeroAwareContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.nn.gradients import output_direction_gradient
from pvi.protocol import ProtocolParams, UniformPathSampler
from pvi.results import write_json
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

EPSILONS = [0.1, 1.0, 10.0]


def mixture_plans(network, honest, layer_index, ceilings, max_zeros=40):
    """Zero the cheapest-exposure neurons, then raise one as little as possible.

    Cheapest exposure under `|w| * (value + eps)` means smallest outgoing weight,
    so we rank by that rather than by saliency.
    """
    architecture = network.architecture
    layer = architecture[layer_index]
    above = architecture[layer_index + 1]
    outgoing = np.abs(network.parameters[above.name][0].astype(np.float64)).sum(axis=0)

    honest_class = int(honest.output.argmax())
    logits = honest.output
    target = int(np.argsort(logits)[::-1][1])
    direction = np.zeros_like(logits, dtype=np.float64)
    direction[target] = 1.0
    direction[honest_class] -= 1.0
    gradient = output_direction_gradient(network, honest, layer_index, direction)
    values = honest[layer_index].astype(np.float64)

    # Zeroing helps when the neuron currently pushes towards the honest class.
    helps_when_zeroed = np.flatnonzero((gradient < 0) & (values > 0))
    cheapest = helps_when_zeroed[np.argsort(outgoing[helps_when_zeroed])]

    plans = []
    for n_zero in (0, 2, 5, 10, 20, max_zeros):
        zeros = [int(v) for v in cheapest[:n_zero]]
        trace = honest
        for v in zeros:
            trace = trace.tampered(layer_index, v, 0.0)
        forged = network.forward_from(trace, layer_index)
        if int(forged.output.argmax()) == target:
            plans.append(TamperPlan(
                layer_index=layer_index, neurons=tuple(zeros),
                new_values=tuple(0.0 for _ in zeros),
                original_values=tuple(float(values[v]) for v in zeros),
                honest_class=honest_class, forged_class=target))
            continue
        # Not enough on its own: raise one neuron by the least amount that
        # finishes the job.  Give the attacker every reasonable option -- any
        # wrong class counts, search widely, and fall back to an unconstrained
        # value if the natural envelope is too tight.
        order = np.argsort(outgoing)          # cheapest exposure first
        raised = False
        for v in order[:192]:
            v = int(v)
            if v in zeros:
                continue
            value = _smallest_flipping_value(
                network, trace, layer_index, v, honest_class, None,
                max_value=float(ceilings[v]), require_nonnegative=True)
            if value is None:
                value = _smallest_flipping_value(
                    network, trace, layer_index, v, honest_class, None,
                    max_value=1e4, require_nonnegative=True)
            if value is None:
                continue
            raised = True
            plans.append(TamperPlan(
                layer_index=layer_index, neurons=tuple(zeros) + (v,),
                new_values=tuple(0.0 for _ in zeros) + (float(value),),
                original_values=tuple(float(values[n]) for n in zeros) + (float(values[v]),),
                honest_class=honest_class, forged_class=target))
            break
        if not raised:
            continue
    return plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    params = ProtocolParams()
    width = network.architecture[args.layer].n_neurons
    ceilings = calibrate_activation_ceilings(
        network, train_x[:3000], layer_index=args.layer, percentile=100.0)

    samplers = {"uniform": lambda: UniformPathSampler()}
    for eps in EPSILONS:
        samplers[f"eps={eps:g}"] = (lambda e=eps: ZeroAwareContributionSampler(network, epsilon=e))

    per = {name: {"pure": [], "mixture": [], "best": []} for name in samplers}
    used = 0
    for query in test_x[: args.queries]:
        honest = network.eval_trace(query)
        mixes = mixture_plans(network, honest, args.layer, ceilings)
        if not mixes:
            continue
        used += 1
        forged_mixes = [apply_plan(network, honest, p) for p in mixes]
        for name, make in samplers.items():
            sampler = make()
            pure = plan_adaptive_stealthy_flip(
                network, honest, layer_index=args.layer, ceilings=ceilings,
                sampler=sampler, params=params)
            pure_det = (1.0 - pure.acceptance) if pure.succeeded else None
            mix_det = min(
                1.0 - acceptance_probability(network, f, params, sampler)
                for f in forged_mixes)
            candidates = [c for c in (pure_det, mix_det) if c is not None]
            if pure_det is not None:
                per[name]["pure"].append(pure_det)
            per[name]["mixture"].append(mix_det)
            per[name]["best"].append(min(candidates))
        print(f"query {used}: " + "  ".join(
            f"{n}: {per[n]['best'][-1]:.4f}" for n in samplers), flush=True)

    print(f"\nlayer {args.layer} (width {width}), {used} queries, 1/N = {1/width:.6f}\n")
    header = f"{'sampler':>10} | {'pure':>9} {'MIXTURE':>9} | {'best':>9} {'worst q':>9} {'vs 1/N':>8}"
    print(header); print("-" * len(header))
    rows = {}
    for name in samplers:
        d = per[name]
        best = np.array(d["best"])
        row = {"pure_mean": float(np.mean(d["pure"])) if d["pure"] else float("nan"),
               "mixture_mean": float(np.mean(d["mixture"])),
               "best_mean": float(best.mean()), "best_worst": float(best.min()),
               "ratio_worst": float(best.min() * width)}
        rows[name] = row
        print(f"{name:>10} | {row['pure_mean']:>9.5f} {row['mixture_mean']:>9.5f} | "
              f"{row['best_mean']:>9.5f} {row['best_worst']:>9.5f} {row['ratio_worst']:>7.1f}x")

    base = rows["uniform"]["best_worst"]
    print(f"\nuniform worst query = {base:.6f}")
    for name, row in rows.items():
        if name == "uniform":
            continue
        verdict = "still beats uniform" if row["best_worst"] > base else "NO LONGER beats uniform"
        print(f"  {name:>9}: worst query {row['best_worst']:.6f} -> {verdict}")

    write_json(RESULTS / "mixture_attack.json",
               {"layer": args.layer, "width": width, "queries": used, "samplers": rows})
    print(f"\nwrote {RESULTS / 'mixture_attack.json'}")


if __name__ == "__main__":
    main()
