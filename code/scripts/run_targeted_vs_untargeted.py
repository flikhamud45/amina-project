"""Revision-1, Phase 0.5: does a specific target label shrink U?

``DEFENCE_NOTES.md``'s ``|U|`` numbers (e.g. "454 of 512 neurons admit a flip") are
computed with ``target_class=None`` in ``_smallest_flipping_value``: the adversary
wins by flipping the prediction to *any* other class. A real backdoor usually needs
a specific target label. ``target_class`` is already wired through
``attacks/tamper.py``, so this script just asks the question the existing code was
never pointed at: how much smaller is ``U`` when the adversary must hit one
pre-chosen class, and does that raise the Theorem 1/2 detection floor in practice.

Usage
-----
    python scripts/run_targeted_vs_untargeted.py [--queries N] [--layer L]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pvi.attacks.tamper import _smallest_flipping_value
from pvi.data import load_classification
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"


def usable_neurons(network, honest, layer_index: int, target_class: int | None) -> list[int]:
    width = network.architecture[layer_index].n_neurons
    honest_class = int(honest.output.argmax())
    usable = []
    for neuron in range(width):
        value = _smallest_flipping_value(
            network, honest, layer_index, neuron, honest_class, target_class,
            max_value=1e4, require_nonnegative=True,
        )
        if value is not None:
            usable.append(neuron)
    return usable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    n_classes = dataset.n_classes
    queries = dataset.test_x.reshape(len(dataset.test_x), -1)[: args.queries]
    width = network.architecture[args.layer].n_neurons

    rows = []
    for index, query in enumerate(queries):
        honest = network.eval_trace(query)
        honest_class = int(honest.output.argmax())

        u_any = usable_neurons(network, honest, args.layer, None)
        per_target = {}
        for target in range(n_classes):
            if target == honest_class:
                continue
            per_target[target] = len(usable_neurons(network, honest, args.layer, target))

        row = {
            "query_index": index,
            "honest_class": honest_class,
            "u_any": len(u_any),
            "per_target": per_target,
            "u_target_min": min(per_target.values()),
            "u_target_max": max(per_target.values()),
            "u_target_mean": float(np.mean(list(per_target.values()))),
        }
        rows.append(row)
        print(
            f"[{index:3d}] honest={honest_class}  |U_any|={row['u_any']:3d}/{width}  "
            f"|U_target| min={row['u_target_min']:3d} "
            f"mean={row['u_target_mean']:6.1f} max={row['u_target_max']:3d}"
        )

    u_any_vals = np.array([r["u_any"] for r in rows], dtype=float)
    u_target_min_vals = np.array([r["u_target_min"] for r in rows], dtype=float)
    u_target_max_vals = np.array([r["u_target_max"] for r in rows], dtype=float)
    u_target_mean_vals = np.array([r["u_target_mean"] for r in rows], dtype=float)

    summary = {
        "layer": args.layer,
        "layer_width": width,
        "n_queries": len(rows),
        "u_any_mean": float(u_any_vals.mean()),
        "u_target_hardest_mean": float(u_target_min_vals.mean()),
        "u_target_easiest_mean": float(u_target_max_vals.mean()),
        "u_target_avg_mean": float(u_target_mean_vals.mean()),
        "bound_untargeted_k1": float(np.mean(1.0 / u_any_vals)),
        "bound_targeted_hardest_target_k1": float(np.mean(1.0 / u_target_min_vals)),
        "bound_targeted_easiest_target_k1": float(np.mean(1.0 / u_target_max_vals)),
        "bound_targeted_avg_target_k1": float(np.mean(1.0 / u_target_mean_vals)),
    }

    print("\n=== summary (Theorem 1/2 bound at k=1, i.e. uniform single-path detection) ===")
    print(f"  layer {args.layer}, width {width}, {len(rows)} queries")
    print(f"  |U_any| (untargeted)                 : mean {summary['u_any_mean']:.1f}  "
          f"-> bound {summary['bound_untargeted_k1']:.5f}")
    print(f"  |U_target| easiest target for attacker: mean {summary['u_target_easiest_mean']:.1f}  "
          f"-> bound {summary['bound_targeted_easiest_target_k1']:.5f}")
    print(f"  |U_target| averaged over targets      : mean {summary['u_target_avg_mean']:.1f}  "
          f"-> bound {summary['bound_targeted_avg_target_k1']:.5f}")
    print(f"  |U_target| hardest target for attacker: mean {summary['u_target_hardest_mean']:.1f}  "
          f"-> bound {summary['bound_targeted_hardest_target_k1']:.5f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "targeted_vs_untargeted.json", {"summary": summary, "rows": rows})
    print(f"\nwrote {RESULTS / 'targeted_vs_untargeted.json'}")


if __name__ == "__main__":
    main()
