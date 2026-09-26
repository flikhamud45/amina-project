"""Revision-1, Phase 3: do the findings hold at 8x the width?

Every theorem and measurement in `DEFENCE_NOTES.md` is stated in terms of a
layer's width `N`, never assuming MNIST-MLP scale specifically. This script
is the check: train (or load, if already cached) `LARGE_MLP_SPEC`
(``784 -> 4096 -> 2048 -> 10``, 8x `mlp_architecture_for`'s hidden widths) and
re-run the two key Revision-1 measurements against it --

1. the Theorem 4 epsilon-sweep (does any additive floor on contribution
   weighting ever beat uniform's exact minimax bound?), and
2. the per-layer sumcheck-style prototype (does the idealised batched check
   still catch a single-neuron tamper with certainty?)

-- to confirm the qualitative conclusions are a property of the *mechanism*,
not an artefact of testing at N=512.

This is CPU-trainable but slow at this width; pass ``--device cuda`` (or
``auto``) once GPU access is available. Without one, this script still runs
end to end on CPU, just slowly -- use ``--epochs`` and ``--max-train`` to
smoke-test the code path quickly rather than fully training.

Usage
-----
    python scripts/run_gpu_scale_check.py [--device auto] [--force]
                                           [--epochs N] [--max-train N]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import _smallest_flipping_value, apply_plan, plan_single_neuron_flip
from pvi.data import load_classification
from pvi.defences import LocalContributionSampler, ZeroAwareContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip, visit_probabilities_under
from pvi.experiments.analysis import acceptance_probability
from pvi.nn.architecture import apply_activation
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol import ProtocolParams
from pvi.training import load_network, save_network, train_network
from pvi.zoo import LARGE_MLP_SPEC

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"

EPSILONS = [0.0, 0.1, 1.0, 10.0, 100.0]


def layer_residual(network: TracedNetwork, trace: Trace, layer_index: int) -> np.ndarray:
    layer = network.architecture[layer_index]
    weight, bias = network.parameters[layer.name]
    parents = trace[layer_index - 1].astype(np.float64)
    recomputed = apply_activation(
        parents @ weight.astype(np.float64).T + bias.astype(np.float64), layer.activation
    )
    claimed = trace[layer_index].astype(np.float64)
    return claimed - np.asarray(recomputed, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--force", action="store_true", help="retrain even if cached")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count")
    parser.add_argument("--max-train", type=int, default=None, help="cap training set size")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    path = MODELS / f"{LARGE_MLP_SPEC.name}.npz"
    if path.exists() and not args.force:
        print(f"[load] {path}")
        network = load_network(LARGE_MLP_SPEC.architecture(), path)
    else:
        dataset = LARGE_MLP_SPEC.dataset()
        if args.max_train:
            dataset = dataclasses.replace(
                dataset,
                train_x=dataset.train_x[: args.max_train],
                train_y=dataset.train_y[: args.max_train],
            )
        config = dataclasses.replace(LARGE_MLP_SPEC.train_config, device=args.device)
        if args.epochs is not None:
            config = dataclasses.replace(config, epochs=args.epochs)
        print(f"[train] {LARGE_MLP_SPEC.name} on {config.device}, "
              f"{len(dataset.train_x)} examples, {config.epochs} epochs")
        network, report = train_network(LARGE_MLP_SPEC.architecture(), dataset, config)
        save_network(network, path, report)

    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    params = ProtocolParams()
    width = network.architecture[args.layer].n_neurons
    print(f"\nlayer {args.layer}, width {width} "
          f"({width // 512}x the width Theorem 4 was originally measured at)\n")

    # --- 1. Theorem 4's exact minimax sweep, at 8x the width ---
    print("=== Theorem 4 sweep (exact minimax, single query) ===")
    honest0 = network.eval_trace(test_x[0])
    usable = [
        n for n in range(width)
        if _smallest_flipping_value(
            network, honest0, args.layer, n, int(honest0.output.argmax()), None,
            max_value=1e4, require_nonnegative=True,
        ) is not None
    ]
    usable_array = np.asarray(usable)
    bound = 1.0 / max(len(usable), 1)
    print(f"|U| = {len(usable)} / {width}, uniform bound 1/|U| = {bound:.6f}")
    for epsilon in EPSILONS:
        sampler = ZeroAwareContributionSampler(network, epsilon=epsilon)
        visits = visit_probabilities_under(network, honest0, args.layer, sampler)
        restricted = visits[usable_array] if len(usable_array) else visits
        beats = restricted.min() > bound
        print(f"  epsilon={epsilon:>7.3g}  min_v q(v) = {restricted.min():.6f}  "
              f"{'BEATS' if beats else 'does not beat'} uniform")

    # --- 2. sumcheck-style prototype, at 8x the width ---
    print("\n=== sumcheck-style prototype (idealised, single-layer batch) ===")
    honest_traces = [network.eval_trace(q) for q in test_x[: args.queries]]
    honest_c = np.array([
        rng.standard_normal(width) @ layer_residual(network, t, args.layer)
        for t in honest_traces
    ])
    tau = 10.0 * float(np.abs(honest_c).max())
    print(f"honest |C|: max {np.abs(honest_c).max():.3e} -> tau = {tau:.3e}")

    naive_forged = []
    for q in test_x[: args.queries]:
        honest = network.eval_trace(q)
        plan = plan_single_neuron_flip(network, honest, layer_index=args.layer)
        if plan is not None:
            naive_forged.append(apply_plan(network, honest, plan))
    naive_c = np.array([
        rng.standard_normal(width) @ layer_residual(network, t, args.layer)
        for t in naive_forged
    ])
    detect_sumcheck = float((np.abs(naive_c) > tau).mean()) if len(naive_c) else float("nan")
    detect_uniform = 1.0 - float(np.mean(
        [acceptance_probability(network, t, params) for t in naive_forged]
    )) if naive_forged else float("nan")
    print(f"naive single-neuron tamper ({len(naive_forged)} traces): "
          f"sumcheck detection {detect_sumcheck:.6f}, uniform sampling {detect_uniform:.6f}")

    print("\nConclusion so far (fill in after a real GPU run): Theorem 4 should still "
          "show no epsilon beating uniform, and the sumcheck prototype should still show "
          "~1.0 detection independent of width -- both mechanisms are architecture-size-"
          "independent by construction, so this is a confirmation, not a new finding, "
          "unless something surprising shows up.")


if __name__ == "__main__":
    main()
