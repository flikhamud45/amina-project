"""Revision-1, Phase 0/1: does a JOINT (not per-neuron) plausibility check help?

Every sampler tried so far (``LocalContributionSampler``, ``ZeroAwareContributionSampler``,
``GradientSaliencySampler``, and the informal "cumulative path-sensitivity" idea)
scores a candidate parent using only *that neuron's own* claimed value and weight.
``DEFENCE_NOTES.md``'s Theorem 4 argues, and this script's sibling experiments
confirm, that any such per-neuron rule cannot distinguish a maliciously-set zero
from a naturally-occurring one, because they are literally the same value and a
natural zero is common (ReLU sparsity).

A genuinely different mechanism: check the *joint* claimed activation vector of a
whole layer against a plausibility model calibrated on many honest executions
(cheap and one-time, like ``expected_saliency_importance``'s calibration set) --
e.g. a low-rank PCA subspace fit to natural layer activations, using
reconstruction error as an anomaly score. This uses only information the layer's
own commitment leaf already reveals (the *entire* claimed activation vector, per
SPEC_NOTES.md Section 5 -- ``C_trc`` commits one leaf per layer, not per neuron),
so it costs nothing extra to open. The question this script answers: does the
existing zero-hiding attack (Theorem 3's exploit) actually look anomalous under a
*joint* model, even though it is invisible to every *marginal*/per-neuron check?

This is a first cheap look, not yet an adaptive-attacker test: it checks whether
the mechanism has any signal at all before spending effort wiring it into the
adaptive evasion search.

Usage
-----
    python scripts/run_joint_plausibility_check.py [--layer L] [--rank K]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import apply_plan
from pvi.data import load_classification
from pvi.defences import LocalContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip
from pvi.protocol import ProtocolParams
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--calibration", type=int, default=5000)
    parser.add_argument("--test-queries", type=int, default=60)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    params = ProtocolParams()

    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)

    # --- calibrate a low-rank plausibility model on natural activations only ---
    calib_queries = train_x[: args.calibration]
    natural = network.forward_batch(calib_queries)[args.layer].astype(np.float64)
    mean = natural.mean(axis=0)
    centered = natural - mean
    # Economy SVD: the natural activation manifold's top-`rank` directions.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: args.rank]  # (rank, width)

    def reconstruction_error(vectors: np.ndarray) -> np.ndarray:
        centered_v = vectors - mean
        coeffs = centered_v @ basis.T
        reconstructed = coeffs @ basis
        return np.linalg.norm(centered_v - reconstructed, axis=-1)

    width = network.architecture[args.layer].n_neurons
    explained = (s[: args.rank] ** 2).sum() / (s**2).sum()
    print(f"layer {args.layer} (width {width}), rank-{args.rank} PCA from "
          f"{len(calib_queries)} calibration queries, explains {explained:.4%} of variance\n")

    # --- honest reconstruction error, on FRESH (non-calibration) natural queries ---
    honest_test = test_x[: args.test_queries]
    honest_activations = network.forward_batch(honest_test)[args.layer].astype(np.float64)
    honest_errors = reconstruction_error(honest_activations)

    # --- the existing zero-hiding attack's forged activations ---
    ceilings = calibrate_activation_ceilings(network, train_x[:3000], layer_index=args.layer, percentile=100.0)
    sampler = LocalContributionSampler(network)
    forged_errors = []
    supports = []
    for q in honest_test:
        honest = network.eval_trace(q)
        search = plan_adaptive_stealthy_flip(
            network, honest, layer_index=args.layer, ceilings=ceilings,
            sampler=sampler, params=params,
        )
        if not search.succeeded:
            continue
        forged = apply_plan(network, honest, search.plan)
        forged_errors.append(float(reconstruction_error(forged[args.layer][None, :])[0]))
        supports.append(search.plan.size)

    forged_errors = np.array(forged_errors)
    print(f"honest reconstruction error:  mean {honest_errors.mean():.4f}  "
          f"p50 {np.median(honest_errors):.4f}  p99 {np.percentile(honest_errors, 99):.4f}  "
          f"max {honest_errors.max():.4f}")
    print(f"forged (zero-hiding) error:   mean {forged_errors.mean():.4f}  "
          f"p50 {np.median(forged_errors):.4f}  min {forged_errors.min():.4f}  "
          f"max {forged_errors.max():.4f}  (n={len(forged_errors)}, mean support {np.mean(supports):.1f})")

    # How well would a threshold set from honest data alone catch these forgeries?
    for pct in (95.0, 99.0, 99.9, 100.0):
        threshold = np.percentile(honest_errors, pct)
        caught = float((forged_errors > threshold).mean())
        print(f"  threshold at honest p{pct:>5}: {threshold:.4f}  -> "
              f"catches {caught:.1%} of forged traces "
              f"(false-positive rate on honest: {100.0 - pct:.2f}%)")


if __name__ == "__main__":
    main()
