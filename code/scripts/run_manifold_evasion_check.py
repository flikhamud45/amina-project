"""Revision-1, Phase 0/1: the adaptive counter to the joint-plausibility check.

``run_joint_plausibility_check.py`` found real signal: a PCA plausibility model
on the whole claimed layer vector catches 35.6% of the *existing* zero-hiding
attack at a 5% false-positive rate. But that attack was not built to evade this
check -- it was built to evade ``LocalContributionSampler``.

The mathematical counter is immediate: a perturbation confined to the PCA
subspace produces *zero* extra reconstruction error, by construction, however
large. The open question is whether this can be done with a *small* support
(few neurons touched), since a small support is what makes the perturbation
cheap to hide from ordinary path sampling too (Theorem 1: detection ~ k/N).
Confining a perturbation to a low-rank subspace generally forces it to be
*dense* (most natural, correlated directions involve many neurons at once) --
so the question is quantitative: how much support does manifold-invisibility
actually cost?

Method: for a fixed support size k, choose the k neurons ranked by margin
sensitivity (same ranking ``plan_single_neuron_flip`` uses), then solve the
*exact* small QP for the perturbation on just those k coordinates that flips
the linearised margin by a target amount while minimising the *added* PCA
reconstruction error (closed-form equality-constrained least squares -- no
external solver needed). Verify the flip against the real (nonlinear) forward
pass, growing the margin target if the linear approximation undershoots.
Sweep k and report the resulting (support size, reconstruction error) frontier.

Usage
-----
    python scripts/run_manifold_evasion_check.py [--layer L] [--rank K]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvi.data import load_classification
from pvi.nn.architecture import DenseLayer
from pvi.nn.gradients import output_direction_gradient
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"

SUPPORT_SIZES = [1, 2, 3, 5, 8, 13, 20, 35, 50, 80]


def fit_pca(network, calib_queries: np.ndarray, layer_index: int, rank: int):
    natural = network.forward_batch(calib_queries)[layer_index].astype(np.float64)
    mean = natural.mean(axis=0)
    _, s, vt = np.linalg.svd(natural - mean, full_matrices=False)
    basis = vt[:rank]
    width = natural.shape[1]
    projector = np.eye(width) - basis.T @ basis  # onto the orthogonal complement
    return mean, projector


def solve_min_residual_delta(
    projector: np.ndarray,
    residual: np.ndarray,
    support: np.ndarray,
    gradient: np.ndarray,
    target_margin: float,
    ridge: float = 1e-6,
) -> np.ndarray:
    """delta (on `support` only) minimising ||projector @ delta + residual||^2
    subject to gradient[support]^T delta = target_margin. Closed-form KKT."""
    a = projector[:, support]
    g = gradient[support]
    h = a.T @ a + ridge * np.eye(len(support))
    rhs = np.stack([a.T @ residual, g], axis=1)  # (k, 2)
    solved = np.linalg.solve(h, rhs)
    delta_u, h_inv_g = solved[:, 0], solved[:, 1]
    delta_u = -delta_u
    denom = g @ h_inv_g
    t = (target_margin - g @ delta_u) / denom if abs(denom) > 1e-12 else 0.0
    delta = delta_u + t * h_inv_g
    full = np.zeros(projector.shape[0])
    full[support] = delta
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--calibration", type=int, default=5000)
    parser.add_argument("--test-queries", type=int, default=30)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)

    mean, projector = fit_pca(network, train_x[: args.calibration], args.layer, args.rank)
    width = network.architecture[args.layer].n_neurons

    honest_natural = network.forward_batch(test_x[: args.test_queries])[args.layer].astype(np.float64)
    honest_errors = np.linalg.norm((honest_natural - mean) @ projector.T, axis=-1)
    print(f"layer {args.layer} (width {width}), rank-{args.rank} PCA\n"
          f"honest reconstruction error: mean {honest_errors.mean():.4f}, "
          f"p95 {np.percentile(honest_errors, 95):.4f}\n")

    print(f"{'support k':>10} | {'flip rate':>9} | {'mean recon. error':>18} | "
          f"{'vs honest p95':>14} | {'uniform exposure k/N':>20}")
    print("-" * 85)

    layer = network.architecture[args.layer]
    assert isinstance(layer, DenseLayer)

    for k in SUPPORT_SIZES:
        errors, flips = [], []
        for query in test_x[: args.test_queries]:
            honest = network.eval_trace(query)
            honest_class = int(honest.output.argmax())
            logits = honest.output
            target_class = int(np.argsort(logits)[::-1][1])
            direction = np.zeros_like(logits, dtype=np.float64)
            direction[target_class] = 1.0
            direction[honest_class] -= 1.0
            gradient = output_direction_gradient(network, honest, args.layer, direction)

            support = np.argsort(np.abs(gradient))[::-1][:k]
            a_honest = honest[args.layer].astype(np.float64)
            residual = (a_honest - mean) @ projector.T

            margin_needed = float(logits[honest_class] - logits[target_class]) + 1e-3
            scale = 1.0
            flipped = False
            for _ in range(12):
                delta = solve_min_residual_delta(
                    projector, residual, support, gradient, margin_needed * scale,
                )
                trial = honest
                for neuron in support:
                    value = float(a_honest[neuron] + delta[neuron])
                    if layer.activation == "relu":
                        value = max(value, 0.0)
                    trial = trial.tampered(args.layer, int(neuron), value)
                forged = network.forward_from(trial, args.layer)
                if int(forged.output.argmax()) == target_class:
                    flipped = True
                    break
                scale *= 1.7

            if flipped:
                final_activation = forged[args.layer].astype(np.float64)
                error = float(np.linalg.norm((final_activation - mean) @ projector.T))
                errors.append(error)
            flips.append(flipped)

        flip_rate = float(np.mean(flips))
        mean_error = float(np.mean(errors)) if errors else float("nan")
        p95 = np.percentile(honest_errors, 95)
        print(f"{k:>10} | {flip_rate:>9.2%} | {mean_error:>18.4f} | "
              f"{'BELOW p95' if mean_error < p95 else 'above p95':>14} | {k / width:>20.4f}")


if __name__ == "__main__":
    main()
