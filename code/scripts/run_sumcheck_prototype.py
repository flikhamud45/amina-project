"""Revision-1, Phase 2 (stretch goal): a per-layer random-linear-combination check.

Every candidate in Phase 0/1 tried to choose *which* neurons to check more
cleverly. All of them failed against an adversary that knows the rule
(DEFENCE_NOTES.md Section 5's corollary: any trace-independent-by-construction
budget of k << N openings caps detection at k/|U|). The only way left to beat
that ceiling without full SNARK cost is to change *what a single check
verifies*, not how neurons are picked.

The idea: instead of opening one node's weight row and recomputing its local
relation, draw a random challenge vector `r` over the *whole* layer and check
one scalar,

    C = sum_j r_j * (a~_j - phi(sum_i w_ij a~_i))

which is exactly zero if every local relation in the layer holds, and is a
uniformly-random-projection of the (nonzero) residual vector otherwise --
caught with overwhelming probability regardless of *where* in the layer the
inconsistency sits, by simple linear algebra (a random linear functional
applied to a fixed nonzero vector is zero with probability 0 for continuous
challenges, or <= 1/|F| for challenges drawn from a size-|F| field). This is
NOT the classic sumcheck protocol (that reduces a claimed sum over a large
Boolean hypercube to O(log n) rounds via a genuine interactive/succinct
argument); it is the trivial special case where the "sum" is already a
concrete, small vector, and there's nothing to compress *yet*.

**What this script honestly measures, and what it does not.** It measures the
SOUNDNESS gain, assuming the prover can prove the value of `C` in O(1) --
i.e. assuming a succinct proof for "this random linear combination of a
whole layer's affine-plus-ReLU relations equals this claimed scalar" already
exists. It does NOT build that proof. As implemented, computing `C` still
means the verifier holds every weight and every claimed activation of the
layer -- the same total information as checking every node individually, so
by itself this buys NOTHING over just checking all N nodes and taking their
logical AND. The entire value of a real construction here is a genuine
succinct argument (a sumcheck-protocol-style reduction over a low-degree
extension of the weight matrix and activation vector, or a polynomial
commitment such as KZG) that lets the verifier accept `C`'s value with
O(log N) or O(1) communication instead of opening every row. That
construction is real cryptographic engineering -- pairing-based commitments
or a multi-round sumcheck -- and is out of scope for a numpy prototype. This
script answers "is the soundness worth building that for", not "here is the
protocol".

Usage
-----
    python scripts/run_sumcheck_prototype.py [--layer L] [--trials N]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.data import load_classification
from pvi.defences import LocalContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.nn.architecture import apply_activation
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol import ProtocolParams
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"


def layer_residual(network: TracedNetwork, trace: Trace, layer_index: int) -> np.ndarray:
    """a~_j - phi(sum_i w_ij a~_i) for every j in the layer, using the REAL weights.

    This is the idealised-verifier step: it assumes access to the full weight
    matrix of the layer, which a real verifier never holds -- see the module
    docstring.
    """
    layer = network.architecture[layer_index]
    weight, bias = network.parameters[layer.name]
    parents = trace[layer_index - 1].astype(np.float64)
    recomputed = apply_activation(
        parents @ weight.astype(np.float64).T + bias.astype(np.float64), layer.activation
    )
    claimed = trace[layer_index].astype(np.float64)
    return claimed - np.asarray(recomputed, dtype=np.float64)


def batched_combination(residual: np.ndarray, rng: np.random.Generator) -> float:
    r = rng.standard_normal(len(residual))
    return float(r @ residual)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=30)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    params = ProtocolParams()

    width = network.architecture[args.layer].n_neurons
    print(f"layer {args.layer}, width {width}\n")

    # --- calibrate: how large is C on HONEST traces, purely from float noise? ---
    honest_traces = [network.eval_trace(q) for q in test_x[: args.queries]]
    honest_c = np.array([
        batched_combination(layer_residual(network, t, args.layer), rng)
        for t in honest_traces for _ in range(args.trials // args.queries)
    ])
    tau = 10.0 * float(np.abs(honest_c).max())
    print(f"honest |C|: mean {np.abs(honest_c).mean():.3e}, max {np.abs(honest_c).max():.3e} "
          f"-> tolerance tau = {tau:.3e} (10x headroom, {args.trials} trials)")
    completeness = float((np.abs(honest_c) <= tau).mean())
    print(f"honest completeness at this tau: {completeness:.6f}\n")

    # --- naive single-neuron tamper ---
    naive_forged = []
    for q in test_x[: args.queries]:
        honest = network.eval_trace(q)
        plan = plan_single_neuron_flip(network, honest, layer_index=args.layer)
        if plan is not None:
            naive_forged.append(apply_plan(network, honest, plan))

    naive_c = np.array([
        batched_combination(layer_residual(network, t, args.layer), rng)
        for t in naive_forged for _ in range(max(args.trials // max(len(naive_forged), 1), 1))
    ])
    naive_detect_sumcheck = float((np.abs(naive_c) > tau).mean())
    naive_detect_uniform = 1.0 - float(np.mean(
        [acceptance_probability(network, t, params) for t in naive_forged]
    ))
    print(f"naive single-neuron tamper ({len(naive_forged)} traces):")
    print(f"  batched per-layer check detection: {naive_detect_sumcheck:.6f}")
    print(f"  uniform single-path sampling detection: {naive_detect_uniform:.6f}\n")

    # --- the 13-neuron zero-hiding backdoor attack from Theorem 3 ---
    ceilings = calibrate_activation_ceilings(network, train_x[:3000], layer_index=args.layer, percentile=100.0)
    sampler = LocalContributionSampler(network)
    zero_forged = []
    for q in test_x[: args.queries]:
        honest = network.eval_trace(q)
        search = plan_adaptive_stealthy_flip(
            network, honest, layer_index=args.layer, ceilings=ceilings,
            sampler=sampler, params=params,
        )
        if search.succeeded:
            zero_forged.append(apply_plan(network, honest, search.plan))

    zero_c = np.array([
        batched_combination(layer_residual(network, t, args.layer), rng)
        for t in zero_forged for _ in range(max(args.trials // max(len(zero_forged), 1), 1))
    ])
    zero_detect_sumcheck = float((np.abs(zero_c) > tau).mean())
    zero_detect_contribution = 1.0 - float(np.mean(
        [acceptance_probability(network, t, params, sampler) for t in zero_forged]
    ))
    zero_detect_uniform = 1.0 - float(np.mean(
        [acceptance_probability(network, t, params) for t in zero_forged]
    ))
    print(f"13-neuron zero-hiding backdoor ({len(zero_forged)} traces, the exact\n"
          f"attack that achieves 0.000000 detection under contribution weighting):")
    print(f"  batched per-layer check detection:      {zero_detect_sumcheck:.6f}")
    print(f"  contribution-weighted path sampling:    {zero_detect_contribution:.6f}")
    print(f"  uniform single-path sampling:            {zero_detect_uniform:.6f}")


if __name__ == "__main__":
    main()
