"""Revision-1, Phase 4: the batched per-layer check against the real attacks.

``protocol/batched.py`` is the candidate defence: one Ligero-style commitment
opening per layer instead of per-node Merkle row openings, plus a sign witness
on the zero set that closes Theorem 3's blind spot in its arithmetic form.

This script measures it against the *actual* attacks this repository already
implements, on the real trained MNIST MLP -- not a synthetic tamper:

1. the naive single-neuron flip (``plan_single_neuron_flip``),
2. the stealthy envelope-confined flip (``plan_stealthy_flip``),
3. the adaptive zero-hiding backdoor (``plan_adaptive_stealthy_flip`` against
   ``LocalContributionSampler``) -- the one that achieves detection exactly
   ``0.000000`` under contribution weighting and ~0.03 under uniform sampling.

and reports honest completeness plus proof size against the sampling baseline.

Usage
-----
    python scripts/run_batched_defence.py [--queries N] [--layer L] [--scale-bits B]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip, plan_stealthy_flip
from pvi.data import load_classification
from pvi.defences import LocalContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ProtocolParams
from pvi.protocol.batched import (
    BatchedWeightCommitment,
    fixed_point_layer,
    prove_layer,
    verify_layer,
)
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"


def honest_view(network, layer_index, trace, scale_bits):
    """The honest fixed-point execution of this layer.

    The committed model *is* the fixed-point model, so the honest trace is the
    integer execution -- not the float trace rounded, which would differ by
    quantisation and fail completeness for the wrong reason.
    """
    layer = network.architecture[layer_index]
    weight, bias = network.parameters[layer.name]
    return fixed_point_layer(
        weight, bias, trace[layer_index - 1], scale_bits=scale_bits
    )


def tampered_view(network, layer_index, trace, plan, scale_bits):
    """The honest fixed-point layer with the attack's tamper applied to it."""
    layer = network.architecture[layer_index]
    weight, bias = network.parameters[layer.name]
    honest = honest_view(network, layer_index, trace, scale_bits)
    outputs = np.array(honest.pre_activations(), dtype=np.int64)
    outputs = np.maximum(outputs, 0)
    scale = float(1 << (2 * scale_bits))
    for neuron, value in zip(plan.neurons, plan.new_values):
        outputs[int(neuron)] = int(round(float(value) * scale))
    return fixed_point_layer(
        weight, bias, trace[layer_index - 1],
        activations_out=outputs, scale_bits=scale_bits,
    )


def run_check(view, commitment, n_queries):
    proof = prove_layer(view, commitment, n_queries=n_queries)
    accepted = verify_layer(
        proof, commitment.digest, commitment.params, commitment,
        view.inputs, view.outputs, prime=view.prime, n_queries=n_queries,
    )
    return accepted, proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=8)
    parser.add_argument("--code-queries", type=int, default=24)
    args = parser.parse_args()

    network = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_full.npz")
    dataset = load_classification(name="mnist", seed=0)
    train_x = dataset.train_x.reshape(len(dataset.train_x), -1)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    params = ProtocolParams()
    width = network.architecture[args.layer].n_neurons

    layer = network.architecture[args.layer]
    weight, bias = network.parameters[layer.name]
    reference = fixed_point_layer(
        weight, bias, network.eval_trace(test_x[0])[args.layer - 1],
        scale_bits=args.scale_bits,
    )
    started = time.perf_counter()
    commitment = BatchedWeightCommitment(reference.matrix, prime=reference.prime)
    commit_seconds = time.perf_counter() - started

    print(f"layer {args.layer}, width {width}, scale 2^{args.scale_bits}, "
          f"field prime {reference.prime}")
    print(f"commitment: {commitment.n_columns} columns, built once in {commit_seconds:.1f}s\n")

    # --- honest completeness -------------------------------------------------
    honest_accepts, proof_sizes = 0, []
    for query in test_x[: args.queries]:
        trace = network.eval_trace(query)
        view = honest_view(network, args.layer, trace, args.scale_bits)
        accepted, proof = run_check(view, commitment, args.code_queries)
        honest_accepts += bool(accepted)
        proof_sizes.append(proof.size_bytes())
    completeness = honest_accepts / max(args.queries, 1)
    print(f"honest completeness: {completeness:.6f} ({honest_accepts}/{args.queries}), "
          f"mean proof {np.mean(proof_sizes) / 1024:.1f} kB")

    # --- the three attacks ---------------------------------------------------
    ceilings = calibrate_activation_ceilings(
        network, train_x[:3000], layer_index=args.layer, percentile=100.0
    )
    sampler = LocalContributionSampler(network)

    def planned(kind, trace):
        if kind == "naive":
            return plan_single_neuron_flip(network, trace, layer_index=args.layer)
        if kind == "stealthy":
            return plan_stealthy_flip(
                network, trace, layer_index=args.layer, ceilings=ceilings
            )
        search = plan_adaptive_stealthy_flip(
            network, trace, layer_index=args.layer, ceilings=ceilings,
            sampler=sampler, params=params,
        )
        return search.plan if search.succeeded else None

    rows = {}
    for kind in ("naive", "stealthy", "zero_hiding"):
        caught, total, uniform_det, contrib_det, supports = 0, 0, [], [], []
        for query in test_x[: args.queries]:
            trace = network.eval_trace(query)
            plan = planned(kind, trace)
            if plan is None:
                continue
            forged = apply_plan(network, trace, plan)
            if int(forged.output.argmax()) == int(trace.output.argmax()):
                continue
            total += 1
            supports.append(plan.size)
            view = tampered_view(network, args.layer, trace, plan, args.scale_bits)
            accepted, _ = run_check(view, commitment, args.code_queries)
            caught += not accepted
            uniform_det.append(1.0 - acceptance_probability(network, forged, params))
            contrib_det.append(
                1.0 - acceptance_probability(network, forged, params, sampler)
            )
        rows[kind] = {
            "n_traces": total,
            "mean_support": float(np.mean(supports)) if supports else float("nan"),
            "batched_detection": caught / max(total, 1),
            "uniform_sampling_detection": float(np.mean(uniform_det)) if uniform_det else float("nan"),
            "contribution_sampling_detection": float(np.mean(contrib_det)) if contrib_det else float("nan"),
        }
        r = rows[kind]
        print(f"\n{kind} ({total} traces, mean support {r['mean_support']:.1f}):")
        print(f"  batched per-layer check : {r['batched_detection']:.6f}")
        print(f"  uniform path sampling   : {r['uniform_sampling_detection']:.6f}")
        print(f"  contribution weighting  : {r['contribution_sampling_detection']:.6f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer": args.layer,
        "width": width,
        "scale_bits": args.scale_bits,
        "prime": reference.prime,
        "code_queries": args.code_queries,
        "commit_seconds": commit_seconds,
        "honest_completeness": completeness,
        "mean_proof_bytes": float(np.mean(proof_sizes)),
        "attacks": rows,
    }
    with open(RESULTS / "batched_defence.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {RESULTS / 'batched_defence.json'}")


if __name__ == "__main__":
    main()
