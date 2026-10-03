"""Smarter path sampling, and why it does not stop the attack (report, Sec. 3.3 and 4.2).

Blocks
------
1. **Defence sweep.**  Four samplers -- uniform, static importance, local
   contribution, gradient saliency -- against two adversaries: the naive
   single-neuron tamper, and one that adapts its choice of neuron to the sampler.
2. **The zero blind spot.**  Contribution weighting gives weight zero to a zero
   activation; checked against the real protocol rather than only the analysis.

Usage
-----
    python experiments/3_sampling_fixes/run.py [--queries N]

Writes ``artifacts/results/defence.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.data import load_classification
from pvi.defences import (
    GradientSaliencySampler,
    LocalContributionSampler,
    StaticImportanceSampler,
    weight_magnitude_importance,
)
from pvi.defences.adaptive import plan_adaptive_stealthy_flip, plan_evasive_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ModelCommitment, ProtocolParams, Prover, UniformPathSampler, Verifier, run_protocol
from pvi.results import write_json
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

LAYER = 1
PERCENTILE = 99.0


def build_samplers(network) -> dict:
    importance = {
        index: weight_magnitude_importance(network, index)
        for index, _ in enumerate(network.architecture.layers)
        if index < len(network.architecture) - 1
    }
    return {
        "uniform": UniformPathSampler(),
        "static-importance": StaticImportanceSampler(importance),
        "local-contribution": LocalContributionSampler(network),
        "gradient-saliency": GradientSaliencySampler(network),
    }


# --------------------------------------------------------------------------- #
# 1. Defence sweep
# --------------------------------------------------------------------------- #


def block_sweep(network, queries, params, n_queries: int) -> dict:
    print("\n[1/2] defence sweep: samplers against adversaries")
    samplers = build_samplers(network)
    out: dict[str, dict] = {}

    for name, sampler in samplers.items():
        naive, evasive = [], []
        for index in range(n_queries):
            honest = network.eval_trace(queries[index])

            plan = plan_single_neuron_flip(network, honest, layer_index=LAYER)
            if plan is not None:
                forged = apply_plan(network, honest, plan)
                naive.append(acceptance_probability(network, forged, params, sampler))

            search = plan_evasive_flip(
                network, honest, layer_index=LAYER, sampler=sampler, params=params,
                candidate_neurons=60,
            )
            if search.succeeded:
                evasive.append(search.acceptance)

        row = {
            "detection_naive": 1.0 - float(np.mean(naive)),
            "detection_evasive": 1.0 - float(np.mean(evasive)),
        }
        out[name] = row
        print(
            f"      {name:<20} naive {row['detection_naive']:.5f}  "
            f"evasive {row['detection_evasive']:.5f}"
        )
    return out


# --------------------------------------------------------------------------- #
# 2. The zero blind spot
# --------------------------------------------------------------------------- #


def block_zero_blind_spot(network, queries, ceilings, params, n_queries: int) -> dict:
    print("\n[2/2] the zero blind spot")
    sampler = LocalContributionSampler(network)
    natural = network.forward_batch(queries[:3000])[LAYER]

    commitment = ModelCommitment(network)
    prover = Prover(network, commitment, params, sampler=sampler)
    verifier = Verifier.from_commitment(commitment, params, sampler=sampler)

    all_zero, supports, contrib_acc, zero_fracs = [], [], [], []
    protocol_runs, protocol_accepts = 0, 0

    for index in range(min(n_queries, 40)):
        honest = network.eval_trace(queries[index])
        search = plan_adaptive_stealthy_flip(
            network, honest, layer_index=LAYER, ceilings=ceilings,
            sampler=sampler, params=params,
        )
        if not search.succeeded:
            continue
        plan = search.plan
        forged = apply_plan(network, honest, plan)
        all_zero.append(all(value == 0.0 for value in plan.new_values))
        supports.append(plan.size)
        contrib_acc.append(acceptance_probability(network, forged, params, sampler))
        zero_fracs.append(float((natural[:, list(plan.neurons)] == 0).mean()))

        for _ in range(50):
            protocol_runs += 1
            protocol_accepts += run_protocol(
                prover, verifier, queries[index], trace=forged
            ).accepted

    result = {
        "fraction_of_plans_using_only_zeros": float(np.mean(all_zero)),
        "mean_support": float(np.mean(supports)),
        "detection_contribution_weighted": 1.0 - float(np.mean(contrib_acc)),
        "natural_zero_fraction_at_chosen_neurons": float(np.mean(zero_fracs)),
        "protocol_runs": protocol_runs,
        "protocol_accepts": protocol_accepts,
    }
    print(
        f"      plans using only zeroed activations: "
        f"{result['fraction_of_plans_using_only_zeros']:.0%}, mean support "
        f"{result['mean_support']:.1f}/{network.architecture[LAYER].n_neurons}"
    )
    print(
        f"      detection under contribution weighting: "
        f"{result['detection_contribution_weighted']:.6f}"
    )
    print(
        f"      real protocol under contribution weighting: "
        f"{protocol_accepts}/{protocol_runs} accepted"
    )
    print(
        f"      chosen neurons are naturally zero on "
        f"{result['natural_zero_fraction_at_chosen_neurons']:.1%} of natural inputs"
    )
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=40)
    args = parser.parse_args()

    architecture = mlp_architecture_for(10)
    network = load_network(architecture, MODELS / "mlp_mnist_full.npz")
    dataset = load_classification("mnist")
    queries = np.ascontiguousarray(
        dataset.test_x.reshape(len(dataset.test_x), -1), dtype=np.float32
    )
    params = ProtocolParams(n_paths=1)
    ceilings = np.percentile(
        network.forward_batch(queries[:3000])[LAYER], PERCENTILE, axis=0
    )

    width = architecture[LAYER].n_neurons
    print(f"model {architecture.layer_widths}, tampering layer {LAYER} (width {width})")
    started = time.perf_counter()

    results = {
        "config": vars(args),
        "layer": LAYER,
        "layer_width": width,
        "sweep": block_sweep(network, queries, params, args.queries),
        "zero_blind_spot": block_zero_blind_spot(
            network, queries, ceilings, params, args.queries
        ),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "defence.json", results)
    print(f"\ndone in {time.perf_counter() - started:.0f}s")
    print(f"wrote {RESULTS / 'defence.json'}")


if __name__ == "__main__":
    main()
