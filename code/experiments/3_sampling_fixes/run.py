"""Smarter path sampling, and why it does not stop the attack (report, Sec. 3.2 and 4.2).

Blocks
------
1. **Defence sweep.**  Four samplers -- uniform, static importance, local
   contribution, gradient saliency -- against three adversaries: the naive
   single-neuron tamper, one that adapts its choice of neuron to the sampler, and
   one that additionally keeps every forged activation inside its neuron's natural
   range.
2. **The zero blind spot.**  Contribution weighting gives weight zero to a zero
   activation; checked against the real protocol rather than only the analysis.
3. **Minimax check.**  The minimax theorem (report, Theorem 3.1) predicts that a
   trace-independent sampler's worst-case detection is ``min_{v in U} q(v) <= 1/|U|``.
   We compute ``q`` and ``U`` exactly and compare.
4. **Budget curve.**  Detection against the number of weight rows opened, for each
   sampler, which is the quantity the efficiency claim is stated in.

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

from pvi.attacks.tamper import (
    apply_plan,
    plan_single_neuron_flip,
    plan_stealthy_flip,
)
from pvi.data import load_classification
from pvi.defences import (
    GradientSaliencySampler,
    LocalContributionSampler,
    StaticImportanceSampler,
    weight_magnitude_importance,
)
from pvi.defences.adaptive import (
    plan_adaptive_stealthy_flip,
    plan_evasive_flip,
    visit_probabilities_under,
)
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import (
    ModelCommitment,
    ProtocolParams,
    Prover,
    UniformPathSampler,
    Verifier,
    run_protocol,
)
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

LAYER = 1


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


def block_sweep(network, queries, ceilings, params, n_queries: int) -> dict:
    print("\n[1/4] defence sweep: samplers against adversaries")
    samplers = build_samplers(network)
    out: dict[str, dict] = {}

    for name, sampler in samplers.items():
        naive, evasive, stealthy, adaptive, supports = [], [], [], [], []
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

            stealth_plan = plan_stealthy_flip(
                network, honest, layer_index=LAYER, ceilings=ceilings
            )
            if stealth_plan is not None:
                forged = apply_plan(network, honest, stealth_plan)
                if int(forged.output.argmax()) != int(honest.output.argmax()):
                    stealthy.append(
                        acceptance_probability(network, forged, params, sampler)
                    )

            combined = plan_adaptive_stealthy_flip(
                network, honest, layer_index=LAYER, ceilings=ceilings,
                sampler=sampler, params=params,
            )
            if combined.succeeded:
                adaptive.append(combined.acceptance)
                supports.append(combined.plan.size)

        row = {
            "detection_naive": 1.0 - float(np.mean(naive)),
            "detection_evasive": 1.0 - float(np.mean(evasive)),
            "detection_stealthy": 1.0 - float(np.mean(stealthy)),
            "detection_adaptive_stealthy": 1.0 - float(np.mean(adaptive)),
            "adaptive_support_mean": float(np.mean(supports)),
        }
        out[name] = row
        print(
            f"      {name:<20} naive {row['detection_naive']:.5f}  "
            f"evasive {row['detection_evasive']:.5f}  "
            f"stealthy {row['detection_stealthy']:.5f}  "
            f"adaptive+stealthy {row['detection_adaptive_stealthy']:.5f} "
            f"(k={row['adaptive_support_mean']:.1f})"
        )
    return out


# --------------------------------------------------------------------------- #
# 2. The zero blind spot
# --------------------------------------------------------------------------- #


def block_zero_blind_spot(network, queries, ceilings, params, n_queries: int) -> dict:
    print("\n[2/4] the zero blind spot (Theorem 3)")
    sampler = LocalContributionSampler(network)
    natural = network.forward_batch(queries[:3000])[LAYER]

    commitment = ModelCommitment(network)
    prover = Prover(network, commitment, params, sampler=sampler)
    verifier = Verifier.from_commitment(commitment, params, sampler=sampler)

    all_zero, supports, contrib_acc, uniform_acc, zero_fracs = [], [], [], [], []
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
        uniform_acc.append(acceptance_probability(network, forged, params))
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
        "detection_uniform": 1.0 - float(np.mean(uniform_acc)),
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
        f"      detection: contribution-weighted "
        f"{result['detection_contribution_weighted']:.6f} vs uniform "
        f"{result['detection_uniform']:.6f}"
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
# 3. Minimax check
# --------------------------------------------------------------------------- #


def block_minimax(network, queries, params) -> dict:
    print("\n[3/4] minimax check (Theorem 2)")
    samplers = build_samplers(network)
    honest = network.eval_trace(queries[0])
    width = network.architecture[LAYER].n_neurons

    # U: neurons at which a single-node tamper flips the output.
    usable = []
    for neuron in range(width):
        from pvi.attacks.tamper import _smallest_flipping_value

        value = _smallest_flipping_value(
            network, honest, LAYER, neuron, int(honest.output.argmax()), None,
            max_value=1e4, require_nonnegative=True,
        )
        if value is not None:
            usable.append(neuron)
    usable_array = np.asarray(usable)

    out: dict[str, dict] = {"usable_neurons": int(len(usable)), "layer_width": width}
    for name, sampler in samplers.items():
        visits = visit_probabilities_under(network, honest, LAYER, sampler)
        restricted = visits[usable_array] if len(usable_array) else visits
        out[name] = {
            "min_visit_probability_over_U": float(restricted.min()),
            "mean_visit_probability": float(visits.mean()),
            "bound_one_over_U": 1.0 / max(len(usable), 1),
        }
        print(
            f"      {name:<20} min_v q(v) over U = "
            f"{out[name]['min_visit_probability_over_U']:.6f}  "
            f"(bound 1/|U| = {out[name]['bound_one_over_U']:.6f})"
        )
    print(f"      |U| = {len(usable)} of {width} neurons admit a flipping value")
    return out


# --------------------------------------------------------------------------- #
# 4. Budget curve
# --------------------------------------------------------------------------- #


def block_budget(network, queries, ceilings) -> dict:
    print("\n[4/4] detection versus weight rows opened")
    samplers = build_samplers(network)
    honest = network.eval_trace(queries[0])
    out: dict[str, dict] = {}

    plans: dict[str, object] = {}
    for name, sampler in samplers.items():
        base = ProtocolParams(n_paths=1, check_full_input=True)
        search = plan_adaptive_stealthy_flip(
            network, honest, layer_index=LAYER, ceilings=ceilings,
            sampler=sampler, params=base,
        )
        plans[name] = search.plan if search.succeeded else None

    for name, sampler in samplers.items():
        plan = plans[name]
        if plan is None:
            continue
        forged = apply_plan(network, honest, plan)
        row = {}
        for n_paths in (1, 5, 25, 100, 250, 500):
            params = ProtocolParams(n_paths=n_paths, check_full_input=True)
            row[str(n_paths)] = {
                "detection": 1.0 - acceptance_probability(network, forged, params, sampler),
                "weight_rows": n_paths * 3,
            }
        out[name] = row
        print(
            f"      {name:<20} "
            + ", ".join(f"{k}p -> {v['detection']:.4f}" for k, v in row.items())
        )
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    architecture = mlp_architecture_for(10)
    network = load_network(architecture, MODELS / "mlp_mnist_full.npz")
    dataset = load_classification("mnist")
    queries = np.ascontiguousarray(
        dataset.test_x.reshape(len(dataset.test_x), -1), dtype=np.float32
    )
    params = ProtocolParams(n_paths=1, check_full_input=True)
    ceilings = np.percentile(
        network.forward_batch(queries[:3000])[LAYER], args.percentile, axis=0
    )

    width = architecture[LAYER].n_neurons
    print(f"model {architecture.layer_widths}, tampering layer {LAYER} (width {width})")
    print(f"uniform baseline detection 1/N = {1 / width:.6f}")
    started = time.perf_counter()

    results = {
        "config": vars(args),
        "layer": LAYER,
        "layer_width": width,
        "uniform_baseline": 1.0 / width,
        "sweep": block_sweep(network, queries, ceilings, params, args.queries),
        "zero_blind_spot": block_zero_blind_spot(
            network, queries, ceilings, params, args.queries
        ),
        "minimax": block_minimax(network, queries, params),
        "budget": block_budget(network, queries, ceilings),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "defence.json", results)
    print(f"\ndone in {time.perf_counter() - started:.0f}s")
    print(f"wrote {RESULTS / 'defence.json'}")


if __name__ == "__main__":
    main()
