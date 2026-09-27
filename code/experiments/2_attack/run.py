"""The single-neuron attack and backdoor on the original protocol (report, Table 1).

Blocks
------
1. **Baselines.**  The paper's own attacks (a substitute model, Section 7.2,
   Appendices E and F) versus ours: how many trace nodes are inconsistent, which is
   what predicts detection, and the probability that the verifier accepts.
2. **The backdoor.**  Clean-accuracy gap, attack success rate, and verifier
   acceptance on triggered queries.
3. **Opening budget.**  Acceptance of the single-neuron tamper with 250 paths.
4. **Stealth.**  The same attack with the forged activation capped at a percentile of
   the neuron's natural range.

Acceptance is computed exactly by the dynamic program in
``pvi.experiments.analysis``; the backdoor is also run on the real protocol.

Usage
-----
    python experiments/2_attack/run.py [--queries N] [--challenges N]

Writes ``artifacts/results/attack.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pvi.attacks import (
    BackdoorAdversary,
    PatchTrigger,
    calibrate_activation_ceilings,
    evaluate_plan,
    gradient_forge,
    injection_forge,
    inverse_transform_forge,
    logit_swap_target,
    plan_single_neuron_flip,
)
from pvi.attacks.tamper import plan_stealthy_flip
from pvi.data import load_classification
from pvi.experiments.analysis import acceptance_probability, inconsistent_nodes
from pvi.protocol import ModelCommitment, ProtocolParams, Prover, Verifier, run_protocol
from pvi.results import write_json
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

TARGET_CLASS = 0
OPENING_BUDGET_PATHS = 250


# --------------------------------------------------------------------------- #
# 1. Baselines
# --------------------------------------------------------------------------- #


def block_baselines(network, queries, params, n_examples: int = 8) -> dict:
    print("\n[1/4] baseline forging strategies")
    substitute = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_halfB.npz")
    rows: list[dict] = []

    for index in range(n_examples):
        query = queries[index]
        honest = network.eval_trace(query)
        target = logit_swap_target(honest.output)

        attempts = [
            inverse_transform_forge(network, query, target, params=params),
            injection_forge(network, query, target, injection_layer=2, steps=1500, params=params),
            gradient_forge(network, query, target, steps=3000, weight_decay=0.0, params=params),
        ]
        for attempt in attempts:
            rows.append(
                {
                    "attack": attempt.name,
                    "acceptance": attempt.acceptance_probability,
                    "support": attempt.n_inconsistent,
                    "n_nodes": attempt.n_nodes,
                }
            )

        # Substitute model, honest trace (plain other-model soundness).
        foreign = substitute.eval_trace(query)
        report = inconsistent_nodes(network, foreign, params)
        rows.append(
            {
                "attack": "substitute model, honest trace (Def. 1)",
                "acceptance": acceptance_probability(network, foreign, params),
                "support": report.total,
                "n_nodes": int(sum(report.per_layer_widths.values())),
            }
        )

        # Ours.
        for layer_index in (1, 2):
            plan = plan_single_neuron_flip(network, honest, layer_index=layer_index)
            if plan is None:
                continue
            outcome = evaluate_plan(network, honest, plan, params)
            rows.append(
                {
                    "attack": f"single-neuron tamper, layer {layer_index} (ours)",
                    "acceptance": outcome.acceptance_probability,
                    "support": outcome.n_inconsistent,
                    "n_nodes": outcome.n_nodes,
                }
            )

    summary: dict[str, dict] = {}
    for row in rows:
        entry = summary.setdefault(row["attack"], {"acceptance": [], "support": []})
        entry["acceptance"].append(row["acceptance"])
        entry["support"].append(row["support"])

    out: dict[str, dict] = {}
    for attack, entry in summary.items():
        out[attack] = {
            "acceptance_mean": float(np.mean(entry["acceptance"])),
            "support_mean": float(np.mean(entry["support"])),
        }
        print(
            f"      {attack:<48} acceptance {out[attack]['acceptance_mean']:.4f}  "
            f"support {out[attack]['support_mean']:.0f}/{rows[0]['n_nodes']}"
        )
    return out


# --------------------------------------------------------------------------- #
# 2. The backdoor
# --------------------------------------------------------------------------- #


def block_backdoor(network, queries, labels, params, challenges: int) -> dict:
    print("\n[2/4] trigger-conditional backdoor")
    trigger = PatchTrigger(input_shape=(1, 28, 28), size=3, value=1.0)
    adversary = BackdoorAdversary(
        network, trigger, layer_index=1, target_class=TARGET_CLASS
    )

    honest_pred = network.predict(queries)
    served = [adversary.serve(q) for q in queries]
    served_pred = np.asarray([r.served_class for r in served])

    triggered_queries = trigger.apply_batch(queries)
    triggered = [adversary.serve(q) for q in triggered_queries]
    triggered_honest = np.asarray([r.honest_class for r in triggered])
    triggered_served = np.asarray([r.served_class for r in triggered])

    # Standard backdoor convention: ASR is measured on inputs not already in the
    # target class, since for those the adversary has nothing to do.
    eligible = triggered_honest != TARGET_CLASS
    asr = float((triggered_served[eligible] == TARGET_CLASS).mean())

    commitment = ModelCommitment(network)
    prover = Prover(network, commitment, params)
    verifier = Verifier.from_commitment(commitment, params)

    trig_accepts, trig_runs, exact = 0, 0, []
    for query, response in zip(triggered_queries[:100], triggered[:100]):
        if not response.tampered:
            continue
        exact.append(acceptance_probability(network, response.trace, params))
        for _ in range(challenges):
            trig_runs += 1
            trig_accepts += run_protocol(
                prover, verifier, query, trace=response.trace
            ).accepted

    result = {
        "clean_identical_to_committed": float((honest_pred == served_pred).mean()),
        "clean_accuracy_gap": float((served_pred == labels).mean() - (honest_pred == labels).mean()),
        "attack_success_rate": asr,
        "triggered_acceptance_empirical": trig_accepts / trig_runs,
        "triggered_acceptance_exact": float(np.mean(exact)),
        "triggered_runs": trig_runs,
    }
    print(
        f"      clean: served == committed on {result['clean_identical_to_committed']:.1%} "
        f"of queries, accuracy gap {result['clean_accuracy_gap']:+.6f}"
    )
    print(f"      triggered: ASR {asr:.4f}")
    print(
        f"      verifier acceptance on triggered queries "
        f"{result['triggered_acceptance_empirical']:.4f} "
        f"(exact {result['triggered_acceptance_exact']:.4f})"
    )
    return result


# --------------------------------------------------------------------------- #
# 3. Opening budget
# --------------------------------------------------------------------------- #


def block_opening_budget(network, queries) -> dict:
    print(f"\n[3/4] acceptance with {OPENING_BUDGET_PATHS} paths")
    honest = network.eval_trace(queries[0])
    plan = plan_single_neuron_flip(network, honest, layer_index=1)
    params = ProtocolParams(n_paths=OPENING_BUDGET_PATHS)
    outcome = evaluate_plan(network, honest, plan, params)
    print(f"      {OPENING_BUDGET_PATHS} paths -> {outcome.acceptance_probability:.3f}")
    return {"n_paths": OPENING_BUDGET_PATHS, "acceptance_exact": outcome.acceptance_probability}


# --------------------------------------------------------------------------- #
# 4. Stealth
# --------------------------------------------------------------------------- #


def block_stealth(network, queries, params, n_queries: int) -> dict:
    """Can the attack survive an activation-envelope check?

    The single-neuron tamper needs a value far outside anything the neuron produces
    on natural data, so a verifier that range-checked opened activations against a
    published envelope would catch it -- a defence the protocol does not specify, but
    which this attack motivates.  The stealthy planner confines every forged value to
    the neuron's own envelope, paying a few extra inconsistent nodes for it.
    """
    print("\n[4/4] stealth: forged activations confined to natural per-neuron ranges")
    out: dict[str, dict] = {}

    natural = network.forward_batch(queries[:3000])[1]
    for percentile in (100.0, 99.9, 99.0, 95.0):
        ceilings = (
            natural.max(axis=0)
            if percentile == 100.0
            else calibrate_activation_ceilings(
                network, queries[:3000], layer_index=1, percentile=percentile
            )
        )
        supports, accepts = [], []
        for index in range(min(n_queries, 60)):
            honest = network.eval_trace(queries[index])
            plan = plan_stealthy_flip(network, honest, layer_index=1, ceilings=ceilings)
            if plan is None:
                continue
            outcome = evaluate_plan(network, honest, plan, params)
            if outcome.forged_class == outcome.honest_class:
                continue
            supports.append(outcome.n_inconsistent)
            accepts.append(outcome.acceptance_probability)
        out[f"p{percentile}"] = {
            "mean_support": float(np.mean(supports)) if supports else float("nan"),
            "acceptance_mean": float(np.mean(accepts)) if accepts else float("nan"),
        }
        print(
            f"      cap p{percentile:<5}: k={out[f'p{percentile}']['mean_support']:.1f}, "
            f"acceptance {out[f'p{percentile}']['acceptance_mean']:.4f}"
        )
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--challenges", type=int, default=25)
    args = parser.parse_args()

    architecture = mlp_architecture_for(10)
    network = load_network(architecture, MODELS / "mlp_mnist_full.npz")
    dataset = load_classification("mnist")
    queries = np.ascontiguousarray(
        dataset.test_x.reshape(len(dataset.test_x), -1), dtype=np.float32
    )
    labels = dataset.test_y
    params = ProtocolParams(n_paths=1)

    print(f"model: {architecture.layer_widths}, {len(queries)} test queries available")
    started = time.perf_counter()

    results = {
        "config": vars(args),
        "layer_widths": list(architecture.layer_widths),
        "baselines": block_baselines(network, queries, params),
        "backdoor": block_backdoor(
            network, queries[: args.queries], labels[: args.queries], params,
            args.challenges,
        ),
        "opening_budget": block_opening_budget(network, queries),
        "stealth": block_stealth(network, queries, params, min(args.queries, 100)),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "attack.json", results)
    print(f"\ndone in {time.perf_counter() - started:.0f}s")
    print(f"wrote {RESULTS / 'attack.json'}")


if __name__ == "__main__":
    main()
