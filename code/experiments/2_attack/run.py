"""The single-neuron attack and backdoor on the original protocol (report, Table 1).

Blocks
------
1. **Baselines.**  The paper's own forging strategies (Sections 2, 7.2, E, F) versus
   ours, scored on both the paper's metric (separation magnitude) and the one that
   predicts detection (support of the inconsistency).
2. **The backdoor.**  Clean-accuracy gap, attack success rate, and verifier
   acceptance, separately for clean and triggered queries.
3. **Evasion map.**  Acceptance against the verifier's opening budget, the tampered
   layer's width, and backdoor locality (how many neurons the tamper is spread over).
4. **Audit accounting.**  Probability the adversary is ever caught, as a function of
   trigger rarity and audit size.
5. **Stealth.**  The same attack with the forged activation capped at a percentile of
   the neuron's natural range.

Every acceptance figure is reported twice: exactly, from the dynamic program in
``pvi.experiments.analysis``, and empirically, by running the real protocol.

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
    audit_detection_probability,
    calibrate_activation_ceilings,
    evaluate_plan,
    gradient_forge,
    injection_forge,
    inverse_transform_forge,
    logit_swap_target,
    plan_single_neuron_flip,
    plan_spread_flip,
    strawman_detection_probability,
)
from pvi.data import load_classification
from pvi.experiments.analysis import acceptance_probability, inconsistent_nodes
from pvi.protocol import (
    ModelCommitment,
    ProtocolParams,
    Prover,
    Verifier,
    run_protocol,
)
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

TARGET_CLASS = 0


def empirical_acceptance(prover, verifier, query, trace, trials: int) -> float:
    return (
        sum(run_protocol(prover, verifier, query, trace=trace).accepted for _ in range(trials))
        / trials
    )


# --------------------------------------------------------------------------- #
# 1. Baselines
# --------------------------------------------------------------------------- #


def block_baselines(network, queries, params, n_examples: int = 8) -> dict:
    print("\n[1/5] baseline forging strategies")
    substitute = load_network(mlp_architecture_for(10), MODELS / "mlp_mnist_halfB.npz")
    rows: list[dict] = []

    for index in range(n_examples):
        query = queries[index]
        honest = network.eval_trace(query)
        target = logit_swap_target(honest.output)

        attempts = [
            inverse_transform_forge(network, query, target, method="pinv", params=params),
            inverse_transform_forge(
                network, query, target, method="regularised", params=params
            ),
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
                    "mean_separation": attempt.mean_separation,
                    "output_changed": attempt.output_differs_from_honest,
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
                "mean_separation": float("nan"),
                "output_changed": bool(
                    int(foreign.output.argmax()) != int(honest.output.argmax())
                ),
                "strawman_detection": strawman_detection_probability(network, foreign, params),
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
                    "mean_separation": float(np.abs(plan.deltas[0])),
                    "output_changed": outcome.forged_class != outcome.honest_class,
                    "strawman_detection": strawman_detection_probability(
                        network, outcome.trace, params
                    ),
                }
            )

    summary: dict[str, dict] = {}
    for row in rows:
        entry = summary.setdefault(
            row["attack"],
            {"acceptance": [], "support": [], "separation": [], "changed": [], "strawman": []},
        )
        entry["acceptance"].append(row["acceptance"])
        entry["support"].append(row["support"])
        entry["separation"].append(row["mean_separation"])
        entry["changed"].append(row["output_changed"])
        if "strawman_detection" in row:
            entry["strawman"].append(row["strawman_detection"])

    out: dict[str, dict] = {}
    for attack, entry in summary.items():
        finite = [s for s in entry["separation"] if np.isfinite(s)]
        out[attack] = {
            "acceptance_mean": float(np.mean(entry["acceptance"])),
            "support_mean": float(np.mean(entry["support"])),
            "support_fraction": float(np.mean(entry["support"]) / rows[0]["n_nodes"]),
            "separation_mean": float(np.mean(finite)) if finite else None,
            "output_changed_rate": float(np.mean(entry["changed"])),
        }
        if entry["strawman"]:
            out[attack]["strawman_detection_mean"] = float(np.mean(entry["strawman"]))
        print(
            f"      {attack:<48} acceptance {out[attack]['acceptance_mean']:.4f}  "
            f"support {out[attack]['support_mean']:.0f}/{rows[0]['n_nodes']}"
        )
    return out


# --------------------------------------------------------------------------- #
# 2. The backdoor
# --------------------------------------------------------------------------- #


def block_backdoor(network, dataset, queries, labels, params, challenges: int) -> dict:
    print("\n[2/5] trigger-conditional backdoor")
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

    clean_accepts, clean_runs = 0, 0
    for query, response in zip(queries[:100], served[:100]):
        for _ in range(challenges):
            clean_runs += 1
            clean_accepts += run_protocol(
                prover, verifier, query, trace=response.trace
            ).accepted

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
        "clean_accuracy_committed": float((honest_pred == labels).mean()),
        "clean_accuracy_served": float((served_pred == labels).mean()),
        "clean_accuracy_gap": float((served_pred == labels).mean() - (honest_pred == labels).mean()),
        "tamper_rate_on_triggered": float(np.mean([r.tampered for r in triggered])),
        "attack_success_rate": asr,
        "clean_acceptance_empirical": clean_accepts / clean_runs,
        "clean_runs": clean_runs,
        "triggered_acceptance_empirical": trig_accepts / trig_runs,
        "triggered_acceptance_exact": float(np.mean(exact)),
        "triggered_runs": trig_runs,
        "tampered_layer_width": network.architecture[1].n_neurons,
    }
    print(
        f"      clean: served == committed on {result['clean_identical_to_committed']:.1%} "
        f"of queries, accuracy gap {result['clean_accuracy_gap']:+.6f}"
    )
    print(f"      triggered: ASR {asr:.4f}")
    print(
        f"      verifier acceptance -- clean {result['clean_acceptance_empirical']:.4f}, "
        f"triggered {result['triggered_acceptance_empirical']:.4f} "
        f"(exact {result['triggered_acceptance_exact']:.4f})"
    )
    return result


# --------------------------------------------------------------------------- #
# 3. Evasion map
# --------------------------------------------------------------------------- #


def block_evasion_map(network, queries, n_queries: int, challenges: int) -> dict:
    print("\n[3/5] evasion versus opening budget, layer width and locality")
    architecture = network.architecture

    by_layer: dict[str, dict] = {}
    for layer_index in (1, 2):
        width = architecture[layer_index].n_neurons
        exact, empirical, deltas, flips = [], [], [], 0
        for index in range(n_queries):
            query = queries[index]
            honest = network.eval_trace(query)
            plan = plan_single_neuron_flip(network, honest, layer_index=layer_index)
            if plan is None:
                continue
            params = ProtocolParams(n_paths=1, check_full_input=True)
            outcome = evaluate_plan(network, honest, plan, params)
            if outcome.forged_class == outcome.honest_class:
                continue
            flips += 1
            exact.append(outcome.acceptance_probability)
            deltas.append(abs(plan.deltas[0]))
        by_layer[str(layer_index)] = {
            "width": width,
            "flip_rate": flips / n_queries,
            "acceptance_exact_mean": float(np.mean(exact)),
            "acceptance_predicted": 1.0 - 1.0 / width,
            "mean_abs_delta": float(np.mean(deltas)),
        }
        print(
            f"      layer {layer_index} (width {width}): flip rate "
            f"{flips / n_queries:.2f}, acceptance {np.mean(exact):.4f} "
            f"(1-1/N = {1 - 1 / width:.4f}), mean |delta| {np.mean(deltas):.2f}"
        )

    # Opening budget.
    budget: dict[str, dict] = {}
    query = queries[0]
    honest = network.eval_trace(query)
    plan = plan_single_neuron_flip(network, honest, layer_index=1)
    width = architecture[1].n_neurons
    for n_paths in (1, 2, 5, 10, 25, 50, 100, 250):
        params = ProtocolParams(n_paths=n_paths, check_full_input=True)
        outcome = evaluate_plan(network, honest, plan, params)
        commitment = ModelCommitment(network)
        prover = Prover(network, commitment, params)
        verifier = Verifier.from_commitment(commitment, params)
        trials = max(200, challenges * 10)
        budget[str(n_paths)] = {
            "acceptance_exact": outcome.acceptance_probability,
            "acceptance_predicted": (1.0 - 1.0 / width) ** n_paths,
            "acceptance_empirical": empirical_acceptance(
                prover, verifier, query, outcome.trace, trials
            ),
            "weight_rows_opened": n_paths * 3,
        }
    print(
        "      opening budget: "
        + ", ".join(
            f"{k} paths -> {v['acceptance_exact']:.3f}" for k, v in budget.items()
        )
    )

    # Locality: spread the tamper over k neurons.
    locality: dict[str, dict] = {}
    for k in (1, 2, 4, 8, 16, 32):
        accepts, supports, found = [], [], 0
        for index in range(min(n_queries, 30)):
            honest = network.eval_trace(queries[index])
            if k == 1:
                spread = plan_single_neuron_flip(network, honest, layer_index=1)
            else:
                spread = plan_spread_flip(network, honest, layer_index=1, n_neurons=k)
            if spread is None:
                continue
            params = ProtocolParams(n_paths=1, check_full_input=True)
            outcome = evaluate_plan(network, honest, spread, params)
            if outcome.forged_class == outcome.honest_class:
                continue
            found += 1
            accepts.append(outcome.acceptance_probability)
            # Report the support we actually achieved: clipping at the ReLU floor can
            # leave a planned neuron unchanged, in which case it stays consistent.
            supports.append(outcome.n_inconsistent)
        if accepts:
            measured = float(np.mean(supports))
            locality[str(k)] = {
                "requested_neurons": k,
                "measured_support": measured,
                "acceptance_mean": float(np.mean(accepts)),
                "acceptance_predicted": 1.0 - measured / architecture[1].n_neurons,
                "success_rate": found / min(n_queries, 30),
            }
    print(
        "      locality: "
        + ", ".join(
            f"k={k} (support {v['measured_support']:.1f}) -> {v['acceptance_mean']:.3f}"
            for k, v in locality.items()
        )
    )

    return {"by_layer": by_layer, "opening_budget": budget, "locality": locality}


# --------------------------------------------------------------------------- #
# 4. Audit accounting
# --------------------------------------------------------------------------- #


def block_audit(width: int) -> dict:
    print("\n[4/5] audit-level detection")
    table: dict[str, dict[str, float]] = {}
    for trigger_rate in (1.0, 1e-2, 1e-4, 0.0):
        row = {}
        for n_queries in (1, 10**3, 10**6, 10**9):
            row[str(n_queries)] = audit_detection_probability(
                width, n_paths=1, trigger_rate=trigger_rate, n_audited_queries=n_queries
            )
        table[str(trigger_rate)] = row
        print(
            f"      trigger rate {trigger_rate:>7}: "
            + ", ".join(f"{k} queries -> {v:.4f}" for k, v in row.items())
        )
    return table


# --------------------------------------------------------------------------- #
# 5. Stealth
# --------------------------------------------------------------------------- #


def block_stealth(network, queries, params, n_queries: int) -> dict:
    """Can the attack survive an activation-envelope check?

    The single-neuron tamper needs a value far outside anything the neuron produces
    on natural data, so a verifier that range-checked opened activations against a
    published envelope would catch it -- a defence the protocol does not specify, but
    which this attack motivates.  The stealthy planner confines every forged value to
    the neuron's own envelope, paying a few extra inconsistent nodes for it.
    """
    from pvi.attacks.tamper import plan_stealthy_flip

    print("\n[5/5] stealth: forged activations confined to natural per-neuron ranges")
    out: dict[str, dict] = {}

    natural = network.forward_batch(queries[:3000])[1]
    print(
        f"      layer-1 natural activations: global max {natural.max():.2f}, "
        f"median per-neuron p99.9 {np.median(np.percentile(natural, 99.9, axis=0)):.2f}"
    )

    # How far outside the envelope does the unconstrained single-neuron attack sit?
    naive_values, naive_percentiles = [], []
    for index in range(min(n_queries, 60)):
        honest = network.eval_trace(queries[index])
        plan = plan_single_neuron_flip(
            network, honest, layer_index=1, target_class=TARGET_CLASS
        )
        if plan is None:
            continue
        neuron, value = plan.neurons[0], plan.new_values[0]
        naive_values.append(value)
        naive_percentiles.append(float((natural[:, neuron] <= value).mean() * 100.0))
    out["unconstrained_single_neuron"] = {
        "mean_forged_value": float(np.mean(naive_values)),
        "mean_percentile_within_neuron": float(np.mean(naive_percentiles)),
        "natural_global_max": float(natural.max()),
    }
    print(
        f"      unconstrained single-neuron tamper needs mean value "
        f"{np.mean(naive_values):.1f} -- above the global natural max: detectable"
    )

    for percentile in (100.0, 99.9, 99.0, 95.0):
        ceilings = (
            natural.max(axis=0)
            if percentile == 100.0
            else calibrate_activation_ceilings(
                network, queries[:3000], layer_index=1, percentile=percentile
            )
        )
        row: dict[str, dict] = {}
        for label, target in (("targeted", TARGET_CLASS), ("untargeted", None)):
            successes, supports, accepts, eligible = 0, [], [], 0
            for index in range(min(n_queries, 60)):
                honest = network.eval_trace(queries[index])
                if target is not None and int(honest.output.argmax()) == target:
                    continue
                eligible += 1
                plan = plan_stealthy_flip(
                    network,
                    honest,
                    layer_index=1,
                    ceilings=ceilings,
                    target_class=target,
                )
                if plan is None:
                    continue
                outcome = evaluate_plan(network, honest, plan, params)
                if outcome.forged_class == outcome.honest_class:
                    continue
                if target is not None and outcome.forged_class != target:
                    continue
                successes += 1
                supports.append(outcome.n_inconsistent)
                accepts.append(outcome.acceptance_probability)
            row[label] = {
                "attack_success_rate": successes / max(eligible, 1),
                "mean_support": float(np.mean(supports)) if supports else float("nan"),
                "acceptance_mean": float(np.mean(accepts)) if accepts else float("nan"),
            }
        out[f"p{percentile}"] = row
        print(
            f"      cap p{percentile:<5}: "
            f"untargeted ASR {row['untargeted']['attack_success_rate']:.3f} "
            f"(k={row['untargeted']['mean_support']:.1f}, "
            f"acceptance {row['untargeted']['acceptance_mean']:.4f}) | "
            f"targeted ASR {row['targeted']['attack_success_rate']:.3f} "
            f"(k={row['targeted']['mean_support']:.1f}, "
            f"acceptance {row['targeted']['acceptance_mean']:.4f})"
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
    params = ProtocolParams(n_paths=1, check_full_input=True)

    print(f"model: {architecture.layer_widths}, {len(queries)} test queries available")
    started = time.perf_counter()

    results = {
        "config": vars(args),
        "layer_widths": list(architecture.layer_widths),
        "baselines": block_baselines(network, queries, params),
        "backdoor": block_backdoor(
            network, dataset, queries[: args.queries], labels[: args.queries], params,
            args.challenges,
        ),
        "evasion": block_evasion_map(network, queries, args.queries, args.challenges),
        "audit": block_audit(architecture[1].n_neurons),
        "stealth": block_stealth(network, queries, params, min(args.queries, 100)),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "attack.json", results)
    print(f"\ndone in {time.perf_counter() - started:.0f}s")
    print(f"wrote {RESULTS / 'attack.json'}")


if __name__ == "__main__":
    main()
