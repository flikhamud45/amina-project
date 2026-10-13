"""The single-neuron attack and its backdoor against the path protocol on the MNIST MLP (Sec. 3.5, ``sec:break``).

The adversary runs the committed model honestly, overwrites one neuron (ranked by the gradient of the decision
margin, with the smallest flipping value found by bisection) and recomputes every later layer with the true weights
(``pvi.attacks``).  Acceptance under ``RandPathTest`` is computed exactly by the dynamic program of
``pvi.baseline.analysis``; the backdoor also runs on the real protocol.  Four blocks:

1. **Baselines** (``--baseline-queries`` queries).  The forgeries Anchuri et al. consider (a substitute model's
   honest trace; the inverse transform, logit-swap injection and gradient reconstruction of their Appendices E and
   F and Section 7.2) against ours (one neuron of each of ``--baseline-layers``): inconsistent trace nodes
   (``support_mean``) and acceptance (``acceptance_mean``).  Paper: "One tampered neuron passes with the predicted
   probability 1-1/N (99.6--99.8%); the substitute models of Anchuri et al. corrupt hundreds of the MLP's 778 trace
   nodes".
2. **Backdoor** (``--queries`` test queries).  A ``--trigger-size`` square patch in the image corner tells the
   adversary to force ``--target-class`` with one neuron of ``--layer``; clean queries get the honest trace.
   Records ``clean_identical_to_committed``, ``clean_accuracy_gap``, ``attack_success_rate`` (on triggered queries
   not already in the target class), ``triggered_acceptance_exact`` and ``triggered_acceptance_empirical`` (real
   protocol runs, ``--challenges`` per tampered query among the first ``--empirical-queries``).  Paper: "passed
   99.96% of 2,300 end-to-end runs (predicted: 99.8%)".
3. **Opening budget.**  Exact acceptance of the single-neuron edit with ``--opening-paths`` paths (61% with 250:
   ``tab:sok`` and ``ext:attack-measure``).
4. **Stealth.**  Every forged value capped at a percentile (``--stealth-percentiles``) of the neuron's range on the
   first ``--calib-images`` test images, on ``--stealth-queries`` queries: edited neurons (``mean_support``) and
   acceptance.  Paper (``ext:attack-measure``): "below its neuron's 95th percentile takes about five neurons and
   passes 99% of the time".  The calibration images include the attacked queries; ``adaptive_game.py`` repeats the
   measurement with held-out ranges.

Usage (from ``code/``)::

    python -m experiments.attack.attack_mnist
    python -m experiments.attack.attack_mnist --queries 20 --challenges 5 --stealth-queries 10 --out /tmp/attack.json

Writes ``--out`` (default ``artifacts/results/attack.json``): ``config``, ``env``, ``layer_widths``, ``baselines``,
``backdoor``, ``opening_budget``, ``stealth`` (keys ``p<percentile>``).  Uses ``artifacts/models/mlp_mnist_full.npz``
(and ``mlp_mnist_halfB.npz`` as the substitute) and MNIST; CPU only, about 2 min.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.common.paths import MNIST_MODELS, RESULTS
from experiments.common.records import environment, write_json
from pvi.attacks import (BackdoorAdversary, PatchTrigger, calibrate_activation_ceilings, evaluate_plan,
                         gradient_forge, injection_forge, inverse_transform_forge, logit_swap_target,
                         plan_single_neuron_flip)
from pvi.attacks.tamper import plan_stealthy_flip
from pvi.baseline.analysis import acceptance_probability, inconsistent_nodes
from pvi.baseline.path.commitments import ModelCommitment
from pvi.baseline.path.params import ProtocolParams
from pvi.baseline.path.prover import Prover
from pvi.baseline.path.session import run_protocol
from pvi.baseline.path.verifier import Verifier
from pvi.mnist.data import load_classification
from pvi.mnist.training import load_network
from pvi.mnist.zoo import mlp_architecture_for

MODEL = "mlp_mnist_full"
SUBSTITUTE = "mlp_mnist_halfB"


# --------------------------------------------------------------------------- #
# 1. Baselines
# --------------------------------------------------------------------------- #


def block_baselines(network, substitute, queries, params, args) -> dict:
    print("\n[1/4] baseline forging strategies")
    rows: list[dict] = []
    for index in range(args.baseline_queries):
        query = queries[index]
        honest = network.eval_trace(query)
        target = logit_swap_target(honest.output)

        attempts = [
            inverse_transform_forge(network, query, target, params=params),
            injection_forge(network, query, target, injection_layer=args.injection_layer,
                            steps=args.injection_steps, params=params),
            gradient_forge(network, query, target, steps=args.gradient_steps, weight_decay=0.0, params=params),
        ]
        for attempt in attempts:
            rows.append({"attack": attempt.name, "acceptance": attempt.acceptance_probability,
                         "support": attempt.n_inconsistent, "n_nodes": attempt.n_nodes})

        # Substitute model, honest trace (plain other-model soundness).
        foreign = substitute.eval_trace(query)
        report = inconsistent_nodes(network, foreign, params)
        rows.append({"attack": "substitute model, honest trace (Def. 1)",
                     "acceptance": acceptance_probability(network, foreign, params),
                     "support": report.total, "n_nodes": int(sum(report.per_layer_widths.values()))})

        # Ours.
        for layer_index in args.baseline_layers:
            plan = plan_single_neuron_flip(network, honest, layer_index=layer_index)
            if plan is None:
                continue
            outcome = evaluate_plan(network, honest, plan, params)
            rows.append({"attack": f"single-neuron tamper, layer {layer_index} (ours)",
                         "acceptance": outcome.acceptance_probability,
                         "support": outcome.n_inconsistent, "n_nodes": outcome.n_nodes})

    summary: dict[str, dict] = {}
    for row in rows:
        entry = summary.setdefault(row["attack"], {"acceptance": [], "support": []})
        entry["acceptance"].append(row["acceptance"])
        entry["support"].append(row["support"])

    out: dict[str, dict] = {}
    for attack, entry in summary.items():
        out[attack] = {"acceptance_mean": float(np.mean(entry["acceptance"])),
                       "support_mean": float(np.mean(entry["support"]))}
        print(f"      {attack:<48} acceptance {out[attack]['acceptance_mean']:.4f}  "
              f"support {out[attack]['support_mean']:.0f}/{rows[0]['n_nodes']}")
    return out


# --------------------------------------------------------------------------- #
# 2. The backdoor
# --------------------------------------------------------------------------- #


def block_backdoor(network, queries, labels, params, args) -> dict:
    print("\n[2/4] trigger-conditional backdoor")
    trigger = PatchTrigger(input_shape=(1, 28, 28), size=args.trigger_size, value=args.trigger_value)
    adversary = BackdoorAdversary(network, trigger, layer_index=args.layer, target_class=args.target_class)

    honest_pred = network.predict(queries)
    served = [adversary.serve(q) for q in queries]
    served_pred = np.asarray([r.served_class for r in served])

    triggered_queries = trigger.apply_batch(queries)
    triggered = [adversary.serve(q) for q in triggered_queries]
    triggered_honest = np.asarray([r.honest_class for r in triggered])
    triggered_served = np.asarray([r.served_class for r in triggered])

    # Standard backdoor convention: the success rate counts inputs not already in the target class, since for
    # those the adversary has nothing to do.
    eligible = triggered_honest != args.target_class
    asr = float((triggered_served[eligible] == args.target_class).mean())

    commitment = ModelCommitment(network)
    prover = Prover(network, commitment, params)
    verifier = Verifier.from_commitment(commitment, params)

    trig_accepts, trig_runs, exact = 0, 0, []
    n = args.empirical_queries
    for query, response in zip(triggered_queries[:n], triggered[:n]):
        if not response.tampered:
            continue
        exact.append(acceptance_probability(network, response.trace, params))
        for _ in range(args.challenges):
            trig_runs += 1
            trig_accepts += run_protocol(prover, verifier, query, trace=response.trace).accepted

    result = {
        "clean_identical_to_committed": float((honest_pred == served_pred).mean()),
        "clean_accuracy_gap": float((served_pred == labels).mean() - (honest_pred == labels).mean()),
        "attack_success_rate": asr,
        "triggered_acceptance_empirical": trig_accepts / trig_runs,
        "triggered_acceptance_exact": float(np.mean(exact)),
        "triggered_runs": trig_runs,
    }
    print(f"      clean: served == committed on {result['clean_identical_to_committed']:.1%} "
          f"of queries, accuracy gap {result['clean_accuracy_gap']:+.6f}")
    print(f"      triggered: ASR {asr:.4f}")
    print(f"      verifier acceptance on triggered queries {result['triggered_acceptance_empirical']:.4f} "
          f"(exact {result['triggered_acceptance_exact']:.4f})")
    return result


# --------------------------------------------------------------------------- #
# 3. Opening budget
# --------------------------------------------------------------------------- #


def block_opening_budget(network, queries, args) -> dict:
    paths = args.opening_paths
    print(f"\n[3/4] acceptance with {paths} paths")
    honest = network.eval_trace(queries[0])
    plan = plan_single_neuron_flip(network, honest, layer_index=args.layer)
    outcome = evaluate_plan(network, honest, plan, ProtocolParams(n_paths=paths))
    print(f"      {paths} paths -> {outcome.acceptance_probability:.3f}")
    return {"n_paths": paths, "acceptance_exact": outcome.acceptance_probability}


# --------------------------------------------------------------------------- #
# 4. Stealth
# --------------------------------------------------------------------------- #


def block_stealth(network, queries, params, args) -> dict:
    """Can the attack survive an activation-envelope check?

    The single-neuron edit needs a value far outside anything the neuron produces on natural data, so a verifier
    that range-checked opened activations against a published envelope would catch it, a defence the protocol does
    not specify but which this attack motivates.  The stealthy planner confines every forged value to the neuron's
    own envelope and pays for it with a few more edited neurons.
    """
    print("\n[4/4] stealth: forged activations confined to natural per-neuron ranges")
    out: dict[str, dict] = {}
    calib = queries[:args.calib_images]
    natural = network.forward_batch(calib)[args.layer]
    for percentile in args.stealth_percentiles:
        ceilings = (natural.max(axis=0) if percentile == 100.0 else
                    calibrate_activation_ceilings(network, calib, layer_index=args.layer, percentile=percentile))
        supports, accepts = [], []
        for index in range(args.stealth_queries):
            honest = network.eval_trace(queries[index])
            plan = plan_stealthy_flip(network, honest, layer_index=args.layer, ceilings=ceilings)
            if plan is None:
                continue
            outcome = evaluate_plan(network, honest, plan, params)
            if outcome.forged_class == outcome.honest_class:
                continue
            supports.append(outcome.n_inconsistent)
            accepts.append(outcome.acceptance_probability)
        key = f"p{percentile}"
        out[key] = {"mean_support": float(np.mean(supports)) if supports else float("nan"),
                    "acceptance_mean": float(np.mean(accepts)) if accepts else float("nan")}
        print(f"      cap p{percentile:<5}: k={out[key]['mean_support']:.1f}, "
              f"acceptance {out[key]['acceptance_mean']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=200, help="test queries of the backdoor")
    parser.add_argument("--challenges", type=int, default=25, help="real protocol runs per tampered query")
    parser.add_argument("--empirical-queries", type=int, default=100,
                        help="the backdoor's real runs use the tampered queries among the first this many")
    parser.add_argument("--baseline-queries", type=int, default=8, help="queries of the baselines block")
    parser.add_argument("--baseline-layers", type=int, nargs="+", default=[1, 2],
                        help="hidden layers of our single-neuron edit in the baselines block")
    parser.add_argument("--injection-layer", type=int, default=2, help="layer of the logit-swap injection")
    parser.add_argument("--injection-steps", type=int, default=1500, help="optimiser steps of the injection forgery")
    parser.add_argument("--gradient-steps", type=int, default=3000,
                        help="optimiser steps of the gradient reconstruction")
    parser.add_argument("--layer", type=int, default=1,
                        help="hidden layer of the backdoor, the opening budget and the stealth block")
    parser.add_argument("--target-class", type=int, default=0, help="the backdoor's target class")
    parser.add_argument("--trigger-size", type=int, default=3, help="side of the square trigger patch, pixels")
    parser.add_argument("--trigger-value", type=float, default=1.0, help="pixel value of the trigger patch")
    parser.add_argument("--opening-paths", type=int, default=250, help="paths of the opening-budget block")
    parser.add_argument("--stealth-percentiles", type=float, nargs="+", default=[100.0, 99.9, 99.0, 95.0],
                        help="caps of the stealth block, percentiles of each neuron's natural range")
    parser.add_argument("--stealth-queries", type=int, default=60, help="queries of the stealth block")
    parser.add_argument("--calib-images", type=int, default=3000,
                        help="test images whose activations set the stealth block's ranges")
    parser.add_argument("--models", default=str(MNIST_MODELS), help="folder of the trained models")
    parser.add_argument("--out", default=str(RESULTS / "attack.json"))
    args = parser.parse_args()

    architecture = mlp_architecture_for(10)
    network = load_network(architecture, Path(args.models) / f"{MODEL}.npz")
    substitute = load_network(architecture, Path(args.models) / f"{SUBSTITUTE}.npz")
    dataset = load_classification("mnist")
    queries = np.ascontiguousarray(dataset.test_x.reshape(len(dataset.test_x), -1), dtype=np.float32)
    labels = dataset.test_y
    params = ProtocolParams(n_paths=1)

    print(f"model: {architecture.layer_widths}, {len(queries)} test queries available")
    started = time.perf_counter()
    results = {
        "config": {k: v for k, v in vars(args).items() if k not in ("models", "out")},
        "env": environment("cpu"),
        "layer_widths": list(architecture.layer_widths),
        "baselines": block_baselines(network, substitute, queries, params, args),
        "backdoor": block_backdoor(network, queries[:args.queries], labels[:args.queries], params, args),
        "opening_budget": block_opening_budget(network, queries, args),
        "stealth": block_stealth(network, queries, params, args),
    }
    write_json(args.out, results)
    print(f"\ndone in {time.perf_counter() - started:.0f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
