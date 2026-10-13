"""Value-aware path samplers against an adversary who knows the rule, on the MNIST MLP (Sec. 3.5, ``sec:break``).

Two blocks, on hidden layer ``--layer`` of ``mlp_mnist_full``, with exact single-path detection
(``pvi.baseline.analysis``):

1. **Sampler sweep** (``sweep``).  Uniform, static importance (outgoing weight magnitude), local contribution
   (``|w_ij a_i|``) and gradient saliency (``pvi.baseline.samplers``) against the plain single-neuron edit
   (``detection_naive``) and against an adversary that picks, among ``--candidate-neurons`` candidates, the edit the
   sampler detects least (``detection_evasive``, ``pvi.baseline.adaptive.plan_evasive_flip``).  Paper: "Uniform
   sampling detects 0.20% of attacks.  The contribution-weighted sampler [...] detects 9.2% of the best
   single-neuron edits"; static importance 0.11% and gradient saliency 0.016% (``ext:attack-measure``).
2. **The zero blind spot** (``zero_blind_spot``).  Contribution weighting gives weight zero to a claimed zero, so
   an adversary that sets neurons to zero (within the ``--percentile`` ceilings of the first ``--calib-images``
   test images) is never opened: ``mean_support`` edited neurons, ``detection_contribution_weighted``, and
   ``--protocol-runs`` real protocol runs per query (``protocol_runs``, ``protocol_accepts``); and how often the
   zeroed neurons are zero on natural inputs (``natural_zero_fraction_at_chosen_neurons``).  Paper: "zeroing about 14
   of the 512 neurons flips the prediction and passed 2,000 of 2,000 runs"; "65% of inputs".

Usage (from ``code/``)::

    python -m experiments.attack.samplers
    python -m experiments.attack.samplers --queries 5 --out /tmp/defence.json

Writes ``--out`` (default ``artifacts/results/defence.json``): ``config``, ``env``, ``layer``, ``layer_width``,
``sweep.<sampler>.{detection_naive,detection_evasive}`` and ``zero_blind_spot``.  Uses
``artifacts/models/mlp_mnist_full.npz`` and MNIST; CPU only, about 2 min.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.common.paths import MNIST_MODELS, RESULTS
from experiments.common.records import environment, write_json
from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.baseline.adaptive import plan_adaptive_stealthy_flip, plan_evasive_flip
from pvi.baseline.analysis import acceptance_probability
from pvi.baseline.path.commitments import ModelCommitment
from pvi.baseline.path.params import ProtocolParams
from pvi.baseline.path.prover import Prover
from pvi.baseline.path.sampling import UniformPathSampler
from pvi.baseline.path.session import run_protocol
from pvi.baseline.path.verifier import Verifier
from pvi.baseline.samplers import (GradientSaliencySampler, LocalContributionSampler, StaticImportanceSampler,
                                   weight_magnitude_importance)
from pvi.mnist.data import load_classification
from pvi.mnist.training import load_network
from pvi.mnist.zoo import mlp_architecture_for

MODEL = "mlp_mnist_full"


def build_samplers(network) -> dict:
    importance = {index: weight_magnitude_importance(network, index)
                  for index, _ in enumerate(network.architecture.layers) if index < len(network.architecture) - 1}
    return {
        "uniform": UniformPathSampler(),
        "static-importance": StaticImportanceSampler(importance),
        "local-contribution": LocalContributionSampler(network),
        "gradient-saliency": GradientSaliencySampler(network),
    }


# --------------------------------------------------------------------------- #
# 1. Sampler sweep
# --------------------------------------------------------------------------- #


def block_sweep(network, queries, params, args) -> dict:
    print("\n[1/2] defence sweep: samplers against adversaries")
    out: dict[str, dict] = {}
    for name, sampler in build_samplers(network).items():
        naive, evasive = [], []
        for index in range(args.queries):
            honest = network.eval_trace(queries[index])
            plan = plan_single_neuron_flip(network, honest, layer_index=args.layer)
            if plan is not None:
                forged = apply_plan(network, honest, plan)
                naive.append(acceptance_probability(network, forged, params, sampler))
            search = plan_evasive_flip(network, honest, layer_index=args.layer, sampler=sampler, params=params,
                                       candidate_neurons=args.candidate_neurons)
            if search.succeeded:
                evasive.append(search.acceptance)
        row = {"detection_naive": 1.0 - float(np.mean(naive)), "detection_evasive": 1.0 - float(np.mean(evasive))}
        out[name] = row
        print(f"      {name:<20} naive {row['detection_naive']:.5f}  evasive {row['detection_evasive']:.5f}")
    return out


# --------------------------------------------------------------------------- #
# 2. The zero blind spot
# --------------------------------------------------------------------------- #


def block_zero_blind_spot(network, queries, ceilings, params, args) -> dict:
    print("\n[2/2] the zero blind spot")
    sampler = LocalContributionSampler(network)
    natural = network.forward_batch(queries[:args.calib_images])[args.layer]

    commitment = ModelCommitment(network)
    prover = Prover(network, commitment, params, sampler=sampler)
    verifier = Verifier.from_commitment(commitment, params, sampler=sampler)

    all_zero, supports, contrib_acc, zero_fracs = [], [], [], []
    protocol_runs, protocol_accepts = 0, 0
    for index in range(min(args.queries, args.blind_spot_queries)):
        honest = network.eval_trace(queries[index])
        search = plan_adaptive_stealthy_flip(network, honest, layer_index=args.layer, ceilings=ceilings,
                                             sampler=sampler, params=params)
        if not search.succeeded:
            continue
        plan = search.plan
        forged = apply_plan(network, honest, plan)
        all_zero.append(all(value == 0.0 for value in plan.new_values))
        supports.append(plan.size)
        contrib_acc.append(acceptance_probability(network, forged, params, sampler))
        zero_fracs.append(float((natural[:, list(plan.neurons)] == 0).mean()))
        for _ in range(args.protocol_runs):
            protocol_runs += 1
            protocol_accepts += run_protocol(prover, verifier, queries[index], trace=forged).accepted

    result = {
        "fraction_of_plans_using_only_zeros": float(np.mean(all_zero)),
        "mean_support": float(np.mean(supports)),
        "detection_contribution_weighted": 1.0 - float(np.mean(contrib_acc)),
        "natural_zero_fraction_at_chosen_neurons": float(np.mean(zero_fracs)),
        "protocol_runs": protocol_runs,
        "protocol_accepts": protocol_accepts,
    }
    print(f"      plans using only zeroed activations: {result['fraction_of_plans_using_only_zeros']:.0%}, "
          f"mean support {result['mean_support']:.1f}/{network.architecture[args.layer].n_neurons}")
    print(f"      detection under contribution weighting: {result['detection_contribution_weighted']:.6f}")
    print(f"      real protocol under contribution weighting: {protocol_accepts}/{protocol_runs} accepted")
    print(f"      chosen neurons are naturally zero on {result['natural_zero_fraction_at_chosen_neurons']:.1%} "
          f"of natural inputs")
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=40, help="test queries of the sweep")
    parser.add_argument("--blind-spot-queries", type=int, default=40,
                        help="the zero blind spot uses the first min(--queries, this) queries")
    parser.add_argument("--layer", type=int, default=1, help="the attacked hidden layer")
    parser.add_argument("--candidate-neurons", type=int, default=60,
                        help="neurons the evasive adversary tries per query")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="ceiling of the zeroing adversary, a percentile of each neuron's natural range")
    parser.add_argument("--calib-images", type=int, default=3000,
                        help="test images whose activations set the ceilings and the natural zero fraction")
    parser.add_argument("--protocol-runs", type=int, default=50, help="real protocol runs per zeroing edit")
    parser.add_argument("--models", default=str(MNIST_MODELS), help="folder of the trained models")
    parser.add_argument("--out", default=str(RESULTS / "defence.json"))
    args = parser.parse_args()

    architecture = mlp_architecture_for(10)
    network = load_network(architecture, Path(args.models) / f"{MODEL}.npz")
    dataset = load_classification("mnist")
    queries = np.ascontiguousarray(dataset.test_x.reshape(len(dataset.test_x), -1), dtype=np.float32)
    params = ProtocolParams(n_paths=1)
    ceilings = np.percentile(network.forward_batch(queries[:args.calib_images])[args.layer], args.percentile, axis=0)

    width = architecture[args.layer].n_neurons
    print(f"model {architecture.layer_widths}, tampering layer {args.layer} (width {width})")
    started = time.perf_counter()
    results = {
        "config": {k: v for k, v in vars(args).items() if k not in ("models", "out")},
        "env": environment("cpu"),
        "layer": args.layer,
        "layer_width": width,
        "sweep": block_sweep(network, queries, params, args),
        "zero_blind_spot": block_zero_blind_spot(network, queries, ceilings, params, args),
    }
    write_json(args.out, results)
    print(f"\ndone in {time.perf_counter() - started:.0f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
