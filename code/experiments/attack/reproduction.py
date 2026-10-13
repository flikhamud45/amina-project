"""The path protocol of Anchuri et al. on its own threat model: completeness and other-model soundness.

Backs "The implementation reproduces the published behaviour" (Sec. 3.5, ``sec:break``) and the paragraph
*Reproduction* of the extended version (``ext:attack-measure``).  ``RandPathTest`` (``pvi.baseline.path``) runs in
the four substitute-model settings of ``pvi.mnist.zoo.SETUPS`` (different tasks with one shared class; the same task
on disjoint training data, as an MLP and as a CNN; an 8-bit quantised copy of the committed model):

1. **Completeness.**  The honest prover runs ``--challenges`` times on each of ``--queries`` test queries.  Each
   query (at most ``--zero-tolerance-queries``) also runs once against a verifier without the protocol's tolerance
   of 1e-4, the exact-equality test of the protocol's description, which rejects many honest runs because a
   layer's matrix product and a single neuron's dot product round differently.
2. **Other-model soundness.**  How often the verifier of the committed model rejects the substitute's *honest*
   trace, and how often the two models agree on the prediction.

Usage (from ``code/``)::

    python -m experiments.attack.reproduction                     # 150 queries x 20 challenges per setting
    python -m experiments.attack.reproduction --queries 20 --challenges 5 --setups quantised

Writes ``--out`` (default ``artifacts/results/reproduction.json``): ``config``, ``env`` and, per setting,
``setups.<key>.correctness`` (``runs``, ``accepted``, ``acceptance_rate``, ``tolerance``, ``zero_tolerance_runs``,
``zero_tolerance_acceptance_rate``) and ``setups.<key>.other_model`` (``prediction_agreement``, ``runs``,
``detection_rate``).  Uses the stored models of ``artifacts/models/`` and MNIST; CPU only, about 1 min.  The
challenges are fresh on every run, so a rerun moves the rates by a few tenths of a percent (the zero-tolerance rate,
from one run per query, by a few points).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common.paths import MNIST_MODELS, RESULTS
from experiments.common.records import environment, write_json
from pvi.baseline.path.commitments import ModelCommitment
from pvi.baseline.path.params import ProtocolParams
from pvi.baseline.path.prover import Prover
from pvi.baseline.path.session import run_protocol
from pvi.baseline.path.verifier import Verifier
from pvi.mnist.network import TracedNetwork
from pvi.mnist.training import load_network
from pvi.mnist.zoo import SETUPS, SubstitutionSetup, quantise_network


def load_setup(setup: SubstitutionSetup, models: Path) -> tuple[TracedNetwork, TracedNetwork]:
    """``(M, M~)`` for a setting, from the weights in ``models`` (``train_mnist.py``)."""
    honest = load_network(setup.honest.architecture(), models / f"{setup.honest.name}.npz")
    if setup.is_derived():
        return honest, quantise_network(honest, bits=setup.quantise_bits or 8)
    substitute = load_network(setup.substitute.architecture(), models / f"{setup.substitute.name}.npz")
    return honest, substitute


def evaluation_queries(setup: SubstitutionSetup, count: int, seed: int = 0) -> np.ndarray:
    dataset = setup.eval_dataset()
    pool = dataset.test_x.reshape(len(dataset.test_x), -1)
    index = np.random.default_rng(seed).choice(len(pool), size=min(count, len(pool)), replace=False)
    return np.ascontiguousarray(pool[index], dtype=np.float32)


def parties(network: TracedNetwork, params: ProtocolParams) -> tuple[Prover, Verifier]:
    commitment = ModelCommitment(network)
    return Prover(network, commitment, params), Verifier.from_commitment(commitment, params)


def measure_correctness(network: TracedNetwork, queries: np.ndarray, challenges: int, zero_tolerance: int) -> dict:
    params = ProtocolParams()
    prover, verifier = parties(network, params)
    accepted, runs = 0, 0
    for query in queries:
        for _ in range(challenges):
            accepted += int(run_protocol(prover, verifier, query).accepted)
            runs += 1

    # The exact-equality test: no tolerance on the local check.
    exact_prover, exact_verifier = parties(network, ProtocolParams(abs_tolerance=0.0))
    sample = queries[:zero_tolerance]
    exact_accepted = sum(run_protocol(exact_prover, exact_verifier, q).accepted for q in sample)
    return {
        "runs": runs,
        "accepted": accepted,
        "acceptance_rate": accepted / runs,
        "tolerance": params.abs_tolerance,
        "zero_tolerance_runs": len(sample),
        "zero_tolerance_acceptance_rate": exact_accepted / len(sample),
    }


def measure_other_model(honest: TracedNetwork, substitute: TracedNetwork, queries: np.ndarray,
                        challenges: int) -> dict:
    """``RandPathTest`` against the substitute's *honest* trace (the protocol's own threat model)."""
    prover, verifier = parties(honest, ProtocolParams())
    agree, detections, runs = 0, 0, 0
    for query in queries:
        trace = substitute.eval_trace(query)
        agree += int(trace.output.argmax() == honest.eval_trace(query).output.argmax())
        for _ in range(challenges):
            detections += int(not run_protocol(prover, verifier, query, trace=trace).accepted)
            runs += 1
    return {"prediction_agreement": agree / len(queries), "runs": runs, "detection_rate": detections / runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=150, help="test queries per setting")
    parser.add_argument("--challenges", type=int, default=20, help="fresh challenges per query")
    parser.add_argument("--zero-tolerance-queries", type=int, default=200,
                        help="at most this many queries run once against the zero-tolerance verifier")
    parser.add_argument("--setups", nargs="*", default=None, metavar="KEY",
                        help=f"settings to run (default all: {' '.join(s.key for s in SETUPS)})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the query selection")
    parser.add_argument("--models", default=str(MNIST_MODELS), help="folder of the trained models")
    parser.add_argument("--out", default=str(RESULTS / "reproduction.json"))
    args = parser.parse_args()

    setups = [s for s in SETUPS if not args.setups or s.key in args.setups]
    config = {k: v for k, v in vars(args).items() if k not in ("models", "out")}
    results: dict = {"config": config, "env": environment("cpu"), "setups": {}}
    for setup in setups:
        print(f"\n== {setup.key}: {setup.title}  [{setup.paper_reference}]")
        honest, substitute = load_setup(setup, Path(args.models))
        queries = evaluation_queries(setup, args.queries, args.seed)
        correctness = measure_correctness(honest, queries, args.challenges, args.zero_tolerance_queries)
        other = measure_other_model(honest, substitute, queries, args.challenges)
        print(f"   honest accepted {correctness['acceptance_rate']:.4f} of {correctness['runs']} runs; "
              f"zero-tolerance verifier accepts {correctness['zero_tolerance_acceptance_rate']:.2f}")
        print(f"   substitute: prediction agreement {other['prediction_agreement']:.4f}, "
              f"detected {other['detection_rate']:.4f} of {other['runs']} runs")
        results["setups"][setup.key] = {"title": setup.title, "paper_reference": setup.paper_reference,
                                        "correctness": correctness, "other_model": other}
    write_json(args.out, results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
