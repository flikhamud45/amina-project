"""Reproduce the protocol of Anchuri et al. on its own threat model (report, Sec. 4.2).

For each of four substitute-model settings from the paper (``pvi.zoo.SETUPS``):

1. **Completeness.**  The honest prover is accepted on every run.  The same runs
   are replayed against a zero-tolerance verifier (the idealised exact-equality
   test of the paper's Figure 3), which rejects many honest runs because a layer's
   matrix product and a single neuron's dot product round differently.
2. **Other-model soundness.**  How often ``RandPathTest`` rejects the substitute
   model's honest trace, and how often the two models agree on the prediction.
3. **Cost.**  Prover and verifier time and proof size for one path.

Usage
-----
    python experiments/1_reproduction/run.py [--queries N] [--challenges N] [--setups KEY ...]

Writes ``artifacts/results/reproduction.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pvi.nn.network import TracedNetwork
from pvi.protocol import ModelCommitment, ProtocolParams, Prover, Verifier, run_protocol, sample_challenge
from pvi.results import write_json
from pvi.training import load_network
from pvi.zoo import SETUPS, SubstitutionSetup, quantise_network

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"


def load_setup(setup: SubstitutionSetup) -> tuple[TracedNetwork, TracedNetwork]:
    """Return ``(M, M~)`` for a setup (trained by experiments/0_train_models)."""
    honest = load_network(setup.honest.architecture(), MODELS / f"{setup.honest.name}.npz")
    if setup.is_derived():
        return honest, quantise_network(honest, bits=setup.quantise_bits or 8)
    substitute = load_network(setup.substitute.architecture(), MODELS / f"{setup.substitute.name}.npz")
    return honest, substitute


def evaluation_queries(setup: SubstitutionSetup, count: int, seed: int = 0) -> np.ndarray:
    dataset = setup.eval_dataset()
    pool = dataset.test_x.reshape(len(dataset.test_x), -1)
    index = np.random.default_rng(seed).choice(len(pool), size=min(count, len(pool)), replace=False)
    return np.ascontiguousarray(pool[index], dtype=np.float32)


def parties(network: TracedNetwork, params: ProtocolParams) -> tuple[Prover, Verifier]:
    commitment = ModelCommitment(network)
    return Prover(network, commitment, params), Verifier.from_commitment(commitment, params)


def measure_correctness(network: TracedNetwork, queries: np.ndarray, challenges: int, n_paths: int) -> dict:
    params = ProtocolParams(n_paths=n_paths, check_full_input=True)
    prover, verifier = parties(network, params)
    accepted, runs, residuals = 0, 0, []
    for query in queries:
        for _ in range(challenges):
            result = run_protocol(prover, verifier, query)
            accepted += int(result.accepted)
            runs += 1
            residuals.extend(result.residuals)
    residuals = np.asarray(residuals)

    # The idealised exact-equality test: zero tolerance on the local check.
    exact = ProtocolParams(n_paths=n_paths, abs_tolerance=0.0, rel_tolerance=0.0, check_full_input=True)
    exact_prover, exact_verifier = parties(network, exact)
    sample = queries[:200]
    exact_accepted = sum(run_protocol(exact_prover, exact_verifier, q).accepted for q in sample)

    return {
        "runs": runs,
        "accepted": accepted,
        "acceptance_rate": accepted / runs,
        "residual_max": float(residuals.max()),
        "tolerance": params.abs_tolerance,
        "zero_tolerance_runs": len(sample),
        "zero_tolerance_acceptance_rate": exact_accepted / len(sample),
    }


def measure_other_model(honest: TracedNetwork, substitute: TracedNetwork, queries: np.ndarray,
                        challenges: int, n_paths: int) -> dict:
    """RandPathTest against the substitute's *honest* trace (the paper's threat model)."""
    prover, verifier = parties(honest, ProtocolParams(n_paths=n_paths, check_full_input=True))
    agree, detections, runs = 0, 0, 0
    for query in queries:
        trace = substitute.eval_trace(query)
        agree += int(trace.output.argmax() == honest.eval_trace(query).output.argmax())
        for _ in range(challenges):
            detections += int(not run_protocol(prover, verifier, query, trace=trace).accepted)
            runs += 1
    return {"prediction_agreement": agree / len(queries), "runs": runs, "detection_rate": detections / runs}


def measure_cost(network: TracedNetwork, queries: np.ndarray, n_paths: int) -> dict:
    prover, verifier = parties(network, ProtocolParams(n_paths=n_paths, check_full_input=True))
    prove, verify, sizes = [], [], []
    for query in queries[:100]:
        challenge = sample_challenge()
        t0 = time.perf_counter()
        round1, state = prover.prove1(query)
        round2 = prover.prove2(state, challenge)
        t1 = time.perf_counter()
        result = verifier.verify(query, round1, round2, challenge)
        t2 = time.perf_counter()
        prove.append(t1 - t0)
        verify.append(t2 - t1)
        sizes.append(result.proof_size_bytes)
    return {"prove_ms": float(np.mean(prove) * 1e3), "verify_ms": float(np.mean(verify) * 1e3),
            "proof_kb": float(np.mean(sizes) / 1024)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=150)
    parser.add_argument("--challenges", type=int, default=20)
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--setups", nargs="*", default=None)
    args = parser.parse_args()

    setups = [s for s in SETUPS if not args.setups or s.key in args.setups]
    results: dict = {"config": vars(args), "setups": {}}
    for setup in setups:
        print(f"\n== {setup.key}: {setup.title}  [{setup.paper_reference}]")
        honest, substitute = load_setup(setup)
        queries = evaluation_queries(setup, args.queries)

        correctness = measure_correctness(honest, queries, args.challenges, args.paths)
        other = measure_other_model(honest, substitute, queries, args.challenges, args.paths)
        cost = measure_cost(honest, queries, args.paths)
        print(f"   honest accepted {correctness['acceptance_rate']:.4f} of {correctness['runs']} runs; "
              f"zero-tolerance verifier accepts {correctness['zero_tolerance_acceptance_rate']:.2f}")
        print(f"   substitute: prediction agreement {other['prediction_agreement']:.4f}, "
              f"detected {other['detection_rate']:.4f} of {other['runs']} runs")
        print(f"   one path: prove {cost['prove_ms']:.2f} ms, verify {cost['verify_ms']:.2f} ms, "
              f"proof {cost['proof_kb']:.1f} kB")
        results["setups"][setup.key] = {"title": setup.title, "paper_reference": setup.paper_reference,
                                        "correctness": correctness, "other_model": other, "cost": cost}

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "reproduction.json", results)
    print(f"\nwrote {RESULTS / 'reproduction.json'}")


if __name__ == "__main__":
    main()
