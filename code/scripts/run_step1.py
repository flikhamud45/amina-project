"""Step 1: reproduce the protocol of Anchuri et al. and validate it.

Runs four blocks of experiments and writes machine-readable results plus figures.

1. **Correctness (Definition 4).**  The honest prover is accepted on every run.
   We also report the distribution of local-check residuals, which is what
   justifies the verifier's tolerance -- and what a zero-tolerance verifier would
   fall foul of.

2. **Other-model soundness (Definition 1, Section 6).**  For each substitution
   setup: how functionally similar the two models are, the Equation (1) separation
   values per layer, the per-layer Jensen-Shannon divergence, and the rate at
   which ``RandPathTest`` rejects the substitute's honest trace.

3. **Soundness parameters (Appendix I).**  The empirical estimation procedure for
   ``delta_out``, ``delta_trace``, ``eps_sep`` and ``eps_tst``, yielding the bound
   of Theorem 1.

4. **Cost (Section 8).**  Prover and verifier wall-clock time and proof size.

Usage
-----
    python scripts/run_step1.py [--queries N] [--challenges N] [--setups KEY ...]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pvi.experiments.estimation import (
    build_separation_dataset,
    select_parameters,
    valid_layers,
)
from pvi.experiments.separation import (
    equation1_separation,
    js_divergence,
    layer_js_divergences,
    verifier_residuals,
)
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol import (
    ModelCommitment,
    ProtocolParams,
    Prover,
    UniformPathSampler,
    Verifier,
    run_protocol,
    sample_challenge,
)
from pvi.training import load_network
from pvi.zoo import SETUPS, SubstitutionSetup, quantise_network
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"
FIGURES = ROOT / "artifacts" / "figures"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_setup(setup: SubstitutionSetup) -> tuple[TracedNetwork, TracedNetwork]:
    """Return ``(M, M~)`` for a setup, training-free."""
    architecture = setup.honest.architecture()
    honest = load_network(architecture, MODELS / f"{setup.honest.name}.npz")
    if setup.is_derived():
        return honest, quantise_network(honest, bits=setup.quantise_bits or 8)
    assert setup.substitute is not None
    substitute = load_network(
        setup.substitute.architecture(), MODELS / f"{setup.substitute.name}.npz"
    )
    return honest, substitute


def evaluation_queries(setup: SubstitutionSetup, count: int, seed: int = 0) -> np.ndarray:
    dataset = setup.eval_dataset()
    pool = dataset.test_x.reshape(len(dataset.test_x), -1)
    rng = np.random.default_rng(seed)
    index = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
    return np.ascontiguousarray(pool[index], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. Correctness
# --------------------------------------------------------------------------- #


def measure_correctness(
    network: TracedNetwork, queries: np.ndarray, challenges: int, n_paths: int
) -> dict:
    params = ProtocolParams(n_paths=n_paths, check_full_input=True)
    commitment = ModelCommitment(network)
    prover = Prover(network, commitment, params)
    verifier = Verifier.from_commitment(commitment, params)

    accepted, runs = 0, 0
    residuals: list[float] = []
    for query in queries:
        for _ in range(challenges):
            result = run_protocol(prover, verifier, query)
            accepted += int(result.accepted)
            runs += 1
            residuals.extend(result.residuals)

    residual_array = np.asarray(residuals)

    # How the same runs would fare under a zero-tolerance verifier, i.e. the
    # idealised exact-equality test of Figure 3.
    exact_params = ProtocolParams(
        n_paths=n_paths, abs_tolerance=0.0, rel_tolerance=0.0, check_full_input=True
    )
    exact_prover = Prover(network, commitment, exact_params)
    exact_verifier = Verifier.from_commitment(commitment, exact_params)
    exact_accepted = sum(
        run_protocol(exact_prover, exact_verifier, query).accepted
        for query in queries[: min(len(queries), 200)]
    )
    exact_runs = min(len(queries), 200)

    return {
        "runs": runs,
        "accepted": accepted,
        "acceptance_rate": accepted / runs,
        "nodes_checked": int(residual_array.size),
        "residual_max": float(residual_array.max()),
        "residual_mean": float(residual_array.mean()),
        "residual_p999": float(np.quantile(residual_array, 0.999)),
        "tolerance": params.abs_tolerance,
        "headroom_factor": float(params.abs_tolerance / max(residual_array.max(), 1e-30)),
        "zero_tolerance_runs": exact_runs,
        "zero_tolerance_accepted": int(exact_accepted),
        "zero_tolerance_acceptance_rate": exact_accepted / exact_runs,
    }


# --------------------------------------------------------------------------- #
# 2. Other-model soundness
# --------------------------------------------------------------------------- #


def measure_other_model_soundness(
    honest_net: TracedNetwork,
    substitute_net: TracedNetwork,
    queries: np.ndarray,
    challenges: int,
    n_paths: int,
) -> dict:
    architecture = honest_net.architecture
    params = ProtocolParams(n_paths=n_paths, check_full_input=True)
    commitment = ModelCommitment(honest_net)
    prover = Prover(honest_net, commitment, params)
    verifier = Verifier.from_commitment(commitment, params)

    honest_traces = [honest_net.eval_trace(q) for q in queries]
    substitute_traces = [substitute_net.eval_trace(q) for q in queries]

    honest_pred = np.asarray([int(t.output.argmax()) for t in honest_traces])
    substitute_pred = np.asarray([int(t.output.argmax()) for t in substitute_traces])
    output_distance = np.asarray(
        [
            float(np.linalg.norm(h.output.astype(np.float64) - s.output.astype(np.float64)))
            for h, s in zip(honest_traces, substitute_traces)
        ]
    )

    # Equation (1), per layer, pooled over queries.
    eq1: dict[int, dict[str, float]] = {}
    for layer_index in range(1, len(architecture)):
        pooled = np.concatenate(
            [
                equation1_separation(honest_net, h, s, layer_index=layer_index)
                for h, s in zip(honest_traces, substitute_traces)
            ]
        )
        eq1[layer_index] = {
            "layer": architecture[layer_index].name,
            "mean": float(pooled.mean()),
            "min": float(pooled.min()),
            "max": float(pooled.max()),
            "median": float(np.median(pooled)),
        }

    # The residual RandPathTest actually tests, on the substitute's trace.
    residual_stats: dict[int, dict[str, float]] = {}
    for layer_index in range(1, len(architecture)):
        pooled = np.concatenate(
            [
                verifier_residuals(honest_net, s, layer_index=layer_index)[layer_index]
                for s in substitute_traces
            ]
        )
        residual_stats[layer_index] = {
            "layer": architecture[layer_index].name,
            "mean": float(pooled.mean()),
            "min": float(pooled.min()),
            "fraction_above_tolerance": float(
                (pooled > params.abs_tolerance).mean()
            ),
        }

    divergences = layer_js_divergences(honest_traces, substitute_traces)

    # Appendix I.1 computes JS "for path-selected neurons".  Since paths select
    # uniformly, that is a subsample of the same population and should give the
    # same answer; we record both so the claim can be checked rather than assumed.
    sampler = UniformPathSampler()
    rng = np.random.default_rng(0)
    sampled_paths = [sampler.sample(architecture, rng) for _ in range(50)]
    path_divergences: dict[str, float] = {}
    for layer_index in range(1, len(architecture)):
        selected = np.asarray([p[layer_index] for p in sampled_paths])
        left = np.concatenate([t[layer_index][selected] for t in honest_traces])
        right = np.concatenate([t[layer_index][selected] for t in substitute_traces])
        path_divergences[str(layer_index)] = js_divergence(left, right)

    detections, runs = 0, 0
    for query, trace in zip(queries, substitute_traces):
        for _ in range(challenges):
            runs += 1
            detections += int(not run_protocol(prover, verifier, query, trace=trace).accepted)

    return {
        "functional": {
            "prediction_agreement": float((honest_pred == substitute_pred).mean()),
            "output_distance_mean": float(output_distance.mean()),
            "output_distance_min": float(output_distance.min()),
        },
        "equation1_by_layer": eq1,
        "verifier_residual_by_layer": residual_stats,
        "js_divergence_by_layer": {str(k): float(v) for k, v in divergences.items()},
        "js_divergence_path_selected": path_divergences,
        "detection": {
            "runs": runs,
            "detections": detections,
            "detection_rate": detections / runs,
        },
    }


# --------------------------------------------------------------------------- #
# 3. Appendix I
# --------------------------------------------------------------------------- #


def estimate_soundness_parameters(
    honest_net: TracedNetwork,
    queries: np.ndarray,
    honest_traces: list[Trace],
    substitute_traces: list[Trace],
    divergences: dict[str, float],
    *,
    repetitions: int,
) -> dict:
    valid = valid_layers({int(k): v for k, v in divergences.items()}, threshold=0.05)
    if not valid:
        return {"status": "no layer passed the JS filter", "valid_layers": []}

    dataset = build_separation_dataset(honest_traces, substitute_traces, valid)

    params = ProtocolParams(n_paths=1, check_full_input=True)
    commitment = ModelCommitment(honest_net)
    prover = Prover(honest_net, commitment, params)
    verifier = Verifier.from_commitment(commitment, params)

    def accept_fn(index: int, trace: Trace) -> bool:
        return run_protocol(prover, verifier, queries[index], trace=trace).accepted

    estimate = select_parameters(
        dataset,
        accept_fn,
        target_eps_sep=0.01,
        target_eps_tst=0.05,
        repetitions=repetitions,
    )
    out: dict = {
        "valid_layers": list(valid),
        "n_queries": len(dataset),
        "d_out_mean": float(dataset.d_out.mean()),
        "d_trace_mean": float(dataset.d_trace.mean()),
    }
    if estimate is None:
        out["status"] = "no candidate met the target test error"
    else:
        out["status"] = "ok"
        out["estimate"] = asdict(estimate)
        out["soundness_error"] = estimate.soundness_error
    return out


# --------------------------------------------------------------------------- #
# 4. Cost
# --------------------------------------------------------------------------- #


def measure_cost(network: TracedNetwork, queries: np.ndarray, n_paths: int) -> dict:
    params = ProtocolParams(n_paths=n_paths, check_full_input=True)

    started = time.perf_counter()
    commitment = ModelCommitment(network)
    commit_model_s = time.perf_counter() - started

    prover = Prover(network, commitment, params)
    verifier = Verifier.from_commitment(commitment, params)

    sample = queries[: min(len(queries), 100)]
    prove1_s, prove2_s, verify_s, sizes, weight_rows = [], [], [], [], []
    for query in sample:
        challenge = sample_challenge()

        t0 = time.perf_counter()
        round1, state = prover.prove1(query)
        t1 = time.perf_counter()
        round2 = prover.prove2(state, challenge)
        t2 = time.perf_counter()
        result = verifier.verify(query, round1, round2, challenge)
        t3 = time.perf_counter()

        prove1_s.append(t1 - t0)
        prove2_s.append(t2 - t1)
        verify_s.append(t3 - t2)
        sizes.append(result.proof_size_bytes)
        weight_rows.append(result.n_weight_rows_opened)

    return {
        "commit_model_ms": commit_model_s * 1e3,
        "prove1_ms_mean": float(np.mean(prove1_s) * 1e3),
        "prove2_ms_mean": float(np.mean(prove2_s) * 1e3),
        "verify_ms_mean": float(np.mean(verify_s) * 1e3),
        "proof_size_kb_mean": float(np.mean(sizes) / 1024),
        "weight_rows_opened_mean": float(np.mean(weight_rows)),
        "trace_entries": network.architecture.n_trace_entries,
        "n_paths": n_paths,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_separation(results: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for key, block in results["setups"].items():
        eq1 = block["other_model"]["equation1_by_layer"]
        layers = sorted(eq1, key=int)
        axes[0].plot(
            [int(k) for k in layers],
            # Equation (1) is identically zero at layer 1: both of its terms use
            # M's weights and the input layer is the shared query.  Clamp for the
            # log axis rather than silently dropping the point.
            [max(eq1[k]["mean"], 1e-12) for k in layers],
            marker="o",
            label=key,
        )
        js = block["other_model"]["js_divergence_by_layer"]
        js_layers = sorted(js, key=int)
        axes[1].plot(
            [int(k) for k in js_layers], [js[k] for k in js_layers], marker="s", label=key
        )

    axes[0].set_xlabel("layer index")
    axes[0].set_ylabel("mean separation, Eq. (1)")
    axes[0].set_yscale("log")
    axes[0].set_title("Trace separation by layer")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.05, color="k", ls="--", lw=1, label="Appendix I filter")
    axes[1].set_xlabel("layer index")
    axes[1].set_ylabel("Jensen-Shannon divergence")
    axes[1].set_title("Activation-distribution divergence")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "step1_separation.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=150)
    parser.add_argument("--challenges", type=int, default=20)
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--setups", nargs="*", default=None)
    args = parser.parse_args()

    setups = SETUPS
    if args.setups:
        wanted = set(args.setups)
        setups = tuple(s for s in SETUPS if s.key in wanted)
        if not setups:
            raise SystemExit(f"no setup matched {sorted(wanted)}")

    results: dict = {
        "config": vars(args),
        "setups": {},
    }

    for setup in setups:
        print(f"\n{'=' * 78}\n{setup.key}: {setup.title}  [{setup.paper_reference}]\n{'=' * 78}")
        honest_net, substitute_net = load_setup(setup)
        queries = evaluation_queries(setup, args.queries)
        print(f"  {len(queries)} queries, architecture {honest_net.architecture.layer_widths}")

        print("  [1/4] correctness ...")
        correctness = measure_correctness(honest_net, queries, args.challenges, args.paths)
        print(
            f"        acceptance {correctness['acceptance_rate']:.4f} over "
            f"{correctness['runs']} runs; max residual {correctness['residual_max']:.3e} "
            f"({correctness['headroom_factor']:.0f}x below tolerance)"
        )
        print(
            f"        zero-tolerance verifier acceptance: "
            f"{correctness['zero_tolerance_acceptance_rate']:.4f}"
        )

        print("  [2/4] other-model soundness ...")
        other = measure_other_model_soundness(
            honest_net, substitute_net, queries, args.challenges, args.paths
        )
        print(
            f"        prediction agreement {other['functional']['prediction_agreement']:.4f}; "
            f"detection rate {other['detection']['detection_rate']:.4f} "
            f"over {other['detection']['runs']} runs"
        )

        print("  [3/4] Appendix I parameter estimation ...")
        honest_traces = [honest_net.eval_trace(q) for q in queries]
        substitute_traces = [substitute_net.eval_trace(q) for q in queries]
        estimation = estimate_soundness_parameters(
            honest_net,
            queries,
            honest_traces,
            substitute_traces,
            other["js_divergence_by_layer"],
            repetitions=args.repetitions,
        )
        print(f"        {estimation.get('status')}")
        if estimation.get("estimate"):
            e = estimation["estimate"]
            print(
                f"        delta_out={e['delta_out']:.4f} delta_trace={e['delta_trace']:.6f} "
                f"eps_tst={e['eps_tst']:.4f} -> soundness error <= "
                f"{estimation['soundness_error']:.4f}"
            )

        print("  [4/4] cost ...")
        cost = measure_cost(honest_net, queries, args.paths)
        print(
            f"        prove {cost['prove1_ms_mean'] + cost['prove2_ms_mean']:.2f} ms, "
            f"verify {cost['verify_ms_mean']:.2f} ms, "
            f"proof {cost['proof_size_kb_mean']:.1f} kB, "
            f"{cost['weight_rows_opened_mean']:.1f} weight rows opened"
        )

        results["setups"][setup.key] = {
            "title": setup.title,
            "paper_reference": setup.paper_reference,
            "layer_widths": list(honest_net.architecture.layer_widths),
            "correctness": correctness,
            "other_model": other,
            "estimation": estimation,
            "cost": cost,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "step1.json"
    write_json(out_path, results)
    plot_separation(results)
    print(f"\nwrote {out_path}")
    print(f"wrote {FIGURES / 'step1_separation.png'}")


if __name__ == "__main__":
    main()
