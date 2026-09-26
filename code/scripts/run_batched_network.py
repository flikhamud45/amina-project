"""Whole-network batched check: correctness and cost, at two model scales.

`scripts/run_batched_defence.py` checks a single layer. This one runs the
chained protocol over *every* layer -- the verifier recomputing each layer's
input from the previous layer's verified output, and the identity/logit layer
handled with its own constraint -- and reports what it costs.

The comparison that matters once every layer is checked is no longer the
sampling scheme's proof (there is nothing left to sample), but the two things
this construction sits between: downloading the model, and a full SNARK.

Usage
-----
    python scripts/run_batched_network.py [--model full|large] [--queries N]
                                          [--scale-bits B] [--code-queries T]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pvi.data import load_classification
from pvi.protocol.batched import (
    VerifierRandomness,
    BatchedWeightCommitment,
    fixed_point_network,
    prove_network,
    verify_network,
)
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for, mlp_architecture_large_for
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "results"

MODEL_CHOICES = {
    "full": ("mlp_mnist_full", mlp_architecture_for),
    "large": ("mlp_mnist_large", mlp_architecture_large_for),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="full", choices=sorted(MODEL_CHOICES))
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--scale-bits", type=int, default=8)
    parser.add_argument("--code-queries", type=int, default=24)
    parser.add_argument(
        "--challenges", default="verifier", choices=["verifier", "fiat-shamir"],
        help="'verifier' draws fresh randomness (no grinding, ~24-bit soundness "
             "as the parameters claim); 'fiat-shamir' hashes the prover's own "
             "message, which is non-interactive but grindable.",
    )
    args = parser.parse_args()

    name, architecture_fn = MODEL_CHOICES[args.model]
    network = load_network(architecture_fn(10), MODELS / f"{name}.npz")
    dataset = load_classification(name="mnist", seed=0)
    test_x = dataset.test_x.reshape(len(dataset.test_x), -1)
    arch = network.architecture
    sb, nq = args.scale_bits, args.code_queries

    widths = [layer.n_neurons for layer in arch.layers[1:]]
    activations = [layer.activation for layer in arch.layers[1:]]
    n_params = sum(w.size + b.size for w, b in network.parameters.values())
    print(f"model {name}: widths {widths}, activations {activations}")
    print(f"{n_params} parameters = {n_params * 4 / 1024 / 1024:.2f} MB as float32")
    print(f"scale 2^{sb}, {nq} spot-checked columns\n")

    interactive = args.challenges == "verifier"
    # A fresh draw per query: that is what "the verifier picks the challenge"
    # means, and reusing one across queries would replay challenges.
    def new_randomness():
        return VerifierRandomness() if interactive else None

    print(f"challenges: {args.challenges}"
          f"{' (fresh per interaction)' if interactive else ' (hash-derived, grindable)'}")

    views0 = fixed_point_network(network, test_x[0], scale_bits=sb)
    headroom = [int(np.abs(v.pre_activations()).max()) for v in views0]
    print(f"max|z| per layer {headroom} vs field half-width {views0[0].prime // 2}")

    started = time.perf_counter()
    commitments = [BatchedWeightCommitment(v.matrix, prime=v.prime) for v in views0]
    commit_seconds = time.perf_counter() - started
    print(f"commitment (one-off, all layers): {commit_seconds:.1f}s\n")

    accepted, sizes, per_layer, prove_ms, verify_ms = 0, [], None, [], []
    for query in test_x[: args.queries]:
        views = fixed_point_network(network, query, scale_bits=sb)
        t0 = time.perf_counter()
        randomness = new_randomness()
        proofs = prove_network(views, commitments, n_queries=nq, randomness=randomness)
        prove_ms.append((time.perf_counter() - t0) * 1e3)
        outputs = [v.outputs for v in views]
        t0 = time.perf_counter()
        ok = verify_network(arch, query, outputs, proofs, commitments,
                            scale_bits=sb, n_queries=nq, randomness=randomness)
        verify_ms.append((time.perf_counter() - t0) * 1e3)
        accepted += bool(ok)
        sizes.append(sum(p.size_bytes() for p in proofs))
        per_layer = [p.size_bytes() / 1024 for p in proofs]

    print(f"honest whole network accepted: {accepted}/{args.queries}")
    print(f"proof {np.mean(sizes)/1024:.1f} kB  (per layer: "
          f"{[round(x, 1) for x in per_layer]})")
    print(f"prover {np.mean(prove_ms):.1f} ms, verifier {np.mean(verify_ms):.1f} ms")

    # tamper at each layer in turn
    caught_rows = {}
    for index, activation in enumerate(activations):
        caught = total = 0
        for query in test_x[: args.queries]:
            views = fixed_point_network(network, query, scale_bits=sb)
            z = np.array(views[index].pre_activations(), dtype=np.int64)
            bad = np.maximum(z, 0) if activation == "relu" else z.copy()
            if activation == "relu":
                live = np.flatnonzero(bad > 0)
                if not len(live):
                    continue
                bad[live[0]] = 0                     # the zero-hiding tamper
            else:
                bad[0] = int(bad.min()) - 1000
            total += 1
            tv = fixed_point_network(network, query, scale_bits=sb,
                                     tampered={index + 1: bad})
            randomness = new_randomness()
            proofs = prove_network(tv, commitments, n_queries=nq, randomness=randomness)
            ok = verify_network(arch, query, [v.outputs for v in tv], proofs,
                                commitments, scale_bits=sb, n_queries=nq,
                                randomness=randomness)
            caught += not ok
        caught_rows[f"layer{index + 1}_{activation}"] = caught / max(total, 1)
        print(f"tamper at layer {index+1} ({activation:>8}): caught {caught}/{total}")

    proof_kb = float(np.mean(sizes)) / 1024
    model_mb = n_params * 4 / 1024 / 1024
    print(f"\nproof is {model_mb * 1024 / proof_kb:.1f}x smaller than the model")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": name, "widths": widths, "activations": activations,
        "parameters": int(n_params), "scale_bits": sb, "code_queries": nq,
        "max_abs_z_per_layer": headroom,
        "commit_seconds": commit_seconds,
        "honest_accepted": accepted, "queries": args.queries,
        "proof_kb": proof_kb, "proof_kb_per_layer": per_layer,
        "prove_ms": float(np.mean(prove_ms)), "verify_ms": float(np.mean(verify_ms)),
        "challenges": args.challenges,
        "tamper_detection": caught_rows,
        "model_mb": model_mb,
        "proof_smaller_than_model_x": model_mb * 1024 / proof_kb,
    }
    write_json(RESULTS / f"batched_network_{name}.json", payload)
    print(f"wrote {RESULTS / f'batched_network_{name}.json'}")


if __name__ == "__main__":
    main()
