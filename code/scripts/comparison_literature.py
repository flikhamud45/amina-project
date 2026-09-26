"""Curated, numeric literature table for the comparison figures.

``artifacts/comparison/literature/reported_benchmarks.csv`` holds every number
the literature search extracted (413 rows, each with its table/page and a
verbatim snippet, each re-checked against the paper by a second reader).  This
script picks the rows the figures use, converts them to SI units (seconds,
bytes), and *links every curated row back to the catalogue row it came from*:
a curated entry whose source text cannot be found in the catalogue is an error.

Writes ``tables/reported_curated.csv`` and ``tables/systems.csv``.

    python scripts/comparison_literature.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "artifacts" / "comparison" / "literature"
TABLES = ROOT / "artifacts" / "comparison" / "tables"

KB, MB, MIB = 1e3, 1e6, 2 ** 20

# (system, model, params, seq, prover_s, verifier_s, proof_bytes, hardware, match=(system, model, text))
# ``match`` locates the catalogue row: the canonical system, a substring of its
# model field, and a substring of its prover/verifier/proof text.
CURATED = [
    # ---- CNNs / MLPs --------------------------------------------------------------------
    ("zkCNN", "LeNet-5 MNIST", 61.7e3, None, 0.441, 5.80e-3, 71.3 * KB, "AMD EPYC 7R32, 1 core", ("zkCNN", "LeNet", "0.441 s")),
    ("zkCNN", "VGG-11 CIFAR-10", 9.75e6, None, 47.8, 39.3e-3, 304 * KB, "AMD EPYC 7R32, 1 core", ("zkCNN", "VGG11", "47.8 s")),
    ("zkCNN", "VGG-16 CIFAR-10", 15.2e6, None, 88.3, 59.3e-3, 341 * KB, "AMD EPYC 7R32, 1 core", ("zkCNN", "VGG16 (13 conv", "88.3 s")),
    ("vCNN (re-run by zkCNN)", "LeNet-5 MNIST", 61.7e3, None, 5.49, 84e-3, 0.34 * KB, "AMD EPYC 7R32", ("zkCNN", "baseline vCNN, re-run", "5.49 s")),
    ("ZKML", "ResNet-18 CIFAR-10 (ZKML's 281K-param variant)", 280.9e3, None, 52.9, 11.84e-3, 15744, "AWS r6i.8xlarge, 32 vCPU", ("ZKML", "ResNet-18", "52.9 s (KZG)")),
    ("ZKML", "VGG-16 CIFAR-10", 15.2e6, None, 637.14, 9.62e-3, 12064, "AWS r6i.8xlarge, 32 vCPU", ("ZKML", "VGG16", "637.14 s (KZG)")),
    ("ZKML", "MobileNet v2 ImageNet", 3.5e6, None, 1225.5, 17.67e-3, 17664, "AWS r6i.16xlarge, 64 vCPU", ("ZKML", "MobileNet v2", "1225.5 s (KZG)")),
    ("ZKML", "GPT-2 (distilled)", 81.3e6, None, 3651.67, 18.70, 28128, "AWS r6i.32xlarge, 128 vCPU", ("ZKML", "GPT-2 (distilled)", "3651.67 s (KZG)")),
    ("ZENO", "VGG-16 CIFAR-10 (19.9M-FLOP variant)", None, None, 48.0, None, None, "Xeon Gold 5218", ("ZENO", "VggNet-16", "ZENO 48 s")),
    ("ZENO", "ResNet-18 CIFAR-10 (32.4M-FLOP variant)", None, None, 102.0, None, None, "Xeon Gold 5218", ("ZENO", "ResNet-18", "ZENO 102 s")),
    ("Bionetta", "MNIST MLP (2M)", 2.0e6, None, 3.05, 0.010, 0.88 * KB, "Xeon E5-2665, 16 threads", ("Bionetta", "MNIST dense MLP - Bionetta", "3.05 s")),
    ("Bionetta", "LeNet-5", None, None, 3.75, 0.010, 0.88 * KB, "Xeon E5-2665, 16 threads", ("Bionetta", "LeNet5 - Bionetta", "3.75 s")),
    ("Bionetta", "ResNet-18 (Bionetta's version)", None, None, 14.10, 0.015, 0.88 * KB, "Xeon E5-2665, 16 threads", ("Bionetta", "ResNet18 - Bionetta", "14.10 s")),
    ("EZKL (run by Bionetta)", "MNIST MLP (2M)", 2.0e6, None, 1310, 5.40, 127 * KB, "Xeon E5-2665, 16 threads", ("EZKL", "MNIST dense MLP", "1310 s")),
    ("EZKL (run by Bionetta)", "LeNet-5", None, None, 535, 2.00, 127 * KB, "Xeon E5-2665, 16 threads", ("EZKL", "LeNet5", "535 s")),
    ("EZKL (run by Bionetta)", "ResNet-18 (Bionetta's version)", None, None, 6840, 25.30, 180 * KB, "Xeon E5-2665, 16 threads", ("EZKL", "ResNet18", "6840 s")),
    ("ZEN", "LeNet-5-small CIFAR-10", None, None, 125.53, 0.023, 192, "Azure M32ls, 32 cores", ("ZEN", "LeNet-5-small", "125.53 s")),
    ("Mystique (interactive)", "ResNet-50 CIFAR-10", 23.5e6, None, 158, None, 0.53e9, "2x AWS m5.2xlarge, 200 Mbps", ("Mystique", "ResNet-50; public model", "158 s")),
    ("zkPyTorch", "VGG-16 CIFAR-10", 15.2e6, None, 2.2, None, None, "single CPU core", ("zkPyTorch", "VGG-16", "2.2 sec")),
    # ---- language models ----------------------------------------------------------------
    ("zkLLM", "OPT-125M", 125e6, 2048, 73.9, 0.342, 141 * KB, "A100 40GB", ("zkLLM", "OPT-125M", "73.9 s")),
    ("zkLLM", "OPT-350M", 350e6, 2048, 111, 0.593, 144 * KB, "A100 40GB", ("zkLLM", "OPT-350M", "111 s")),
    ("zkLLM", "OPT-1.3B", 1.3e9, 2048, 221, 0.899, 147 * KB, "A100 40GB", ("zkLLM", "OPT-1.3B", "221 s")),
    ("zkLLM", "OPT-2.7B", 2.7e9, 2048, 352, 1.41, 152 * KB, "A100 40GB", ("zkLLM", "OPT-2.7B", "352 s")),
    ("zkLLM", "OPT-6.7B", 6.7e9, 2048, 548, 2.08, 157 * KB, "A100 40GB", ("zkLLM", "OPT-6.7B", "548 s")),
    ("zkLLM", "OPT-13B", 13e9, 2048, 713, 3.71, 160 * KB, "A100 40GB", ("zkLLM", "OPT-13B", "713 s")),
    ("zkLLM", "Llama-2-7B", 6.74e9, 2048, 620, 2.36, 183 * KB, "A100 40GB", ("zkLLM", "Llama-2-7B", "620 s")),
    ("zkLLM", "Llama-2-13B", 13.0e9, 2048, 803, 3.95, 188 * KB, "A100 40GB", ("zkLLM", "Llama-2-13B", "803 s")),
    ("zkLLM (re-run by Anchuri)", "Llama-2-7B", 6.74e9, None, 388.3, None, None, "RTX 3090", ("zkLLM", "re-run by Anchuri", "388.3 s")),
    ("zkLLM (re-run by zkGPT)", "GPT-2", 124e6, None, 15.8, 0.54, 126 * KB, "A100", ("zkLLM", "GPT-2 (zkLLM re-run", "15.8 s")),
    ("zkGPT", "GPT-2", 124e6, None, 21.8, 0.35, 101 * KB, "Xeon 6126, 32 threads", ("zkGPT", "all optimisations", "21.8 s")),
    ("DeepProve", "GPT-2", 124e6, 64, 0.57 * 60, 1.35, 20.66 * MIB, "Ryzen 9 7950X3D, 16 cores", ("DeepProve", "BaseFold seq 64 (secondary)", "0.57 min")),
    ("DeepProve", "GPT-2", 124e6, 128, 0.85 * 60, 1.45, 21.82 * MIB, "Ryzen 9 7950X3D, 16 cores", ("DeepProve", "BaseFold seq 128 (secondary)", "0.85 min")),
    ("DeepProve", "GPT-2", 124e6, 256, 1.46 * 60, 1.55, 23.02 * MIB, "Ryzen 9 7950X3D, 16 cores", ("DeepProve", "BaseFold seq 256 (secondary)", "1.46 min")),
    ("DeepProve", "GPT-2", 124e6, 512, 2.94 * 60, 1.65, 24.33 * MIB, "Ryzen 9 7950X3D, 16 cores", ("DeepProve", "BaseFold seq 512 (secondary)", "2.94 min")),
    ("ZKTorch", "GPT-2 (ZKML's distilled model)", 81.3e6, None, 599, 12, None, "Xeon Platinum 8358, 64 threads", ("ZKTorch", "GPT-2", "599")),
    ("ZKTorch", "Llama-2-7B (1 token)", 6.74e9, 1, 2645.50, 100.14, 22.85 * MB, "Xeon Platinum 8358, 64 threads", ("ZKTorch", "LLaMA-2-7B", "2645.50 s")),
    ("Jolt Atlas", "GPT-2", 124e6, None, 38, None, None, "Apple M3 laptop", ("Jolt Atlas", "GPT-2", "~38 s")),
    ("SLP (5 sampled chunks)", "GPT-2", 124e6, None, 28.0, 0.16, 916 * 1024, "Apple M3 Pro laptop", ("SLP: Seal, Then Sample", "GPT-2", "28.0 s")),
    ("LAMP (1 layer, matmuls only)", "GPT-2-medium, one layer", 12.58e6, 1024, 230.82, 6.52, 3342724, "AMD EPYC 7B13, 32 cores", ("LAMP", "GPT-2", "230.82 s")),
    ("Anchuri et al. (1 path)", "Llama-2-7B", 6.74e9, 64, 5.8e-3, 12.44e-3, 3.4 * MB, "RTX 3090", ("Anchuri et al.", "verification", "12.44 ms")),
    ("Maverick (verif.-only, 1 thr.)", "Qwen3-4B", 4.02e9, 8, 0.3545, 0.0871, 36.08 * MB, "AWS c8i (client 1 thread)", ("Maverick", "verification-only mode", "354.5 ms")),
    ("Maverick (verif.-only, 8 thr.)", "Qwen3-4B", 4.02e9, 8, 0.3144, 0.0128, 36.08 * MB, "AWS c8i (client 8 threads)", ("Maverick", "verification-only mode", "314.4 ms")),
    ("Maverick (standard, 1 thr.)", "Qwen3-4B", 4.02e9, 8, 0.3633, 0.0896, 57.83 * MB, "AWS c8i (client 1 thread)", ("Maverick", "standard mode (8-token", "363.3 ms")),
]

SYSTEMS = [
    # system, family, setting, error type, stated error / security, trust assumptions
    ("Anchuri et al.", "sampling", "C", "empirical", "trace separation (no negligible bound); single-node tamper detected w.p. 1/N per path", "separation assumption; CM honest"),
    ("Ours (whole-network check)", "freivalds+RS", "C / K", "statistical", "<= sum_l [p^-r + ((k-1)/n)^t], set to 2^-lambda", "CM honest (as in Anchuri); interactive, or Fiat-Shamir with +64 bits"),
    ("SafetyNets", "interactive proof (sumcheck)", "K", "statistical", "3b sum n_i / p < 2^-30", "quadratic activations only"),
    ("Slalom", "freivalds + TEE", "K", "statistical", "40 bits per layer (p = 2^24-3, k = 2)", "SGX enclave trusted"),
    ("Maverick", "sparse freivalds + code", "K", "statistical", "kappa = 40 bits", "client knows weights (preprocessing)"),
    ("LAMP", "freivalds + code + SNARK", "C", "computational", "(k-1)/|F| + 4(1-delta)^t + ..., t=128, BN254", "commit-and-prove"),
    ("zkCNN", "GKR/sumcheck SNARK", "C", "computational", "O(d log|C|/|F|), BLS12-381 ('128-bit')", "discrete log, ROM"),
    ("zkLLM", "sumcheck + tlookup SNARK", "C", "computational", "negligible, |F| ~ 2^254 (BLS12-381)", "discrete log, ROM"),
    ("zkGPT", "GKR/sumcheck SNARK", "C", "computational", "~100 bits (BN254)", "discrete log, ROM"),
    ("DeepProve", "sumcheck + logup-GKR", "C", "computational", "round-by-round sound; not quantified", "ROM"),
    ("ZKML", "halo2 (Plonkish)", "C", "computational", "inherits halo2; not quantified", "KZG setup or IPA"),
    ("ZKTorch", "KZG poly-IOPs + folding", "C", "computational", "union bound, 'negligible' (BN254)", "AGM, KZG setup"),
    ("EZKL", "halo2 (Plonkish)", "C", "computational", "inherits halo2; not stated", "KZG setup"),
    ("Bionetta", "Groth16 / UltraGroth", "C", "computational", "GGM + ROM; not quantified", "trusted setup per model"),
    ("ZEN / ZENO / vCNN", "Groth16 R1CS", "C", "computational", "knowledge soundness (GGM/KEA); not quantified", "trusted setup"),
    ("Mystique", "VOLE interactive ZK", "C", "computational+statistical", "lambda = 128, rho >= 40", "designated verifier, interactive"),
    ("NanoZK", "halo2 per layer", "C", "computational", "3(L+2) 2^-128", "IPA, SHA-256 chain"),
    ("TOPLOC", "activation fingerprint", "K", "empirical", "FPR/FNR 0% in evaluation", "verifier re-runs the model"),
    ("SVIP", "hidden-state proxy", "K", "empirical", "FNR < 5%, FPR < 3% per query", "secret proxy task"),
    ("SLP", "sampled SNARK chunks", "C", "computational+sampling", "per audit <= q(M,b,s) + eps; coverage 3/161 on 70B", "BN254 HyperKZG"),
    ("Verde / opML / TAO", "refereed / optimistic", "C", "trust", "1-of-k honest (Verde/opML); empirical thresholds (TAO)", "an honest provider or validator"),
]


def _load_catalogue() -> list[dict]:
    with open(LIT / "reported_benchmarks.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _match(rows, system, model_sub, text) -> dict:
    hits = [r for r in rows if r["canonical_system"].startswith(system) and model_sub in (r["model"] or "")
            and any(text in (r[f] or "") for f in ("prover_time", "verifier_time", "proof_size"))]
    if not hits:
        raise SystemExit(f"curated entry not found in catalogue: {system!r} / {model_sub!r} / {text!r}")
    return hits[0]


def main() -> None:
    rows = _load_catalogue()
    out = []
    for system, model, params, seq, p, v, pi, hw, match in CURATED:
        src = _match(rows, *match)
        stated = (src["params"] or "").strip()
        source = "not stated" if params is None else ("paper" if stated and "FLOP" not in stated else "architecture")
        out.append({"system": system, "model": model, "params": params, "params_source": source, "seq": seq,
                    "prover_s": p,
                    "verifier_s": v, "proof_bytes": pi, "hardware": hw, "provenance": "reported",
                    "catalog_row_id": src["row_id"], "source_location": src["source_location"],
                    "venue": src["venue_year"], "snippet": src["snippet"]})
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "reported_curated.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(TABLES / "systems.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "family", "setting", "error_type", "stated_error", "trust"])
        w.writerows(SYSTEMS)
    print(f"{len(out)} curated rows (all linked to catalogue rows), {len(SYSTEMS)} systems")


if __name__ == "__main__":
    main()
