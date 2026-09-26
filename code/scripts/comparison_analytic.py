"""Formula-derived costs and error bounds (``tables/analytic.csv``).

For every LLM configuration and sequence length, and for every security level,
this evaluates the closed-form model of ``pvi.fullcheck.analytic`` (proof bytes
per part, multiply-add counts, soundness bits) and the number of paths
Anchuri et al.'s protocol needs against a single-node tamper.  These rows are
``provenance=formula``; the benchmark's measured rows check the formulas (the
byte formula is exact -- see the tests) and the analytic rows extend them to
settings too large to run here, such as zkLLM's 2048-token prompts.

    python scripts/comparison_analytic.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from pvi.fullcheck.analytic import decoder_shapes, proof_bytes, sampling_paths, soundness_error_bits, work
from pvi.fullcheck.protocol import params_for
from pvi.fullcheck.transformer import CONFIGS, decoder_param_count

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "comparison" / "tables"

# Anchuri et al. report ~3.4 MB per path for Llama-2-7B (incl. weights, Sec. 8.2);
# per-path bytes for other models are measured by the benchmark (CNNs) or scaled
# here as (layers x widest input row) x (value + weight bytes) for LLMs.
ANCHURI_LLAMA7_PATH_BYTES = 3.4e6


def main() -> None:
    rows = []
    for name, cfg in CONFIGS.items():
        n_params = decoder_param_count(cfg)
        widest = max(cfg.d_ff, cfg.d_model, cfg.n_heads * cfg.head_dim)
        per_path = ANCHURI_LLAMA7_PATH_BYTES * (cfg.n_layers * cfg.d_model) / (32 * 4096)
        for seq in (1, 8, 64, 128, 256, 512, 1024, 2048):
            shapes = decoder_shapes(cfg, seq)
            trace = 2 * seq * cfg.n_layers * (4 * cfg.d_model + 3 * cfg.d_ff)   # fp16 activations
            open_all = 2 * n_params + trace
            for lam in (40, 80, 128):
                for mode, fs in (("C", False), ("C", True), ("Kpre", False)):
                    p = params_for(lam, len(shapes), rate=4, fiat_shamir=fs)
                    b = proof_bytes(shapes, p, mode)
                    w = work(shapes, p, mode)
                    rows.append({
                        "model": name, "params": n_params, "seq": seq, "lam": lam, "mode": mode,
                        "challenges": "fs" if fs else "int", "reps": p.reps, "columns": p.columns,
                        "soundness_bits": round(soundness_error_bits(shapes, p, mode), 2),
                        **{f"bytes_{k}": v for k, v in b.items()}, "bytes_total": sum(b.values()),
                        **{f"macs_{k}": v for k, v in w.items()},
                        "model_bytes_int8": n_params, "model_bytes_fp16": 2 * n_params,
                        "anchuri_width": widest, "anchuri_paths": sampling_paths(lam, widest),
                        "anchuri_path_bytes_est": per_path,
                        # k paths never cost more than opening everything once (fp16 weights,
                        # as in the paper, plus the committed trace); the naive k x one-path
                        # figure is kept separately for reference only.
                        "anchuri_bytes_naive": sampling_paths(lam, widest) * per_path,
                        "anchuri_open_all_bytes": open_all,
                        "anchuri_bytes_est": min(sampling_paths(lam, widest) * per_path, open_all),
                        "provenance": "formula",
                    })
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "analytic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} analytic rows")


if __name__ == "__main__":
    main()
