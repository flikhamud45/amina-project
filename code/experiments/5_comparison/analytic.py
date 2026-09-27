"""The path test's cost of reaching 2^-40 on Llama-2-7B (report Sec. 4.2), ``tables/analytic.csv``.

A single tampered neuron in the widest layer (the MLP's ``d_ff`` inputs) is on one
random path with probability 1/width, so 2^-40 needs ``paths_for(40, 1 / width)``
paths; that many paths open every committed row, i.e. the whole model (fp16, as in
Anchuri et al.) plus the trace.

    python experiments/5_comparison/analytic.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from pvi.fullcheck.sampling import paths_for
from pvi.fullcheck.transformer import CONFIGS, decoder_param_count

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "artifacts" / "comparison" / "tables"


def main() -> None:
    name, seq, lam = "llama2-7b", 64, 40
    cfg = CONFIGS[name]
    n_params = decoder_param_count(cfg)
    widest = max(cfg.d_ff, cfg.d_model, cfg.n_heads * cfg.head_dim)
    trace = 2 * seq * cfg.n_layers * (4 * cfg.d_model + 3 * cfg.d_ff)   # fp16 activations
    row = {"model": name, "params": n_params, "seq": seq, "lam": lam, "anchuri_width": widest,
           "anchuri_paths": paths_for(lam, 1.0 / widest), "anchuri_open_all_bytes": 2 * n_params + trace}
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "analytic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
