"""The path test's cost of soundness ``2^-lam`` against one edited neuron on a decoder (Sec. 3.5, ``sec:break``).

A path visits at most one neuron of a given layer and picks it uniformly, so a single edited neuron of the widest
layer (for Llama-2-7B the FFN's ``d_ff = 11,008`` inputs of the down projection) is caught by one path with
probability at most ``1 / width``, and ``2^-lam`` needs at least ``paths_for(lam, 1 / width)`` paths.  That many
paths open every committed row: the whole model in fp16 (the weights of Anchuri et al.) plus the fp16 trace of
``--seq`` tokens.  Backs "at least 305,193 [paths] in the FFN layers of Llama-2-7B (width 11,008), about 13.7 GB of
fp16 openings at 64 tokens" (Sec. 3.5; ``tab:sok-derivation`` and ``ext:attack-measure`` of the extended version).

Usage (from ``code/``)::

    python -m experiments.attack.path_test_bound                                 # Llama-2-7B, 64 tokens, lam = 40
    python -m experiments.attack.path_test_bound --model llama2-7b opt-6.7b --seq 64 --lam 40 --out /tmp/analytic.csv

Writes one CSV row per model to ``--out`` (default ``artifacts/comparison/tables/analytic.csv``; the run package's
attack tier writes ``logs/<name>/attack/analytic.csv``, which ``experiments/paper/numbers.py`` reads for the macros
``PathsLlama``, ``WidthLlama`` and ``OpenAllLlama``): ``model``, ``params`` (weight-product parameters), ``seq``,
``lam``, ``anchuri_width``, ``anchuri_paths``, ``anchuri_open_all_bytes``; and prints each row as one JSON line.  Shapes
only (``pvi.models.decoder.CONFIGS``): no weights, no GPU, a few seconds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.common.paths import BENCH
from experiments.common.records import emit
from pvi.baseline.path_int import paths_for
from pvi.models.decoder import CONFIGS, decoder_param_count


def bound(name: str, seq: int, lam: int) -> dict:
    """The CSV row of one model."""
    cfg = CONFIGS[name]
    n_params = decoder_param_count(cfg)
    widest = max(cfg.d_ff, cfg.d_model, cfg.n_heads * cfg.head_dim)
    trace = 2 * seq * cfg.n_layers * (4 * cfg.d_model + 3 * cfg.d_ff)   # fp16 activations
    return {"model": name, "params": n_params, "seq": seq, "lam": lam, "anchuri_width": widest,
            "anchuri_paths": paths_for(lam, 1.0 / widest), "anchuri_open_all_bytes": 2 * n_params + trace}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=["llama2-7b"], choices=sorted(CONFIGS),
                        help="decoder shapes of pvi.models.decoder.CONFIGS")
    parser.add_argument("--seq", type=int, default=64, help="tokens of the opened trace")
    parser.add_argument("--lam", type=int, default=40, help="soundness 2^-lam")
    parser.add_argument("--out", default=str(BENCH / "tables" / "analytic.csv"))
    args = parser.parse_args()

    rows = [bound(name, args.seq, args.lam) for name in args.model]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        emit(row)


if __name__ == "__main__":
    main()
