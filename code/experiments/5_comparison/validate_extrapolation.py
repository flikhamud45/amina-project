"""Check the linear block extrapolation of the LLM suite against full builds (report Sec. 4.1).

Reads ``tables_<platform>/measured_summary.csv`` (run ``aggregate.py --platform <p>`` first) and,
for every (model, seq, mode, challenges, lam, variant, metric) with a full build and at least
two other block counts (the report's: 1 and 2 blocks), compares

* the report's two-point rule  ``m1 + (L - 1) * (m2 - m1)``  with the measured full build, and
* a least-squares line through every measured block count (curvature shows up as residuals).

    python experiments/5_comparison/validate_extrapolation.py --platform rtx2080ti-v2
writes ``tables_<platform>/llm_extrapolation_validation.csv`` and prints the worst cases.  The
text's error bounds are the totals of ``text_numbers.py``; this is the per-part detail.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "comparison"
METRICS = {"prove_forward", "prove_fold", "prove_open", "fs_hash", "verify_derive", "verify_fold",
           "verify_products", "verify_columns", "bytes_total", "bytes_claims", "bytes_paths",
           "gpu_peak_memory", "gpu_peak_reserved", "host_peak_rss"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM") or "rtx2080ti-v2",
                    help="read tables_<platform>/ (default: $PVI_PLATFORM, else rtx2080ti-v2, the report's numbers)")
    TABLES = BASE / f"tables_{ap.parse_args().platform}"
    by: dict[tuple, dict[int, float]] = defaultdict(dict)
    full_l: dict[tuple, int] = {}
    with open(TABLES / "measured_summary.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["suite"] != "llm" or not r["cell"].startswith("defence_") or r["metric"] not in METRICS:
                continue
            if r.get("batch") or r.get("attack") or r.get("stage"):
                continue
            key = (r["model"], r["cfg_seq"], r["cfg_mode"], r["cfg_challenges"], r["cfg_lam"],
                   r.get("cfg_variant", "") + f"|rate{r.get('cfg_rate', '')}|thr{r.get('cfg_threads', '')}",
                   r.get("cfg_lean", ""),   # never fit lean and non-lean builds together
                   r.get("prover_hw", ""), r["metric"])
            by[key][int(r["cfg_n_layers"])] = float(r["median"])
            full_l[key] = int(r["cfg_n_layers_full"])
    out = []
    for key, pts in sorted(by.items()):
        L = full_l[key]
        if L not in pts or len(pts) < 3:
            continue
        row = dict(zip(("model", "seq", "mode", "challenges", "lam", "variant", "lean", "prover_hw", "metric"), key))
        row.update(n_layers_full=L, builds=" ".join(str(l) for l in sorted(pts)), measured_full=pts[L])
        if 1 in pts and 2 in pts:
            ext = pts[1] + (L - 1) * (pts[2] - pts[1])
            row.update(extrapolated_2pt=ext, rel_err_2pt=(ext - pts[L]) / pts[L] if pts[L] else None)
        xs, ys = zip(*sorted(pts.items()))
        n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        icpt = my - slope * mx
        resid = [(y - (icpt + slope * x)) / y for x, y in zip(xs, ys) if y]
        row.update(fit_per_block=slope, fit_fixed=icpt, fit_max_rel_resid=max(map(abs, resid)) if resid else None)
        out.append(row)
    if not out:
        raise SystemExit("no model with a full build and at least three block counts")
    path = TABLES / "llm_extrapolation_validation.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in out for k in r)))
        w.writeheader()
        w.writerows(out)
    worst = sorted((r for r in out if r.get("rel_err_2pt") is not None), key=lambda r: -abs(r["rel_err_2pt"]))
    for r in worst[:25]:
        print(f"{r['model']:10s} T{r['seq']:>5s} {r['mode']:4s} {r['challenges']:3s} lam{r['lam']:>3s} {r['metric']:17s} "
              f"full {r['measured_full']:.4g}  2-pt {r['extrapolated_2pt']:.4g} ({100 * r['rel_err_2pt']:+.1f}%)  "
              f"line-fit max resid {100 * (r['fit_max_rel_resid'] or 0):.1f}%")
    print(f"{len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
