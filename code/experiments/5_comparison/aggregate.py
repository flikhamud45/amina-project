"""Rebuild ``artifacts/comparison/tables/`` from the raw benchmark records.

Deterministic and idempotent: it only reads ``raw/**/*.jsonl`` (finished cells)
and writes

* ``measured_long.csv``    -- every raw record, one row each, config flattened;
* ``measured_summary.csv`` -- per (suite, model, cell, metric, group) statistics;
* ``llm_full_model.csv``   -- full-model LLM costs: measured when all layers were
  built, otherwise extrapolated linearly from 1- and 2-block builds
  (``full = m1 + (L - 1) * (m2 - m1)``; every block has identical shapes).

    python experiments/5_comparison/aggregate.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "comparison" / "raw"
TABLES = ROOT / "artifacts" / "comparison" / "tables"
GROUP_KEYS = ("batch", "attack", "lam", "op", "stage", "k")
ADDITIVE = {"prove_forward", "prove_fold", "prove_open", "verify_derive", "verify_products", "verify_columns",
            "verify_fold", "fs_hash", "bytes_claims", "bytes_u", "bytes_columns", "bytes_paths", "bytes_total",
            "commit_total", "verifier_precompute", "bytes_paths_multiproof", "bytes_total_multiproof"}


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(RAW.rglob("*.jsonl")):
        if not path.with_suffix(".done").exists():
            continue
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                if row["metric"] == "env":
                    continue
                cfg = row.pop("config", {}) or {}
                for k, v in cfg.items():
                    row["cfg_" + k] = v
                rows.append(row)
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    for r in rows:
        if not isinstance(r["value"], (int, float)):
            continue
        key = (r["suite"], r["model"], r["cell"], r["metric"]) + tuple(str(r.get(k, "")) for k in GROUP_KEYS)
        groups[key].append(float(r["value"]))
        meta[key] = {k: v for k, v in r.items()
                     if k.startswith("cfg_") or k in ("hw", "prover_hw", "verifier_hw", "host", "git_sha", "unit",
                                                      "bits")}
    out = []
    for key, vals in sorted(groups.items()):
        vals.sort()
        q = lambda f: vals[min(len(vals) - 1, int(f * (len(vals) - 1) + 0.5))]
        out.append({"suite": key[0], "model": key[1], "cell": key[2], "metric": key[3],
                    **{k: key[4 + i] for i, k in enumerate(GROUP_KEYS)},
                    "n": len(vals), "median": statistics.median(vals), "mean": statistics.fmean(vals),
                    "p25": q(0.25), "p75": q(0.75), "min": vals[0], "max": vals[-1], **meta[key]})
    return out


def multiproof_adjust(summary: list[dict]) -> list[dict]:
    """Runs made before multiproofs sent one Merkle path per opened column.  Their
    Merkle term is replaced by the exact *expected* multiproof size for the same
    trees and number of columns (the columns are uniform, so this is the mean of
    what a multiproof run would send); the measured value is kept alongside."""
    from pvi.fullcheck.analytic import decoder_shapes, expected_multiproof_nodes
    from pvi.fullcheck.transformer import CONFIGS

    extra = []
    for s in summary:
        if not (s["suite"] == "llm" and s["cell"].startswith("defence_") and s.get("cfg_mode") == "C"
                and s.get("cfg_merkle", "") != "multiproof" and s["metric"] in ("bytes_paths", "bytes_total")):
            continue
        cfg = CONFIGS[s["model"]]
        shapes = decoder_shapes(cfg, int(s["cfg_seq"]), n_layers=int(s["cfg_n_layers"]))
        t, rate = int(s["cfg_columns"]), int(s["cfg_rate"])
        mp = sum(32 * expected_multiproof_nodes(sh.n_points(rate), t) for sh in shapes)
        if s["metric"] == "bytes_paths":
            extra.append(dict(s, metric="bytes_paths_multiproof", median=mp, mean=mp, p25=mp, p75=mp, min=mp, max=mp,
                              provenance="expected multiproof for the measured trees"))
        else:
            paths = next(x for x in summary if x["cell"] == s["cell"] and x["model"] == s["model"]
                         and x["metric"] == "bytes_paths" and x["suite"] == "llm")
            v = s["median"] - paths["median"] + mp
            extra.append(dict(s, metric="bytes_total_multiproof", median=v, mean=v, p25=v, p75=v, min=v, max=v,
                              provenance="measured data + expected multiproof"))
    return extra


def llm_full(summary: list[dict]) -> list[dict]:
    """Full-model LLM costs per (model, seq, mode, challenges, lam, metric)."""
    by = defaultdict(dict)
    for s in summary:
        if s["suite"] != "llm" or not s["cell"].startswith("defence_") or s["metric"] not in ADDITIVE:
            continue
        key = (s["model"], s["cfg_seq"], s["cfg_mode"], s["cfg_challenges"], s["cfg_lam"], s["metric"],
               s.get("cfg_threads", ""), s.get("cfg_rate", ""), s.get("cfg_variant", ""), s.get("cfg_device", ""))
        by[key][int(s["cfg_n_layers"])] = s
    out = []
    for (model, seq, mode, chal, lam, metric, threads, rate, variant, device), builds in sorted(
            by.items(), key=lambda kv: str(kv[0])):
        any_ = next(iter(builds.values()))
        L = int(any_["cfg_n_layers_full"])
        base = {"model": model, "seq": seq, "mode": mode, "challenges": chal, "lam": lam, "metric": metric,
                "threads": threads, "rate": rate, "variant": variant, "device": device,
                "prover_hw": any_.get("prover_hw"), "verifier_hw": any_.get("verifier_hw"),
                "n_layers_full": L, "params_full": any_.get("cfg_params_full"), "hw": any_.get("hw")}
        if L in builds:
            out.append(dict(base, value=builds[L]["median"], provenance="measured"))
        elif 1 in builds and 2 in builds:
            if builds[1].get("host") != builds[2].get("host") and builds[1].get("prover_hw") != builds[2].get("prover_hw"):
                continue  # never extrapolate across different machines
            m1, m2 = builds[1]["median"], builds[2]["median"]
            out.append(dict(base, value=m1 + (L - 1) * (m2 - m1), provenance="extrapolated_from_1_and_2_blocks",
                            value_1block=m1, value_2blocks=m2))
    # the one-time commitment, from the commit cells
    cb = defaultdict(dict)
    for s in summary:
        if s["suite"] == "llm" and s["cell"].startswith("commit_") and s["metric"] == "commit_total":
            key = (s["model"], s["cfg_seq"], s.get("cfg_rate", ""), s.get("cfg_variant", ""))
            cb[key][int(s["cfg_n_layers"])] = s
    for (model, seq, rate, variant), b in sorted(cb.items(), key=lambda kv: str(kv[0])):
        any_ = next(iter(b.values()))
        if not any_.get("cfg_n_layers_full"):
            continue
        L = int(any_["cfg_n_layers_full"])
        base = {"model": model, "seq": seq, "mode": "C", "challenges": "", "lam": "", "metric": "commit_total",
                "threads": "", "rate": rate, "variant": variant, "device": any_.get("cfg_device", ""),
                "n_layers_full": L, "params_full": any_.get("cfg_params_full"),
                "prover_hw": any_.get("prover_hw"), "verifier_hw": any_.get("verifier_hw")}
        if L in b:
            out.append(dict(base, value=b[L]["median"], provenance="measured"))
        elif 1 in b and 2 in b:
            m1, m2 = b[1]["median"], b[2]["median"]
            out.append(dict(base, value=m1 + (L - 1) * (m2 - m1), provenance="extrapolated_from_1_and_2_blocks",
                            value_1block=m1, value_2blocks=m2))
    return out


def main() -> None:
    rows = load_rows()
    _write(TABLES / "measured_long.csv", rows)
    summary = summarise(rows)
    summary += multiproof_adjust(summary)
    _write(TABLES / "measured_summary.csv", summary)
    full = llm_full(summary)
    if full:
        _write(TABLES / "llm_full_model.csv", full)
    print(f"{len(rows)} raw records -> {len(summary)} summary rows, {len(full)} full-model LLM rows")


if __name__ == "__main__":
    main()
