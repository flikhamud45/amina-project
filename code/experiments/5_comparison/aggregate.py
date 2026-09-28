"""Rebuild one platform's tables (``artifacts/comparison/tables_<platform>/``) from its raw records.

Deterministic and idempotent: it only reads ``raw_<platform>/**/*.jsonl`` (finished cells)
and writes

* ``measured_summary.csv`` -- per (suite, model, cell, metric, group) statistics;
* ``llm_full_model.csv``   -- full-model LLM costs: measured when all layers were
  built, otherwise extrapolated linearly from 1- and 2-block builds
  (``full = m1 + (L - 1) * (m2 - m1)``; every block has identical shapes).

    python experiments/5_comparison/aggregate.py --platform rtx2080ti-v2  # the report's numbers
    python experiments/5_comparison/aggregate.py --platform h100  # raw_h100/ -> tables_h100/
    python experiments/5_comparison/aggregate.py                  # raw/ -> tables/ (the earlier run)

Each hardware platform has its own raw root and tables directory, so the records of
different machines never enter the same median or the same extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "comparison"
RAW = BASE / "raw"          # set by main() from --platform
TABLES = BASE / "tables"


def dirs_for(platform: str) -> tuple[Path, Path]:
    """(raw root, tables directory) of one platform; '' (alias rtx2080ti) is the earlier run's."""
    p = "" if platform in ("", "rtx2080ti") else platform
    return BASE / (f"raw_{p}" if p else "raw"), BASE / (f"tables_{p}" if p else "tables")
GROUP_KEYS = ("batch", "attack", "lam", "op", "stage", "k")
ADDITIVE = {"prove_forward", "prove_fold", "prove_open", "verify_derive", "verify_products", "verify_columns",
            "verify_fold", "verify_upload", "fs_hash", "bytes_total", "bytes_total_multiproof"}


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(RAW.rglob("*.jsonl")):
        if not path.with_suffix(".done").exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["metric"] == "env":
                    continue
                cfg = row.pop("config", {}) or {}
                for k, v in cfg.items():
                    row["cfg_" + k] = v
                rows.append(row)
    return rows


def check_one_machine(rows: list[dict]) -> None:
    """A root holds one machine: one prover, one client CPU (its thread count may differ, e.g.
    the _thr1 cells), and a GPU client only on the prover's GPU model -- all matching the
    root's ``PLATFORM.json``.  Checked over the whole root: a per-cell check can never fire,
    since every cell is written by one job on one machine."""
    provers, cpus, clients = set(), set(), set()
    for r in rows:
        provers.add(r.get("prover_hw"))
        client, sep, cpu = str(r.get("verifier_hw")).rpartition(" (GPU client) + ")
        cpus.add(re.sub(r" x\d+ threads$", "", cpu))
        if sep:
            clients.add(client)
    lock = RAW / "PLATFORM.json"
    if lock.exists():
        have = json.loads(lock.read_text(encoding="utf-8"))
        provers.add(have.get("gpu") or have.get("cpu"))
        cpus.add(have.get("cpu"))
    if len(provers) > 1 or len(cpus) > 1 or not clients <= provers:
        raise SystemExit(f"{RAW.name} holds records of several machines (or not of its PLATFORM.json): provers "
                         f"{sorted(map(str, provers))}, client CPUs {sorted(map(str, cpus))}, GPU clients "
                         f"{sorted(map(str, clients))}")


def _write(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    hw: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if not isinstance(r["value"], (int, float)):
            continue
        key = (r["suite"], r["model"], r["cell"], r["metric"]) + tuple(str(r.get(k, "")) for k in GROUP_KEYS)
        groups[key].append(float(r["value"]))
        hw[key].add((r.get("prover_hw"), r.get("verifier_hw")))
        meta[key] = {k: v for k, v in r.items()
                     if k.startswith("cfg_") or k in ("prover_hw", "verifier_hw", "host", "git_sha", "unit",
                                                      "bits")}
    mixed = sorted(k for k, v in hw.items() if len(v) > 1)
    if mixed:  # a median over two machines is not a measurement of either
        raise SystemExit(f"records of different hardware in one group, e.g. {mixed[0][:4]}: "
                         f"{sorted(hw[mixed[0]], key=str)}")
    out = []
    for key, vals in sorted(groups.items()):
        out.append({"suite": key[0], "model": key[1], "cell": key[2], "metric": key[3],
                    **{k: key[4 + i] for i, k in enumerate(GROUP_KEYS)},
                    "n": len(vals), "median": statistics.median(vals), "mean": statistics.fmean(vals),
                    **{k: v for k, v in meta[key].items() if k != "_hw"}})
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
                and s.get("cfg_merkle", "") != "multiproof" and s["metric"] == "bytes_total"):
            continue
        cfg = CONFIGS[s["model"]]
        shapes = decoder_shapes(cfg, n_layers=int(s["cfg_n_layers"]))
        t, rate = int(s["cfg_columns"]), int(s["cfg_rate"])
        # rounded: the last digits of this expectation differ between platforms' maths libraries
        mp = round(sum(32 * expected_multiproof_nodes(sh.n_points(rate), t) for sh in shapes), 3)
        paths = next(x for x in summary if x["cell"] == s["cell"] and x["model"] == s["model"]
                     and x["metric"] == "bytes_paths" and x["suite"] == "llm")
        v = round(s["median"] - paths["median"] + mp, 3)
        extra.append(dict(s, metric="bytes_total_multiproof", median=v, mean=v,
                          provenance="measured data + expected multiproof"))
    return extra


def llm_full(summary: list[dict]) -> list[dict]:
    """Full-model LLM costs per (model, seq, mode, challenges, lam, metric, hardware).  A full
    build gives a ``measured`` row; with 1- and 2-block builds on the same machine it also
    carries what the linear extrapolation would have said (``llm_extrapolation_check.csv``).
    Without a full build, 1- and 2-block builds give an ``extrapolated`` row."""
    by = defaultdict(dict)
    for s in summary:
        if s["suite"] != "llm" or not s["cell"].startswith("defence_") or s["metric"] not in ADDITIVE:
            continue
        key = (s["model"], s["cfg_seq"], s["cfg_mode"], s["cfg_challenges"], s["cfg_lam"], s["metric"],
               s.get("cfg_threads", ""), s.get("cfg_rate", ""), s.get("cfg_variant", ""), s.get("cfg_device", ""),
               s.get("prover_hw", ""), s.get("verifier_hw", ""),   # never merge two machines' builds
               s.get("batch", ""))                                 # nor batched with single-prompt runs
        by[key][int(s["cfg_n_layers"])] = s
    out = []
    for (model, seq, mode, chal, lam, metric, threads, rate, variant, device, _hw, _vhw, batch), builds in sorted(
            by.items(), key=lambda kv: str(kv[0])):
        leans = {str(s.get("cfg_lean", "")) for s in builds.values()}
        if len(leans) > 1:  # e.g. sweep.sh and strong_gpu.sh into one root: the builds ran different code
            raise SystemExit(f"{model} T{seq} {mode}:{chal} lam{lam} {metric} variant '{variant}': builds "
                             f"{sorted(builds)} mix lean and non-lean runs ({sorted(leans)}); tag one of them")
        any_ = next(iter(builds.values()))
        L = int(any_["cfg_n_layers_full"])
        base = {"model": model, "seq": seq, "mode": mode, "challenges": chal, "lam": lam, "metric": metric,
                "threads": threads, "rate": rate, "variant": variant, "device": device,
                "prover_hw": None, "verifier_hw": None,
                "n_layers_full": L, "params_full": any_.get("cfg_params_full")}
        if batch != "":
            base["batch"] = batch

        def hw(s):  # the hardware of the build(s) a row is computed from, not of an arbitrary build
            return {"prover_hw": s.get("prover_hw"), "verifier_hw": s.get("verifier_hw")}

        if L in builds:
            row = dict(base, **hw(builds[L]), value=builds[L]["median"], provenance="measured")
            if L not in (1, 2) and 1 in builds and 2 in builds:   # what extrapolation would have said, for the check
                m1, m2 = builds[1]["median"], builds[2]["median"]
                row.update(value_1block=m1, value_2blocks=m2, extrapolated=m1 + (L - 1) * (m2 - m1))
            out.append(row)
        elif 1 in builds and 2 in builds:
            if hw(builds[1]) != hw(builds[2]):  # never extrapolate across different machines
                raise SystemExit(f"1- and 2-block builds of {model} T{seq} {mode}:{chal} lam{lam} ran on different "
                                 f"hardware: {hw(builds[1])} vs {hw(builds[2])}")
            m1, m2 = builds[1]["median"], builds[2]["median"]
            out.append(dict(base, **hw(builds[1]), value=m1 + (L - 1) * (m2 - m1),
                            provenance="extrapolated_from_1_and_2_blocks", value_1block=m1, value_2blocks=m2))
    return out


def main() -> None:
    global RAW, TABLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM", ""),
                    help="read raw_<platform>/, write tables_<platform>/ ('' or rtx2080ti: raw/ -> tables/)")
    RAW, TABLES = dirs_for(ap.parse_args().platform)
    if not RAW.is_dir():
        raise SystemExit(f"no raw records at {RAW}")
    rows = load_rows()
    check_one_machine(rows)
    summary = summarise(rows)
    summary += multiproof_adjust(summary)
    full = llm_full(summary)   # before any write: a refused lean/non-lean mix leaves the tables as they were
    _write(TABLES / "measured_summary.csv", summary)
    if full:
        _write(TABLES / "llm_full_model.csv", full)
        check = [dict(r, rel_error=(r["extrapolated"] - r["value"]) / r["value"] if r["value"] else None,
                      ratio=r["extrapolated"] / r["value"] if r["value"] else None)
                 for r in full if r["provenance"] == "measured" and "extrapolated" in r]
        if check:
            _write(TABLES / "llm_extrapolation_check.csv", check)
    else:
        check = []
    hws = sorted({str(s.get("prover_hw")) for s in summary})
    print(f"{RAW.name}: {len(rows)} raw records -> {len(summary)} summary rows, {len(full)} full-model LLM rows, "
          f"{len(check)} measured-vs-extrapolated pairs; prover {', '.join(hws)}")


if __name__ == "__main__":
    main()
