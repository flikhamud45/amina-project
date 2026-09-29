"""Soundness and completeness counts of the report (Sec. 4.3), straight from one platform's raw records.

    python experiments/5_comparison/count_outcomes.py --platform l40s          # the report's counts (Sec. 4.3) and record total (Sec. 4.1)
    python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2  # the second platform's, also quoted there

Honest queries: every ``accepted`` record of a defence cell (the batched CNN and LLM
queries included).  Attacks: every ``rejected`` record of the tamper cells and every
``tamper_rejected`` record (the one tampered query of each LLM defence cell), with the
check that rejected them (``stage``).  Every tag counts (``_gpuv``, ``_batch``, ``_thr1``,
``_thr12``, ``_tf32``, ``_nofix`` and ``_nolean`` too).  Only finished cells (with a
``.done`` marker) count, plus the rejections that stopped a job: bench.py keeps such a
cell's records as ``<cell>.rejected-<run_id>.jsonl``, and each rejected honest query in
them counts.  ``--platform`` defaults to ``$PVI_PLATFORM``, else ``l40s`` (the report's counts).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "comparison"
SUITES = ("cnn", "llm")


def count(root: Path) -> dict:
    c = {k: Counter() for k in ("honest", "attacks", "stages", "by_type")}
    records, rejected = 0, []
    for path in sorted(root.glob("*/*/*.jsonl")):
        if not path.with_suffix(".done").exists():
            continue
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["metric"] == "env":
                continue
            records += 1
            if r["metric"] == "accepted" and r["cell"].startswith("defence_"):
                c["honest"][(suite, r["value"])] += 1
            elif r["metric"] in ("rejected", "tamper_rejected"):
                c["attacks"][(suite, r["value"])] += 1
                c["by_type"][(suite, r.get("attack") or "one pre-activation +1", r["value"])] += 1
                if r["value"]:
                    c["stages"][r.get("stage")] += 1
    # an honest rejection stopped its job before the cell finished: its evidence is kept apart
    for path in sorted(root.glob("*/*/*.rejected-*.jsonl")):
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["metric"] == "accepted" and r["value"] == 0:
                c["honest"][(suite, 0)] += 1
                rejected.append(f"{path.relative_to(root)} trial {r.get('trial')}")
    return dict(c, records=records, rejected=rejected)


def honest(c: dict, value: int, suite: str | None = None) -> int:
    return sum(n for (s, v), n in c["honest"].items() if v == value and suite in (None, s))


def report(name: str, c: dict) -> None:
    a = c["attacks"]
    print(f"== {name}: {c['records']:,} records")
    print(f"honest queries accepted: {honest(c, 1)} / {honest(c, 0) + honest(c, 1)}")
    for suite in SUITES:
        print(f"{suite} honest queries accepted: {honest(c, 1, suite)} / {honest(c, 0, suite) + honest(c, 1, suite)}")
    for suite in SUITES:
        print(f"{suite} attacks rejected: {a[(suite, 1)]} / {a[(suite, 0)] + a[(suite, 1)]}")
    for (suite, kind, value), n in sorted(c["by_type"].items()):
        print(f"   {suite:3s} {kind:28s} {'rejected' if value else 'ACCEPTED'}: {n}")
    print("rejected by:", ", ".join(f"{k} {v}" for k, v in c["stages"].most_common()))
    for line in c["rejected"]:
        print("   HONEST QUERY REJECTED (job stopped):", line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM") or "l40s",
                    help="count raw_<platform>/ (default: $PVI_PLATFORM, else l40s, the report's counts)")
    root = BASE / f"raw_{ap.parse_args().platform}"
    if not (root / "PLATFORM.json").exists():
        raise SystemExit(f"{root} is not a platform root (no PLATFORM.json)")
    report(root.name, count(root))


if __name__ == "__main__":
    main()
