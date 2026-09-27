"""Soundness and completeness counts of the report (Sec. 4.3), straight from the raw records.

    python experiments/5_comparison/count_outcomes.py                       # every platform root
    python experiments/5_comparison/count_outcomes.py --platform h100       # one root
    python experiments/5_comparison/count_outcomes.py --platform h100 --variant ''   # untagged cells only
    python experiments/5_comparison/count_outcomes.py --tex ../report/tables/counts.tex

Honest queries: every ``accepted`` record of a defence cell.  Attacks: every
``rejected`` record of the CNN tamper cells and every ``tamper_rejected`` record of the
LLM cells, with the check that rejected them (``stage``).  Only finished cells (with a
``.done`` marker) count, plus the rejections that stopped a job: bench.py keeps such a
cell's records as ``<cell>.rejected-<run_id>.jsonl``, and each rejected honest query in
them counts (so ``--tex`` refuses).  By default every platform root (``raw/``,
``raw_<platform>/``, each with a ``PLATFORM.json``) is counted and the totals are summed;
the smoke job's throw-away ``raw_smoke*/`` roots are never counted by default.  ``--tex``
writes the totals as LaTeX macros for the report and refuses if any honest query was
rejected or any attack accepted, since the report says "all".  ``--variant`` keeps only
the cells of one ``--tag`` ('' = the untagged cells; default: every variant).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "comparison"
SUITES = ("cnn", "llm")


def root_for(platform: str) -> Path:
    return BASE / ("raw" if platform in ("", "rtx2080ti") else f"raw_{platform}")


def default_roots() -> list[Path]:
    return sorted(p for p in BASE.glob("raw*") if p.is_dir() and (p / "PLATFORM.json").exists()
                  and not p.name.startswith("raw_smoke"))


def count(root: Path, variant: str | None = None) -> dict:
    c = {k: Counter() for k in ("honest", "attacks", "stages", "by_type")}
    records, run_ids, ids, rejected = 0, set(), {"honest": set(), "attacks": set()}, []

    def keep(r) -> bool:
        return variant is None or (r.get("config") or {}).get("variant", "") == variant

    for path in sorted(root.glob("*/*/*.jsonl")):
        if not path.with_suffix(".done").exists():
            continue
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if not keep(r):
                continue
            run_ids.add(r.get("run_id"))
            if r["metric"] == "env":
                continue
            records += 1
            # the queries and the attacks are seeded, so another platform repeats the same instances
            # (with fresh challenges): ``ids`` identifies an instance independently of the platform
            inst = (suite, r["model"], r["cell"], r.get("attack"), r.get("trial"), r.get("batch"))
            if r["metric"] == "accepted" and r["cell"].startswith("defence_"):
                c["honest"][(suite, r["value"])] += 1
                ids["honest"].add(inst)
            elif r["metric"] in ("rejected", "tamper_rejected"):
                c["attacks"][(suite, r["value"])] += 1
                ids["attacks"].add(inst)
                c["by_type"][(suite, r.get("attack") or "one pre-activation +1", r["value"])] += 1
                if r["value"]:
                    c["stages"][r.get("stage")] += 1
    # an honest rejection stopped its job before the cell finished: its evidence is kept apart
    for path in sorted(root.glob("*/*/*.rejected-*.jsonl")):
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if keep(r) and r["metric"] == "accepted" and r["value"] == 0:
                c["honest"][(suite, 0)] += 1
                rejected.append(f"{path.relative_to(root)} trial {r.get('trial')}")
    return dict(c, records=records, run_ids=run_ids, ids=ids, rejected=rejected)


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
    for line in c.get("rejected", []):
        print("   HONEST QUERY REJECTED (job stopped):", line)


def tex(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", nargs="*", default=None,
                    help="roots to count (default: every root except the smoke job's raw_smoke*)")
    ap.add_argument("--variant", default=None, help="only cells of this --tag ('' = untagged; default: all)")
    ap.add_argument("--tex", default="", help="write the totals as LaTeX macros to this file")
    args = ap.parse_args()
    if args.platform is not None and not args.platform:
        raise SystemExit("--platform needs at least one platform name")
    roots = [root_for(p) for p in args.platform] if args.platform is not None else default_roots()
    if not roots:
        raise SystemExit(f"no platform roots to count under {BASE}")
    for root in roots:
        if not (root / "PLATFORM.json").exists():
            raise SystemExit(f"{root} is not a platform root (no PLATFORM.json)")
    total = {k: Counter() for k in ("honest", "attacks", "stages", "by_type")}
    total["records"], total["rejected"], seen, distinct = 0, [], {}, {"honest": set(), "attacks": set()}
    for root in roots:
        c = count(root, args.variant)
        dup = c["run_ids"] & set(seen)
        if dup:  # e.g. a copy of raw/ left next to it: counting it twice would inflate the report
            raise SystemExit(f"{root.name} repeats runs of {seen[next(iter(dup))]}; remove the copy")
        seen.update({r: root.name for r in c["run_ids"]})
        report(root.name, c)
        for k in ("honest", "attacks", "stages", "by_type"):
            total[k].update(c[k])
        total["records"] += c["records"]
        total["rejected"] += c["rejected"]
        for k in distinct:
            distinct[k] |= c["ids"][k]
    if len(roots) > 1:
        report("total", dict(total, rejected=[]))
    print(f"distinct instances: {len(distinct['honest'])} honest queries, {len(distinct['attacks'])} attacks")
    if args.tex:
        a, s = total["attacks"], total["stages"]
        bad = honest(total, 0) + a[("cnn", 0)] + a[("llm", 0)]
        if bad:
            raise SystemExit(f"{bad} honest rejections or accepted attacks: the report cannot say 'all'")
        # machines, not roots: two roots of one GPU + CPU (e.g. a re-run) are one platform
        machines = {tuple(json.loads((r / "PLATFORM.json").read_text(encoding="utf-8")).get(k) for k in ("gpu", "cpu"))
                    for r in roots}
        macros = {"NRecords": total["records"], "NHonest": honest(total, 1), "NAttacksCNN": a[("cnn", 1)],
                  "NAttacksLLM": a[("llm", 1)], "NAttacks": a[("cnn", 1)] + a[("llm", 1)],
                  "NFreivalds": s["freivalds"], "NColumnsCode": s["columns_code"],
                  "NColumnsMerkle": s["columns_merkle"], "NHonestDistinct": len(distinct["honest"]),
                  "NAttacksDistinct": len(distinct["attacks"]), "NPlatforms": len(machines)}
        Path(args.tex).write_text("".join(f"\\newcommand{{\\{k}}}{{{tex(v)}}}\n" for k, v in macros.items()),
                                  encoding="utf-8", newline="\n")
        print("wrote", args.tex, f"({', '.join(r.name for r in roots)})")


if __name__ == "__main__":
    main()
