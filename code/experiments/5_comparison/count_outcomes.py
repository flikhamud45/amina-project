"""Soundness and completeness counts of the report (Sec. 4.3), straight from the raw records.

    python experiments/5_comparison/count_outcomes.py                       # every platform root
    python experiments/5_comparison/count_outcomes.py --platform h100       # one root
    python experiments/5_comparison/count_outcomes.py --tex ../report/tables/counts.tex

Honest queries: every ``accepted`` record of a defence cell.  Attacks: every
``rejected`` record of the CNN tamper cells and every ``tamper_rejected`` record of the
LLM cells, with the check that rejected them (``stage``).  Only finished cells (with a
``.done`` marker) count.  By default every platform root (``raw/``, ``raw_<platform>/``,
each with a ``PLATFORM.json``) is counted and the totals are summed; ``--tex`` writes
them as LaTeX macros for the report and refuses if any honest query was rejected or
any attack accepted, since the report says "all".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "comparison"


def root_for(platform: str) -> Path:
    return BASE / ("raw" if platform in ("", "rtx2080ti") else f"raw_{platform}")


def count(root: Path) -> dict:
    c = {k: Counter() for k in ("honest", "attacks", "stages", "by_type")}
    records, run_ids, ids = 0, set(), {"honest": set(), "attacks": set()}
    for path in sorted(root.glob("*/*/*.jsonl")):
        if not path.with_suffix(".done").exists():
            continue
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            run_ids.add(r.get("run_id"))
            if r["metric"] == "env":
                continue
            records += 1
            # the queries and the attacks are seeded, so another platform repeats the same instances
            # (with fresh challenges): ``ids`` identifies an instance independently of the platform
            inst = (suite, r["model"], r["cell"], r.get("attack"), r.get("trial"), r.get("batch"))
            if r["metric"] == "accepted" and r["cell"].startswith("defence_"):
                c["honest"][r["value"]] += 1
                ids["honest"].add(inst)
            elif r["metric"] in ("rejected", "tamper_rejected"):
                c["attacks"][(suite, r["value"])] += 1
                ids["attacks"].add(inst)
                c["by_type"][(suite, r.get("attack") or "one pre-activation +1", r["value"])] += 1
                if r["value"]:
                    c["stages"][r.get("stage")] += 1
    return dict(c, records=records, run_ids=run_ids, ids=ids)


def report(name: str, c: dict) -> None:
    h, a = c["honest"], c["attacks"]
    print(f"== {name}: {c['records']:,} records")
    print(f"honest queries accepted: {h[1]} / {h[0] + h[1]}")
    for suite in ("cnn", "llm"):
        print(f"{suite} attacks rejected: {a[(suite, 1)]} / {a[(suite, 0)] + a[(suite, 1)]}")
    for (suite, kind, value), n in sorted(c["by_type"].items()):
        print(f"   {suite:3s} {kind:28s} {'rejected' if value else 'ACCEPTED'}: {n}")
    print("rejected by:", ", ".join(f"{k} {v}" for k, v in c["stages"].most_common()))


def tex(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", nargs="*", default=None, help="roots to count (default: every root)")
    ap.add_argument("--tex", default="", help="write the totals as LaTeX macros to this file")
    args = ap.parse_args()
    roots = ([root_for(p) for p in args.platform] if args.platform is not None else
             sorted(p for p in BASE.glob("raw*") if p.is_dir() and (p / "PLATFORM.json").exists()))
    total = {k: Counter() for k in ("honest", "attacks", "stages", "by_type")}
    total["records"], seen, distinct = 0, {}, {"honest": set(), "attacks": set()}
    for root in roots:
        c = count(root)
        dup = c["run_ids"] & set(seen)
        if dup:  # e.g. a copy of raw/ left next to it: counting it twice would inflate the report
            raise SystemExit(f"{root.name} repeats runs of {seen[next(iter(dup))]}; remove the copy")
        seen.update({r: root.name for r in c["run_ids"]})
        report(root.name, c)
        for k in ("honest", "attacks", "stages", "by_type"):
            total[k].update(c[k])
        total["records"] += c["records"]
        for k in distinct:
            distinct[k] |= c["ids"][k]
    if len(roots) > 1:
        report("total", total)
    print(f"distinct instances: {len(distinct['honest'])} honest queries, {len(distinct['attacks'])} attacks")
    if args.tex:
        h, a, s = total["honest"], total["attacks"], total["stages"]
        bad = h[0] + a[("cnn", 0)] + a[("llm", 0)]
        if bad:
            raise SystemExit(f"{bad} honest rejections or accepted attacks: the report cannot say 'all'")
        macros = {"NRecords": total["records"], "NHonest": h[1], "NAttacksCNN": a[("cnn", 1)],
                  "NAttacksLLM": a[("llm", 1)], "NAttacks": a[("cnn", 1)] + a[("llm", 1)],
                  "NFreivalds": s["freivalds"], "NColumnsCode": s["columns_code"],
                  "NColumnsMerkle": s["columns_merkle"], "NHonestDistinct": len(distinct["honest"]),
                  "NAttacksDistinct": len(distinct["attacks"]), "NPlatforms": len(roots)}
        Path(args.tex).write_text("".join(f"\\newcommand{{\\{k}}}{{{tex(v)}}}\n" for k, v in macros.items()),
                                  encoding="utf-8", newline="\n")
        print("wrote", args.tex, f"({', '.join(r.name for r in roots)})")


if __name__ == "__main__":
    main()
