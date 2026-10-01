"""Soundness and completeness counts of the report (Sec. 5.2 and 5.4), straight from one platform's raw records.

    python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
        # the basic protocol's counts (abstract, Sec. 5.2, conclusion) -> \\NHonest, \\NAttacks, ...
    python experiments/5_comparison/count_outcomes.py --platform l40s_improved --prefix Opt \\
        --tex ../report/tables/counts_opt.tex
        # the optimised run's counts (Sec. 5.4) -> \\OptNHonest, \\OptNAttacks, \\OptNAttacksPlans, ...
    python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2   # print only

Honest queries: every ``accepted`` record of a defence cell (the batched CNN and LLM
queries included).  Attacks: every ``rejected`` record of the tamper cells and every
``tamper_rejected`` record (the one tampered query of each LLM defence cell), with the
check that rejected them (``stage``).  Every tag counts (``_gpuv``, ``_batch``, ``_thr1``,
``_thr12``, ``_tf32``, ``_nofix``, ``_nolean`` and the optimised run's tags too).  Only finished
cells (with a ``.done`` marker) count, plus the rejections that stopped a job: bench.py keeps such a
cell's records as ``<cell>.rejected-<run_id>.jsonl``, and each rejected honest query in
them counts.  The records of the cells built under the planning rule (``--policy auto``,
``config.policy_requested``) are also counted apart (``--policy``), and the "+1 on one
pre-activation" LLM attacks on 1- and 2-block builds of the larger shapes apart from the
full builds.  ``--platform`` defaults to ``$PVI_PLATFORM``, else ``l40s`` (the report's basic run).

``--tex`` writes the counts as LaTeX macros (``\\<prefix>NHonest`` and so on, numbers with
``{,}``), which ``main.tex`` inputs; it refuses if an honest query was rejected or an attack
accepted, since the text says "all".
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "comparison"
SUITES = ("cnn", "llm")
PLUS_ONE = "one pre-activation +1"   # the attack of every LLM defence cell (no 'attack' field)


def count(root: Path, policy: str = "auto") -> dict:
    c = {k: Counter() for k in ("honest", "attacks", "stages", "by_type", "policy", "partial")}
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
                kind = r.get("attack") or PLUS_ONE
                cfg = r.get("config") or {}
                c["attacks"][(suite, r["value"])] += 1
                c["by_type"][(suite, kind, r["value"])] += 1
                if cfg.get("policy_requested") == policy:
                    c["policy"][(suite, r["value"])] += 1
                if kind == PLUS_ONE:
                    part = (cfg.get("n_layers") or 0) < (cfg.get("n_layers_full") or 0)
                    c["partial"][(r["model"], part, r["value"])] += 1
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


def attacks(c: dict, value: int, suite: str | None = None, key: str = "attacks") -> int:
    return sum(n for (s, v), n in c[key].items() if v == value and suite in (None, s))


def report(name: str, c: dict, policy: str) -> None:
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
    for suite in SUITES:
        n = attacks(c, 1, suite, "policy") + attacks(c, 0, suite, "policy")
        if n:
            print(f"{suite} attacks on cells built under --policy {policy}: {attacks(c, 1, suite, 'policy')} rejected / {n}")
    full, blocks, large = plus_one(c)
    if full + blocks:
        print(f"llm '{PLUS_ONE}' rejected: {full} on full builds, {blocks} on 1-2 block builds, {large} of them "
              "of shapes built only so (the 30-70B shapes)")
    for line in c["rejected"]:
        print("   HONEST QUERY REJECTED (job stopped):", line)


def plus_one(c: dict):
    """Rejected '+1' LLM attacks: (on full builds, on 1-2 block builds, on 1-2 block builds of the shapes
    that have no full build on this platform)."""
    built = {m for (m, part, v) in c["partial"] if not part}
    full = sum(n for (m, part, v), n in c["partial"].items() if v and not part)
    blocks = sum(n for (m, part, v), n in c["partial"].items() if v and part)
    large = sum(n for (m, part, v), n in c["partial"].items() if v and part and m not in built)
    return full, blocks, large


def macros(c: dict, prefix: str) -> dict:
    """The counts the text quotes, by macro name (without the backslash)."""
    if honest(c, 0) or attacks(c, 0):
        raise SystemExit(f"{honest(c, 0)} honest queries rejected and {attacks(c, 0)} attacks accepted: "
                         "the text's 'all accepted / all rejected' would be wrong")
    m = {"NRecords": c["records"], "NHonest": honest(c, 1), "NHonestCNN": honest(c, 1, "cnn"),
         "NHonestLLM": honest(c, 1, "llm"), "NAttacks": attacks(c, 1), "NAttacksCNN": attacks(c, 1, "cnn"),
         "NAttacksLLM": attacks(c, 1, "llm"), "NFreivalds": c["stages"]["freivalds"],
         "NColumnsCode": c["stages"]["columns_code"], "NColumnsMerkle": c["stages"]["columns_merkle"],
         "NAttacksPlusOne": sum(plus_one(c)[:2]),     # the '+1 on one pre-activation' LLM attacks ...
         "NAttacksPlusOnePartial": plus_one(c)[2]}     # ... of which on 1-2 block builds of the 30-70B shapes
    if attacks(c, 1, "cnn", "policy"):   # image-model attacks under the planning rule (the optimised run)
        m["NAttacksPlans"] = attacks(c, 1, "cnn", "policy")
    return {prefix + k: v for k, v in m.items()}


def write_tex(path: Path, name: str, values: dict) -> None:
    lines = [f"% generated by code/experiments/5_comparison/count_outcomes.py from {name}; do not edit"]
    lines += ["\\newcommand{\\" + k + "}{" + f"{v:,}".replace(",", "{,}") + "}" for k, v in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(values)} macros)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM") or "l40s",
                    help="count raw_<platform>/ (default: $PVI_PLATFORM, else l40s, the report's basic counts)")
    ap.add_argument("--policy", default="auto",
                    help="also count apart the attacks on cells built under this commitment policy (default auto)")
    ap.add_argument("--prefix", default="", help="macro name prefix for --tex (the optimised run: Opt)")
    ap.add_argument("--tex", type=Path, help="write the counts as LaTeX macros to this file")
    args = ap.parse_args()
    root = BASE / f"raw_{args.platform}"
    if not (root / "PLATFORM.json").exists():
        raise SystemExit(f"{root} is not a platform root (no PLATFORM.json)")
    c = count(root, args.policy)
    report(root.name, c, args.policy)
    if args.tex:
        write_tex(args.tex, root.name, macros(c, args.prefix))


if __name__ == "__main__":
    main()
