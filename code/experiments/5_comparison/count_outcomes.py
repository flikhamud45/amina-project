"""Soundness and completeness counts of the report (Sec. 4.3), straight from the raw records.

    python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
        # the basic protocol's counts (abstract, Sec. 4.3, conclusion) -> \\NHonest, \\NAttacks, ...
    python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt \\
        --tex ../report/tables/counts_opt.tex
        # the optimised protocol's runs, added up (abstract, Sec. 4.3) -> \\OptNHonest, \\OptNAttacks, ...
        # (add --definition once l40s_improved2 is stored: the text says "on the optimised one", and the
        # unfiltered roots also hold basic-format, plan-only and by-product cells)
    python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2   # print only

``--platform`` takes one platform or a comma-separated list, whose counts are added.  A listed
platform that has no raw root yet (a run still to come) is skipped with a warning, and ``--tex`` then
writes every count as ``\\pending{<count so far>}`` (red in main.tex, which defines \\pending): the
total is not final until that run is stored.
``--definition`` counts only the defence and tamper cells of the optimised protocol's definition
(``paper_assets.definition_tagsets()``: no basic-format cells, stand-ins or by-products of the
optimised roots); ``--require-tag TAG`` only the cells whose name carries TAG.

Honest queries: every ``accepted`` record of a defence cell (the batched CNN and LLM
queries included).  Attacks: every ``rejected`` record of the tamper cells and every
``tamper_rejected`` record (the one tampered query of each LLM defence cell), with the
check that rejected them (``stage``).  Every tag counts (``_gpuv``, ``_batch``, ``_thr1``,
``_thr12``, ``_tf32``, ``_nofix``, ``_nolean`` and the optimised runs' tags too).  Only finished
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
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "comparison"
SUITES = ("cnn", "llm")
PLUS_ONE = "one pre-activation +1"   # the attack of every LLM defence cell (no 'attack' field)
KINDS = ("honest", "attacks", "stages", "by_type", "policy", "partial")


BASE_PART = re.compile(r"^(int|fs|lam\d+|rate\d+|T\d+|L\d+)$")


def cell_tags(cell: str):
    """(mode, tag set) of a defence or tamper cell ('tamper_C_int_lam40_T64_L32_wire_prune_polauto' ->
    ('C', {'wire', 'prune', 'polauto'})), or (None, None) for any other cell."""
    parts = cell.split("_")
    if parts[0] not in ("defence", "tamper") or len(parts) < 2:
        return None, None
    return parts[1], frozenset(p for p in parts[2:] if not BASE_PART.match(p))


def _kept(suite: str, cell: str, tags, definition) -> bool:
    if not set(tags) <= set(cell.split("_")):
        return False
    if definition is None:
        return True
    mode, ts = cell_tags(cell)
    return mode is not None and ts in definition.get((suite, mode), set())


def count(root: Path, policy: str = "auto", tags=(), definition=None) -> dict:
    """``tags``: count only the cells whose name carries all of them.  ``definition``: {(suite, mode): {tag
    sets}} (paper_assets.definition_tagsets()): count only the defence and tamper cells of the optimised
    protocol's definition, as check_opt2.py's uniform cells."""
    c = {k: Counter() for k in KINDS}
    records, rejected = 0, []
    for path in sorted(root.glob("*/*/*.jsonl")):
        if not path.with_suffix(".done").exists() or not _kept(path.parts[-3], path.stem, tags, definition):
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
        if not _kept(path.parts[-3], path.name.split(".rejected-")[0], tags, definition):
            continue
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["metric"] == "accepted" and r["value"] == 0:
                c["honest"][(suite, 0)] += 1
                rejected.append(f"{root.name}/{path.relative_to(root)} trial {r.get('trial')}")
    return dict(c, records=records, rejected=rejected)


def merge(counts: list[dict]) -> dict:
    """Several platforms' counts, added up."""
    out = {k: Counter() for k in KINDS}
    out.update(records=0, rejected=[])
    for c in counts:
        for k in KINDS:
            out[k].update(c[k])
        out["records"] += c["records"]
        out["rejected"] += c["rejected"]
    return out


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
    that have no full build on these platforms)."""
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
    if attacks(c, 1, "cnn", "policy"):   # image-model attacks under the planning rule (the optimised runs)
        m["NAttacksPlans"] = attacks(c, 1, "cnn", "policy")
    return {prefix + k: v for k, v in m.items()}


def write_tex(path: Path, name: str, values: dict, pending: str = "") -> None:
    """``pending``: the runs still to come; every count is then wrapped in \\pending{} (a partial total)."""
    lines = [f"% generated by code/experiments/5_comparison/count_outcomes.py from {name}; do not edit"]
    if pending:
        lines.append(f"% partial: {pending} not yet run, so every count is \\pending{{}} (red in main.tex)")
    wrap = (lambda x: "\\pending{" + x + "}") if pending else (lambda x: x)
    lines += ["\\newcommand{\\" + k + "}{" + wrap(f"{v:,}".replace(",", "{,}")) + "}" for k, v in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(values)} macros{', pending' if pending else ''})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM") or "l40s",
                    help="count raw_<platform>/, or several (comma-separated, added up; one not yet run is skipped "
                         "and the macros are written as pending) (default: $PVI_PLATFORM, else l40s)")
    ap.add_argument("--policy", default="auto",
                    help="also count apart the attacks on cells built under this commitment policy (default auto)")
    ap.add_argument("--prefix", default="", help="macro name prefix for --tex (the optimised runs: Opt)")
    ap.add_argument("--definition", action="store_true",
                    help="count only the defence and tamper cells of the optimised protocol's definition "
                         "(paper_assets.OPTIMISED), without by-products and stand-ins")
    ap.add_argument("--require-tag", action="append", default=[], metavar="TAG",
                    help="count only the cells whose name carries this tag (repeatable; e.g. wire: the optimised "
                         "protocol's cells only, without l40s_improved's basic-format and plan-only cells)")
    ap.add_argument("--tex", type=Path, help="write the counts as LaTeX macros to this file")
    args = ap.parse_args()
    names = [p.strip() for p in args.platform.split(",") if p.strip()]
    roots, absent = [], []
    for name in names:
        root = BASE / f"raw_{name}"
        if (root / "PLATFORM.json").exists():
            roots.append(root)
        elif len(names) > 1 and not root.exists():
            print(f"warning: no {root.name}/ (not yet run): skipped; the counts are partial", file=sys.stderr)
            absent.append(root.name)
        else:
            raise SystemExit(f"{root} is not a platform root (no PLATFORM.json)")
    if not roots:
        raise SystemExit(f"none of {names} has a raw root yet")
    definition = None
    if args.definition:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import paper_assets   # noqa: E402  (the one definition of the optimised protocol's cells)
        definition = paper_assets.definition_tagsets()
    c = merge([count(root, args.policy, args.require_tag, definition) for root in roots])
    label = " + ".join(r.name for r in roots) + (f" (cells tagged {', '.join(args.require_tag)})"
                                                  if args.require_tag else "") +         (" (the optimised definition's cells)" if definition else "")
    report(label, c, args.policy)
    if args.tex:
        write_tex(args.tex, label, macros(c, args.prefix), pending=", ".join(absent))


if __name__ == "__main__":
    main()
