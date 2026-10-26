"""The completeness and soundness counts of the benchmark, straight from its raw records.

Produces the counts of Sec. ``sec:eval-setup`` ("Attacks and soundness tests": the base verifier accepted all 3,792
honest queries and rejected all 2,512 attacks, Freivalds' check rejecting 1,601, the column check 853, the Merkle check
56 and the range check 2), from the defence and tamper cells that ``costs/bench.py`` writes to
``artifacts/comparison/raw_<platform>/``.  ``paper/numbers.py`` calls :func:`count_runs` and :func:`macros` on the
final run's root (``\\FinalNHonest``, ...).

    python -m experiments.comparison.count_outcomes --platform l40s_improved,l40s_improved2,l40s_improved3 \\
        --definition --policy auto --prob-max 255
        # the paper's counts: the stored runs of the base implementation (8-bit decoders), added up, their defining
        # cells only
    python -m experiments.comparison.count_outcomes --platform l40s
        # the first L40S run (the per-op commitment, uncompressed proofs; the attack's path-test cells)
    python -m experiments.comparison.count_outcomes --platform rtx2080ti-v2
        # the second platform's verdicts

``--platform`` takes one platform or a comma-separated list, whose counts are added (a cell finished in a later
platform's root replaces the same cell of an earlier one, as in the tables).  A listed platform that has no raw root
yet (a run still to come) is skipped with a warning: the counts are then partial.  ``--definition`` counts only the
defining defence and tamper cells under the plan ``--policy``, the language models' at the attention probabilities'
scale ``--prob-max`` (default 2047, the final run's; 255 for the stored runs' cells), as
``experiments.paper.bench.definition_tagsets`` defines them (no cells of the uncompressed proofs, stand-ins or
by-products of those roots); ``--require-tag TAG`` only the cells whose name carries TAG.

Honest queries: every ``accepted`` record of a defence cell (the batched image and language-model queries included).
Attacks: every ``rejected`` record of the tamper cells and every ``tamper_rejected`` record (the one tampered query of
each language-model defence cell), with the check that rejected them (``stage``).  Every tag counts (``_gpuv``,
``_batch``, ``_thr1``, ...).  Only finished cells (with a ``.done`` marker) count, plus the rejections that stopped a
job: ``bench.py`` keeps such a cell's records as ``<cell>.rejected-<run_id>.jsonl``, and each rejected honest query
in them counts.  The attacks on the cells built under the commitment plan ``--policy`` (the records'
``config.policy_requested``) are also counted apart, and the "+1 on one pre-activation" attacks on language models on
1- and 2-block builds of the larger shapes apart from the full builds.

Prints the counts (records, honest queries accepted per suite, attacks rejected per suite and type, the checks that
rejected them, the "+1" attacks by build); reads only, in seconds to a minute on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from experiments.common.paths import BENCH
from experiments.paper.bench import DEFAULT_POLICY as PAPER_POLICY
from experiments.paper.bench import DEFAULT_PROB_MAX as PAPER_PROB_MAX

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


def finished_cells(root: Path) -> set:
    """(suite, model, cell) of every finished cell (a .done marker) of one raw root."""
    return {(p.parts[-3], p.parts[-2], p.stem) for p in root.glob("*/*/*.done")}


def count(root: Path, policy: str = PAPER_POLICY, tags=(), definition=None, skip=frozenset()) -> dict:
    """``tags``: count only the cells whose name carries all of them.  ``definition``: {(suite, mode): {tag
    sets}} (``experiments.paper.bench.definition_tagsets``): count only the defining defence and tamper cells.  ``skip``: (suite, model, cell) of finished cells not to count (a later run holds
    them: :func:`count_runs`)."""
    c = {k: Counter() for k in KINDS}
    records, rejected = 0, []
    for path in sorted(root.glob("*/*/*.jsonl")):
        if not path.with_suffix(".done").exists() or not _kept(path.parts[-3], path.stem, tags, definition):
            continue
        if (path.parts[-3], path.parts[-2], path.stem) in skip:
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


def count_runs(roots: list[Path], policy: str = PAPER_POLICY, tags=(), definition=None) -> dict:
    """Several runs' counts, added up, where a cell finished in a later root replaces the same cell of an earlier
    one (a re-measured cell would otherwise count twice).
    Honest rejections (.rejected-* files) always count."""
    out, later = [], set()
    for root in reversed(roots):
        out.append(count(root, policy, tags, definition, skip=frozenset(later)))
        later |= finished_cells(root)
    return merge(out[::-1])


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
            print(f"{suite} attacks on cells built under --policy {policy}: {attacks(c, 1, suite, 'policy')} "
                  f"rejected / {n}")
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
    """The counts the text quotes, by macro name (without the backslash; ``paper/numbers.py`` writes them).  Refuses
    if an honest query was rejected or an attack accepted, since the text says "all"."""
    if honest(c, 0) or attacks(c, 0):
        raise SystemExit(f"{honest(c, 0)} honest queries rejected and {attacks(c, 0)} attacks accepted: "
                         "the text's 'all accepted / all rejected' would be wrong")
    m = {"NRecords": c["records"], "NHonest": honest(c, 1), "NHonestCNN": honest(c, 1, "cnn"),
         "NHonestLLM": honest(c, 1, "llm"), "NAttacks": attacks(c, 1), "NAttacksCNN": attacks(c, 1, "cnn"),
         "NAttacksLLM": attacks(c, 1, "llm"), "NFreivalds": c["stages"]["freivalds"],
         "NColumnsCode": c["stages"]["columns_code"], "NColumnsMerkle": c["stages"]["columns_merkle"],
         "NRange": c["stages"]["range_or_shape"],     # rejected by the range (and shape) check
         "NAttacksPlusOne": sum(plus_one(c)[:2]),     # the '+1 on one pre-activation' LLM attacks ...
         "NAttacksPlusOnePartial": plus_one(c)[2]}     # ... of which on 1-2 block builds of the 30-70B shapes
    if attacks(c, 1, "cnn", "policy"):   # image-model attacks on the cells built under --policy
        m["NAttacksPlans"] = attacks(c, 1, "cnn", "policy")
    return {prefix + k: v for k, v in m.items()}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--platform", required=True,
                    help="count raw_<platform>/, or several (comma-separated, added up; one not yet run is skipped "
                         "and the counts are partial)")
    ap.add_argument("--base", type=Path, default=BENCH,
                    help="the folder holding the raw roots (default artifacts/comparison)")
    ap.add_argument("--policy", default=PAPER_POLICY,
                    help=f"also count apart the attacks on cells built under this commitment plan (default "
                         f"{PAPER_POLICY}, the paper's: experiments.paper.bench.DEFAULT_POLICY; the stored runs of the "
                         "base implementation: auto, the planning rule)")
    ap.add_argument("--definition", action="store_true",
                    help="count only the defining defence and tamper cells under --policy and --prob-max "
                         "(experiments.paper.bench.definition_tagsets), without by-products and stand-ins")
    ap.add_argument("--prob-max", type=int, default=PAPER_PROB_MAX,
                    help=f"with --definition: the language models' attention probabilities' scale (default "
                         f"{PAPER_PROB_MAX}, the final run's: cells tagged pmax{PAPER_PROB_MAX}; the stored runs: 255, "
                         "cells without the tag)")
    ap.add_argument("--require-tag", action="append", default=[], metavar="TAG",
                    help="count only the cells whose name carries this tag (repeatable; e.g. wire: the cells in the "
                         "compact encoding only)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    names = [p.strip() for p in args.platform.split(",") if p.strip()]
    roots = []
    for name in names:
        root = args.base / f"raw_{name}"
        if (root / "PLATFORM.json").exists():
            roots.append(root)
        elif len(names) > 1 and not root.exists():
            print(f"warning: no {root.name}/ (not yet run): skipped; the counts are partial", file=sys.stderr)
        else:
            raise SystemExit(f"{root} is not a platform root (no PLATFORM.json)")
    if not roots:
        raise SystemExit(f"none of {names} has a raw root yet")
    definition = None
    if args.definition:
        from experiments.paper import bench    # the one definition of the cost tables' cells
        definition = bench.definition_tagsets(args.policy, args.prob_max)
    c = count_runs(roots, args.policy, args.require_tag, definition)
    label = (" + ".join(r.name for r in roots)
             + (f" (cells tagged {', '.join(args.require_tag)})" if args.require_tag else "")
             + (" (the defining cells)" if definition else ""))
    report(label, c, args.policy)


if __name__ == "__main__":
    main()
