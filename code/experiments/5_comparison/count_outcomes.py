"""Soundness and completeness counts of the report (Sec. 4.3), straight from the raw records.

    python experiments/5_comparison/count_outcomes.py

Honest queries: every ``accepted`` record of a defence cell.  Attacks: every
``rejected`` record of the CNN tamper cells and every ``tamper_rejected`` record of the
LLM cells, with the check that rejected them (``stage``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "artifacts" / "comparison" / "raw"


def main() -> None:
    honest, attacks, stages = Counter(), Counter(), Counter()
    by_type = Counter()
    for path in sorted(RAW.glob("*/*/*.jsonl")):
        suite = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["metric"] == "accepted" and r["cell"].startswith("defence_"):
                honest[r["value"]] += 1
            elif r["metric"] in ("rejected", "tamper_rejected"):
                attacks[(suite, r["value"])] += 1
                by_type[(suite, r.get("attack") or "one pre-activation +1", r["value"])] += 1
                if r["value"]:
                    stages[r.get("stage")] += 1

    print(f"honest queries accepted: {honest[1]} / {honest[0] + honest[1]}")
    for suite in ("cnn", "llm"):
        ok, n = attacks[(suite, 1)], attacks[(suite, 0)] + attacks[(suite, 1)]
        print(f"{suite} attacks rejected: {ok} / {n}")
    for (suite, kind, value), n in sorted(by_type.items()):
        print(f"   {suite:3s} {kind:28s} {'rejected' if value else 'ACCEPTED'}: {n}")
    print("rejected by:", ", ".join(f"{k} {v}" for k, v in stages.most_common()))


if __name__ == "__main__":
    main()
