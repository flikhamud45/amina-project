"""The hardware-independent values of two raw benchmark roots agree: proof sizes do not depend on the hardware.

Backs the extended version's "Proof sizes do not depend on the hardware" (``app:eval-setup``): every value below is
an integer fact about the model, the graph or the protocol, so it must be identical on any GPU and CPU.  On the stored
roots, the second platform (an RTX 2080 Ti prover with a Xeon Silver 4114 verifier) reproduces all 1,978 such values
of the L40S run:

    python -m experiments.comparison.fingerprint_check artifacts/comparison/raw_l40s \\
        artifacts/comparison/raw_rtx2080ti-v2          # 1978 fingerprints compared, 0 problems
    python -m experiments.comparison.fingerprint_check <stored raw root> <new raw root>
        # a new platform against a stored one, before anything long runs on it

The values compared (:data:`EXACT`): the parameter count, the int8 model's bytes and accuracy, the path test's facts
(the bytes to open everything, the detection probabilities, the paths needed and their bytes), the security bits, the
bytes of the claims, of ``u`` and of the opened columns (the Merkle term depends on the random columns), and every
verdict; and, within ``--accuracy-tolerance``, the float model's accuracy (cuDNN may change its last digits).  A
mismatch means different weights, data, code or a GPU/CPU disagreement.  It also prints every honest rejection and
every accepted attack of the second root.  Exits 1 on any problem, and also when nothing could be compared (a wrong
or partial stored root).  Cells are compared by name, so a decoder cell of another attention probabilities' scale (the
final run's ``_pmax2047``, 11 bits) has no counterpart in the stored roots' 8-bit cells.  Reads only; a minute on a
laptop CPU for the stored roots.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXACT = {"n_params", "model_bytes_int8", "int8_accuracy", "open_all_bytes", "p_detect_penultimate",
         "p_detect_min_node", "paths_needed_penultimate", "paths_bytes_shared", "paths_bytes_shared_k",
         "soundness_bits", "bytes_claims", "bytes_u", "bytes_columns", "accepted", "tamper_rejected", "rejected"}
CLOSE = ("float_accuracy",)
"""Compared within ``--accuracy-tolerance``: the float model on cuDNN, whose last digits may change."""


def load(root: Path) -> dict:
    """{(suite, model, cell): {(metric, batch, attack, k, stage): {values}}} of the finished cells of ``root``."""
    out = defaultdict(lambda: defaultdict(set))
    for p in root.glob("*/*/*.jsonl"):
        if not p.with_suffix(".done").exists():
            continue
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["metric"] == "env":
                continue
            key = (r["suite"], r["model"], r["cell"])
            sub = (r["metric"], r.get("batch"), r.get("attack"), r.get("k"), r.get("stage"))
            out[key][sub].add(r["value"] if not isinstance(r["value"], float) else round(r["value"], 12))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stored", type=Path, help="the reference raw root (e.g. artifacts/comparison/raw_l40s)")
    ap.add_argument("new", type=Path, help="the raw root to check against it")
    ap.add_argument("--accuracy-tolerance", type=float, default=0.002,
                    help="the largest difference of the float model's accuracy (default 0.002)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    old, new = load(args.stored), load(args.new)
    bad = checked = 0
    for key, metrics in sorted(new.items()):
        for sub, vals in sorted(metrics.items(), key=str):
            m = sub[0]
            if m in ("accepted",) and vals != {1}:
                print("HONEST REJECTION", key, sub, vals)
                bad += 1
            if m in ("rejected", "tamper_rejected") and vals != {1}:
                print("ATTACK ACCEPTED", key, sub, vals)
                bad += 1
            ref = old.get(key, {}).get(sub)
            if ref is None or m not in EXACT | set(CLOSE):
                continue
            checked += 1
            if m in CLOSE:
                ok = abs(min(vals) - min(ref)) <= args.accuracy_tolerance
            elif m in ("accepted", "rejected", "tamper_rejected"):
                ok = vals == ref
            else:  # per-query facts: every new value must be one of the stored ones
                ok = vals <= ref
            if not ok:
                bad += 1
                print("MISMATCH", key, sub, "stored", sorted(ref)[:4], "new", sorted(vals)[:4])
    print(f"{checked} fingerprints compared, {bad} problems")
    if not checked:   # a wrong or partial stored root must not pass as agreement
        print(f"nothing compared: is {args.stored} the stored raw root, and does {args.new} hold finished cells?")
    sys.exit(1 if bad or not checked else 0)


if __name__ == "__main__":
    main()
