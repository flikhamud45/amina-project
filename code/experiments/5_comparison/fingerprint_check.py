"""Compare the hardware-independent numbers of a new raw tree with the stored one.

    python fingerprint_check.py <stored raw dir> <new raw dir>

Everything below is an integer fact about the model, the graph or the protocol, so it
must be identical on any GPU: a mismatch means different weights, data, code or a
GPU/CPU disagreement -- stop and report it before running anything long.
Also prints every honest rejection and every accepted attack in the new tree.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

EXACT = {"n_params", "model_bytes_int8", "int8_accuracy", "open_all_bytes", "p_detect_penultimate",
         "p_detect_min_node", "paths_needed_penultimate", "paths_bytes_shared", "paths_bytes_shared_k",
         "soundness_bits", "bytes_claims", "bytes_u", "bytes_columns", "accepted", "tamper_rejected", "rejected"}
CLOSE = {"float_accuracy": 0.002}          # float model on cuDNN: last-digit changes are expected


def load(root: Path) -> dict:
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


def main() -> None:
    old, new = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
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
                ok = abs(min(vals) - min(ref)) <= CLOSE[m]
            elif m in ("accepted", "rejected", "tamper_rejected"):
                ok = vals == ref
            else:  # per-query facts: every new value must be one of the stored ones
                ok = vals <= ref
            if not ok:
                bad += 1
                print("MISMATCH", key, sub, "stored", sorted(ref)[:4], "new", sorted(vals)[:4])
    print(f"{checked} fingerprints compared, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
