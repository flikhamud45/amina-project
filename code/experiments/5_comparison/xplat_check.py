"""Check that two platforms measured the same thing: every number that does not depend on
the hardware must agree between their raw roots.

    python experiments/5_comparison/xplat_check.py rtx2080ti h100

Exact: the integer models (parameters, int8 accuracy), the path-test baseline (visit
probabilities, paths and bytes for 2^-40), the security parameters (r, t, bits) and every
proof byte count except the Merkle multiproof, whose size depends on which columns are
drawn (checked to 10%).  LLM cells of raw/ made before multiproofs (gpt2, opt-125m, opt-1.3b,
opt-6.7b) sent one Merkle path per column: their bytes_paths and bytes_total are not
compared at all, only their claims, u and columns (exact), so "0 disagreements" says
nothing about their Merkle term.  Honest queries must all be accepted and attacks all
rejected on both.  Timings are not compared.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "comparison"
EXACT = {"n_params", "model_bytes_int8", "int8_accuracy", "p_detect_penultimate", "p_detect_min_node",
         "open_all_bytes", "paths_needed_penultimate", "paths_bytes_shared", "paths_bytes_shared_k",
         "bytes_claims", "bytes_u", "bytes_columns", "flip_success"}
REL = {"soundness_bits": 1e-9, "float_accuracy": 0.002, "bytes_paths": 0.10, "bytes_total": 0.02}
ALL_ONE = {"accepted", "rejected", "tamper_rejected"}


def root(platform: str) -> Path:
    return BASE / ("raw" if platform in ("", "rtx2080ti") else f"raw_{platform}")


def load(r: Path):
    vals, cfg = defaultdict(list), {}
    for p in sorted(r.glob("*/*/*.jsonl")):
        if not p.with_suffix(".done").exists():
            continue
        for line in p.open(encoding="utf-8"):
            x = json.loads(line)
            if x["metric"] == "env":
                continue
            c = x.get("config") or {}
            if x["metric"] in ("bytes_paths", "bytes_total") and c.get("merkle") != "multiproof":
                continue  # pre-multiproof runs: not comparable byte for byte
            vals[(x["suite"], x["model"], x["cell"], x["metric"]) + tuple(str(x.get(k) or "") for k in
                                                                          ("batch", "attack", "lam", "k"))].append(x["value"])
            cfg[(x["suite"], x["model"], x["cell"])] = tuple(c.get(k) for k in ("reps", "columns", "rate", "lam", "seq"))
    return vals, cfg


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    (va, ca), (vb, cb) = load(root(a)), load(root(b))
    bad, n = [], 0
    for cell in sorted(set(ca) & set(cb)):
        n += 1
        if ca[cell] != cb[cell]:
            bad.append(f"parameters of {cell}: {ca[cell]} vs {cb[cell]}")
    for key in sorted(set(va) & set(vb)):
        m = key[3]
        if m in ALL_ONE:
            ok = min(va[key]) == min(vb[key]) == 1
        elif m in EXACT:
            ok = statistics.median(va[key]) == statistics.median(vb[key])
        elif m in REL:
            x, y = statistics.median(va[key]), statistics.median(vb[key])
            ok = abs(x - y) <= REL[m] * max(abs(x), abs(y), 1e-300)
        else:
            continue
        n += 1
        if not ok:
            bad.append(f"{key}: {statistics.median(va[key])} vs {statistics.median(vb[key])}")
    common = set(ca) & set(cb)
    metrics = {k[:4] for k in va} & {k[:4] for k in vb}
    for key in sorted((set(va) ^ set(vb))):  # e.g. an attack that one platform skipped (a missing model file)
        if key[:3] not in common:
            continue
        # attack types and honest queries must match one to one; for the other exact metrics only a
        # metric missing altogether counts (older bench versions also stored extra lambdas)
        if key[3] in ALL_ONE or (key[3] in EXACT and key[:4] not in metrics):
            bad.append(f"{key} measured on only one platform")
    only = sorted(set(ca) ^ set(cb))
    print(f"{a} vs {b}: {n} checks on {len(set(ca) & set(cb))} common cells, {len(bad)} disagreements; "
          f"{len(only)} cells in only one root")
    for line in bad[:50]:
        print("  DIFFER", line)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
