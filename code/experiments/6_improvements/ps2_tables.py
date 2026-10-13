"""Plan 2, PS2: the fixed part of a mode-C proof (u, opened columns, Merkle paths) under lower-rate commitment plans.

At short prompts the fixed part dominates a mode-C query, and it falls with the code rate of the commitment. This
prints, per model, commitment policy and Fiat-Shamir setting, the fixed bytes per query (from ``analytic.proof_bytes``,
which reproduces stored records within about 1%) and the one-time setup size (encoded entries), with the setup time
at a stated encoding rate.

    cd code && PYTHONPATH=src python experiments/6_improvements/ps2_tables.py --models gpt2 opt-1.3b llama2-7b llama2-13b qwen3-4b
"""
from __future__ import annotations

import argparse
import json

from pvi.commit.cost import decoder_shapes, proof_bytes
from pvi.commit.plans import plan_commitment
from pvi.v0.params import params_for
from pvi.models.decoder import CONFIGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "opt-1.3b", "llama2-7b", "llama2-13b", "qwen3-4b"])
    ap.add_argument("--policies", nargs="+", default=["cnn16c", "cnn18c", "R16c", "R64c"])
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--rate-entries-per-s", type=float, default=67e6, help="measured setup encoding rate")
    args = ap.parse_args()
    for name in args.models:
        cfg = CONFIGS[name]
        ops = decoder_shapes(cfg, prune_last=True)
        for policy in args.policies:
            plan = plan_commitment(ops, policy, model_ops=ops)
            entries = sum(m.n_points * m.n_rows for m in plan.matrices) if plan is not None else None
            for fs in (False, True):
                params = params_for(args.lam, len(ops), plan=plan, fiat_shamir=fs)
                b = proof_bytes(ops, params, 64, plan=plan, mode="C")
                fixed = b["u"] + b["columns"] + b["paths"]
                print(json.dumps({"model": name, "policy": policy, "fs": fs, "reps": params.reps,
                                  "fixed_MB": round(fixed / 1e6, 2), "u_MB": round(b["u"] / 1e6, 2),
                                  "columns_MB": round(b["columns"] / 1e6, 2), "paths_MB": round(b["paths"] / 1e6, 2),
                                  "setup_entries": entries,
                                  "setup_s_est": round(entries / args.rate_entries_per_s) if entries else None}),
                      flush=True)


if __name__ == "__main__":
    main()
