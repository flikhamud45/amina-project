"""The bytes of every message ``run_query(wire=True)`` encodes, hashed, in two checkouts: the same or not.

    cd code && export PYTHONPATH=$PWD/src
    python experiments/6_improvements/wire_hashes.py --base ../../base/code/src \\
        --out results/codec_bytes_identity.jsonl
    python experiments/6_improvements/wire_hashes.py --out hashes.jsonl lenet5_C_paper qwen3-4b_L4_T8_Kpre

For fixed models (random weights; decoders with seed 0), commitment plans, modes, queries and seeds
(``run_query(..., seed=i)``: the same challenges, so the same opened columns), the first 20 hex digits of
the SHA-256 of each message ``claimcodec.encode`` (the claims, ``PVC3``), ``pack_rows`` (a plan's lookup
rows) and ``pack_field`` (``u``, the opened columns) return, with each query's verdict and byte counts.
With ``--base`` (another checkout's ``src``) the cases run in two processes, one with each checkout's
``src`` on ``PYTHONPATH`` (both with this checkout's model builders, ``ab_verifier.py``), and each line
holds both checkouts' hashes, verdicts and byte counts and whether they are ``identical``; without it,
this checkout's alone.  Cases: the names below (default: all).  Nothing here writes to the benchmark's roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]

# (name, model, blocks, tokens, mode, policy, lookups, prune_last, lean, lambda, queries)
CASES = [(f"lenet5_C_{p}", "lenet5", None, None, "C", p, False, False, False, 128, 4)
         for p in ("paper", "cnn16", "cnn17c", "R16")] + [
    ("lenet5_Kpre", "lenet5", None, None, "Kpre", None, False, False, False, 128, 4),
    ("vgg16_C_paper", "vgg16", None, None, "C", "paper", False, False, False, 128, 2),
    ("vgg16_Kpre", "vgg16", None, None, "Kpre", None, False, False, False, 128, 2),
    ("gpt2_T64_Kpre", "gpt2", 12, 64, "Kpre", None, False, False, False, 128, 3),
    ("gpt2_T64_Kpre_lookups_prune", "gpt2", 12, 64, "Kpre", None, True, True, False, 128, 2),
    ("gpt2_T64_C_paper", "gpt2", 12, 64, "C", "paper", False, False, False, 128, 2),
    ("gpt2_L2_T64_C_tightc_prune", "gpt2", 2, 64, "C", "tightc", False, True, False, 128, 2),
    ("qwen3-4b_L4_T8_Kpre", "qwen3-4b", 4, 8, "Kpre", None, False, False, False, 40, 3),
    ("qwen3-4b_L4_T8_Kpre_lookups_prune", "qwen3-4b", 4, 8, "Kpre", None, True, True, False, 40, 2),
    ("qwen3-4b_L4_T8_Kpre_lean", "qwen3-4b", 4, 8, "Kpre", None, False, False, True, 40, 2),
    ("llama2-7b_L1_T16_Kpre", "llama2-7b", 1, 16, "Kpre", None, False, False, False, 40, 2),
]


def dump(out: Path, want: list[str]) -> None:
    """Runs the cases with the ``pvi`` on ``sys.path``, one JSON line per query into ``out``."""
    import torch

    sys.path.insert(0, str(HERE))
    import ab_verifier as ab
    import pvi.fullcheck as fullcheck
    from pvi.fullcheck import claimcodec as cc
    from pvi.fullcheck import protocol as proto
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    torch.set_num_threads(4)
    record = []

    def hashed(name, real):
        def call(values, *a, **k):
            blob = real(values, *a, **k)
            record.append((name, len(blob), hashlib.sha256(blob).hexdigest()[:20]))
            return blob
        return call

    proto.claimcodec.encode = hashed("claims", cc.encode)
    proto.claimcodec.pack_rows = hashed("rows", cc.pack_rows)
    proto.claimcodec.pack_field = hashed("field", cc.pack_field)
    with out.open("w") as f:
        for name, model, layers, seq, mode, policy, lookups, prune, lean, lam, n_q in CASES:
            if want and name not in want:
                continue
            t0 = time.time()
            if model in ab.CNNS:
                graph, x, _ = ab.build(fullcheck, ab.Case(model, "C", lam))
                queries = ab.random_queries(fullcheck, graph, x, n_q)
                model_ops, n_checks = None, len(graph.mat_ops)
            else:
                cfg = CONFIGS[model]
                graph = build_decoder(cfg, n_layers=layers, calib_tokens=min(seq, 32), seed=0, prune_last=prune)
                model_ops = decoder_shapes(cfg, prune_last=prune)
                n_checks = len(model_ops)
                tokens = torch.randint(0, cfg.vocab, (n_q, seq), generator=torch.Generator().manual_seed(1))
                queries = list(tokens.split(1))
            if mode == "C":
                coms = proto.commit_graph(graph, 4, policy=policy, model_ops=model_ops)
                params = proto.params_for(lam, n_checks, rate=4, plan=coms.plan)
                v = proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                                   tables=coms.table_publics, lean=lean)
            else:
                coms = None
                params = proto.params_for(lam, n_checks, rate=4)
                v = proto.Verifier(graph.public(), params, mode, lookups=lookups, lean=lean,
                                   weights={op.name: (op.weight, op.bias) for op in graph.mat_ops})
                v.precompute(proto.Challenger())
            prover = proto.Prover(graph, commitments=coms, lean=lean)
            for i, q in enumerate(queries):
                record.clear()
                r = proto.run_query(prover, v, q, seed=i, wire=True)
                f.write(json.dumps({"case": name, "query": i, "accepted": r["accepted"], "rejected_at": r["rejected_at"],
                                    "bytes": r["bytes"], "messages": list(record)}) + "\n")
                f.flush()
            print(name, f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)


def _run(src: Path, out: Path, cases: list[str]) -> list[dict]:
    env = dict(os.environ, PYTHONPATH=str(src))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--dump", "--out", str(out), *cases], env=env,
                   cwd=src.parent, check=True)
    return [json.loads(line) for line in out.read_text().splitlines()]


def _git(path: Path) -> str | None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=path)
    return sha.stdout.strip() or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("--base", type=Path, help="the other checkout's src")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dump", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.dump:
        return dump(args.out, args.cases)
    if args.base is None:
        return dump(args.out, args.cases)
    with tempfile.TemporaryDirectory() as tmp:
        this = _run(CODE / "src", Path(tmp) / "this.jsonl", args.cases)
        base = _run(args.base.resolve(), Path(tmp) / "base.jsonl", args.cases)
    if [(r["case"], r["query"]) for r in this] != [(r["case"], r["query"]) for r in base]:
        raise SystemExit("the checkouts ran different queries")
    shas = {"this": _git(CODE), "base": _git(args.base.resolve())}
    same = 0
    with args.out.open("w") as f:
        for a, b in zip(base, this):
            row = {"case": a["case"], "query": a["query"],
                   "identical": a["messages"] == b["messages"] and a["bytes"] == b["bytes"]
                   and (a["accepted"], a["rejected_at"]) == (b["accepted"], b["rejected_at"]),
                   "accepted": [a["accepted"], b["accepted"]], "bytes": [a["bytes"], b["bytes"]],
                   "sha256_20_base": [m[2] for m in a["messages"]], "sha256_20_this": [m[2] for m in b["messages"]],
                   "messages": [m[:2] for m in b["messages"]], "git": shas}
            same += row["identical"]
            f.write(json.dumps(row) + "\n")
    print(f"{same} of {len(this)} queries identical "
          f"({sum(len(r['messages']) for r in this)} messages, {len({r['case'] for r in this})} cases)")
    if same != len(this):
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
