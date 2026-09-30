"""Proof bytes, soundness and setup size of the commitment plans (``pvi.fullcheck.plans``).

    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --out results/plan_bytes.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run lenet5 vgg16 gpt2:64,512:12 \\
        --policies paper tight cnn16 R16 --out results/plan_bytes_run.csv

Without ``--run``: one row per (model, prompt length, policy, challenges) at ``--lam`` from the
shapes alone (``analytic.proof_bytes``: claims, ``u`` and opened columns exact, Merkle paths their
expectation; ``analytic.setup_size``), for the benchmark's CNNs (random-init, the bytes depend on
the shapes only) and the decoders of Table 4.  With ``--run model[:seqs[:blocks[:vocab]]]``: the
model is built and committed under every policy and ``--queries`` honest queries are run with
``run_query`` (interactive and Fiat--Shamir); each row holds the measured bytes (paths: the
median) next to the model's, and every measured claim, ``u`` and column byte count must equal it.
Nothing here writes to the stored benchmark roots.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import time
from pathlib import Path

import torch

import pvi.fullcheck as fullcheck
from pvi.fullcheck.analytic import decoder_claim_columns, decoder_shapes, proof_bytes, setup_size
from pvi.fullcheck.plans import next_pow2, plan_commitment
from pvi.fullcheck.protocol import Prover, Verifier, commit_graph, params_for, run_query, soundness_bits
from pvi.fullcheck.transformer import CONFIGS, build_decoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_verifier import Case, build  # noqa: E402

CNNS = ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar")
DECODERS = (("gpt2", 64), ("gpt2", 512), ("llama2-7b", 1), ("llama2-7b", 64), ("llama2-7b", 2048), ("qwen3-4b", 8),
            ("opt-125m", 2048), ("opt-350m", 2048), ("opt-1.3b", 2048), ("opt-2.7b", 2048), ("opt-6.7b", 2048),
            ("opt-13b", 2048))
POLICIES = ("paper", "tight", "cnn16", "cnn17", "cnn18", "R8", "R16", "R64")


def cnn_graph(name: str):
    """The benchmark's CNN (random init, as ``ab_verifier.py`` builds it), a query, and each weight
    op's claim columns."""
    graph, x, _ = build(fullcheck, Case(name, "C", 128))
    _, claims = graph.forward(x)
    return graph, x, {k: z.shape[1] for k, z in claims.items()}


def analytic_rows(lam: int, policies) -> list[dict]:
    cases = [(name, None, *cnn_graph(name)[::2]) for name in CNNS]
    cases += [(name, seq, None, decoder_claim_columns(CONFIGS[name], seq)) for name, seq in DECODERS]
    rows = []
    for name, seq, graph, claim_cols in cases:
        ops = graph.mat_ops if graph is not None else decoder_shapes(CONFIGS[name])
        for policy in policies:
            plan = plan_commitment(ops, policy)
            setup = setup_size(ops, plan=plan)
            for chal in ("int", "fs"):
                params = params_for(lam, len(ops), fiat_shamir=chal == "fs", plan=plan)
                cols = None if plan is None else plan.op_columns(params.group_columns)
                shapes = plan.shapes() if plan is not None else [(op.row_length, 4 * next_pow2(op.row_length))
                                                                  for op in ops]
                b = proof_bytes(ops, params, claim_cols, plan=plan)
                rows.append({"model": name, "seq": seq, "policy": policy, "challenges": chal, "lam": lam,
                             "reps": params.reps, **{f"bytes_{k}": v for k, v in b.items()},
                             "bytes_total": sum(b.values()),
                             "soundness_bits": soundness_bits(params, shapes, "C", columns=cols),
                             "columns_opened": sum(cols) if cols else params.columns * len(ops), **setup})
    base = {(r["model"], r["seq"], r["challenges"]): r["bytes_total"] for r in rows if r["policy"] == "paper"}
    for r in rows:
        r["ratio_vs_paper"] = r["bytes_total"] / base[(r["model"], r["seq"], r["challenges"])]
    return rows


def run_rows(spec: str, lam: int, policies, queries: int) -> list[dict]:
    """Every policy's measured bytes on the built model, next to the model's: ``model`` (a CNN) or
    ``model:seqs[:blocks[:vocab]]`` (a decoder: comma-separated prompt lengths; ``vocab`` shrinks the
    vocabulary of a cheap validation build -- the model is then evaluated for that vocabulary)."""
    name, *rest = spec.split(":")
    if name in CNNS:
        graph, x, claim_cols = cnn_graph(name)
        cases, n_checks, layers, model_ops = [(None, [x] * queries, claim_cols)], len(graph.mat_ops), None, None
    else:
        cfg = CONFIGS[name]
        if len(rest) > 2:
            cfg = dataclasses.replace(cfg, vocab=int(rest[2]))
        seqs = [int(v) for v in rest[0].split(",")] if rest else [64]
        layers = int(rest[1]) if len(rest) > 1 else cfg.n_layers
        graph = build_decoder(cfg, n_layers=layers, calib_tokens=min(max(seqs), 32), seed=0)
        model_ops = decoder_shapes(cfg)
        n_checks = len(model_ops)
        cases = [(seq, list(torch.randint(0, cfg.vocab, (queries, seq), generator=torch.Generator().manual_seed(1))
                            .split(1)),
                  {op.name: c for op, c in zip(graph.mat_ops, decoder_claim_columns(cfg, seq, layers).values())})
                 for seq in seqs]
    rows = []
    for policy in policies:
        t0 = time.perf_counter()
        coms = commit_graph(graph, 4, policy=policy, model_ops=model_ops)
        commit_s = time.perf_counter() - t0
        prover = Prover(graph, commitments=coms)
        for seq, xs, claim_cols in cases:
            for chal in ("int", "fs"):
                params = params_for(lam, n_checks, fiat_shamir=chal == "fs", plan=coms.plan)
                v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics)
                res = [run_query(prover, v, q, seed=None if chal == "fs" else i) for i, q in enumerate(xs)]
                if not all(r["accepted"] for r in res):
                    raise SystemExit(f"{spec} {policy} {chal}: an honest query was rejected")
                got = {k: statistics.median(r["bytes"][k] for r in res) for k in res[0]["bytes"]}
                want = proof_bytes(graph.mat_ops, params, claim_cols, plan=coms.plan)
                for k in ("claims", "u", "columns"):
                    if any(r["bytes"][k] != want[k] for r in res):
                        raise SystemExit(f"{spec} {policy} {chal}: measured {k} bytes differ from the model")
                rows.append({"model": name, "seq": seq, "layers": layers, "vocab": None if name in CNNS else cfg.vocab,
                             "policy": policy, "challenges": chal, "lam": lam,
                             **{f"bytes_{k}": v_ for k, v_ in got.items()},
                             "bytes_total": statistics.median(sum(r["bytes"].values()) for r in res),
                             "model_paths": want["paths"], "model_total": sum(want.values()), "queries": len(res),
                             "commit_s": commit_s, **setup_size(graph.mat_ops, plan=coms.plan)})
                print(rows[-1], flush=True)
        coms = prover = v = None
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--run", nargs="*", default=[], help="model[:seqs[:blocks[:vocab]]]: build, commit, measure")
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = [r for spec in args.run for r in run_rows(spec, args.lam, args.policies, args.queries)] if args.run \
        else analytic_rows(args.lam, args.policies)
    if args.out:
        path = Path(__file__).resolve().parent / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    if not args.run:
        for r in rows:
            print(f"{r['model']:>15} {str(r['seq']):>5} {r['policy']:>6} {r['challenges']:>3} "
                  f"{r['bytes_total'] / 1e3:12.1f} kB  x{r['ratio_vs_paper']:.3f}  {r['soundness_bits']:.1f} bits  "
                  f"trees {r['trees']:>3}  entries {r['encoded_entries'] / 1e6:10.1f}M  "
                  f"max_n 2^{r['max_n'].bit_length() - 1}")


if __name__ == "__main__":
    main()
