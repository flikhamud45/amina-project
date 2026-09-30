"""Proof bytes, soundness and setup size of the commitment plans (``pvi.fullcheck.plans``).

    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --out results/plan_bytes.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run lenet5 vgg16 gpt2:64,512:12 \\
        --policies paper tight cnn16 R16 --out results/plan_bytes_run.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run lenet5 --policies paper cnn16 \\
        --wire off on --out results/combined_run_lenet5.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2 --claims-only \\
        --policies paper R16 R64 --wire off on --out results/combined_model_llama.csv

Without ``--run``: one row per (model, prompt length, policy, challenges) at ``--lam`` from the
shapes alone (``analytic.proof_bytes``: claims, ``u`` and opened columns exact, Merkle paths their
expectation; ``analytic.setup_size``), for the benchmark's CNNs (random-init, the bytes depend on
the shapes only) and the decoders of Table 4.  With ``--run model[:seqs[:blocks[:vocab]]]``: the
model is built and committed under every policy and ``--queries`` honest queries are run with
``run_query`` (interactive and Fiat--Shamir); each row holds the measured bytes (paths: the
median) next to the model's, and every measured claim, ``u`` and column byte count must equal it.
``--wire off on`` runs every policy without and with the compact wire encoding
(``run_query(wire=True)``; the model then takes the measured size of the encoded claims, which
depends on their values).  ``--claims-only`` commits nothing, for setups too large for this
machine: the claims of the same queries are measured and every other part is the model's.  A
decoder built at several block counts (``gpt2:64:1,2``) also gets rows of the whole model: the
model, with the claim bytes extrapolated linearly from the last two builds (exact without wire,
which is checked).  Nothing here writes to the stored benchmark roots.
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
from pvi.fullcheck import claimcodec
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


def _claim_bytes(graph, xs) -> dict[bool, float]:
    """The median bytes of the claims of queries ``xs`` as ``run_query`` counts them: 4 each (``False``),
    or with wire (``True``) their encoding."""
    sizes = {False: [], True: []}
    for x in xs:
        zs = [z for z in Prover(graph).claims(x).values()]
        sizes[False].append(4 * sum(z.numel() for z in zs))
        sizes[True].append(len(claimcodec.encode(zs)))
    return {wire: statistics.median(v) for wire, v in sizes.items()}


def _build_rows(graph, cases, policies, lam: int, n_checks: int, wires, claims_only: bool, head: dict,
                model_ops=None) -> list:
    """The rows of one build (``head``: its model, layers and vocabulary): per policy, prompt, wire and
    challenge kind, the median bytes of ``run_query`` next to the model's (``claims_only``: the
    model's, on the measured claims)."""
    rows = []
    measured = {seq: _claim_bytes(graph, xs) for seq, xs, _ in cases} if claims_only else {}
    for policy in policies:
        t0 = time.perf_counter()
        coms = None if claims_only else commit_graph(graph, 4, policy=policy, model_ops=model_ops)
        commit_s = None if claims_only else time.perf_counter() - t0
        plan = plan_commitment(graph.mat_ops, policy, model_ops=model_ops) if claims_only else coms.plan
        prover = None if claims_only else Prover(graph, commitments=coms)
        for seq, xs, claim_cols in cases:
            for wire in wires:
                for chal in ("int", "fs"):
                    params = params_for(lam, n_checks, fiat_shamir=chal == "fs", plan=plan)
                    if claims_only:
                        want = proof_bytes(graph.mat_ops, params, claim_cols, plan=plan,
                                           wire_claims=measured[seq][True] if wire else None)
                        if want["claims"] != measured[seq][wire]:
                            raise SystemExit(f"{policy}: the measured claim bytes differ from the model")
                        got = {f"bytes_{k}": v_ for k, v_ in want.items()}
                        got.update(bytes_total=sum(want.values()), source="model")
                    else:
                        v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics)
                        res = [run_query(prover, v, q, seed=None if chal == "fs" else i, wire=wire)
                               for i, q in enumerate(xs)]
                        if not all(r["accepted"] for r in res):
                            raise SystemExit(f"{policy} {chal} wire={wire}: an honest query was rejected")
                        for r in res:          # with wire, the model takes the size of the encoded claims
                            want = proof_bytes(graph.mat_ops, params, claim_cols, plan=plan,
                                               wire_claims=r["bytes"]["claims"] if wire else None)
                            if any(r["bytes"][k] != want[k] for k in ("claims", "u", "columns")):
                                raise SystemExit(f"{policy} {chal} wire={wire}: measured bytes differ from the model")
                        got = {f"bytes_{k}": statistics.median(r["bytes"][k] for r in res) for k in res[0]["bytes"]}
                        got.update(bytes_total=statistics.median(sum(r["bytes"].values()) for r in res),
                                   model_paths=want["paths"], model_total=sum(want.values()), source="run_query")
                    rows.append({"model": head["model"], "seq": seq, "layers": head["layers"], "vocab": head["vocab"],
                                 "policy": policy, "wire": wire, "challenges": chal, "lam": lam, **got,
                                 "queries": len(xs), "commit_s": commit_s, **setup_size(graph.mat_ops, plan=plan)})
    return rows


def _whole_model_rows(rows: list, cfg, lam: int) -> list:
    """The whole model's rows from those of its two largest builds: the model under the whole model's
    plan, with the claim bytes extrapolated linearly in the blocks (exact without wire: checked)."""
    model_ops = decoder_shapes(cfg)
    b0, b1 = sorted({r["layers"] for r in rows})[-2:]
    at = {(r["layers"], r["seq"], r["policy"], r["wire"], r["challenges"]): r["bytes_claims"] for r in rows}
    out = []
    for (layers, seq, policy, wire, chal), c1 in at.items():
        if layers != b1:
            continue
        claims = c1 + (cfg.n_layers - b1) * (c1 - at[b0, seq, policy, wire, chal]) / (b1 - b0)
        whole = plan_commitment(model_ops, policy)
        params = params_for(lam, len(model_ops), fiat_shamir=chal == "fs", plan=whole)
        b = proof_bytes(model_ops, params, decoder_claim_columns(cfg, seq), plan=whole,
                        wire_claims=claims if wire else None)
        if not wire and abs(b["claims"] - claims) > 0.5:
            raise SystemExit(f"{cfg.name}: the claims of the builds do not extrapolate to the whole model's")
        cols = None if whole is None else whole.op_columns(params.group_columns)
        shapes = whole.shapes() if whole is not None else [(op.row_length, op.n_points(4)) for op in model_ops]
        out.append({"model": cfg.name, "seq": seq, "layers": cfg.n_layers, "vocab": cfg.vocab, "policy": policy,
                    "wire": wire, "challenges": chal, "lam": lam, **{f"bytes_{k}": v for k, v in b.items()},
                    "bytes_total": sum(b.values()), "source": f"model, claims extrapolated from L{b0} and L{b1}",
                    "soundness_bits": soundness_bits(params, shapes, "C", columns=cols),
                    **setup_size(model_ops, plan=whole)})
    return out


def run_rows(spec: str, lam: int, policies, queries: int, wires=(False,), claims_only: bool = False) -> list[dict]:
    """Every policy's measured bytes on the built model, next to the model's: ``model`` (a CNN) or
    ``model:seqs[:blocks[:vocab]]`` (a decoder: comma-separated prompt lengths and block counts;
    ``vocab`` shrinks the vocabulary of a cheap validation build -- the model is then evaluated for
    that vocabulary).  ``wires``: without and/or with the compact wire encoding; ``claims_only``:
    nothing committed, every part but the claims from the model."""
    name, *rest = spec.split(":")
    if name in CNNS:
        graph, x, claim_cols = cnn_graph(name)
        rows = _build_rows(graph, [(None, [x] * queries, claim_cols)], policies, lam, len(graph.mat_ops), wires,
                           claims_only, {"model": name, "layers": None, "vocab": None})
        for r in rows:
            print(r, flush=True)
        return rows
    cfg = CONFIGS[name]
    if len(rest) > 2:
        cfg = dataclasses.replace(cfg, vocab=int(rest[2]))
    seqs = [int(v) for v in rest[0].split(",")] if rest else [64]
    blocks = [int(v) for v in rest[1].split(",")] if len(rest) > 1 else [cfg.n_layers]
    model_ops = decoder_shapes(cfg)
    rows = []
    for layers in blocks:
        graph = build_decoder(cfg, n_layers=layers, calib_tokens=min(max(seqs), 32), seed=0)
        cases = [(seq, list(torch.randint(0, cfg.vocab, (queries, seq), generator=torch.Generator().manual_seed(1))
                            .split(1)),
                  {op.name: c for op, c in zip(graph.mat_ops, decoder_claim_columns(cfg, seq, layers).values())})
                 for seq in seqs]
        for row in _build_rows(graph, cases, policies, lam, len(model_ops), wires, claims_only,
                               {"model": name, "layers": layers, "vocab": cfg.vocab}, model_ops):
            if not row["wire"]:        # the whole model's bytes, which 1- and 2-block builds extrapolate to
                whole = plan_commitment(model_ops, row["policy"])
                row["whole_model_total"] = sum(proof_bytes(
                    model_ops, params_for(lam, len(model_ops), fiat_shamir=row["challenges"] == "fs", plan=whole),
                    decoder_claim_columns(cfg, row["seq"]), plan=whole).values())
            rows.append(row)
            print(row, flush=True)
        graph = None
    if len(blocks) > 1:
        for row in _whole_model_rows(rows, cfg, lam):
            rows.append(row)
            print(row, flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--run", nargs="*", default=[], help="model[:seqs[:blocks[:vocab]]]: build, commit, measure")
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--wire", nargs="*", choices=["off", "on"], default=["off"],
                    help="--run: without (off) and/or with (on) the compact wire encoding; a bare --wire: on")
    ap.add_argument("--claims-only", action="store_true",
                    help="--run: commit nothing, measure the claims and take every other part from the model")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wires = [w == "on" for w in args.wire or ["on"]]
    if not args.run and (wires != [False] or args.claims_only):
        raise SystemExit("--wire and --claims-only measure the claims: they go with --run")
    rows = [r for spec in args.run for r in run_rows(spec, args.lam, args.policies, args.queries, wires,
                                                      args.claims_only)] if args.run \
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
