"""Proof bytes, soundness and setup size of the commitment plans (``pvi.fullcheck.plans``).

    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --out results/plan_bytes.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run lenet5 vgg16 gpt2:64,512:12 \\
        --policies paper tight cnn16 R16 --out results/plan_bytes_run.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run lenet5 --policies paper cnn16 \\
        --wire off on --out results/combined_run_lenet5.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2 --claims-only \\
        --policies paper R16 R64 --wire off on --out results/combined_model_llama.csv
    cd code && PYTHONPATH=src python experiments/6_improvements/plan_bytes.py --run qwen3-4b:8:4,8 --policies \\
        --kpre off on --prune-last --wire off on --out results/lookup_kpre_qwen.csv

Without ``--run``: one row per (model, prompt length, policy, challenges) at ``--lam`` from the
shapes alone (``analytic.proof_bytes``: claims, ``u`` and opened columns exact, Merkle paths their
expectation; ``analytic.setup_size``), for the benchmark's CNNs (random-init, the bytes depend on
the shapes only) and the decoders of Table 4.  With ``--run model[:seqs[:blocks[:vocab]]]``: the
model is built and committed under every policy and ``--queries`` honest queries (distinct random
inputs: ``ab_verifier.random_queries`` for a CNN, random prompts for a decoder) are run with
``run_query`` (interactive and Fiat--Shamir); each row holds the measured bytes (paths: the
median) next to the model's, and every measured claim, ``u`` and column byte count must equal it.
``--wire off on`` runs every policy without and with the compact wire encoding
(``run_query(wire=True)``; the model then takes the measured size of the encoded claims, which
depends on their values).  ``--claims-only`` commits nothing, for setups too large for this
machine: the claims of the same queries are measured and every other part is the model's.  A
decoder built at several block counts (``gpt2:64:1,2``) also gets rows of the whole model: the
model, with the claim bytes extrapolated linearly from the last two builds (exact without wire,
which is checked).  ``--prune-last`` builds (and models) the decoders with their last block at the
last position (``build_decoder(..., prune_last=True)``).  ``--kpre off on`` adds rows of mode Kpre
(the proof is its claims; ``policy`` "-"), with the verifier reading the embedding rows itself
(``Verifier(lookups=True)``) off and/or on.  A ``c`` policy's lookup tables send their claims as
int8 rows and one multiproof each, which the rows count exactly (the model takes the ids a query
looks up; for the whole model, a learned position table's ``0 .. T-1`` and the expectation of
distinct uniform tokens).  Nothing here writes to the stored benchmark roots.
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
from pvi.fullcheck.commitment import next_pow2
from pvi.fullcheck.plans import plan_commitment
from pvi.fullcheck.protocol import (Challenger, Prover, Verifier, commit_graph, params_for, run_query,
                                    soundness_bits)
from pvi.fullcheck.transformer import CONFIGS, build_decoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_verifier import Case, build, random_queries  # noqa: E402

CNNS = ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar")
DECODERS = (("gpt2", 64), ("gpt2", 512), ("llama2-7b", 1), ("llama2-7b", 64), ("llama2-7b", 2048), ("qwen3-4b", 8),
            ("opt-125m", 2048), ("opt-350m", 2048), ("opt-1.3b", 2048), ("opt-2.7b", 2048), ("opt-6.7b", 2048),
            ("opt-13b", 2048))
POLICIES = ("paper", "tight", "cnn16", "cnn17", "cnn18", "R8", "R16", "R64", "tightc", "cnn16c", "cnn17c", "cnn18c",
            "R8c", "R16c", "R64c")


def cnn_graph(name: str):
    """The benchmark's CNN (random init, as ``ab_verifier.py`` builds it), a query, and each weight
    op's claim columns."""
    graph, x, _ = build(fullcheck, Case(name, "C", 128))
    _, claims = graph.forward(x)
    return graph, x, {k: z.shape[1] for k, z in claims.items()}


def analytic_rows(lam: int, policies, prune_last: bool = False, kpre=()) -> list[dict]:
    """The byte model of every CNN and decoder of Table 4: per policy (mode C) and, with ``kpre``, mode
    Kpre with the verifier reading the embedding rows itself off and/or on."""
    cases = [(name, None, *cnn_graph(name)[::2]) for name in CNNS]
    cases += [(name, seq, None, decoder_claim_columns(CONFIGS[name], seq, prune_last=prune_last))
              for name, seq in DECODERS]
    rows = []
    for name, seq, graph, claim_cols in cases:
        ops = graph.mat_ops if graph is not None else decoder_shapes(CONFIGS[name], prune_last=prune_last)
        ids = _position_ids(CONFIGS[name], seq) if graph is None else None
        for policy in policies:
            plan = plan_commitment(ops, policy)
            setup = setup_size(ops, plan=plan)
            for chal in ("int", "fs"):
                params = params_for(lam, len(ops), fiat_shamir=chal == "fs", plan=plan)
                cols = None if plan is None else plan.matrix_columns(params.group_columns)
                shapes = plan.shapes() if plan is not None else [(op.row_length, 4 * next_pow2(op.row_length))
                                                                  for op in ops]
                b = proof_bytes(ops, params, claim_cols, plan=plan, table_ids=ids)
                rows.append({"model": name, "seq": seq, "mode": "C", "policy": policy, "prune_last": prune_last,
                             "challenges": chal, "lam": lam, "reps": params.reps,
                             **{f"bytes_{k}": v for k, v in b.items()}, "bytes_total": sum(b.values()),
                             "soundness_bits": soundness_bits(params, shapes, "C", columns=cols),
                             "columns_opened": sum(cols) if cols else params.columns * len(ops), **setup})
        for lookups in kpre:
            params = params_for(lam, len(ops))
            b = proof_bytes(ops, params, claim_cols, mode="Kpre", lookups=lookups)
            rows.append({"model": name, "seq": seq, "mode": "Kpre", "policy": "-", "lookups": lookups,
                         "prune_last": prune_last, "challenges": "int", "lam": lam, "reps": params.reps,
                         **{f"bytes_{k}": v for k, v in b.items()}, "bytes_total": sum(b.values()),
                         "soundness_bits": soundness_bits(params, [(op.row_length, 0) for op in ops
                                                                   if not (lookups and op.layout == "embed")], "K")})
    base = {(r["model"], r["seq"], r["challenges"]): r["bytes_total"] for r in rows if r["policy"] == "paper"}
    for r in rows:
        if r["mode"] == "C" and (r["model"], r["seq"], r["challenges"]) in base:
            r["ratio_vs_paper"] = r["bytes_total"] / base[(r["model"], r["seq"], r["challenges"])]
    return rows


def _position_ids(cfg, seq: int) -> dict[str, list[int]]:
    """``{"pos": 0 .. seq - 1}``: what a decoder's learned position table looks up (named as
    ``decoder_shapes``), none for a model without one."""
    return {"pos": list(range(seq))} if cfg.pos == "learned" else {}


def query_claims(graph, x) -> tuple[list[torch.Tensor], int]:
    """The honest claims of query ``x`` (in weight-op order) and the bytes of their wire encoding."""
    zs = list(Prover(graph).claims(x).values())
    return zs, len(claimcodec.encode(zs))


def _variant_claims(graph, x, tables=frozenset(), own=frozenset()) -> tuple[dict[bool, int], dict[str, list[int]]]:
    """The claims of query ``x`` as ``run_query`` sends them -- ``False``: 4 bytes each, a lookup table's
    (``tables``) 1; ``True``: the ``PVC3`` bytes of the others and the tables' int8 rows -- none of the
    ops in ``own`` (rows the verifier reads itself), and the ids each table looks up."""
    ids: dict[str, list[int]] = {}

    def watch(op, xin):
        if op.name in tables:
            ids[op.name] = xin.reshape(-1).tolist()

    _, claims = graph.forward(x, watch=watch)
    sent = [op.name for op in graph.mat_ops if op.name not in own]
    plain = sum((1 if name in tables else 4) * claims[name].numel() for name in sent)
    wired = (len(claimcodec.encode([claims[name] for name in sent if name not in tables]))
             + len(claimcodec.pack_rows([claims[name] for name in sent if name in tables])))
    return {False: plain, True: wired}, ids


def _medians(parts: list[dict]) -> dict:
    """``bytes_<part>``: the median of each part over ``parts``, and ``bytes_total`` the median total."""
    out = {f"bytes_{k}": statistics.median(b[k] for b in parts) for k in parts[0]}
    out["bytes_total"] = statistics.median(sum(b.values()) for b in parts)
    return out


def _build_rows(graph, cases, policies, lam: int, n_checks: int, wires, claims_only: bool, head: dict,
                model_ops=None, kpre=()) -> list:
    """The rows of one build (``head``: its model, layers, vocabulary and pruning): per variant -- each
    policy in mode C, and mode Kpre with lookups off and/or on (``kpre``) -- prompt, wire and challenge
    kind, the median bytes of ``run_query`` next to the model's (``claims_only``: the model's, on the
    measured claims)."""
    rows = []
    variants = [("C", policy, False) for policy in policies] + [("Kpre", "-", lookups) for lookups in kpre]
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    measured: dict = {}              # (prompt, tables, own) -> each query's claim bytes and looked-up ids
    for mode, policy, lookups in variants:
        t0 = time.perf_counter()
        coms = None if claims_only or mode != "C" else commit_graph(graph, 4, policy=policy, model_ops=model_ops)
        commit_s = None if coms is None else time.perf_counter() - t0
        plan = None if mode != "C" else (plan_commitment(graph.mat_ops, policy, model_ops=model_ops)
                                         if claims_only else coms.plan)
        tables = frozenset(m.name for m in plan.tables) if plan is not None else frozenset()
        own = frozenset(op.name for op in graph.mat_ops if lookups and op.layout == "embed")
        prover = None if claims_only else Prover(graph, commitments=coms)
        for seq, xs, claim_cols in cases:
            if (seq, tables, own) not in measured:
                measured[seq, tables, own] = [_variant_claims(graph, q, tables, own) for q in xs]
            sent = measured[seq, tables, own]
            for wire in wires:
                for chal in (("int", "fs") if mode == "C" else ("int",)):
                    params = params_for(lam, n_checks, fiat_shamir=chal == "fs", plan=plan)
                    wants = [proof_bytes(graph.mat_ops, params, claim_cols, plan=plan, mode=mode, lookups=lookups,
                                         wire_claims=sizes[True] if wire else None, table_ids=ids)
                             for sizes, ids in sent]
                    if any(want["claims"] != sizes[wire] for want, (sizes, _) in zip(wants, sent)):
                        raise SystemExit(f"{policy}: the measured claim bytes differ from the model")
                    if claims_only:
                        got = dict(_medians(wants), source="model")
                    else:
                        if mode == "C":
                            v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                                         tables=coms.table_publics)
                        else:
                            v = Verifier(graph.public(), params, "Kpre", weights=weights, lookups=lookups)
                            v.precompute(Challenger(seed=0))
                        res = [run_query(prover, v, q, seed=None if chal == "fs" else i, wire=wire)
                               for i, q in enumerate(xs)]
                        if not all(r["accepted"] for r in res):
                            raise SystemExit(f"{mode} {policy} {chal} wire={wire}: an honest query was rejected")
                        if any(r["bytes"][k] != w[k] for r, w in zip(res, wants) for k in ("claims", "u", "columns")):
                            raise SystemExit(f"{mode} {policy} {chal} wire={wire}: the bytes differ from the model")
                        got = dict(_medians([r["bytes"] for r in res]), model_paths=statistics.median(
                            w["paths"] for w in wants), model_total=statistics.median(sum(w.values()) for w in wants),
                                   source="run_query")
                    rows.append({**head, "seq": seq, "mode": mode, "policy": policy, "lookups": lookups, "wire": wire,
                                 "challenges": chal, "lam": lam, **got, "queries": len(xs), "commit_s": commit_s,
                                 **(setup_size(graph.mat_ops, plan=plan) if mode == "C" else {})})
    return rows


def _whole_model_rows(rows: list, cfg, lam: int) -> list:
    """The whole model's rows from those of its two largest builds: the model under the whole model's
    plan, with the claim bytes extrapolated linearly in the blocks (exact without wire: checked)."""
    prune_last = rows[0]["prune_last"]
    model_ops = decoder_shapes(cfg, prune_last=prune_last)
    b0, b1 = sorted({r["layers"] for r in rows})[-2:]
    variant = ("seq", "mode", "policy", "lookups", "wire", "challenges")
    at = {(r["layers"], *(r[k] for k in variant)): r["bytes_claims"] for r in rows}
    out = []
    for (layers, seq, mode, policy, lookups, wire, chal), c1 in at.items():
        if layers != b1:
            continue
        claims = c1 + (cfg.n_layers - b1) * (c1 - at[b0, seq, mode, policy, lookups, wire, chal]) / (b1 - b0)
        whole = plan_commitment(model_ops, policy) if mode == "C" else None
        params = params_for(lam, len(model_ops), fiat_shamir=chal == "fs", plan=whole)
        b = proof_bytes(model_ops, params, decoder_claim_columns(cfg, seq, prune_last=prune_last), plan=whole,
                        mode=mode, lookups=lookups, wire_claims=claims if wire else None,
                        table_ids=_position_ids(cfg, seq))
        if not wire and abs(b["claims"] - claims) > 0.5:
            raise SystemExit(f"{cfg.name}: the claims of the builds do not extrapolate to the whole model's")
        if mode == "C":
            cols = None if whole is None else whole.matrix_columns(params.group_columns)
            shapes = whole.shapes() if whole is not None else [(op.row_length, op.n_points(4)) for op in model_ops]
            bits = soundness_bits(params, shapes, "C", columns=cols)
        else:
            bits = soundness_bits(params, [(op.row_length, 0) for op in model_ops
                                           if not (lookups and op.layout == "embed")], "K")
        out.append({"model": cfg.name, "seq": seq, "layers": cfg.n_layers, "vocab": cfg.vocab, "prune_last": prune_last,
                    "mode": mode, "policy": policy, "lookups": lookups, "wire": wire, "challenges": chal, "lam": lam,
                    **{f"bytes_{k}": v for k, v in b.items()}, "bytes_total": sum(b.values()),
                    "source": f"model, claims extrapolated from L{b0} and L{b1}", "soundness_bits": bits,
                    **(setup_size(model_ops, plan=whole) if mode == "C" else {})})
    return out


def run_rows(spec: str, lam: int, policies, queries: int, wires=(False,), claims_only: bool = False,
             prune_last: bool = False, kpre=()) -> list[dict]:
    """Every variant's measured bytes on the built model, next to the model's: ``model`` (a CNN) or
    ``model:seqs[:blocks[:vocab]]`` (a decoder: comma-separated prompt lengths and block counts;
    ``vocab`` shrinks the vocabulary of a cheap validation build -- the model is then evaluated for
    that vocabulary).  ``wires``: without and/or with the compact wire encoding; ``claims_only``:
    nothing committed, every part but the claims from the model; ``prune_last``: the decoder's last
    block pruned; ``kpre``: mode Kpre rows, with lookups off and/or on."""
    name, *rest = spec.split(":")
    if name in CNNS:
        graph, x, claim_cols = cnn_graph(name)
        xs = random_queries(fullcheck, graph, x, queries)
        rows = _build_rows(graph, [(None, xs, claim_cols)], policies, lam, len(graph.mat_ops), wires,
                           claims_only, {"model": name, "layers": None, "vocab": None, "prune_last": False},
                           kpre=kpre)
        for r in rows:
            print(r, flush=True)
        return rows
    cfg = CONFIGS[name]
    if len(rest) > 2:
        cfg = dataclasses.replace(cfg, vocab=int(rest[2]))
    seqs = [int(v) for v in rest[0].split(",")] if rest else [64]
    blocks = [int(v) for v in rest[1].split(",")] if len(rest) > 1 else [cfg.n_layers]
    model_ops = decoder_shapes(cfg, prune_last=prune_last)
    rows = []
    for layers in blocks:
        graph = build_decoder(cfg, n_layers=layers, calib_tokens=min(max(seqs), 32), seed=0, prune_last=prune_last)
        cases = [(seq, list(torch.randint(0, cfg.vocab, (queries, seq), generator=torch.Generator().manual_seed(1))
                            .split(1)),
                  {op.name: c for op, c in zip(graph.mat_ops, decoder_claim_columns(cfg, seq, layers,
                                                                                    prune_last).values())})
                 for seq in seqs]
        head = {"model": name, "layers": layers, "vocab": cfg.vocab, "prune_last": prune_last}
        for row in _build_rows(graph, cases, policies, lam, len(model_ops), wires, claims_only, head, model_ops, kpre):
            if not row["wire"]:        # the whole model's bytes, which 1- and 2-block builds extrapolate to
                whole = plan_commitment(model_ops, row["policy"]) if row["mode"] == "C" else None
                row["whole_model_total"] = sum(proof_bytes(
                    model_ops, params_for(lam, len(model_ops), fiat_shamir=row["challenges"] == "fs", plan=whole),
                    decoder_claim_columns(cfg, row["seq"], prune_last=prune_last), plan=whole, mode=row["mode"],
                    lookups=row["lookups"], table_ids=_position_ids(cfg, row["seq"])).values())
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
    ap.add_argument("--policies", nargs="*", default=list(POLICIES), help="mode-C commitment plans (none: no C rows)")
    ap.add_argument("--run", nargs="*", default=[], help="model[:seqs[:blocks[:vocab]]]: build, commit, measure")
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--wire", nargs="*", choices=["off", "on"], default=["off"],
                    help="--run: without (off) and/or with (on) the compact wire encoding; a bare --wire: on")
    ap.add_argument("--claims-only", action="store_true",
                    help="--run: commit nothing, measure the claims and take every other part from the model")
    ap.add_argument("--prune-last", action="store_true",
                    help="decoders: the last block at the last position only (build_decoder(prune_last=True))")
    ap.add_argument("--kpre", nargs="*", choices=["off", "on"], default=[],
                    help="also mode-Kpre rows, the verifier reading the embedding rows itself off and/or on")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wires = [w == "on" for w in args.wire or ["on"]]
    kpre = [k == "on" for k in args.kpre]
    if not args.run and (wires != [False] or args.claims_only):
        raise SystemExit("--wire and --claims-only measure the claims: they go with --run")
    rows = [r for spec in args.run for r in run_rows(spec, args.lam, args.policies, args.queries, wires,
                                                      args.claims_only, args.prune_last, kpre)] if args.run \
        else analytic_rows(args.lam, args.policies, args.prune_last, kpre)
    if args.out:
        path = Path(__file__).resolve().parent / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    if not args.run:
        for r in rows:
            print(f"{r['model']:>15} {str(r['seq']):>5} {r['mode']:>4} {r['policy']:>6} {r['challenges']:>3} "
                  f"{r['bytes_total'] / 1e3:12.1f} kB  x{r.get('ratio_vs_paper', float('nan')):.3f}  "
                  f"{r['soundness_bits']:.1f} bits" + (f"  trees {r['trees']:>3}  entries "
                                                      f"{r['encoded_entries'] / 1e6:10.1f}M  max_n "
                                                      f"2^{r['max_n'].bit_length() - 1}" if r["mode"] == "C" else ""))


if __name__ == "__main__":
    main()
