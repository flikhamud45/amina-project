"""Decoder weight-op shapes, the expected size of a Merkle multiproof, and a query's proof size.

* ``decoder_shapes`` lists the weight ops of a decoder (see
  ``transformer.build_decoder``) without building it: the benchmark sizes
  ``(r, t)`` for the full model's op count, and ``aggregate.py`` needs the
  codeword length of every committed matrix;
* ``expected_multiproof_nodes`` is the exact expected number of hashes in one
  multiproof for ``t`` distinct uniform columns, which ``aggregate.py`` uses as
  the Merkle term of LLM runs made before multiproofs (checked by Monte Carlo in
  ``tests/test_fullcheck.py``);
* ``proof_bytes`` and ``setup_size`` give a query's proof bytes and the commitment's size for
  weight-op shapes alone, under the report's parameters or a commitment plan
  (:mod:`pvi.fullcheck.plans`), for models too large to run (checked against ``run_query`` in
  ``tests/test_plans.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["OpShape", "decoder_shapes", "decoder_claim_columns", "expected_multiproof_nodes", "proof_bytes",
           "setup_size"]


@dataclass(frozen=True)
class OpShape:
    name: str
    n_rows: int
    row_length: int
    layout: str = "linear"       # "embed" for the lookup tables (as ``MatOp.layout``)

    def n_points(self, rate: int) -> int:
        return rate * (1 << max(0, (self.row_length - 1).bit_length()))


def decoder_shapes(cfg, n_layers: int | None = None) -> list[OpShape]:
    """Weight-op shapes of a decoder (see ``transformer.build_decoder``) without building it."""
    L = cfg.n_layers if n_layers is None else n_layers
    d, dh, hq, hkv, f = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff
    b = 1 if cfg.bias else 0
    e = getattr(cfg, "embed_dim", 0) or d
    ops = [OpShape("embed", e, cfg.vocab, "embed")]
    if e != d:
        ops.append(OpShape("proj_in", d, e))
    if cfg.pos == "learned":
        ops.append(OpShape("pos", d, cfg.max_pos, "embed"))
    for i in range(L):
        ops += [OpShape(f"q{i}", hq * dh, d + b), OpShape(f"k{i}", hkv * dh, d + b),
                OpShape(f"v{i}", hkv * dh, d + b), OpShape(f"o{i}", d, hq * dh + b)]
        if cfg.mlp == "swiglu":
            ops += [OpShape(f"gate{i}", f, d + b), OpShape(f"up{i}", f, d + b), OpShape(f"down{i}", d, f + b)]
        else:
            ops += [OpShape(f"fc1{i}", f, d + b), OpShape(f"fc2{i}", d, f + b)]
    if e != d:
        ops.append(OpShape("proj_out", e, d))
    ops.append(OpShape("head", cfg.vocab, e))
    return ops


def decoder_claim_columns(cfg, seq: int, n_layers: int | None = None) -> dict[str, int]:
    """Each weight op's claim columns for a prompt of ``seq`` tokens, keyed as :func:`decoder_shapes`:
    ``seq``, but 1 for the LM head and OPT-350M's project_out, which read the last position only."""
    return {s.name: 1 if s.name in ("proj_out", "head") else seq for s in decoder_shapes(cfg, n_layers)}


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def expected_multiproof_nodes(n: int, t: int) -> float:
    """Expected hashes in a multiproof for ``t`` distinct uniform leaves of ``n``.

    At a level whose nodes cover ``s`` leaves, a sibling hash is sent for each
    pair of nodes where exactly one of the two covers an opened leaf; there are
    ``n / 2s`` pairs and that happens with probability
    ``2 [C(n-s, t) - C(n-2s, t)] / C(n, t)``.
    """
    t = min(t, n)
    total, s = 0.0, 1
    base = _log_comb(n, t)
    while s < n:
        a = math.exp(_log_comb(n - s, t) - base)
        b = math.exp(_log_comb(n - 2 * s, t) - base) if n - 2 * s >= t else 0.0
        total += (n / s) * (a - b)
        s *= 2
    return total


def _n_points(op, plan, rate: int) -> int:
    return rate * (1 << max(0, (op.row_length - 1).bit_length())) if plan is None else plan.op(op.name).n_points


def _trees(ops, plan, rate: int) -> list[tuple[str, int, int]]:
    """``(name, codeword length, rows)`` of every Merkle tree holding one of ``ops``."""
    if plan is None:
        return [(op.name, _n_points(op, None, rate), op.n_rows) for op in ops]
    rows = {op.name: op.n_rows for op in ops}
    return [(g, plan.op(members[0]).n_points, sum(rows[m] for m in members if m in rows))
            for g, members in plan.groups if any(m in rows for m in members)]


def proof_bytes(ops, params, claim_columns, *, plan=None, rate: int = 4, mode: str = "C") -> dict[str, float]:
    """One query's proof bytes as ``run_query`` counts them: the claims, ``u`` and the opened columns
    exactly, and the Merkle multiproofs' expected size (the column indices are random).  ``ops``:
    weight-op shapes (``decoder_shapes`` or a graph's ``MatOp``s), whose claims have
    ``claim_columns`` columns (an int for all of them, e.g. the prompt length of a decoder, or
    ``{name: columns}``); ``plan``: a commitment plan (its groups restricted to ``ops``), else the
    report's trees at ``rate``.  Modes K and Kpre send only the claims."""
    def cols(op):
        return claim_columns if isinstance(claim_columns, int) else claim_columns[op.name]

    out = {"claims": 4 * sum(op.n_rows * cols(op) for op in ops), "u": 0, "columns": 0, "paths": 0.0}
    if mode == "C":
        trees = [(n, min(params.columns if plan is None else params.columns_for(g), n), rows)
                 for g, n, rows in _trees(ops, plan, rate)]
        out["u"] = 4 * params.reps * sum(op.row_length for op in ops)
        out["columns"] = 4 * sum(t * rows for _, t, rows in trees)
        out["paths"] = 32 * sum(expected_multiproof_nodes(n, t) for n, t, _ in trees)
    return out


def setup_size(ops, *, plan=None, rate: int = 4) -> dict[str, int]:
    """The one-time commitment of ``ops``: encoded field entries (``sum_l N_l n_l``, what the NTTs
    produce and the column digests hash), Merkle leaves, trees and the longest codeword."""
    trees = _trees(ops, plan, rate)
    return {"encoded_entries": sum(op.n_rows * _n_points(op, plan, rate) for op in ops),
            "leaves": sum(n for _, n, _ in trees), "trees": len(trees), "max_n": max(n for _, n, _ in trees)}
