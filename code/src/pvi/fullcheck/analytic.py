"""Decoder weight-op shapes, the expected size of a Merkle multiproof, and a query's proof size.

* ``decoder_shapes`` lists the weight ops of a decoder (see
  ``transformer.build_decoder``) without building it: the benchmark sizes
  ``(r, t)`` for the full model's op count, and ``aggregate.py`` needs the
  codeword length of every committed matrix;
* ``expected_multiproof_nodes`` is the exact expected number of hashes in one
  multiproof for ``t`` distinct uniform columns, which ``aggregate.py`` uses as
  the Merkle term of LLM runs made before multiproofs (only the earliest run,
  ``raw/``, which the report does not use; checked by Monte Carlo in
  ``tests/test_fullcheck.py``), and ``expected_lookup_nodes`` its analogue for the
  rows of a lookup table (a tree whose last leaves pad it to a power of two);
* ``proof_bytes`` and ``setup_size`` give a query's proof bytes and the commitment's size for
  weight-op shapes alone, under the report's parameters or a commitment plan
  (:mod:`pvi.fullcheck.plans`), in the default form or the compact wire encoding, for models too
  large to run (checked against ``run_query`` in ``tests/test_plans.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .claimcodec import field_size
from .commitment import multiproof_size, next_pow2

__all__ = ["OpShape", "decoder_shapes", "decoder_claim_columns", "expected_multiproof_nodes",
           "expected_lookup_nodes", "proof_bytes", "setup_size"]


@dataclass(frozen=True)
class OpShape:
    name: str
    n_rows: int
    row_length: int
    layout: str = "linear"       # "embed" for the lookup tables (as ``MatOp.layout``)
    input: str | None = None     # the tensor it reads (ops reading one tensor may share a col matrix)
    has_bias: bool = False       # as ``MatOp.has_bias`` (``row_length`` counts its column)

    @property
    def inputs(self) -> tuple[str | None]:
        """As ``MatOp.inputs`` (``None``: unknown, never shared)."""
        return (self.input,)

    def n_points(self, rate: int) -> int:
        return rate * next_pow2(self.row_length)


def decoder_shapes(cfg, n_layers: int | None = None, prune_last: bool = False) -> list[OpShape]:
    """Weight-op shapes of a decoder (see ``transformer.build_decoder``) without building it
    (``prune_last``: its last block's q reads the last position alone, k and v every position)."""
    L = cfg.n_layers if n_layers is None else n_layers
    d, dh, hq, hkv, f = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff
    b = 1 if cfg.bias else 0
    e = getattr(cfg, "embed_dim", 0) or d

    def block(name, rows, k, src):       # a projection of a block, with the config's bias
        return OpShape(name, rows, k + b, input=src, has_bias=bool(b))

    ops = [OpShape("embed", e, cfg.vocab, "embed", "tokens")]
    if e != d:
        ops.append(OpShape("proj_in", d, e, input="embedded"))
    if cfg.pos == "learned":
        ops.append(OpShape("pos", d, cfg.max_pos, "embed", "positions"))
    for i in range(L):
        a, m = f"attn_in{i}", f"mlp_in{i}"
        aq = f"attn_last{i}" if prune_last and i == L - 1 else a
        ops += [block(f"q{i}", hq * dh, d, aq), block(f"k{i}", hkv * dh, d, a), block(f"v{i}", hkv * dh, d, a),
                block(f"o{i}", d, hq * dh, f"attn_out{i}")]
        if cfg.mlp == "swiglu":
            ops += [block(f"gate{i}", f, d, m), block(f"up{i}", f, d, m), block(f"down{i}", d, f, f"mlp_act{i}")]
        else:
            ops += [block(f"fc1{i}", f, d, m), block(f"fc2{i}", d, f, f"mlp_act{i}")]
    if e != d:
        ops.append(OpShape("proj_out", e, d, input="final"))
    ops.append(OpShape("head", cfg.vocab, e, input="final" if e == d else "final_projected"))
    return ops


def decoder_claim_columns(cfg, seq: int, n_layers: int | None = None, prune_last: bool = False) -> dict[str, int]:
    """Each weight op's claim columns for a prompt of ``seq`` tokens, keyed as :func:`decoder_shapes`:
    ``seq``, but 1 for the LM head and OPT-350M's project_out, which read the last position only, and
    with ``prune_last`` for every op of the last block but k and v."""
    last = (cfg.n_layers if n_layers is None else n_layers) - 1
    pruned = {f"{p}{last}" for p in ("q", "o", "gate", "up", "down", "fc1", "fc2")} if prune_last else set()
    return {s.name: 1 if s.name in ("proj_out", "head") or s.name in pruned else seq
            for s in decoder_shapes(cfg, n_layers, prune_last)}


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


def expected_lookup_nodes(v: int, m: int) -> float:
    """Expected hashes in the multiproof of ``m`` distinct uniform rows of a lookup table of ``v`` rows,
    whose tree pads them with leaves ``v .. next_pow2(v) - 1`` that are never opened.

    At a level whose nodes cover ``s`` leaves, each pair of siblings covers ``2 s`` leaves, ``a`` and
    ``b`` of them rows; exactly one of the two covers an opened row with probability ``[C(v-b, m) +
    C(v-a, m) - 2 C(v-a-b, m)] / C(v, m)``: ``floor(v / 2s)`` pairs of ``s`` and ``s`` rows, at most one
    pair with fewer, and none past the rows (:func:`expected_multiproof_nodes` for ``v = n``)."""
    n, m = next_pow2(v), min(m, v)
    base = _log_comb(v, m)

    def exactly_one(a: int, b: int) -> float:
        return (math.exp(_log_comb(v - b, m) - base) + math.exp(_log_comb(v - a, m) - base)
                - 2 * math.exp(_log_comb(v - a - b, m) - base))

    total, s = 0.0, 1
    while s < n:
        full, rest = divmod(v, 2 * s)
        total += full * exactly_one(s, s) + (exactly_one(min(s, rest), max(0, rest - s)) if rest else 0.0)
        s *= 2
    return total


def _matrices(ops, plan, rate: int) -> list[tuple[str, int, int]]:
    """``(name, rows, codeword length)`` of every encoded matrix of ``ops``: each op's own (the
    report), or the plan's matrices of ``ops``."""
    if plan is None:
        return [(op.name, op.n_rows, rate * next_pow2(op.row_length)) for op in ops]
    names = {op.name for op in ops}
    return [(m.name, m.n_rows, m.n_points) for m in plan.coded if m.ops[0] in names]


def _tables(ops, plan) -> list:
    """The plan's lookup tables of ``ops``."""
    names = {op.name for op in ops}
    return [] if plan is None else [m for m in plan.tables if m.name in names]


def _trees(ops, plan, rate: int) -> list[tuple[str, int, int]]:
    """``(name, codeword length, rows)`` of every Merkle tree holding one of ``ops``."""
    if plan is None:
        return [(name, n, rows) for name, rows, n in _matrices(ops, None, rate)]
    rows = {name: r for name, r, _ in _matrices(ops, plan, rate)}
    return [(g, plan.matrix(members[0]).n_points, sum(rows[m] for m in members if m in rows))
            for g, members in plan.groups if any(m in rows for m in members)]


def proof_bytes(ops, params, claim_columns, *, plan=None, rate: int = 4, mode: str = "C",
                wire_claims: int | None = None, lookups: bool = False,
                table_ids: dict | None = None) -> dict[str, float]:
    """One query's proof bytes as ``run_query`` counts them: the claims, ``u`` and the opened columns
    exactly, and the Merkle multiproofs' expected size (the column indices are random).  ``ops``:
    weight-op shapes (``decoder_shapes`` or a graph's ``MatOp``s), whose claims have
    ``claim_columns`` columns (an int for all of them, e.g. the prompt length of a decoder, or
    ``{name: columns}``); ``plan``: a commitment plan (its groups restricted to ``ops``), else the
    report's trees at ``rate``; under a plan ``u`` is that of its row-layout ops, and a tree opens
    the rows of its matrices (a col-layout one's: ``k``).  A lookup table's claims (mode C, under a
    plan) are its looked-up rows, a byte per entry, and its multiproof counts with the paths: exact
    for the ids ``table_ids[name]`` it looks up, else the expectation for distinct uniform ids
    (:func:`expected_lookup_nodes`).  Modes K and Kpre send only the claims, and with ``lookups`` none
    for the embedding ops (the verifier reads their rows).  ``wire_claims``: the size of the claims
    in the compact wire encoding (``PVC3``, which depends on their values: measured, with a plan's
    table rows as int8 bytes), for the proof of ``run_query(wire=True)``, whose ``u`` and opened
    columns travel as two runs of 31-bit field elements (:func:`claimcodec.field_size`)."""
    if lookups and mode == "C":
        raise ValueError("lookups are modes K and Kpre's; mode C looks rows up in a plan's tables")

    def cols(op):
        return claim_columns if isinstance(claim_columns, int) else claim_columns[op.name]

    tables = _tables(ops, plan) if mode == "C" else []
    table_names = {m.name for m in tables}
    sent = [op for op in ops if not (lookups and op.layout == "embed")]
    size = (lambda numel: 4 * numel) if wire_claims is None else field_size
    claims = sum((1 if op.name in table_names else 4) * op.n_rows * cols(op) for op in sent) \
        if wire_claims is None else wire_claims
    out = {"claims": claims, "u": 0, "columns": 0, "paths": 0.0}
    if mode == "C":
        trees = [(n, min(params.columns if plan is None else params.columns_for(g), n), rows)
                 for g, n, rows in _trees(ops, plan, rate)]
        folded = [op for op in ops if plan is None or plan.matrix_of(op.name).layout == "row"]
        of = {op.name: op for op in ops}
        looked_up = [multiproof_size(set(table_ids[m.name]), m.n_points.bit_length() - 1)
                     if table_ids is not None and m.name in table_ids else
                     expected_lookup_nodes(m.row_length, cols(of[m.name])) for m in tables]
        out["u"] = size(params.reps * sum(op.row_length for op in folded))
        out["columns"] = size(sum(t * rows for _, t, rows in trees))
        out["paths"] = 32 * sum(expected_multiproof_nodes(n, t) for n, t, _ in trees) + 32 * sum(looked_up)
    return out


def setup_size(ops, *, plan=None, rate: int = 4) -> dict[str, int]:
    """The one-time commitment of ``ops``: encoded field entries (``sum_l N_l n_l``, what the NTTs
    produce and the column digests hash), Merkle leaves and trees (a plan's lookup tables' too, whose
    rows are hashed as they are) and the longest codeword."""
    trees, tables = _trees(ops, plan, rate), _tables(ops, plan)
    return {"encoded_entries": sum(rows * n for _, rows, n in _matrices(ops, plan, rate)),
            "leaves": sum(n for _, n, _ in trees) + sum(m.n_points for m in tables),
            "trees": len(trees) + len(tables), "max_n": max(n for _, n, _ in trees)}
