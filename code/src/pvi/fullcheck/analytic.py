"""Closed-form cost and error model of the whole-network check.

Everything is a function of the weight-op shapes only.  For weight op ``l``
let ``N_l`` be its output rows, ``k_l`` its row length (inputs + bias),
``M_l`` the number of columns of its claimed output (spatial positions or
tokens, times the batch), ``n_l = rate * next_pow2(k_l)`` the codeword length,
``r`` the Freivalds repetitions and ``t`` the opened columns.

Per query (mode C):

* proof bytes   = sum_l 4 N_l M_l            (claimed pre-activations)
                + sum_l 4 r k_l              (folded rows ``u_l``)
                + sum_l 4 t N_l              (opened columns)
                + sum_l 32 E[multiproof_l]   (one Merkle multiproof per op, <= t log2 n_l)
* prover work   = forward pass + r sum_l N_l k_l (fold) + t sum_l N_l k_l (columns)
* verifier work = r sum_l (N_l + k_l) M_l  (Freivalds)  + r t sum_l k_l (codeword at the t columns)
                + r t sum_l N_l (column checks) + cheap ops (recomputed)
* soundness     <= sum_l [ p^-r + ((k_l - 1)/n_l)^t ]

Mode K drops the ``u``, column and path terms (the verifier folds itself);
mode Kpre also drops the verifier's fold (precomputed).

``tests/test_fullcheck.py`` checks the byte formula against the protocol's
measured proof sizes exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .field import LOG2_P
from .protocol import SecurityParams

__all__ = ["OpShape", "graph_shapes", "decoder_shapes", "proof_bytes", "work", "soundness_error_bits",
           "sampling_paths", "expected_multiproof_nodes"]


@dataclass(frozen=True)
class OpShape:
    name: str
    n_rows: int
    row_length: int
    n_cols: int
    gather: bool = False
    """An embedding lookup: committed like any matrix, but computed as a gather."""

    def n_points(self, rate: int) -> int:
        return rate * (1 << max(0, (self.row_length - 1).bit_length()))


def graph_shapes(graph, x) -> list[OpShape]:
    """Shapes of every weight op of an :class:`IntGraph` on input ``x``."""
    env, _ = graph.forward(x)
    out = []
    for op in graph.mat_ops:
        out.append(OpShape(op.name, op.n_rows, op.row_length, op.n_cols(env[op.inputs[0]]),
                           gather=op.layout == "embed"))
    return out


def decoder_shapes(cfg, seq: int, n_layers: int | None = None, batch: int = 1, all_logits: bool = False) -> list[OpShape]:
    """Weight-op shapes of a decoder (see ``transformer.build_decoder``) without building it."""
    L = cfg.n_layers if n_layers is None else n_layers
    d, dh, hq, hkv, f = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff
    b = 1 if cfg.bias else 0
    m = seq * batch
    ops = [OpShape("embed", d, cfg.vocab, m, gather=True)]
    if cfg.pos == "learned":
        ops.append(OpShape("pos", d, cfg.max_pos, m, gather=True))
    for i in range(L):
        ops += [OpShape(f"q{i}", hq * dh, d + b, m), OpShape(f"k{i}", hkv * dh, d + b, m),
                OpShape(f"v{i}", hkv * dh, d + b, m), OpShape(f"o{i}", d, hq * dh + b, m)]
        if cfg.mlp == "swiglu":
            ops += [OpShape(f"gate{i}", f, d + b, m), OpShape(f"up{i}", f, d + b, m),
                    OpShape(f"down{i}", d, f + b, m)]
        else:
            ops += [OpShape(f"fc1{i}", f, d + b, m), OpShape(f"fc2{i}", d, f + b, m)]
    ops.append(OpShape("head", cfg.vocab, d, m if all_logits else batch))
    return ops


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


def proof_bytes(shapes: list[OpShape], p: SecurityParams, mode: str = "C") -> dict[str, int]:
    claims = sum(4 * s.n_rows * s.n_cols for s in shapes)
    if mode != "C":
        return {"claims": claims, "u": 0, "columns": 0, "paths": 0}
    u = sum(4 * p.reps * s.row_length for s in shapes)
    cols = sum(4 * min(p.columns, s.n_points(p.rate)) * s.n_rows for s in shapes)
    paths = round(sum(32 * expected_multiproof_nodes(s.n_points(p.rate), p.columns) for s in shapes))
    return {"claims": claims, "u": u, "columns": cols, "paths": paths}


def work(shapes: list[OpShape], p: SecurityParams, mode: str = "C") -> dict[str, float]:
    """Multiply-add counts (field or integer) for each party, weight ops only."""
    k_eff = lambda s: 1 if s.gather else s.row_length  # embeddings are gathers  # noqa: E731
    fwd = sum(s.n_rows * k_eff(s) * s.n_cols for s in shapes)
    fold = p.reps * sum(s.n_rows * s.row_length for s in shapes)
    cols = p.columns * sum(s.n_rows * s.row_length for s in shapes)
    freivalds = p.reps * sum((s.n_rows + k_eff(s)) * s.n_cols for s in shapes)
    reenc = p.reps * sum(s.row_length * min(p.columns, s.n_points(p.rate)) for s in shapes)
    colchk = p.reps * p.columns * sum(s.n_rows for s in shapes)
    out = {"prover_forward": fwd, "verifier_freivalds": freivalds}
    if mode == "C":
        out.update(prover_fold=fold, prover_open=cols, verifier_reencode=reenc, verifier_columns=colchk)
    elif mode == "K":
        out.update(verifier_fold=fold)
    return out


def soundness_error_bits(shapes: list[OpShape], p: SecurityParams, mode: str = "C") -> float:
    total = 0.0
    for s in shapes:
        err = 2.0 ** (-p.reps * LOG2_P)
        if mode == "C":
            n, k = s.n_points(p.rate), s.row_length
            t = min(p.columns, n)
            err += 2.0 ** sum(math.log2(max(k - 1 - i, 1e-300) / (n - i)) for i in range(t)) if k > 1 else 0.0
        total += err
    bits = -math.log2(total)
    return bits - (p.grinding_bits if p.fiat_shamir else 0)


def sampling_paths(lam: float, width: int) -> int:
    """Paths the sampling protocol needs against a single-node tamper at a layer of this width."""
    return math.ceil(lam / -math.log2(1.0 - 1.0 / width))
