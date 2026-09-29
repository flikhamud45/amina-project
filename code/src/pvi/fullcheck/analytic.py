"""Decoder weight-op shapes and the expected size of a Merkle multiproof.

* ``decoder_shapes`` lists the weight ops of a decoder (see
  ``transformer.build_decoder``) without building it: the benchmark sizes
  ``(r, t)`` for the full model's op count, and ``aggregate.py`` needs the
  codeword length of every committed matrix;
* ``expected_multiproof_nodes`` is the exact expected number of hashes in one
  multiproof for ``t`` distinct uniform columns, which ``aggregate.py`` uses as
  the Merkle term of LLM runs made before multiproofs (only the earliest run,
  ``raw/``, which the report does not use; checked by Monte Carlo in
  ``tests/test_fullcheck.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["OpShape", "decoder_shapes", "expected_multiproof_nodes"]


@dataclass(frozen=True)
class OpShape:
    name: str
    n_rows: int
    row_length: int

    def n_points(self, rate: int) -> int:
        return rate * (1 << max(0, (self.row_length - 1).bit_length()))


def decoder_shapes(cfg, n_layers: int | None = None) -> list[OpShape]:
    """Weight-op shapes of a decoder (see ``transformer.build_decoder``) without building it."""
    L = cfg.n_layers if n_layers is None else n_layers
    d, dh, hq, hkv, f = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff
    b = 1 if cfg.bias else 0
    e = getattr(cfg, "embed_dim", 0) or d
    ops = [OpShape("embed", e, cfg.vocab)]
    if e != d:
        ops.append(OpShape("proj_in", d, e))
    if cfg.pos == "learned":
        ops.append(OpShape("pos", d, cfg.max_pos))
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
