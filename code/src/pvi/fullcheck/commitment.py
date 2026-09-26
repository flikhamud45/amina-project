"""Merkle trees and the Ligero-style weight commitment.

The weight commitment lets a verifier who holds only a 32-byte digest of a
weight matrix ``A`` (with the bias as its last column) learn ``u = chi^T A`` for
a challenge ``chi`` of its choice, without ever seeing ``A``:

1. The committer Reed--Solomon encodes every row of ``A`` (length ``k``) into a
   codeword of length ``n = rate * next_pow2(k)`` and Merkle-commits the
   *columns* of the encoded matrix ``E``.
2. At query time the prover sends ``u``.  The verifier re-encodes ``u`` itself
   and spot-checks ``t`` random columns: each must satisfy
   ``<chi, E[:, c]> == Enc(u)[c]``.

Encoding is linear, so an honest ``u`` passes every column.  A wrong ``u``
gives a codeword that differs from ``Enc(chi^T A)`` in at least ``n - k + 1``
positions, so each random column catches it with probability at least
``(n - k + 1) / n``; ``t`` distinct columns leave it at most ``((k-1)/n)**t``.

As in the base protocol, the committed model is the agreed ground truth (the
paper's ``C_M``), so the committer is honest and no proximity test is needed;
see ``DEFENCE_NOTES.md``.

The prover does not keep the encoded matrix.  An opened column is recomputed
from the weights with one small matrix product (``A @ V[:, C]`` for the
Vandermonde columns ``C``), which costs ``t * N * k`` multiply-adds per matrix and
avoids storing ``rate`` times the model.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .field import P, field_matmul_mod, power_table, root_of_unity, rs_encode, small_matmul_mod, to_field

__all__ = [
    "HASH_BYTES",
    "MerkleTree",
    "verify_path",
    "column_leaf",
    "CommitmentPublic",
    "WeightCommitment",
    "vandermonde_columns",
]

HASH_BYTES = 32


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"pvi/node" + left + right).digest()


def column_leaf(tag: bytes, index: int, column: np.ndarray) -> bytes:
    """Leaf digest of one encoded column (little-endian uint32 field elements)."""
    h = hashlib.sha256(b"pvi/col" + tag + int(index).to_bytes(8, "big"))
    h.update(np.ascontiguousarray(column, dtype="<u4").tobytes())
    return h.digest()


class MerkleTree:
    """A binary SHA-256 Merkle tree over a power-of-two number of leaf digests."""

    def __init__(self, leaves: list[bytes]) -> None:
        n = len(leaves)
        if n < 1 or n & (n - 1):
            raise ValueError(f"Merkle tree needs a power-of-two leaf count, got {n}")
        self.levels = [list(leaves)]
        while len(self.levels[-1]) > 1:
            prev = self.levels[-1]
            self.levels.append([_node(prev[i], prev[i + 1]) for i in range(0, len(prev), 2)])

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def path(self, index: int) -> list[bytes]:
        out = []
        for level in self.levels[:-1]:
            out.append(level[index ^ 1])
            index >>= 1
        return out


def verify_path(root: bytes, index: int, leaf: bytes, path: list[bytes]) -> bool:
    node = leaf
    for sibling in path:
        node = _node(node, sibling) if index % 2 == 0 else _node(sibling, node)
        index >>= 1
    return node == root


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def vandermonde_columns(n_points: int, row_length: int, columns: torch.Tensor,
                        device: torch.device | str = "cpu") -> torch.Tensor:
    """``V[j, c] = w**(j*c)`` for ``j < row_length`` and the requested columns.

    ``Enc(row)[c] = sum_j row[j] * V[j, c]`` with ``w`` the primitive
    ``n_points``-th root of unity used by :func:`rs_encode`.
    """
    table = power_table(root_of_unity(n_points), n_points, device)
    j = torch.arange(row_length, dtype=torch.int64, device=device)[:, None]
    c = columns.to(device=device, dtype=torch.int64)[None, :]
    return table[(j * c) % n_points]


@dataclass(frozen=True)
class CommitmentPublic:
    """What a verifier holds for one committed matrix: no weights at all."""

    tag: bytes
    root: bytes
    n_rows: int
    row_length: int
    n_points: int

    @property
    def depth(self) -> int:
        return int(math.log2(self.n_points))


@dataclass
class WeightCommitment:
    """Prover-side commitment to ``A = [W | b]`` (``W`` int8 ``[N, K]``, ``b`` int64 ``[N]``)."""

    tag: bytes
    weight: torch.Tensor
    bias: torch.Tensor | None
    rate: int
    tree: MerkleTree = field(repr=False)
    n_points: int
    commit_seconds: float

    @classmethod
    def build(cls, tag: bytes, weight: torch.Tensor, bias: torch.Tensor | None, *,
              rate: int = 4, device: torch.device | str = "cpu",
              row_chunk: int = 256) -> "WeightCommitment":
        start = time.perf_counter()
        weight = weight.to(torch.int8).cpu()
        n_rows, k = weight.shape
        row_length = k + (1 if bias is not None else 0)
        n_points = rate * _next_pow2(row_length)
        # Encode a few rows at a time (the NTT needs ~4x its input in scratch),
        # gather the codeword column-major on the host, then hash each column once.
        row_chunk = max(1, min(row_chunk, (1 << 25) // n_points))
        cols = np.empty((n_points, n_rows), dtype="<u4")
        for r0 in range(0, n_rows, row_chunk):
            rows = weight[r0:r0 + row_chunk].to(device=device, dtype=torch.int64)
            if bias is not None:
                rows = torch.cat([rows, bias[r0:r0 + row_chunk].to(device=device, dtype=torch.int64)[:, None]], 1)
            cols[:, r0:r0 + rows.shape[0]] = rs_encode(rows, n_points).to(torch.int32).cpu().numpy().T
            del rows
        tree = MerkleTree([hashlib.sha256(b"pvi/col" + tag + c.to_bytes(8, "big") + cols[c].tobytes()).digest()
                           for c in range(n_points)])
        del cols
        return cls(tag=tag, weight=weight, bias=None if bias is None else bias.cpu().to(torch.int64),
                   rate=rate, tree=tree, n_points=n_points,
                   commit_seconds=time.perf_counter() - start)

    @property
    def public(self) -> CommitmentPublic:
        return CommitmentPublic(self.tag, self.tree.root, self.weight.shape[0],
                                self.row_length, self.n_points)

    @property
    def row_length(self) -> int:
        return self.weight.shape[1] + (1 if self.bias is not None else 0)

    def fold(self, chi: torch.Tensor, device: torch.device | str = "cpu", row_chunk: int = 8192,
             weight: torch.Tensor | None = None) -> torch.Tensor:
        """``u = chi @ A mod P`` for challenge rows ``chi`` of shape ``[r, N]``.

        ``weight`` may be a copy of the (same) int8 weights already resident on
        ``device`` -- the prover's forward pass keeps one -- to avoid moving the
        matrix across the bus on every query.
        """
        w_all = self.weight if weight is None else weight
        chi = chi.to(device)
        u_w = None
        for r0 in range(0, w_all.shape[0], row_chunk):
            w = w_all[r0:r0 + row_chunk].to(device)
            part = small_matmul_mod(w.T, chi[:, r0:r0 + row_chunk].T.contiguous()).T  # [r, K]
            u_w = part if u_w is None else (u_w + part) % P
        if self.bias is None:
            return u_w.cpu()
        b = to_field(self.bias.to(device))
        u_b = field_matmul_mod(chi, b[:, None])  # [r, 1]; chi * b needs the limb product
        return torch.cat([u_w, u_b], 1).cpu()

    def open(self, columns: torch.Tensor, device: torch.device | str = "cpu",
             weight: torch.Tensor | None = None):
        """Recompute encoded columns ``E[:, columns]`` and their Merkle paths."""
        w_all = self.weight if weight is None else weight
        k = w_all.shape[1]
        v = vandermonde_columns(self.n_points, self.row_length, columns, device)
        cols = torch.cat([small_matmul_mod(w_all[r0:r0 + 16384].to(device), v[:k])
                          for r0 in range(0, w_all.shape[0], 16384)], 0)
        if self.bias is not None:
            cols = (cols + (to_field(self.bias.to(device))[:, None] * v[k][None, :]) % P) % P
        paths = [self.tree.path(int(c)) for c in columns.tolist()]
        return cols.cpu(), paths
