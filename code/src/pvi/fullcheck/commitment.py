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
paper's ``C_M``), so the committer is honest and no proximity test is needed
(report, Theorem 3.2).

The prover does not keep the encoded matrix.  An opened column is recomputed
from the weights with one small matrix product (``A @ V[:, C]`` for the
Vandermonde columns ``C``), which costs ``t * N * k`` multiply-adds per matrix and
avoids storing ``rate`` times the model.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch

from .field import (_LIMB_MAX, NP_SMALL, P, exact_chunk, field_matmul_mod, np_field_matmul_mod, power_table,
                    root_of_unity, rs_encode, small_matmul_mod, to_field)

__all__ = [
    "HASH_BYTES",
    "MerkleTree",
    "multiproof",
    "multiproof_size",
    "verify_multiproof",
    "verify_multiproofs",
    "column_leaf",
    "column_rows",
    "row_leaves",
    "map_threaded",
    "CommitmentPublic",
    "WeightCommitment",
    "column_leaves",
    "group_leaf",
    "GroupPublic",
    "GroupCommitment",
    "vandermonde_columns",
    "codeword_at",
]

HASH_BYTES = 32
MERKLE_PROCESSES = os.environ.get("PVI_MERKLE_PROCESSES") == "1"
"""Opt-in (``PVI_MERKLE_PROCESSES=1``): a multi-threaded verifier checks its multiproofs in
worker processes (:func:`verify_multiproofs`).  Off by default: the workers are started by a
fork server (Linux, macOS) or spawned (Windows), never forked from the verifier itself, which
runs threads (torch's, the leaf hashing's, the streaming verifier's host thread) and could
deadlock a forked child; so the calling script needs an ``if __name__ == "__main__"`` guard
(both start methods import it), and extra processes are a deployment choice."""
MERKLE_PROCESSES_MIN_HASHES = 6000
"""Fewer sibling hashes than this in one check are not worth the inter-process traffic."""


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"pvi/node" + left + right).digest()


def column_leaf(tag: bytes, index: int, column: np.ndarray) -> bytes:
    """Leaf digest of one encoded column (little-endian uint32 field elements)."""
    h = hashlib.sha256(b"pvi/col" + tag + int(index).to_bytes(8, "big"))
    h.update(np.ascontiguousarray(column, dtype="<u4").tobytes())
    return h.digest()


def column_rows(opened: torch.Tensor) -> np.ndarray:
    """The int64 columns ``opened`` ``[N, t]`` as the 32-bit rows ``[t, N]`` that :func:`column_leaf`
    hashes (the low 32 bits of each value): one transposing cast for all columns instead of a
    strided gather and a cast per column."""
    if opened.shape[0] <= 1 << 14:
        rows = opened.cpu().numpy().T.astype(np.int32, order="C")
    else:                                                            # tall columns: torch transposes faster
        rows = opened.to(torch.int32).T.contiguous().cpu().numpy()
    if sys.byteorder != "little":  # pragma: no cover - column_leaf hashes little-endian words
        rows = rows.astype("<u4")
    return rows


def row_leaves(tag: bytes, indices: list[int], rows: np.ndarray) -> dict[int, bytes]:
    """``{c: column_leaf(tag, c, column)}`` for the columns given as :func:`column_rows` ``[t, N]``."""
    prefix = b"pvi/col" + tag
    out = {}
    for j, c in enumerate(indices):
        h = hashlib.sha256(prefix + int(c).to_bytes(8, "big"))
        h.update(rows[j])
        out[c] = h.digest()
    return out


LEAF_THREAD_BYTES = 1 << 14
"""Columns of at least this many bytes are hashed on threads (:func:`map_threaded`): ``hashlib``
releases the GIL while it hashes more than 2 KiB, but below ~16 KiB per column the threads'
overhead outweighs the gain (laptop: 4 threads are 1.6x faster at 16 KiB, 2.9x at 125 KiB,
and slower at 12 KiB)."""
_THREAD_POOLS: dict = {}


def map_threaded(fn, items: list, sizes: list[int], workers: int) -> list:
    """``[fn(x) for x in items]``: the items of ``sizes >= LEAF_THREAD_BYTES`` on a pool of
    ``workers`` threads, the others meanwhile on the calling thread."""
    big = [i for i, s in enumerate(sizes) if s >= LEAF_THREAD_BYTES] if workers > 1 else []
    if not big:
        return [fn(x) for x in items]
    if workers not in _THREAD_POOLS:
        from concurrent.futures import ThreadPoolExecutor
        _THREAD_POOLS[workers] = ThreadPoolExecutor(max_workers=workers)
    futures = {i: _THREAD_POOLS[workers].submit(fn, items[i]) for i in big}
    out = [None if i in futures else fn(x) for i, x in enumerate(items)]
    for i, fut in futures.items():
        out[i] = fut.result()
    return out


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


def multiproof_size(indices, depth: int) -> int:
    """Number of hashes a multiproof for ``indices`` needs (no tree required)."""
    known = sorted(set(int(i) for i in indices))
    total = 0
    for _ in range(depth):
        kset = set(known)
        total += sum(1 for i in known if (i ^ 1) not in kset)
        known = sorted({i >> 1 for i in known})
    return total


def multiproof(tree: MerkleTree, indices) -> list[bytes]:
    """One authentication set for several leaves: every sibling hash the verifier
    cannot compute itself, each sent once, level by level in index order."""
    known = sorted(set(int(i) for i in indices))
    proof = []
    for level in tree.levels[:-1]:
        kset = set(known)
        proof += [level[i ^ 1] for i in known if (i ^ 1) not in kset]
        known = sorted({i >> 1 for i in known})
    return proof


def verify_multiproof(root: bytes, depth: int, leaves: dict[int, bytes], proof: list[bytes]) -> bool:
    """Rebuild the root from ``{index: leaf digest}`` and a :func:`multiproof`."""
    known = dict(leaves)
    it = iter(proof)
    try:
        for _ in range(depth):
            nxt: dict[int, bytes] = {}
            for i in sorted(known):
                if i % 2 == 0:
                    right = known[i + 1] if (i + 1) in known else next(it)
                    nxt[i >> 1] = _node(known[i], right)
                elif (i - 1) not in known:
                    nxt[i >> 1] = _node(next(it), known[i])
            known = nxt
    except StopIteration:
        return False
    return next(it, None) is None and list(known.items()) == [(0, root)]


def _verify_jobs(jobs: list) -> list[tuple[bool, Exception | None]]:
    out = []
    for job in jobs:
        try:
            out.append((verify_multiproof(*job), None))
        except Exception as exc:   # e.g. a proof entry that is not bytes: the caller raises it in op order
            out.append((False, exc))
    return out


_PROCESS_POOLS: dict = {}


def _process_pool(workers: int):
    """The worker processes of :func:`verify_multiproofs` (one pool per worker count), started
    as :data:`MERKLE_PROCESSES` says."""
    if workers not in _PROCESS_POOLS:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        _PROCESS_POOLS[workers] = ProcessPoolExecutor(max_workers=workers,
                                                      mp_context=multiprocessing.get_context(method))
    return _PROCESS_POOLS[workers]


def verify_multiproofs(jobs: list, workers: int = 1) -> list[tuple[bool, Exception | None]]:
    """``(verify_multiproof(*job), exception or None)`` for every ``(root, depth, leaves,
    proof)`` job, in order.

    SHA-256 of a 72-byte node holds the GIL, so threads cannot share this work.  With
    :data:`MERKLE_PROCESSES`, ``workers > 1`` and enough hashes, the jobs go to ``workers``
    worker processes in batches balanced by proof length.  A batch that does not reach a
    worker or come back (a proof entry that cannot be pickled, a pool that broke) is checked
    in this process instead, so every result is the same function of the same job.
    """
    if (not MERKLE_PROCESSES or workers <= 1 or len(jobs) < 2
            or sum(len(job[3]) for job in jobs) < MERKLE_PROCESSES_MIN_HASHES):
        return _verify_jobs(jobs)
    from concurrent.futures.process import BrokenProcessPool

    batches, load = [[] for _ in range(workers)], [0] * workers
    for i in sorted(range(len(jobs)), key=lambda i: -len(jobs[i][3])):   # longest first, to the least loaded
        w = load.index(min(load))
        batches[w].append(i)
        load[w] += len(jobs[i][2]) + len(jobs[i][3])
    batches = [b for b in batches if b]
    pool, broken = _process_pool(workers), False
    futures = []
    for b in batches:
        try:
            futures.append(pool.submit(_verify_jobs, [jobs[i] for i in b]))
        except RuntimeError:            # the pool broke meanwhile (a BrokenProcessPool) or was shut down
            futures.append(None)
            broken = True
    out: list = [None] * len(jobs)
    for b, fut in zip(batches, futures):
        results = None
        if fut is not None:
            try:
                results = fut.result()
            except Exception as exc:    # a job that cannot be pickled, or a worker that died
                broken = broken or isinstance(exc, BrokenProcessPool)
        if results is None:             # checked here (_verify_jobs itself raises nothing)
            results = _verify_jobs([jobs[i] for i in b])
        for i, res in zip(b, results):
            out[i] = res
    if broken:
        _PROCESS_POOLS.pop(workers, None)     # the next call starts new workers
    return out


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
    return table[(j * c) & (n_points - 1)]    # n_points is a power of two and j * c >= 0


def codeword_at(u: torch.Tensor, n_points: int, columns: torch.Tensor) -> torch.Tensor:
    """``Enc(u)[..., columns]``, the same field elements as ``field_matmul_mod(u,
    vandermonde_columns(n_points, k, columns))``.

    ``u`` is ``[r, k]``, or ``[G, r, k]`` with ``columns`` ``[G, t]`` (G matrices at
    once).  Baby-step giant-step: with ``j = a S + b`` (``b < S ~ sqrt(k)``, ``a < G' =
    ceil(k / S)``), ``Enc(u)[c] = sum_a w**(a S c) sum_b u[a S + b] w**(b c)``.  The inner
    sums are one limb product with an ``S x t`` table of powers, the outer sum an
    elementwise product with a ``G' x t`` table: ``(S + G') t`` table entries instead of
    the ``k t`` of the Vandermonde block (69x fewer for GPT-2's embedding).
    """
    r, k = u.shape[-2], u.shape[-1]
    s = 1 << (max(k, 1).bit_length() + 1) // 2          # a power of two >= sqrt(k)
    g = -(-k // s)
    if (u.device.type == "cpu" and u.dtype == torch.int64 and columns.device.type == "cpu"
            and 3 * u.numel() + (s + g) * columns.numel() <= NP_SMALL):
        return torch.from_numpy(_np_codeword_at(u.numpy(), n_points, columns.to(torch.int64).numpy(), s, g))
    dev = u.device
    table = power_table(root_of_unity(n_points), n_points, dev)
    c = columns.to(device=dev, dtype=torch.int64).unsqueeze(-2)                  # [..., 1, t]
    steps = torch.arange(max(s, g), dtype=torch.int64, device=dev)[:, None]
    powers = table[torch.cat([steps[:s] * c, steps[:g] * (s * c)], -2) & (n_points - 1)]   # [..., S + G', t]
    baby, giant = powers[..., :s, :], powers[..., s:, :]
    u = torch.nn.functional.pad(u, (0, g * s - k))
    inner = field_matmul_mod(u.reshape(*u.shape[:-2], r * g, s), baby).reshape(*u.shape[:-2], r, g, -1)
    inner *= giant.unsqueeze(-3)                  # products of two field elements < 2**62
    inner %= P
    return inner.sum(-2) % P                      # G' < 2**32 terms below 2**31


def _np_codeword_at(u: np.ndarray, n_points: int, columns: np.ndarray, s: int, g: int) -> np.ndarray:
    """:func:`codeword_at` in numpy, for small CPU operands: the same integer steps."""
    r, k = u.shape[-2], u.shape[-1]
    table = power_table(root_of_unity(n_points), n_points).numpy()
    c = columns[..., None, :]
    steps = np.arange(max(s, g), dtype=np.int64)[:, None]
    powers = table[np.concatenate([steps[:s] * c, steps[:g] * (s * c)], -2) & (n_points - 1)]
    baby, giant = powers[..., :s, :], powers[..., s:, :]
    u = np.concatenate([u, np.zeros((*u.shape[:-1], g * s - k), dtype=np.int64)], -1)
    inner = np_field_matmul_mod(u.reshape(*u.shape[:-2], r * g, s), baby, True, exact_chunk(_LIMB_MAX, P - 1))
    inner = inner.reshape(*u.shape[:-2], r, g, -1) * giant[..., None, :, :]
    return np.remainder(inner, P).sum(-2) % P


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


def _encoded_rows(weight: torch.Tensor, bias: torch.Tensor | None, n_points: int, device, row_chunk: int):
    """``(r0, E[r0:r0 + c])``: the codeword rows of ``[W | b]`` as host int32 arrays ``[c, n_points]``,
    a few rows at a time (the NTT needs ~4x its input in scratch)."""
    row_chunk = max(1, min(row_chunk, (1 << 25) // n_points))
    for r0 in range(0, weight.shape[0], row_chunk):
        rows = weight[r0:r0 + row_chunk].to(device=device, dtype=torch.int64)
        if bias is not None:
            rows = torch.cat([rows, bias[r0:r0 + row_chunk].to(device=device, dtype=torch.int64)[:, None]], 1)
        yield r0, rs_encode(rows, n_points).to(torch.int32).cpu().numpy()
        del rows


def column_leaves(tag: bytes, weight: torch.Tensor, bias: torch.Tensor | None, n_points: int, *,
                  device: torch.device | str = "cpu", row_chunk: int = 256,
                  max_host_bytes: int | None = None) -> list[bytes]:
    """The :func:`column_leaf` digest of every column of ``Enc([W | b])`` (length ``n_points``).

    The codeword is gathered column-major on the host (``4 N n_points`` bytes) and each column
    hashed once -- unless that exceeds ``max_host_bytes``: then every column keeps a running
    SHA-256 that is fed one block of at most ``max_host_bytes`` of codeword rows at a time, so a
    long codeword of a tall matrix (a vocabulary-sized LM head at a high rate) needs no buffer of
    its whole encoding.  Both hash the same bytes in the same order: the same digests."""
    n_rows = weight.shape[0]
    if max_host_bytes is None or 4 * n_rows * n_points <= max_host_bytes:
        cols = np.empty((n_points, n_rows), dtype="<u4")
        for r0, block in _encoded_rows(weight, bias, n_points, device, row_chunk):
            cols[:, r0:r0 + block.shape[0]] = block.T
        return [column_leaf(tag, c, cols[c]) for c in range(n_points)]
    hashes = [hashlib.sha256(b"pvi/col" + tag + int(c).to_bytes(8, "big")) for c in range(n_points)]
    block_rows = max(1, max_host_bytes // (4 * n_points))
    pending: list[np.ndarray] = []

    def flush() -> None:
        cols = np.ascontiguousarray(np.concatenate(pending).T, dtype="<u4")     # [n_points, rows]
        pending.clear()
        for h, col in zip(hashes, cols):
            h.update(col)

    for _, block in _encoded_rows(weight, bias, n_points, device, row_chunk):
        pending.append(block)
        if sum(b.shape[0] for b in pending) >= block_rows:
            flush()
    if pending:
        flush()
    return [h.digest() for h in hashes]


@dataclass
class WeightCommitment:
    """Prover-side commitment to ``A = [W | b]`` (``W`` int8 ``[N, K]``, ``b`` int64 ``[N]``).

    ``tree`` is ``None`` for a member of a :class:`GroupCommitment`, whose tree binds its columns."""

    tag: bytes
    weight: torch.Tensor
    bias: torch.Tensor | None
    rate: int
    tree: MerkleTree | None = field(repr=False)
    n_points: int

    @classmethod
    def build(cls, tag: bytes, weight: torch.Tensor, bias: torch.Tensor | None, *,
              rate: int = 4, device: torch.device | str = "cpu",
              row_chunk: int = 256) -> "WeightCommitment":
        weight = weight.to(torch.int8).cpu()
        n_points = rate * _next_pow2(weight.shape[1] + (1 if bias is not None else 0))
        tree = MerkleTree(column_leaves(tag, weight, bias, n_points, device=device, row_chunk=row_chunk))
        return cls(tag=tag, weight=weight, bias=None if bias is None else bias.cpu().to(torch.int64),
                   rate=rate, tree=tree, n_points=n_points)

    @classmethod
    def member(cls, tag: bytes, weight: torch.Tensor, bias: torch.Tensor | None, n_points: int) -> "WeightCommitment":
        """A matrix encoded at ``n_points`` whose columns a group's tree binds (no tree of its own)."""
        weight = weight.to(torch.int8).cpu()
        rate = n_points // _next_pow2(weight.shape[1] + (1 if bias is not None else 0))
        return cls(tag=tag, weight=weight, bias=None if bias is None else bias.cpu().to(torch.int64),
                   rate=rate, tree=None, n_points=n_points)

    @property
    def public(self) -> CommitmentPublic:
        """The verifier's view (the root is ``b""`` for a group member: its group holds the root)."""
        return CommitmentPublic(self.tag, b"" if self.tree is None else self.tree.root, self.weight.shape[0],
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

    def columns_at(self, v: torch.Tensor, device: torch.device | str = "cpu",
                   weight: torch.Tensor | None = None) -> torch.Tensor:
        """``E[:, C]`` on ``device`` from the Vandermonde columns ``v = V[:, C]`` (at least
        ``row_length`` rows; row ``j`` is the same for every matrix of this codeword length)."""
        w_all = self.weight if weight is None else weight
        k = w_all.shape[1]
        cols = torch.cat([small_matmul_mod(w_all[r0:r0 + 16384].to(device), v[:k])
                          for r0 in range(0, w_all.shape[0], 16384)], 0)
        if self.bias is not None:
            cols = (cols + (to_field(self.bias.to(device))[:, None] * v[k][None, :]) % P) % P
        return cols

    def open(self, columns: torch.Tensor, device: torch.device | str = "cpu",
             weight: torch.Tensor | None = None):
        """Recompute encoded columns ``E[:, columns]`` and one Merkle multiproof for them."""
        v = vandermonde_columns(self.n_points, self.row_length, columns, device)
        return self.columns_at(v, device, weight).cpu(), multiproof(self.tree, columns.tolist())


# -- shared trees ---------------------------------------------------------------------------

def group_leaf(tag: bytes, index: int, member_digests) -> bytes:
    """Leaf ``index`` of a group's tree: the :func:`column_leaf` digests of column ``index`` of
    every member, in the group's (public) member order.  The digests are 32 bytes each and the
    members are public, so for a group the leaf input has one fixed length."""
    h = hashlib.sha256(b"pvi/group" + tag + int(index).to_bytes(8, "big"))
    for d in member_digests:
        h.update(d)
    return h.digest()


@dataclass(frozen=True)
class GroupPublic:
    """What a verifier holds for one group of a commitment plan: the root of the tree over all its
    members' columns and the member op names, in leaf order (their shapes are their own
    :class:`CommitmentPublic`, whose root is empty)."""

    tag: bytes
    root: bytes
    n_points: int
    members: tuple[str, ...]

    @property
    def depth(self) -> int:
        return int(math.log2(self.n_points))


@dataclass
class GroupCommitment:
    """Several matrices of one codeword length under ONE Merkle tree: leaf ``c`` is the
    :func:`group_leaf` of their columns ``c``.  Member ``name`` is committed under the tag
    ``name.encode()``, as a matrix on its own tree is, so its column digests are those of its own
    commitment.  One set of column indices opens every member, with ONE multiproof."""

    tag: bytes
    members: dict[str, WeightCommitment]     # in leaf order
    n_points: int
    tree: MerkleTree = field(repr=False)

    @classmethod
    def build(cls, tag: bytes, matrices: dict[str, tuple[torch.Tensor, torch.Tensor | None]], n_points: int, *,
              device: torch.device | str = "cpu", max_host_bytes: int = 1 << 28) -> "GroupCommitment":
        """``matrices``: ``{name: (W, b)}`` in leaf order.  Each member's column digests are fed
        into ``n_points`` running leaf hashes as soon as they are computed, so no member's
        encoding outlives its own digests (and a tall one is hashed in blocks of
        ``max_host_bytes``, see :func:`column_leaves`)."""
        leaves = [hashlib.sha256(b"pvi/group" + tag + int(c).to_bytes(8, "big")) for c in range(n_points)]
        members = {}
        for name, (w, b) in matrices.items():
            m = WeightCommitment.member(name.encode(), w, b, n_points)
            for h, d in zip(leaves, column_leaves(m.tag, m.weight, m.bias, n_points, device=device,
                                                  max_host_bytes=max_host_bytes)):
                h.update(d)
            members[name] = m
        return cls(tag=tag, members=members, n_points=n_points, tree=MerkleTree([h.digest() for h in leaves]))

    @property
    def public(self) -> GroupPublic:
        return GroupPublic(self.tag, self.tree.root, self.n_points, tuple(self.members))

    def open(self, columns: torch.Tensor, device: torch.device | str = "cpu", weights: list | None = None):
        """Every member's ``E[:, columns]``, stacked in member order (``[sum N, t]``), and one multiproof;
        ``weights`` may hold device-resident copies of the members' int8 weights, in member order."""
        v = vandermonde_columns(self.n_points, max(m.row_length for m in self.members.values()), columns, device)
        cols = [m.columns_at(v, device, None if weights is None else weights[i])
                for i, m in enumerate(self.members.values())]
        return torch.cat(cols, 0).cpu(), multiproof(self.tree, columns.tolist())
