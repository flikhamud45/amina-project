"""A prover-side cache of the weight commitment's Merkle leaves.

The one-time setup of mode C encodes every row of every weight matrix and hashes every column of every codeword:
``protocol.commit_graph`` spends almost all its time computing the leaf digests of the commitment's trees (a
group's leaf ``c`` is the :func:`commitment.group_leaf` of its members' column digests ``c``; a matrix on a tree of
its own has the :func:`commitment.column_leaf` digests; a lookup table the :func:`commitment.table_leaf` digests of
its rows). The digests depend only on the committed integers (weights and biases), the tags and the plan's layout,
so a later process that commits the same matrices under the same plan can read them back and rebuild every tree
(``MerkleTree(leaves)``, a fraction of a second per tree) instead of encoding the model again. Nothing that
leaves the prover changes: the roots, the openings and every proof byte are those of a fresh commitment
(``tests/test_setup_cache.py``).

* **Opt-in.** The folder comes from ``commit_graph(cache=...)`` (in the scripts, ``--setup-cache DIR``); ``None``,
  the default, means no cache (the setup proof's timing never uses one).
* **Key.** A BLAKE2b-256 over a format version, the base rate, the plan's policy and layout (groups, their member
  matrices in leaf order, each matrix's layout, members and codeword length, the lookup tables) and, for every
  weight op in the graph's order, its name (the tag), shapes, and a BLAKE2b-256 of its int8 weights and int64 bias.
  Different weights, biases, calibration, pruning or plan give a different key.
* **Files.** ``<folder>/<key>.npz``: one ``uint8 [n, 32]`` array per tree and the tree names as JSON, saved with
  ``numpy.savez`` (no pickle) to a temporary name and renamed into place. A Llama-2-13B commitment under cnn18c is
  about 51 MB.
* **Check on reading.** Before trusting a file, :func:`load` recomputes ``checks`` random leaves of every tree from
  the weights (the encoded columns, exactly as an opening computes them, and their digests) and compares them; any
  mismatch, a missing tree or an unreadable file makes it a miss and the commitment is computed afresh (and the file
  rewritten). A wrong leaf could not make a verifier accept anything (it recomputes the leaves of the opened columns
  from their values and checks them against the root), it would only make honest proofs fail; the check keeps that
  from happening silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

VERSION = b"pvi/setup-cache/v1"
"""Bump when a leaf definition (``column_leaf``, ``group_leaf``, ``table_leaf``, the encoding) changes."""


def folder(cache) -> Path | None:
    """The cache folder for ``commit_graph(cache=...)``: ``None``, ``False`` or ``""`` -> none; a path -> that
    folder."""
    return Path(cache) if cache else None


def _tensor_digest(t: torch.Tensor | None, dtype) -> bytes:
    if t is None:
        return b"none"
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype(dtype, copy=False))
    h = hashlib.blake2b(digest_size=32)
    h.update(str(a.shape).encode())
    h.update(memoryview(a).cast("B"))
    return h.digest()


def key(graph, rate: int, plan) -> str:
    """The cache key (hex) of committing ``graph``'s weight ops at base rate ``rate`` under ``plan`` (``None``: the
    report's layout, one tree per op)."""
    h = hashlib.blake2b(digest_size=32)
    h.update(VERSION + b"|rate=%d|" % rate)
    if plan is None:
        h.update(b"plan=paper")
    else:
        h.update(f"plan={plan.policy}".encode())
        for g, members in plan.groups:
            h.update(f"|group={g}:{','.join(members)}".encode())
        for m in plan.coded + plan.tables:
            h.update(f"|matrix={m.name}:{m.layout}:{m.n_rows}:{m.row_length}:{m.n_points}:{','.join(m.members)}"
                     .encode())
    for op in graph.mat_ops:
        h.update(f"|op={op.name}:{tuple(op.weight.shape)}".encode())
        h.update(_tensor_digest(op.weight, np.int8))
        h.update(_tensor_digest(op.bias, np.int64))
    return h.hexdigest()


def save(where: Path, k: str, trees: dict[str, list[bytes]]) -> Path:
    """Write ``trees`` (``{tree name: its leaf digests}``) as ``<where>/<k>.npz``, atomically."""
    where.mkdir(parents=True, exist_ok=True)
    names = list(trees)
    arrays = {f"t{i}": np.frombuffer(b"".join(trees[n]), dtype=np.uint8).reshape(-1, 32) for i, n in enumerate(names)}
    meta = json.dumps({"key": k, "version": VERSION.decode(), "names": names})
    fd, tmp = tempfile.mkstemp(prefix=f".{k}.", suffix=".npz", dir=where)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, meta=np.array(meta), **arrays)
        os.replace(tmp, where / f"{k}.npz")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return where / f"{k}.npz"


def read(where: Path, k: str) -> dict[str, list[bytes]] | None:
    """``{tree name: leaf digests}`` from ``<where>/<k>.npz``, or ``None`` if it is missing or unreadable."""
    path = where / f"{k}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("key") != k or meta.get("version") != VERSION.decode():
                return None
            out = {}
            for i, name in enumerate(meta["names"]):
                a = z[f"t{i}"]
                if a.ndim != 2 or a.shape[1] != 32:
                    return None
                out[name] = [bytes(row) for row in a]
            return out
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):   # (JSONDecodeError is a ValueError)
        return None


def _sample(n: int, checks: int) -> list[int]:
    return sorted({secrets.randbelow(n) for _ in range(min(checks, n))})


def check_group(tag: bytes, members: dict, n_points: int, leaves: list[bytes], device, checks: int) -> bool:
    """Recompute ``checks`` random leaves of a group (its members' encoded columns and digests) and compare."""
    from pvi.commit.commitment import column_rows, group_leaf, row_leaves, vandermonde_columns
    if len(leaves) != n_points:
        return False
    idx = _sample(n_points, checks)
    v = vandermonde_columns(n_points, max(m.row_length for m in members.values()), torch.tensor(idx), device)
    digests = [row_leaves(m.tag, idx, column_rows(m.columns_at(v, device))) for m in members.values()]
    return all(group_leaf(tag, c, [d[c] for d in digests]) == leaves[c] for c in idx)


def check_matrix(com, leaves: list[bytes], device, checks: int) -> bool:
    """The same for a matrix on a tree of its own (its column digests)."""
    from pvi.commit.commitment import column_rows, row_leaves, vandermonde_columns
    if len(leaves) != com.n_points:
        return False
    idx = _sample(com.n_points, checks)
    v = vandermonde_columns(com.n_points, com.row_length, torch.tensor(idx), device)
    d = row_leaves(com.tag, idx, column_rows(com.columns_at(v, device)))
    return all(d[c] == leaves[c] for c in idx)


def check_table(tag: bytes, weight: torch.Tensor, leaves: list[bytes], checks: int) -> bool:
    """The same for a lookup table (its rows' digests; the padding leaves are checked too)."""
    from pvi.commit.commitment import padding_leaf, table_leaf
    rows = weight.to(torch.int8).T.contiguous().cpu().numpy()
    v = rows.shape[0]
    if len(leaves) < v or len(leaves) & (len(leaves) - 1):
        return False
    for j in _sample(len(leaves), checks):
        want = table_leaf(tag, j, rows[j]) if j < v else padding_leaf(tag, j)
        if want != leaves[j]:
            return False
    return True
