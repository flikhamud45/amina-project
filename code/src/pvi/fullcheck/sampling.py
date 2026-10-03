"""Anchuri et al.'s RandPathTest on the same integer CNN graphs (the baseline).

This lets the original protocol and our defence be measured on identical
models, inputs and hardware.  It follows Figure 3 of the paper and its neuron
model ``a_j = phi(sum_{i in G_j} w_ij a_i)``:

* a *neuron* is an output of an activation-level operation: a requantised
  (ReLU) layer output, a residual ``add + ReLU`` output, a pooling output, or a
  final logit.  Pre-activations are never committed -- as in the paper's
  Section 8 "hybrid" scheme, the check recomputes them from the parents;
* the prover Merkle-commits the trace, one leaf per *row* (the channel vector at
  one spatial position, or a whole vector for dense layers), as in Section 8;
  flatten is a view, not a separate tensor;
* the verifier holds a commitment ``C_M`` to the weights (one leaf per output
  row of every weight matrix);
* one path runs from a uniformly random output node back to the input.  At each
  node the prover opens the node's row, its parents' rows and the node's weight
  row(s); the verifier checks that one node and continues to a parent chosen
  uniformly from ``G_j``.  For a residual neuron ``G_j`` is the union of the
  main branch's receptive field and the shortcut's parents.

``visit_probabilities`` computes, exactly, the probability that one path visits
each neuron.  A single tampered neuron (honestly re-propagated above it, which
is the attack) fails only its own check, so that probability *is* the per-path
detection probability; ``tests/test_fullcheck.py`` checks it by Monte Carlo.
``shared_path_bytes`` measures the cost of ``k`` paths *with openings shared*
between paths (a row or weight row is sent once per proof, however many paths
touch it), which is capped by opening everything.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .commitment import HASH_BYTES, MerkleTree, multiproof, multiproof_size, verify_multiproof
from .graph import CheapOp, IntGraph, MatOp

__all__ = ["TraceCommitment", "sample_path", "shared_path_bytes", "visit_probabilities",
           "paths_for", "neuron_tensors"]


def _pad_pow2(leaves: list[bytes]) -> list[bytes]:
    n = 1 << max(0, (len(leaves) - 1).bit_length())
    return leaves + [hashlib.sha256(b"pvi/pad" + i.to_bytes(8, "big")).digest() for i in range(len(leaves), n)]


def _rows(t: torch.Tensor) -> np.ndarray:
    """``[1, C, H, W]`` -> ``[H*W, C]``; ``[1, N]`` -> ``[1, N]``."""
    if t.dim() == 4:
        return t[0].reshape(t.shape[1], -1).T.contiguous().numpy()
    return t.reshape(1, -1).numpy()


def _leaf(tag: bytes, i: int, row: np.ndarray, width: int) -> bytes:
    dtype = "<i4" if width == 4 else "i1"
    return hashlib.sha256(b"pvi/trace" + tag + i.to_bytes(8, "big") + row.astype(dtype).tobytes()).digest()


def _wleaf(name: str, i: int, w: np.ndarray, b: int) -> bytes:
    return hashlib.sha256(b"pvi/w" + name.encode() + i.to_bytes(8, "big") + w.astype("i1").tobytes()
                          + np.int64(b).tobytes()).digest()


def _producers(graph: IntGraph) -> dict[str, object]:
    return {op.output: op for op in graph.ops}


def _base(prod: dict, name: str) -> str:
    """Follow flatten views back to the tensor that holds the values."""
    while isinstance(prod.get(name), CheapOp) and prod[name].note == "flatten":
        name = prod[name].inputs[0]
    return name


def neuron_tensors(graph: IntGraph) -> list[str]:
    """The committed tensors: every neuron layer (and the final logits)."""
    out = []
    for op in graph.ops:
        if isinstance(op, MatOp):
            if op.output == graph.output_name:
                out.append(op.output)
        elif op.note != "flatten":
            out.append(op.output)
    return out


@dataclass
class _Committed:
    rows: np.ndarray
    width: int
    tree: MerkleTree


class TraceCommitment:
    """The prover's commitment to one execution trace plus the weight commitment ``C_M``."""

    def __init__(self, graph: IntGraph, env: dict[str, torch.Tensor]) -> None:
        self.graph = graph
        self.prod = _producers(graph)
        self.tensors: dict[str, _Committed] = {}
        for name in neuron_tensors(graph):
            rows = _rows(env[name].cpu())
            w = 4 if name == graph.output_name else 1
            tree = MerkleTree(_pad_pow2([_leaf(name.encode(), i, r, w) for i, r in enumerate(rows)]))
            self.tensors[name] = _Committed(rows, w, tree)
        self.weights: dict[str, tuple[np.ndarray, np.ndarray, MerkleTree]] = {}
        for op in graph.mat_ops:
            w = op.weight.numpy().astype(np.int64)
            b = op.bias.numpy() if op.bias is not None else np.zeros(w.shape[0], dtype=np.int64)
            self.weights[op.name] = (w, b, MerkleTree(_pad_pow2([_wleaf(op.name, i, w[i], b[i])
                                                                  for i in range(w.shape[0])])))
        # opening every leaf of a tree needs no sibling hashes at all
        self.open_all_bytes = (sum(c.rows.size * c.width for c in self.tensors.values())
                               + sum(w.shape[0] * (w.shape[1] + 8) for w, _, _ in self.weights.values()))

    def merkle_bytes(self, rows: dict, wrows: dict) -> int:
        """Multiproof size for the opened rows of every trace and weight tree."""
        n = sum(multiproof_size(r, self.tensors[k].tree.depth) for k, r in rows.items() if r)
        n += sum(multiproof_size(r, self.weights[k][2].depth) for k, r in wrows.items() if r)
        return n * HASH_BYTES


class _Walk:
    """Opens values for one path (or several, sharing the ``seen`` caches).

    Data bytes are counted as rows are opened; when verifying, the Merkle
    authentication for everything a path opened is one multiproof per tree,
    built and checked by :meth:`finish`.
    """

    def __init__(self, tc: TraceCommitment, env: dict, verify: bool, seen_rows: set, seen_w: set) -> None:
        self.tc, self.env, self.verify = tc, env, verify
        self.seen_rows, self.seen_w = seen_rows, seen_w
        self.input_rows = _rows(env[tc.graph.input_name].cpu())
        self.bytes = 0
        self.ok = True
        self.new_rows: dict[str, dict[int, bytes]] = {}
        self.new_w: dict[str, dict[int, bytes]] = {}

    def row(self, tensor: str, r: int) -> np.ndarray:
        if tensor == self.tc.graph.input_name:     # the verifier's own query: nothing to open
            return self.input_rows[r]
        c = self.tc.tensors[tensor]
        key = (tensor, r)
        if key not in self.seen_rows:
            self.seen_rows.add(key)
            vals = c.rows[r]
            self.bytes += vals.size * c.width
            if self.verify:
                self.new_rows.setdefault(tensor, {})[r] = _leaf(tensor.encode(), r, vals, c.width)
        return c.rows[r]

    def weight(self, op: MatOp, r: int) -> tuple[np.ndarray, int]:
        w, b, tree = self.tc.weights[op.name]
        key = (op.name, r)
        if key not in self.seen_w:
            self.seen_w.add(key)
            self.bytes += w.shape[1] + 8
            if self.verify:
                self.new_w.setdefault(op.name, {})[r] = _wleaf(op.name, r, w[r], b[r])
        return w[r], int(b[r])

    def finish(self) -> None:
        """Multiproofs for this path's newly opened rows: build (prover), check (verifier)."""
        if not self.verify:
            return
        for trees, opened in ((({k: self.tc.tensors[k].tree for k in self.new_rows}), self.new_rows),
                              (({k: self.tc.weights[k][2] for k in self.new_w}), self.new_w)):
            for k, leaves in opened.items():
                tree = trees[k]
                proof = multiproof(tree, list(leaves))
                self.ok &= verify_multiproof(tree.root, tree.depth, leaves, proof)

    def value(self, tensor: str, flat: int) -> int:
        base = _base(self.tc.prod, tensor)
        shape = self.env[base].shape
        if len(shape) == 4:
            hw = shape[2] * shape[3]
            return int(self.row(base, flat % hw)[flat // hw])
        return int(self.row(base, 0)[flat])

    def preact(self, op: MatOp, c: int, y: int, x: int):
        """Recompute ``z = W[c] . patch + b[c]``; return ``(z, parent_sampler, n_parents)``."""
        wrow, brow = self.weight(op, c)
        src = op.inputs[0]
        base = _base(self.tc.prod, src)
        if op.layout == "conv":
            k, s, p = op.conv
            _, ci, h, w_ = self.env[src].shape
            taps = [(dy, dx, y * s - p + dy, x * s - p + dx) for dy in range(k) for dx in range(k)
                    if 0 <= y * s - p + dy < h and 0 <= x * s - p + dx < w_]
            patch = np.zeros((ci, k, k), dtype=np.int64)
            for dy, dx, yy, xx in taps:
                patch[:, dy, dx] = self.row(base, yy * w_ + xx)
            z = int(wrow @ patch.reshape(-1)) + brow

            def sampler(rng, taps=taps, ci=ci, h=h, w_=w_, base=base):
                _, _, yy, xx = taps[int(rng.integers(len(taps)))]
                return base, int(rng.integers(ci)) * h * w_ + yy * w_ + xx

            return z, sampler, ci * len(taps)
        shape = self.env[base].shape
        if len(shape) == 4:
            hw = shape[2] * shape[3]
            vec = np.stack([self.row(base, r_) for r_ in range(hw)]).T.reshape(-1)  # torch flatten order
        else:
            vec = self.row(base, 0).astype(np.int64)
        z = int(wrow @ vec) + brow
        n = vec.size
        return z, (lambda rng, base=base, n=n: (base, int(rng.integers(n)))), n


def _one_hot_apply(fn, shape, c: int, *vals: int) -> int:
    """Apply a per-channel elementwise op to single values placed at channel ``c``."""
    if len(shape) == 4:
        args = [torch.zeros(1, shape[1], 1, 1, dtype=torch.int64) for _ in vals]
        for a, v in zip(args, vals):
            a[0, c, 0, 0] = v
        return int(fn(*args)[0, c, 0, 0])
    args = [torch.zeros(1, shape[1], dtype=torch.int64) for _ in vals]
    for a, v in zip(args, vals):
        a[0, c] = v
    return int(fn(*args)[0, c])


def sample_path(tc: TraceCommitment, env: dict[str, torch.Tensor], rng: np.random.Generator, *,
                verify: bool = True, seen_rows: set | None = None, seen_w: set | None = None) -> dict:
    """Open one random path.  Returns ``ok`` and the data bytes it newly opened."""
    graph, prod = tc.graph, tc.prod
    wk = _Walk(tc, env, verify, set() if seen_rows is None else seen_rows, set() if seen_w is None else seen_w)
    name = graph.output_name
    flat = int(rng.integers(env[name].numel()))
    while name != graph.input_name:
        op = prod[name]
        shape = tuple(env[name].shape)
        if len(shape) == 4:
            hw = shape[2] * shape[3]
            c, pos = divmod(flat, hw)
            y, x = divmod(pos, shape[3])
        else:
            c, y, x = flat, 0, 0
        claimed = wk.value(name, flat)
        if isinstance(op, MatOp):                       # the final logits
            z, sampler, _ = wk.preact(op, c, y, x)
            wk.ok &= z == claimed
            name, flat = sampler(rng)
        elif op.note.startswith("requant"):             # a = relu(requant(W a' + b))
            z, sampler, _ = wk.preact(prod[op.inputs[0]], c, y, x)
            wk.ok &= _one_hot_apply(op.fn, shape, c, z) == claimed
            name, flat = sampler(rng)
        elif op.note == "residual add+relu":            # a = relu(W2 h + b2 + shortcut)
            z_main, s_main, n_main = wk.preact(prod[op.inputs[0]], c, y, x)
            sc = op.inputs[1]
            if isinstance(prod.get(sc), MatOp):         # 1x1 downsample convolution
                z_sc, s_sc, n_sc = wk.preact(prod[sc], c, y, x)
            else:                                       # identity shortcut: one parent
                z_sc, n_sc = wk.value(sc, flat), 1
                s_sc = lambda rng, sc=sc, flat=flat: (_base(prod, sc), flat)
            wk.ok &= _one_hot_apply(op.fn, shape, c, z_main, z_sc) == claimed
            name, flat = (s_main if rng.integers(n_main + n_sc) < n_main else s_sc)(rng)
        elif op.note in ("maxpool", "global avgpool"):
            src = op.inputs[0]
            _, ci, h, w_ = env[src].shape
            if op.note == "global avgpool":
                window = [(yy, xx) for yy in range(h) for xx in range(w_)]
            else:
                k, s, p = op.params["k"], op.params["s"], op.params["p"]
                window = [(y * s - p + dy, x * s - p + dx) for dy in range(k) for dx in range(k)
                          if 0 <= y * s - p + dy < h and 0 <= x * s - p + dx < w_]
            base = _base(prod, src)
            vals = np.array([int(wk.row(base, yy * w_ + xx)[c]) for yy, xx in window])
            if op.note == "global avgpool":
                n = h * w_
                wk.ok &= (int(vals.sum()) + n // 2) // n == claimed
            else:
                wk.ok &= int(vals.max()) == claimed
            yy, xx = window[int(rng.integers(len(window)))]
            name, flat = base, c * h * w_ + yy * w_ + xx
        else:
            raise ValueError(f"no path rule for {op.name} ({op.note})")
        name = _base(prod, name)
    wk.finish()
    return {"ok": bool(wk.ok), "data_bytes": wk.bytes}


def shared_path_bytes(tc: TraceCommitment, env: dict, checkpoints: list[int], seed: int = 1) -> dict[int, int]:
    """Bytes of ``k`` paths sent as one proof, per ``k``: every row and weight row
    once, plus one Merkle multiproof per tree for everything opened.

    Stops early once every committed row and weight row has been opened: from
    there on the proof is the whole trace and model, ``tc.open_all_bytes``.
    """
    rng = np.random.default_rng(seed)
    seen_rows: set = set()
    seen_w: set = set()
    total_rows = sum(c.rows.shape[0] for c in tc.tensors.values())
    total_w = sum(w.shape[0] for w, _, _ in tc.weights.values())
    out: dict[int, int] = {}
    data = done = 0

    def union_bytes() -> int:
        rows: dict[str, list] = {}
        for k, r in seen_rows:
            rows.setdefault(k, []).append(r)
        wrows: dict[str, list] = {}
        for k, r in seen_w:
            wrows.setdefault(k, []).append(r)
        return data + tc.merkle_bytes(rows, wrows)

    for k in sorted(checkpoints):
        while done < k and (len(seen_rows) < total_rows or len(seen_w) < total_w):
            # only the data part is additive; the Merkle part is recomputed on the union
            data += sample_path(tc, env, rng, verify=False, seen_rows=seen_rows, seen_w=seen_w)["data_bytes"]
            done += 1
        saturated = len(seen_rows) >= total_rows and len(seen_w) >= total_w
        out[k] = tc.open_all_bytes if saturated else union_bytes()
    return out


def visit_probabilities(graph: IntGraph, env: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exact probability that one random path visits each node, per committed tensor."""
    prod = _producers(graph)
    mass = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in env.items()}
    out = env[graph.output_name]
    mass[graph.output_name] = torch.full_like(out, 1.0 / out.numel(), dtype=torch.float64)

    def add(name: str, m: torch.Tensor) -> None:
        base = _base(prod, name)
        mass[base] += m.reshape(mass[base].shape)

    def spread(op: MatOp, m_over_n: torch.Tensor) -> None:
        """Every parent in ``op``'s receptive field gets ``m / n_parents(position)``."""
        xin = env[op.inputs[0]]
        if op.layout == "conv":
            k, s, p = op.conv
            h, w = xin.shape[2], xin.shape[3]
            ho, wo = m_over_n.shape[2], m_over_n.shape[3]
            pad_out = (h - ((ho - 1) * s - 2 * p + k), w - ((wo - 1) * s - 2 * p + k))
            ker = torch.ones(m_over_n.shape[1], xin.shape[1], k, k, dtype=torch.float64)
            add(op.inputs[0], F.conv_transpose2d(m_over_n, ker, stride=s, padding=p, output_padding=pad_out))
        else:
            add(op.inputs[0], torch.full(xin.shape, float(m_over_n.sum()), dtype=torch.float64))

    def n_parents(op: MatOp) -> torch.Tensor:
        """Receptive-field size per output position (``[1,1,Ho,Wo]`` for conv, scalar otherwise)."""
        xin = env[op.inputs[0]]
        if op.layout == "conv":
            k, s, p = op.conv
            ones = torch.ones(1, xin.shape[1], xin.shape[2], xin.shape[3], dtype=torch.float64)
            return F.conv2d(ones, torch.ones(1, xin.shape[1], k, k, dtype=torch.float64), stride=s, padding=p)
        return torch.tensor(float(op.n_in), dtype=torch.float64)

    for name in reversed(neuron_tensors(graph)):
        op = prod[name]
        m = mass[name]
        if isinstance(op, MatOp):
            spread(op, m / n_parents(op))
        elif op.note.startswith("requant"):
            z_op = prod[op.inputs[0]]
            spread(z_op, m / n_parents(z_op))
        elif op.note == "residual add+relu":
            main = prod[op.inputs[0]]
            sc = op.inputs[1]
            n_main = n_parents(main)
            sc_op = prod.get(sc)
            n_sc = n_parents(sc_op) if isinstance(sc_op, MatOp) else torch.tensor(1.0, dtype=torch.float64)
            share = m / (n_main + n_sc)
            spread(main, share)
            if isinstance(sc_op, MatOp):
                spread(sc_op, share)
            else:
                add(sc, share)
        elif op.note == "maxpool":
            kk, s, p = op.params["k"], op.params["s"], op.params["p"]
            x = torch.zeros_like(env[op.inputs[0]], dtype=torch.float64, requires_grad=True)
            F.avg_pool2d(x, kk, s, p, count_include_pad=False).backward(m)
            add(op.inputs[0], x.grad)
        elif op.note == "global avgpool":
            x = env[op.inputs[0]]
            add(op.inputs[0], (m / (x.shape[2] * x.shape[3]))[:, :, None, None].expand_as(x))
        else:
            raise ValueError(f"no path rule for {op.name} ({op.note})")
    return mass


def paths_for(lam: float, p_detect: float) -> int:
    """Paths needed so a single-node tamper escapes with probability ``<= 2**-lam``."""
    if p_detect >= 1:
        return 1
    return math.ceil(lam / -math.log2(1.0 - p_detect))
