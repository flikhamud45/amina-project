"""Integer computation graphs: the exact semantics both parties agree on.

A model is a list of operations over named integer tensors (int64 storage,
small values).  Two kinds matter to the protocol:

* ``MatOp`` -- a product with a committed weight matrix: dense layers,
  convolutions, embeddings and the LM head.  These are all of the model's heavy
  compute.  The prover *claims* their outputs (int32 pre-activations) and the
  verifier checks every one of them with a random linear combination.
* cheap operations -- requantisation, ReLU, pooling, residual additions,
  normalisation, softmax, look-up-table activations, RoPE and attention.  The
  verifier *recomputes* these itself from values it has already checked, so
  they need no argument at all.

Every operation is deterministic integer arithmetic, so the prover's GPU and
the verifier's CPU produce bit-identical results (tested).  Matrix products
run as float32 GEMMs over int8-valued operands with the contraction dimension
chunked so every partial sum stays below ``2**24`` and is therefore exact.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import types
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F

__all__ = [
    "Op",
    "MatOp",
    "CheapOp",
    "IntGraph",
    "exact_matmul",
    "int_scalar",
    "mul_add_half",
    "requant",
    "INT8_MAX",
]

INT8_MAX = 127
# PVI_LEGACY_WEIGHT_KEY=1 restores the weight cache of the earliest run (raw/): its key was 'cuda' for the
# prover and 'cuda:0' for the tensors, so every committed-weights query re-uploaded the model twice.
# Only for the _nofix controls of bench.py, which measure what the fix changed.
_LEGACY_WEIGHT_KEY = os.environ.get("PVI_LEGACY_WEIGHT_KEY") == "1"
_FP32_EXACT = 1 << 24


def exact_matmul(w: torch.Tensor, x: torch.Tensor, *, max_w: int = 128, max_x: int = 128) -> torch.Tensor:
    """Exact (batched) ``w @ x`` for small-integer operands via chunked float32 GEMMs."""
    k = w.shape[-1]
    chunk = max(1, _FP32_EXACT // (max_w * max_x))
    out = None
    for s in range(0, k, chunk):
        part = (w[..., s:s + chunk].to(torch.float32) @ x[..., s:s + chunk, :].to(torch.float32))
        part = part.to(torch.int64)
        out = part if out is None else out + part
    return out


@lru_cache(maxsize=None)
def int_scalar(value: int, device: str) -> torch.Tensor:
    """A 0-d int64 tensor on ``device`` (cached; callers only read it)."""
    return torch.tensor(value, dtype=torch.int64, device=device)


def mul_add_half(z: torch.Tensor, mult: torch.Tensor, shift: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """``z * mult + 2**(shift-1)``, fresh or written into ``out``: one ``addcmul`` pass for
    int64 operands (the same int64 values as the multiply and the add)."""
    if z.dtype == torch.int64 and torch.is_tensor(mult) and mult.dtype == torch.int64:
        return torch.addcmul(int_scalar(1 << (shift - 1), str(z.device)), z, mult, out=out)
    if out is None:
        return z * mult + (1 << (shift - 1))
    torch.mul(z, mult, out=out)
    out += 1 << (shift - 1)
    return out


def requant(z: torch.Tensor, mult: torch.Tensor, shift: int, lo: int, hi: int,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """``clamp(round(z * mult / 2**shift), lo, hi)`` with round-half-up, exactly: fresh, or
    written into ``out``.

    The shift and the clamp run in place on the product."""
    out = mul_add_half(z, mult, shift, out=out)
    out >>= shift
    return out.clamp_(lo, hi)


@dataclass
class Op:
    name: str
    inputs: tuple[str, ...]
    output: str

    def public(self) -> "Op":
        return self


@dataclass
class MatOp(Op):
    """``z = W @ unfold(x) + b`` with ``W`` int8 ``[N, K]`` and ``b`` int64 ``[N]``.

    ``layout`` is ``"linear"`` (``x[..., K]``), ``"conv"`` (``x[B, C, H, W]``, with
    ``conv = (kernel, stride, padding)``) or ``"embed"`` (``x`` holds token ids and
    ``W`` is the transposed embedding table ``[d, vocab]``).
    """

    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    layout: str = "linear"
    conv: tuple[int, int, int] | None = None
    n_rows: int = 0
    n_in: int = 0
    has_bias: bool = False
    max_input: int = INT8_MAX
    _dev: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.weight is not None:
            self.n_rows, self.n_in = self.weight.shape
            self.has_bias = self.bias is not None

    @property
    def row_length(self) -> int:
        return self.n_in + (1 if self.has_bias else 0)

    def public(self) -> "MatOp":
        clone = copy.copy(self)
        clone.weight = None
        clone.bias = None
        clone._dev = {}
        return clone

    # -- shapes ---------------------------------------------------------------
    def unfold(self, x: torch.Tensor, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        """Input tensor -> ``X`` of shape ``[K, M]``; token ids for embeddings.

        The values are small integers, so float32 (used on the prover's GPU)
        represents them exactly.
        """
        if self.layout == "conv":
            k, s, p = self.conv
            cols = F.unfold(x.to(torch.float32), k, padding=p, stride=s)  # [B, K, L]
            return cols.permute(1, 0, 2).reshape(cols.shape[1], -1).to(dtype)
        if self.layout == "embed":
            return x.reshape(-1).to(torch.int64)
        return x.reshape(-1, x.shape[-1]).T.to(dtype)

    def n_cols(self, x: torch.Tensor) -> int | None:
        """Number of columns ``M`` of ``X`` for input ``x``, or ``None`` if ``x`` is malformed."""
        if self.layout == "conv":
            k, s, p = self.conv
            if x.dim() != 4 or x.shape[1] * k * k != self.n_in:
                return None
            ho = (x.shape[2] + 2 * p - k) // s + 1
            wo = (x.shape[3] + 2 * p - k) // s + 1
            return x.shape[0] * ho * wo
        if self.layout == "embed":
            return x.numel()
        if x.shape[-1] != self.n_in:
            return None
        return x.numel() // self.n_in

    def fold(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Claimed ``Z [N, M]`` -> output tensor shaped like the op's output.

        Linear layers and embeddings return a transposed *view* of ``z``: their readers
        are elementwise (requantisation, residual additions), so a transposing copy would
        be one more pass for the same values.  Nothing modifies a fold output in place.
        """
        if self.layout == "conv":
            k, s, p = self.conv
            b, _, h, w = x.shape
            ho = (h + 2 * p - k) // s + 1
            wo = (w + 2 * p - k) // s + 1
            return z.reshape(self.n_rows, b, ho, wo).permute(1, 0, 2, 3).contiguous()
        if self.layout == "embed":
            return z.T.reshape(*x.shape, self.n_rows)
        return z.T.reshape(*x.shape[:-1], self.n_rows)

    # -- prover ---------------------------------------------------------------
    def _weights_on(self, device) -> tuple[torch.Tensor, torch.Tensor | None]:
        # One cache key per physical device: torch.device("cuda") (the Prover's) and a
        # tensor's cuda:0 are the same GPU, and a mismatch re-uploads every weight matrix.
        device = torch.device(device)
        if _LEGACY_WEIGHT_KEY:
            pass
        elif device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif device.type == "cpu":
            device = torch.device("cpu")
        key = str(device)
        if key not in self._dev:
            self._dev.clear()
            w = self.weight.to(device)
            b = None if self.bias is None else self.bias.to(device=device, dtype=torch.int64)
            self._dev[key] = (w, b)
        return self._dev[key]

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Exact ``Z [N, M]`` on ``x``'s device."""
        w, b = self._weights_on(x.device)
        if self.layout == "embed":
            z = w[:, x.reshape(-1)].to(torch.int64)
        else:
            z = exact_matmul(w, self.unfold(x, torch.float32), max_x=self.max_input + 1)
        if b is not None:
            z = z + b[:, None]
        return z


@dataclass
class CheapOp(Op):
    """An operation the verifier recomputes: ``output = fn(*inputs)``."""

    fn: Callable[..., torch.Tensor] | None = None
    note: str = ""
    params: dict = field(default_factory=dict)


def _with_tensors_on(fn, device: torch.device, copies: dict):
    """``fn`` with every tensor among its default arguments and closure cells replaced by its
    copy on ``device`` (``copies``: ``id(tensor) -> (tensor, copy)``, shared across calls so a
    tensor is copied once); ``fn`` itself when it captured no tensor or is not a plain function."""
    if not isinstance(fn, types.FunctionType):
        return fn

    def on(v):
        if not torch.is_tensor(v):
            return v
        if id(v) not in copies:
            copies[id(v)] = (v, v.to(device))
        return copies[id(v)][1]

    defaults = tuple(map(on, fn.__defaults__ or ()))
    kwdefaults = {k: on(v) for k, v in (fn.__kwdefaults__ or {}).items()}
    cells = tuple(types.CellType(on(c.cell_contents)) if torch.is_tensor(_cell_value(c)) else c
                  for c in fn.__closure__ or ())
    if (all(a is b for a, b in zip(defaults, fn.__defaults__ or ()))
            and all(v is fn.__kwdefaults__[k] for k, v in kwdefaults.items())
            and all(a is b for a, b in zip(cells, fn.__closure__ or ()))):
        return fn
    new = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, defaults or None, cells or None)
    new.__kwdefaults__ = kwdefaults or None
    new.__qualname__, new.__doc__ = fn.__qualname__, fn.__doc__
    return new


def _cell_value(cell):
    try:
        return cell.cell_contents
    except ValueError:            # an empty cell
        return None


@dataclass
class IntGraph:
    """An ordered list of operations; ``inputs[0]`` of the first op is the query."""

    ops: list[Op]
    input_name: str = "x"
    output_name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def mat_ops(self) -> list[MatOp]:
        return [op for op in self.ops if isinstance(op, MatOp)]

    def public(self) -> "IntGraph":
        return IntGraph([op.public() for op in self.ops], self.input_name,
                        self.output_name, dict(self.meta))

    def with_constants_on(self, device) -> tuple["IntGraph", list[tuple[torch.Tensor, torch.Tensor]]]:
        """This graph with the tensors its cheap ops captured (requantisation multipliers, norm
        gains, look-up tables) copied to ``device`` once, and the ``(source, copy)`` pairs.

        The cheap ops are new ``CheapOp`` objects whose functions are the same code with the
        copies as their default arguments (or closure cells); this graph and its ops are not
        modified.  Each ``c.to(x.device)`` the functions do is then a no-op instead of a host
        round trip per call on a GPU."""
        copies: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        ops = [dataclasses.replace(op, fn=_with_tensors_on(op.fn, torch.device(device), copies))
               if isinstance(op, CheapOp) else op for op in self.ops]
        pairs = [pair for pair in copies.values() if pair[0] is not pair[1]]
        return IntGraph(ops, self.input_name, self.output_name, dict(self.meta)), pairs

    def claim_columns(self, x: torch.Tensor) -> list[int] | None:
        """Every weight op's column count ``M`` on the query ``x``, in op order (``None`` if ``x``
        is malformed for one of them): the ops run on meta tensors, which carry shapes and no
        values, so nothing is computed or allocated."""
        graph, _ = self.with_constants_on("meta")
        env = {self.input_name: x.to("meta")}
        out = []
        for op in graph.ops:
            if isinstance(op, MatOp):
                xin = env[op.inputs[0]]
                m = op.n_cols(xin)
                if m is None:
                    return None
                out.append(m)
                env[op.output] = op.fold(torch.empty(op.n_rows, m, dtype=torch.int64, device="meta"), xin)
            else:
                env[op.output] = op.fn(*[env[n] for n in op.inputs])
        return out

    def n_params(self) -> int:
        return sum(op.n_rows * op.row_length for op in self.mat_ops)

    def last_use(self) -> dict[str, int]:
        """Index of the last op that reads each tensor (the graph output is never freed)."""
        last: dict[str, int] = {}
        for i, op in enumerate(self.ops):
            for n in op.inputs:
                last[n] = i
        last[self.output_name] = len(self.ops)
        return last

    def forward(self, x: torch.Tensor, *, tamper: Callable[[MatOp, torch.Tensor], torch.Tensor] | None = None,
                weights_override: dict[str, MatOp] | None = None, free: bool = False,
                claims_device=None, send: Callable[[torch.Tensor], object] | None = None,
                watch: Callable[[MatOp, torch.Tensor], None] | None = None) -> tuple[dict, dict]:
        """Honest (or tampered) execution.  Returns ``(env, claims)``.

        ``claims[op.name]`` is the ``[N, M]`` pre-activation matrix the prover
        sends.  ``tamper(op, Z)`` may return a modified ``Z``; everything after it
        is re-propagated honestly from the modified value, which is exactly the
        trace-tampering attack.  ``weights_override`` runs some ops with other
        weights (the model-substitution attack) while claiming the committed model.
        The lean prover (``Prover(lean=True)``) passes ``free=True``, which drops every
        tensor once no later op reads it, and ``claims_device="cpu"``, which moves each
        claim to host memory as soon as it is computed: the same integers, much less
        GPU memory.
        ``send(Z)``, if given, is what is kept of each claim instead of ``Z`` (moved to
        ``claims_device``), e.g. the streaming verifier's int32 wire format.  ``watch(op, x)``, if
        given, sees each weight op's input as the op runs (e.g. the token ids a lookup table is
        opened at).
        """
        env = {self.input_name: x}
        claims: dict[str, torch.Tensor] = {}
        last = self.last_use() if free else None
        for i, op in enumerate(self.ops):
            if isinstance(op, MatOp):
                runner = (weights_override or {}).get(op.name, op)
                xin = env[op.inputs[0]]
                if watch is not None:
                    watch(op, xin)
                z = runner.compute(xin)
                if tamper is not None:
                    z = tamper(op, z)
                env[op.output] = op.fold(z, xin)
                if send is not None:
                    claims[op.name] = send(z)
                else:
                    claims[op.name] = z if claims_device is None else z.to(claims_device)
                del z
            else:
                env[op.output] = op.fn(*[env[n] for n in op.inputs])
            if free:  # drop every tensor no later op reads (``env`` is then partial)
                for n in op.inputs:
                    if last.get(n, -1) <= i:
                        env.pop(n, None)
        return env, claims
