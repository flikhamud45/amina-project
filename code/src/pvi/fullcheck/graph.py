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
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

__all__ = [
    "Op",
    "MatOp",
    "CheapOp",
    "IntGraph",
    "exact_matmul",
    "requant",
    "INT8_MAX",
]

INT8_MAX = 127
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


def requant(z: torch.Tensor, mult: torch.Tensor, shift: int, lo: int, hi: int) -> torch.Tensor:
    """``clamp(round(z * mult / 2**shift), lo, hi)`` with round-half-up, exactly."""
    out = (z * mult + (1 << (shift - 1))) >> shift
    return out.clamp(lo, hi)


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
        """Claimed ``Z [N, M]`` -> output tensor shaped like the op's output."""
        if self.layout == "conv":
            k, s, p = self.conv
            b, _, h, w = x.shape
            ho = (h + 2 * p - k) // s + 1
            wo = (w + 2 * p - k) // s + 1
            return z.reshape(self.n_rows, b, ho, wo).permute(1, 0, 2, 3).contiguous()
        if self.layout == "embed":
            return z.T.reshape(*x.shape, self.n_rows).contiguous()
        return z.T.reshape(*x.shape[:-1], self.n_rows).contiguous()

    # -- prover ---------------------------------------------------------------
    def _weights_on(self, device) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    def n_params(self) -> int:
        return sum(op.n_rows * op.row_length for op in self.mat_ops)

    def forward(self, x: torch.Tensor, *, tamper: Callable[[MatOp, torch.Tensor], torch.Tensor] | None = None,
                weights_override: dict[str, MatOp] | None = None) -> tuple[dict, dict]:
        """Honest (or tampered) execution.  Returns ``(env, claims)``.

        ``claims[op.name]`` is the ``[N, M]`` pre-activation matrix the prover
        sends.  ``tamper(op, Z)`` may return a modified ``Z``; everything after it
        is re-propagated honestly from the modified value, which is exactly the
        trace-tampering attack.  ``weights_override`` runs some ops with other
        weights (the model-substitution attack) while claiming the committed model.
        """
        env = {self.input_name: x}
        claims: dict[str, torch.Tensor] = {}
        for op in self.ops:
            if isinstance(op, MatOp):
                runner = (weights_override or {}).get(op.name, op)
                xin = env[op.inputs[0]]
                z = runner.compute(xin)
                if tamper is not None:
                    z = tamper(op, z)
                claims[op.name] = z
                env[op.output] = op.fold(z, xin)
            else:
                env[op.output] = op.fn(*[env[n] for n in op.inputs])
        return env, claims
