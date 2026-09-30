"""The whole-network check: every weight product is verified, nothing is sampled.

Message flow for one query ``x`` (``mode="C"``, the verifier holds only the
32-byte commitment of each weight matrix -- the setting of Anchuri et al.):

1. prover -> verifier: the claimed pre-activations ``Z_l`` of every weight op.
2. verifier: recomputes every cheap op (requant, ReLU, pooling, residual adds,
   norms, softmax, attention, ...) from ``x`` and the claims, which gives it the
   input ``X_l`` of every weight op without trusting the prover for any of them.
   It then draws ``r`` random rows ``chi_l`` per weight op.
3. prover -> verifier: ``u_l = chi_l^T [W_l | b_l]``.
4. verifier: checks ``chi_l^T Z_l == u_l^T [X_l ; 1]`` for every op and every
   column of ``Z_l`` (Freivalds), then draws ``t`` column indices per op.
5. prover -> verifier: the opened columns and their Merkle paths; the verifier
   checks each against its commitment and against ``Enc(u_l)``.

``mode="K"`` is the setting of SafetyNets, Slalom and Maverick, where the
verifier knows the weights: it computes ``u_l`` itself (fresh ``chi`` per query)
and steps 3 and 5 disappear.  ``mode="Kpre"`` additionally precomputes ``u_l``
for a secret ``chi`` once, so a query costs the verifier only step 2 and the two
inner products; the proof is just the claims (Slalom's and Maverick's
"preprocessing" mode).

Soundness (a false claim accepted), per weight op, is at most ``p**-r``
(Freivalds: a nonzero column of ``Z - A X`` survives a uniformly random row with
probability ``1/p``) plus, in mode C, ``((k-1)/n)**t`` (a wrong ``u`` survives
``t`` distinct columns of a Reed--Solomon code of distance ``n-k+1``), with a
union bound over ops.  Two preconditions make the *integer* claim, not just its
residue, the thing that is checked: claims are range-checked to ``|z| < 2**29``
(so two in-range integers with equal residues are equal, since ``2 * 2**29 < p``),
and every weight op is checked at commitment time to have honest outputs inside
that range (:func:`claim_bound`).  Completeness is exact: every quantity is an
integer and both parties compute cheap ops with the same integer rules.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
import time
import weakref
from dataclasses import dataclass, field

import numpy as np
import torch

from .commitment import (HASH_BYTES, CommitmentPublic, WeightCommitment, codeword_at, column_rows, map_threaded,
                         row_leaves, verify_multiproofs)
from .field import (_LIMB_MAX, LOG2_P, P, combine_limbs, exact_chunk, exact_gemm_i64, field_matmul_mod,
                    int8_field_matmul, int8_left, int8_ok, int8_right, int8_small_matmul, limbs_f64,
                    small_matmul_mod, to_field)
from .graph import IntGraph, MatOp

__all__ = [
    "SecurityParams",
    "params_for",
    "soundness_bits",
    "claim_bound",
    "Challenger",
    "Prover",
    "Verifier",
    "run_query",
    "Z_BOUND",
    "commit_graph",
]

Z_BOUND = 1 << 29
"""Claims must satisfy ``-2**29 < z < 2**29``.  Because ``2 * 2**29 < p``, two
in-range integers with the same residue mod ``p`` are equal, so checking
``Z = A X`` mod ``p`` pins down the integer claim the verifier then uses."""


@dataclass(frozen=True)
class SecurityParams:
    lam: float
    reps: int
    rate: int
    columns: int
    fiat_shamir: bool = False
    grinding_bits: int = 64


def params_for(lam: float, n_checks: int, *, rate: int = 4, fiat_shamir: bool = False,
               grinding_bits: int = 64) -> SecurityParams:
    """Smallest ``(r, t)`` whose union bound over ``n_checks`` ops is ``<= 2**-lam``.

    With Fiat--Shamir the prover can grind offline, so ``grinding_bits`` more are
    required (security against ``2**grinding_bits`` hash evaluations).
    """
    bits = lam + math.log2(2 * max(1, n_checks)) + (grinding_bits if fiat_shamir else 0)
    reps = max(1, math.ceil(bits / LOG2_P))
    columns = max(1, math.ceil(bits / math.log2(rate)))
    return SecurityParams(lam, reps, rate, columns, fiat_shamir, grinding_bits)


def soundness_bits(params: SecurityParams, shapes: list[tuple[int, int]], mode: str = "C") -> float:
    """``-log2`` of the total soundness error.  ``shapes`` holds ``(k, n)`` per op."""
    total = 0.0
    for k, n in shapes:
        err = 2.0 ** (-params.reps * LOG2_P)
        if mode == "C":
            t = min(params.columns, n)
            # distinct columns: prod_{i<t} (k-1-i)/(n-i) <= ((k-1)/n)**t
            log_col = sum(math.log2(max(k - 1 - i, 1e-300) / (n - i)) for i in range(t)) if k > 1 else -math.inf
            err += 2.0 ** log_col
        total += err
    bits = -math.log2(total)
    return bits - (params.grinding_bits if params.fiat_shamir else 0)


def claim_bound(op: MatOp) -> int:
    """Largest ``|z|`` an honest execution of ``op`` can produce."""
    w = op.weight.to(torch.int64)
    if op.layout == "embed":
        return int(w.abs().max())
    bound = (op.max_input + 1) * w.abs().sum(1)
    if op.bias is not None:
        bound = bound + op.bias.abs()
    return int(bound.max())


def commit_graph(graph: IntGraph, rate: int, device="cpu") -> dict[str, WeightCommitment]:
    """Commit every weight op, after checking honest claims fit the range check."""
    for op in graph.mat_ops:
        b = claim_bound(op)
        if b >= Z_BOUND:
            raise ValueError(f"{op.name}: honest claims can reach {b} >= Z_BOUND = {Z_BOUND}")
    return {op.name: WeightCommitment.build(op.name.encode(), op.weight, op.bias, rate=rate, device=device)
            for op in graph.mat_ops}


class Challenger:
    """Verifier randomness.

    Interactive: every challenge is expanded with SHAKE-256 from a fresh 256-bit
    key drawn from the operating system's CSPRNG (``secrets``) at the moment the
    challenge is issued.  In particular the column indices are keyed only after
    the prover has sent ``u``, so nothing the prover saw earlier predicts them.
    Fiat--Shamir: the key is the running SHA-256 transcript of the statement
    (parameters, commitments, input) and of every prover message so far.
    ``seed`` makes interactive keys reproducible (tests only).
    """

    def __init__(self, *, fiat_shamir: bool = False, seed: int | None = None) -> None:
        self.fiat_shamir = fiat_shamir
        self._state = hashlib.sha256(b"pvi/fullcheck/v2")
        self._seed = seed
        self._count = 0

    def absorb(self, label: bytes, blob: bytes) -> None:
        """Length-prefixed, labelled absorption (an injective transcript encoding)."""
        self._state.update(len(label).to_bytes(2, "big") + label + len(blob).to_bytes(8, "big"))
        self._state.update(blob)

    def _key(self, label: str) -> bytes:
        if self.fiat_shamir:
            h = self._state.copy()
            h.update(b"challenge/" + label.encode())
            return h.digest()
        if self._seed is None:
            return secrets.token_bytes(32)
        self._count += 1
        return hashlib.sha256(f"test-seed/{self._seed}/{self._count}/{label}".encode()).digest()

    @staticmethod
    def _stream(key: bytes, label: str, n_bytes: int) -> bytes:
        return hashlib.shake_256(key + label.encode()).digest(n_bytes)

    def folding(self, name: str, n_rows: int, reps: int) -> torch.Tensor:
        """``reps x n_rows`` uniform field elements (rejection sampling, no bias)."""
        n = reps * n_rows
        key = self._key("fold/" + name)
        parts, have, block = [], 0, 0
        while have < n:
            want = int((n - have) * 1.08) + 64
            words = np.frombuffer(self._stream(key, f"fold/{name}/{block}", 4 * want), dtype="<u4")
            words = words.astype(np.int64)
            words = words[words < 2 * P] % P
            parts.append(words)
            have += words.size
            block += 1
        return torch.from_numpy(np.concatenate(parts)[:n].reshape(reps, n_rows).copy())

    def columns(self, name: str, n_points: int, t: int) -> torch.Tensor:
        """``t`` distinct uniform indices in ``[0, n_points)``, sorted."""
        t = min(t, n_points)
        key = self._key("cols/" + name)
        limit = (1 << 64) - ((1 << 64) % n_points)
        chosen: set[int] = set()
        block = 0
        while len(chosen) < t:
            vals = np.frombuffer(self._stream(key, f"cols/{name}/{block}", 8 * (4 * t + 64)), dtype="<u8")
            for v in vals.tolist():
                if v < limit:
                    chosen.add(v % n_points)
                    if len(chosen) == t:
                        break
            block += 1
        return torch.tensor(sorted(chosen), dtype=torch.int64)


def _tensor_blob(t: torch.Tensor) -> bytes:
    shape = struct.pack(f"<{t.dim()}Q", *t.shape)
    return struct.pack("<B", t.dim()) + shape + t.detach().to("cpu", torch.int64).contiguous().numpy().tobytes()


class Prover:
    def __init__(self, graph: IntGraph, *, device="cpu", commitments: dict[str, WeightCommitment] | None = None,
                 lean: bool = False):
        self.graph = graph
        self.device = torch.device(device)
        self.commitments = commitments or {}
        self.lean = lean  # free dead activations and stream each claim to the host during the forward pass
        self._ops = {op.name: op for op in graph.mat_ops}

    def claims(self, x: torch.Tensor, *, send=None, **forward_kwargs) -> dict[str, torch.Tensor]:
        """The claims of query ``x``, on the host.  ``send(z)`` (see ``IntGraph.forward``), if given,
        is applied to each claim on the prover's device as it is computed, and its results are
        returned as they are."""
        if self.lean:
            forward_kwargs = dict(forward_kwargs, free=True, claims_device="cpu")
        _, claims = self.graph.forward(x.to(self.device), send=send, **forward_kwargs)
        return claims if send is not None else {k: v.to("cpu") for k, v in claims.items()}

    def _weight(self, name: str) -> torch.Tensor:
        # the int8 weights the forward pass already keeps on the device
        return self._ops[name]._weights_on(self.device)[0]

    def fold(self, chis: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.commitments[name].fold(chi, self.device, weight=self._weight(name))
                for name, chi in chis.items()}

    def open(self, cols: dict[str, torch.Tensor]):
        return {name: self.commitments[name].open(idx, self.device, weight=self._weight(name))
                for name, idx in cols.items()}


_RHS_CHUNK = 8192
"""Contraction block of the right-hand sides: that of ``small_matmul_mod``, which the
straightforward ``u^T X`` uses (int8-sized ``X`` times field elements: every block sum is
below ``2**53``), so the products below are exact on exactly the same inputs."""
_IM2COL_MAX = 1 << 16
"""Convolutions whose unfolded input has at most this many entries use im2col and one GEMM
(fewest calls); larger ones the unfold-free shifted GEMM (less memory traffic)."""
_BATCH_BYTES = 1 << 25
"""Budget (float64 operand bytes) of one batched product in the verifier's checks."""


def _conv_rhs(op: MatOp, u: torch.Tensor, xin: torch.Tensor) -> torch.Tensor:
    """``u^T unfold(x)`` without unfolding: ONE GEMM of the ``r k^2`` filter rows of ``u``
    with the padded input (``[r k^2, C] @ [C, B Hp Wp]``), then ``k^2`` shifted strided
    slices are added.  Each output is the sum of the same ``C k^2`` integer terms as in
    the unfolded product, exact in float64 when ``C k^2 <= _RHS_CHUNK``."""
    kk, s, p = op.conv
    b, c, h, w = xin.shape
    r = u.shape[0]
    ho, wo = (h + 2 * p - kk) // s + 1, (w + 2 * p - kk) // s + 1
    filt = u[:, :c * kk * kk].to(torch.float64).reshape(r, c, kk, kk).permute(0, 2, 3, 1).reshape(r * kk * kk, c)
    xf = torch.nn.functional.pad(xin.to(torch.float64), (p, p, p, p))
    hp, wp = h + 2 * p, w + 2 * p
    y = (filt @ xf.transpose(0, 1).reshape(c, b * hp * wp)).view(r, kk, kk, b, hp, wp)
    acc = y[:, 0, 0, :, :s * (ho - 1) + 1:s, :s * (wo - 1) + 1:s].clone()
    for dy in range(kk):
        for dx in range(kk):
            if dy or dx:
                acc += y[:, dy, dx, :, dy:dy + s * (ho - 1) + 1:s, dx:dx + s * (wo - 1) + 1:s]
    return torch.remainder(acc.reshape(r, b * ho * wo).to(torch.int64), P)


def _rhs(op: MatOp, u: torch.Tensor, xin: torch.Tensor) -> torch.Tensor:
    """``u^T [X ; 1]`` over the field, shape ``[r, M]``.

    Linear layers multiply ``u`` with the ``[M, K]`` input as it is (no int64 unfold, no
    transposed copies); convolutions with ``C k^2 <= _RHS_CHUNK`` use im2col (small) or
    :func:`_conv_rhs`.  Every product keeps the ``_RHS_CHUNK`` blocks of the
    straightforward product, so both are exact on the same inputs.
    """
    k = op.n_in
    if op.layout == "embed":
        out = u[:, xin.reshape(-1)]
    elif op.layout == "linear":
        out = torch.remainder(exact_gemm_i64(u[:, :k].to(torch.float64), xin.reshape(-1, k).T, _RHS_CHUNK), P)
    elif k > _RHS_CHUNK:
        x = op.unfold(xin)  # [K, M] small ints
        out = small_matmul_mod(x.T.contiguous(), u[:, :k].T.contiguous()).T
    elif k * op.n_cols(xin) <= _IM2COL_MAX:
        kk, s, p = op.conv
        cols = torch.nn.functional.unfold(xin.to(torch.float64), kk, padding=p, stride=s)   # [B, K, L]
        y = (u[:, :k].to(torch.float64) @ cols).transpose(0, 1).reshape(u.shape[0], -1)
        out = torch.remainder(y.to(torch.int64), P)
    else:
        out = _conv_rhs(op, u, xin)
    if op.has_bias:
        out = (out + u[:, k:k + 1]) % P
    return out


def _min_max(a: torch.Tensor) -> tuple[int, int]:
    """Exact ``(min, max)`` of a non-empty integer tensor: numpy's reductions on the CPU
    (about 2x torch's there), ``aminmax`` elsewhere."""
    if a.device.type == "cpu":
        n = a.numpy()
        return int(n.min()), int(n.max())
    lo, hi = torch.stack(torch.aminmax(a)).tolist()     # one device-to-host copy
    return lo, hi


def _in_range(z: torch.Tensor, bound: int) -> bool:
    """``-bound < z < bound`` everywhere (without abs(), which overflows on INT64_MIN)."""
    if z.numel() == 0:
        return True
    lo, hi = _min_max(z)
    return lo > -bound and hi < bound


def _in_field(a: torch.Tensor) -> bool:
    """``0 <= a < P`` everywhere."""
    if a.numel() == 0:
        return True
    lo, hi = _min_max(a)
    return lo >= 0 and hi < P


def _stamp(t: torch.Tensor) -> tuple:
    """Identifies ``t`` as it is now, without keeping it alive: a weak reference and the
    version counter that every in-place change of ``t`` (or of a view of it) increments.

    A write that bypasses the counter (through ``.numpy()``, ``.data`` or DLPack) is not
    seen.  What is stamped is the verifier's own state: the claims it received, which must not
    change while one query is checked in any case (derive's inputs were computed from them; a
    verifier fed over a wire deserialises them into tensors nobody else holds), and in mode
    Kpre its secret ``chi`` and ``u``."""
    return weakref.ref(t), t._version


def _same(stamp: tuple, t: torch.Tensor) -> bool:
    """``t`` is the very tensor ``stamp`` was taken of, and it has not been modified since."""
    return stamp[0]() is t and stamp[1] == t._version


def _leading_passes(ops: list, check) -> tuple[int, Exception | None]:
    """How many leading ``ops`` pass ``check``, and the exception the next one raised (if any)."""
    for i, op in enumerate(ops):
        try:
            if not check(op):
                return i, None
        except Exception as exc:   # raised by the caller once the ops before it are decided
            return i, exc
    return len(ops), None


# On a GPU the checks do not copy a verdict back per op (each copy waits for the device):
# every range check and comparison leaves a boolean on the device, and all of them come back
# with ONE copy once the work is queued.

def _defer(device) -> bool:
    """Whether checks on ``device`` leave their verdicts there: everywhere but on the CPU,
    where reading a verdict costs nothing (the tests run the deferred forms on the CPU too)."""
    return torch.device(device).type != "cpu"


def _outside(a: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Some entry of ``a`` outside ``[lo, hi]``, as a device boolean (``False`` if ``a`` is empty)."""
    if a.numel() == 0:
        return torch.zeros((), dtype=torch.bool, device=a.device)
    amin, amax = torch.aminmax(a)
    return (amin < lo) | (amax > hi)


def _out_of_range(z: torch.Tensor, bound: int) -> torch.Tensor:
    """Some ``|z| >= bound``: :func:`_in_range`, deferred."""
    return _outside(z, 1 - bound, bound - 1)


def _disagrees(u: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``u`` is not in the field or ``chi^T Z != u^T [X ; 1]``: one op's Freivalds check, deferred."""
    return _outside(u, 0, P - 1) | (lhs != rhs).any()


def _columns_verdict(codes: list[bool], merkle: list, n_ok: int, exc: Exception | None, n_ops: int) -> str | None:
    """The column check's verdict, op by op -- shape, code, Merkle -- for the ``n_ok`` leading ops
    with well-formed openings (``exc``: what the next one raised): ``codes[i]`` and ``merkle[i] =
    (ok, exception)`` of op ``i`` (``merkle`` may end at the first code failure)."""
    for i in range(n_ok):
        if not codes[i]:
            return "columns_code"
        ok, err = merkle[i]
        if err is not None:
            raise err
        if not ok:
            return "columns_merkle"
    if exc is not None:
        raise exc
    return "columns_shape" if n_ok < n_ops else None


def _to_host(flags: list[torch.Tensor]) -> list[bool]:
    """Device booleans (0-d or 1-d, on one device) as a flat list, with ONE device-to-host copy."""
    return torch.cat([f.reshape(-1) for f in flags]).tolist() if flags else []


def _to_device(t: torch.Tensor, device) -> torch.Tensor:
    """``t`` on ``device``; a host tensor bound for a GPU goes through pinned memory, so the copy
    does not wait for the device."""
    if torch.device(device).type != "cuda" or t.device.type != "cpu":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def _upload_rows(rows: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    """Host int32 arrays on ``device``: per shape, one staging buffer (pinned) and one
    non-blocking copy (a copy from pageable memory would wait for the device each time)."""
    by_shape: dict = {}
    for name, a in rows.items():
        by_shape.setdefault(a.shape, []).append(name)
    out = {}
    for shape, names in by_shape.items():
        staged = torch.empty((len(names), *shape), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        for j, name in enumerate(names):
            staged[j].numpy()[...] = rows[name]
        out.update(zip(names, staged.to(device, non_blocking=True)))
    return out


@dataclass
class Verifier:
    graph: IntGraph
    params: SecurityParams
    mode: str = "C"
    publics: dict[str, CommitmentPublic] = field(default_factory=dict)
    weights: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = field(default_factory=dict)
    lean: bool = False
    device: str = "cpu"      # "cuda" / "cuda:1": a client with a GPU (Merkle hashing stays on the CPU)
    stream: bool = False     # run_query with the streaming verifier (pvi.fullcheck.pipeline): same verdicts
    _pre: dict = field(default_factory=dict)
    _wdev: dict = field(default_factory=dict)   # mode K on a GPU: the weights stay on the device
    _ranged: dict = field(default_factory=dict)  # name -> (weakref, _version) of the claim derive() range-checked
    _kept: dict = field(default_factory=dict)    # Kpre: stacked operands of the fixed chi and u
    _consts: list = field(default_factory=list)  # a GPU client: the cheap ops' constants (source, copy, versions)

    def __post_init__(self) -> None:
        if torch.device(self.device).type != "cpu":   # the cheap ops' constants live on the device
            self.graph, pairs = self.graph.with_constants_on(self.device)
            self._consts = [(src, dst, (src._version, dst._version)) for src, dst in pairs]

    def _refresh_constants(self) -> None:
        """Copy again every device constant whose source (or copy) was modified since."""
        for i, (src, dst, versions) in enumerate(self._consts):
            if (src._version, dst._version) != versions:
                dst.copy_(src)
                self._consts[i] = (src, dst, (src._version, dst._version))

    def precompute(self, challenger: Challenger) -> None:
        """Mode Kpre: fix a secret ``chi`` per op and precompute ``u``."""
        for op in self.graph.mat_ops:
            chi = challenger.folding(op.name, op.n_rows, self.params.reps).to(self.device)
            self._pre[op.name] = (chi, self._fold_local(op, chi))
        self._wdev.clear()   # Kpre never folds again: keep no device copy of the model after this
        _sync(self.device)   # the precompute is timed by its callers

    def _fold_local(self, op: MatOp, chi: torch.Tensor) -> torch.Tensor:
        w, b = self.weights[op.name]
        if chi.device.type != "cpu":   # upload once, not on every query
            key = (op.name, str(chi.device))
            if key not in self._wdev:
                self._wdev[key] = (w.to(chi.device), None if b is None else b.to(chi.device))
            w, b = self._wdev[key]
        u = small_matmul_mod(w.T.to(torch.int64).contiguous(), chi.T.contiguous()).T
        if b is not None:
            u = torch.cat([u, field_matmul_mod(chi, to_field(b)[:, None])], 1)
        return u

    def derive(self, x: torch.Tensor, claims: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Recompute every cheap op; return each weight op's input, or ``None`` to reject.

        The range checks of claims on a GPU are deferred: they come back with one copy at
        the end instead of one per weight op."""
        out = self._derive(x, claims)
        if out is None or any(_to_host(out[1])):
            return None
        self._ranged = out[2]
        return out[0]

    def _derive(self, x: torch.Tensor, claims, *, dtype: torch.dtype = torch.int64, visit=None):
        """The derive loop: ``None`` to reject, else ``(inputs, pending, ranged)``.

        ``claims.get(name)`` gives each claim, of ``dtype``.  A claim on the CPU is range-checked
        at once; a claim on a device leaves its (deferred) verdict in ``pending``, and the ops
        after it run on whatever it holds: every cheap op is a total function of int64
        tensors, so they only compute values that are never used, and an exception there is
        reported as the rejection it follows.  ``visit(op, z, x)``, if given, is called for
        each weight op once its claim has the right shape; the op's input is then not kept
        (``inputs`` stays empty) and every tensor is freed after its last reader.
        """
        env = {self.graph.input_name: x}
        inputs: dict[str, torch.Tensor] = {}
        pending: list[torch.Tensor] = []
        ranged: dict[str, tuple] = {}
        self._ranged = {}
        self._refresh_constants()
        free = self.lean or visit is not None
        last = self.graph.last_use() if free else None
        try:
            for i, op in enumerate(self.graph.ops):
                if free and i > 0:  # free what no later op reads (weight-op inputs stay in ``inputs``)
                    for n in self.graph.ops[i - 1].inputs:
                        if last.get(n, -1) <= i - 1:
                            env.pop(n, None)
                if isinstance(op, MatOp):
                    z = claims.get(op.name)
                    xin = env[op.inputs[0]]
                    m = op.n_cols(xin)
                    if z is None or m is None or z.dtype != dtype or tuple(z.shape) != (op.n_rows, m):
                        return None
                    if not _defer(z.device):
                        if not _in_range(z, Z_BOUND):
                            return None
                    elif z.numel():
                        pending.append(_out_of_range(z, Z_BOUND))
                    ranged[op.name] = _stamp(z)
                    if visit is None:
                        inputs[op.name] = xin
                    else:
                        visit(op, z, xin)
                    y = op.fold(z, xin)
                    env[op.output] = y if y.dtype == torch.int64 else y.to(torch.int64)
                else:
                    env[op.output] = op.fn(*[env[n] for n in op.inputs])
        except Exception:
            if any(_to_host(pending)):
                return None
            raise
        return inputs, pending, ranged

    # -- the checks ---------------------------------------------------------------------
    # The checks of different weight ops are independent, and a transformer repeats a few
    # shapes in every block, so same-shape ops are checked with ONE batched product instead
    # of ~15 small torch calls each.  Each op's result is the same integers as its own
    # product, and the verdict (and any exception) is that of checking op by op in order.

    def _kept_or_built(self, key, tensors: list, build):
        """``build(tensors)``; in mode Kpre, where ``chi`` and ``u`` are the verifier's own fixed
        values, kept across queries while exactly these, unmodified, tensors come back."""
        if self.mode != "Kpre":
            return build(tensors)
        hit = self._kept.get(key)
        if hit is not None and len(hit[0]) == len(tensors) and all(map(_same, hit[0], tensors)):
            return hit[1]
        out = build(tensors)
        self._kept[key] = ([_stamp(t) for t in tensors], out)
        return out

    def _field_products(self, pairs: dict, bound: int) -> dict:
        """``{name: left @ right mod P}`` for ``pairs[name] = (left, right)``, every ``|right| <
        bound``; same-shape pairs go through one batched limb product."""
        out, groups = {}, {}
        for name, (left, right) in pairs.items():
            groups.setdefault((left.shape, right.shape, left.device, right.device), []).append(name)
        chunk = exact_chunk(_LIMB_MAX, bound - 1)
        for (lshape, rshape, _, dev), names in groups.items():
            if len(names) == 1 and self.mode != "Kpre":        # nothing to share or to keep
                out[names[0]] = field_matmul_mod(*pairs[names[0]], right_bound=bound)
                continue
            step = max(1, _BATCH_BYTES // (8 * (rshape.numel() + 3 * lshape.numel())))
            for i in range(0, len(names), step):
                part = tuple(names[i:i + step])
                limbs = self._kept_or_built(("limbs",) + part, [pairs[n][0] for n in part],
                                            lambda ts: limbs_f64(torch.stack(ts)))
                rf = torch.empty((len(part),) + tuple(rshape), dtype=torch.float64, device=dev)
                for j, name in enumerate(part):
                    rf[j].copy_(pairs[name][1])
                out.update(zip(part, combine_limbs(exact_gemm_i64(limbs, rf, chunk), lshape[0])))
        return out

    def _lhs(self, mats, claims, chis) -> dict:
        """``chi_l Z_l mod P``.  A claim that derive() range-checked (this very tensor, not
        modified since) is multiplied as a signed integer; any other is reduced first."""
        ranged, other = {}, {}
        for op in mats:
            z, seen = claims[op.name], self._ranged.get(op.name)
            if seen is not None and _same(seen, z):
                ranged[op.name] = (chis[op.name], z)
            else:
                other[op.name] = (chis[op.name], to_field(z))
        return {**self._signed_lhs(ranged), **self._field_products(other, P)}

    def _signed_lhs(self, pairs: dict) -> dict:
        """``{name: chi @ Z mod P}`` for ``pairs[name] = (chi, Z)`` with signed claims ``|Z| < 2**29``:
        int8 GEMMs where :func:`int8_ok` (exact for every int32 claim), else limb products."""
        if not pairs or not int8_ok(next(iter(pairs.values()))[1].device):
            return self._field_products(pairs, Z_BOUND)
        return {name: int8_field_matmul(self._kept_or_built(("chi8", name), [chi], lambda ts: int8_left(ts[0])), z)
                for name, (chi, z) in pairs.items()}

    def _rhs_all(self, mats, inputs, us, *, int8: bool = True) -> tuple[dict, list]:
        """``u_l^T [X_l ; 1]``, and a device flag per input the product took to be int8-valued.

        Linear ops reading the same input (q, k, v; gate, up) share one product with their ``u``
        rows stacked: an int8 GEMM of the input where :func:`int8_ok` (unless ``int8=False``),
        whose flag is set if the input is not int8-valued (the caller then computes again with
        ``int8=False``); else a limb product, with inputs of the same shape batched."""
        out, shared = {}, {}
        use8 = int8 and bool(mats) and int8_ok(us[mats[0].name].device)
        for op in mats:
            xin = inputs[op.name]
            if op.layout == "linear" and (use8 or op.n_in <= _RHS_CHUNK) and us[op.name].device == xin.device:
                shared.setdefault(id(xin), (xin, []))[1].append(op)
            else:
                out[op.name] = _rhs(op, us[op.name], xin)
        if use8:
            return out, [self._rhs_int8(xin, ops, us, out) for xin, ops in shared.values()]
        groups = {}
        for xin, ops in shared.values():
            key = (ops[0].n_in, xin.numel() // ops[0].n_in, tuple(us[op.name].shape[0] for op in ops), xin.device)
            groups.setdefault(key, []).append((xin, ops))
        for (k, m, rows, dev), items in groups.items():
            step = max(1, _BATCH_BYTES // (8 * k * (m + sum(rows))))
            for i in range(0, len(items), step):
                part = items[i:i + step]
                names = tuple(op.name for _, ops in part for op in ops)
                uf = self._kept_or_built(("u",) + names, [us[n] for n in names], lambda ts, g=len(part): torch.cat(
                    [t[:, :k] for t in ts]).to(torch.float64).reshape(g, sum(rows), k))
                # X = x^T [K, M] per input; the batch buffer takes the layout the inputs are stored
                # in (row- or column-major: a linear layer's output is a transposed view of its
                # claim, see MatOp.fold), so the copy reads memory in order
                xs = [xin.reshape(m, k).T for xin, _ in part]
                by_rows = not xs[0].is_contiguous()
                xf = torch.empty(len(part), *((m, k) if by_rows else (k, m)), dtype=torch.float64, device=dev)
                for j, x in enumerate(xs):
                    xf[j].copy_(x.T if by_rows else x)
                y = torch.remainder((uf @ (xf.transpose(1, 2) if by_rows else xf)).to(torch.int64), P)   # exact
                for j, (_, ops) in enumerate(part):
                    for op, part_rows in zip(ops, y[j].split(rows)):
                        out[op.name] = (part_rows + us[op.name][:, k:]) % P if op.has_bias else part_rows
        return out, []

    def _rhs_int8(self, xin: torch.Tensor, ops: list, us, out: dict) -> torch.Tensor:
        """Fills ``out`` with ``u^T [X ; 1]`` of the linear ``ops`` reading ``xin`` (one int8 GEMM);
        returns the device flag "``X`` is not int8-valued", which makes this product wrong."""
        k = ops[0].n_in
        x = xin.reshape(-1, k)
        names = tuple(op.name for op in ops)
        operand = self._kept_or_built(("u8",) + names, [us[n] for n in names],
                                      lambda ts: int8_right(torch.cat([t[:, :k] for t in ts])))
        y, r0 = int8_small_matmul(operand, x.to(torch.int8)), 0
        for op in ops:
            r = us[op.name].shape[0]
            out[op.name] = (y[r0:r0 + r] + us[op.name][:, k:]) % P if op.has_bias else y[r0:r0 + r]
            r0 += r
        return _outside(x, -128, 127)

    def _u_ok(self, op: MatOp, us, *, values: bool = True) -> bool:
        """``u`` has the dtype and shape of the op's fold (and, with ``values``, is in the field)."""
        u = us.get(op.name)
        return (u is not None and u.dtype == torch.int64 and tuple(u.shape) == (self.params.reps, op.row_length)
                and (not values or _in_field(u)))

    def check_products(self, claims, inputs, chis, us) -> bool:
        """Freivalds for every weight op (``claims`` are the ones derive() accepted): the verdict
        of checking op by op -- ``u`` well formed, then ``chi^T Z == u^T [X ; 1]``.  On a GPU the
        field check of every ``u`` and every comparison come back with one copy."""
        mats = self.graph.mat_ops
        on_cpu = not _defer(self.device)
        n_ok, exc = _leading_passes(mats, lambda op: self._u_ok(op, us, values=on_cpu))
        good = mats[:n_ok]
        lhs = self._lhs(good, claims, chis)
        rhs, unchecked = self._rhs_all(good, inputs, us)
        if on_cpu:
            if any(_to_host(unchecked)):          # an input that is not int8-valued (int8 GEMMs only)
                rhs, _ = self._rhs_all(good, inputs, us, int8=False)
            agree = all(torch.equal(lhs[op.name], rhs[op.name]) for op in good)
        else:
            def disagree(rhs):
                return [_disagrees(us[op.name], lhs[op.name], rhs[op.name]) for op in good]

            flags, n = _to_host(unchecked + disagree(rhs)), len(unchecked)
            if any(flags[:n]):
                flags, n = _to_host(disagree(self._rhs_all(good, inputs, us, int8=False)[0])), 0
            agree = not any(flags[n:])
        if not agree:
            return False
        if exc is not None:
            raise exc
        return n_ok == len(mats)

    def _columns_shape_ok(self, op: MatOp, us, cols, openings, *, wire: bool = False) -> bool:
        """The opening of ``op`` is well formed: int64 columns ``[N, t]``, or with ``wire`` the
        int32 rows ``[t, N]`` of the streaming verifier (``None``: not representable); in the field."""
        pub = self.publics[op.name]
        opened, proof = openings[op.name]
        idx = cols[op.name]
        if wire and opened is None:
            return False
        shape, dtype = ((len(idx), pub.n_rows), torch.int32) if wire else ((pub.n_rows, len(idx)), torch.int64)
        return not (tuple(opened.shape) != shape or len(proof) > len(idx) * pub.depth
                    or opened.dtype != dtype or not _in_field(opened)
                    or us[op.name].shape[1] != pub.row_length)

    def _codewords(self, mats, us, cols) -> dict:
        """``Enc(u)[columns]`` per op.  ``Enc(u)`` is only needed at the ``t`` opened columns: it
        is evaluated there directly (:func:`codeword_at`), same-shape ops at once."""
        enc, groups = {}, {}
        for op in mats:
            u = us[op.name]
            groups.setdefault((u.shape, self.publics[op.name].n_points, len(cols[op.name]), u.device),
                              []).append(op.name)
        for (ushape, n_points, t, dev), names in groups.items():
            step = max(1, _BATCH_BYTES // (8 * ushape[-1] * (3 * ushape[0] + 2 * t)))
            for i in range(0, len(names), step):
                part = names[i:i + step]
                if len(part) == 1:
                    enc[part[0]] = codeword_at(us[part[0]], n_points, _to_device(cols[part[0]], dev))
                else:
                    res = codeword_at(torch.stack([us[n] for n in part]), n_points,
                                      _to_device(torch.stack([cols[n] for n in part]), dev))
                    enc.update(zip(part, res))
        return enc

    def _codes_ok(self, mats, chis, us, cols, openings) -> list[bool]:
        """``chi^T (opened columns) == Enc(u)[columns]`` per op."""
        enc = self._codewords(mats, us, cols)
        opened = self._field_products({op.name: (chis[op.name], openings[op.name][0].to(us[op.name].device))
                                       for op in mats}, P)
        return [torch.equal(opened[op.name], enc[op.name]) for op in mats]

    def _code_flags(self, mats, chis, us, cols, rows: dict) -> list[torch.Tensor]:
        """:meth:`_codes_ok` on a device, negated and left there: ``rows`` holds each op's opened
        columns as the int32 rows ``[t, N]`` already on the device."""
        enc = self._codewords(mats, us, cols)
        opened = self._field_products({op.name: (chis[op.name], rows[op.name].T) for op in mats}, P)
        return [(opened[op.name] != enc[op.name]).any() for op in mats]

    def _merkle(self, mats, cols, openings, rows: dict | None = None) -> list[tuple[bool, Exception | None]]:
        """``(multiproof verified, exception)`` per op: its opened columns (``rows``, or cast from
        ``openings``) hashed into leaves, large ones on threads, then its Merkle multiproof."""
        def leaves(op):
            r = rows[op.name] if rows is not None else column_rows(openings[op.name][0])
            return row_leaves(self.publics[op.name].tag, cols[op.name].tolist(), r)

        workers = torch.get_num_threads()
        hashed = map_threaded(leaves, mats, [4 * self.publics[op.name].n_rows for op in mats], workers)
        return verify_multiproofs([(self.publics[op.name].root, self.publics[op.name].depth, lv, openings[op.name][1])
                                   for op, lv in zip(mats, hashed)], workers)

    def check_columns(self, chis, us, cols, openings) -> str | None:
        """``None`` if every opened column is consistent, else the failing check: the verdict
        (and any exception) of checking op by op -- shape, then code, then Merkle.  On a GPU the
        code checks run on the device while the host checks the Merkle paths, and come back
        with one copy."""
        mats = self.graph.mat_ops
        n_ok, exc = _leading_passes(mats, lambda op: self._columns_shape_ok(op, us, cols, openings))
        good = mats[:n_ok]
        if not _defer(self.device):
            codes = self._codes_ok(good, chis, us, cols, openings)
            # the Merkle checks the verdict needs: those of the ops before the first code failure
            merkle = self._merkle(good[:codes.index(False)] if False in codes else good, cols, openings)
        else:
            rows = {op.name: column_rows(openings[op.name][0]) for op in good}
            flags = self._code_flags(good, chis, us, cols, _upload_rows(rows, self.device))
            merkle = self._merkle(good, cols, openings, rows)
            codes = [not bad for bad in _to_host(flags)]
        return _columns_verdict(codes, merkle, n_ok, exc, len(mats))


def _sync(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)   # the given GPU (e.g. a verifier on cuda:1), not the current one


def _absorb_statement(ch: Challenger, verifier: Verifier, x: torch.Tensor) -> None:
    p = verifier.params
    ch.absorb(b"params", struct.pack("<IIIi", p.reps, p.columns, p.rate, p.grinding_bits) + verifier.mode.encode())
    for op in verifier.graph.mat_ops:
        pub = verifier.publics.get(op.name)
        blob = struct.pack("<QQ", op.n_rows, op.row_length)
        if pub is not None:
            blob += pub.tag + pub.root + struct.pack("<Q", pub.n_points)
        ch.absorb(b"op/" + op.name.encode(), blob)
    ch.absorb(b"x", _tensor_blob(x.cpu()))


def run_query(prover: Prover, verifier: Verifier, x: torch.Tensor, *, seed: int | None = None,
              forward_kwargs: dict | None = None) -> dict:
    """One full interaction.  Returns acceptance, the rejecting check, timings (s)
    and proof bytes.  With Fiat--Shamir, ``fs_hash`` is paid by both parties.  A verifier
    with ``stream=True`` checks with the streaming verifier (:mod:`pvi.fullcheck.pipeline`):
    the same verdicts, with the verifier's overlapped work timed as one ``verify_total``."""
    if verifier.stream:
        from .pipeline import run_query_streaming
        return run_query_streaming(prover, verifier, x, seed=seed, forward_kwargs=forward_kwargs)
    p = verifier.params
    mode = verifier.mode
    ch = Challenger(fiat_shamir=p.fiat_shamir, seed=seed)
    t: dict[str, float] = {}

    _sync(prover.device)
    t0 = time.perf_counter()
    claims = prover.claims(x, **(forward_kwargs or {}))
    _sync(prover.device)
    t["prove_forward"] = time.perf_counter() - t0

    out = {"accepted": False, "rejected_at": None, "timings": t}
    b_claims = sum(z.numel() for z in claims.values()) * 4
    out["bytes"] = {"claims": b_claims, "u": 0, "columns": 0, "paths": 0}

    t0 = time.perf_counter()
    if p.fiat_shamir:
        _absorb_statement(ch, verifier, x)
        for k in sorted(claims):
            ch.absorb(b"claim/" + k.encode(), _tensor_blob(claims[k]))
    t["fs_hash"] = time.perf_counter() - t0

    vdev = torch.device(verifier.device)
    x_v, claims_v = x.cpu(), claims
    if vdev.type != "cpu":        # a GPU client: receiving the proof includes uploading it
        _sync(vdev)
        t0 = time.perf_counter()
        x_v, claims_v = x.to(vdev), {k: v.to(vdev) for k, v in claims.items()}
        _sync(vdev)
        t["verify_upload"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    inputs = verifier.derive(x_v, claims_v)
    _sync(vdev)
    t["verify_derive"] = time.perf_counter() - t0
    if inputs is None:
        out["rejected_at"] = "range_or_shape"
        return out

    mats = verifier.graph.mat_ops
    if mode == "Kpre":
        chis = {op.name: verifier._pre[op.name][0] for op in mats}
        us = {op.name: verifier._pre[op.name][1] for op in mats}
        t["prove_fold"] = t["verify_fold"] = 0.0
    else:
        chis = {op.name: ch.folding(op.name, op.n_rows, p.reps).to(vdev) for op in mats}
        _sync(prover.device)
        t0 = time.perf_counter()
        if mode == "C":
            us = prover.fold(chis)
            _sync(prover.device)
            t["prove_fold"] = time.perf_counter() - t0
            t["verify_fold"] = 0.0
        else:  # K: the verifier folds its own copy of the weights
            us = {op.name: verifier._fold_local(op, chis[op.name]) for op in mats}
            _sync(vdev)
            t["verify_fold"] = time.perf_counter() - t0
            t["prove_fold"] = 0.0
    if mode == "C":
        out["bytes"]["u"] = sum(u.numel() for u in us.values()) * 4
        if vdev.type != "cpu":
            t0 = time.perf_counter()
            us = {k: v.to(vdev) for k, v in us.items()}
            _sync(vdev)
            t["verify_upload"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    ok = verifier.check_products(claims_v, inputs, chis, us)
    _sync(vdev)
    t["verify_products"] = time.perf_counter() - t0
    if not ok:
        out["rejected_at"] = "freivalds"
        return out

    if mode == "C":
        if p.fiat_shamir:
            t0 = time.perf_counter()
            for k in sorted(us):
                ch.absorb(b"u/" + k.encode(), _tensor_blob(us[k]))
            t["fs_hash"] += time.perf_counter() - t0
        cols = {op.name: ch.columns(op.name, verifier.publics[op.name].n_points, p.columns) for op in mats}
        _sync(prover.device)
        t0 = time.perf_counter()
        openings = prover.open(cols)
        _sync(prover.device)
        t["prove_open"] = time.perf_counter() - t0
        out["bytes"]["columns"] = sum(o[0].numel() for o in openings.values()) * 4
        out["bytes"]["paths"] = sum(len(proof) * HASH_BYTES for _, proof in openings.values())
        t0 = time.perf_counter()
        reason = verifier.check_columns(chis, us, cols, openings)
        _sync(vdev)
        t["verify_columns"] = time.perf_counter() - t0
        if reason is not None:
            out["rejected_at"] = reason
            return out
    out["accepted"] = True
    return out
