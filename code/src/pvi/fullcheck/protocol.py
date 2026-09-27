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
from dataclasses import dataclass, field

import numpy as np
import torch

from .commitment import (HASH_BYTES, CommitmentPublic, WeightCommitment, column_leaf, vandermonde_columns,
                         verify_multiproof)
from .field import LOG2_P, P, field_matmul_mod, small_matmul_mod, to_field
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

    def claims(self, x: torch.Tensor, **forward_kwargs) -> dict[str, torch.Tensor]:
        if self.lean:
            forward_kwargs = dict(forward_kwargs, free=True, claims_device="cpu")
        _, claims = self.graph.forward(x.to(self.device), **forward_kwargs)
        return {k: v.to("cpu") for k, v in claims.items()}

    def _weight(self, name: str) -> torch.Tensor:
        # the int8 weights the forward pass already keeps on the device
        return self._ops[name]._weights_on(self.device)[0]

    def fold(self, chis: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.commitments[name].fold(chi, self.device, weight=self._weight(name))
                for name, chi in chis.items()}

    def open(self, cols: dict[str, torch.Tensor]):
        return {name: self.commitments[name].open(idx, self.device, weight=self._weight(name))
                for name, idx in cols.items()}


def _rhs(op: MatOp, u: torch.Tensor, xin: torch.Tensor) -> torch.Tensor:
    """``u^T [X ; 1]`` over the field, shape ``[r, M]``."""
    k = op.n_in
    if op.layout == "embed":
        out = u[:, xin.reshape(-1)]
    else:
        x = op.unfold(xin)  # [K, M] small ints
        out = small_matmul_mod(x.T.contiguous(), u[:, :k].T.contiguous()).T
    if op.has_bias:
        out = (out + u[:, k:k + 1]) % P
    return out


def _in_range(z: torch.Tensor, bound: int) -> bool:
    # written without abs(): torch.abs(INT64_MIN) overflows back to INT64_MIN
    return not bool(((z >= bound) | (z <= -bound)).any())


@dataclass
class Verifier:
    graph: IntGraph
    params: SecurityParams
    mode: str = "C"
    publics: dict[str, CommitmentPublic] = field(default_factory=dict)
    weights: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = field(default_factory=dict)
    lean: bool = False
    device: str = "cpu"      # "cuda" / "cuda:1": a client with a GPU (Merkle hashing stays on the CPU)
    _pre: dict = field(default_factory=dict)
    _wdev: dict = field(default_factory=dict)   # mode K on a GPU: the weights stay on the device

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
        """Recompute every cheap op; return each weight op's input, or ``None`` to reject."""
        env = {self.graph.input_name: x}
        inputs: dict[str, torch.Tensor] = {}
        last = self.graph.last_use() if self.lean else None
        for i, op in enumerate(self.graph.ops):
            if last is not None and i > 0:  # free what no later op reads (weight-op inputs stay in ``inputs``)
                for n in self.graph.ops[i - 1].inputs:
                    if last.get(n, -1) <= i - 1:
                        env.pop(n, None)
            if isinstance(op, MatOp):
                z = claims.get(op.name)
                xin = env[op.inputs[0]]
                m = op.n_cols(xin)
                if (z is None or m is None or z.dtype != torch.int64 or tuple(z.shape) != (op.n_rows, m)
                        or not _in_range(z, Z_BOUND)):
                    return None
                inputs[op.name] = xin
                env[op.output] = op.fold(z, xin)
            else:
                env[op.output] = op.fn(*[env[n] for n in op.inputs])
        return inputs

    def check_products(self, claims, inputs, chis, us) -> bool:
        for op in self.graph.mat_ops:
            u = us.get(op.name)
            if (u is None or u.dtype != torch.int64 or tuple(u.shape) != (self.params.reps, op.row_length)
                    or bool(((u < 0) | (u >= P)).any())):
                return False
            lhs = field_matmul_mod(chis[op.name], to_field(claims[op.name]))
            if not torch.equal(lhs, _rhs(op, us[op.name], inputs[op.name])):
                return False
        return True

    def check_columns(self, chis, us, cols, openings) -> str | None:
        """``None`` if every opened column is consistent, else the failing check."""
        for op in self.graph.mat_ops:
            pub = self.publics[op.name]
            opened, proof = openings[op.name]
            idx = cols[op.name]
            if (tuple(opened.shape) != (pub.n_rows, len(idx)) or len(proof) > len(idx) * pub.depth
                    or opened.dtype != torch.int64 or bool(((opened < 0) | (opened >= P)).any())
                    or us[op.name].shape[1] != pub.row_length):
                return "columns_shape"
            # Enc(u) is only needed at the t opened columns: evaluate it there
            # directly (r*k*t work) instead of re-encoding the whole codeword.
            dev = us[op.name].device
            enc = field_matmul_mod(us[op.name], vandermonde_columns(pub.n_points, pub.row_length, idx, dev))
            if not torch.equal(field_matmul_mod(chis[op.name], opened.to(dev)), enc):
                return "columns_code"
            col_np = opened.to(torch.int64).cpu().numpy()
            leaves = {c: column_leaf(pub.tag, c, col_np[:, j]) for j, c in enumerate(idx.tolist())}
            if not verify_multiproof(pub.root, pub.depth, leaves, proof):
                return "columns_merkle"
        return None


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
    and proof bytes.  With Fiat--Shamir, ``fs_hash`` is paid by both parties."""
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
