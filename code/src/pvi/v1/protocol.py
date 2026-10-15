"""V1's query: the cut operations' requantised values, proved with a windowed logUp-GKR and no new commitment.

The messages, in order (under Fiat-Shamir each is absorbed before the next challenge is drawn):

* the values frame (``PVC3``: the sent values ``s`` of the cut operations, the claims ``Z`` of the clear ones, and
  the lookup rows of a commitment plan, as in v0), the exceptions frame ``PVX1`` (every clamped entry outside its
  normal window, with its exact ``Z``) and the table's multiplicities ``m`` (``PVC3`` of one row);
* the logUp point ``alpha``, in ``F_{p^8}`` outside ``F_p + xi F_p``, and in mode C v0's fold challenges ``chi`` of
  the clear operations;
* one fractional-sum GKR transcript per instance (:mod:`pvi.v1.gkr`), round by round;
* the folds at each instance's final point: in mode C ``u = chi^T A`` (8 base-field planes) for a cut operation in
  the row layout and ``y = A xbar`` for a cut transposed matrix, in Kpre ``y = A xbar`` for every cut operation;
  and v0's folded rows ``u`` of the clear operations;
* in mode C, the column indices and the openings, as in v0.

The verifier's checks, in order, with their rejection labels: (D1) and (D2) ``range_or_shape`` (decoding and the
derivation, with the cut operations' values and inputs in their ranges), (D3) ``token``, the lookup checks, (D4)
``cut_exception``, (D5) ``cut_multiplicities``, (G) ``cut_gkr``, (F1) ``freivalds`` (the clear operations), (F2)
``kpre_y``, (F3) ``cut_final``, (F4) ``cut_logup``, then the column checks (C).  The verifier here is the
non-streaming one (CPU or GPU tensors, with the GKR and the final checks on the CPU); mode K and the streaming
verifier do not support V1 (``NotImplementedError``).

Interactive challenges are the verifier's own (drawn from a CSPRNG at the moment each is needed).  Since both
parties run in one process, the GKR's round challenges are drawn when the prover asks for them, after its message,
and the verifier checks against the ones it drew (:class:`_Recorder`, :class:`_Replayer`).  Under Fiat-Shamir the
prover runs the GKR on a fork of the transcript and the verifier re-derives every challenge from its own.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import torch

from pvi import options
from pvi.commit.commitment import HASH_BYTES, multiproof_size
from pvi.core import contention
from pvi.core import extfield as ef
from pvi.core.field import P, field_matmul_mod, min_max, to_field
from pvi.v0 import claimcodec
from pvi.v0.prover import Prover
from pvi.v0.query import (absorb_statement, decoded_claims, encoded_message, field_message, open_message, passed,
                          phase_done, proof_hashes)
from pvi.v0.transcript import Challenger
from pvi.v0.verifier import Verifier, in_field, sync_device
from pvi.v1 import cut as cutmod
from pvi.v1 import gkr, logup
from pvi.v1 import products

__all__ = ["run_cut_query", "cut_proof_bytes", "cut_plan", "pack_exceptions", "unpack_exceptions", "EXC_MAGIC"]

EXC_MAGIC = b"PVX1"
_FINAL_ROWS = 256
"""Rows of a cut operation per block of the baseline final check (at ``T = 2,048``, ``[L ; W]`` of 256 rows is 8 MB
of float64)."""


# ------------------------------------------------------------------ plans and the exceptions frame
def cut_plan(graph, columns, T: int, matrices, lmax: int, weights: dict | None = None) -> cutmod.CutPlan:
    """The query's cut plan.  ``matrices``: a ``CommitmentPlan`` (the prover's), the verifier's ``publics`` (a dict
    of ``CommitmentPublic``, whose transposed matrices carry ``members``), or ``None``; ``weights``: a mode K or Kpre
    verifier's, for (P4)."""
    return cutmod.CutPlan.from_graph(graph, columns, T, _cols_of(matrices), lmax=lmax, weights=weights)


def _cols_of(matrices):
    if matrices is None:
        return None
    if hasattr(matrices, "matrices"):
        return {m.name: tuple(m.members) for m in matrices.matrices if m.layout == "col"}
    return {name: tuple(pub.members) for name, pub in matrices.items() if getattr(pub, "members", ())}


def pack_exceptions(plan: cutmod.CutPlan, exc: dict) -> bytes:
    """``PVX1``: the magic, every cut operation's exception count (u32, in plan order), then per operation its flat
    indices (u32, ascending) and exact claims (i32)."""
    counts = [int(exc[o.name][0].numel()) for o in plan.ops]
    parts = [EXC_MAGIC, struct.pack(f"<{len(counts)}I", *counts)]
    for o in plan.ops:
        idx, z = exc[o.name]
        parts.append(idx.cpu().numpy().astype("<u4").tobytes())
        parts.append(z.cpu().numpy().astype("<i4").tobytes())
    return b"".join(parts)


def unpack_exceptions(buf, plan: cutmod.CutPlan) -> dict | None:
    """The exceptions of :func:`pack_exceptions`'s bytes as ``{op: (idx, z)}`` int64 tensors, or ``None`` if the
    bytes are not such a frame for ``plan`` (lengths; every count at most the operation's entries)."""
    buf = bytes(buf)
    n = len(plan.ops)
    head = 4 + 4 * n
    if len(buf) < head or buf[:4] != EXC_MAGIC:
        return None
    counts = struct.unpack_from(f"<{n}I", buf, 4)
    if any(c > o.n_rows * plan.T for c, o in zip(counts, plan.ops)) or len(buf) != head + 8 * sum(counts):
        return None
    out, at = {}, head
    for c, o in zip(counts, plan.ops):
        idx = np.frombuffer(buf, dtype="<u4", count=c, offset=at).astype(np.int64)
        z = np.frombuffer(buf, dtype="<i4", count=c, offset=at + 4 * c).astype(np.int64)
        out[o.name] = (torch.from_numpy(idx.copy()), torch.from_numpy(z.copy()))
        at += 8 * c
    return out


def exceptions_ok(plan: cutmod.CutPlan, exc: dict, values: dict) -> bool:
    """Check (D4) (``cut_exception``): only requantisations list exceptions, at strictly increasing indices inside
    the operation, each with ``|z| < 2^29`` and ``rq(z)`` equal to the sent value there."""
    for o in plan.ops:
        idx, z = exc[o.name]
        if not idx.numel():
            continue
        if o.kind != "requant":
            return False
        if bool((idx[1:] <= idx[:-1]).any()) or int(idx[-1]) >= o.n_rows * plan.T:
            return False
        if bool((z.abs() >= cutmod.CLAIM_LIMIT).any()):
            return False
        s = values[o.name].reshape(-1)[idx.to(values[o.name].device)].cpu()
        if not torch.equal(cutmod.requant_values(z, o), s.to(torch.int64)):
            return False
    return True


# ------------------------------------------------------------------ challenges of the GKR in one process
class _Recorder:
    """The prover's challenger in an interactive GKR: each challenge is the verifier's, drawn (``ch.ext``) when the
    prover asks for it, after the message it follows, and recorded for the verifier."""

    def __init__(self, ch: Challenger) -> None:
        self.ch, self.log = ch, []

    def absorb(self, label: bytes, blob: bytes) -> None:
        pass

    def ext(self, name: str, count: int = 1, *, nonbase: bool = False) -> torch.Tensor:
        v = self.ch.ext(name, count, nonbase=nonbase)
        self.log.append((name, v.clone()))
        return v


class _Replayer:
    """The verifier's challenger in an interactive GKR: the challenges it drew, in order."""

    def __init__(self, ch: Challenger, log: list) -> None:
        self.ch, self.log, self.i = ch, log, 0

    def absorb(self, label: bytes, blob: bytes) -> None:
        self.ch.absorb(label, blob)

    def ext(self, name: str, count: int = 1, *, nonbase: bool = False) -> torch.Tensor:
        label, v = self.log[self.i]
        if label != name:
            raise RuntimeError("the GKR's challenges were drawn in another order")
        self.i += 1
        return v.clone()


# ------------------------------------------------------------------ algebra helpers
def _planes(v: torch.Tensor) -> torch.Tensor:
    """A vector over ``F_{p^8}`` ``[n, 8]`` as its 8 base-field planes ``[8, n]``."""
    return v.T.contiguous()


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum_i a_i b_i`` in ``F_{p^8}`` for ``a, b`` ``[n, 8]`` (one exact 8 x 8 product,
    ``pvi.core.extfield.inner``)."""
    return ef.inner(a.cpu(), b.cpu())


def _xbar(op, xin: torch.Tensor, e: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
    """``[X ; 1] e`` ``[K (+1), 8]`` of a linear operation's input ``xin`` (its ``[M, K]`` rows) at the column point
    ``e``: the product by value of the optimised verifier (:func:`pvi.v1.products.xbar`, the same integers) when the
    input is small, else ``pvi.core.extfield.int_times`` for entries below ``2^31`` and the general product otherwise;
    operations reading the same input at the same point (q, k, v; gate, up) share it through ``cache``."""
    key = (xin.data_ptr(), tuple(xin.shape), id(e))
    if cache is not None and key in cache:
        out = cache[key]
    else:
        out = products.xbar(xin.reshape(-1, op.n_in), e) if products.optimised() else None
        if out is None:
            x = xin.reshape(-1, op.n_in).cpu()
            if x.numel() and int(x.abs().max()) >= 1 << 31:          # not an int32-sized input: the general product
                out = ef.small_times(x.T.contiguous().to(torch.int64), e.cpu())
            else:
                out = ef.int_times(x.to(torch.float64).T, e.cpu(), 31)
        if cache is not None:
            cache[key] = out
    return torch.cat([out, (e.sum(0) % P)[None]]) if op.has_bias else out


def _points(plan: cutmod.CutPlan, beta: int, rho: torch.Tensor):
    """Instance ``beta``'s column point ``e = eq(r_col)[0:T]`` ``[T, 8]`` and row table ``eq(r_row)``."""
    n_row, n_col = plan.n_vars(beta)
    return ef.eq_table(rho[:n_col])[:plan.T], ef.eq_table(rho[n_col:])


# ------------------------------------------------------------------ the query
def run_cut_query(prover: Prover, verifier: Verifier, x: torch.Tensor, *, seed: int | None = None,
                  forward_kwargs: dict | None = None, cheat: dict | None = None, sampler=None) -> dict:
    """One V1 query (always on the wire: ``PVC3`` values, packed field elements).  Returns what
    :func:`pvi.v0.query.run_query` returns, with the cut's bytes (``cut_exc``, ``cut_mult``, ``cut_gkr``) and
    timings.

    ``cheat`` (tests only) makes the prover dishonest at one message: ``"witness"(values, witness, exc, mult)``
    edits its sent values, leaves, exceptions or counts in place; ``"gkr"(blobs)``, ``"fold"(parts)`` return
    other transcripts or fold messages; ``forward_kwargs={"tamper": ...}`` runs the trace-tampering attack."""
    p = verifier.params
    if verifier.mode not in ("C", "Kpre") or verifier.stream:
        raise NotImplementedError("cut: modes C and Kpre, non-streaming verifier only in v1")
    if x.dim() != 2 or x.shape[0] != 1:
        raise NotImplementedError("cut: one prompt per query")
    ch = Challenger(fiat_shamir=p.fiat_shamir, seed=seed)
    out = {"accepted": False, "rejected_at": None, "timings": {}, "contention": {}}
    T = x.shape[1]
    cols = verifier.claim_columns(x)
    if cols is None:
        out["rejected_at"] = "range_or_shape"
        return out
    plan = cut_plan(verifier.graph, cols, T, verifier.publics if verifier.groups else None, p.cut_lmax,
                    verifier.weights or None)
    mine = cut_plan(prover.graph, prover.graph.claim_columns(x), T, getattr(prover.commitments, "plan", None),
                    p.cut_lmax)
    if mine.digest() != plan.digest():
        raise ValueError("prover and verifier derive different cut plans")
    verifier._cut_plan, verifier.sampler = plan, sampler
    try:
        out["rejected_at"] = _query(prover, verifier, x, ch, out, plan, forward_kwargs or {}, cheat or {})
    finally:
        verifier._cut_plan = verifier.sampler = None
    out["accepted"] = out["rejected_at"] is None
    out["cut"] = {"n_ops": len(plan.ops), "n_instances": len(plan.instances), "n_leaves": plan.n_leaves(),
                  "table": plan.table_size(), "n_rounds": sum(n * (n - 1) // 2 for n in cutmod.instance_bits(plan))}
    return out


def cut_proof_bytes(prover: Prover, verifier: Verifier, x: torch.Tensor, *, seed: int | None = None) -> dict:
    """V1's proof bytes on the query ``x``, message by message, without running the GKR, the folds or the openings
    (for models whose GKR the eager prover cannot hold).  The values, the exceptions and the multiplicities are
    produced and encoded exactly as in :func:`run_cut_query`; the GKR transcripts, the fold message and the opened
    columns have sizes fixed by the cut plan and the parameters; the multiproofs are sized
    (``pvi.commit.commitment.multiproof_size``) at columns drawn as a query draws them.
    ``tests/test_v1_protocol.py`` checks every size except that of the paths against a real query."""
    p = verifier.params
    T = x.shape[1]
    plan = cut_plan(verifier.graph, verifier.claim_columns(x), T, verifier.publics if verifier.groups else None,
                    p.cut_lmax, verifier.weights or None)
    verifier._cut_plan = plan
    try:
        _, offsets = plan.table()
        narrow = claimcodec.narrow if prover.lean and prover.device.type != "cpu" else None
        zs = prover.claims(passed(x), send=narrow, to_host=False)
        mult = torch.zeros(plan.table_size(), dtype=torch.int64)
        exc, values = {}, {}
        sent = {op.name for op in verifier._sent_ops()}
        mult = mult.to(prover.device)
        for o in plan.ops:
            s, d, w, idx, ez, counts = cutmod.split_counts(zs[o.name].to(prover.device), o, offsets,
                                                           plan.table_size())
            mult += counts
            exc[o.name] = (idx.cpu(), ez.cpu())
            values[o.name] = s
            del d, w, counts
        mult = mult.cpu()
        for k, z in zs.items():
            if k in sent and k not in values:
                values[k] = z
        rows = (encoded_message(claimcodec.pack_rows, [values[name] for name in verifier.tables])
                if verifier.tables else b"")
        blob = encoded_message(claimcodec.encode, [values[op.name] for op in prover.graph.mat_ops
                                            if op.name in sent and op.name not in verifier.tables])
        proofs = prover.open_tables() if verifier.tables else {}
        clear_rows = verifier._row_ops()
        items = _fold_items(verifier, plan, _cut_col_matrices(verifier, plan))
        n_fold = (sum(p.reps * op.row_length for op in clear_rows) if verifier.mode == "C" else 0)
        n_fold += sum(8 * _fold_len(verifier, plan, key, members) for key, members in items)
        out = {"claims": len(blob) + len(rows), "cut_exc": len(pack_exceptions(plan, exc)),
               "cut_mult": len(encoded_message(claimcodec.encode, [mult[None]])),
               "cut_gkr": sum(1 + claimcodec.field_size(8 * gkr.n_elements(n)) for n in cutmod.instance_bits(plan)),
               "u": claimcodec.field_size(n_fold), "columns": 0,
               "paths": HASH_BYTES * sum(len(proof_hashes(pr)) for pr in proofs.values())}
        if verifier.mode == "C":
            cols = verifier.column_challenges(Challenger(seed=seed))
            units = verifier._column_units()
            out["columns"] = claimcodec.field_size(sum(
                len(cols[name]) * sum(verifier.publics[m].n_rows for m in members) for name, members in units))
            out["paths"] += HASH_BYTES * sum(multiproof_size(cols[name].tolist(), verifier._tree(name).depth)
                                             for name, _ in units)
        out["n_leaves"], out["n_instances"], out["table"] = plan.n_leaves(), len(plan.instances), plan.table_size()
        out["n_exceptions"] = sum(int(i.numel()) for i, _ in exc.values())
        return out
    finally:
        verifier._cut_plan = None


def _timed(out: dict, name: str):
    """A context timing one phase into ``out["timings"][name]`` (added to)."""
    class _T:
        def __enter__(self):
            self.c0, self.t0 = contention.begin(), time.perf_counter()

        def __exit__(self, *exc):
            out["timings"][name] = out["timings"].get(name, 0.0) + time.perf_counter() - self.t0
            phase_done(out, name, self.c0)
    return _T()


def _query(prover: Prover, verifier: Verifier, x: torch.Tensor, ch: Challenger, out: dict, plan, forward_kwargs,
           cheat: dict):
    p = verifier.params
    tau, offsets = plan.table()
    cut_names = plan.names()
    # ---------------------------------------------------------------- the values, exceptions, multiplicities (prover)
    query = passed(x)
    sync_device(prover.device)
    with _timed(out, "prove_forward"):
        # as in the wire path of pvi.v0.query.run_query: a lean GPU prover keeps each claim as int32 (claimcodec.narrow)
        narrow = claimcodec.narrow if prover.lean and prover.device.type != "cpu" else None
        zs = prover.claims(query, send=narrow, to_host=False, **forward_kwargs)
        sync_device(prover.device)
    witness, exc = {}, {}
    mult = torch.zeros(plan.table_size(), dtype=torch.int64, device=prover.device)
    with _timed(out, "prove_cut_split"):
        sent_vals = {}
        for o in plan.ops:                  # on the prover's device (a lean prover's host claims go up one by one)
            s, d, w, idx, ez, counts = cutmod.split_counts(zs[o.name].to(prover.device), o, offsets,
                                                           plan.table_size())
            mult += counts
            witness[o.name] = (d.to(torch.int32).cpu(), w.to(torch.int32).cpu())     # 0 <= delta < W <= 2^16
            exc[o.name] = (idx.cpu(), ez.cpu())
            # where the prover keeps its claims (a lean one: the host, int32), so the encoder sees them all alike
            sent_vals[o.name] = s.to(zs[o.name].device, zs[o.name].dtype)
            del d, w, counts
        mult = mult.cpu()
        sync_device(prover.device)
    sent = {op.name for op in verifier._sent_ops()}
    values = {k: (sent_vals[k] if k in cut_names else z) for k, z in zs.items() if k in sent}
    if "witness" in cheat:
        cheat["witness"](values, witness, exc, mult)
    proofs = passed(prover.open_tables()) if verifier.tables else {}
    with _timed(out, "prove_encode"):
        rows = (encoded_message(claimcodec.pack_rows, [values[name] for name in verifier.tables])
                if verifier.tables else None)
        blob = encoded_message(claimcodec.encode, [values[op.name] for op in prover.graph.mat_ops
                                            if op.name in sent and op.name not in verifier.tables])
        exc_blob = pack_exceptions(plan, exc)
        mult_blob = encoded_message(claimcodec.encode, [mult[None]])
        sync_device(prover.device)
    out["bytes"] = {"claims": len(blob) + len(rows or b""), "cut_exc": len(exc_blob), "cut_mult": len(mult_blob),
                    "cut_gkr": 0, "u": 0, "columns": 0,
                    "paths": HASH_BYTES * sum(len(proof_hashes(pr)) for pr in proofs.values())}
    with _timed(out, "fs_hash"):
        if p.fiat_shamir:
            absorb_statement(ch, verifier, x)
            ch.absorb(b"cut", plan.digest())
            ch.absorb_large(b"claims/" + claimcodec.MAGIC, blob)
            if rows is not None:
                ch.absorb_large(b"rows/I8", rows)
            for name in verifier.tables:
                hashes = proof_hashes(proofs.get(name))
                ch.absorb(b"lookup/" + name.encode(), struct.pack("<Q", len(hashes)) + b"".join(hashes))
            ch.absorb_large(b"cut/exc", exc_blob)
            ch.absorb_large(b"cut/mult", mult_blob)

    # ---------------------------------------------------------------- the values, exceptions, multiplicities (verifier)
    narrow = products.optimised()            # the cut operations' values kept as int8 (int16) once in range
    claims_v = decoded_claims(verifier, blob, rows, x, out, torch.int32 if narrow else torch.int64)
    if claims_v is None:
        return "range_or_shape"
    if narrow:
        with _timed(out, "verify_decode"):
            claims_v = _narrowed(plan, claims_v)
    x_v = x.cpu()
    if not verifier.check_tokens(x_v, claims_v):
        return "token"
    vdev = torch.device(verifier.device)
    if vdev.type != "cpu":
        with _timed(out, "verify_upload"):
            x_v, claims_v = x.to(vdev), {k: v.to(vdev) for k, v in claims_v.items()}
            sync_device(vdev)
    with _timed(out, "verify_derive"):
        inputs = verifier.derive(x_v, claims_v)
        sync_device(vdev)
    if inputs is None:
        return "range_or_shape"
    if verifier.tables:
        with _timed(out, "verify_lookups"):
            reason = verifier.check_lookups(claims_v, inputs, proofs)
        if reason is not None:
            return reason
    with _timed(out, "verify_cut_m1"):
        exc_v = unpack_exceptions(exc_blob, plan)
        exc_ok = exc_v is not None and exceptions_ok(plan, exc_v, claims_v)
        mult_v = None
        if exc_ok:
            try:
                mult_v = claimcodec.decode_torch(mult_blob, [1], [plan.table_size()], dtype=torch.int64)[0][0]
            except claimcodec.ClaimCodecError:
                mult_v = None
    if not exc_ok:
        return "cut_exception"
    if mult_v is None or not logup.multiplicities_ok(mult_v, plan.table_size(), plan.n_leaves()):
        return "cut_multiplicities"

    # ---------------------------------------------------------------- the logUp point and the GKR transcripts
    alpha = ch.ext("cut/alpha", nonbase=True)[0]
    chis = ({name: chi.to(vdev) for name, chi in verifier.fold_challenges(ch, x).items()}
            if verifier.mode == "C" else {})
    ch_p = ch.fork() if p.fiat_shamir else _Recorder(ch)
    on_gpu = False
    if options.current().gkr_triton and prover.device.type == "cuda":
        from pvi.v1 import gkr_triton
        on_gpu = gkr_triton.available(prover.device)
    if on_gpu:
        # free the forward pass's cached blocks: the Triton GKR's fraction tree alone takes up to 128 bytes per leaf
        # (numerators and denominators as 8 int32 planes each, over all its layers)
        torch.cuda.empty_cache()
    with _timed(out, "prove_gkr"):
        trs = []
        for b, names in enumerate(plan.instances):
            n_row, n_col = plan.n_vars(b)
            d = torch.cat([witness[o][0] for o in names])
            w = torch.cat([witness[o][1] for o in names])
            if on_gpu:
                trs.append(gkr_triton.prove_leaves(d, w, n_col, n_row + n_col, alpha, ch_p, f"cut/{b}/",
                                                   device=prover.device))
            else:
                pq = logup.leaves(d, w, n_col, n_row + n_col, alpha)
                trs.append(gkr.prove(*pq, ch_p, f"cut/{b}/"))
                del pq
        sync_device(prover.device)
    gkr_blobs = [tr.to_bytes() for tr in trs]
    if "gkr" in cheat:
        gkr_blobs = cheat["gkr"](gkr_blobs)
    out["bytes"]["cut_gkr"] = sum(len(g) for g in gkr_blobs)
    ns = cutmod.instance_bits(plan)
    with _timed(out, "verify_gkr"):
        try:
            parsed = [gkr.Transcript.from_bytes(g, n) for g, n in zip(gkr_blobs, ns)]
        except (ValueError, claimcodec.ClaimCodecError):
            parsed = None
        ch_v = ch if p.fiat_shamir else _Replayer(ch, ch_p.log)
        outs = None if parsed is None else logup.verify_instances(parsed, ns, ch_v)
    if outs is None:
        return "cut_gkr"

    # ---------------------------------------------------------------- the folds (prover)
    clear_rows = verifier._row_ops()
    col_cut = _cut_col_matrices(verifier, plan)
    with _timed(out, "prove_fold"):
        us_clear = (prover.fold(passed({op.name: chis[op.name] for op in clear_rows}))
                    if verifier.mode == "C" else {})
        folds = {}                                          # name -> planes [8, n]
        for b, tr in enumerate(trs):
            e, chi_full = _points(plan, b, tr.point)
            for o in plan.instance_ops(b):
                if verifier.mode == "C" and o.layout == "row":
                    chi = chi_full[o.row_offset:o.row_offset + o.n_rows]
                    folds[o.name] = prover.fold({o.name: _planes(chi).to(prover.device)})[o.name].cpu()
                else:                                       # y = A xbar = Z e (|Z| < 2^29), where Z is kept
                    folds[o.name] = _planes(ef.int_times(zs[o.name], e, 30)).cpu()
        order = [op.name for op in clear_rows] if verifier.mode == "C" else []
        items = _fold_items(verifier, plan, col_cut)
        parts = [us_clear[n] for n in order] + [torch.cat([folds[o] for o in members], 1) for _, members in items]
        if "fold" in cheat:
            parts = cheat["fold"](parts)
        fold_blob = encoded_message(claimcodec.pack_field, parts)
    out["bytes"]["u"] = len(fold_blob)

    # ---------------------------------------------------------------- the folds (verifier)
    shapes = ([(p.reps, op.row_length) for op in clear_rows] if verifier.mode == "C" else [])
    shapes += [(8, _fold_len(verifier, plan, key, members)) for key, members in items]
    got = field_message(fold_blob, shapes, out, torch.int64)
    if got is None or not all(in_field(t) for t in got[len(order) if verifier.mode == "C" else 0:]):
        # malformed, or a cut fold outside [0, p) (the clear operations' u are checked by (F1)): canonical encodings
        return "freivalds" if verifier.mode == "C" and clear_rows and got is None else "cut_final"
    us = {n: u.to(vdev) for n, u in zip(order, got[:len(order)])}
    cut_folds = dict(zip((key for key, _ in items), got[len(order):]))
    if verifier.mode == "Kpre":
        chis = {op.name: verifier._pre[op.name][0] for op in clear_rows}
        us = {op.name: verifier._pre[op.name][1] for op in clear_rows}
        if p.fiat_shamir:
            ch.absorb_large(b"u/F31", fold_blob)
    with _timed(out, "verify_products"):
        ok = verifier.check_products(claims_v, inputs, chis, us)
        if ok and verifier.mode == "C":
            lefts, sources = verifier.column_operands(claims_v, inputs, chis, us)
    if not ok:
        return "freivalds"
    with _timed(out, "verify_final"):
        xbars, ys = {}, {}
        mats = {op.name: op for op in verifier.graph.mat_ops}
        points = [_points(plan, b, o[2]) for b, o in enumerate(outs)]      # each instance's e and eq(r_row), once
        xcache: dict = {}
        for b in range(len(outs)):
            e, _ = points[b]
            for o in plan.instance_ops(b):
                xbars[o.name] = _xbar(mats[o.name], inputs[o.name], e, xcache)
        del xcache
        for key, members in items:
            planes = cut_folds[key]
            if not (verifier.mode == "C" and plan.op(members[0]).layout == "row"):     # y of these members
                at = 0
                for m in members:
                    n = plan.op(m).n_rows
                    ys[m] = planes[:, at:at + n].T.contiguous()
                    at += n
        if verifier.mode == "Kpre" and not _kpre_y_ok(verifier, plan, ys, xbars):
            return "kpre_y"
        if not _final_ok(verifier, plan, outs, alpha, claims_v, exc_v, cut_folds, items, ys, xbars, points):
            return "cut_final"
    with _timed(out, "verify_logup"):
        if not logup.logup_ok([(o[0], o[1]) for o in outs], mult_v, alpha, tau):
            return "cut_logup"
    if verifier.mode != "C":
        return None

    # ---------------------------------------------------------------- the columns
    for key, members in items:
        planes = cut_folds[key].to(vdev)
        if plan.op(members[0]).layout == "row":
            o = plan.op(members[0])
            _, chi_full = points[o.instance]
            lefts[key] = _planes(chi_full[o.row_offset:o.row_offset + o.n_rows]).to(vdev)
            sources[key] = planes
        else:
            lefts[key] = _planes(xbars[members[0]]).to(vdev)
            sources[key] = planes
    cols_idx, openings = open_message(prover, verifier, ch, us, out, u_blob=fold_blob)
    with _timed(out, "verify_columns"):
        reason = verifier.check_columns(lefts, sources, cols_idx, openings, wire=True)
        sync_device(vdev)
    return reason


def _narrowed(plan, claims: dict) -> dict:
    """The decoded claims (int32) as the optimised verifier keeps them: a cut operation's sent values, when every one
    lies in the operation's public range (a requantisation's clamp, a residual operation's ``R_Delta``; one exact
    min/max), in the narrowest of int8 and int16 that holds them (a requantisation's always fit int8); every other
    claim widened to int64 as the decoder does.  A cut operation with a value out of range, or one needing more than
    16 bits, stays int64, so that the range check (D1) of the derivation rejects it with the same label at the same
    point.  The derivation then reads the narrow values, range-checks them again and widens only the tensor their
    readers take; the exceptions and the final check read them as they are."""
    cut = {o.name: o for o in plan.ops}
    out = {}
    for name, z in claims.items():
        o = cut.get(name)
        if o is not None and z.numel():
            lo, hi = (o.lo_, o.hi) if o.kind == "requant" else cutmod.delta_range(o.mult, o.sh)
            amin, amax = min_max(z)
            nd = next((d for d in (torch.int8, torch.int16)
                       if torch.iinfo(d).min <= amin and amax <= torch.iinfo(d).max), None)
            if lo <= amin and amax <= hi and nd is not None:
                out[name] = z.to(nd)
                continue
        out[name] = z.to(torch.int64)
    return out


def _cut_col_matrices(verifier: Verifier, plan) -> dict:
    """Mode C: the cut transposed matrices ``{name: members}``, in plan order."""
    out = {}
    for o in plan.ops:
        if o.layout == "col":
            out.setdefault(o.matrix, []).append(o.name)
    return out


def _fold_items(verifier: Verifier, plan, col_cut: dict) -> list[tuple[str, list[str]]]:
    """The cut part of the fold message, in order, as ``(key, members)``: in mode C, the ``u`` of an operation in the
    row layout and the ``y`` of a transposed matrix (its members' rows together); in Kpre, every operation's
    ``y``."""
    items, seen = [], set()
    for o in plan.ops:
        if verifier.mode == "C" and o.layout == "col":
            if o.matrix not in seen:
                seen.add(o.matrix)
                items.append((o.matrix, col_cut[o.matrix]))
        else:
            items.append((o.name, [o.name]))
    return items


def _fold_len(verifier: Verifier, plan, key: str, members: list[str]) -> int:
    o = plan.op(members[0])
    if verifier.mode == "C" and o.layout == "row":
        return o.n_in + int(o.has_bias)
    return sum(plan.op(m).n_rows for m in members)


def _kpre_y_ok(verifier: Verifier, plan, ys: dict, xbars: dict) -> bool:
    """Check (F2) (``kpre_y``): ``chi_s^T y = u_s xbar`` in ``F_{p^8}`` for every cut operation and every secret
    vector ``chi_s`` with its folded row ``u_s``."""
    for o in plan.ops:
        chi_s, u_s = (t.cpu() for t in verifier._pre[o.name])
        y, xb = ys[o.name], xbars[o.name]
        left = field_matmul_mod(to_field(chi_s), to_field(y))
        right = field_matmul_mod(to_field(u_s), to_field(xb))     # small: r x (K + 1) times [K + 1, 8]
        if not torch.equal(left, right):
            return False
    return True


def _lw_e_old(vals: torch.Tensor, idx: torch.Tensor, z: torch.Tensor, o, e: torch.Tensor, T: int):
    """The baseline ``(L e, W e)`` ``[N, 8]`` of a cut operation (also for spans past ``pvi.v1.products.SPAN``):
    ``[L ; W]`` gathered in float64 from the window table, one exact GEMM with ``e``'s limbs (by (D1), (D4) and (P3),
    ``|L| < 2^29 + 2^16 < 2^30`` and ``W <= 2^16``), in row blocks of :data:`_FINAL_ROWS`."""
    le_parts, we_parts = [], []
    for r0 in range(0, o.n_rows, _FINAL_ROWS):      # row blocks: the window matrix stays in cache
        r1 = min(o.n_rows, r0 + _FINAL_ROWS)
        sel = (idx >= r0 * T) & (idx < r1 * T)
        lw = cutmod.public_lw_f64(vals[r0:r1], (idx[sel] - r0 * T).to(vals.device), z[sel].to(vals.device), o)
        lwe = ef.int_times(lw, e, 30).cpu()
        le_parts.append(lwe[:r1 - r0])
        we_parts.append(lwe[r1 - r0:])
        del lw
    if not le_parts:
        return torch.zeros(0, ef.D, dtype=torch.int64), torch.zeros(0, ef.D, dtype=torch.int64)
    return torch.cat(le_parts), torch.cat(we_parts)


def _final_claims(verifier, plan, outs, alpha, values, exc, cut_folds, items, ys, xbars, points):
    """Per instance, in order (a generator): ``(I(rho), q)``, the leaf layer's extensions at the GKR's final point
    computed from the sent values, the exceptions and the folds, which check (F3) compares with ``(P_hat, Q_hat)``:
    ``q = alpha I - sum_l (S_l - chi (L e) + xi chi (W e)) + (1 - I)``, where ``S_l`` is ``u_l xbar_l`` in the row
    layout of mode C and ``chi^T y_l`` otherwise.  ``L e`` and ``W e`` ``[N, 8]`` are the optimised verifier's
    products by value (:func:`pvi.v1.products.lw_times`, the same integers), or :func:`_lw_e_old` for the baseline
    verifier and for a residual operation whose values span past ``pvi.v1.products.SPAN``; ``chi (L e)`` and
    ``chi (W e)`` are then inner products."""
    us_row = {}
    for key, members in items:
        o = plan.op(members[0])
        if verifier.mode == "C" and o.layout == "row":
            us_row[o.name] = cut_folds[key].T.contiguous()            # [K (+1), 8]
    x_gen = ef.gen()
    vdev = torch.device(verifier.device)
    new = products.optimised()
    for b, (p0, q0, rho, p_hat, q_hat) in enumerate(outs):
        e, chi_full = points[b]
        total = ef.const(0)
        for o in plan.instance_ops(b):
            chi = chi_full[o.row_offset:o.row_offset + o.n_rows]
            idx, z = exc[o.name]
            vals = values[o.name].to(vdev)
            got = products.lw_times(vals, idx, z, o, e) if new else None
            le_vec, we_vec = got if got is not None else _lw_e_old(vals, idx, z, o, e, plan.T)
            s_l = _dot(us_row[o.name], xbars[o.name]) if o.name in us_row else _dot(chi, ys[o.name])
            le, we = _dot(chi, le_vec), _dot(chi, we_vec)
            total = ef.add(total, ef.add(ef.sub(s_l, le), ef.mul(x_gen, we)))
        i_b = logup.indicator(rho, plan.rows(b), plan.T, plan.n_vars(b)[1])
        yield i_b, ef.add(ef.sub(ef.mul(alpha, i_b), total), ef.sub(ef.const(1), i_b))


def _final_ok(verifier, plan, outs, alpha, values, exc, cut_folds, items, ys, xbars, points) -> bool:
    """Check (F3) (``cut_final``): every instance's final claims are the leaf layer's extensions computed from the
    sent values, the exceptions and the folds (:func:`_final_claims`), instance by instance up to the first that is
    not."""
    claims = _final_claims(verifier, plan, outs, alpha, values, exc, cut_folds, items, ys, xbars, points)
    for (i_b, want_q), (_, _, _, p_hat, q_hat) in zip(claims, outs):
        if not (ef.equal(p_hat, i_b) and ef.equal(q_hat, want_q)):
            return False
    return True
