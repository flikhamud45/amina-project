"""The streaming verifier: :func:`protocol.run_query`'s checks, verdicts and rejection labels,
with the verifier's work arranged for a GPU client.  Optional: ``Verifier(stream=True)``.

* The claims travel as int32 (:func:`wire_claim`: the 4 bytes per value the proof size
  already counts), narrowed on the prover's device and received into pinned host memory.
  The verifier uploads them one weight op at a time on a side CUDA stream, ``AHEAD`` ops
  before derive needs them, so the upload overlaps derive.
* Each weight op is checked as soon as its claim is on the device: its range check, both sides
  of Freivalds' check (``chi^T Z``, and ``u^T [X ; 1]`` once per input shared by q/k/v or
  gate/up; int8 tensor cores where ``int8_ok``), then its fold into the next op's input.
  Nothing is kept for a later products pass and every tensor is freed after its last reader,
  so the device holds a window of claims and one block's activations, not the whole proof.
* In mode C the opened columns travel as int32 rows ``[t, N]`` (:func:`wire_openings`); their
  code checks are queued on the device first and their Merkle checks run on a host thread
  while the device derives.
* Every verdict stays on the device until ONE copy decides the query.

The messages and their order are those of ``run_query`` (claims, chi, u, columns, openings),
every challenge drawn after the message it follows, with the same transcript under
Fiat--Shamir; only the verifier's local computations are reordered, each a function of
messages it already holds.  The columns are asked for before the products are checked, which
changes no label: a query whose products fail is rejected at ``freivalds`` either way.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .commitment import HASH_BYTES
from .graph import MatOp
from .protocol import (Challenger, _absorb_statement, _columns_verdict, _disagrees, _leading_passes, _min_max,
                       _sync, _tensor_blob, _to_device, _to_host, _upload_rows)

__all__ = ["AHEAD", "wire_claim", "wire_openings", "verify_streaming", "run_query_streaming"]

AHEAD = 16
"""Weight ops whose claims are uploaded ahead of derive (about two transformer blocks)."""
_INT32 = (-(1 << 31), (1 << 31) - 1)
_NOT_INT8 = object()     # a weight op's input was not int8-valued: verify again without int8 GEMMs


def _fits_int32(t: torch.Tensor) -> bool:
    if t.numel() == 0:
        return True
    lo, hi = _min_max(t)
    return lo >= _INT32[0] and hi <= _INT32[1]


def wire_claim(z, pin: bool = False) -> torch.Tensor | None:
    """A claim as the streaming verifier receives it: int32 on the host (in pinned memory with
    ``pin``, ready for a non-blocking upload), narrowed where it was computed; ``None`` when it
    is not an int64 tensor of int32 values, which cannot be sent (``run_query`` rejects such a
    claim at ``range_or_shape``, and so does the streaming verifier)."""
    if not torch.is_tensor(z) or z.dtype != torch.int64 or not _fits_int32(z):
        return None
    z32 = z.to(torch.int32)
    if not (pin and torch.cuda.is_available()):
        return z32.cpu()
    return torch.empty(z.shape, dtype=torch.int32, pin_memory=True).copy_(z32)


def wire_openings(openings: dict) -> dict:
    """``{name: (rows, proof)}``: each opening's int64 columns ``[N, t]`` as the int32 rows ``[t, N]``
    the Merkle leaves hash, or ``None`` when they are not a 2-d int64 tensor of int32 values
    (rejected at ``columns_shape``, as ``run_query`` rejects them); anything that is not a pair
    is passed on unchanged (and raises where ``run_query`` raises)."""
    out = {}
    for name, item in openings.items():
        if isinstance(item, tuple) and len(item) == 2:
            o, proof = item
            ok = torch.is_tensor(o) and o.dtype == torch.int64 and o.dim() == 2 and _fits_int32(o)
            item = (o.to(torch.int32).T.contiguous() if ok else None, proof)
        out[name] = item
    return out


class _Uploads:
    """``get(name)`` of the claims on ``device``, asked for in weight-op order and uploaded
    ``AHEAD`` ops earlier: on a side CUDA stream (the compute stream waits for each copy), or
    directly off the GPU."""

    def __init__(self, claims: dict, names: list[str], device: torch.device) -> None:
        self.claims, self.names, self.device = claims, names, device
        self.index = {n: i for i, n in enumerate(names)}
        self.side = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.ready: dict[str, tuple] = {}
        self.issued = 0

    def _issue(self, upto: int) -> None:
        while self.issued < min(upto, len(self.names)):
            name = self.names[self.issued]
            self.issued += 1
            z = self.claims.get(name)
            if not torch.is_tensor(z):                 # the verifier rejects it
                self.ready[name] = (z, None)
            elif self.side is None:
                self.ready[name] = (z.to(self.device), None)
            else:
                with torch.cuda.stream(self.side):
                    zd = z.to(self.device, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(self.side)
                self.ready[name] = (zd, done)

    def get(self, name: str):
        self._issue(self.index[name] + 1 + AHEAD)
        z, done = self.ready.pop(name)
        if done is not None:
            main = torch.cuda.current_stream(self.device)
            main.wait_event(done)
            z.record_stream(main)     # allocated on the side stream, read on this one
        return z


def verify_streaming(verifier, x: torch.Tensor, claims: dict, chis: dict, us: dict,
                     cols: dict | None = None, openings: dict | None = None) -> str | None:
    """``None`` to accept, else the check that rejects, with ``run_query``'s labels: the verdict
    of ``derive``, ``check_products`` and (with ``openings``) ``check_columns`` on these messages.
    ``claims`` from :func:`wire_claim`, ``openings`` from :func:`wire_openings`."""
    reason = _verify(verifier, x, claims, chis, us, cols, openings, int8=True)
    if reason is _NOT_INT8:        # (the graphs here clamp every weight op's input to int8)
        reason = _verify(verifier, x, claims, chis, us, cols, openings, int8=False)
    return reason


def _verify(verifier, x, claims, chis, us, cols, openings, *, int8: bool):
    dev = torch.device(verifier.device)
    mats = verifier.graph.mat_ops
    index = {op.name: i for i, op in enumerate(mats)}
    us = {k: _to_device(v, dev) if torch.is_tensor(v) else v for k, v in us.items()}
    n_u, u_exc = _leading_passes(mats, lambda op: verifier._u_ok(op, us, values=False))
    sharing: dict[str, list[MatOp]] = {}     # linear ops reading one input: one right-hand side
    for op in mats[:n_u]:
        if op.layout == "linear":
            sharing.setdefault(op.inputs[0], []).append(op)
    right: dict = {}
    x_flags: list = []
    flags: list = []

    def visit(op: MatOp, z: torch.Tensor, xin: torch.Tensor) -> None:
        if index[op.name] >= n_u:          # the verdict no longer depends on this product
            return
        if op.name not in right:
            group = sharing.get(op.inputs[0], [op]) if op.layout == "linear" else [op]
            out, unchecked = verifier._rhs_all(group, {o.name: xin for o in group}, us, int8=int8)
            right.update(out)
            x_flags.extend(unchecked)
        left = verifier._signed_lhs({op.name: (chis[op.name], z)})[op.name]
        flags.append(_disagrees(us[op.name], left, right.pop(op.name)))

    with ThreadPoolExecutor(1) as host:
        # the column checks compare with Enc(u): only started when every u is well formed (with one
        # that is not, the verdict is freivalds or u_exc, and the columns are never looked at)
        well_formed = openings is not None and n_u == len(mats)
        columns = _start_columns(verifier, chis, us, cols, openings, host) if well_formed else None
        derived = verifier._derive(_to_device(x, dev), _Uploads(claims, [op.name for op in mats], dev),
                                   dtype=torch.int32, visit=visit)
        if derived is None:
            return "range_or_shape"
        pending = derived[1]
        got = _to_host(pending + x_flags + flags + (columns[2] if columns else []))   # the one round trip
        a, b, c = len(pending), len(pending) + len(x_flags), len(pending) + len(x_flags) + len(flags)
        if any(got[:a]):
            return "range_or_shape"
        if any(got[a:b]):
            return _NOT_INT8
        if any(got[b:c]) or (u_exc is None and n_u < len(mats)):
            return "freivalds"
        if u_exc is not None:
            raise u_exc
        if columns is None:
            return None
        n_ok, exc, _, merkle = columns
        return _columns_verdict([not bad for bad in got[c:]], merkle.result(), n_ok, exc, len(mats))


def _start_columns(verifier, chis, us, cols, openings, host: ThreadPoolExecutor):
    """The column checks, started: ``(n_ok, exception, code flags on the device, Merkle future)``
    for the leading ops whose openings are well formed."""
    mats = verifier.graph.mat_ops
    n_ok, exc = _leading_passes(mats, lambda op: verifier._columns_shape_ok(op, us, cols, openings, wire=True))
    good = mats[:n_ok]
    rows = {op.name: openings[op.name][0].numpy() for op in good}
    dev = torch.device(verifier.device)
    on_dev = _upload_rows(rows, dev) if dev.type != "cpu" else {op.name: openings[op.name][0] for op in good}
    code_flags = verifier._code_flags(good, chis, us, cols, on_dev)
    return n_ok, exc, code_flags, host.submit(verifier._merkle, good, cols, openings, rows)


def run_query_streaming(prover, verifier, x: torch.Tensor, *, seed: int | None = None,
                        forward_kwargs: dict | None = None) -> dict:
    """:func:`protocol.run_query` with :func:`verify_streaming`: the same messages, challenges,
    transcript, verdicts, labels and proof bytes.  ``verify_total`` times the verifier's work
    from the moment it holds the messages (its uploads included); the prover's phases and
    ``fs_hash`` are timed as in ``run_query``."""
    p, mode = verifier.params, verifier.mode
    ch = Challenger(fiat_shamir=p.fiat_shamir, seed=seed)
    t: dict[str, float] = {}
    vdev = torch.device(verifier.device)

    _sync(prover.device)
    t0 = time.perf_counter()
    claims = prover.claims(x, send=lambda z: wire_claim(z, pin=vdev.type == "cuda"), **(forward_kwargs or {}))
    _sync(prover.device)
    t["prove_forward"] = time.perf_counter() - t0
    out = {"accepted": False, "rejected_at": None, "timings": t,
           "bytes": {"claims": 4 * sum(z.numel() for z in claims.values() if z is not None),
                     "u": 0, "columns": 0, "paths": 0}}

    t0 = time.perf_counter()
    if p.fiat_shamir:
        _absorb_statement(ch, verifier, x)
        for k in sorted(claims):       # int32 claims hash as the int64 ones of run_query
            ch.absorb(b"claim/" + k.encode(), b"" if claims[k] is None else _tensor_blob(claims[k]))
    t["fs_hash"] = time.perf_counter() - t0

    mats = verifier.graph.mat_ops
    cols = openings = None
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
            t["prove_fold"], t["verify_fold"] = time.perf_counter() - t0, 0.0
        else:  # K: the verifier folds its own copy of the weights
            us = {op.name: verifier._fold_local(op, chis[op.name]) for op in mats}
            _sync(vdev)
            t["verify_fold"], t["prove_fold"] = time.perf_counter() - t0, 0.0
    if mode == "C":
        out["bytes"]["u"] = sum(u.numel() for u in us.values()) * 4
        t0 = time.perf_counter()
        if p.fiat_shamir:
            for k in sorted(us):
                ch.absorb(b"u/" + k.encode(), _tensor_blob(us[k]))
        t["fs_hash"] += time.perf_counter() - t0
        cols = {op.name: ch.columns(op.name, verifier.publics[op.name].n_points, p.columns) for op in mats}
        _sync(prover.device)
        t0 = time.perf_counter()
        opened = prover.open(cols)
        _sync(prover.device)
        out["bytes"]["columns"] = sum(o[0].numel() for o in opened.values()) * 4
        out["bytes"]["paths"] = sum(len(proof) * HASH_BYTES for _, proof in opened.values())
        openings = wire_openings(opened)
        t["prove_open"] = time.perf_counter() - t0

    _sync(vdev)
    t0 = time.perf_counter()
    reason = verify_streaming(verifier, x, claims, chis, us, cols, openings)
    _sync(vdev)
    t["verify_total"] = time.perf_counter() - t0
    out["rejected_at"] = reason
    out["accepted"] = reason is None
    return out
