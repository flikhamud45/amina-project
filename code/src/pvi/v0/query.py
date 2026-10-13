"""One query end to end: the messages in order (claims, folds, openings), their wire encodings and byte
counts, the Fiat--Shamir absorption, and the verdict with its timings."""

from __future__ import annotations

import struct
import time

import numpy as np
import torch

from pvi.v0 import claimcodec
from pvi.core import contention
from pvi.core import hostmem
from pvi.kernels import native as native_kernels
from pvi.commit.commitment import HASH_BYTES
from pvi.v0.streaming import wire_claim, wire_openings, wire_rows
from pvi.v0.transcript import Challenger
from pvi.v0.prover import Prover
from pvi.v0.verifier import Verifier, _GPU_DECODE, _sync


def _tensor_blob(t: torch.Tensor) -> bytes:
    shape = struct.pack(f"<{t.dim()}Q", *t.shape)
    return struct.pack("<B", t.dim()) + shape + t.detach().to("cpu", torch.int64).contiguous().numpy().tobytes()


def _phase_done(out: dict, name: str, start) -> None:
    """``out["contention"][name]``: the contention evidence of the timed phase ``name`` that
    ``start = contention.begin()`` opened (:mod:`pvi.core.contention`; measurement only, both
    snapshots outside the timed window; a phase timed in parts sums them)."""
    contention.add(out.setdefault("contention", {}), name, start)


def _absorb_statement(ch: Challenger, verifier: Verifier, x: torch.Tensor) -> None:
    p = verifier.params
    ch.absorb(b"graph", verifier._digest or verifier.graph.digest())    # every op and cheap-op constant
    ch.absorb(b"params", struct.pack("<IIIi", p.reps, p.columns, p.rate, p.grinding_bits) + verifier.mode.encode())
    for op in verifier.graph.mat_ops:
        pub = verifier.publics.get(op.name)
        blob = struct.pack("<QQ", op.n_rows, op.row_length)
        if pub is not None:
            blob += pub.tag + pub.root + struct.pack("<Q", pub.n_points)
        ch.absorb(b"op/" + op.name.encode(), blob)
    for name, pub in verifier.publics.items() if verifier.groups else ():
        if pub.members:            # a plan's col-layout matrix: its code and the ops it stacks (in order)
            blob = struct.pack("<QQQH", pub.n_rows, pub.row_length, pub.n_points, len(pub.tag)) + pub.tag
            ch.absorb(b"col/" + name.encode(), blob + b"".join(struct.pack("<H", len(m)) + m.encode()
                                                                for m in pub.members))
    for name, g in verifier.groups.items():      # a commitment plan: its trees, members (in order) and t
        blob = struct.pack("<H", len(g.tag)) + g.tag + g.root + struct.pack("<QQQ", g.n_points, p.columns_for(name),
                                                                            len(g.members))
        ch.absorb(b"group/" + name.encode(), blob + b"".join(struct.pack("<H", len(m)) + m.encode() for m in g.members))
    for name, table in verifier.tables.items():   # a plan's lookup tables: the root over each one's rows
        ch.absorb(b"table/" + name.encode(), struct.pack("<H", len(table.tag)) + table.tag + table.root
                  + struct.pack("<QQ", table.n_rows, table.n_tokens))
    if verifier.lookups:                          # modes K and Kpre: the embedding ops the verifier reads itself
        ch.absorb(b"lookups", b"".join(struct.pack("<H", len(op.name)) + op.name.encode()
                                       for op in verifier._own_rows()))
    ch.absorb(b"x", _tensor_blob(x.cpu()))
    if verifier.sampler is not None:              # E2: the client's seed and sampling rule
        ch.absorb(b"sampler", verifier.sampler.digest())


# -- one query ---------------------------------------------------------------------------
# The prover's messages, each challenge drawn after the message it follows (under Fiat--Shamir
# from the transcript that absorbed it), each message's proof bytes counted and its prover time
# recorded in ``out``.  run_query checks each message as it arrives and stops asking at the
# first failing check; the streaming verifier receives them all, then checks.  Both parties run
# in one process, so what one hands the other is passed as a copy (:func:`_passed`): the query, the
# challenges and the column indices the prover receives, and every message of the prover's that
# the verifier reads after the prover has run again (the claims, ``u`` and the tables'
# multiproofs) -- as a wire, a decoder or a GPU client's upload gives each party its own tensors.

def _passed(message, pin: bool = False):
    """``message`` (a tensor, or a dict of them) as the other party receives it: a copy of every
    tensor (with ``pin``, in pinned host memory: a GPU client's) and of every list among the values,
    so that neither party can change in place, when it runs again, what the other holds."""
    def copy(t):
        return hostmem.empty(t.shape, t.dtype).copy_(t) if pin else t.clone()

    if torch.is_tensor(message):
        return copy(message)
    return {k: copy(v) if torch.is_tensor(v) else list(v) if isinstance(v, list) else v for k, v in message.items()}


def _encoded(encode, values) -> bytes:
    """``encode(values)``, the bytes of a message in the compact encoding; none for values it cannot
    carry (:class:`claimcodec.Unencodable`), which a prover can only replace with other bytes: no
    bytes, a message the verifier rejects as malformed at that message's check (``range_or_shape``,
    ``freivalds``, ``columns_shape``), as it rejects those values without wire."""
    try:
        return encode(values)
    except claimcodec.Unencodable:
        return b""


def _proof_hashes(proof) -> list[bytes]:
    """The hashes of a multiproof as they travel (nothing if it is not a list of byte strings)."""
    ok = isinstance(proof, (list, tuple)) and all(isinstance(h, bytes) for h in proof)
    return list(proof) if ok else []


def _claims_message(prover: Prover, verifier: Verifier, x: torch.Tensor, ch: Challenger, out: dict,
                    forward_kwargs: dict, send=None, wire: bool = False) -> tuple[dict | bytes, bytes | None, dict]:
    """``(claims, rows, proofs)``: the claims of every weight op but those whose rows the verifier reads
    itself (``send``: the wire format they travel in; ``wire``: the ``PVC3`` bytes of all but a plan's
    lookup tables, encoded where the prover keeps them, and ``rows`` the tables' as int8 bytes, else
    ``None``) and the multiproof of every lookup table, absorbed with the statement.  A table's claims
    count one byte each."""
    t = out["timings"]
    query = _passed(x)
    if wire and send is None and prover.lean and prover.device.type != "cpu":
        # a lean GPU prover keeps each claim on its device as int32 (not the host's int64) and encodes
        # them there: about 2.3 bytes a claim come back instead of 8
        send = claimcodec.narrow
    _sync(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    claims = prover.claims(query, send=send, to_host=not wire, **forward_kwargs)
    _sync(prover.device)
    t["prove_forward"] = time.perf_counter() - t0
    _phase_done(out, "prove_forward", c0)
    sent = {op.name for op in verifier._sent_ops()}
    if len(sent) < len(claims):
        claims = {k: z for k, z in claims.items() if k in sent}
    proofs, rows = {}, None
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        proofs = prover.open_tables()
        t["prove_lookups"] = time.perf_counter() - t0
        _phase_done(out, "prove_lookups", c0)
        proofs = _passed(proofs)
    if wire:
        c0 = contention.begin()
        t0 = time.perf_counter()
        if verifier.tables:
            rows = _encoded(claimcodec.pack_rows, [claims[name] for name in verifier.tables])
        claims = _encoded(claimcodec.encode, [claims[op.name] for op in prover.graph.mat_ops
                                              if op.name in sent and op.name not in verifier.tables])
        _sync(prover.device)
        t["prove_encode"] = time.perf_counter() - t0
        _phase_done(out, "prove_encode", c0)
    size = len(claims) + len(rows or b"") if wire else sum(z.numel() * (1 if k in verifier.tables else 4)
                                                          for k, z in claims.items())
    out["bytes"] = {"claims": size, "u": 0, "columns": 0,
                    "paths": HASH_BYTES * sum(len(_proof_hashes(p)) for p in proofs.values())}
    c0 = contention.begin()
    t0 = time.perf_counter()
    # Kpre is one message to a designated verifier whose chi and u are its secret key: no challenge reads the
    # transcript, so there is nothing to absorb (FS1)
    if verifier.params.fiat_shamir and verifier.mode != "Kpre":
        _absorb_statement(ch, verifier, x)
        if wire:
            ch.absorb_large(b"claims/" + claimcodec.MAGIC, claims)
            if rows is not None:
                ch.absorb_large(b"rows/I8", rows)
        else:
            for k in sorted(claims):       # int32 wire claims hash as the int64 ones
                ch.absorb(b"claim/" + k.encode(), _tensor_blob(claims[k]))
        for name in verifier.tables:       # the tables' multiproofs, in table order
            hashes = _proof_hashes(proofs.get(name))
            ch.absorb(b"lookup/" + name.encode(), struct.pack("<Q", len(hashes)) + b"".join(hashes))
    t["fs_hash"] = time.perf_counter() - t0
    _phase_done(out, "fs_hash", c0)
    return claims, rows, proofs


def _decoded_claims(verifier: Verifier, blob: bytes, rows: bytes | None, x: torch.Tensor, out: dict,
                    dtype: torch.dtype, pin: bool = False) -> dict | None:
    """The verifier's claims from their ``PVC3`` bytes and a plan's lookup tables' from their int8
    ``rows`` (``None``: malformed), as ``dtype`` host tensors (int32 in pinned memory with ``pin``),
    timed as ``verify_decode``.  Each claim must have the shape the query gives it, which the decoders
    check before they allocate the claims.  A verifier on a GPU with Triton decodes the claims there
    (``claimcodec.decode_device``: int32 device tensors, the same integers and checks; ``PVI_GPU_DECODE=0``
    decodes on the host)."""
    sent = verifier._sent_ops()
    mats = [op for op in sent if op.name not in verifier.tables]
    tables = [op for op in sent if op.name in verifier.tables]
    vdev = torch.device(verifier.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    cols = verifier.claim_columns(x)
    zs = None
    if cols is not None:
        m_of = dict(zip((op.name for op in verifier.graph.mat_ops), cols))
        shapes = [op.n_rows for op in mats], [m_of[op.name] for op in mats]
        try:
            if vdev.type == "cuda" and _GPU_DECODE and claimcodec.decode_device_ok(vdev):
                dec = claimcodec.decode_device(blob, *shapes, vdev)
                dec = dec if dtype == torch.int32 else [d.to(dtype) for d in dec]
                _sync(vdev)
            else:
                dec = claimcodec.decode_torch(blob, *shapes, dtype=dtype, workers=torch.get_num_threads(), pin=pin)
            zs = dict(zip((op.name for op in mats), dec))
            if tables:
                zs.update(zip((op.name for op in tables), claimcodec.unpack_rows(
                    rows, [(op.n_rows, m_of[op.name]) for op in tables], dtype=dtype, pin=pin)))
        except claimcodec.ClaimCodecError:
            zs = None
    out["timings"]["verify_decode"] = out["timings"].get("verify_decode", 0.0) + time.perf_counter() - t0
    _phase_done(out, "verify_decode", c0)
    return zs


def _field_message(blob: bytes, shapes: list[tuple[int, int]], out: dict, dtype: torch.dtype) -> list | None:
    """The field elements of a 31-bit packed message as ``dtype`` host tensors of ``shapes``
    (``None``: malformed), timed as ``verify_decode``."""
    c0 = contention.begin()
    t0 = time.perf_counter()
    try:
        flat = claimcodec.unpack_field(blob, sum(a * b for a, b in shapes), workers=torch.get_num_threads())
    except claimcodec.ClaimCodecError:
        flat = None
    parts = None
    if flat is not None:
        flat = torch.from_numpy(flat.view(np.int32)).to(dtype)
        parts, o = [], 0
        for a, b in shapes:
            parts.append(flat[o:o + a * b].view(a, b))
            o += a * b
    out["timings"]["verify_decode"] = out["timings"].get("verify_decode", 0.0) + time.perf_counter() - t0
    _phase_done(out, "verify_decode", c0)
    return parts


def _fold_message(prover: Prover, verifier: Verifier, ch: Challenger, x: torch.Tensor, out: dict,
                  wire: bool = False) -> tuple[dict, dict, bytes | None]:
    """``(chi, u, the bytes of u with wire)``: fresh challenges (:meth:`Verifier.fold_challenges`:
    under a commitment plan also each col-layout matrix's ``chi'``) and the prover's ``u`` of every
    op checked with Freivalds (mode C; with ``wire`` 31-bit packed, and ``{}`` if the packed ``u``
    is malformed), the verifier's own (K), or the precomputed pair (Kpre)."""
    p, t = verifier.params, out["timings"]
    mats = verifier._row_ops()
    if verifier.mode == "Kpre":
        t["prove_fold"] = t["verify_fold"] = 0.0
        return ({op.name: verifier._pre[op.name][0] for op in mats},
                {op.name: verifier._pre[op.name][1] for op in mats}, None)
    vdev = torch.device(verifier.device)
    chis = {name: chi.to(vdev) for name, chi in verifier.fold_challenges(ch, x).items()}
    sent = _passed({op.name: chis[op.name] for op in mats}) if verifier.mode == "C" else None
    _sync(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    blob = None
    if verifier.mode == "C":
        us = prover.fold(sent)
        _sync(prover.device)
        t["prove_fold"] = time.perf_counter() - t0
        _phase_done(out, "prove_fold", c0)
        t["verify_fold"] = 0.0
        out["bytes"]["u"] = sum(u.numel() for u in us.values()) * 4
        if wire:
            c0 = contention.begin()
            t0 = time.perf_counter()
            blob = _encoded(claimcodec.pack_field, [us[op.name] for op in mats])
            t["prove_encode"] += time.perf_counter() - t0
            _phase_done(out, "prove_encode", c0)
            out["bytes"]["u"] = len(blob)
            parts = _field_message(blob, [(p.reps, op.row_length) for op in mats], out, torch.int64)
            us = {} if parts is None else {op.name: u for op, u in zip(mats, parts)}
        else:
            us = _passed(us)
    else:  # K: the verifier folds its own copy of the weights
        us = {op.name: verifier._fold_local(op, chis[op.name]) for op in mats}
        _sync(vdev)
        t["verify_fold"] = time.perf_counter() - t0
        _phase_done(out, "verify_fold", c0)
        t["prove_fold"] = 0.0
    return chis, us, blob


def _open_message(prover: Prover, verifier: Verifier, ch: Challenger, us: dict, out: dict,
                  package=None, u_blob: bytes | None = None) -> tuple[dict, dict]:
    """Mode C: ``u`` absorbed (its bytes ``u_blob`` with wire), the column indices, and the openings
    (``package``: the wire format they travel in).  With ``u_blob`` the opened columns of every tree
    (each weight op's, or each group's: its members' rows side by side) travel 31-bit packed, tree
    after tree in the order of the challenges and column by column, and arrive as the int32 rows
    ``[t, N]`` of :func:`pipeline.wire_openings` (``None`` if the packed columns are malformed)."""
    p, t = verifier.params, out["timings"]
    if p.fiat_shamir:
        c0 = contention.begin()
        t0 = time.perf_counter()
        if u_blob is not None:
            ch.absorb_large(b"u/F31", u_blob)
        else:
            for k in sorted(us):
                ch.absorb(b"u/" + k.encode(), _tensor_blob(us[k]))
        t["fs_hash"] += time.perf_counter() - t0
        _phase_done(out, "fs_hash", c0)
    cols = verifier.column_challenges(ch)
    sent = _passed(cols)
    _sync(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    opened = prover.open(sent)
    _sync(prover.device)
    if u_blob is None:
        openings = opened if package is None else package(opened)
        t["prove_open"] = time.perf_counter() - t0
        _phase_done(out, "prove_open", c0)
        out["bytes"]["columns"] = sum(o[0].numel() for o in opened.values()) * 4
    else:
        t["prove_open"] = time.perf_counter() - t0
        _phase_done(out, "prove_open", c0)
        c0 = contention.begin()
        t0 = time.perf_counter()
        blob = _encoded(claimcodec.pack_field, [opened[name][0].T for name in cols])
        t["prove_encode"] += time.perf_counter() - t0
        _phase_done(out, "prove_encode", c0)
        out["bytes"]["columns"] = len(blob)
        units = verifier._column_units()
        rows = _field_message(blob, [(len(cols[name]), sum(verifier.publics[m].n_rows for m in members))
                                     for name, members in units], out, torch.int32)
        openings = {name: (None if rows is None else rows[i], opened[name][1]) for i, (name, _) in enumerate(units)}
    out["bytes"]["paths"] += sum(len(proof) * HASH_BYTES for _, proof in opened.values())
    return cols, openings


def run_query(prover: Prover, verifier: Verifier, x: torch.Tensor, *, seed: int | None = None,
              forward_kwargs: dict | None = None, wire: bool = False, sampler=None) -> dict:
    """One full interaction.  Returns acceptance, the rejecting check, timings (s)
    and proof bytes.  With Fiat--Shamir, ``fs_hash`` is paid by both parties.

    A verifier with ``stream=True`` receives every message (the claims and openings in the wire
    formats of :mod:`pvi.v0.streaming`), then checks them with :meth:`Verifier.verify_streaming`:
    the same verdict and label, its overlapped work timed as one ``verify_total``.  An accepted
    query has the same proof bytes and Fiat--Shamir transcript; a rejected one has also received
    (counted, and absorbed) the messages this function no longer asks for once a check fails --
    in mode C, ``u`` and the openings -- so the transcript here is a prefix of the streaming one.

    ``wire=True`` sends the proof in the compact encoding of :mod:`pvi.v0.claimcodec`: the
    claims as ``PVC3`` bytes, encoded where the prover keeps them (``prove_encode``), and ``u`` and
    the opened columns 31-bit packed.  ``sampler`` (a ``token_sampler.TokenSampler``, generation graphs): the
    response's tokens were sampled with the client's seed, and the token rule checks that sampler's choices
    instead of the greedy ones (the seed is part of the statement).  The verifier decodes them (``verify_decode``) before it
    checks anything and rejects a malformed message at the check that rejects a malformed message
    of the same kind (``range_or_shape`` for the claims, ``freivalds`` for ``u``, ``columns_shape``
    for the columns) -- so too a prover's values that the encoding cannot carry (a claim outside the
    range check, a looked-up row outside int8, a field element of 32 bits), which travel as no
    bytes; ``bytes`` counts the encoded sizes, and with Fiat--Shamir the transcript absorbs these
    bytes.  A GPU client uploads the decoded claims as int32 and widens them there."""
    verifier.sampler = sampler
    try:
        return _run_query(prover, verifier, x, seed, forward_kwargs, wire)
    finally:
        verifier.sampler = None


def _run_query(prover: Prover, verifier: Verifier, x: torch.Tensor, seed, forward_kwargs, wire) -> dict:
    ch = Challenger(fiat_shamir=verifier.params.fiat_shamir, seed=seed)
    out = {"accepted": False, "rejected_at": None, "timings": {}, "contention": {}}
    if verifier.stream:
        out["rejected_at"] = _run_streaming(prover, verifier, x, ch, out, forward_kwargs or {}, wire)
        out["accepted"] = out["rejected_at"] is None
        return out
    t = out["timings"]
    claims, rows, proofs = _claims_message(prover, verifier, x, ch, out, forward_kwargs or {}, wire=wire)

    vdev = torch.device(verifier.device)
    x_v, claims_v = x.cpu(), claims
    if wire:                      # a GPU client uploads int32 claims
        cpu32 = vdev.type == "cpu" and native_kernels.int32_claims()          # (the native kernels read int32 claims)
        claims_v = _decoded_claims(verifier, claims, rows, x, out, torch.int32 if vdev.type != "cpu" or cpu32
                                   else torch.int64)
        if claims_v is None:
            out["rejected_at"] = "range_or_shape"
            return out
    else:                         # the prover's own tensors, which the checks after its fold read again
        claims_v = _passed(claims)
    if not verifier.check_tokens(x_v, claims_v):
        out["rejected_at"] = "token"
        return out
    if vdev.type != "cpu":        # a GPU client: receiving the proof includes uploading it
        _sync(vdev)
        c0 = contention.begin()
        t0 = time.perf_counter()
        x_v, claims_v = x.to(vdev), {k: v.to(vdev) for k, v in claims_v.items()}
        if wire:
            claims_v = {k: v.to(torch.int64) for k, v in claims_v.items()}
        _sync(vdev)
        t["verify_upload"] = time.perf_counter() - t0
        _phase_done(out, "verify_upload", c0)

    c0 = contention.begin()
    t0 = time.perf_counter()
    inputs = verifier.derive(x_v, claims_v)
    _sync(vdev)
    t["verify_derive"] = time.perf_counter() - t0
    _phase_done(out, "verify_derive", c0)
    if inputs is None:
        out["rejected_at"] = "range_or_shape"
        return out
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_lookups(claims_v, inputs, proofs)
        t["verify_lookups"] = time.perf_counter() - t0
        _phase_done(out, "verify_lookups", c0)
        if reason is not None:
            out["rejected_at"] = reason
            return out

    chis, us, u_blob = _fold_message(prover, verifier, ch, x, out, wire)
    if verifier.mode == "C" and vdev.type != "cpu":
        c0 = contention.begin()
        t0 = time.perf_counter()
        us = {k: v.to(vdev) for k, v in us.items()}
        _sync(vdev)
        t["verify_upload"] += time.perf_counter() - t0
        _phase_done(out, "verify_upload", c0)

    c0 = contention.begin()
    t0 = time.perf_counter()
    ok = verifier.check_products(claims_v, inputs, chis, us)
    if ok and verifier.mode == "C":     # the code checks' two sides (a col-layout matrix's: the verifier's own)
        lefts, sources = verifier.column_operands(claims_v, inputs, chis, us)
    _sync(vdev)
    t["verify_products"] = time.perf_counter() - t0
    _phase_done(out, "verify_products", c0)
    if not ok:
        out["rejected_at"] = "freivalds"
        return out

    if verifier.mode == "C":
        cols, openings = _open_message(prover, verifier, ch, us, out, u_blob=u_blob)
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_columns(lefts, sources, cols, openings, wire=wire)
        _sync(vdev)
        t["verify_columns"] = time.perf_counter() - t0
        _phase_done(out, "verify_columns", c0)
        if reason is not None:
            out["rejected_at"] = reason
            return out
    out["accepted"] = True
    return out


def _run_streaming(prover: Prover, verifier: Verifier, x: torch.Tensor, ch: Challenger, out: dict,
                   forward_kwargs: dict, wire: bool) -> str | None:
    """:func:`run_query` with ``stream=True``: every message, then :meth:`Verifier.verify_streaming`.
    ``verify_total`` times the verifier's work from the moment it holds the messages (its uploads
    included; with ``wire``, after ``verify_decode``, which gives it the int32 claims and rows of
    :mod:`pvi.v0.streaming`'s wire formats)."""
    vdev = torch.device(verifier.device)
    pin = vdev.type == "cuda"
    claims, rows, proofs = _claims_message(prover, verifier, x, ch, out, forward_kwargs, wire=wire,
                                           send=None if wire else lambda z: wire_claim(z, pin=pin))
    if not wire:   # the verifier's own copies (read after the prover runs again); a lookup table's as its int8 rows
        claims = {k: wire_rows(z, pin=pin) if k in verifier.tables else z for k, z in _passed(claims, pin).items()}
    chis, us, u_blob = _fold_message(prover, verifier, ch, x, out, wire)
    cols = openings = None
    if verifier.mode == "C":
        cols, openings = _open_message(prover, verifier, ch, us, out, package=wire_openings, u_blob=u_blob)
    if wire:
        claims = _decoded_claims(verifier, claims, rows, x, out, torch.int32, pin=vdev.type == "cuda")
        if claims is None:
            return "range_or_shape"
    _sync(vdev)
    c0 = contention.begin()
    t0 = time.perf_counter()
    reason = verifier.verify_streaming(x, claims, chis, us, cols, openings, proofs)
    _sync(vdev)
    out["timings"]["verify_total"] = time.perf_counter() - t0
    _phase_done(out, "verify_total", c0)
    return reason
