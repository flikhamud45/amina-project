"""One query end to end: the messages in order (claims, folds, openings), their wire encodings and byte
counts, the Fiat--Shamir absorption, and the verdict with its timings."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from pvi import options
from pvi.commit.commitment import HASH_BYTES
from pvi.core import contention, hostmem
from pvi.kernels import native as native_kernels
from pvi.v0 import claimcodec
from pvi.v0.prover import Prover
from pvi.v0.streaming import wire_claim, wire_openings, wire_rows
from pvi.v0.transcript import Challenger
from pvi.v0.verifier import Verifier, sync_device


def _tensor_blob(t: torch.Tensor) -> bytes:
    shape = struct.pack(f"<{t.dim()}Q", *t.shape)
    return struct.pack("<B", t.dim()) + shape + t.detach().to("cpu", torch.int64).contiguous().numpy().tobytes()


def phase_done(out: dict, name: str, start) -> None:
    """``out["contention"][name]``: the contention evidence of the timed phase ``name`` that
    ``start = contention.begin()`` opened (:mod:`pvi.core.contention`; measurement only, both
    snapshots outside the timed window; a phase timed in parts sums them)."""
    contention.add(out.setdefault("contention", {}), name, start)


def absorb_statement(ch: Challenger, verifier: Verifier, x: torch.Tensor) -> None:
    """The statement, absorbed into the Fiat--Shamir transcript ``ch`` before any prover message: the graph's
    digest, the parameters and the mode, every weight op's shape and commitment, a commitment plan's col-layout
    matrices, groups and lookup tables, the embedding ops a verifier with ``lookups`` reads itself, the query
    ``x`` and, for sampled generation, the sampler (its rule and the client's seed)."""
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
    if verifier.sampler is not None:              # sampled generation: the client's seed and sampling rule
        ch.absorb(b"sampler", verifier.sampler.digest())


# -- one query ---------------------------------------------------------------------------
# The prover's messages, each challenge drawn after the message it follows (under Fiat--Shamir
# from the transcript that absorbed it), each message's proof bytes counted and its prover time
# recorded in ``out``.  run_query checks each message as it arrives and stops asking at the
# first failing check; the streaming verifier receives them all, then checks.  Both parties run
# in one process, so what one hands the other is passed as a copy (:func:`passed`): the query, the
# challenges and the column indices the prover receives, and every message of the prover's that
# the verifier reads after the prover has run again (the claims, ``u`` and the tables'
# multiproofs) -- as a wire, a decoder or a GPU client's upload gives each party its own tensors.

def passed(message, pin: bool = False):
    """``message`` (a tensor, or a dict of them) as the other party receives it: a copy of every
    tensor (with ``pin``, in pinned host memory: a GPU client's) and of every list among the values,
    so that neither party can change in place, when it runs again, what the other holds."""
    def copy(t):
        return hostmem.empty(t.shape, t.dtype).copy_(t) if pin else t.clone()

    if torch.is_tensor(message):
        return copy(message)
    return {k: copy(v) if torch.is_tensor(v) else list(v) if isinstance(v, list) else v for k, v in message.items()}


def encoded_message(encode, values) -> bytes:
    """``encode(values)``, the bytes of a message in the compact encoding; none for values it cannot
    carry (:class:`claimcodec.Unencodable`), which a prover can only replace with other bytes: no
    bytes, a message the verifier rejects as malformed at that message's check (``range_or_shape``,
    ``freivalds``, ``columns_shape``), as it rejects those values without wire."""
    try:
        return encode(values)
    except claimcodec.Unencodable:
        return b""


def proof_hashes(proof) -> list[bytes]:
    """The hashes of a multiproof as they travel (nothing if it is not a list of byte strings)."""
    ok = isinstance(proof, (list, tuple)) and all(isinstance(h, bytes) for h in proof)
    return list(proof) if ok else []


@dataclass
class WireProof:
    """The prover's messages of one query in the compact encoding, as the verifier receives them: the
    claims (``PVC3`` bytes), a plan's lookup tables' rows (int8 bytes) and their multiproofs (``lookups``,
    by table), and in mode C ``u`` and the opened columns (31-bit packed) with each tree's multiproof
    (``paths``).  What a client holds of one proof (:func:`verify_wire`)."""

    claims: bytes
    rows: bytes | None = None
    lookups: dict = field(default_factory=dict)
    u: bytes | None = None
    columns: bytes | None = None
    paths: dict = field(default_factory=dict)

    def nbytes(self) -> int:
        """The bytes of the proof as it travels (the multiproofs' hashes included)."""
        hashes = sum(len(proof_hashes(p)) for p in (*self.lookups.values(), *self.paths.values()))
        return sum(len(b or b"") for b in (self.claims, self.rows, self.u, self.columns)) + HASH_BYTES * hashes


def _claims_message(prover: Prover, verifier: Verifier, x: torch.Tensor, ch: Challenger, out: dict,
                    forward_kwargs: dict, send=None, wire: bool = False) -> tuple[dict | bytes, bytes | None, dict]:
    """``(claims, rows, proofs)``: the claims of every weight op but those whose rows the verifier reads
    itself (``send``: the wire format they travel in; ``wire``: the ``PVC3`` bytes of all but a plan's
    lookup tables, encoded where the prover keeps them, and ``rows`` the tables' as int8 bytes, else
    ``None``) and the multiproof of every lookup table, absorbed with the statement.  A table's claims
    count one byte each."""
    t = out["timings"]
    query = passed(x)
    if wire and send is None and prover.lean and prover.device.type != "cpu":
        # a lean GPU prover keeps each claim on its device as int32 (not the host's int64) and encodes
        # them there: only the encoding comes back (about 2.3 bytes a claim on the random-weight decoders),
        # not 8 bytes a claim
        send = claimcodec.narrow
    sync_device(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    claims = prover.claims(query, send=send, to_host=not wire, **forward_kwargs)
    sync_device(prover.device)
    t["prove_forward"] = time.perf_counter() - t0
    phase_done(out, "prove_forward", c0)
    sent = {op.name for op in verifier._sent_ops()}
    if len(sent) < len(claims):
        claims = {k: z for k, z in claims.items() if k in sent}
    proofs, rows = {}, None
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        proofs = prover.open_tables()
        t["prove_lookups"] = time.perf_counter() - t0
        phase_done(out, "prove_lookups", c0)
        proofs = passed(proofs)
    if wire:
        c0 = contention.begin()
        t0 = time.perf_counter()
        if verifier.tables:
            rows = encoded_message(claimcodec.pack_rows, [claims[name] for name in verifier.tables])
        claims = encoded_message(claimcodec.encode, [claims[op.name] for op in prover.graph.mat_ops
                                              if op.name in sent and op.name not in verifier.tables])
        sync_device(prover.device)
        t["prove_encode"] = time.perf_counter() - t0
        phase_done(out, "prove_encode", c0)
    size = len(claims) + len(rows or b"") if wire else sum(z.numel() * (1 if k in verifier.tables else 4)
                                                          for k, z in claims.items())
    out["bytes"] = {"claims": size, "u": 0, "columns": 0,
                    "paths": HASH_BYTES * sum(len(proof_hashes(p)) for p in proofs.values())}
    c0 = contention.begin()
    t0 = time.perf_counter()
    absorb_claims(ch, verifier, x, claims, rows, proofs, wire)
    t["fs_hash"] = time.perf_counter() - t0
    phase_done(out, "fs_hash", c0)
    return claims, rows, proofs


def absorb_claims(ch: Challenger, verifier: Verifier, x: torch.Tensor, claims, rows: bytes | None, proofs: dict,
                  wire: bool) -> None:
    """The first message absorbed into the Fiat--Shamir transcript ``ch``, as both parties absorb it: the
    statement, the claims (their ``PVC3`` bytes with ``wire``, and a plan's lookup tables' ``rows``) and the
    tables' multiproofs ``proofs``.  Nothing in mode Kpre, which is one message to a designated verifier whose
    ``chi`` and ``u`` are its secret key (no challenge reads the transcript), or without Fiat--Shamir."""
    if not verifier.params.fiat_shamir or verifier.mode == "Kpre":
        return
    absorb_statement(ch, verifier, x)
    if wire:
        ch.absorb_large(b"claims/" + claimcodec.MAGIC, claims)
        if rows is not None:
            ch.absorb_large(b"rows/I8", rows)
    else:
        for k in sorted(claims):       # int32 wire claims hash as the int64 ones
            ch.absorb(b"claim/" + k.encode(), _tensor_blob(claims[k]))
    for name in verifier.tables:       # the tables' multiproofs, in table order
        hashes = proof_hashes(proofs.get(name))
        ch.absorb(b"lookup/" + name.encode(), struct.pack("<Q", len(hashes)) + b"".join(hashes))


def decoded_claims(verifier: Verifier, blob: bytes, rows: bytes | None, x: torch.Tensor, out: dict,
                    dtype: torch.dtype, pin: bool = False) -> dict | None:
    """The verifier's claims from their ``PVC3`` bytes and a plan's lookup tables' from their int8
    ``rows`` (``None``: malformed), as ``dtype`` host tensors (int32 in pinned memory with ``pin``),
    timed as ``verify_decode``.  Each claim must have the shape the query gives it, which the decoders
    check before they allocate the claims.  A verifier on a GPU with Triton decodes the claims there
    (``claimcodec.decode_device``: int32 device tensors, the same integers and checks; with the option
    ``gpu_decode`` off on the host)."""
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
            if vdev.type == "cuda" and options.current().gpu_decode and claimcodec.decode_device_ok(vdev):
                dec = claimcodec.decode_device(blob, *shapes, vdev)
                dec = dec if dtype == torch.int32 else [d.to(dtype) for d in dec]
                sync_device(vdev)
            else:
                dec = claimcodec.decode_torch(blob, *shapes, dtype=dtype, workers=torch.get_num_threads(), pin=pin)
            zs = dict(zip((op.name for op in mats), dec))
            if tables:
                zs.update(zip((op.name for op in tables), claimcodec.unpack_rows(
                    rows, [(op.n_rows, m_of[op.name]) for op in tables], dtype=dtype, pin=pin)))
        except claimcodec.ClaimCodecError:
            zs = None
    out["timings"]["verify_decode"] = out["timings"].get("verify_decode", 0.0) + time.perf_counter() - t0
    phase_done(out, "verify_decode", c0)
    return zs


def field_message(blob: bytes, shapes: list[tuple[int, int]], out: dict, dtype: torch.dtype) -> list | None:
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
    phase_done(out, "verify_decode", c0)
    return parts


def _fold_message(prover: Prover, verifier: Verifier, ch: Challenger, x: torch.Tensor, out: dict,
                  wire: bool = False) -> tuple[dict, dict, bytes | None]:
    """``(chi, u, the bytes of u with wire)``: fresh challenges (:meth:`Verifier.fold_challenges`:
    under a commitment plan also each col-layout matrix's ``chi'``) and the prover's ``u`` of every
    op checked with Freivalds (mode C; with ``wire`` 31-bit packed, and ``{}`` if the packed ``u``
    is malformed), the verifier's own (K), or the precomputed pair (Kpre)."""
    t = out["timings"]
    mats = verifier._row_ops()
    if verifier.mode == "Kpre":
        t["prove_fold"] = t["verify_fold"] = 0.0
        return ({op.name: verifier._pre[op.name][0] for op in mats},
                {op.name: verifier._pre[op.name][1] for op in mats}, None)
    vdev = torch.device(verifier.device)
    chis = {name: chi.to(vdev) for name, chi in verifier.fold_challenges(ch, x).items()}
    sent = passed({op.name: chis[op.name] for op in mats}) if verifier.mode == "C" else None
    sync_device(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    blob = None
    if verifier.mode == "C":
        us = prover.fold(sent)
        sync_device(prover.device)
        t["prove_fold"] = time.perf_counter() - t0
        phase_done(out, "prove_fold", c0)
        t["verify_fold"] = 0.0
        out["bytes"]["u"] = sum(u.numel() for u in us.values()) * 4
        if wire:
            c0 = contention.begin()
            t0 = time.perf_counter()
            blob = encoded_message(claimcodec.pack_field, [us[op.name] for op in mats])
            t["prove_encode"] += time.perf_counter() - t0
            phase_done(out, "prove_encode", c0)
            out["bytes"]["u"] = len(blob)
            us = received_u(verifier, blob, out)
        else:
            us = passed(us)
    else:  # K: the verifier folds its own copy of the weights
        us = {op.name: verifier._fold_local(op, chis[op.name]) for op in mats}
        sync_device(vdev)
        t["verify_fold"] = time.perf_counter() - t0
        phase_done(out, "verify_fold", c0)
        t["prove_fold"] = 0.0
    return chis, us, blob


def received_u(verifier: Verifier, blob: bytes, out: dict) -> dict:
    """Mode C's ``u`` of every op checked with Freivalds from its 31-bit packed bytes, as int64 host
    tensors (``{}`` if the bytes are malformed), timed as ``verify_decode``."""
    mats = verifier._row_ops()
    parts = field_message(blob, [(verifier.params.reps, op.row_length) for op in mats], out, torch.int64)
    return {} if parts is None else {op.name: u for op, u in zip(mats, parts)}


def received_openings(verifier: Verifier, cols: dict, blob: bytes, paths: dict, out: dict) -> dict:
    """Mode C's openings from the 31-bit packed bytes of the opened columns (every tree's, tree after
    tree in the order of the challenges ``cols``) and each tree's multiproof ``paths``: ``{tree: (int32
    rows [t, N] or None if the bytes are malformed, its multiproof)}``, timed as ``verify_decode``."""
    units = verifier._column_units()
    rows = field_message(blob, [(len(cols[name]), sum(verifier.publics[m].n_rows for m in members))
                                for name, members in units], out, torch.int32)
    return {name: (None if rows is None else rows[i], paths.get(name)) for i, (name, _) in enumerate(units)}


def open_message(prover: Prover, verifier: Verifier, ch: Challenger, us: dict, out: dict,
                  package=None, u_blob: bytes | None = None, keep: "WireProof | None" = None) -> tuple[dict, dict]:
    """Mode C: ``u`` absorbed (its bytes ``u_blob`` with wire), the column indices, and the openings
    (``package``: the wire format they travel in).  With ``u_blob`` the opened columns of every tree
    (each weight op's, or each group's: its members' rows side by side) travel 31-bit packed, tree
    after tree in the order of the challenges and column by column, and arrive as the int32 rows
    ``[t, N]`` of :func:`pvi.v0.streaming.wire_openings` (``None`` if the packed columns are malformed);
    ``keep`` (with ``u_blob``) receives the multiproofs, and those bytes only if they are malformed:
    well-formed ones are freed here and packed again from the openings after the column checks
    (:func:`kept_columns`), so that a kept proof adds nothing to the peak memory of those checks."""
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
        phase_done(out, "fs_hash", c0)
    cols = verifier.column_challenges(ch)
    sent = passed(cols)
    sync_device(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    opened = prover.open(sent)
    sync_device(prover.device)
    if u_blob is None:
        openings = opened if package is None else package(opened)
        t["prove_open"] = time.perf_counter() - t0
        phase_done(out, "prove_open", c0)
        out["bytes"]["columns"] = sum(o[0].numel() for o in opened.values()) * 4
    else:
        t["prove_open"] = time.perf_counter() - t0
        phase_done(out, "prove_open", c0)
        c0 = contention.begin()
        t0 = time.perf_counter()
        blob = encoded_message(claimcodec.pack_field, [opened[name][0].T for name in cols])
        t["prove_encode"] += time.perf_counter() - t0
        phase_done(out, "prove_encode", c0)
        out["bytes"]["columns"] = len(blob)
        paths = {name: proof for name, (_, proof) in opened.items()}
        openings = received_openings(verifier, cols, blob, paths, out)
        if keep is not None:
            keep.paths = paths
            if any(rows is None for rows, _ in openings.values()):
                keep.columns = blob
    out["bytes"]["paths"] += sum(len(proof) * HASH_BYTES for _, proof in opened.values())
    return cols, openings


def kept_columns(cols: dict, openings: dict) -> bytes:
    """The 31-bit packed opened columns of :func:`open_message` again, from their well-formed openings
    (the int32 rows ``[t, N]`` of every tree, in the order of the challenges ``cols``): the same bytes,
    since packing is canonical (the padding bits are zero)."""
    return encoded_message(claimcodec.pack_field, [openings[name][0] for name in cols])


def run_query(prover: Prover, verifier: Verifier, x: torch.Tensor, *, seed: int | None = None,
              forward_kwargs: dict | None = None, wire: bool = False, sampler=None, keep_proof: bool = False) -> dict:
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
    the opened columns 31-bit packed.  The verifier decodes them (``verify_decode``) before it
    checks anything and rejects a malformed message at the check that rejects a malformed message
    of the same kind (``range_or_shape`` for the claims, ``freivalds`` for ``u``, ``columns_shape``
    for the columns) -- so too a prover's values that the encoding cannot carry (a claim outside the
    range check, a looked-up row outside int8, a field element of 32 bits), which travel as no
    bytes; ``bytes`` counts the encoded sizes, and with Fiat--Shamir the transcript absorbs these
    bytes.  A GPU client uploads the decoded claims as int32 and widens them there.

    ``sampler`` (a :class:`pvi.v0.token_sampler.TokenSampler`, generation graphs): the response's tokens
    were sampled with the client's seed, and the token rule checks that sampler's choices instead of the
    greedy ones (the seed is part of the statement).

    ``keep_proof=True`` (with ``wire``, not streaming) also returns the prover's messages as the verifier
    received them (``out["proof"]``, a :class:`WireProof`), which :func:`verify_wire` checks again; a
    rejected query's holds the messages up to the rejecting check."""
    if keep_proof and (not wire or verifier.stream):
        raise ValueError("keep_proof keeps the compact encoding's messages: it needs wire=True and a verifier "
                         "without stream")
    verifier.sampler = sampler
    try:
        return _run_query(prover, verifier, x, seed, forward_kwargs, wire, keep_proof)
    finally:
        verifier.sampler = None


def _run_query(prover: Prover, verifier: Verifier, x: torch.Tensor, seed, forward_kwargs, wire,
               keep_proof: bool = False) -> dict:
    ch = Challenger(fiat_shamir=verifier.params.fiat_shamir, seed=seed)
    out = {"accepted": False, "rejected_at": None, "timings": {}, "contention": {}}
    if verifier.stream:
        out["rejected_at"] = _run_streaming(prover, verifier, x, ch, out, forward_kwargs or {}, wire)
        out["accepted"] = out["rejected_at"] is None
        return out
    t = out["timings"]
    claims, rows, proofs = _claims_message(prover, verifier, x, ch, out, forward_kwargs or {}, wire=wire)
    keep = None
    if keep_proof:
        keep = out["proof"] = WireProof(claims, rows, proofs)

    vdev = torch.device(verifier.device)
    x_v, claims_v = x.cpu(), claims
    if wire:                      # a GPU client uploads int32 claims
        cpu32 = vdev.type == "cpu" and native_kernels.int32_claims()          # (the native kernels read int32 claims)
        claims_v = decoded_claims(verifier, claims, rows, x, out, torch.int32 if vdev.type != "cpu" or cpu32
                                   else torch.int64)
        if claims_v is None:
            out["rejected_at"] = "range_or_shape"
            return out
    else:                         # the prover's own tensors, which the checks after its fold read again
        claims_v = passed(claims)
    if not verifier.check_tokens(x_v, claims_v):
        out["rejected_at"] = "token"
        return out
    if vdev.type != "cpu":        # a GPU client: receiving the proof includes uploading it
        sync_device(vdev)
        c0 = contention.begin()
        t0 = time.perf_counter()
        x_v, claims_v = x.to(vdev), {k: v.to(vdev) for k, v in claims_v.items()}
        if wire:
            claims_v = {k: v.to(torch.int64) for k, v in claims_v.items()}
        sync_device(vdev)
        t["verify_upload"] = time.perf_counter() - t0
        phase_done(out, "verify_upload", c0)

    c0 = contention.begin()
    t0 = time.perf_counter()
    inputs = verifier.derive(x_v, claims_v)
    sync_device(vdev)
    t["verify_derive"] = time.perf_counter() - t0
    phase_done(out, "verify_derive", c0)
    if inputs is None:
        out["rejected_at"] = "range_or_shape"
        return out
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_lookups(claims_v, inputs, proofs)
        t["verify_lookups"] = time.perf_counter() - t0
        phase_done(out, "verify_lookups", c0)
        if reason is not None:
            out["rejected_at"] = reason
            return out

    chis, us, u_blob = _fold_message(prover, verifier, ch, x, out, wire)
    if keep is not None:
        keep.u = u_blob
    if verifier.mode == "C" and vdev.type != "cpu":
        c0 = contention.begin()
        t0 = time.perf_counter()
        us = {k: v.to(vdev) for k, v in us.items()}
        sync_device(vdev)
        t["verify_upload"] += time.perf_counter() - t0
        phase_done(out, "verify_upload", c0)

    c0 = contention.begin()
    t0 = time.perf_counter()
    ok = verifier.check_products(claims_v, inputs, chis, us)
    if ok and verifier.mode == "C":     # the code checks' two sides (a col-layout matrix's: the verifier's own)
        lefts, sources = verifier.column_operands(claims_v, inputs, chis, us)
    sync_device(vdev)
    t["verify_products"] = time.perf_counter() - t0
    phase_done(out, "verify_products", c0)
    if not ok:
        out["rejected_at"] = "freivalds"
        return out

    if verifier.mode == "C":
        cols, openings = open_message(prover, verifier, ch, us, out, u_blob=u_blob, keep=keep)
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_columns(lefts, sources, cols, openings, wire=wire)
        sync_device(vdev)
        t["verify_columns"] = time.perf_counter() - t0
        phase_done(out, "verify_columns", c0)
        if keep is not None and keep.columns is None:
            keep.columns = kept_columns(cols, openings)
        if reason is not None:
            out["rejected_at"] = reason
            return out
    out["accepted"] = True
    return out


def verify_wire(verifier: Verifier, x: torch.Tensor, proof: WireProof, *, sampler=None) -> dict:
    """A CPU verifier's checks of one query from the prover's messages alone (``proof``, as
    ``run_query(..., wire=True, keep_proof=True)`` keeps them): what a client that received the proof
    computes, with no prover in the process.  The checks of :func:`run_query`, in its order and with its
    rejection labels, timings and verdicts: the challenges come from the Fiat--Shamir transcript that the
    messages give (mode C needs Fiat--Shamir: an interactive proof answers challenges drawn while it ran),
    mode Kpre's are the verifier's key, and mode K's verifier folds its own weights; ``sampler`` as in
    :func:`run_query`.  Returns ``accepted``, ``rejected_at``, ``timings`` and ``contention``."""
    if verifier.stream or torch.device(verifier.device).type != "cpu":
        raise ValueError("verify_wire runs the CPU verifier without stream")
    if verifier.mode == "C" and not verifier.params.fiat_shamir:
        raise ValueError("mode C's proof answers interactive challenges: verify_wire needs Fiat-Shamir")
    verifier.sampler = sampler
    try:
        return _verify_wire(verifier, x, proof)
    finally:
        verifier.sampler = None


def _verify_wire(verifier: Verifier, x: torch.Tensor, proof: WireProof) -> dict:
    ch = Challenger(fiat_shamir=verifier.params.fiat_shamir)
    out = {"accepted": False, "rejected_at": None, "timings": {}, "contention": {}}
    t = out["timings"]
    c0 = contention.begin()
    t0 = time.perf_counter()
    absorb_claims(ch, verifier, x, proof.claims, proof.rows, proof.lookups, True)
    t["fs_hash"] = time.perf_counter() - t0
    phase_done(out, "fs_hash", c0)
    x_v = x.cpu()
    claims_v = decoded_claims(verifier, proof.claims, proof.rows, x, out,
                              torch.int32 if native_kernels.int32_claims() else torch.int64)
    if claims_v is None:
        out["rejected_at"] = "range_or_shape"
        return out
    if not verifier.check_tokens(x_v, claims_v):
        out["rejected_at"] = "token"
        return out
    c0 = contention.begin()
    t0 = time.perf_counter()
    inputs = verifier.derive(x_v, claims_v)
    t["verify_derive"] = time.perf_counter() - t0
    phase_done(out, "verify_derive", c0)
    if inputs is None:
        out["rejected_at"] = "range_or_shape"
        return out
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_lookups(claims_v, inputs, proof.lookups)
        t["verify_lookups"] = time.perf_counter() - t0
        phase_done(out, "verify_lookups", c0)
        if reason is not None:
            out["rejected_at"] = reason
            return out
    mats = verifier._row_ops()
    t["verify_fold"] = 0.0
    if verifier.mode == "Kpre":
        chis = {op.name: verifier._pre[op.name][0] for op in mats}
        us = {op.name: verifier._pre[op.name][1] for op in mats}
    else:
        chis = verifier.fold_challenges(ch, x)
        if verifier.mode == "C":
            us = received_u(verifier, proof.u or b"", out)
        else:                 # K: the verifier folds its own copy of the weights
            c0 = contention.begin()
            t0 = time.perf_counter()
            us = {op.name: verifier._fold_local(op, chis[op.name]) for op in mats}
            t["verify_fold"] = time.perf_counter() - t0
            phase_done(out, "verify_fold", c0)
    c0 = contention.begin()
    t0 = time.perf_counter()
    ok = verifier.check_products(claims_v, inputs, chis, us)
    if ok and verifier.mode == "C":
        lefts, sources = verifier.column_operands(claims_v, inputs, chis, us)
    t["verify_products"] = time.perf_counter() - t0
    phase_done(out, "verify_products", c0)
    if not ok:
        out["rejected_at"] = "freivalds"
        return out
    if verifier.mode == "C":
        c0 = contention.begin()
        t0 = time.perf_counter()
        ch.absorb_large(b"u/F31", proof.u or b"")
        t["fs_hash"] += time.perf_counter() - t0
        phase_done(out, "fs_hash", c0)
        cols = verifier.column_challenges(ch)
        openings = received_openings(verifier, cols, proof.columns or b"", proof.paths, out)
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_columns(lefts, sources, cols, openings, wire=True)
        t["verify_columns"] = time.perf_counter() - t0
        phase_done(out, "verify_columns", c0)
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
        claims = {k: wire_rows(z, pin=pin) if k in verifier.tables else z for k, z in passed(claims, pin).items()}
    chis, us, u_blob = _fold_message(prover, verifier, ch, x, out, wire)
    cols = openings = None
    if verifier.mode == "C":
        cols, openings = open_message(prover, verifier, ch, us, out, package=wire_openings, u_blob=u_blob)
    if wire:
        claims = decoded_claims(verifier, claims, rows, x, out, torch.int32, pin=vdev.type == "cuda")
        if claims is None:
            return "range_or_shape"
    sync_device(vdev)
    c0 = contention.begin()
    t0 = time.perf_counter()
    reason = verifier.verify_streaming(x, claims, chis, us, cols, openings, proofs)
    sync_device(vdev)
    out["timings"]["verify_total"] = time.perf_counter() - t0
    phase_done(out, "verify_total", c0)
    return reason
