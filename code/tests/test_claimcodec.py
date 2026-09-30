"""The compact wire encoding of the proof: ``pvi.fullcheck.claimcodec``.

Slot streams, Rice vectors and 31-bit field elements round trip; the vectorised, lane-by-lane,
threaded and torch (GPU prover) paths give the same bytes and values; every malformed input is
rejected with ``ClaimCodecError`` (truncations, trailing bytes, header fields, padding,
exceptions) and random corruptions never make the decoder raise anything else.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from pvi.fullcheck import claimcodec as cc
from pvi.fullcheck.protocol import P, Z_BOUND
from pvi.fullcheck.transformer import DecoderConfig, build_decoder

LIM = Z_BOUND - 1


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return request.param


def _rt(zs, **kw) -> bytes:
    blob = cc.encode(zs, **kw)
    out = cc.decode(blob, [z.shape[0] for z in zs], max_cols=max([z.shape[1] for z in zs] + [1]))
    assert len(out) == len(zs)
    for a, b in zip(zs, out):
        assert b.dtype == np.int32 and b.shape == a.shape
        assert np.array_equal(np.asarray(a, dtype=np.int64), b.astype(np.int64))
    return blob


# -- slot streams and Rice vectors --------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 8, 13, 16, 17, 24, 26, 30, 31, 32])
def test_pack32_round_trip(k):
    rng = np.random.default_rng(k)
    for n in [0, 1, 7, 31, 32, 33, 64, 1000]:
        v = rng.integers(0, 1 << k, n, dtype=np.uint64).astype(np.uint32)
        b = cc.pack32(v, k)
        assert len(b) == 4 * k * ((n + 31) // 32)
        w, off = cc.unpack32(b, 0, n, k)
        assert off == len(b) and np.array_equal(w, v)


@pytest.mark.parametrize("n", [33, cc._SMALL + 33])
def test_pack32_rejects_nonzero_padding(n):
    v = np.arange(n, dtype=np.uint32) % 64
    b = bytearray(cc.pack32(v, 6))
    b[-1] |= 0x80          # the last word's top bits hold value 32 G - 1, a padding slot
    with pytest.raises(cc.ClaimCodecError):
        cc.unpack32(bytes(b), 0, n, 6)


@pytest.mark.parametrize("k", [1, 3, 8, 13, 17, 26, 31, 32])
def test_the_vectorised_lane_and_threaded_packers_agree(k):
    rng = np.random.default_rng(k)
    for n in [cc._SMALL - 5, cc._SMALL, cc._SMALL + 7, cc._THREADED + 77]:
        v = rng.integers(0, 1 << k, n, dtype=np.uint64).astype(np.uint32)
        b = cc.pack32(v, k)
        g = (n + 31) // 32
        vv = np.zeros(32 * g, np.uint32)
        vv[:n] = v
        assert b == cc._pack32_small(vv.reshape(32, g), k, g)
        w = np.frombuffer(b, "<u4").reshape(k, g)
        assert np.array_equal(cc._unpack32_small(w, k, g).reshape(-1)[:n], v)
        for workers in (1, 3, 4):
            assert np.array_equal(cc.unpack32(b, 0, n, k, workers=workers)[0], v)


@pytest.mark.parametrize("k", [1, 5, 17, 26, 30, 31, 32])
def test_the_torch_packer_gives_the_bytes_of_numpy(k):
    g = torch.Generator().manual_seed(k)
    for n in [1, 31, 32, 100, 4097]:
        v = torch.randint(0, 1 << k, (n,), generator=g, dtype=torch.int64)
        assert cc._pack32_torch(v, k) == cc.pack32(v.numpy().astype(np.uint32), k)


def test_rice_round_trip():
    rng = np.random.default_rng(0)
    for v in [np.zeros(0, np.int64), np.zeros(5, np.int64), rng.geometric(0.1, 1000) - 1,
              rng.integers(0, 1 << 29, 300), np.array([(1 << 36) - 1, 0, 3]), np.array([1 << 33] * 40)]:
        b = cc._rice_encode(v)
        w, off = cc._rice_decode(b, 0, v.size)
        assert off == len(b) and np.array_equal(w, v)
    with pytest.raises(ValueError):
        cc._rice_encode(np.array([1 << 36]))


# -- claims: round trips ------------------------------------------------------------------------

def test_round_trip_distributions_and_shapes():
    rng = np.random.default_rng(1)
    zs = [
        np.round(rng.normal(0, 3e4, (64, 8))).astype(np.int64),            # LLM-like, 8 tokens
        np.round(rng.normal(5e3, 2e3, (16, 100))).astype(np.int64) + rng.integers(-9e3, 9e3, (16, 1)),  # row means
        rng.integers(-127, 128, (32, 4)),                                     # embedding look-ups
        np.full((5, 7), 12345, dtype=np.int64),                               # constant
        np.zeros((3, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.int64),   # empty
        np.array([[LIM, -LIM, 0, 1, -1]]),                                    # range-check extremes
        np.array([[LIM, LIM], [-LIM, -LIM]]),                                 # extreme row means
        np.round(rng.standard_cauchy((40, 40)) * 1e3).clip(-LIM, LIM).astype(np.int64),   # heavy tails
        rng.integers(-LIM, LIM, (10, 10)),                                    # incompressible
    ]
    for centre in (False, True):
        assert len(_rt(zs, centre=centre)) < 4 * sum(z.size for z in zs)
    assert len(_rt([])) == 12


def test_bases_are_always_in_the_decoders_range():
    for b in range(cc.BMAX + 1):
        for c in (-(1 << 30), -LIM - (1 << 29), -LIM, -5, 0, 7, LIM, LIM + (1 << 29), 1 << 30):
            lo = cc._base(c, b)
            assert -(1 << 30) < lo <= (1 << 30) - (1 << b)


def test_the_encoder_refuses_claims_outside_the_range_check():
    for z in (1 << 29, -(1 << 29)):
        with pytest.raises(ValueError):
            cc.encode([np.array([[z]])])


def test_numpy_torch_and_decoded_forms_agree():
    rng = np.random.default_rng(4)
    zs = [np.round(rng.normal(0, 3e4, (64, 8))).astype(np.int64), rng.integers(-127, 128, (8, 5))]
    blob = cc.encode(zs)
    assert blob == cc.encode([torch.from_numpy(z) for z in zs]) == cc.encode([torch.from_numpy(z).int() for z in zs])
    wide = cc.decode_torch(blob, [64, 8], max_cols=8)
    narrow = cc.decode_torch(blob, [64, 8], max_cols=8, dtype=torch.int32)
    threaded = cc.decode(blob, [64, 8], max_cols=8, workers=4)
    for z, a, b, c in zip(zs, wide, narrow, threaded):
        assert a.dtype == torch.int64 and b.dtype == torch.int32
        assert torch.equal(a, torch.from_numpy(z)) and torch.equal(b.long(), a) and np.array_equal(c, z)


def test_decoder_claims_shrink_and_round_trip():
    cfg = DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64)
    graph = build_decoder(cfg, calib_tokens=16, seed=3)
    tokens = torch.randint(0, cfg.vocab, (1, 16), generator=torch.Generator().manual_seed(5))
    _, claims = graph.forward(tokens)
    zs = [claims[op.name] for op in graph.mat_ops]
    blob = cc.encode(zs)
    out = cc.decode_torch(blob, [op.n_rows for op in graph.mat_ops], max_cols=tokens.numel())
    assert all(torch.equal(a, b) for a, b in zip(zs, out))
    assert len(blob) < 0.75 * 4 * sum(z.numel() for z in zs)


def test_threaded_decoding_gives_the_same_claims_and_rejections(monkeypatch):
    # every step split over the workers (lanes, exceptions, ops), in jobs above the pool's threshold
    monkeypatch.setattr(cc, "_SMALL", 1 << 9)
    monkeypatch.setattr(cc, "_THREADED", 1 << 11)
    rng = np.random.default_rng(7)
    zs = [np.round(rng.standard_cauchy((64, 900)) * 2e3).clip(-LIM, LIM).astype(np.int64),
          np.round(rng.normal(0, 3e4, (160, 400))).astype(np.int64) + rng.integers(-9e4, 9e4, (160, 1)),
          rng.integers(-127, 128, (32, 700))]
    rows = [z.shape[0] for z in zs]
    blob = cc.encode(zs)
    for workers in (1, 2, 3, 4):
        assert all(np.array_equal(a, b) for a, b in zip(zs, cc.decode(blob, rows, 900, workers=workers)))
    bad = bytearray(blob)
    bad[-1] ^= 0x80                                           # the payloads' last unary level: padding
    for workers in (1, 4):
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(bytes(bad), rows, 900, workers=workers)


def test_the_gpu_prover_encodes_the_bytes_of_the_cpu(device):
    rng = np.random.default_rng(6)
    zs = [torch.from_numpy(np.round(rng.normal(0, 3e4, (256, 64))).astype(np.int64)),
          torch.from_numpy(rng.integers(-127, 128, (64, 33)))]
    assert cc.encode([z.to(device) for z in zs]) == cc.encode(zs)


def test_field_packing_round_trip_and_size():
    g = torch.Generator().manual_seed(0)
    ts = [torch.randint(0, P, (3, 77), generator=g), torch.randint(0, P, (5, 2), generator=g)]
    b = cc.pack_field(ts)
    assert len(b) == cc.field_size(3 * 77 + 10) == 4 * 31 * ((241 + 31) // 32)
    assert np.array_equal(cc.unpack_field(b, 241, workers=2), torch.cat([t.reshape(-1) for t in ts]).numpy())
    for bad, n in ((b + b"\0", 241), (b[:-1], 241), (b, 240), (b, 272), (bytearray(b), 241)):
        with pytest.raises(cc.ClaimCodecError):
            cc.unpack_field(bad, n)
    with pytest.raises(ValueError):
        cc.pack_field([torch.tensor([1 << 31])])


# -- claims: malformed input --------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(2)
    zs = [np.round(rng.normal(0, 2e4, (24, 9))).astype(np.int64),
          np.round(rng.normal(3e3, 1e3, (6, 40))).astype(np.int64) + rng.integers(-5e3, 5e3, (6, 1)),
          rng.integers(-127, 128, (10, 3)), rng.integers(-LIM, LIM, (2, 2))]
    blob = cc.encode(zs)
    # op headers at bytes 8 (plain), 18 (centred: the row means' B and lo, then the residuals'), 33, 43
    assert [blob[12], blob[22], blob[37], blob[47]] == [0, cc._F_CENTRED, 0, 0]
    return zs, blob, [z.shape[0] for z in zs], 40


def _decode_or_reject(blob, rows, max_cols):
    """Anything but a ClaimCodecError or well-formed int32 matrices is a decoder bug."""
    try:
        out = cc.decode(blob, rows, max_cols)
    except cc.ClaimCodecError:
        return None
    assert [z.shape[0] for z in out] == rows and all(z.dtype == np.int32 for z in out)
    assert all(z.shape[1] <= max_cols for z in out)
    return out


def test_every_truncation_is_rejected(sample):
    _, blob, rows, mc = sample
    for n in range(len(blob)):
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(blob[:n], rows, mc)


def test_trailing_bytes_magic_op_count_and_type_are_rejected(sample):
    _, blob, rows, mc = sample
    for bad, r in [(blob + b"\0", rows), (b"PVC2" + blob[4:], rows), (blob, rows[:-1]), (blob, rows + [3]),
                   (bytearray(blob), rows), (memoryview(blob), rows), ("PVC3", rows)]:
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(bad, r, mc)


def test_header_fields_are_validated(sample):
    _, blob, rows, mc = sample
    # op 0 (plain) at byte 8: u32 M, flags, u8 B, i32 lo; op 1 (centred) at byte 18
    def bad(at, value):
        b = bytearray(blob)
        b[at:at + len(value)] = value
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(bytes(b), rows, mc)

    bad(8, struct.pack("<I", mc + 1))                        # more columns than the query has
    for flags in (2, 3, 0x80):                               # other flag bits
        bad(12, bytes([flags]))
    bad(13, bytes([cc.BMAX + 1]))                            # width too large
    bad(14, struct.pack("<i", (1 << 30) - 1))                # base: lo + 2**B > 2**30
    bad(14, struct.pack("<i", -(1 << 30)))                   # base: lo <= -2**30
    bad(23, bytes([cc.BMAX + 1]))                            # the row means' width
    bad(24, struct.pack("<i", 1 << 30))                      # the row means' base
    head = 8 + 10 + 15 + 10 + 10
    assert head % 4 == 1                                     # three bytes of padding follow the headers
    bad(head, b"\1")


def test_a_stream_longer_than_the_input_and_too_many_exceptions_are_rejected():
    blob = cc.encode([np.arange(12).reshape(3, 4) * 1000])
    b = bytearray(blob)
    b[8:12] = struct.pack("<I", 4000)                        # 3 x 4000 slots of the op's width
    with pytest.raises(cc.ClaimCodecError, match="longer than the input"):
        cc.decode(bytes(b), [3], 4000)
    zero = cc._assemble([(1000, [(0, 0, torch.zeros(3000, dtype=torch.int32))])])   # B = 0: no slot bytes
    assert zero[-4:] == struct.pack("<I", 0)
    with pytest.raises(cc.ClaimCodecError, match="exceptions"):
        cc.decode(zero[:-4] + struct.pack("<I", 2000), [3], 1000)
    assert cc.decode(zero, [3], 1000)[0].shape == (3, 1000)


def test_random_corruptions_never_crash(sample):
    zs, blob, rows, mc = sample
    rng = np.random.default_rng(3)
    changed = 0
    for _ in range(600):
        b = bytearray(blob)
        for _ in range(int(rng.integers(1, 4))):
            b[int(rng.integers(0, len(b)))] ^= 1 << int(rng.integers(0, 8))
        out = _decode_or_reject(bytes(b), rows, mc)
        if out is not None and not all(np.array_equal(a, c) for a, c in zip(zs, out)):
            changed += 1
    # a flipped slot bit is a different (well-formed) claim: the protocol's checks catch it
    assert changed > 0


def _one_exception(h: int, b: int, slot: int = 1, lo: int = 0) -> bytes:
    """One op, one value: width ``b``, base ``lo``, the given slot, one exception of high part ``h``."""
    head = cc.MAGIC + struct.pack("<I", 1) + struct.pack("<IB", 1, 0) + struct.pack("<Bi", b, lo)
    head += bytes(-len(head) % 4)
    hz = ((h << 1) ^ (h >> 63)) - 1
    slots = cc.pack32(np.array([slot if b else 0], dtype=np.uint32), b)
    return head + slots + struct.pack("<I", 1) + cc._rice_encode(np.array([0])) + cc._rice_encode(np.array([hz]))


def test_exceptions_are_range_checked():
    assert cc.decode(_one_exception(5, 0), [1], 1)[0][0, 0] == 5
    assert cc.decode(_one_exception(-3, 4), [1], 1)[0][0, 0] == 1 - 3 * 16
    assert cc.decode(_one_exception((1 << 28) - 1, 2), [1], 1)[0][0, 0] == 1 + ((1 << 28) - 1) * 4   # < 2**30
    assert cc.decode(_one_exception(-(1 << 28), 2), [1], 1)[0][0, 0] == 1 - (1 << 30)                 # > -2**30
    assert cc.decode(_one_exception(1, 29, 0, -(1 << 29)), [1], 1)[0][0, 0] == 0
    for h, b in (((1 << 28), 2), (-(1 << 28) - 1, 2), ((1 << 29), 1), ((1 << 33), 0), (-(1 << 34), 3)):
        with pytest.raises(cc.ClaimCodecError, match="exception out of range"):
            cc.decode(_one_exception(h, b), [1], 1)


def test_exception_positions_are_checked():
    ok = _one_exception(5, 0)
    tail = struct.pack("<I", 1) + cc._rice_encode(np.array([0])) + cc._rice_encode(np.array([9]))
    assert ok.endswith(tail)
    for gaps in ([1], [(1 << 36) - 1]):                       # position 1 of 1 value, far beyond
        bad = ok[:-len(tail)] + struct.pack("<I", 1) + cc._rice_encode(np.array(gaps)) + cc._rice_encode(np.array([9]))
        with pytest.raises(cc.ClaimCodecError, match="position"):
            cc.decode(bad, [1], 1)
    # a sum of gaps past 2**63 (at least 2**27 exceptions of the largest gap) wraps: never in range
    with pytest.raises(cc.ClaimCodecError, match="position"):
        cc._patch(np.zeros(10, np.int32), np.full(4, 1 << 62, np.int64), np.zeros(4, np.int64), [(0, 1, 0)], 10, 1)
    b = bytearray(ok)
    b[-len(tail) + 4] = 33                                    # the gaps' Rice parameter k > 32
    with pytest.raises(cc.ClaimCodecError, match="Rice"):
        cc.decode(bytes(b), [1], 1)
