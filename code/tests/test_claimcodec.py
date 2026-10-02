"""The compact wire encoding of the proof: ``pvi.fullcheck.claimcodec``.

Slot streams, Rice vectors and 31-bit field elements round trip; the vectorised, lane-by-lane,
threaded and torch (GPU prover) paths give the same bytes and values; every malformed input is
rejected with ``ClaimCodecError`` (truncations, trailing bytes, header fields, padding,
exceptions) and random corruptions never make the decoder raise anything else.  The host and the
device encoders give the bytes of the per-op encoder that defined the format
(``claimcodec_reference``) on every input, threaded or not; the device encoder's kernels grow with
the widths, not the ops, and it waits for the device four times at most (on a GPU: checked in
CUDA's sync debug mode).  In parts (a proof of more than ``_CHUNK`` values; here with tiny parts) the
device encoder gives the same bytes on every input, its temporaries bounded by the part.
"""

from __future__ import annotations

import collections
import struct
import threading
import tracemalloc

import claimcodec_reference as ref
import numpy as np
import pytest
import torch

from pvi.fullcheck import claimcodec as cc
from pvi.fullcheck import commitment, opcount
from pvi.fullcheck.protocol import P, Z_BOUND
from pvi.fullcheck.transformer import DecoderConfig, build_decoder

LIM = Z_BOUND - 1
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return request.param


def _rt(zs, **kw) -> bytes:
    blob = cc.encode(zs, **kw)
    out = cc.decode(blob, [z.shape[0] for z in zs], [z.shape[1] for z in zs])
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
        words = cc._d2h(cc._pack_dev(v, k))                      # int32 [k, G]: the words' bytes
        assert words.astype("<i4", copy=False).tobytes() == cc.pack32(v.numpy().astype(np.uint32), k)


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
        with pytest.raises(cc.Unencodable):
            cc.encode([np.array([[z]])])
    for rows in (torch.tensor([[128]]), torch.tensor([[-129]])):    # nor rows outside int8
        with pytest.raises(cc.Unencodable):
            cc.pack_rows([rows])


def test_numpy_torch_and_decoded_forms_agree():
    rng = np.random.default_rng(4)
    zs = [np.round(rng.normal(0, 3e4, (64, 8))).astype(np.int64), rng.integers(-127, 128, (8, 5))]
    blob = cc.encode(zs)
    assert blob == cc.encode([torch.from_numpy(z) for z in zs]) == cc.encode([torch.from_numpy(z).int() for z in zs])
    wide = cc.decode_torch(blob, [64, 8], [8, 5])
    narrow = cc.decode_torch(blob, [64, 8], [8, 5], dtype=torch.int32)
    threaded = cc.decode(blob, [64, 8], [8, 5], workers=4)
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
    out = cc.decode_torch(blob, [op.n_rows for op in graph.mat_ops], graph.public().claim_columns(tokens))
    assert all(torch.equal(a, b) for a, b in zip(zs, out))
    assert len(blob) < 0.75 * 4 * sum(z.numel() for z in zs)


def test_threaded_decoding_gives_the_same_claims_and_rejections(monkeypatch):
    # every step split over the workers (lanes, exceptions, ops), in jobs above the pool's threshold
    monkeypatch.setattr(cc, "_SMALL", 1 << 9)
    monkeypatch.setattr(cc, "_THREADED", 1 << 11)
    monkeypatch.setattr(cc, "_SPLIT", 1 << 6)
    rng = np.random.default_rng(7)
    zs = [np.round(rng.standard_cauchy((64, 900)) * 2e3).clip(-LIM, LIM).astype(np.int64),
          np.round(rng.normal(0, 3e4, (160, 400))).astype(np.int64) + rng.integers(-9e4, 9e4, (160, 1)),
          rng.integers(-127, 128, (32, 700))]
    rows, cols = [z.shape[0] for z in zs], [z.shape[1] for z in zs]
    blob = cc.encode(zs)
    for workers in (1, 2, 3, 4):
        assert all(np.array_equal(a, b) for a, b in zip(zs, cc.decode(blob, rows, cols, workers=workers)))
    bad = bytearray(blob)
    bad[-1] ^= 0x80                                           # the payloads' last unary level: padding
    for workers in (1, 4):
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(bytes(bad), rows, cols, workers=workers)


@pytest.mark.parametrize("shape", [(4096, 64), (1 << 18, 1), (600, 600)])
def test_a_width_0_stream_of_a_threaded_size_round_trips(shape):
    # B = 0 has no slot stream, so none of its (at least _THREADED) values goes to a worker
    z = np.zeros(shape, np.int64)
    if shape == (600, 600):
        z[3, 7] = 1 << 20                                     # mostly constant: B = 0 and one exception
    blob = _rt([z])
    assert (blob[12], blob[13]) == (0, 0) and z.size >= cc._THREADED     # one plain op at B = 0
    assert np.array_equal(cc.decode(blob, [shape[0]], [shape[1]], workers=4)[0], z)


def test_row_constant_claims_centre_to_a_width_0_stream_and_round_trip():
    z = np.repeat(np.random.default_rng(0).integers(-5000, 5000, (4096, 1)), 64, 1)
    blob = _rt([z])
    assert blob[12] == cc._F_CENTRED and blob[18] == 0        # the residuals: all 0, at B = 0


def _zeros_header(m: int) -> bytes:
    """One op of ``M = m`` columns at ``B = 0``, no exception: 24 bytes, whatever its size."""
    head = cc.MAGIC + struct.pack("<I", 1) + struct.pack("<IB", m, 0) + struct.pack("<Bi", 0, 0)
    return head + bytes(-len(head) % 4) + struct.pack("<I", 0)


def test_a_width_0_header_decodes_only_to_the_expected_shape():
    out = _decode_or_reject(_zeros_header(64), [4096], [64])
    assert out is not None and not out[0].any()               # well formed: 4096 x 64 zeros
    # 2**36 values claimed in 24 bytes: rejected in the header, before anything is allocated
    tracemalloc.start()
    try:
        with pytest.raises(cc.ClaimCodecError, match="column count"):
            cc.decode(_zeros_header(1 << 24), [4096], [64])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4096 * 64 * 4


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
    for bad in (1 << 31, -1):
        with pytest.raises(cc.Unencodable):
            cc.pack_field([torch.tensor([bad])])


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
    return zs, blob, [z.shape[0] for z in zs], [z.shape[1] for z in zs]


def _decode_or_reject(blob, rows, cols):
    """Anything but a ClaimCodecError or well-formed int32 matrices of the given shapes is a decoder bug."""
    try:
        out = cc.decode(blob, rows, cols)
    except cc.ClaimCodecError:
        return None
    assert [z.shape for z in out] == list(zip(rows, cols)) and all(z.dtype == np.int32 for z in out)
    return out


def test_every_truncation_is_rejected(sample):
    _, blob, rows, cols = sample
    for n in range(len(blob)):
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(blob[:n], rows, cols)


def test_trailing_bytes_magic_op_count_and_type_are_rejected(sample):
    _, blob, rows, cols = sample
    for bad, r in [(blob + b"\0", rows), (b"PVC2" + blob[4:], rows), (blob, rows[:-1]), (blob, rows + [3]),
                   (bytearray(blob), rows), (memoryview(blob), rows), ("PVC3", rows)]:
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(bad, r, cols)


def test_header_fields_are_validated(sample):
    _, blob, rows, cols = sample
    # op 0 (plain) at byte 8: u32 M, flags, u8 B, i32 lo; op 1 (centred) at byte 18
    def bad(at, value):
        b = bytearray(blob)
        b[at:at + len(value)] = value
        with pytest.raises(cc.ClaimCodecError):
            cc.decode(bytes(b), rows, cols)

    bad(8, struct.pack("<I", cols[0] + 1))                   # more columns than the query gives the op
    bad(8, struct.pack("<I", cols[0] - 1))                   # fewer
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
        cc.decode(bytes(b), [3], [4000])
    zero = cc._assemble([(1000, [(0, 0, torch.zeros(3000, dtype=torch.int32))])])   # B = 0: no slot bytes
    assert zero[-4:] == struct.pack("<I", 0)
    with pytest.raises(cc.ClaimCodecError, match="exceptions"):
        cc.decode(zero[:-4] + struct.pack("<I", 2000), [3], [1000])
    assert cc.decode(zero, [3], [1000])[0].shape == (3, 1000)


def test_random_corruptions_never_crash(sample):
    zs, blob, rows, cols = sample
    rng = np.random.default_rng(3)
    changed = 0
    for _ in range(600):
        b = bytearray(blob)
        for _ in range(int(rng.integers(1, 4))):
            b[int(rng.integers(0, len(b)))] ^= 1 << int(rng.integers(0, 8))
        out = _decode_or_reject(bytes(b), rows, cols)
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
    assert cc.decode(_one_exception(5, 0), [1], [1])[0][0, 0] == 5
    assert cc.decode(_one_exception(-3, 4), [1], [1])[0][0, 0] == 1 - 3 * 16
    assert cc.decode(_one_exception((1 << 28) - 1, 2), [1], [1])[0][0, 0] == 1 + ((1 << 28) - 1) * 4   # < 2**30
    assert cc.decode(_one_exception(-(1 << 28), 2), [1], [1])[0][0, 0] == 1 - (1 << 30)                 # > -2**30
    assert cc.decode(_one_exception(1, 29, 0, -(1 << 29)), [1], [1])[0][0, 0] == 0
    for h, b in (((1 << 28), 2), (-(1 << 28) - 1, 2), ((1 << 29), 1), ((1 << 33), 0), (-(1 << 34), 3)):
        with pytest.raises(cc.ClaimCodecError, match="exception out of range"):
            cc.decode(_one_exception(h, b), [1], [1])


def test_exception_positions_are_checked():
    ok = _one_exception(5, 0)
    tail = struct.pack("<I", 1) + cc._rice_encode(np.array([0])) + cc._rice_encode(np.array([9]))
    assert ok.endswith(tail)
    for gaps in ([1], [(1 << 36) - 1]):                       # position 1 of 1 value, far beyond
        bad = ok[:-len(tail)] + struct.pack("<I", 1) + cc._rice_encode(np.array(gaps)) + cc._rice_encode(np.array([9]))
        with pytest.raises(cc.ClaimCodecError, match="position"):
            cc.decode(bad, [1], [1])
    # a sum of gaps past 2**63 (at least 2**27 exceptions of the largest gap) wraps: never in range
    with pytest.raises(cc.ClaimCodecError, match="position"):
        cc._patch(np.zeros(10, np.int32), np.full(4, 1 << 62, np.int64), np.zeros(4, np.int64), [(0, 1, 0)], 10, 1)
    b = bytearray(ok)
    b[-len(tail) + 4] = 33                                    # the gaps' Rice parameter k > 32
    with pytest.raises(cc.ClaimCodecError, match="Rice"):
        cc.decode(bytes(b), [1], [1])


# -- the host and device encoders against the reference -------------------------------------------

def _claim_sets() -> dict:
    """Claim lists covering the planner's cases: sampling steps (by rows and by values), centring and its
    boundary (M = 1, 2), widths 0..30, exception-heavy and incompressible ops, extremes, empty ops,
    dtypes and layouts, many ops of few widths, and decoder claims."""
    rng = np.random.default_rng(11)

    def normal(shape, sd, mean=0.0):
        return np.round(rng.normal(mean, sd, shape)).clip(-LIM, LIM).astype(np.int64)

    def rows_off(n, m, sd):
        return normal((n, m), sd) + rng.integers(-40 * sd, 40 * sd, (n, 1))

    sets = {
        "llm": [normal((64, 8), 3e4), normal((512, 8), 3e4), normal((3000, 1), 2e5), normal((1, 3000), 50)],
        "row_means": [rows_off(300, 64, 2e3), rows_off(70, 3, 50), rows_off(20, 2, 9)],
        "sampling": [normal((70_000, 1), 1e4), normal((300, 400), 1e3), normal((257, 255), 77),
                     normal((513, 2), 5e3), normal((1, 70_000), 3)],
        "tails": [np.round(rng.standard_cauchy((90, 90)) * 1e3).clip(-LIM, LIM).astype(np.int64),
                  normal((50, 60), 1e7), rng.integers(-LIM, LIM, (33, 31))],
        "narrow": [np.zeros((40, 9), np.int64), np.full((7, 7), 1234, np.int64), rng.integers(-1, 2, (80, 5)),
                   rng.integers(-127, 128, (32, 4)), np.repeat(rng.integers(-5000, 5000, (256, 1)), 16, 1)],
        "extremes": [np.array([[LIM, -LIM, 0, 1, -1]]), np.array([[LIM, LIM], [-LIM, -LIM]]),
                     np.full((3, 4), -LIM, np.int64)],
        "empty": [np.zeros((0, 4), np.int64), normal((5, 6), 100), np.zeros((3, 0), np.int64),
                  np.zeros((0, 0), np.int64)],
        "all_empty": [np.zeros((0, 4), np.int64), np.zeros((3, 0), np.int64)],
        "dtypes": [torch.from_numpy(normal((30, 8), 3e3)).int(), normal((12, 5), 10).astype(np.int16),
                   torch.from_numpy(normal((9, 40), 2e4)).T, torch.from_numpy(normal((16, 16), 3e2))[::2, 1::3]],
        "one_op": [normal((4096, 64), 2e4)],
    }
    many = [normal((int(rng.integers(1, 90)), int(rng.integers(1, 12))), float(rng.choice([1e2, 1e4])))
            for _ in range(60)]
    sets["many_ops"] = many * 5                                   # 300 ops, few widths
    for kind, cfg in (("gpt", DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64)),
                      ("qwen", DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                                             pos="rope", bias=False, qk_norm=True))):
        graph = build_decoder(cfg, calib_tokens=16, seed=3)
        tokens = torch.randint(0, cfg.vocab, (1, 16), generator=torch.Generator().manual_seed(5))
        _, claims = graph.forward(tokens)
        sets[f"decoder_{kind}"] = [claims[op.name] for op in graph.mat_ops]
    return sets


@pytest.fixture(scope="module")
def claim_sets():
    return _claim_sets()


def test_the_host_and_device_encoders_give_the_reference_bytes(claim_sets):
    for name, zs in claim_sets.items():
        for centre in (True, False):
            want = ref.encode(zs, centre=centre)
            for workers in (1, 4):
                assert cc.encode(zs, centre=centre, impl="host", workers=workers) == want, (name, centre, workers)
            assert cc.encode(zs, centre=centre, impl="device") == want, (name, centre)   # torch ops on CPU tensors


def test_the_encoders_threaded_in_small_jobs_give_the_reference_bytes(claim_sets, monkeypatch):
    # every step of the host encoder on the workers (stats, clipping, fills, scans, packing), the lane packer
    monkeypatch.setattr(cc, "_JOB", 1 << 8)
    monkeypatch.setattr(cc, "_SMALL", 1 << 6)
    for name, zs in claim_sets.items():
        want = ref.encode(zs)
        for workers in (2, 3, 8):
            assert cc.encode(zs, impl="host", workers=workers) == want, (name, workers)
        assert cc.encode(zs, impl="device") == want, name


def test_the_encoders_refuse_the_claims_the_reference_refuses():
    for z in (1 << 29, -(1 << 29), 1 << 31, -(1 << 33), 1 << 62):
        zs = [np.zeros((3, 3), np.int64), np.array([[0, z], [1, 2]])]
        with pytest.raises(ref.Unencodable):
            ref.encode(zs)
        for impl in ("host", "device"):
            with pytest.raises(cc.Unencodable):
                cc.encode(zs, impl=impl)
    assert cc.encode([], impl="host") == cc.encode([], impl="device") == ref.encode([])


def test_the_planners_costs_and_centres_are_the_references():
    rng = np.random.default_rng(3)
    for trial in range(300):
        n = int(rng.integers(0, 3000))
        x = (np.round(rng.normal(0, 10.0 ** rng.uniform(0, 6), n)) * rng.choice([1, 1, 7])).astype(np.int64)
        x = x.clip(-LIM, LIM)
        c = ref._median(x)
        if n:
            srt = np.sort(x)
            assert int(cc._round_half(np.int64(srt[(n - 1) // 2] + srt[n // 2]))) == c, trial
            assert int(cc._medians(x.copy(), np.array([0, n]))[0]) == c, trial
        scale = float(rng.uniform(0.5, 300))
        d = x - c
        hist = np.bincount(np.frexp(((d << 1) ^ (d >> 63)).astype(np.float64))[1], minlength=64)
        mine = cc._costs(hist[None], [n], [scale])[0]
        assert np.array_equal(mine, ref._width_costs(x, c, scale)), trial          # float for float
        assert int(np.argmin(mine)) == ref._choose(x, c, scale)[0]
        bins = np.full(n, -1022, dtype=np.int64)
        got = cc._hist(x, np.array([0, n]), np.array([c]), bins, 1, np.empty(n, np.int64), np.empty(n, np.int64))
        assert np.array_equal(got[0], hist), trial


def test_the_rice_parameter_and_bytes_are_the_references():
    rng = np.random.default_rng(5)
    vs = [rng.geometric(p, int(rng.integers(1, 5000))) - 1 for p in (0.9, 0.5, 0.1, 0.01, 0.001)]
    vs += [rng.integers(0, 1 << b, 777) for b in (1, 5, 20, 33, 36)]
    vs += [np.zeros(9, np.int64), np.array([(1 << 36) - 1])]
    for v in vs:
        v = np.asarray(v, dtype=np.int64)
        assert cc._rice_k(v) == ref._rice_k(v)
        assert cc._rice_encode(v) == ref._rice_encode(v)


def test_the_lane_packer_gives_the_references_bytes():
    rng = np.random.default_rng(8)
    for k in (1, 7, 16, 17, 19, 30, 31, 32):
        for n in (cc._SMALL + 1, cc._SMALL + 777, 3 * cc._JOB + 5):
            v = rng.integers(0, 1 << k, n, dtype=np.uint64).astype(np.uint32)
            want = ref.pack32(v, k)
            for workers in (1, 3):
                assert cc.pack32(v, k, workers) == want, (k, n, workers)


def _device_encode(zs):
    """``encode(zs, impl="device")`` with its kernels counted (aten ops) and its waits for the device."""
    with opcount.codec_waits(cc) as waits, opcount.KernelOps() as ops:
        blob = cc.encode(zs, impl="device")
    return blob, ops.n, dict(waits)


def _header_widths(blob: bytes, n_ops: int) -> set:
    widths, off = set(), 8
    for _ in range(n_ops):
        flags = blob[off + 4]
        off += 5
        for _ in range(2 if flags & cc._F_CENTRED else 1):
            widths.add(blob[off])
            off += 5
    return widths


def test_the_device_encoders_kernels_grow_with_the_widths_not_the_ops(claim_sets):
    # decoder claims (centred ops, several widths) repeated 1, 4 and 16 times: the same widths, 4-16x the ops;
    # a new shape set (cold: its tables built on the device) and the same again (warm)
    base = claim_sets["decoder_gpt"]
    counts = []
    for reps in (1, 4, 16):
        zs = [z.clone() for _ in range(reps) for z in base]
        cc._layout.cache_clear()
        (blob, cold, cold_waits), (again, n, waits) = _device_encode(zs), _device_encode(zs)
        assert blob == again == ref.encode(zs)
        assert waits == cold_waits == {"_d2h": 3, "_nonzero_dev": 1}     # statistics, Rice statistics, the encoding
        counts.append((n, cold, len(_header_widths(blob, len(zs)))))
    assert len({w for _, _, w in counts}) == 1 and counts[0][2] >= 3
    assert counts[0][:2] == counts[1][:2] == counts[2][:2], counts
    n, cold, widths = counts[0]
    assert n <= 118 + 12 * (widths + 2), counts                  # fixed work, ~10 a packed stream (Rice: 2 more)
    assert cold - n <= 40, counts                                 # the tables: a fixed number of ops, once
    with opcount.KernelOps() as old:                              # the reference: ~24 aten ops an op
        ref.encode([z.clone() for _ in range(16) for z in base])
    assert old.n > 9 * n


def test_the_device_tables_are_the_host_index(claim_sets):
    # built on the device from per-op arrays: the host encoder's index arrays, element for element
    for name, zs in claim_sets.items():
        lay = cc._Layout(tuple((int(z.shape[0]), int(z.shape[1])) for z in zs))
        if lay.total == 0:
            continue
        lay.host_index()
        n = lay.n_ops
        op_of_frow = np.repeat(np.arange(n, dtype=np.int64), lay.rows)
        for centre in (False, True):
            tb = lay.on(torch.device("cpu"), centre)
            assert np.array_equal(tb["sample_idx"].numpy(), lay.sample_idx), name
            seg = [lay.op_of_sample]
            if centre:
                seg += [n + lay.op_of_sample, 2 * n + lay.op_of_row, 3 * n + op_of_frow]
            segkey = np.concatenate([(k << 32) + (1 << 31) for k in seg])
            assert np.array_equal(torch.repeat_interleave(tb["segbase"], tb["seg_size"]).numpy(), segkey), name
            assert tb["n_keys"] == segkey.size and tb["n_seg"] == len(seg) * n
            if centre:
                assert np.array_equal(tb["srow"].numpy(), lay.srow), name
                assert np.array_equal(tb["sample_row"].numpy(), np.repeat(np.arange(lay.roff[-1]), lay.row_m)), name
                assert np.array_equal(tb["frow"].numpy(), cc._cum0(lay.cols[op_of_frow])), name


def test_a_new_shape_set_releases_the_device_tables_of_the_last(claim_sets):
    # only the layout last used on a device keeps its tables there (not every cached shape set's)
    first, second = claim_sets["llm"], claim_sets["row_means"]
    shapes = [tuple((int(z.shape[0]), int(z.shape[1])) for z in zs) for zs in (first, second)]
    cc.encode(first, impl="device")
    assert cc._layout(shapes[0])._dev
    assert cc.encode(second, impl="device") == ref.encode(second)
    assert not cc._layout(shapes[0])._dev and cc._layout(shapes[1])._dev
    assert cc.encode(first, impl="device") == ref.encode(first)       # rebuilt
    assert cc._layout(shapes[0])._dev and not cc._layout(shapes[1])._dev


# -- the device encoder in parts (proofs of more than _CHUNK claim values) ---------------------------------

def _numel(zs) -> int:
    return sum(int(np.prod(tuple(z.shape))) for z in zs)


def _one_pass(zs, monkeypatch, **kw) -> bytes:
    """The device encoder in one pass however many claims (the encoder of commit c8be5eb)."""
    monkeypatch.setattr(cc, "_CHUNK", 1 << 40)
    return cc.encode(zs, impl="device", **kw)


def _in_parts(zs, chunk: int, monkeypatch, **kw) -> bytes:
    """The device encoder in parts of ``chunk`` values (when there are more claims)."""
    monkeypatch.setattr(cc, "_CHUNK", chunk)
    return cc.encode(zs, impl="device", **kw)


def _normal(rng, shape, sd, mean=0.0):
    return np.round(rng.normal(mean, sd, shape)).clip(-LIM, LIM).astype(np.int64)


def _rows_off(rng, n, m, sd):
    """Rows of different means: centred ops."""
    return _normal(rng, (n, m), sd) + rng.integers(-40 * sd, 40 * sd, (n, 1))


def _centred_ops(blob: bytes, n_ops: int) -> int:
    off, count = 8, 0
    for _ in range(n_ops):
        flags = blob[off + 4]
        count += bool(flags & cc._F_CENTRED)
        off += 5 + 5 * (2 if flags & cc._F_CENTRED else 1)
    return count


def test_the_device_encoder_in_parts_gives_the_bytes_of_one_pass(claim_sets, monkeypatch):
    # every claim set, centred or not, in parts of one value, of 37, of a third and of all values but one: the
    # bytes of the one-pass encoder and of the per-op reference
    for name, zs in claim_sets.items():
        n = _numel(zs)
        for centre in (True, False):
            want = ref.encode(zs, centre=centre)
            assert _one_pass(zs, monkeypatch, centre=centre) == want, name
            for chunk in sorted({1, 37, n // 3 + 1, n - 1}):
                if 0 < chunk < n and n // chunk <= 400:
                    assert _in_parts(zs, chunk, monkeypatch, centre=centre) == want, (name, centre, chunk)


@pytest.mark.parametrize("chunk", [1, 32, 33, 64, 100])
def test_parts_around_and_across_their_boundaries(chunk, monkeypatch):
    # claim totals and single-width streams just below, at and above one and two parts, lanes that end inside a
    # block of columns (padding), streams of a whole number of blocks and one column more; one op of one column
    # or row, or ops of several widths with centred rows; values with exceptions in every part
    rng = np.random.default_rng(chunk)
    cols = max(1, chunk // 32)
    sizes = {chunk - 1, chunk, chunk + 1, 2 * chunk - 1, 2 * chunk, 2 * chunk + 1, 3 * chunk + 31,
             64 * cols, 64 * cols + 1, 64 * cols - 1, 96 * cols + 33}
    for total in sorted(s for s in sizes if s > 0):
        k = total // 6
        cases = [[_normal(rng, (total, 1), 3e4)], [_normal(rng, (1, total), 50)],
                 [np.round(rng.standard_cauchy((total, 1)) * 1e3).clip(-LIM, LIM).astype(np.int64)]]
        if k:
            cases.append([_rows_off(rng, k, 3, 2e3), _normal(rng, (total - 3 * k, 1), 1e2)])
        for zs in cases:
            assert _numel(zs) == total
            for centre in (True, False):
                assert _in_parts(zs, chunk, monkeypatch, centre=centre) == ref.encode(zs, centre=centre), \
                    (chunk, total, [z.shape for z in zs], centre)


def test_parts_over_empty_rows_long_rows_ops_above_a_part_and_far_exceptions(monkeypatch):
    # rows of no values (M = 0) and ops of no rows between the others, rows longer than a part (a window of its
    # own), ops whose statistics alone exceed a part (a sort group of their own), centred ops whose residuals and
    # row means span several parts and blocks, exceptions in every part, and two exceptions thousands of values
    # (hundreds of parts) apart, whose gap is carried across the parts between them
    rng = np.random.default_rng(21)
    far = np.full((3000, 1), 77, np.int64)
    far[0, 0], far[-1, 0] = 5000, -9000                       # a constant op (B = 0) but its first and last values
    zs = [np.zeros((5, 0), np.int64), _normal(rng, (3, 400), 1e4), np.zeros((0, 7), np.int64),
          _rows_off(rng, 300, 9, 2e3), far, _rows_off(rng, 64, 40, 50),
          np.round(rng.standard_cauchy((50, 30)) * 3e3).clip(-LIM, LIM).astype(np.int64),
          np.array([[LIM, -LIM, 0], [-LIM, LIM, 1]]), np.zeros((4, 0), np.int64)]
    lanes, whole = [], []
    real_lanes, real_dev = cc._pack_lanes, cc._pack_dev
    monkeypatch.setattr(cc, "_pack_lanes", lambda vv, k: (lanes.append(k), real_lanes(vv, k))[1])
    monkeypatch.setattr(cc, "_pack_dev", lambda v, k: (whole.append(k), real_dev(v, k))[1])
    for centre in (True, False):
        want = ref.encode(zs, centre=centre)
        if centre:
            assert _centred_ops(want, len(zs)) >= 2
        for chunk in (8, 50, 150, 700):
            lanes.clear()
            whole.clear()
            assert _in_parts(zs, chunk, monkeypatch, centre=centre) == want, (centre, chunk)
            assert lanes and whole, (centre, chunk)            # blocks of lanes, and whole streams
    assert cc.decode(want, [z.shape[0] for z in zs], [z.shape[1] for z in zs])[4][-1, 0] == -9000


def test_parts_of_every_claim_dtype_give_the_bytes_of_one_pass(monkeypatch):
    # int32 (narrow's: no copy), int64, a narrower or a boolean claim (widened slice by slice), transposed and
    # strided claims, and their mixtures
    rng = np.random.default_rng(5)
    base = [torch.from_numpy(_rows_off(rng, 40, 12, 3e3)), torch.from_numpy(_normal(rng, (90, 7), 2e4)),
            torch.from_numpy(rng.integers(-127, 128, (33, 5)))]
    forms = {"int32": [cc.narrow(z) for z in base], "int64": base,
             "mixed": [base[0].int(), base[1], base[2].to(torch.int16)],
             "bool": [base[0], (base[1] > 0), base[2].to(torch.uint8)],
             "views": [base[0].T.contiguous().T, base[1][::1, :], torch.from_numpy(_normal(rng, (66, 10), 9e3))[::2]]}
    for name, zs in forms.items():
        for centre in (True, False):
            want = _one_pass(zs, monkeypatch, centre=centre)
            assert want == cc.encode([z.numpy() for z in zs], centre=centre, impl="host"), name
            for chunk in (16, 100, 500):
                assert _in_parts(zs, chunk, monkeypatch, centre=centre) == want, (name, centre, chunk)


def test_parts_with_wide_indices_give_the_same_bytes(claim_sets, monkeypatch):
    # the int64 indices of more than 2**31 values (forced here)
    real = cc._Parts.__init__

    def wide(self, *a, **k):
        real(self, *a, **k)
        self.it, self.huge = torch.int64, 1 << 62

    monkeypatch.setattr(cc._Parts, "__init__", wide)
    for name in ("row_means", "decoder_gpt", "tails"):
        zs = claim_sets[name]
        assert _in_parts(zs, _numel(zs) // 5 + 1, monkeypatch) == ref.encode(zs), name


def test_parts_refuse_the_claims_one_pass_refuses(monkeypatch):
    for z in (1 << 29, -(1 << 29), 1 << 31, -(1 << 33), 1 << 62):
        zs = [np.zeros((30, 3), np.int64), np.array([[0, z], [1, 2]]), np.ones((20, 2), np.int64)]
        for chunk in (4, 50):
            with pytest.raises(cc.Unencodable):
                _in_parts(zs, chunk, monkeypatch)
            with pytest.raises(cc.Unencodable):
                _in_parts([cc.narrow(torch.from_numpy(a)) for a in zs], chunk, monkeypatch)


def test_the_rice_parameter_from_bit_lengths_and_top_bits_is_rice_k():
    rng = np.random.default_rng(9)
    vs = [rng.geometric(p, int(rng.integers(1, 4000))) - 1 for p in (0.9, 0.5, 0.1, 0.01, 0.001)]
    vs += [rng.integers(0, 1 << b, int(rng.integers(1, 900))) for b in range(0, 37)]
    vs += [np.zeros(7, np.int64), np.array([(1 << 36) - 1]), np.array([1, 2, 3]), np.array([15, 16, 17]),
           np.array([1 << 33] * 9), rng.integers(0, 16, 50)]
    for v in vs:
        v = np.asarray(v, dtype=np.int64)
        counts = np.bincount(cc._rice_bins(torch.from_numpy(v)).numpy(), minlength=cc._RBITS * 16)
        assert cc._rice_k_bins(v.size, counts) == cc._rice_k(v) == ref._rice_k(v), v[:5]
    with pytest.raises(ValueError):
        cc._rice_k_bins(2, np.bincount(cc._rice_bins(torch.tensor([3, 1 << 36])).numpy(), minlength=cc._RBITS * 16))


def test_the_parts_temporaries_are_bounded_by_the_part_not_the_claims(monkeypatch):
    # every tensor an op creates while encoding (views aside): in parts, at most a few times the part's bytes
    # (plus the samples' and rows' tables, small here), whatever the claims; in one pass, several times the claims
    from torch.utils._python_dispatch import TorchDispatchMode

    class Largest(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.bytes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if not any(str(func).startswith("aten." + v) for v in opcount.VIEWS):
                for t in out if isinstance(out, (tuple, list)) else (out,):
                    if torch.is_tensor(t):
                        self.bytes = max(self.bytes, t.numel() * t.element_size())
            return out

    monkeypatch.setattr(cc, "_PLAN_ROWS", 4)          # few samples: the tables stay small next to the claims
    monkeypatch.setattr(cc, "_PLAN_VALUES", 256)
    rng = np.random.default_rng(4)
    zs = [cc.narrow(torch.from_numpy(_rows_off(rng, 64, 500, 3e3) if i % 2 else _normal(rng, (64, 500), 2e4)))
          for i in range(8)]
    n = _numel(zs)                                     # 256,000 int32 claim values: 1 MB
    chunk = 2048
    results = {}
    for way in ("one pass", "in parts"):
        cc._layout.cache_clear()                       # the tables are built (and measured) again
        with Largest() as largest:
            results[way] = _one_pass(zs, monkeypatch) if way == "one pass" else _in_parts(zs, chunk, monkeypatch)
        results[way + " bytes"] = largest.bytes
    assert results["one pass"] == results["in parts"]
    assert results["one pass bytes"] >= 8 * n          # e.g. the int64 cumulative sums of every claim
    assert results["in parts bytes"] <= 8 * 2 * chunk + 8 * 8 * 500, results     # ~32 KiB here, not ~2 MiB


def test_parts_wait_for_the_device_a_few_times_a_part(monkeypatch):
    rng = np.random.default_rng(6)
    zs = [_normal(rng, (400, 25), 3e4), _rows_off(rng, 100, 30, 1e3)]
    chunk = 500
    monkeypatch.setattr(cc, "_CHUNK", chunk)
    with opcount.codec_waits(cc) as waits:
        assert cc.encode(zs, impl="device") == ref.encode(zs)
    parts = -(-(_numel(zs) + 100) // chunk)           # the global order: the claims and a centred op's row means
    assert waits["_nonzero_dev"] <= 2 * parts                                   # (a) and (c)
    assert waits["_d2h"] <= 2 + parts + 2 * parts + 2, dict(waits)             # stats, Rice, (c), (b) blocks


@cuda_only
def test_the_gpu_encoder_gives_the_reference_bytes(claim_sets):
    for name, zs in claim_sets.items():
        on = [(z if torch.is_tensor(z) else torch.from_numpy(np.asarray(z))).cuda() for z in zs]
        for centre in (True, False):
            assert cc.encode(on, centre=centre) == ref.encode(zs, centre=centre), (name, centre)
    for z in (1 << 29, 1 << 40):
        with pytest.raises(cc.Unencodable):
            cc.encode([torch.tensor([[0, z]], device="cuda")])
        with pytest.raises(cc.Unencodable):
            cc.encode([cc.narrow(torch.tensor([[0, z]], device="cuda"))])


@cuda_only
def test_the_gpu_encoder_in_parts_gives_the_reference_bytes_in_bounded_memory(claim_sets, monkeypatch):
    for name, zs in claim_sets.items():
        on = [(z if torch.is_tensor(z) else torch.from_numpy(np.asarray(z))).cuda() for z in zs]
        n = _numel(zs)
        for chunk in sorted({37, n // 3 + 1}):
            if 0 < chunk < n and n // chunk <= 400:
                for centre in (True, False):
                    assert _in_parts(on, chunk, monkeypatch, centre=centre) == ref.encode(zs, centre=centre), \
                        (name, chunk, centre)
    # 25 M int32 claims (100 MB, as a lean prover keeps them) in parts of 2**20 values: the encoder's own
    # memory stays near 40 bytes a part's value plus the tables, against about 40 bytes a claim in one pass
    g = torch.Generator(device="cuda").manual_seed(0)
    zs = [cc.narrow(torch.randn(4096, 1024, generator=g, device="cuda").mul_(3e4).round_().to(torch.int64)
                    + (i % 2) * torch.randint(-9000, 9000, (4096, 1), generator=g, device="cuda"))
          for i in range(6)]
    extra, blobs = {}, {}
    for way, chunk in (("one pass", 1 << 40), ("in parts", 1 << 20)):
        monkeypatch.setattr(cc, "_CHUNK", chunk)
        cc._layout.cache_clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        blobs[way] = cc.encode(zs)
        torch.cuda.synchronize()
        extra[way] = torch.cuda.max_memory_allocated() - before
    assert blobs["one pass"] == blobs["in parts"] == cc.encode([z.cpu() for z in zs], impl="host")
    assert extra["in parts"] < 100 * 2**20 < 600 * 2**20 < extra["one pass"], extra


@cuda_only
def test_the_gpu_encoder_waits_four_times_and_launches_kernels_per_width(claim_sets, monkeypatch):
    """In CUDA's sync debug mode "error" any other host round trip raises, on a new shape set (its tables
    built on the device) and on a known one; the kernels (counted by the profiler, in another run) are
    the same for 4x and 16x the ops of the same widths, up to ``cat``'s batches of inputs."""
    base = [z.cuda() for z in claim_sets["decoder_gpt"]]
    waits = []

    def allowed(real):
        def call(t):
            torch.cuda.set_sync_debug_mode(0)
            try:
                waits.append(real.__name__)
                return real(t)
            finally:
                torch.cuda.set_sync_debug_mode("error")
        return call

    kernels = []
    for reps in (1, 4, 16):
        zs = [z.clone() for _ in range(reps) for z in base]
        want = ref.encode([z.cpu() for z in zs])
        cc._layout.cache_clear()
        for run in ("cold", "warm"):                              # cold: the tables built on the device
            waits.clear()
            monkeypatch.setattr(cc, "_d2h", allowed(cc._d2h))
            monkeypatch.setattr(cc, "_nonzero_dev", allowed(cc._nonzero_dev))
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                assert cc.encode(zs) == want, run
            finally:
                torch.cuda.set_sync_debug_mode(0)
                monkeypatch.undo()
            assert sorted(waits) == ["_d2h", "_d2h", "_d2h", "_nonzero_dev"], run
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:   # a run of its own
            assert cc.encode(zs) == want
            torch.cuda.synchronize()
        kernels.append(sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA))
    widths = len(_header_widths(cc.encode(base), len(base)))
    assert kernels[2] <= kernels[0] + 16 and kernels[0] <= 250 + 30 * (widths + 2), kernels


@cuda_only
def test_field_elements_packed_on_a_gpu_are_the_hosts_bytes():
    g = torch.Generator().manual_seed(4)
    ts = [torch.randint(0, P, (5, 333), generator=g), torch.randint(0, P, (77, 3), generator=g).T]
    want = ref.pack32(torch.cat([t.reshape(-1) for t in ts]).numpy().astype(np.uint32), 31)
    assert cc.pack_field([t.cuda() for t in ts]) == cc.pack_field(ts) == want
    for bad in (-1, 1 << 31, 1 << 40):
        for dev in ("cpu", "cuda"):
            with pytest.raises(cc.Unencodable):
                cc.pack_field([torch.tensor([[1, bad]], device=dev)])


@cuda_only
def test_claims_decoded_into_pinned_memory_are_the_claims():
    rng = np.random.default_rng(2)
    zs = [np.round(rng.normal(0, 3e4, (300, 9))).astype(np.int64), rng.integers(-127, 128, (64, 9))]
    blob = cc.encode(zs)
    out = cc.decode_torch(blob, [300, 64], [9, 9], dtype=torch.int32, pin=True)
    assert all(t.is_pinned() and torch.equal(t.long(), torch.from_numpy(z)) for t, z in zip(out, zs))


def test_random_corruptions_never_crash_the_split_decoder(sample, monkeypatch):
    # the two Rice vectors located by counting their levels' bits and decoded as parallel jobs, on corrupted
    # inputs: the same claims or the same rejection as the one-thread decoder (default sizes), never another
    # exception; every job of the split decoder on the pool's threads (lanes, Rice vectors, patches, means)
    zs, blob, rows, cols = sample
    rng = np.random.default_rng(13)
    bad = []
    for _ in range(400):
        b = bytearray(blob)
        for _ in range(int(rng.integers(1, 4))):
            b[int(rng.integers(0, len(b)))] ^= 1 << int(rng.integers(0, 8))
        bad.append(bytes(b))
    want = [_decode_or_reject(b, rows, cols) for b in bad]
    for name, value in (("_SMALL", 1 << 4), ("_THREADED", 1), ("_SPLIT", 1)):
        monkeypatch.setattr(cc, name, value)
    monkeypatch.setattr(commitment, "LEAF_THREAD_BYTES", 1)        # no job too small for the pool
    caller, ran = threading.get_ident(), collections.Counter()

    def where(fn):
        def job(*a, **k):
            ran[fn.__name__, threading.get_ident() != caller] += 1
            return fn(*a, **k)
        return job

    for fn in ("_rice_job", "_unpack_lanes", "_patch_slice", "_add_means"):
        monkeypatch.setattr(cc, fn, where(getattr(cc, fn)))
    assert all(np.array_equal(a, b) for a, b in zip(zs, cc.decode(blob, rows, cols, workers=3)))
    assert ran[("_rice_job", True)] == 2 and ran[("_rice_job", False)] == 0, ran
    assert ran[("_patch_slice", True)] >= 2 and ran[("_unpack_lanes", True)] >= 3, ran
    assert ran[("_add_means", True)] >= 1, ran
    for b, one in zip(bad, want):
        try:
            split = cc.decode(b, rows, cols, workers=3)
        except cc.ClaimCodecError:
            split = None
        assert (one is None) == (split is None)
        assert one is None or all(np.array_equal(p, q) for p, q in zip(one, split))
    assert ran[("_rice_job", True)] > 400, ran                    # most corruptions reach the Rice vectors
