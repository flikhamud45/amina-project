"""The GPU verifier's decoder in bounded memory (``--decode-memory low|auto``: ``claimcodec.decode_device_parts``).

It gives :func:`claimcodec.decode_device`'s claims and rejections on every input, whatever the part size (here on the
CPU too, with a torch twin of the Triton stream unpacker, and on a GPU where there is one); the default path is
untouched (``decode_memory`` default, no record); ``auto`` keeps the default decoder when its predicted peak fits;
a streaming query with the low-memory decoder has the default path's verdict, proof bytes and transcript, and records
the path taken; and on a GPU its memory beyond the claims is a fraction of the default decoder's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pvi.v0 as proto
import shapes
from pvi import options
from pvi.kernels import codec_triton
from pvi.models.decoder import build_decoder
from pvi.v0 import claimcodec as cc
from pvi.v0 import query as v0_query
from test_claimcodec import _claim_sets, _one_exception

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _unpack_stream_torch(words: torch.Tensor, out: torch.Tensor, err: torch.Tensor, n: int, b: int) -> None:
    """:func:`codec_triton.unpack_stream` in torch (any device): the values of a ``b``-bit stream into ``out[:n]``
    (cast to its dtype as the kernel's store is), ``err`` set to 1 on a non-zero padding slot."""
    g = words.shape[1]
    w = words.to(torch.int64) & 0xFFFFFFFF
    lane = torch.arange(32, device=words.device)
    bitpos = lane * b
    wi, sh = bitpos >> 5, bitpos & 31
    low = w[wi]
    high = torch.where(((sh + b) > 32)[:, None], w[(wi + 1).clamp(max=b - 1)], 0)
    val = ((low >> sh[:, None]) | (high << (32 - sh)[:, None])) & ((1 << b) - 1)
    v = (lane[:, None] * g + torch.arange(g, device=words.device)).reshape(-1)
    val = val.reshape(-1)
    inside = v < n
    out[v[inside]] = val[inside].to(out.dtype)
    if bool((val[~inside] != 0).any()):
        err.fill_(1)


@pytest.fixture
def torch_unpacker(monkeypatch):
    """Both device decoders on the CPU: the Triton stream unpacker replaced by its torch twin."""
    monkeypatch.setattr(codec_triton, "unpack_stream", _unpack_stream_torch)


def _outcome(fn, *args):
    """``fn``'s claims on the host as numpy, or its ClaimCodecError's message (anything else is a bug)."""
    try:
        return [z.cpu().numpy() for z in fn(*args)]
    except cc.ClaimCodecError as e:
        return str(e)


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return len(a) == len(b) and all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a, b))


def _shapes(zs):
    return [int(np.asarray(z).shape[0]) for z in zs], [int(np.asarray(z).shape[1]) for z in zs]


def _many_exceptions():
    """Claims with about a third of their values exceptions (wide tails), several ops and a centred one."""
    rng = np.random.default_rng(7)
    a = np.round(rng.standard_t(1.5, (40, 70)) * 300).clip(-(1 << 28), 1 << 28).astype(np.int64)
    b = np.round(rng.standard_t(1.2, (9, 333)) * 50).clip(-(1 << 28), 1 << 28).astype(np.int64) \
        + rng.integers(-4000, 4000, (9, 1))
    return [a, b, rng.integers(-127, 128, (5, 6))]


def _check_all(device, chunks):
    sets = dict(_claim_sets(), many=_many_exceptions())
    for name, zs in sets.items():
        rows, cols = _shapes(zs) if zs else ([], [])
        for centre in (True, False):
            blob = cc.encode(zs, centre=centre, impl="host")
            want = [z for z in cc.decode(blob, rows, cols)]
            default = _outcome(cc.decode_device, blob, rows, cols, device)
            assert _same(default, want), (name, centre)
            for chunk in chunks:
                assert _same(_outcome(cc.decode_device_parts, blob, rows, cols, device, chunk), want), \
                    (name, centre, chunk)


def _corruptions(blob: bytes, n: int, seed: int) -> list[bytes]:
    cases = [blob[:k] for k in range(0, len(blob), max(1, len(blob) // 200))] + [blob + b"\0", b"PVC2" + blob[4:]]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        b = bytearray(blob)
        for _ in range(int(rng.integers(1, 4))):
            b[int(rng.integers(0, len(b)))] ^= 1 << int(rng.integers(0, 8))
        cases.append(bytes(b))
    return cases


def _check_rejections(device, n: int):
    zs = _many_exceptions()
    rows, cols = _shapes(zs)
    blob = cc.encode(zs, impl="host")
    for i, case in enumerate(_corruptions(blob, n, 4)):
        default = _outcome(cc.decode_device, case, rows, cols, device)
        for chunk in (5, 1 << 23):
            assert _same(_outcome(cc.decode_device_parts, case, rows, cols, device, chunk), default), (i, chunk)
    for h in (-(1 << 31), (1 << 31) - 1, 1 << 20, 5, -3):
        for b in (0, 3):
            one = _one_exception(h, b)
            assert _same(_outcome(cc.decode_device_parts, one, [1], [1], device, 1),
                         _outcome(cc.decode_device, one, [1], [1], device)), (h, b)


# -- on the CPU (the stream unpacker's torch twin) ----------------------------------------------------------

def test_the_twin_unpacker_gives_the_host_streams(torch_unpacker):
    rng = np.random.default_rng(1)
    for k in (1, 3, 8, 17, 30, 31, 32):
        for n in (1, 31, 33, 1000):
            v = rng.integers(0, 1 << k, n, dtype=np.uint64).astype(np.uint32)
            words = torch.from_numpy(np.frombuffer(cc.pack32(v, k), dtype=np.int32).copy()).view(k, -1)
            out, err = torch.empty(n, dtype=torch.int64), torch.zeros(1, dtype=torch.int32)
            codec_triton.unpack_stream(words, out, err, n, k)
            assert np.array_equal(out.numpy(), v.astype(np.int64)) and int(err) == 0


def test_the_low_memory_decoder_gives_the_default_decoders_claims_in_any_parts(torch_unpacker):
    _check_all("cpu", (1, 3, 64, 1 << 23))


def test_the_low_memory_decoder_rejects_what_the_default_decoder_rejects(torch_unpacker):
    _check_rejections("cpu", 300)


def test_the_predicted_peak_reads_the_header_only():
    zs = _many_exceptions()
    rows, cols = _shapes(zs)
    blob = cc.encode(zs, impl="host")
    total = sum(r * c for r, c in zip(rows, cols))
    peak = cc.device_decode_peak(blob, rows, cols)
    assert peak is not None and peak > len(blob) + 4 * total + cc._PEAK_PER_EXCEPTION * total // 10
    assert cc.device_decode_peak(blob[:9], rows, cols) is None
    assert cc.device_decode_peak(blob, rows, [c + 1 for c in cols]) is None
    with pytest.raises(ValueError):
        cc.decode_device_parts(blob, rows, cols, "cpu", 0)


def test_the_option_defaults_to_the_measured_decoder_and_records_nothing_then(set_options, monkeypatch):
    assert options.Options().decode_memory == "default"
    called = []
    monkeypatch.setattr(cc, "decode_device_parts", lambda *a: called.append(a))
    graph = build_decoder(shapes.LLAMA, calib_tokens=12, seed=3)
    x = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(5))
    prover, verifier = _streaming(graph, "C")
    out = proto.run_query(prover, verifier, x, seed=1, wire=True)
    assert out["accepted"] and "decode_memory" not in out and not called
    set_options(decode_memory="low")       # a verifier on the CPU never takes the GPU's decoder
    out = proto.run_query(prover, verifier, x, seed=1, wire=True)
    assert out["accepted"] and out["decode_memory"] == "default" and not called


def _streaming(graph, mode, device="cpu"):
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=True)
    coms = proto.commit_graph(graph, params.rate) if mode == "C" else {}
    prover = proto.Prover(graph, commitments=coms)
    v = proto.Verifier(graph.public(), params, mode, publics={k: c.public for k, c in coms.items()},
                       weights={op.name: (op.weight, op.bias) for op in graph.mat_ops}, device=device, stream=True)
    if mode == "Kpre":
        v.precompute(proto.Challenger(seed=9))
    return prover, v


@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_a_query_with_the_low_memory_decoder_has_the_default_proof(mode, torch_unpacker, set_options, monkeypatch):
    """The CPU stands in for the GPU (the low-memory decoder forced on it): the same verdict, proof bytes and
    transcript as the default path, and the path recorded; a tampered encoding is rejected alike."""
    graph = build_decoder(shapes.LLAMA, calib_tokens=12, seed=3)
    x = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(5))
    prover, verifier = _streaming(graph, mode)
    base = proto.run_query(prover, verifier, x, seed=1, wire=True)
    set_options(decode_memory="low", decode_chunk=7)
    monkeypatch.setattr(v0_query, "device_decode_low", lambda *a: True)
    low = proto.run_query(prover, verifier, x, seed=1, wire=True)
    assert base["accepted"] and low["accepted"] and low["decode_memory"] == "low"
    assert (low["transcript"], low["bytes"]) == (base["transcript"], base["bytes"])
    real = cc.encode

    def tampered(zs, **kw):            # one claim off by one: rejected at the same check on both paths
        zs = [z.clone() for z in zs]
        zs[1].view(-1)[3] += 1
        return real(zs, **kw)
    monkeypatch.setattr(cc, "encode", tampered)
    bad_low = proto.run_query(prover, verifier, x, seed=1, wire=True)
    set_options(decode_memory="default")
    bad = proto.run_query(prover, verifier, x, seed=1, wire=True)
    assert not bad["accepted"] and (bad_low["accepted"], bad_low["rejected_at"]) == (False, bad["rejected_at"])


# -- on a GPU ------------------------------------------------------------------------------------------------

def _gpu_ok():
    if not cc.decode_device_ok("cuda"):
        pytest.skip("no Triton")


@cuda_only
def test_on_a_gpu_the_low_memory_decoder_gives_the_default_decoders_claims():
    _gpu_ok()
    _check_all("cuda", (1, 3, 64, 1 << 23))


@cuda_only
def test_on_a_gpu_the_low_memory_decoder_rejects_what_the_default_decoder_rejects():
    _gpu_ok()
    _check_rejections("cuda", 200)


@cuda_only
def test_on_a_gpu_auto_keeps_the_default_decoder_when_it_fits(set_options, monkeypatch):
    _gpu_ok()
    graph = build_decoder(shapes.LLAMA, calib_tokens=12, seed=3)
    x = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(5))
    prover, verifier = _streaming(graph, "C", "cuda")
    set_options(decode_memory="auto")
    out = proto.run_query(prover, verifier, x, seed=1, wire=True)
    assert out["accepted"] and out["decode_memory"] == "default"
    assert v0_query.device_decode_low(verifier, b"PVC3", x) is False     # malformed: the default decoder rejects
    # a predicted peak beyond the GPU (the tiny model's own is far below any budget): auto takes the low decoder,
    # and the prediction it replaces was read from this query's header (not None)
    real = []

    def huge(buf, rows, cols):
        real.append(cc.device_decode_peak(buf, rows, cols))
        return 1 << 50
    monkeypatch.setattr(v0_query.claimcodec, "device_decode_peak", huge)
    low = proto.run_query(prover, verifier, x, seed=1, wire=True)
    assert real and all(p is not None and p > 0 for p in real)
    assert low["accepted"] and low["decode_memory"] == "low" and low["transcript"] == out["transcript"]


@cuda_only
def test_on_a_gpu_the_low_memory_decoder_takes_a_fraction_of_the_default_memory():
    """About 4 M claims with a third of them exceptions: beyond the int32 claims, the default decoder holds the input
    and about eleven int64 vectors of the exceptions, the low-memory one the largest stream, then 1.3 bytes of
    Rice values per exception and its parts' temporaries."""
    _gpu_ok()
    rng = np.random.default_rng(3)
    zs = [np.round(rng.standard_t(1.5, (512, 2048)) * 300).clip(-(1 << 28), 1 << 28).astype(np.int64)
          for _ in range(4)]
    rows, cols = _shapes(zs)
    blob = cc.encode(zs, impl="host")
    total = sum(r * c for r, c in zip(rows, cols))
    extra = {}
    for way, fn in (("default", lambda: cc.decode_device(blob, rows, cols, "cuda")),
                    ("low", lambda: cc.decode_device_parts(blob, rows, cols, "cuda", 1 << 18))):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        got = fn()
        torch.cuda.synchronize()
        extra[way] = torch.cuda.max_memory_allocated() - before - 4 * total
        extra[way + " claims"] = [z.cpu().numpy() for z in got]
        del got
    assert _same(extra["default claims"], extra["low claims"])
    print("decoder memory beyond the claims (MiB):", extra["default"] / 2**20, extra["low"] / 2**20)
    assert extra["low"] < extra["default"] / 3
