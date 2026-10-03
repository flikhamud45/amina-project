"""Pinned host memory within the driver's limit (``pvi.fullcheck.hostmem``).

The L40S nodes refuse a single pinned allocation of more than 2 GiB; above ``PIN_MAX`` the defence takes
pageable memory instead.  These tests force every buffer above the limit (``PIN_MAX = 0``) and check that
nothing but the memory changes: the encoder's bytes, the decoder's values and the verdicts, labels and proof
sizes of wire queries with the batched and the streaming verifier on a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import test_wire as tw
from pvi.fullcheck import claimcodec as cc
from pvi.fullcheck import hostmem
from pvi.fullcheck import protocol as proto

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_a_buffer_above_the_limit_is_pageable(monkeypatch):
    monkeypatch.setattr(hostmem, "PIN_MAX", 4 * 10 - 1)
    t = hostmem.empty(10, torch.int32)
    assert t.shape == (10,) and t.dtype == torch.int32 and not t.is_pinned()
    assert not hostmem.empty((2, 5), torch.int32).is_pinned()
    a = torch.arange(10, dtype=torch.int32)
    assert hostmem.pinned(a) is a
    assert not hostmem.empty(1, torch.int32, pin=False).is_pinned()


def test_the_limit_is_what_the_driver_grants():
    assert hostmem.PIN_MAX == 2 ** 31
    assert hostmem._nbytes((2 ** 29,), torch.int32) == hostmem.PIN_MAX
    assert hostmem._nbytes(3, torch.int64) == 24 and hostmem._nbytes((2, 3), torch.uint8) == 6


@cuda_only
def test_a_buffer_within_the_limit_is_pinned(monkeypatch):
    monkeypatch.setattr(hostmem, "PIN_MAX", 4 * 10)
    assert hostmem.empty(10, torch.int32).is_pinned()
    assert hostmem.pinned(torch.arange(10, dtype=torch.int32)).is_pinned()
    assert not hostmem.empty(11, torch.int32).is_pinned()


@cuda_only
def test_pageable_buffers_give_the_gpu_encoders_and_the_decoders_values(monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(0)
    zs = [cc.narrow(torch.randn(512, 96, generator=g, device="cuda").mul_(3e4).round_().to(torch.int64)
                    + (i % 2) * torch.randint(-9000, 9000, (512, 1), generator=g, device="cuda"))
          for i in range(5)]
    rows, cols = [z.shape[0] for z in zs], [z.shape[1] for z in zs]
    want = cc.encode([z.cpu() for z in zs], impl="host")
    for chunk in (1 << 40, 4096):                    # one pass, and in parts (with host slabs and buffers)
        monkeypatch.setattr(cc, "_ONE_PASS", chunk)
        monkeypatch.setattr(cc, "_CHUNK", chunk)
        for limit in (hostmem.PIN_MAX, 0):
            monkeypatch.setattr(hostmem, "PIN_MAX", limit)
            assert cc.encode(zs) == want, (chunk, limit)
    for limit in (2 ** 31, 0):
        monkeypatch.setattr(hostmem, "PIN_MAX", limit)
        got = cc.decode_torch(want, rows, cols, dtype=torch.int32, pin=True)
        assert all(a.is_pinned() == (limit > 0) for a in got)
        assert all(torch.equal(a, z.cpu().to(torch.int32)) for a, z in zip(got, zs))
        assert all(torch.equal(a.cuda(non_blocking=True), z.to(torch.int32)) for a, z in zip(got, zs))


@cuda_only
@pytest.mark.parametrize("kind", ["lenet5", "llama"])
@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_wire_queries_on_a_gpu_are_unchanged_by_pageable_buffers(kind, mode, monkeypatch):
    graph, x = tw._graph(kind)
    results = {}
    for limit in (2 ** 31, 0):
        monkeypatch.setattr(hostmem, "PIN_MAX", limit)
        prover, pair = tw._setup(graph, mode, device="cuda")
        results[limit] = [tw._both(prover, pair, x, seed=s) for s in (1, 2)]
    for a, b in zip(results[2 ** 31], results[0]):
        assert a["accepted"] and b["accepted"]
        assert (a["rejected_at"], a["bytes"]) == (b["rejected_at"], b["bytes"])


@cuda_only
def test_a_tampered_claim_is_rejected_with_pageable_buffers(monkeypatch):
    graph, x = tw._graph("llama")
    monkeypatch.setattr(hostmem, "PIN_MAX", 0)
    prover, pair = tw._setup(graph, "C", device="cuda")
    real = cc.encode

    def tampered(zs, **kw):
        zs = [z.clone() for z in zs]
        zs[1].view(-1)[3] += 1
        return real(zs, **kw)

    monkeypatch.setattr(proto.claimcodec, "encode", tampered)
    for v in pair:
        r = proto.run_query(prover, v, x, seed=1, wire=True)
        assert not r["accepted"]
