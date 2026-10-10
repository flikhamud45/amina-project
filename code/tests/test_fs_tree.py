"""Fiat--Shamir hashing at memory speed (plan 2, FS1): the one-level tree absorption of large messages and the
designated-verifier mode Kpre, which absorbs nothing."""
import hashlib
import struct

import pytest
import torch

from pvi.fullcheck import protocol as proto


def _challenge(ch: proto.Challenger) -> bytes:
    return ch._key("probe")


def _tree_reference(label: bytes, blob: bytes, chunk: int) -> bytes:
    """The tree absorption written out serially, for a Challenger that absorbed nothing before."""
    head = b"pvi/leaf/v1" + len(label).to_bytes(2, "big") + label
    leaves = [hashlib.shake_256(head + j.to_bytes(8, "big") + blob[a:a + chunk]).digest(32)
              for j, a in enumerate(range(0, len(blob), chunk))]
    ref = proto.Challenger(fiat_shamir=True)
    ref.absorb(label + b"/tree", struct.pack(">QQQ", len(leaves), chunk, len(blob)) + b"".join(leaves))
    return _challenge(ref)


@pytest.fixture
def small_tree(monkeypatch):
    monkeypatch.setattr(proto, "_TREE_MIN", 1 << 12)
    monkeypatch.setattr(proto, "_TREE_CHUNK", 1 << 10)


def test_small_messages_are_absorbed_as_before(small_tree):
    blob = bytes(range(256)) * 15                       # below the threshold
    a, b = proto.Challenger(fiat_shamir=True), proto.Challenger(fiat_shamir=True)
    a.absorb(b"m", blob)
    b.absorb_large(b"m", blob)
    assert _challenge(a) == _challenge(b)


@pytest.mark.parametrize("threads", ["1", "8"])
def test_tree_absorption_matches_its_serial_reference(small_tree, monkeypatch, threads):
    monkeypatch.setenv("PVI_HASH_THREADS", threads)
    g = torch.Generator().manual_seed(1)
    for n in (1 << 12, (1 << 12) + 1, 10_000, 1 << 15):
        blob = torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8).numpy().tobytes()
        ch = proto.Challenger(fiat_shamir=True)
        ch.absorb_large(b"claims/PVC3", blob)
        assert _challenge(ch) == _tree_reference(b"claims/PVC3", blob, 1 << 10)
        ch2 = proto.Challenger(fiat_shamir=True)
        ch2.absorb_large(b"claims/PVC3", memoryview(blob))             # a read-only view gives the same
        assert _challenge(ch2) == _challenge(ch)


def test_every_chunk_and_the_framing_bind(small_tree):
    blob = bytearray(range(256)) * 40                  # 10,240 bytes: 10 chunks
    base = proto.Challenger(fiat_shamir=True)
    base.absorb_large(b"m", bytes(blob))
    seen = {_challenge(base)}
    for pos in (0, 5_000, len(blob) - 1):              # a bit flip in the first, a middle and the last chunk
        b = bytearray(blob)
        b[pos] ^= 1
        ch = proto.Challenger(fiat_shamir=True)
        ch.absorb_large(b"m", bytes(b))
        seen.add(_challenge(ch))
    for variant in (bytes(blob) + b"\0", bytes(blob)[:-1]):        # one byte more or less
        ch = proto.Challenger(fiat_shamir=True)
        ch.absorb_large(b"m", variant)
        seen.add(_challenge(ch))
    other_label = proto.Challenger(fiat_shamir=True)
    other_label.absorb_large(b"n", bytes(blob))
    seen.add(_challenge(other_label))
    direct = proto.Challenger(fiat_shamir=True)                     # the same bytes absorbed directly
    direct.absorb(b"m", bytes(blob))
    seen.add(_challenge(direct))
    assert len(seen) == 8                              # base, 3 flips, 2 lengths, another label, direct


def test_kpre_under_fiat_shamir_absorbs_nothing_and_keeps_its_verdicts():
    from pvi.fullcheck.transformer import CONFIGS, build_decoder
    cfg = CONFIGS["gpt2"]
    g = build_decoder(cfg, n_layers=2, calib_tokens=8, seed=0, prune_last=True)
    params = proto.params_for(40, len(g.mat_ops), fiat_shamir=True)
    v = proto.Verifier(g.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in g.mat_ops})
    v.precompute(proto.Challenger())
    prover = proto.Prover(g)
    x = torch.randint(0, cfg.vocab, (1, 16), generator=torch.Generator().manual_seed(3))
    out = proto.run_query(prover, v, x, wire=True)
    assert out["accepted"] and out["timings"]["fs_hash"] < 1e-3
    name = g.mat_ops[3].name

    def tamper(op, z):
        if op.name != name:
            return z
        z = z.clone()
        z.view(-1)[7] += 1
        return z
    bad = proto.run_query(prover, v, x, wire=True, forward_kwargs={"tamper": tamper})
    assert not bad["accepted"]
