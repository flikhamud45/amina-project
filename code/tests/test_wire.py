"""The protocol with the compact wire encoding of the proof: ``run_query(wire=True)``.

In modes C, K and Kpre, interactive and Fiat--Shamir, for a CNN and decoders, with the batched
verifier (and its deferred and int8 forms), the streaming verifier and a GPU client: honest
queries are accepted with smaller proofs; tampered claims, forged ``u``, forged or misplaced opened
columns, forged paths and malformed encodings are rejected at the check that rejects them without
``wire``; the Fiat--Shamir transcript absorbs the encoded bytes; the default flow runs none of the
codec; the security parameters are those of the default flow, which meet ``lambda`` on every model;
and bench.py records wire cells only under a ``_wire`` tag.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pvi.fullcheck import claimcodec as cc
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.analytic import decoder_shapes
from pvi.fullcheck.models import build_float_model
from pvi.fullcheck.quantize import quantize_input, quantize_model
from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder

P = proto.P
_ENCODE, _PACK_FIELD = cc.encode, cc.pack_field     # the codec's own, which tests replace in the protocol


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return request.param


def _plain(zs) -> bytes:
    """A well-formed encoding of any claims ``|z| < 2**30``: every op one segment at ``B = 30``."""
    return cc._assemble([(z.shape[1], [(30, -(1 << 29), torch.as_tensor(z).reshape(-1).to(torch.int32))])
                         for z in zs])


# -- the protocol with wire=True -------------------------------------------------------------------

_TINY = {
    "gpt2": DecoderConfig("tiny-gpt2", 64, 2, 4, 4, 16, 256, 97),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                           bias=False, tied=False),
    "qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                          rope_theta=1e6, bias=False, qk_norm=True),
}
KINDS = ["lenet5", *_TINY]
SETTINGS = [("C", False), ("C", True), ("K", False), ("K", True), ("Kpre", False), ("Kpre", True)]
_GRAPHS: dict = {}


def _graph(kind):
    if kind not in _GRAPHS:
        g = torch.Generator().manual_seed(1)
        if kind == "lenet5":
            torch.manual_seed(0)
            graph = quantize_model(build_float_model("lenet5", 10).eval(), torch.randn(16, 1, 28, 28, generator=g))
            x = quantize_input(graph, torch.randn(1, 1, 28, 28, generator=g))
        else:
            graph = build_decoder(_TINY[kind], calib_tokens=12, seed=3)
            x = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(5))
        _GRAPHS[kind] = graph, x
    return _GRAPHS[kind]


def _setup(graph, mode, fiat_shamir=False, device="cpu"):
    """A prover and the pair (batched verifier, streaming verifier)."""
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir)
    coms = proto.commit_graph(graph, params.rate) if mode == "C" else {}
    prover = proto.Prover(graph, commitments=coms)
    kw = dict(publics={k: c.public for k, c in coms.items()},
              weights={op.name: (op.weight, op.bias) for op in graph.mat_ops}, device=device)
    pair = [proto.Verifier(graph.public(), params, mode, **kw), proto.Verifier(graph.public(), params, mode, stream=True, **kw)]
    if mode == "Kpre":
        for v in pair:
            v.precompute(proto.Challenger(seed=9))
    return prover, pair


def _both(prover, pair, x, seed, **kw):
    """``run_query(wire=True)`` with the batched and the streaming verifier: the same verdict and
    label, and on an accepted query the same proof bytes.  Returns the batched one's result."""
    a, b = (proto.run_query(prover, v, x, seed=seed, wire=True, **kw) for v in pair)
    assert (a["accepted"], a["rejected_at"]) == (b["accepted"], b["rejected_at"])
    assert b["bytes"]["claims"] == a["bytes"]["claims"]
    if a["accepted"]:
        assert b["bytes"] == a["bytes"]
    else:        # the streaming verifier also received the messages run_query no longer asks for
        assert all(a["bytes"][k] in (0, b["bytes"][k]) for k in a["bytes"])
    for r in (a, b):
        assert "prove_encode" in r["timings"] and "verify_decode" in r["timings"]
    return a


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode,fiat_shamir", SETTINGS)
def test_honest_wire_queries_are_accepted_with_smaller_proofs(kind, mode, fiat_shamir):
    graph, x = _graph(kind)
    prover, pair = _setup(graph, mode, fiat_shamir)
    plain = proto.run_query(prover, pair[0], x, seed=1)
    wired = _both(prover, pair, x, seed=1)
    assert plain["accepted"] and wired["accepted"]
    assert wired["bytes"]["claims"] < 0.8 * plain["bytes"]["claims"]
    params = pair[0].params
    if mode == "C":        # the parameters are the default flow's: t columns per op, of every row
        n_cols = [min(params.columns, pair[0].publics[op.name].n_points) for op in graph.mat_ops]
        assert wired["bytes"]["u"] == cc.field_size(params.reps * sum(op.row_length for op in graph.mat_ops))
        assert wired["bytes"]["columns"] == cc.field_size(sum(op.n_rows * t for op, t in zip(graph.mat_ops, n_cols)))
        assert wired["bytes"]["columns"] < plain["bytes"]["columns"]
        if not fiat_shamir:  # the same challenges: the same multiproofs
            assert wired["bytes"]["paths"] == plain["bytes"]["paths"]
    else:
        assert wired["bytes"]["u"] == wired["bytes"]["columns"] == wired["bytes"]["paths"] == 0


def _claim_attacks(mats):
    mid = mats[len(mats) // 2]

    def at(op, change):
        def tamper(o, z):
            return change(z.clone()) if o.name == op.name else z
        return {"tamper": tamper}

    def plus_one(z):
        z.view(-1)[z.numel() // 3] += 1
        return z

    def setting(value):
        def change(z):
            z.view(-1)[-1] = value
            return z
        return change

    return {"honest": {}, "first+1": at(mats[0], plus_one), "mid+1": at(mid, plus_one),
            "last+1": at(mats[-1], plus_one), "1-Z_BOUND": at(mid, setting(1 - proto.Z_BOUND))}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode,fiat_shamir", SETTINGS)
def test_tampered_claims_are_rejected_where_the_default_flow_rejects_them(kind, mode, fiat_shamir):
    graph, x = _graph(kind)
    prover, pair = _setup(graph, mode, fiat_shamir)
    for i, (name, kw) in enumerate(_claim_attacks(graph.mat_ops).items()):
        got = _both(prover, pair, x, seed=i, forward_kwargs=kw)["rejected_at"]
        want = proto.run_query(prover, pair[0], x, seed=i, forward_kwargs=kw)["rejected_at"]
        assert got == want == (None if name == "honest" else "freivalds"), name


@pytest.mark.parametrize("kind", ["lenet5", "qwen"])
@pytest.mark.parametrize("mode,fiat_shamir", SETTINGS)
def test_malformed_or_out_of_range_claim_bytes_are_rejected_at_range_or_shape(kind, mode, fiat_shamir, monkeypatch):
    graph, x = _graph(kind)
    prover, pair = _setup(graph, mode, fiat_shamir)
    encode = cc.encode
    n_ops = len(graph.mat_ops)

    def at_bound(zs):                      # well formed, but a claim at the range check's bound
        zs = [z.clone() for z in zs]
        zs[n_ops // 2].view(-1)[0] = proto.Z_BOUND
        return _plain(zs)

    def more_columns(zs):                  # op 0 claims one more column: more than the query has
        zs = list(zs)
        zs[0] = torch.cat([zs[0], zs[0][:, :1]], 1)
        return encode(zs)

    def fewer_columns(zs):                 # well formed, but op 0 claims one column fewer than it has
        return encode([zs[0][:, :-1], *zs[1:]])

    variants = {"plain": _plain, "trailing": lambda zs: encode(zs) + b"\0", "truncated": lambda zs: encode(zs)[:-1],
                "magic": lambda zs: b"PVC0" + encode(zs)[4:], "ops": lambda zs: encode(zs[:-1]),
                "at bound": at_bound, "more columns": more_columns, "fewer columns": fewer_columns}
    for i, (name, fake) in enumerate(variants.items()):
        monkeypatch.setattr(proto.claimcodec, "encode", fake)
        assert _both(prover, pair, x, seed=i)["rejected_at"] == (None if name == "plain" else "range_or_shape"), name


def _corrupt_field_message(monkeypatch, which: str, change, reps: int):
    """The prover's 31-bit packed ``u`` (``which="u"``) or opened columns, as ``change`` makes them
    (the tensors of ``u`` have ``reps`` rows, those of the columns ``t`` rows)."""
    def patched(tensors):
        out = _PACK_FIELD(tensors)
        return change(out) if (tensors[0].shape[0] == reps) == (which == "u") else out
    monkeypatch.setattr(proto.claimcodec, "pack_field", patched)


def _fold_and_open_attacks(graph):
    mats = graph.mat_ops
    victim, twin = mats[1].name, mats[2].name

    def fold(change):
        return "fold", lambda fold_: lambda chis: {k: (change(u) if k == victim else u) for k, u in fold_(chis).items()}

    def opening(changes):
        return "open", lambda open_: lambda cols: {k: (changes[k](*o) if k in changes else o)
                                                   for k, o in open_(cols).items()}

    def entry(value):
        def change(o, pr):
            o = o.clone()
            o[0, 0] = value(o[0, 0])
            return o, pr
        return change

    return {"u+1": fold(lambda u: (u + (torch.arange(u.numel()).reshape(u.shape) == 0)) % P),
            "u=P": fold(lambda u: torch.where(torch.arange(u.numel()).reshape(u.shape) == 0, P, u)),
            "u narrow": fold(lambda u: u[:, :-1]), "u extra row": fold(lambda u: torch.cat([u, u[:1]])),
            "opened+1": opening({victim: entry(lambda v: (v + 1) % P)}), "opened=P": opening({victim: entry(lambda v: P)}),
            "opened narrow": opening({victim: lambda o, pr: (o[:, :-1], pr)}),
            "columns swapped": opening({victim: lambda o, pr: (o[:, [1, 0] + list(range(2, o.shape[1]))], pr)}),
            "path forged": opening({victim: lambda o, pr: (o, [bytes(32)] + pr[1:])}),
            "path short": opening({victim: lambda o, pr: (o, pr[:-1])}),
            "path long": opening({victim: lambda o, pr: (o, pr + [bytes(32)])}),
            "merkle then code": opening({victim: lambda o, pr: (o, [bytes(32)] + pr[1:]),
                                         twin: entry(lambda v: (v + 1) % P)})}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_forged_u_and_openings_are_rejected_where_the_default_flow_rejects_them(kind, fiat_shamir):
    graph, x = _graph(kind)
    prover, pair = _setup(graph, "C", fiat_shamir)
    real = {"fold": prover.fold, "open": prover.open}
    labels = {}
    for i, (name, (what, patched)) in enumerate(_fold_and_open_attacks(graph).items()):
        setattr(prover, what, patched(real[what]))
        try:
            labels[name] = _both(prover, pair, x, seed=i)["rejected_at"]
            assert labels[name] == proto.run_query(prover, pair[0], x, seed=i)["rejected_at"], name
        finally:
            setattr(prover, what, real[what])
    assert {labels[k] for k in ("u+1", "u=P", "u narrow", "u extra row")} == {"freivalds"}
    assert labels["opened+1"] == labels["columns swapped"] == "columns_code"
    assert labels["opened=P"] == labels["opened narrow"] == "columns_shape"
    assert labels["path forged"] == labels["path short"] == labels["path long"] == "columns_merkle"
    assert labels["merkle then code"] == "columns_merkle"


def _as_unpacked(tensors: list, change) -> list:
    """The flat tensors the verifier unpacks from the packed ``tensors`` after ``change``."""
    flat = torch.from_numpy(cc.unpack_field(change(_PACK_FIELD(tensors)), sum(t.numel() for t in tensors))
                            .view(np.int32)).long()
    return list(flat.split([t.numel() for t in tensors]))


@pytest.mark.parametrize("kind", ["lenet5", "gpt2"])
@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_malformed_field_messages_are_rejected(kind, fiat_shamir, monkeypatch):
    graph, x = _graph(kind)
    prover, pair = _setup(graph, "C", fiat_shamir)
    reps = pair[0].params.reps
    assert reps < pair[0].params.columns
    cases = [("u", lambda b: b + b"\0", "freivalds"), ("u", lambda b: b[:-4], "freivalds"),
             ("columns", lambda b: b + b"\0", "columns_shape"), ("columns", lambda b: b[:-4], "columns_shape")]
    if (reps * sum(op.row_length for op in graph.mat_ops)) % 32:   # then element 32 G - 1 is padding
        cases.append(("u", lambda b: b[:-1] + bytes([b[-1] ^ 0x80]), "freivalds"))   # its top bit
    for i, (which, change, label) in enumerate(cases):
        _corrupt_field_message(monkeypatch, which, change, reps)
        assert _both(prover, pair, x, seed=i)["rejected_at"] == label, (which, label)
    monkeypatch.undo()
    # a flipped bit gives another element: rejected where the default flow rejects that element
    flip = lambda b: b[:9] + bytes([b[9] ^ 1]) + b[10:]
    real = {"fold": prover.fold, "open": prover.open}

    def fold(chis):
        us = real["fold"](chis)
        return {k: a.view(u.shape) for (k, u), a in zip(us.items(), _as_unpacked(list(us.values()), flip))}

    def open_(cols):
        opened = real["open"](cols)
        rows = _as_unpacked([o.T for o, _ in opened.values()], flip)
        return {k: (a.view(o.shape[1], o.shape[0]).T, pr) for (k, (o, pr)), a in zip(opened.items(), rows)}

    for i, (which, what, as_sent) in enumerate((("u", "fold", fold), ("columns", "open", open_))):
        _corrupt_field_message(monkeypatch, which, flip, reps)
        got = _both(prover, pair, x, seed=10 + i)["rejected_at"]
        monkeypatch.undo()
        setattr(prover, what, as_sent)
        try:
            assert got is not None and got == proto.run_query(prover, pair[0], x, seed=10 + i)["rejected_at"], which
        finally:
            setattr(prover, what, real[what])


@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_random_corruptions_of_the_claim_bytes_are_never_accepted(mode, monkeypatch):
    graph, x = _graph("lenet5")
    prover, pair = _setup(graph, mode)
    encode = cc.encode
    rng = np.random.default_rng(11)
    labels = set()
    for i in range(40):
        def corrupt(zs):
            b = bytearray(encode(zs))
            b[int(rng.integers(8, len(b)))] ^= 1 << int(rng.integers(0, 8))
            return bytes(b)
        monkeypatch.setattr(proto.claimcodec, "encode", corrupt)
        out = proto.run_query(prover, pair[0], x, seed=i, wire=True)
        assert not out["accepted"]
        labels.add(out["rejected_at"])
    assert labels <= {"range_or_shape", "freivalds"} and "freivalds" in labels


@pytest.mark.parametrize("mode", ["C", "K", "Kpre"])
def test_fiat_shamir_absorbs_the_encoded_bytes(mode, monkeypatch):
    absorbed = []
    absorb = proto.Challenger.absorb
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        absorbed.append((label, bytes(blob))), absorb(self, label, blob)))
    graph, x = _graph("qwen")
    prover, pair = _setup(graph, mode, fiat_shamir=True)
    claims = [prover.claims(x)[op.name] for op in graph.mat_ops]
    transcripts = {}
    for name, encode in (("pvc", _ENCODE), ("plain", _plain)):     # two encodings of the same claims
        monkeypatch.setattr(proto.claimcodec, "encode", encode)
        for v in pair:
            absorbed.clear()
            assert proto.run_query(prover, v, x, wire=True)["accepted"]
            transcripts.setdefault(name, []).append(list(absorbed))
    at = -2 if mode == "C" else -1        # the claims: the last message, or the one before u
    for name, (a, b) in transcripts.items():
        assert a == b                     # the streaming verifier's transcript is the batched one's
        labels = [label for label, _ in a]
        assert labels[:2] == [b"params", b"op/" + graph.mat_ops[0].name.encode()] and labels[at - 1] == b"x"
        assert a[at] == (b"claims/PVC3", (_ENCODE if name == "pvc" else _plain)(claims))
        if mode == "C":
            assert labels[-1] == b"u/F31"
            assert len(a[-1][1]) == cc.field_size(pair[0].params.reps * sum(op.row_length for op in graph.mat_ops))
        assert not any(label.startswith(b"claim/") for label in labels)
    assert transcripts["pvc"][0][at] != transcripts["plain"][0][at]


def test_the_default_flow_is_unchanged(monkeypatch):
    """Without ``wire`` nothing of the codec runs, and the transcript absorbs the int64 claims."""
    def refuse(*a, **kw):
        raise AssertionError("the codec ran without wire=True")
    for name in ("encode", "decode_torch", "pack_field", "unpack_field"):
        monkeypatch.setattr(proto.claimcodec, name, refuse)
    graph, x = _graph("gpt2")
    for mode in ("C", "Kpre"):
        prover, pair = _setup(graph, mode, fiat_shamir=True)
        for v in pair:
            out = proto.run_query(prover, v, x)
            assert out["accepted"] and "prove_encode" not in out["timings"] and "verify_decode" not in out["timings"]
            assert out["bytes"]["claims"] == 4 * sum(z.numel() for z in prover.claims(x).values())


@pytest.mark.parametrize("paths", ["deferred", "int8", "deferred_int8"])
def test_the_deferred_and_int8_checks_give_the_same_wire_verdicts(paths, monkeypatch):
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind in ("lenet5", "qwen"):
        graph, x = _graph(kind)
        for mode in ("C", "Kpre"):
            prover, pair = _setup(graph, mode)
            for i, (name, kw) in enumerate(_claim_attacks(graph.mat_ops).items()):
                got = _both(prover, pair, x, seed=i, forward_kwargs=kw)["rejected_at"]
                assert got == (None if name == "honest" else "freivalds"), (kind, mode, name)
        prover, pair = _setup(graph, "C")
        real = {"fold": prover.fold, "open": prover.open}
        for i, (name, (what, patched)) in enumerate(_fold_and_open_attacks(graph).items()):
            setattr(prover, what, patched(real[what]))
            try:
                assert _both(prover, pair, x, seed=i)["rejected_at"] is not None, name
            finally:
                setattr(prover, what, real[what])


def test_a_lean_prover_encodes_on_the_host():
    graph, x = _graph("llama")
    params = proto.params_for(40, len(graph.mat_ops))
    prover = proto.Prover(graph, lean=True)
    v = proto.Verifier(graph.public(), params, "K", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops},
                       lean=True)
    assert proto.run_query(prover, v, x, wire=True)["accepted"]


def test_a_gpu_client_gives_the_cpu_wire_verdicts(device):
    for kind in ("lenet5", "qwen"):
        graph, x = _graph(kind)
        for mode in ("C", "Kpre"):
            outcomes = []
            for dev in ("cpu", device):
                prover, pair = _setup(graph, mode, device=dev)
                outcomes.append([(r["accepted"], r["rejected_at"], r["bytes"]) for r in
                                 (_both(prover, pair, x, seed=i, forward_kwargs=kw)
                                  for i, kw in enumerate(_claim_attacks(graph.mat_ops).values()))])
            assert outcomes[0] == outcomes[1]


def test_a_gpu_prover_encodes_on_its_device_and_sends_the_cpu_provers_bytes(device, monkeypatch):
    graph, x = _graph("qwen")
    encoded_on = []
    monkeypatch.setattr(proto.claimcodec, "encode", lambda zs: (encoded_on.append({z.device.type for z in zs}),
                                                                _ENCODE(zs))[1])
    for mode in ("C", "Kpre"):
        prover, pair = _setup(graph, mode)
        on_device = proto.Prover(graph, commitments=prover.commitments, device=device)
        for i, kw in enumerate(_claim_attacks(graph.mat_ops).values()):
            a, b = (proto.run_query(p, pair[0], x, seed=i, wire=True, forward_kwargs=kw) for p in (prover, on_device))
            assert (a["accepted"], a["rejected_at"], a["bytes"]) == (b["accepted"], b["rejected_at"], b["bytes"])
    assert encoded_on[1::2] == [{torch.device(device).type}] * (len(encoded_on) // 2)


# -- the security parameters ----------------------------------------------------------------------

def _cnn_shapes(kind: str, shape: tuple) -> list[tuple[int, int]]:
    torch.manual_seed(0)
    graph = quantize_model(build_float_model(kind, 10).eval(), torch.randn(2, *shape))
    return [op.row_length for op in graph.mat_ops]


def test_the_parameters_meet_lambda_on_every_model():
    """``wire`` changes no parameter (``params_for``), and those meet lambda on the benchmark's models,
    in every mode, interactive and Fiat--Shamir (bench.py: CNNs sized for their own op count,
    decoders for the full model's)."""
    mlp = np.load(Path(__file__).resolve().parents[1] / "artifacts" / "models" / "mlp_mnist_full.npz")
    lengths = {"mlp_mnist": [mlp[k].shape[1] + 1 for k in mlp.files if k.endswith(".weight")],
               "lenet5": _cnn_shapes("lenet5", (1, 28, 28)), "vgg16": _cnn_shapes("vgg16", (3, 32, 32)),
               "resnet18_cifar": _cnn_shapes("resnet18_cifar", (3, 32, 32))}
    assert len(lengths["mlp_mnist"]) == 3 and len(lengths["lenet5"]) == 5
    for model in ("gpt2", "qwen3-4b", "llama2-7b"):
        lengths[model] = [s.row_length for s in decoder_shapes(CONFIGS[model])]
    for model, ks in lengths.items():
        for lam in (40, 80, 128):
            for fs in (False, True):
                params = proto.params_for(lam, len(ks), fiat_shamir=fs)
                shapes = [(k, params.rate * (1 << max(0, (k - 1).bit_length()))) for k in ks]
                for mode in ("C", "K"):
                    assert proto.soundness_bits(params, shapes, mode) >= lam, (model, lam, fs, mode)
                assert params.reps == math.ceil((lam + math.log2(2 * len(ks)) + (64 if fs else 0)) / proto.LOG2_P)


# -- bench.py --------------------------------------------------------------------------------------

def test_bench_wire_cells_carry_a_wire_tag(monkeypatch):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "experiments" / "4_defence_benchmark" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_wire_test", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    for argv, needs in ((["--wire"], "_wire"), (["--tag", "_wire"], "_wire")):
        monkeypatch.setattr(sys, "argv", ["bench.py", "cnn", "--model", "lenet5", *argv])
        with pytest.raises(SystemExit, match=needs):
            bench.main()
    assert bench.build_parser().parse_args(["cnn", "--model", "x", "--wire"]).wire
