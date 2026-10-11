"""The serving harness ``experiments/7_reexec/serve_ab.py`` (plan 2, A7).

U1 -- the integer model served without proofs, every claim dropped -- gives exactly the logits the proof claims
(decoded from the proof's bytes as the verifier decodes them), for a batch proved as one statement and for each prompt
proved on its own against the batched forward pass; and a proof produced again for the same prompts is byte-identical
(the "proof on request" row), accepted by ``run_query`` with the same byte count.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from pvi.fullcheck.protocol import run_query

CODE = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sa = _load("serve_ab", "experiments/7_reexec/serve_ab.py")


class _Args:
    """The options :func:`serve_ab.build` reads."""

    def __init__(self, policy: str = "paper", modes=("C", "Kpre")):
        self.modes, self.policy, self.lam = list(modes), policy, 128
        self.verifier_device, self.lookups, self.accept = "cpu", True, True


_CASES = [("tiny-llama", "paper"), ("tiny-opt", "paper"), ("tiny-qwen", "R16c"), ("tiny-gpt", "tightc")]
"""Each family once; the c plans add lookup tables (int8 rows and multiproofs) and col-layout matrices to the proof."""


@pytest.fixture(scope="module", params=_CASES, ids=[f"{m}-{p}" for m, p in _CASES])
def built(request):
    name, policy = request.param
    return sa.build(name, 0, _Args(policy), CPU)


def test_u1_logits_are_the_claimed_logits(built):
    m = built
    x = sa._tokens(m.cfg, 12, 3)
    u1 = sa.logit_rows(sa.serve_u1(m.graph, x, CPU))
    _, claims = m.graph.forward(x)                      # the claims the prover computes, before any encoding
    assert torch.equal(claims[sa.head_op(m.graph).name].T.to(torch.int64), u1)
    for mode in ("C", "Kpre"):
        p, v = m.provers[mode], m.verifiers[mode]
        parts, _, _ = sa.prove(p, v, x)                 # the batch as one statement
        assert torch.equal(sa.claimed_logits(v, m.graph, parts, x), u1)
        for i in range(x.shape[0]):                     # each prompt on its own, against the batched U1
            q = x[i:i + 1]
            parts, _, _ = sa.prove(p, v, q)
            assert torch.equal(sa.claimed_logits(v, m.graph, parts, q), u1[i:i + 1])


def test_proof_on_request_is_byte_identical_and_accepted(built):
    m = built
    x = sa._tokens(m.cfg, 12, 2)
    other = x.clone()
    other[0, 5] = (other[0, 5] + 1) % m.cfg.vocab
    for mode in ("C", "Kpre"):
        p, v = m.provers[mode], m.verifiers[mode]
        a_parts, a_open, a_out = sa.prove(p, v, x)
        first = [bytes(b) for b in sa.messages(a_parts, a_open, v)]
        sa.serve_u1(m.graph, other, CPU)                # serving other queries in between changes nothing
        sa.prove(p, v, other)
        b_parts, b_open, _ = sa.prove(p, v, x)
        again = [bytes(b) for b in sa.messages(b_parts, b_open, v)]
        assert first == again
        assert sum(map(len, first)) == sum(a_out["bytes"].values())
        assert sa.digest(first) == sa.digest(again)
        o_parts, o_open, _ = sa.prove(p, v, other)      # one changed token: other claims, other bytes
        assert sa.digest(sa.messages(o_parts, o_open, v)) != sa.digest(first)
        res = run_query(p, v, x, wire=True)
        assert res["accepted"], res["rejected_at"]
        assert sum(res["bytes"].values()) == sum(map(len, first))
        assert len(first[0]) > 0
        if mode == "C":     # the columns and their paths too (u may be empty: a c plan's col-layout ops send none)
            assert len(first) == 6 and len(first[4]) > 0 and len(first[5]) > 0
        else:
            assert len(first) == 3


def test_build_zero_layers_is_the_full_model():
    m = sa.build("tiny-llama", 0, _Args(modes=("Kpre",)), CPU)
    assert m.layers == m.cfg.n_layers == 2
    assert len(m.graph.mat_ops) == 2 + 7 * 2       # embedding, LM head, and q, k, v, o, gate, up, down per block


def test_prove_refuses_an_interactive_mode_c_proof():
    m = sa.build("tiny-llama", 0, _Args(modes=("C",)), CPU)
    v = m.verifiers["C"]
    interactive = sa.TimedChallenges(v.graph, sa.params_for(128, 8, rate=sa.RATE), "C", publics=v.publics,
                                     lean=True)
    with pytest.raises(ValueError):
        sa.prove(m.provers["C"], interactive, sa._tokens(m.cfg, 8, 1))


@pytest.mark.parametrize("name", ["tiny-gpt", "tiny-opt", "tiny-llama", "tiny-qwen"])
def test_float_decoder_pruned_last_block_gives_the_same_logits(name):
    reexec = sa._load("reexec", sa.HERE / "reexec.py")
    cfg = sa.config(name)
    x = sa._tokens(cfg, 12, 3)
    full = reexec.FloatDecoder(cfg, 2, torch.float64, CPU)(x)
    pruned = reexec.FloatDecoder(cfg, 2, torch.float64, CPU, prune_last=True)(x)
    assert pruned.shape == full.shape == (3, cfg.vocab)
    assert torch.allclose(pruned, full, rtol=1e-9, atol=1e-9)


def test_bootstrap_median_and_ratio():
    r = sa.boot([5.0, 1.0, 3.0, 2.0, 4.0])
    assert r["median"] == 3.0 and r["ci95"][0] <= 3.0 <= r["ci95"][1]
    assert sa.boot([2.0] * 6, [1.0] * 6) == {"ratio": 2.0, "ci95": [2.0, 2.0]}


def _cli(monkeypatch, capsys, *argv) -> tuple[int, list[dict]]:
    """``serve_ab.main()`` on ``argv``: its status and its JSON lines (torch's thread count restored after it)."""
    monkeypatch.setattr(sys, "argv", ["serve_ab.py", *argv])
    threads = torch.get_num_threads()
    try:
        status = sa.main()
    finally:
        torch.set_num_threads(threads)
    return status, [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA (CUDA events, claims narrowed and encoded on the device)"))])
def test_cli_lines(device, monkeypatch, capsys):
    status, lines = _cli(monkeypatch, capsys, "--model", "tiny-llama", "--cells", "8x1", "8x2", "--reps", "2",
                         "--batch-proof", "seq", "joint", "--device", device, "--threads", "1")
    assert status == 0
    prove = [r for r in lines if r["row"] == "prove"]
    assert {(r["seq"], r["batch"], r["mode"], r["batch_proof"]) for r in prove} == {
        (8, 1, "C", "single"), (8, 1, "Kpre", "single"), (8, 2, "C", "seq"), (8, 2, "C", "joint"),
        (8, 2, "Kpre", "seq"), (8, 2, "Kpre", "joint")}
    for r in prove:
        assert "skipped" not in r and r["ok"], r["checks"]
        assert r["checks"]["accepted"] and r["checks"]["deterministic"] and r["checks"]["u1_equals_claims"]
        assert len(r["times_s"]["prove"]) == 2 and r["proof_bytes"] > 0
        assert r["timer"] == ("cuda_events" if device == "cuda" else "perf_counter")
    req = [r for r in lines if r["row"] == "on_request"]
    assert len(req) == 2 and all(p["deterministic"] for r in req for p in r["later_proof"].values())
    seq_c = next(r for r in prove if (r["batch"], r["mode"], r["batch_proof"]) == (2, "C", "seq"))
    single_c = next(r for r in prove if (r["batch"], r["mode"]) == (1, "C"))
    assert seq_c["proof_sha256"] == single_c["proof_sha256"]     # prompt 0 of every cell is the B = 1 prompt


def test_out_of_memory_is_skipped_not_failed(monkeypatch, capsys):
    """A variant that runs out of GPU memory (simulated: every proof of a 2-prompt statement) is recorded as
    skipped and the run goes on, with status 0; an error that is not about memory is not taken for one."""
    real = sa.prove

    def prove(prover, verifier, x):
        if x.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 44.00 GiB")
        return real(prover, verifier, x)

    monkeypatch.setattr(sa, "prove", prove)
    status, lines = _cli(monkeypatch, capsys, "--model", "tiny-llama", "--modes", "Kpre", "--cells", "8x2",
                         "--reps", "2", "--batch-proof", "seq", "joint", "--device", "cpu", "--threads", "1")
    assert status == 0
    prove_lines = {r["batch_proof"]: r for r in lines if r["row"] == "prove"}
    assert prove_lines["joint"]["skipped"] == "out of memory"
    assert "skipped" not in prove_lines["seq"] and prove_lines["seq"]["ok"]
    (req,) = [r for r in lines if r["row"] == "on_request"]
    assert list(req["later_proof"]) == ["Kpre/seq"]
    assert sa._is_oom(RuntimeError("CUDA error: out of memory")) and not sa._is_oom(RuntimeError("shape mismatch"))
