"""The attention probabilities' scale ``pmax``: ``p = round(pmax e / tot)``, 0..pmax.

``build_decoder`` defaults to :data:`pvi.models.decoder.PROB_MAX` = 255 (8 bits); the certified models and the cost
measurements' random-weight graphs (``experiments.common.cli.PAPER.prob_max``) use
:data:`pvi.models.decoder.CERTIFIED_PROB_MAX` = 2047 (11 bits), which keeps a long row's probability mass. Checked:
every torch path of :func:`pvi.models.decoder.integer_attention` gives ``reference.attention``'s integers at both
scales (the int32 path, the int64 fallback, grouped-query attention, queries of the last positions only, the output
requantisation), and at the longest prompt of the run (16,384 tokens) on rows at the int32 path's bounds (the
largest row sum of probabilities, a row total of about ``2**29``); ``build_decoder(prob_max=255)`` is the default
graph, digest included, and another scale is bound by the digest; the 8- and 11-bit graphs of one seed hold the same
weights and shapes, and :func:`pvi.models.decoder.prob_max_of` reads a graph's scale; the cost measurements' default
is the certified models' scale; the probability-mass diagnostic (``pvi.models.hf_opt.attention_mass``) reads the
probabilities the reference computes; and a 2047 decoder is proved and its tampered claims rejected like any other.
"""

from __future__ import annotations

import pytest
import torch

import pvi.v0 as proto
import reference as ref
from pvi.core.graph import CheapOp, MatOp, OpFn
from pvi.models import decoder as tr
from pvi.models.decoder import build_decoder
from test_plans import _TINY

LONG_T = 16384
"""The longest prompt of the run (the long tier's 16,384 tokens)."""
LONG_M_S = 1 << 24
"""A score multiplier of the int32 path (``tr._int32_scores``) under which the long rows' low scores look up 0."""
TOP_KEYS = 4093
"""Keys at the top score in :func:`long_rows`'s first head: ``p = round(2047 / 4093) = 1`` on each, the largest row
sum of 11-bit probabilities (``pmax + n / 2``)."""


def _qkv(b, tq, t, hq, hkv, dh, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-127, 128, (b, tq, hq * dh), generator=g)
    k = torch.randint(-127, 128, (b, t, hkv * dh), generator=g)
    v = torch.randint(-127, 128, (b, t, hkv * dh), generator=g)
    return q, k, v


@pytest.mark.parametrize("pmax", [255, 2047])
@pytest.mark.parametrize("hq,hkv,tq,t", [(4, 4, 33, 33), (8, 2, 5, 40), (2, 2, 1, 70)])
@pytest.mark.parametrize("m_s", [1 << 14, 1 << 17, 3 << 19])
def test_every_torch_path_gives_the_reference_integers(pmax, hq, hkv, tq, t, m_s):
    dh = 16
    q, k, v = _qkv(1, tq, t, hq, hkv, dh, seed=t + hq + m_s % 97)
    for m_o in (None, 1 << 21, 3 << 18):
        want = ref.attention(q, k, v, m_s, m_o, hq, hkv, dh, pmax)
        assert torch.equal(tr.integer_attention(q, k, v, m_s, m_o, hq, hkv, dh, pmax), want)
    # the int32 path and the int64 fallback, head by head
    qh = q.reshape(1, tq, hq, dh).transpose(1, 2)
    kh = k.reshape(1, t, hkv, dh).transpose(1, 2).repeat_interleave(hq // hkv, 1)
    vh = v.reshape(1, t, hkv, dh).transpose(1, 2).repeat_interleave(hq // hkv, 1)
    notmask = tr._causal_notmask(t, 1, "cpu", tq)
    want = ref.attention_heads(qh, kh, vh, m_s, None, pmax)
    if tr._int32_scores(m_s, dh, t):                     # small m_s: the gap table would be too long
        assert torch.equal(tr._attention_core(qh, kh, vh, tr._exp_lut(m_s, "cpu"), notmask, pmax), want)
    assert torch.equal(tr._attention_int64(qh, kh, vh, m_s, notmask, pmax), want)


def long_rows(t: int = LONG_T, rows: int = 4, dh: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``q [1, 3, rows, dh]`` (the queries of the last ``rows`` positions), ``k, v [1, 3, t, dh]``, int8-valued, at the
    bounds of the exact int32 attention under :data:`LONG_M_S`: head 0 has :data:`TOP_KEYS` keys at the top score and
    every other one ``254 * 127`` below it (looked up as 0), with ``v = 127``; head 1 has every score equal, so a row's
    total is ``n 2**15``, about ``2**29``; head 2 is random."""
    g = torch.Generator().manual_seed(t + rows)
    q = torch.zeros(1, 3, rows, dh, dtype=torch.int64)
    k = torch.zeros(1, 3, t, dh, dtype=torch.int64)
    v = torch.full((1, 3, t, dh), 127, dtype=torch.int64)
    q[:, 0, :, 0] = 127
    k[:, 0, :, 0] = -127
    k[:, 0, :TOP_KEYS, 0] = 127
    q[:, 2] = torch.randint(-127, 128, (rows, dh), generator=g)
    k[:, 2] = torch.randint(-127, 128, (t, dh), generator=g)
    v[:, 1:] = torch.randint(-127, 128, (2, t, dh), generator=g)
    return q, k, v


def last_rows_notmask(t: int, rows: int) -> torch.Tensor:
    """The causal mask's complement for the queries of the last ``rows`` positions: ``[rows, t]`` (no ``t x t``
    tensor)."""
    return torch.arange(t)[None, :] > (t - rows + torch.arange(rows))[:, None]


def reference_rows(q, k, v, m_s: int, pmax: int, notmask: torch.Tensor) -> torch.Tensor:
    """``reference.attention_heads``'s raw ``P V`` in int64 throughout, for few query rows against many keys."""
    s = (q @ k.transpose(-1, -2)).masked_fill(notmask, -(1 << 40))
    d = (s.amax(-1, keepdim=True) - s).clamp(max=1 << 31)
    e = tr.EXP_TABLE[((d * m_s + (1 << 29)) >> 30).clamp(0, 255)].masked_fill(notmask, 0)
    tot = e.sum(-1, keepdim=True)
    return torch.div(e * (2 * pmax) + tot, 2 * tot, rounding_mode="floor") @ v


@pytest.mark.parametrize("pmax", [255, 2047])
def test_the_exact_paths_at_the_longest_prompt_and_the_bounds(pmax):
    """At 16,384 keys the int32 path applies (``T < 2**15``), and the int32 path and the int64 fallback give the
    reference's integers on rows at its bounds: the largest row sum of probabilities (``P V`` up to
    ``4093 * 127``, below ``2**24``) and a row total of ``n 2**15`` (``2 tot`` and ``tot + 2 pmax e`` below
    ``2**31``)."""
    q, k, v = long_rows()
    rows, t = q.shape[2], k.shape[2]
    assert tr._int32_scores(LONG_M_S, 128, t) and tr._int32_scores(1 << 20, 128, t)
    notmask = last_rows_notmask(t, rows)
    want = reference_rows(q, k, v, LONG_M_S, pmax, notmask)
    if pmax == 2047:
        assert bool((want[0, 0] == TOP_KEYS * 127).all())             # p = 1 on every top key
    assert not want[0, 1].any()                                        # p = round(pmax / n) = 0 at n > 2 pmax
    got = tr._attention_core(q, k, v, tr._exp_lut(LONG_M_S, "cpu"), notmask, pmax)
    assert torch.equal(got, want)
    assert torch.equal(tr._attention_int64(q, k, v, LONG_M_S, notmask, pmax), want)


def test_the_scale_of_a_graph_and_the_cost_measurements_default():
    """:func:`pvi.models.decoder.prob_max_of` reads the scale a graph's attention ops share (None for an image model);
    the cost measurements' protocol default is the certified models' scale (``cli.PAPER``, and its torch-free copy
    in ``experiments.paper.bench``); the 8- and 11-bit graphs of one seed hold the same weight ops (weights, biases,
    shapes) and the same op kinds, so their proofs have the same shape."""
    from experiments.common import cli
    from experiments.paper import bench as paper_bench
    from pvi.models.quantize import random_cnn
    cfg = _TINY["qwen"]
    g8 = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=255)
    g11 = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=2047)
    assert (tr.prob_max_of(g8), tr.prob_max_of(g11), tr.prob_max_of(random_cnn("lenet5")[0])) == (255, 2047, None)
    assert cli.PAPER.prob_max == tr.CERTIFIED_PROB_MAX == paper_bench.DEFAULT_PROB_MAX == 2047
    assert paper_bench.STORED_PROB_MAX == tr.PROB_MAX
    assert len(g8.ops) == len(g11.ops)
    for a, b in zip(g8.ops, g11.ops):
        assert (type(a), a.name, a.inputs, a.output) == (type(b), b.name, b.inputs, b.output)
        if isinstance(a, MatOp):
            assert torch.equal(a.weight, b.weight) and (a.bias is None) == (b.bias is None)
            assert a.bias is None or torch.equal(a.bias, b.bias)
        elif isinstance(a, CheapOp) and isinstance(a.fn, OpFn):
            assert a.fn.kind == b.fn.kind
            assert ("pmax" in a.fn.consts) == (a.fn.kind == tr.attention_op.op_kind)


def test_the_scales_differ_and_keep_more_mass_at_11_bits():
    """A diffuse row (equal scores over many keys) rounds to 0 at 8 bits once ``tot / e > 510``; at 11 bits the mass
    survives up to 4094 keys."""
    t, dh = 2000, 16
    q = torch.zeros(1, 1, dh, dtype=torch.int64)
    k = torch.zeros(1, t, dh, dtype=torch.int64)
    v = torch.ones(1, t, dh, dtype=torch.int64)
    raw8 = tr.integer_attention(q, k, v, 1 << 20, None, 1, 1, dh, 255)
    raw11 = tr.integer_attention(q, k, v, 1 << 20, None, 1, 1, dh, 2047)
    assert int(raw8[0, 0, 0]) == 0                       # every p = round(255 / 2000) = 0
    assert int(raw11[0, 0, 0]) == t                      # every p = round(2047 / 2000) = 1


def test_the_default_scale_is_the_8_bit_graph_and_another_scale_is_bound_by_the_digest():
    cfg = _TINY["qwen"]
    base = build_decoder(cfg, calib_tokens=12, seed=3)
    same = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=255)
    other = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=2047)
    assert base.digest() == same.digest() and other.digest() != base.digest()
    x = torch.randint(0, cfg.vocab, (1, 12), generator=torch.Generator().manual_seed(5))
    assert torch.equal(base.forward(x)[0][base.output_name], same.forward(x)[0][same.output_name])


@pytest.mark.parametrize("pmax", [255, 2047])
@pytest.mark.parametrize("hq,hkv,tq,t", [(4, 4, 40, 40), (8, 2, 5, 40), (12, 12, 1, 70)])
def test_the_probability_mass_is_that_of_the_reference_probabilities(pmax, hq, hkv, tq, t):
    """``hf_opt.attention_mass`` is the mean over rows and heads of ``sum_j p_j / pmax`` for the reference's
    probabilities: with ``v = 1`` the reference's raw ``P V`` is each row's sum of probabilities."""
    from pvi.models.hf_opt import attention_mass
    dh, m_s = 16, 1 << 17
    q, k, _ = _qkv(1, tq, t, hq, hkv, dh, seed=t + hq)
    sums = ref.attention(q, k, torch.ones(1, t, hkv * dh, dtype=torch.int64), m_s, None, hq, hkv, dh, pmax)
    want = float(sums.reshape(1, tq, hq, dh)[..., 0].double().mean()) / pmax
    got = attention_mass(q, k, m_s=m_s, heads=hq, kv_heads=hkv, dh=dh, pmax=pmax)
    assert abs(got - want) < 1e-12, (got, want)


def test_a_diffuse_row_keeps_its_mass_at_11_bits_only():
    from pvi.models.hf_opt import attention_mass
    t, dh = 2000, 16
    q = torch.zeros(1, 1, dh, dtype=torch.int64)
    k = torch.zeros(1, t, dh, dtype=torch.int64)
    assert attention_mass(q, k, m_s=1 << 20, heads=1, kv_heads=1, dh=dh, pmax=255) == 0.0
    assert attention_mass(q, k, m_s=1 << 20, heads=1, kv_heads=1, dh=dh, pmax=2047) == t / 2047


@pytest.mark.parametrize("mode", ["C", "Kpre"])
@pytest.mark.parametrize("kind", ["gpt", "qwen"])
def test_an_11_bit_decoder_is_proved_and_tampering_rejected(kind, mode):
    cfg = _TINY[kind]
    graph = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=2047)
    x = torch.randint(0, cfg.vocab, (1, 12), generator=torch.Generator().manual_seed(5))
    params = proto.params_for(40, len(graph.mat_ops))
    coms = proto.commit_graph(graph, params.rate) if mode == "C" else {}
    prover = proto.Prover(graph, commitments=coms)
    verifier = proto.Verifier(graph.public(), params, mode, publics={k_: c.public for k_, c in coms.items()},
                              weights={op.name: (op.weight, op.bias) for op in graph.mat_ops})
    if mode == "Kpre":
        verifier.precompute(proto.Challenger(seed=9))
    assert proto.run_query(prover, verifier, x, seed=1)["accepted"]
    victim = graph.mat_ops[len(graph.mat_ops) // 2].name

    def tamper(op, z):
        if op.name == victim:
            z = z.clone()
            z[0, 0] += 1
        return z

    assert not proto.run_query(prover, verifier, x, seed=2, forward_kwargs={"tamper": tamper})["accepted"]


@pytest.mark.parametrize("family", ["opt", "llama", "qwen"])
def test_the_real_builders_take_the_scale(family):
    """``build_opt_from_hf`` and ``build_llama_from_hf`` with ``prob_max``: 255 is the 8-bit graph (digest, logits);
    2047, the certified models' default, is another graph (bound by the digest) whose output multipliers are
    calibrated at that scale, and it runs."""
    transformers = pytest.importorskip("transformers")
    from pvi.models import hf_llama as rl
    from pvi.models import hf_opt as rw
    torch.manual_seed(0)
    if family == "opt":
        cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                     num_attention_heads=4, max_position_embeddings=64, word_embed_proj_dim=32,
                                     do_layer_norm_before=True)
        model, build = transformers.OPTForCausalLM(cfg).eval(), rw.build_opt_from_hf
    elif family == "llama":
        cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                       tie_word_embeddings=False, initializer_range=0.15)
        model, build = transformers.LlamaForCausalLM(cfg).eval(), rl.build_llama_from_hf
    else:
        cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                       num_attention_heads=8, num_key_value_heads=2, head_dim=16,
                                       max_position_embeddings=64, tie_word_embeddings=True, initializer_range=0.15)
        model, build = transformers.Qwen3ForCausalLM(cfg).eval(), rl.build_llama_from_hf
    calib = torch.randint(3, 97, (1, 24), generator=torch.Generator().manual_seed(1))
    x = torch.randint(3, 97, (1, 20), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        base, _ = build(model, calib, norm_gain=4, prob_max=255)            # the 8-bit graph
        same, _ = build(model, calib, norm_gain=4, prob_max=tr.PROB_MAX)
        other, _ = build(model, calib, norm_gain=4, prob_max=2047)
        certified, _ = build(model, calib, norm_gain=4)                      # the default: 11 bits
    assert base.digest() == same.digest() and other.digest() != base.digest()
    assert certified.digest() == other.digest() and tr.CERTIFIED_PROB_MAX == 2047
    assert torch.equal(base.forward(x)[0][base.output_name], same.forward(x)[0][same.output_name])
    assert other.forward(x)[0][other.output_name].shape == base.forward(x)[0][base.output_name].shape
