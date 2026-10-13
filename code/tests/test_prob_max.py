"""The attention probabilities' scale ``pmax`` (Plan 2 F2a): ``p = round(pmax e / tot)``, 0..pmax.

Today's graphs use :data:`transformer.PROB_MAX` = 255 (8 bits); F2a measures what 11 bits (2047) recover at long context.
Here: every torch path of :func:`transformer._attention` gives ``reference.attention``'s integers at both scales (the
int32 path, the int64 fallback, grouped-query attention, queries of the last positions only, the output
requantisation); ``build_decoder(prob_max=255)`` is today's graph, digest included, and another scale is bound by the
digest; and a 2047 decoder is proved and its tampered claims rejected like any other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import pvi.v0 as proto
import reference as ref
from pvi.models import decoder as tr
from pvi.models.decoder import build_decoder
from test_plans import _TINY


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
        assert torch.equal(tr._attention(q, k, v, m_s, m_o, hq, hkv, dh, pmax), want)
    # the int32 path and the int64 fallback, head by head
    qh = q.reshape(1, tq, hq, dh).transpose(1, 2)
    kh = k.reshape(1, t, hkv, dh).transpose(1, 2).repeat_interleave(hq // hkv, 1)
    vh = v.reshape(1, t, hkv, dh).transpose(1, 2).repeat_interleave(hq // hkv, 1)
    notmask = tr._causal_notmask(t, 1, "cpu", tq)
    want = ref.attention_heads(qh, kh, vh, m_s, None, pmax)
    if tr._int32_scores(m_s, dh, t):                     # small m_s: the gap table would be too long
        assert torch.equal(tr._attention_core(qh, kh, vh, tr._exp_lut(m_s, "cpu"), notmask, pmax), want)
    assert torch.equal(tr._attention_int64(qh, kh, vh, m_s, notmask, pmax), want)


def test_the_scales_differ_and_keep_more_mass_at_11_bits():
    """A diffuse row (equal scores over many keys) rounds to 0 at 8 bits once ``tot / e > 510``; at 11 bits the mass
    survives up to 4094 keys."""
    t, dh = 2000, 16
    q = torch.zeros(1, 1, dh, dtype=torch.int64)
    k = torch.zeros(1, t, dh, dtype=torch.int64)
    v = torch.ones(1, t, dh, dtype=torch.int64)
    raw8 = tr._attention(q, k, v, 1 << 20, None, 1, 1, dh, 255)
    raw11 = tr._attention(q, k, v, 1 << 20, None, 1, 1, dh, 2047)
    assert int(raw8[0, 0, 0]) == 0                       # every p = round(255 / 2000) = 0
    assert int(raw11[0, 0, 0]) == t                      # every p = round(2047 / 2000) = 1


def test_the_default_scale_is_todays_graph_and_another_scale_is_bound_by_the_digest():
    cfg = _TINY["qwen"]
    today = build_decoder(cfg, calib_tokens=12, seed=3)
    same = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=255)
    other = build_decoder(cfg, calib_tokens=12, seed=3, prob_max=2047)
    assert today.digest() == same.digest() and other.digest() != today.digest()
    x = torch.randint(0, cfg.vocab, (1, 12), generator=torch.Generator().manual_seed(5))
    assert torch.equal(today.forward(x)[0][today.output_name], same.forward(x)[0][same.output_name])


def test_opt_builder_attention_var_is_the_graph_attention():
    """F2a's experiment builder (``experiments/9_quality/opt_builder.attention_var``) computes the same integers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "9_quality"))
    import opt_builder as ob
    q, k, v = _qkv(1, 40, 40, 4, 4, 16, seed=7)
    for pmax in (255, 2047):
        for m_o in (None, 1 << 21):
            got = ob.attention_var(q, k, v, 1 << 17, m_o, 4, 16, pmax, ob.exp_table(16))
            assert torch.equal(got, tr._attention(q, k, v, 1 << 17, m_o, 4, 4, 16, pmax))


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
    """``build_opt_from_hf`` and ``build_llama_from_hf`` with ``prob_max``: 255 is the graph of before F2a (digest,
    logits); 2047, the certified models' default, is another graph (bound by the digest) whose output multipliers are
    calibrated at that scale, and it runs."""
    transformers = pytest.importorskip("transformers")
    from pvi.models import hf_llama as rl
    from pvi.models import hf_opt as rw
    torch.manual_seed(0)
    if family == "opt":
        cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64, num_attention_heads=4,
                                     max_position_embeddings=64, word_embed_proj_dim=32, do_layer_norm_before=True)
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
        today, _ = build(model, calib, norm_gain=4, prob_max=255)            # the 8-bit graph of before F2a
        same, _ = build(model, calib, norm_gain=4, prob_max=tr.PROB_MAX)
        other, _ = build(model, calib, norm_gain=4, prob_max=2047)
        certified, _ = build(model, calib, norm_gain=4)                      # the default: 11 bits
    assert today.digest() == same.digest() and other.digest() != today.digest()
    assert certified.digest() == other.digest() and tr.CERTIFIED_PROB_MAX == 2047
    assert torch.equal(today.forward(x)[0][today.output_name], same.forward(x)[0][same.output_name])
    assert other.forward(x)[0][other.output_name].shape == today.forward(x)[0][today.output_name].shape
