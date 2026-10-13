"""``experiments.quality.llama_attrib``: the op-level attribution of the integer Llama / Qwen3 graph against the fp32
checkpoint, and the q/k-norm balance it led to (``hf_llama``'s ``qk_balance``), on tiny randomly initialised Hugging
Face models (no download).

The tiny models are ``test_real_llama.py``'s (2 blocks, d = 64, head_dim 16, gains in [0.5, 1.5], weights of std
0.15; calibration 2 x 32 random tokens, evaluation 4 x 32). Measured on the smoothed recipe (seeds 0 / 1; the
unpruned graph, ``prune_last``, the prefix (5, 17) and ``glu_exp = 8``; blocks 0 and 1):

=============================================  ====================  ====================
quantity (over every tensor of both blocks)     accumulated           teacher-forced
=============================================  ====================  ====================
``vs_hf`` mean cosine, smallest                 0.987 / 0.981         0.996 / 0.995
``local`` mean cosine, smallest                 0.9994 / 0.9968       0.9994 / 0.9970
``local`` relative L2 error, largest            0.049 / 0.134         0.049 / 0.131
score error an op adds (nats), largest          0.025 / 0.037         0.033 / 0.048
attention score error (nats), largest           0.17 / 0.19           0.054 / 0.069
=============================================  ====================  ====================

(seed 1's 0.13 is the pruned Llama block's SwiGLU product: four positions whose products use few int8 levels.) The
main recipe (``hf_llama.MAIN_RECIPE``: the smoothed recipe with the SwiGLU product balance, here with the prefix) tracks
as closely, but for the product's own error on the tiny Llama (mean cosine 0.9989, relative error 0.074 in seed 0's
first block); the attribution undoes the balance's factors (without that, the up projection is flagged).  A
broken builder is caught at its op: a q (k) norm gain on the wrong channels, or a SwiGLU product off by 2x (local
relative error 1.0 at an unchanged cosine: only the relative error sees a scale), flags that op and nothing before
it.

The realistic case: a tiny Qwen3 with Qwen3-0.6B-Base's real layer-0 q/k-norm gains (``REAL_Q_NORM_L0``,
``REAL_K_NORM_L0``, read from the checkpoint's safetensors header and the two 256-byte tensors alone: k's gain is
96.5, 38.3 and 26.3 on channels 50, 115 and 88 -- median 2.3 -- where q's is 0.003, 0.18 and 0.02), head_dim 128,
1 block, d = 128, 4 / 2 heads, random weights; calibration 4 x 64, evaluation 2 x 64. Without the balance the
first block's output falls to a mean cosine 0.872 / 0.783 (seeds 0 / 1, smoothed, teacher-forced; scalar gain 4:
0.827 / 0.737) -- the shape of the real Qwen3-0.6B-Base's first block (0.889 smoothed, 0.865 scalar, in
``llama_diag.py``'s diagnosis of the graph before the balance) -- and the attribution puts it on the k norm: it adds
0.90 / 1.10 nats to the attention scores' error (q norm 0.24 / 0.27, every other op <= 0.15), 31% / 35% of its int8
outputs are 0. With the balance (the default): k norm 0.11 / 0.14 nats, q norm 0.11 / 0.12, the block's output 0.987
/ 0.983 (scalar: 0.955 / 0.939, where the norm's gain-4 resolution then dominates: 0.31 nats).
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.quality import llama_attrib as la  # noqa: E402
from pvi.models import hf_llama  # noqa: E402

SMOOTH = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
VARIANTS = {"plain": {}, "prune": {"prune_last": True}, "prefix": {"prefix_tokens": (5, 17)}, "exp8": {"glu_exp": 8},
            "main": {"glu_smooth": hf_llama.MAIN_RECIPE["glu_smooth"], "prefix_tokens": (5, 17)}}
"""The graphs compared: ``SMOOTH`` alone, pruned, with a prefix, with the per-token exponent, and the main recipe
(``hf_llama.MAIN_RECIPE``: ``SMOOTH`` with the SwiGLU product balance) with a prefix."""
_CACHE: dict = {}

REAL_Q_NORM_L0 = (
    4.53125, 1.24219, -0.734375, 1.70312, 2.59375, 1.54688, 1.09375, 1.46094, 2.76562, 2, 1.84375, -0.0324707,
    0.730469, 0.757812, 1.89844, 2.07812, 1.9375, 1.875, 2.1875, 2.40625, 2.42188, 1.76562, 1.77344, 1.90625, 1.04688,
    2.15625, 2.25, 1.74219, 2.10938, 1.60938, 1.88281, 1.6875, 11.8125, 1.95312, 1.66406, 1.67188, 1.3125, 2.01562,
    1.38281, 2.0625, 1.875, 2, 1.1875, 2.01562, 2.10938, 2.59375, 1.59375, 3.15625, 3.14062, 1.01562, 0.00259399,
    1.6875, 1.20312, 3.25, 4.84375, 2.125, 2.5, 1.66406, 1.92969, 2.53125, 1.86719, 2.48438, 2.625, 1.875, 1.625,
    2.28125, 0.964844, 2.375, 2.10938, 1.125, 1.20312, 2.53125, 2.51562, 2.125, 1.71094, 2.1875, 2, 0.980469, 1.99219,
    1.99219, 2.09375, 1.97656, 2.15625, 3.17188, 2.48438, 2.03125, 1.80469, 2.01562, 0.0233154, 2.15625, 1.94531,
    2.23438, 2, 1.59375, 1.84375, 1.67969, 0.15332, 1.82812, 1.30469, 1.67188, 1.59375, 2.04688, 2.14062, 2.1875,
    1.59375, 2.15625, 2.03125, 1.63281, 2.125, 1.71875, 2.64062, 1.67188, 1.13281, 3.46875, 1.29688, 0.183594, 3.125,
    1.57031, 1.67188, 1.82031, 1.48438, 2.375, 2.1875, 1.77344, 2.34375, 1.83594, 1.70312, 2.39062)
REAL_K_NORM_L0 = (
    1.29688, 2.32812, 4.40625, 2.01562, 1.75781, 2.53125, 4.28125, 2.40625, 1.60938, 2.25, 2.26562, 0.0263672, 5.34375,
    5.4375, 2.39062, 2.17188, 2.29688, 2.4375, 2.0625, 1.80469, 1.90625, 2.32812, 2.54688, 2.35938, 1.27344, 2.07812,
    2.07812, 2.51562, 2.15625, 2.85938, 2.40625, 2.70312, 0.162109, 2.3125, 2.59375, 2.71875, 3.53125, 2.28125, 3.375,
    2.21875, 2.5, 2.32812, 3.89062, 2.29688, 2.17188, 1.8125, 2.9375, 1.50781, 1.51562, 4.59375, 96.5, 1.49219, 4.0625,
    1.4375, 1.00781, 2.28125, 1.89062, 2.84375, 2.39062, 1.88281, 2.60938, 1.9375, 1.86719, 2.57812, 0.976562,
    2.57812, 3.71875, 1.90625, 1.80469, 3.34375, 3.73438, 1.14844, 1.77344, 2.15625, 2.5625, 2.23438, 1.17969,
    0.730469, 2.26562, 2.23438, 2.125, 2.29688, 2.14062, 1.25, 1.82812, 2.15625, 2.54688, 2.28125, 26.25, 2.04688,
    2.29688, 1.99219, 2.26562, 2.875, 2.48438, 2.6875, 2.875, 2.48438, 3.5625, 2.8125, 2.84375, 2.23438, 2.04688,
    2.17188, 2.875, 2.15625, 2.21875, 2.875, 2.20312, 2.73438, 1.79688, 2.84375, 4.21875, 1.34375, 1.21094, 38.25,
    1.51562, 2.98438, 2.79688, 2.53125, 3.10938, 2.03125, 2.14062, 2.73438, 2, 2.59375, 2.79688, 2.01562)
"""Qwen/Qwen3-0.6B-Base's ``model.layers.0.self_attn.q_norm.weight`` and ``k_norm.weight`` (bf16, exact)."""


def _tiny(kind: str, seed: int = 0):
    """``test_real_llama.py``'s tiny models."""
    torch.manual_seed(seed)
    if kind == "llama":
        cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                       rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.15)
        model = transformers.LlamaForCausalLM(cfg)
    else:
        cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                       num_attention_heads=8, num_key_value_heads=2, head_dim=16,
                                       max_position_embeddings=64, rms_norm_eps=1e-6, tie_word_embeddings=True,
                                       rope_theta=1e6, initializer_range=0.15)
        model = transformers.Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model.eval()


def _realistic_qwen3(seed: int = 0):
    """A 1-block Qwen3 of head_dim 128 (d = 128, 4 / 2 heads) whose q/k-norm gains are Qwen3-0.6B-Base's layer 0."""
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=128, intermediate_size=384, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                                   max_position_embeddings=256, rms_norm_eps=1e-6, tie_word_embeddings=True,
                                   rope_theta=1e6, initializer_range=0.15)
    model = transformers.Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
        at = model.model.layers[0].self_attn
        at.q_norm.weight.copy_(torch.tensor(REAL_Q_NORM_L0))
        at.k_norm.weight.copy_(torch.tensor(REAL_K_NORM_L0))
    return model.eval()


def _data(t: int, n_calib: int, n_eval: int):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 97, (n_calib, t), generator=g), torch.randint(0, 97, (n_eval, t), generator=g)


def _setup(kind: str, variant: str = "plain"):
    key = (kind, variant)
    if key not in _CACHE:
        model = _CACHE.setdefault(kind, _tiny(kind))
        calib, x = _data(32, 2, 4)
        with torch.no_grad():
            graph, info = hf_llama.build_llama_from_hf(model, calib, **SMOOTH, **VARIANTS[variant])
        _CACHE[key] = model, graph, info, calib, x
    return _CACHE[key]


def _attribute(model, graph, info, x, mode, prefix=None, blocks=(0, 1)):
    with torch.no_grad():
        return la.attribute(graph, info, model, x, list(blocks), mode, prefix)


# ------------------------------------------------------------------ the map between the graph and the checkpoint

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_role_map_names_every_tensor_of_every_block(kind, variant):
    _, graph, info, _, _ = _setup(kind, variant)
    assert la.check_roles(graph, info) == []
    want = {r for r, *_ in hf_llama.ROLES} - (set() if kind == "qwen3" else {"q_norm", "k_norm"}) | {"in"}
    assert all(set(roles) == want for roles in info["roles"]) and len(info["roles"]) == 2
    assert all(n in info["scales"] for roles in info["roles"] for n in roles.values())
    # the blocks' op ranges tile the graph from the first block's norm to the last block's residual add
    ranges = [la.block_ops(graph, info, b) for b in range(2)]
    assert ranges[0][1] == ranges[1][0] and graph.ops[ranges[0][0]].note == "norm"
    assert graph.ops[ranges[1][1] - 1].output == info["blocks"][-1] == info["roles"][1]["res_mlp"]
    # the check can fail: two roles swapped, or a role on an op of another kind
    swapped = dict(info, roles=[dict(info["roles"][0], q=info["roles"][0]["k"], k=info["roles"][0]["q"])] +
                   info["roles"][1:])
    assert la.check_roles(graph, swapped)
    other = dict(info, roles=[dict(info["roles"][0], attn=info["roles"][0]["mlp_norm"])] + info["roles"][1:])
    assert any("not 'attention'" in p for p in la.check_roles(graph, other))


def test_every_role_has_a_checkpoint_source_and_an_exact_op():
    assert set(la.HF_SOURCE) - {"in"} == set(la.FLOAT_OP) == set(la.ROLES) == set(la.KIND)
    assert la.ROLES.index("attn_raw") == la.ROLES.index("attn") - 1


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("prefix", [None, (5, 17)])
def test_the_exact_ops_on_the_checkpoints_inputs_give_its_outputs(kind, prefix):
    # Every hooked activation is what FLOAT_OP says it is: the exact float64 op applied to the checkpoint's own
    # inputs gives the checkpoint's output (to float32 rounding), the attention (float_attention, grouped and causal,
    # on the reconstructed RoPE outputs and with the prefix's keys and values) included; and the reconstructed post-RoPE
    # keys are HF's KV cache.
    model = _tiny(kind)
    _, x = _data(32, 2, 4)
    cfg = hf_llama.config_of(model.config)
    hq, hkv, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    with torch.no_grad():
        hf = la.hf_activations(model, x, [0, 1], prefix)
        joint = x if prefix is None else torch.cat([torch.tensor([prefix]).expand(x.shape[0], -1), x], 1)
        cache = model(joint, use_cache=True).past_key_values
    n = len(prefix or ())

    def close(a, b):
        return float((a - b).norm() / b.norm()) < 1e-5

    for b in (0, 1):
        h, layer = hf[b], model.model.layers[b]
        at, mlp = layer.self_attn, layer.mlp
        assert close(la._rms(h["in"], layer.input_layernorm), h["attn_norm"])
        for r, p in (("q_proj", at.q_proj), ("k_proj", at.k_proj), ("v_proj", at.v_proj)):
            assert close(la._linear(h["attn_norm"], p), h[r])
        if kind == "qwen3":
            assert close(la._head_rms(h["q_proj"], at.q_norm, hq, dh), h["q_norm"])
            assert close(la._head_rms(h["k_proj"], at.k_norm, hkv, dh), h["k_norm"])
        k_hf = cache[b][0].transpose(1, 2).reshape(x.shape[0], -1, hkv * dh).double()
        assert close(h["k_rope"], k_hf[:, n:])
        k, v = h["k_rope"], h["v_proj"]
        if prefix:
            assert close(h["k_rope_prefix"], k_hf[:, :n])
            k, v = torch.cat([h["k_rope_prefix"], k], 1), torch.cat([h["v_proj_prefix"], v], 1)
        assert close(la.float_attention(h["q_rope"], k, v, hq, hkv, dh), h["attn"])
        assert close(la._linear(h["attn"], at.o_proj), h["o_proj"])
        assert close(h["in"] + h["o_proj"], h["res_attn"])
        assert close(la._rms(h["res_attn"], layer.post_attention_layernorm), h["mlp_norm"])
        assert close(la._linear(h["mlp_norm"], mlp.gate_proj), h["gate_proj"])
        assert close(la._linear(h["mlp_norm"], mlp.up_proj), h["up_proj"])
        assert close(torch.nn.functional.silu(h["gate_proj"]), h["silu"])
        assert close(h["silu"] * h["up_proj"], h["glu"])
        assert close(la._linear(h["glu"], mlp.down_proj), h["down_proj"])
        assert close(h["res_attn"] + h["down_proj"], h["res_mlp"])
    assert torch.equal(hf[1]["in"], hf[0]["res_mlp"])


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("variant", ["plain", "prune", "prefix", "main"])
def test_the_runner_and_teacher_forcing_give_the_graphs_own_integers(kind, variant):
    # the accumulated run's tensors are the plain forward's, integer for integer; and teacher forcing a block with the
    # graph's own residual (in real units) reproduces the accumulated block exactly
    model, graph, info, _, x = _setup(kind, variant)
    env_full, _ = graph.forward(x)
    env, last = {graph.input_name: x}, graph.last_use()
    keep = {n for roles in info["roles"] for n in roles.values()}
    la._run(graph, env, 0, la.block_ops(graph, info, 1)[1], keep, last)
    assert all(torch.equal(env[n], env_full[n]) for n in keep)
    prefix = VARIANTS[variant].get("prefix_tokens")
    with torch.no_grad():
        hf = la.hf_activations(model, x, [1], prefix)
    acc = la.attribute(graph, info, model, x, [1], "accumulated", prefix, hf)[0]
    hf[1]["in"] = env_full[info["roles"][1]["in"]].double() * info["s_residual"]
    tf = la.attribute(graph, info, model, x, [1], "teacher", prefix, hf)[0]
    assert tf["rows"] == acc["rows"] and tf["pattern"] == acc["pattern"]


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_a_one_block_build_has_the_full_builds_first_block(kind):
    # --blocks 0 builds one block: the same block 0, integer for integer, with the same scales
    model, graph, info, calib, x = _setup(kind, "prefix")
    with torch.no_grad():
        one, info1 = hf_llama.build_llama_from_hf(model, calib, n_layers=1, prefix_tokens=(5, 17), **SMOOTH)
    assert info1["roles"][0] == info["roles"][0] and info1["s_residual"] == info["s_residual"]
    names = list(info["roles"][0].values())
    assert all(torch.equal(torch.as_tensor(info1["scales"][n]), torch.as_tensor(info["scales"][n])) for n in names)
    a, _ = graph.forward(x)
    b, _ = one.forward(x)
    assert all(torch.equal(a[n], b[n]) for n in names)


# ------------------------------------------------------------------ fidelity of the smoothed graph, op by op

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("mode", ["accumulated", "teacher"])
def test_every_tensor_of_the_smoothed_graph_tracks_the_checkpoint(kind, variant, mode):
    model, graph, info, _, x = _setup(kind, variant)
    res = _attribute(model, graph, info, x, mode, VARIANTS[variant].get("prefix_tokens"))
    # (on these random weights the product balance leaves the SwiGLU product a little less resolution: its own mean
    # cosine 0.9989 on the tiny Llama's first block, relative error 0.074)
    local_cos = 0.998 if variant == "main" else 0.999
    for r in res:
        for role, row in r["rows"].items():
            assert row["vs_hf"]["cos_mean"] >= (0.98 if mode == "accumulated" else 0.99), (r["block"], role, row)
            assert row["local"]["cos_mean"] >= local_cos and row["local"]["rel"] <= 0.08, (r["block"], role, row)
        steps = la.score_steps(r["pattern"])
        assert max(v for k, v in steps.items() if k != "in") <= 0.08, steps
        assert r["pattern"]["qk"]["score_err_nats"] <= (0.25 if mode == "accumulated" else 0.1), r["pattern"]["qk"]
        assert la.first_flag(r, 0.1, 0.99, 0.1) is None
    # the accumulated block outputs are llama_diag.py's per-block cosines (hf_llama.block_cosines)
    if mode == "accumulated":
        prefix = VARIANTS[variant].get("prefix_tokens")
        with torch.no_grad():
            diag = hf_llama.block_cosines(hf_llama.residual_states(graph, info, x),
                                          hf_llama.hf_residual_states(model, x, prefix=prefix))
        for r in res:
            out = r["rows"]["res_mlp"]["vs_hf"]
            assert abs(out["cos_mean"] - diag[r["block"] + 1]["mean"]) < 1e-9
            assert abs(out["cos_min"] - diag[r["block"] + 1]["min"]) < 1e-9


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("mode", ["accumulated", "teacher"])
def test_the_product_balance_is_undone_in_the_checkpoints_units(kind, mode):
    # glu_smooth divides up's row j by g_j and multiplies the down projection's column j by it; the attribution brings
    # up's tensors and the product back by g_j (above: every op's own error stays small).  Without that, the exact up
    # projection and the exact down projection see inputs in the balanced units: the up projection is flagged
    model, graph, info, _, x = _setup(kind, "main")
    g = info["glu_smooth_g"]
    assert all(t is not None for t in g) and float(max(t.max() / t.min() for t in g)) > 1.5
    unbalanced = dict(info, glu_smooth_g=[None] * len(g))
    for b in (0, 1):
        right = _attribute(model, graph, info, x, mode, (5, 17), blocks=(b,))[0]
        wrong = _attribute(model, graph, unbalanced, x, mode, (5, 17), blocks=(b,))[0]
        assert la.first_flag(right, 0.1, 0.99, 0.1) is None and la.first_flag(wrong, 0.1, 0.99, 0.1) == "up_proj"
        for role in ("up_proj", "down_proj"):
            assert wrong["rows"][role]["local"]["rel"] > 0.1 > 0.02 > right["rows"][role]["local"]["rel"], role
        assert wrong["rows"]["down_proj"]["vs_hf"] == right["rows"]["down_proj"]["vs_hf"]


# ------------------------------------------------------------------ a broken builder is caught at its op

@pytest.mark.parametrize("side", ["q", "k"])
def test_a_wrong_q_or_k_norm_gain_is_caught_at_that_norm(side, monkeypatch):
    model, _, _, calib, x = _setup("qwen3")
    right = hf_llama._Builder.head_norm

    def wrong(self, z, heads, dh, gamma, prefix, note, c=None):
        return right(self, z, heads, dh, gamma.flip(0) if note == f"{side} norm" else gamma, prefix, note, c=c)

    monkeypatch.setattr(hf_llama._Builder, "head_norm", wrong)
    with torch.no_grad():
        graph, info = hf_llama.build_llama_from_hf(model, calib, **SMOOTH)
    for mode in ("teacher", "accumulated"):
        r = _attribute(model, graph, info, x, mode, blocks=(0,))[0]
        assert la.first_flag(r, 0.1, 0.99, 0.1) == f"{side}_norm", mode
        assert r["rows"][f"{side}_norm"]["local"]["rel"] > 0.2


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_a_swiglu_product_off_by_two_is_caught_at_the_product(kind, monkeypatch):
    model, _, _, calib, x = _setup(kind)
    from pvi.core import graph as core_graph
    right = core_graph.requant_mul                                  # the product op's integer map (mul_requant/1)
    monkeypatch.setattr(core_graph, "requant_mul", lambda a, b, m, *rest: right(a, b, m * 2, *rest))
    with torch.no_grad():
        graph, info = hf_llama.build_llama_from_hf(model, calib, **SMOOTH)
    for mode in ("teacher", "accumulated"):
        r = _attribute(model, graph, info, x, mode, blocks=(0,))[0]
        assert la.first_flag(r, 0.1, 0.99, 0.1) == "glu", mode
        glu = r["rows"]["glu"]["local"]
        assert 0.8 < glu["rel"] < 1.1 and glu["cos_mean"] > 0.99      # a scale: the cosine alone would miss it


# ------------------------------------------------------------------ Qwen3-0.6B's q/k-norm gains and the balance

@pytest.mark.parametrize("mode", ["teacher", "accumulated"])
def test_qwen3_06b_k_norm_gains_cost_the_scores_and_the_balance_restores_them(mode):
    model = _realistic_qwen3()
    calib, x = _data(64, 4, 2)
    with torch.no_grad():
        old, old_info = hf_llama.build_llama_from_hf(model, calib, qk_balance=False, **SMOOTH)
        new, new_info = hf_llama.build_llama_from_hf(model, calib, **SMOOTH)            # qk_balance: the default
    a = _attribute(model, old, old_info, x, mode, blocks=(0,))[0]
    b = _attribute(model, new, new_info, x, mode, blocks=(0,))[0]
    sa, sb = la.score_steps(a["pattern"]), la.score_steps(b["pattern"])
    # without the balance: the k norm's int8 range is set by its 96.5 gain; it adds most of the scores' error
    assert la.worst(a)["score_step"] == "k_norm" and sa["k_norm"] > 0.6 and a["rows"]["k_norm"]["zero"] > 0.2
    assert max(v for k, v in sa.items() if k not in ("q_norm", "k_norm")) < 0.2
    assert a["pattern"]["qk"]["score_err_nats"] > 1.0 and a["rows"]["res_mlp"]["vs_hf"]["cos_mean"] < 0.9
    # with it: every op adds less than 0.2 nats, and the block's output is back near the checkpoint's
    assert max(sb.values()) < 0.2 and b["rows"]["k_norm"]["zero"] < 0.1
    assert b["pattern"]["qk"]["score_err_nats"] < 0.4 and b["rows"]["res_mlp"]["vs_hf"]["cos_mean"] > 0.98
    # the per-RoPE-pair factor: q's channel scale times k's is the same in every channel (the product m_s uses)
    roles, sc = new_info["roles"][0], new_info["scales"]
    c = new_info["qk_balance_c"][0]
    assert torch.equal(c[:64], c[64:]) and float(c.max() / c.min()) > 4
    prod = sc[roles["q_rope"]][:128] * sc[roles["k_rope"]][:128]
    assert float((prod / prod[0] - 1).abs().max()) < 1e-12
    assert old_info["qk_balance_c"] == [None] and new_info["qk_balance"] and not old_info["qk_balance"]


def test_the_balance_leaves_llama_graphs_and_the_structure_alone():
    model, _, _, calib, _ = _setup("llama")
    with torch.no_grad():
        on, info = hf_llama.build_llama_from_hf(model, calib, **SMOOTH)
        off, _ = hf_llama.build_llama_from_hf(model, calib, qk_balance=False, **SMOOTH)
    assert on.digest() == off.digest() and not info["qk_balance"] and info["qk_balance_c"] == [None, None]
    qwen, _, _, qcalib, _ = _setup("qwen3")
    with torch.no_grad():
        on, _ = hf_llama.build_llama_from_hf(qwen, qcalib, **SMOOTH)
        off, _ = hf_llama.build_llama_from_hf(qwen, qcalib, qk_balance=False, **SMOOTH)
    assert on.digest() != off.digest()                  # other gain constants: the digest binds them
    shape = hf_llama.config_of(qwen.config)
    assert hf_llama.matches_benchmark_graph(on, shape) == [] == hf_llama.matches_benchmark_graph(off, shape)
