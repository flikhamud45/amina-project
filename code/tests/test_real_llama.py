"""pvi.fullcheck.real_llama (plan 2, F2b-Q): integer Llama / Qwen3 decoders from real weights, on tiny randomly
initialised Hugging Face models (no download).

The models have 2 blocks, d = 64 and grouped-query attention (Llama: 4 query heads over 2 key/value heads, head_dim
16; Qwen3: 8 over 2, head_dim 16, so the query width 128 differs from d as in Qwen3-0.6B), random RMSNorm and q/k-norm
gains in [0.5, 1.5], and weights of std 0.15. Hugging Face's default std 0.02 leaves the attention nearly uniform, so
a wrong RoPE pairing would go unnoticed; at 0.15 the thresholds below separate the builder from a builder with the
wrong head grouping or RoPE pairing by a wide margin (measured on two seeds per family, smoothed recipe, 4 x 32
random tokens):

==========================  ===================  ======================  =====================
quantity                    this builder         wrong GQA grouping      wrong RoPE pairing
==========================  ===================  ======================  =====================
logit cosine, mean          0.992 - 0.995        0.18 - 0.28             0.52 - 0.78
logit cosine, min           0.949 - 0.981        < 0                     -0.31 - 0.27
top-1 agreement             0.852 - 0.883        0.05 - 0.09             0.16 - 0.43
block cosine, min           0.946 - 0.981        < 0                     -0.11 - 0.35
==========================  ===================  ======================  =====================

Hence: mean logit cosine >= 0.98, minimum >= 0.9, top-1 >= 0.75 and every block's minimum cosine >= 0.9 for the
smoothed graph; the scalar-gain graph (gain 4, no clipping) measured 0.960 - 0.970 mean, so it is held to a mean
>= 0.94. A Qwen3 q/k-norm gain moved to the other half of its RoPE pair measured a mean of 0.684 - 0.867 and top-1
0.344 - 0.492 (the same two seeds), below the same thresholds.
"""

from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.fullcheck import real_llama as rl                                 # noqa: E402
from pvi.fullcheck.graph import CheapOp, MatOp                              # noqa: E402
from pvi.fullcheck.transformer import CONFIGS, _rope, decoder_param_count   # noqa: E402

SMOOTH = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
_CACHE: dict = {}


def _tiny(kind: str, seed: int = 0):
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


def _setup(kind: str, recipe: str = "smooth", prune: bool = False):
    key = (kind, recipe, prune)
    if key not in _CACHE:
        model = _CACHE.setdefault(kind, _tiny(kind))
        g = torch.Generator().manual_seed(1)
        calib = torch.randint(0, 97, (2, 32), generator=g)
        x = torch.randint(0, 97, (4, 32), generator=g)
        kw = SMOOTH if recipe == "smooth" else dict(norm_gain=4.0)
        graph, info = rl.build_llama_from_hf(model, calib, prune_last=prune, **kw)
        _CACHE[key] = model, graph, info, calib, x
    return _CACHE[key]


def _scores(graph, info, model, x):
    with torch.no_grad():
        ref = model(x).logits.double()
    got = rl.real_logits(graph, info, x, all_positions=True)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    top1 = float((got.argmax(-1) == ref.argmax(-1)).double().mean())
    return float(cos.mean()), float(cos.min()), top1


# ------------------------------------------------------------------ structure

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("prune", [False, True])
def test_the_graph_is_the_benchmark_graph_of_its_shape(kind, prune):
    model, graph, info, _, _ = _setup(kind, prune=prune)
    shape = rl.config_of(model.config, "tiny")
    assert rl.same_shape(info["config"], shape) == [] and shape.qk_norm == (kind == "qwen3")
    assert shape.tied == (kind == "qwen3") and (shape.n_heads, shape.n_kv_heads) == ((4, 2) if kind == "llama"
                                                                                       else (8, 2))
    # op kinds, names, wiring, shapes, biases, notes and claim columns: no difference, not even an LM head bias
    assert rl.matches_benchmark_graph(graph, shape, prune_last=prune) == []
    assert all(op.weight.dtype == torch.int8 for op in graph.mat_ops)
    assert rl.matches_benchmark_graph(graph, shape, prune_last=not prune) != []      # the check can fail


def test_the_ladders_configs_are_the_checkpoints_shapes():
    # the published config.json values of Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, Qwen/Qwen3-4B-Base and
    # TinyLlama/TinyLlama_v1.1 (shapes only; max_position_embeddings is not used by a RoPE graph)
    qwen = dict(vocab_size=151936, num_key_value_heads=8, head_dim=128, rope_theta=1e6, tie_word_embeddings=True,
                max_position_embeddings=32768)
    hf = {"qwen3-0.6b": transformers.Qwen3Config(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
                                                 intermediate_size=3072, **qwen),
          "qwen3-1.7b": transformers.Qwen3Config(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                                                 intermediate_size=6144, **qwen),
          "qwen3-4b": transformers.Qwen3Config(hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
                                               intermediate_size=9728, **qwen),
          "tinyllama-1.1b": transformers.LlamaConfig(hidden_size=2048, num_hidden_layers=22, num_attention_heads=32,
                                                     num_key_value_heads=4, intermediate_size=5632, vocab_size=32000,
                                                     tie_word_embeddings=False, max_position_embeddings=2048)}
    for name, cfg in hf.items():
        assert rl.same_shape(rl.config_of(cfg, name), CONFIGS[name]) == [], name
    for name, published in [("qwen3-0.6b", 0.6e9), ("qwen3-1.7b", 1.7e9), ("tinyllama-1.1b", 1.1e9)]:
        n = decoder_param_count(CONFIGS[name])
        assert abs(n - published) / published < 0.06, (name, n)


# ------------------------------------------------------------------ exactness

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_the_integer_forward_and_the_builder_are_deterministic(kind):
    model, graph, info, calib, x = _setup(kind)
    env1, claims1 = graph.forward(x)
    env2, claims2 = graph.forward(x)
    assert torch.equal(env1[graph.output_name], env2[graph.output_name])
    assert all(torch.equal(claims1[k], claims2[k]) for k in claims1)
    again, info2 = rl.build_llama_from_hf(model, calib, **SMOOTH)
    assert again.digest() == graph.digest() and info2["logit_scale"] == info["logit_scale"]
    assert all(torch.equal(a.weight, b.weight) for a, b in zip(again.mat_ops, graph.mat_ops))


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_lean_chunks_and_all_positions_give_the_same_integers(kind):
    _, graph, info, _, x = _setup(kind)
    flat = torch.cat([z for _, z in rl.lean_logits(graph, x, chunk=1 << 20)], 1)
    chunked = torch.cat([z for _, z in rl.lean_logits(graph, x, chunk=5)], 1)
    assert torch.equal(flat, chunked) and flat.shape[:2] == x.shape
    # one pass at every position is one query per prefix (and the plain forward at the last position)
    for t in (1, 9, x.shape[1]):
        env, _ = graph.forward(x[:, :t])
        assert torch.equal(env[graph.output_name][:, -1], flat[:, t - 1])
    assert torch.equal(rl.real_logits(graph, info, x)[:, 0], flat[:, -1].double() * info["logit_scale"])


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_prune_last_gives_the_unpruned_last_position_logits(kind):
    _, graph, _, _, x = _setup(kind)
    _, pruned, _, _, _ = _setup(kind, prune=True)
    assert sum(isinstance(o, CheapOp) and o.note == "last position" for o in pruned.ops) == 3
    for t in (x.shape[1], 9):
        a, _ = graph.forward(x[:, :t])
        b, claims = pruned.forward(x[:, :t])
        assert torch.equal(a[graph.output_name], b[pruned.output_name])
    # the pruned block's q, attention output, residuals and MLP claim one column per query
    assert min(c.shape[1] for c in claims.values()) == x.shape[0]
    with pytest.raises(ValueError):
        next(rl.lean_logits(pruned, x))


def test_rope_is_hugging_faces_rotate_half():
    # the integer RoPE of int8 values is HF's apply_rotary_pos_emb of the same values, to the rounding
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
    cfg = transformers.LlamaConfig(hidden_size=64, num_attention_heads=4, rope_theta=1e4,
                                   max_position_embeddings=64)
    x = torch.randint(-90, 91, (1, 40, 4, 16), generator=torch.Generator().manual_seed(3))
    cos, sin = LlamaRotaryEmbedding(cfg)(x.double(), torch.arange(40)[None])
    ref, _ = apply_rotary_pos_emb(x.double().transpose(1, 2), x.double().transpose(1, 2), cos.double(),
                                  sin.double())
    got = _rope(x, 1e4)
    assert (got - ref.transpose(1, 2)).abs().max() <= 1.0


# ------------------------------------------------------------------ fidelity

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_the_logits_and_every_block_track_the_float_model(kind):
    model, graph, info, _, x = _setup(kind)
    mean, low, top1 = _scores(graph, info, model, x)
    assert mean >= 0.98 and low >= 0.9 and top1 >= 0.75, (mean, low, top1)
    blocks = rl.block_cosines(rl.residual_states(graph, info, x), rl.hf_residual_states(model, x))
    assert len(blocks) == 3 and min(b["min"] for b in blocks) >= 0.9, blocks
    _, scalar, s_info, _, _ = _setup(kind, "scalar")
    assert _scores(scalar, s_info, model, x)[0] >= 0.94


def _tiled_heads(model):
    """A model whose interleaved head grouping (HF's ``repeat_kv``) is ``model``'s tiled one (query head ``j`` on
    key/value head ``j % n_kv``): its graph is what a builder with the wrong grouping would give for ``model``."""
    other = copy.deepcopy(model)
    c = model.config
    hq, hkv = c.num_attention_heads, c.num_key_value_heads
    dh = getattr(c, "head_dim", None) or c.hidden_size // hq
    rep = hq // hkv
    idx = torch.cat([torch.arange(j * dh, (j + 1) * dh) for j in
                     ((p % rep) * hkv + p // rep for p in range(hq))])
    with torch.no_grad():
        for layer in other.model.layers:
            layer.self_attn.q_proj.weight.copy_(layer.self_attn.q_proj.weight[idx])
            layer.self_attn.o_proj.weight.copy_(layer.self_attn.o_proj.weight[:, idx])
    return other


def _paired_rope(model):
    """A model whose half-rotation RoPE is ``model``'s rotation of adjacent pairs ``(2i, 2i+1)``."""
    other = copy.deepcopy(model)
    c = model.config
    hq, hkv = c.num_attention_heads, c.num_key_value_heads
    dh = getattr(c, "head_dim", None) or c.hidden_size // hq
    within = torch.cat([torch.arange(0, dh, 2), torch.arange(1, dh, 2)])
    with torch.no_grad():
        for layer in other.model.layers:
            at = layer.self_attn
            for proj, heads in ((at.q_proj, hq), (at.k_proj, hkv)):
                proj.weight.copy_(proj.weight[torch.cat([j * dh + within for j in range(heads)])])
            if hasattr(at, "q_norm"):
                at.q_norm.weight.copy_(at.q_norm.weight[within])
                at.k_norm.weight.copy_(at.k_norm.weight[within])
    return other


def _swapped_qk_gains(model):
    """A model whose q/k-norm gain of channel ``j`` is ``model``'s gain of its RoPE partner ``j +- head_dim/2``: its
    graph is what a builder that put each gain on the wrong side of the rotation would give for ``model``."""
    other = copy.deepcopy(model)
    with torch.no_grad():
        for layer in other.model.layers:
            for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm):
                half = norm.weight.shape[0] // 2
                norm.weight.copy_(torch.cat([norm.weight[half:], norm.weight[:half]]))
    return other


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("wrong", [_tiled_heads, _paired_rope])
def test_a_wrong_head_grouping_or_rope_pairing_fails_the_thresholds(kind, wrong):
    model, _, _, calib, x = _setup(kind)
    graph, info = rl.build_llama_from_hf(wrong(model), calib, **SMOOTH)
    mean, low, top1 = _scores(graph, info, model, x)
    assert mean < 0.9 and top1 < 0.5, (mean, low, top1)


def test_misplaced_qk_norm_gains_fail_the_thresholds():
    # the q/k-norm gains are per channel before RoPE: a gain on the other half of its pair is detected (seed 0:
    # logit cosine 0.684, top-1 0.344; seed 1: 0.867, 0.492 -- against 0.994 / 0.883 and 0.995 / 0.859 when right)
    model, _, _, calib, x = _setup("qwen3")
    graph, info = rl.build_llama_from_hf(_swapped_qk_gains(model), calib, **SMOOTH)
    mean, low, top1 = _scores(graph, info, model, x)
    assert mean < 0.9 and top1 < 0.5, (mean, low, top1)


# ------------------------------------------------------------------ the protocol on real graphs

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("prune", [False, True])
def test_v0_accepts_real_graphs_in_modes_c_and_kpre_and_rejects_a_tamper(kind, prune):
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query

    _, graph, _, _, x = _setup(kind, prune=prune)
    ids = x[:1, :16]
    params = params_for(20, len(graph.mat_ops))
    coms = commit_graph(graph, params.rate)               # also checks the claim range bound
    prover = Prover(graph, commitments=coms)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    kpre = Verifier(graph.public(), params, "Kpre", weights=weights)
    kpre.precompute(Challenger(seed=11))
    for v in (Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()}), kpre):
        for seed in range(2):
            assert run_query(prover, v, ids, seed=seed)["accepted"]
        for target in (graph.mat_ops[1].name, graph.mat_ops[len(graph.mat_ops) // 2].name, graph.mat_ops[-1].name):
            def tamper(op, z, target=target):
                if op.name == target:
                    z = z.clone()
                    z.view(-1)[0] -= 3
                return z
            assert not run_query(prover, v, ids, seed=1, forward_kwargs={"tamper": tamper})["accepted"]


# ------------------------------------------------------------------ the opt-in per-token exponent (glu_exp)

def test_the_per_token_exponent_is_the_smallest_clip_free_one():
    g = torch.Generator().manual_seed(4)
    a = torch.randint(-127, 128, (2, 6, 40), generator=g)
    b = torch.randint(-127, 128, (2, 6, 40), generator=g)
    a[0, 2] //= 40                                           # small rows and a massive one
    b[1, 4] = 127
    mult = 1 << 27
    y = a * b * mult
    e = rl._exponents(y, 12)
    out = rl.swiglu_exp(a, b, mult, 12)
    assert out.abs().max() <= 127 and e.min() >= 0 and len(torch.unique(e)) > 1
    exact = y.double() / (1 << 30)
    assert ((out.double() * 2.0 ** e.double() - exact).abs() <= 2.0 ** e.double() / 2).all()   # round-half-up
    lower = (y + (1 << (29 + 0))) >> 30                      # e - 1 would clip some entry of each row with e > 0
    for i, j in torch.nonzero(e[..., 0] > 0).tolist():
        sh = 30 + int(e[i, j, 0]) - 1
        assert ((y[i, j] + (1 << (sh - 1))) >> sh).abs().max() > 127
    assert torch.equal(out[e[..., 0] == 0], lower[e[..., 0] == 0].clamp(-127, 127))
    # the residual add after it: z's row t carries 2^e_t
    z = torch.randint(-(1 << 20), 1 << 20, (2, 6, 8), generator=g)
    r = torch.randint(-(1 << 20), 1 << 20, (2, 6, 8), generator=g)
    m = 3 << 20
    got = rl.residual_exp(r, z, a, b, mult, m, 12)
    want = r + torch.floor(z.double() * m * 2.0 ** e.double() / (1 << 30) + 0.5).to(torch.int64)
    assert torch.equal(got, want)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_glu_exp_graphs_track_the_model_and_pass_v0_and_v1(kind):
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query

    model, base, _, calib, x = _setup(kind)
    graph, info = rl.build_llama_from_hf(model, calib, glu_exp=8, **SMOOTH)
    assert info["glu_emax"] == [8, 8]
    # exactly two ops per block differ from the benchmark graph: the product's note, the residual add's wiring
    diff = rl.matches_benchmark_graph(graph, rl.config_of(model.config))
    assert len(diff) == 4 and sum("per-token" in d for d in diff) == 2 and sum("wiring" in d for d in diff) == 2
    assert [type(o) for o in graph.ops] == [type(o) for o in base.ops]
    mean, low, top1 = _scores(graph, info, model, x)
    assert mean >= 0.98 and low >= 0.9 and top1 >= 0.75, (mean, low, top1)
    pruned, _ = rl.build_llama_from_hf(model, calib, glu_exp=8, prune_last=True, **SMOOTH)
    assert torch.equal(graph.forward(x)[0][graph.output_name], pruned.forward(x)[0][pruned.output_name])
    ids = x[:1, :16]
    params = params_for(20, len(graph.mat_ops), cut=True)
    coms = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=coms)
    kpre = Verifier(graph.public(), params, "Kpre", weights={o.name: (o.weight, o.bias) for o in graph.mat_ops})
    kpre.precompute(Challenger(seed=11))
    readers = graph.readers()
    down = [o for o in graph.mat_ops if any(getattr(r, "note", "").endswith("per-token shift")
                                            for r in readers.get(o.output, []))]
    assert len(down) == 2

    def tamper(op, z):
        if op.name == down[0].name:
            z = z.clone()
            z.view(-1)[0] += 1
        return z
    for v in (Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()}), kpre):
        assert run_query(prover, v, ids, seed=1)["accepted"]
        assert not run_query(prover, v, ids, seed=1, forward_kwargs={"tamper": tamper})["accepted"]
    v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    out = cp.run_cut_query(prover, v, ids, seed=1)
    assert out["accepted"], out["rejected_at"]
    assert not cp.run_cut_query(prover, v, ids, seed=1, forward_kwargs={"tamper": tamper})["accepted"]


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_v1_accepts_a_real_graph_and_rejects_a_tamper(kind):
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.protocol import Prover, Verifier, commit_graph, params_for

    _, graph, _, _, x = _setup(kind)
    assert all(isinstance(o.params["requant"]["mult"], int) for o in graph.ops
               if isinstance(o, CheapOp) and o.params.get("requant"))
    params = params_for(20, len(graph.mat_ops), cut=True)
    coms = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=coms)
    v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    out = cp.run_cut_query(prover, v, x[:1, :16], seed=1)
    assert out["accepted"], out["rejected_at"]
    assert isinstance(graph.mat_ops[0], MatOp) and graph.mat_ops[0].layout == "embed"
    for target in (graph.mat_ops[1].name, graph.mat_ops[len(graph.mat_ops) // 2].name, graph.mat_ops[-1].name):
        def tamper(op, z, target=target):
            if op.name == target:
                z = z.clone()
                z.view(-1)[0] -= 3
            return z
        assert not cp.run_cut_query(prover, v, x[:1, :16], seed=1, forward_kwargs={"tamper": tamper})["accepted"]
