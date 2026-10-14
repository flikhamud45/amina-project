"""``pvi.models.hf_llama``: integer Llama / Qwen3 decoders from real weights, on tiny randomly initialised Hugging
Face models (no download).

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

The opt-in fixed prefix (``prefix_tokens``) is held to the same thresholds against HF on ``[prefix | x]`` (measured,
smoothed, two seeds per family, prefixes (5,) and (5, 17): mean 0.992 - 0.996, minimum 0.907 - 0.986, top-1 0.828 -
0.961, block minimum 0.935 - 0.986; scalar mean 0.960 - 0.974), and to an exact equality: the prefixed graph on x is,
integer for integer, the same ops on [prefix | x] with RoPE at the joint positions and the prefix's keys and values
replaced by the graph's constants (``_glued``).
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.core.graph import CheapOp, IntGraph, MatOp  # noqa: E402
from pvi.models import decoder as tr  # noqa: E402
from pvi.models import hf_llama as rl  # noqa: E402
from pvi.models.decoder import CONFIGS, integer_attention, _rope, decoder_param_count  # noqa: E402

SMOOTH = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
PREFIX = (5, 17)
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


def _setup(kind: str, recipe: str = "smooth", prune: bool = False, prefix: tuple | None = None):
    key = (kind, recipe, prune, prefix)
    if key not in _CACHE:
        model = _CACHE.setdefault(kind, _tiny(kind))
        g = torch.Generator().manual_seed(1)
        calib = torch.randint(0, 97, (2, 32), generator=g)
        x = torch.randint(0, 97, (4, 32), generator=g)
        kw = SMOOTH if recipe == "smooth" else dict(norm_gain=4.0)
        graph, info = rl.build_llama_from_hf(model, calib, prune_last=prune, prefix_tokens=prefix, **kw)
        _CACHE[key] = model, graph, info, calib, x
    return _CACHE[key]


def _scores(graph, info, model, x, prefix: tuple | None = None):
    """Logit cosine (mean, min over positions) and top-1 agreement of the graph against the HF model, which runs on
    ``[prefix | x]`` (the prefix's positions dropped) when the graph was built with that prefix."""
    n = len(prefix or ())
    with torch.no_grad():
        ref = model(torch.cat([torch.tensor([prefix or ()], dtype=torch.int64).expand(x.shape[0], -1), x], 1)
                    ).logits[:, n:].double()
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
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier

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
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.v1 import protocol as cp

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
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.verifier import Verifier
    from pvi.v1 import protocol as cp

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


# ------------------------------------------------------------------ the opt-in fixed prefix (prefix_tokens)

def _defaults(fn) -> dict:
    """A cheap op's public constants by name (its ``OpFn``'s keyword arguments)."""
    return dict(fn.consts)


def _attention_ops(graph) -> list:
    return [o for o in graph.ops if isinstance(o, CheapOp) and o.note == "attention"]


def _glued(graph: IntGraph) -> IntGraph:
    """A prefixed graph's ops, weights and constants as a graph of ``[prefix | query]``: RoPE at the positions of
    the joint sequence (no offset), and plain causal attention over the joint sequence
    (``pvi.models.decoder.integer_attention``) whose first ``P`` keys and values are replaced by the prefix's int8
    constants. The prefix positions' other values are computed from the prefix tokens and only feed the keys and
    values that are replaced."""
    n = graph.meta["prefix_len"]
    ops = []
    for op in graph.ops:
        if isinstance(op, CheapOp) and op.note == "rope":
            d = _defaults(op.fn)
            assert d.get("offset", d.get("base")) == n
            if len(op.inputs) == 2:          # a pruned q: the last positions of the joint sequence
                fn = lambda a, x, H=d["heads"], dh=d["dh"], th=d["theta"]: _rope(          # noqa: E731
                    a.reshape(*a.shape[:-1], H, dh), th, x.shape[1] - a.shape[1]).reshape(a.shape)
            else:
                fn = lambda a, H=d["heads"], dh=d["dh"], th=d["theta"]: _rope(              # noqa: E731
                    a.reshape(*a.shape[:-1], H, dh), th).reshape(a.shape)
            op = dataclasses.replace(op, fn=fn)
        elif isinstance(op, CheapOp) and op.note == "attention":
            def fn(q, k, v, d=_defaults(op.fn)):
                kp, vp = (c.to(torch.int64).expand(k.shape[0], -1, -1) for c in (d["k_prefix"], d["v_prefix"]))
                return integer_attention(q, torch.cat([kp, k[:, n:]], 1), torch.cat([vp, v[:, n:]], 1), d["m_s"],
                                         d["m_o"], d["heads"], d["kv_heads"], d["dh"], d["pmax"])
            op = dataclasses.replace(op, fn=fn)
        ops.append(op)
    return IntGraph(ops, graph.input_name, graph.output_name)


def _joint(prefix, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.tensor([prefix], dtype=torch.int64).expand(x.shape[0], -1), x], 1)


def _with_constant(graph: IntGraph, layer: int, name: str, fix) -> IntGraph:
    """``graph`` with the constant ``name`` of its ``layer``-th attention op replaced by ``fix(constant)``: the same
    kind, name and other constants."""
    ops, seen = [], 0
    for op in graph.ops:
        if isinstance(op, CheapOp) and op.note == "attention":
            if seen == layer:
                consts = dict(op.fn.consts)
                consts[name] = fix(consts[name])
                op = dataclasses.replace(op, fn=op.fn.with_consts(consts))
            seen += 1
        ops.append(op)
    return IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta))


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("prune", [False, True])
def test_a_prefixed_graph_is_the_benchmark_graph_of_its_shape(kind, prune):
    model, graph, info, _, x = _setup(kind, prune=prune, prefix=PREFIX)
    _, plain, _, _, _ = _setup(kind, prune=prune)
    assert graph.meta == {"prefix_len": 2, "prefix_tokens": [5, 17]} and plain.meta == {}
    assert info["prefix_tokens"] == [5, 17] and info["prefix_len"] == 2
    assert rl.matches_benchmark_graph(graph, rl.config_of(model.config, "tiny"), prune_last=prune) == []
    # the prefix adds no claim: the weight ops and their claim columns are the graph's without a prefix
    assert [(o.name, o.n_rows, o.n_in, o.layout) for o in graph.mat_ops] == \
        [(o.name, o.n_rows, o.n_in, o.layout) for o in plain.mat_ops]
    for t in (1, 9, x.shape[1]):
        assert graph.claim_columns(x[:, :t]) == plain.claim_columns(x[:, :t])
    # per layer, two int8 constants [1, P, Hkv head_dim], the default arguments of the attention op
    hkv_dh = model.config.num_key_value_heads * 16
    for op, (kp, vp) in zip(_attention_ops(graph), info["prefix_kv"]):
        d = _defaults(op.fn)
        assert d["k_prefix"] is kp and d["v_prefix"] is vp and kp.dtype == vp.dtype == torch.int8
        assert kp.shape == vp.shape == (1, 2, hkv_dh)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("prefix", [(5,), PREFIX])
def test_the_prefixed_forward_is_the_graph_on_prefix_and_query_with_constant_keys_and_values(kind, prefix):
    # The exact equality: for a query x [B, T], the prefixed graph's integers on x -- every weight op's claims, the
    # residual stream at every block boundary and the logits at every position -- equal those of _glued(graph) on
    # [prefix | x] at x's positions P .. P+T-1: the same ops and constants on the joint sequence, RoPE at its own
    # positions, plain causal attention whose first P keys and values are the graph's int8 constants.
    _, graph, info, _, x = _setup(kind, prefix=prefix)
    n, b = len(prefix), x.shape[0]
    joint = _joint(prefix, x)
    glued = _glued(graph)
    _, claims = graph.forward(x)
    _, claims_j = glued.forward(joint)
    for op in graph.mat_ops:                     # columns (window, position); the LM head's: the last position
        mine = claims[op.name].reshape(op.n_rows, b, -1)
        assert torch.equal(mine, claims_j[op.name].reshape(op.n_rows, b, -1)[:, :, -mine.shape[2]:]), op.name
    ours, theirs = rl.residual_states(graph, info, x), rl.residual_states(glued, info, joint)
    assert len(ours) == 3 and all(torch.equal(a, c[:, n:]) for a, c in zip(ours, theirs))
    flat = torch.cat([z for _, z in rl.lean_logits(graph, x)], 1)
    assert torch.equal(flat, torch.cat([z for _, z in rl.lean_logits(glued, joint)], 1)[:, n:])
    # the check can fail: the query's RoPE at the positions 0 .. T-1 (no offset) gives other integers
    shifted = IntGraph([g if isinstance(o, CheapOp) and o.note == "rope" else o for o, g in zip(graph.ops, glued.ops)],
                       graph.input_name, graph.output_name)
    assert not torch.equal(flat, torch.cat([z for _, z in rl.lean_logits(shifted, x)], 1))


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_prune_last_with_a_prefix_gives_the_unpruned_last_position_logits(kind):
    _, graph, _, _, x = _setup(kind, prefix=PREFIX)
    _, pruned, _, _, _ = _setup(kind, prune=True, prefix=PREFIX)
    for t in (x.shape[1], 9):
        a, _ = graph.forward(x[:, :t])
        b, claims = pruned.forward(x[:, :t])
        assert torch.equal(a[graph.output_name], b[pruned.output_name])
    assert min(c.shape[1] for c in claims.values()) == x.shape[0]
    # and they are the glued graph's on [prefix | x] (the pruned q's RoPE at the last joint position)
    j, _ = _glued(pruned).forward(_joint(PREFIX, x))
    assert torch.equal(pruned.forward(x)[0][pruned.output_name], j[pruned.output_name])


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_a_prefixed_graph_is_deterministic_and_independent_of_the_attention_path(kind, monkeypatch):
    model, graph, info, calib, x = _setup(kind, prefix=PREFIX)
    env1, claims1 = graph.forward(x)
    env2, claims2 = graph.forward(x)
    assert torch.equal(env1[graph.output_name], env2[graph.output_name])
    assert all(torch.equal(claims1[k], claims2[k]) for k in claims1)
    again, info2 = rl.build_llama_from_hf(model, calib, prefix_tokens=PREFIX, **SMOOTH)
    assert again.digest() == graph.digest() and again.meta == graph.meta
    assert all(torch.equal(a, c) for p, q in zip(info["prefix_kv"], info2["prefix_kv"]) for a, c in zip(p, q))
    # the integer int64 attention path (int64 scores and gaps, last-rows causal mask) gives the same integers as the
    # default int32 path; the GPU's integers are compared with the CPU's by
    # ``python -m experiments.quality.ppl_llama --check``
    monkeypatch.setattr(tr, "_int32_scores", lambda *a: False)
    env3, claims3 = graph.forward(x)
    assert torch.equal(env3[graph.output_name], env1[graph.output_name])
    assert all(torch.equal(claims3[k], claims1[k]) for k in claims1)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_the_prefix_constants_are_the_checkpoints_keys_and_values_on_the_layers_scales(kind):
    # the constants are HF's KV cache of the prefix alone (keys after the k norm and RoPE at the positions 0..P-1)
    # rounded on (s_k, s_v); and (s_k, s_v) are the real scales of the k and v the graph computes for the query:
    # dequantised, those match HF's keys and values of [prefix | x] at the query's positions in size (norm ratio
    # 0.990 - 1.017 measured, two seeds per family) and direction (relative error 0.018 - 0.094)
    model, graph, info, _, x = _setup(kind, prefix=PREFIX)
    with torch.no_grad():
        alone = model(torch.tensor([PREFIX]), use_cache=True).past_key_values
        joint = model(_joint(PREFIX, x), use_cache=True).past_key_values
    env, _ = graph.forward(x)
    for i, (op, (kp, vp), (sk, sv)) in enumerate(zip(_attention_ops(graph), info["prefix_kv"], info["prefix_scales"])):
        for c, real, s in ((kp, alone[i][0], sk), (vp, alone[i][1], sv)):
            real = real.transpose(1, 2).reshape(c.shape).double()
            assert torch.equal(c.to(torch.int64), (real / s).round().clamp(-127, 127).to(torch.int64))
        for got, s, ref in ((env[op.inputs[1]], sk, joint[i][0]), (env[op.inputs[2]], sv, joint[i][1])):
            ref = ref[:, :, len(PREFIX):].transpose(1, 2).reshape(got.shape).double()
            real = got.double() * s
            ratio, err = float(real.norm() / ref.norm()), float((real - ref).norm() / ref.norm())
            assert 0.95 <= ratio <= 1.05 and err <= 0.15, (i, ratio, err)
    assert all(max(c) < 1.2 for c in info["prefix_clip"])           # (a random model's prefix barely clips)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_prefixed_logits_and_blocks_track_the_float_model_on_prefix_and_query(kind):
    # the thresholds of test_the_logits_and_every_block_track_the_float_model, against HF on [prefix | x]
    model, graph, info, _, x = _setup(kind, prefix=PREFIX)
    mean, low, top1 = _scores(graph, info, model, x, PREFIX)
    assert mean >= 0.98 and low >= 0.9 and top1 >= 0.75, (mean, low, top1)
    blocks = rl.block_cosines(rl.residual_states(graph, info, x), rl.hf_residual_states(model, x, prefix=PREFIX))
    assert len(blocks) == 3 and min(b["min"] for b in blocks) >= 0.9, blocks
    _, scalar, s_info, _, _ = _setup(kind, "scalar", prefix=PREFIX)
    assert _scores(scalar, s_info, model, x, PREFIX)[0] >= 0.94


def test_the_residual_scale_ignores_the_prefix_positions():
    # a prefix token whose embedding is 40x larger (the windows lack it): the fp residual maximum behind s_res is the
    # windows' own positions' (after the prefix), not the larger one at the prefix's position
    model = copy.deepcopy(_setup("llama")[0])
    calib = _setup("llama")[3]
    tok = min(set(range(97)) - set(calib.reshape(-1).tolist()))
    with torch.no_grad():
        model.model.embed_tokens.weight[tok] *= 40
    top, pre = rl._hidden_max(model, calib, torch.tensor([[tok]]))
    states = [s for i in range(calib.shape[0]) for s in rl.hf_residual_states(model, calib[i:i + 1], prefix=(tok,))]
    assert top == max(float(s.abs().max()) for s in states) and pre > 1.5 * top
    _, info = rl.build_llama_from_hf(model, calib, prefix_tokens=(tok,), **SMOOTH)
    assert info["residual_real_max"] == top * 1.25 and info["prefix_hidden_max"] == pre


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_the_digest_binds_the_prefix_and_v0_rejects_a_prover_with_other_constants(kind):
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.verifier import Verifier

    model, graph, _, calib, x = _setup(kind, prefix=PREFIX)
    _, plain, _, _, _ = _setup(kind)
    other, _ = rl.build_llama_from_hf(model, calib, prefix_tokens=(5, 18), **SMOOTH)
    swapped, _ = rl.build_llama_from_hf(model, calib, prefix_tokens=PREFIX[::-1], **SMOOTH)
    assert len({graph.digest(), plain.digest(), other.digest(), swapped.digest()}) == 4
    # the digest changes with one constant alone: layer 0's prefix keys (values) negated, the same function otherwise
    assert _with_constant(graph, 0, "k_prefix", lambda c: c).digest() == graph.digest()
    params = params_for(20, len(graph.mat_ops))
    coms = commit_graph(graph, params.rate)
    v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    assert run_query(Prover(graph, commitments=coms), v, x[:1, :16], seed=1)["accepted"]
    for name in ("k_prefix", "v_prefix"):
        tampered = _with_constant(graph, 0, name, lambda c: -c)
        assert tampered.digest() != graph.digest()
        # a prover that computes with other prefix constants (same weights, same commitments) is rejected: the
        # verifier recomputes the attention with the graph's
        assert not run_query(Prover(tampered, commitments=coms), v, x[:1, :16], seed=1)["accepted"]


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@pytest.mark.parametrize("prune", [False, True])
def test_v0_and_v1_accept_prefixed_real_graphs_and_reject_a_tamper(kind, prune):
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.v1 import protocol as cp

    _, graph, _, _, x = _setup(kind, prune=prune, prefix=PREFIX)
    ids = x[:1, :16]
    targets = (graph.mat_ops[1].name, graph.mat_ops[len(graph.mat_ops) // 2].name, graph.mat_ops[-1].name)

    def tampers():
        for target in targets:
            def tamper(op, z, target=target):
                if op.name == target:
                    z = z.clone()
                    z.view(-1)[0] -= 3
                return z
            yield {"tamper": tamper}

    params = params_for(20, len(graph.mat_ops))
    coms = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=coms)
    kpre = Verifier(graph.public(), params, "Kpre", weights={o.name: (o.weight, o.bias) for o in graph.mat_ops})
    kpre.precompute(Challenger(seed=11))
    for v in (Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()}), kpre):
        assert all(run_query(prover, v, ids, seed=s)["accepted"] for s in range(2))
        assert not any(run_query(prover, v, ids, seed=1, forward_kwargs=f)["accepted"] for f in tampers())
    params = params_for(20, len(graph.mat_ops), cut=True)
    coms = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=coms)
    v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    out = cp.run_cut_query(prover, v, ids, seed=1)
    assert out["accepted"], out["rejected_at"]
    assert not any(cp.run_cut_query(prover, v, ids, seed=1, forward_kwargs=f)["accepted"] for f in tampers())


def test_prefixquant_selects_the_frequent_outlier_tokens_then_the_bos():
    # windows of 10 tokens, 3 blocks: the initial token is an outlier in every window from block 1 on, token 7 at
    # its first occurrence (three windows), the BOS (1) once (never counted: it ends the prefix anyway), token 9 in
    # one window of the last block. Outliers per window: block 0: 0, block 1: 2.0, block 2: 2.25 -> o = 3
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(10, 50, (4, 10), generator=g)
    ids[:3, 3], ids[0, 5], ids[1, 6] = 7, 9, 1
    maxes = [torch.rand(4, 10, generator=g) + 1 for _ in range(3)]
    for m in maxes[1:]:
        m[:, 0], m[:3, 3], m[1, 6] = 1000.0, 500.0, 400.0
    maxes[2][0, 5] = 200.0
    prefix, stats = rl._prefixquant_select(maxes, ids, 1, 64.0)
    assert prefix == [7, 9, 1] and stats["o"] == 3 and stats["outliers_per_block"] == [0.0, 2.0, 2.25]
    assert stats["outlier_tokens"] == [(7, 6), (9, 1)]
    maxes[2][0, 5] = 50.0                         # R ~ 33 < eta: not an outlier; o = 2
    assert rl._prefixquant_select(maxes, ids, 1, 64.0)[0] == [7, 1]
    # a random model has no massive activations: the prefix is the BOS alone (the configuration's, or given)
    model, _, _, calib, _ = _setup("llama")
    prefix, stats = rl.prefixquant_tokens(model, calib)
    assert prefix == [model.config.bos_token_id] == [1] and stats["o"] <= 1
    qwen = _setup("qwen3")[0]
    with pytest.raises(ValueError):
        rl.prefixquant_tokens(qwen, calib)        # Qwen3Config's default has no bos_token_id (the checkpoint's has)
    assert rl.prefixquant_tokens(qwen, calib, bos=96)[0] == [96]


# ------------------------------------------------------------------ the opt-in SwiGLU balance (glu_smooth)

@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_glu_smooth_is_exact_in_real_numbers_and_keeps_the_structure(kind):
    """``glu_smooth`` moves a per-channel factor from the product (through the up projection's rows) into the down
    projection's columns: the float model with the same rescaling gives the checkpoint's logits, the graph keeps the
    benchmark structure, tracks the float model at least as well as the plain smoothed graph, and v0 accepts it."""
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    model, base, b_info, calib, x = _setup(kind)
    graph, info = rl.build_llama_from_hf(model, calib, glu_smooth=0.5, **SMOOTH)
    g = info["glu_smooth_g"]
    assert len(g) == 2 and all(t is not None and t.min() >= 1 / 256 and t.max() <= 256 for t in g)
    assert not all(torch.allclose(t, torch.ones_like(t)) for t in g)
    for layer, gl in zip(model.model.layers, g):              # the box: the down projection's largest weight stays
        down = layer.mlp.down_proj.weight.double()
        assert float((down.abs().amax(0) * gl).max()) <= float(down.abs().max()) * (1 + 1e-9)
    shape = rl.config_of(model.config, "tiny")
    assert rl.matches_benchmark_graph(graph, shape) == []
    # the rescaling itself is exact in real numbers: apply it to a float copy and compare logits
    other = copy.deepcopy(model)
    with torch.no_grad():
        for layer, gl in zip(other.model.layers, g):
            layer.mlp.up_proj.weight.div_(gl.float()[:, None])
            layer.mlp.down_proj.weight.mul_(gl.float()[None, :])
        a, b = model(x).logits, other(x).logits
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    assert _scores(graph, info, model, x)[0] >= 0.98
    params = params_for(20, len(graph.mat_ops))
    v = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops})
    v.precompute(Challenger(seed=3))
    assert run_query(Prover(graph), v, x[:1, :16], seed=0)["accepted"]
    off, _ = rl.build_llama_from_hf(model, calib, **SMOOTH)
    assert off.digest() == base.digest() != graph.digest()           # off by default: the graph without the balance


def test_glu_smooth_box_keeps_both_weight_scales():
    """``_glu_smooth``'s box on a dead channel (gate about 0, so its product is about 0 and the plain balance gives it
    a factor near 0): up's largest row as quantised (input scale folded) and the down projection's largest column do
    not grow, so no row takes over up's int8 scale."""
    gen = torch.Generator().manual_seed(5)
    d, ff = 16, 24
    h = torch.randint(-127, 128, (64, d), generator=gen).to(torch.int64)
    s_in = torch.rand(d, generator=gen, dtype=torch.float64) * 1e-2 + 1e-3
    wg = torch.randn(ff, d, generator=gen) * 0.1
    wu = torch.randn(ff, d, generator=gen) * 0.1
    wd = torch.randn(d, ff, generator=gen) * 0.1
    wg[3] *= 1e-6                                          # channel 3: gate ~ 0, so silu(gate) up ~ 0
    wg[7] *= 30                                            # channel 7: a large product
    g = rl._glu_smooth(h, s_in, wg, wu, wd, 0.5)
    u = (wu.double() * s_in[None, :]).abs().amax(1)
    dc = wd.double().abs().amax(0)
    assert float((u / g).max()) <= float(u.max()) * (1 + 1e-9)
    assert float((dc * g).max()) <= float(dc.max()) * (1 + 1e-9)
    assert abs(float(g[3]) - float(u[3] / u.max())) < 1e-12  # the box binds the dead channel...
    assert float(g[7]) > 1                                 # ...and the balance still shrinks the large product
