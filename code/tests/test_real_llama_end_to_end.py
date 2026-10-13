"""A real Llama-family checkpoint's graph end to end, on a tiny random Llama (no download):
``experiments.common.checkpoints.real_graph``'s Llama branch is ``build_llama_from_hf(**MAIN_RECIPE)`` with the
calibration's prefix, keeps the benchmark structure (``v1_bytes``' shape check), and its graph serves
``generation/gen_bench.py --real``'s generated responses: ``with_lm_positions``, greedy and sampled tokens (the
sampler's weight from the LM head's real logit scale), proved and verified in modes C and Kpre, and a wrong token
rejected."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

import pvi.v0 as proto  # noqa: E402
from experiments.common import checkpoints  # noqa: E402
from experiments.generation import gen_bench as gb  # noqa: E402
from experiments.v1 import v1_bytes as v1b  # noqa: E402
from pvi.models import hf_llama as rl  # noqa: E402
from pvi.models.decoder import greedy_tokens, with_lm_positions  # noqa: E402
from pvi.v0.token_sampler import TokenSampler, sample_tokens  # noqa: E402

PREFIX = [5, 17]


@pytest.fixture(scope="module")
def built():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.15)
    model = transformers.LlamaForCausalLM(cfg).eval()
    calib = torch.randint(0, 97, (2, 32), generator=torch.Generator().manual_seed(1))
    graph, info = checkpoints.real_graph(model, SimpleNamespace(ids=calib, prefix=PREFIX), True)
    return model, calib, graph, info


def test_the_llama_branch_is_the_main_recipe_with_the_prefix(built):
    model, calib, graph, info = built
    direct, _ = rl.build_llama_from_hf(model, calib, prune_last=True, prefix_tokens=PREFIX, **rl.MAIN_RECIPE)
    assert graph.digest() == direct.digest()
    assert info["glu_smooth_g"][0] is not None and info["prefix_tokens"] == PREFIX
    assert rl.matches_benchmark_graph(graph, rl.config_of(model.config, "tiny"), prune_last=True) == []
    assert info["logit_scale"] > 0


def test_the_shape_check_takes_the_llama_head_as_built(built):
    """v1_bytes' shape check: a real Llama head has the benchmark's row length (no folded bias, unlike OPT's)."""
    model, calib, graph, info = built
    cfg = rl.config_of(model.config, "tiny")
    res = v1b.shape_check(graph, cfg, cfg.n_layers, True, True, calib[:1, :16])
    assert res["problems"] == []


def test_forged_logits_of_a_sampled_response_are_caught_by_the_weight_check(built):
    """gen_bench's forged_logits tamper with the real logit scale's sampler: the margin is sized from that sampler's
    weight ``a`` (not ``--sampler-a``), so the forgery makes the rule choose the wrong token and only Freivalds'
    check catches it (with a margin sized from ``--sampler-a = 4096`` the noise can still outweigh the forged logit,
    and the token check catches it instead)."""
    model, calib, graph, info = built
    sampler = TokenSampler.for_temperature(b"\x09" * 32, 1.0, float(info["logit_scale"]))
    assert gb.forge_margin("sample", sampler) * sampler.a >= 1 << 21
    assert gb.forge_margin("greedy", None) == 1
    p, g = 10, 4
    prompt = torch.randint(0, 97, (1, p), generator=torch.Generator().manual_seed(5))
    x = sample_tokens(graph, prompt, g - 1, sampler)
    gen = with_lm_positions(graph, g)
    params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=True)
    v = proto.Verifier(gen.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in gen.mat_ops},
                       lean=True, lookups=True)
    v.precompute(proto.Challenger())
    prover = proto.Prover(gen, lean=True)
    i_bad = (g - 1) // 2
    bad = gb.wrong_token(graph, x, p, i_bad, "sample", sampler, "cpu")
    want, head, delta = int(bad[0, p + i_bad]), gen.mat_ops[-1].name, gb.forge_margin("sample", sampler)

    def forge(op, z):
        if op.name == head:
            z = z.clone()
            z[want, i_bad] = z[:, i_bad].max() + delta
        return z

    res = proto.run_query(prover, v, bad, wire=True, sampler=sampler, forward_kwargs={"tamper": forge})
    assert not res["accepted"] and res["rejected_at"] == "freivalds"


@pytest.mark.parametrize("rule", ["greedy", "sample"])
@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_generated_responses_prove_and_verify(built, rule, mode):
    model, calib, graph, info = built
    prompt = torch.randint(0, 97, (1, 10), generator=torch.Generator().manual_seed(3))
    g = 4
    sampler = TokenSampler.for_temperature(b"\x07" * 32, 1.0, float(info["logit_scale"])) if rule == "sample" else None
    x = (sample_tokens(graph, prompt, g - 1, sampler) if sampler is not None else greedy_tokens(graph, prompt, g - 1))
    assert x.shape == (1, 10 + g - 1)
    gen = with_lm_positions(graph, g)
    if mode == "C":
        coms = proto.commit_graph(gen, 4, policy="R8c")
        params = proto.params_for(40, len(gen.mat_ops), plan=coms.plan, fiat_shamir=True)
        v = proto.Verifier(gen.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                           tables=coms.table_publics, lean=True)
        prover = proto.Prover(gen, commitments=coms, lean=True)
    else:
        params = proto.params_for(40, len(gen.mat_ops), fiat_shamir=True)
        v = proto.Verifier(gen.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in gen.mat_ops},
                           lean=True, lookups=True)
        v.precompute(proto.Challenger())
        prover = proto.Prover(gen, lean=True)
    assert proto.run_query(prover, v, x, wire=True, sampler=sampler)["accepted"]
    bad = x.clone()
    bad[0, 11] = (int(bad[0, 11]) + 1) % 97                 # a generated token replaced (the rest kept)
    res = proto.run_query(prover, v, bad, wire=True, sampler=sampler)
    assert not res["accepted"]
