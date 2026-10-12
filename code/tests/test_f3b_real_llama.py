"""F3b's plumbing on a tiny random Llama (no download): ``v1_bytes.real_graph``'s Llama branch is
``build_llama_from_hf(**MAIN_RECIPE)`` with the calibration's prefix, keeps the benchmark structure, and its graph serves
``gen_bench.py --real``'s generated responses: ``with_lm_positions``, greedy and sampled tokens (the sampler's weight
from the LM head's real logit scale), proved and verified in modes C and Kpre, and a wrong token rejected."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.fullcheck import protocol as proto  # noqa: E402
from pvi.fullcheck import real_llama as rl  # noqa: E402
from pvi.fullcheck.token_sampler import TokenSampler, sample_tokens  # noqa: E402
from pvi.fullcheck.transformer import greedy_tokens, with_lm_positions  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
PREFIX = [5, 17]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, CODE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.15)
    model = transformers.LlamaForCausalLM(cfg).eval()
    calib = torch.randint(0, 97, (2, 32), generator=torch.Generator().manual_seed(1))
    v1b = _load("v1_bytes", "experiments/8_v1/v1_bytes.py")
    graph, info = v1b.real_graph(model, SimpleNamespace(ids=calib, prefix=PREFIX), None, True)
    return model, calib, graph, info


def test_the_llama_branch_is_the_main_recipe_with_the_prefix(built):
    model, calib, graph, info = built
    direct, _ = rl.build_llama_from_hf(model, calib, prune_last=True, prefix_tokens=PREFIX, **rl.MAIN_RECIPE)
    assert graph.digest() == direct.digest()
    assert info["glu_smooth_g"][0] is not None and info["prefix_tokens"] == PREFIX
    assert rl.matches_benchmark_graph(graph, rl.config_of(model.config, "tiny"), prune_last=True) == []
    assert info["logit_scale"] > 0


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
        coms = proto.commit_graph(gen, 4, policy="R8c", cache=False)
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
