"""Checks for the strong-GPU plan: lean forward, lean derive, verifier on another device."""

import pytest
import torch

from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
from pvi.fullcheck.transformer import DecoderConfig, build_decoder

_TINY = {
    "gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                           pos="rope", bias=False, tied=False),
    "qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                          pos="rope", rope_theta=1e6, bias=False, qk_norm=True),
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("family", list(_TINY))
def test_lean_forward_gives_identical_claims(family):
    g = build_decoder(_TINY[family], calib_tokens=12, seed=3)
    x = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(5))
    env_full, a = g.forward(x)
    env_lean, b = g.forward(x, free=True, claims_device="cpu")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert len(env_lean) < len(env_full) and g.output_name in env_lean


@pytest.mark.parametrize("family", list(_TINY))
@pytest.mark.parametrize("mode", ["C", "K", "Kpre"])
@pytest.mark.parametrize("vdev", DEVICES)
def test_lean_prover_and_device_verifier(family, mode, vdev):
    g = build_decoder(_TINY[family], calib_tokens=12, seed=3)
    x = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(5))
    p = params_for(40, len(g.mat_ops))
    coms = commit_graph(g, p.rate) if mode == "C" else None
    pdev = "cuda" if torch.cuda.is_available() else "cpu"
    prover = Prover(g, device=pdev, commitments=coms, lean=True)
    if mode == "C":
        v = Verifier(g.public(), p, "C", publics={k: c.public for k, c in coms.items()}, device=vdev, lean=True)
    else:
        v = Verifier(g.public(), p, mode, weights={o.name: (o.weight, o.bias) for o in g.mat_ops},
                     device=vdev, lean=True)
        if mode == "Kpre":
            v.precompute(Challenger())
    res = run_query(prover, v, x, seed=1)
    assert res["accepted"], res["rejected_at"]
    victim = g.mat_ops[len(g.mat_ops) // 2].name

    def tamper(o, z):
        if o.name == victim:
            z = z.clone()
            z.view(-1)[0] += 1
        return z

    bad = run_query(prover, v, x, seed=2, forward_kwargs={"tamper": tamper})
    assert not bad["accepted"] and bad["rejected_at"] == "freivalds"
    if vdev != "cpu":
        assert "verify_upload" in res["timings"]


def test_added_configs_have_published_parameter_counts():
    from pvi.fullcheck.transformer import CONFIGS, decoder_param_count
    for name, published in [("opt-350m", 331e6), ("opt-2.7b", 2.7e9), ("opt-13b", 13e9),
                            ("opt-30b", 30e9), ("opt-66b", 66e9), ("llama2-70b", 69e9)]:
        n = decoder_param_count(CONFIGS[name])
        assert abs(n - published) / published < 0.06, (name, n)
