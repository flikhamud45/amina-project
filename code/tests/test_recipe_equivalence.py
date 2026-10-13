"""F2a's recipe search runs on its own builder (``experiments/9_quality/opt_builder.build_opt``); the certified model is
the paper builder's (``real_weights.build_opt_from_hf``).  At the recipe F2a found ("smooth-max": SmoothQuant through
the norm gains with each channel's maximum, no clipping, 11-bit attention probabilities), the two give the same int8
weights and the same integer logits, so the search's numbers hold for the certified model and the switch is a change
of the paper builder's parameters alone (``smooth=alpha, smooth_pct=100, pct=100, pct_att=100``)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.models import hf_opt as rw                                   # noqa: E402

CODE = Path(__file__).resolve().parents[1]


def _opt_builder():
    sys.path.insert(0, str(CODE / "experiments" / "9_quality"))
    spec = importlib.util.spec_from_file_location("opt_builder", CODE / "experiments" / "9_quality" / "opt_builder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("alpha", [0.5, 0.6])
def test_smooth_max_is_the_same_graph_in_both_builders(alpha):
    ob = _opt_builder()
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                 num_attention_heads=4, max_position_embeddings=64, word_embed_proj_dim=32,
                                 do_layer_norm_before=True)
    model = transformers.OPTForCausalLM(cfg).eval()
    calib = torch.randint(3, 97, (2, 48))
    ids = torch.randint(3, 97, (1, 40))
    with torch.no_grad():
        gp, ip = rw.build_opt_from_hf(model, calib, norm_gain=4, smooth=alpha, smooth_pct=100.0, pct=100.0,
                                      pct_att=100.0)
        go, io = ob.build_opt(model, calib, norm_mode="smooth", norm_gain=4.0, alpha=alpha, smooth_pct=100.0,
                              pct=100.0, pct_att=100.0, prob_bits=11)
        assert ip["logit_scale"] == io["logit_scale"]
        assert [o.name for o in gp.mat_ops] == [o.name for o in go.mat_ops]
        assert all(torch.equal(a.weight, b.weight) for a, b in zip(gp.mat_ops, go.mat_ops))
        assert torch.equal(rw.real_logits(gp, ip, ids, all_positions=True), rw.real_logits(go, io, ids,
                                                                                           all_positions=True))
