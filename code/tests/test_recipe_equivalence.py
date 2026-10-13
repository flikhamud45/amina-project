"""The OPT recipes and the extended version's variants of ``pvi.models.hf_opt.build_opt_from_hf`` on a tiny OPT whose
weights are drawn by Python's ``random`` (independent of the transformers version's initialisation).

* Stored expectations: the certified recipe, the earlier recipes (norm gain 4; alpha 0.55 with percentile
  statistics), the per-row power-of-two weight scales (``row_pow2``) and cross-layer equalisation (``cle_mlp``) build
  the integer graphs whose fingerprints are stored here -- the graph digest, every weight and bias, the logit scale
  and the integer logits at every position of a fixed query.  The variants' values were checked equal, integer for
  integer, to those of the recipe-search builder they come from (the source of the extended version's per-row and
  equalisation numbers) before it was merged into ``hf_opt``.
* Against the certified path: the variants' options at their defaults give the certified graph; a per-row graph keeps
  the benchmark graph's structure, its vector multipliers take at most ``row_pow2`` values a power of two apart, and
  protocol v0 accepts it and rejects a tampered claim; equalisation changes fc1 and fc2 only.
"""

import hashlib
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.core.graph import CheapOp  # noqa: E402
from pvi.models import hf_opt  # noqa: E402
from pvi.models.decoder import DecoderConfig  # noqa: E402

LEGACY = dict(norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
CASES = {
    "certified": dict(hf_opt.CERTIFIED_RECIPE),
    "certified-p8": dict(hf_opt.CERTIFIED_RECIPE, prob_max=255),
    "certified-pruned": dict(hf_opt.CERTIFIED_RECIPE, prune_last=True),
    "gain4-p8": dict(norm_gain=4, prob_max=255),
    "legacy-p8": dict(LEGACY, prob_max=255),
    "legacy": dict(LEGACY),
    "rows4-legacy-p8": dict(LEGACY, row_pow2=4, prob_max=255),
    "rows4-certified": dict(hf_opt.CERTIFIED_RECIPE, row_pow2=4),
    "cle0r2-legacy-p8": dict(LEGACY, cle_mlp=0.0, cle_ratio=2.0, prob_max=255),
    "cle0.5r4-certified": dict(hf_opt.CERTIFIED_RECIPE, cle_mlp=0.5, cle_ratio=4.0),
}
STORED = {
    "certified": "7393d1e868ea5a4a",
    "certified-p8": "36760eed322a80f1",
    "certified-pruned": "f64507119fe45391",
    "cle0.5r4-certified": "3e53f2478b33762e",
    "cle0r2-legacy-p8": "5cf71d1dcc8cb3fb",
    "gain4-p8": "14c4789f4a704848",
    "legacy": "f897e8fa946682b1",
    "legacy-p8": "a2b077e971d86ee0",
    "rows4-certified": "53db4521701d76eb",
    "rows4-legacy-p8": "09ab5a532b9b553d",
}
"""``fingerprint`` of each case's graph (graph digest format ``pvi/graph/v2``)."""


def tiny_opt():
    """A 2-block OPT (d = 32, 4 heads, ffn 64, vocabulary 97) with weights from ``random.Random(0)``: LayerNorm gains
    in [0.5, 1.5], and Gaussian weights of std 0.3 (biases 0.05), with a few large channels in the token embedding,
    as OPT's outlier features."""
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64, num_attention_heads=4,
                                 max_position_embeddings=64, word_embed_proj_dim=32, do_layer_norm_before=True)
    model = transformers.OPTForCausalLM(cfg).eval()
    rng = random.Random(0)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            n = p.numel()
            if "layer_norm" in name and name.endswith("weight"):
                vals = [rng.uniform(0.5, 1.5) for _ in range(n)]
            else:
                std = 0.05 if name.endswith("bias") else 0.3
                vals = [rng.gauss(0.0, std) for _ in range(n)]
            p.copy_(torch.tensor(vals, dtype=torch.float64).reshape(p.shape))
        model.model.decoder.embed_tokens.weight[:, [3, 17]] *= 6.0
    return model


def _ids(seed, shape):
    rng = random.Random(seed)
    return torch.tensor([[rng.randrange(3, 97) for _ in range(shape[1])] for _ in range(shape[0])])


@pytest.fixture(scope="module")
def setup():
    return tiny_opt(), _ids(1, (2, 48)), _ids(2, (1, 30))


def fingerprint(graph, info, x) -> str:
    """SHA-256 (first 16 hex digits) of ``graph``'s digest, its weights and biases, the logit scale and its integer
    logits at every position of ``x``."""
    h = hashlib.sha256(graph.digest())
    for op in graph.mat_ops:
        h.update(op.name.encode() + op.weight.contiguous().numpy().tobytes())
        if op.bias is not None:
            h.update(op.bias.contiguous().numpy().tobytes())
    h.update(repr(info["logit_scale"]).encode())
    env, _ = hf_opt.every_position(graph).forward(x)
    h.update(env[graph.output_name].to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _build(setup, **kw):
    model, calib, _ = setup
    with torch.no_grad():
        return hf_opt.build_opt_from_hf(model, calib, **kw)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_recipes_build_the_stored_graphs(setup, case):
    graph, info = _build(setup, **CASES[case])
    assert fingerprint(graph, info, setup[2]) == STORED[case]


def test_the_variants_off_are_the_certified_graph(setup):
    a, ia = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    b, ib = _build(setup, **hf_opt.CERTIFIED_RECIPE, row_pow2=0, cle_mlp=None)
    x = setup[2]
    assert fingerprint(a, ia, x) == fingerprint(b, ib, x)


def _config(model) -> DecoderConfig:
    c = model.config
    return DecoderConfig("tiny-opt", c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_attention_heads,
                         c.hidden_size // c.num_attention_heads, c.ffn_dim, c.vocab_size, mlp="relu",
                         max_pos=c.max_position_embeddings)


@pytest.mark.parametrize("buckets", [2, 4])
def test_per_row_scales_keep_the_structure_and_v0_accepts_them(setup, buckets):
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.verifier import Verifier

    model, _, x = setup
    graph, info = _build(setup, **hf_opt.CERTIFIED_RECIPE, row_pow2=buckets)
    base, _ = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    assert hf_opt.matches_benchmark_graph(graph, _config(model)) == hf_opt.matches_benchmark_graph(base, _config(model))
    vec = [o for o in graph.ops if isinstance(o, CheapOp) and "vec_requant" in o.params]
    assert len(vec) == 6 * model.config.num_hidden_layers          # q, k, v, out-proj, fc1, fc2 per block
    for o in vec:
        m = o.fn.consts["mult"]
        assert m.numel() > 1 and torch.unique(m).numel() <= buckets
        ratios = m.double() / m.double().min()                       # power-of-two row scales (up to rounding)
        assert torch.allclose(ratios, torch.pow(2.0, torch.round(torch.log2(ratios))), rtol=1e-6)
    for a, b in zip(graph.mat_ops[:2], base.mat_ops[:2]):            # the embeddings keep one scale
        assert torch.equal(a.weight, b.weight)
    assert not any(isinstance(o, CheapOp) and "vec_requant" in o.params for o in base.ops)
    params = params_for(20, len(graph.mat_ops))
    coms = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=coms)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    assert run_query(prover, verifier, x[:, :16], seed=0)["accepted"]
    victim = graph.mat_ops[4].name

    def tamper(op, z):
        if op.name == victim:
            z = z.clone()
            z[0, 0] += 1
        return z

    assert not run_query(prover, verifier, x[:, :16], seed=1, forward_kwargs={"tamper": tamper})["accepted"]


def test_equalisation_changes_fc1_and_fc2_only(setup):
    graph, _ = _build(setup, **hf_opt.CERTIFIED_RECIPE, cle_mlp=0.0, cle_ratio=2.0)
    base, _ = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    mats, ref = graph.mat_ops, base.mat_ops
    for a, b in zip(mats[:6], ref[:6]):        # the embeddings and block 0's q, k, v and out-projection
        assert torch.equal(a.weight, b.weight)
    assert not torch.equal(mats[6].weight, ref[6].weight)              # block 0's fc1
    assert [o.name for o in mats] == [o.name for o in ref]


def test_probability_mass_runs_the_graph(setup):
    graph, info = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    lean = hf_opt.every_position(graph)
    x = setup[2]
    out, mass = hf_opt.probability_mass(lean, x)
    env, _ = lean.forward(x)
    assert torch.equal(out, env[graph.output_name])
    assert len(mass) == sum(isinstance(o, CheapOp) and o.note == "attention" for o in graph.ops) == 2
    assert all(0.99 < m <= 1.01 for m in mass)


# ------------------------------------------------------------------ the scripts on the tiny model
def _ppl_args(**kw):
    import argparse
    base = dict(model="facebook/opt-1.3b", graphs=["scalar", "smooth"], recipe="certified", alphas=None, gain=None,
                gains=None, row_buckets=4, cle_mlp=None, cle_ratio=32.0, smooth_opts="{}", scalar_opts="{}",
                pbits=[8, 11], split="test", calibration="train", seq=48, windows=0, calib_windows=2, draws=[0],
                mass_windows=1, device="cpu", ref_device=None, ref_cache=None, tag="t", out=None)
    return argparse.Namespace(**dict(base, **kw))


def test_ppl_opt_cells_are_the_paper_graphs_by_default():
    from experiments.quality import ppl_opt
    got = ppl_opt.cells(_ppl_args())
    assert [c["cell"] for c in got] == ["scalar-g3-p8", "scalar-g3-p11", "smooth-certified-a0.6-p8",
                                        "smooth-certified-a0.6-p11"]
    assert got[3]["kw"] == dict(hf_opt.CERTIFIED_RECIPE, prob_max=2047) and got[1]["kw"] == dict(norm_gain=3.0,
                                                                                                    prob_max=2047)
    legacy = ppl_opt.cells(_ppl_args(graphs=["rows"], recipe="legacy", pbits=[8], cle_mlp=0.0, cle_ratio=2.0))
    assert [c["cell"] for c in legacy] == ["rows4-legacy-a0.55-cle0r2-p8"]
    assert legacy[0]["kw"] == dict(LEGACY, row_pow2=4, cle_mlp=0.0, cle_ratio=2.0, prob_max=255)


def test_ppl_opt_check_passes_on_the_tiny_model(setup):
    from experiments.quality import ppl_opt
    model, calib, x = setup
    report = ppl_opt.check(model, x, calib)
    assert report["ok"], report
    certified, _ = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    assert report["certified-p11"]["digest"] == certified.digest().hex()


def test_ppl_opt_runs_every_cell_on_the_tiny_model(setup, monkeypatch, capsys):
    import json
    from experiments.quality import ppl_opt
    model, calib, x = setup
    wins = [_ids(10 + i, (1, 48)) for i in range(3)]
    monkeypatch.setattr(ppl_opt, "load_data", lambda tok, args: (wins, {0: calib}, "random windows"))
    ppl_opt.run(model, None, _ppl_args(graphs=["scalar", "smooth", "rows"]))
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert [r["cell"] for r in rows] == ["fp32", "scalar-g3-p8", "scalar-g3-p11", "smooth-certified-a0.6-p8",
                                         "smooth-certified-a0.6-p11", "rows4-certified-a0.6-p8",
                                         "rows4-certified-a0.6-p11"]
    cert = rows[4]
    graph, info = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    assert cert["digest"] == graph.digest().hex() and cert["windows"] == 3 and len(cert["mass_per_layer"]) == 2
    nll = sum(ppl_opt.nll_argmax(hf_opt.real_logits(graph, info, w, all_positions=True), 1.0, w)[0] for w in wins)
    assert abs(cert["ppl"] - __import__("math").exp(nll / (3 * 47))) < 1e-9 and 0 < cert["top1"] <= 1


def test_digest_check_prints_the_certified_digest_and_roots(setup, monkeypatch, capsys):
    import json
    import sys
    from experiments.common import checkpoints
    from experiments.quality import digest_check
    model, calib, _ = setup
    monkeypatch.setattr(checkpoints, "load_opt", lambda name: (model, None))
    monkeypatch.setattr(checkpoints, "certified_calibration", lambda tok: calib)
    monkeypatch.setattr(sys, "argv", ["digest_check", "--threads", "2", "--commit"])
    digest_check.main()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    graph, _ = _build(setup, **hf_opt.CERTIFIED_RECIPE)
    assert out["digest"] == graph.digest().hex() and out["policy"] == "cnn18c" and out["n_roots"] > 0
