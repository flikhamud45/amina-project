"""The real-weight OPT loader (plan D.3): a tiny random OPT, so no download is needed."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.fullcheck.graph import CheapOp, MatOp          # noqa: E402
from pvi.fullcheck.real_weights import build_opt_from_hf, real_logits   # noqa: E402


@pytest.fixture(scope="module")
def tiny_opt():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                 num_attention_heads=4, max_position_embeddings=64,
                                 word_embed_proj_dim=32, do_layer_norm_before=True)
    model = transformers.OPTForCausalLM(cfg).eval()
    ids = torch.randint(3, 97, (1, 24))
    graph, info = build_opt_from_hf(model, ids, norm_gain=4)
    return model, graph, info, ids


def test_all_positions_equals_one_query_per_prefix(tiny_opt):
    model, graph, info, ids = tiny_opt
    every = real_logits(graph, info, ids, all_positions=True)[0]
    for t in (1, 7, ids.shape[1]):
        last = real_logits(graph, info, ids[:, :t])[0, -1]
        assert torch.equal(every[t - 1], last)


def test_graph_is_integer_and_tracks_the_float_model(tiny_opt):
    model, graph, info, ids = tiny_opt
    assert all(m.weight.dtype == torch.int8 for m in graph.ops if isinstance(m, MatOp))
    assert sum(isinstance(o, CheapOp) and o.note == "last position" for o in graph.ops) == 1
    with torch.no_grad():
        ref = model(ids).logits[0].double()
    got = real_logits(graph, info, ids, all_positions=True)[0]
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    assert cos.min() > 0.9, cos


def test_defence_accepts_honest_queries_on_real_weights(tiny_opt):
    from pvi.fullcheck.protocol import Prover, Verifier, commit_graph, params_for, run_query

    _, graph, _, ids = tiny_opt
    params = params_for(20, len(graph.mat_ops))
    commitments = commit_graph(graph, params.rate)        # also checks the claim range bound
    prover = Prover(graph, commitments=commitments)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in commitments.items()})
    for seed in range(2):
        assert run_query(prover, verifier, ids[:, :16], seed=seed)["accepted"]
