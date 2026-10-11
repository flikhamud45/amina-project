"""F2c's zero-shot harness (experiments/9_quality/zeroshot.py) without downloads: the harness's task formats, the
encoding of a request, the scoring and its metrics, and right padding on tiny integer OPTs (both builders), which
must give exactly the logits of each request run alone."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "experiments" / "9_quality"))
_spec = importlib.util.spec_from_file_location("zeroshot", CODE / "experiments" / "9_quality" / "zeroshot.py")
zs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zs)


class _CharTok:
    """One id per character (ord), no special tokens."""

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return self._Out([ord(c) for c in text])


def test_hellaswag_preprocess_matches_the_harness():
    assert zs.hellaswag_preprocess(" How to cook [title] Boil water. [step] Add salt.  ") == \
        "How to cook. Boil water. Add salt."
    assert zs.hellaswag_preprocess("a  b") == "a b"


def test_task_formats():
    lam = zs.make_docs("lambada_openai", [{"text": "the cat sat on the mat"}])[0]
    assert lam["requests"] == [("the cat sat on the", " mat")] and lam["choices"] is None
    hs = zs.make_docs("hellaswag", [{"activity_label": "Baking", "ctx_a": "She mixes flour.", "ctx_b": "then she",
                                     "endings": ["bakes it.", "eats [title] it."], "label": "1"}])[0]
    assert hs["requests"][0] == ("Baking: She mixes flour. Then she", " bakes it.")
    assert hs["requests"][1][1] == " eats. it." and hs["gold"] == 1 and hs["choices"] == ["bakes it.", "eats. it."]
    pq = zs.make_docs("piqa", [{"goal": "Open a jar", "sol1": "twist", "sol2": "pull", "label": 1}])[0]
    assert pq["requests"] == [("Question: Open a jar\nAnswer:", " twist"), ("Question: Open a jar\nAnswer:", " pull")]
    assert pq["gold"] == 1
    arc = zs.make_docs("arc_easy", [{"question": "2+2?", "choices": {"text": ["3", "4"], "label": ["A", "B"]},
                                     "answerKey": "B"}])[0]
    assert arc["requests"][1] == ("Question: 2+2?\nAnswer:", " 4") and arc["gold"] == 1
    wg = zs.make_docs("winogrande", [{"sentence": "Tom gave Bob a gift because _ was kind.", "option1": "Tom",
                                      "option2": "Bob", "answer": "1"}])[0]
    assert wg["requests"] == [("Tom gave Bob a gift because Tom", " was kind."),
                              ("Tom gave Bob a gift because Bob", " was kind.")]
    assert wg["gold"] == 0 and wg["choices"] is None


def test_hellaswag_subset_is_fixed_and_sorted():
    docs = [{"i": i} for i in range(50)]
    a = zs.select("hellaswag", docs, 10, 0)
    assert a == zs.select("hellaswag", docs, 10, 0) and len(a) == 10
    assert [d["i"] for d in a] == sorted(d["i"] for d in a)
    assert zs.select("piqa", docs, 10, 3) == docs[:3]


def test_encode_pair_moves_trailing_spaces():
    tok = _CharTok()
    ctx, cont = zs.encode_pair(tok, "Answer: ", "yes")
    assert ctx == [ord(c) for c in "Answer:"] and cont == [ord(c) for c in " yes"]
    ctx, cont = zs.encode_pair(tok, "ab", " c")
    assert ctx == [97, 98] and cont == [32, 99]


def _bigram_fn(table):
    def fn(ids):
        return table[ids], 0.5
    return fn


def test_score_is_independent_of_batching_and_padding():
    torch.manual_seed(0)
    table = torch.randn(40, 40, dtype=torch.float64)
    enc = [([1, 2, 3], [4, 5]), ([7], [8]), ([9, 10, 11, 12, 13], [14]), ([2, 2], [3, 3, 3])]
    batched = zs.score(enc, _bigram_fn(table), budget=64, pad_id=0)
    alone = [zs.score([e], _bigram_fn(table), budget=64, pad_id=0)[0] for e in enc]
    assert batched == alone
    c, k = enc[0]
    lp = (table * 0.5).log_softmax(-1)
    want = float(lp[3, 4] + lp[4, 5])           # position t predicts token t + 1
    assert math.isclose(batched[0][0], want, rel_tol=0, abs_tol=1e-12)
    assert batched[0][1] == (int(lp[3].argmax()) == 4 and int(lp[4].argmax()) == 5)


def test_metrics_acc_and_acc_norm():
    docs = [{"requests": [("q", " a"), ("q", " bbbb")], "gold": 1, "choices": ["a", "bbbb"]},
            {"requests": [("q", " a"), ("q", " b")], "gold": 0, "choices": ["a", "b"]}]
    res = [(-1.0, False), (-2.0, False), (-3.0, False), (-1.0, False)]
    m = zs.metrics("piqa", docs, res)
    assert m["correct"] == [0, 0] and m["correct_norm"] == [1, 0] and m["acc_norm"] == 0.5
    lam = zs.metrics("lambada_openai", [{"requests": [("x", " y")], "gold": 0, "choices": None}] * 2,
                     [(-1.0, True), (-3.0, False)])
    assert lam["acc"] == 0.5 and math.isclose(lam["ppl"], math.exp(2.0))


def test_paired_ci():
    assert zs.paired_ci([1, 0, 1], [1, 0, 1]) == [0.0, 0.0]
    lo, hi = zs.paired_ci([1] * 20 + [0] * 20, [0] * 40)
    assert 0.0 < lo < 0.5 < hi <= 1.0


def _tiny_opt():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                 num_attention_heads=4, max_position_embeddings=64,
                                 word_embed_proj_dim=32, do_layer_norm_before=True)
    return transformers.OPTForCausalLM(cfg).eval()


@pytest.mark.parametrize("builder", ["paper", "f2a"])
def test_right_padding_is_exact_on_the_integer_graph(builder):
    from pvi.fullcheck import real_weights as rw
    model = _tiny_opt()
    calib = torch.randint(3, 97, (2, 32))
    opts = dict(norm_gain=4) if builder == "paper" else dict(zs.F2A_BEST)
    with torch.no_grad():
        graph, info = zs.build_int(model, calib if builder == "f2a" else calib[:1], builder, opts)
    fn = zs.int_logits_fn(graph, info, "cpu")
    g = torch.Generator().manual_seed(1)
    enc = [(torch.randint(3, 97, (n,), generator=g).tolist(), torch.randint(3, 97, (m,), generator=g).tolist())
           for n, m in ((5, 3), (12, 1), (2, 4), (20, 2), (7, 7))]
    batched = zs.score(enc, fn, budget=128, pad_id=1)
    alone = [zs.score([e], fn, budget=128, pad_id=1)[0] for e in enc]
    assert batched == alone
    ids = torch.tensor([(enc[3][0] + enc[3][1])[:-1]])
    ours, scale = fn(ids)
    assert torch.equal(ours.double() * scale, rw.real_logits(graph, info, ids, all_positions=True))
    with torch.no_grad():
        ref = zs.score(enc, zs.float_logits_fn(model, "cpu"), budget=128, pad_id=1)
    assert all(abs(a[0] - b[0]) < 2.0 for a, b in zip(batched, ref))     # the integer model tracks the float one


def test_split_opt_gives_the_one_device_logits():
    """``split_opt`` over two devices (here both the CPU) gives the unsplit model's logits; the head is untied."""
    import copy
    model = _tiny_opt()
    ids = torch.randint(3, 97, (2, 12), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = model(ids).logits
        split = zs.split_opt(copy.deepcopy(model), ["cpu", "cpu"])
        got = split(ids).logits
    assert torch.equal(got, want)
    assert split.lm_head.weight.data_ptr() != split.model.decoder.embed_tokens.weight.data_ptr()


def test_pair_matches_a_joint_run(tmp_path, capsys):
    rows_ref = {"task": "piqa", "variant": "ref", "n": 3, "acc": 2 / 3, "correct": [1, 1, 0], "pick": [0, 1, 1],
                "acc_norm": 1 / 3, "correct_norm": [1, 0, 0]}
    rows_int = {"task": "piqa", "variant": "int", "n": 3, "acc": 1 / 3, "correct": [1, 0, 0], "pick": [0, 0, 1],
                "acc_norm": 1 / 3, "correct_norm": [1, 0, 0]}
    (tmp_path / "r.jsonl").write_text(__import__("json").dumps(rows_ref) + "\n")
    (tmp_path / "i.jsonl").write_text(__import__("json").dumps(rows_int) + "\n")
    zs.pair(str(tmp_path / "r.jsonl"), str(tmp_path / "i.jsonl"))
    row = __import__("json").loads(capsys.readouterr().out.strip())
    assert abs(row["diff_acc"] + 1 / 3) < 1e-12 and row["same_pick"] == 2 / 3 and row["diff_acc_norm"] == 0
