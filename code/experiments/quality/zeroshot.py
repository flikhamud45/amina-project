"""Plan 2, F2c: zero-shot accuracy of the certified integer OPT against its checkpoint.

    cd code && PYTHONPATH=src python experiments/9_quality/zeroshot.py --model facebook/opt-1.3b --device cuda \\
        --out zeroshot_opt-1.3b.jsonl
    cd code && PYTHONPATH=src python experiments/9_quality/zeroshot.py --model facebook/opt-125m --check

The tasks are those of lm-evaluation-harness 0.4 (zero-shot), re-implemented here so that the integer model needs no
harness wrapper; prompts, choices and metrics follow the harness's task files:

* ``lambada_openai``: EleutherAI/lambada_openai, test (5,153). Context: every word but the last; continuation: a space
  and the last word. ``acc``: the continuation is the greedy one; ``ppl``: exp of minus the mean log-likelihood.
* ``hellaswag``: Rowan/hellaswag, validation, the first ``--hellaswag-n`` documents of a fixed permutation (seed
  1234; 2,000 by default, 0 for all 10,042). Query ``preprocess(activity_label + ": " + ctx_a + " " +
  ctx_b.capitalize())``, choices ``preprocess(ending)`` (the harness's preprocess). ``acc`` and ``acc_norm``.
* ``piqa``: ybisk/piqa, validation (1,838; the hub's parquet conversion). ``"Question: {goal}\\nAnswer:"``, choices
  ``sol1``, ``sol2``. ``acc`` and ``acc_norm``.
* ``arc_easy`` / ``arc_challenge``: allenai/ai2_arc, test (2,376 / 1,172). ``"Question: {question}\\nAnswer:"``,
  choices ``choices.text``. ``acc`` and ``acc_norm``.
* ``winogrande``: allenai/winogrande, winogrande_xl, validation (1,267). Partial scoring: the context is the sentence
  up to the blank with each option filled in, the continuation the rest of the sentence. ``acc``.

A multiple-choice continuation is a space and the choice. ``acc_norm`` divides each choice's log-likelihood by the
choice's length in characters (the harness's ``acc_norm``). Log-likelihoods use the harness's ``_encode_pair``:
trailing spaces of the context move to the continuation, context and context + continuation are tokenised without
special tokens, and the continuation's ids are the tail of the joint encoding; the model reads context +
continuation without its last token, and the continuation's log-probabilities are read at its positions.

Requests are batched, longest first, up to ``--budget`` tokens per batch, and right-padded. That is exact for a
causal model: no position reads a later one, and the integer graph's constants do not depend on the input (``--check``
compares every padded row with the same request run alone, and the integer graph's logits with
``real_weights.real_logits``). The reference is the checkpoint in ``--ref-dtype`` (fp16 on a GPU by default), its
log-softmax in float32.

The integer model is ``opt_builder.build_opt`` (``--builder f2a``, F2a's recipe search; default: the smoothed graph
with channel maxima, no clipping, alpha 0.5 and 11-bit probabilities, F2a round 1's best) or the paper builder
``real_weights.build_opt_from_hf`` (``--builder paper``, default ``real_weights.CERTIFIED_RECIPE``); ``--opts`` (JSON)
replaces the recipe. Calibration: draw
``--draw`` of ``--calib-windows`` windows of ``--calib-seq`` tokens of the WikiText-2 *train* split (as F2a).

Output: one JSON line per task and model (stdout, and appended to ``--out``): accuracy, ``acc_norm``, the number of
documents, every document's correctness (so runs can be pooled), and for the integer model the paired difference to
the reference with a 95% bootstrap interval over documents and the share of documents on which both choose the same
answer.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

TASKS = ("lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande")
SOURCES = {   # task: (dataset repository, file, revision)
    "lambada_openai": ("EleutherAI/lambada_openai", "default/test/default.parquet", None),
    "hellaswag": ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet", None),
    "piqa": ("ybisk/piqa", "plain_text/validation/0000.parquet", "refs/convert/parquet"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet", None),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", None),
    "winogrande": ("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet", None),
}
F2A_BEST = dict(norm_mode="smooth", norm_gain=4.0, alpha=0.5, smooth_pct=100.0, pct=100.0, pct_att=100.0,
                prob_bits=11)
MAX_LENGTH = 2048


# ------------------------------------------------------------------ documents

def load_rows(task: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, name, rev = SOURCES[task]
    return pq.read_table(hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", revision=rev)).to_pylist()


def hellaswag_preprocess(text: str) -> str:
    """lm-eval's ``hellaswag.utils.preprocess``."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def make_docs(task: str, rows: list[dict]) -> list[dict]:
    """Per document: ``requests`` [(context, continuation)], ``gold``, and ``choices`` (the strings whose lengths
    normalise ``acc_norm``; None where the task has no ``acc_norm``)."""
    out = []
    for r in rows:
        if task == "lambada_openai":
            words = r["text"].split(" ")
            out.append({"requests": [(" ".join(words[:-1]), " " + words[-1])], "gold": 0, "choices": None})
        elif task == "hellaswag":
            ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
            query = hellaswag_preprocess(r["activity_label"] + ": " + ctx)
            ch = [hellaswag_preprocess(e) for e in r["endings"]]
            out.append({"requests": [(query, " " + c) for c in ch], "gold": int(r["label"]), "choices": ch})
        elif task == "piqa":
            ch = [r["sol1"], r["sol2"]]
            out.append({"requests": [(f"Question: {r['goal']}\nAnswer:", " " + c) for c in ch],
                        "gold": int(r["label"]), "choices": ch})
        elif task in ("arc_easy", "arc_challenge"):
            ch = list(r["choices"]["text"])
            out.append({"requests": [(f"Question: {r['question']}\nAnswer:", " " + c) for c in ch],
                        "gold": list(r["choices"]["label"]).index(r["answerKey"]), "choices": ch})
        elif task == "winogrande":
            s = r["sentence"]
            i = s.index("_")
            cont = " " + s[i + 1:].strip()
            out.append({"requests": [(s[:i] + o, cont) for o in (r["option1"], r["option2"])],
                        "gold": {"1": 0, "2": 1}[r["answer"]], "choices": None})
        else:
            raise ValueError(task)
    return out


def select(task: str, docs: list[dict], hellaswag_n: int, limit: int) -> list[dict]:
    if task == "hellaswag" and hellaswag_n:
        perm = torch.randperm(len(docs), generator=torch.Generator().manual_seed(1234)).tolist()
        docs = [docs[i] for i in sorted(perm[:hellaswag_n])]
    return docs[:limit] if limit else docs


def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-eval's ``HFLM._encode_pair`` (no special tokens)."""
    n = len(context) - len(context.rstrip())
    if n:
        continuation = context[-n:] + continuation
        context = context[:-n]
    whole = tok(context + continuation, add_special_tokens=False).input_ids
    ctx = tok(context, add_special_tokens=False).input_ids
    return ctx, whole[len(ctx):]


# ------------------------------------------------------------------ scoring

def score(encoded: list[tuple[list[int], list[int]]], logits_fn, budget: int, pad_id: int) -> list[tuple[float, bool]]:
    """``(log-likelihood, greedy)`` of each ``(context ids, continuation ids)``. ``logits_fn(ids [B, L])`` returns
    ``(logits [B, L, V], scale)``: real logits are ``logits * scale`` (1 for a float model)."""
    def length(i):
        c, k = encoded[i]
        return min(len(c) + len(k), MAX_LENGTH + 1) - 1
    order = sorted(range(len(encoded)), key=lambda i: (-length(i), i))
    out: list[tuple[float, bool] | None] = [None] * len(encoded)
    a = 0
    while a < len(order):
        width = length(order[a])
        batch = order[a:a + max(1, budget // max(width, 1))]
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        for row, j in enumerate(batch):
            c, k = encoded[j]
            seq = (c + k)[-(MAX_LENGTH + 1):][:-1]
            ids[row, :len(seq)] = torch.tensor(seq)
        logits, scale = logits_fn(ids)
        for row, j in enumerate(batch):
            k = encoded[j][1]
            n = length(j)
            lp = (logits[row, n - len(k):n].double() * scale).log_softmax(-1)
            tgt = torch.tensor(k, device=lp.device)
            out[j] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
        del logits
        a += len(batch)
    return out


def metrics(task: str, docs: list[dict], results: list[tuple[float, bool]]) -> dict:
    """Per-document correctness (``acc``, ``acc_norm``) and the chosen answers, from the flat ``results``."""
    acc, norm, pick, pick_norm, lls = [], [], [], [], []
    i = 0
    for d in docs:
        r = results[i:i + len(d["requests"])]
        i += len(d["requests"])
        if task == "lambada_openai":
            acc.append(int(r[0][1]))
            pick.append(int(r[0][1]))
            lls.append(r[0][0])
            continue
        ll = [x[0] for x in r]
        p = max(range(len(ll)), key=lambda c: (ll[c], -c))
        acc.append(int(p == d["gold"]))
        pick.append(p)
        if d["choices"] is not None:
            nl = [x / len(c) for x, c in zip(ll, d["choices"])]
            q = max(range(len(nl)), key=lambda c: (nl[c], -c))
            norm.append(int(q == d["gold"]))
            pick_norm.append(q)
    assert i == len(results)
    out = {"n": len(docs), "acc": sum(acc) / len(acc), "correct": acc, "pick": pick}
    if norm:
        out.update(acc_norm=sum(norm) / len(norm), correct_norm=norm, pick_norm=pick_norm)
    if lls:
        out["ppl"] = math.exp(-sum(lls) / len(lls))
    return out


def paired_ci(a: list[int], b: list[int], reps: int = 2000) -> list[float]:
    """95% bootstrap interval of mean(a) - mean(b) over documents (paired)."""
    g = torch.Generator().manual_seed(0)
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    idx = torch.randint(0, len(x), (reps, len(x)), generator=g)
    m = x[idx].mean(1)
    return [float(m.quantile(0.025)), float(m.quantile(0.975))]


# ------------------------------------------------------------------ models

def split_opt(model, devices: list[str]):
    """An OPT checkpoint spread over several devices in order (embeddings on the first, the blocks in equal runs,
    the final norm and the LM head on the last), for a reference that one GPU cannot hold (OPT-6.7B in fp16 on two
    11 GB cards). Forward pre-hooks move every tensor argument to the module's device; the LM head gets its own copy
    of the tied embedding. The arithmetic is the one-device model's (each op runs on one device)."""
    import math
    dec = model.model.decoder
    head = model.lm_head
    head.weight = torch.nn.Parameter(head.weight.detach().clone())          # untie: the head moves to another device

    def mover(dev):
        def hook(mod, args, kwargs):
            return (tuple(a.to(dev) if torch.is_tensor(a) else a for a in args),
                    {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()})
        return hook
    first, last = devices[0], devices[-1]
    for part in (dec.embed_tokens, dec.embed_positions, getattr(dec, "project_in", None)):
        if part is not None:
            part.to(first)
    per = math.ceil(len(dec.layers) / len(devices))
    for i, layer in enumerate(dec.layers):
        dev = devices[min(i // per, len(devices) - 1)]
        layer.to(dev)
        layer.register_forward_pre_hook(mover(dev), with_kwargs=True)
    for part in (getattr(dec, "final_layer_norm", None), getattr(dec, "project_out", None), head):
        if part is not None:
            part.to(last)
            part.register_forward_pre_hook(mover(last), with_kwargs=True)
    return model


def pair(ref_path: str, int_path: str) -> None:
    """Paired differences between a reference-only run (``--no-int``) and an integer-only run (``--no-ref``) on the
    same documents: one JSON line per task, as the integer rows of a joint run."""
    refs = {r["task"]: r for r in map(json.loads, open(ref_path, encoding="utf-8")) if r["variant"] == "ref"}
    for got in map(json.loads, open(int_path, encoding="utf-8")):
        if got["variant"] != "int" or got["task"] not in refs:
            continue
        ref = refs[got["task"]]
        assert ref["n"] == got["n"], got["task"]
        row = {"task": got["task"], "n": got["n"], "acc_ref": ref["acc"], "acc_int": got["acc"],
               "diff_acc": got["acc"] - ref["acc"], "diff_acc_ci95": paired_ci(got["correct"], ref["correct"]),
               "same_pick": sum(int(a == b) for a, b in zip(got["pick"], ref["pick"])) / got["n"]}
        if "acc_norm" in got:
            row.update(acc_norm_ref=ref["acc_norm"], acc_norm_int=got["acc_norm"],
                       diff_acc_norm=got["acc_norm"] - ref["acc_norm"],
                       diff_acc_norm_ci95=paired_ci(got["correct_norm"], ref["correct_norm"]))
        print(json.dumps(row), flush=True)


def float_logits_fn(model, device: str):
    def fn(ids):
        with torch.no_grad():
            return model(ids.to(device)).logits.float(), 1.0
    return fn


def int_logits_fn(graph, info: dict, device: str):
    from pvi.core.graph import CheapOp, IntGraph, identity_op, op_fn
    ops = [CheapOp(o.name, o.inputs, o.output, fn=op_fn(identity_op), note=o.note, params=o.params)
           if isinstance(o, CheapOp) and o.note == "last position" else o for o in graph.ops]
    lean, _ = IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta)).with_constants_on(device)
    scale = info["logit_scale"]

    def fn(ids):
        with torch.no_grad():
            env, _ = lean.forward(ids.to(device), free=True, send=lambda z: None)
            return env.pop(lean.output_name), scale
    return fn


def build_int(model, calib: torch.Tensor, builder: str, opts: dict):
    if builder == "paper":
        from pvi.models import hf_opt as rw
        return rw.build_opt_from_hf(model, calib, **opts)
    import opt_builder as ob
    return ob.build_opt(model, calib, **opts)


# ------------------------------------------------------------------ main

def _emit(args, row: dict) -> None:
    short = {k: v for k, v in row.items() if not isinstance(v, list)}
    print(json.dumps(short), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def check(args, tok, logits_ref, logits_int, graph, info) -> None:
    """Right padding is exact (every padded row equals the request alone) for the integer model and within
    rounding for the float one, and the integer logits equal ``real_weights.real_logits(..., all_positions=True)``."""
    from pvi.models import hf_opt as rw
    docs = make_docs("hellaswag", load_rows("hellaswag")[:4]) + make_docs("piqa", load_rows("piqa")[:4])
    enc = [encode_pair(tok, c, k) for d in docs for c, k in d["requests"]]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for name, fn in (("ref", logits_ref), ("int", logits_int)):
        batched = score(enc, fn, args.budget, pad)
        alone = [score([e], fn, args.budget, pad)[0] for e in enc]
        diff = max(abs(a[0] - b[0]) for a, b in zip(batched, alone))
        same = sum(a[1] == b[1] for a, b in zip(batched, alone))
        exact = name == "int"
        print(f"CHECK padding {name}: {len(enc)} requests, max |ll batched - alone| = {diff:.3g}, greedy equal "
              f"{same}/{len(enc)} ({'must be 0' if exact else 'float model: rounding only'})", flush=True)
        if exact:
            assert diff == 0 and same == len(enc), (name, diff)
    ids = torch.tensor([(enc[0][0] + enc[0][1])[:-1]])
    ours, scale = logits_int(ids)
    theirs = rw.real_logits(graph, info, ids, all_positions=True)
    assert torch.equal(ours.cpu().double() * scale, theirs.cpu()), "integer logits differ from real_logits"
    print("CHECK OK: right padding exact; integer logits equal real_logits", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--hellaswag-n", type=int, default=2000, help="HellaSwag documents (0: all 10,042)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many documents per task (0: all)")
    ap.add_argument("--builder", choices=["f2a", "paper"], default="f2a")
    ap.add_argument("--opts", default=None, help="JSON recipe for the builder (default: F2a round 1's best)")
    ap.add_argument("--calib-seq", type=int, default=2048)
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--ref-dtype", choices=["fp16", "bf16", "fp32"], default=None,
                    help="the checkpoint's dtype (default: fp16 on a GPU, fp32 on a CPU)")
    ap.add_argument("--no-int", action="store_true", help="the reference only (validation against published numbers)")
    ap.add_argument("--no-ref", action="store_true",
                    help="the integer model only (a model whose fp16 copy and integer graph do not fit one GPU: run "
                         "the reference in another process with --no-int; the records pool by document)")
    ap.add_argument("--budget", type=int, default=2048, help="tokens per batch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ref-device", default=None, help="where the reference runs (default: --device)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="padding exactness and the integer logits, then stop")
    ap.add_argument("--ref-gpus", type=int, default=1, help="spread the reference over this many GPUs (split_opt)")
    ap.add_argument("--pair", nargs=2, metavar=("REF_JSONL", "INT_JSONL"), default=None,
                    help="paired differences of a --no-int run and a --no-ref run, then stop")
    args = ap.parse_args()
    if args.pair:
        return pair(*args.pair)
    args.ref_device = args.ref_device or args.device
    assert not (args.no_int and args.no_ref), "--no-int and --no-ref leave nothing to run"
    assert not (args.check and (args.no_int or args.no_ref)), "--check needs both models"
    torch.set_num_threads(args.threads)

    import ppl_2048 as p2
    from common import load_opt
    model, tok = load_opt(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    from pvi.models.hf_opt import CERTIFIED_RECIPE
    opts = json.loads(args.opts) if args.opts else (dict(F2A_BEST) if args.builder == "f2a" else dict(CERTIFIED_RECIPE))
    head = {"model": args.model, "tag": args.tag, "host": platform.node(), "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    graph = info = None
    if not args.no_int:
        train = tok(p2._text("train")[:4_000_000], return_tensors="pt").input_ids[0]
        calib = torch.cat(p2._windows(train, args.calib_seq, args.calib_windows,
                                      start=args.draw * args.calib_windows))
        t0 = time.perf_counter()
        with torch.no_grad():
            graph, info = build_int(model, calib, args.builder, opts)
        head.update(builder=args.builder, opts=opts, draw=args.draw, calib=[args.calib_windows, args.calib_seq],
                    digest=graph.digest().hex(), build_s=round(time.perf_counter() - t0, 1))
        print(json.dumps({k: head[k] for k in ("builder", "opts", "digest", "build_s")}), flush=True)

    logits_int = int_logits_fn(graph, info, args.device) if graph is not None else None
    logits_ref = None
    if not args.no_ref:                        # after the build, which reads the fp32 model
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
            args.ref_dtype or ("fp16" if args.ref_device.startswith("cuda") else "fp32")]
        if args.ref_gpus > 1:
            devs = [f"cuda:{i}" for i in range(args.ref_gpus)]
            logits_ref = float_logits_fn(split_opt(model.to(dtype=dtype), devs), devs[0])
        else:
            logits_ref = float_logits_fn(model.to(dtype=dtype, device=args.ref_device), args.ref_device)
        head["ref_dtype"] = str(dtype).replace("torch.", "")
    else:
        del model
    if args.check:
        return check(args, tok, logits_ref, logits_int, graph, info)

    for task in args.tasks:
        docs = select(task, make_docs(task, load_rows(task)), args.hellaswag_n, args.limit)
        enc = [encode_pair(tok, c, k) for d in docs for c, k in d["requests"]]
        ref = None
        if logits_ref is not None:
            t0 = time.perf_counter()
            ref = metrics(task, docs, score(enc, logits_ref, args.budget, pad))
            _emit(args, dict(head, task=task, variant="ref", seconds=round(time.perf_counter() - t0, 1), **ref))
        if logits_int is None:
            continue
        t0 = time.perf_counter()
        got = metrics(task, docs, score(enc, logits_int, args.budget, pad))
        row = dict(head, task=task, variant="int", seconds=round(time.perf_counter() - t0, 1), **got)
        if ref is not None:
            row.update(diff_acc=got["acc"] - ref["acc"], diff_acc_ci95=paired_ci(got["correct"], ref["correct"]),
                       same_pick=sum(int(a == b) for a, b in zip(got["pick"], ref["pick"])) / len(docs))
            if "acc_norm" in got:
                row.update(diff_acc_norm=got["acc_norm"] - ref["acc_norm"],
                           diff_acc_norm_ci95=paired_ci(got["correct_norm"], ref["correct_norm"]))
        _emit(args, row)


if __name__ == "__main__":
    main()
