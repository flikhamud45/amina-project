"""Plan 2, F2b-Q / F2b-L: the certified integer Qwen3 and Llama models' quality on WikiText-2.

    cd code && PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base \\
        --seq 128 --windows 16 --device cuda --structure --out q06_128.jsonl
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model TinyLlama/TinyLlama_v1.1 \\
        --seq 2048 --device cuda --out tl_2048.jsonl
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --check
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --smoke \\
        --seq 128 --windows 2 --calib-windows 1 --device cpu --threads 2 --chunk 64     # a tiny random model

* Text and windows are ``ppl_2048.py``'s: WikiText-2 (raw) joined with blank lines, tokenised once; every
  non-overlapping ``--seq``-token window of the test split (``--windows`` keeps the first ones: 16 x 128 is the
  short setting, every 2,048-token window the long one), each scored on its own, all positions in one pass.
* Calibration: draw ``d`` is ``--calib-windows`` consecutive ``--calib-seq``-token windows (default ``--seq``) of
  the *train* split, starting at window ``d * --calib-windows``; no evaluated token is seen.
* Reference: the checkpoint in fp32 (``--ref-dtype`` float16 / bfloat16: a second copy loaded in that type), its LM
  head applied a ``--chunk`` of positions at a time; ``--ref-cache`` keeps its NLL and argmaxes.
* Graphs (``real_llama.build_llama_from_hf``): ``scalar`` (one norm gain per ``--gains`` value, no clipping) and
  ``smooth`` (SmoothQuant through the norm gains, alpha per ``--alphas``, ``--smooth-pct`` 99.9, ``--pct`` 99.99 at
  q/k/v/gate/up and the q/k norms, ``--pct-att`` 99.99 at the attention output, ``--pct-glu`` (default ``--pct``) at
  the SwiGLU product). ``--no-head-eq`` turns off the head equalisation in front of Qwen3's q/k norms.
  ``--glu-exps E ...`` adds ``smooth-expE`` cells: the smoothed graph with the opt-in per-token exponent ``0 .. E``
  at the SwiGLU product (``real_llama.glu_exp``; ``--pct-glu`` then picks the token maximum that takes exponent E).
* Lean evaluation (``real_llama.lean_logits``): the integer forward frees each tensor after its last reader and keeps
  no claim; the LM head takes ``--chunk`` positions at a time, reduced at once to NLL and argmax.
* Per cell, also: top-1 agreement with the reference, the per-block cosine of the residual stream against the fp32
  checkpoint (mean and minimum over positions, on the first ``--cos-windows`` windows), ``opt_builder.graph_report``
  (claim bound, multipliers outside V1's P3), the graph digest and the structural diff (``--structure``: against
  ``build_decoder(CONFIGS[--config])``, which builds the full random benchmark graph).
* Output: one JSON line per cell (stdout, and appended to ``--out``): perplexity, ratio to the reference with a 95%
  bootstrap interval over windows, top-1, block cosines, per-window NLL, seconds, peak host and GPU memory.
* ``--check``: on one 128-token window, for the smoothed graph and its ``glu_exp = 8`` variant, the lean chunked
  logits equal the plain forward's, the pruned graph's last-position logits equal the unpruned graph's and, on a
  GPU, the GPU's integers equal the CPU's; the smoothed graphs' structural diffs against ``CONFIGS[--config]`` are
  empty.
* ``--protocol``: the checkpoint's first 2 blocks as a real graph (unpruned and pruned) on a 64-token window: the
  2-block benchmark structure, v0 in modes C and Kpre and V1 accept honest queries and reject tampered claims.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_builder as ob                                   # noqa: E402
import ppl_2048 as p2                                      # noqa: E402

from pvi.fullcheck import real_llama as rl                 # noqa: E402
from pvi.fullcheck.transformer import CONFIGS              # noqa: E402

CONFIG_OF = {"Qwen3-0.6B-Base": "qwen3-0.6b", "Qwen3-1.7B-Base": "qwen3-1.7b", "Qwen3-4B-Base": "qwen3-4b",
             "TinyLlama_v1.1": "tinyllama-1.1b"}


# ------------------------------------------------------------------ models

def load_llama(name: str, dtype: torch.dtype = torch.float32):
    """A Llama / Qwen3 checkpoint in ``dtype``, eval mode, and its tokenizer. As ``common.load_opt``: the model is
    built on the meta device and the state dict assigned to it (no second copy), from safetensors where the repo
    has them, else from ``pytorch_model.bin`` with ``weights_only=True`` (transformers refuses ``torch.load`` under
    torch < 2.6); the rotary embedding's buffers are not in the state dict, so it is built again on the CPU."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    meta = ["*.json", "*.model", "*.txt"]
    path = snapshot_download(name, allow_patterns=meta + ["*.safetensors"])
    shards = _shards(path, "model.safetensors")
    if not shards:
        path = snapshot_download(name, allow_patterns=meta + ["*.bin"])
        shards = _shards(path, "pytorch_model.bin")
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    sd = {}
    for shard in shards:
        if shard.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd.update(load_file(shard))
        else:
            sd.update(torch.load(shard, map_location="cpu", weights_only=True))
    sd = {k: v for k, v in sd.items() if not k.endswith("rotary_emb.inv_freq")}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del sd
    missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_word_embeddings)]
    assert not missing and not unexpected, (missing, unexpected)
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    if cfg.tie_word_embeddings:
        model.tie_weights()
    for p_ in model.parameters():                   # bf16 checkpoints -> dtype, one tensor at a time
        p_.data = p_.data.to(dtype)
    model = model.eval()
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
    return model, AutoTokenizer.from_pretrained(path)


def _shards(path: str, single: str) -> list[str]:
    index = os.path.join(path, single + ".index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return [os.path.join(path, s) for s in sorted(set(json.load(f)["weight_map"].values()))]
    return [os.path.join(path, single)] if os.path.exists(os.path.join(path, single)) else []


def smoke_model(name: str, seed: int = 0):
    """A 2-block random model of ``name``'s family and vocabulary (d = 64, grouped-query heads, random norm gains,
    weights of std 0.15) and ``name``'s tokenizer: only the configuration and the tokenizer are downloaded."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download(name, allow_patterns=["*.json", "*.model", "*.txt"])
    cfg = AutoConfig.from_pretrained(path).to_dict()
    cfg.pop("layer_types", None)
    cfg.update(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, initializer_range=0.15, max_window_layers=2)
    cfg = AutoConfig.for_model(**cfg)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    with torch.no_grad():
        for n, p_ in model.named_parameters():
            if "norm" in n:
                p_.uniform_(0.5, 1.5)
    return model.eval(), AutoTokenizer.from_pretrained(path)


# ------------------------------------------------------------------ evaluation

def _nll_argmax_chunk(logits: torch.Tensor, ids: torch.Tensor, a: int):
    """Summed NLL of the positions ``a .. a + n - 1`` of ``logits [1, n, V]`` (real units) against the next tokens
    of ``ids [1, T]`` (the last position has none), and the argmaxes of all ``n``."""
    lg = logits[0].double()
    tgt = ids[0, a + 1:a + 1 + lg.shape[0]].to(lg.device)
    nll = float(F.cross_entropy(lg[:tgt.numel()], tgt, reduction="sum")) if tgt.numel() else 0.0
    return nll, lg.argmax(-1).cpu()


def int_eval(graph, scale: float, w: torch.Tensor, device: str, chunk: int):
    """NLL and argmaxes of the integer graph on the window ``w [1, T]`` (lean forward, chunked LM head)."""
    nll, am = 0.0, []
    for a, z in rl.lean_logits(graph, w.to(device), chunk=chunk):
        x, m = _nll_argmax_chunk(z.double() * scale, w, a)
        del z
        nll += x
        am.append(m)
    return nll, torch.cat(am)


def fp_eval(model, w: torch.Tensor, chunk: int):
    """The same for the checkpoint: its final hidden states, then its LM head on ``chunk`` positions at a time."""
    hs = model.model(w.to(model.device)).last_hidden_state
    nll, am = 0.0, []
    for a in range(0, hs.shape[1], chunk):
        x, m = _nll_argmax_chunk(model.lm_head(hs[:, a:a + chunk]).float(), w, a)
        nll += x
        am.append(m)
    return nll, torch.cat(am)


def reference(args, model, wins) -> dict:
    """The reference's NLL and argmaxes per window (cached in ``--ref-cache`` with its key)."""
    key = {"model": args.model, "seq": args.seq, "windows": len(wins), "dtype": args.ref_dtype,
           "smoke": args.smoke}
    cache = Path(args.ref_cache) if args.ref_cache else None
    if cache is not None and cache.exists():
        ref = torch.load(cache, weights_only=True)
        if ref.get("key") == key:
            return ref
    device = args.ref_device or args.device
    t0 = time.perf_counter()
    if args.ref_dtype == "float32":
        m = model.to(device)
    elif args.smoke:                # the smoke model itself in that type (no checkpoint is downloaded)
        m = copy.deepcopy(model).to(device=device, dtype=getattr(torch, args.ref_dtype))
    else:
        m = load_llama(args.model, getattr(torch, args.ref_dtype))[0].to(device)
    nll, am = [], []
    with torch.no_grad():
        for w in wins:
            x, a = fp_eval(m, w, args.chunk)
            nll.append(x)
            am.append(a)
    if m is model:
        model.to("cpu")
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ref = {"key": key, "nll": nll, "argmax": torch.stack(am), "seconds": time.perf_counter() - t0}
    if cache is not None:
        torch.save(ref, cache)
    return ref


def cells(args) -> list[dict]:
    head_eq = not args.no_head_eq
    out = [{"graph": "scalar", "gain": g, "alpha": None,
            "kw": dict(norm_gain=g, head_eq=head_eq)} for g in (args.gains if "scalar" in args.graphs else [])]
    out += [{"graph": "smooth" if not e else f"smooth-exp{e}", "gain": None, "alpha": a,
             "kw": dict(smooth=a, smooth_pct=args.smooth_pct, pct=args.pct, pct_att=args.pct_att,
                        pct_glu=args.pct_glu, head_eq=head_eq, glu_exp=e)}
            for e in ([0] if "smooth" in args.graphs else []) + list(args.glu_exps) for a in args.alphas]
    return out


def emit(args, row: dict) -> None:
    line = json.dumps(row)
    print(line, flush=True)
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
             if k in ("cell", "draw", "gain", "alpha", "ppl", "ratio", "ratio_ci95", "top1", "block_cos_min",
                      "structure_diff", "seconds")}
    print(json.dumps(short), file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--config", default=None, help="the CONFIGS key of its shape (default: from --model)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0, help="0: every window of the test split")
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--calib-seq", type=int, default=None, help="default: --seq")
    ap.add_argument("--train-chars", type=int, default=4_000_000, help="characters of the train split tokenised")
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--graphs", nargs="*", default=["scalar", "smooth"], help="none: the reference only")
    ap.add_argument("--gains", type=float, nargs="+", default=[4.0])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.55])
    ap.add_argument("--pct", type=float, default=99.99)
    ap.add_argument("--pct-att", type=float, default=99.99)
    ap.add_argument("--pct-glu", type=float, default=None)
    ap.add_argument("--smooth-pct", type=float, default=99.9)
    ap.add_argument("--no-head-eq", action="store_true")
    ap.add_argument("--glu-exps", type=int, nargs="*", default=[],
                    help="also the smoothed graph with a per-token exponent 0..E at the SwiGLU product, per E")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="where the integer forward runs")
    ap.add_argument("--calib-device", default="cpu", help="where the integer calibration pass runs")
    ap.add_argument("--ref-device", default=None, help="where the reference runs (default: --device)")
    ap.add_argument("--ref-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--cos-windows", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=256, help="positions per LM-head chunk")
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="a 2-block random model with the checkpoint's vocabulary")
    ap.add_argument("--check", action="store_true", help="exactness checks on one 128-token window")
    ap.add_argument("--protocol", action="store_true", help="2-block real graphs through v0 (C, Kpre) and V1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = smoke_model(args.model) if args.smoke else load_llama(args.model)
    name = args.config or CONFIG_OF.get(args.model.split("/")[-1])
    shape = rl.config_of(model.config)
    if args.check:
        return check(args, model, tok, name)
    if args.protocol:
        return protocol(args, model, tok)

    test = tok(p2._text("test"), return_tensors="pt").input_ids[0]
    wins = p2._windows(test, args.seq, args.windows)
    train = tok(p2._text("train")[:args.train_chars], return_tensors="pt").input_ids[0]
    counts = [w.shape[1] - 1 for w in wins]
    ref = reference(args, model, wins)
    fp = math.exp(sum(ref["nll"]) / sum(counts))
    head = {"model": args.model, "smoke": args.smoke, "config": name, "seq": args.seq, "windows": len(wins),
            "tokens": sum(counts), "dataset": "wikitext-2-raw-v1 test, '\\n\\n'.join", "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "device": args.device,
            "calib_windows": args.calib_windows, "calib_seq": args.calib_seq or args.seq, "ref_dtype": args.ref_dtype,
            "shape_vs_config": (rl.same_shape(shape, CONFIGS[name]) if name in CONFIGS else None)}
    emit(args, dict(head, cell="reference", ppl=fp, nll=ref["nll"], seconds=ref["seconds"]))
    todo = cells(args)
    if not todo:
        return
    with torch.no_grad():
        hf_states = [rl.hf_residual_states(model, w) for w in wins[:args.cos_windows]]
    structure_done = False
    for draw in args.draws:
        cs = args.calib_seq or args.seq
        calib = torch.cat(p2._windows(train, cs, args.calib_windows, start=draw * args.calib_windows))
        for cell in todo:
            t0 = time.perf_counter()
            with torch.no_grad():
                graph, info = rl.build_llama_from_hf(model, calib, device=args.calib_device, **cell["kw"])
            build_s = time.perf_counter() - t0
            extra = {"digest": graph.digest().hex(), "report": ob.graph_report(graph)}
            if args.structure and not structure_done:
                # against the named benchmark configuration when the checkpoint has its shape (a smoke model
                # does not: then against a benchmark graph of the model's own shape)
                same = name in CONFIGS and not rl.same_shape(shape, CONFIGS[name])
                extra["structure_against"] = name if same else "own shape"
                extra["structure_diff"] = rl.matches_benchmark_graph(graph, name if same else shape)
                structure_done = True
            lean, _ = graph.with_constants_on(args.device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            nll, agree = [], 0
            with torch.no_grad():
                for w, r_am in zip(wins, ref["argmax"]):
                    x, am = int_eval(lean, info["logit_scale"], w, args.device, args.chunk)
                    nll.append(x)
                    agree += int((am[:-1] == r_am[:-1]).sum())
                blocks = []
                for w, states in zip(wins, hf_states):
                    ours = rl.residual_states(lean, info, w.to(args.device))
                    blocks.append(rl.block_cosines(ours, states))
                    del ours
            per_block = [{"mean": sum(b[i]["mean"] for b in blocks) / len(blocks),
                          "min": min(b[i]["min"] for b in blocks)} for i in range(len(blocks[0]))] if blocks else []
            ppl = math.exp(sum(nll) / sum(counts))
            emit(args, dict(head, cell=f"{cell['graph']}", gain=cell["gain"], alpha=cell["alpha"], draw=draw,
                            opts={k: v for k, v in cell["kw"].items()}, ppl=ppl, ref_ppl=fp, ratio=ppl / fp,
                            ratio_ci95=p2._bootstrap(nll, ref["nll"], counts), top1=agree / sum(counts),
                            block_cos=per_block, block_cos_min=min((b["min"] for b in per_block), default=None),
                            block_cos_mean_min=min((b["mean"] for b in per_block), default=None),
                            s_residual=info["s_residual"], logit_scale=info["logit_scale"],
                            embed_mult=info["embed_mult"], embed_clip=info["embed_clip"], nll=nll, build_s=build_s,
                            seconds=time.perf_counter() - t0, peak_rss_GB=p2._peak_rss_gb(),
                            peak_gpu_GB=p2._peak_gpu_gb(), **extra))
            del graph, lean, info
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def check(args, model, tok, name: str | None) -> None:
    """Exactness on one 128-token test window, with the smoothed recipe calibrated on one train window, for the
    graph the main cells evaluate (``smooth``) and for its opt-in per-token-exponent variant (``smooth-exp8``,
    ``glu_exp = 8``, whose two extra op kinds the ``smooth-expE`` cells run on the GPU): the lean chunked logits
    equal one pass, the last position equals the plain forward, the pruned graph's logits equal the unpruned graph's
    and, on a GPU, the GPU's integers equal the CPU's. The ``smooth`` graphs' structural diffs are taken against
    ``build_decoder(CONFIGS[name])`` when the checkpoint has that shape (``shape_vs_config == []``), else against a
    benchmark graph of the model's own shape (a smoke model)."""
    seq = 128
    test = tok(p2._text("test"), return_tensors="pt").input_ids[0]
    w = p2._windows(test, seq, 1)[0]
    calib = p2._windows(tok(p2._text("train")[:200_000], return_tensors="pt").input_ids[0], seq, 1)[0]
    shape = rl.config_of(model.config)
    vs_config = rl.same_shape(shape, CONFIGS[name]) if name in CONFIGS else None
    against = name if vs_config == [] else shape
    report = {"model": args.model, "smoke": args.smoke, "seq": seq, "shape_vs_config": vs_config,
              "structure_against": name if vs_config == [] else "own shape"}
    ok = args.smoke or vs_config == []              # a real checkpoint must have its CONFIGS entry's shape
    for label, extra in (("smooth", {}), ("smooth-exp8", {"glu_exp": 8})):
        kw = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99, head_eq=not args.no_head_eq, **extra)
        with torch.no_grad():
            graph, info = rl.build_llama_from_hf(model, calib, **kw)
            pruned, _ = rl.build_llama_from_hf(model, calib, prune_last=True, **kw)
            chunks = torch.cat([z for _, z in rl.lean_logits(graph, w, chunk=args.chunk)], 1)
            whole = torch.cat([z for _, z in rl.lean_logits(graph, w, chunk=1 << 30)], 1)
            env, _ = graph.forward(w)
            env_p, _ = pruned.forward(w)
            cell = {"chunked_equals_whole": bool(torch.equal(chunks, whole)),
                    "last_equals_forward": bool(torch.equal(whole[:, -1], env[graph.output_name][:, -1])),
                    "pruned_equals_unpruned": bool(torch.equal(env_p[pruned.output_name], env[graph.output_name])),
                    "gpu_equals_cpu": None, "glu_emax": info["glu_emax"]}
            del env, env_p, chunks
            if not extra:
                cell["structure_diff"] = rl.matches_benchmark_graph(graph, against)
                cell["pruned_structure_diff"] = rl.matches_benchmark_graph(pruned, against, prune_last=True)
            del pruned
            if torch.cuda.is_available():
                lean, _ = graph.with_constants_on("cuda")
                gpu = torch.cat([z.cpu() for _, z in rl.lean_logits(lean, w.cuda(), chunk=args.chunk)], 1)
                cell["gpu_equals_cpu"] = bool(torch.equal(gpu, whole))
                del lean, gpu
                torch.cuda.empty_cache()
            nll, _ = int_eval(graph, info["logit_scale"], w, "cpu", args.chunk)
            cell["ppl"] = math.exp(nll / (seq - 1))
            del graph, whole
        report[label] = cell
        ok = ok and all(v for k, v in cell.items() if k.endswith(("_equals_whole", "_equals_forward",
                                                                  "_equals_unpruned", "_equals_cpu"))
                        and v is not None)
        ok = ok and cell.get("structure_diff", []) == [] and cell.get("pruned_structure_diff", []) == []
    with torch.no_grad():
        report["ref_ppl"] = math.exp(fp_eval(model, w, args.chunk)[0] / (seq - 1))
    print(json.dumps(report, indent=1), flush=True)
    print("CHECK", "PASSED" if ok else "FAILED", flush=True)


def protocol(args, model, tok) -> None:
    """The checkpoint's first 2 blocks as a real graph (``n_layers=2``, then the final norm and the LM head; the
    smoothed recipe calibrated on one 64-token train window), unpruned and pruned, on a 64-token test window: the
    structure is the 2-block benchmark graph's, v0 accepts in modes C and Kpre (lambda = 40) and rejects a tampered
    claim (the first q projection, the middle weight op, the LM head), and V1 (mode C) accepts and rejects the same
    tampers."""
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
    seq = 64
    test = tok(p2._text("test"), return_tensors="pt").input_ids[0]
    w = p2._windows(test, seq, 1)[0]
    calib = p2._windows(tok(p2._text("train")[:200_000], return_tensors="pt").input_ids[0], seq, 1)[0]
    kw = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99, head_eq=not args.no_head_eq)
    report, ok = {"model": args.model, "smoke": args.smoke, "seq": seq}, True
    for prune in (False, True):
        t0 = time.perf_counter()
        with torch.no_grad():
            graph, _ = rl.build_llama_from_hf(model, calib, n_layers=2, prune_last=prune, **kw)
        cell = {"structure_diff": rl.matches_benchmark_graph(graph, rl.config_of(model.config), n_layers=2,
                                                             prune_last=prune)}
        targets = [graph.mat_ops[1].name, graph.mat_ops[len(graph.mat_ops) // 2].name, graph.mat_ops[-1].name]

        def tampers():
            for target in targets:
                def tamper(op, z, target=target):
                    if op.name == target:
                        z = z.clone()
                        z.view(-1)[0] -= 3
                    return z
                yield target, {"tamper": tamper}

        commitments = {}                   # one commitment per rate (modes C and V1 share it)
        for mode in ("C", "Kpre", "V1"):
            params = params_for(40, len(graph.mat_ops), cut=mode == "V1")
            if mode == "Kpre":
                prover = Prover(graph)
                v = Verifier(graph.public(), params, "Kpre", weights={o.name: (o.weight, o.bias)
                                                                       for o in graph.mat_ops})
                v.precompute(Challenger(seed=11))
            else:
                if params.rate not in commitments:
                    commitments[params.rate] = commit_graph(graph, params.rate)
                coms = commitments[params.rate]
                prover = Prover(graph, commitments=coms)
                v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
            run = (lambda kw_=None, s=1: cp.run_cut_query(prover, v, w, seed=s, forward_kwargs=kw_)) if mode == "V1" \
                else (lambda kw_=None, s=1: run_query(prover, v, w, seed=s, forward_kwargs=kw_))
            cell[mode] = {"honest_accepted": [bool(run(None, s)["accepted"]) for s in (1, 2)],
                          "tamper_rejected": {t: not run(f)["accepted"] for t, f in tampers()}}
            ok = ok and all(cell[mode]["honest_accepted"]) and all(cell[mode]["tamper_rejected"].values())
        cell["seconds"] = time.perf_counter() - t0
        ok = ok and cell["structure_diff"] == []
        report["pruned" if prune else "unpruned"] = cell
        del graph
    print(json.dumps(report, indent=1), flush=True)
    print("PROTOCOL", "PASSED" if ok else "FAILED", flush=True)


if __name__ == "__main__":
    main()
