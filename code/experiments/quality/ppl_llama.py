"""Perplexity of the integer Llama-family models (Llama, Qwen3) on WikiText-2 (raw): the Llama-family ratios of
Sec. ``sec:quality`` and of the extended version's discussion (with the main recipe at 2,048 tokens TinyLlama x1.12,
Qwen3-1.7B x1.40 and Qwen3-4B x5.3; with the per-token exponent, a possible improvement, Qwen3-1.7B x1.20 and
Qwen3-4B x3.0), the ratios of Llama-2-7B and Llama-2-13B (the run package's quality tier, with ``--llama-licence``),
their committed alphas (and Llama-2-7B's product balance) among their neighbours on the test and the validation split,
and the check that Llama-2-7B's integer graph has exactly the structure of the benchmark graph whose costs the paper
reports.  OpenLLaMA-7B (``openlm-research/open_llama_7b_v2``), a licence-free model of Llama-2-7B's shapes, takes
Llama-2-7B's place in a run without the Llama-2 licence (x1.073 with the same recipe); the paper does not report it.

    python -m experiments.quality.ppl_llama --model meta-llama/Llama-2-7b-hf --alphas 0.45 0.55 0.65 \\
        --glu-smooths 0.1 0.2 0.3 --structure --device cuda --out llama2-7b.jsonl       # add --split validation
    python -m experiments.quality.ppl_llama --model meta-llama/Llama-2-13b-hf --alphas 0.5 0.6 0.7 \\
        --ref-dtype float16 --structure --device cuda --out llama2-13b.jsonl            # add --split validation
    python -m experiments.quality.ppl_llama --model openlm-research/open_llama_7b_v2 --config llama2-7b \\
        --structure --device cuda --ref-device cpu --ref-cache ref_open_llama_7b_v2.pt --out open_llama_7b_v2.jsonl
    python -m experiments.quality.ppl_llama --model TinyLlama/TinyLlama_v1.1 --device cuda --out tinyllama.jsonl
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-4B-Base --device cuda --ref-device cpu \\
        --graphs main smooth --glu-exps 8 --out qwen3-4b.jsonl
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-4B-Base --device cuda --ref-device cpu \\
        --glu-exps 8 --exp-only --out qwen3-4b.jsonl          # the per-token-exponent cells only
    python -m experiments.quality.ppl_llama --model openlm-research/open_llama_7b_v2 --device cuda --split validation \\
        --alphas 0.45 0.55 0.65 --glu-smooths 0.1 0.2 0.3 --out open_llama_7b_v2_validation.jsonl
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-0.6B-Base --check
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-0.6B-Base --protocol
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-0.6B-Base --smoke --seq 128 --windows 2 \\
        --calib-windows 1 --device cpu --threads 2 --chunk 64                # a tiny random model (laptop)

* Text and windows: WikiText-2 (raw) ``--split`` -- ``test`` (the reported ratios) or ``validation`` (to check a
  checkpoint's smoothing alpha and product balance on text no choice used) -- joined with blank lines
  (``checkpoints.wikitext``), tokenised once -- with a prefix without the tokenizer's special tokens, since the prefix
  takes the BOS's place and no window holds one; every non-overlapping ``--seq``-token window (``--windows`` keeps the
  first ones; 0: all), each scored on its own at every position in one pass.
* Calibration: draw ``d`` is ``--calib-windows`` consecutive ``--calib-seq``-token windows (default ``--seq``) of the
  train split's first ``--train-chars`` characters from window ``d * --calib-windows`` on; no evaluated token is
  seen.  With the defaults, draw 0 is ``checkpoints.llama_calibration``: for OpenLLaMA-7B and Llama-2-7B (whose
  settings are ``MAIN_RECIPE``'s) the main cell is then the unpruned graph of the real-weight cost cells
  (``checkpoints.real_graph``).
* The fixed public prefix (``--prefix``, part of the main recipe): ``auto`` (the default, and the value of a bare
  ``--prefix``) is PrefixQuant's rule (``hf_llama.prefixquant_tokens``: the high-frequency outlier tokens of the down
  projections' inputs, then the BOS) on the first draw's calibration windows; ``bos`` the configuration's BOS alone;
  comma-separated token ids; ``none`` no prefix.  Its keys and values enter every attention op as int8 constants
  (``build_llama_from_hf(prefix_tokens=)``).  Only the windows' own tokens are scored, exactly the targets of a run
  without a prefix, and the fp32 reference runs on the same ``[prefix | window]`` (its prefix positions dropped), so
  the ratio compares like with like; every record says so (``prefix_tokens``, ``prefix_scoring``).
* Graphs (``--graphs``, ``hf_llama.build_llama_from_hf``): ``main`` (the default), the main algorithm's recipe
  ``hf_llama.MAIN_RECIPE``: SmoothQuant through the norms' gains at alpha ``--alphas`` with the ``--smooth-pct``
  (99.9th) percentile of each channel, requantisation at the ``--pct`` (99.99th) percentile at q/k/v/gate/up and the
  q/k norms, ``--pct-att`` at the attention output and ``--pct-glu`` (default ``--pct``) at the SwiGLU product, the
  SwiGLU product's balance ``--glu-smooths``, the prefix, and for Qwen3 the q/k-norm balance per RoPE pair and the
  head equalisation in front of the q/k norms; alpha and the product balance default to the checkpoint's
  (:data:`pvi.models.hf_llama.MAIN_SETTINGS`).  ``smooth``: the same without the product balance.  ``scalar``: one norm gain per
  ``--gains``, no clipping.  ``--glu-exps E ...`` adds to every smoothed cell its variant with the per-token exponent
  ``0 .. E`` at the SwiGLU product (``glu_exp``; ``--pct-glu`` then picks the token maximum that takes exponent E): a
  possible improvement, not the main algorithm, since it changes two ops per block and V1's cut; ``--exp-only``
  keeps only those variants (a run whose cells without the exponent an earlier run of the same settings measured).
  ``--no-qk-balance``: Qwen3's graphs without the q/k-norm balance.  Every graph keeps the benchmark graph's ops and
  shapes (but ``glu_exp``'s two ops per block): only constants change.
* Reference: the checkpoint in fp32 on ``--ref-device`` (``--ref-dtype`` float16 / bfloat16: a second copy loaded in
  that type), its LM head applied a ``--chunk`` of positions at a time; ``--ref-cache`` keeps its NLL and argmaxes
  (keyed by the model, the split, the windows, the type and the prefix).
* Lean evaluation (``hf_llama.lean_logits``): the integer forward on ``--device`` frees each tensor after its last
  reader and keeps no claim; the LM head takes ``--chunk`` positions at a time, reduced at once to NLL and argmax.
  The integer calibration pass runs on ``--calib-device``.
* Per cell also: the top-1 agreement with the reference, the per-block cosine of the residual stream against the
  fp32 checkpoint (mean and minimum over positions, on the first ``--cos-windows`` windows), ``checkpoints.
  graph_report`` (the claim bound, the multipliers outside V1's range), the graph's digest and weights hash and, with
  ``--structure``, the structural diff against ``build_decoder(CONFIGS[--config])`` (the full random benchmark graph;
  against a benchmark graph of the model's own shape when the checkpoint has no ``CONFIGS`` entry's shape).
* ``--check``: on one 128-token test window, for the main graph and its per-token-exponent variant (exponent
  ``--glu-exps``' first value, default 8), the lean chunked logits equal one pass, the last position equals the plain
  forward, the pruned graph's logits equal the unpruned graph's and, on a GPU, the GPU's integers equal the CPU's; the
  main graph's structural diffs (unpruned and pruned) are empty.  Prints one JSON line and ``CHECK PASSED`` (or exits
  non-zero).
* ``--protocol``: the checkpoint's first 2 blocks as a real graph (main recipe, ``--prune-last``) on a 64-token
  window, under the protocol flags (default the paper's configuration, ``experiments.common.cli.PAPER``): the 2-block
  benchmark structure, and v0 in modes C and Kpre and V1 (mode C) accept an honest query and reject a tampered claim
  of the first q projection, the middle weight op and the LM head.  Prints one JSON line and ``PROTOCOL PASSED``.
  The protocol flags apply to this check only.

Output: one JSON line per cell on stdout (and appended to ``--out``): first the reference (``cell: "reference"``,
``ppl``, per-window ``nll``), then per integer graph ``cell`` (``main-a<alpha>-glus<balance>``, ``smooth-a<alpha>``,
``scalar-g<gain>``, with ``-exp<E>`` for the per-token exponent), ``graph``, ``gain``, ``alpha``, ``glu_smooth``,
``glu_exp``, ``draw``, ``opts`` (the builder's keyword arguments but the prefix), ``ppl``, ``ref_ppl``, ``ratio`` with
``ratio_ci95`` (a 95% bootstrap interval over windows), ``top1``, ``block_cos`` (``block_cos_min``,
``block_cos_mean_min``), the residual and logit scales, the embedding's multiplier and clipping, per-window ``nll``,
``digest``, ``weights_sha256``, ``report``, ``build_s``, ``seconds``, the peak memory (``peaks``, ``peak_rss_GB``,
``peak_gpu_GB``), with a prefix its keys' and values' reach (``prefix_clip``, ``prefix_hidden_max``), with
``--structure`` ``structure_against`` and ``structure_diff``; every line carries the run's head (model, shape,
``split``, windows, tokens, calibration, prefix and its choice, ``env``: ``records.environment``).  A short summary per
cell goes to stderr.

Hardware and time (2,048 tokens, every test window; the integer forward on an RTX 2080 Ti, the calibration on the
CPU of a Xeon node with 250 GB): a 7B cell (OpenLLaMA-7B) takes about 47 min, mostly the calibration build on the
CPU, and its fp32 reference about 4 h on the CPU (``--ref-device cpu``; ``--ref-cache`` keeps it); Qwen3-4B's fp32
reference does not fit the 2080 Ti either (about 2.7 h on the CPU) and its build needs about 128 GB of host memory;
TinyLlama, Qwen3-0.6B and Qwen3-1.7B need 64 GB.  The run package's Llama-2 grids run on an L40S (48 GB) with the
reference on the GPU, one job per model and split: Llama-2-7B's nine cells against fp32 (192 GB of host memory,
10-12 h), Llama-2-13B's three against an fp16 copy (256 GB, 12 h); so do OpenLLaMA-7B's nine validation cells
without the Llama-2 licence.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.common import records
from experiments.common.checkpoints import (graph_report, load_llama, prefix_of, smoke_model, token_ids, weights_sha,
                                            wikitext, windows)
from experiments.common.cli import add_protocol_arguments, add_runtime_arguments, apply_runtime, protocol_of
from experiments.common.stats import ppl_ratio_ci
from pvi.models import hf_llama
from pvi.models.decoder import CONFIGS

CONFIG_OF = {"Qwen3-0.6B-Base": "qwen3-0.6b", "Qwen3-1.7B-Base": "qwen3-1.7b", "Qwen3-4B-Base": "qwen3-4b",
             "TinyLlama_v1.1": "tinyllama-1.1b", "open_llama_7b_v2": "llama2-7b", "Llama-2-7b-hf": "llama2-7b",
             "Llama-2-13b-hf": "llama2-13b"}
"""The benchmark configuration (``pvi.models.decoder.CONFIGS``) of each checkpoint's shape."""
PREFIX_SCORING = ("the windows' own tokens only (the targets of a run without a prefix); the fp32 reference runs on "
                  "the same [prefix | window]; the text is tokenised without the tokenizer's special tokens")
CHECK_SEQ, PROTOCOL_SEQ, CHECK_CHARS = 128, 64, 200_000
"""``--check``'s and ``--protocol``'s window lengths, and the train characters their calibration window comes from."""


def prefix_spec(value: str | None) -> str | None:
    """The ``--prefix`` value for :func:`~experiments.common.checkpoints.prefix_of` (``none``: no prefix)."""
    return None if not value or value == "none" else value


def _joint(w: torch.Tensor, prefix) -> torch.Tensor:
    return w if not prefix else torch.cat([torch.tensor([prefix], dtype=w.dtype), w], 1)


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
    for a, z in hf_llama.lean_logits(graph, w.to(device), chunk=chunk):
        x, m = _nll_argmax_chunk(z.double() * scale, w, a)
        del z
        nll += x
        am.append(m)
    return nll, torch.cat(am)


def fp_eval(model, w: torch.Tensor, chunk: int, prefix=None):
    """The same for the checkpoint: its final hidden states, then its LM head on ``chunk`` positions at a time. With
    a ``prefix`` (token ids) the checkpoint runs on ``[prefix | w]`` and only ``w``'s positions are scored, against
    the same targets."""
    hs = model.model(_joint(w, prefix).to(model.device)).last_hidden_state[:, len(prefix or ()):]
    nll, am = 0.0, []
    for a in range(0, hs.shape[1], chunk):
        x, m = _nll_argmax_chunk(model.lm_head(hs[:, a:a + chunk]).float(), w, a)
        nll += x
        am.append(m)
    return nll, torch.cat(am)


def reference(args, model, wins, prefix=None) -> dict:
    """The reference's NLL and argmaxes per window (cached in ``--ref-cache`` with its key), on ``[prefix | window]``
    with a prefix."""
    key = {"model": args.model, "split": args.split, "seq": args.seq, "windows": len(wins), "dtype": args.ref_dtype,
           "smoke": args.smoke}
    if prefix:                      # (the key of a run without a prefix has no such entry)
        key["prefix"] = list(prefix)
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
            x, a = fp_eval(m, w, args.chunk, prefix)
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


def cells(args, prefix=None) -> list[dict]:
    """The integer graphs of a run, in ``--graphs`` order: ``{"cell", "graph", "gain", "alpha", "glu_smooth",
    "glu_exp", "kw"}``, ``kw`` the keyword arguments of ``build_llama_from_hf``; with ``--exp-only`` only the smoothed
    cells' per-token-exponent variants."""
    common = dict(qk_balance=not args.no_qk_balance, prefix_tokens=list(prefix) if prefix else None)
    stats = dict(smooth_pct=args.smooth_pct, pct=args.pct, pct_att=args.pct_att, pct_glu=args.pct_glu)
    own = hf_llama.main_recipe(args.model)
    exps = list(args.glu_exps) if args.exp_only else [0] + list(args.glu_exps)
    out = []
    for graph in args.graphs:
        if graph == "scalar":
            if not args.exp_only:
                out += [{"cell": f"scalar-g{g:g}", "graph": graph, "gain": g, "alpha": None, "glu_smooth": None,
                         "glu_exp": 0, "kw": dict(norm_gain=g, **common)} for g in args.gains]
            continue
        for a in args.alphas or [own["smooth"]]:
            for gs in (args.glu_smooths or [own["glu_smooth"]]) if graph == "main" else [None]:
                label = f"main-a{a:g}-glus{gs:g}" if graph == "main" else f"smooth-a{a:g}"
                for e in exps:
                    kw = dict(hf_llama.main_recipe(args.model, a), **stats, glu_smooth=gs, glu_exp=e, **common)
                    out.append({"cell": label + (f"-exp{e}" if e else ""), "graph": graph, "gain": None, "alpha": a,
                                "glu_smooth": gs,
                                "glu_exp": e, "kw": kw})
    return out


def _emit(args, row: dict) -> None:
    records.emit(row, args.out)
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
             if k in ("cell", "draw", "prefix_tokens", "ppl", "ratio", "ratio_ci95", "top1", "block_cos_min",
                      "structure_diff", "seconds")}
    print(json.dumps(short), file=sys.stderr, flush=True)


def _gb(v):
    return None if v is None else v / 2 ** 30


def run(args, model, tok, spec) -> None:
    """The perplexity cells of ``args`` (module docstring)."""
    text = token_ids(tok, wikitext(args.split), spec)
    wins = windows(text, args.seq, args.windows)
    train = token_ids(tok, wikitext("train")[:args.train_chars], spec)
    counts = [w.shape[1] - 1 for w in wins]
    cs = args.calib_seq or args.seq
    calibs = {draw: torch.cat(windows(train, cs, args.calib_windows, start=draw * args.calib_windows))
              for draw in args.draws}
    with torch.no_grad():           # one prefix for every draw and the reference: chosen on the first draw's windows
        prefix, prefix_info = prefix_of(spec, model, calibs[args.draws[0]], tok)
    ref = reference(args, model, wins, prefix)
    fp = math.exp(sum(ref["nll"]) / sum(counts))
    name = args.config or CONFIG_OF.get(args.model.split("/")[-1])
    shape = hf_llama.config_of(model.config)
    env = records.environment(args.device)
    head = {"model": args.model, "smoke": args.smoke, "config": name, "split": args.split, "seq": args.seq,
            "windows": len(wins), "tokens": sum(counts), "dataset": f"wikitext-2-raw-v1 {args.split}, '\\n\\n'.join",
            "host": env["host"],
            "gpu": env["gpu"], "device": args.device, "calib_windows": args.calib_windows, "calib_seq": cs,
            "ref_dtype": args.ref_dtype, "shape_vs_config": (hf_llama.same_shape(shape, CONFIGS[name])
                                                             if name in CONFIGS else None),
            "prefix_spec": spec, "prefix_tokens": prefix, "prefix_len": len(prefix or ()), "env": env}
    if prefix:       # (joint_len above max_pos: the last positions lie past the trained context, for both models)
        head.update(prefix_choice=prefix_info, prefix_scoring=PREFIX_SCORING, joint_len=args.seq + len(prefix),
                    max_pos=getattr(model.config, "max_position_embeddings", None))
    _emit(args, dict(head, cell="reference", ppl=fp, nll=ref["nll"], seconds=ref["seconds"]))
    todo = cells(args, prefix)
    if not todo:
        return
    with torch.no_grad():
        hf_states = [hf_llama.hf_residual_states(model, w, prefix=prefix) for w in wins[:args.cos_windows]]
    structure_done = False
    for draw in args.draws:
        calib = calibs[draw]
        for cell in todo:
            records.reset_peaks(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                graph, info = hf_llama.build_llama_from_hf(model, calib, device=args.calib_device, **cell["kw"])
            build_s = time.perf_counter() - t0
            extra = {"digest": graph.digest().hex(), "weights_sha256": weights_sha(graph),
                     "report": graph_report(graph), "residual_real_max": info["residual_real_max"]}
            if prefix:       # per layer max |real| / (127 s) of the prefix's keys and values (above 1: clipped)
                extra.update(prefix_clip=info["prefix_clip"], prefix_hidden_max=info["prefix_hidden_max"])
            if args.structure and not structure_done:
                # against the named benchmark configuration when the checkpoint has its shape (a smoke model
                # does not: then against a benchmark graph of the model's own shape)
                same = name in CONFIGS and not hf_llama.same_shape(shape, CONFIGS[name])
                extra["structure_against"] = name if same else "own shape"
                extra["structure_diff"] = hf_llama.matches_benchmark_graph(graph, name if same else shape)
                structure_done = True
            lean, _ = graph.with_constants_on(args.device)
            t0 = time.perf_counter()
            nll, agree = [], 0
            with torch.no_grad():
                for w, r_am in zip(wins, ref["argmax"]):
                    x, am = int_eval(lean, info["logit_scale"], w, args.device, args.chunk)
                    nll.append(x)
                    agree += int((am[:-1] == r_am[:-1]).sum())
                blocks = []
                for w, states in zip(wins, hf_states):
                    ours = hf_llama.residual_states(lean, info, w.to(args.device))
                    blocks.append(hf_llama.block_cosines(ours, states))
                    del ours
            per_block = [{"mean": sum(b[i]["mean"] for b in blocks) / len(blocks),
                          "min": min(b[i]["min"] for b in blocks)} for i in range(len(blocks[0]))] if blocks else []
            ppl = math.exp(sum(nll) / sum(counts))
            peaks = records.peaks(args.device)
            _emit(args, dict(head, cell=cell["cell"], graph=cell["graph"], gain=cell["gain"], alpha=cell["alpha"],
                             glu_smooth=cell["glu_smooth"], glu_exp=cell["glu_exp"], draw=draw,
                             opts={k: v for k, v in cell["kw"].items() if k != "prefix_tokens"}, ppl=ppl, ref_ppl=fp,
                             ratio=ppl / fp, ratio_ci95=ppl_ratio_ci(nll, ref["nll"], counts),
                             top1=agree / sum(counts), block_cos=per_block,
                             block_cos_min=min((b["min"] for b in per_block), default=None),
                             block_cos_mean_min=min((b["mean"] for b in per_block), default=None),
                             s_residual=info["s_residual"], logit_scale=info["logit_scale"],
                             embed_mult=info["embed_mult"], embed_clip=info["embed_clip"], nll=nll, build_s=build_s,
                             seconds=time.perf_counter() - t0, peaks=peaks, peak_rss_GB=_gb(peaks["host_peak_rss"]),
                             peak_gpu_GB=_gb(peaks.get("gpu_peak_memory")), **extra))
            del graph, lean, info
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ------------------------------------------------------------------ checks
def _first(args) -> tuple[float | None, float | None]:
    """The checks' smoothing alpha and product balance: the first of ``--alphas`` and ``--glu-smooths`` (None: the
    checkpoint's)."""
    return (args.alphas or [None])[0], (args.glu_smooths or [None])[0]


def _check_window(tok, spec, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(the first seq-token test window, the first seq-token window of the train split's first CHECK_CHARS
    characters)``, both ``[1, seq]``."""
    w = windows(token_ids(tok, wikitext("test"), spec), seq, 1)[0]
    return w, windows(token_ids(tok, wikitext("train")[:CHECK_CHARS], spec), seq, 1)[0]


def check(args, model, tok, spec) -> dict:
    """Exactness on one :data:`CHECK_SEQ`-token test window, with the main recipe calibrated on one train window,
    for the main graph (``main``) and its per-token-exponent variant (``main-exp<E>``, whose two extra op kinds the
    ``-exp`` cells run on the GPU): the lean chunked logits equal one pass, the last position equals the plain
    forward, the pruned graph's logits equal the unpruned graph's and, on a GPU, the GPU's integers equal the CPU's.
    The ``main`` graphs' structural diffs are taken against ``build_decoder(CONFIGS[--config])`` when the checkpoint
    has that shape (``shape_vs_config == []``), else against a benchmark graph of the model's own shape (a smoke
    model).  With a prefix, every graph is built with it (chosen on the calibration window) and the reference runs
    on ``[prefix | window]``.  Returns the report; ``report["ok"]`` says whether every check holds."""
    seq = CHECK_SEQ
    w, calib = _check_window(tok, spec, seq)
    name = args.config or CONFIG_OF.get(args.model.split("/")[-1])
    shape = hf_llama.config_of(model.config)
    vs_config = hf_llama.same_shape(shape, CONFIGS[name]) if name in CONFIGS else None
    against = name if vs_config == [] else shape
    with torch.no_grad():
        prefix, prefix_info = prefix_of(spec, model, calib, tok)
    report = {"model": args.model, "smoke": args.smoke, "seq": seq, "shape_vs_config": vs_config,
              "structure_against": name if vs_config == [] else "own shape", "prefix_tokens": prefix,
              "prefix_choice": prefix_info}
    ok = args.smoke or vs_config == []              # a real checkpoint must have its CONFIGS entry's shape
    exp = args.glu_exps[0] if args.glu_exps else 8
    base = dict(hf_llama.main_recipe(args.model, *_first(args)),
                smooth_pct=args.smooth_pct, pct=args.pct, pct_att=args.pct_att, pct_glu=args.pct_glu,
                qk_balance=not args.no_qk_balance, prefix_tokens=prefix)
    for label, extra in (("main", {}), (f"main-exp{exp}", {"glu_exp": exp})):
        with torch.no_grad():
            graph, info = hf_llama.build_llama_from_hf(model, calib, **base, **extra)
            pruned, _ = hf_llama.build_llama_from_hf(model, calib, prune_last=True, **base, **extra)
            chunks = torch.cat([z for _, z in hf_llama.lean_logits(graph, w, chunk=args.chunk)], 1)
            whole = torch.cat([z for _, z in hf_llama.lean_logits(graph, w, chunk=1 << 30)], 1)
            env, _ = graph.forward(w)
            env_p, _ = pruned.forward(w)
            cell = {"opts": {k: v for k, v in dict(base, **extra).items() if k != "prefix_tokens"},
                    "chunked_equals_whole": bool(torch.equal(chunks, whole)),
                    "last_equals_forward": bool(torch.equal(whole[:, -1], env[graph.output_name][:, -1])),
                    "pruned_equals_unpruned": bool(torch.equal(env_p[pruned.output_name], env[graph.output_name])),
                    "gpu_equals_cpu": None, "glu_emax": info["glu_emax"], "meta": graph.meta}
            if prefix:
                cell["prefix_clip"] = info["prefix_clip"]
            del env, env_p, chunks
            if not extra:
                cell["structure_diff"] = hf_llama.matches_benchmark_graph(graph, against)
                cell["pruned_structure_diff"] = hf_llama.matches_benchmark_graph(pruned, against, prune_last=True)
            del pruned
            if torch.cuda.is_available():
                lean, _ = graph.with_constants_on("cuda")
                gpu = torch.cat([z.cpu() for _, z in hf_llama.lean_logits(lean, w.cuda(), chunk=args.chunk)], 1)
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
        report["ref_ppl"] = math.exp(fp_eval(model, w, args.chunk, prefix)[0] / (seq - 1))
    report["ok"] = bool(ok)
    return report


def protocol(args, model, tok, spec) -> dict:
    """The checkpoint's first 2 blocks as a real graph (``n_layers=2``, then the final norm and the LM head; the main
    recipe calibrated on one :data:`PROTOCOL_SEQ`-token train window, ``--prune-last``) on a 64-token test window,
    under the protocol flags: the structure is the 2-block benchmark graph's; v0 in mode C (the commitment plan
    ``--policy``) and in mode Kpre (the secret vectors from an interactive challenger) and V1 (mode C, the same
    commitment) accept an honest query and reject a tampered claim of the first q projection, of the middle weight op
    and of the LM head.  With a prefix, the graph carries it (chosen on the calibration window).  Returns the report;
    ``report["ok"]`` says whether every check holds."""
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.v1 import protocol as v1_protocol
    p = protocol_of(args)
    w, calib = _check_window(tok, spec, PROTOCOL_SEQ)
    with torch.no_grad():
        prefix, prefix_info = prefix_of(spec, model, calib, tok)
        graph, _ = hf_llama.build_llama_from_hf(model, calib, n_layers=2, prune_last=p.prune_last,
                                                qk_balance=not args.no_qk_balance, prefix_tokens=prefix,
                                                **hf_llama.main_recipe(args.model, *_first(args)))
    report = {"model": args.model, "smoke": args.smoke, "seq": PROTOCOL_SEQ, "prefix_tokens": prefix,
              "prefix_choice": prefix_info, "protocol": dataclasses.asdict(p), "meta": graph.meta,
              "structure_diff": hf_llama.matches_benchmark_graph(graph, hf_llama.config_of(model.config), n_layers=2,
                                                                 prune_last=p.prune_last)}
    ok = report["structure_diff"] == []
    targets = [graph.mat_ops[1].name, graph.mat_ops[len(graph.mat_ops) // 2].name, graph.mat_ops[-1].name]

    def tamper_of(target):
        def tamper(op, z):
            if op.name == target:
                z = z.clone()
                z.view(-1)[0] -= 3
            return z
        return {"tamper": tamper}

    t0 = time.perf_counter()
    coms = commit_graph(graph, p.rate, policy=p.policy, cache=args.setup_cache)
    weights = {o.name: (o.weight, o.bias) for o in graph.mat_ops}
    for mode in ("C", "Kpre", "V1"):
        params = params_for(p.lam, len(graph.mat_ops), rate=p.rate, fiat_shamir=p.fiat_shamir,
                            plan=None if mode == "Kpre" else coms.plan, cut=mode == "V1")
        if mode == "Kpre":
            prover = Prover(graph, lean=p.lean)
            v = Verifier(graph.public(), params, "Kpre", weights=weights, lean=p.lean, lookups=p.lookups)
            v.precompute(Challenger())
        else:
            prover = Prover(graph, commitments=coms, lean=p.lean)
            v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                         tables=coms.table_publics, lean=p.lean)

        def query(kw=None, prover=prover, v=v, mode=mode):
            if mode == "V1":
                return v1_protocol.run_cut_query(prover, v, w, forward_kwargs=kw)
            return run_query(prover, v, w, forward_kwargs=kw, wire=p.wire)

        honest = query()
        report[mode] = {"honest_accepted": bool(honest["accepted"]), "honest_rejected_at": honest["rejected_at"],
                        "tamper_rejected": {t: not query(tamper_of(t))["accepted"] for t in targets}}
        ok = ok and report[mode]["honest_accepted"] and all(report[mode]["tamper_rejected"].values())
    report["seconds"] = time.perf_counter() - t0
    report["ok"] = bool(ok)
    return report


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="openlm-research/open_llama_7b_v2", help="a Llama or Qwen3 checkpoint")
    ap.add_argument("--config", default=None, help="the CONFIGS key of its shape (default: from --model)")
    ap.add_argument("--split", choices=["test", "validation"], default="test",
                    help="the evaluated split (validation: check a checkpoint's alpha and product balance on text no "
                         "choice used)")
    ap.add_argument("--seq", type=int, default=2048, help="tokens per window")
    ap.add_argument("--windows", type=int, default=0, help="evaluated windows (0: every window of the split)")
    ap.add_argument("--calib-windows", type=int, default=4, help="calibration windows per draw")
    ap.add_argument("--calib-seq", type=int, default=None, help="tokens per calibration window (default: --seq)")
    ap.add_argument("--train-chars", type=int, default=4_000_000, help="characters of the train split tokenised")
    ap.add_argument("--draws", type=int, nargs="+", default=[0], help="calibration draws")
    ap.add_argument("--prefix", nargs="?", const="auto", default="auto",
                    help="the fixed public prefix: auto (PrefixQuant's rule, the main recipe's), bos, comma-separated "
                         "token ids, or none")
    ap.add_argument("--graphs", nargs="*", choices=["main", "smooth", "scalar"], default=["main"],
                    help="main: the main recipe; smooth: without the SwiGLU product balance; scalar: one norm gain "
                         "(given without a value: the reference only)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="smoothing alphas of the main and smooth graphs (default: the checkpoint's, hf_llama.MAIN_SETTINGS)")
    ap.add_argument("--glu-smooths", type=float, nargs="+", default=None,
                    help="the main graphs' SwiGLU product balance alphas (default: the checkpoint's, hf_llama.MAIN_SETTINGS)")
    ap.add_argument("--gains", type=float, nargs="+", default=[4.0], help="the scalar graphs' norm gains")
    ap.add_argument("--pct", type=float, default=hf_llama.MAIN_RECIPE["pct"],
                    help="the requantisation percentile at q/k/v/gate/up and the q/k norms")
    ap.add_argument("--pct-att", type=float, default=hf_llama.MAIN_RECIPE["pct_att"],
                    help="the requantisation percentile at the attention output")
    ap.add_argument("--pct-glu", type=float, default=None,
                    help="the requantisation percentile at the SwiGLU product (default: --pct)")
    ap.add_argument("--smooth-pct", type=float, default=hf_llama.MAIN_RECIPE["smooth_pct"],
                    help="the percentile of each channel the smoothing reads")
    ap.add_argument("--no-qk-balance", action="store_true", help="Qwen3: no q/k-norm balance per RoPE pair")
    ap.add_argument("--glu-exps", type=int, nargs="*", default=[],
                    help="also each smoothed graph with the per-token exponent 0..E at the SwiGLU product, per E (a "
                         "possible improvement, not the main algorithm)")
    ap.add_argument("--exp-only", action="store_true",
                    help="with --glu-exps, only the per-token-exponent variants (not the cells without it)")
    ap.add_argument("--calib-device", default="cpu", help="where the integer calibration pass runs")
    ap.add_argument("--ref-device", default=None, help="where the reference runs (default: --device)")
    ap.add_argument("--ref-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--ref-cache", default=None, help="a file keeping the reference's NLL and argmaxes between runs")
    ap.add_argument("--cos-windows", type=int, default=1, help="windows of the per-block cosines")
    ap.add_argument("--chunk", type=int, default=256, help="positions per LM-head chunk")
    ap.add_argument("--structure", action="store_true", help="the structural diff against the benchmark graph")
    ap.add_argument("--out", default=None, help="a JSONL file the records are appended to")
    ap.add_argument("--smoke", action="store_true", help="a 2-block random model with the checkpoint's vocabulary")
    ap.add_argument("--check", action="store_true", help="exactness checks on one 128-token window, then stop")
    ap.add_argument("--protocol", action="store_true",
                    help="the first 2 blocks as a real graph through v0 (C, Kpre) and V1 under the protocol flags, "
                         "then stop")
    add_protocol_arguments(ap)
    add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    if args.glu_smooths and "main" not in args.graphs:
        ap.error("--glu-smooths sets the main graphs' product balance: add main to --graphs")
    if args.exp_only and not args.glu_exps:
        ap.error("--exp-only keeps the per-token-exponent variants: give their exponents with --glu-exps")
    apply_runtime(args)
    spec = prefix_spec(args.prefix)
    model, tok = smoke_model(args.model) if args.smoke else load_llama(args.model)
    for flag, fn, word in ((args.check, check, "CHECK"), (args.protocol, protocol, "PROTOCOL")):
        if flag:
            report = fn(args, model, tok, spec)
            records.emit(dict(report, env=records.environment(args.device)), args.out)
            print(word, "PASSED" if report["ok"] else "FAILED", flush=True)
            raise SystemExit(0 if report["ok"] else 1)
    run(args, model, tok, spec)


if __name__ == "__main__":
    main()
