"""The real-weight option (``--real``) of the scripts that time one part of the protocol on a decoder of a model's
shapes (``experiments/improvements/*_ab.py``, ``gpu_profile.py``, ``experiments/v1/v1_phases.py``): the integer graph
of a Hugging Face checkpoint, or of its first blocks, queried on WikiText-2 windows, as ``costs/perf.py`` and
``v1/v1_bytes.py`` build their real-weight cells (:func:`experiments.common.checkpoints.load_real` and
:func:`~experiments.common.checkpoints.real_graph`).

* :func:`add_arguments`: ``--real``, ``--recipe``, ``--smooth`` and ``--calib-seq``, as ``perf.py`` has them;
* :func:`check`: the refusals that need no checkpoint -- shapes without an integer builder of a real checkpoint (GPT-2;
  OPT-350M, whose embedding is projected), the legacy recipe on a Llama-family model, ``--prob-max`` (a checkpoint's
  recipe fixes it), more blocks than the model has, and a prompt longer than the model's trained context;
* :func:`load`: the checkpoint in fp32, its calibration (OPT: ``hf_opt.CERTIFIED_RECIPE``'s four 2,048-token train
  windows, or the legacy recipe's window; a Llama-family checkpoint: ``hf_llama.MAIN_RECIPE``'s and its fixed public
  prefix) and its queries, checked against the shapes ``--model`` names and against the checkpoint's own context
  (``max_position_embeddings``; the prefix's positions come before a Llama-family query's);
* :func:`graph`: the integer graph of the checkpoint's first ``n`` blocks (0: every block), the final norm and the LM
  head, built as if the checkpoint had those blocks only (:func:`first_blocks`: every calibration statistic is that
  of the first ``n`` blocks; a Llama-family checkpoint keeps the prefix chosen on the whole model);
* :func:`model_ops`: the whole model's weight-op shapes as the real graph has them, for mode C's plan of a graph of
  a few blocks (a real OPT's LM head has one more column than the benchmark's: its final LayerNorm's folded bias);
* :func:`fields`: the records' ``real`` (the checkpoint; None for random weights), ``recipe`` (``certified`` or
  ``legacy`` for OPT, ``main`` for a Llama-family checkpoint; None for random weights), ``smooth`` and ``calib_seq``
  (the legacy recipe's; else None).
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
from types import SimpleNamespace

import torch

from experiments.common import checkpoints, cli

__all__ = ["add_arguments", "builder_of", "check", "load", "first_blocks", "graph", "model_ops", "fields"]


def add_arguments(ap) -> None:
    """``--real``, ``--recipe``, ``--smooth`` and ``--calib-seq`` (``costs/perf.py``'s options)."""
    ap.add_argument("--real", default="", help="a Hugging Face checkpoint (e.g. facebook/opt-1.3b): its integer graph "
                    "queried on WikiText-2 windows; --model names its shapes")
    ap.add_argument("--recipe", choices=["certified", "legacy"], default="certified",
                    help="with an OPT --real: certified = hf_opt.CERTIFIED_RECIPE calibrated on four 2,048-token "
                         "train windows; legacy = the percentile recipe (norm gain 4, --smooth, --calib-seq)")
    ap.add_argument("--smooth", type=float, default=None, help="with --recipe legacy: SmoothQuant strength through "
                    "the norm gains (0.55 with experiments.common.checkpoints.SMOOTH's clipping)")
    ap.add_argument("--calib-seq", type=int, default=0,
                    help="with --recipe legacy: the calibration window's tokens (0: --seq)")


def builder_of(cfg) -> str | None:
    """The integer builder of a real checkpoint of ``cfg``'s shapes: ``opt`` (``hf_opt.build_opt_from_hf``),
    ``llama`` (``hf_llama.build_llama_from_hf``: Llama and Qwen3) or None (GPT-2, OPT-350M)."""
    if cfg.norm == "rmsnorm" and cfg.mlp == "swiglu" and cfg.pos == "rope":
        return "llama"
    if cfg.norm == "layernorm" and cfg.mlp == "relu" and cfg.pos == "learned" and not cfg.embed_dim:
        return "opt"
    return None


def check(args, cfg, *, layers=(0,), seqs=(), prob_max=None) -> None:
    """Refuse (``SystemExit``) the settings that have no real-weight graph, before anything is loaded (module
    docstring); ``layers``: the block counts the script builds (0: every block), ``seqs``: its prompt lengths,
    ``prob_max``: its ``--prob-max`` (None: the script has none)."""
    if not args.real:
        if args.recipe != "certified" or args.smooth is not None or args.calib_seq:
            raise SystemExit("--recipe, --smooth and --calib-seq are the options of a real checkpoint: pass --real")
        return
    if args.recipe == "certified" and (args.smooth is not None or args.calib_seq):
        raise SystemExit("--smooth and --calib-seq are the legacy recipe's: pass --recipe legacy")
    family = builder_of(cfg)
    if family is None:
        raise SystemExit(f"--real: {cfg.name}'s shapes have no integer builder of a real checkpoint (OPT and "
                         f"Llama-family checkpoints only); its cells run on random weights of its shapes")
    if family == "llama" and args.recipe != "certified":
        raise SystemExit("--recipe legacy is an OPT recipe: a Llama-family checkpoint is built with "
                         "hf_llama.MAIN_RECIPE")
    if args.recipe == "legacy" and not args.calib_seq and len(set(seqs)) != 1:
        raise SystemExit("--recipe legacy here needs --calib-seq (no single --seq to take it from)")
    if prob_max is not None and prob_max != cli.PAPER.prob_max:
        raise SystemExit("--prob-max is for random-weight decoders: a checkpoint's recipe fixes its scale")
    for n in layers:
        if not 0 <= n <= cfg.n_layers:
            raise SystemExit(f"--layers {n}: {cfg.name} has {cfg.n_layers} blocks")
    for s in seqs:
        if s > cfg.max_pos:
            raise SystemExit(f"--seq {s} is beyond {cfg.name}'s trained context of {cfg.max_pos} positions")


def _owner(model):
    """The module that holds a loaded checkpoint's block list (``layers``)."""
    return model.model.decoder if model.config.model_type == "opt" else model.model


def _shape_problems(hf, cfg) -> list[str]:
    """How a loaded checkpoint's configuration ``hf`` differs from the shapes ``cfg``."""
    if hf.model_type == "opt":
        got = {"d_model": hf.hidden_size, "n_layers": hf.num_hidden_layers, "n_heads": hf.num_attention_heads,
               "d_ff": hf.ffn_dim, "vocab": hf.vocab_size}
        problems = [f"{k}: {v} vs {getattr(cfg, k)}" for k, v in got.items() if v != getattr(cfg, k)]
        if hf.word_embed_proj_dim != hf.hidden_size:
            problems.append(f"a projected embedding of {hf.word_embed_proj_dim} dimensions")
        return problems
    from pvi.models import hf_llama
    return hf_llama.same_shape(hf_llama.config_of(hf), cfg)


def load(args, cfg, *, seq: int, n_queries: int, layers=(0,)) -> SimpleNamespace:
    """The checkpoint ``args.real`` (fp32), its calibration and ``n_queries`` WikiText-2 test windows of ``seq``
    tokens (:func:`experiments.common.checkpoints.load_real` with ``args.recipe`` and ``args.calib_seq``), checked
    against ``cfg`` and against the checkpoint's context; ``layers``: the block counts the script builds, of which
    only the first ``max(layers)`` blocks are kept (0: every block).  Returns ``model``, ``queries``, ``calib``,
    ``recipe`` (:func:`experiments.common.checkpoints.recipe_name`), ``calib_seq`` (the legacy recipe's window, else
    None) and ``prefix_len`` (a Llama-family checkpoint's fixed prefix, else 0)."""
    model, queries, calib = checkpoints.load_real(args.real, seq, n_queries, args.calib_seq, args.recipe)
    hf = model.config
    family = {"opt": "opt", "llama": "llama", "qwen3": "llama"}.get(hf.model_type)
    if family != builder_of(cfg):
        raise SystemExit(f"--real {args.real}: a checkpoint of model type {hf.model_type!r}, while --model "
                         f"{cfg.name} names shapes of {builder_of(cfg)}")
    problems = _shape_problems(hf, cfg)
    if problems:
        raise SystemExit(f"--real {args.real} is not of --model {cfg.name}'s shapes: {'; '.join(problems)}")
    prefix_len = len(getattr(calib, "prefix", None) or [])
    if seq + prefix_len > hf.max_position_embeddings:
        raise SystemExit(f"--seq {seq}{f' after a prefix of {prefix_len} tokens' if prefix_len else ''} is beyond "
                         f"{args.real}'s trained context of {hf.max_position_embeddings} positions")
    keep = 0 if 0 in layers else max(layers)
    if keep and keep < hf.num_hidden_layers:   # the blocks no build reads (the prefix is chosen on every block)
        owner = _owner(model)
        owner.layers = torch.nn.ModuleList(list(owner.layers)[:keep])
        hf.num_hidden_layers = keep
        gc.collect()
    legacy = args.recipe == "legacy" and family == "opt"
    return SimpleNamespace(model=model, queries=queries, calib=calib,
                           recipe=checkpoints.recipe_name(calib, args.recipe),
                           calib_seq=(args.calib_seq or seq) if legacy else None, prefix_len=prefix_len)


@contextlib.contextmanager
def first_blocks(model, n: int):
    """``model`` (a loaded OPT or Llama-family checkpoint) cut to its first ``n`` blocks (0, or every block: as it
    is) while the ``with`` block runs: its block list and ``config.num_hidden_layers``, then put back."""
    owner, hf = _owner(model), model.config
    blocks, count = owner.layers, hf.num_hidden_layers
    if n and n < len(blocks):
        owner.layers, hf.num_hidden_layers = torch.nn.ModuleList(list(blocks)[:n]), n
    try:
        yield model
    finally:
        owner.layers, hf.num_hidden_layers = blocks, count


def graph(real: SimpleNamespace, args, n_layers: int, prune_last: bool):
    """The integer graph of the checkpoint's first ``n_layers`` blocks (0: every block) under ``args.recipe`` and
    ``args.smooth`` (:func:`experiments.common.checkpoints.real_graph`), calibrated as if the checkpoint had those
    blocks only; ``prune_last``: its last block at the last position only."""
    with first_blocks(real.model, n_layers) as model:
        g, _ = checkpoints.real_graph(model, real.calib, prune_last, args.recipe, args.smooth)
    return g


def model_ops(cfg, prune_last: bool, real: bool) -> list:
    """The whole model's weight-op shapes (:func:`pvi.commit.cost.decoder_shapes`) as its graph has them: a real
    OPT checkpoint's LM head reads its final LayerNorm's folded bias (one more column, ``has_bias``), a real
    Llama-family head has the benchmark's shape."""
    from pvi.commit.cost import decoder_shapes
    ops = decoder_shapes(cfg, prune_last=prune_last)
    if real and builder_of(cfg) == "opt":
        ops = [dataclasses.replace(s, row_length=s.row_length + 1, has_bias=True) if s.name == "head" else s
               for s in ops]
    return ops


def fields(args, real: SimpleNamespace | None = None) -> dict:
    """The records' ``real``, ``recipe``, ``smooth`` and ``calib_seq`` (``real``: :func:`load`'s result; None for
    random weights)."""
    legacy = real is not None and real.calib_seq is not None
    return {"real": args.real or None, "recipe": real.recipe if real is not None else None,
            "smooth": args.smooth if legacy else None, "calib_seq": real.calib_seq if legacy else None}
