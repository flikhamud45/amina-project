"""Real-weight cells: the ``--real`` option of every script whose cells otherwise take random weights of a model's
shapes, the settings it refuses, the checkpoint's integer graph (of every block or of the first ``n``) and its
WikiText-2 prompts, and the rule that decides which of the paper's cells are real (:func:`main_checkpoint`).

A checkpoint is loaded in fp32 and calibrated by :func:`experiments.common.checkpoints.load_real` and its integer graph
built by :func:`experiments.common.checkpoints.real_graph`: an OPT checkpoint with the certified recipe
(``hf_opt.CERTIFIED_RECIPE``, calibrated on four 2,048-token windows of the WikiText-2 train split) or, with
``--recipe legacy``, the percentile recipe; a Llama-family checkpoint (Llama, Qwen3) with ``hf_llama.MAIN_RECIPE`` and
its fixed public prefix.  A build of ``n`` blocks (``--layers n``, ``bench.py --builds``) is the checkpoint's first
``n`` blocks, then the final norm and the LM head, for both families: the residual stream's scale is set on the whole
checkpoint, so these blocks are the full graph's first ``n``, op for op.  The prompts are the WikiText-2 test split's
windows from its start, the queries of ``perf.py --real``: ``B`` prompts of ``T`` tokens are the first ``B`` windows of
``T`` tokens (:func:`prompts`).

* :func:`add_arguments`: ``--real``, and with ``recipes`` ``--recipe``, ``--smooth`` and ``--calib-seq``;
* :func:`check` and :func:`check_args`: the refusals that need no checkpoint -- shapes without an integer builder of a
  real checkpoint (GPT-2; OPT-350M, whose embedding is projected), the legacy recipe on a Llama-family model or its
  options without it, an attention probabilities' scale other than the paper's (a checkpoint's recipe fixes it), a
  block count outside the model's, and a prompt longer than the model's trained context;
* :func:`checkpoint_config`: the same refusals of a checkpoint's ``config.json`` alone (its family and shapes), for
  the scripts that run the checkpoint's float weights;
* :func:`load`: the checkpoint, its calibration and queries, checked against the shapes ``--model`` names and against
  the checkpoint's own context (``max_position_embeddings``; a Llama-family recipe's prefix comes before each query);
* :func:`model_ops`: the whole model's weight-op shapes as a real graph has them, for mode C's plan of a build of a
  few blocks (a real OPT's LM head has one more column than the benchmark's: its final LayerNorm's folded bias);
* :func:`fields`: the records' ``real`` and ``recipe`` (and with the recipe options ``smooth`` and ``calib_seq``);
* :func:`main_checkpoint`: the checkpoint of a paper cell (``package/run.sh``'s ``real_checkpoint`` implements the
  same rule), or None where the cell keeps random weights.
"""
from __future__ import annotations

import dataclasses

import torch

from experiments.common import checkpoints, cli

__all__ = ["HELP", "add_arguments", "family", "check", "check_args", "partial", "hf_config", "checkpoint_config",
           "Checkpoint", "load", "prompts", "model_ops", "fields", "recipe_fields", "CHECKPOINTS", "CONTEXT",
           "LICENSED", "LONG_CURVE", "main_checkpoint"]

HELP = ("decoders: a Hugging Face checkpoint (e.g. meta-llama/Llama-2-7b-hf), its integer graph queried on WikiText-2 "
        "windows; --model names its shapes (default: random weights)")


def add_arguments(ap, help: str = HELP, recipes: bool = False) -> None:
    """``--real`` (``""``: random weights) and, with ``recipes``, ``--recipe``, ``--smooth`` and ``--calib-seq``."""
    ap.add_argument("--real", default="", metavar="CHECKPOINT", help=help)
    if not recipes:
        return
    ap.add_argument("--recipe", choices=["certified", "legacy"], default="certified",
                    help="with an OPT --real: certified = hf_opt.CERTIFIED_RECIPE calibrated on four 2,048-token "
                         "train windows; legacy = the percentile recipe (norm gain 4, --smooth, --calib-seq)")
    ap.add_argument("--smooth", type=float, default=None, help="with --recipe legacy: SmoothQuant strength through "
                    "the norm gains (0.55 with experiments.common.checkpoints.SMOOTH's clipping)")
    ap.add_argument("--calib-seq", type=int, default=0,
                    help="with --recipe legacy: the calibration window's tokens (0: --seq)")


def family(cfg) -> str | None:
    """The integer builder of a real checkpoint of ``cfg``'s shapes: ``opt`` (``hf_opt.build_opt_from_hf``),
    ``llama`` (``hf_llama.build_llama_from_hf``: Llama and Qwen3) or None (GPT-2; OPT-350M, whose embedding is
    projected)."""
    if cfg.norm == "rmsnorm" and cfg.mlp == "swiglu" and cfg.pos == "rope":
        return "llama"
    if cfg.norm == "layernorm" and cfg.mlp == "relu" and cfg.pos == "learned" and not cfg.embed_dim:
        return "opt"
    return None


def check(cfg, *, seqs=(), layers=(), prob_max: int | None = None, recipe: str = "certified",
          smooth: float | None = None, calib_seq: int = 0) -> str:
    """Refuse (``SystemExit``) a real-weight cell of ``cfg``'s shapes that cannot be built or would be mislabelled,
    before anything is loaded: no builder, the legacy recipe's options without it or on a Llama-family model, an
    attention probabilities' scale ``prob_max`` other than the paper's, a block count of ``layers`` outside
    ``0 .. cfg.n_layers`` (0: every block), a prompt of ``seqs`` beyond the trained context.  Returns the family
    (:func:`family`)."""
    fam = family(cfg)
    if fam is None:
        raise SystemExit(f"--real: {cfg.name}'s shapes have no integer builder of a real checkpoint (OPT: "
                         "pvi.models.hf_opt; Llama and Qwen3: pvi.models.hf_llama); its cells take random weights "
                         "only")
    if recipe == "certified" and (smooth is not None or calib_seq):
        raise SystemExit("--smooth and --calib-seq are the legacy recipe's: pass --recipe legacy")
    if fam == "llama" and recipe != "certified":
        raise SystemExit("--recipe legacy is an OPT recipe: a Llama-family checkpoint is built with "
                         "hf_llama.MAIN_RECIPE")
    if recipe == "legacy" and not calib_seq and len(set(seqs)) != 1:
        raise SystemExit("--recipe legacy here needs --calib-seq (no single --seq to take it from)")
    if prob_max is not None and prob_max != cli.PAPER.prob_max:
        raise SystemExit("--prob-max is for random-weight decoders: a checkpoint's recipe fixes its scale")
    for n in layers:
        if not 0 <= (n or 0) <= cfg.n_layers:
            raise SystemExit(f"--layers {n}: {cfg.name} has {cfg.n_layers} blocks")
    long = sorted({s for s in seqs if s > cfg.max_pos})
    if long:
        raise SystemExit(f"--real: prompts of {', '.join(map(str, long))} tokens exceed {cfg.name}'s trained "
                         f"context of {cfg.max_pos} positions")
    return fam


def check_args(args, cfg, *, seqs=(), layers=(), prob_max: int | None = None) -> str | None:
    """:func:`check` of a script's parsed ``args`` (``real`` and, where the script has them, ``recipe``, ``smooth``
    and ``calib_seq``); without ``--real`` only the recipe options are refused (they need a checkpoint).  Returns the
    family, or None for random weights."""
    recipe, smooth, calib_seq = (getattr(args, k, d) for k, d in (("recipe", "certified"), ("smooth", None),
                                                                    ("calib_seq", 0)))
    if not args.real:
        if recipe != "certified" or smooth is not None or calib_seq:
            raise SystemExit("--recipe, --smooth and --calib-seq are the options of a real checkpoint: pass --real")
        return None
    return check(cfg, seqs=seqs, layers=layers, prob_max=prob_max, recipe=recipe, smooth=smooth, calib_seq=calib_seq)


def partial(cfg, layers: int) -> int | None:
    """The block count to pass to :func:`experiments.common.checkpoints.real_graph` for ``--layers layers`` of a
    checkpoint of ``cfg``'s shapes: None for the full model (``layers`` 0 or ``cfg.n_layers``), else ``layers`` (the
    checkpoint's first blocks, OPT and Llama-family alike); refuses a block count outside ``1 .. cfg.n_layers``."""
    if not layers or layers == cfg.n_layers:
        return None
    if not 1 <= layers <= cfg.n_layers:
        raise SystemExit(f"--layers {layers}: {cfg.name} has {cfg.n_layers} blocks")
    return layers


# ------------------------------------------------------------------ a checkpoint's configuration alone
def hf_config(name: str):
    """The Hugging Face configuration of the checkpoint ``name`` (its ``config.json`` alone)."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(name)


def _shapes(hf, name: str):
    """The ``DecoderConfig`` (named ``name``) of an OPT, Llama or Qwen3 configuration's shapes."""
    from pvi.models import hf_llama
    from pvi.models.decoder import DecoderConfig
    if hf.model_type != "opt":
        return hf_llama.config_of(hf, name)
    d, h = hf.hidden_size, hf.num_attention_heads
    return DecoderConfig(name, d, hf.num_hidden_layers, h, h, d // h, hf.ffn_dim, hf.vocab_size,
                         mlp=hf.activation_function, bias=bool(getattr(hf, "enable_bias", True)),
                         tied=bool(hf.tie_word_embeddings), max_pos=hf.max_position_embeddings,
                         embed_dim=0 if hf.word_embed_proj_dim == d else hf.word_embed_proj_dim)


def _problems(hf, name: str, cfg) -> list[str]:
    """How the checkpoint ``name`` of configuration ``hf`` differs from ``cfg``: another builder family, or other
    shapes (the context aside)."""
    fam = {"opt": "opt", "llama": "llama", "qwen3": "llama"}.get(hf.model_type)
    if fam is None or fam != family(cfg):
        return [f"a checkpoint of model type {hf.model_type!r}, while --model {cfg.name} names shapes of "
                f"{family(cfg)}"]
    from pvi.models import hf_llama
    got = _shapes(hf, cfg.name)
    problems = hf_llama.same_shape(dataclasses.replace(got, max_pos=cfg.max_pos), cfg)
    if fam == "opt" and got.embed_dim:
        problems.append(f"a projected embedding of {got.embed_dim} dimensions")
    return problems


def checkpoint_config(name: str, cfg, families=("opt", "llama")):
    """``cfg`` with the checkpoint ``name``'s trained context as ``max_pos``, from its ``config.json`` alone
    (:func:`hf_config`); refuses (``SystemExit``) a checkpoint of a family outside ``families`` (``opt``, ``llama``:
    Llama and Qwen3), of another family than ``cfg``'s shapes, or of other shapes."""
    hf = hf_config(name)
    if family(cfg) is None:
        check(cfg)
    problems = _problems(hf, name, cfg)
    if problems:
        raise SystemExit(f"--real {name} is not of --model {cfg.name}'s shapes: {'; '.join(problems)}")
    if family(cfg) not in families:
        raise SystemExit(f"--real {name}: a {hf.model_type} checkpoint; this script runs "
                         f"{' and '.join(families)} checkpoints only")
    return dataclasses.replace(cfg, max_pos=hf.max_position_embeddings)


# ------------------------------------------------------------------ the loaded checkpoint
class Checkpoint:
    """A loaded checkpoint: ``name``, ``family``, the fp32 ``model`` (None once :meth:`free` dropped it), its
    calibration ``calib`` (:func:`experiments.common.checkpoints.load_real`), ``queries`` (its WikiText-2 test
    windows ``[1, seq]``), ``stream`` (those windows joined, 1-D), ``recipe`` (``main``, ``certified`` or ``legacy``:
    :func:`experiments.common.checkpoints.recipe_name`), ``calib_seq`` (the legacy recipe's window, else None),
    ``prefix_len`` (a Llama-family checkpoint's fixed prefix, else 0) and ``n_layers`` (its blocks)."""

    def __init__(self, name: str, fam: str, model, queries, calib, recipe: str, calib_seq: int | None) -> None:
        self.name, self.family, self.model, self.queries, self.calib = name, fam, model, queries, calib
        self.recipe, self.calib_seq = recipe, calib_seq
        self.stream = torch.cat([q.reshape(-1) for q in queries]) if queries else torch.zeros(0, dtype=torch.int64)
        self.prefix_len = len(getattr(calib, "prefix", None) or [])
        self.n_layers = int(model.config.num_hidden_layers)

    def graph(self, prune_last: bool, n_layers: int | None = None, smooth: float | None = None):
        """``(graph, info)``: the integer graph of the first ``n_layers`` blocks (None, 0 or the checkpoint's: all),
        under the checkpoint's recipe (``smooth``: the legacy recipe's SmoothQuant strength)."""
        if self.model is None:
            raise RuntimeError("the checkpoint's fp32 model was freed")
        part = None if not n_layers or n_layers == self.n_layers else n_layers
        recipe = "legacy" if self.recipe == "legacy" else "certified"
        return checkpoints.real_graph(self.model, self.calib, prune_last, recipe, smooth, n_layers=part)

    def free(self) -> None:
        """Drop the fp32 model (the graphs built from it keep their own integer weights)."""
        self.model = None


def load(name: str, cfg, *, seq: int, n_queries: int, recipe: str = "certified", calib_seq: int = 0,
         generated: int = 0) -> Checkpoint:
    """The checkpoint ``name`` (fp32), its calibration (``recipe``; the legacy one on a window of ``calib_seq``
    tokens, 0: ``seq``) and ``n_queries`` WikiText-2 test windows of ``seq`` tokens
    (:func:`experiments.common.checkpoints.load_real`).  Refuses a checkpoint of another family or other shapes than
    ``cfg``'s, and prompts of ``seq`` tokens, with ``generated`` more positions after them, that would run past its
    ``max_position_embeddings`` behind its fixed prefix."""
    model, queries, calib = checkpoints.load_real(name, seq, n_queries, calib_seq, recipe)
    hf = model.config
    problems = _problems(hf, name, cfg)
    if problems:
        raise SystemExit(f"--real {name} is not of --model {cfg.name}'s shapes: {'; '.join(problems)}")
    prefix = len(getattr(calib, "prefix", None) or [])
    ctx = getattr(hf, "max_position_embeddings", None)
    if ctx is not None and seq + generated + prefix > ctx:
        what = f"{seq} tokens" + (f" and {generated} generated" if generated else "")
        behind = f" behind its {prefix}-token fixed prefix" if prefix else ""
        raise SystemExit(f"--real {name}: prompts of {what}{behind} exceed its trained context of {ctx} positions")
    fam = family(cfg)
    legacy = recipe == "legacy" and fam == "opt"
    return Checkpoint(name, fam, model, list(queries), calib, checkpoints.recipe_name(calib, recipe),
                      (calib_seq or seq) if legacy else None)


def prompts(ck: Checkpoint, seq: int, n: int) -> torch.Tensor:
    """``[n, seq]``: the first ``n`` WikiText-2 test windows of ``seq`` tokens (``perf.py --real``'s queries)."""
    if n * seq > ck.stream.numel():
        raise ValueError(f"{n} prompts of {seq} tokens need {n * seq} tokens, {ck.stream.numel()} were loaded")
    return ck.stream[:n * seq].reshape(n, seq).clone()


def model_ops(cfg, prune_last: bool, real: bool = True) -> list:
    """The whole model's weight-op shapes (:func:`pvi.commit.cost.decoder_shapes`) as its graph has them: a real
    OPT checkpoint's LM head reads its final LayerNorm's folded bias (one more column, ``has_bias``); a real
    Llama-family head, and every random-weight graph, has the benchmark's shapes."""
    from pvi.commit.cost import decoder_shapes
    ops = decoder_shapes(cfg, prune_last=prune_last)
    if real and family(cfg) == "opt":
        ops = [dataclasses.replace(s, row_length=s.row_length + 1, has_bias=True) if s.name == "head" else s
               for s in ops]
    return ops


def fields(ck: Checkpoint | None, smooth: float | None = None, recipe_options: bool = False) -> dict:
    """The record's ``real`` (the checkpoint; None for random weights) and ``recipe`` (None for random weights); with
    ``recipe_options`` also ``smooth`` and ``calib_seq`` (the legacy recipe's; else None)."""
    out = {"real": ck.name if ck is not None else None, "recipe": ck.recipe if ck is not None else None}
    if recipe_options:
        legacy = ck is not None and ck.calib_seq is not None
        out.update(smooth=smooth if legacy else None, calib_seq=ck.calib_seq if legacy else None)
    return out


def recipe_fields(ck: Checkpoint | None, args) -> dict:
    """:func:`fields` with ``smooth`` and ``calib_seq``, of a script with the recipe options (``args.smooth``)."""
    return fields(ck, args.smooth, recipe_options=True)


# ------------------------------------------------------------------ the paper's cells
CHECKPOINTS = {"llama2-7b": "meta-llama/Llama-2-7b-hf", "llama2-13b": "meta-llama/Llama-2-13b-hf",
               "opt-125m": "facebook/opt-125m", "opt-1.3b": "facebook/opt-1.3b", "opt-6.7b": "facebook/opt-6.7b",
               "qwen3-4b": "Qwen/Qwen3-4B-Base"}
"""The checkpoint of each model of the paper's cost cells that has one (GPT-2 and the image models: none)."""
CONTEXT = {"llama2-7b": 4096, "llama2-13b": 4096, "opt-125m": 2048, "opt-1.3b": 2048, "opt-6.7b": 2048,
           "qwen3-4b": 32768}
"""Each checkpoint's trained context (its ``config.json``'s ``max_position_embeddings``)."""
LICENSED = ("llama2-7b", "llama2-13b")
"""The models whose checkpoint needs its licence accepted (``run.sh --llama-licence``)."""
PREFIX_MAX = 64
"""An upper bound on a Llama-family recipe's fixed prefix (PrefixQuant's rule picks a few tokens), which runs in front
of each prompt."""
LONG_CURVE = ("l-", "g-long-", "v1-codec-llama7-", "gate-l-")
"""The jobs (``run.sh``'s names without ``sp-``) of the curves that reach beyond Llama-2's trained context, whose every
cell keeps random weights: the long tier, the GPU verifier's long cells, V1's codec curve and the gate's long cells."""


def main_checkpoint(model: str, seq: int = 0, licence: bool = True, gen: int = 0,
                    long_curve: bool = False) -> str | None:
    """The checkpoint of the paper's cell of ``model`` at a prompt of ``seq`` tokens (``gen``: tokens generated after
    it; ``seq`` 0: a cell without a prompt, such as the setup's), or None where the cell keeps random weights of the
    model's shapes: a model without a checkpoint (GPT-2, the image models), Llama-2 without ``licence`` (``run.sh
    --llama-licence``), a prompt that with the generated tokens and, for a Llama-family model, the fixed prefix does
    not fit the checkpoint's trained context (:data:`CONTEXT`; Llama-2: cells up to 2,048 tokens are real, the
    4,096-token cells random), and every cell of a curve that reaches beyond that context (``long_curve``,
    :data:`LONG_CURVE`: the long-prompt cells from 2,048 to 16,384 tokens, their re-execution and GPU cells, and V1's
    codec curve), so that each curve is measured on one kind of weights.  ``run.sh``'s ``rw`` implements the same
    rule for the jobs (``tests/test_run_package.py`` compares them)."""
    name = CHECKPOINTS.get(model)
    if name is None or long_curve or (model in LICENSED and not licence):
        return None
    from pvi.models.decoder import CONFIGS
    prefix = PREFIX_MAX if family(CONFIGS[model]) == "llama" else 0
    if seq + max(gen - 1, 0) + prefix > CONTEXT[model]:
        return None
    return name
