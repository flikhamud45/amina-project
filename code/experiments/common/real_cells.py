"""Real-weight cells of the cost scripts whose cells otherwise take random weights of a model's shapes
(``costs/bench.py``, ``costs/setup_timing.py``, ``deployment/serve_ab.py``): the ``--real`` option, the settings it
refuses, and the checkpoint's integer graph and WikiText-2 prompts, built as ``costs/perf.py`` and ``v1/v1_bytes.py``
build them.

A checkpoint is loaded in fp32 and calibrated by :func:`experiments.common.checkpoints.load_real` and its integer graph
built by :func:`experiments.common.checkpoints.real_graph`: an OPT checkpoint with the certified recipe
(``hf_opt.CERTIFIED_RECIPE``, calibrated on four 2,048-token windows of the WikiText-2 train split), a Llama-family
checkpoint (Llama, Qwen3) with ``hf_llama.MAIN_RECIPE`` and its fixed public prefix.  The prompts are the WikiText-2
test split's windows from its start, the queries of ``perf.py --real``: ``B`` prompts of ``T`` tokens are the first
``B`` windows of ``T`` tokens (:func:`prompts`), whatever other prompt lengths the job measures.

Refused: a decoder family without an integer builder of a real checkpoint (GPT-2; OPT-350M, whose embedding is
projected), a checkpoint whose blocks or width are not those of ``--model``, a prompt longer than the checkpoint's
trained context (``DecoderConfig.max_pos``, and the checkpoint's own ``max_position_embeddings`` with the fixed
prefix in front), an OPT build of fewer blocks than the checkpoint's (the OPT builder takes every block; a Llama-family
build of ``n`` blocks is the checkpoint's first ``n``), and an attention probabilities' scale other than the paper's
(a checkpoint's recipe fixes it).
"""
from __future__ import annotations

import torch

from experiments.common import checkpoints, cli

__all__ = ["HELP", "add_argument", "family", "check", "Checkpoint", "load", "prompts", "model_ops", "fields"]

HELP = ("decoders: a Hugging Face checkpoint (e.g. meta-llama/Llama-2-7b-hf), its integer graph queried on WikiText-2 "
        "windows; --model names its shapes (default: random weights)")


def add_argument(ap, help: str = HELP) -> None:
    """The ``--real`` option (``""``: random weights)."""
    ap.add_argument("--real", default="", metavar="CHECKPOINT", help=help)


def family(cfg) -> str | None:
    """``"opt"`` or ``"llama"`` (Llama, Qwen3): the integer builder of a real checkpoint of ``cfg``'s shapes; None for
    a family without one (GPT-2)."""
    if cfg.norm == "layernorm" and cfg.mlp == "relu":
        return "opt"
    if cfg.norm == "rmsnorm" and cfg.mlp == "swiglu":
        return "llama"
    return None


def check(cfg, *, seqs=(), n_layers=(), prob_max: int | None = None) -> str:
    """Refuse (``SystemExit``) a real-weight cell of ``cfg``'s shapes that cannot be built or would be mislabelled:
    no builder, a prompt of ``seqs`` beyond the trained context, an OPT build of ``n_layers`` other than the full
    model, an attention probabilities' scale other than the paper's.  Returns the family (:func:`family`)."""
    fam = family(cfg)
    if fam is None:
        raise SystemExit(f"--real: {cfg.name} has no integer builder of a real checkpoint (OPT: pvi.models.hf_opt; "
                         "Llama and Qwen3: pvi.models.hf_llama); its cells take random weights only")
    if cfg.embed_dim:
        raise SystemExit(f"--real: {cfg.name} projects its embedding (project_in/out), which hf_opt does not build")
    long = sorted({s for s in seqs if s > cfg.max_pos})
    if long:
        raise SystemExit(f"--real: prompts of {', '.join(map(str, long))} tokens exceed {cfg.name}'s trained "
                         f"context of {cfg.max_pos} tokens")
    if fam == "opt" and any(n != cfg.n_layers for n in n_layers):
        raise SystemExit(f"--real: the OPT builder (hf_opt.build_opt_from_hf) takes every block of the checkpoint: "
                         f"build the full model ({cfg.n_layers} blocks)")
    if prob_max is not None and prob_max != cli.PAPER.prob_max:
        raise SystemExit("--prob-max is for random-weight decoders: a checkpoint's recipe fixes its scale")
    return fam


class Checkpoint:
    """A loaded checkpoint: ``name``, ``family``, the fp32 ``model`` (None once :meth:`free` dropped it), its
    calibration ``calib`` (:func:`experiments.common.checkpoints.load_real`), ``recipe`` (``main`` or ``certified``,
    :func:`experiments.common.checkpoints.recipe_name`) and ``stream``, the first WikiText-2 test tokens (1-D)."""

    def __init__(self, name: str, fam: str, model, calib, stream: torch.Tensor) -> None:
        self.name, self.family, self.model, self.calib, self.stream = name, fam, model, calib, stream
        self.recipe = checkpoints.recipe_name(calib, "certified")
        self.n_layers = int(model.config.num_hidden_layers)

    def graph(self, prune_last: bool, n_layers: int | None = None):
        """``(graph, info)``: the integer graph of the first ``n_layers`` blocks (None or the checkpoint's: all)."""
        if self.model is None:
            raise RuntimeError("the checkpoint's fp32 model was freed")
        full = n_layers is None or n_layers == self.n_layers
        return checkpoints.real_graph(self.model, self.calib, prune_last, "certified", None,
                                      n_layers=None if full else n_layers)

    def free(self) -> None:
        """Drop the fp32 model (the graphs built from it keep their own integer weights)."""
        self.model = None


def load(name: str, cfg, fam: str, *, seq: int, tokens: int) -> Checkpoint:
    """The checkpoint ``name`` of ``cfg``'s shapes (family ``fam``) with at least ``tokens`` WikiText-2 test tokens
    for prompts of at most ``seq`` tokens (:func:`experiments.common.checkpoints.load_real`, its queries of ``seq``
    tokens joined into one stream).  Refuses a checkpoint of another family, block count or width than ``cfg``'s, and
    prompts of ``seq`` tokens that would run past the checkpoint's ``max_position_embeddings`` behind its fixed
    prefix."""
    model, queries, calib = checkpoints.load_real(name, seq, max(1, -(-tokens // seq)))
    hf = model.config
    got = "opt" if hf.model_type == "opt" else "llama"
    if got != fam or (hf.num_hidden_layers, hf.hidden_size) != (cfg.n_layers, cfg.d_model):
        raise SystemExit(f"--real {name}: a {hf.model_type} checkpoint of {hf.num_hidden_layers} blocks of width "
                         f"{hf.hidden_size}, but --model {cfg.name} names {cfg.n_layers} blocks of width "
                         f"{cfg.d_model}")
    ctx = getattr(hf, "max_position_embeddings", None)
    prefix = len(getattr(calib, "prefix", None) or [])
    if ctx is not None and seq + prefix > ctx:
        raise SystemExit(f"--real {name}: prompts of {seq} tokens behind its {prefix}-token fixed prefix exceed its "
                         f"trained context of {ctx} positions")
    stream = torch.cat([q.reshape(-1) for q in queries])
    return Checkpoint(name, fam, model, calib, stream)


def prompts(ck: Checkpoint, seq: int, n: int) -> torch.Tensor:
    """``[n, seq]``: the first ``n`` WikiText-2 test windows of ``seq`` tokens (``perf.py --real``'s queries)."""
    if n * seq > ck.stream.numel():
        raise ValueError(f"{n} prompts of {seq} tokens need {n * seq} tokens, {ck.stream.numel()} were loaded")
    return ck.stream[:n * seq].reshape(n, seq).clone()


def model_ops(fam: str, cfg, prune_last: bool):
    """The whole model's weight-op shapes a build's commitment plan is made for: ``decoder_shapes`` (a real
    Llama-family graph's weight ops have the benchmark's shapes), or None for OPT, whose real LM head has one more
    column (its final LayerNorm's folded bias) and whose builds are always the full model (planned from its own
    ops, as ``perf.py --real``)."""
    if fam == "opt":
        return None
    from pvi.commit.cost import decoder_shapes
    return decoder_shapes(cfg, prune_last=prune_last)


def fields(ck: Checkpoint | None) -> dict:
    """The record's ``real`` (the checkpoint; None for random weights) and ``recipe`` (None for random weights)."""
    return {"real": ck.name if ck is not None else None, "recipe": ck.recipe if ck is not None else None}
