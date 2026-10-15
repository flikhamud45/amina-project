"""Real checkpoints and their data: OPT and Llama-family models in fp32, WikiText-2 (raw) windows, the certified
models' calibration, and their integer graphs.

Two tokenisations of the test split are in use, each kept as the experiments that report it ran it:
:func:`load_windows` joins the split's lines without a separator (the real-weight cost cells' queries), and
:func:`wikitext` joins them with blank lines (the quality experiments).  Calibration never sees an evaluation token:
the certified recipes calibrate on the train split (:func:`certified_calibration`, :func:`llama_calibration`), and
:func:`load_windows` takes its calibration windows from the end of the test split, disjoint from the evaluation
windows at its start.

Every checkpoint and WikiText-2 file is the snapshot of the Hugging Face cache (``HF_HOME``) that the hub resolves
for the repository's main branch; :data:`SOURCES` keeps the revision of each one loaded, for the records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from pvi.models.hf_opt import CERTIFIED_RECIPE, LEGACY_RECIPE

__all__ = ["wikitext", "windows", "load_windows", "certified_calibration", "is_opt", "load_opt", "load_llama",
           "smoke_model", "token_ids", "prefix_of", "llama_calibration", "load_real", "real_graph", "recipe_name",
           "graph_report", "weights_sha", "Scorer", "SMOOTH", "OPT_RECIPES", "SOURCES"]

_SPLITS = {"test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
           "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
           "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet"}
CALIB_CHARS = 4_000_000
"""The certified recipes calibrate on windows of the first 4,000,000 characters of the train split."""
SMOOTH = dict(smooth_pct=99.9, pct=99.99, pct_att=99.99)
"""The clipping of the OPT percentile recipe (``recipe="legacy"`` with ``--smooth alpha``): SmoothQuant through the
norm gains, 99.99th-percentile clipping; norm gain 4 either way."""
OPT_RECIPES = {"certified": dict(CERTIFIED_RECIPE), "legacy": dict(LEGACY_RECIPE)}
"""The integer OPT recipes the experiments take by name (``--recipe``): ``certified`` (``hf_opt.CERTIFIED_RECIPE``)
and ``legacy``, the percentile recipe (``hf_opt.LEGACY_RECIPE``)."""
SOURCES: dict[str, str] = {}
"""The revision of every checkpoint and WikiText-2 file this process has loaded from the Hugging Face cache:
``{repository, or repository/file: the snapshot's commit hash}`` (:func:`experiments.common.records.environment`
records it)."""


def _snapshot(name: str, path: str) -> str:
    """``path``, a folder or file of the Hugging Face cache, with its snapshot noted in :data:`SOURCES` for ``name``."""
    m = re.search(r"[\\/]snapshots[\\/]([0-9a-f]{40})(?:[\\/]|$)", str(path))
    if m:
        SOURCES[name] = m.group(1)
    return path


# ------------------------------------------------------------------ WikiText-2
def _parquet(split: str, path: str | None = None) -> str:
    if path:
        return path
    if split not in _SPLITS:
        raise ValueError(f"WikiText-2 has the splits {', '.join(_SPLITS)}, not {split!r}")
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    repo, file = "Salesforce/wikitext", _SPLITS[split]
    hit = try_to_load_from_cache(repo, file, repo_type="dataset")     # the cache's main revision, no download
    if not isinstance(hit, str):
        hit = hf_hub_download(repo_id=repo, filename=file, repo_type="dataset")
    return _snapshot(f"{repo}/{file}", hit)


def wikitext(split: str, path: str | None = None, sep: str = "\n\n") -> str:
    """WikiText-2 (raw) ``split`` ("test", "validation" or "train") with its lines joined by ``sep``."""
    import pyarrow.parquet as pq
    return sep.join(pq.read_table(_parquet(split, path)).column("text").to_pylist())


def windows(ids: torch.Tensor, seq: int, n: int | None = None, start: int = 0) -> list[torch.Tensor]:
    """Consecutive non-overlapping ``[1, seq]`` windows of ``ids`` from window ``start`` on (``n``: at most that many;
    default all)."""
    total = ids.numel() // seq - start
    n = total if not n else min(n, total)
    return [ids[(start + i) * seq:(start + i + 1) * seq].unsqueeze(0) for i in range(n)]


def load_windows(tok, parquet: str | None = None, *, seq: int = 128, n: int = 16, calib_windows: int = 1):
    """``(evaluation windows, calibration windows)`` of the test split joined without a separator: ``n`` ``[1, seq]``
    windows from its start and ``calib_windows`` from its end (window ``j`` the ``j``-th counted back), disjoint as
    long as ``(n + calib_windows) seq`` is below the split's length (about 287k OPT tokens)."""
    ids = tok(wikitext("test", parquet, sep=""), return_tensors="pt").input_ids[0]
    assert (n + calib_windows) * seq < ids.numel()
    evals = [ids[i * seq:(i + 1) * seq].unsqueeze(0) for i in range(n)]
    total = ids.numel()
    calib = [ids[total - (j + 1) * seq:total - j * seq].unsqueeze(0) for j in range(calib_windows)]
    return evals, calib


def certified_calibration(tok, n: int = 4, seq: int = 2048, draw: int = 0) -> torch.Tensor:
    """The certified OPT models' calibration ``[n, seq]`` (``hf_opt.CERTIFIED_RECIPE``): windows ``draw * n ..`` of
    ``seq`` tokens of the train split's first :data:`CALIB_CHARS` characters, tokenised once."""
    ids = tok(wikitext("train")[:CALIB_CHARS], return_tensors="pt").input_ids[0]
    first = draw * n
    assert (first + n) * seq <= ids.numel()
    return torch.stack([ids[(first + i) * seq:(first + i + 1) * seq] for i in range(n)])


# ------------------------------------------------------------------ checkpoints
def is_opt(name: str) -> bool:
    """Whether ``name`` is an OPT checkpoint (its configuration's ``model_type``); a name without a Hugging Face
    configuration counts as OPT."""
    from transformers import AutoConfig
    try:
        return AutoConfig.from_pretrained(name).model_type == "opt"
    except OSError:
        return True


def _shards(path: str, single: str) -> list[str]:
    index = os.path.join(path, single + ".index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return [os.path.join(path, s) for s in sorted(set(json.load(f)["weight_map"].values()))]
    return [os.path.join(path, single)] if os.path.exists(os.path.join(path, single)) else []


def load_opt(name: str):
    """``(model, tokenizer)``: OPT in fp32, eval mode.  The official OPT repositories ship ``pytorch_model.bin``
    only, which recent transformers refuse to ``torch.load`` under torch < 2.6, so the state dict is loaded here with
    ``weights_only=True`` (no pickle code runs) and assigned to a model built on the meta device (no second copy)."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM
    path = _snapshot(name, snapshot_download(name, allow_patterns=["*.json", "*.bin", "*.txt"]))
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = OPTForCausalLM(cfg)
    sd = {}
    for shard in _shards(path, "pytorch_model.bin"):
        sd.update(torch.load(shard, map_location="cpu", weights_only=True))
    sd = {(k if k.startswith(("model.", "lm_head.")) else "model." + k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del sd
    missing = [m for m in missing if m != "lm_head.weight"]          # tied to the token embedding
    assert not missing and not unexpected, (missing, unexpected)
    model.tie_weights()
    for p_ in model.parameters():                   # fp16 checkpoints -> fp32, one tensor at a time
        p_.data = p_.data.float()
    model = model.eval()
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
    return model, AutoTokenizer.from_pretrained(path)


def load_llama(name: str, dtype: torch.dtype = torch.float32):
    """``(model, tokenizer)``: a Llama-family checkpoint (Llama, Qwen3) in ``dtype``, eval mode, built on the meta
    device with the state dict assigned (no second copy), from safetensors where the repository has them, else from
    ``pytorch_model.bin`` with ``weights_only=True``.  The rotary embedding's buffers are not in the state dict, so it
    is built again on the CPU."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    meta = ["*.json", "*.model", "*.txt"]
    path = _snapshot(name, snapshot_download(name, allow_patterns=meta + ["*.safetensors"]))
    shards = _shards(path, "model.safetensors")
    if not shards:
        path = _snapshot(name, snapshot_download(name, allow_patterns=meta + ["*.bin"]))
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


def smoke_model(name: str, seed: int = 0):
    """A 2-block random model of ``name``'s family and vocabulary (d = 64, grouped-query heads, random norm gains,
    weights of std 0.15) and ``name``'s tokenizer: only the configuration and the tokenizer are downloaded."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = _snapshot(name, snapshot_download(name, allow_patterns=["*.json", "*.model", "*.txt"]))
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


# ------------------------------------------------------------------ the Llama recipe's fixed prefix
def token_ids(tok, text: str, prefix_spec) -> torch.Tensor:
    """``text`` tokenised as one sequence; with a prefix, without the tokenizer's special tokens (the prefix takes
    the place of its BOS, so no window holds one)."""
    return tok(text, return_tensors="pt", add_special_tokens=not prefix_spec).input_ids[0]


def prefix_of(spec, model, calib: torch.Tensor, tok=None) -> tuple[list[int] | None, dict]:
    """The fixed public prefix's token ids for ``spec`` and a record of the choice: None or ``""`` (no prefix),
    ``auto`` (``hf_llama.prefixquant_tokens`` on the calibration windows ``calib``), ``bos`` (the configuration's
    BOS alone) or comma-separated ids."""
    from pvi.models import hf_llama
    if not spec:
        return None, {}
    if spec == "auto":
        ids, stats = hf_llama.prefixquant_tokens(model, calib)
        stats["outlier_tokens"] = [list(t) for t in stats["outlier_tokens"]]
    elif spec == "bos":
        if model.config.bos_token_id is None:
            raise ValueError("the configuration has no bos_token_id: give the prefix's ids")
        ids, stats = [int(model.config.bos_token_id)], {"rule": "bos"}
    else:
        ids, stats = [int(s) for s in spec.split(",")], {"rule": "ids"}
    stats["spec"] = spec
    if tok is not None:
        stats["text"] = [tok.decode([i]) for i in ids]
    return ids, stats


def llama_calibration(model, tok, n: int = 4, seq: int = 2048, name: str = "") -> SimpleNamespace:
    """The main Llama recipe's calibration (``hf_llama.MAIN_RECIPE``): ``ids``, ``n`` windows of ``seq`` tokens of the
    train split's first :data:`CALIB_CHARS` characters (tokenised without special tokens), and ``prefix``, the fixed
    public prefix PrefixQuant's rule picks on them (``prefix_info``: the record of the choice), and ``recipe``, the
    main recipe at the checkpoint ``name``'s settings (``hf_llama.main_recipe``)."""
    train = token_ids(tok, wikitext("train")[:CALIB_CHARS], "auto")
    ids = torch.cat(windows(train, seq, n, start=0))
    with torch.no_grad():
        prefix, info = prefix_of("auto", model, ids, tok)
    from pvi.models import hf_llama
    return SimpleNamespace(ids=ids, prefix=prefix, prefix_info=info, recipe=hf_llama.main_recipe(name))


# ------------------------------------------------------------------ real queries and their integer graphs
def load_real(name: str, seq: int, n_queries: int, calib_seq: int = 0, recipe: str = "certified"):
    """``(model, queries, calib)`` for the real-weight cost cells.  A Llama-family checkpoint: ``n_queries`` test
    windows of ``seq`` tokens (blank-line joined, without special tokens) and :func:`llama_calibration`.  OPT:
    ``n_queries`` windows of :func:`load_windows`, and the calibration of ``recipe`` -- ``certified``
    (:func:`certified_calibration`) or ``legacy``, one window of ``calib_seq`` tokens (default ``seq``) from the end
    of the test split, disjoint from every query."""
    if not is_opt(name):
        model, tok = load_llama(name)
        test = token_ids(tok, wikitext("test"), "auto")
        queries = windows(test, seq, n_queries)[:n_queries]
        return model, queries, llama_calibration(model, tok, name=name)
    model, tok = load_opt(name)
    queries, _ = load_windows(tok, seq=seq, n=n_queries, calib_windows=0)
    if recipe == "certified":
        return model, queries, certified_calibration(tok)
    cs = calib_seq or seq
    _, calib = load_windows(tok, seq=cs, n=-(-n_queries * seq // cs), calib_windows=1)
    return model, queries, calib[0]


def recipe_name(calib, recipe: str) -> str:
    """The recipe a real graph was built with, for the records: ``main`` (``hf_llama.MAIN_RECIPE``) for a
    Llama-family checkpoint (its calibration carries the fixed prefix), else the OPT recipe."""
    return "main" if hasattr(calib, "prefix") else recipe


def real_graph(model, calib, prune_last: bool, recipe: str = "certified", smooth: float | None = None):
    """The integer graph of ``model`` calibrated on ``calib`` (from :func:`load_real`) on the CPU.  A Llama-family
    checkpoint: ``build_llama_from_hf`` with ``MAIN_RECIPE`` and the calibration's prefix.  OPT: ``build_opt_from_hf``
    with ``CERTIFIED_RECIPE`` (``recipe="certified"``), or the percentile recipe (``legacy``: norm gain 4, with
    ``smooth`` SmoothQuant through the norm gains and :data:`SMOOTH`'s clipping)."""
    if hasattr(calib, "prefix"):
        from pvi.models import hf_llama
        return hf_llama.build_llama_from_hf(model, calib.ids, prune_last=prune_last, prefix_tokens=calib.prefix,
                                            **getattr(calib, "recipe", hf_llama.MAIN_RECIPE))
    from pvi.models.hf_opt import CERTIFIED_RECIPE, build_opt_from_hf
    if recipe == "certified":
        return build_opt_from_hf(model, calib, prune_last=prune_last, **CERTIFIED_RECIPE)
    kw = dict(norm_gain=4) if smooth is None else dict(norm_gain=4, smooth=smooth, **SMOOTH)
    return build_opt_from_hf(model, calib, prune_last=prune_last, **kw)


# ------------------------------------------------------------------ checks of a built graph
def graph_report(graph, info: dict | None = None) -> dict:
    """The constraints a certified graph must meet: the honest claim bound (below ``2^29``), the multipliers of the
    requantisations that follow linear weight ops (V1 cuts only those, and needs ``2^14 <= m <= 2^29``; the
    embedding's residual add is not one), per-row multipliers (how many ops, how many distinct values), and the norm
    gains' range (``info``: the builder's, for the ops' names and gains)."""
    from pvi.core.graph import CheapOp, MatOp
    from pvi.v0.params import claim_bound
    worst_claim = max(claim_bound(op) for op in graph.mat_ops)
    producer = {op.output: op for op in graph.ops}
    mults, vec_ops, distinct, outside = [], 0, [], []
    for op in graph.ops:
        if not isinstance(op, CheapOp):
            continue
        z = producer.get(op.inputs[-1])
        linear = isinstance(z, MatOp) and z.layout == "linear"
        if "requant" in op.params:
            m = int(op.params["requant"]["mult"])
            if linear:
                mults.append(m)
            if linear and not (1 << 14 <= m <= 1 << 29):
                comp = info["meta"][op.name].get("comp") if info else ""
                outside.append(f"{op.name}({comp}, layer {info['meta'][op.name].get('layer') if info else ''}):"
                               f" 2^{math.log2(m):.2f}")
        elif "vec_requant" in op.params:
            vec_ops += 1
            mv = op.fn.consts["mult"]
            distinct.append(int(torch.unique(mv).numel()))
            mults.extend([int(mv.min()), int(mv.max())])
            bad = int(((mv < (1 << 14)) | (mv > (1 << 29))).sum())
            if bad:
                comp = info["meta"][op.name].get("comp") if info else ""
                outside.append(f"{op.name}({comp}, vector): {bad} rows outside")
    gains = [m["G"] for m in info["meta"].values() if m.get("kind") == "norm"] if info else []
    return {"max_claim_bound_log2": round(math.log2(worst_claim), 3), "claims_ok": worst_claim < (1 << 29),
            "linear_mult_min_log2": round(math.log2(max(1, min(mults))), 3),
            "linear_mult_max_log2": round(math.log2(max(mults)), 3),
            "linear_mults_outside_v1_range": outside, "vector_mult_ops": vec_ops,
            "distinct_mults_per_vector_op_max": max(distinct) if distinct else 1,
            "distinct_mults_per_vector_op_mean": (sum(distinct) / len(distinct)) if distinct else 1,
            "norm_gain_min": float(min(float(g.min()) for g in gains)) if gains else None,
            "norm_gain_max": float(max(float(g.max()) for g in gains)) if gains else None}


def weights_sha(graph) -> str:
    """SHA-256 of a graph's weight ops (names, int8 weights and biases, in order): two builds with the same hash
    commit the same matrices."""
    h = hashlib.sha256()
    for op in graph.mat_ops:
        h.update(op.name.encode() + op.weight.contiguous().numpy().tobytes())
        if op.bias is not None:
            h.update(op.bias.contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ perplexity
class Scorer:
    """Perplexity and top-1 agreement (with a reference's argmaxes) over fixed windows."""

    def __init__(self, wins):
        self.wins = wins
        self.ref_argmax = None

    def score(self, logits_fn):
        tot, n, am = 0.0, 0, []
        for w in self.wins:
            lg = logits_fn(w)[0, :-1].double()
            tgt = w[0, 1:]
            tot += float(F.cross_entropy(lg, tgt, reduction="sum"))
            n += tgt.numel()
            am.append(lg.argmax(-1))
        ppl = math.exp(tot / n)
        agree = None
        if self.ref_argmax is not None:
            agree = sum(int((a == b).sum()) for a, b in zip(am, self.ref_argmax)) / sum(a.numel() for a in am)
        return ppl, agree, am
