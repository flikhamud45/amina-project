"""Where the integer Llama-family graph leaves the fp32 checkpoint, block by block and position by position: the
diagnosis behind the statement of Sec. ``sec:quality`` and of the extended version's discussion that Qwen3-4B forms a
second massive activation, at the first token after the fixed prefix, which no public prefix takes and which one
static int8 scale of the residual stream cannot hold.

    python -m experiments.quality.llama_diag --model Qwen/Qwen3-4B-Base --recipes main smooth55 scalar4
    python -m experiments.quality.llama_diag --model Qwen/Qwen3-4B-Base --prefix 151643,151643 --recipes smooth55
    python -m experiments.quality.llama_diag --model Qwen/Qwen3-0.6B-Base --prefix none --no-qk-balance \\
        --recipes scalar4 smooth55                                     # the graphs before the prefix and the balance
    python -m experiments.quality.llama_diag --model Qwen/Qwen3-0.6B-Base --smoke --threads 2    # laptop

For one WikiText-2 test window (the first ``--seq`` tokens of the test split), with calibration on ``--calib-windows``
windows of ``--seq`` tokens from the start of the train split: per block boundary, the cosine between the integer
residual stream (dequantised) and the fp32 checkpoint's, as a mean over positions, its minimum, at position 0 and
over positions 1..; the fp32 stream's largest |value| and its position and channel (the massive activations of Qwen
and Llama models sit at the first token); and the integer stream's clipping (entries at ``+-RES_MAX``).  Recipes
(``--recipes``): ``main``, the main recipe at the checkpoint's settings (``hf_llama.main_recipe``); ``smooth55``,
smoothing at alpha 0.55 with the 99.9 / 99.99 percentiles and no SwiGLU product balance; ``scalar4``, one norm gain 4.

The fixed public prefix (``--prefix``, as ``ppl_llama.py``'s: ``auto``, the main recipe's and the default, is
PrefixQuant's rule on the calibration windows; ``bos``; comma-separated ids; ``none``): the graphs carry it, the text
is tokenised without the tokenizer's special tokens, and the fp32 reference runs on ``[prefix | window]``.  Positions
are then the window's (position 0 is the window's first token, at RoPE position ``P``); the output adds the prefix's
tokens and how they were chosen (``auto``: the outlier tokens per block and their counts), the fp32 stream's largest
|value| at the prefix positions per block (where the first token's massive activation then sits), and per layer how
far the prefix's keys and values reach on their layer's scales (``max |real| / (127 s)``; above 1 they clip).  A
second massive activation shows as an ``fp_max`` at window position 0 above the prefix's ``fp_prefix_max``.
``--no-qk-balance``: Qwen3's graphs without the q/k-norm balance (the lines are tagged ``-nobal``).  The op-level
attribution is ``llama_attrib.py``.

Output: one JSON line per recipe on stdout (and appended to ``--out``): ``model``, ``recipe``, ``opts`` (the
builder's keyword arguments), ``prefix_tokens``, ``qk_balance``, ``s_residual``, ``res_max_real``, ``RES_MAX``,
``blocks`` (per boundary ``block``, ``cos_mean``, ``cos_min``, ``cos_pos0``, ``cos_rest_mean``, ``fp_max``,
``fp_max_pos``, ``fp_max_ch``, ``int_clip``, ``fp_prefix_max``), with a prefix ``prefix_hidden_max`` and
``prefix_clip``, and ``env``; with a prefix a first line with its choice.  A table per recipe goes to stderr.

Hardware: the CPU only (one 128-token window, 4 x 128 calibration tokens; the reported diagnoses ran on a Xeon node),
64 GB of host memory for the models up to Qwen3-1.7B; Qwen3-4B's fp32 checkpoint alone takes 16 GB.
"""
from __future__ import annotations

import argparse
import sys

import torch

from experiments.common import records
from experiments.common.checkpoints import load_llama, prefix_of, smoke_model, token_ids, wikitext
from experiments.quality.ppl_llama import prefix_spec
from pvi.models import hf_llama
from pvi.models.decoder import RES_MAX

RECIPES = {"scalar4": dict(norm_gain=4.0),
           "smooth55": dict(norm_gain=4.0, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)}
"""The diagnosis's comparison recipes: one norm gain 4, and smoothing at alpha 0.55 without the SwiGLU product
balance (the recipe of the reported Qwen3-4B diagnosis); ``main`` is :func:`recipe`'s."""
TEXT_CHARS = 400_000
"""Characters of each split tokenised: enough for the windows, whose tokens are those of the whole split's."""


def recipe(name: str, model_name: str) -> dict:
    """The keyword arguments of ``build_llama_from_hf`` for the recipe ``name`` (``main``: the main recipe at the
    checkpoint ``model_name``'s settings, ``hf_llama.main_recipe``; else :data:`RECIPES`)."""
    return hf_llama.main_recipe(model_name) if name == "main" else dict(RECIPES[name])


def data(tok, spec, seq: int, calib_windows: int, window: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(the test window [1, seq] number window, the calibration [calib_windows, seq])`` from the starts of the test
    and train splits, tokenised as ``ppl_llama.py`` does (without special tokens with a prefix)."""
    test = token_ids(tok, wikitext("test")[:TEXT_CHARS], spec)
    train = token_ids(tok, wikitext("train")[:TEXT_CHARS], spec)
    w = test[window * seq:(window + 1) * seq].unsqueeze(0)
    return w, torch.stack([train[i * seq:(i + 1) * seq] for i in range(calib_windows)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base", help="a Llama or Qwen3 checkpoint")
    ap.add_argument("--seq", type=int, default=128, help="tokens of the test window and of each calibration window")
    ap.add_argument("--calib-windows", type=int, default=4, help="calibration windows")
    ap.add_argument("--recipes", nargs="+", default=["main"], choices=["main", *RECIPES])
    ap.add_argument("--prefix", nargs="?", const="auto", default="auto",
                    help="the fixed public prefix: auto (PrefixQuant's rule, the main recipe's), bos, comma-separated "
                         "token ids, or none")
    ap.add_argument("--no-qk-balance", action="store_true", help="Qwen3: no q/k-norm balance per RoPE pair")
    ap.add_argument("--smoke", action="store_true", help="a 2-block random model with the checkpoint's vocabulary")
    ap.add_argument("--threads", type=int, default=8, help="torch's intra-op threads")
    ap.add_argument("--out", default=None, help="a JSONL file the records are appended to")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    spec = prefix_spec(args.prefix)
    model, tok = smoke_model(args.model) if args.smoke else load_llama(args.model)
    w, calib = data(tok, spec, args.seq, args.calib_windows)
    env = records.environment("cpu")
    with torch.no_grad():
        prefix, choice = prefix_of(spec, model, calib, tok)
        joint = w if not prefix else torch.cat([torch.tensor([prefix], dtype=w.dtype), w], 1)
        full = hf_llama.hf_residual_states(model, joint)              # [prefix | window]: split below
    n = len(prefix or ())
    ref = [h[:, n:] for h in full]
    pre_max = [float(h[:, :n].abs().max()) if n else None for h in full]
    del full
    if prefix:
        records.emit({"model": args.model, "prefix_tokens": prefix, "prefix_choice": choice, "env": env}, args.out)
        print(f"  prefix {prefix} = {choice.get('text')} ({choice.get('rule')}"
              + (f", o = {choice['o']}, outliers per block {[round(c, 2) for c in choice['outliers_per_block']]}, "
                 f"outlier tokens {[(tok.decode([t]), c) for t, c in choice['outlier_tokens'][:8]]}"
                 if choice.get("rule") == "prefixquant" else "") + ")", file=sys.stderr, flush=True)
    mx = [(float(h.abs().max()), *[int(v) for v in divmod(int(h.abs().reshape(-1).argmax()), h.shape[-1])])
          for h in ref]
    for name in args.recipes:
        kw = recipe(name, args.model)
        with torch.no_grad():
            graph, info = hf_llama.build_llama_from_hf(model, calib, prefix_tokens=prefix,
                                                       qk_balance=not args.no_qk_balance, **kw)
            ours = hf_llama.residual_states(graph, info, w)
        rows = []
        for i, (a, b) in enumerate(zip(ours, ref)):
            b = b[:, -a.shape[1]:]
            c = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]).double(),
                                                      b.reshape(-1, b.shape[-1]).double(), dim=-1)
            rows.append({"block": i, "cos_mean": round(float(c.mean()), 4), "cos_min": round(float(c.min()), 4),
                         "cos_pos0": round(float(c[0]), 4), "cos_rest_mean": round(float(c[1:].mean()), 4),
                         "fp_max": round(mx[i][0], 1), "fp_max_pos": mx[i][1], "fp_max_ch": mx[i][2],
                         "int_clip": int((a.abs() >= RES_MAX * info["s_residual"] * (1 - 1e-9)).sum()),
                         "fp_prefix_max": None if pre_max[i] is None else round(pre_max[i], 1)})
        rec = {"model": args.model, "recipe": name, "opts": kw, "prefix_tokens": prefix,
               "qk_balance": info["qk_balance"], "s_residual": info.get("s_residual"),
               "res_max_real": info.get("residual_real_max"), "RES_MAX": RES_MAX, "blocks": rows}
        if prefix:
            rec.update(prefix_hidden_max=info["prefix_hidden_max"],
                       prefix_clip=[[round(k, 3), round(v, 3)] for k, v in info["prefix_clip"]])
        records.emit(dict(rec, env=env), args.out)
        tag = name + ("+prefix" if prefix else "") + ("" if info["qk_balance"] or info["arch"] != "qwen3" else "-nobal")
        for r in rows:
            print(f"  {tag} block {r['block']:2d}: cos mean {r['cos_mean']:+.3f} min {r['cos_min']:+.3f} "
                  f"pos0 {r['cos_pos0']:+.3f} rest {r['cos_rest_mean']:+.3f} | fp max {r['fp_max']:.1f} at "
                  f"pos {r['fp_max_pos']} ch {r['fp_max_ch']}"
                  + (f" | fp prefix max {r['fp_prefix_max']:.1f}" if prefix else "")
                  + (f" | int clip {r['int_clip']}" if r["int_clip"] else ""), file=sys.stderr, flush=True)
        if prefix:
            fp_top = info["residual_real_max"] / info["residual_headroom"]
            print(f"  {tag} s_res {info['s_residual']:.3g} (fp max {fp_top:.1f} over the "
                  f"window positions, {info['prefix_hidden_max']:.1f} at the prefix); prefix k/v reach per layer "
                  f"(max |real| / (127 s)): {[(round(k, 2), round(v, 2)) for k, v in info['prefix_clip']]}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
