"""F2b-Q diagnosis: where the integer Llama/Qwen3 graph leaves the checkpoint, block by block and position by position.

    cd code && PYTHONPATH=src python experiments/9_quality/llama_diag.py --model Qwen/Qwen3-0.6B-Base --seq 128
    cd code && PYTHONPATH=src python experiments/9_quality/llama_diag.py --model Qwen/Qwen3-0.6B-Base --seq 128 --prefix

For one WikiText-2 test window, with calibration on train windows as in ``ppl_llama.py``: per block boundary, the
cosine between the integer residual stream (dequantised) and the fp32 checkpoint's, as a mean over positions, its
minimum, at position 0 and over positions 1..; the fp32 stream's largest |value| and its position and channel (the
massive activations of Qwen/Llama models sit at the first token); and the integer stream's clipping (entries at
``+-RES_MAX``). Recipes: the scalar gain and the smoothed graph, each also with the first position's residual kept
out of the calibration statistics' influence (by comparing position 0 against the rest).

``--prefix [SPEC]`` (``ppl_llama.py``'s: ``auto`` = PrefixQuant's rule, the default of a bare ``--prefix``; ``bos``;
or ids): the graphs carry the fixed prefix, the text is tokenised without the tokenizer's special tokens, and the fp32
reference runs on ``[prefix | window]``. Positions are then the window's (position 0 is the window's first token, at
RoPE position ``P``); the output adds the prefix's tokens and how they were chosen (``auto``: the outlier tokens per
block and their counts), the fp32 stream's largest |value| at the prefix positions per block (where the massive
activation should now sit), and per layer how far the prefix's keys and values reach on their layer's scales
(``max |real| / (127 s)``; above 1 they clip).

``--no-qk-balance``: Qwen3's graphs without the q/k balance per RoPE pair (``real_llama``'s ``qk_balance``, on by
default; the lines are tagged ``-nobal``), the graphs of job 1009531. The op-level attribution is ``llama_attrib.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppl_llama as pl                                      # noqa: E402

from pvi.fullcheck import real_llama as rl                 # noqa: E402
from pvi.fullcheck.transformer import RES_MAX               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--recipes", nargs="+", default=["scalar4", "smooth55"])
    ap.add_argument("--smoke", action="store_true", help="a tiny random model of the family (laptop check)")
    ap.add_argument("--prefix", nargs="?", const="auto", default=None,
                    help="a fixed public prefix: auto (PrefixQuant's rule; a bare --prefix), bos, or ids 'a,b,..'")
    ap.add_argument("--no-qk-balance", action="store_true", help="Qwen3: no q/k balance per RoPE pair (the old graph)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = pl.smoke_model(args.model) if args.smoke else pl.load_llama(args.model)
    test = pl.token_ids(tok, pl.p2._text("test")[:400_000], args.prefix)
    train = pl.token_ids(tok, pl.p2._text("train")[:400_000], args.prefix)
    w = test[:args.seq].unsqueeze(0)
    calib = torch.stack([train[i * args.seq:(i + 1) * args.seq] for i in range(args.calib_windows)])
    with torch.no_grad():
        prefix, choice = pl.prefix_of(args.prefix, model, calib, tok)
        full = rl.hf_residual_states(model, pl._joint(w, prefix))      # [prefix | window]: split below
    n = len(prefix or ())
    ref = [h[:, n:] for h in full]
    pre_max = [float(h[:, :n].abs().max()) if n else None for h in full]
    del full
    if prefix:
        print(json.dumps({"model": args.model, "prefix_tokens": prefix, "prefix_choice": choice}), flush=True)
        print(f"  prefix {prefix} = {choice.get('text')} ({choice.get('rule')}"
              + (f", o = {choice['o']}, outliers per block {[round(c, 2) for c in choice['outliers_per_block']]}, "
                 f"outlier tokens {[(tok.decode([t]), c) for t, c in choice['outlier_tokens'][:8]]}"
                 if choice.get("rule") == "prefixquant" else "") + ")", file=sys.stderr, flush=True)
    mx = [(float(h.abs().max()), *[int(v) for v in divmod(int(h.abs().reshape(-1).argmax()), h.shape[-1])]) for h in ref]
    recipes = {"scalar4": dict(norm_gain=4.0), "smooth55": dict(norm_gain=4.0, smooth=0.55, smooth_pct=99.9, pct=99.99,
                                                                     pct_att=99.99)}
    for name in args.recipes:
        with torch.no_grad():
            graph, info = rl.build_llama_from_hf(model, calib, prefix_tokens=prefix,
                                                 qk_balance=not args.no_qk_balance, **recipes[name])
            ours = rl.residual_states(graph, info, w)
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
        rec = {"model": args.model, "recipe": name, "prefix_tokens": prefix, "qk_balance": info["qk_balance"],
               "s_residual": info.get("s_residual"),
               "res_max_real": info.get("residual_real_max"), "RES_MAX": RES_MAX, "blocks": rows}
        if prefix:
            rec.update(prefix_hidden_max=info["prefix_hidden_max"],
                       prefix_clip=[[round(k, 3), round(v, 3)] for k, v in info["prefix_clip"]])
        print(json.dumps(rec), flush=True)
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
