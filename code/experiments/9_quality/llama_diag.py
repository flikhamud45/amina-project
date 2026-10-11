"""F2b-Q diagnosis: where the integer Llama/Qwen3 graph leaves the checkpoint, block by block and position by position.

    cd code && PYTHONPATH=src python experiments/9_quality/llama_diag.py --model Qwen/Qwen3-0.6B-Base --seq 128

For one WikiText-2 test window, with calibration on train windows as in ``ppl_llama.py``: per block boundary, the
cosine between the integer residual stream (dequantised) and the fp32 checkpoint's, as a mean over positions, its
minimum, at position 0 and over positions 1..; the fp32 stream's largest |value| and its position and channel (the
massive activations of Qwen/Llama models sit at the first token); and the integer stream's clipping (entries at
``+-RES_MAX``). Recipes: the scalar gain and the smoothed graph, each also with the first position's residual kept
out of the calibration statistics' influence (by comparing position 0 against the rest).
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
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = pl.smoke_model(args.model) if args.smoke else pl.load_llama(args.model)
    test = tok(pl.p2._text("test")[:400_000], return_tensors="pt").input_ids[0]
    train = tok(pl.p2._text("train")[:400_000], return_tensors="pt").input_ids[0]
    w = test[:args.seq].unsqueeze(0)
    calib = torch.stack([train[i * args.seq:(i + 1) * args.seq] for i in range(args.calib_windows)])
    with torch.no_grad():
        ref = rl.hf_residual_states(model, w)
    mx = [(float(h.abs().max()), *[int(v) for v in divmod(int(h.abs().reshape(-1).argmax()), h.shape[-1])]) for h in ref]
    recipes = {"scalar4": dict(norm_gain=4.0), "smooth55": dict(norm_gain=4.0, smooth=0.55, smooth_pct=99.9, pct=99.99,
                                                                     pct_att=99.99)}
    for name in args.recipes:
        with torch.no_grad():
            graph, info = rl.build_llama_from_hf(model, calib, **recipes[name])
            ours = rl.residual_states(graph, info, w)
        rows = []
        for i, (a, b) in enumerate(zip(ours, ref)):
            b = b[:, -a.shape[1]:]
            c = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]).double(),
                                                      b.reshape(-1, b.shape[-1]).double(), dim=-1)
            rows.append({"block": i, "cos_mean": round(float(c.mean()), 4), "cos_min": round(float(c.min()), 4),
                         "cos_pos0": round(float(c[0]), 4), "cos_rest_mean": round(float(c[1:].mean()), 4),
                         "fp_max": round(mx[i][0], 1), "fp_max_pos": mx[i][1], "fp_max_ch": mx[i][2]})
        print(json.dumps({"model": args.model, "recipe": name, "s_residual": info.get("s_residual"),
                          "res_max_real": info.get("residual_real_max"), "RES_MAX": RES_MAX, "blocks": rows}), flush=True)
        for r in rows:
            print(f"  {name} block {r['block']:2d}: cos mean {r['cos_mean']:+.3f} min {r['cos_min']:+.3f} "
                  f"pos0 {r['cos_pos0']:+.3f} rest {r['cos_rest_mean']:+.3f} | fp max {r['fp_max']:.1f} at "
                  f"pos {r['fp_max_pos']} ch {r['fp_max_ch']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
