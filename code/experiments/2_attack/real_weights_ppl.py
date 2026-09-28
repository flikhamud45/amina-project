"""D.3: perplexity of the integer OPT built from real weights, vs the fp32 checkpoint.

WikiText-2 (raw) test split, non-overlapping windows of ``--seq`` tokens; calibration
uses a disjoint window taken from the end of the split. Every knob swept here is a
public constant of a cheap op, so every row has the benchmark graph's exact cost.

    python experiments/2_attack/real_weights_ppl.py --model facebook/opt-125m \
        --config opt-125m --parquet PATH/test.parquet --windows 16
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvi.fullcheck.real_weights import build_opt_from_hf, matches_benchmark_graph, real_logits

ROOT = Path(__file__).resolve().parents[2]


def nll(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int, int]:
    """Summed next-token NLL, token count and argmax hits for one window."""
    lg = logits[0, :-1].double()
    tgt = ids[0, 1:]
    return float(F.cross_entropy(lg, tgt, reduction="sum")), tgt.numel(), lg.argmax(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--config", default="opt-125m")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--pct", type=float, nargs="+", default=[100.0, 99.99, 99.9])
    ap.add_argument("--norm-gain", type=float, nargs="+", default=[32, 16, 8])
    ap.add_argument("--device", default="cpu", help="where the integer graph runs (the fp32 model stays on CPU)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "results" / "real_weights_ppl.json"))
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    text = "".join(pq.read_table(args.parquet).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    calib = ids[-args.seq:].unsqueeze(0)
    wins = [ids[i * args.seq:(i + 1) * args.seq].unsqueeze(0) for i in range(args.windows)]

    def score(fn):
        tot, n, hits = 0.0, 0, []
        for w in wins:
            lg = fn(w)
            s, c, am = nll(lg, w)
            tot, n = tot + s, n + c
            hits.append(am)
        return math.exp(tot / n), hits

    with torch.no_grad():
        ppl_fp, am_fp = score(lambda w: model(w).logits)
    print(f"fp32 ppl {ppl_fp:.2f}", flush=True)

    rows = []
    for pct, gain in itertools.product(args.pct, args.norm_gain):
        t0 = time.time()
        g, info = build_opt_from_hf(model, calib, pct=pct, norm_gain=gain, device=args.device)
        probs = matches_benchmark_graph(g, args.config)
        ppl, am = score(lambda w: real_logits(g, info, w.to(args.device), all_positions=True).cpu())
        del g
        agree = sum(int((a == b).sum()) for a, b in zip(am, am_fp)) / sum(a.numel() for a in am)
        row = {"pct": pct, "norm_gain": gain, "embed_mult": info["embed_mult"], "ppl": ppl,
               "top1_agreement_with_fp32": agree, "graph_differences": probs,
               "seconds": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)

    best = min(rows, key=lambda r: r["ppl"])
    out = {"model": args.model, "config": args.config, "dataset": "wikitext-2-raw-v1 test",
           "seq": args.seq, "windows": args.windows, "tokens": args.windows * (args.seq - 1),
           "fp32_ppl": ppl_fp, "rows": rows, "best": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("best", json.dumps(best))


if __name__ == "__main__":
    main()
