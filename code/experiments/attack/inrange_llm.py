"""In-range edits on a real LLM against per-neuron range checks (plan G3.5; mock review, Reviewer A, weakness 2
and question 3).

``real_llm.py`` changes the next token with ONE edited FFN neuron, but only 12.5% of those forged values stay
inside the neuron's natural range, so a cheap per-neuron range check would flag most of them. This script asks
the question a hardened spot-checker raises: if the verifier also checks that every activation it opens lies in a
range calibrated on held-out text, how many neurons must an attacker edit (each within its range) to change the
next token, and how often does a path, or a value-aware sampler, then open one of them?

* Ranges: the FFN activations (the post-ReLU inputs of ``fc2``) of the chosen layers over ``--calib-windows``
  windows of ``--calib-seq`` tokens of WikiText-2 *train* (disjoint from the prompts), every position: per neuron
  the maximum (``max``) and the 99.9th percentile (``p999``); the minimum is 0 (ReLU).
* Attack (greedy, untargeted): at the last position of one layer, repeatedly move the neurons with the largest
  first-order decrease of the margin (top-1 logit minus runner-up) to the bound of their range in the helpful
  direction (0, or the range's top), until the next token changes or ``--cap`` neurons are edited. Every edited
  value is inside the range, so the range check passes. ``--moves down`` allows only decreases (to 0): a zeroed
  neuron has weight 0 under contribution weighting of the claimed trace, so that sampler never opens it.
* Detection: per random path the edited layer is opened at one neuron, uniform over ``d_ff`` (Anchuri et al.),
  so a path catches the edit with probability ``k / d_ff``; a contribution-weighted sampler opens neuron ``j``
  with probability proportional to ``|a_j| ||W_fc2[:, j]||`` on the claimed (edited) trace. Paths needed for
  ``1 - 2**-40``: ``ln(2**40) / -ln(1 - p)``.

    cd code && PYTHONPATH=src python experiments/2_attack/inrange_llm.py --model facebook/opt-1.3b --layers 6 12 18 23

One JSON line per (layer, range) with the success rate, the number of edited neurons and the detection figures;
the per-prompt records go to ``artifacts/results/inrange_llm_<model>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_llm import PROMPTS, last_logits, prefix_cache          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Hook:
    """Pre-hook on ``fc2``: overwrite some activations of the last position, and/or expose that row as a leaf."""

    def __init__(self) -> None:
        self.idx: torch.Tensor | None = None
        self.val: torch.Tensor | None = None
        self.capture = False
        self.act: torch.Tensor | None = None
        self.record: list | None = None

    def __call__(self, module, inputs):
        x = inputs[0]
        if self.record is not None:
            self.record.append(x.detach().float().reshape(-1, x.shape[-1]))
            return None
        if self.idx is None and not self.capture:
            return None
        flat = x.reshape(-1, x.shape[-1])
        last = flat[-1].float().clone()
        if self.idx is not None:
            last[self.idx] = self.val
        if self.capture:
            last = last.detach().requires_grad_(True)
            self.act = last
        out = torch.cat([flat[:-1], last.to(x.dtype)[None]], 0)
        return (out.reshape(x.shape),)


def _calibration_text(n_tokens: int, tok) -> torch.Tensor:
    """The first ``n_tokens`` tokens of WikiText-2 (raw) *train*, which neither the prompts nor the quality
    evaluation (test split) use."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="Salesforce/wikitext", filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "".join(pq.read_table(path).column("text").to_pylist()[:20000])
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= n_tokens, ids.numel()
    return ids[:n_tokens]


def calibrate(model, tok, fc2s: dict, windows: int, seq: int) -> dict:
    """Per layer: (max, 99.9th percentile) of each fc2 input over the calibration windows, every position."""
    ids = _calibration_text(windows * seq, tok).view(windows, seq).to(next(model.parameters()).device)
    hooks = {}
    for layer, fc2 in fc2s.items():
        h = Hook()
        h.record = []
        hooks[layer] = (h, fc2.register_forward_pre_hook(h))
    with torch.no_grad():
        for w in range(windows):
            model(input_ids=ids[w:w + 1])
    out = {}
    for layer, (h, handle) in hooks.items():
        handle.remove()
        a = torch.cat(h.record, 0)
        a = a.cpu()
        out[layer] = {"max": a.max(0).values, "p999": torch.quantile(a, 0.999, dim=0), "tokens": a.shape[0]}
    return out


def attack(model, hook: Hook, last_ids, past, lo: torch.Tensor, hi: torch.Tensor, cap: int, down: bool) -> dict:
    """Greedy in-range untargeted edit at one layer's last position (see the module docstring); with ``down``
    only decreasing moves (towards 0), which a contribution-weighted sampler of the claimed trace cannot see."""
    hook.idx = hook.val = None
    hook.capture = True
    with torch.no_grad():
        honest = last_logits(model, last_ids, past)
    a0 = hook.act.detach().clone()
    top = int(honest.argmax())
    vals = a0.clone()
    edited: list[int] = []
    steps = 0
    while True:
        hook.idx = torch.tensor(edited, dtype=torch.long) if edited else None
        hook.val = vals[edited] if edited else None
        hook.capture = True
        logits = last_logits(model, last_ids, past)
        a = hook.act
        cur = int(logits.argmax())
        if cur != top:
            break
        if len(edited) >= cap:
            break
        masked = logits.detach().clone()
        masked[top] = -float("inf")
        second = int(masked.argmax())
        margin = logits[top] - logits[second]
        (g,) = torch.autograd.grad(margin, a)
        av = a.detach()
        gain = torch.where(g > 0, g * (av - lo), -g * (hi - av))
        if down:
            gain = torch.where(g > 0, gain, torch.full_like(gain, -float("inf")))
        if edited:
            gain[edited] = -float("inf")
        m = 1 if len(edited) < 32 else 4
        picks = torch.topk(gain, min(m, cap - len(edited))).indices.tolist()
        if gain[picks[0]] <= 0:
            break                                   # no in-range move lowers the margin to first order
        for j in picks:
            vals[j] = lo[j] if g[j] > 0 else hi[j]
            edited.append(j)
        steps += 1
    hook.idx = hook.val = None
    hook.capture = False
    return {"success": cur != top, "k": len(edited), "steps": steps, "honest": top, "forged": cur,
            "edited": edited, "values": vals[edited].tolist() if edited else [], "a_tampered": vals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23])
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--calib-seq", type=int, default=128)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cuda for the large models (fp32 weights on the GPU)")
    ap.add_argument("--moves", choices=["any", "down"], default="any",
                    help="down: only decrease activations (the adaptive attacker against contribution weighting)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT / "experiments" / "9_quality"))
    from common import load_opt                    # weights_only state-dict loading (torch < 2.6)
    model, tok = load_opt(args.model)
    model = model.to(args.device)
    layers = model.model.decoder.layers
    fc2s = {i: layers[i].fc2 for i in args.layers}
    t0 = time.perf_counter()
    ranges = calibrate(model, tok, fc2s, args.calib_windows, args.calib_seq)
    print(json.dumps({"calibration_s": round(time.perf_counter() - t0, 1),
                      "tokens": ranges[args.layers[0]]["tokens"]}), flush=True)
    records = []
    for layer in args.layers:
        fc2 = fc2s[layer]
        d_ff = fc2.in_features
        col = fc2.weight.detach().float().norm(dim=0).cpu()            # ||W_fc2[:, j]||
        hook = Hook()
        handle = fc2.register_forward_pre_hook(hook)
        for kind in ("max", "p999"):
            hi = ranges[layer][kind].to(args.device)
            lo = torch.zeros_like(hi)
            rows = []
            for p in PROMPTS[:args.prompts]:
                ids = tok(p, return_tensors="pt").input_ids.to(args.device)
                past = prefix_cache(model, ids)
                r = attack(model, hook, ids[:, -1:], past, lo, hi, args.cap, args.moves == "down")
                a_t = r.pop("a_tampered").cpu()
                w = a_t.abs() * col
                p_unif = r["k"] / d_ff
                p_contrib = float(w[r["edited"]].sum() / w.sum()) if r["edited"] else 0.0
                r.update(prompt=p, layer=layer, range=kind, d_ff=d_ff, p_uniform=p_unif, p_contribution=p_contrib)
                rows.append(r)
            ok = [r for r in rows if r["success"]]
            ks = [r["k"] for r in ok]

            def paths(pp):
                return None if pp <= 0 else math.ceil(40 * math.log(2) / -math.log1p(-pp))
            summary = {"model": args.model, "moves": args.moves, "layer": layer, "range": kind, "d_ff": d_ff,
                       "prompts": len(rows),
                       "success": len(ok), "k_median": float(np.median(ks)) if ks else None,
                       "k_min": min(ks) if ks else None, "k_max": max(ks) if ks else None,
                       "p_uniform_median": float(np.median([r["p_uniform"] for r in ok])) if ok else None,
                       "p_contribution_median": float(np.median([r["p_contribution"] for r in ok])) if ok else None,
                       "paths_2e-40_uniform_median": paths(float(np.median([r["p_uniform"] for r in ok]))) if ok else None,
                       "seconds": round(time.perf_counter() - t0, 1)}
            print(json.dumps(summary), flush=True)
            records.extend(rows)
        handle.remove()
    out = ROOT / "artifacts" / "results" / f"inrange_llm_{args.model.split('/')[-1]}_{args.moves}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "records": records}, indent=1))


if __name__ == "__main__":
    main()
