"""The adaptive game on a real LLM: the LP mixture over value-aware samplers (plan G3-min, item 3).

This is the OPT counterpart of ``adaptive_game.py``.  The edit sits at one FFN layer, at the last
position, so -- conditional on a path entering that FFN at that position -- detection is decided by
one step of the sampler over the ``d_ff`` fc2-input neurons.  Detection conditional on entering the
FFN is an upper bound on the sampler's unconditional detection, so it is conservative for the claim.

Nothing new is searched for here.  The edits are those of ``experiments/2_attack/inrange_llm.py``
(imported, not copied), which build on ``real_llm.py``'s ``prefix_cache`` and ``last_logits``:

* ``any``: greedy in-range moves of the neurons with the largest first-order margin decrease, each
  to the bound of its range (0, or the cap's ceiling), until the next token changes;
* ``down``: the same with decreases only (to 0), which contribution weighting never opens.

The additions are the range caps, the sampler family and the LP mixture.  For each cap ``p`` the
``any`` edit is run with ceilings at the ``p``-th percentile of held-out activations (WikiText-2
*train*), and edits found at a tighter cap are carried to every looser cap, where they stay in range;
the ``down`` edit does not depend on the cap.  Each sampler's one-step detection of each edit fills
the matrix, the per-sampler value is the mean over prompts of the adversary's best edit, and the LP
mixture is the same per-prompt formulation as on the MNIST MLP (``adaptive_game.solve_mixture``).

One-step detection of an edit with neuron set ``E`` and claimed fc2 input ``a``:

* uniform:            ``|E| / d_ff``;
* static importance:  ``sum_{j in E} ||W[:,j]|| / sum_j ||W[:,j]||`` (outgoing magnitude, value-blind);
* contribution:       ``sum_{j in E} |a_j| ||W[:,j]|| / sum_j |a_j| ||W[:,j]||``;
* floor eps:          the same with ``|a_j| + eps`` in place of ``|a_j|``;
* tail-first:         uniform among neurons with ``a_j > 0`` at or above the tail percentile (else uniform);
* zero-first:         uniform among neurons with ``a_j == 0`` (else uniform).

The adversary here has only the two existing edits, so every number is an upper bound on the
sampler's worst case, and a loose one for samplers that target a value the two edits happen to write
(``down`` writes exact zeros; ``any`` writes the ceiling).  On the MNIST MLP the obvious replies --
lowering to a value just above zero, and reusing a tighter cap's plan -- remove those gains.

Smoke test (laptop, OPT-125M only, <=2 GB, 2 threads):

    cd code && PYTHONPATH=src OMP_NUM_THREADS=2 \
      python experiments/3_sampling_fixes/adaptive_game_llm.py \
      --model facebook/opt-125m --layers 6 --prompts 2 --calib-windows 8 --caps 90 95 99 100 \
      --out artifacts/results/adaptive_game_opt-125m_smoke.json

Full run (cluster, OPT-1.3B):

    cd code && PYTHONPATH=src OMP_NUM_THREADS=8 \
      python experiments/3_sampling_fixes/adaptive_game_llm.py \
      --model facebook/opt-1.3b --layers 6 12 18 23 --prompts 40 \
      --calib-windows 64 --calib-seq 128 --caps 90 95 99 100 --threads 8 --device cuda \
      --out artifacts/results/adaptive_game_opt-1.3b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "2_attack"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inrange_llm import Hook, _calibration_text, attack  # noqa: E402
from real_llm import PROMPTS, prefix_cache  # noqa: E402

FLOOR_EPSILONS = (0.01, 0.1, 1.0, 10.0, 100.0)
NAMES = ["uniform", "static-importance", "contribution",
         *[f"floor-{e:g}" for e in FLOOR_EPSILONS], "tail-first", "zero-first"]
EDIT_CAP = 60  # neurons, as in G3.5's stored runs (at most 60 edited)


def calibrate_percentiles(model, tok, fc2s: dict, windows: int, seq: int, percentiles: list[float]) -> dict:
    """Per layer and percentile: each fc2 input's held-out ceiling (minimum 0, ReLU).

    Activations of the fc2 inputs over ``windows`` windows of ``seq`` tokens of WikiText-2 *train*
    (disjoint from the prompts), every position.  Quantiles are taken neuron-chunk by neuron-chunk
    with NumPy, which avoids ``torch.quantile``'s input-size limit at ``d_ff = 8192``.
    """
    ids = _calibration_text(windows * seq, tok).view(windows, seq).to(next(model.parameters()).device)
    hooks = {}
    for layer, fc2 in fc2s.items():
        h = Hook()
        h.record = []
        hooks[layer] = (h, fc2.register_forward_pre_hook(h))
    with torch.no_grad():
        for w in range(windows):
            model(input_ids=ids[w : w + 1])
    out = {}
    for layer, (h, handle) in hooks.items():
        handle.remove()
        acts = torch.cat(h.record, 0).cpu().float().numpy()
        h.record = None
        ceilings = {}
        for p in percentiles:
            ceilings[f"{p:g}"] = np.concatenate(
                [np.percentile(acts[:, s : s + 1024], p, axis=0) for s in range(0, acts.shape[1], 1024)]
            )
        out[layer] = {"tokens": int(acts.shape[0]), "ceilings": ceilings}
        del acts
    return out


def sampler_detections(a: np.ndarray, col: np.ndarray, edited: list[int], thr: np.ndarray) -> np.ndarray:
    """One-step detection of every sampler (order ``NAMES``) for claimed input ``a`` and edit ``edited``."""
    d_ff = a.shape[0]
    k = len(edited)
    a_abs = np.abs(a)
    det = {"uniform": k / d_ff,
           "static-importance": float(col[edited].sum() / col.sum()) if k else 0.0}
    w = a_abs * col
    det["contribution"] = float(w[edited].sum() / w.sum()) if w.sum() > 0 else k / d_ff
    for eps in FLOOR_EPSILONS:
        we = (a_abs + eps) * col
        det[f"floor-{eps:g}"] = float(we[edited].sum() / we.sum())
    tail = (a >= thr) & (a > 0.0)
    det["tail-first"] = float(tail[edited].sum()) / int(tail.sum()) if tail.any() else k / d_ff
    zeros = a == 0.0
    det["zero-first"] = float(zeros[edited].sum()) / int(zeros.sum()) if zeros.any() else k / d_ff
    return np.array([det[n] for n in NAMES])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23])
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--calib-seq", type=int, default=128)
    ap.add_argument("--caps", type=float, nargs="+", default=[90.0, 95.0, 99.0, 100.0])
    ap.add_argument("--tail-percentile", type=float, default=95.0, help="tail-first sampler cut")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "results" / "adaptive_game_llm.json"))
    args = ap.parse_args()

    from adaptive_game import s_for_target, solve_mixture  # the MNIST game's LP, same formulation

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT / "experiments" / "9_quality"))
    from common import load_opt  # noqa: E402

    t0 = time.perf_counter()
    model, tok = load_opt(args.model)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    decoder_layers = model.model.decoder.layers
    fc2s = {i: decoder_layers[i].fc2 for i in args.layers}
    caps = sorted(args.caps)
    ranges = calibrate_percentiles(
        model, tok, fc2s, args.calib_windows, args.calib_seq, sorted(set(caps) | {args.tail_percentile})
    )
    print(json.dumps({"model": args.model, "calibration_s": round(time.perf_counter() - t0, 1),
                      "tokens": ranges[args.layers[0]]["tokens"]}), flush=True)

    impl_cols = list(range(len(NAMES)))  # every sampler here is implementable
    prompts = PROMPTS[: args.prompts]
    report: dict = {
        "model": args.model, "layers": args.layers, "caps": caps, "prompts": len(prompts),
        "tail_percentile": args.tail_percentile, "calib_tokens": ranges[args.layers[0]]["tokens"],
        "note": ("One-step FFN detection conditional on the path entering that FFN at the last position "
                 "(an upper bound on unconditional detection). Per-sampler value: mean over prompts of the "
                 "adversary's best edit among the existing 'any' (per cap, carried to looser caps) and "
                 "'down' edits of inrange_llm.py; an upper bound on the sampler's worst case."),
        "per_layer": {},
    }

    for layer in args.layers:
        fc2 = fc2s[layer]
        d_ff = fc2.in_features
        col = fc2.weight.detach().float().norm(dim=0).cpu().numpy()  # ||W_fc2[:, j]||
        thr = np.asarray(ranges[layer]["ceilings"][f"{args.tail_percentile:g}"])
        hook = Hook()
        handle = fc2.register_forward_pre_hook(hook)
        # rows[prompt] = list of (detection vector, support) carried across ascending caps
        rows: list[list[tuple[np.ndarray, int]]] = [[] for _ in prompts]
        cap_results = {}
        for ci, cap in enumerate(caps):
            hi = torch.tensor(ranges[layer]["ceilings"][f"{cap:g}"], dtype=torch.float32).to(args.device)
            lo = torch.zeros_like(hi)
            for pi, text in enumerate(prompts):
                ids = tok(text, return_tensors="pt").input_ids.to(args.device)
                past = prefix_cache(model, ids)
                moves = ["any"] + (["down"] if ci == 0 else [])  # 'down' (to 0) is cap-independent
                for mv in moves:
                    r = attack(model, hook, ids[:, -1:], past, lo, hi, EDIT_CAP, mv == "down")
                    a_t = r.pop("a_tampered").cpu().float().numpy()
                    if r["success"] and r["edited"]:
                        rows[pi].append((sampler_detections(a_t, col, list(r["edited"]), thr),
                                         len(r["edited"])))
            detection = [np.vstack([d for d, _k in rs]) for rs in rows if rs]
            supports = [min(k for _d, k in rs) for rs in rows if rs]
            if not detection:
                cap_results[f"{cap:g}"] = {"success_rate": 0.0}
                continue
            per_sampler = {NAMES[i]: float(np.mean([m[:, i].min() for m in detection])) for i in range(len(NAMES))}
            alpha, value = solve_mixture(detection, impl_cols)
            best = max((n for n in NAMES if n != "uniform"), key=lambda n: per_sampler[n])
            cap_results[f"{cap:g}"] = {
                "success_rate": len(detection) / len(prompts),
                "mean_min_support": float(np.mean(supports)),
                "d_ff": d_ff,
                "samplers": {n: {"detection_per_path": per_sampler[n], "s_40": s_for_target(per_sampler[n])}
                             for n in NAMES},
                "best_single_implementable": best,
                "best_single_sampler": best,
                "mixture": {"weights": {NAMES[i]: float(alpha[i]) for i in range(len(NAMES)) if alpha[i] > 1e-6},
                            "detection_per_path": value, "s_40": s_for_target(value)},
            }
            print(f"layer {layer} (d_ff={d_ff}) cap {cap:g}: success {len(detection)}/{len(prompts)}  "
                  f"unif {per_sampler['uniform']:.2e}  best[{best}] {per_sampler[best]:.4f}  "
                  f"mix {value:.4f}  [{time.perf_counter() - t0:.0f}s]", flush=True)
        handle.remove()
        report["per_layer"][str(layer)] = {"d_ff": d_ff, "caps": cap_results}

    report["seconds"] = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out} in {report['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
