"""The sampler game on a real LLM, OPT-1.3B: the LP mixture over value-aware samplers against the in-range edits
(Sec. 3.5, ``sec:break``; ``ext:attack-inrange``, paragraph *The sampler game on OPT-1.3B*).

This is the OPT counterpart of ``adaptive_game.py``.  The edit sits at one FFN layer, at the last position, so,
conditional on a path entering that FFN at that position, detection is decided by one step of the sampler over the
``d_ff`` fc2-input neurons.  Detection conditional on entering the FFN is an upper bound on the sampler's
unconditional detection.

No new edit is searched for.  The edits are those of ``inrange_llm.py`` (:func:`inrange_llm.attack`):

* ``any``: greedy in-range moves of the neurons with the largest first-order margin decrease, each to the bound of
  its range (0, or the cap's ceiling), until the next token changes or ``--edit-cap`` neurons are edited;
* ``down``: the same with decreases only (to 0), which contribution weighting never opens.

For each cap ``p`` (``--caps``) the ``any`` edit runs with ceilings at the ``p``-th percentile of held-out activations
(``--calib-windows`` windows of ``--calib-seq`` tokens of WikiText-2 *train*), and edits found at a tighter cap are
carried to every looser cap, where they stay in range; the ``down`` edit does not depend on the cap.  Each sampler's
one-step detection of each edit fills the matrix, the per-sampler value is the mean over prompts of the adversary's
best edit, and the LP mixture is the per-prompt formulation of the MNIST game (``adaptive_game.solve_mixture``).
One-step detection of an edit with neuron set ``E`` and claimed fc2 input ``a`` (``W`` the ``fc2`` weight):

* uniform:            ``|E| / d_ff``;
* static importance:  ``sum_{j in E} ||W[:,j]|| / sum_j ||W[:,j]||`` (outgoing magnitude, value-blind);
* contribution:       ``sum_{j in E} |a_j| ||W[:,j]|| / sum_j |a_j| ||W[:,j]||``;
* floor ``delta``:    the same with ``|a_j| + delta`` in place of ``|a_j|`` (``--floors``);
* tail-first:         uniform among neurons with ``a_j > 0`` at or above the ``--tail-percentile`` ceiling (else
  uniform);
* zero-first:         uniform among neurons with ``a_j == 0`` (else uniform).

The adversary has only the two fixed edits, so every number is an upper bound on the sampler's worst case, and a
loose one for samplers that target a value the two edits happen to write (``down`` writes exact zeros; ``any``
writes the ceiling).  On the MNIST MLP the adversary's replies, lowering to a value just above zero and the greedy
guided by the sampler's own detection, remove those gains; here neither is run.

Paper (``ext:attack-inrange``): "With caps at the 90th, 95th, 99th and 100th percentile, the mixture detects 3.1%,
23%, 38% and 48% at layer 6; 0.23%, 1.2%, 5.4% and 6.8% at layer 12; 0.50%, 1.2%, 1.6% and 2.2% at layer 18; and
0.09%, 0.09%, 0.08% and 0.08% at layer 23, against 27.7/d_ff = 0.34% and 0.02--0.19% for uniform sampling"; Sec. 3.5
quotes the cap-100 column.

Usage (from ``code/``; ``requirements-llm.txt`` and SciPy; checkpoint and WikiText-2 are downloaded to the Hugging
Face cache)::

    python -m experiments.attack.adaptive_game_llm --threads 8 --device cuda       # the paper's run
    python -m experiments.attack.adaptive_game_llm --model facebook/opt-125m --layers 6 --prompts 2 \\
        --calib-windows 8 --out /tmp/adaptive_game_opt-125m.json                   # a laptop check, not a result

Prints one JSON line with the calibration and one line per (layer, cap).  Writes ``--out`` (default
``artifacts/results/adaptive_game_<model>.json``): ``model``, ``layers``, ``caps``, ``prompts``, ``tail_percentile``,
``calib_tokens``, ``edit_cap``, ``floors``, ``seed``, ``note``, ``env``, ``seconds`` and, per layer and cap,
``success_rate``, ``mean_min_support``, ``d_ff``, ``samplers.<name>.{detection_per_path,s_40}``,
``best_single_sampler`` and ``mixture`` (``weights``, ``detection_per_path``, ``s_40``).  OPT-1.3B in fp32 needs
about 5.3 GB of GPU memory: one RTX 2080 Ti, about 13 min, with 8 cores and 32 GB of host memory.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from experiments.attack.adaptive_game import s_for_target, solve_mixture
from experiments.attack.attack_llm import PROMPTS, prefix_cache
from experiments.attack.inrange_llm import Hook, attack, calibration_tokens
from experiments.common.checkpoints import load_opt
from experiments.common.paths import RESULTS
from experiments.common.records import environment, write_json

FLOOR_EPSILONS = (0.01, 0.1, 1.0, 10.0, 100.0)
"""The floors of the MNIST game's family."""
EDIT_CAP = 60
"""Edited neurons per edit, at most: the most that an ``any`` edit of ``tab:inrange`` needs on OPT-1.3B
(``inrange_llm.py`` itself caps at 256)."""


def sampler_names(floors) -> list[str]:
    return ["uniform", "static-importance", "contribution", *[f"floor-{e:g}" for e in floors],
            "tail-first", "zero-first"]


def calibrate_percentiles(model, tok, fc2s: dict, windows: int, seq: int, percentiles: list[float]) -> dict:
    """Per layer and percentile: each fc2 input's held-out ceiling (minimum 0, ReLU).

    Activations of the fc2 inputs over ``windows`` windows of ``seq`` tokens of WikiText-2 *train* (disjoint from
    the prompts), every position.  Quantiles are taken neuron-chunk by neuron-chunk with NumPy, which avoids
    ``torch.quantile``'s input-size limit at ``d_ff = 8192``.
    """
    ids = calibration_tokens(windows * seq, tok).view(windows, seq).to(next(model.parameters()).device)
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


def sampler_detections(a: np.ndarray, col: np.ndarray, edited: list[int], thr: np.ndarray, floors) -> np.ndarray:
    """One-step detection of every sampler (order :func:`sampler_names`) for claimed input ``a`` and edit
    ``edited``."""
    d_ff = a.shape[0]
    k = len(edited)
    a_abs = np.abs(a)
    det = {"uniform": k / d_ff,
           "static-importance": float(col[edited].sum() / col.sum()) if k else 0.0}
    w = a_abs * col
    det["contribution"] = float(w[edited].sum() / w.sum()) if w.sum() > 0 else k / d_ff
    for eps in floors:
        we = (a_abs + eps) * col
        det[f"floor-{eps:g}"] = float(we[edited].sum() / we.sum())
    tail = (a >= thr) & (a > 0.0)
    det["tail-first"] = float(tail[edited].sum()) / int(tail.sum()) if tail.any() else k / d_ff
    zeros = a == 0.0
    det["zero-first"] = float(zeros[edited].sum()) / int(zeros.sum()) if zeros.any() else k / d_ff
    return np.array([det[n] for n in sampler_names(floors)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/opt-1.3b", help="an OPT checkpoint of the Hugging Face hub")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23], help="decoder layers attacked")
    ap.add_argument("--prompts", type=int, default=40, help="the first this many of the 40 prompts")
    ap.add_argument("--calib-windows", type=int, default=64, help="calibration windows of WikiText-2 train")
    ap.add_argument("--calib-seq", type=int, default=128, help="tokens per calibration window")
    ap.add_argument("--caps", type=float, nargs="+", default=[90.0, 95.0, 99.0, 100.0],
                    help="caps as percentiles of the held-out activations")
    ap.add_argument("--edit-cap", type=int, default=EDIT_CAP, help="edited neurons per edit, at most")
    ap.add_argument("--floors", type=float, nargs="+", default=list(FLOOR_EPSILONS),
                    help="floors of the zero-aware contribution samplers")
    ap.add_argument("--tail-percentile", type=float, default=95.0, help="the tail-first sampler's cut")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu, or cuda (fp32 weights on the GPU)")
    ap.add_argument("--out", default=None, help="default artifacts/results/adaptive_game_<model>.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = args.out or str(RESULTS / f"adaptive_game_{args.model.split('/')[-1]}.json")
    floors = tuple(args.floors)
    names = sampler_names(floors)

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

    impl_cols = list(range(len(names)))  # every sampler here is implementable
    prompts = PROMPTS[: args.prompts]
    report: dict = {
        "model": args.model, "layers": args.layers, "caps": caps, "prompts": len(prompts),
        "tail_percentile": args.tail_percentile, "calib_tokens": ranges[args.layers[0]]["tokens"],
        "edit_cap": args.edit_cap, "floors": list(floors), "seed": args.seed,
        "note": ("One-step FFN detection conditional on the path entering that FFN at the last position "
                 "(an upper bound on unconditional detection). Per-sampler value: mean over prompts of the "
                 "adversary's best edit among the existing 'any' (per cap, carried to looser caps) and "
                 "'down' edits of inrange_llm.py; an upper bound on the sampler's worst case."),
        "env": environment(args.device),
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
                    r = attack(model, hook, ids[:, -1:], past, lo, hi, args.edit_cap, mv == "down")
                    a_t = r.pop("a_tampered").cpu().float().numpy()
                    if r["success"] and r["edited"]:
                        rows[pi].append((sampler_detections(a_t, col, list(r["edited"]), thr, floors),
                                         len(r["edited"])))
            detection = [np.vstack([d for d, _k in rs]) for rs in rows if rs]
            supports = [min(k for _d, k in rs) for rs in rows if rs]
            if not detection:
                cap_results[f"{cap:g}"] = {"success_rate": 0.0}
                continue
            per_sampler = {names[i]: float(np.mean([m[:, i].min() for m in detection])) for i in range(len(names))}
            alpha, value = solve_mixture(detection, impl_cols)
            best = max((n for n in names if n != "uniform"), key=lambda n: per_sampler[n])
            cap_results[f"{cap:g}"] = {
                "success_rate": len(detection) / len(prompts),
                "mean_min_support": float(np.mean(supports)),
                "d_ff": d_ff,
                "samplers": {n: {"detection_per_path": per_sampler[n], "s_40": s_for_target(per_sampler[n])}
                             for n in names},
                "best_single_implementable": best,
                "best_single_sampler": best,
                "mixture": {"weights": {names[i]: float(alpha[i]) for i in range(len(names)) if alpha[i] > 1e-6},
                            "detection_per_path": value, "s_40": s_for_target(value)},
            }
            print(f"layer {layer} (d_ff={d_ff}) cap {cap:g}: success {len(detection)}/{len(prompts)}  "
                  f"unif {per_sampler['uniform']:.2e}  best[{best}] {per_sampler[best]:.4f}  "
                  f"mix {value:.4f}  [{time.perf_counter() - t0:.0f}s]", flush=True)
        handle.remove()
        report["per_layer"][str(layer)] = {"d_ff": d_ff, "caps": cap_results}

    report["seconds"] = time.perf_counter() - t0
    write_json(out, report)
    print(f"\nwrote {out} in {report['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
