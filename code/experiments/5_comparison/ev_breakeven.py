"""Plan 2 EV (a): the deployment case as two break-even panels, from measured cells and the byte model.

    cd code && PYTHONPATH=src python experiments/5_comparison/ev_breakeven.py --inputs experiments/5_comparison/ev_inputs_draft.json \
        --out artifacts/results/ev_breakeven_draft

* **Panel 1, client bytes against the number of queries Q.** Downloading the int8 weights once (``W`` bytes), against
  ``Q`` mode-C proofs of ``b`` bytes each: the crossover is ``Q* = W / b``. ``b`` = the measured wire claims plus the
  fixed part (``u``, opened columns, Merkle paths) of ``analytic.proof_bytes`` for the full model, interactive and
  under Fiat-Shamir, for each commitment plan; at 2,048 tokens also V1 (claims divided by the measured V1 ratio).
* **Panel 2, a Kpre client's time against re-execution, by link speed L.** The client receives the proof (``B`` bytes)
  and verifies it in ``t_v``; re-executing takes ``t_r`` (RX's strongest exact baseline on the same threads). The
  client is faster when ``t_v + 8 B / L < t_r``, i.e. above the break-even link ``L* = 8 B / (t_r - t_v)`` (none if
  ``t_v >= t_r``).

Writes ``<out>.json`` (every crossover with its inputs), ``<out>.pdf`` (the figure, one column wide, in the paper's
style: 7 pt text) and ``<out>.tex`` (the crossovers as comment lines). ``--figure <path>`` writes the figure there
as well (the paper's ``figures/ev_breakeven.pdf``). The figure draws the first commitment plan of the inputs (the
final one) under Fiat-Shamir: mode-C bytes at 64 and 2,048 tokens and with V1 at 2,048; the Kpre client's time at
2,048 tokens for v0 and V1 against the strongest exact re-execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pvi.commit.cost import decoder_shapes, proof_bytes
from pvi.commit.plans import plan_commitment
from pvi.v0.protocol import params_for
from pvi.models.decoder import CONFIGS


def crossover_queries(weights_bytes: float, proof_bytes_per_query: float) -> float:
    """``Q*`` with ``Q* b = W``: below it the proofs cost the client fewer bytes than the weights."""
    if proof_bytes_per_query <= 0:
        raise ValueError("a proof has a positive size")
    return weights_bytes / proof_bytes_per_query


def breakeven_link_gbps(proof_bytes: float, verify_s: float, reexec_s: float) -> float | None:
    """``L*`` in Gb/s with ``t_v + 8 B / L* = t_r``, or ``None`` when verifying alone takes as long as re-executing."""
    if verify_s >= reexec_s:
        return None
    return 8 * proof_bytes / (reexec_s - verify_s) / 1e9


def fixed_part(model: str, policy: str, fs: bool, seq: int) -> dict:
    """The mode-C fixed part of one query for the full model (pruned last block)."""
    ops = decoder_shapes(CONFIGS[model], prune_last=True)
    plan = plan_commitment(ops, policy, model_ops=ops)
    params = params_for(128, len(ops), plan=plan, fiat_shamir=fs)
    b = proof_bytes(ops, params, seq, plan=plan, mode="C")
    return {"u": b["u"], "columns": b["columns"], "paths": b["paths"], "fixed": b["u"] + b["columns"] + b["paths"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure", default="", help="also write the figure to this path")
    args = ap.parse_args()
    cfg = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    model = cfg["model"]
    W = cfg["weights_int8_GB"]["value"] * 1e9
    out = {"inputs": args.inputs, "model": model, "weights_bytes": W, "panel1": [], "panel2": []}

    for seq in (64, 2048):
        claims = cfg["claims_wire_MB"][str(seq)]["value"] * 1e6
        for policy in cfg["mode_c_policies"]:
            for fs in (False, True):
                f = fixed_part(model, policy, fs, seq)
                b = claims + f["fixed"]
                out["panel1"].append({"seq": seq, "protocol": "v0", "policy": policy, "fs": fs, "claims": claims,
                                      **f, "proof_bytes": b, "Q_star": crossover_queries(W, b)})
        r = cfg["v1_ratio_2048"]["value"]
        if seq == 2048:
            for fs in (False, True):
                f = fixed_part(model, cfg["mode_c_policies"][0], fs, seq)
                b = claims / r + f["fixed"]
                out["panel1"].append({"seq": seq, "protocol": "v1", "policy": cfg["mode_c_policies"][0], "fs": fs,
                                      "claims": claims / r, **f, "proof_bytes": b, "Q_star": crossover_queries(W, b)})

        t_r = cfg["reexec_s"][str(seq)]["strongest_exact"]
        for fs in (False, True):
            t_v = cfg["verifier_s"][str(seq)]["fs" if fs else "int"]
            out["panel2"].append({"seq": seq, "protocol": "v0 Kpre", "fs": fs, "proof_bytes": claims, "verify_s": t_v,
                                  "reexec_s": t_r, "L_star_gbps": breakeven_link_gbps(claims, t_v, t_r)})
        if seq == 2048:
            for key, t_v in cfg["v1_verifier_s_2048"].items():
                if key.startswith("_"):
                    continue
                label = {"measured_torch": "V1 Kpre (before V-V1N)", "v1n_torch": "V1 Kpre (V-V1N parts 1-4)",
                         "v1n_native": "V1 Kpre"}.get(key, key)
                out["panel2"].append({"seq": seq, "protocol": label, "fs": True,
                                      "proof_bytes": claims / cfg["v1_ratio_2048"]["value"], "verify_s": t_v["value"],
                                      "reexec_s": t_r, "L_star_gbps": breakeven_link_gbps(
                                          claims / cfg["v1_ratio_2048"]["value"], t_v["value"], t_r)})

    Path(args.out + ".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    macros = []
    for row in out["panel1"]:
        if row["seq"] == 64 and row["protocol"] == "v0":
            macros.append(f"% Q* at 64 tokens, {row['policy']}, {'FS' if row['fs'] else 'interactive'}: "
                          f"{row['Q_star']:.1f} ({row['proof_bytes'] / 1e6:.1f} MB per query)")
    for row in out["panel2"]:
        L = row["L_star_gbps"]
        macros.append(f"% L* {row['protocol']} at {row['seq']} tokens, {'FS' if row['fs'] else 'interactive'}: "
                      + (f"{L:.2f} Gb/s" if L is not None else "none (verifier slower than re-execution)"))
    Path(args.out + ".tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print("\n".join(macros))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                             "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 7, "xtick.labelsize": 7,
                             "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                             "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
                             "ytick.major.pad": 2, "axes.labelpad": 2, "pdf.fonttype": 42, "ps.fonttype": 42})
        ours, ours_k = "#1f5f99", "#7fb2e5"
        kpre = "\U0001d5aa" + "$_{\\mathrm{pre}}$"      # upright sans-serif K, as paper_assets.KPRE
        final = cfg["mode_c_policies"][0]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.375, 1.9))
        qs = [10 ** (i / 20) for i in range(0, 61)]
        a1.plot(qs, [W / 1e9] * len(qs), "k--", lw=0.9, label="weights")
        style = {(64, "v0"): (ours, "-", "64"), (2048, "v0"): (ours, ":", "2,048"),
                 (2048, "v1"): (ours_k, "-", "2,048, V1")}
        for row in out["panel1"]:
            key = (row["seq"], row["protocol"])
            if row["policy"] == final and row["fs"] and key in style:
                color, ls, lab = style[key]
                a1.plot(qs, [q * row["proof_bytes"] / 1e9 for q in qs], color=color, ls=ls, lw=1.1, label=lab)
        a1.set_xscale("log"); a1.set_yscale("log")
        a1.set_xlabel("queries $Q$"); a1.set_ylabel("client bytes (GB)")
        a1.legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.2)
        links = [10 ** (i / 20 - 1) for i in range(0, 61)]
        for row in out["panel2"]:
            if row["seq"] == 2048 and row["fs"] and row["protocol"] in ("v0 Kpre", "V1 Kpre"):
                v1 = row["protocol"].startswith("V1")
                a2.plot(links, [row["verify_s"] + 8 * row["proof_bytes"] / (L * 1e9) for L in links],
                        color=ours_k if v1 else ours, lw=1.1, label="V1" if v1 else "v0")
        t_r = cfg["reexec_s"]["2048"]["strongest_exact"]
        a2.plot(links, [t_r] * len(links), "k--", lw=0.9, label="re-exec.")
        a2.set_xscale("log"); a2.set_yscale("log"); a2.set_ylim(10, 1e3)
        a2.set_xlabel("link (Gb/s)"); a2.set_ylabel(kpre + " client time (s)")
        a2.legend(loc="upper right", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.2)
        fig.tight_layout(pad=0.3, w_pad=0.8)
        fig.savefig(args.out + ".pdf")
        if args.figure:
            fig.savefig(args.figure)

if __name__ == "__main__":
    main()
