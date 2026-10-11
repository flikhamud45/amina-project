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

Writes ``<out>.json`` (every crossover with its inputs), ``<out>.pdf`` (the figure) and ``<out>.tex`` (macros).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pvi.fullcheck.analytic import decoder_shapes, proof_bytes
from pvi.fullcheck.plans import plan_commitment
from pvi.fullcheck.protocol import params_for
from pvi.fullcheck.transformer import CONFIGS


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
                         "v1n_native": "V1 Kpre (V-V1N, native products)"}.get(key, key)
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
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
        qs = [10 ** (i / 20) for i in range(0, 61)]
        a1.plot(qs, [W / 1e9] * len(qs), "k-", lw=1.2, label="download int8 weights")
        for row in out["panel1"]:
            if row["seq"] == 64 and row["protocol"] == "v0" and (row["policy"], row["fs"]) in (
                    (cfg["mode_c_policies"][0], False), (cfg["mode_c_policies"][0], True), (cfg["mode_c_policies"][-1], True)):
                a1.plot(qs, [q * row["proof_bytes"] / 1e9 for q in qs],
                        label=f"mode C, 64 tok., {row['policy']}{', FS' if row['fs'] else ''}")
        a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("queries Q"); a1.set_ylabel("client bytes (GB)")
        a1.legend(fontsize=6, loc="upper left")
        links = [10 ** (i / 20 - 1) for i in range(0, 61)]
        for row in out["panel2"]:
            if row["seq"] == 2048 and (row["fs"] or row["protocol"].startswith("V1")):
                a2.plot(links, [row["verify_s"] + 8 * row["proof_bytes"] / (L * 1e9) for L in links],
                        label=f"{row['protocol']}")
        t_r = cfg["reexec_s"]["2048"]["strongest_exact"]
        a2.plot(links, [t_r] * len(links), "k--", lw=1.2, label="re-execution (strongest exact)")
        a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("link (Gb/s)"); a2.set_ylabel("client time, 2,048 tok. (s)")
        a2.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        fig.savefig(args.out + ".pdf")


if __name__ == "__main__":
    main()
