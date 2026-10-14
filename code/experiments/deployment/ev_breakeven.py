"""When a client gains from the protocol: the break-even of fig:breakeven, from measured inputs and the byte model.

Produces ``fig:breakeven`` (Sec. ``sec:eval-reexec``, "Break-even") and the crossovers the discussion quotes for
Llama-2-7B under Fiat--Shamir (Sec. ``sec:discussion``, "Who the protocol serves": the number of 64-token queries below
which mode-C proofs cost fewer bytes than one download of the int8 weights, one query at 2,048 tokens or 2.3 with V1,
and the links above which a Kpre client verifies faster than it re-executes; "Proof size": the prompt length at which
one proof reaches the size of the weights; the extended version's ``ext:disc-real``, "V1": the link below which V1's
client is faster than v0's).  It computes from ``--inputs`` (``ev_inputs.json``: measured claims, verifier and
re-execution times, typed in with their sources; see ``experiments/deployment/README.md``) and the byte model of mode
C's fixed part (:func:`pvi.commit.cost.proof_bytes`); nothing is measured here.

    python -m experiments.deployment.ev_breakeven
    python -m experiments.deployment.ev_breakeven --figure ../paper_sp2027/figures/ev_breakeven.pdf
    python -m experiments.deployment.ev_breakeven --policy cnn18c R64c --no-figure

* **Client bytes against the number of queries Q** (``panel1``; the figure's left panel).  Downloading the int8
  weights once (``W`` bytes), against ``Q`` mode-C proofs of ``b`` bytes each: the crossover is ``Q* = W / b``.
  ``b`` = the measured claims in the compact encoding plus the fixed part (``u``, opened columns, Merkle paths) of
  ``proof_bytes`` for the full model with its last block pruned (the measured claims are the pruned graph's),
  interactive and under Fiat--Shamir, for each ``--policy``; at 2,048 tokens also V1 (the claims divided by the
  measured ratio of v0's claims to V1's).  The claims are a Kpre cell's (the verifier reads the embedding rows
  itself); mode C also sends the looked-up embedding rows of the plan's lookup table, ``T x d_model`` bytes (about
  0.1% of the proof), which are left out.
* **A Kpre client's time against re-execution, by link speed L** (``panel2``; the right panel).  The client receives
  the proof (``B`` bytes, the claims) and verifies it in ``t_v``; re-executing takes ``t_r`` (the fastest exact
  re-execution on the same threads).  The client is faster when ``t_v + 8 B / L < t_r``, i.e. above the break-even
  link ``L* = 8 B / (t_r - t_v)`` (none if ``t_v >= t_r``).
* **V1 against v0** (``v1_against_v0``): V1's client (fewer bytes, a slower verifier) is faster than v0's below
  ``L = 8 (B_v0 - B_V1) / (t_V1 - t_v0)``.
* **The prompt length at which one proof weighs as much as the weights** (``prompt_crossover``): the mode-C proof
  bytes taken linear in the prompt length through the two measured lengths (the claims grow with the prompt, the
  fixed part barely), solved for ``W``; V1's by the same line with its claims divided by the ratio (measured at 2,048
  tokens only, so beyond that length a byte-model estimate).

Writes ``<out>.json`` (every crossover with its inputs), ``<out>.tex`` (the crossovers as LaTeX comment lines, also
printed) and ``<out>.pdf`` (the figure, one column wide in the paper's style: 7 pt text; ``--no-figure`` skips it,
``--figure <path>`` writes it there as well).  The figure draws the first ``--policy`` (the paper's plan ``cnn18c``)
under Fiat--Shamir: mode-C bytes at 64 and 2,048 tokens and with V1 at 2,048; the Kpre client's time at 2,048 tokens
for v0 and V1 against the fastest exact re-execution.  It needs matplotlib and takes a second on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.common import cli
from experiments.common.paths import RESULTS
from pvi.commit.cost import decoder_shapes, proof_bytes
from pvi.commit.plans import plan_commitment
from pvi.models.decoder import CONFIGS
from pvi.v0.params import params_for

INPUTS = Path(__file__).resolve().with_name("ev_inputs.json")
"""The stored inputs: Llama-2-7B, measured on an RTX 2080 Ti prover and 8 threads of a Xeon Silver 4114."""


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


def crossover_tokens(weights_bytes: float, short: tuple[int, float], long: tuple[int, float]) -> float | None:
    """The prompt length at which a proof of ``bytes(T)``, linear through ``short = (T1, bytes)`` and ``long = (T2,
    bytes)``, reaches ``weights_bytes`` (``None`` if the proof does not grow with the prompt)."""
    (t1, b1), (t2, b2) = short, long
    per_token = (b2 - b1) / (t2 - t1)
    if per_token <= 0:
        return None
    return t1 + (weights_bytes - b1) / per_token


def fixed_part(model: str, policy: str, fs: bool, seq: int, lam: int = cli.PAPER.lam, rate: int = cli.RATE) -> dict:
    """The mode-C fixed part of one query for the full model (last block pruned) under the plan ``policy``."""
    ops = decoder_shapes(CONFIGS[model], prune_last=True)
    plan = plan_commitment(ops, policy, rate=rate, model_ops=ops)
    params = params_for(lam, len(ops), rate=rate, plan=plan, fiat_shamir=fs)
    b = proof_bytes(ops, params, seq, plan=plan, rate=rate, mode="C")
    return {"u": b["u"], "columns": b["columns"], "paths": b["paths"], "fixed": b["u"] + b["columns"] + b["paths"]}


def crossovers(cfg: dict, policies: list[str], lam: int = cli.PAPER.lam, rate: int = cli.RATE) -> dict:
    """Every crossover of the inputs ``cfg`` (the parsed ``ev_inputs.json``) for the plans ``policies``."""
    model = cfg["model"]
    W = cfg["weights_int8_GB"]["value"] * 1e9
    r = cfg["v1_ratio_2048"]["value"]
    seqs = sorted(int(s) for s in cfg["claims_wire_MB"])
    out = {"model": model, "weights_bytes": W, "lam": lam, "rate": rate, "policies": list(policies),
           "panel1": [], "panel2": [], "v1_against_v0": [], "prompt_crossover": []}
    for seq in seqs:
        claims = cfg["claims_wire_MB"][str(seq)]["value"] * 1e6
        for policy in policies:
            for fs in (False, True):
                f = fixed_part(model, policy, fs, seq, lam, rate)
                b = claims + f["fixed"]
                out["panel1"].append({"seq": seq, "protocol": "v0", "policy": policy, "fs": fs, "claims": claims,
                                      **f, "proof_bytes": b, "Q_star": crossover_queries(W, b)})
        if seq == 2048:
            for fs in (False, True):
                f = fixed_part(model, policies[0], fs, seq, lam, rate)
                b = claims / r + f["fixed"]
                out["panel1"].append({"seq": seq, "protocol": "v1", "policy": policies[0], "fs": fs,
                                      "claims": claims / r, **f, "proof_bytes": b, "Q_star": crossover_queries(W, b)})
        t_r = cfg["reexec_s"][str(seq)]["exact"]
        for fs in (False, True):
            t_v = cfg["verifier_s"][str(seq)]["fs" if fs else "int"]
            out["panel2"].append({"seq": seq, "protocol": "v0 Kpre", "fs": fs, "proof_bytes": claims, "verify_s": t_v,
                                  "reexec_s": t_r, "L_star_gbps": breakeven_link_gbps(claims, t_v, t_r)})
        if seq == 2048:              # V1's verifier is measured under Fiat--Shamir
            t_v1 = cfg["v1_verifier_s_2048"]["value"]
            out["panel2"].append({"seq": seq, "protocol": "V1 Kpre", "fs": True, "proof_bytes": claims / r,
                                  "verify_s": t_v1, "reexec_s": t_r,
                                  "L_star_gbps": breakeven_link_gbps(claims / r, t_v1, t_r)})
            t_v0 = cfg["verifier_s"][str(seq)]["fs"]
            out["v1_against_v0"].append({"seq": seq, "fs": True, "v0_proof_bytes": claims, "v1_proof_bytes": claims / r,
                                         "v0_verify_s": t_v0, "v1_verify_s": t_v1,
                                         "v1_faster_below_gbps": breakeven_link_gbps(claims - claims / r, t_v0, t_v1)})
    lo, hi = seqs[0], seqs[-1]
    for policy in policies:
        for fs in (False, True):
            for protocol, div in (("v0", 1.0), ("v1", r)):
                if protocol == "v1" and policy != policies[0]:
                    continue
                ends = [(t, cfg["claims_wire_MB"][str(t)]["value"] * 1e6 / div
                         + fixed_part(model, policy, fs, t, lam, rate)["fixed"]) for t in (lo, hi)]
                out["prompt_crossover"].append({"protocol": protocol, "policy": policy, "fs": fs,
                                                "proof_bytes_at": {str(t): b for t, b in ends},
                                                "tokens": crossover_tokens(W, *ends)})
    return out


def comment_lines(out: dict) -> list[str]:
    """The crossovers as LaTeX comment lines."""
    mode = {False: "interactive", True: "FS"}
    lines = []
    for row in out["panel1"]:
        if row["seq"] == 64 and row["protocol"] == "v0":
            lines.append(f"% Q* at 64 tokens, {row['policy']}, {mode[row['fs']]}: "
                         f"{row['Q_star']:.1f} ({row['proof_bytes'] / 1e6:.1f} MB per query)")
    for row in out["panel2"]:
        L = row["L_star_gbps"]
        lines.append(f"% L* {row['protocol']} at {row['seq']} tokens, {mode[row['fs']]}: "
                     + (f"{L:.2f} Gb/s" if L is not None else "none (verifier slower than re-execution)"))
    for row in out["v1_against_v0"]:
        L = row["v1_faster_below_gbps"]
        lines.append(f"% V1's Kpre client faster than v0's at {row['seq']} tokens, {mode[row['fs']]}: "
                     + (f"below {L:.2f} Gb/s" if L is not None else "never"))
    for row in out["prompt_crossover"]:
        t = row["tokens"]
        lines.append(f"% one {row['protocol']} mode-C proof as large as the weights, {row['policy']}, "
                     f"{mode[row['fs']]}: " + (f"{t:,.0f} tokens" if t is not None else "never"))
    return lines


def draw(out: dict, cfg: dict, paths: list[str]) -> None:
    """The figure: the first plan under Fiat--Shamir, written to every one of ``paths``."""
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
    kpre = "\U0001d5aa" + "$_{\\mathrm{pre}}$"      # upright sans-serif K, the paper's mode Kpre
    final, W = out["policies"][0], out["weights_bytes"]
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
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("queries $Q$")
    a1.set_ylabel("client bytes (GB)")
    a1.legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.2)
    links = [10 ** (i / 20 - 1) for i in range(0, 61)]
    for row in out["panel2"]:
        if row["seq"] == 2048 and row["fs"] and row["protocol"] in ("v0 Kpre", "V1 Kpre"):
            v1 = row["protocol"].startswith("V1")
            a2.plot(links, [row["verify_s"] + 8 * row["proof_bytes"] / (L * 1e9) for L in links],
                    color=ours_k if v1 else ours, lw=1.1, label="V1" if v1 else "v0")
    t_r = cfg["reexec_s"]["2048"]["exact"]
    a2.plot(links, [t_r] * len(links), "k--", lw=0.9, label="re-exec.")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_ylim(10, 1e3)
    a2.set_xlabel("link (Gb/s)")
    a2.set_ylabel(kpre + " client time (s)")
    a2.legend(loc="upper right", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", default=str(INPUTS),
                    help="the measured inputs (default: experiments/deployment/ev_inputs.json)")
    ap.add_argument("--out", default=str(RESULTS / "ev_breakeven"),
                    help="the output prefix: <out>.json, <out>.tex, <out>.pdf (default artifacts/results/ev_breakeven)")
    ap.add_argument("--policy", nargs="+", default=[cli.PAPER.policy],
                    help="mode C's commitment plans of the byte model (the figure draws the first)")
    ap.add_argument("--lam", type=int, default=cli.PAPER.lam, help="security level in bits")
    ap.add_argument("--rate", type=int, default=cli.RATE, help="Reed-Solomon rate of the weight commitment")
    ap.add_argument("--no-figure", action="store_true", help="write the JSON and the comment lines only")
    ap.add_argument("--figure", default="", help="also write the figure to this path")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    out = {"inputs": args.inputs, **crossovers(cfg, args.policy, args.lam, args.rate)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = comment_lines(out)
    Path(args.out + ".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not args.no_figure:
        draw(out, cfg, [args.out + ".pdf"] + ([args.figure] if args.figure else []))


if __name__ == "__main__":
    main()
