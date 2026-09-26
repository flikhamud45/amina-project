"""Results tables for the write-up, generated from ``artifacts/comparison/tables``.

    python scripts/comparison_report_tables.py
    # -> artifacts/comparison/tables/report_tables.md  and  report_tables.tex

Every number in COMPARISON_ANALYSIS.md and in the report's tables comes from
here, so it can always be traced to a stored measurement.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from comparison_figures import CNN_ORDER, DEFAULT, LLM_LABEL, PROVE, VERIFY, Measured, llm_rows, _llm_bytes, _read, _f

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "comparison" / "tables"
NAMES = {"mlp_mnist": "MLP (MNIST)", "lenet5": "LeNet-5 (MNIST)", "vgg11": "VGG-11 (CIFAR-10)",
         "vgg16": "VGG-16 (CIFAR-10)", "resnet18_cifar": "ResNet-18 (CIFAR-10)",
         "resnet18_224": "ResNet-18 (224px, dogs vs cats)"}


def t(sec):
    if sec is None:
        return "–"
    if sec < 1e-3:
        return f"{sec * 1e6:.0f} µs"
    if sec < 1:
        return f"{sec * 1e3:.1f} ms"
    if sec < 120:
        return f"{sec:.2f} s"
    return f"{sec / 60:.1f} min"


def b(n):
    if n is None:
        return "–"
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= k:
            return f"{n / k:.3g} {unit}"
    return f"{n:.0f} B"


def p(x):
    return "–" if x is None else f"{x * 100:.2f}%"


def params(n):
    if n is None:
        return "–"
    return f"{n / 1e9:.2f}B" if n >= 1e9 else (f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K")


class Doc:
    def __init__(self) -> None:
        self.md: list[str] = []
        self.tex: list[str] = []

    def table(self, title: str, header: list[str], rows: list[list[str]], label: str) -> None:
        self.md += [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        self.md += ["| " + " | ".join(r) + " |" for r in rows] + [""]
        esc = lambda s: (s.replace("%", r"\%").replace("µ", r"$\mu$").replace("λ", r"$\lambda$")
                         .replace("≤", r"$\le$").replace("×", r"$\times$").replace("_", r"\_"))
        self.tex += [f"% {title}", r"\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}", r"\toprule",
                     " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
        self.tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
        self.tex += [r"\bottomrule", r"\end{tabular}", f"% label: {label}", ""]


def main() -> None:
    M = Measured()
    d = Doc()
    lam = DEFAULT["lam"]

    # -- models and fidelity -------------------------------------------------------------
    rows = []
    for m in CNN_ORDER:
        if M.get(m, "facts", "n_params") is None:
            continue
        rows.append([NAMES[m], params(M.get(m, "facts", "n_params")), p(M.get(m, "facts", "float_accuracy")),
                     p(M.get(m, "facts", "int8_accuracy")), p(M.get(m, "facts", "float_int8_agreement")),
                     f"{M.get(m, 'facts', 'claim_values_per_query'):,.0f}"])
    d.table("Models: accuracy of the float model and of the exact int8 model the protocol verifies",
            ["model", "params", "float acc.", "int8 acc.", "agreement", "claimed values / query"], rows, "tab:models")

    # -- CNN costs: ours vs the sampling protocol, same model, same machine ---------------------
    rows = []
    for m in CNN_ORDER:
        c = f"defence_C_int_lam{lam}_rate4"
        k = f"defence_Kpre_int_lam{lam}_rate4"
        if M.get(m, c, "prove_forward") is None:
            continue
        per_path = M.get(m, "sampling", "path_bytes")
        shared40 = M.get(m, "sampling", "paths_bytes_shared", lam=40)
        rows.append([NAMES[m], t(M.total(m, c, PROVE)), t(M.total(m, c, VERIFY)), b(M.get(m, c, "bytes_total")),
                     t(M.total(m, k, VERIFY)), b(M.get(m, k, "bytes_total")),
                     b(per_path), f"{M.get(m, 'sampling', 'paths_needed_penultimate', lam=40):,.0f}",
                     b(shared40)])
    d.table(f"CNNs, one query (RTX 2080 Ti prover, Xeon 4114 verifier): ours at λ={lam} vs Anchuri et al.'s path test",
            ["model", "prove (C)", "verify (C)", "proof (C)", "verify (Kpre)", "proof (Kpre)",
             "[1]: 1 path", "[1]: paths for 2^-40", "[1]: 2^-40, shared"], rows, "tab:cnn")

    # -- soundness experiments -----------------------------------------------------------------
    attacks = ["single_value", "penultimate_neuron", "output_logit", "substituted_model_1pct",
               "substituted_model_M_tilde", "forged_fold", "forged_column"]
    rows = []
    for m in CNN_ORDER:
        if M.get(m, "sampling", "p_detect_penultimate") is None:
            continue
        cells = [NAMES[m], f"1/{1 / M.get(m, 'sampling', 'p_detect_penultimate'):,.0f}",
                 f"1/{1 / M.get(m, 'sampling', 'p_detect_min_node'):,.0f}"]
        for a in attacks:
            rate, n = M.attack_rate(m, "tamper_C_int_lam40", a)
            cells.append("–" if rate is None else f"{rate * n:.0f}/{n}")
        rows.append(cells)
    d.table("Detection. [1]: per-path probability of catching a single tampered neuron (exact); "
            "ours: attacks rejected / attempted (λ=40, committed weights)",
            ["model", "[1] attack neuron", "[1] worst neuron", "single value", "penult. neuron", "logit",
             "1% weights", "M~ model", "forged fold", "forged column"], rows, "tab:detect")

    # -- the attack itself ---------------------------------------------------------------------
    rows = []
    for m in CNN_ORDER:
        s = M.get(m, "attack_float", "flip_success", "mean")
        if s is None:
            continue
        rows.append([NAMES[m], f"{M.get(m, 'attack_float', 'penultimate_width'):.0f}", p(s),
                     p(M.get(m, "attack_float", "flip_success_live", "mean")),
                     f"{M.get(m, 'attack_float', 'flip_delta_over_unit_max_live'):.2g}×"
                     if M.get(m, "attack_float", "flip_delta_over_unit_max_live") else "–"])
    d.table("The single-neuron attack on the float model (one penultimate neuron set to a large value)",
            ["model", "penultimate width", "flips (any neuron)", "flips (live neurons)",
             "value / neuron's max (median)"], rows, "tab:attack")

    # -- LLMs ------------------------------------------------------------------------------------
    rows = []
    c_rows, k_rows = llm_rows("C"), llm_rows("Kpre")
    for (model, seq), dd in sorted(c_rows.items(), key=lambda kv: (kv[1]["prove_forward"][2], kv[0][1])):
        if "prove_forward" not in dd:
            continue
        pr = sum(dd[k][0] for k in PROVE if k in dd)
        vr = sum(dd[k][0] for k in VERIFY if k in dd)
        kk = k_rows.get((model, seq), {})
        kv = sum(kk[k][0] for k in VERIFY if k in kk) if kk else None
        prov = dd["prove_forward"][1]
        rows.append([LLM_LABEL.get(model, model), params(dd["prove_forward"][2]), str(seq), t(pr), t(vr),
                     b(_llm_bytes(dd)), t(kv), b(kk.get("bytes_total", (None,))[0]),
                     "measured" if prov == "measured" else "extrap."])
    d.table(f"Language models, one prompt (next-token logits), λ={lam}",
            ["model", "params", "tokens", "prove (C)", "verify (C)", "proof (C)", "verify (Kpre)", "proof (Kpre)",
             "source"], rows, "tab:llm")

    # -- published rows used for comparison ---------------------------------------------------
    rows = []
    for r in _read("reported_curated.csv"):
        rows.append([r["system"], r["model"], params(_f(r["params"])), r["seq"] or "–",
                     t(_f(r["prover_s"])), t(_f(r["verifier_s"])), b(_f(r["proof_bytes"])), r["hardware"],
                     r["source_location"][:40]])
    d.table("Published results used in the comparison (as reported; hardware differs per row)",
            ["system", "model", "params", "tokens", "prove", "verify", "proof", "hardware", "source"], rows,
            "tab:published")

    # -- error types -------------------------------------------------------------------------------
    rows = [[r["system"], r["setting"], r["error_type"], r["stated_error"], r["trust"]]
            for r in _read("systems.csv")]
    d.table("What each system's error bound means", ["system", "setting", "error type", "stated error", "assumes"],
            rows, "tab:errors")

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "report_tables.md").write_text("\n".join(d.md), encoding="utf-8")
    (TABLES / "report_tables.tex").write_text("\n".join(d.tex), encoding="utf-8")
    print(f"wrote {len([l for l in d.md if l.startswith('###')])} tables")


if __name__ == "__main__":
    main()
