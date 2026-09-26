"""All result tables (Markdown and LaTeX), generated from ``artifacts/comparison/tables``.

    python experiments/5_comparison/report_tables.py
    # -> artifacts/comparison/tables/report_tables.md  and  report_tables.tex

A superset of the report's tables, including the like-for-like ratio table
(``MATCHES`` below, also used by paper_assets.py) and every published row compared.
"""

from __future__ import annotations

import csv
import math

from common import (CNN_ORDER, DEFAULT, LLM_LABEL, PROVE, VERIFY, Measured, llm_rows, _llm_bytes, _llm_cost,
                    _read, _f)

from common import TABLES
NAMES = {"mlp_mnist": "MLP (MNIST)", "lenet5": "LeNet-5 (MNIST)", "vgg11": "VGG-11 (CIFAR-10)",
         "vgg16": "VGG-16 (CIFAR-10)", "resnet18_cifar": "ResNet-18 (CIFAR-10)",
         "resnet18_224": "ResNet-18 (224px, dogs vs cats)"}


# (published system, published model, our model, our prompt length or None for CNNs, our mode, note)
MATCHES = [
    ("zkCNN", "LeNet-5 MNIST", "lenet5", None, "C", "same model"),
    ("vCNN (re-run by zkCNN)", "LeNet-5 MNIST", "lenet5", None, "C", "same model"),
    ("Bionetta", "LeNet-5", "lenet5", None, "C", "same architecture"),
    ("EZKL (run by Bionetta)", "LeNet-5", "lenet5", None, "C", "same architecture"),
    ("zkCNN", "VGG-11 CIFAR-10", "vgg11", None, "C", "same model"),
    ("zkCNN", "VGG-16 CIFAR-10", "vgg16", None, "C", "same model"),
    ("ZKML", "VGG-16 CIFAR-10", "vgg16", None, "C", "same model"),
    ("zkPyTorch", "VGG-16 CIFAR-10", "vgg16", None, "C", "same model; prover time only"),
    ("ZENO", "VGG-16 CIFAR-10 (19.9M-FLOP variant)", "vgg16", None, "C", "their variant is smaller"),
    ("ZKML", "ResNet-18 CIFAR-10 (ZKML's 281K-param variant)", "resnet18_cifar", None, "C",
     "their variant has 281K params, ours 11.2M"),
    ("Bionetta", "ResNet-18 (Bionetta's version)", "resnet18_cifar", None, "C", "their own ResNet-18 variant"),
    ("zkGPT", "GPT-2", "gpt2", 64, "C", "their prompt length not in the table"),
    ("zkLLM (re-run by zkGPT)", "GPT-2", "gpt2", 64, "C", "their prompt length not in the table"),
    ("DeepProve", "GPT-2", "gpt2", 64, "C", "same prompt length"),
    ("DeepProve", "GPT-2", "gpt2", 128, "C", "same prompt length"),
    ("DeepProve", "GPT-2", "gpt2", 256, "C", "same prompt length"),
    ("DeepProve", "GPT-2", "gpt2", 512, "C", "same prompt length"),
    ("zkLLM", "OPT-125M", "opt-125m", 2048, "C", "same prompt length"),
    ("zkLLM", "OPT-1.3B", "opt-1.3b", 2048, "C", "same prompt length"),
    ("zkLLM", "OPT-6.7B", "opt-6.7b", 2048, "C", "same prompt length"),
    ("zkLLM", "Llama-2-7B", "llama2-7b", 2048, "C", "same prompt length"),
    ("zkLLM (re-run by Anchuri)", "Llama-2-7B", "llama2-7b", 2048, "C", "their prompt length not stated"),
    ("ZKTorch", "Llama-2-7B (1 token)", "llama2-7b", 64, "C", "they prove 1 token, we prove 64"),
    ("Maverick (verif.-only, 1 thr.)", "Qwen3-4B", "qwen3-4b", 8, "Kpre", "λ=40 and 1 client thread, as Maverick"),
]


def ratio(a, b, up, down):
    if not a or not b:
        return "–"
    r = a / b
    if 0.99 < r < 1.01:
        return "equal"
    x = r if r >= 1 else 1 / r
    return (f"{x:,.0f}×" if x >= 1000 else f"{x:.3g}×") + " " + (up if r >= 1 else down)


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

    # -- CNN costs: ours vs the sampling protocol and vs downloading the model ------------
    rows = []
    for m in CNN_ORDER:
        c = f"defence_C_int_lam{lam}_rate4"
        k = f"defence_Kpre_int_lam{lam}_rate4"
        if M.get(m, c, "prove_forward") is None:
            continue
        rows.append([NAMES[m], t(M.total(m, c, PROVE)), t(M.total(m, c, VERIFY)), b(M.get(m, c, "bytes_total")),
                     b(M.get(m, c, "bytes_claims_zlib") and M.get(m, c, "bytes_total") - M.get(m, c, "bytes_claims")
                       + M.get(m, c, "bytes_claims_zlib")),
                     t(M.total(m, k, VERIFY)), b(M.get(m, k, "bytes_total")),
                     b(M.get(m, "facts", "model_bytes_int8")), t(M.get(m, "facts", "cpu_float_inference"))])
    d.table(f"CNNs, one query, λ={lam} (prover RTX 2080 Ti, verifier Xeon 4114 x8): ours vs downloading the model",
            ["model", "prove (C)", "verify (C)", "proof (C)", "proof (C), zlib", "verify (Kpre)", "proof (Kpre)",
             "int8 model", "CPU re-run"], rows, "tab:cnn")

    rows = []
    for m in CNN_ORDER:
        if M.get(m, "sampling", "path_bytes") is None:
            continue
        rows.append([NAMES[m], b(M.get(m, "sampling", "path_bytes")), t(M.get(m, "sampling", "path_verify")),
                     f"1/{1 / M.get(m, 'sampling', 'p_detect_penultimate'):,.0f}",
                     f"{M.get(m, 'sampling', 'paths_needed_penultimate', lam=40):,.0f}",
                     b(M.get(m, "sampling", "paths_bytes_shared", lam=40)),
                     b(M.get(m, "sampling", "open_all_bytes")),
                     b(M.get(m, f"defence_C_int_lam40_rate4", "bytes_total"))])
    d.table("Anchuri et al.'s path test on the same models: cost of reaching 2^-40 against the single-neuron attack",
            ["model", "1 path", "verify 1 path", "detect / path", "paths for 2^-40", "k paths, shared",
             "open everything", "ours at λ=40 (C)"], rows, "tab:sampling")

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

    # -- like-for-like ratios -----------------------------------------------------------------------
    curated = _read("reported_curated.csv")
    c_llm = llm_rows("C")
    k_thr1 = llm_rows("Kpre", lam=40, threads="1", variant="_thr1")
    rows = []
    for system, model, ours, seq, mode, note in MATCHES:
        # the published row with the same prompt length if there is one, else the only row for that model
        cand = [r for r in curated if (r["system"], r["model"]) == (system, model)]
        same = [r for r in cand if seq and r["seq"] == str(seq)]
        r = same[0] if same else (cand[0] if len(cand) == 1 else None)
        if r is None:
            continue
        if seq is None:
            cell = f"defence_{mode}_int_lam{lam}_rate4"
            us = (M.total(ours, cell, PROVE), M.total(ours, cell, VERIFY), M.get(ours, cell, "bytes_total"))
            label = NAMES[ours]
        else:
            dd = (c_llm if mode == "C" else k_thr1).get((ours, seq))
            if not dd:
                continue
            us = _llm_cost(dd)[:3]
            label = f"{LLM_LABEL[ours]}, {seq} tok."
        them = (_f(r["prover_s"]), _f(r["verifier_s"]), _f(r["proof_bytes"]))
        rows.append([f"{system}: {model}", f"{label} ({mode})", ratio(them[0], us[0], "faster", "slower"),
                     ratio(them[1], us[1], "faster", "slower"), ratio(us[2], them[2], "larger", "smaller"),
                     r["hardware"], note])
    d.table("Like-for-like ratios against published systems (ours at λ=128 unless noted; their hardware differs)",
            ["published", "ours", "prover", "verifier", "proof", "their hardware", "note"], rows, "tab:ratios")

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
