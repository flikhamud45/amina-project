"""The numbers ``report/main.tex`` quotes in running text, recomputed from one platform's tables.

    python experiments/5_comparison/text_numbers.py                  # the RTX 2080 Ti (tables/)
    python experiments/5_comparison/text_numbers.py --platform h100  # tables_h100/

Generated tables and figures follow the data by themselves; these sentences do not.  For
every quoted number this prints the line of main.tex it is on, the value recomputed from
the tables, and the text as it stands, so a platform switch is a mechanical edit.  The
counts of Section 4.3 are macros (count_outcomes.py --tex) and are not repeated here.
On any platform other than the report's own, the ratio ranges use measured (full-depth)
rows only, and the zkLLM ranges cover every zkLLM model that platform measured (Figure 5's
list; Table 4's for the proof sizes).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_assets as pa  # noqa: E402

MAIN = pa.REPORT / "main.tex"


def rng(vals, fmt=lambda v: f"{v:.0f}"):
    vals = [v for v in vals if v is not None]
    if not vals:   # e.g. a platform without full-depth builds for this sentence
        return "(no measured rows)"
    return f"{fmt(min(vals))}--{fmt(max(vals))}"


def x(v):
    """A ratio as the text prints it: 1.4, 16, 7,024."""
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def numbers():
    M = pa.Measured()
    c, k = pa.llm_rows("C"), pa.llm_rows("Kpre")
    cur = pa._read("reported_curated.csv")
    rep = {(r["system"], r["model"], r["seq"]): r for r in cur}
    cnn_c, cnn_k = f"defence_C_int_lam{pa.LAM}_rate4", f"defence_Kpre_int_lam{pa.LAM}_rate4"
    out = []

    def add(anchor, what, value):
        out.append((anchor, what, value))

    # 4.1 and 4.2: hardware-independent (a check that the new platform ran the same models)
    gap = [(M.get(m, "facts", "float_accuracy") or 0) - (M.get(m, "facts", "int8_accuracy") or 0) for m in pa.CNN_ORDER]
    add(r"within 0\.4 points", "max |float - int8| accuracy (points)", f"{100 * max(map(abs, gap)):.1f}")
    pdet = [M.get(m, "sampling", "p_detect_penultimate") for m in pa.CNN_ORDER]
    add(r"probability 1/84", "one path, attacked neuron: 1/p range", rng([1 / p for p in pdet]))
    add(r"1/28 million", "least-visited neuron: 1/p (millions)",
        f"{1 / min(M.get(m, 'sampling', 'p_detect_min_node') for m in pa.CNN_ORDER) / 1e6:.0f}")
    add(r"2\{,\}316--14\{,\}182 paths", "paths for 2^-40",
        rng([M.get(m, "sampling", "paths_needed_penultimate", lam=40) for m in pa.CNN_ORDER], lambda v: f"{v:,.0f}"))
    # 4.3 bits against bytes
    for lam, anchor in ((40, r"48--58 bits"), (128, r"150--153 bits")):
        cell = f"defence_C_int_lam{lam}_rate4"
        add(anchor, f"CNN security bits at lambda={lam}", rng([M.get(m, cell, "soundness_bits") for m in pa.CNN_ORDER],
                                                              lambda v: f"{v:.0f}"))
        add(anchor, f"CNN bytes at lambda={lam}", rng([M.get(m, cell, "bytes_total") for m in pa.CNN_ORDER], pa.b))
    # 4.4 image models
    add(r"11--105\\,ms per query", "CNN prover (C)", rng([M.total(m, cnn_c, pa.PROVE) for m in pa.CNN_ORDER], pa.t))
    add(r"the verifier 21--370\\,ms", "CNN verifier (C)", rng([M.total(m, cnn_c, pa.VERIFY) for m in pa.CNN_ORDER], pa.t))
    add(r"4--8\$\\times\$ the known-weights prover", "C prover / Kpre prover",
        rng([M.total(m, cnn_c, pa.PROVE) / M.total(m, cnn_k, pa.PROVE) for m in pa.CNN_ORDER], x))
    add(r"2\.4--4\.4\$\\times\$ smaller", "int8 model / proof (VGGs, CIFAR ResNet)",
        rng([M.get(m, "facts", "model_bytes_int8") / M.get(m, cnn_c, "bytes_total")
             for m in ("vgg11", "vgg16", "resnet18_cifar")], lambda v: f"{v:.1f}"))
    # 4.4 language models
    for (model, seq), anchor in (((("gpt2", 64)), r"proved in 0\.44"), ((("llama2-7b", 64)), r"Llama-2-7B takes 11\.5"),
                                 ((("llama2-7b", 2048)), r"takes 32\\,s to prove")):
        p, v, by, n = pa._llm_cost(c[(model, seq)])
        add(anchor, f"{model} T{seq}: prove / verify / proof", f"{pa.t(p)} / {pa.t(v)} / {pa.b(by)}")
    add(r"about 9\$\\times\$ smaller than\s*$", "Llama-2-7B T64: int8 weights / proof",
        x(pa._llm_cost(c[("llama2-7b", 64)])[3] / pa._llm_cost(c[("llama2-7b", 64)])[2]))
    add(r"21\.8\\,MB for GPT-2", "Kpre proof GPT-2 T64 / Llama-2-7B T64",
        f"{pa.b(pa._llm_cost(k[('gpt2', 64)])[2])} / {pa.b(pa._llm_cost(k[('llama2-7b', 64)])[2])}")
    frozen = pa.is_frozen()   # the submitted text's ranges include extrapolated rows
    ratio = {key: pa._llm_cost(c[key])[0] / pa._llm_cost(k[key])[0] for key in list(c) if key in k
             and (frozen or (pa._measured(c[key]) and pa._measured(k[key])))}
    add(r"1\.2--1\.9\$\\times\$", "C prover / Kpre prover at 2048 tokens",
        rng([v for (m, s), v in ratio.items() if s == 2048], x))
    add(r"10--30\$\\times\$ for billion", "C prover / Kpre prover, >=1B params, <=64 tokens",
        rng([v for (m, s), v in ratio.items() if s <= 64 and (pa._llm_cost(c[(m, s)])[3] or 0) >= 1e9], x))
    # against zkSNARKs (the same selection as Table 4, plus the zkLLM rows of Figure 5)
    def ours(model, seq):
        if seq is None:
            return M.total(model, cnn_c, pa.PROVE), M.total(model, cnn_c, pa.VERIFY), M.get(model, cnn_c, "bytes_total")
        return pa._llm_cost(c[(model, seq)])[:3]

    fac = {}
    table4 = pa.matches(c)
    for system, pub, model, seq, mode in table4:
        if mode != "C":
            continue
        r = rep.get((system, pub, str(seq) if seq else "")) or next(v for kk, v in rep.items() if kk[:2] == (system, pub))
        fac[(system, pub, seq)] = [(pa._f(r[f]) or 0) / o if o else None
                                   for f, o in zip(("prover_s", "verifier_s", "proof_bytes"), ours(model, seq))]
    cnn = [v[0] for (s, p, q), v in fac.items() if q is None]
    llm = [v[0] for (s, p, q), v in fac.items() if q is not None]
    add(r"Our prover is 27--7\{,\}024", "prover speed-up, CNN systems", rng(cnn, x))
    add(r"11--277\$\\times\$ faster on", "prover speed-up, LLM systems", rng(llm, x))
    fig5 = pa.zkllm_models([pa.TABLES])          # Figure 5's models (the stored tables: four)
    zk = [pa._f(rep[("zkLLM", n, "2048")]["prover_s"]) / pa._llm_cost(c[(m, 2048)])[0] for m, n in fig5]
    add(r"still 11--49\$\\times\$ faster", "prover speed-up vs zkLLM at 2048 (also Contributions, l.157)", rng(zk, x))
    add(r"from 78\$\\times\$ at 64 tokens", "DeepProve GPT-2 speed-up at 64 / 512",
        f"{x(fac[('DeepProve', 'GPT-2', 64)][0])} / {x(fac[('DeepProve', 'GPT-2', 512)][0])}")
    zkv = [pa._llm_cost(c[(m, 2048)])[1] / pa._f(rep[("zkLLM", n, "2048")]["verifier_s"]) for m, n in fig5]
    add(r"65--141\$\\times\$ slower than zkLLM", "verifier slow-down vs zkLLM at 2048", rng(zkv, x))
    small = [ours("lenet5", None)[1] / 0.0058, ours("vgg16", None)[1] / 0.0593,
             pa._llm_cost(c[("gpt2", 64)])[1] / pa._f(rep[("zkGPT", "GPT-2", "")]["verifier_s"]),
             pa._llm_cost(c[("gpt2", 64)])[1] / pa._f(rep[("zkLLM (re-run by zkGPT)", "GPT-2", "")]["verifier_s"])]
    add(r"1\.8--5\$\\times\$ slower than the sum-check", "verifier slow-down vs zkCNN/zkGPT/zkLLM(GPT-2)", rng(small, x))
    add(r"20\$\\times\$ slower than ZKML", "verifier slow-down vs ZKML VGG-16", x(1 / fac[("ZKML", "VGG-16 CIFAR-10", None)][1]))
    dp = {s: pa._llm_cost(c[("gpt2", s)])[1] / pa._f(rep[("DeepProve", "GPT-2", str(s))]["verifier_s"])
          for s in (64, 128, 256, 512)}
    add(r"faster than DeepProve up to 256", "our verifier / DeepProve's at 64,128,256,512 (<1: ours faster)",
        ", ".join(f"{v:.2f}" for v in dp.values()))
    zkp = [pa._llm_cost(c[(m, 2048)])[2] / pa._f(rep[("zkLLM", n, "2048")]["proof_bytes"])
           for system, n, m, _, _ in table4 if system == "zkLLM"]   # Table 4's zkLLM rows
    add(r"64\{,\}000\$\\times\$ zkLLM", "largest proof ratio vs zkLLM at 2048", f"{max(zkp):,.0f}")
    # Maverick (1 client thread, lambda = 40, Kpre)
    k1 = pa.llm_rows("Kpre", lam=40, threads="1", variant="_thr1" + pa.DEFAULT["variant"])[("qwen3-4b", 8)]
    p1, v1, b1, _ = pa._llm_cost(k1)
    mav = rep[("Maverick (verif.-only, 1 thr.)", "Qwen3-4B", "8")]
    add(r"exactly the size of Maverick", "our Kpre proof, Qwen3-4B T8", pa.b(b1))
    add(r"verifies it 16\$\\times\$ faster", "our verifier / Maverick's (1 thread); ours in s",
        f"{x(v1 / pa._f(mav['verifier_s']))}x; {v1:.2f} s")
    add(r"its prover is 1\.4\$\\times\$ faster", "our prover / Maverick's", x(p1 / pa._f(mav["prover_s"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM", ""))
    pa.TABLES = pa.tables_dir(ap.parse_args().platform)
    lines = MAIN.read_text(encoding="utf-8").splitlines()
    print(f"numbers from {pa.TABLES.name}/ against {MAIN.name}")
    for anchor, what, value in numbers():
        hit = next(((i + 1, l.strip()) for i, l in enumerate(lines) if re.search(anchor, l)), (None, "(anchor not found)"))
        print(f"l.{hit[0]}: {what}\n    computed: {value}\n    text:     {hit[1][:110]}")


if __name__ == "__main__":
    main()
