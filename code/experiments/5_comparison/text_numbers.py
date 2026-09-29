"""The numbers ``report/main.tex`` quotes in running text, recomputed from one platform's tables.

    python experiments/5_comparison/text_numbers.py --platform l40s          # the report's numbers

Generated tables and figures follow the data by themselves; these sentences do not.  For
every quoted number this prints the line of main.tex it is on (found by a search on the
wording around it, not on the number), the value recomputed from the tables, and the text
as it stands, so a re-measurement (a new platform) is a mechanical edit.  The counts of
Section 4.3 (in the abstract and the conclusion too) and the record total of Section 4.1
are typed by hand and not repeated here: take them from ``count_outcomes.py --platform <p>``.
The ratio ranges use full-depth builds only, and the zkLLM ranges cover the zkLLM rows of
Table 4 (Figure 5's models).  zkLLM's own run on our GPU (Section 4.4) is not in the
tables: its times are in ``code/artifacts/results/zkllm_l40s/``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_assets as pa  # noqa: E402

MAIN = pa.REPORT / "main.tex"


def rng(vals, fmt=lambda v: f"{v:.0f}"):
    vals = [v for v in vals if v is not None]
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
    # 4.1: the 1-2 block extrapolation against every full build of the platform (on the totals, as in
    # Table 3), and the lean prover's peak GPU memory
    groups = {}
    for r in pa._read("llm_extrapolation_check.csv"):
        key = tuple(r[f] for f in ("model", "seq", "mode", "challenges", "lam", "threads", "variant"))
        groups.setdefault(key, {})[r["metric"]] = r

    def err(d, parts):
        full = sum(pa._f(d[p]["value"]) or 0 for p in parts if p in d)
        return 100 * (sum(pa._f(d[p]["extrapolated"]) or 0 for p in parts if p in d) / full - 1) if full else None

    # the headline selection only (the batched and 1-thread cells have 1-2 block builds too)
    groups = {k: d for k, d in groups.items() if (k[5], k[6]) == (pa.DEFAULT["threads"], pa.DEFAULT["variant"])}
    be = [err(d, ("bytes_total",)) for d in groups.values()]
    pe = {}
    for key, d in groups.items():
        e = err(d, pa.PROVE)
        if e is not None and abs(e) > pe.get(key[0], -1):
            pe[key[0]] = abs(e)
    ve = [err(d, pa.VERIFY) for d in groups.values()]
    add(r"this rule predicts the proof", "extrapolated vs full: max |error| of proof size (%)",
        f"{max(abs(v) for v in be if v is not None):.3f}")
    add(r"misses the times by up to", "extrapolated vs full: max |error| of verifier; of prover (model) (%)",
        f"{max(abs(v) for v in ve if v is not None):.0f}; {max(pe.values()):.0f} ({max(pe, key=pe.get)})")
    cell = f"defence_C_int_lam{pa.LAM}_T2048_L32"   # the full 7B builds
    summary = pa._read("measured_summary.csv")

    def llm(model, metric):
        return next((pa._f(r["median"]) for r in summary if (r["suite"], r["model"], r["cell"], r["metric"],
                     r.get("batch", "")) == ("llm", model, cell, metric, "")), None)

    mem = llm("llama2-7b", "gpu_peak_memory")
    add(r"GiB for our prover", "Llama-2-7B T2048: our peak GPU memory", f"{mem / 2 ** 30:.1f} GiB")
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
    add(r"ms per query, the verifier", "CNN prover (C)", rng([M.total(m, cnn_c, pa.PROVE) for m in pa.CNN_ORDER], pa.t))
    add(r"ms per query, the verifier", "CNN verifier (C)", rng([M.total(m, cnn_c, pa.VERIFY) for m in pa.CNN_ORDER], pa.t))
    add(r"the known-weights prover, which only runs", "C prover / Kpre prover",
        rng([M.total(m, cnn_c, pa.PROVE) / M.total(m, cnn_k, pa.PROVE) for m in pa.CNN_ORDER], x))
    add(r"2\.4--4\.4\$\\times\$ smaller", "int8 model / proof (VGGs, CIFAR ResNet)",
        rng([M.get(m, "facts", "model_bytes_int8") / M.get(m, cnn_c, "bytes_total")
             for m in ("vgg11", "vgg16", "resnet18_cifar")], lambda v: f"{v:.1f}"))
    # 4.4 language models
    for (model, seq), anchor in (((("gpt2", 64)), r"64-token prompt is proved in"), ((("llama2-7b", 64)), r"^Llama-2-7B takes"),
                                 ((("llama2-7b", 2048)), r"tokens Llama-2-7B takes")):
        p, v, by, n = pa._llm_cost(c[(model, seq)])
        add(anchor, f"{model} T{seq}: prove / verify / proof", f"{pa.t(p)} / {pa.t(v)} / {pa.b(by)}")
    add(r"about 9\$\\times\$ smaller than\s*$", "Llama-2-7B T64: int8 weights / proof",
        x(pa._llm_cost(c[("llama2-7b", 64)])[3] / pa._llm_cost(c[("llama2-7b", 64)])[2]))
    if ("llama2-7b", 4096) in c:
        p, v, by, n = pa._llm_cost(c[("llama2-7b", 4096)])
        add(r"at 4\{,\}096 tokens:", "llama2-7b T4096: prove / verify / proof", f"{pa.t(p)} / {pa.t(v)} / {pa.b(by)}")
    if ("llama2-70b", 64) in c:
        _, _, by, n = pa._llm_cost(c[("llama2-70b", 64)])
        add(r"the 1- and 2-block builds give a", "llama2-70b T64 (1-2 blocks): C proof; int8 weights / proof",
            f"{pa.b(by)}; {x(n / by)}")
    add(r"21\.8\\,MB for GPT-2", "Kpre proof GPT-2 T64 / Llama-2-7B T64",
        f"{pa.b(pa._llm_cost(k[('gpt2', 64)])[2])} / {pa.b(pa._llm_cost(k[('llama2-7b', 64)])[2])}")
    ratio = {key: pa._llm_cost(c[key])[0] / pa._llm_cost(k[key])[0] for key in list(c) if key in k
             and pa._measured(c[key]) and pa._measured(k[key])}
    add(r"the known-weights one at 2\{,\}048", "C prover / Kpre prover at 2048 tokens",
        rng([v for (m, s), v in ratio.items() if s == 2048], x))
    add(r"\\times\$ for billion-parameter", "C prover / Kpre prover, >=1B params, <=64 tokens",
        rng([v for (m, s), v in ratio.items() if s <= 64 and (pa._llm_cost(c[(m, s)])[3] or 0) >= 1e9], x))
    # against zkSNARKs (the same selection as Table 4, whose zkLLM rows are Figure 5's)
    def ours(model, seq):
        if seq is None:
            return M.total(model, cnn_c, pa.PROVE), M.total(model, cnn_c, pa.VERIFY), M.get(model, cnn_c, "bytes_total")
        return pa._llm_cost(c[(model, seq)])[:3]

    fac = {}
    for system, pub, model, seq, mode in pa.MATCHES:
        if mode != "C":
            continue
        r = rep.get((system, pub, str(seq) if seq else "")) or next(v for kk, v in rep.items() if kk[:2] == (system, pub))
        fac[(system, pub, seq)] = [(pa._f(r[f]) or 0) / o if o else None
                                   for f, o in zip(("prover_s", "verifier_s", "proof_bytes"), ours(model, seq))]
    cnn = [v[0] for (s, p, q), v in fac.items() if q is None]
    llm = [v[0] for (s, p, q), v in fac.items() if q is not None]
    add(r"\(not stated\)\. Our prover is", "prover speed-up, CNN systems", rng(cnn, x))
    add(r"\\times\$ faster on\s*$", "prover speed-up, LLM systems", rng(llm, x))
    zk = [pa._f(rep[("zkLLM", n, "2048")]["prover_s"]) / pa._llm_cost(c[(m, 2048)])[0] for m, n in pa.ZKLLM]
    add(r"A100, and our prover is", "prover speed-up vs zkLLM at 2048", rng(zk, x))
    zk13 = [pa._f(rep[("zkLLM", "Llama-2-13B", "2048")]["prover_s"]) / pa._llm_cost(r[("llama2-13b", 2048)])[0]
            for r in (c, k) if pa._measured(r.get(("llama2-13b", 2048)))]
    add(r"so there we use the paper's", "prover speed-up vs zkLLM's paper, Llama-2-13B T2048 (C, Kpre)", rng(zk13, x))
    add(r"15\.0\\,s \(Kpre\)", "llama2-7b T2048 prover (C / Kpre), against zkLLM on our GPU",
        " / ".join(pa.t(pa._llm_cost(r[("llama2-7b", 2048)])[0]) for r in (c, k)))
    add(r"\\times\$ at 64 tokens to", "DeepProve GPT-2 speed-up at 64 / 512",
        f"{x(fac[('DeepProve', 'GPT-2', 64)][0])} / {x(fac[('DeepProve', 'GPT-2', 512)][0])}")
    zkv = [pa._llm_cost(c[(m, 2048)])[1] / pa._f(rep[("zkLLM", n, "2048")]["verifier_s"]) for m, n in pa.ZKLLM]
    add(r"slower than zkLLM at 2\{,\}048 tokens", "verifier slow-down vs zkLLM at 2048", rng(zkv, x))
    # the verifier on the GPU (Section 4.4), where it was run
    g = {mode: pa.llm_rows(mode, variant="_gpuv") for mode in ("C", "Kpre")}
    cpu = {"C": c, "Kpre": k}
    gv = {(mode, key): (pa._llm_cost(cpu[mode][key])[1], pa._llm_cost(rows[key])[1])
          for mode, rows in g.items() for key in rows if key in cpu[mode] and "verify_products" in rows[key]}
    zk_name = dict(pa.ZKLLM)
    add(r"where we also\s*$", "GPU verifier / zkLLM's at 2048", rng(
        [gpu / pa._f(rep[("zkLLM", zk_name[m], "2048")]["verifier_s"]) for (_, (m, s)), (_, gpu) in gv.items()
         if s == 2048 and m in zk_name], x))
    add(r"^21--41", "CPU / GPU verifier at 2048 tokens", rng([a / b for (_, (m, s)), (a, b) in gv.items() if s == 2048], x))
    if ("C", ("llama2-7b", 2048)) in gv:
        add(r"^21--41", "llama2-7b T2048 verifier, GPU (C / Kpre) and CPU (C / Kpre)",
            " / ".join(pa.t(gv[(mode, ("llama2-7b", 2048))][1]) for mode in ("C", "Kpre")) + "; " +
            " / ".join(pa.t(gv[(mode, ("llama2-7b", 2048))][0]) for mode in ("C", "Kpre")))
    add(r"At 64 tokens or fewer it gains", "CPU / GPU verifier at <= 64 tokens",
        rng([a / b for (_, (m, s)), (a, b) in gv.items() if s <= 64], lambda v: f"{v:.1f}"))
    # batching (Section 4.4): Llama-2-7B, 64 tokens, committed weights, per prompt
    bat = {}
    for r in pa._read("llm_full_model.csv"):
        if (r["model"], r["seq"], r["mode"], r["challenges"], r["variant"], r["threads"], r.get("batch")) == \
                ("llama2-7b", "64", "C", "int", "_batch", pa.DEFAULT["threads"], "8") \
                and pa._f(r["lam"]) == pa.LAM:
            bat[r["metric"]] = (float(r["value"]), r["provenance"])
    if bat:
        add(r"With 8 prompts of 64 tokens", "llama2-7b T64 C batch of 8: prover / proof per prompt",
            f"{pa.t(sum(bat[p][0] for p in pa.PROVE if p in bat) / 8)} / {pa.b(bat['bytes_total'][0] / 8)}")
    small = [ours("lenet5", None)[1] / 0.0058, ours("vgg16", None)[1] / 0.0593,
             pa._llm_cost(c[("gpt2", 64)])[1] / pa._f(rep[("zkGPT", "GPT-2", "")]["verifier_s"]),
             pa._llm_cost(c[("gpt2", 64)])[1] / pa._f(rep[("zkLLM (re-run by zkGPT)", "GPT-2", "")]["verifier_s"])]
    add(r"about as fast as the sum-check", "verifier slow-down vs zkCNN/zkGPT/zkLLM(GPT-2)", rng(small, x))
    add(r"slower than ZKML, and", "verifier slow-down vs ZKML VGG-16", x(1 / fac[("ZKML", "VGG-16 CIFAR-10", None)][1]))
    dp = {s: pa._llm_cost(c[("gpt2", s)])[1] / pa._f(rep[("DeepProve", "GPT-2", str(s))]["verifier_s"])
          for s in (64, 128, 256, 512)}
    add(r"faster than DeepProve up to", "our verifier / DeepProve's at 64,128,256,512 (<1: ours faster)",
        ", ".join(f"{v:.2f}" for v in dp.values()))
    zkp = [pa._llm_cost(c[(m, 2048)])[2] / pa._f(rep[("zkLLM", n, "2048")]["proof_bytes"]) for m, n in pa.ZKLLM]
    add(r"000\$\\times\$ zkLLM's", "largest proof ratio vs zkLLM at 2048", f"{max(zkp):,.0f}")
    # Maverick (1 client thread, lambda = 40, Kpre)
    k1 = pa.llm_rows("Kpre", lam=40, threads="1", variant="_thr1")[("qwen3-4b", 8)]
    p1, v1, b1, _ = pa._llm_cost(k1)
    mav = rep[("Maverick (verif.-only, 1 thr.)", "Qwen3-4B", "8")]
    add(r"exactly the size of Maverick", "our Kpre proof, Qwen3-4B T8", pa.b(b1))
    add(r"faster than our Python verifier", "our verifier / Maverick's (1 thread); ours in s",
        f"{x(v1 / pa._f(mav['verifier_s']))}x; {v1:.2f} s")
    add(r"faster than Maverick's\.", "Maverick's prover / ours", x(pa._f(mav["prover_s"]) / p1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=pa.PLATFORM,
                    help="read tables_<platform>/ (default: $PVI_PLATFORM, else l40s, the report's numbers)")
    pa.TABLES = pa.tables_dir(ap.parse_args().platform)
    gaps = pa.missing(pa.Measured())
    if gaps:   # the report's text needs every number Figures 3-5 and Tables 2-4 draw
        raise SystemExit(f"{pa.TABLES.name}: {len(gaps)} missing\n  " + "\n  ".join(gaps))
    lines = MAIN.read_text(encoding="utf-8").splitlines()
    print(f"numbers from {pa.TABLES.name}/ against {MAIN.name}")
    for anchor, what, value in numbers():
        hit = next(((i + 1, l.strip()) for i, l in enumerate(lines) if re.search(anchor, l)), (None, "(anchor not found)"))
        print(f"l.{hit[0]}: {what}\n    computed: {value}\n    text:     {hit[1][:110]}")


if __name__ == "__main__":
    main()
