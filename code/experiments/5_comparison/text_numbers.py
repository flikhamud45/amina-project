"""The numbers ``report/main.tex`` quotes in running text, recomputed from the two runs' tables.

    python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved   # the report
    python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved --tex <draft.tex>

Generated tables and figures follow the data by themselves; these sentences do not.  For
every quoted number this prints the line of main.tex it is on (found by a search on the
wording around it, not on the number) and the next one, the value recomputed from the tables,
and the text as it stands, so a re-measurement is a mechanical edit; an anchor that is not found
is reported (exit status 1).  ``--platform`` is the basic protocol (Sections 3.4, 5.2 and 5.3),
``--optimised`` the optimised protocol on the same machines (Sections 5.2, 5.4 and 5.5).  The
counts of Sections 5.2 and 5.4 (abstract and conclusion too) are macros written by
``count_outcomes.py --tex``, and are not repeated here.  The ratio ranges use full-depth builds
only.  zkLLM's own run on our GPU is read from ``artifacts/results/zkllm_l40s/summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
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


def numbers(M):
    """The basic protocol's numbers (tables_<platform>/)."""
    c, k = pa.llm_rows("C"), pa.llm_rows("Kpre")
    cnn_c, cnn_k = f"defence_C_int_lam{pa.LAM}_rate4", f"defence_Kpre_int_lam{pa.LAM}_rate4"
    out = []

    def add(anchor, what, value):
        out.append((anchor, what, value))

    # 5.1: hardware-independent (a check that a new platform ran the same models)
    gap = [(M.get(m, "facts", "float_accuracy") or 0) - (M.get(m, "facts", "int8_accuracy") or 0) for m in pa.CNN_ORDER]
    add(r"changed accuracy by at most", "max |float - int8| accuracy (points)", f"{100 * max(map(abs, gap)):.1f}")
    # 5.1: the 1-2 block extrapolation of proof sizes against every full build of the platform
    groups = {}
    for r in pa._read("llm_extrapolation_check.csv"):
        key = tuple(r[f] for f in ("model", "seq", "mode", "challenges", "lam", "threads", "variant"))
        groups.setdefault(key, {})[r["metric"]] = r
    groups = {kk: d for kk, d in groups.items() if (kk[5], kk[6]) == (pa.DEFAULT["threads"], pa.DEFAULT["variant"])}
    be = [100 * (pa._f(d["bytes_total"]["extrapolated"]) / pa._f(d["bytes_total"]["value"]) - 1)
          for d in groups.values() if "bytes_total" in d]
    add(r"we extrapolate proof sizes", "extrapolated vs full: max |error| of proof size (%)",
        f"{max(abs(v) for v in be):.3f}")
    # 3.4: the path protocol on the image models
    pdet = [M.get(m, "sampling", "p_detect_penultimate") for m in pa.CNN_ORDER]
    add(r"One path catches it with probability", "one path, attacked neuron: 1/p (LeNet-5; range)",
        f"{1 / M.get('lenet5', 'sampling', 'p_detect_penultimate'):.0f}; {rng([1 / p for p in pdet])}")
    add(r"1/28 million", "least-visited neuron: 1/p (millions)",
        f"{1 / min(M.get(m, 'sampling', 'p_detect_min_node') for m in pa.CNN_ORDER) / 1e6:.0f}")
    add(r"takes 2\{,\}316--14\{,\}182 paths|\$2\^\{-40\}\$ takes", "paths for 2^-40",
        rng([M.get(m, "sampling", "paths_needed_penultimate", lam=40) for m in pa.CNN_ORDER], lambda v: f"{v:,.0f}"))
    an = next(r for r in pa._read("analytic.csv", pa.BASE / "tables") if r["model"] == "llama2-7b")
    add(r"needs about 305|takes paths that open about", "Llama-2-7B T64: 1/N; paths for 2^-40; bytes opened",
        f"1/{int(an['anchuri_width']):,}; {int(an['anchuri_paths']):,}; {pa.b(float(an['anchuri_open_all_bytes']))}")
    # 5.2: bits against bytes (Figure 3)
    for lam, anchor in ((40, r"rounded up \("), (128, r"rounded up \(")):
        cell = f"defence_C_int_lam{lam}_rate4"
        add(anchor, f"basic: CNN security bits at lambda={lam}",
            rng([M.get(m, cell, "soundness_bits") for m in pa.CNN_ORDER]))
    add(r"it sends .*fewer bytes than the paths|at 40 bits its proof is", "paths' bytes for 2^-40 / basic proof at lambda=40",
        rng([M.get(m, "sampling", "paths_bytes_shared", lam=40) / M.get(m, "defence_C_int_lam40_rate4", "bytes_total")
             for m in pa.CNN_ORDER], lambda v: f"{v:.1f}"))
    # 5.3 image models
    add(r"With committed weights \(Table", "CNN prover / verifier / proof (C)",
        " / ".join(rng([f(m) for m in pa.CNN_ORDER], fmt) for f, fmt in (
            (lambda m: M.total(m, cnn_c, pa.PROVE), pa.t), (lambda m: M.total(m, cnn_c, pa.VERIFY), pa.t),
            (lambda m: M.get(m, cnn_c, "bytes_total"), pa.b))))
    add(r"2\.4--4\.4\$\\times\$ smaller", "int8 model / proof (VGGs, CIFAR ResNet; 224px ResNet)",
        rng([M.get(m, "facts", "model_bytes_int8") / M.get(m, cnn_c, "bytes_total")
             for m in ("vgg11", "vgg16", "resnet18_cifar")], lambda v: f"{v:.1f}") +
        f"; {M.get('resnet18_224', 'facts', 'model_bytes_int8') / M.get('resnet18_224', cnn_c, 'bytes_total'):.2f}")
    # 5.3 language models
    for (model, seq), anchor in ((("gpt2", 64), r"64-token prompt is proved in"), (("llama2-7b", 64), r"with a 62\.1"),
                                 (("llama2-7b", 2048), r"tokens Llama-2-7B is still proved")):
        p, v, by, n = pa._llm_cost(c[(model, seq)])
        add(anchor, f"{model} T{seq}: prove / verify / proof", f"{pa.t(p)} / {pa.t(v)} / {pa.b(by)}")
    add(r"smaller than its int8 weights\. At", "Llama-2-7B T64: int8 weights / proof",
        x(pa._llm_cost(c[("llama2-7b", 64)])[3] / pa._llm_cost(c[("llama2-7b", 64)])[2]))
    ratio = {key: pa._llm_cost(c[key])[0] / pa._llm_cost(k[key])[0] for key in list(c) if key in k
             and pa._measured(c[key]) and pa._measured(k[key])}
    add(r"Committed weights make the prover|The committed-weights prover takes", "C prover / Kpre prover, >=1B params, <=64 tokens; at 2048",
        rng([v for (m, s), v in ratio.items() if s <= 64 and (pa._llm_cost(c[(m, s)])[3] or 0) >= 1e9], x) + "; " +
        rng([v for (m, s), v in ratio.items() if s == 2048], x))
    g = {mode: pa.llm_rows(mode, variant="_gpuv") for mode in ("C", "Kpre")}
    cpu = {"C": c, "Kpre": k}
    gv = {(mode, key): (pa._llm_cost(cpu[mode][key])[1], pa._llm_cost(rows[key])[1])
          for mode, rows in g.items() for key in rows if key in cpu[mode] and "verify_products" in rows[key]}
    add(r"a GPU verifier is", "CPU / GPU verifier at 2048 tokens; at <= 64 tokens",
        rng([a / b_ for (_, (m, s)), (a, b_) in gv.items() if s == 2048], x) + "; " +
        rng([a / b_ for (_, (m, s)), (a, b_) in gv.items() if s <= 64], lambda v: f"{v:.1f}"))
    bat = {}
    for r in pa._read("llm_full_model.csv"):
        if (r["model"], r["seq"], r["mode"], r["challenges"], r["variant"], r["threads"], r.get("batch")) == \
                ("llama2-7b", "64", "C", "int", "_batch", pa.DEFAULT["threads"], "8") and pa._f(r["lam"]) == pa.LAM:
            bat[r["metric"]] = float(r["value"])
    add(r"in batches of eight", "llama2-7b T64 C batch of 8: prover / proof per prompt",
        f"{pa.t(sum(bat[p] for p in pa.PROVE if p in bat) / 8)} / {pa.b(bat['bytes_total'] / 8)}")
    _, _, by70, n70 = pa._llm_cost(c[("llama2-70b", 64)])
    add(r"For Llama-2-70B the extrapolated", "llama2-70b T64 (1-2 blocks): C proof; int8 weights / proof",
        f"{pa.b(by70)}; {x(n70 / by70)}")
    return out


def numbers_opt(M, MO):
    """The optimised protocol's numbers (tables_<optimised>/), and the comparison of Table 5."""
    out = []

    def add(anchor, what, value):
        out.append((anchor, what, value))

    # 5.2: the optimised points of Figure 3
    cells = {lam: pa.CNN_OPT_CELL.format(ch="int", lam=lam) for lam in (40, 128)}
    run = [m for m in pa.CNN_ORDER if MO.get(m, cells[128], "bytes_total") is not None]
    add(r"closer to the target", "optimised: CNN security bits at lambda=40; 128",
        "; ".join(rng([MO.get(m, cells[lam], "soundness_bits") for m in run]) for lam in (40, 128)))
    below = {m: (MO.get(m, cells[128], "bytes_total"), M.get(m, "sampling", "paths_bytes_shared", lam=40)) for m in run}
    if not all(a < p for a, p in below.values()):
        raise SystemExit(f"Figure 3's claim fails: optimised 2^-128 bytes vs paths' 2^-40 bytes {below}")
    add(r"on every model we ran it on|on every model it was run on, reaches", "optimised 2^-128 bytes < paths' 2^-40 bytes, per model",
        ", ".join(f"{pa.SHORT[m]} {pa.b(a)} < {pa.b(p)}" for m, (a, p) in below.items()))
    # 5.4 / Table 4
    rows = pa.opt_pairs(M, MO)
    short = [(lab, bf, af) for lab, m, s, bf, af in rows if s != 2048 and m != "qwen3-4b"]
    proof = rng([bf[2] / af[2] for _, bf, af in short], lambda v: f"{v:.1f}")
    verify = rng([bf[1] / af[1] for _, bf, af in short], lambda v: f"{v:.1f}")
    for anchor in (r"prompts of up to 512 tokens, the proof|the proof becomes",
                   r"below 2\{,\}048 tokens the proof shrinks",
                   r"optimisations that keep the bound and shrink the"):
        add(anchor, "Table 4 below 2,048 tokens (Kpre row excluded): proof smaller; verifier faster",
            f"{proof}x; {verify}x")
    eng = pa.llm_cost(("gpt2", 64), {}, MO.tables)
    add(r"engineering alone, which also speeds up", "GPT-2 T64, basic proof, basic -> released verifier",
        f"{pa.t(pa.llm_cost(('gpt2', 64), {})[1])} -> {pa.t(eng[1])}")
    share = [pa.llm_cost((m, 2048), {"mode": "Kpre", "variant": "_gpuv_stream"}, MO.tables)[2] /
             pa.llm_cost((m, 2048), {"variant": "_gpuv_stream"}, MO.tables)[2]
             for (m, s) in pa.OPT_LLM if s == 2048]
    add(r"tokens the claims are", "Kpre bytes / C bytes at 2,048 tokens (the claims' share, %)",
        rng([100 * v for v in share], lambda v: f"{v:.1f}"))
    l7b = pa.llm_cost(("llama2-7b", 2048), pa.BASIC_LLM[("llama2-7b", 2048)])
    l7o = pa.llm_cost(("llama2-7b", 2048), pa.OPT_LLM[("llama2-7b", 2048)], MO.tables)
    l7k = pa.llm_cost(("llama2-7b", 2048), {"mode": "Kpre", "variant": "_gpuv_stream"}, MO.tables)
    add(r"Llama-2-7B is proved in", "Llama-2-7B T2048 GPU verifier: prover; verifier (basic -> optimised)",
        f"{pa.t(l7b[0])} -> {pa.t(l7o[0])}; {pa.t(l7b[1])} -> {pa.t(l7o[1])}")
    add(r"committed-weights prover then takes", "Llama-2-7B T2048 streaming: C prover / Kpre prover",
        x(l7o[0] / l7k[0]))

    def commit(T, model, cell):
        r = T.idx.get(("llm", model, cell, "commit_total", "", "", "", "", ""))
        return pa._f(r["median"]) if r else None
    g_new = [commit(MO, "gpt2", f"commit_T{s}_L12")for s in (64, 512)]
    g_opt = [commit(MO, "gpt2", cell) for cell in ("commit_T64_L12_wire_prune_polauto", "commit_T64_L12_wire_gpuv_prune_polauto",
                                                    "commit_T512_L12_wire_prune_polauto")]
    l_old = [commit(M, "llama2-7b", f"commit_T{s}_L32") for s in (1, 64)]
    l_opt = [commit(MO, "llama2-7b", f"commit_T{s}_L32_wire_polauto") for s in (1, 64)]
    add(r"one-time setup is longer|price is a longer one-time setup", "commit: GPT-2 (released code, untagged) -> optimised; Llama-2-7B basic -> optimised",
        f"{rng(g_new, lambda v: f'{v:.0f}')} s -> {rng(g_opt, lambda v: f'{v:.0f}')} s; "
        f"{rng(l_old, lambda v: f'{v / 60:.1f}')} -> {rng(l_opt, lambda v: f'{v / 60:.1f}')} min")
    add(r"optimised codes make up to", "commit slow-down, GPT-2 (same seq, released code untagged -> optimised)",
        rng([commit(MO, "gpt2", f"commit_T{s}_L12_wire_prune_polauto") / commit(MO, "gpt2", f"commit_T{s}_L12")
             for s in (64, 512)], x))
    l13 = pa.llm_cost(("llama2-13b", 2048), pa.OPT_LLM[("llama2-13b", 2048)], MO.tables)
    rss = MO.idx.get(("llm", "llama2-13b", "defence_C_int_lam128_T2048_L40_gpuv_stream", "host_peak_rss",
                      "", "", "", "", ""))
    add(r"streaming verifier takes", "Llama-2-13B T2048 C streaming verifier; / Llama-2-7B's; host peak RSS",
        f"{pa.t(l13[1])}; {x(l13[1] / l7o[1])}; {pa._f(rss['median']) / 2 ** 30:.0f} GiB")
    # 5.5 / Table 5
    rr = {(s, w): cells_ for s, w, cells_, _ in pa.ratio_rows(MO)}
    q = {key: [a / o if a and o else None for a, o in v] for key, v in rr.items()}
    pro = {(s, w): v[0] for (s, w), v in q.items() if v[0] and not s.endswith("$^c$") and not s.startswith("Maverick")}
    lo, hi = min(pro, key=pro.get), max(pro, key=pro.get)
    for anchor in (r"Ours is faster than every zkSNARK prover", r"On matched models our prover is"):
        add(anchor, "prover speed-up over the zkSNARK rows of Table 5 (min, max)",
            f"{x(pro[lo])} ({lo[0]}, {lo[1]}) -- {x(pro[hi])} ({hi[0]}, {hi[1]})")
    zk7 = q[("zkLLM$^c$", "Llama-2-7B (2{,}048)")][0]
    basic7 = pa.zkllm_l40s_whole_model_s() / pa.llm_cost(("llama2-7b", 2048), {})[0]
    for anchor in (r"this gives about", r"faster than zkLLM's\s*$"):
        add(anchor, "zkLLM's code on our L40S (s); / optimised prover; / basic prover (CPU-verifier run)",
            f"{pa.zkllm_l40s_whole_model_s():.0f}; {x(zk7)}; {x(basic7)}")
    zk = {row["model"]: float(row["per_layer_s"]) for row in csv.DictReader(open(pa.ZKLLM_L40S, encoding="utf-8"))}
    add(r"whose time per layer grows", "zkLLM demo time per layer, Llama-2-13B / Llama-2-7B",
        x(zk["llama2-13b"] / zk["llama2-7b"]))
    slower = {s: 1 / v[1] for (s, w), v in q.items() if v[1] and v[1] < 1}
    add(r"Our verifier is faster than all but", "verifier slow-downs (theirs/ours < 1)",
        ", ".join(f"{s} {w}: {x(1 / q[(s, w)][1])}" for (s, w) in q if q[(s, w)][1] and q[(s, w)][1] < 1))
    add(r"beats zkLLM's on four of five", "zkLLM rows: GPU verifier faster on n of 5; Llama-2-13B slower by",
        f"{sum(1 for (s, w), v in q.items() if s == 'zkLLM' and v[1] >= 1)} of "
        f"{sum(1 for s, w in q if s == 'zkLLM')}; {x(1 / q[('zkLLM', 'Llama-2-13B (2{,}048)')][1])}")
    trip = {}
    for s, w, cells_, _ in pa.ratio_rows(MO):
        if (s, w) in (("zkCNN", "LeNet-5"), ("DeepProve", "GPT-2 (64)")):
            trip[s] = cells_
    add(r"the optimised protocol wins on all three costs",
        "ours vs zkCNN LeNet-5 / DeepProve GPT-2 (64), Fiat-Shamir: prover, verifier, proof",
        "; ".join(f"{s}: " + ", ".join(f"{f(o)} vs {f(a)}" for (a, o), f in zip(v, (pa.t, pa.t, pa.b)))
                  for s, v in trip.items()))
    bolds = [f"{s} {w}" for (s, w), v in q.items() if all(r_ is not None and r_ >= 1 for r_ in v)]
    add(r"wins on all three costs", "rows better on all three (bold)", ", ".join(bolds))
    short_lose = [1 / v[2] for (s, w), v in q.items() if v[2] and v[2] < 1 and s != "zkLLM"]
    zkl = [1 / v[2] for (s, w), v in q.items() if v[2] and s == "zkLLM"]
    add(r"Elsewhere the proof is our", "proof larger: other short-input rows; zkLLM rows",
        f"{rng(short_lose, x)}; {rng(zkl, x)}")
    mav = q[("Maverick$^b$", "Qwen3-4B (8)")]
    mo = rr[("Maverick$^b$", "Qwen3-4B (8)")]
    add(r"Maverick's own setting", "Maverick: our proof smaller by; our prover faster by", f"{x(mav[2])}; {x(mav[0])}")
    ms = lambda v: f"{v * 1e3:.1f} ms"   # noqa: E731
    add(r"of matrix checks plus the", "Maverick's verifier (matrix checks + non-linear replay); ours",
        f"{ms(mo[1][0])} ({ms(mo[1][0] - pa.MAVERICK_NONLINEAR_S)} + {ms(pa.MAVERICK_NONLINEAR_S)}); {ms(mo[1][1])}")
    # Section 1 and 6: the optimised Llama-2-7B proof at 64 tokens
    l64 = pa.llm_cost(("llama2-7b", 64), pa.OPT_LLM[("llama2-7b", 64)], MO.tables)
    add(r"is 762\\,MB, or|proof at \$2\^\{-128\}\$ is", "Llama-2-7B T64 proof: basic, optimised",
        f"{pa.b(pa.llm_cost(('llama2-7b', 64), {})[2])}, {pa.b(l64[2])}")
    add(r"optimised proof is far smaller than the model|far smaller than the",
        "Llama-2-7B T64: int8 weights / optimised proof", x(l64[3] / l64[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=pa.PLATFORM,
                    help="the basic protocol's run (default: $PVI_PLATFORM, else l40s)")
    ap.add_argument("--optimised", default="", help="the optimised protocol's run (the report: l40s_improved)")
    ap.add_argument("--tex", type=Path, default=MAIN, help="the text to check (default: report/main.tex)")
    args = ap.parse_args()
    pa.TABLES = pa.tables_dir(args.platform)
    M = pa.Measured()
    MO = None
    if args.optimised:
        pa.OPT_TABLES = pa.tables_dir(args.optimised)
        MO = pa.Measured(pa.OPT_TABLES)
    gaps = pa.missing(M, MO)
    if gaps:   # the report's text needs every number Figures 3-4 and Tables 2-5 draw
        raise SystemExit(f"{len(gaps)} missing\n  " + "\n  ".join(gaps))
    lines = args.tex.read_text(encoding="utf-8").splitlines()
    print(f"numbers from {pa.TABLES.name}/" + (f" and {pa.OPT_TABLES.name}/" if MO else "") + f" against {args.tex.name}")
    lost = 0
    for anchor, what, value in numbers(M) + (numbers_opt(M, MO) if MO else []):
        i = next((i for i, l in enumerate(lines) if re.search(anchor, l)), None)
        if i is None:
            lost += 1
            print(f"l.?: {what}\n    computed: {value}\n    text:     (anchor not found: {anchor})")
            continue
        text = " ".join(l.strip() for l in lines[i:i + 2])
        print(f"l.{i + 1}: {what}\n    computed: {value}\n    text:     {text[:200]}")
    if lost:
        raise SystemExit(f"{lost} anchors not found")


if __name__ == "__main__":
    main()
