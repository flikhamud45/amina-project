"""Check every number ``report/main.tex`` quotes in running text against the stored tables.

    python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2
    python experiments/5_comparison/text_numbers.py ... --tex <draft.tex>        # check another file
    python experiments/5_comparison/text_numbers.py ... --list                    # only print the values

Generated tables and figures follow the data by themselves; these sentences do not.  Each check
finds its sentence by a search on the wording around the number (not on the number), takes that
line and the next three (LaTeX comments removed), and compares what the text prints with the value
recomputed from the tables, the same way ``paper_assets.py`` computes the tables (the optimised
protocol's cells as defined in ``paper_assets.OPTIMISED``, the basic protocol's on ``--platform``).
The exit status is 1 if any check fails:

* ``MISMATCH``: a final number differs from the text;
* ``UNWRAP``: a final number is still inside ``\\pending{}`` (after the re-run: make it black);
* ``UNMARKED``: a number that depends on a pending cell is not inside ``\\pending{}``;
* ``ANCHOR``: the sentence is no longer found (update the anchor below with the text).

A number that depends on a cell still pending is expected inside ``\\pending{}``; its placeholder is
the value the tables show (the basic protocol's value of the pending cells, element by element).  A
different red placeholder is reported (``placeholder``) but does not fail: the value is temporary.
Numbers shown from an acceptable stand-in (``paper_assets.OPTIMISED[...]["acceptable"]``) may be
black or red.  ``\\pendingclaim{}`` is transparent: the numbers inside it are checked as usual.  The
counts (abstract, Sec. 5.2, conclusion) are recomputed from the raw records by ``count_outcomes.py``;
an optimised run that has no raw root yet makes the optimised counts pending.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_outcomes as co  # noqa: E402
import paper_assets as pa  # noqa: E402

MAIN = pa.REPORT / "main.tex"
FINAL, PENDING, ACCEPTABLE = "final", "pending", "acceptable"
UNITS = ("min", "ms", "kB", "MB", "GB", "GiB", "s", "B", "×", "%")


# ------------------------------------------------------------------------------------- the sources
class Src:
    """Where one element's inputs come from: the optimised protocol (with pending / acceptable tracking)
    or the basic protocol (the placeholders)."""

    def __init__(self, optimised: bool):
        self.opt = optimised
        self.pending = False
        self.acceptable = False

    def _track(self, fn, *a, **k):
        n = len(pa.PENDING)
        v, p = fn(*a, **k, where=("text", "", ""))
        new = pa.PENDING[n:]
        del pa.PENDING[n:]
        self.pending |= p
        self.acceptable |= any(e["status"] == ACCEPTABLE for e in new)
        return v

    def cnn(self, model, metric, mode="C", ch="int", lam=None):
        if not self.opt:
            return pa.basic_cnn(model, metric, mode, ch, lam)
        return self._track(pa.opt_cnn, model, metric, mode, ch, lam)

    def llm(self, model, seq, metric, mode="C", ch="int", full_only=True):
        if not self.opt:
            return pa.basic_llm(model, seq, metric, mode, ch, full_only)
        return self._track(pa.opt_llm, model, seq, metric, mode, ch, full_only=full_only)


class V:
    """One element's value and state: final, pending (the basic placeholder) or acceptable."""

    def __init__(self, value, state):
        self.value, self.state = value, state


def elem(fn, *args) -> V:
    """Evaluate fn(src, *args) on the optimised protocol; if any input is pending, on the basic protocol."""
    s = Src(True)
    v = fn(s, *args)
    if s.pending:
        return V(fn(Src(False), *args), PENDING)
    return V(v, ACCEPTABLE if s.acceptable else FINAL)


def fixed(value) -> V:
    """A number that does not depend on the optimised runs (the basic run, the literature, analytic)."""
    return V(value, FINAL)


def worst(*states):
    return PENDING if PENDING in states else (ACCEPTABLE if ACCEPTABLE in states else FINAL)


# ------------------------------------------------------------------------------------- formatting
def x(v):
    """A ratio as the text prints it: 1.4, 16, 7,024."""
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def x2(v):
    """A large ratio rounded to two significant figures, as the text prints it: 32,000, 96,000, 5,200."""
    if v < 1000:
        return x(v)
    e = 10 ** (int(math.log10(v)) - 1)
    return f"{round(v / e) * e:,.0f}"


def tex2plain(s: str) -> str:
    return s.replace("{,}", ",").replace("\\,", "")


def unit_split(s: str):
    """'27.4\\,ms' -> ('27.4', 'ms')."""
    s = tex2plain(s)
    m = re.match(r"^([\d.,]+)(\D*)$", s)
    return (m.group(1), m.group(2)) if m else (s, "")


class Tok:
    """One number the text must print: its segments (text, state), joined by an en dash for a range,
    and the unit after the last segment (part of the match)."""

    def __init__(self, segs, unit="", mid=""):
        self.segs, self.unit, self.mid = segs, unit, mid   # mid: the first endpoint's unit, if it differs

    @property
    def state(self):
        return worst(*(st for _, st in self.segs))

    def show(self):
        return (self.mid + "–").join(f"⟦{t}⟧" if st == PENDING else t for t, st in self.segs) + self.unit

    def plain(self):
        return (self.mid + "–").join(t for t, _ in self.segs) + self.unit


def one(v: V, fmt=x, unit="") -> Tok:
    if v.value is None:
        return Tok([(None, v.state)], unit)
    s = fmt(v.value)
    if fmt in (pa.t, pa.b):
        s, unit = unit_split(s)
    return Tok([(tex2plain(s), v.state)], unit)


def rng(vals: list, fmt=x, unit="") -> list:
    """A range 'lo--hi' over elements (V): one token, or two when the endpoints' units differ (49.5 kB--
    12.1 MB: the first endpoint's unit is ``mid``).  The range is pending if any element is, acceptable if
    any is, else final.  In a pending range an endpoint that a final (or acceptable) element attains may be
    black or red: the text may mark only the end a pending cell supplies ('3.8--\\pending{27.4}').  That is
    safe, because once the re-run stores the pending cell the range is recomputed, and an endpoint it moves
    fails as MISMATCH."""
    state = worst(*(v.state for v in vals))
    vals = [v for v in vals if v.value is not None]
    lo, hi = min(vals, key=lambda v: v.value), max(vals, key=lambda v: v.value)
    a, b_ = fmt(lo.value), fmt(hi.value)

    def end_state(shown):   # the endpoint's own state: pending only if every element printing it is
        if state != PENDING:
            return state
        return PENDING if all(v.state == PENDING for v in vals if fmt(v.value) == shown) else ACCEPTABLE
    sa, sb = end_state(a), end_state(b_)
    if fmt in (pa.t, pa.b):
        (a, ua), (b_, ub) = unit_split(a), unit_split(b_)
        if ua != ub:
            return [Tok([(a, sa), (b_, sb)], ub, mid=ua)]
        unit = ua
    a, b_ = tex2plain(a), tex2plain(b_)
    if a == b_:
        return [Tok([(a, worst(sa, sb))], unit)]
    return [Tok([(a, sa), (b_, sb)], unit)]


def ends(vals: list, fmt=x, unit="") -> list:
    """The two endpoints of rng() as separate tokens, for a text that names them apart ('by 30x (zkLLM) to
    32,000x (ZKML)')."""
    (tok,) = rng(vals, fmt, unit)
    if len(tok.segs) == 1:
        return [tok]
    return [Tok([tok.segs[0]], tok.mid or tok.unit), Tok([tok.segs[1]], tok.unit)]


# ------------------------------------------------------------------------------------- the text
def _unbrace(s: str, macro: str, left: str, right: str) -> str:
    """Replace \\macro{...} (balanced braces) by left ... right."""
    out, i, key = [], 0, "\\" + macro + "{"
    while True:
        j = s.find(key, i)
        if j < 0:
            return "".join(out) + s[i:]
        out.append(s[i:j])
        k, depth = j + len(key), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        out.append(left + s[j + len(key):k - 1] + right)
        i = k


def strip_comment(line: str) -> str:
    m = re.search(r"(?<!\\)%", line)
    return line[:m.start()] if m else line


def normalise(s: str) -> str:
    """LaTeX to the plain form the tokens use: \\pending{x} -> ⟦x⟧ (units moved outside), \\pendingclaim
    removed, {,} -> ',', no thin spaces, $\\times$ -> ×, -- -> –."""
    s = _unbrace(s, "pendingclaim", "", "")
    s = _unbrace(s, "pending", "⟦", "⟧")
    s = s.replace("{,}", ",").replace("\\,", "").replace("~", " ").replace("\\%", "%")
    s = re.sub(r"\$?\\times\$?", "×", s).replace("$", "")
    s = s.replace("---", "—").replace("--", "–")
    s = re.sub(r"⟦([^⟧]*?)\s*(" + "|".join(UNITS) + r")⟧", r"⟦\1⟧\2", s)
    # a red range \pending{a--b} marks both endpoints: ⟦a–b⟧ -> ⟦a⟧–⟦b⟧ (a unit after a stays outside)
    s = re.sub(r"⟦([\d.,/]+)\s*((?:" + "|".join(UNITS) + r")?)\s*–\s*([\d.,/]+)⟧", r"⟦\1⟧\2–⟦\3⟧", s)
    return re.sub(r"\s+", " ", s)


def _pattern(tok: Tok, any_state=False) -> str:
    """A regex for the token in a normalised window.  Each endpoint follows its own state: final black,
    pending red (⟦x⟧), acceptable either.  ``any_state``: any colour, with groups o<i>/c<i> capturing the
    brackets of endpoint i (judge() uses them to say which endpoint has the wrong colour)."""
    parts = []
    for i, (t, st) in enumerate(tok.segs):
        num = r"[\d.,/]+" if t is None else re.escape(t)
        if any_state:
            parts.append(f"(?P<o{i}>⟦?){num}(?P<c{i}>⟧?)")
        elif st == PENDING:
            parts.append(f"⟦{num}⟧")
        elif st == ACCEPTABLE:
            parts.append(f"⟦?{num}⟧?")
        else:
            parts.append(f"(?<!⟦){num}(?!⟧)")
    pat = f"{re.escape(tok.mid)}–".join(parts) + re.escape(tok.unit)
    return r"(?<![\d.,])" + pat + (r"(?![\d])" if not tok.unit else "")


def expand_macros(text: str) -> str:
    """Replace the generated macros (tables/counts*.tex, tables/hardware.tex) by their definitions."""
    defs = {}
    for name in ("counts.tex", "counts_opt.tex", "hardware.tex"):
        path = pa.TABS / name
        if path.exists():
            for m in re.finditer(r"^\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", path.read_text(encoding="utf-8"), re.M):
                defs[m.group(1)] = m.group(2)
    for k in sorted(defs, key=len, reverse=True):
        text = re.sub(r"\\" + k + r"(?![A-Za-z])(\{\})?", lambda _m, v=defs[k]: v, text)
    return text


def judge(tok: Tok, text: str):
    """(status, detail) of one token in a normalised window."""
    if re.search(_pattern(tok), text):
        return "ok", ""
    found_any = re.search(_pattern(tok, any_state=True), text) if tok.segs[0][0] is not None else None
    if found_any:   # the value is there in the wrong colour: say which endpoint
        red = [bool(found_any.group(f"o{i}") or found_any.group(f"c{i}")) for i in range(len(tok.segs))]
        if any(st == PENDING and not r for (_, st), r in zip(tok.segs, red)):
            return "UNMARKED", f"{tok.plain()} depends on a pending cell: wrap {tok.show()} in \\pending{{}}"
        if any(st == FINAL and r for (_, st), r in zip(tok.segs, red)):
            return "UNWRAP", f"{tok.plain()} is final: remove its \\pending{{}}"
    if tok.state == PENDING:
        if "⟦" in text:
            return "placeholder", f"red placeholder differs from the tables' {tok.show()}"
        return "UNMARKED", f"no \\pending{{}} in the sentence, but it depends on a pending cell ({tok.show()})"
    return "MISMATCH", f"expected {tok.show()}"


# ------------------------------------------------------------------------------------- the checks
COUNT_TAGS: list = []   # --require-tag: count only the cells carrying these tags (as count_outcomes.py)
COUNT_DEFINITION: list = []   # --definition: count only the optimised definition's cells (one element: True)


def counts(names: list[str], tags=()):
    """(values, state) of count_outcomes' macros over the listed runs (pending if one has no raw root)."""
    roots = [pa.BASE / f"raw_{n}" for n in names]
    have = [r for r in roots if (r / "PLATFORM.json").exists()]
    definition = pa.definition_tagsets() if COUNT_DEFINITION and names != [pa._platform_name(pa.TABLES)] else None
    c = co.merge([co.count(r, tags=tags, definition=definition) for r in have])
    m = co.macros(c, "")
    return m, (FINAL if len(have) == len(roots) else PENDING)


def checks(M, opt_names: list[str]):
    """[(anchor regex, what, [Tok])]: every number of the text, recomputed."""
    out = []

    def add(anchor, what, toks):
        out.append((anchor, what, toks if isinstance(toks, list) else [toks]))

    C = lambda m: f"{m:,}"   # noqa: E731
    bc, _ = counts([pa._platform_name(pa.TABLES)])
    oc, ost = counts(opt_names, COUNT_TAGS) if opt_names else ({}, PENDING)
    oget = lambda k: Tok([(C(oc[k]) if k in oc else None, ost)])   # noqa: E731
    # abstract, Sec. 5.2, conclusion: the counts
    for anchor in (r"attacks on a basic version and all", r"attacks on the basic protocol and all"):
        add(anchor, "attacks rejected: basic run; optimised runs", [Tok([(C(bc["NAttacks"]), FINAL)]), oget("NAttacks")])
    add(r"In the optimised runs the verifier accepted", "optimised: honest accepted, attacks, image-model attacks",
        [oget("NHonest"), oget("NAttacks"), oget("NAttacksCNN")])
    # (optional: the clause is deleted once the counts use --definition, where every image-model attack is
    # on commitments built under the planning rule)
    add(r"?built under the planning", "optimised: image-model attacks under the planning rule", oget("NAttacksPlans"))
    add(r"middle-layer pre-activation", "optimised: '+1' language-model attacks", oget("NAttacksPlusOne"))
    add(r"Freivalds' check rejected", "optimised: rejected by Freivalds; by the code check", [oget("NFreivalds"),
                                                                                            oget("NColumnsCode")])
    add(r"the Merkle check rejected all|forged folded rows", "optimised: rejected by the Merkle check",
        oget("NColumnsMerkle"))
    add(r"basic protocol's run accepted", "basic: honest accepted, attacks rejected, on language models",
        [Tok([(C(bc["NHonest"]), FINAL)]), Tok([(C(bc["NAttacks"]), FINAL)]), Tok([(C(bc["NAttacksLLM"]), FINAL)])])

    # 3.4: the path protocol on the image models (basic run; hardware-independent)
    pdet = [M.get(m, "sampling", "p_detect_penultimate") for m in pa.CNN_ORDER]
    add(r"One path catches it with probability", "one path, attacked neuron: 1/p (LeNet-5; largest)",
        [Tok([(f"1/{1 / M.get('lenet5', 'sampling', 'p_detect_penultimate'):.0f}", FINAL)]),
         Tok([(f"1/{1 / min(pdet):.0f}", FINAL)])])
    add(r"least-visited neuron", "least-visited neuron: 1/p (millions)",
        Tok([(f"1/{1 / min(M.get(m, 'sampling', 'p_detect_min_node') for m in pa.CNN_ORDER) / 1e6:.0f}", FINAL)],
            " million"))
    add(r"\$2\^\{-40\}\$ takes|Reaching \$2\^\{-40\}\$ takes", "paths for 2^-40",
        rng([fixed(M.get(m, "sampling", "paths_needed_penultimate", lam=40)) for m in pa.CNN_ORDER],
            lambda v: f"{v:,.0f}"))
    an = next(r for r in pa._read("analytic.csv", pa.BASE / "tables") if r["model"] == "llama2-7b")
    width, paths, opened = int(an["anchuri_width"]), int(an["anchuri_paths"]), float(an["anchuri_open_all_bytes"])
    add(r"On Llama-2-7B one path detects", "Llama-2-7B T64: 1/N; paths for 2^-40 (thousands); bytes opened",
        [Tok([(f"1/{width:,}", FINAL)]), Tok([(f"{round(paths, -3):,}", FINAL)]),
         one(fixed(opened), pa.b)])
    add(r"?single-neuron attack on Llama-2-7B with probability", "one path on Llama-2-7B detects 1/N",
        Tok([(f"1/{width:,}", FINAL)]))
    # Sec. 1: the paths' bytes against the optimised Llama-2-7B proof at 64 tokens
    add(r"takes paths that open about", "paths' bytes for 2^-40; optimised Llama-2-7B T64 proof",
        [one(fixed(opened), pa.b), one(elem(lambda s: s.llm("llama2-7b", 64, "bytes")), pa.b)])

    # 5.1
    gap = [(M.get(m, "facts", "float_accuracy") or 0) - (M.get(m, "facts", "int8_accuracy") or 0) for m in pa.CNN_ORDER]
    for anchor in (r"quantisation changed accuracy by at most", r"changed CNN accuracy by at most"):
        add(anchor, "max |float - int8| accuracy (points)", Tok([(f"{100 * max(map(abs, gap)):.1f}", FINAL)]))
    def extrap_errors(dirs, optimised):
        """|extrapolated / full - 1| of the proof bytes (%) over llm_extrapolation_check.csv: the basic run's
        untagged rows, or the optimised runs' rows with the compact encoding (the definition's builds)."""
        groups = {}
        for d in dirs:
            for r in pa._read("llm_extrapolation_check.csv", d):
                key = tuple(r[f] for f in ("model", "seq", "mode", "challenges", "lam", "threads", "variant"))
                groups.setdefault(key, {})[r["metric"]] = r
        return [abs(100 * (pa._f(d["bytes_total"]["extrapolated"]) / pa._f(d["bytes_total"]["value"]) - 1))
                for kk, d in groups.items() if "bytes_total" in d and kk[5] == pa.DEFAULT["threads"]
                and (("wire" in pa.tagset(kk[6])) if optimised else kk[6] == "")]
    be_opt = extrap_errors(pa.opt_dirs(), True)
    be = be_opt or extrap_errors([pa.TABLES], False)   # the basic check stands in until the re-run's (red)
    add(r"we extrapolate proof sizes|extrapolated size is within", "extrapolated vs full proof size: max |error| (%, 2 "
        "decimals; optimised builds, else the basic run's as placeholder)",
        Tok([(f"{max(be):.2f}", FINAL if be_opt else PENDING)], "%"))

    # 5.2: Figure 3
    bits = lambda lam: lambda s, m: s.cnn(m, "bits", lam=lam)   # noqa: E731
    add(r"lands close to", "optimised: CNN security bits at lambda=40; 128",
        [*rng([elem(bits(40), m) for m in pa.CNN_ORDER]), *rng([elem(bits(128), m) for m in pa.CNN_ORDER])])
    add(r"overshoots its target", "basic: CNN security bits at lambda=40; 128",
        [*rng([fixed(pa.basic_cnn(m, "bits", lam=40)) for m in pa.CNN_ORDER]),
         *rng([fixed(pa.basic_cnn(m, "bits", lam=128)) for m in pa.CNN_ORDER])])
    add(r"it sends [\d.]+--[\d.]+\$\times\$ fewer bytes|it sends\s+\S+ fewer bytes",
        "basic at lambda=40: paths' bytes for 2^-40 / proof",
        rng([fixed(M.get(m, "sampling", "paths_bytes_shared", lam=40) / pa.basic_cnn(m, "bytes", lam=40))
             for m in pa.CNN_ORDER], lambda v: f"{v:.1f}"))

    # 5.3 image models (Table 2, optimised)
    cnn = lambda k: [elem(lambda s, m: s.cnn(m, k), m) for m in pa.CNN_ORDER]   # noqa: E731
    i8 = lambda s, m: M.get(m, "facts", "model_bytes_int8") / s.cnn(m, "bytes")   # noqa: E731
    add(r"With committed weights \(Table", "CNN prover; verifier; proof (C, optimised)",
        [*rng(cnn("prove"), pa.t), *rng(cnn("verify"), pa.t), *rng(cnn("bytes"), pa.b)])
    add(r"smaller than the int8 weights for|smaller than the int8 weights, by",
        "int8 weights / proof (the MNIST and CIFAR models)",
        rng([elem(i8, m) for m in ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar")], lambda v: f"{v:.1f}"))
    # 5.3 language models (Table 3, optimised)
    L = lambda m, sq, k, mode="C": elem(lambda s: s.llm(m, sq, k, mode))   # noqa: E731
    l64 = lambda s: pa.opt_llm("llama2-7b", 64, "params")[0] / s.llm("llama2-7b", 64, "bytes")   # noqa: E731
    add(r"With a 64-token prompt GPT-2 is proved in", "GPT-2 T64 prover; Llama-2-7B T64 prover, proof, int8/proof",
        [one(L("gpt2", 64, "prove"), pa.t), one(L("llama2-7b", 64, "prove"), pa.t), one(L("llama2-7b", 64, "bytes"), pa.b),
         one(elem(l64), x, "×")])
    add(r"tokens Llama-2-7B is proved in", "Llama-2-7B T2048 prover, proof, CPU verifier",
        [one(L("llama2-7b", 2048, "prove"), pa.t), one(L("llama2-7b", 2048, "bytes"), pa.b),
         one(L("llama2-7b", 2048, "verify"), pa.t)])
    gpu_rows = [(m, s) for m, s in pa.TAB_LLM_PICK if pa.basic_llm(m, s, "gpu") is not None or
                pa.opt_llm(m, s, "gpu")[0] is not None]
    ratio = lambda m, sq: lambda s: s.llm(m, sq, "verify") / s.llm(m, sq, "gpu")   # noqa: E731
    add(r"a GPU verifier is", "C: CPU / GPU verifier at 2,048 tokens; at 64 (Table 3 rows with both)",
        [*rng([elem(ratio(m, s)) for m, s in gpu_rows if s == 2048]),
         *rng([elem(ratio(m, s)) for m, s in gpu_rows if s == 64], lambda v: f"{v:.1f}")])

    def batch(s):
        tags = frozenset({"wire", "batch", "prune", "polauto"}) if s.opt else frozenset({"batch"})
        rows = {}
        for r in pa._llm_table(pa._llm_dirs(pa.opt_dirs() if s.opt else None)):
            if (r["model"], r["seq"], r["mode"], r["challenges"], r.get("threads"), r.get("batch")) == \
                    ("llama2-7b", "64", "C", "int", pa.DEFAULT["threads"], "8") and pa.tagset(r["variant"]) == tags \
                    and pa._f(r["lam"]) == pa.LAM:
                rows[r["metric"]] = float(r["value"])
        if s.opt and not rows:
            s.pending = True
            return None
        return sum(rows[p] for p in pa.PROVE if p in rows) / 8, rows["bytes_total"] / 8
    bv = elem(batch)
    add(r"in batches of eight", "Llama-2-7B T64 C, batch of 8: prover, proof per prompt",
        [one(V(bv.value[0], bv.state), lambda v: f"{v:.2f}", "s"), one(V(bv.value[1], bv.state), pa.b)])
    p70 = lambda s: s.llm("llama2-70b", 64, "bytes", full_only=False)   # noqa: E731
    n70 = pa.llm_cost(("llama2-70b", 64), {}, None)[3]
    add(r"For Llama-2-70B the extrapolated", "Llama-2-70B T64 (1-2 blocks): proof; int8 weights / proof",
        [one(elem(p70), pa.b), one(elem(lambda s: n70 / p70(s)), x, "×")])

    # 5.4 / Table 4
    rows = pa.opt_pairs()
    short = [(lab, bf, af, pd) for lab, m, s, bf, af, pd in rows if s != 2048 and m != "qwen3-4b"]
    acc = {r["row"] for r in pa.pending_rows(ACCEPTABLE) if r["asset"] == "Table 4"}
    t4 = lambda i: [V(bf[i] / af[i], PENDING if pd[i] else (ACCEPTABLE if pa._plain(lab) in acc else FINAL))  # noqa: E731
                    for lab, bf, af, pd in short]
    add(r"below 2\{,\}048 tokens, the proof shrinks|below 2\{,\}048 tokens the proof shrinks",
        "Table 4 below 2,048 tokens (Qwen3-4B excluded): proof smaller; verifier faster",
        [*rng(t4(2), lambda v: f"{v:.1f}"), *rng(t4(1), lambda v: f"{v:.1f}")])
    add(r"verifier gain in setting", "Table 4 below 2,048 tokens (Qwen3-4B excluded): verifier faster (Sec. 5.4)",
        rng(t4(1), lambda v: f"{v:.1f}"))
    add(r"Optimisations keep the bound and shrink", "Table 4 below 2,048 tokens (Qwen3-4B excluded): proof smaller",
        rng(t4(2), lambda v: f"{v:.1f}"))
    eng = pa.llm_cost(("gpt2", 64), {}, pa.opt_dirs())
    add(r"speeds up the basic proofs", "GPT-2 T64 basic proof: basic run's verifier -> released code's",
        [Tok([(f"{pa.basic_llm('gpt2', 64, 'verify') * 1e3:.0f}", FINAL)]), one(fixed(eng[1] if eng else None), pa.t)])
    # the two MNIST models, whose prover slows down: '(4.3 to 5.3 ms for LeNet-5 and 3.1 to 3.8 ms for the MLP)
    # ... (3.0 and 1.9 ms without the compact encoding)'
    pol = {m: opt_measured_cost(m, "defence_C_int_lam128_rate4_polauto") for m in ("lenet5", "mlp_mnist")}
    bare = lambda v: Tok([(pa.t(v).split("\\")[0], FINAL)])   # noqa: E731  (a number whose unit follows later)
    add(r"except on the two MNIST models", "LeNet-5, MLP prover basic -> optimised; optimised without the compact "
        "encoding", [bare(pa.basic_cnn("lenet5", "prove")), one(elem(lambda s: s.cnn("lenet5", "prove")), pa.t),
                     bare(pa.basic_cnn("mlp_mnist", "prove")), one(elem(lambda s: s.cnn("mlp_mnist", "prove")), pa.t),
                     bare(pol["lenet5"]), one(fixed(pol["mlp_mnist"]), pa.t)])
    l7 = [r for r in rows if r[1] == "llama2-7b" and r[2] == 2048][0]
    add(r"the optimised protocol proves Llama-2-7B", "Table 4 row g: prover, verifier, proof optimised; basic",
        [one(V(l7[4][0], PENDING if l7[5][0] else FINAL), pa.t), one(V(l7[4][1], PENDING if l7[5][1] else FINAL), pa.t),
         one(V(l7[4][2], PENDING if l7[5][2] else FINAL), pa.b),
         one(fixed(l7[3][0]), pa.t), one(fixed(l7[3][1]), pa.t), one(fixed(l7[3][2]), pa.b)])
    stream = {mode: pa.llm_cost(("llama2-7b", 2048), {"mode": mode, "variant": "_gpuv_stream"}, pa.opt_dirs())
              for mode in ("C", "Kpre")}
    if stream["C"]:
        add(r"already take", "basic proof format, released prover + streaming GPU verifier: prover, verifier; C/Kpre prover",
            [one(fixed(stream["C"][0]), pa.t), one(fixed(stream["C"][1]), pa.t),
             one(fixed(stream["C"][0] / stream["Kpre"][0]), x, "×")])

    def commit(T, model, cell):
        r = T.idx.get(("llm", model, cell, "commit_total", "", "", "", "", ""))
        return pa._f(r["median"]) if r else None
    MO = pa.opt_measured()
    g_new = [commit(MO, "gpt2", f"commit_T{s}_L12") for s in (64, 512)]
    g_opt = [commit(MO, "gpt2", c) for c in ("commit_T64_L12_wire_prune_polauto", "commit_T64_L12_wire_gpuv_prune_polauto",
                                             "commit_T64_L12_wire_gpuv_stream_prune_polauto",
                                             "commit_T512_L12_wire_prune_polauto")]
    l_old = [commit(M, "llama2-7b", f"commit_T{s}_L32") for s in (1, 64)]
    l_def = [commit(MO, "llama2-7b", f"commit_T{s}_L32_wire_prune_polauto") for s in (1, 64)]
    l_acc = [commit(MO, "llama2-7b", f"commit_T{s}_L32_wire_polauto") for s in (1, 64)]
    l_state = FINAL if None not in l_def else ACCEPTABLE
    l_opt = l_def if None not in l_def else l_acc
    mean = lambda v: sum(v) / len(v)   # noqa: E731
    add(r"one-time setup (is )?longer", "setup: GPT-2 optimised; GPT-2 released code; Llama-2-7B optimised, basic (min)",
        [*rng([fixed(v) for v in g_opt if v], lambda v: f"{v:.0f}", "s"), *rng([fixed(v) for v in g_new], lambda v: f"{v:.0f}", "s"),
         Tok([(f"{mean(l_opt) / 60:.1f}", l_state)]), Tok([(f"{mean(l_old) / 60:.1f}", FINAL)], "min")])
    add(r"longer codewords make up to", "setup slow-down, GPT-2 (same length, released code -> optimised)",
        Tok([(x(max(commit(MO, "gpt2", f"commit_T{s}_L12_wire_prune_polauto") / commit(MO, "gpt2", f"commit_T{s}_L12")
                    for s in (64, 512))), FINAL)], "×"))

    # 5.5 / Table 5
    # Table 5's rows; system() drops the footnote letters of SYSTEM_MARK (zkCNN$^a$ -> zkCNN), not Maverick's
    # d, zkLLM's own code's e (pa.ZKLLM_OWN) or the interactive asterisk
    rr = pa.ratio_rows()
    marks = {m for m in pa.SYSTEM_MARK.values()}
    system = lambda s: re.sub(r"\$\^\{?([a-z,]+)\}?\$$",   # noqa: E731
                              lambda m: "" if set(m.group(1).split(",")) <= marks else m.group(0), s)
    st = lambda p: PENDING if p else FINAL   # noqa: E731
    q = {(s, w): [V(a / o if a and o else None, st(p)) for (a, o), p in zip(cells, pend)] for s, w, cells, _, pend in rr}
    zk = [v[0] for (s, w), v in q.items() if s != pa.ZKLLM_OWN and not s.startswith("Maverick")]
    add(r"Our prover is .*faster than the zkSNARK", "prover: theirs/ours over the zkSNARK rows (contributions)", rng(zk, x2))
    add(r"Ours is faster than every zkSNARK prover", "prover: theirs/ours over the zkSNARK rows (smallest; largest)",
        ends(zk, x2))
    ours7 = elem(lambda s: s.llm("llama2-7b", 2048, "prove"))
    zk7 = pa.zkllm_l40s_whole_model_s()
    add(r"faster than zkLLM's code", "zkLLM's code on our L40S / our Llama-2-7B T2048 prover",
        one(V(zk7 / ours7.value, ours7.state), x))
    add(r"this gives about", "zkLLM's code on our L40S (s); / our prover; our prover",
        [Tok([(f"{zk7:.0f}", FINAL)], "s"), one(V(zk7 / ours7.value, ours7.state), x), one(ours7, pa.t)])
    zrows = {row["model"]: float(row["per_layer_s"]) for row in csv.DictReader(open(pa.ZKLLM_L40S, encoding="utf-8"))}
    add(r"grows out of", "zkLLM demo time per layer, Llama-2-13B / Llama-2-7B",
        Tok([(x(zrows["llama2-13b"] / zrows["llama2-7b"]), FINAL)], "×"))
    slow = [one(V(1 / v[1].value, FINAL), x, "×") for k, v in q.items()
            if v[1].state == FINAL and v[1].value and v[1].value < 1]
    # the zkLLM rows while any is red: the range of ours/theirs over the red rows where ours is slower, as
    # Table 5 prints them (the basic GPU verifier's placeholders; a row with none, '--', is left out)
    zslow = [V(1 / v[1].value, PENDING) for k, v in q.items()
             if k[0] == "zkLLM" and v[1].state == PENDING and v[1].value and v[1].value < 1]
    if any(v[1].state == PENDING for k, v in q.items() if k[0] == "zkLLM"):
        zslow_tok = rng(zslow, lambda v: f"{v:.1f}", "×") if zslow else [Tok([(None, PENDING)], "×")]
        slow += zslow_tok
        add(r"the GPU verifier is", "Limitations: GPU verifier slower than zkLLM's (zkLLM rows, red)", zslow_tok)
    add(r"Our verifier is slower only than", "verifier: final rows where ours is slower (ours/theirs); zkLLM rows",
        slow)
    # Sec. 5.4: the streaming verifier on the basic proof format (the stored _gpuv_stream cells), Llama-2-13B
    l13 = pa.llm_cost(("llama2-13b", 2048), {"mode": "C", "variant": "_gpuv_stream"}, pa.opt_dirs())
    if l13 and stream["C"]:
        rss = pa.opt_measured().idx.get(("llm", "llama2-13b", "defence_C_int_lam128_T2048_L40_gpuv_stream",
                                         "host_peak_rss", "", "", "", "", ""))
        add(r"On Llama-2-13B the same streaming verifier", "basic proof format, streaming GPU verifier: Llama-2-13B;"
            " / Llama-2-7B's; params ratio; host memory (GiB)",
            [one(fixed(l13[1]), pa.t), one(fixed(l13[1] / stream["C"][1]), x, "×"),
             one(fixed(l13[3] / stream["C"][3]), x, "×"),
             Tok([(f"{pa._f(rss['median']) / 2 ** 30:.0f}" if rss else None, FINAL)], "GiB")])
    mav = [cells for s, w, cells, _, _ in rr if s.startswith("Maverick")][0]
    snippet = [r for r in pa._read("reported_curated.csv") if r["system"] == pa.MAVERICK][0]["snippet"].split()
    mav_ours = elem(lambda s: s.llm("qwen3-4b", 8, "verify", "Kpre"))
    # the part of ours that decodes the compact encoding (verify_decode of the same cell)
    mav_spec = pa.OPTIMISED_SPECIAL[("qwen3-4b", 8, "Kpre")]
    mav_row = pa.llm_rows("Kpre", lam=mav_spec["lam"], threads=mav_spec["threads"], tables=pa.opt_dirs(),
                          tags=mav_spec["tags"]).get(("qwen3-4b", 8), {}) if pa.opt_dirs() else {}
    dec = mav_row.get("verify_decode", (None,))[0]
    add(r"Maverick's 124\.5|Maverick's .*is its", "Maverick's verifier, matrix checks, replay; client total; ours; "
        "of it decoding",
        [Tok([(f"{mav[1][0] * 1e3:.1f}", FINAL)], "ms"), Tok([(f"{(mav[1][0] - pa.MAVERICK_NONLINEAR_S) * 1e3:.1f}", FINAL)], "ms"),
         Tok([(f"{pa.MAVERICK_NONLINEAR_S * 1e3:.1f}", FINAL)], "ms"), Tok([(snippet[3], FINAL)], "ms"),
         Tok([(f"{mav_ours.value * 1e3:.1f}", mav_ours.state)], "ms"),
         Tok([(f"{dec * 1e3:.1f}" if dec is not None else None, mav_ours.state if dec is not None else PENDING)], "ms")])
    nw = pa.llm_cost(("qwen3-4b", 8), dict(mode="Kpre", lam=40, threads="1", tags=("thr1", "prune", "lookups")),
                     pa.opt_dirs())
    if nw:
        add(r"Without the compact encoding our Qwen3-4B", "ours without the compact encoding vs Maverick: prover, verifier, proof",
            [Tok([(f"{nw[0] * 1e3:.0f}", FINAL)]), Tok([(f"{mav[0][0] * 1e3:.1f}", FINAL)], "ms"),
             Tok([(f"{nw[1] * 1e3:.1f}", FINAL)]), Tok([(f"{mav[1][0] * 1e3:.1f}", FINAL)], "ms"),
             Tok([(f"{nw[2] / 1e6:.2f}", FINAL)]), Tok([(f"{mav[2][0] / 1e6:.2f}", FINAL)], "MB")])
    trip = {system(s): (cells, pend) for s, w, cells, _, pend in rr
            if (system(s), w) in (("zkCNN", "LeNet-5"), ("DeepProve", "GPT-2 (64)"))}
    toks = []
    for s, (cells, pend) in trip.items():
        for i, ((a, o), p) in enumerate(zip(cells, pend)):
            if s == "DeepProve" and i < 2:   # the text prints 91 and 105 ms, 34.2 and 1.35 s
                toks.append(Tok([(f"{o * 1e3:.0f}", st(p))], "ms"))
                toks.append(Tok([(f"{a:g}" if a >= 1 else f"{a:.2f}", FINAL)], "s"))
                continue
            fmt = pa.b if i == 2 else pa.t
            toks += [one(V(o, st(p)), fmt), one(fixed(a), fmt)]
    add(r"costs: 5\.4|wins on all three\s*$|optimised protocol wins on all three", "ours vs zkCNN LeNet-5, DeepProve GPT-2 (64): FS",
        toks)
    lose = [v[2] for (s, w), v in q.items() if v[2].value and v[2].value < 1 and system(s) != "zkLLM"]
    zkl = [v[2] for (s, w), v in q.items() if s == "zkLLM"]
    add(r"Elsewhere our proof is larger", "proof larger: other short-input rows; zkLLM rows",
        [*rng([V(1 / v.value, v.state) for v in lose], x2), *rng([V(1 / v.value, v.state) for v in zkl], x2)])
    zp = [_f for r in pa._read("reported_curated.csv") if r["system"] == "zkLLM" and r["model"] in ("OPT-6.7B", "Llama-2-7B")
          and r["seq"] == "2048" for _f in [pa._f(r["proof_bytes"])]]
    add(r"against zkLLM's 157", "zkLLM's proofs for the 7B models", rng([fixed(v) for v in zp], pa.b))
    cpu2048 = [elem(lambda s, m=m: s.llm(m, 2048, "verify")) for m, sq in pa.TAB_LLM_PICK if sq == 2048]
    add(r"CPU verifier needs up to", "largest CPU verifier at 2,048 tokens (Table 3)",
        one(max(cpu2048, key=lambda v: v.value), pa.t))
    add(r"optimised proof is far smaller than the model|far smaller than the model", "Llama-2-7B T64: int8 weights / proof",
        one(elem(l64), x, "×"))
    return out


def opt_measured_cost(model, cell):
    c = pa.opt_measured().cost(model, cell)
    return c[0]


def run(tex: Path, opt_names, listing=False) -> int:
    M = pa.basic_measured()
    pa.PENDING.clear()
    pa.DRY = True
    pa.build_all(figures=False)   # fills the pending registry the way the tables see it (nothing is written)
    found = checks(M, opt_names)
    raw = tex.read_text(encoding="utf-8").splitlines()
    # comment-only lines (the % PENDING notes) are dropped, so a sentence's window is its own text
    kept = [(n, expand_macros(strip_comment(l))) for n, l in enumerate(raw, 1) if strip_comment(l).strip()]
    numbers, lines = [n for n, _ in kept], [l for _, l in kept]
    bad = 0
    print(f"numbers from {pa.TABLES.name}/ and {[d.name for d in pa.opt_dirs()]} against {tex.name}"
          f" (⟦x⟧: inside \\pending{{}})")
    for anchor, what, toks in found:
        optional = anchor.startswith("?")
        anchor = anchor.lstrip("?")
        print(f"- {what}\n    computed: " + "; ".join(t.show() if t.segs[0][0] is not None else "⟦?⟧ (no placeholder)"
                                                    for t in toks))
        if listing:
            continue
        # the anchor may run over a line break: search each line joined with the next
        idx = [i for i in range(len(lines)) if re.search(anchor, " ".join(l.strip() for l in lines[i:i + 2]))]
        if not idx:
            bad += 0 if optional else 1
            print(f"    {'absent' if optional else 'ANCHOR':9s}not found: {anchor}"
                  + (" (optional: the sentence may have been cut)" if optional else ""))
            continue
        i = idx[0]
        window = normalise(" ".join(lines[i:i + 4]))
        res = [(t, *judge(t, window)) for t in toks]
        failed = [(t, s, d) for t, s, d in res if s in ("MISMATCH", "UNWRAP", "UNMARKED")]
        bad += len(failed)
        print(f"    l.{numbers[i]}: {window[:220]}")
        for t, s, d in res:
            if s != "ok":
                print(f"    {s:9s}{d}")
        if not any(s != "ok" for _, s, _ in res):
            print("    ok")
    if bad:
        print(f"{bad} failed checks")
    return 1 if bad else 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=pa.PLATFORM, help="the basic protocol's run (default: $PVI_PLATFORM, else l40s)")
    ap.add_argument("--optimised", default="", help="the optimised protocol's runs, comma-separated, as for paper_assets.py")
    ap.add_argument("--tex", type=Path, default=MAIN, help="the text to check (default: report/main.tex)")
    ap.add_argument("--list", action="store_true", help="only print the recomputed values")
    ap.add_argument("--require-tag", action="append", default=[], metavar="TAG",
                    help="count the optimised runs' cells carrying this tag only (as count_outcomes.py --require-tag)")
    ap.add_argument("--definition", action="store_true",
                    help="count the optimised runs' definition cells only (as count_outcomes.py --definition)")
    args = ap.parse_args()
    pa.TABLES = pa.tables_dir(args.platform)
    names = [s.strip() for s in args.optimised.split(",") if s.strip()]
    pa.OPT_TABLES = [d for d in (pa.tables_dir(n, required=False) for n in names) if d is not None]
    COUNT_TAGS[:] = args.require_tag
    COUNT_DEFINITION[:] = [True] if args.definition else []
    gaps = pa.missing()
    if gaps:
        raise SystemExit(f"{len(gaps)} missing\n  " + "\n  ".join(gaps))
    raise SystemExit(run(args.tex, names, args.list))


if __name__ == "__main__":
    main()
