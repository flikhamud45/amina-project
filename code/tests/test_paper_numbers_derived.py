"""``experiments/paper``: the numbers that follow from formulas, from stored literature runs and from summaries of the
run's records, on the synthetic run of ``test_paper_numbers.py``.

* The in-range attack's bounds skip a layer without a successful edit, so that one such layer leaves the rest of the
  attack family intact.
* ``tab:paths`` and ``tab:paths-acc`` take their edit sizes from the run's in-range documents (the stored ones give
  the fallback), and their header names those sizes by the same macros.
* The spot-checking schemes' values of ``tab:sok`` and ``tab:sok-derivation`` follow from the formulas the paper
  states (``attack/sok.py``), equal the fallback, and a run's path count that differs from the formula's fails the
  build.
* zkLLM's stored runs on the run's GPU model give same-GPU prover factors, and none on another GPU; the run's own
  zkllm tier (logs/<name>/zkllm/) takes their place when it holds both models on the run's GPU, and the text then
  says so (``ZkllmSameGpuRun``); at the tier's other prompt lengths a length that zkLLM's demo did not prove prints
  as failed, and ``tab:zkllm-lengths`` agrees with the macros.
* ``tab:eval-compare``'s rows are chosen by ``numbers.py --compare-rows``; ``tab:eval-compare-full`` keeps every row.
* The tampered queries of the real tier's and generation cells, V1's verifier with GPU tensors, the timed runs
  of ``tab:options`` with their spread and V1's bound on the measured builds alone (not the gate's or the smoke's).
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest
from test_paper_numbers import PAPER, RESULTS, _args, _bench, _ctx, _logs, _write, gen, perf, reexec, v1

from experiments.attack import sok as S
from experiments.paper import bench as B
from experiments.paper import macros as M
from experiments.paper import numbers as N
from experiments.paper import records as R
from experiments.paper import tables as T

FALLBACK = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (PAPER / "fallback" / "numbers.tex").read_text(
    encoding="utf-8")))


@pytest.fixture
def run_dir(tmp_path):
    _bench(tmp_path)
    return tmp_path, _logs(tmp_path)


def _macros(ctx):
    problems = []
    m, _ = M.build(ctx, problems)
    return {k: x.value for k, x in m.items()}, problems


def _fallback_rows(name: str) -> list[str]:
    return [r.strip() for r in (PAPER / "fallback" / f"{name}.tex").read_text(encoding="utf-8").splitlines()[1:]]


def _inrange_doc(src: Path, dst: Path, edit) -> None:
    """A copy of a stored ``inrange_llm.py`` document with ``edit(record)`` applied to each of its records."""
    d = json.loads(src.read_text(encoding="utf-8"))
    for r in d["records"]:
        edit(r)
    dst.write_text(json.dumps(d), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ in-range edits
def test_a_layer_without_a_successful_edit_leaves_the_attack_family_intact(run_dir):
    """A layer and range where no in-range edit succeeded has no median: the bounds over the cells skip it instead
    of failing the whole family (no ``PARSE`` problem, and the macros computed after it are filled)."""
    tmp, logs = run_dir
    att = logs / "attack"

    def fail(layer, rng):
        def edit(r):
            if r.get("layer") == layer and r.get("range") == rng:
                r["success"] = False
        return edit

    _inrange_doc(RESULTS / "inrange_llm_opt-1.3b_any.json", att / "inrange_llm_opt-1.3b_any.json", fail(23, "max"))
    _inrange_doc(RESULTS / "inrange_llm_opt-6.7b_any.json", att / "inrange_llm_opt-6.7b_any.json", fail(31, "p999"))
    ctx, problems = _ctx(tmp, logs)
    v, problems = _macros(ctx)
    assert not [p for p in problems if p.startswith("PARSE")]
    others = [T.inrange_cell(ctx.run.file("inrange_llm_opt-1.3b_any.json"), layer, rng)
              for layer in T.INRANGE_LAYERS for rng in ("max", "p999")]
    assert others[-2]["p_uniform"] is None and others[-2]["changed"] == 0
    assert v["InrangePMedianMax"] == M.ceil_sci(max(x["p_uniform"] for x in others if x["p_uniform"] is not None))
    assert v["InrangeChanged"] == "at least 0 of the 40"
    for name in ("InrangeOptLPMedianMax", "InrangeOptLKRange", "InrangePathsOptM", "InrangePathsOptL",
                 "InrangeDownKMedian", "InrangeKMax"):
        assert v[name] is not None, name
    assert M._medians([{"p": None}, {"p": 0.2}, {"p": 0.1}], "p") == [0.2, 0.1]
    assert M._medians([{"p": None}], "p") == [None]


# ------------------------------------------------------------------------------------------------ tab:paths
def test_the_path_tables_take_the_runs_edit_sizes(run_dir, tmp_path):
    """The stored in-range documents regenerate the fallback's ``tab:paths`` and ``tab:paths-acc`` (k = 1, 2, 6, 16,
    60); other edit sizes move the columns with ``\\InrangeKMedian``, ``\\InrangeDownKMedian`` and ``\\InrangeKMax``,
    and a run without the documents leaves those columns empty."""
    tmp, logs = run_dir
    ctx, _ = _ctx(tmp, logs)
    assert [r.strip() for r in T.tab_paths(ctx).rows] == _fallback_rows("tab_paths")
    assert [r.strip() for r in T.tab_paths_acc(ctx).rows] == _fallback_rows("tab_paths_acc")
    assert [k for _, k, _ in T.paths_columns(ctx.run)] == [1, 2, 6, 16, 60]
    # every successful edit of OPT-1.3B rewritten to 3 neurons (lowering only: 8), the largest any-direction one 41
    att = logs / "attack"
    first = {"done": False}

    def three(r):
        if r.get("success"):
            r["k"] = 41 if not first["done"] else 3
            first["done"] = True

    _inrange_doc(RESULTS / "inrange_llm_opt-1.3b_any.json", att / "inrange_llm_opt-1.3b_any.json", three)
    _inrange_doc(RESULTS / "inrange_llm_opt-1.3b_down.json", att / "inrange_llm_opt-1.3b_down.json",
                 lambda r: r.update(k=8) if r.get("success") else None)
    ctx, _ = _ctx(tmp, logs)
    v, _ = _macros(ctx)
    assert (v["InrangeKMedian"], v["InrangeDownKMedian"], v["InrangeKMax"]) == ("3", "8", "41")
    tb = T.tab_paths(ctx)
    cells = {(c["row"], c["column"]): c["value"] for c in tb.cells}
    assert cells["OPT-1.3B", "k median"] == S.paths_needed(40, 3, 8192)
    assert cells["OPT-6.7B", "k down"] == S.paths_needed(40, 8, 16384)
    assert cells["Llama-2-7B", "k max"] == S.paths_needed(40, 41, 11008)
    assert cells["Llama-2-7B", "k=1"] == 305_193 and cells["OPT-1.3B", "k=16"] == 14_182
    acc = {(c["row"], c["column"]): c["value"] for c in T.tab_paths_acc(ctx).cells}
    assert acc["Llama-2-7B s=10^3", "k median"] == pytest.approx(1000 * math.log1p(-3 / 11008) / math.log(10))
    # no in-range document: the measured columns are empty, the fixed ones are not
    empty = T.Ctx(R.Run([], []), B.Bench(None), {}, byte_model=False)
    tb = T.tab_paths(empty)
    assert len(tb.missing) == 3 * len(T.PATHS_MODELS) and tb.filled == 2 * len(T.PATHS_MODELS)
    assert tb.rows[0] == r"OPT-1.3B & 8{,}192 & 227{,}117 & -- & -- & 14{,}182 & -- \\"


def test_an_even_number_of_edits_gives_the_lower_median(monkeypatch):
    """The column of an even number of edits is an edit size that occurred (the lower median), never half a neuron."""
    sizes = {"any": [2, 3], "down": [4, 7]}      # each layer (and range) the same two edits: medians 2.5 and 5.5

    class Files:
        def file(self, name):
            return "down" if name.endswith("_down.json") else "any"

    monkeypatch.setattr(T, "inrange_cell", lambda doc, layer, rng: {"k_all": sizes[doc]})
    ks = T.inrange_ks(Files())
    assert (ks["median"], ks["down"], ks["max"]) == (2, 4, 3)


def test_the_acceptance_format():
    assert T.accept_text(math.log10(0.02551)) == "0.026"
    assert T.accept_text(math.log10(0.5797)) == "0.580"
    assert T.accept_text(math.log10(4.235e-3)) == r"$4.2\cdot10^{-3}$"
    assert T.accept_text(-63.17) == r"$6.8\cdot10^{-64}$"
    assert (T.accept_text(-237.4), T.accept_text(-159.3), T.accept_text(-1000.0)) == (
        r"$<10^{-200}$", r"$<10^{-150}$", r"$<10^{-1000}$")
    assert S.accept_log10(60, 11008, 1e5) == pytest.approx(-237.36, abs=0.01)


def test_the_paper_names_the_path_tables_edit_sizes_by_their_macros():
    """``tab:paths`` and ``tab:paths-acc`` read their bodies from the run, and their headers and caption print the
    edit sizes with the macros the text uses, never as literals."""
    tex = (PAPER / "sections" / "ext_proofs.tex").read_text(encoding="utf-8")
    heads = re.findall(r"Model & \$[Ns]\$ & (.*?) \\\\", tex)
    want = r"$k=1$ & $k=\InrangeKMedian{}$ & $k=\InrangeDownKMedian{}$ & $k=16$ & $k=\InrangeKMax{}$"
    assert heads == [want, want]
    assert r"\pvitable{tab_paths}" in tex and r"\pvitable{tab_paths_acc}" in tex
    tables = tex.split(r"\caption{Uniform paths needed")[1].split(r"\label{tab:paths-acc}")[0]
    assert "$k=2$" not in tables and "$60$" not in tables
    assert r"$k=\InrangeKMedian{}$, $\InrangeDownKMedian{}$" in tables


# ------------------------------------------------------------------------------------------------ tab:sok
def test_the_spot_checking_values_follow_from_the_stated_formulas():
    """``attack/sok.py`` gives the values of ``tab:sok-derivation`` from the parameters that the table states (the
    fallback's ``\\Sok...``), its command prints them, and the tables print the macros, not the numbers."""
    got = {r["name"]: r["printed"] for r in S.values()}
    assert got == {"PathOne": "99.80", "PathMany": "61.3", "PathsLenet": "2,316", "PathsWide": "14,182",
                   "PathsLlama": "305,193", "PathsOptL": "454,248", "SlpAudit": "98.14", "SlpAuditsHalf": "37",
                   "SlpAuditsMost": "160", "SlpTiny": "88.9", "CommitOpened": "87.5", "CommitRate": "99.375",
                   "CommitRateAbout": "99.4", "NanoAudit": "50"}
    c = M.Collector()
    M.sok(c, None)
    assert {k: x.value for k, x in c.m.items()} == {k: FALLBACK[k] for k in c.m}
    assert S.main(["--json"]) == 0
    derivation = (PAPER / "sections" / "ext_attack.tex").read_text(encoding="utf-8")
    for stated in (r"$N{=}512$, $s{=}1$", r"$N{=}512$, $s{=}250$", r"LeNet-5 ($N{=}84$)", r"($N{=}512$)",
                   r"($N{=}11{,}008$)", r"($N{=}16{,}384$)", r"$1-3/161$", r"\ln(158/161)", r"$1-5/45$", r"$1-10/80$",
                   r"$1-0.05\cdot10/80$"):
        assert stated in derivation, stated
    sok_table = (PAPER / "sections" / "attack.tex").read_text(encoding="utf-8")
    table = derivation.split(r"\label{tab:sok-derivation}")[1].split(r"\end{table*}")[0]
    for literal in ("98.14", "99.80", "61.3", "87.5", "99.4", "37 / 160", "14{,}182", "454{,}248"):
        assert literal not in sok_table and literal not in table


def test_a_path_count_that_differs_from_the_formula_fails_the_build(run_dir):
    """``tab:sok`` prints the formula's path counts as those of our models: the attack documents' and the analytic
    bound's counts must equal them (a text claim: a warning in a rehearsal)."""
    tmp, logs = run_dir
    ctx, _ = _ctx(tmp, logs)
    v, problems = _macros(ctx)
    assert v["AttackOptLPaths"] == v["SokPathsOptL"] and v["PathsLlama"] == v["SokPathsLlama"]
    assert not [p for p in problems if p.startswith(M.TEXT_CLAIM)]
    rows = list(csv.DictReader(open(logs / "attack" / "analytic.csv", encoding="utf-8")))
    rows[0]["anchuri_paths"] = "305192"
    with open(logs / "attack" / "analytic.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ctx, _ = _ctx(tmp, logs)
    _, problems = _macros(ctx)
    assert [p for p in problems if p.startswith(M.TEXT_CLAIM) and "PathsLlama" in p]
    assert N.demote_text_claims(problems)[0] == [p for p in problems if not p.startswith(M.TEXT_CLAIM)]


# ------------------------------------------------------------------------------------------------ zkLLM on one GPU
def _bench_int(root: Path, gpu: str = "NVIDIA L40S") -> Path:
    """A ``tables_<name>/`` with the interactive mode-C cells of Llama-2-7B and Llama-2-13B at 2,048 tokens (the prover
    10.0 s and 20.0 s), at the final run's 11-bit attention probabilities (``pmax2047``)."""
    t = root / "tables_int"
    t.mkdir(parents=True)
    head = ["model", "seq", "mode", "challenges", "lam", "metric", "threads", "rate", "variant", "device", "prover_hw",
            "verifier_hw", "n_layers_full", "params_full", "value", "provenance", "batch"]
    with open(t / "llm_full_model.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for m, layers, prove in (("llama2-7b", 32, 10.0), ("llama2-13b", 40, 20.0)):
            for metric, value in (("prove_forward", prove), ("verify_products", 1.0), ("bytes_total", 6e9)):
                w.writerow([m, 2048, "C", "int", 128, metric, 8, 4, "_wire_prune_pmax2047_polcnn18c", "cuda", gpu,
                            "AMD EPYC 9334 32-Core Processor x8 threads", layers, 7e9, value, "measured", ""])
    with open(t / "measured_summary.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["suite", "model", "cell", "metric", "batch", "attack", "lam", "stage", "k", "n",
                                 "median", "mean", "prover_hw", "verifier_hw"])
    return t


def test_zkllm_on_the_same_gpu(tmp_path):
    """zkLLM's stored runs (``artifacts/results/zkllm_l40s/``) are read with the published values; their prover over
    ours on the same prompts gives the same-GPU factors when the run's GPU is theirs, and nothing on another GPU."""
    lit = B.literature()
    row = lit[(B.ZKLLM_RUNS, "Llama-2-7B", "2048")]
    assert (row["prover_s"], row["prover_hw"], row["cpu"]) == ("843.84", "NVIDIA L40S",
                                                                "AMD EPYC 9334 32-Core Processor")
    assert lit[(B.ZKLLM_RUNS, "Llama-2-13B", "2048")]["prover_s"] == "2628.40"
    assert ("zkLLM", "Llama-2-13B", "2048") in lit                         # the published A100 row stays
    for gpu, want in (("NVIDIA L40S", ("844", "2{,}628", "84", "131", "84--131")),
                      ("NVIDIA GeForce RTX 2080 Ti", (None,) * 5)):
        bench = B.Bench(_bench_int(tmp_path / gpu.replace(" ", "_"), gpu))
        rec = R.Rec("cpuv", "c-x", "1", "x", 1, "perf", dict(perf("llama2-7b", 64, "Kpre", 1.0, 1.0),
                                                              env={"gpu": gpu, "cpu": "AMD EPYC 9334"}))
        ctx = T.Ctx(R.Run([rec]), bench, lit, "cnn18c", "x", byte_model=False)
        c = M.Collector()
        M.zkllm_same_gpu(c, ctx)
        assert tuple(c.m[k].value for k in ("ZkllmSameGpuProverLlama", "ZkllmSameGpuProverLlamaL",
                                            "ZkllmSameGpuFactorLlama", "ZkllmSameGpuFactorLlamaL",
                                            "ZkllmSameGpuFactorRange")) == want
        assert c.m["ZkllmSameGpuRun"].value == ("in separate runs" if want[0] else None)
    assert (FALLBACK["ZkllmSameGpuProverLlama"], FALLBACK["ZkllmSameGpuProverLlamaL"]) == ("844", "2{,}628")
    assert B.zkllm_runs(tmp_path / "none") == {}


def _zkllm_run(folder: Path, gpu: str, models=(("llama2-7b", 32, 28.125), ("llama2-13b", 40, 67.5))) -> Path:
    """A run's ``logs/<name>/zkllm/`` as the zkllm tier writes it (``summarise.py --csv``, each run's ``env.txt``)."""
    folder.mkdir(parents=True)
    rows = ["model,seq,layers,run,per_layer_s,whole_model_s"]
    for m, layers, per in models:
        rows.append(f"{m},2048,{layers},{m}-T2048,{per:.4f},{per * layers:.2f}")
        (folder / f"{m}-T2048").mkdir()
        (folder / f"{m}-T2048" / "env.txt").write_text(
            f"n-1\n{gpu}, 580.126.09, 46068 MiB\nModel name:    AMD EPYC 9334 32-Core Processor\n"
            "993311ea8e2346b90efacb35337b2e3cfe0d6f8d\narch: sm_89\n", encoding="utf-8")
    (folder / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


def _zkllm_record(model: str, ok: bool, job: str = "60") -> R.Rec:
    """The record a zkllm tier's prove job writes (``summarise.py --record``): ``ok`` when every proof binary exited 0
    and every proved layer printed each verifier message."""
    d = {"kind": "zkllm", "model": model, "seq": 2048, "layers": 32 if model == "llama2-7b" else 40, "reps": 3,
         "layer_ids": [0, 1], "layers_expected": 6, "layers_proved": 6 if ok else 5, "binary_failures": 0 if ok else 1,
         "ok": ok, "env": {"gpu": "NVIDIA L40S", "cpu": "AMD EPYC 9334 32-Core Processor"}}
    return R.Rec("zkllm", f"zk-{model}", job, f"sp-zk-{model}-{job}.out", 1, "zkllm", d)


def test_zkllm_on_the_same_gpu_from_the_run_itself(tmp_path):
    """The run's own zkllm tier (logs/<name>/zkllm/summary.csv) gives the same-GPU values when it holds both models on
    the run's GPU, each with an ok record of its prove job ("in the same run"); on another GPU, with one model only,
    or with a prove job whose record is not ok (a proof binary failed: its times are not zkLLM's) or missing, the
    stored runs do ("in separate runs")."""
    stored = ("844", "2{,}628", "84", "131", "84--131", "in separate runs")
    both = (("llama2-7b", True, "60"), ("llama2-13b", True, "60"))
    cases = (("NVIDIA L40S", None, both, ("900", "2{,}700", "90", "135", "90--135", "in the same run")),
             ("NVIDIA A100-SXM4-80GB", None, both, stored),
             ("NVIDIA L40S", (("llama2-7b", 32, 28.125),), both, stored),
             ("NVIDIA L40S", None, (("llama2-7b", True, "60"), ("llama2-13b", False, "60")), stored),
             ("NVIDIA L40S", None, (("llama2-7b", True, "60"),), stored),
             # the tier run again: the later prove job's ok record replaces the failed one
             ("NVIDIA L40S", None, (("llama2-7b", True, "60"), ("llama2-13b", False, "60"), ("llama2-13b", True, "61")),
              ("900", "2{,}700", "90", "135", "90--135", "in the same run")))
    for i, (zk_gpu, models, oks, want) in enumerate(cases):
        folder = _zkllm_run(tmp_path / f"logs{i}" / "zkllm", zk_gpu, *(m for m in (models,) if m))
        lit = B.literature(run_zkllm=folder)
        assert lit[(B.ZKLLM_RUN, "Llama-2-7B", "2048")]["prover_hw"] == zk_gpu
        assert lit[(B.ZKLLM_RUN, "Llama-2-7B", "2048")]["source"].startswith(f"logs/logs{i}/zkllm/summary.csv")
        bench = B.Bench(_bench_int(tmp_path / f"bench{i}", "NVIDIA L40S"))
        rec = R.Rec("cpuv", "c-x", "1", "x", 1, "perf", dict(perf("llama2-7b", 64, "Kpre", 1.0, 1.0),
                                                              env={"gpu": "NVIDIA L40S", "cpu": "AMD EPYC 9334"}))
        zk = [_zkllm_record(m, ok, job) for m, ok, job in oks]
        ctx = T.Ctx(R.Run([rec, *zk]), bench, lit, "cnn18c", "x", byte_model=False)
        c = M.Collector()
        M.zkllm_same_gpu(c, ctx)
        got = tuple(c.m[k].value for k in ("ZkllmSameGpuProverLlama", "ZkllmSameGpuProverLlamaL",
                                           "ZkllmSameGpuFactorLlama", "ZkllmSameGpuFactorLlamaL",
                                           "ZkllmSameGpuFactorRange", "ZkllmSameGpuRun"))
        assert got == want, (zk_gpu, oks, got)
    assert B.literature(run_zkllm=tmp_path / "no-run").keys() == B.literature().keys()
    assert FALLBACK["ZkllmSameGpuRun"] == "in separate runs"


# --------------------------------------------------------------------------------------------- zkLLM at several lengths
def _bench_lengths(root: Path, seqs=(64, 512, 1024, 2048), skip=()) -> Path:
    """A ``tables_<name>/`` with the interactive mode-C cells of Llama-2-7B and Llama-2-13B at ``seqs`` (the prover
    ``seq / 100`` s and ``seq / 50`` s; none at the ``(model, seq)`` of ``skip``) on the L40S."""
    t = root / "tables_len"
    t.mkdir(parents=True)
    head = ["model", "seq", "mode", "challenges", "lam", "metric", "threads", "rate", "variant", "device", "prover_hw",
            "verifier_hw", "n_layers_full", "params_full", "value", "provenance", "batch"]
    with open(t / "llm_full_model.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for m, layers, per in (("llama2-7b", 32, 100.0), ("llama2-13b", 40, 50.0)):
            for seq in seqs:
                if (m, seq) in skip:
                    continue
                for metric, value in (("prove_forward", seq / per), ("verify_products", 1.0), ("bytes_total", 6e9)):
                    w.writerow([m, seq, "C", "int", 128, metric, 8, 4, "_wire_prune_pmax2047_polcnn18c", "cuda",
                                "NVIDIA L40S", "AMD EPYC 9334 32-Core Processor x8 threads", layers, 7e9, value,
                                "measured", ""])
    with open(t / "measured_summary.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["suite", "model", "cell", "metric", "batch", "attack", "lam", "stage", "k", "n",
                                 "median", "mean", "prover_hw", "verifier_hw"])
    return t


ZK_LENGTHS = {   # (model, seq): (zkLLM's whole-model seconds, ok)
    ("llama2-7b", 2048): (900.0, True), ("llama2-13b", 2048): (2700.0, True),
    ("llama2-7b", 1024): (400.0, True), ("llama2-13b", 1024): (1200.0, True),
    ("llama2-7b", 512): (10.0, False), ("llama2-13b", 512): (500.0, True),
    ("llama2-7b", 64): (1.0, False), ("llama2-13b", 64): (2.0, False)}


def _zkllm_lengths(folder: Path, runs=ZK_LENGTHS) -> list:
    """The run's ``logs/<name>/zkllm/`` and the records of the zkllm tier's jobs at several lengths: the 2,048-token
    jobs (required) and the short jobs (``--allow-failed-lengths``: required false)."""
    folder.mkdir(parents=True)
    rows, recs = ["model,seq,layers,run,per_layer_s,whole_model_s,ok"], []
    for (m, seq), (whole, ok) in runs.items():
        layers = 32 if m == "llama2-7b" else 40
        run = f"{m}-T{seq}"
        rows.append(f"{m},{seq},{layers},{run},{whole / layers:.4f},{whole:.2f},{str(ok).lower()}")
        (folder / run).mkdir()
        (folder / run / "env.txt").write_text(
            "n-1\nNVIDIA L40S, 580.126.09, 46068 MiB\nModel name:    AMD EPYC 9334 32-Core Processor\n"
            "993311ea8e2346b90efacb35337b2e3cfe0d6f8d\narch: sm_89\n", encoding="utf-8")
        job = f"zk-{m}" + ("" if seq == 2048 else "-short")
        d = {"kind": "zkllm", "model": m, "seq": seq, "layers": layers, "reps": 3, "layer_ids": [0, 1],
             "layers_expected": 6, "layers_proved": 6 if ok else 2, "binary_failures": 0 if ok else 2, "ok": ok,
             "required": seq == 2048, "errors": [] if ok else ["2 what(): N is not power of 2, or D is not "
                                                                "divisible by N"],
             "env": {"gpu": "NVIDIA L40S", "cpu": "AMD EPYC 9334 32-Core Processor"}}
        recs.append(R.Rec("zkllm", job, "70", f"sp-{job}-70.out", 1, "zkllm", d))
    (folder / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return recs


def test_zkllm_at_several_prompt_lengths(tmp_path):
    """The zkllm tier's runs at 64, 512, 1,024 and 2,048 tokens against our interactive mode-C cells at the same
    lengths: a length whose record is not ok prints ``fails`` (no factor) and is no failed check when its job allowed
    it; the factors over every proved length, the shortest length below 2,048 tokens that zkLLM proved for both models
    and the factors there; ``tab:zkllm-lengths`` prints the same values; a shorter length that zkLLM proved for both
    models without our cell makes the text's "the shortest length that it proves for both models" a failed check."""
    recs = _zkllm_lengths(tmp_path / "logs" / "zkllm")
    assert not [m for r in recs for m in R.checks(r)]                     # the failed short lengths are results
    lit = B.literature(run_zkllm=tmp_path / "logs" / "zkllm")
    ctx = T.Ctx(R.Run(recs), B.Bench(_bench_lengths(tmp_path / "b")), lit, "cnn18c", "x", byte_model=False)
    c = M.Collector()
    M.zkllm_same_gpu(c, ctx)
    v = {k: x.value for k, x in c.m.items()}
    # ours: 7B seq/100 s, 13B seq/50 s; zkLLM's: ZK_LENGTHS
    assert (v["ZkllmSameGpuProverLlama"], v["ZkllmSameGpuFactorLlama"], v["ZkllmSameGpuFactorLlamaL"]) == \
        ("900", "44", "66")
    assert v["ZkllmSameGpuRun"] == "in the same run" and v["ZkllmSameGpuOursLlamaL"] == "41"
    assert (v["ZkllmSameGpuProverLlamaTOneZeroTwoFour"], v["ZkllmSameGpuFactorLlamaTOneZeroTwoFour"],
            v["ZkllmSameGpuFactorLlamaLTOneZeroTwoFour"]) == ("400", "39", "59")
    assert (v["ZkllmSameGpuProverLlamaTFiveOneTwo"], v["ZkllmSameGpuFactorLlamaTFiveOneTwo"]) == ("fails", "--")
    assert (v["ZkllmSameGpuProverLlamaLTFiveOneTwo"], v["ZkllmSameGpuFactorLlamaLTFiveOneTwo"]) == ("500", "49")
    assert v["ZkllmSameGpuProverLlamaLTSixFour"] == "fails" and v["ZkllmSameGpuOursLlamaTSixFour"] == "0.64"
    assert v["ZkllmSameGpuShortSeq"] == "1{,}024" and v["ZkllmSameGpuFactorShortRange"] == "39--59"
    assert v["ZkllmSameGpuFactorLengthRange"] == "39--66" and not c.problems
    tb = T.tab_zkllm_lengths(ctx)
    assert tb.rows[0].startswith(r"64 & fails & 640\,ms & -- & fails & 1.3\,s & -- \\")
    assert tb.rows[1] == r"512 & fails & 5.1\,s & -- & 8.3\,min & 10.2\,s & 49$\times$ \\"
    assert tb.rows[3] == r"2{,}048 & 15.0\,min & 20.5\,s & 44$\times$ & 45.0\,min & 41.0\,s & 66$\times$ \\"
    assert not tb.missing
    texts, values = {"tab_zkllm_lengths": tb.text()}, {k: x.value for k, x in c.m.items() if x.value is not None}
    assert N.same_value(texts, values) == []
    # without our 512-token cells, 1,024 tokens is no longer the shortest length zkLLM proved for both models (512,
    # had its 7B run succeeded): a failed check of the text's statement
    runs = {**ZK_LENGTHS, ("llama2-7b", 512): (250.0, True)}
    recs = _zkllm_lengths(tmp_path / "logs2" / "zkllm", runs)
    bench = B.Bench(_bench_lengths(tmp_path / "b2", skip=(("llama2-7b", 512), ("llama2-13b", 512))))
    ctx = T.Ctx(R.Run(recs), bench, B.literature(run_zkllm=tmp_path / "logs2" / "zkllm"), "cnn18c", "x",
                byte_model=False)
    c = M.Collector()
    M.zkllm_same_gpu(c, ctx)
    assert c.m["ZkllmSameGpuShortSeq"].value == "1{,}024"
    assert [p for p in c.problems if p.startswith(M.TEXT_CLAIM) and "ZkllmSameGpuShortSeq" in p]
    # without the tier's short lengths (the stored runs only): the 2,048-token values, and no other length
    ctx = T.Ctx(R.Run([]), B.Bench(_bench_lengths(tmp_path / "b3")), B.literature(), "cnn18c", "x", byte_model=False)
    c = M.Collector()
    M.zkllm_same_gpu(c, ctx)
    assert c.m["ZkllmSameGpuProverLlama"].value == "844" and c.m["ZkllmSameGpuRun"].value == "in separate runs"
    assert c.m["ZkllmSameGpuProverLlamaTSixFour"].value is None and c.m["ZkllmSameGpuShortSeq"].value is None
    assert T.tab_zkllm_lengths(ctx).missing


def test_the_comparison_rows_are_chosen_without_code_edits(run_dir, tmp_path):
    """``numbers.py --compare-rows``: ``tab:eval-compare`` prints the chosen rows (in the tables' order, with a rule
    between groups) and ``tab:eval-compare-full`` every row, whichever are chosen, with the same cells; the text's
    comparison numbers follow the chosen rows; a file of keys works as a list; an unknown key is refused."""
    assert T.compare_selection(None) == T.compare_selection("default") == T.COMPARE_DEFAULT
    assert T.compare_selection("all") == T.COMPARE_KEYS and len(T.COMPARE_KEYS) == len(set(T.COMPARE_KEYS)) == 16
    assert set(T.COMPARE_DEFAULT) < set(T.COMPARE_KEYS)
    assert T.compare_selection("zkllm-llama2-13b, zkcnn-vgg16") == ("zkcnn-vgg16", "zkllm-llama2-13b")
    keys = tmp_path / "rows.txt"
    keys.write_text("# the body's rows\nzkcnn-lenet5\nmaverick-qwen3-4b-8  # Maverick\n", encoding="utf-8")
    assert T.compare_selection(str(keys)) == ("zkcnn-lenet5", "maverick-qwen3-4b-8")
    with pytest.raises(ValueError, match="unknown rows.*zkllm-gpt5"):
        T.compare_selection("zkllm-gpt5")
    # the fallback's body rows are the default rows, and the full table's rows hold them
    body, full = _fallback_rows("tab_eval_compare"), _fallback_rows("tab_eval_compare_full")
    assert sum(r.endswith("\\\\") and r != r"\midrule" for r in body) == len(T.COMPARE_DEFAULT)
    assert sum(r.endswith("\\\\") for r in full) == len(T.COMPARE_KEYS) and set(body) <= set(full)
    tmp, logs = run_dir
    ctx, _ = _ctx(tmp, logs)
    default, every = T.tab_eval_compare(ctx), T.tab_eval_compare_full(ctx)
    assert len([r for r in default.rows if r != r"\midrule"]) == len(T.COMPARE_DEFAULT)
    assert len([r for r in every.rows if r != r"\midrule"]) == len(T.COMPARE_KEYS)
    assert every.rows.count(r"\midrule") == 2 and default.rows.count(r"\midrule") == 2
    ctx.compare_rows = ("zkcnn-vgg16", "zkllm-llama2-7b", "zkllm-llama2-13b")
    some = T.tab_eval_compare(ctx)
    assert [r.split(" & ")[0] for r in some.rows] == ["zkCNN$^a$", r"\midrule", "zkLLM", "zkLLM"]
    assert set(some.rows) - {r"\midrule"} <= set(every.rows)
    assert {(x["row"], x["column"]) for x in some.cells} < {(x["row"], x["column"]) for x in every.cells}
    # the command: the chosen rows' table, the full table and cells.csv with every row
    assert N.main(_args(tmp, logs, "--allow-missing", "--compare-rows", "zkcnn-vgg16,zkllm-llama2-13b")) in (0, 1)
    out = tmp / "out"
    body = [r for r in (out / "tab_eval_compare.tex").read_text(encoding="utf-8").splitlines()
            if r and not r.startswith("%")]
    assert len(body) == 3 and body[1] == r"\midrule"
    assert "rows (numbers.py --compare-rows): zkcnn-vgg16,zkllm-llama2-13b" in \
        (out / "tab_eval_compare.tex").read_text(encoding="utf-8")
    cells = list(csv.DictReader(open(out / "cells.csv", encoding="utf-8")))
    assert {r["row"] for r in cells if r["table"] == "tab:eval-compare-full"} == set(T.COMPARE_KEYS)
    with pytest.raises(SystemExit):
        N.main(_args(tmp, logs, "--compare-rows", "no-such-row"))


# ------------------------------------------------------------------------------------------------ the run's records
def test_the_tampered_queries_and_the_setup_proofs(run_dir):
    """The real tier's and generation cells' tampered queries (perf.py and V1 per kind, gen_bench.py per cell) and
    the setup tier's honest proofs; an accepted tamper or a rejected setup proof leaves its count empty."""
    tmp, logs = run_dir
    tam = {"trace": {"attempts": 3, "rejected": 3, "labels": {}}, "single": {"attempts": 3, "rejected": 3}}
    _write(logs, "sp-r-opt67-v1-2048-60.out", [dict(v1("opt-6.7b", 2048, "Kpre", 32, 1e10, 6e9, label="real",
                                                       real="facebook/opt-6.7b"), tampers=tam)])
    ctx, _ = _ctx(tmp, logs)
    v, _ = _macros(ctx)
    # perf.py: 9 (the real Llama-2-7B); gen_bench.py: 19 responses of the gen tier and 1 of the real tier; V1: 6
    assert (v["OutcomeTampers"], v["OutcomeTampersVOne"], v["OutcomeTampersGen"], v["OutcomeSetupProofs"]) == \
        ("35", "6", "20", "1")
    bad = dict(gen("llama2-7b", "C", "greedy", 64, 1.0), tampers={"wrong_token": {"rejected": False}}, ok=False)
    _write(logs, "sp-gen-llama2-7b-61.out", [bad])
    _write(logs, "sp-setup-gpt2-62.out", [{"kind": "setup", "label": "setup", "model": "gpt2", "layers": None,
                                           "amortise": 100, "accepted": False, "env": {}}])
    ctx, _ = _ctx(tmp, logs)
    v, _ = _macros(ctx)
    assert (v["OutcomeTampers"], v["OutcomeTampersGen"], v["OutcomeSetupProofs"]) == (None, None, None)
    assert v["OutcomeTampersVOne"] == "6"
    assert M._tampered(R.Rec("gen", "g", "1", "x", 1, "gen", bad)) == (1, 0)


def test_v1_with_the_verifier_tensors_on_the_gpu(run_dir):
    """The gpuv tier's V1 cells (label gpuv-v1, the verifier on cuda, the full model): V1's optimised verifier and
    v0's on the same queries; a build of fewer blocks fills nothing."""
    tmp, logs = run_dir
    _write(logs, "sp-g-v1-opt-1.3b-63.out", [v1("opt-1.3b", 2048, "Kpre", 2, 1e9, 5e8, label="gpuv-v1",
                                                argv=["--verifier-device", "cuda"])])
    ctx, _ = _ctx(tmp, logs)
    v, _ = _macros(ctx)
    assert (v["VOneGpuVerifyLlama"], v["VOneGpuVZeroVerifyLlama"]) == ("1.1", "5.1")
    assert (v["VOneGpuVerifyOptM"], v["VOneGpuVZeroVerifyOptM"]) == (None, None)
    assert v["VOneVerifyLlama"] == "45.4"                  # the CPU verifier's cell is another record


def test_the_timed_runs_of_the_options_table(run_dir):
    """``tab:options``' cells: their timed runs (5 for perf.py, reexec.py and llamacpp.py, 3 for V1's) and their
    largest relative min--max spread, rounded up; a cell whose record has no range leaves the spread empty."""
    tmp, logs = run_dir
    ctx, _ = _ctx(tmp, logs)
    cells = M.options_timings(ctx.run)
    assert len(cells) == 4 + 2 + 2 + 2 + 1     # Kpre and C at 64 and 2,048; CPU and GPU re-execution; llama.cpp; V1
    assert {r.kind for _, _, r in cells} >= {"llamacpp"}
    v, _ = _macros(ctx)
    assert (v["OptionsRuns"], v["OptionsSpread"]) == ("3--5", "4")    # llama.cpp: (1.02 - 0.98) / 1.0
    wide = dict(perf("llama2-7b", 2048, "Kpre", 19.6, 6.43e9), verify_s_range=[18.6, 20.6])
    _write(logs, "sp-c-llama2-7b-2048-64.out", [wide])
    ctx, _ = _ctx(tmp, logs)
    v, _ = _macros(ctx)
    assert v["OptionsSpread"] == "11"                    # 2.0 / 19.6 = 10.2%
    rx = reexec("llama2-7b", 64, "int_fp32gemm", 4.0)          # the fastest exact cell at 64 tokens
    del rx["range_s"]
    _write(logs, "sp-c-rx-llama2-7b-65.out", [rx])
    ctx, _ = _ctx(tmp, logs)
    v, _ = _macros(ctx)
    assert v["OptionsSpread"] is None and v["OptionsRuns"] == "3--5"
    assert M._spread([1.0, 1.2], 1.1) == pytest.approx(0.2 / 1.1) and M._spread(None, 1.0) is None


def test_the_gate_and_the_smoke_fill_no_v1_bound(run_dir):
    """V1's multiplicities and GKR transcripts (``VOneExtrasMax``) are the measured builds': a larger gate or smoke
    proof leaves the bound unchanged."""
    tmp, logs = run_dir
    for name, label in (("sp-gate-v1-llama7-2k-2-66.out", "gate"), ("sp-smoke-67.out", "")):
        big = v1("llama2-7b", 2048, "Kpre", 2, 2.5e8, 1.1e8, label=label)
        big["v1_bytes"] = dict(big["v1_bytes"], cut_mult=5e6)
        _write(logs, name, [big])
    ctx, _ = _ctx(tmp, logs)
    assert {r.tier for r in ctx.run.kind("v1")} >= {"gate", "smoke"}
    v, _ = _macros(ctx)
    assert v["VOneExtrasMax"] == "0.91"
