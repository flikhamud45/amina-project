"""Plan 2 F6: the final-run package's harnesses and the package itself (``slurm/sp2027.sh`` v2).

* ``gen_bench.py`` (the gen tier) on a tiny decoder: every greedy and sampled response is accepted in modes C and
  Kpre under Fiat-Shamir, a wrong token is rejected by the token rule and logits forged to choose it by the LM head's
  check, and the line carries the timings with their ``[min, max]`` and the peaks;
* ``setup_timing.py`` (the setup tier): the honest setup proof is accepted, its bytes and parameters are recorded;
* ``reexec.py --max-pos`` refuses a model with learned positions (before building anything); ``h2d_bandwidth.py``
  measures on a GPU and exits 2 without one; ``perf.py``'s ``[min, max]`` are each query's own totals;
* the package's dry run (``PVI_DRYRUN=1``): every v0/V1/generation cell runs under Fiat-Shamir, the only interactive
  bench cells are the ones a table compares with an interactive system, every mode-C cell takes ``PVI_C_POLICY``,
  the headline cells time 5 queries, no course-report control or attack job is left, every job is chained after the
  previous one and the jobs after smoke wait for it, and the rehearsal submits small cells of every tier only.

The smoke tier runs this file on the final platform.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
SP = REPO / "code" / "experiments" / "4_defence_benchmark" / "slurm" / "sp2027.sh"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cli(mod, monkeypatch, capsys, script: str, *argv) -> tuple[int, list[dict]]:
    monkeypatch.setattr(sys, "argv", [script, *argv])
    threads = torch.get_num_threads()
    try:
        status = mod.main()
    finally:
        torch.set_num_threads(threads)
    return status, [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]


def test_gen_bench_greedy_and_sampled_responses_and_both_tamper_cells(monkeypatch, capsys):
    gb = _load("gen_bench", "experiments/6_improvements/gen_bench.py")
    status, lines = _cli(gb, monkeypatch, capsys, "gen_bench.py", "--model", "tiny-llama", "--prompt", "6",
                         "--gen", "3", "--decode", "greedy", "sample", "--modes", "C", "Kpre", "--policy", "auto",
                         "--fs", "--queries", "2", "--device", "cpu", "--threads", "1", "--sampler-a", "4096")
    assert status == 0
    assert {(r["mode"], r["decode"]) for r in lines} == {("C", "greedy"), ("C", "sample"), ("Kpre", "greedy"),
                                                         ("Kpre", "sample")}
    for r in lines:
        assert r["kind"] == "gen" and r["ok"] and r["accepted"] == 3 and r["positions"] == 6 + 3 - 1
        assert r["fs"] and r["gen"] == 3 and r["proof_bytes"] > 0
        assert r["prove_s_range"][0] <= r["prove_s"] <= r["prove_s_range"][1]
        assert r["verify_s_range"][0] <= r["verify_s"] <= r["verify_s_range"][1]
        assert r["tampers"]["wrong_token"]["rejected"] and r["tampers"]["wrong_token"]["label"] == "token"
        forged = r["tampers"]["forged_logits"]
        assert forged["rejected"] and forged["label"] in forged["expected"], forged
        assert "peak" in r and "build_peak" in r
    sampled = next(r for r in lines if r["decode"] == "sample")
    assert sampled["sampler"] == {"a": 4096, "top_k": 0}


def test_setup_timing_accepts_the_honest_setup_proof(monkeypatch, capsys):
    st = _load("setup_timing", "experiments/6_improvements/setup_timing.py")
    status, lines = _cli(st, monkeypatch, capsys, "setup_timing.py", "--model", "tiny-llama", "--policy", "auto",
                         "--amortise", "100", "1000", "--reps", "1", "--device", "cpu", "--threads", "1")
    assert status == 0 and [r["amortise"] for r in lines] == [100, 1000]
    for r in lines:
        assert r["kind"] == "setup" and r["accepted"] and r["proof_bytes"] > 0 and r["s"] >= 7
        assert r["query_columns_bytes_untrusted"] >= r["query_columns_bytes_honest"] > 0
        assert r["prove_s"] > 0 and r["verify_s"] > 0 and r["commit_s"] > 0


def test_reexec_refuses_a_longer_context_for_learned_positions(monkeypatch):
    rx = _load("reexec", "experiments/7_reexec/reexec.py")
    monkeypatch.setattr(sys, "argv", ["reexec.py", "--model", "gpt2", "--seq", "64", "--max-pos", "4096",
                                      "--device", "cpu", "--threads", "1"])
    with pytest.raises(SystemExit, match="learned positions"):
        rx.main()


def test_h2d_bandwidth(monkeypatch, capsys):
    h2d = _load("h2d_bandwidth", "experiments/6_improvements/h2d_bandwidth.py")
    if not torch.cuda.is_available():
        monkeypatch.setattr(sys, "argv", ["h2d_bandwidth.py"])
        assert h2d.main() == 2
        return
    status, lines = _cli(h2d, monkeypatch, capsys, "h2d_bandwidth.py", "--sizes-mb", "16", "--reps", "2")
    assert status == 0 and len(lines) == 4
    assert all(r["kind"] == "h2d" and r["gb_s"] > 0 for r in lines)


def test_perf_spread_is_each_querys_own_totals():
    perf = _load("perf", "experiments/6_improvements/perf.py")
    runs = [{"timings": {"prove_forward": 1.0, "fs_hash": 0.5, "verify_derive": 2.0}},
            {"timings": {"prove_forward": 3.0, "fs_hash": 0.5, "verify_derive": 1.0}}]
    assert perf._spread(runs) == {"prove_s_range": [1.5, 3.5], "verify_s_range": [1.5, 2.5]}


# ------------------------------------------------------------------------------------------ the package
def _dry(**env) -> list[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    # none of the caller's PVI_* switches (e.g. the smoke job's own environment), only the ones given here
    full = {**{k: v for k, v in os.environ.items() if not k.startswith("PVI_")}, "PVI_PLATFORM": "l40s_sp2027",
            "PVI_DRYRUN": "1", "HF_HOME": "/tmp/hf", "CXX": "g++", "PVI_OPTIONAL": "1", "PVI_LLAMA_LICENCE": "1",
            **env}
    out = subprocess.run([bash, str(SP.relative_to(REPO)), "all"], cwd=REPO, env=full, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return [line for line in out.stdout.splitlines() if line.startswith("sbatch")]


def _job(line: str) -> str:
    return re.search(r" -J (\S+)", line).group(1)


def test_the_package_runs_fiat_shamir_and_the_policy_everywhere():
    lines = _dry(PVI_C_POLICY="cnn18c")
    assert len(lines) > 100
    for line in lines:
        job = _job(line)
        for script in ("perf.py", "v1_bytes.py", "gen_bench.py"):
            if script in line:
                assert " --fs" in line, job
        # every mode-C cell of the Python harnesses (and the setup proof) takes PVI_C_POLICY
        if re.search(r"--modes [^-]*\bC\b", line) or "setup_timing.py" in line:
            assert "--policy cnn18c" in line, job
        if "bench.sbatch llm" in line:
            modes = re.search(r"--modes (\S+)", line).group(1)
            if ":int" in modes:     # interactive bench cells: zkLLM's rows, Maverick's, C4a's interactive byte model
                assert re.search(r"-(c|cg)-(opt-125m|opt-1\.3b|llama2-7b|llama2-13b)-2k$", job) \
                    or "thr1" in job or job.endswith("-cint"), job
            if re.search(r"\bC:", modes) and "--policy" in line:
                assert "--policy cnn18c" in line, job
        if "bench.sbatch cnn" in line and "--policy" in line:
            assert "--policy cnn18c" in line, job
        assert "_nofix" not in line and "_nolean" not in line and "inrange_llm" not in line, job
        assert "--exclusive" in line and "--dependency=" in line, job


def test_the_package_times_five_queries_on_the_headline_cells_and_chains_every_job():
    lines = _dry()
    jobs = [_job(line) for line in lines]
    for line in lines:
        job = _job(line)
        if job.startswith("sp-c-") and ("perf.py" in line or "v1_bytes.py" in line or "reexec.py" in line):
            assert re.search(r"--(queries|reps) 5\b", line), job
        if job in ("sp-v1-opt13-64", "sp-v1-opt13-2048", "sp-v1-llama7-64", "sp-v1-llama7-2048"):
            assert "--reps 5" in line, job
        if job.startswith("sp-serve-"):
            assert "--reps 10" in line, job
    deps = [re.search(r"--dependency=(\S+)", line).group(1) for line in lines]
    assert deps[0] == "singleton" and deps[1].startswith("afterok:DRY1,")       # nothing before smoke succeeds
    assert all(d.startswith("afterany:DRY") for d in deps[2:]), deps
    ids = [int(re.search(r"DRY(\d+)", d).group(1)) for d in deps[1:]]
    assert ids == list(range(1, len(lines))), "each job after the previous one"
    assert len(set(jobs)) == len(jobs)
    for tier in ("sp-smoke", "sp-c-", "sp-v1-", "sp-serve-", "l40s_sp2027-c-", "sp-g-", "sp-gen-",
                 "l40s_sp2027-b-", "sp-r-", "sp-setup-", "sp-q-", "sp-l-"):
        assert any(j.startswith(tier) for j in jobs), tier


def test_the_rehearsal_submits_small_cells_of_every_tier():
    lines = _dry(PVI_REHEARSAL="1")
    jobs = [_job(line) for line in lines]
    assert len(lines) <= 30
    for prefix in ("sp-smoke", "sp-c-", "sp-v1-", "sp-serve-", "l40s_sp2027-r-c-", "sp-g-", "sp-gen-",
                   "l40s_sp2027-r-b-", "sp-r-", "sp-setup-", "sp-q-", "sp-l-"):
        assert any(j.startswith(prefix) for j in jobs), prefix
    big = ("opt-1.3b", "opt-6.7b", "llama2-13b", "qwen3-4b")
    for line in lines:
        assert not any(f"--model {m} " in line for m in big), _job(line)
        if "--model llama2-7b" in line:            # the long tier's one block at 4,608 tokens
            assert "--layers 1" in line, _job(line)
        assert "--time=" in line
