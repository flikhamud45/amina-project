"""The run package (``experiments/package/run.sh``) and the harnesses its tiers call.

* ``gen_bench`` (the gen tier) on a tiny decoder: every greedy and sampled response is accepted in modes C and Kpre
  under Fiat-Shamir, a wrong token is rejected by the token rule and logits forged to choose it by the LM head's
  check, and the record carries the timings with their ``[min, max]`` and the peaks;
* ``setup_timing`` (the setup tier): the honest setup proof is accepted, its bytes and parameters are recorded;
* ``reexec --max-pos`` refuses a model with learned positions (before building anything); ``h2d_bandwidth``
  measures on a GPU and exits 2 without one; ``perf``'s ``[min, max]`` are each query's own totals;
* the package's dry run (``--dry-run``): no v0, V1 or generation cell turns Fiat-Shamir off, the only interactive
  benchmark cells are those a table compares with an interactive system, every mode-C cell takes ``--policy``, the
  headline cells time 5 queries, every job is chained after the previous one and the jobs after smoke wait for it,
  every tier submits jobs, the cells behind the text's generation and recipe numbers are there, and the rehearsal
  submits small cells only.

The smoke tier runs this file on the final platform.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.costs import perf, setup_timing
from experiments.generation import gen_bench
from experiments.reexec import h2d_bandwidth, reexec

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
RUN = Path("code") / "experiments" / "package" / "run.sh"
TIERS = ("smoke", "cpuv", "v1", "serve", "bench", "gpuv", "gen", "batch", "real", "setup", "improve", "quality", "tv",
         "attack", "long")


def _cli(mod, monkeypatch, capsys, script: str, *argv) -> tuple[int, list[dict]]:
    monkeypatch.setattr(sys, "argv", [script, *argv])
    threads = torch.get_num_threads()
    try:
        status = mod.main()
    finally:
        torch.set_num_threads(threads)
    return status, [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]


def test_gen_bench_greedy_and_sampled_responses_and_both_tamper_cells(monkeypatch, capsys):
    status, lines = _cli(gen_bench, monkeypatch, capsys, "gen_bench.py", "--model", "tiny-llama", "--prompt", "6",
                         "--gen", "3", "--decode", "greedy", "sample", "--modes", "C", "Kpre", "--policy", "auto",
                         "--queries", "2", "--device", "cpu", "--threads", "1", "--sampler-a", "4096")
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
    status, lines = _cli(setup_timing, monkeypatch, capsys, "setup_timing.py", "--model", "tiny-llama", "--policy",
                         "auto", "--amortise", "100", "1000", "--reps", "1", "--device", "cpu", "--threads", "1")
    assert status == 0 and [r["amortise"] for r in lines] == [100, 1000]
    for r in lines:
        assert r["kind"] == "setup" and r["accepted"] and r["proof_bytes"] > 0 and r["s"] >= 7
        assert r["query_columns_bytes_untrusted"] >= r["query_columns_bytes_honest"] > 0
        assert r["prove_s"] > 0 and r["verify_s"] > 0 and r["commit_s"] > 0


def test_reexec_refuses_a_longer_context_for_learned_positions(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["reexec.py", "--model", "gpt2", "--seq", "64", "--max-pos", "4096",
                                      "--device", "cpu", "--threads", "1"])
    with pytest.raises(SystemExit, match="learned positions"):
        reexec.main()


def test_h2d_bandwidth(monkeypatch, capsys):
    if not torch.cuda.is_available():
        monkeypatch.setattr(sys, "argv", ["h2d_bandwidth.py"])
        assert h2d_bandwidth.main() == 2
        return
    status, lines = _cli(h2d_bandwidth, monkeypatch, capsys, "h2d_bandwidth.py", "--sizes-mb", "16", "--reps", "2")
    assert status == 0 and len(lines) == 4
    assert all(r["kind"] == "h2d" and r["gb_s"] > 0 for r in lines)


def test_perf_spread_is_each_querys_own_totals():
    runs = [{"timings": {"prove_forward": 1.0, "fs_hash": 0.5, "verify_derive": 2.0}},
            {"timings": {"prove_forward": 3.0, "fs_hash": 0.5, "verify_derive": 1.0}}]
    assert perf._spread(runs) == {"prove_s_range": [1.5, 3.5], "verify_s_range": [1.5, 2.5]}


# ------------------------------------------------------------------------------------------ the package
def _dry(*flags) -> list[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    env = {**os.environ, "HF_HOME": "/tmp/hf", "CXX": "g++"}
    out = subprocess.run([bash, str(RUN), "--platform", "final", "--dry-run", "--optional", "--llama-licence",
                          *flags, "all"], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return [line for line in out.stdout.splitlines() if line.startswith("sbatch")]


def _job(line: str) -> str:
    return re.search(r" -J (\S+)", line).group(1)


def test_the_package_runs_fiat_shamir_and_the_policy_everywhere():
    lines = _dry("--policy", "cnn18c")
    assert len(lines) > 100
    for line in lines:
        job = _job(line)
        assert "--no-fs" not in line and "--fs " not in line, job          # Fiat-Shamir is every script's default
        if re.search(r"--modes [^-]*\bC\b", line) or "costs.setup_timing" in line:
            assert "--policy cnn18c" in line, job
        if "costs.bench" in line:
            modes = re.search(r"--modes (\S+)", line).group(1)
            if ":int" in modes:     # interactive cells: zkLLM's rows, Maverick's, the batched byte model, the images
                assert re.search(r"-(c|cg)-(opt-125m|opt-1\.3b|llama2-7b|llama2-13b)-2k$", job) or "thr1" in job \
                    or job.endswith("-cint") or " cnn " in line, job
            if re.search(r"\bC:", modes):
                assert "--policy cnn18c" in line, job
            assert "--platform final" in line, job
        assert "--exclusive" in line and "--dependency=" in line, job
        switches = set(re.findall(r"PVI_[A-Z_]+", line)) - {"PVI_NATIVE_DIR", "PVI_TEST_DEVICE"}
        assert not switches, (job, switches)            # the options are flags; only host locations stay variables


def test_the_package_times_five_queries_on_the_headline_cells_and_chains_every_job():
    lines = _dry()
    jobs = [_job(line) for line in lines]
    for line in lines:
        job = _job(line)
        if job.startswith("sp-c-") and any(s in line for s in ("costs.perf", "v1.v1_bytes", "reexec.reexec")):
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
    for prefix in ("sp-smoke", "sp-c-", "sp-v1-", "sp-serve-", "sp-b-", "sp-g-", "sp-gen-", "sp-bt-", "sp-r-",
                   "sp-setup-", "sp-i-", "sp-q-", "sp-tv-", "sp-a-", "sp-l-"):
        assert any(j.startswith(prefix) for j in jobs), prefix


def test_the_rehearsal_submits_small_cells_of_every_tier():
    lines = _dry("--rehearsal")
    jobs = [_job(line) for line in lines]
    assert len(lines) <= 40
    for prefix in ("sp-smoke", "sp-c-", "sp-v1-", "sp-serve-", "sp-b-", "sp-g-", "sp-gen-", "sp-bt-", "sp-r-",
                   "sp-setup-", "sp-i-", "sp-q-", "sp-tv-", "sp-a-", "sp-l-"):
        assert any(j.startswith(prefix) for j in jobs), prefix
    big = ("opt-1.3b", "opt-6.7b", "llama2-13b", "qwen3-4b", "facebook/opt-1.3b", "facebook/opt-6.7b")
    for line in lines:
        assert not any(f"--model {m} " in line for m in big), _job(line)
        if "--model llama2-7b" in line:            # the long tier's one block at 4,608 tokens
            assert "--layers 1" in line, _job(line)
        assert "--time=" in line


def test_the_gen_and_quality_cells_of_the_text():
    """The gen tier's cells (GPT-2's per-token prover after a 64-token prompt; Llama-2-7B and Qwen3-4B after a
    512-token prompt, greedy and sampled) and the OPT-125M recipe studies of the quality tier; the rehearsal keeps its
    own small generation cell."""
    lines = {_job(line): line for line in _dry()}
    gpt2 = lines["sp-gen-gpt2"].split("generation.gen_bench", 1)[1]
    assert "--model gpt2 --prompt 64 --gen 256 --decode greedy --modes C Kpre --policy cnn18c" in gpt2
    assert "--layers" not in gpt2 and "--label gen" in gpt2
    for m in ("llama2-7b", "qwen3-4b"):
        assert f"--model {m} --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre" in lines[f"sp-gen-{m}"]
    rec = lines["sp-q-opt125-recipes"]
    assert "--model facebook/opt-125m --recipe legacy --graphs smooth --pbits 11" in rec
    assert "--model facebook/opt-125m --graphs smooth --alphas 0.3 0.4 0.5 0.6 0.7 --pbits 11" in rec
    rehearsal = {_job(line): line for line in _dry("--rehearsal")}
    assert "--layers 2" in rehearsal["sp-gen-gpt2"] and "sp-q-opt125-recipes" not in rehearsal


def test_every_tier_is_documented_and_reachable():
    text = (REPO / RUN).read_text()
    for tier in TIERS:
        assert f"tier_{tier}()" in text and re.search(rf"^#\s+{tier}\s", text, re.M), tier
