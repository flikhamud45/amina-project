"""The run package (``experiments/package/run.sh``) and the harnesses its tiers call.

* ``gen_bench`` (the gen tier) on a tiny decoder: every greedy and sampled response is accepted in modes C and Kpre
  under Fiat-Shamir, a wrong token is rejected by the token rule and logits forged to choose it by the LM head's
  check, and the record carries the timings with their ``[min, max]`` and the peaks;
* ``setup_timing`` (the setup tier): the honest setup proof is accepted, its bytes and parameters are recorded;
* ``reexec --max-pos`` refuses a model with learned positions (before building anything); ``h2d_bandwidth``
  measures on a GPU and exits 2 without one; ``perf``'s ``[min, max]`` are each query's own totals;
  ``perf --client-memory`` checks every timed proof again in the client's own process, with the in-process verdicts,
  and records that process's memory;
* the package's dry run (``--dry-run``): no v0, V1 or generation cell turns Fiat-Shamir off, the only interactive
  benchmark cells are those a table compares with an interactive system, every mode-C cell takes ``--policy``, the
  headline cells time 5 queries, every job is chained after the previous one and the jobs after smoke wait for it,
  every tier submits jobs, the cells behind the text's generation, recipe and quality numbers are there (no quality
  cell measured twice where one job suffices), and the rehearsal submits small cells only, which run every script of
  every tier on the paths the full cells take (Llama-2-7B's grid path, or OpenLLaMA-7B's tokenizer without
  --llama-licence); every CPU-timing
  cell requires the native kernels and is pinned to one NUMA node, and the dry run lists every job even on a host
  without the run's settings; every cost cell builds its decoders at the default 11-bit attention probabilities but
  the improve tier's A/B of 8 against 11 bits (in the full run and the rehearsal); Llama-2-7B's CPU-verifier cells
  measure the client's own memory; the gate tier (the full-size checks before the run) is not in 'all', is refused
  with --rehearsal, and its V1 cells reach GKR instances of 2^lmax leaves (exactly 2^lmax on OPT-1.3B), its long cells
  16,384 tokens;
* every job starts by logging its node and checking the GPU and CPU models (``experiments/package/node.py``; exit
  status 86 on another model), and ``--gres`` and ``--exclude-nodes`` reach every job and allocation;
* the package's submission against stand-ins for SLURM: a queued job of the same platform is not submitted again and
  the chain and the smoke gate pass through it; ``--one-job`` submits the tiers as one whole-node job that runs them
  with ``--local``, and a ``--local`` run again runs the smoke and only the jobs that did not end successfully; a
  ``--local`` run checks its node first (on another model it runs nothing), runs every job past a failed one, needs
  no smoke when the smoke tier is not selected, and ends with a summary that names the command to submit again;
  nothing is submitted without a GPU request;
  nothing is submitted when the partition lacks the memory, the compiler (or, for the cpuv tier's llama.cpp build,
  CMake) is missing, the interpreter lacks the quality tier's tokenizer libraries, a Hugging Face cache variable
  overrides ``HF_HOME`` with an unwritable folder or the bench tier's models and datasets are; ``bench.py`` itself
  stops without the substitute model; ``experiments/package/pin.py`` picks one thread of each core of one NUMA node,
  and every record keeps the affinity, the library versions and the checkpoints' revisions.

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
from experiments.package import pin
from experiments.paper.records import LOG, tier_of
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
HOST_VARIABLES = {"PVI_NATIVE_DIR", "PVI_PY_INCLUDE", "PVI_TEST_DEVICE", "PVI_CIFAR_ROOT", "PVI_IMAGENET_ROOT",
                  "PVI_CACHE"}
"""The ``PVI_`` variables a job may carry: places on the host (and the smoke's test device), never a switch."""
CPU_TIERS = ("cpuv", "v1", "bench", "batch", "gen", "real", "setup", "improve", "long")
"""The tiers with CPU-timing jobs (pinned to 8 cores of one NUMA node); the others have none."""


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    return bash


def _dry(*flags, env: dict | None = None, out: list | None = None, tiers: tuple = ("all",),
         status: int = 0) -> list[str]:
    """The ``sbatch`` lines of ``run.sh --dry-run`` of ``tiers`` (its output appended to ``out``), which must end
    with the exit status ``status``."""
    env ={**os.environ, "HF_HOME": "/tmp/hf", "CXX": "g++"} if env is None else env
    run = subprocess.run([_bash(), str(RUN), "--platform", "final", "--dry-run", "--optional", "--llama-licence",
                          *flags, *tiers], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == status, run.stderr
    if out is not None:
        out.append(run)
    return [line for line in run.stdout.splitlines() if line.startswith("sbatch")]


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
            if ":int" in modes:     # interactive: zkLLM's rows and lengths, Maverick's, the batched bytes, the images
                assert re.search(r"-(c|cg)-(opt-125m|opt-1\.3b|llama2-7b|llama2-13b)-2k$", job) or "thr1" in job \
                    or job.endswith("-cint") or job.endswith("-zk") or " cnn " in line, job
            if re.search(r"\bC:", modes):
                assert "--policy cnn18c" in line, job
            assert "--platform final" in line, job
        assert "--exclusive" in line and "--dependency=" in line and "--comment=final" in line, job
        switches = set(re.findall(r"\bPVI_[A-Z_]+", line)) - HOST_VARIABLES
        assert not switches, (job, switches)            # the options are flags; only host locations stay variables


def test_every_cpu_timing_cell_requires_the_native_kernels_and_is_pinned():
    """A cell that times the CPU verifier or a re-execution on the CPU passes --require-native (it stops instead of
    timing the torch path) and starts by pinning itself to 8 cores of one NUMA node; the CPU re-execution includes
    the int8 GEMMs of torch._int_mm; the GPU, serving, quality, tv and attack jobs are not pinned."""
    pinned = 0
    for line in _dry():
        job, cmd = _job(line), line.split("--wrap", 1)[1]
        cpu_verifier = re.search(r"costs\.perf|v1\.v1_bytes|generation\.gen_bench|costs\.bench", cmd) \
            and "--verifier-device cuda" not in cmd and "--codec-only" not in cmd
        cpu_reexec = "reexec.reexec" in cmd and "--device cpu" in cmd
        if (cpu_verifier or cpu_reexec) and job != "sp-smoke":
            assert "--require-native" in cmd and "-m experiments.package.pin --cores 8 --pid $$ && " in cmd, job
        if cpu_reexec:                  # glibc's allocator defaults, or the like-for-like row with the verifier's
            assert " int_mm " in cmd and ("--no-malloc-tune" in cmd or "--malloc-tune" in cmd), job
            assert "MALLOC_" not in cmd, job                    # the allocator setting is a flag, not a variable
        if "reexec.llamacpp" in cmd:     # llama.cpp on the same 8 pinned threads (never its default thread count)
            assert "-m experiments.package.pin --cores 8 --pid $$ && " in cmd and "--threads 8 " in cmd, job
        if "experiments.package.pin" in cmd:
            pinned += 1
            assert tier_of(job[len("sp-"):]) in CPU_TIERS, job
    assert pinned > 50


def test_the_dry_run_lists_every_job_without_the_hosts_settings():
    """Without HF_HOME, CXX and the interpreter the dry run still lists every job (to the summary line) and names
    what the run would refuse."""
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HOME", "CXX", "HF_TOKEN")}
    runs: list = []
    lines = _dry("--python", "/no/python", env=env, out=runs)
    assert len(lines) == len(_dry()) and runs[0].stdout.splitlines()[-1].startswith("# smoke cpuv ")
    for text in ("export HF_HOME=", "export CXX=", "--no-native-ok", "is not an interpreter", "HF_TOKEN"):
        assert text in runs[0].stderr, text


def test_the_help_prints_the_whole_header():
    out = subprocess.run([_bash(), str(RUN), "--help"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.rstrip().endswith("is the runner's guide."), out.stdout[-300:]
    assert "--no-pin" in out.stdout and "--no-native-ok" in out.stdout
    assert all(f in out.stdout for f in ("--require-gpu", "--require-cpu", "--gres", "--exclude-nodes"))


FAKE = {"squeue": '#!/bin/bash\n[ -z "${FAKE_QUEUE:-}" ] || printf \'%s\\n\' "$FAKE_QUEUE"\n',
        "sbatch": '#!/bin/bash\necho "$*" >> "$FAKE_LOG"\ncat > /dev/null\nwc -l < "$FAKE_LOG" | tr -d " "\n',
        "sinfo": '#!/bin/bash\nprintf \'%s\\n\' $FAKE_MEM\n',
        "fakepy": '#!/bin/bash\ncase "$*" in *experiments.package.node*) [ -z "${FAKE_WRONG_NODE:-}" ] || '
                  '{ echo "wrong node: GPU NVIDIA A100 / CPU Intel Xeon, expected GPU L40S"; exit 86; } ;; esac\n'
                  'case "$*" in *"${FAKE_FAIL:-@none@}"*) exit 3 ;; esac\n'
                  'case "$*" in *"${FAKE_SLOW:-@none@}"*) sleep 100 ;; esac\necho "{}"\n',
        "cmake": '#!/bin/bash\necho "cmake version 3.31.6"\n'}
"""Stand-ins for SLURM's commands, the interpreter and CMake: ``squeue`` lists ``$FAKE_QUEUE`` (``name|platform|id``
lines), ``sbatch`` appends its arguments to ``$FAKE_LOG`` and answers with its line number as the job id, ``sinfo``
gives each node's memory in MB (``$FAKE_MEM``); the interpreter prints ``{}`` and succeeds, except for a command
line that contains ``$FAKE_FAIL`` (exit status 3), one that contains ``$FAKE_SLOW`` (which first sleeps 100 s)
and the node check with ``$FAKE_WRONG_NODE`` set (exit status 86, ``wrong node``); ``cmake`` (the cpuv tier's
pre-flight) prints its version."""

HF_CACHES = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE",
             "TRANSFORMERS_OFFLINE")


SIGNAL_WHEN = ('bash "$0" "$@" > out.txt 2> err.txt & p=$!; until grep -q "$SIGNAL_AFTER" out.txt; do sleep 0.2; '
               'done; kill -TERM $p; wait $p; st=$?; cat out.txt; cat err.txt >&2; exit $st')
"""Runs ``run.sh`` in the background and sends it SIGTERM once its output holds ``$SIGNAL_AFTER``."""


def _submit(tmp_path, *argv, queue: str = "", mem: str = "1031000", cxx: bool = True, extra: dict | None = None,
            signal_after: str | None = None):
    """``run.sh argv`` from a copy of the package under ``tmp_path`` (kept between calls with the same
    ``tmp_path``), submitting to :data:`FAKE`'s SLURM, with the variables ``extra`` added to the environment (with
    ``signal_after``, SIGTERM is sent to it once its output holds that text: :data:`SIGNAL_WHEN`):
    ``(exit status, stdout, stderr, [sbatch's arguments of each job])``."""
    bash = _bash()
    root, bin_ = tmp_path / "root", tmp_path / "bin"
    (root / RUN).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / RUN, root / RUN)
    bin_.mkdir(exist_ok=True)
    for name, text in FAKE.items():
        (bin_ / name).write_text(text, newline="\n")
        (bin_ / name).chmod(0o755)
    log = tmp_path / "sbatch.log"
    log.write_text("")
    py = (bin_ / "fakepy").as_posix()
    drop = ("PVI_CACHE", "PVI_CIFAR_ROOT", "PVI_IMAGENET_ROOT", "FAKE_FAIL", "FAKE_SLOW", "FAKE_WRONG_NODE",
            *HF_CACHES)
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(PATH=f"{bin_}{os.pathsep}{os.environ['PATH']}", FAKE_QUEUE=queue, FAKE_MEM=mem, FAKE_LOG=log.as_posix(),
               CXX=py if cxx else "", HF_HOME=(tmp_path / "hf").as_posix(), SBATCH_GRES="gpu:l40s:1")
    env.update(extra or {})
    cmd = [bash, str(RUN), "--python", py, *argv]
    if signal_after is not None:
        env["SIGNAL_AFTER"] = signal_after
        cmd = [bash, "-c", SIGNAL_WHEN, *cmd[1:]]
    out = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=120)
    return out.returncode, out.stdout, out.stderr, log.read_text().splitlines()


def test_a_queued_job_of_the_same_platform_is_skipped_and_the_chain_and_gate_pass_through_it(tmp_path):
    queue = "sp-smoke|smoke|901\nsp-c-gpt2-64|rehearsal|902\nsp-other|(null)|903"
    status, out, err, jobs = _submit(tmp_path, "--platform", "rehearsal", "--rehearsal", "--no-exclusive", "smoke",
                                     "cpuv", queue=queue)
    assert status == 0, err
    names = [_job(" " + j) for j in jobs]
    assert names == ["sp-smoke", "sp-c-rx-gpt2", "sp-c-v1-gpt2-64", "sp-c-llamacpp-llama2-7b-L1"]    # another
    # platform's sp-smoke is not this one
    assert all("--comment=rehearsal" in j and "--exclusive" not in j for j in jobs)
    assert "--dependency=afterok:1,afterany:902,singleton" in jobs[1]    # after the smoke's success and the queued job
    assert "--dependency=afterany:2,singleton" in jobs[2] and "--dependency=afterany:3,singleton" in jobs[3]
    assert "skip sp-c-gpt2-64: already queued (job 902)" in out and "# 1 jobs already queued" in out


def test_a_queued_smoke_still_gates_the_run(tmp_path):
    status, out, err, jobs = _submit(tmp_path, "--platform", "final", "smoke", "cpuv", queue="sp-smoke|final|901")
    assert status == 0, err
    assert len(jobs) == 14 and "--dependency=afterok:901,singleton" in jobs[0], jobs[0]


def test_one_job_submits_the_tiers_as_one_whole_node_job_that_runs_them_locally(tmp_path):
    """--one-job: one job with the node to itself, its time limit the sum of the jobs' limits capped at --max-hours,
    queued again when pre-empted, whose script runs the same command with --local; a queued one is not submitted
    again, and with --no-exclusive it asks for the largest job's memory instead of the node."""
    argv = ("--platform", "final", "--llama-licence", "--one-job", "smoke", "cpuv", "long")
    status, out, err, jobs = _submit(tmp_path / "a", *argv)
    assert status == 0 and len(jobs) == 1, err
    for part in ("-J sp-node-smoke-cpuv-long", "--comment=final", "--exclusive", "--mem=0", "--time=24:00:00",
                 "--requeue", "--signal=B:TERM@300", "--open-mode=append", "--dependency=singleton",
                 "logs/final/node-smoke-cpuv-long-%j.log", "--gres=gpu:l40s:1"):
        assert part in jobs[0], (part, jobs[0])
    assert "# one job: smoke cpuv long: 27 jobs, job limits 56 h, time limit 24 h" in out, out
    status, out, err, jobs = _submit(tmp_path / "b", "--dry-run", "--max-hours", "12", *argv)
    line = next(s for s in out.splitlines() if s.startswith("sbatch"))
    assert status == 0 and not jobs and "--time=12:00:00" in line, err
    assert re.search(r' && exec bash code/experiments/package/run\.sh --local --python \S+ --max-hours 12 '
                     r'--platform final --llama-licence smoke cpuv long --from-one-job"$', line), line
    status, out, err, jobs = _submit(tmp_path / "c", *argv, queue="sp-node-smoke-cpuv-long|final|77")
    assert status == 0 and not jobs and "skip sp-node-smoke-cpuv-long: already queued (job 77)" in out, err
    status, out, err, jobs = _submit(tmp_path / "d", "--no-exclusive", *argv)
    assert status == 0 and "--mem=256G" in jobs[0] and "--exclusive" not in jobs[0], jobs
    assert "--gres=gpu:l40s:1" in jobs[0]                   # the jobs that time nothing get the same GPU
    status, out, err, jobs = _submit(tmp_path / "e", "--local", *argv)
    assert status == 2 and not jobs and "--one-job submits a job that runs --local itself" in err


def test_a_local_run_again_runs_the_smoke_and_the_jobs_that_did_not_end_successfully(tmp_path):
    """--local (the script of a --one-job allocation): a job that ends successfully leaves a marker, and the same
    command run again (after a time limit or a pre-emption) runs the smoke, which checks the node, and only the jobs
    without a marker: one that failed, one whose marker was deleted.  Every log is named as the records expect, and a
    later run's log sorts later."""
    argv = ("--platform", "t", "--rehearsal", "--local", "smoke", "cpuv")
    logs = tmp_path / "root" / "logs" / "t"
    status, out, err, _ = _submit(tmp_path, *argv, extra={"FAKE_FAIL": "reexec.reexec"})
    assert status == 1 and "sp-c-rx-gpt2: exit status 3" in err, err
    assert re.search(r"^#   FAILED  sp-c-rx-gpt2 \(exit status 3\)  logs/t/sp-c-rx-gpt2-\d+\.out$", out, re.M), out
    assert re.search(r"^#   bash code/experiments/package/run\.sh --python \S+ --platform t --rehearsal --local smoke "
                     r"cpuv$", out, re.M), out
    assert sorted(p.name for p in (logs / "done").iterdir()) == ["sp-c-gpt2-64", "sp-c-llamacpp-llama2-7b-L1",
                                                                 "sp-c-v1-gpt2-64", "sp-smoke"]
    status, out, err, _ = _submit(tmp_path, *argv)
    assert status == 0 and re.findall(r"running (sp-\S+)", out) == ["sp-smoke", "sp-c-rx-gpt2"], out
    assert "# every job of this submission has ended successfully" in out
    assert "skip sp-c-gpt2-64: ended successfully earlier" in out and "# 3 jobs had ended successfully" in out
    (logs / "done" / "sp-c-gpt2-64").unlink()
    status, out, err, _ = _submit(tmp_path, *argv)
    assert status == 0 and re.findall(r"running (sp-\S+)", out) == ["sp-smoke", "sp-c-gpt2-64"], out
    smokes = sorted(logs.glob("sp-smoke-*.out"), key=lambda p: p.stat().st_mtime_ns)
    ids = [int(LOG.match(p.name)["job"]) for p in smokes]
    assert len(ids) == 3 and ids == sorted(ids) and all("SMOKE OK" in p.read_text() for p in smokes)
    assert all(LOG.match(p.name) for p in logs.glob("*.out"))


def test_a_submission_without_the_smoke_runs_every_job_past_a_failure_and_names_its_command(tmp_path):
    """A --one-job allocation (--local --from-one-job) without the smoke tier starts at once (no SMOKE OK needed),
    runs every job although one before them failed (no job reads another's output), and its summary names each job
    with its log, the failed one with its exit status, and the --one-job command, which, submitted again, runs only
    the failed job."""
    argv = ("--platform", "t", "--rehearsal", "--local", "--from-one-job", "cpuv")
    status, out, err, _ = _submit(tmp_path, *argv, extra={"FAKE_FAIL": "reexec.reexec"})
    ran = ["sp-c-gpt2-64", "sp-c-rx-gpt2", "sp-c-v1-gpt2-64", "sp-c-llamacpp-llama2-7b-L1"]
    assert status == 1 and re.findall(r"running (sp-\S+)", out) == ran, (out, err)
    assert "sp-smoke" not in out and "SMOKE OK" not in out + err
    assert "3 jobs ended successfully, 1 failed, 0 had ended successfully in an earlier run" in out
    assert len(re.findall(r"^#   ok      sp-c-\S+  logs/t/sp-c-\S+\.out$", out, re.M)) == 3, out
    assert re.search(r"^#   FAILED  sp-c-rx-gpt2 \(exit status 3\)  logs/t/sp-c-rx-gpt2-\d+\.out$", out, re.M), out
    command = re.search(r"^#   (bash code/experiments/package/run\.sh .*)$", out, re.M).group(1)
    assert re.fullmatch(r"bash code/experiments/package/run\.sh --python \S+ --platform t --rehearsal --one-job cpuv",
                        command), command
    status, out, err, _ = _submit(tmp_path, *argv)
    assert status == 0 and re.findall(r"running (sp-\S+)", out) == ["sp-c-rx-gpt2"], out


def test_a_local_run_stopped_by_a_signal_writes_its_summary(tmp_path):
    """A --one-job allocation (--local --from-one-job) that SLURM signals (its time limit, through --signal, or a
    pre-emption) stops its running job at once and writes its summary: the jobs that ended successfully, the stopped
    one as failed with exit status 143 and its log, and the command to submit again, which runs only the stopped job
    and those after it."""
    argv = ("--platform", "t", "--rehearsal", "--local", "--from-one-job", "cpuv")
    status, out, err, _ = _submit(tmp_path, *argv, extra={"FAKE_SLOW": "reexec.reexec"},
                                  signal_after="running sp-c-rx-gpt2")
    assert status == 143 and "run.sh: stopped by SIGTERM" in err, (status, out, err)
    assert re.findall(r"running (sp-\S+)", out) == ["sp-c-gpt2-64", "sp-c-rx-gpt2"], out
    assert "1 jobs ended successfully, 1 failed" in out and "stopped by SIGTERM" in out, out
    assert re.search(r"^#   FAILED  sp-c-rx-gpt2 \(exit status 143: stopped by a signal\)  "
                     r"logs/t/sp-c-rx-gpt2-\d+\.out$", out, re.M), out
    assert re.search(r"^#   bash code/experiments/package/run\.sh --python \S+ --platform t --rehearsal --one-job "
                     r"cpuv$", out, re.M), out
    status, out, err, _ = _submit(tmp_path, *argv)
    assert status == 0 and re.findall(r"running (sp-\S+)", out) == ["sp-c-rx-gpt2", "sp-c-v1-gpt2-64",
                                                                      "sp-c-llamacpp-llama2-7b-L1"], out


def test_the_tv_jobs_use_the_cpus_they_ask_for_and_have_room_over_their_estimates():
    """The tv tier runs on the CPU in the 8 CPUs every job and --one-job allocation asks for: no more worker processes
    or threads than those, and limits above the documented times (OPT-1.3B's analysis took 43 min with 16 workers;
    OPT-125M's Monte Carlo on one worker about an hour)."""
    lines = {_job(line): line for line in _dry()}
    for job, hours in (("sp-tv-opt-1.3b", 4), ("sp-tv-checks", 3)):
        line = lines[job]
        assert "--cpus-per-task=8" in line and f"--time={hours}:00:00" in line, line
        for flag in ("--workers", "--threads"):
            assert all(int(n) <= 8 for n in re.findall(rf"{flag} (\d+)", line)), line
    assert "--workers 8 --threads 8" in lines["sp-tv-opt-1.3b"]


def test_the_qllama_tier_is_in_all_and_runs_next_to_quality():
    """The Llama-2 grids are a tier of their own (qllama, in 'all' after quality), so that they run on a second GPU
    next to the quality tier; their jobs keep the quality tier's names (q-...), whose records numbers.py reads."""
    run = subprocess.run([_bash(), str(RUN), "--platform", "final", "--dry-run", "--llama-licence", "all"], cwd=REPO,
                         env={**os.environ, "HF_HOME": "/tmp/hf", "CXX": "g++"}, capture_output=True, text=True,
                         timeout=120)
    tiers = re.search(r"^# (smoke .*): \d+ jobs", run.stdout, re.M).group(1).split()
    assert tiers[tiers.index("quality") + 1] == "qllama", tiers
    qllama = re.search(r"^# tier qllama: (\d+) jobs, job limits (\d+) h, estimate (\S+) h$", run.stdout, re.M)
    assert qllama and qllama.group(1) == "12" and qllama.group(3) == "40-48", run.stdout[-2000:]
    for tier in ("quality", "qllama"):
        jobs = [_job(x) for x in _dry(tiers=(tier,))]
        assert jobs and all(tier_of(j[len("sp-"):]) == "quality" for j in jobs), (tier, jobs)
    assert not any(re.search(r"Llama-2|open_llama", x) for x in _dry(tiers=("quality",)))


def test_a_failed_smoke_still_stops_its_own_invocation(tmp_path):
    """With the smoke tier, nothing after it runs unless it ends in SMOKE OK, and the summary says so."""
    status, out, err, _ = _submit(tmp_path, "--platform", "t", "--rehearsal", "--local", "smoke", "cpuv",
                                  extra={"FAKE_FAIL": "-m pytest"})
    assert status == 1 and re.findall(r"running (sp-\S+)", out) == ["sp-smoke"], out
    assert "the smoke did not end in SMOKE OK: the jobs after it were not run" in out
    assert "#   FAILED  sp-smoke (exit status 3)" in out and "--local smoke cpuv" in out


def test_a_node_of_another_model_runs_nothing(tmp_path):
    """--local (a --one-job allocation) checks its node before the first job: on another GPU or CPU model it runs no
    job, exits with status 86 and names the command to submit again with the node excluded."""
    status, out, err, _ = _submit(tmp_path, "--platform", "t", "--rehearsal", "--local", "--from-one-job", "cpuv",
                                  extra={"FAKE_WRONG_NODE": "1"})
    assert status == 86 and "running" not in out, out
    assert "wrong node: GPU NVIDIA A100 / CPU Intel Xeon" in out and "the node check failed (exit status 86)" in err
    assert "--exclude-nodes" in out and "--rehearsal --one-job cpuv" in out
    assert not list((tmp_path / "root" / "logs" / "t").glob("sp-*.out"))


def test_every_job_checks_its_node_and_asks_for_the_same_gpu():
    """Every job's first command logs its node and checks the GPU and CPU models (default the paper's, L40S and EPYC
    9334; 'any' checks nothing), and --gres and --exclude-nodes reach every job and every --one-job allocation, also
    one without --exclusive (the tiers that time nothing); the dry run prints the requirement."""
    check = "-m experiments.package.node --require-gpu L40S --require-cpu EPYC\\ 9334 && "
    runs: list = []
    lines = _dry(out=runs)
    assert all(check in line for line in lines) and "# node: GPU 'L40S', CPU 'EPYC 9334'" in runs[0].stdout
    for line in lines:                                  # the check comes before the pinning and any work
        first = re.search(r"-m (experiments\.\S+|pytest)", line.split("--wrap", 1)[1]).group(1)
        assert first == "experiments.package.node", _job(line)
    lines = _dry("--require-gpu", "any", "--require-cpu", "any", "--gres", "gpu:l40s:1", "--exclude-nodes", "t-806")
    assert all("--require-gpu any --require-cpu any && " in line for line in lines)
    assert all("--gres=gpu:l40s:1" in line and "--exclude=t-806" in line for line in lines)
    for flags in ((), ("--no-exclusive",)):
        one = _dry(*flags, "--one-job", "--gres", "gpu:l40s:1", "--exclude-nodes", "t-806", tiers=("tv", "attack"))
        assert len(one) == 1 and "--gres=gpu:l40s:1 --exclude=t-806" in one[0], one
        assert "--exclude-nodes t-806 tv attack --from-one-job" in one[0]
    _dry("--require-gpu", "", status=2)


def test_nothing_is_submitted_without_a_gpu_request(tmp_path):
    status, _, err, jobs = _submit(tmp_path / "a", "--platform", "final", "smoke", extra={"SBATCH_GRES": ""})
    assert status == 2 and not jobs and "no GPU requested" in err, err
    status, _, err, jobs = _submit(tmp_path / "b", "--platform", "final", "--gres", "gpu:l40s:1", "smoke",
                                   extra={"SBATCH_GRES": ""})
    assert status == 0 and len(jobs) == 1 and "--gres=gpu:l40s:1" in jobs[0], err
    status, _, err, jobs = _submit(tmp_path / "c", "--platform", "final", "--gres", "gpu:1", "smoke")
    assert status == 0 and "names no GPU type 'L40S'" in err, err


def test_the_node_check(monkeypatch, capsys):
    """``node.py``: one ``node:`` line of the node's fields; exit status 86 and a ``wrong node`` line when the GPU's
    name or the CPU's model name lacks the required part (letters compared without case; 'any' requires nothing); a
    job that sees no GPU has GPU none."""
    from experiments.package import node
    seen = {"host": "n-801", "gpu": "NVIDIA L40S", "gpu_count": 1, "cuda_visible_devices": "0", "gpu_uuid": None,
            "gpu_bus_id": None, "driver": "535.183.01", "cpu": "AMD EPYC 9334 32-Core Processor", "slurm_job": "1",
            "slurm_nodelist": "n-801"}
    monkeypatch.setattr(node, "describe", lambda: dict(seen))
    assert node.main(["--require-gpu", "l40s", "--require-cpu", "EPYC 9334"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("node: {") and json.loads(out.split("node: ", 1)[1]) == seen and "wrong" not in out
    assert node.main([]) == 0 and node.main(["--require-gpu", "any", "--require-cpu", "any"]) == 0
    capsys.readouterr()
    seen.update(cpu="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", host="t-806")
    assert node.main(["--require-gpu", "L40S", "--require-cpu", "EPYC 9334"]) == node.WRONG_NODE == 86
    assert ("wrong node: GPU NVIDIA L40S / CPU Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, expected GPU 'L40S' / CPU "
            "'EPYC 9334' (host t-806") in capsys.readouterr().out
    seen.update(gpu=None, cpu="AMD EPYC 9334 32-Core Processor")
    assert node.main(["--require-gpu", "L40S"]) == 86 and "wrong node: GPU none / CPU AMD" in capsys.readouterr().out
    assert node.matches("NVIDIA L40S", "l40s") and not node.matches(None, "L40S") and node.matches(None, "any")


def test_the_node_description_on_this_host():
    from experiments.package import node
    d = node.describe()
    assert set(d) == {"host", "gpu", "gpu_count", "cuda_visible_devices", "gpu_uuid", "gpu_bus_id", "driver", "cpu",
                      "slurm_job", "slurm_nodelist"}
    assert d["host"] and (d["gpu"] is None) == (d["gpu_count"] == 0)


def test_a_hugging_face_cache_variable_the_jobs_cannot_write_stops_the_submission(tmp_path):
    """HF_HUB_CACHE, TRANSFORMERS_CACHE and HF_DATASETS_CACHE take precedence over HF_HOME: an unwritable one stops
    run.sh before it submits anything; a writable one (or one still to be created in a writable folder) does not."""
    status, _, err, jobs = _submit(tmp_path / "a", "--platform", "final", "--rehearsal", "real",
                                   extra={"HF_HUB_CACHE": "/proc/no-cache"})
    assert status == 2 and not jobs and "HF_HUB_CACHE=/proc/no-cache takes precedence over HF_HOME" in err, err
    status, _, err, jobs = _submit(tmp_path / "b", "--platform", "final", "--rehearsal", "real",
                                   extra={"HF_HUB_CACHE": (tmp_path / "b" / "hf" / "hub").as_posix()})
    assert status == 0 and len(jobs) == 4, err


def test_nothing_is_submitted_when_the_partition_lacks_the_memory_or_the_compiler(tmp_path):
    status, _, err, jobs = _submit(tmp_path / "a", "--platform", "final", "smoke", "cpuv", mem="64000 131000")
    assert status == 2 and not jobs and "pass --mem-max 127" in err, err
    status, _, err, jobs = _submit(tmp_path / "b", "--platform", "final", "--rehearsal", "smoke", mem="64000 131000")
    assert status == 0 and len(jobs) == 1, err                          # the rehearsal asks for 64 GB at most
    status, _, err, jobs = _submit(tmp_path / "c", "--platform", "final", "smoke", cxx=False)
    assert status == 2 and not jobs and "--no-native-ok" in err, err
    status, _, err, jobs = _submit(tmp_path / "d", "--platform", "final", "--no-native-ok", "smoke", cxx=False)
    assert status == 0 and len(jobs) == 1, err
    # the cpuv tier builds llama.cpp: CMake (CMAKE, else cmake on the PATH) is checked before anything is submitted
    status, _, err, jobs = _submit(tmp_path / "e", "--platform", "final", "--rehearsal", "cpuv",
                                   extra={"CMAKE": "/no/cmake"})
    assert status == 2 and not jobs and "CMake ('/no/cmake') not found" in err and "cmake==3.31.6" in err, err
    status, _, err, jobs = _submit(tmp_path / "f", "--platform", "final", "--rehearsal", "smoke",
                                   extra={"CMAKE": "/no/cmake"})
    assert status == 0 and len(jobs) == 1, err                          # no cpuv tier: no CMake needed
    cmake = (tmp_path / "g" / "bin" / "cmake").as_posix()                # FAKE's stand-in, given by its path
    status, out, err, _ = _submit(tmp_path / "g", "--dry-run", "--platform", "final", "--rehearsal", "cpuv",
                                  extra={"CMAKE": cmake})
    wraps = [x for x in out.splitlines() if x.startswith("sbatch")]
    assert status == 0 and len(wraps) == 4 and all(f"CMAKE={cmake} " in x for x in wraps), err  # passed to the jobs


def test_the_qllama_tier_needs_the_tokenizer_libraries(tmp_path):
    """OpenLLaMA-7B ships only its SentencePiece model: run.sh submits nothing for the qllama tier without
    --llama-licence (OpenLLaMA-7B's jobs) while the interpreter cannot import sentencepiece and google.protobuf, and
    asks for neither without that tier (the quality tier included) or with --llama-licence (Llama-2 ships its fast
    tokenizer)."""
    fail = {"FAKE_FAIL": "import sentencepiece, google.protobuf"}
    status, _, err, jobs = _submit(tmp_path / "a", "--platform", "final", "--rehearsal", "qllama", extra=fail)
    assert status == 2 and not jobs and "cannot import sentencepiece and google.protobuf" in err, err
    for i, tier in enumerate(("cpuv", "quality")):
        status, _, err, jobs = _submit(tmp_path / f"b{i}", "--platform", "final", "--rehearsal", tier, extra=fail)
        assert status == 0 and jobs, err
    status, _, err, jobs = _submit(tmp_path / "c", "--platform", "final", "--rehearsal", "quality", "qllama")
    assert status == 0 and len(jobs) == 8 and any(" -J sp-q-openllama7-smoke " in j for j in jobs), err
    status, _, err, jobs = _submit(tmp_path / "d", "--platform", "final", "--rehearsal", "--llama-licence", "quality",
                                   "qllama", extra=fail)
    assert status == 0 and len(jobs) == 8 and any(" -J sp-q-llama-access " in j for j in jobs), err
    assert not any("openllama" in j for j in jobs), err


def test_the_bench_tier_needs_the_substitute_model_and_the_imagenet_layout(tmp_path):
    status, _, err, jobs = _submit(tmp_path, "--platform", "final", "bench")
    assert status == 2 and not jobs
    for text in ("resnet18_224_squirrel.pt", "ImageNet(dogs_cats: no ", "/train/152", "ImageNet(dogs_squirrels",
                 "README.md, Sec. 1"):
        assert text in err, (text, err)


def test_bench_stops_before_the_image_cells_without_the_substitute_model(tmp_path, monkeypatch):
    from experiments.costs import bench
    monkeypatch.setattr(bench, "RAW", tmp_path / "raw")
    monkeypatch.setattr(bench, "CNN_MODELS", tmp_path / "models")
    args = bench.build_parser().parse_args(["cnn", "--model", "resnet18_224", "--platform", "t", "--tampers", "3"])
    with pytest.raises(SystemExit, match="resnet18_224_squirrel"):
        bench.suite_cnn(args, {"device": "cpu"})


def _topology(root, nodes: dict, siblings: int) -> str:
    """A kernel topology under ``root``: ``nodes`` ``{node: [cpus]}``, CPU ``c`` on core ``c % siblings``."""
    for n, cpus in nodes.items():
        (root / "node" / f"node{n}").mkdir(parents=True)
        (root / "node" / f"node{n}" / "cpulist").write_text(pin.compact(cpus) + "\n")
        for c in cpus:
            topo = root / "cpu" / f"cpu{c}" / "topology"
            topo.mkdir(parents=True)
            (topo / "physical_package_id").write_text("0\n")
            (topo / "core_id").write_text(f"{c % siblings}\n")
    return str(root)


def test_pin_takes_one_thread_of_each_core_of_the_lowest_node_that_has_enough(tmp_path):
    assert pin.parse_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11] and pin.compact([11, 0, 1, 2, 3, 8, 10]) \
        == "0-3,8,10-11"
    # two nodes of 4 cores with 2 hardware threads each (CPU c and c + 8 share a core)
    root = _topology(tmp_path / "sys", {0: [0, 1, 2, 3, 8, 9, 10, 11], 1: [4, 5, 6, 7, 12, 13, 14, 15]}, 8)
    assert pin.choose(4, range(16), root) == (0, [0, 1, 2, 3])
    assert pin.choose(4, [4, 5, 6, 7, 12, 13, 14, 15], root) == (1, [4, 5, 6, 7])
    assert pin.choose(2, [8, 9, 0], root) == (0, [0, 9])
    assert pin.choose(4, [0, 8, 1, 9, 2, 10, 4, 5], root) is None       # 3 cores on node 0, 2 on node 1
    assert pin.nodes_of([3, 12], root) == [0, 1]
    assert pin.choose(2, [5, 6, 7], str(tmp_path / "none")) == (0, [5, 6])   # no topology: each CPU a core
    assert pin.choose(4, range(16), root, prefer=1) == (1, [4, 5, 6, 7])          # the GPU's node first
    assert pin.choose(4, [0, 1, 2, 3, 4], root, prefer=1) == (0, [0, 1, 2, 3])    # unless it lacks the cores


@pytest.mark.skipif(os.name == "nt", reason="a PCI address holds ':', which a Windows file name cannot")
def test_pin_reads_the_gpus_numa_node_from_its_pci_address(tmp_path):
    dev = tmp_path / "pci" / "0000:41:00.0"
    dev.mkdir(parents=True)
    (dev / "numa_node").write_text("3\n")
    assert pin.pci_node("00000000:41:00.0", str(tmp_path / "pci")) == 3
    (dev / "numa_node").write_text("-1\n")
    assert pin.pci_node("00000000:41:00.0", str(tmp_path / "pci")) is None
    assert pin.pci_node("00000000:42:00.0", str(tmp_path / "pci")) is None


def test_pin_sets_the_affinity_of_the_given_process(monkeypatch, capsys):
    got = {}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: got.update({pid: list(cpus)}), raising=False)
    monkeypatch.setattr(pin, "gpu_bus_ids", lambda: ["00000000:41:00.0"])
    monkeypatch.setattr(pin, "pci_node", lambda bus_id: 2)
    monkeypatch.setattr(pin, "choose", lambda n, allowed, prefer=None: (prefer, list(range(16, 16 + n))))
    assert pin.main(["--cores", "8", "--pid", "4242"]) == 0
    assert got == {4242: list(range(16, 24))}
    assert "pinned to CPUs 16-23 (8 cores of NUMA node 2; the GPU's node)" in capsys.readouterr().out
    monkeypatch.setattr(pin, "choose", lambda n, allowed, prefer=None: None)
    assert pin.main(["--pid", "4242"]) == 0 and "not pinned" in capsys.readouterr().out


def test_records_keep_the_affinity_library_versions_and_checkpoint_revisions(monkeypatch):
    from experiments.common import checkpoints, records
    monkeypatch.setattr(checkpoints, "SOURCES", {})
    sha = "0123456789abcdef0123456789abcdef01234567"
    checkpoints._snapshot("org/model", f"/hf/hub/models--org--model/snapshots/{sha}")
    checkpoints._snapshot("org/local", "/somewhere/else")
    env = records.environment("cpu")
    assert env["sources"]["org/model"] == sha and "org/local" not in env["sources"]
    assert {"numpy", "transformers", "triton"} <= set(env["versions"]) and isinstance(env["env_vars"], dict)
    if hasattr(os, "sched_getaffinity"):
        assert env["cpu_affinity_list"] == pin.compact(os.sched_getaffinity(0))
        assert env["cpu_affinity"] == len(os.sched_getaffinity(0)) and isinstance(env["cpu_numa_nodes"], list)


def test_the_package_times_five_queries_on_the_headline_cells_and_chains_every_job():
    lines = _dry()
    jobs = [_job(line) for line in lines]
    for line in lines:
        job = _job(line)
        if job.startswith("sp-c-") and any(s in line for s in ("costs.perf", "v1.v1_bytes", "reexec.reexec",
                                                                "reexec.llamacpp")):
            assert re.search(r"--(queries|reps) 5\b", line), job
        if job in ("sp-v1-opt13-64", "sp-v1-opt13-2048", "sp-v1-llama7-64", "sp-v1-llama7-2048"):
            assert "--reps 5" in line, job
        if job.startswith("sp-serve-"):
            assert "--reps 10" in line, job
    missing = [t for t in re.findall(r"\btests/\w+\.py", lines[0]) if not (CODE / t).is_file()]
    assert jobs[0] == "sp-smoke" and not missing, missing           # every test file the smoke runs exists
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
    assert len(lines) <= 56 and len(set(jobs)) == len(jobs)
    for prefix in ("sp-smoke", "sp-c-", "sp-v1-", "sp-serve-", "sp-b-", "sp-g-", "sp-gen-", "sp-bt-", "sp-r-",
                   "sp-setup-", "sp-i-", "sp-q-", "sp-tv-", "sp-a-", "sp-l-"):
        assert any(j.startswith(prefix) for j in jobs), prefix
    big = ("opt-1.3b", "opt-6.7b", "llama2-13b", "qwen3-4b", "facebook/opt-1.3b", "facebook/opt-6.7b")
    for line in lines:
        assert not any(f"--model {m} " in line for m in big), _job(line)
        for cmd in re.split(r"&&|;", line):                    # a 7B checkpoint lends its tokenizer to a smoke model
            if re.search(r"--model (openlm-research|meta-llama)/", cmd):
                assert "--smoke" in cmd, _job(line)
        if "--model llama2-7b" in line:            # the long tier's one block at 4,608 tokens
            assert "--layers 1" in line, _job(line)
        assert "--time=" in line


FLAGS = ("--real", "--no-wire", "--codec-only", "--exp-only", "--client-memory")
"""The switches that select a script's path: a checkpoint, the encoding off, V1's codec profile only, the
per-token-exponent cells only, the client's own process."""
VALUED = ("--recipe", "--prefix", "--v1-verifier", "--split")
"""The options whose values select a path: the recipe, the fixed prefix, V1's verifiers, the evaluated split."""


def _paths(script: str, cmd: str) -> set:
    """The paths one command takes: its verifier's device (``cuda`` with ``--verifier-device cuda``, else ``cpu``),
    each switch of :data:`FLAGS`, each option of :data:`VALUED` with its values, and ``--model`` for
    ``sampler_tv`` (a real model's logits; without it, its toy distributions only)."""
    words = cmd.replace('"', " ").split()
    out = {"cuda" if "--verifier-device cuda" in cmd else "cpu"}
    for i, w in enumerate(words):
        if w in FLAGS or (w == "--model" and script == "generation.sampler_tv"):
            out.add(w)
        elif w in VALUED:
            values = []
            for v in words[i + 1:]:
                if v.startswith("--"):
                    break
                values.append(v)
            out.add(" ".join([w, *values]))
    return out


def _scripts(lines: list[str]) -> dict:
    """``{tier: {(script, path)}}`` of a dry run: the script of every command of a job (``-m
    experiments.<section>.<name>``) with each path it takes (:func:`_paths`); the node check that starts every job
    (``experiments.package.node``) is not a tier's script."""
    out: dict = {}
    for line in lines:
        tier = tier_of(_job(line)[len("sp-"):])
        for cmd in re.split(r"&&|;", line.split("--wrap", 1)[1]):
            m = re.search(r"-m experiments\.(\w+\.\w+)", cmd)
            if m and m.group(1) != "package.node":
                out.setdefault(tier, set()).update((m.group(1), p) for p in _paths(m.group(1), cmd))
    return out


def test_the_rehearsal_runs_every_script_of_every_tier():
    """Every script a tier's full cells run, on each path they take it (the verifier's device, a checkpoint, the
    encoding off, the codec profile, the recipe, the fixed prefix, V1's verifiers, sampler_tv on a model), is in the
    tier's rehearsal cells, so that no script meets a path for the first time in the final run."""
    assert _paths("generation.sampler_tv", 'x --model m --recipe legacy --v1-verifier optimised baseline --no-wire"') \
        == {"cpu", "--model", "--recipe legacy", "--v1-verifier optimised baseline", "--no-wire"}
    full, rehearsal = _scripts(_dry()), _scripts(_dry("--rehearsal"))
    assert set(full) == set(TIERS)
    for tier in TIERS:
        missing = full[tier] - rehearsal.get(tier, set())
        assert not missing, (tier, sorted(missing))


def test_the_cost_cells_use_11_bit_probabilities_but_the_a_b_of_the_scales():
    """No job passes ``--prob-max`` (every script's default is the certified models' 2047, ``cli.PAPER``) but the
    improve tier's A/B, which times 255 and 2047 on the same queries in modes Kpre and C with the CPU verifier: two
    Llama-2-7B blocks at 2,048 tokens in the full run, two GPT-2 blocks in the rehearsal."""
    from experiments.common import cli
    assert cli.PAPER.prob_max == 2047
    for flags, job, cell in (((), "sp-i-probbits-llama2-7b", "--model llama2-7b --layers 2 --seq 2048"),
                             (("--rehearsal",), "sp-i-probbits-gpt2", "--model gpt2 --layers 2 --seq 64")):
        lines = {_job(line): line for line in _dry(*flags)}
        assert [j for j, line in lines.items() if "--prob-max" in line] == [job]
        ab = lines[job].split("costs.perf", 1)[1]
        assert cell in ab and "--modes Kpre C" in ab and "--prob-max 255 2047" in ab and "--label prob-bits" in ab
        assert "--verifier-device cpu" in ab and "--require-native" in ab


def test_the_gen_and_quality_cells_of_the_text():
    """The gen tier's cells (GPT-2's per-token prover after a 64-token prompt; Llama-2-7B and Qwen3-4B after a
    512-token prompt, greedy and sampled) and the quality tier's: the OPT-125M recipe studies (the alpha study without
    the certified cell, alpha 0.6, which q-opt125 measures) and attribution, Qwen3-4B's per-token-exponent cells
    without the main cell the job's first command measured, the reproducible build (OPT-125M with 1 and 8 threads,
    OPT-1.3B's commitment on the GPU and on the CPU) and, with --llama-licence, the Llama-2 grids of the text
    (macros.LLAMA_GRIDS) on the test split (with the structural diff) and the validation split, Llama-2-13B against
    fp16, and no OpenLLaMA-7B job; the rehearsal keeps its own small generation and attribution cells and runs the
    grid path on a Llama-2-7B smoke model (OpenLLaMA-7B's without --llama-licence)."""
    from experiments.paper.macros import DIGESTS, LLAMA_GRIDS
    lines = {_job(line): line for line in _dry()}
    gpt2 = lines["sp-gen-gpt2"].split("generation.gen_bench", 1)[1]
    assert "--model gpt2 --prompt 64 --gen 256 --decode greedy --modes C Kpre --policy cnn18c" in gpt2
    assert "--layers" not in gpt2 and "--label gen" in gpt2
    for m in ("llama2-7b", "qwen3-4b"):
        assert f"--model {m} --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre" in lines[f"sp-gen-{m}"]
    rec = lines["sp-q-opt125-recipes"]
    assert "--model facebook/opt-125m --recipe legacy --graphs smooth --pbits 11" in rec
    assert "--model facebook/opt-125m --graphs smooth --alphas 0.3 0.4 0.5 0.7 --pbits 11" in rec
    assert re.search(r"quality\.ppl_opt --device cuda --model facebook/opt-125m\"", lines["sp-q-opt125"])
    assert "--alphas 0.3 0.4 0.5 0.6 0.7 --pbits 11 --draws 0 1" in lines["sp-q-opt13-recipes"]   # draw 1's 0.6
    first, second = lines["sp-q-qwen4"].split("--wrap", 1)[1].split(" && ")[-2:]
    assert "--glu-exps" not in first and "--glu-exps 8 --exp-only" in second
    # one job per model, split and alpha (each with every balance of its grid), whose limit leaves room over its
    # estimate (about 4 h); together they are exactly the grid
    for (model, _, alphas, balances), job in zip(LLAMA_GRIDS, ("sp-q-llama7", "sp-q-llama13")):
        for split, suffix in (("test", ""), ("validation", "-validation")):
            jobs = sorted(j for j in lines if re.fullmatch(rf"{job}{suffix}-a[0-9.]+", j))
            assert jobs == sorted(f"{job}{suffix}-a{a:g}" for a in alphas), (job, split, jobs)
            for a in alphas:
                line = lines[f"{job}{suffix}-a{a:g}"]
                cmd = line.split("--wrap", 1)[1]
                assert cmd.count("quality.ppl_llama") == 1 and f"--model {model}" in cmd, line
                assert re.search(rf"--alphas {a:g}( |\")", cmd), line
                if len(balances) > 1:
                    assert f"--glu-smooths {' '.join(f'{g:g}' for g in balances)}" in cmd, line
                else:
                    assert "--glu-smooths" not in cmd, line
                assert ("--split validation" in cmd) == (split == "validation"), line
                assert ("--structure" in cmd) == (split == "test"), line
                assert ("--ref-dtype float16" in cmd) == (model == "meta-llama/Llama-2-13b-hf"), line
                hours = int(re.search(r"--time=(\d+):00:00", line)[1])
                assert 6 <= hours <= 8, line
    assert not any("openllama" in j for j in lines)
    builds = re.findall(r"quality\.digest_check (--model \S+ --threads \d+[^&\"]*)", lines["sp-q-digest"])
    assert [b.strip() for b in builds] == [
        "--model facebook/opt-125m --threads 1", "--model facebook/opt-125m --threads 8 --commit --commit-device cuda",
        "--model facebook/opt-1.3b --threads 8 --commit --commit-device cuda",
        "--model facebook/opt-1.3b --threads 8 --commit --commit-device cpu"]
    assert [m for m, _, _ in DIGESTS] == ["facebook/opt-125m", "facebook/opt-1.3b"]
    rehearsal = {_job(line): line for line in _dry("--rehearsal")}
    assert "--layers 2" in rehearsal["sp-gen-gpt2"] and "sp-q-opt125-recipes" not in rehearsal
    access = rehearsal["sp-q-llama-access"]
    assert "--model meta-llama/Llama-2-7b-hf --smoke --split validation" in access and "sp-q-openllama7-smoke" \
        not in rehearsal
    assert "--alphas 0.45 0.55 --glu-smooths 0.1 0.2 --ref-dtype float16" in access
    # the attribution of OPT-125M's perplexity (tab:quality-attr): the script's defaults are the paper's settings (16
    # windows of 128 tokens, the scalar graph with norm gain 4 and 8-bit probabilities, the feature sub-rows), on the
    # CPU and not pinned; the rehearsal's cell takes two windows without the sub-rows
    attr = lines["sp-q-opt125-attr"].split("--wrap", 1)[1]
    assert attr.endswith('experiments.quality.attribution --device cpu --threads 8 --model facebook/opt-125m"'), attr
    assert "experiments.package.pin" not in attr and "--time=1:00:00" in lines["sp-q-opt125-attr"]
    assert "--mem=64G" in lines["sp-q-opt125-attr"]
    assert rehearsal["sp-q-opt125-attr"].endswith("--model facebook/opt-125m --windows 2 --skip-sub\"")
    assert [j for j, x in lines.items() if "experiments.quality.attribution" in x] == ["sp-q-opt125-attr"]


def test_every_tier_is_documented_and_reachable():
    text = (REPO / RUN).read_text()
    for tier in (*TIERS, "qllama", "zkllm", "gate"):
        assert f"tier_{tier}()" in text and re.search(rf"^#\s+{tier}\s", text, re.M), tier


def test_the_gate_tier_checks_the_full_sizes_once_before_the_run():
    """The gate tier: never in 'all' and refused with --rehearsal; its records land in a tier of their own.  Every V1
    cell runs the whole query (--full) at the default lmax, and its GKR instances reach 2^lmax leaves in each mode
    (from the shapes, as the cut packs them): two Llama-2-7B blocks at 2,048 tokens in modes Kpre and C with both CPU
    verifiers and with the GPU verifier, honest and tampered queries; OPT-1.3B at 2,048 tokens, whose instances hold
    exactly 2^lmax leaves; the certified OPT-125M (real weights); the full Llama-2-7B at 2,048 tokens.  Its long cells
    take 16,384 tokens, the long tier's largest prompt; the CPU cells are pinned and need the native kernels."""
    from experiments.paper.tables import v1_shape_model
    from pvi.v1.cut import LMAX
    assert not any(_job(line).startswith("sp-gate-") for line in _dry())
    runs: list = []
    _dry("--rehearsal", tiers=("smoke", "gate"), status=2, out=runs)
    assert "without --rehearsal" in runs[0].stderr
    jobs = {_job(line): line.split("--wrap", 1)[1] for line in _dry(tiers=("gate",))}
    assert jobs and {tier_of(job[len("sp-"):]) for job in jobs} == {"gate"}
    v1 = {job: cmd for job, cmd in jobs.items() if "v1.v1_bytes" in cmd}
    exact = set()
    for job, cmd in v1.items():
        assert "--full" in cmd and "--lmax" not in cmd and "--codec-only" not in cmd and "--tampers" in cmd, job
        model, seq = re.search(r"--model (\S+)", cmd).group(1), int(re.search(r"--seq (\d+)", cmd).group(1))
        layers = re.search(r"--layers (\d+)", cmd)
        for mode in re.search(r"--modes ([A-Za-z ]+?) --", cmd).group(1).split():
            shape = v1_shape_model(model, seq, LMAX, "cnn18c" if mode == "C" else None,
                                   int(layers.group(1)) if layers else None)
            assert shape["max_bits"] == LMAX, (job, mode)
            if max(shape["instance_leaves"]) == 1 << LMAX:
                exact.add((model, mode))
    assert exact == {("opt-1.3b", "Kpre"), ("opt-1.3b", "C")}
    two = v1["sp-gate-v1-llama7-2k-2"]
    assert "--layers 2 --modes Kpre C" in two and "--v1-verifier optimised baseline" in two
    assert "--verifier-device cuda" in v1["sp-gate-g-v1-llama7-2k-2"]
    assert "--real facebook/opt-125m" in v1["sp-gate-r-opt125-v1-2k"]
    assert "--layers" not in v1["sp-gate-v1-llama7-2k"] and "--model llama2-7b --seq 2048" in v1["sp-gate-v1-llama7-2k"]
    long = {job: cmd for job, cmd in jobs.items() if job.startswith("sp-gate-l-")}
    assert len(long) == 2 and all("--seq 16384" in cmd and "--layers 2" in cmd for cmd in long.values())
    assert "--max-pos 16384" in long["sp-gate-l-rx-llama7-16k"] and " int_mm " in long["sp-gate-l-rx-llama7-16k"]
    for job, cmd in jobs.items():
        cpu = "--verifier-device cuda" not in cmd
        assert ("--require-native" in cmd and "experiments.package.pin" in cmd) == cpu, job


def test_the_cpuv_tier_measures_the_clients_own_memory():
    """Llama-2-7B's CPU-verifier cells (64 and 2,048 tokens, modes Kpre and C) check their timed proofs again in the
    client's own process (perf.py --client-memory), and the rehearsal's cell takes the same path."""
    full = {_job(line): line for line in _dry()}
    assert all("--client-memory" in full[f"sp-c-llama2-7b-{s}"] for s in (64, 2048))
    assert "--modes Kpre C" in full["sp-c-llama2-7b-2048"]
    assert "--client-memory" in {_job(line): line for line in _dry("--rehearsal")}["sp-c-gpt2-64"]


def test_perf_checks_the_timed_proofs_again_in_the_clients_own_process(monkeypatch, capsys, tmp_path):
    """``perf.py --client-memory`` on a tiny decoder: a separate process checks every timed proof again from its
    bytes, with the in-process verdicts, and the record keeps that process's peak resident set and its resident set
    after the imports (Linux) and nothing of the key or the proofs is left in ``--client-dir``; the option refuses a
    GPU verifier and mode C without Fiat-Shamir."""
    status, lines = _cli(perf, monkeypatch, capsys, "perf.py", "--model", "tiny-llama", "--seq", "12", "--modes", "C",
                         "Kpre", "--queries", "2", "--device", "cpu", "--threads", "1", "--policy", "auto",
                         "--client-memory", "--client-dir", str(tmp_path))
    assert status == 0 and sorted(r["mode"] for r in lines) == ["C", "Kpre"]
    for r in lines:
        client = r["client_memory"]
        assert client["agree"] and client["queries"] == client["accepted"] == r["accepted"] == 2, client
        assert "error" not in client and client["verify_s"] > 0
        if os.path.exists("/proc/self/status"):
            assert 0 < client["baseline_rss"] <= client["peak_rss"] and client["key_rss"] <= client["peak_rss"]
        else:
            assert client["peak_rss"] is None
    assert not list(tmp_path.iterdir())
    for argv in (("--verifier-device", "cuda"), ("--modes", "C", "--no-fs")):
        monkeypatch.setattr(sys, "argv", ["perf.py", "--model", "tiny-llama", "--client-memory", *argv])
        with pytest.raises(SystemExit, match="--client-memory"):
            perf.main()


def test_the_zkllm_tier_runs_zkllms_demo_with_the_llama_licence_only(tmp_path):
    """The optional zkllm tier: in 'all' with --optional (after long), only with --llama-licence (else no job and a
    message); zkLLM's build, then Llama-2-7B and Llama-2-13B at 2,048 tokens (one job each, every length required),
    then each model at the other lengths of --zkllm-seqs (default 64, 512 and 1,024 tokens; one job each, a length
    that zkLLM's demo does not prove allowed to fail), 3 repetitions of layers 0 and 1, in logs/<name>/zkllm/
    (whole-node jobs, not pinned); the bench tier's interactive mode-C cells of both models at the same other lengths;
    --zkllm-seqs changes both; the rehearsal's one job builds for Llama-2-7B and proves one layer once at 2,048
    tokens; --zkllm-home, --zkllm-scratch, --zkllm-python and --cuda-home reach the jobs; --one-job zkllm is one
    whole-node job; the pre-flight names what the build lacks."""
    runs: list = []
    jobs = {_job(line): line.split("--wrap", 1)[1] for line in _dry(out=runs)}
    names = [j for j in jobs if j.startswith("sp-zk-")]
    assert names == ["sp-zk-build", "sp-zk-llama2-7b", "sp-zk-llama2-13b", "sp-zk-llama2-7b-short",
                     "sp-zk-llama2-13b-short"] and list(jobs)[-5:] == names
    assert runs[0].stdout.splitlines()[-1].startswith("# smoke cpuv ") and " long zkllm: " in runs[0].stdout
    assert "# tier zkllm: 5 jobs, job limits 14 h, estimate 6-9 h" in runs[0].stdout
    assert "run_zkllm.sh build --home " in jobs["sp-zk-build"] and "/logs/final/zkllm/home" in jobs["sp-zk-build"]
    for size, job in ((7, "sp-zk-llama2-7b"), (13, "sp-zk-llama2-13b")):
        assert "run_zkllm.sh prove --home " in jobs[job] and "/logs/final/zkllm" in jobs[job]
        assert f"--size {size} --seqs 2048 --reps 3 --layers 0,1\"" in jobs[job]
        short = jobs[f"{job}-short"]
        assert "run_zkllm.sh prove --home " in short and "/logs/final/zkllm" in short
        assert f"--size {size} --seqs 64,512,1024 --reps 3 --layers 0,1 --allow-failed-lengths" in short
    assert all("experiments.package.pin" not in jobs[j] for j in names)
    for m in ("llama2-7b", "llama2-13b"):
        ours = jobs[f"sp-b-ci-{m}-zk"]
        assert f"--model {m} --seq 64 512 1024 --builds full --queries 3 --lams 128 --modes C:int --policy" in ours
        assert "experiments.package.pin" in ours and "--require-native" in ours      # a CPU-verifier cell
    full = {_job(line): line for line in _dry(tiers=("zkllm",))}
    assert "--time=2:00:00" in full["sp-zk-llama2-7b"] and "--time=4:00:00" in full["sp-zk-llama2-13b"]
    assert "--time=2:00:00" in full["sp-zk-llama2-7b-short"] and "--time=4:00:00" in full["sp-zk-llama2-13b-short"]
    bench = {_job(line): line for line in _dry(tiers=("bench",))}
    assert "--time=2:00:00" in bench["sp-b-ci-llama2-7b-zk"] and "--time=3:00:00" in bench["sp-b-ci-llama2-13b-zk"]
    # --zkllm-seqs: other lengths for both sides; without 2,048 no 2,048-token job; 2,048 alone: no short job
    got = {_job(line): line for line in _dry("--zkllm-seqs", "256,2048,4096", tiers=("zkllm", "bench"))}
    assert "--seqs 256,4096 " in got["sp-zk-llama2-13b-short"] and "--seq 256 4096 " in got["sp-b-ci-llama2-7b-zk"]
    got = {_job(line) for line in _dry("--zkllm-seqs", "512", tiers=("zkllm",))}
    assert got == {"sp-zk-build", "sp-zk-llama2-7b-short", "sp-zk-llama2-13b-short"}
    got = {_job(line) for line in _dry("--zkllm-seqs", "2048", tiers=("zkllm", "bench"))}
    assert "sp-zk-llama2-7b" in got and not any(j.endswith("-short") or j.startswith("sp-b-ci-") for j in got)
    bad = subprocess.run([_bash(), str(RUN), "--platform", "final", "--dry-run", "--llama-licence", "--zkllm-seqs",
                          "64,x", "zkllm"], cwd=REPO, env={**os.environ, "HF_HOME": "/tmp/hf", "CXX": "g++"},
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "--zkllm-seqs" in bad.stderr
    # without --llama-licence: nothing, and why
    out = subprocess.run([_bash(), str(RUN), "--platform", "final", "--dry-run", "--optional", "zkllm"], cwd=REPO,
                         env={**os.environ, "HF_HOME": "/tmp/hf", "CXX": "g++"}, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0 and "sbatch" not in out.stdout and "--llama-licence" in out.stderr, out.stderr
    reh = {_job(line): line for line in _dry("--rehearsal", tiers=("zkllm",))}
    assert list(reh) == ["sp-zk-rehearsal"]
    assert "build --home " in reh["sp-zk-rehearsal"] and "--models 7 && " in reh["sp-zk-rehearsal"]
    assert "--size 7 --seqs 2048 --reps 1 --layers 0" in reh["sp-zk-rehearsal"]
    opts = {_job(line): line for line in _dry("--zkllm-home", "/zk/home", "--zkllm-scratch", "/zk/scratch",
                                              "--zkllm-python", "/zk/python3", "--cuda-home", "/zk/cuda",
                                              tiers=("zkllm",))}
    assert "--home /zk/home --python /zk/python3 --cuda-home /zk/cuda" in opts["sp-zk-build"]
    assert "--home /zk/home --out " in opts["sp-zk-llama2-13b"]
    assert "--cuda-home /zk/cuda --scratch /zk/scratch --size 13" in opts["sp-zk-llama2-13b"]
    # one whole-node job
    status, out_, err, _ = _submit(tmp_path / "a", "--dry-run", "--platform", "final", "--llama-licence",
                                   "--one-job", "zkllm")
    assert status == 0 and "-J sp-node-zkllm " in out_ and "--exclusive --mem=0" in out_, err
    assert "# one job: zkllm: 5 jobs, job limits 14 h, time limit 14 h" in out_, out_
    # the pre-flight: no CUDA compiler and no token stop the submission
    status, _, err, jobs_ = _submit(tmp_path / "b", "--platform", "final", "--llama-licence", "--cuda-home",
                                    "/no/cuda", "zkllm", extra={"HF_TOKEN": ""})
    assert status == 2 and not jobs_ and "no CUDA compiler (nvcc)" in err and "export HF_TOKEN=" in err, err
    # a home that holds the build and its environment needs neither the compilers nor the interpreter
    home = tmp_path / "c" / "root" / "logs" / "final" / "zkllm" / "home"
    (home / "zkllm").mkdir(parents=True)
    (home / "env" / "bin").mkdir(parents=True)
    (home / "zkllm" / "self-attn.real").write_text("")
    (home / "env" / "bin" / "python").write_text("#!/bin/bash\n", newline="\n")
    (home / "env" / "bin" / "python").chmod(0o755)
    status, _, err, jobs_ = _submit(tmp_path / "c", "--platform", "final", "--llama-licence", "--cuda-home",
                                    "/no/cuda", "zkllm", extra={"HF_TOKEN": "hf_x"})
    assert status == 0 and len(jobs_) == 5, err
