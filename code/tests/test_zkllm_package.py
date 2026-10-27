"""zkLLM's demo on the run's GPU (``experiments/comparison/zkllm/``, the run package's ``zkllm`` tier).

* ``summarise.py`` gives the stored runs' figures (26.37 s and 65.71 s per layer, 844 s and 2,628 s for the whole
  models), their records (each verifier message once per proved layer, no failed binary: ``ok``) and their
  environment (the L40S, the EPYC 9334, zkLLM's commit);
* ``run_zkllm.sh prove`` with stand-ins for zkLLM's scripts and binaries, the GPU tools and GNU time: the run's
  folder holds what the stored runs hold, ``summary.csv`` and the record are written, the scratch folder is deleted,
  and a run in which a layer's verifier did not report is not ``ok`` (exit status 1, and ``records.checks`` reports
  it); several prompt lengths share one setup, and a length at which a proof binary fails is recorded as failed
  (with zkLLM's error line, ``required`` false under ``--allow-failed-lengths``, which ``records.checks`` accepts)
  while the other lengths are proved;
* the record is read as a ``zkllm`` record of the ``zkllm`` tier.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.comparison.zkllm import summarise as S
from experiments.paper import records as R

CODE = Path(__file__).resolve().parents[1]
PKG = CODE / "experiments" / "comparison" / "zkllm"
STORED = CODE / "artifacts" / "results" / "zkllm_l40s"


def test_summarise_gives_the_stored_runs_figures_and_records():
    assert round(S.per_layer_s(STORED / "llama2-7b-T2048"), 2) == 26.37
    assert round(S.per_layer_s(STORED / "llama2-13b-T2048"), 2) == 65.71
    for run, whole in (("llama2-7b-T2048", 843.84), ("llama2-13b-T2048", 2628.4)):
        rec = S.record(STORED / run)
        assert rec["kind"] == "zkllm" and rec["ok"] and rec["whole_model_s"] == whole and rec["seq"] == 2048
        assert rec["verdicts"] == {"qkv": 6, "attn": 6, "ffn": 6} and rec["binary_failures"] == 0
        env = rec["env"]
        assert (env["gpu"], env["cpu"]) == ("NVIDIA L40S", "AMD EPYC 9334 32-Core Processor")
        assert env["zkllm_commit"].startswith("993311e")
        assert R.kind_of(rec) == "zkllm" and R.tier_of("zk-llama2-7b") == "zkllm"
        assert not R.checks(R.Rec("zkllm", "zk-x", "1", "x", 1, "zkllm", rec))
    bad = dict(S.record(STORED / "llama2-7b-T2048"), ok=False)
    assert R.checks(R.Rec("zkllm", "zk-x", "1", "x", 1, "zkllm", bad))
    assert S.run_name(Path("llama2-13b-T2048-12")) == ("llama2-13b", 2048)


FAKE_PY = r"""#!/bin/bash
# stand-in for zkLLM's environment's python: zkLLM's scripts append what its shims would log
log() { echo "BIN $1 0.50 s  maxrss 1000 KB  exit ${3:-0}  args $2" >> "$ZK_TIMELOG"; }
arg() { local k=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$k" ] && { echo "$2"; return; }; shift; done; }
case "$1" in
  *summarise.py) exec "$REAL_PY" "$@" ;;
  -c) echo 1.0 ;;
  llama-ppgen.py) mkdir -p "zkllm-workdir/Llama-2-${2}b" && echo pp > "zkllm-workdir/Llama-2-${2}b/pp.bin"
                  echo x >> "$FAKE_SETUPS"; log ppgen 131072 ;;
  llama-commit.py) log commit-param x ;;
  *mkinput.py) echo input > "$5" ;;
  llama-rmsnorm.py) log rmsnorm "$4"; echo x > "$(arg --output_file "$@")" ;;
  llama-self-attn.py) log self-attn linear
                      if [ "${FAKE_FAIL_SEQ:-}" = "$4" ]; then log self-attn attn 134
                        echo "terminate called after throwing an instance of 'std::runtime_error'"
                        echo "  what():  N is not power of 2, or D is not divisible by N"; exit 0; fi
                      log self-attn attn
                      echo "QKV linear proof successfully verified!"
                      [ "${FAKE_SILENT_LAYER:-}" = "$3" ] || echo "Self attention proof successfully verified!" ;;
  llama-skip-connection.py) [ -f "$(arg --block_output_file "$@")" ] || { echo "Input or output file does not exist"
                            exit 1; }
                            log skip-connection x; echo x > "$(arg --output_file "$@")" ;;
  llama-ffn.py) log ffn x; echo "SwiGLU proof complete."; echo x > "$(arg --output_file "$@")" ;;
  *) echo "fake python: $*" >&2; exit 3 ;;
esac
"""
FAKE_TOOLS = {
    "nvidia-smi": r"""#!/bin/bash
case "$*" in
  *compute_cap*) echo 8.9 ;;
  *-lms*) echo "timestamp, memory.used [MiB], utilization.gpu [%]"
          while :; do echo "2026/01/01 00:00:00.000, 1000 MiB, 50 %"; sleep 0.2; done ;;
  *name,driver_version*) echo "NVIDIA L40S, 535.183.01, 46068 MiB" ;;
  *) echo "GPU 0: NVIDIA L40S" ;;
esac
""",
    "lscpu": '#!/bin/bash\necho "Model name:                           AMD EPYC 9334 32-Core Processor"\n',
    "gnutime": r"""#!/bin/bash
# stand-in for GNU time -v -o FILE CMD...
shift; out=$2; shift 2
"$@"; rc=$?
printf '\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.50\n\tExit status: %s\n' "$rc" > "$out"
exit $rc
""",
}


def _home(tmp_path: Path) -> tuple[Path, Path]:
    """A zkLLM home with a built, shimmed checkout (a git repository) and a stand-in environment; the tools' folder."""
    if shutil.which("bash") is None or shutil.which("git") is None:
        pytest.skip("no bash or git")
    home, tools = tmp_path / "home", tmp_path / "bin"
    z = home / "zkllm"
    (home / "env" / "bin").mkdir(parents=True)
    z.mkdir(parents=True)
    tools.mkdir()
    for name, text in {**FAKE_TOOLS, "python": FAKE_PY}.items():
        path = (home / "env" / "bin" / name) if name == "python" else tools / name
        path.write_text(text, newline="\n")
        path.chmod(0o755)
    (z / "Makefile").write_text("all:\n", newline="\n")
    git = ["git", "-C", str(z), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(z)], check=True)
    subprocess.run([*git, "add", "Makefile"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    (z / "self-attn.real").write_text("")
    (home / "BUILD.txt").write_text("arch: sm_89\n", newline="\n")
    (home / "wikitext-2-raw-v1-test.parquet").write_text("")
    return home, tools


def _prove(tmp_path: Path, home: Path, tools: Path, *extra: str, env: dict | None = None):
    out, scratch = tmp_path / "logs" / "zkllm", tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    e = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    e.update(PATH=f"{tools}{os.pathsep}{os.environ['PATH']}", REAL_PY=Path(sys.executable).as_posix(),
             CUDA_HOME=str(tools), FAKE_SETUPS=(tmp_path / "setups.txt").as_posix(), **(env or {}))
    (tools / "bin").mkdir(exist_ok=True)
    (tools / "bin" / "nvcc").write_text("#!/bin/bash\n", newline="\n")
    (tools / "bin" / "nvcc").chmod(0o755)
    cmd = [shutil.which("bash"), (PKG / "run_zkllm.sh").as_posix(), "prove", "--home", home.as_posix(), "--size", "7", "--out",
           out.as_posix(), "--scratch", scratch.as_posix(), "--need-gb", "0", "--time-cmd",
           (tools / "gnutime").as_posix(), *extra]
    run = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=300)
    return run, out, scratch


def test_prove_writes_the_runs_folder_summary_and_record(tmp_path):
    home, tools = _home(tmp_path)
    run, out, scratch = _prove(tmp_path, home, tools, "--seq", "2048", "--reps", "2", "--layers", "0,1")
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    r = out / "llama2-7b-T2048"
    for name in ("bin_times.log", "env.txt", "verdicts.txt", "run.json", "logs.tar.gz", "nvidia-smi.csv.gz",
                 "time-ppgen.txt", "time-commit.txt", "time-input.txt", "time-r2-L1-ffn.txt", "workdir-size.txt"):
        assert (r / name).exists(), name
    assert not list(r.glob("*.bin")) and not list(r.glob("log-*.txt"))
    assert not list(scratch.iterdir())                                 # the weights' folder is deleted
    rec = json.loads([s for s in run.stdout.splitlines() if s.startswith("{")][-1])
    assert rec["ok"] and rec["layers_expected"] == 4 and rec["layers_proved"] == 4 and rec["reps"] == 2
    assert rec["verdicts"] == {"qkv": 4, "attn": 4, "ffn": 4} and rec["layer_ids"] == [0, 1]
    # six binaries per layer ran (the first skip connection cannot), 0.5 s each, the skip counted twice: 3.5 s
    assert rec["per_layer_s"] == 3.5 and rec["whole_model_s"] == 112.0
    assert rec["env"]["gpu"] == "NVIDIA L40S" and rec["env"]["arch"] == "sm_89"
    assert rec["env"]["zkllm_unchanged"] == "yes" and rec["env"]["cpu"] == "AMD EPYC 9334 32-Core Processor"
    assert rec["required"] and rec["errors"] == []
    rows = (out / "summary.csv").read_text().splitlines()
    assert rows[0] == "model,seq,layers,run,per_layer_s,whole_model_s,ok"
    assert rows[1] == "llama2-7b,2048,32,llama2-7b-T2048,3.5000,112.00,true"


def test_prove_shares_the_setup_across_lengths_and_records_a_failed_length(tmp_path):
    """Three lengths in one job: one setup (ppgen and commit once, in the first length's folder); the length at which
    a proof binary fails stops after its first repetition and is recorded as failed, with zkLLM's error line, and the
    lengths after it are proved; with --allow-failed-lengths the exit status is 0 and the failed record is not
    required (records.checks accepts it), without it the exit status is 1 and the record is a failed check."""
    for allow in (True, False):
        base = tmp_path / ("allow" if allow else "strict")       # a fresh home and run folder for each
        base.mkdir()
        home, tools = _home(base)
        extra = ("--allow-failed-lengths",) if allow else ()
        run, out, scratch = _prove(base, home, tools, "--seqs", "64,512,1024", "--reps", "2", "--layers", "0,1",
                                   *extra, env={"FAKE_FAIL_SEQ": "512"})
        assert run.returncode == (0 if allow else 1), run.stdout[-2000:] + run.stderr[-2000:]
        assert (base / "setups.txt").read_text().count("x") == 1          # one setup for the three lengths
        assert (out / "llama2-7b-T64" / "time-commit.txt").exists()
        assert not (out / "llama2-7b-T1024" / "time-commit.txt").exists()
        assert "setup: llama2-7b-T64" in (out / "llama2-7b-T1024" / "env.txt").read_text()
        recs = {r["seq"]: r for r in (json.loads(s) for s in run.stdout.splitlines() if s.startswith("{"))}
        assert sorted(recs) == [64, 512, 1024]
        assert recs[64]["ok"] and recs[1024]["ok"] and recs[1024]["layers_proved"] == 4
        bad = recs[512]
        assert not bad["ok"] and bad["binary_failures"] == 2 and bad["layers_proved"] == 2 and bad["reps"] == 2
        assert any("D is not divisible by N" in e for e in bad["errors"]), bad["errors"]
        assert all(r["required"] is not allow for r in recs.values())
        assert bool(R.checks(R.Rec("zkllm", "zk-x", "1", "x", 1, "zkllm", bad))) is not allow
        rows = (out / "summary.csv").read_text().splitlines()
        assert [r.split(",")[1::5] for r in rows[1:]] == [["1024", "true"], ["512", "false"], ["64", "true"]]
        assert not list(scratch.iterdir()) and not list(out.glob("*/*.bin"))
    # at no length proved, --allow-failed-lengths still gives exit status 1
    base = tmp_path / "none"
    base.mkdir()
    home, tools = _home(base)
    run, _, _ = _prove(base, home, tools, "--seqs", "512", "--reps", "1", "--layers", "0",
                       "--allow-failed-lengths", env={"FAKE_FAIL_SEQ": "512"})
    assert run.returncode == 1, run.stderr[-2000:]


def test_prove_fails_when_a_layer_is_not_verified(tmp_path):
    home, tools = _home(tmp_path)
    run, _, _ = _prove(tmp_path, home, tools, "--reps", "1", "--layers", "0,1", env={"FAKE_SILENT_LAYER": "1"})
    assert run.returncode == 1, run.stderr[-2000:]
    rec = json.loads([s for s in run.stdout.splitlines() if s.startswith("{")][-1])
    assert not rec["ok"] and rec["verdicts"]["attn"] == 1


def test_prove_refuses_a_home_without_a_build(tmp_path):
    home, tools = _home(tmp_path)
    (home / "zkllm" / "self-attn.real").unlink()
    run, _, _ = _prove(tmp_path, home, tools)
    assert run.returncode == 2 and "run_zkllm.sh build --home" in run.stderr
