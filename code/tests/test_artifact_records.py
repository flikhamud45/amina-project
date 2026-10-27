"""The artifact-evaluation records of the run: what each job leaves besides its numbers.

* The proof's fingerprint: ``run_query`` returns, under Fiat--Shamir, the transcript's digest (mode Kpre's from its
  one message), equal on two runs of the same query, different on another query, the same with the streaming
  verifier, None for interactive challenges; V1's ``run_cut_query`` likewise; ``perf.py`` records each timed query's
  digest, totals and parts and the warm-up, ``records.roots_sha256`` fingerprints ``C_M``.
* ``comparison/proof_digests.py``: the same cells proved on two devices agree, a record carries every query's
  digest, and ``--compare`` finds equal and differing runs; ``paper/macros.hardware_digests`` counts the cells and
  names the devices, and a differing cell is a text check that fails.
* The node's environment (``package/envsnap.py``): a snapshot folder with its files and ``env.json``, written once
  per tag; the inputs' hashes and the stored networks' check; the fields on which two nodes differ, which
  ``numbers.py --check`` reports as warnings.
* The GPU telemetry (``package/telemetry.py``): the summary of samples (throttle reasons by NVML's bits, power,
  energy), read back from the summary, the CSV or its compressed copy; a sampler on a stand-in for ``nvidia-smi``
  writes its files and stops when its watched process has ended; ``numbers.py --check`` warns on a timed job slowed
  by a power or thermal limit and not on a job that times nothing.
* The GPU's energy counter (``common/energy.py``) through a stand-in for NVML; ``serve_ab``'s energy per call.
* ``trained_mlp``'s weights' SHA-256 in the records' sources; the benchmark's digest rows; the raw timings of
  ``reexec.py``, ``setup_timing.py`` and ``gen_bench.py``.
"""
from __future__ import annotations

import gzip
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.common import checkpoints, records
from experiments.common.models import decoder_config
from experiments.comparison import proof_digests as PD
from experiments.package import envsnap, telemetry
from experiments.paper import macros as M
from experiments.paper import numbers as N
from experiments.paper import records as R
from experiments.paper import tables as T
from experiments.paper.bench import Bench

CODE = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _keep_the_options():
    """The scripts' ``main`` sets the implementation options and torch's threads for the process: restored after
    each test, so that no later test sees them."""
    from pvi import options
    before, threads = options.current(), torch.get_num_threads()
    yield
    options.configure(before)
    torch.set_num_threads(threads)


def _tiny(mode: str, fs: bool = True, stream: bool = False):
    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    cfg = decoder_config("tiny-gpt")
    graph = build_decoder(cfg, n_layers=2, calib_tokens=8, seed=0, prune_last=True, prob_max=2047)
    ops = decoder_shapes(cfg, prune_last=True)
    coms = commit_graph(graph, 4, device="cpu", policy="per-op", model_ops=ops) if mode == "C" else None
    params = params_for(40, len(ops), rate=4, plan=None, fiat_shamir=fs)
    if mode == "C":
        v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                     tables=coms.table_publics, lean=True, stream=stream)
    else:
        v = Verifier(graph.public(), params, mode, weights={o.name: (o.weight, o.bias) for o in graph.mat_ops},
                     lean=True, lookups=True, stream=stream)
        v.precompute(Challenger())
    return Prover(graph, device="cpu", commitments=coms, lean=True), v, coms, cfg


def _prompt(cfg, seed: int) -> torch.Tensor:
    return torch.randint(0, cfg.vocab, (1, 8), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------------------ the proof's fingerprint
@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_a_fiat_shamir_query_returns_the_proofs_fingerprint(mode):
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    prover, v, coms, cfg = _tiny(mode)
    a, b = (run_query(prover, v, _prompt(cfg, 1), wire=True) for _ in range(2))
    other = run_query(prover, v, _prompt(cfg, 2), wire=True)
    assert a["accepted"] and len(a["transcript"]) == 64 and a["transcript"] == b["transcript"]
    assert other["transcript"] != a["transcript"]
    sp, sv, _, _ = _tiny(mode, stream=True)
    assert run_query(sp, sv, _prompt(cfg, 1), wire=True)["transcript"] == a["transcript"]      # the same proof
    ip, iv, _, _ = _tiny(mode, fs=False)
    assert run_query(ip, iv, _prompt(cfg, 1), wire=True)["transcript"] is None                 # random challenges
    ch = Challenger(fiat_shamir=True)
    ch.absorb(b"x", b"1")
    before = ch.transcript_digest()
    assert ch.transcript_digest() == before                    # reading it leaves the transcript as it was
    ch.absorb(b"x", b"2")
    assert ch.transcript_digest() != before
    if coms is not None:
        digest, n = records.roots_sha256(coms)
        assert len(digest) == 64 and n > 0 and records.roots_sha256(coms)[0] == digest


def test_a_v1_query_returns_its_fingerprint():
    from pvi.v1 import protocol as cp
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder
    cfg = decoder_config("tiny-llama")
    graph = build_decoder(cfg, n_layers=2, calib_tokens=8, seed=0, prune_last=True, prob_max=2047)
    params = params_for(40, len(decoder_shapes(cfg, prune_last=True)), rate=4, fiat_shamir=True, cut=True,
                        cut_lmax=20)
    v = Verifier(graph.public(), params, "Kpre", weights={o.name: (o.weight, o.bias) for o in graph.mat_ops},
                 lean=True)
    v.precompute(Challenger())
    prover = Prover(graph, device="cpu", lean=True)
    a, b = (cp.run_cut_query(prover, v, _prompt(cfg, 1)) for _ in range(2))
    assert a["accepted"] and a["transcript"] and a["transcript"] == b["transcript"]


def test_per_query_keeps_every_query():
    runs = [{"timings": {"prove_forward": 1.0}, "transcript": "a"}, {"timings": {"verify_derive": 2.0}}]
    out = records.per_query(runs, [1.0, 0.0], [0.0, 2.0])
    assert out == {"timings_all": {"prove_forward": [1.0, 0.0], "verify_derive": [0.0, 2.0]},
                   "prove_s_all": [1.0, 0.0], "verify_s_all": [0.0, 2.0], "transcripts": ["a", None]}
    assert "transcripts" not in records.per_query([{"timings": {}, "transcript": None}])


def test_the_trained_networks_hash_reaches_the_records(monkeypatch):
    from experiments.common.models import note_file
    monkeypatch.setattr(checkpoints, "SOURCES", {})
    path = CODE / "artifacts" / "models" / "mlp_mnist_full.npz"
    note_file("models/mlp_mnist_full.npz", path)
    note_file("models/missing.pt", CODE / "no" / "such.pt")
    stored = dict(line.split()[::-1] for line in (path.parent / "MODELS.sha256").read_text().splitlines())
    assert checkpoints.SOURCES == {"models/mlp_mnist_full.npz": "sha256:" + stored["mlp_mnist_full.npz"]}
    assert records.environment("cpu")["sources"]["models/mlp_mnist_full.npz"].startswith("sha256:")


def test_the_benchmark_records_the_digests():
    from experiments.costs import bench
    rows = []

    class Rec:
        def rec(self, metric, value, unit="", trial=None, **extra):
            rows.append((metric, value, trial))
    res = {"timings": {"prove_forward": 1.0}, "bytes": {"claims": 4}, "accepted": True, "transcript": "f" * 64}
    bench._record_query(Rec(), res, 3)
    assert ("proof_transcript", "f" * 64, 3) in rows
    rows.clear()
    bench._record_query(Rec(), {**res, "transcript": None}, 0)
    assert not any(m == "proof_transcript" for m, _, _ in rows)
    from experiments.comparison import fingerprint_check
    assert {"proof_transcript", "commit_roots_sha256"} <= fingerprint_check.EXACT


# ------------------------------------------------------------------------------------------ proofs on two devices
def test_proof_digests_on_two_devices_and_the_comparison(tmp_path, capsys):
    argv = ["--cells", "mlp_mnist", "tiny-gpt:2:8", "--v1-cells", "tiny-llama:2:8:Kpre", "--modes", "Kpre",
            "--devices", "cpu", "cpu:0", "--queries", "2", "--lmax", "20", "--threads", "2"]
    assert PD.main(argv) == 0
    recs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [(r["model"], r["mode"], r["protocol"]) for r in recs] == [
        ("mlp_mnist", "Kpre", "v0"), ("tiny-gpt", "Kpre", "v0"), ("tiny-llama", "Kpre", "v1")]
    for r in recs:
        assert r["kind"] == "digests" and r["agree"] and set(r["devices"]) == {"cpu", "cpu:0"}
        assert all(len(t) == 64 for d in r["devices"].values() for t in d["transcripts"])
        assert R.kind_of(r) == "digests"
    log = tmp_path / "a" / "sp-xcheck-1.out"
    log.parent.mkdir()
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    n, problems = PD.compare(PD.load([str(log.parent)]), recs)
    assert n == 3 and not problems
    bad = json.loads(json.dumps(recs[1]))
    bad["devices"]["cpu"]["transcripts"][0] = "0" * 64
    assert PD.compare(recs, [bad])[1] and PD.compare(recs, [{**recs[1], "graph_digest": "x"}])[1]
    assert R.checks(R.Rec("smoke", "xcheck", "1", "x", 1, "digests", {**bad, "agree": False}))


def test_the_hardware_digest_macros():
    def rec(model, gpu="NVIDIA L40S", cpu="AMD EPYC 9334 32-Core Processor", t="a" * 64, agree=True):
        return {"kind": "digests", "model": model, "mode": "Kpre", "protocol": "v0", "layers": None, "seq": None,
                "queries": 1, "fs": True, "gpu": gpu, "cpu": cpu, "agree": agree,
                "devices": {"cuda": {"transcripts": [t]}, "cpu": {"transcripts": [t]}}}
    recs = [R.Rec("smoke", "xcheck", "1", "sp-xcheck-1.out", i + 1, "digests", rec(m))
            for i, m in enumerate(("mlp_mnist", "gpt2"))]
    ctx = T.Ctx(R.Run(recs), Bench(None), {})
    c = M.Collector()
    M.hardware_digests(c, ctx)
    assert c.m["HwDigestCells"].value == "2" and not c.problems
    assert c.m["HwDigestDevices"].value == "the NVIDIA L40S and on the AMD EPYC 9334"
    ctx.xcheck = [rec("gpt2", "NVIDIA GeForce RTX 2080 Ti", "Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz")]
    c = M.Collector()
    M.hardware_digests(c, ctx)
    assert c.m["HwDigestDevices"].value.endswith(", and on a second platform's NVIDIA GeForce RTX 2080 Ti and "
                                                 "Intel Xeon Silver 4114")
    ctx.xcheck = [rec("gpt2", "NVIDIA GeForce RTX 2080 Ti", "Intel Xeon", t="b" * 64)]
    c = M.Collector()
    M.hardware_digests(c, ctx)
    assert c.m["HwDigestCells"].value is None and c.problems and c.problems[0].startswith(M.TEXT_CLAIM)


# ------------------------------------------------------------------------------------------ the node's environment
def test_the_environment_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("PVI_RUN_ID", "77")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    out = tmp_path / "env"
    assert envsnap.main(["--out", str(out)]) == 0
    folders = list(out.iterdir())
    assert len(folders) == 1 and folders[0].name.endswith("-77")
    files = {p.name for p in folders[0].iterdir()}
    assert {"env.json", "pip-freeze.txt", "lscpu.txt", "nvidia-smi-q.txt", "git.txt", "cxx.txt", "cpufreq.txt",
            "numactl.txt", "free.txt", "uname.txt", "environ.txt"} <= files
    d = json.loads((folders[0] / "env.json").read_text())
    assert set(envsnap.COMPARED) <= set(d) and d["torch"] and d["python"]
    (folders[0] / "lscpu.txt").write_text("kept")
    assert envsnap.main(["--out", str(out)]) == 0 and (folders[0] / "lscpu.txt").read_text() == "kept"   # once
    other = out / "n-2-5"
    other.mkdir()
    (other / "env.json").write_text(json.dumps({**d, "driver": "595.99.02"}))
    snaps = envsnap.snapshots(out)
    assert len(snaps) == 2 and list(envsnap.differences(snaps)) == ["driver"]
    summary, warn = N.env_report(tmp_path)
    assert summary and summary[0].startswith("environment: 2 node snapshots")
    assert len(warn) == 1 and warn[0].startswith("WARNING env: the nodes differ in driver")


def test_the_inputs_hashes(tmp_path, monkeypatch):
    code = tmp_path / "code"
    (code / "artifacts" / "models" / "cnn").mkdir(parents=True)
    (code / "artifacts" / "models" / "cnn" / "lenet5.pt").write_bytes(b"weights")
    (code / "artifacts" / "models" / "a.npz").write_bytes(b"a")
    (code / "artifacts" / "models" / "MODELS.sha256").write_text(
        f"{records.file_sha256.__wrapped__(str(code / 'artifacts' / 'models' / 'a.npz'))}  a.npz\n{'0' * 64}  b.npz\n")
    monkeypatch.setattr(envsnap, "CODE", code)
    monkeypatch.delenv("PVI_CACHE", raising=False)
    monkeypatch.delenv("PVI_CIFAR_ROOT", raising=False)
    out = tmp_path / "env"
    assert envsnap.write_inputs(out) == out / "inputs.sha256"
    lines = (out / "inputs.sha256").read_text().splitlines()
    assert lines == [f"{records.file_sha256.__wrapped__(str(code / 'artifacts/models/cnn/lenet5.pt'))}  "
                     "artifacts/models/cnn/lenet5.pt"]
    assert (out / "models-check.txt").read_text().splitlines() == ["a.npz: OK", "b.npz: FAILED"]
    assert envsnap.write_inputs(out) is None                       # once per run folder
    (out / "x-1").mkdir()
    (out / "x-1" / "env.json").write_text(json.dumps({"host": "x"}))
    _, warn = N.env_report(tmp_path)
    assert warn == ["WARNING env: b.npz: FAILED (MODELS.sha256)"]


# ------------------------------------------------------------------------------------------ the GPU's telemetry
ROWS = [{"timestamp": f"2026/10/27 10:00:0{i}.000", "clocks.sm": str(2520 - 100 * (i == 2)), "power.draw": "300",
         "temperature.gpu": str(60 + i), "utilization.gpu": "100",
         "clocks_event_reasons.active": "0x0000000000000004" if i == 2 else "0x0000000000000000"} for i in range(4)]


def test_the_telemetry_summary():
    s = telemetry.summarise(ROWS, 1.0)
    assert s["samples"] == 4 and s["span_s"] == pytest.approx(3.0)
    assert s["slowdown"] == {"sw_power_cap": 0.25} and s["throttled"] and s["reasons"] == {"sw_power_cap": 0.25}
    assert s["sm_clock_mhz"] == {"min": 2420.0, "median": 2520.0, "max": 2520.0} and s["temperature_max_c"] == 63
    assert s["energy_j"] == pytest.approx(900.0) and s["power_w"] == {"mean": 300.0, "max": 300.0}
    quiet = telemetry.summarise([{**r, "clocks_throttle_reasons.active": "0x1"} for r in
                                 ({k: v for k, v in r.items() if k != "clocks_event_reasons.active"} for r in ROWS)])
    assert quiet["reasons"] == {"gpu_idle": 1.0} and not quiet["throttled"]       # an older driver's field name


def _csv(rows) -> str:
    keys = list(rows[0])
    return ",".join(keys) + "\n" + "".join(", ".join(r[k] for k in keys) + "\n" for r in rows)


def test_the_telemetry_files_and_the_check(tmp_path):
    logs = tmp_path / "final"
    tel = logs / "telemetry"
    tel.mkdir(parents=True)
    for name in ("sp-c-llama2-7b-64-11.out", "sp-q-opt125-12.out", "sp-b-k-gpt2-64-13.out"):
        (logs / name).write_text("node: {}\n")
    (tel / "sp-c-llama2-7b-64-11.csv").write_text(_csv(ROWS))                       # a sampler killed: its CSV
    with gzip.open(tel / "sp-q-opt125-12.csv.gz", "wt") as fh:                       # times nothing: not checked
        fh.write(_csv(ROWS))
    (tel / "sp-c-llama2-7b-64-11.apps.txt").write_text("# at the start (x)\n123, python, 400 MiB\n# at the end\n")
    s = telemetry.read(tel / "sp-c-llama2-7b-64-11")
    assert s["partial"] and s["throttled"] and s["apps_at_start"] == ["123, python, 400 MiB"]
    assert telemetry.read(tel / "sp-q-opt125-12")["samples"] == 4 and telemetry.read(tel / "none") is None
    summary, warn = N.telemetry_report(logs)
    assert summary == ["telemetry: 1 timed jobs sampled, 1 with a power or thermal slowdown, 1 timed job logs "
                       "without telemetry"]
    assert len(warn) == 2 and all("sp-c-llama2-7b-64-11" in w for w in warn)
    assert "sw_power_cap in 25% of the samples" in warn[0] and "other processes" in warn[1]


def test_a_sampler_writes_its_files_and_stops_with_its_job(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "gpu_id", lambda: "GPU-0")
    monkeypatch.setattr(telemetry, "known_fields", lambda gpu: ["timestamp", "power.draw",
                                                                "clocks_event_reasons.active"])
    monkeypatch.setattr(telemetry, "_apps", lambda gpu: "")
    line = "2026/10/27 10:00:00.000, 250.5, 0x0000000000000000"
    code = f"import time\nfor _ in range(50):\n    print({line!r}, flush=True)\n    time.sleep(0.05)\n"
    monkeypatch.setattr(telemetry.Sampler, "command", lambda self: [sys.executable, "-c", code])
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    s = telemetry.Sampler(tmp_path / "t" / "sp-x-1", 1, job.pid)
    s.start()
    s.wait()                                          # until the watched process has ended
    job.wait()
    summary = s.stop()
    assert summary["samples"] > 0 and summary["power_w"]["max"] == 250.5 and not summary["throttled"]
    assert (tmp_path / "t" / "sp-x-1.csv.gz").is_file() and not (tmp_path / "t" / "sp-x-1.csv").exists()
    assert telemetry.read(tmp_path / "t" / "sp-x-1")["samples"] == summary["samples"]
    monkeypatch.setattr(telemetry, "gpu_id", lambda: None)                  # no nvidia-smi: a summary that says so
    assert telemetry.main(["--out", str(tmp_path / "u" / "sp-y-2"), "--watch", str(job.pid)]) == 0
    assert json.loads((tmp_path / "u" / "sp-y-2.json").read_text())["samples"] == 0


# ------------------------------------------------------------------------------------------ the energy counter
class FakeNvml:
    """A stand-in for NVML's library: one GPU whose counter grows by 1,000 mJ per read."""

    def __init__(self, ok=True):
        self.mj, self.ok = 5_000, ok

    def nvmlInit_v2(self):
        return 0

    def nvmlDeviceGetHandleByUUID(self, uuid, ref):
        return 0 if uuid.value == b"GPU-1" else 13

    def nvmlDeviceGetHandleByIndex_v2(self, index, ref):
        return 0

    def nvmlDeviceGetTotalEnergyConsumption(self, handle, ref):
        if not self.ok:
            return 3
        self.mj += 1000
        ref._obj.value = self.mj
        return 0


def test_the_energy_counter():
    from experiments.common.energy import EnergyMeter, joules_per_call
    m = EnergyMeter("GPU-1", lib=FakeNvml())
    assert m.available and m.read() == 7.0 and m.read() == 8.0
    assert not EnergyMeter("GPU-2", lib=FakeNvml()).available
    gone = EnergyMeter("GPU-1", lib=FakeNvml(ok=False))
    assert not gone.available and gone.read() is None and "no energy counter" in gone.why
    got = joules_per_call(EnergyMeter("GPU-1", lib=FakeNvml()), lambda: None, lambda: None, 0.01, max_calls=4)
    assert got["calls"] >= 1 and got["joules"] == pytest.approx(1.0 / got["calls"])
    assert joules_per_call(None, lambda: None, lambda: None, 1.0) is None


def test_serving_energy_per_call():
    from experiments.deployment import serve_ab
    assert serve_ab._energy([1.0, None, 3.0], None) == {"joules": 2.0, "method": "timed", "calls": 2,
                                                        "joules_all": [1.0, 3.0]}
    assert serve_ab._energy([], {"joules": 0.5, "calls": 9, "seconds": 0.2}) == {"joules": 0.5, "method": "loop",
                                                                                "calls": 9}
    assert serve_ab._energy([None], None) is None and serve_ab._delta(1.0, 3.5) == 2.5
    assert serve_ab.build_parser().parse_args(["--model", "x"]).energy_s == 2.0


def test_the_serving_energy_macro():
    def rec(mode, p, u):
        return R.Rec("serve", "serve-opt13", "1", "sp-serve-opt13-1.out", 1, "serve",
                     {"row": "prove", "model": "opt-1.3b", "seq": 2048, "batch": 1, "mode": mode, "batch_proof": "seq",
                      "layers": None, "energy_j": {"prover": {"joules": p}, "fp16": {"joules": u}}})
    c = M.Collector()
    M.serving(c, T.Ctx(R.Run([rec("Kpre", 30.0, 1.5), rec("C", 60.0, 1.5)]), Bench(None), {}))
    assert c.m["ServeOptMEnergyOverUTwoRange"].value == "20--40"


# ------------------------------------------------------------------------------------------ raw timings
def test_reexec_and_setup_keep_every_repetition(monkeypatch, capsys):
    from experiments.costs import setup_timing
    from experiments.reexec import reexec
    for mod, argv in ((reexec, ["reexec.py", "--model", "tiny-gpt", "--seq", "8", "--layers", "1", "--variants", "int",
                                "--device", "cpu", "--reps", "2", "--threads", "1"]),
                      (setup_timing, ["setup_timing.py", "--model", "tiny-gpt", "--layers", "1", "--amortise", "10",
                                      "--reps", "2", "--device", "cpu", "--threads", "1"])):
        monkeypatch.setattr(sys, "argv", argv)
        assert mod.main() == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    rx, st = lines
    assert len(rx["times_s"]["1"]) == 2 and rx["measured_s"]["1"] == pytest.approx(statistics.median(rx["times_s"]["1"]))
    assert len(st["prove_s_all"]) == 2 == len(st["verify_s_all"])
    assert st["prove_s"] == pytest.approx(statistics.median(st["prove_s_all"]))


# ------------------------------------------------------------------------------------------ the package's jobs
def test_every_job_snapshots_its_node_and_samples_its_gpu():
    """Every job's script, after its node check and before any pinning or work: the environment snapshot and the GPU
    sampler, stopped at the job's end with the job's exit status kept; the smoke keeps its pytest log and JUnit XML
    per node and job, and its tier's xcheck job proves the fixed cells on the GPU and the CPU; both can be turned
    off, and a sampling interval below one second is refused."""
    from tests.test_run_package import STOP, _dry, _job
    lines = _dry()
    for line in lines:
        cmd = line.split("--wrap", 1)[1]
        node, snap, tele = (cmd.index(f"-m experiments.package.{m} ") for m in ("node", "envsnap", "telemetry"))
        assert node < snap < tele and cmd.endswith(STOP), _job(line)
        assert f"telemetry/{_job(line)}-${{PVI_JOB_ID:-${{SLURM_JOB_ID:-local}}}} --interval 1 --watch $$ &" in cmd
        if "package.pin" in cmd:      # the pinning moves the sampler off the pinned cores
            assert tele < cmd.index("-m experiments.package.pin") and "--away $TPID && " in cmd, _job(line)
    jobs = {_job(x): x for x in lines}
    smoke = jobs["sp-smoke"]
    assert "--junitxml=" in smoke and "smoke_pytest-$(hostname)-${PVI_JOB_ID:-${SLURM_JOB_ID:-local}}.txt" in smoke
    assert "comparison.proof_digests --devices cuda cpu" in jobs["sp-xcheck"] and "--time=3:00:00" in jobs["sp-xcheck"]
    off = _dry("--telemetry-interval", "0", "--no-env-snapshot")
    assert not any("package.telemetry" in x or "package.envsnap" in x or "--away" in x for x in off)
    assert any("--pid $$ && " in x for x in off)
    _dry("--telemetry-interval", "0.5", status=2)


def test_a_process_that_ended_but_is_not_reaped_counts_as_ended(tmp_path, monkeypatch):
    stat = tmp_path / "stat"
    real_open = open
    monkeypatch.setattr(telemetry, "open", lambda p, *a, **k: real_open(stat, *a, **k), raising=False)
    stat.write_text("4242 (python (x) y) Z 1 4242 4242 0 -1\n")
    assert telemetry._zombie(4242)
    stat.write_text("4242 (python (x) y) S 1 4242 4242 0 -1\n")
    assert not telemetry._zombie(4242)
    stat.unlink()
    assert not telemetry._zombie(4242)
