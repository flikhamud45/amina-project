"""The benchmark of the report (Sec. 4.3-4.4): our defence and Anchuri et al.'s path
test, on the models the literature benchmarks.  The prover runs on the GPU (the CPU if
there is none) and the verifier on the CPU, or on the prover's GPU with
``--verifier-device cuda`` (the ``_gpuv`` cells).  Every measurement is appended, one JSON
object per line, to ``artifacts/comparison/raw_<platform>/<suite>/<model>/<cell>.jsonl``;
nothing is aggregated here, so figures can be re-made later without re-running.

    python experiments/4_defence_benchmark/bench.py cnn --model vgg16 --platform <name>
    python experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 64 --platform <name>

A cell that has a ``.done`` marker is skipped, so an interrupted job resumes where it
stopped.  An honest query that is rejected stops the job and keeps the cell's records as
``<cell>.rejected-<run_id>.jsonl`` (``count_outcomes.py`` counts them).
``--policy <name>`` (a commitment plan of ``pvi.fullcheck.plans``: tight, cnn<e>, R<rate>, each also
with the suffix c: the col layouts and lookup tables; or auto, the c policy it picks for the model,
recorded as the cells' ``policy``, with ``policy_requested`` auto) commits under that plan and runs only the cells
it changes -- the commitment (``commit_*``, with its setup time and size) and the mode-C cells --
named with a ``_pol<name>`` suffix.  ``--prune-last`` (LLM suite) builds the decoders with their last
block at the last position (``build_decoder(..., prune_last=True)``), every cell named with a
``_prune`` suffix; ``--lookups`` runs the K and Kpre cells with a verifier that reads the embedding
rows itself (``Verifier(lookups=True)``), named with a ``_lookups`` suffix.
Every timing row (``s``) of a query, a commitment, a build or a precomputation also carries a
``contention`` field (:mod:`pvi.fullcheck.contention`): this process's run-queue wait and on-CPU time while
that phase ran, the load average at its start and end, and its context switches -- evidence that the
machine was quiet (the aggregates ignore it; a contention check over these fields reads it).
The ``pvi`` package must be this checkout's ``src/pvi`` (``export PYTHONPATH=<checkout>/code/src``,
as the sbatch scripts do): with a shared venv that has another clone's pvi installed,
bench.py refuses to run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pvi


def _require_this_pvi() -> Path:
    """Refuse to run on another checkout's ``pvi`` (e.g. a shared venv with pvi installed
    editable from a different clone): the records would say this code ran when it did not."""
    here = (Path(__file__).resolve().parents[2] / "src" / "pvi").resolve()
    got = Path(pvi.__file__).resolve().parent
    if got != here:
        raise SystemExit(f"pvi is imported from {got}, not from this checkout ({here}): "
                         f"export PYTHONPATH={here.parent}")
    return got


if __name__ == "__main__":  # before the imports below, which an older pvi may not even have
    _require_this_pvi()

from pvi.fullcheck import contention  # noqa: E402
from pvi.fullcheck.analytic import setup_size  # noqa: E402
from pvi.fullcheck.field import P  # noqa: E402
from pvi.fullcheck.graph import MatOp  # noqa: E402
from pvi.fullcheck.plans import plan_commitment  # noqa: E402
from pvi.fullcheck.protocol import (  # noqa: E402
    Challenger,
    Prover,
    Verifier,
    Z_BOUND,
    commit_graph,
    params_for,
    run_query,
    soundness_bits,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "comparison"
RAW = BASE / "raw"  # set by main() from --platform / $PVI_PLATFORM (see raw_root)
MODELS = ROOT / "artifacts" / "fullcheck" / "models"
LAMBDAS = (40, 80, 128)
RATE = 4       # Reed-Solomon rate (codeword length / row length) of the weight commitment
TAG = ""  # appended to every cell name (``--tag``), so variant runs never collide
VDEV = "cpu"   # --verifier-device: the client's device (the report's headline: the CPU; Sec. 4.4 also a GPU, _gpuv)
IMPL = "default"   # --verifier-impl: "stream" = the streaming verifier (Verifier(stream=True))
WIRE = False       # --wire: the proof in the compact encoding of pvi.fullcheck.claimcodec
PLATFORM = ""  # ``--platform``: "" is the earliest RTX 2080 Ti run's root ``raw/`` (frozen)


def raw_root(platform: str) -> Path:
    """``raw/`` holds the records of the earliest RTX 2080 Ti run (frozen), ``raw_rtx2080ti-v2/``
    its re-run with the fixed code, ``raw_l40s/`` the report's numbers and ``raw_l40s_improved/`` the
    opt-in improvements on the same hardware (all frozen too); every other prover/verifier pair writes
    its own ``raw_<platform>/``, so records of different hardware can never share a cell (or a median)."""
    p = "" if platform in ("", "rtx2080ti") else platform
    return BASE / (f"raw_{p}" if p else "raw")


def claim_platform(env: dict) -> None:
    """Refuse to write into a root that holds another machine's records.  The first run
    in an empty root creates ``PLATFORM.json``; every run (the first included) must match
    its GPU and CPU.  The lock appears atomically and complete (a hard link to a finished
    temporary file), so jobs that start together cannot each claim the root for their own
    machine, and no job ever reads a half-written lock."""
    lock = RAW / "PLATFORM.json"
    want = {"gpu": env.get("gpu") if str(env.get("device", "")).startswith("cuda") else None,
            "cpu": env.get("cpu")}
    if not lock.exists():
        if any(RAW.glob("*/*/*.jsonl")):
            raise SystemExit(f"{RAW} has records but no PLATFORM.json; create it before adding records")
        RAW.mkdir(parents=True, exist_ok=True)
        tmp = RAW / f"PLATFORM.json.{uuid.uuid4().hex}"
        tmp.write_text(json.dumps(dict(want, platform=PLATFORM, host=env.get("host"), torch=env.get("torch"),
                                       cuda=env.get("cuda"), gpu_total_memory=env.get("gpu_total_memory"),
                                       driver=env.get("driver"), created=time.time()), indent=1) + chr(10))
        try:
            os.link(tmp, lock)          # fails if another job created the lock first
        except FileExistsError:
            pass                        # it did: compare with its lock below
        finally:
            tmp.unlink()
    have = json.loads(lock.read_text())
    if have.get("frozen"):
        raise SystemExit(f"{RAW} is frozen (stored records that must not change). Pass --platform "
                         f"<name> (or export PVI_PLATFORM) to write a separate raw_<name>/ root.")
    if {k: have.get(k) for k in want} != want:
        raise SystemExit(f"{RAW} belongs to {have.get('gpu')} + {have.get('cpu')}; this job runs on "
                         f"{want['gpu']} + {want['cpu']}.  Pass --platform <name> (or export "
                         f"PVI_PLATFORM) to write a separate raw_<name>/ root.")


# --------------------------------------------------------------------------- env
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.fileMode=false", *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def env_info() -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except Exception:
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                                capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:
        driver = None
    return {
        "host": socket.gethostname(), "cpu": cpu, "torch_threads": torch.get_num_threads(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "gpu": gpu, "gpu_total_memory": gpu_mem,
        "driver": driver, "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "platform": PLATFORM,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "pvi_path": str(Path(pvi.__file__).resolve().parent),   # the code that ran (see _require_this_pvi)
        **_env_extra(),
    }


def _env_extra() -> dict:
    """Library versions, TF32 settings, environment variables, CPU affinity and GPU details."""
    import torchvision

    out = {"raw_dir": str(RAW), "cpu_count": os.cpu_count(), "numpy": np.__version__,
           "schedstat": contention.available(),   # the timing rows' contention evidence has run/wait times
           "torchvision": torchvision.__version__,
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "fp32_matmul_precision": torch.get_float32_matmul_precision(),
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "vars": {k: v for k, v in os.environ.items()
                    if k.startswith(("NVIDIA_TF32", "TORCH_ALLOW_TF32", "CUBLAS", "OMP_", "MKL_", "CUDA_VISIBLE",
                                     "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "PVI_"))}}
    try:
        out["cpu_affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:  # Linux: physical cores in the affinity set ('x8 threads' can be 4 cores x 2 hyperthreads)
        topo = "/sys/devices/system/cpu/cpu{}/topology/{}"
        out["cpu_affinity_cores"] = len({tuple(Path(topo.format(c, k)).read_text().strip()
                                               for k in ("physical_package_id", "core_id"))
                                         for c in os.sched_getaffinity(0)})
    except Exception:
        pass
    queries = ((["lscpu"], "lscpu"),
               (["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit,"
                 "mig.mode.current,compute_mode,pci.bus_id", "--format=csv,noheader"], "nvidia_smi"))
    for cmd, key in queries:
        try:
            txt = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            continue
        if key == "lscpu":
            keep = ("Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "NUMA node(s)",
                    "CPU max MHz")
            txt = {a.strip(): b.strip() for a, _, b in (l.partition(":") for l in txt.splitlines()) if a.strip() in keep}
        out[key] = txt
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        out["gpu_props"] = {"total_memory": pr.total_memory, "capability": f"{pr.major}.{pr.minor}",
                            "sms": pr.multi_processor_count}
    return out


def _verifier(*args, **kwargs) -> Verifier:
    """Every verifier of this benchmark runs on ``--verifier-device`` (the report's headline: the CPU;
    Sec. 4.4 also a GPU, tag ``_gpuv``), with ``--verifier-impl`` (the report: the default; ``stream``
    records one ``verify_total`` per query instead of the verify_* phases, which overlap)."""
    return Verifier(*args, device=VDEV, stream=IMPL == "stream", **kwargs)


def _query(prover: Prover, v: Verifier, x: torch.Tensor, **kwargs) -> dict:
    """One interaction of this benchmark: ``run_query``, with ``--wire`` in the compact encoding of the
    proof (``prove_encode`` / ``verify_decode`` recorded, and the encoded sizes as ``bytes_*``)."""
    return run_query(prover, v, x, wire=WIRE, **kwargs)


def _verifier_hw(env: dict) -> str:
    cpu = f"{env.get('cpu')} x{env.get('torch_threads')} threads"
    if VDEV == "cpu":
        return cpu
    return f"{torch.cuda.get_device_name(torch.device(VDEV))} (GPU client) + {cpu}"


def _lock(fh, part: Path) -> None:
    """An exclusive lock on a cell's ``.part`` file: a second job on the same cell stops at
    once instead of interleaving its rows with the first job's.  Where the filesystem has
    no ``flock`` (or on Windows), ``sbatch --dependency=singleton`` is the only guard."""
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(f"{part} is being written by another job")
    except (ImportError, OSError):
        pass


class Recorder:
    """Append-only per-cell record file with an atomic ``.done`` marker.  A ``.part`` file
    left by an interrupted attempt is kept (renamed ``.part.<time>``), never truncated."""

    def __init__(self, suite: str, model: str, cell: str, config: dict, env: dict) -> None:
        self.dir = RAW / suite / model
        self.dir.mkdir(parents=True, exist_ok=True)
        cell = cell + TAG
        self.path = self.dir / f"{cell}.jsonl"
        self.done_path = self.dir / f"{cell}.done"
        on_gpu = str(env.get("device", "")).startswith("cuda")
        prover_hw = env.get("gpu") if on_gpu else env.get("cpu")
        self.base = {"run_id": uuid.uuid4().hex[:12], "suite": suite, "model": model, "cell": cell,
                     "config": dict(config, variant=TAG, device=env.get("device"), merkle="multiproof",
                                    verifier_device=VDEV,
                                    **({"platform": PLATFORM} if PLATFORM else {}),
                                    **({"verifier_impl": IMPL} if IMPL != "default" else {}),
                                    **({"wire": True} if WIRE else {})),
                     "prover_hw": prover_hw, "verifier_hw": _verifier_hw(env),
                     "host": env.get("host"), "git_sha": env.get("git_sha")}
        part = self.path.with_suffix(".jsonl.part")
        self.fh = open(part, "a")
        _lock(self.fh, part)
        if os.fstat(self.fh.fileno()).st_size:   # an earlier attempt's records: keep them aside
            self.fh.close()                       # (Windows cannot rename an open file)
            os.replace(part, part.with_name(f"{part.name}.{time.time():.6f}"))
            try:
                self.fh = open(part, "x")
            except FileExistsError:
                raise SystemExit(f"{part} is being written by another job")
            _lock(self.fh, part)
        self.rec("env", 0, "", **{"env": env})

    def rec(self, metric: str, value, unit: str = "", trial: int | None = None, **extra) -> None:
        row = dict(self.base, metric=metric, value=value, unit=unit, trial=trial, ts=time.time(), **extra)
        self.fh.write(json.dumps(row, default=float) + "\n")
        self.fh.flush()

    def done(self) -> None:
        self.fh.close()
        os.replace(self.path.with_suffix(".jsonl.part"), self.path)
        self.done_path.write_text(json.dumps({"ts": time.time()}))
        for p in (self.path, self.done_path):
            try:
                os.chmod(p, 0o664)
            except OSError:
                pass


def is_done(suite, model, cell) -> bool:
    return (RAW / suite / model / f"{cell}{TAG}.done").exists()


PLAN_CELLS = ("commit_", "defence_C_", "tamper_C_")
"""The cells a commitment policy changes (the others are the same under every policy)."""


def _todo(args, suite, model, cell) -> bool:
    """Run this cell?  Not yet done; under ``--policy`` or ``--lookups`` only the cells the option
    changes."""
    if args.policy != "paper" and not cell.startswith(PLAN_CELLS):
        return False
    if args.lookups and not cell.startswith(("defence_K_", "defence_Kpre_")):
        return False
    return not is_done(suite, model, cell)


def _commit(graph, rate: int, device, policy: str, model_ops=None, cs: dict | None = None):
    """``(commitment, seconds)``: the weight commitment under ``policy``, timed to its end on the GPU;
    its contention evidence goes to ``cs["commit_total"]`` (:func:`_cs`)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    c0 = contention.begin()
    t0 = time.perf_counter()
    coms = commit_graph(graph, rate, device=device, policy=policy, model_ops=model_ops)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if cs is not None:
        contention.add(cs, "commit_total", c0)
    return coms, dt


def _cs(store: dict | None, phase: str) -> dict:
    """The ``contention`` field of a timing row: the evidence of whether its machine was quiet while
    ``phase`` ran (:mod:`pvi.fullcheck.contention`: run-queue wait and on-CPU time of this process's
    threads, the load average at start and end, context switches); nothing if none was taken.  A new
    field: every reader of the records keys on metric, value and the group keys, and ignores it."""
    c = (store or {}).get(phase)
    return {"contention": c} if c is not None else {}


def _plan_config(coms, params=None) -> dict:
    """What a cell records of a commitment plan, and with ``params`` its trees' column counts
    (nothing for the report's commitment)."""
    if coms is None or coms.plan is None:
        return {}
    return {"policy": coms.plan.policy, **({"policy_requested": coms.plan.requested} if coms.plan.requested else {}),
            "trees": len(coms.groups),
            **({"group_columns": dict(params.group_columns)} if params is not None else {})}


def _record_setup(r: "Recorder", coms, ops) -> None:
    """A commitment plan's setup size (``analytic.setup_size``: encoded entries, leaves, trees)."""
    for k, v in setup_size(ops, plan=coms.plan).items():
        r.rec("setup_" + k, v, "")


def _plan_soundness(params, plan, mode: str) -> float:
    return soundness_bits(params, plan.shapes(), mode, columns=plan.matrix_columns(params.group_columns))


def _opening_of(coms, name: str) -> tuple[str, int]:
    """Where op ``name``'s opened columns are: its own opening, or from row ``offset`` of its group's
    (those of its committed matrix: under a ``c`` policy maybe a col-layout one, of ``k`` rows)."""
    if coms.plan is None:
        return name, 0
    matrix = coms.plan.matrix_of(name).name
    g, members = next((g, ms) for g, ms in coms.plan.groups if matrix in ms)
    return g, sum(coms[m].public.n_rows for m in members[:members.index(matrix)])


def _record_query(r: Recorder, res: dict, trial: int, **extra) -> None:
    for k, v in res["timings"].items():
        r.rec(k, v, "s", trial, **extra, **_cs(res.get("contention"), k))
    for k, v in res["bytes"].items():
        r.rec("bytes_" + k, v, "B", trial, **extra)
    r.rec("bytes_total", sum(res["bytes"].values()), "B", trial, **extra)
    r.rec("accepted", int(res["accepted"]), "", trial, **extra)
    if not res["accepted"]:  # an honest query must never fail: GPU/CPU disagreement, stop the job
        # keep the evidence where a re-run cannot overwrite it and count_outcomes.py finds it
        # (no .done: the aggregates never read it)
        r.fh.close()
        kept = r.path.with_name(f"{r.path.stem}.rejected-{r.base['run_id']}.jsonl")
        os.replace(r.path.with_suffix(".jsonl.part"), kept)
        raise SystemExit(f"HONEST QUERY REJECTED at {res['rejected_at']} in {r.path.name} trial {trial} "
                         f"(records kept in {kept.name})")


def _reset_peaks(device) -> None:
    """Start a new peak window for GPU allocations (the prover's GPU, which with
    ``--verifier-device cuda`` also holds the client) and host resident memory."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
    try:  # Linux: "5" resets VmHWM (peak RSS) to the current RSS
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def _record_peaks(r: "Recorder", device, **extra) -> None:
    if device.type == "cuda":
        r.rec("gpu_peak_memory", torch.cuda.max_memory_allocated(device), "B", **extra)
        r.rec("gpu_peak_reserved", torch.cuda.max_memory_reserved(device), "B", **extra)
        st = torch.cuda.memory_stats(device)   # allocator churn (cudaMalloc calls, OOM retries) since the reset
        r.rec("gpu_alloc_retries", st.get("num_alloc_retries", -1), "", **extra)
        r.rec("gpu_device_allocs", st.get("num_device_alloc", -1), "", **extra)
    try:
        with open("/proc/self/status") as fh:
            kb = next(int(l.split()[1]) for l in fh if l.startswith("VmHWM:"))
        r.rec("host_peak_rss", kb * 1024, "B", **extra)
    except (OSError, StopIteration):
        pass


# ------------------------------------------------------------------ CNN models
def _mlp_model() -> tuple[nn.Module, dict]:
    from pvi.data import load_classification
    from pvi.training import load_network
    from pvi.zoo import mlp_architecture_for

    net = load_network(mlp_architecture_for(10), ROOT / "artifacts" / "models" / "mlp_mnist_full.npz")
    layers = [l for l in net.architecture.layers if l.name in net.parameters]
    mods: list[nn.Module] = []
    for i, layer in enumerate(layers):
        w, b = net.parameters[layer.name]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = torch.from_numpy(np.asarray(w, dtype=np.float32))
        lin.bias.data = torch.from_numpy(np.asarray(b, dtype=np.float32))
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ReLU())
    ds = load_classification("mnist")
    data = {"train_x": torch.from_numpy(ds.train_x[:512]).reshape(-1, 784),
            "test_x": torch.from_numpy(ds.test_x).reshape(-1, 784), "test_y": torch.from_numpy(ds.test_y)}
    return nn.Sequential(*mods).eval(), data


def _cnn_model(name: str) -> tuple[nn.Module, dict]:
    from pvi.fullcheck.datasets import load_dataset, normalise
    from pvi.fullcheck.models import MODEL_SPECS, build_float_model

    kind, dataset = MODEL_SPECS[name]
    tx, _, vx, vy, family, n_classes = load_dataset(dataset)
    model = build_float_model(kind, n_classes)
    model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=True))
    crop = (lambda x: x[:, :, 16:240, 16:240]) if family == "imagenet" else (lambda x: x)
    g = torch.Generator().manual_seed(0)
    cal = tx[torch.randperm(len(tx), generator=g)[: (64 if family == "imagenet" else 512)]]
    data = {"train_x": normalise(crop(cal), family), "test_x_uint8": vx, "test_y": vy,
            "norm": lambda x: normalise(crop(x), family)}
    return model.eval(), data


def _test_batches(data, batch):
    if "test_x" in data:
        x = data["test_x"]
        for s in range(0, len(x), batch):
            yield x[s:s + batch], data["test_y"][s:s + batch]
    else:
        x = data["test_x_uint8"]
        for s in range(0, len(x), batch):
            yield data["norm"](x[s:s + batch]), data["test_y"][s:s + batch]


def _penultimate_mat(graph) -> MatOp:
    """The weight op whose output (through cheap ops only) feeds the final layer."""
    prod = {op.output: op for op in graph.ops}
    name = graph.mat_ops[-1].inputs[0]
    while not isinstance(prod[name], MatOp):
        name = prod[name].inputs[0]
    return prod[name]


def suite_cnn(args, env) -> None:
    from pvi.fullcheck.quantize import dequantize_logits, quantize_input, quantize_model
    from pvi.fullcheck.sampling import (TraceCommitment, _base as _base_name, neuron_tensors, paths_for,
                                        shared_path_bytes, visit_probabilities)

    device = torch.device(env["device"])
    name = args.model
    model, data = _mlp_model() if name == "mlp_mnist" else _cnn_model(name)
    graph = quantize_model(model, data["train_x"])
    mats = graph.mat_ops
    n_params = graph.n_params()
    model_f = model.float().to(device)
    xs, _ = next(_test_batches(data, max([args.queries + 64, *args.cnn_batches])))
    q_inputs = quantize_input(graph, xs)          # int8 queries, one per trial

    # ---- model facts and fidelity ------------------------------------------------
    cell = "facts"
    if _todo(args, "cnn", name, cell):
        r = Recorder("cnn", name, cell, {}, env)
        r.rec("n_params", n_params, "")
        r.rec("model_bytes_int8", sum(op.weight.numel() + 4 * (op.bias.numel() if op.bias is not None else 0)
                                      for op in mats), "B")
        correct_f = correct_i = total = 0
        with torch.no_grad():
            for xb, yb in _test_batches(data, 128 if "test_x" in data else 32):
                lf = model_f(xb.to(device)).argmax(1).cpu()
                _, cl = graph.forward(quantize_input(graph, xb).to(device))
                li = dequantize_logits(graph, cl[mats[-1].name]).argmax(1).cpu()
                correct_f += int((lf == yb).sum())
                correct_i += int((li == yb).sum())
                total += len(yb)
        r.rec("float_accuracy", correct_f / total, "", n=total)
        r.rec("int8_accuracy", correct_i / total, "", n=total)
        r.done()

    # ---- the single-neuron attack on the float model --------------------------------
    cell = "attack_float"
    if _todo(args, "cnn", name, cell):
        r = Recorder("cnn", name, cell, {"n": len(xs)}, env)
        seq = isinstance(model_f, nn.Sequential)
        head = model_f[-1] if seq else model_f.head
        feat_fn = (lambda b_: model_f[:-1](b_)) if seq else model_f.features
        with torch.no_grad():
            feats = feat_fn(xs.to(device)).double()
            w, b = head.weight.double(), head.bias.double()
            logits = feats @ w.T + b
            pred = logits.argmax(1)

            def best_flip(i):
                best = None
                l_ = logits[i]
                for c in range(w.shape[0]):
                    if c == int(pred[i]):
                        continue
                    others = torch.cat([w[:c], w[c + 1:]])       # rows k != c
                    lo = torch.cat([l_[:c], l_[c + 1:]])
                    ok_j = (w[c][None, :] > others).all(0)  # c wins for a large enough delta
                    if not bool(ok_j.any()):
                        continue
                    gap = (lo - l_[c])[:, None] / (w[c][None, :] - others)
                    need = gap.clamp_min(0).amax(0) * 1.0001 + 1e-9
                    need = torch.where(ok_j, need, torch.full_like(need, math.inf))
                    j = int(need.argmin())
                    if best is None or float(need[j]) < best[0]:
                        best = (float(need[j]), c, j)
                return best

            for i in range(len(xs)):
                best = best_flip(i)
                if best is None:
                    r.rec("flip_success", 0, "", i)
                    continue
                delta, c, j = best
                new = logits[i] + delta * w[:, j]
                r.rec("flip_success", int(int(new.argmax()) == c), "", i)
        r.done()

    # ---- the sampling baseline (Anchuri et al.) on the integer graph -----------------------
    cell = "sampling"
    if _todo(args, "cnn", name, cell):
        r = Recorder("cnn", name, cell, {}, env)
        env1, _ = graph.forward(q_inputs[:1])
        tc = TraceCommitment(graph, env1)
        r.rec("open_all_bytes", tc.open_all_bytes, "B")
        mass = visit_probabilities(graph, env1)
        prod = {op.output: op for op in graph.ops}
        src = _base_name(prod, mats[-1].inputs[0])
        p_attack = float(mass[src].min())
        p_min = min(float(mass[n].min()) for n in neuron_tensors(graph))
        r.rec("p_detect_penultimate", p_attack, "")
        r.rec("p_detect_min_node", p_min, "")
        k40 = paths_for(40, p_attack)          # paths for 2^-40 against the single-neuron attack
        grid = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
        shared = shared_path_bytes(tc, env1, sorted({k40, *grid}))
        for k_ in grid:   # the cost-vs-security curve of the sampling protocol
            r.rec("paths_bytes_shared_k", shared[k_], "B", k=k_, bits=-k_ * math.log2(1.0 - p_attack))
        r.rec("paths_needed_penultimate", k40, "", lam=40)
        r.rec("paths_bytes_shared", shared[k40], "B", lam=40)
        r.done()

    # ---- our defence: commitment, honest queries, all modes and security levels ----------
    # The commitment is one-time and takes seconds for these models, so it is
    # always rebuilt (the Merkle roots are deterministic).
    rate = RATE
    _reset_peaks(device)
    commit_cs: dict = {}
    coms, commit_s = _commit(graph, rate, device, args.policy, cs=commit_cs)
    cell = f"commit_rate{rate}"
    if coms.plan is not None and _todo(args, "cnn", name, cell):     # a policy's setup cost
        r = Recorder("cnn", name, cell, {"rate": rate, **_plan_config(coms)}, env)
        r.rec("commit_total", commit_s, "s", **_cs(commit_cs, "commit_total"))
        _record_setup(r, coms, mats)
        _record_peaks(r, device)
        r.done()

    prover = Prover(graph, device=device, commitments=coms)
    weights = {op.name: (op.weight, op.bias) for op in mats}
    for lam in LAMBDAS:
        for mode, chal in (("C", "int"), ("C", "fs"), ("K", "int"), ("Kpre", "int")):
            cell = f"defence_{mode}_{chal}_lam{lam}_rate{rate}"
            if not _todo(args, "cnn", name, cell):
                continue
            params = params_for(lam, len(mats), rate=rate, fiat_shamir=(chal == "fs"), plan=coms.plan)
            cfg = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": rate,
                   "columns": params.columns, "threads": torch.get_num_threads(), **_plan_config(coms, params)}
            r = Recorder("cnn", name, cell, cfg, env)
            if coms.plan is None:
                shapes = [(op.row_length, rate * (1 << max(0, (op.row_length - 1).bit_length()))) for op in mats]
                r.rec("soundness_bits", soundness_bits(params, shapes, "C" if mode == "C" else "K"), "bits")
            else:
                r.rec("soundness_bits", _plan_soundness(params, coms.plan, "C"), "bits")
            if mode == "C":
                v = _verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                              tables=coms.table_publics)
            else:
                v = _verifier(graph.public(), params, mode, weights=weights)
                if mode == "Kpre":
                    v.precompute(Challenger())
            _reset_peaks(device)
            _query(prover, v, q_inputs[:1])  # untimed warm-up (lazy CUDA/BLAS initialisation)
            for i in range(args.queries):
                _record_query(r, _query(prover, v, q_inputs[i:i + 1]), i)
            _record_peaks(r, device)
            # batch amortisation: B queries in one interaction (mode C, interactive only)
            if mode == "C" and chal == "int":
                for bsz in args.cnn_batches:
                    if bsz > len(q_inputs):
                        continue
                    _reset_peaks(device)
                    for tr in range(args.batch_trials):
                        _record_query(r, _query(prover, v, q_inputs[:bsz]), tr, batch=bsz)
                    _record_peaks(r, device, batch=bsz)
            r.done()

    # ---- soundness experiments: every attack must be rejected (--tampers 0: none) ------------
    cell = "tamper_C_int_lam40"
    if args.tampers and _todo(args, "cnn", name, cell):
        params = params_for(40, len(mats), rate=rate, plan=coms.plan)
        v = _verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                      tables=coms.table_publics)
        r = Recorder("cnn", name, cell, {"lam": 40, "mode": "C", "reps": params.reps, "columns": params.columns,
                                         **_plan_config(coms, params)}, env)
        g = torch.Generator().manual_seed(123)
        pen = _penultimate_mat(graph)
        x0 = q_inputs[:1]

        def attempt(kind, i, **kw):
            res = _query(prover, v, x0, **kw)
            r.rec("rejected", int(not res["accepted"]), "", i, attack=kind, stage=res["rejected_at"])

        for i in range(args.tampers):
            op = mats[int(torch.randint(len(mats), (1,), generator=g))]
            pos = int(torch.randint(10 ** 9, (1,), generator=g))
            delta = int(torch.randint(1, 1000, (1,), generator=g)) * (1 if i % 2 else -1)

            def t_single(o, z, op=op, pos=pos, delta=delta):
                if o.name == op.name:
                    z = z.clone()
                    z.view(-1)[pos % z.numel()] += delta
                return z

            attempt("single_value", i, forward_kwargs={"tamper": t_single})
            j = int(torch.randint(pen.n_rows, (1,), generator=g))

            def t_pen(o, z, j=j):
                if o.name == pen.name:
                    z = z.clone()
                    z[j] = Z_BOUND // 4
                return z

            attempt("penultimate_neuron", i, forward_kwargs={"tamper": t_pen})

            def t_logit(o, z):
                if o.name == mats[-1].name:
                    z = z.clone()
                    k = int(z[:, 0].argmin())
                    z[k, 0] = int(z[:, 0].max()) + 1000
                return z

            attempt("output_logit", i, forward_kwargs={"tamper": t_logit})
        # model substitution: 1% of every weight matrix nudged by +-1
        for i in range(max(3, args.tampers // 10)):
            other = {}
            for op in mats:
                w = op.weight.clone().to(torch.int16)
                mask = torch.rand(w.shape, generator=g) < 0.01
                w = (w + mask.to(torch.int16) * (2 * torch.randint(0, 2, w.shape, generator=g, dtype=torch.int16) - 1))
                other[op.name] = MatOp(op.name, op.inputs, op.output, weight=w.clamp(-127, 127).to(torch.int8),
                                       bias=op.bias, layout=op.layout, conv=op.conv)
            attempt("substituted_model_1pct", i, forward_kwargs={"weights_override": other})
        if name == "resnet18_224" and (MODELS / "resnet18_224_squirrel.pt").exists():
            m2, d2 = _cnn_model("resnet18_224_squirrel")
            g2 = quantize_model(m2, d2["train_x"])
            other = {a.name: b_ for a, b_ in zip(mats, g2.mat_ops)}
            for i in range(max(3, args.tampers // 10)):
                attempt("substituted_model_M_tilde", i, forward_kwargs={"weights_override": other})
        # Attacks aimed at the two parts of mode C that Freivalds alone cannot cover.
        # (1) forged fold: a wrong logit plus a u' solved so that Freivalds passes;
        #     only the Reed-Solomon column check can reject it.
        # (2) forged column: an honest u plus an opened column shifted by a vector in
        #     the kernel of chi, so the code check passes; only the Merkle path can reject it.
        #     (A col-layout matrix's check folds its columns with w = chi' [X ; 1]^T: a shift in
        #     the kernel of [X ; 1]^T, which the prover knows, passes it for every chi'.)
        real_fold, real_open = prover.fold, prover.open
        final = mats[-1]
        state: dict = {}

        def t_forge(o, z):
            if o.name == final.name:
                z = z.clone()
                k = int(z[:, 0].argmin())
                new = int(z[:, 0].max()) + 1000
                state["k"], state["delta"] = k, new - int(z[k, 0])
                z[k, 0] = new
            return z

        def forged_fold(chis):
            us = real_fold(chis)
            u = us[final.name].clone()
            kb = final.n_in                      # the bias coordinate multiplies the constant 1
            for i_ in range(u.shape[0]):
                u[i_, kb] = (int(u[i_, kb]) + int(chis[final.name][i_, state["k"]]) * state["delta"]) % P
            us[final.name] = u
            return us

        def kernel_vector(left):
            """A nonzero delta with left @ delta = 0 mod P (Gaussian elimination; None: full column rank)."""
            m = [[int(v) % P for v in row] for row in left.tolist()]
            pivots = []
            for col in range(left.shape[1]):
                piv = next((i_ for i_ in range(len(pivots), len(m)) if m[i_][col]), None)
                if piv is None:
                    continue
                r_ = len(pivots)
                m[r_], m[piv] = m[piv], m[r_]
                inv = pow(m[r_][col], P - 2, P)
                m[r_] = [(v_ * inv) % P for v_ in m[r_]]
                for i_ in range(len(m)):
                    if i_ != r_ and m[i_][col]:
                        f_ = m[i_][col]
                        m[i_] = [(vi - f_ * vc) % P for vi, vc in zip(m[i_], m[r_])]
                pivots.append(col)
            free = next((c_ for c_ in range(left.shape[1]) if c_ not in pivots), None)
            if free is None:
                return None
            delta = torch.zeros(left.shape[1], dtype=torch.int64)
            delta[free] = 1
            for i_, c_ in enumerate(pivots):
                delta[c_] = (-m[i_][free]) % P
            return delta

        def folded_with(victim):
            """What the code check folds the victim's opened columns with: its chi, or for a col-layout
            victim [X ; 1]^T of the honest query."""
            matrix = coms.plan.matrix_of(victim) if coms.plan is not None else None
            if matrix is None or matrix.layout == "row":
                return state["chis"][victim]
            first = next(op for op in mats if op.name == matrix.members[0])
            xt = first.unfold(prover.graph.forward(x0.to(prover.device))[0][first.inputs[0]]).T.cpu()
            return torch.cat([xt, torch.ones(xt.shape[0], 1, dtype=torch.int64)], 1) if first.has_bias else xt

        folded_final = coms.plan is None or coms.plan.matrix_of(final.name).layout == "row"
        for i in range(max(3, args.tampers // 10)):
            if folded_final:                 # (a col-layout final op has no u to forge: output_logit covers it)
                prover.fold = forged_fold
                attempt("forged_fold", i, forward_kwargs={"tamper": t_forge})
                prover.fold = real_fold
            victim = mats[i % len(mats)].name

            def capture_fold(chis):
                state["chis"] = chis
                return real_fold(chis)

            def forged_open(cols, victim=victim):
                out = real_open(cols)
                key, off = _opening_of(coms, victim)      # (under a plan: the victim's rows of its group's)
                c, pth = out[key]
                delta = kernel_vector(folded_with(victim))
                if delta is not None:
                    c = c.clone()
                    c[off:off + len(delta), 0] = (c[off:off + len(delta), 0] + delta) % P
                out[key] = (c, pth)
                return out

            prover.fold, prover.open = capture_fold, forged_open
            attempt("forged_column", i)
            prover.fold, prover.open = real_fold, real_open
        r.done()


# ------------------------------------------------------------------ LLM shapes
def suite_llm(args, env) -> None:
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.transformer import CONFIGS, build_decoder, decoder_param_count

    device = torch.device(env["device"])
    cfg = CONFIGS[args.model]
    prune = args.prune_last
    if args.builds == "auto":      # 12 blocks or fewer: all of them; else 1 and 2 (the 30-70B jobs)
        builds = [cfg.n_layers] if cfg.n_layers <= 12 else [1, 2]
    else:                          # e.g. "full" or "1,2,full" (L1, L2 and the full model in one job)
        builds = [cfg.n_layers if b == "full" else int(b) for b in args.builds.split(",")]
    modes = [("C", "int"), ("C", "fs"), ("Kpre", "int"), ("K", "int")] if not args.modes else \
        [tuple(m.split(":")) for m in args.modes.split(",")]
    if args.policy != "paper":        # a commitment policy changes the mode-C cells only
        modes = [m for m in modes if m[0] == "C"]
    if args.lookups:                  # and the verifier's own rows the K and Kpre cells only
        modes = [m for m in modes if m[0] != "C"]
    skipped = []
    for seq in args.seq:
        for n_layers in builds:
            base = f"T{seq}_L{n_layers}"
            todo = [f"defence_{m}_{c}_lam{l}_{base}" for m, c in modes for l in args.lams]
            if args.llm_tampers and n_layers == cfg.n_layers and any(m == "C" for m, _ in modes):
                todo.append(f"tamper_C_int_lam40_{base}")
            if all(is_done("llm", cfg.name, t) for t in todo):
                continue
            if device.type == "cuda":  # a build whose int8 weights alone fill the GPU would only OOM
                need = sum(sh.n_rows * sh.row_length for sh in decoder_shapes(cfg, n_layers=n_layers,
                                                                              prune_last=prune))
                have = torch.cuda.get_device_properties(device).total_memory
                if need > 0.85 * have:     # ... after its build and commit: skip it, run the others
                    print(f"SKIP {cfg.name} T{seq} L{n_layers}: {need / 2**30:.1f} GiB of int8 weights "
                          f"> 85% of this GPU's {have / 2**30:.0f} GiB", flush=True)
                    skipped.append(f"T{seq}_L{n_layers}")
                    continue
            _reset_peaks(device)
            setup_cs: dict = {}
            c0 = contention.begin()
            t0 = time.perf_counter()
            graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, 32), seed=0, prune_last=prune)
            build_s = time.perf_counter() - t0
            contention.add(setup_cs, "build_graph", c0)
            mats = graph.mat_ops
            # the first ``queries`` rows are the single-prompt queries of the stored runs
            tokens = torch.randint(0, cfg.vocab, (max([args.queries, *args.batches]), seq),
                                   generator=torch.Generator().manual_seed(1))
            coms = None
            if any(m == "C" for m, _ in modes):
                coms, commit_s = _commit(graph, RATE, device, args.policy, decoder_shapes(cfg, prune_last=prune),
                                         cs=setup_cs)
                cell = f"commit_{base}"
                if _todo(args, "llm", cfg.name, cell):
                    r = Recorder("llm", cfg.name, cell, {"seq": seq, "n_layers": n_layers, "rate": RATE,
                                                         "n_layers_full": cfg.n_layers, "lean": args.lean,
                                                         "params_full": decoder_param_count(cfg),
                                                         **({"prune_last": True} if prune else {}),
                                                         **_plan_config(coms)}, env)
                    r.rec("build_graph", build_s, "s", **_cs(setup_cs, "build_graph"))
                    r.rec("commit_total", commit_s, "s", **_cs(setup_cs, "commit_total"))
                    if coms.plan is not None:
                        _record_setup(r, coms, mats)
                    _record_peaks(r, device)
                    r.done()
            prover = Prover(graph, device=device, commitments=coms, lean=args.lean)
            weights = {op.name: (op.weight, op.bias) for op in mats}
            for lam in args.lams:
                for mode, chal in modes:
                    cell = f"defence_{mode}_{chal}_lam{lam}_{base}"
                    if not _todo(args, "llm", cfg.name, cell):
                        continue
                    # size (r, t) for the FULL model's op count, so extrapolated rows keep their lambda
                    full_shapes = decoder_shapes(cfg, prune_last=prune)
                    plan = None if coms is None else coms.plan
                    params = params_for(lam, len(full_shapes), rate=RATE, fiat_shamir=(chal == "fs"), plan=plan)
                    conf = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": RATE,
                            "columns": params.columns, "seq": seq, "n_layers": n_layers,
                            "n_layers_full": cfg.n_layers, "params_full": decoder_param_count(cfg),
                            "params_built": graph.n_params(), "lean": args.lean,
                            "threads": torch.get_num_threads(), **({"prune_last": True} if prune else {}),
                            **({"lookups": True} if args.lookups else {}), **_plan_config(coms, params)}
                    r = Recorder("llm", cfg.name, cell, conf, env)
                    if plan is not None:      # the whole model's bound, under the whole model's plan
                        whole = plan_commitment(full_shapes, args.policy, rate=RATE)
                        r.rec("soundness_bits", _plan_soundness(params_for(
                            lam, len(full_shapes), fiat_shamir=(chal == "fs"), plan=whole), whole, "C"), "bits")
                    if mode == "C":
                        v = _verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                                      tables=coms.table_publics, lean=args.lean)
                    else:
                        v = _verifier(graph.public(), params, mode, weights=weights, lean=args.lean,
                                      lookups=args.lookups)
                        if mode == "Kpre":
                            pre_cs: dict = {}
                            c0 = contention.begin()
                            t0 = time.perf_counter()
                            v.precompute(Challenger())   # ends with a sync of the verifier's device
                            dt = time.perf_counter() - t0
                            contention.add(pre_cs, "verifier_precompute", c0)
                            r.rec("verifier_precompute", dt, "s", **_cs(pre_cs, "verifier_precompute"))
                    _reset_peaks(device)
                    _query(prover, v, tokens[:1])  # untimed warm-up
                    for i in range(args.queries):
                        _record_query(r, _query(prover, v, tokens[i:i + 1]), i)
                    _record_peaks(r, device)
                    # batch amortisation: B prompts against one set of u and openings (as the CNN suite)
                    if chal == "int" and mode in ("C", "Kpre"):
                        for bsz in args.batches:
                            _reset_peaks(device)
                            for tr in range(args.batch_trials):
                                _record_query(r, _query(prover, v, tokens[:bsz]), tr, batch=bsz)
                            _record_peaks(r, device, batch=bsz)
                    # one tampered query per cell: must be rejected
                    victim = mats[len(mats) // 2].name

                    def tamper(o, z, victim=victim):
                        if o.name == victim:
                            z = z.clone()
                            z.view(-1)[0] += 1
                        return z

                    res = _query(prover, v, tokens[:1], forward_kwargs={"tamper": tamper})
                    r.rec("tamper_rejected", int(not res["accepted"]), "", stage=res["rejected_at"])
                    r.done()
            # ---- attacks on this build (report Sec. 4.3): random single values anywhere, the top
            # logit, and one random block run with 1%-perturbed weights; mode C, lambda = 40
            cell = f"tamper_C_int_lam40_{base}"
            if (args.llm_tampers and coms is not None and n_layers == cfg.n_layers
                    and _todo(args, "llm", cfg.name, cell)):
                params = params_for(40, len(decoder_shapes(cfg)), rate=RATE, plan=coms.plan)
                v = _verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                              tables=coms.table_publics, lean=args.lean)
                r = Recorder("llm", cfg.name, cell, {"lam": 40, "mode": "C", "reps": params.reps,
                                                     "columns": params.columns, "seq": seq, "n_layers": n_layers,
                                                     "n_layers_full": cfg.n_layers, "lean": args.lean,
                                                     **({"prune_last": True} if prune else {}),
                                                     **_plan_config(coms, params)}, env)
                g = torch.Generator().manual_seed(123)
                x0 = tokens[:1]

                def attempt(kind, i, **kw):
                    res = _query(prover, v, x0, forward_kwargs=kw)
                    r.rec("rejected", int(not res["accepted"]), "", i, attack=kind, stage=res["rejected_at"])

                def t_logit(o, z):
                    if o.name == mats[-1].name:
                        z = z.clone()
                        k = int(z[:, 0].argmin())
                        z[k, 0] = int(z[:, 0].max()) + 1000
                    return z

                for i in range(args.llm_tampers):
                    op = mats[int(torch.randint(len(mats), (1,), generator=g))]
                    pos = int(torch.randint(10 ** 9, (1,), generator=g))
                    delta = int(torch.randint(1, 1000, (1,), generator=g)) * (1 if i % 2 else -1)

                    def t_single(o, z, op=op, pos=pos, delta=delta):
                        if o.name == op.name:
                            z = z.clone()
                            z.view(-1)[pos % z.numel()] += delta
                        return z

                    attempt("single_value", i, tamper=t_single)
                    attempt("output_logit", i, tamper=t_logit)
                    # the report's single-neuron tamper: one coordinate of the op feeding the LM head
                    # (the last block's down/fc2 projection), at the last position, set to Z_BOUND // 4
                    j = int(torch.randint(mats[-2].n_rows, (1,), generator=g))

                    def t_last(o, z, j=j):
                        if o.name == mats[-2].name:
                            z = z.clone()
                            z[j, -1] = Z_BOUND // 4
                        return z

                    attempt("last_hidden", i, tamper=t_last)
                proj = 1 if getattr(cfg, "embed_dim", 0) else 0    # OPT-350M: project_in / project_out
                first = (2 if cfg.pos == "learned" else 1) + proj  # token (and position) embedding
                per_block = (len(mats) - first - 1 - proj) // n_layers   # the LM head is last
                assert per_block * n_layers == len(mats) - first - 1 - proj, "unexpected decoder op layout"
                for i in range(max(3, args.llm_tampers // 10)):
                    b0 = first + per_block * int(torch.randint(n_layers, (1,), generator=g))
                    other = {}
                    for op in mats[b0:b0 + per_block]:
                        w = op.weight.to(torch.int16)
                        mask = (torch.rand(w.shape, generator=g) < 0.01).to(torch.int16)
                        sign = 2 * torch.randint(0, 2, w.shape, generator=g, dtype=torch.int16) - 1
                        other[op.name] = MatOp(op.name, op.inputs, op.output, bias=op.bias, layout=op.layout,
                                               weight=(w + mask * sign).clamp(-127, 127).to(torch.int8))
                    attempt("substituted_block_1pct", i, weights_override=other)
                    other = None
                r.done()
            # drop every reference to this build (``mats`` and ``weights`` keep its MatOps, and with
            # them the GPU weight copies, alive otherwise)
            prover = coms = graph = mats = weights = v = r = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if skipped:   # the job must not look complete
        raise SystemExit(f"{cfg.name}: builds {', '.join(skipped)} do not fit this GPU (the others ran)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["cnn", "llm"])
    ap.add_argument("--model", required=True,
                    help="CNN suite: mlp_mnist or a models.MODEL_SPECS name; LLM suite: a transformer.CONFIGS name")
    ap.add_argument("--queries", type=int, default=30, help="honest queries per defence cell")
    ap.add_argument("--tampers", type=int, default=100,
                    help="CNN suite: attacks per type in the tamper cell (0 = no tamper cell, e.g. timing controls)")
    ap.add_argument("--seq", type=int, nargs="+", default=[64], help="LLM suite: prompt lengths")
    ap.add_argument("--lams", type=int, nargs="+", default=list(LAMBDAS), help="LLM suite: security levels")
    ap.add_argument("--modes", default="",
                    help="LLM suite: e.g. C:int,Kpre:int (default: C:int, C:fs, Kpre:int and K:int)")
    ap.add_argument("--threads", type=int, default=0, help="verifier threads (0: torch's default)")
    ap.add_argument("--policy", default="paper",
                    help="commitment plan (pvi.fullcheck.plans) over the base rate 4: paper (the report), tight, "
                         "cnn<e> or R<rate>, the last three also with the suffix c (col layouts and lookup tables), "
                         "or auto (the c plan of fewest non-claim bytes within a setup budget, recorded as the "
                         "cells' policy); runs the commitment and mode-C cells only, named with a _pol<name> suffix")
    ap.add_argument("--prune-last", action="store_true",
                    help="LLM suite: the last decoder block at the last position only "
                         "(build_decoder(prune_last=True)); every cell named with a _prune suffix")
    ap.add_argument("--lookups", action="store_true",
                    help="modes K and Kpre: the verifier reads the embedding rows itself and the prover sends no "
                         "claims for them (Verifier(lookups=True)); runs those cells only, named with a _lookups "
                         "suffix")
    ap.add_argument("--tag", default="",
                    help="cell-name suffix of a variant run: _thr1, _thr12, _nolean, _nofix, _gpuv, _batch, _tf32, "
                         "_stream or _wire (--policy, --prune-last and --lookups add their own)")
    ap.add_argument("--builds", default="auto",
                    help="LLM block counts: 'auto' (all if <=12, else 1,2), 'full', or e.g. '1,2,full'")
    ap.add_argument("--llm-tampers", type=int, default=0,
                    help="LLM suite, full builds only: attacks per type in a tamper_C_int_lam40_<build> cell (0 = none)")
    ap.add_argument("--lean", action="store_true",
                    help="free dead activations and stream claims to the host (prover and verifier)")
    ap.add_argument("--cnn-batches", type=int, nargs="+", default=[8, 32],
                    help="CNN suite: batch sizes timed in the defence_C_int cells (default 8 32, as stored)")
    ap.add_argument("--batches", type=int, nargs="*", default=[],
                    help="LLM suite: also time B prompts in one interaction (C:int and Kpre:int cells)")
    ap.add_argument("--batch-trials", type=int, default=1, help="repetitions per batch size (both suites)")
    ap.add_argument("--verifier-device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (the report's headline) or cuda: run the client's checks on the prover's GPU "
                         "(needs a _gpuv tag)")
    ap.add_argument("--verifier-impl", default="default", choices=["default", "stream"],
                    help="'stream': the streaming verifier, the same verdicts (needs a _stream tag)")
    ap.add_argument("--wire", action="store_true",
                    help="the proof in the compact encoding of pvi.fullcheck.claimcodec (run_query(wire=True): "
                         "PVC3 claims, 31-bit u and columns; needs a _wire tag)")
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 tensor cores for the float32 GEMMs (exact: operands <= 255; needs a _tf32 tag)")
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM", ""),
                    help="a name for this GPU + CPU: records go to raw_<platform>/ ('' = raw/, the earliest "
                         "RTX 2080 Ti run, frozen)")
    return ap


def main() -> None:
    _require_this_pvi()
    args = build_parser().parse_args()
    global TAG, PLATFORM, RAW, VDEV, IMPL, WIRE
    TAG = args.tag
    VDEV = args.verifier_device
    IMPL = args.verifier_impl
    WIRE = args.wire
    if (IMPL == "stream") != ("_stream" in TAG):
        raise SystemExit("--verifier-impl stream goes with a --tag containing _stream, and only then")
    if WIRE != ("_wire" in TAG):
        raise SystemExit("--wire goes with a --tag containing _wire (the stored cells have none), and only then")
    if VDEV != "cpu" and "_gpuv" not in TAG:
        raise SystemExit("--verifier-device other than cpu needs a --tag containing _gpuv (a different client)")
    if (os.environ.get("PVI_LEGACY_WEIGHT_KEY") == "1") != ("_nofix" in TAG):
        raise SystemExit("PVI_LEGACY_WEIGHT_KEY=1 (the earliest run raw/'s weight re-upload) goes with a _nofix tag, "
                         "and only then")
    if args.tf32 and "_tf32" not in TAG:
        raise SystemExit("--tf32 needs a --tag containing _tf32 (the headline keeps TF32 off)")
    if args.tf32 and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        raise SystemExit("--tf32 with NVIDIA_TF32_OVERRIDE=0: cuBLAS would ignore it and the _tf32 cells would "
                         "hold non-TF32 timings (bench.sbatch: export PVI_TF32=1)")
    if "_pol" in TAG:
        raise SystemExit("--tag must not contain _pol: --policy adds it (a paper run must not take a policy's cells)")
    for flag, sfx in (("--prune-last", "_prune"), ("--lookups", "_lookups")):
        if sfx in TAG:
            raise SystemExit(f"--tag must not contain {sfx}: {flag} adds it")
    if args.suite == "cnn" and (args.prune_last or args.lookups):
        raise SystemExit("--prune-last and --lookups are for the decoders (the llm suite)")
    if args.lookups and args.policy != "paper":
        raise SystemExit("--lookups runs the K and Kpre cells, --policy the mode-C ones: one at a time")
    if args.prune_last:
        TAG += "_prune"
    if args.lookups:
        TAG += "_lookups"
    try:
        plan_commitment([], args.policy)
    except ValueError as exc:
        raise SystemExit(f"--policy: {exc}")
    if args.policy != "paper":        # its cells never share a name (or a median) with the report's
        TAG += f"_pol{args.policy}"
    PLATFORM = args.platform
    RAW = raw_root(PLATFORM)
    if args.threads:
        torch.set_num_threads(args.threads)
    # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE (NGC containers) only sets torch's default, which these
    # assignments override.  NVIDIA_TF32_OVERRIDE=0 (bench.sbatch) is different: cuBLAS then
    # ignores allow_tf32 altogether, hence the --tf32 check above.
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = False
    env = env_info()
    env["device"] = "cuda" if torch.cuda.is_available() else "cpu"   # where the prover runs
    print(json.dumps(env), flush=True)
    print(f"pvi from {env['pvi_path']}; verifier threads {env['torch_threads']} on {env.get('cpu_affinity', '?')} "
          f"CPUs / {env.get('cpu_affinity_cores', '?')} physical cores", flush=True)
    claim_platform(env)
    t0 = time.time()
    (suite_cnn if args.suite == "cnn" else suite_llm)(args, env)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
