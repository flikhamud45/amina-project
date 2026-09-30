"""The benchmark of the report (Sec. 4.3-4.4): our defence and Anchuri et al.'s path
test, on the models the literature benchmarks.  Every measurement is appended, one JSON
object per line, to ``artifacts/comparison/raw_<platform>/<suite>/<model>/<cell>.jsonl``;
nothing is aggregated here, so figures can be re-made later without re-running.

    python experiments/4_defence_benchmark/bench.py cnn --model vgg16
    python experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 64

A cell that has a ``.done`` marker is skipped (resume after pre-emption); pass
``--force`` to redo it.  An honest query that is rejected stops the job and keeps the
cell's records as ``<cell>.rejected-<run_id>.jsonl`` (``count_outcomes.py`` counts them).
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

from pvi.fullcheck.field import P  # noqa: E402
from pvi.fullcheck.graph import MatOp  # noqa: E402
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
TAG = ""  # appended to every cell name (``--tag``), so variant runs never collide
VDEV = "cpu"   # --verifier-device: where the client's checks run (the report: the CPU)
IMPL = "default"   # --verifier-impl: "stream" = the streaming verifier (Verifier(stream=True))
PLATFORM = ""  # ``--platform``: "" is the earlier RTX 2080 Ti run's root ``raw/`` (frozen)


def raw_root(platform: str) -> Path:
    """``raw/`` holds the records of the earliest RTX 2080 Ti run (frozen), ``raw_rtx2080ti-v2/``
    its re-run with the fixed code and ``raw_l40s/`` the report's numbers (both frozen too); every
    other prover/verifier pair writes its own ``raw_<platform>/``, so records of different
    hardware can never share a cell (or a median)."""
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
    """What a cross-cluster comparison needs beyond the GPU and CPU names."""
    import torchvision

    out = {"raw_dir": str(RAW), "cpu_count": os.cpu_count(), "numpy": np.__version__,
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
    """Every verifier of this benchmark runs on ``--verifier-device`` (the report: the CPU), with
    ``--verifier-impl`` (the report: the default; ``stream`` records one ``verify_total`` per query
    instead of the verify_* phases, which overlap)."""
    return Verifier(*args, device=VDEV, stream=IMPL == "stream", **kwargs)


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
                                    **({"verifier_impl": IMPL} if IMPL != "default" else {})),
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


def _todo(args, suite, model, cell) -> bool:
    """Run this cell?  Forced, or not yet done."""
    return args.force or not is_done(suite, model, cell)


def _record_query(r: Recorder, res: dict, trial: int, **extra) -> None:
    for k, v in res["timings"].items():
        r.rec(k, v, "s", trial, **extra)
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


def _gpu_index(dv) -> int | None:
    d = torch.device(dv)
    if d.type != "cuda":
        return None
    return torch.cuda.current_device() if d.index is None else d.index


def _peak_devices(device) -> list[tuple[str, int]]:
    """(metric suffix, GPU index) of every GPU whose memory a cell records: the prover's, and
    the verifier's (``_verifier``) when it is another GPU.  With the verifier on the prover's
    GPU (``--verifier-device cuda``) ``gpu_peak_memory`` is prover and client together."""
    p, v = _gpu_index(device), _gpu_index(VDEV)
    out = [] if p is None else [("", p)]
    if v is not None and v != p:
        out.append(("_verifier", v))
    return out


def _reset_peaks(device) -> None:
    """Start a new peak window for GPU allocations and host resident memory."""
    for _, i in _peak_devices(device):
        torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.reset_accumulated_memory_stats(i)
    try:  # Linux: "5" resets VmHWM (peak RSS) to the current RSS
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def _record_peaks(r: "Recorder", device, **extra) -> None:
    for sfx, i in _peak_devices(device):
        r.rec("gpu_peak_memory" + sfx, torch.cuda.max_memory_allocated(i), "B", **extra)
        r.rec("gpu_peak_reserved" + sfx, torch.cuda.max_memory_reserved(i), "B", **extra)
        st = torch.cuda.memory_stats(i)   # allocator churn (cudaMalloc calls, OOM retries) since the reset
        r.rec("gpu_alloc_retries" + sfx, st.get("num_alloc_retries", -1), "", **extra)
        r.rec("gpu_device_allocs" + sfx, st.get("num_device_alloc", -1), "", **extra)
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

    device = torch.device(args.device)
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
    rate = args.rate
    coms = commit_graph(graph, rate, device=device)

    prover = Prover(graph, device=device, commitments=coms)
    weights = {op.name: (op.weight, op.bias) for op in mats}
    for lam in LAMBDAS:
        for mode, chal in (("C", "int"), ("C", "fs"), ("K", "int"), ("Kpre", "int")):
            cell = f"defence_{mode}_{chal}_lam{lam}_rate{rate}"
            if not _todo(args, "cnn", name, cell):
                continue
            params = params_for(lam, len(mats), rate=rate, fiat_shamir=(chal == "fs"))
            cfg = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": params.rate,
                   "columns": params.columns, "threads": torch.get_num_threads()}
            r = Recorder("cnn", name, cell, cfg, env)
            shapes = [(op.row_length, rate * (1 << max(0, (op.row_length - 1).bit_length()))) for op in mats]
            r.rec("soundness_bits", soundness_bits(params, shapes, "C" if mode == "C" else "K"), "bits")
            if mode == "C":
                v = _verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
            else:
                v = _verifier(graph.public(), params, mode, weights=weights)
                if mode == "Kpre":
                    v.precompute(Challenger())
            _reset_peaks(device)
            run_query(prover, v, q_inputs[:1])  # untimed warm-up (lazy CUDA/BLAS initialisation)
            for i in range(args.queries):
                _record_query(r, run_query(prover, v, q_inputs[i:i + 1]), i)
            _record_peaks(r, device)
            # batch amortisation: B queries in one interaction (mode C, interactive only)
            if mode == "C" and chal == "int":
                for bsz in args.cnn_batches:
                    if bsz > len(q_inputs):
                        continue
                    _reset_peaks(device)
                    for tr in range(args.batch_trials):
                        _record_query(r, run_query(prover, v, q_inputs[:bsz]), tr, batch=bsz)
                    _record_peaks(r, device, batch=bsz)
            r.done()

    # ---- soundness experiments: every attack must be rejected (--tampers 0: none) ------------
    cell = "tamper_C_int_lam40"
    if args.tampers and _todo(args, "cnn", name, cell):
        params = params_for(40, len(mats), rate=rate)
        v = _verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
        r = Recorder("cnn", name, cell, {"lam": 40, "mode": "C", "reps": params.reps, "columns": params.columns}, env)
        g = torch.Generator().manual_seed(123)
        pen = _penultimate_mat(graph)
        x0 = q_inputs[:1]

        def attempt(kind, i, **kw):
            res = run_query(prover, v, x0, **kw)
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

        def kernel_vector(chi):
            """A nonzero delta with chi @ delta = 0 mod P (Gaussian elimination)."""
            r_, n_ = chi.shape
            if n_ <= r_:
                return None
            m = [[int(v) % P for v in chi[i_, :r_].tolist()] + [(-int(chi[i_, r_])) % P] for i_ in range(r_)]
            for col in range(r_):
                piv = next((i_ for i_ in range(col, r_) if m[i_][col]), None)
                if piv is None:
                    return None
                m[col], m[piv] = m[piv], m[col]
                inv = pow(m[col][col], P - 2, P)
                m[col] = [(v_ * inv) % P for v_ in m[col]]
                for i_ in range(r_):
                    if i_ != col and m[i_][col]:
                        f_ = m[i_][col]
                        m[i_] = [(vi - f_ * vc) % P for vi, vc in zip(m[i_], m[col])]
            delta = torch.zeros(n_, dtype=torch.int64)
            delta[:r_] = torch.tensor([m[i_][r_] for i_ in range(r_)], dtype=torch.int64)
            delta[r_] = 1
            return delta

        for i in range(max(3, args.tampers // 10)):
            prover.fold = forged_fold
            attempt("forged_fold", i, forward_kwargs={"tamper": t_forge})
            prover.fold = real_fold
            victim = mats[i % len(mats)].name

            def capture_fold(chis):
                state["chis"] = chis
                return real_fold(chis)

            def forged_open(cols, victim=victim):
                out = real_open(cols)
                c, pth = out[victim]
                delta = kernel_vector(state["chis"][victim])
                if delta is not None:
                    c = c.clone()
                    c[:, 0] = (c[:, 0] + delta) % P
                out[victim] = (c, pth)
                return out

            prover.fold, prover.open = capture_fold, forged_open
            attempt("forged_column", i)
            prover.fold, prover.open = real_fold, real_open
        r.done()


# ------------------------------------------------------------------ LLM shapes
def suite_llm(args, env) -> None:
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.transformer import CONFIGS, build_decoder, decoder_param_count

    device = torch.device(args.device)
    cfg = CONFIGS[args.model]
    if args.builds == "auto":      # the report's default: all blocks up to 12, else 1 and 2
        builds = [cfg.n_layers] if cfg.n_layers <= 12 else [1, 2]
    else:                          # e.g. "full" or "1,2,full" (L1, L2 and the full model in one job)
        builds = [cfg.n_layers if b == "full" else int(b) for b in args.builds.split(",")]
    modes = [("C", "int"), ("C", "fs"), ("Kpre", "int"), ("K", "int")] if not args.modes else \
        [tuple(m.split(":")) for m in args.modes.split(",")]
    skipped = []
    for seq in args.seq:
        for n_layers in builds:
            base = f"T{seq}_L{n_layers}" + ("" if args.rate == 4 else f"_rate{args.rate}")
            todo = [f"defence_{m}_{c}_lam{l}_{base}" for m, c in modes for l in args.lams]
            if args.llm_tampers and n_layers == cfg.n_layers and any(m == "C" for m, _ in modes):
                todo.append(f"tamper_C_int_lam40_{base}")
            if not args.force and all(is_done("llm", cfg.name, t) for t in todo):
                continue
            if device.type == "cuda":  # a build whose int8 weights alone fill the GPU would only OOM
                need = sum(sh.n_rows * sh.row_length for sh in decoder_shapes(cfg, n_layers=n_layers))
                have = torch.cuda.get_device_properties(device).total_memory
                if need > 0.85 * have:     # ... after its build and commit: skip it, run the others
                    print(f"SKIP {cfg.name} T{seq} L{n_layers}: {need / 2**30:.1f} GiB of int8 weights "
                          f"> 85% of this GPU's {have / 2**30:.0f} GiB", flush=True)
                    skipped.append(f"T{seq}_L{n_layers}")
                    continue
            _reset_peaks(device)
            t0 = time.perf_counter()
            graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, 32), seed=0)
            build_s = time.perf_counter() - t0
            mats = graph.mat_ops
            # the first ``queries`` rows are the single-prompt queries of the stored runs
            tokens = torch.randint(0, cfg.vocab, (max([args.queries, *args.batches]), seq),
                                   generator=torch.Generator().manual_seed(1))
            coms = None
            if any(m == "C" for m, _ in modes):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                coms = commit_graph(graph, args.rate, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                commit_s = time.perf_counter() - t0
                cell = f"commit_{base}"
                if _todo(args, "llm", cfg.name, cell):
                    r = Recorder("llm", cfg.name, cell, {"seq": seq, "n_layers": n_layers, "rate": args.rate,
                                                         "n_layers_full": cfg.n_layers, "lean": args.lean,
                                                         "params_full": decoder_param_count(cfg)}, env)
                    r.rec("build_graph", build_s, "s")
                    r.rec("commit_total", commit_s, "s")
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
                    full_shapes = decoder_shapes(cfg)
                    params = params_for(lam, len(full_shapes), rate=args.rate, fiat_shamir=(chal == "fs"))
                    conf = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": args.rate,
                            "columns": params.columns, "seq": seq, "n_layers": n_layers,
                            "n_layers_full": cfg.n_layers, "params_full": decoder_param_count(cfg),
                            "params_built": graph.n_params(), "lean": args.lean,
                            "threads": torch.get_num_threads()}
                    r = Recorder("llm", cfg.name, cell, conf, env)
                    if mode == "C":
                        v = _verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()},
                                     lean=args.lean)
                    else:
                        v = _verifier(graph.public(), params, mode, weights=weights, lean=args.lean)
                        if mode == "Kpre":
                            t0 = time.perf_counter()
                            v.precompute(Challenger())   # ends with a sync of the verifier's device
                            r.rec("verifier_precompute", time.perf_counter() - t0, "s")
                    _reset_peaks(device)
                    run_query(prover, v, tokens[:1])  # untimed warm-up
                    for i in range(args.queries):
                        _record_query(r, run_query(prover, v, tokens[i:i + 1]), i)
                    _record_peaks(r, device)
                    # batch amortisation: B prompts against one set of u and openings (as the CNN suite)
                    if chal == "int" and mode in ("C", "Kpre"):
                        for bsz in args.batches:
                            _reset_peaks(device)
                            for tr in range(args.batch_trials):
                                _record_query(r, run_query(prover, v, tokens[:bsz]), tr, batch=bsz)
                            _record_peaks(r, device, batch=bsz)
                    # one tampered query per cell: must be rejected
                    victim = mats[len(mats) // 2].name

                    def tamper(o, z, victim=victim):
                        if o.name == victim:
                            z = z.clone()
                            z.view(-1)[0] += 1
                        return z

                    res = run_query(prover, v, tokens[:1], forward_kwargs={"tamper": tamper})
                    r.rec("tamper_rejected", int(not res["accepted"]), "", stage=res["rejected_at"])
                    r.done()
            # ---- attacks on this build (report Sec. 4.3): random single values anywhere, the top
            # logit, and one random block run with 1%-perturbed weights; mode C, lambda = 40
            cell = f"tamper_C_int_lam40_{base}"
            if (args.llm_tampers and coms is not None and n_layers == cfg.n_layers
                    and _todo(args, "llm", cfg.name, cell)):
                params = params_for(40, len(decoder_shapes(cfg)), rate=args.rate)
                v = _verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()},
                             lean=args.lean)
                r = Recorder("llm", cfg.name, cell, {"lam": 40, "mode": "C", "reps": params.reps,
                                                     "columns": params.columns, "seq": seq, "n_layers": n_layers,
                                                     "n_layers_full": cfg.n_layers, "lean": args.lean}, env)
                g = torch.Generator().manual_seed(123)
                x0 = tokens[:1]

                def attempt(kind, i, **kw):
                    res = run_query(prover, v, x0, forward_kwargs=kw)
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--tampers", type=int, default=100,
                    help="CNN suite: attacks per type in the tamper cell (0 = no tamper cell, e.g. timing controls)")
    ap.add_argument("--seq", type=int, nargs="+", default=[64])
    ap.add_argument("--lams", type=int, nargs="+", default=list(LAMBDAS))
    ap.add_argument("--modes", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rate", type=int, default=4, help="Reed-Solomon rate (codeword / message length)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for cell names, e.g. _thr1")
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
    ap.add_argument("--verifier-device", default="cpu",
                    help="cpu (the report) or cuda / cuda:1: run the client's checks on a GPU (needs a _gpuv tag)")
    ap.add_argument("--verifier-impl", default="default", choices=["default", "stream"],
                    help="'stream': the streaming verifier, the same verdicts (needs a _stream tag)")
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 tensor cores for the float32 GEMMs (exact: operands <= 255; needs a _tf32 tag)")
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM", ""),
                    help="hardware name, e.g. h100: records go to raw_<platform>/ ('' = raw/, the earlier "
                         "RTX 2080 Ti run, frozen)")
    return ap


def main() -> None:
    _require_this_pvi()
    args = build_parser().parse_args()
    global TAG, PLATFORM, RAW, VDEV, IMPL
    TAG = args.tag
    VDEV = args.verifier_device
    IMPL = args.verifier_impl
    if (IMPL == "stream") != ("_stream" in TAG):
        raise SystemExit("--verifier-impl stream goes with a --tag containing _stream, and only then")
    if VDEV != "cpu" and "_gpuv" not in TAG:
        raise SystemExit("--verifier-device other than cpu needs a --tag containing _gpuv (a different client)")
    if (os.environ.get("PVI_LEGACY_WEIGHT_KEY") == "1") != ("_nofix" in TAG):
        raise SystemExit("PVI_LEGACY_WEIGHT_KEY=1 (the stored runs' weight re-upload) goes with a _nofix tag, and only then")
    if args.tf32 and "_tf32" not in TAG:
        raise SystemExit("--tf32 needs a --tag containing _tf32 (the headline keeps TF32 off)")
    if args.tf32 and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        raise SystemExit("--tf32 with NVIDIA_TF32_OVERRIDE=0: cuBLAS would ignore it and the _tf32 cells would "
                         "hold non-TF32 timings (bench.sbatch: export PVI_TF32=1)")
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
    env["device"] = args.device
    print(json.dumps(env), flush=True)
    print(f"pvi from {env['pvi_path']}; verifier threads {env['torch_threads']} on {env.get('cpu_affinity', '?')} "
          f"CPUs / {env.get('cpu_affinity_cores', '?')} physical cores", flush=True)
    claim_platform(env)
    t0 = time.time()
    (suite_cnn if args.suite == "cnn" else suite_llm)(args, env)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
