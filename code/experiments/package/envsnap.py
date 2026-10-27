"""The node's environment, written from inside the run: one snapshot per node and allocation.

Every job of ``run.sh`` (and a ``--local`` run, a ``--one-job`` allocation among them, before its first job) runs,
after its node check and before its work,

    python -m experiments.package.envsnap --out logs/<name>/env --inputs

which writes ``<out>/<host>-<tag>/`` (``tag``: ``--tag``, else the SLURM job, else ``PVI_RUN_ID``, else ``local``)
unless that folder is complete already (its ``env.json``), so the jobs of one allocation share one snapshot and every
allocation, on whichever node SLURM gave it, leaves its own:

* ``nvidia-smi-q.txt`` (``nvidia-smi -q``: the GPU, its driver, clocks, power limits, ECC and persistence mode),
  ``lscpu.txt``, ``numactl.txt`` (``numactl -H``), ``free.txt`` (``free -g``), ``uname.txt`` (``uname -a``),
  ``os-release.txt``, ``cpufreq.txt`` (each CPU's frequency governor and the boost setting), ``cxx.txt``
  (``$CXX --version``: the compiler of the native ``-march=native`` kernels), ``pip-freeze.txt`` (the interpreter's
  packages), ``git.txt`` (``git rev-parse HEAD`` and ``git status --porcelain``) and ``environ.txt`` (the
  variables of the run that name places, threads and devices; never a token);
* ``env.json``: the fields compared between nodes (:data:`COMPARED`): the GPU, its driver, power limit and maximum
  clocks, the CPU model and a SHA-256 of its flags, the kernel, the governors and boost, the compiler, the torch and
  Python versions, a SHA-256 of the package list, the commit and whether the checkout has changes.

With ``--inputs`` it also writes, once per run folder, ``<out>/inputs.sha256`` (``sha256sum``'s format): the trained
image models (``artifacts/models/cnn/*.pt``), the ImageNet subsets' caches, CIFAR-10's batches and MNIST's files that
the bench tier reads, and ``<out>/models-check.txt``: ``artifacts/models/MODELS.sha256`` checked against the stored
networks (``OK`` / ``FAILED`` per file).  The records name each model's SHA-256 themselves
(:func:`experiments.common.models.note_file`).

``experiments/paper/numbers.py --check`` reads every ``env.json`` (:func:`snapshots`) and prints a ``WARNING env``
line for each field of :data:`COMPARED` on which the nodes differ (the GPU and CPU models are checked, and fail,
through the jobs' node lines).  Nothing here can fail a job: each command that fails or is missing leaves its error
in its file, and the module exits with status 0.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

__all__ = ["COMPARED", "snapshot", "write_inputs", "snapshots", "differences", "main"]

CODE = Path(__file__).resolve().parents[2]
COMPARED = ("gpu", "driver", "gpu_power_limit_w", "gpu_max_sm_mhz", "cpu", "cpu_flags_sha256", "kernel", "governors",
            "boost", "cxx", "torch", "python", "pip_sha256", "git")
"""The fields of ``env.json`` that ``numbers.py --check`` compares between the run's nodes."""
VARIABLES = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE",
             "PYTORCH_CUDA_ALLOC_CONF", "TRITON_CACHE_DIR", "PVI_NATIVE_DIR", "CXX", "CMAKE", "HF_HOME", "TMPDIR",
             "PVI_CIFAR_ROOT", "PVI_IMAGENET_ROOT", "PVI_CACHE", "PVI_PY_INCLUDE", "SLURM_JOB_ID",
             "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "PATH")
"""The variables ``environ.txt`` keeps (none holds a secret)."""


def _run(cmd: list[str], timeout: float = 120) -> str:
    """A command's output, or its error as text."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout + (f"\n[exit status {r.returncode}] {r.stderr.strip()}" if r.returncode else "")
    except (OSError, subprocess.SubprocessError) as e:
        return f"[not run: {type(e).__name__}: {e}]"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _cpufreq() -> tuple[str, list[str], str | None]:
    """``(text, governors, boost)``: each CPU's governor (counted), the boost setting (``cpufreq/boost`` or
    intel_pstate's ``no_turbo``)."""
    govs: dict = {}
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")):
        g = _read(p)
        if g:
            govs[g] = govs.get(g, 0) + 1
    boost = _read("/sys/devices/system/cpu/cpufreq/boost")
    if boost is None:
        nt = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
        boost = None if nt is None else str(1 - int(nt)) if nt.isdigit() else nt
    driver = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
    text = "\n".join([f"governor {g}: {n} CPUs" for g, n in sorted(govs.items())]
                     + [f"boost: {boost}", f"scaling driver: {driver}"])
    return text, sorted(govs), boost


def _cpu_flags() -> tuple[str | None, str | None]:
    """``(the CPU model, SHA-256 of its flags)`` from ``/proc/cpuinfo``."""
    text = _read("/proc/cpuinfo") or ""
    model = next((x.split(":", 1)[1].strip() for x in text.splitlines() if x.startswith("model name")), None)
    flags = next((x.split(":", 1)[1].strip() for x in text.splitlines() if x.startswith("flags")), None)
    digest = hashlib.sha256(" ".join(sorted(flags.split())).encode()).hexdigest() if flags else None
    return model or platform.processor() or None, digest


def _smi_field(q: str, gpu: str) -> str | None:
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", gpu], timeout=60)
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    return None if not first or first.startswith("[") or "not supported" in first.lower() else first


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def snapshot(folder: Path, python: str = sys.executable) -> dict:
    """Write the snapshot's files into ``folder`` and return ``env.json``'s fields (written last, so that a folder
    with ``env.json`` is complete)."""
    from experiments.package.telemetry import gpu_id
    folder.mkdir(parents=True, exist_ok=True)
    gpu = gpu_id() if shutil.which("nvidia-smi") is not None else None     # the job's GPU, as nvidia-smi -i takes it
    smi = gpu is not None

    def put(name: str, text: str) -> None:
        (folder / name).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

    put("nvidia-smi-q.txt", _run(["nvidia-smi", "-q", "-i", gpu]) if smi else "[no nvidia-smi]")
    put("lscpu.txt", _run(["lscpu"]))
    put("numactl.txt", _run(["numactl", "-H"]))
    put("free.txt", _run(["free", "-g"]))
    put("uname.txt", _run(["uname", "-a"]))
    put("os-release.txt", _read("/etc/os-release") or "[no /etc/os-release]")
    freq, governors, boost = _cpufreq()
    put("cpufreq.txt", freq)
    cxx = os.environ.get("CXX")
    cxx_text = _run([cxx, "--version"]) if cxx else "[CXX is not set]"
    put("cxx.txt", cxx_text)
    pip = _run([python, "-m", "pip", "freeze", "--all"], timeout=300)
    put("pip-freeze.txt", pip)
    head = _run(["git", "-c", "safe.directory=*", "-C", str(CODE), "rev-parse", "HEAD"]).strip()
    status = _run(["git", "-c", "safe.directory=*", "-C", str(CODE), "status", "--porcelain", "--untracked-files=no"])
    put("git.txt", f"{head}\n{status}")
    put("environ.txt", "\n".join(f"{k}={os.environ[k]}" for k in VARIABLES if k in os.environ))
    cpu, flags = _cpu_flags()
    pkgs = sorted(x for x in pip.splitlines() if x and not x.startswith("["))
    out = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "gpu_id": gpu, "gpu": _smi_field("name", gpu) if smi else None,
           "driver": _smi_field("driver_version", gpu) if smi else None,
           "gpu_power_limit_w": _smi_field("power.limit", gpu) if smi else None,
           "gpu_max_sm_mhz": _smi_field("clocks.max.sm", gpu) if smi else None,
           "gpu_persistence": _smi_field("persistence_mode", gpu) if smi else None,
           "cpu": cpu, "cpu_flags_sha256": flags, "kernel": platform.release(), "governors": governors,
           "boost": boost, "cxx": (cxx_text.strip().splitlines() or [None])[0] if cxx else None,
           "torch": _version("torch"), "python": platform.python_version(),
           "pip_sha256": hashlib.sha256("\n".join(pkgs).encode()).hexdigest() if pkgs else None,
           "pip_packages": len(pkgs), "git": head if re.fullmatch(r"[0-9a-f]{40}", head) else None,
           "git_dirty": bool(status.strip()) if not status.startswith("[") else None}
    tmp = folder / "env.json.tmp"
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, folder / "env.json")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def input_files() -> list[Path]:
    """The bench tier's model and data files that exist here: the trained image models, the ImageNet subsets'
    caches (``PVI_CACHE``), CIFAR-10's batches (``PVI_CIFAR_ROOT``) and MNIST's files."""
    data = CODE / "data"
    cache = Path(os.environ.get("PVI_CACHE") or CODE / "artifacts" / "cache")
    cifar = Path(os.environ.get("PVI_CIFAR_ROOT") or data / "cifar10") / "cifar-10-batches-py"
    paths = [*sorted((CODE / "artifacts" / "models" / "cnn").glob("*.pt")),
             *(cache / f"{n}.pt" for n in ("dogs_cats", "dogs_squirrels")), *sorted(cifar.glob("*")),
             *sorted((data / "MNIST" / "raw").glob("*"))]
    return [p for p in paths if p.is_file()]


def write_inputs(out: Path) -> Path | None:
    """``<out>/inputs.sha256`` (once: an existing file is kept) and ``<out>/models-check.txt``; the path written, or
    None if it was there."""
    target = out / "inputs.sha256"
    if target.exists():
        return None
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in input_files():
        try:
            rel = p.relative_to(CODE)
        except ValueError:
            rel = p
        lines.append(f"{_sha256(p)}  {rel.as_posix()}")
    check = []
    stored = CODE / "artifacts" / "models" / "MODELS.sha256"
    if stored.is_file():
        for line in stored.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            path = stored.parent / parts[1]
            ok = path.is_file() and _sha256(path) == parts[0]
            check.append(f"{parts[1]}: {'OK' if ok else 'FAILED'}")
    (out / "models-check.txt").write_text("\n".join(check) + "\n", encoding="utf-8")
    handle, tmp = tempfile.mkstemp(dir=str(out), prefix=".inputs")
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, target)
    return target


def snapshots(folder: Path) -> list[dict]:
    """Every ``env.json`` under ``folder`` (``logs/<name>/env``), each with its folder's name as ``snapshot``."""
    out = []
    for p in sorted(Path(folder).glob("*/env.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({**d, "snapshot": p.parent.name})
    return out


def differences(snaps: list[dict]) -> dict[str, dict[str, list[str]]]:
    """``{field: {value: [snapshots]}}`` of the fields of :data:`COMPARED` on which the snapshots differ."""
    out = {}
    for key in COMPARED:
        values: dict = {}
        for s in snaps:
            v = s.get(key)
            values.setdefault(json.dumps(v) if isinstance(v, (list, dict)) else str(v), []).append(s["snapshot"])
        if len(values) > 1:
            out[key] = values
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the run's env folder (logs/<name>/env)")
    ap.add_argument("--tag", default=None, help="the snapshot's tag (default: the SLURM job, else PVI_RUN_ID, else "
                                                "local)")
    ap.add_argument("--inputs", action="store_true", help="also write <out>/inputs.sha256 once per run folder")
    ap.add_argument("--python", default=sys.executable, help="the interpreter whose packages pip-freeze.txt lists")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = Path(args.out)
        tag = args.tag or os.environ.get("SLURM_JOB_ID") or os.environ.get("PVI_RUN_ID") or "local"
        folder = out / f"{socket.gethostname()}-{tag}"
        if (folder / "env.json").is_file():
            print(f"env snapshot: {folder} (written earlier)")
        else:
            d = snapshot(folder, args.python)
            print(f"env snapshot: {folder} (GPU {d.get('gpu')}, driver {d.get('driver')}, CPU {d.get('cpu')})")
        if args.inputs:
            written = write_inputs(out)
            if written is not None:
                print(f"env snapshot: {written}")
    except Exception as e:                              # noqa: BLE001 - the snapshot never fails a job
        print(f"env snapshot: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
