"""Run records: the environment of a run, peak memory, timing, and writing results.

Every timed script prints one JSON object per line (:func:`emit`) and starts it with :func:`environment`, so that a
record says which commit, host, GPU, library versions and implementation options produced it.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from pvi import options
from pvi.core import contention

__all__ = ["git_sha", "environment", "emit", "write_json", "reset_peaks", "peaks", "sync", "timed"]


# ------------------------------------------------------------------ the commit
def _git_dir(path: Path) -> Path | None:
    for d in (path, *path.parents):
        dot = d / ".git"
        if dot.is_dir():
            return dot
        if dot.is_file():                            # a worktree or submodule: "gitdir: <path>"
            text = dot.read_text().strip()
            if text.startswith("gitdir:"):
                g = Path(text.split(":", 1)[1].strip())
                return g if g.is_absolute() else (d / g).resolve()
    return None


def _read_ref(git_dir: Path, ref: str) -> str | None:
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    for d in (git_dir, common):
        f = d / ref
        if f.is_file():
            return f.read_text().strip()
        packed = d / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    return None


def git_sha(path: Path | str = ".", *, short: int = 7) -> str:
    """The abbreviated commit of the repository holding ``path`` (``git rev-parse --short HEAD``); on a host without
    a ``git`` binary, ``HEAD`` and its ref are read from the repository directly.  ``""`` when neither works."""
    path = Path(path).resolve()
    try:
        sha = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", f"--short={short}", "HEAD"],
                             capture_output=True, text=True, cwd=path, timeout=30).stdout.strip()
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        git_dir = _git_dir(path)
        if git_dir is None:
            return ""
        head = (git_dir / "HEAD").read_text().strip()
        full = _read_ref(git_dir, head[4:].strip()) if head.startswith("ref:") else head
        return (full or "")[:short]
    except OSError:
        return ""


def _git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, cwd=path, timeout=30)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------ the environment
def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            return next(line.split(":", 1)[1].strip() for line in fh if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def _gpu_driver() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device=None) -> dict:
    """The run's environment for a record: the commit (and whether the checkout has uncommitted changes), the
    host, CPU and threads, the GPU and its driver, the Python, torch and CUDA versions, the scheduler's job and
    the implementation options with what this host runs (:func:`pvi.options.describe`)."""
    from experiments.common.paths import CODE
    import pvi
    cuda = torch.cuda.is_available()
    out = {"git": git_sha(CODE), "git_dirty": _git_dirty(CODE), "host": socket.gethostname(), "cpu": _cpu_name(),
           "cpu_count": os.cpu_count(), "threads": torch.get_num_threads(),
           "gpu": torch.cuda.get_device_name(0) if cuda else None,
           "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory if cuda else None,
           "driver": _gpu_driver() if cuda else None, "python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "pvi": str(Path(pvi.__file__).resolve().parent), "argv": sys.argv[1:],
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "schedstat": contention.available(),
           "implementation": options.describe(device)}
    try:
        out["cpu_affinity"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return out


# ------------------------------------------------------------------ output
def emit(record: dict, out: Path | str | None = None) -> dict:
    """Print ``record`` as one JSON line (and append it to ``out``)."""
    line = json.dumps(record, default=float)
    print(line, flush=True)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as fh:
            fh.write(line + "\n")
    return record


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as JSON atomically (a temporary file in the same folder, renamed over the target), so that
    an interrupted run never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_result_")
    try:
        with os.fdopen(handle, "w") as fh:
            json.dump(payload, fh, indent=indent)
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


# ------------------------------------------------------------------ memory
def reset_peaks(device) -> None:
    """Start a new peak window: the GPU allocator's statistics (on a GPU) and the host's peak resident set (Linux)."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
    try:                                             # Linux: "5" resets VmHWM (peak RSS) to the current RSS
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def peaks(device) -> dict:
    """Bytes since :func:`reset_peaks`: the host's peak resident set (``None`` where ``/proc`` has none), and on a
    GPU its peak allocated and reserved memory and the allocator's retries."""
    out = {"host_peak_rss": None}
    try:
        with open("/proc/self/status") as fh:
            out["host_peak_rss"] = 1024 * next(int(line.split()[1]) for line in fh if line.startswith("VmHWM:"))
    except (OSError, StopIteration):
        pass
    if torch.device(device).type == "cuda":
        out["gpu_peak_memory"] = torch.cuda.max_memory_allocated(device)
        out["gpu_peak_reserved"] = torch.cuda.max_memory_reserved(device)
        out["gpu_alloc_retries"] = torch.cuda.memory_stats(device).get("num_alloc_retries", -1)
    return out


# ------------------------------------------------------------------ timing
def sync(device) -> None:
    """Wait for the GPU's work (nothing on a CPU)."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device):
    """``(fn(), seconds, wait_frac)``: on a GPU between two CUDA events, the first recorded after a synchronisation
    and the second after ``fn`` returns and the device has finished every stream's work (so the window ends when
    the GPU has finished ``fn``'s work and holds the host work ``fn`` did meanwhile); on a CPU with
    ``perf_counter``.  ``wait_frac``: the share of the window this process waited for a CPU (:mod:`pvi.core.
    contention`; ``None`` where the kernel reports no scheduler statistics)."""
    device = torch.device(device)
    c0 = contention.begin()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        torch.cuda.synchronize(device)               # work fn left on another stream: the end event sees its own only
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1e3
    else:
        t0 = time.perf_counter()
        result = fn()
        seconds = time.perf_counter() - t0
    store: dict = {}
    contention.add(store, "window", c0)
    return result, seconds, (store.get("window") or {}).get("wait_frac")
