"""The GPU's telemetry in the background of a job: clocks, power, temperature, utilisation and throttle reasons.

Every job of ``run.sh`` starts this sampler after its node check and before its work; a CPU-timing job then pins
itself to its cores and moves the sampler, its threads and its ``nvidia-smi`` to the job's other CPUs
(``experiments/package/pin.py --away``), so that the sampler never runs on the timed process's cores:

    python -m experiments.package.telemetry --out logs/<name>/telemetry/sp-<cell>-<job id> --interval 1 --watch <pid>

It runs one ``nvidia-smi --query-gpu=... -lms <interval>`` for the job's GPU (CUDA device 0 under
``CUDA_VISIBLE_DEVICES``, found by its UUID) at ``--interval`` seconds (at least 1, so that the sampling does not
perturb the timings: one sample costs ``nvidia-smi`` a few milliseconds of one CPU), and writes

* ``<out>.csv``: one line per sample (:data:`FIELDS` that this driver knows; the throttle reasons as NVML's bit mask,
  :data:`REASONS`), flushed line by line, compressed to ``<out>.csv.gz`` when the sampler stops;
* ``<out>.apps.txt``: the GPU's compute processes (``nvidia-smi --query-compute-apps``) when the sampler starts,
  before the job's own process exists (any process listed there is another's), and when it stops;
* ``<out>.json``: the summary (:func:`summarise`): the samples, the time span, the share of samples with each
  throttle reason, the SM clock's minimum, median and maximum, the largest temperature, the mean and largest power
  and the energy (the power integrated over the samples), and the processes found at the start.

It stops on SIGTERM or SIGINT, or when the process ``--watch`` (the job's shell) has ended, whichever comes first, so
that a job stopped by its time limit leaves no sampler behind.  Nothing it does can fail the job: without
``nvidia-smi`` or a GPU it writes a summary that says so and exits with status 0.  ``experiments/paper/numbers.py
--check`` reads the summaries (:func:`read`) and prints a ``WARNING telemetry`` line for each timed job during which
the GPU was slowed by a thermal or power limit (:data:`SLOWDOWN`); a warning changes no exit status.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

__all__ = ["FIELDS", "REASONS", "SLOWDOWN", "summarise", "read", "main"]

FIELDS = ("timestamp", "clocks.sm", "clocks.mem", "clocks.max.sm", "power.draw", "power.limit", "temperature.gpu",
          "temperature.memory", "utilization.gpu", "utilization.memory", "memory.used", "pstate",
          "clocks_event_reasons.active")
"""The ``nvidia-smi --query-gpu`` fields sampled (those this driver does not know are left out; the throttle reasons'
mask is ``clocks_throttle_reasons.active`` on drivers that predate ``clocks_event_reasons``)."""
ALIASES = {"clocks_event_reasons.active": "clocks_throttle_reasons.active"}
REASONS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
           0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
           0x80: "hw_power_brake_slowdown", 0x100: "display_clocks_setting"}
"""NVML's clock event (throttle) reasons by bit."""
SLOWDOWN = ("sw_power_cap", "hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown")
"""The reasons that mean the GPU ran below its clocks because of a power or thermal limit."""


def _smi(args: list[str], timeout: float = 30) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_id() -> str | None:
    """The job's GPU as ``nvidia-smi -i`` takes it: the first entry of ``CUDA_VISIBLE_DEVICES`` if it is a UUID, the
    only GPU ``nvidia-smi`` lists, else CUDA device 0's UUID from torch (a short-lived process), else index 0; None
    without ``nvidia-smi``."""
    out = _smi(["--query-gpu=uuid", "--format=csv,noheader"])
    if out is None or out.returncode != 0:
        return None
    uuids = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    first = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if first.startswith(("GPU-", "MIG-")):
        return first
    if len(uuids) == 1:
        return uuids[0]
    code = ("import torch; p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
            "print(getattr(p, 'uuid', '') or '')")
    try:
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        got = ""
    if got:
        return got if got.startswith("GPU-") else f"GPU-{got}"
    return "0"


def known_fields(gpu: str) -> list[str]:
    """:data:`FIELDS` that this ``nvidia-smi`` accepts (each tried alone; an alias where the name is unknown)."""
    out = []
    for f in FIELDS:
        for name in (f, ALIASES.get(f)):
            if name is None:
                continue
            r = _smi([f"--query-gpu={name}", "--format=csv,noheader", "-i", gpu])
            if r is not None and r.returncode == 0 and r.stdout.strip():
                out.append(name)
                break
    return out


def _apps(gpu: str) -> str:
    r = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", gpu])
    return "" if r is None or r.returncode != 0 else r.stdout.strip()


def _num(x: str) -> float | None:
    x = x.strip().split(" ")[0]
    try:
        return float(int(x, 16)) if x.lower().startswith("0x") else float(x)
    except ValueError:
        return None


def _time(x: str) -> float | None:
    """``nvidia-smi``'s timestamp (``2026/10/27 12:00:01.123``) as seconds since the epoch (local time)."""
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            import datetime
            return datetime.datetime.strptime(x.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def summarise(rows: list[dict], interval: float | None = None) -> dict:
    """The summary of samples (``rows``: ``{field: text}`` of :data:`FIELDS`): ``samples``, ``span_s``, the share of
    samples with each throttle reason (``reasons``, by :data:`REASONS` name; only those that occurred),
    ``slowdown`` (the reasons of :data:`SLOWDOWN` that occurred, with their shares), ``throttled`` (any of them),
    the SM clock (``sm_clock_mhz``: min, median, max), ``temperature_max_c``, ``power_w`` (mean, max), ``energy_j``
    (the power integrated over the samples' times, each sample held until the next) and ``utilization_mean``."""
    def col(name):
        for key in (name, ALIASES.get(name)):
            if key and any(key in r for r in rows):
                return [_num(r.get(key, "")) for r in rows]
        return [None] * len(rows)

    out: dict = {"samples": len(rows), "interval_s": interval}
    times = [_time(r.get("timestamp", "")) for r in rows]
    ts = [t for t in times if t is not None]
    out["span_s"] = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    masks = [int(m) for m in col("clocks_event_reasons.active") if m is not None]
    shares = {}
    for bit, name in REASONS.items():
        n = sum(1 for m in masks if m & bit)
        if n:
            shares[name] = n / len(masks)
    out["reasons"] = shares
    out["slowdown"] = {k: v for k, v in shares.items() if k in SLOWDOWN}
    out["throttled"] = bool(out["slowdown"])
    out["reason_samples"] = len(masks)
    sm = [x for x in col("clocks.sm") if x is not None]
    out["sm_clock_mhz"] = {"min": min(sm), "median": statistics.median(sm), "max": max(sm)} if sm else None
    temp = [x for x in col("temperature.gpu") if x is not None]
    out["temperature_max_c"] = max(temp) if temp else None
    power = col("power.draw")
    pw = [x for x in power if x is not None]
    out["power_w"] = {"mean": statistics.fmean(pw), "max": max(pw)} if pw else None
    energy = 0.0
    for (t0, p0), (t1, _) in zip(zip(times, power), zip(times[1:], power[1:])):
        if None not in (t0, t1, p0) and t1 >= t0:
            energy += p0 * (t1 - t0)
    out["energy_j"] = energy if pw else None
    util = [x for x in col("utilization.gpu") if x is not None]
    out["utilization_mean"] = statistics.fmean(util) if util else None
    return out


def _rows(text: str) -> list[dict]:
    return [r for r in csv.DictReader(io.StringIO(text), skipinitialspace=True) if r]


def read(stem: Path) -> dict | None:
    """The summary of a sampler's files (``stem``: the path without its suffixes): ``<stem>.json`` if it is there,
    else the summary of ``<stem>.csv`` or ``<stem>.csv.gz`` (a sampler that was killed before it wrote one), with
    ``apps_at_start`` from ``<stem>.apps.txt``; None without any of them."""
    stem = Path(stem)
    js = stem.with_name(stem.name + ".json")
    if js.is_file():
        try:
            return json.loads(js.read_text(encoding="utf-8"))
        except ValueError:
            pass
    for suffix, opener in ((".csv", open), (".csv.gz", gzip.open)):
        path = stem.with_name(stem.name + suffix)
        if path.is_file():
            try:
                with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
                    out = summarise(_rows(fh.read()))
            except (OSError, EOFError):
                continue                   # a compression cut short: try the other file
            out["apps_at_start"] = _apps_at_start(stem)
            out["partial"] = True
            return out
    return None


def _apps_at_start(stem: Path) -> list[str]:
    path = stem.with_name(stem.name + ".apps.txt")
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines[1:]:
        if line.startswith("#"):
            break
        if line.strip():
            out.append(line.strip())
    return out


class Sampler:
    """One ``nvidia-smi`` loop writing ``<out>.csv``; :meth:`stop` ends it and writes the summary."""

    def __init__(self, out: Path, interval: float, watch: int | None) -> None:
        self.out, self.interval, self.watch = Path(out), max(1.0, float(interval)), watch
        self.proc = None
        self.stopped = threading.Event()
        self.gpu = gpu_id()
        self.fields: list[str] = []
        self.thread = None

    def _path(self, suffix: str) -> Path:
        return self.out.with_name(self.out.name + suffix)

    def start(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        if self.gpu is None:
            return
        self.fields = known_fields(self.gpu)
        self._path(".apps.txt").write_text(f"# at the start ({time.strftime('%F %T')})\n{_apps(self.gpu)}\n",
                                           encoding="utf-8")
        if not self.fields:
            return
        try:
            self.proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None
            return
        self.thread = threading.Thread(target=self._copy, daemon=True)
        self.thread.start()

    def command(self) -> list[str]:
        """The sampling loop: ``nvidia-smi`` printing :attr:`fields` of the job's GPU every :attr:`interval`
        seconds."""
        return ["nvidia-smi", f"--query-gpu={','.join(self.fields)}", "--format=csv,noheader,nounits", "-i", self.gpu,
                "-lms", str(int(self.interval * 1000))]

    def _copy(self) -> None:
        with open(self._path(".csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(self.fields) + "\n")
            fh.flush()
            for line in self.proc.stdout:
                fh.write(line)
                fh.flush()

    def wait(self) -> None:
        """Until a signal or the end of the watched process."""
        while not self.stopped.wait(self.interval):
            if self.watch is not None and not _alive(self.watch):
                break
            if self.proc is not None and self.proc.poll() is not None:
                self.proc = None           # nvidia-smi ended by itself: nothing more to sample
                if self.watch is None:
                    break

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.gpu is not None:
            with open(self._path(".apps.txt"), "a", encoding="utf-8") as fh:
                fh.write(f"# at the end ({time.strftime('%F %T')})\n{_apps(self.gpu)}\n")
        csv_path, summary = self._path(".csv"), None
        if csv_path.is_file():
            if self.thread is not None:
                self.thread.join(timeout=10)      # the copying thread's last line
            summary = summarise(_rows(csv_path.read_text(encoding="utf-8", errors="replace")), self.interval)
            with open(csv_path, "rb") as src, gzip.open(self._path(".csv.gz"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            csv_path.unlink()
        if summary is None:
            summary = {"samples": 0, "interval_s": self.interval, "throttled": None, "slowdown": {}, "reasons": {},
                       "why": "no nvidia-smi or no GPU" if self.gpu is None else "no field could be sampled"}
        summary.update(gpu=self.gpu, fields=self.fields, apps_at_start=_apps_at_start(self.out))
        self._path(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return summary


def _alive(pid: int) -> bool:
    """Whether process ``pid`` still runs (on Windows without ``os.kill``, which would end it)."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259              # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the files' common path without suffix (<out>.csv.gz, .json, "
                                                 ".apps.txt)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples (at least 1; default 1)")
    ap.add_argument("--watch", type=int, default=None, help="stop when this process has ended (the job's shell)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sampler = Sampler(Path(args.out), args.interval, args.watch)

        def stop(signum, frame):
            sampler.stopped.set()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        sampler.start()
        sampler.wait()
        sampler.stop()
    except Exception as e:                              # noqa: BLE001 - the sampler never fails a job
        print(f"telemetry: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
