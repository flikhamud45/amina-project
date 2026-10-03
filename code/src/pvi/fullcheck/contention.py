"""Evidence that a timed phase ran on a quiet machine (measurement only: nothing here changes what runs).

The L40S nodes do not confine a job to its cores (``cpu affinity: 0-127`` on 128-CPU nodes), so other
users' load can slow a phase that runs on the host CPU (the CPU verifier, the prover's encoding, a GPU
client's decoding, the commitment) several-fold.  Around such a phase, :func:`begin` takes a snapshot and
:func:`add` takes a second one and stores the difference under the phase's name:

* ``run_ns`` / ``wait_ns``: summed over every thread of this process, the nanoseconds spent on a CPU and
  waiting on a run queue (runnable but not running): the first and second fields of
  ``/proc/self/task/<tid>/schedstat``.  A thread that started during the phase counts from zero; one
  that ended during it is lost (``threads_exited``; ``cpu_s`` below still holds its CPU time).  Both
  are ``None`` where ``/proc/self/schedstat`` does not exist (Windows, macOS, a kernel without
  schedstats).  ``wait_frac = wait_ns / (run_ns + wait_ns)``: the share of the phase's runnable time
  that its threads waited for a CPU (0 on a quiet machine with enough cores).
* ``load_start`` / ``load_end``: ``/proc/loadavg`` at the phase's start and end, as
  ``[1 min, 5 min, 15 min, runnable now, tasks]`` (``None`` without it).
* ``nivcsw`` / ``nvcsw`` / ``cpu_s``: involuntary and voluntary context switches and user + system CPU
  seconds of the whole process, from ``getrusage`` (``None`` without the ``resource`` module): a fallback
  signal where schedstat is missing.
* ``wall_s`` (the time between the two snapshots, which bracket the timed window), ``threads`` (threads
  seen at the end) and ``segments`` (a phase timed in several parts, e.g. ``prove_encode``, sums its
  parts' deltas, keeps the first part's ``load_start`` and the last part's ``load_end``).

The snapshots are taken outside the timed windows, and every failure is swallowed: contention logging
can never change a timing's window, a verdict, a byte of a proof, or stop a run.
"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

try:
    import resource
except ImportError:          # Windows
    resource = None

PROC = "/proc"               # the procfs root (tests point it at a fake tree)
SUMMED = ("wall_s", "run_ns", "wait_ns", "nivcsw", "nvcsw", "cpu_s")


class Snapshot(NamedTuple):
    t: float                                       # time.perf_counter()
    threads: dict | None                           # {tid: (on-CPU ns, run-queue wait ns)}
    load: list | None                              # /proc/loadavg
    rusage: tuple | None                           # (involuntary csw, voluntary csw, user + system s)


def available() -> bool:
    """Does this kernel report per-task scheduler statistics (``/proc/self/schedstat``)?"""
    return os.path.exists(f"{PROC}/self/schedstat")


def _threads() -> dict | None:
    if not available():
        return None
    tasks = f"{PROC}/self/task"
    try:
        tids = os.listdir(tasks)
    except OSError:
        return None
    out = {}
    for tid in tids:
        try:
            with open(f"{tasks}/{tid}/schedstat", "rb") as fh:
                fields = fh.read().split()
            out[int(tid)] = (int(fields[0]), int(fields[1]))
        except (OSError, ValueError, IndexError):
            continue                               # the thread ended meanwhile
    return out


def _loadavg() -> list | None:
    try:
        with open(f"{PROC}/loadavg") as fh:
            a = fh.read().split()
        running, tasks = a[3].split("/")
        return [float(a[0]), float(a[1]), float(a[2]), int(running), int(tasks)]
    except (OSError, ValueError, IndexError):
        return None


def _rusage() -> tuple | None:
    if resource is None:
        return None
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_nivcsw), int(ru.ru_nvcsw), float(ru.ru_utime + ru.ru_stime)
    except (OSError, ValueError, AttributeError):
        return None


def snapshot() -> Snapshot:
    return Snapshot(time.perf_counter(), _threads(), _loadavg(), _rusage())


def begin() -> Snapshot | None:
    """The snapshot that opens a phase (``None`` if taking it failed: :func:`add` then records nothing)."""
    try:
        return snapshot()
    except Exception:            # pragma: no cover - measurement must never stop a run
        return None


def wait_fraction(d: dict) -> float | None:
    run, wait = d.get("run_ns"), d.get("wait_ns")
    if run is None or wait is None or run + wait <= 0:
        return None
    return wait / (run + wait)


def delta(a: Snapshot, b: Snapshot) -> dict:
    """What happened between two snapshots (see the module docstring for the fields)."""
    run = wait = n = exited = None
    if a.threads is not None and b.threads is not None:
        run = wait = 0
        for tid, (r1, w1) in b.threads.items():
            r0, w0 = a.threads.get(tid, (0, 0))    # a thread that started during the phase: from zero
            if r1 < r0 or w1 < w0:                 # a new thread under a recycled id
                r0 = w0 = 0
            run += r1 - r0
            wait += w1 - w0
        n, exited = len(b.threads), len(a.threads.keys() - b.threads.keys())
    ra, rb = a.rusage, b.rusage
    both = ra is not None and rb is not None
    d = {"wall_s": round(b.t - a.t, 6), "run_ns": run, "wait_ns": wait,
         "nivcsw": rb[0] - ra[0] if both else None, "nvcsw": rb[1] - ra[1] if both else None,
         "cpu_s": round(rb[2] - ra[2], 6) if both else None,
         "threads": n, "threads_exited": exited, "load_start": a.load, "load_end": b.load, "segments": 1}
    d["wait_frac"] = wait_fraction(d)
    return d


def add(store: dict, name: str, start: Snapshot | None) -> None:
    """Close the phase that :func:`begin` opened with ``start`` and add it to ``store[name]`` (a second
    part of the same phase is summed into the first)."""
    if start is None:
        return
    try:
        d = delta(start, snapshot())
        old = store.get(name)
        if old is None:
            store[name] = d
            return
        for k in SUMMED:
            old[k] = None if old.get(k) is None or d[k] is None else old[k] + d[k]
        old["wall_s"] = None if old["wall_s"] is None else round(old["wall_s"], 6)
        old["cpu_s"] = None if old["cpu_s"] is None else round(old["cpu_s"], 6)
        if d["threads"] is not None:
            old["threads"] = max(old.get("threads") or 0, d["threads"])
            old["threads_exited"] = (old.get("threads_exited") or 0) + d["threads_exited"]
        old["load_end"] = d["load_end"]
        old["segments"] = old.get("segments", 1) + 1
        old["wait_frac"] = wait_fraction(old)
    except Exception:            # pragma: no cover - measurement must never stop a run
        pass
