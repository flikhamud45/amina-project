"""Pin a job to ``n`` physical cores of one NUMA node, one hardware thread on each.

The CPU-timing jobs of ``run.sh`` start with ``python -m experiments.package.pin --cores 8 --pid $$``: it sets the
CPU affinity of the job's shell (process ``--pid``), which every command the job runs afterwards inherits, to the
first hardware thread of ``n`` cores of one NUMA node that has ``n`` cores among the CPUs the job may use: the GPU's
node where the job sees one GPU and the kernel reports its node (so that the prover's host-to-device copies stay on
its socket), else the lowest-numbered such node.  On a two-socket EPYC 9334 in NPS4 (eight NUMA nodes of eight
cores) that is one whole node.  Memory is not bound: what does not fit in the node's memory is allocated on the
others.  Where no node has ``n`` such cores, or
the host has no CPU affinity (not Linux), the job runs unpinned and its log says so; every record states the CPUs it
ran on (``cpu_affinity_list`` and ``cpu_numa_nodes`` of :func:`experiments.common.records.environment`).

``--away <pid>...`` keeps processes that the job started before it pinned itself (the GPU telemetry sampler,
``experiments/package/telemetry.py``, and its ``nvidia-smi``) off the chosen cores: each of them, every thread of it
and every process it has started get the job's other CPUs, so that they never preempt the timed process.  Where the
job may use no other CPU they keep their affinity and the log says so.

    python -m experiments.package.pin --cores 8 --pid <pid> [--away <pid>...]

Only the standard library is imported, so the job's first command starts in a fraction of a second.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess

__all__ = ["parse_list", "compact", "numa_nodes", "nodes_of", "gpu_bus_ids", "pci_node", "choose", "descendants",
           "tasks", "move_away", "main"]

SYS = "/sys/devices/system"
"""The kernel's CPU and NUMA topology (a test passes another root)."""
PCI = "/sys/bus/pci/devices"
"""The kernel's PCI devices, each with its ``numa_node`` (a test passes another root)."""
PROC = "/proc"
"""The kernel's processes, each with its ``stat`` and its threads under ``task`` (a test passes another root)."""


def parse_list(text: str) -> list[int]:
    """The CPUs of a kernel CPU list (``"0-7,64-71"``)."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def compact(cpus) -> str:
    """A kernel CPU list of ``cpus`` (``[0, 1, 2, 3, 64]`` -> ``"0-3,64"``)."""
    cpus = sorted(set(cpus))
    runs: list[list[int]] = []
    for c in cpus:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def numa_nodes(root: str = SYS) -> dict[int, set[int]]:
    """``{node: its CPUs}`` (empty where the kernel reports no NUMA topology)."""
    out = {}
    for path in glob.glob(os.path.join(root, "node", "node*", "cpulist")):
        m = re.search(r"node(\d+)$", os.path.dirname(path))
        if m:
            with open(path) as fh:
                out[int(m.group(1))] = set(parse_list(fh.read()))
    return out


def nodes_of(cpus, root: str = SYS) -> list[int]:
    """The NUMA nodes that hold any of ``cpus``."""
    cpus = set(cpus)
    return sorted(n for n, members in numa_nodes(root).items() if members & cpus)


def gpu_bus_ids() -> list[str]:
    """The PCI bus ids of the GPUs this job sees (``nvidia-smi``; a SLURM job with constrained devices sees only its
    own), or none where there is no ``nvidia-smi``."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return out.split()


def pci_node(bus_id: str, root: str = PCI) -> int | None:
    """The NUMA node of the PCI device ``bus_id`` (``nvidia-smi``'s ``00000000:41:00.0``), or None where the kernel
    does not report one."""
    domain, _, rest = bus_id.strip().lower().partition(":")
    try:
        with open(os.path.join(root, f"{domain[-4:]}:{rest}", "numa_node")) as fh:
            node = int(fh.read())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def _core(cpu: int, root: str) -> tuple:
    """The physical core of ``cpu`` (its package and core id; the CPU itself where the kernel does not say)."""
    try:
        topo = os.path.join(root, "cpu", f"cpu{cpu}", "topology")
        with open(os.path.join(topo, "physical_package_id")) as a, open(os.path.join(topo, "core_id")) as b:
            return int(a.read()), int(b.read())
    except (OSError, ValueError):
        return ("cpu", cpu)


def choose(n: int, allowed, root: str = SYS, prefer: int | None = None) -> tuple[int, list[int]] | None:
    """``(node, cpus)``: the NUMA node ``prefer`` if it has ``n`` physical cores among ``allowed``, else the
    lowest-numbered node that has them, and the lowest hardware thread of each of its first ``n`` such cores; None if
    no node has them."""
    allowed = set(allowed)
    nodes = numa_nodes(root) or {0: allowed}
    for node in sorted(nodes, key=lambda m: (m != prefer, m)):
        first: dict[tuple, int] = {}
        for c in sorted(nodes[node] & allowed):
            first.setdefault(_core(c, root), c)
        if len(first) >= n:
            return node, sorted(first.values())[:n]
    return None


def descendants(pid: int, root: str = PROC) -> list[int]:
    """``pid`` and every process below it (from each process's parent in ``<root>/<pid>/stat``)."""
    children: dict[int, list[int]] = {}
    for path in glob.glob(os.path.join(root, "[0-9]*", "stat")):
        try:
            with open(path) as fh:
                fields = fh.read().rpartition(")")[2].split()      # after the command name, which may hold spaces
            children.setdefault(int(fields[1]), []).append(int(os.path.basename(os.path.dirname(path))))
        except (OSError, ValueError, IndexError):
            continue                                                 # a process that ended meanwhile
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        if p not in out:
            out.append(p)
            todo.extend(children.get(p, []))
    return out


def tasks(pid: int, root: str = PROC) -> list[int]:
    """The threads of process ``pid`` (``<root>/<pid>/task``; the process alone where the kernel lists none)."""
    found = [int(t) for t in os.listdir(os.path.join(root, str(pid), "task")) if t.isdigit()] \
        if os.path.isdir(os.path.join(root, str(pid), "task")) else []
    return sorted(found) or [pid]


def move_away(pids, cpus, root: str = PROC, setter=None) -> list[int]:
    """Give ``cpus`` to every thread of each of ``pids`` and of the processes below them; the processes moved.  Two
    passes, so that a thread or process started during the first is moved as well (a thread started after a pass
    inherits its creator's new affinity).  ``setter`` (default :func:`os.sched_setaffinity`) is a test's hook."""
    setter = setter or os.sched_setaffinity
    moved: list[int] = []
    for _ in range(2):
        for pid in pids:
            for p in descendants(pid, root):
                for t in tasks(p, root):
                    try:
                        setter(t, cpus)
                    except OSError:                                  # a thread or process that ended meanwhile
                        continue
                    if p not in moved:
                        moved.append(p)
    return moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cores", type=int, default=8, help="physical cores to pin to (default 8)")
    ap.add_argument("--pid", type=int, default=0, help="the process to pin (default this one); its children "
                                                       "started afterwards inherit the affinity")
    ap.add_argument("--away", type=int, nargs="+", default=[], metavar="PID",
                    help="processes started before the pinning (the GPU sampler) to keep off the chosen cores: "
                         "they, their threads and their children get the job's other CPUs")
    args = ap.parse_args(argv)
    if not hasattr(os, "sched_setaffinity"):
        print("not pinned: this host sets no CPU affinity", flush=True)
        return 0
    allowed = os.sched_getaffinity(args.pid)
    gpus = gpu_bus_ids()
    gpu = pci_node(gpus[0]) if len(gpus) == 1 else None
    got = choose(args.cores, allowed, prefer=gpu)
    if got is None:
        print(f"not pinned: no NUMA node has {args.cores} cores among this job's CPUs {compact(allowed)}", flush=True)
        return 0
    node, cpus = got
    try:
        os.sched_setaffinity(args.pid, cpus)
    except OSError as exc:
        print(f"not pinned: {exc}", flush=True)
        return 0
    where = ("the GPU's node" if node == gpu else "the GPU's node is not known" if gpu is None
             else f"the GPU is on node {gpu}")
    print(f"pinned to CPUs {compact(cpus)} ({args.cores} cores of NUMA node {node}; {where})", flush=True)
    if args.away:
        rest = sorted(set(allowed) - set(cpus))
        if not rest:
            print(f"processes {' '.join(map(str, args.away))} share the pinned CPUs: the job may use no other",
                  flush=True)
        else:
            moved = move_away(args.away, rest)
            print(f"processes {' '.join(map(str, moved)) or '(none left)'} moved to CPUs {compact(rest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
