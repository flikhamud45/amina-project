"""The node a job runs on, and the check that it is a node of the run's platform.

Every job of ``run.sh`` (and a ``--local`` run, a ``--one-job`` allocation among them, before its first job) starts
with

    python -m experiments.package.node --require-gpu NAME --require-cpu NAME

which prints one line ``node: {...}`` (a JSON object: the host, the GPU the job's processes use, the number of GPUs
they see, ``CUDA_VISIBLE_DEVICES``, the GPU's UUID and PCI bus id, the driver, the CPU model and the SLURM job) and
exits with status :data:`WRONG_NODE` and a line ``wrong node: GPU <x> / CPU <y>, expected ...`` when the GPU's name
does not contain ``--require-gpu`` or the CPU's model name does not contain ``--require-cpu`` (letters compared
without case; ``any`` requires nothing).  The GPU is CUDA device 0 as torch numbers the devices under
``CUDA_VISIBLE_DEVICES``, the device every script of the package uses, so the check names the GPU SLURM allocated to
the job and not another GPU of the node; a job that sees no CUDA device has GPU ``none``.  The CPU model is the
``model name`` of ``/proc/cpuinfo`` (``lscpu``'s ``Model name``), the name the records keep
(:func:`experiments.common.records.environment`).  ``experiments/paper/records.py`` reads the line from every job's
log (:func:`experiments.paper.records.nodes`), and ``experiments/paper/numbers.py --check`` fails when the run's jobs
name more than one GPU model or CPU model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess

__all__ = ["WRONG_NODE", "describe", "matches", "main"]

WRONG_NODE = 86
"""The exit status of a job (and of a ``--local`` run) that started on a node of another GPU or CPU model."""


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


def _smi() -> list[dict]:
    """``nvidia-smi``'s GPUs (index, UUID, name, driver, PCI bus id); none where there is no ``nvidia-smi``."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keys = ("index", "uuid", "name", "driver", "bus_id")
    return [dict(zip(keys, (p.strip() for p in line.split(",")))) for line in out.splitlines() if line.count(",") >= 4]


def _torch_gpu() -> tuple[str | None, int, str | None]:
    """``(name, count, uuid)`` of CUDA device 0 as the job's processes see it (``CUDA_VISIBLE_DEVICES``)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, 0, None
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        return torch.cuda.get_device_name(0), torch.cuda.device_count(), (f"GPU-{uuid}" if uuid else None)
    except Exception:                                   # noqa: BLE001 - a broken CUDA set-up is a GPU of none
        return None, 0, None


def describe() -> dict:
    """The node as the job sees it: ``host``, ``gpu`` (CUDA device 0's name, None without one), ``gpu_count``,
    ``cuda_visible_devices``, ``gpu_uuid`` and ``gpu_bus_id`` (from ``nvidia-smi``, where it lists that GPU),
    ``driver``, ``cpu`` (the CPU model), ``slurm_job`` and ``slurm_nodelist``."""
    name, count, uuid = _torch_gpu()
    smi = _smi()
    row = next((r for r in smi if uuid and r["uuid"] == uuid), None)
    return {"host": socket.gethostname(), "gpu": name, "gpu_count": count,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_uuid": uuid,
            "gpu_bus_id": row["bus_id"] if row else None,
            "driver": (row or (smi[0] if smi else {})).get("driver"), "cpu": _cpu_model(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST")}


def matches(value: str | None, required: str | None) -> bool:
    """Whether ``value`` contains ``required`` (letters compared without case); ``any`` or nothing matches all."""
    if not required or required.strip().lower() == "any":
        return True
    return value is not None and required.strip().lower() in value.lower()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--require-gpu", default="any", help="a part of the GPU's name, e.g. L40S ('any': no check)")
    ap.add_argument("--require-cpu", default="any", help="a part of the CPU's model name, e.g. 'EPYC 9334' ('any': "
                                                         "no check)")
    args = ap.parse_args(argv)
    node = describe()
    print("node: " + json.dumps(node), flush=True)
    if matches(node["gpu"], args.require_gpu) and matches(node["cpu"], args.require_cpu):
        return 0
    print(f"wrong node: GPU {node['gpu'] or 'none'} / CPU {node['cpu'] or 'unknown'}, expected GPU "
          f"'{args.require_gpu}' / CPU '{args.require_cpu}' (host {node['host']}; --require-gpu, --require-cpu)",
          flush=True)
    return WRONG_NODE


if __name__ == "__main__":
    raise SystemExit(main())
