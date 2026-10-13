"""The host-to-device and device-to-host copy bandwidth of the node: the floor of a GPU client's time.

A GPU client receives the proof in host memory and uploads it before checking it, so its time has a floor of bytes
over bandwidth, against which the GPU verifier's time is read; with the proof sizes it gives the extended version's
transfer times (``ext_discussion.tex``, the client's view of a 2,048-token proof) and the number pipeline's
``\\HtoDPinnedGBs`` (``paper/numbers.py``, the ``gpuv`` tier's pinned host-to-device bandwidth).  The run package
measures it in its ``smoke`` and ``gpuv`` tiers.

    python -m experiments.reexec.h2d_bandwidth --sizes-mb 64 1024 4096 --reps 10 --label gpuv

For each size, one buffer of that many MiB (uint8) is copied host-to-device and device-to-host (``--directions``),
from pinned and from pageable host memory (``--memory``), ``--reps`` times after one untimed copy; each copy is timed
with CUDA events around a synchronised ``copy_`` (``non_blocking`` for pinned memory, which is how the streaming
verifier uploads).

**Record** (one JSON line per size, direction and memory, on stdout; ``"kind": "h2d"``): ``label``, ``gpu``, ``pcie``
(the driver's current PCIe generation and width, from ``nvidia-smi`` if present), ``host``, ``torch``, ``reps``,
``bytes``, ``direction``, ``memory``, ``s`` (the median seconds), ``s_range`` (``[min, max]``), ``gb_s`` (the median's
GB/s, 10^9 bytes per second), ``env`` (:func:`experiments.common.records.environment`).

Hardware and time: a GPU (without one the script prints ``no GPU`` and exits with status 2); seconds, and pinned host
memory of the largest size.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys

import torch

from experiments.common import records


def link() -> str | None:
    """``nvidia-smi``'s current PCIe generation and width of GPU 0, or ``None``."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                              "--format=csv,noheader", "-i", "0"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def timed_copy(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool) -> float:
    """Seconds of one ``dst.copy_(src)``, CUDA events after a device synchronisation."""
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    dst.copy_(src, non_blocking=non_blocking)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 1e3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes-mb", type=int, nargs="+", default=[64, 1024, 4096], help="buffer sizes in MiB")
    ap.add_argument("--reps", type=int, default=10, help="timed copies per cell, after one untimed copy")
    ap.add_argument("--memory", nargs="+", choices=["pinned", "pageable"], default=["pinned", "pageable"],
                    help="the host buffer's memory")
    ap.add_argument("--directions", nargs="+", choices=["h2d", "d2h"], default=["h2d", "d2h"],
                    help="host to device, device to host")
    ap.add_argument("--label", default="", help="a label copied into every record")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    base = {"kind": "h2d", "label": args.label, "gpu": torch.cuda.get_device_name(0), "pcie": link(),
            "host": platform.node(), "torch": torch.__version__, "reps": args.reps, "env": records.environment(dev)}
    for mb in args.sizes_mb:
        n = mb << 20
        gpu = torch.empty(n, dtype=torch.uint8, device=dev)
        for memory in args.memory:
            host = torch.empty(n, dtype=torch.uint8, pin_memory=memory == "pinned")
            host.fill_(1)
            for direction in args.directions:
                dst, src = (gpu, host) if direction == "h2d" else (host, gpu)
                nb = memory == "pinned"
                timed_copy(dst, src, nb)                       # untimed
                times = [timed_copy(dst, src, nb) for _ in range(args.reps)]
                med = statistics.median(times)
                records.emit({**base, "bytes": n, "direction": direction, "memory": memory, "s": med,
                              "s_range": [min(times), max(times)], "gb_s": n / med / 1e9})
            del host
        del gpu
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
