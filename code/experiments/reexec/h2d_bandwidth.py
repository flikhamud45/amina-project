"""Plan 2 F6 part 6 (EV's PCIe floor): the host-to-device and device-to-host copy bandwidth of this node.

    cd code && PYTHONPATH=src python experiments/6_improvements/h2d_bandwidth.py --sizes-mb 64 1024 4096 --reps 10

A GPU client receives the proof in host memory and uploads it before checking it, so its time has a floor of
bytes / bandwidth (EV frames the GPU verifier against that floor).  For each size, one buffer of that many bytes
(uint8) is copied host-to-device and device-to-host, from pinned and from pageable host memory, ``--reps`` times
after one untimed copy; each copy is timed with CUDA events around a synchronised ``copy_`` (``non_blocking`` for
pinned memory, which is how the streaming verifier uploads).  One JSON line per (size, direction, memory),
``"kind": "h2d"``: the median and ``[min, max]`` of the seconds and the median GB/s (10^9 bytes per second).  The GPU,
the driver's PCIe link (``nvidia-smi``, if present) and the host are recorded.  Without a GPU the script exits 2.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys

import torch


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes-mb", type=int, nargs="+", default=[64, 1024, 4096], help="buffer sizes in MiB")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    base = {"kind": "h2d", "label": args.label, "gpu": torch.cuda.get_device_name(0), "pcie": link(),
            "host": platform.node(), "torch": torch.__version__, "reps": args.reps}
    for mb in args.sizes_mb:
        n = mb << 20
        gpu = torch.empty(n, dtype=torch.uint8, device=dev)
        for memory in ("pinned", "pageable"):
            host = torch.empty(n, dtype=torch.uint8, pin_memory=memory == "pinned")
            host.fill_(1)
            for direction in ("h2d", "d2h"):
                dst, src = (gpu, host) if direction == "h2d" else (host, gpu)
                nb = memory == "pinned"
                timed_copy(dst, src, nb)                       # untimed
                times = [timed_copy(dst, src, nb) for _ in range(args.reps)]
                med = statistics.median(times)
                print(json.dumps({**base, "bytes": n, "direction": direction, "memory": memory, "s": med,
                                  "s_range": [min(times), max(times)], "gb_s": n / med / 1e9}), flush=True)
            del host
        del gpu
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
