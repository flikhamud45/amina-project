"""The GPU's energy counter: joules spent by one GPU between two reads, for the energy per served or proved query.

NVML keeps a counter of the energy a GPU has consumed since the driver was loaded (``nvmlDeviceGetTotalEnergyConsumption``,
millijoules; Volta and later).  :class:`EnergyMeter` reads it through ``ctypes`` from the driver's own library
(``libnvidia-ml.so.1``, ``nvml.dll``), so no Python package is needed, for the GPU that torch numbers 0 under
``CUDA_VISIBLE_DEVICES`` (found by its UUID).  A read takes microseconds and starts no CUDA work, so it can sit at the
edges of a timed window without changing it.  Where there is no NVIDIA driver, no such GPU or no counter (a CPU run,
an older GPU), :meth:`EnergyMeter.read` returns ``None`` and every energy field of a record is ``None``; nothing
fails.

The counter is updated by the driver at its own rate (tens of milliseconds on recent GPUs), so a window much shorter
than that has a coarse energy; :func:`joules_per_call` therefore times back-to-back calls for at least a given number
of seconds.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import time

__all__ = ["EnergyMeter", "joules_per_call"]

_NVML_SUCCESS = 0


def _library():
    """The NVML library of the driver, or None."""
    names = ["nvml.dll"] if os.name == "nt" else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    found = ctypes.util.find_library("nvidia-ml")
    if found:
        names.append(found)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class EnergyMeter:
    """The energy counter of one GPU (default: CUDA device 0 as torch numbers it, by its UUID; ``uuid``: that GPU,
    ``GPU-...``).  ``available`` says whether reads return values; :meth:`read` gives joules since the driver was
    loaded (float), or ``None``; ``why`` says why not."""

    def __init__(self, uuid: str | None = None, lib=None) -> None:
        self.available, self.why, self._lib, self._handle = False, None, None, None
        try:
            self._open(uuid, lib)
        except Exception as e:                       # noqa: BLE001 - no counter is a record without energy
            self.why = f"{type(e).__name__}: {e}"
            self.available = False

    def _open(self, uuid: str | None, lib) -> None:
        lib = lib if lib is not None else _library()
        if lib is None:
            self.why = "no NVML library (no NVIDIA driver)"
            return
        if lib.nvmlInit_v2() != _NVML_SUCCESS:
            self.why = "nvmlInit failed"
            return
        handle = ctypes.c_void_p()
        uuid = uuid or _torch_uuid()
        if uuid is not None:
            st = lib.nvmlDeviceGetHandleByUUID(ctypes.c_char_p(uuid.encode()), ctypes.byref(handle))
        elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
            st = lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(handle))
        else:
            self.why = "the GPU's UUID is unknown and CUDA_VISIBLE_DEVICES renumbers the GPUs"
            return
        if st != _NVML_SUCCESS:
            self.why = f"no NVML handle for the GPU (status {st})"
            return
        self._lib, self._handle = lib, handle
        if self._raw() is None:
            self.why = "the GPU has no energy counter"
            return
        self.available = True

    def _raw(self) -> int | None:
        mj = ctypes.c_ulonglong()
        if self._lib.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(mj)) != _NVML_SUCCESS:
            return None
        return int(mj.value)

    def read(self) -> float | None:
        """Joules since the driver was loaded, or ``None`` without a counter."""
        if not self.available:
            return None
        mj = self._raw()
        return None if mj is None else mj / 1e3


def _torch_uuid() -> str | None:
    """CUDA device 0's UUID as NVML names it (``GPU-<uuid>``), if torch has CUDA and reports it."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        return f"GPU-{uuid}" if uuid else None
    except Exception:                                # noqa: BLE001
        return None


def joules_per_call(meter: EnergyMeter | None, fn, sync, min_s: float, max_calls: int = 1000) -> dict | None:
    """``{"joules", "calls", "seconds"}``: ``fn`` called back to back, after ``sync()``, until at least ``min_s``
    seconds have passed (at least once, at most ``max_calls`` times), and the counter's joules per call; ``None``
    without a counter or with ``min_s <= 0``."""
    if meter is None or not meter.available or min_s <= 0:
        return None
    sync()
    e0, t0, calls = meter.read(), time.perf_counter(), 0
    while True:
        fn()
        calls += 1
        sync()
        dt = time.perf_counter() - t0
        if dt >= min_s or calls >= max_calls:
            break
    e1 = meter.read()
    if e0 is None or e1 is None:
        return None
    return {"joules": (e1 - e0) / calls, "calls": calls, "seconds": dt / calls}
