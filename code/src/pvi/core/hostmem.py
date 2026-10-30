"""Pinned host memory, within what the driver grants.

On the NVIDIA L40S machines of the measurements (drivers 535 and 580, Linux kernel 6.11)
``cudaHostAlloc`` refuses any single allocation of more than 2 GiB with ``invalid argument``
(``cudaHostRegister`` refuses even 1 GiB), whatever ``RLIMIT_MEMLOCK`` is; many smaller pinned
allocations, 48 GiB in all, are granted.  torch's pinned
allocator rounds a request up to a power of two, so every request of at most ``PIN_MAX`` bytes is
granted.  A larger buffer is taken from pageable memory instead: the same values, and a copy from or to
it is still correct, but it waits for the device (a pageable host-to-device copy also returns only once
the host buffer has been read).  Every pinned allocation of the defence goes through these two
functions.
"""

from __future__ import annotations

import math

import torch

PIN_MAX = 1 << 31           # bytes: the largest single pinned allocation (cudaHostAlloc refuses more)


def _nbytes(shape, dtype: torch.dtype) -> int:
    dims = (shape,) if isinstance(shape, int) else tuple(shape)
    return math.prod(dims) * torch.empty((), dtype=dtype).element_size()


def empty(shape, dtype: torch.dtype, pin: bool = True) -> torch.Tensor:
    """``torch.empty(shape, dtype=dtype)`` on the host, pinned with ``pin`` unless it exceeds
    ``PIN_MAX`` bytes."""
    return torch.empty(shape, dtype=dtype, pin_memory=pin and _nbytes(shape, dtype) <= PIN_MAX)


def pinned(t: torch.Tensor) -> torch.Tensor:
    """``t.pin_memory()``, or ``t`` itself above ``PIN_MAX`` bytes."""
    return t.pin_memory() if t.numel() * t.element_size() <= PIN_MAX else t


SLAB = 1 << 30              # bytes of a :class:`Slabs` slab (a power of two: no rounding up)
_ALIGN = 256


class Slabs:
    """Host buffers carved one after another out of slabs of ``slab`` bytes (pinned with ``pin``), so that many
    buffers whose sizes are not powers of two take about their own size of pinned memory instead of torch's
    rounding up of each to a power of two.  A buffer larger than a slab is a buffer of its own (:func:`empty`).
    If the driver refuses a pinned slab, it and every later one are pageable (the same values; copies to and
    from them wait).  The slabs are freed with the buffers that view them."""

    def __init__(self, slab: int = SLAB, pin: bool = True) -> None:
        self.slab_bytes, self.pin = slab, pin
        self.slab, self.at = None, 0

    def empty(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """An uninitialised host tensor of ``shape`` and ``dtype``."""
        n = _nbytes(shape, dtype)
        if n > self.slab_bytes:
            return empty(shape, dtype, pin=self.pin)
        if self.slab is None or self.at + n > self.slab_bytes:
            try:
                self.slab = torch.empty(self.slab_bytes, dtype=torch.uint8, pin_memory=self.pin)
            except RuntimeError:            # the driver granted no more pinned memory
                self.pin = False
                self.slab = torch.empty(self.slab_bytes, dtype=torch.uint8)
            self.at = 0
        out = self.slab[self.at:self.at + n].view(dtype).view(shape)
        self.at += -(-n // _ALIGN) * _ALIGN
        return out
