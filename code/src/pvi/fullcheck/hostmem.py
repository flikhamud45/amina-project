"""Pinned host memory, within what the driver grants.

On the L40S nodes (driver 535 and 580, kernel 6.11) ``cudaHostAlloc`` refuses any single allocation of
more than 2 GiB with ``invalid argument`` (``cudaHostRegister`` refuses even 1 GiB), whatever
``RLIMIT_MEMLOCK`` is; many smaller pinned allocations, 48 GiB in all, are granted.  torch's pinned
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
