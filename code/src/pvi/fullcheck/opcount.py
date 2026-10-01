"""Counting the work a computation launches, for the tests' bounds and the benchmarks' reports.

:class:`KernelOps` counts the aten ops that launch work on a device (views and ``detach`` launch
none), on any device: on CPU tensors the same ops run as on a GPU, so a count there is the count of
a GPU's kernel launches up to ops that one aten op splits into several kernels.  :func:`codec_waits`
counts the device encoder's waits for its device (``claimcodec._d2h`` and ``claimcodec._nonzero_dev``,
its only host round trips).  Neither changes what it counts.
"""

from __future__ import annotations

import contextlib

from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["VIEWS", "KernelOps", "codec_waits"]

VIEWS = ("view", "slice", "detach", "lift_fresh", "alias", "_reshape_alias", "select", "unsqueeze", "expand",
         "as_strided", "t.default", "transpose", "squeeze", "unbind", "split", "reshape")
"""Prefixes (after ``aten.``) of the aten ops that launch no work."""


class KernelOps(TorchDispatchMode):
    """In its body, ``n``: the aten ops dispatched that launch work on a device (all but :data:`VIEWS`)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not any(str(func).startswith("aten." + v) for v in VIEWS):
            self.n += 1
        return func(*args, **(kwargs or {}))


@contextlib.contextmanager
def codec_waits(codec=None):
    """In its body, the calls of ``codec._d2h`` and ``codec._nonzero_dev`` (``codec``: a ``claimcodec``
    module, default this package's), as ``{name: count}``; the functions are restored after it."""
    if codec is None:
        from . import claimcodec as codec
    waits = {"_d2h": 0, "_nonzero_dev": 0}
    real = {name: getattr(codec, name) for name in waits}

    def counted(name):
        def call(t):
            waits[name] += 1
            return real[name](t)
        return call

    for name in waits:
        setattr(codec, name, counted(name))
    try:
        yield waits
    finally:
        for name, fn in real.items():
            setattr(codec, name, fn)
