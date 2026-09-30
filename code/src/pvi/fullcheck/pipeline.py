"""How the proof reaches a GPU client: the wire formats and claim uploads of the streaming
verifier (``Verifier(stream=True)``, whose checks are :meth:`protocol.Verifier.verify_streaming`).

* The claims travel as int32 (:func:`wire_claim`: the 4 bytes per value the proof size
  already counts), narrowed on the prover's device and received into pinned host memory.
  The verifier uploads them one weight op at a time on a side CUDA stream, ``AHEAD`` ops
  before derive needs them (:class:`ClaimUploads`), so the upload overlaps derive.
* In mode C the opened columns travel as int32 rows ``[t, N]`` (:func:`wire_openings`), the
  layout their Merkle leaves hash, so the verifier casts nothing before uploading them.

With ``run_query(wire=True)`` the proof travels in the compact encoding of
:mod:`pvi.fullcheck.claimcodec` instead, and the verifier decodes it into these formats (the
claims into pinned memory on a GPU client) before the streaming checks start.
"""

from __future__ import annotations

import torch

from .field import min_max

__all__ = ["AHEAD", "ClaimUploads", "wire_claim", "wire_openings"]

AHEAD = 16
"""Weight ops whose claims are uploaded ahead of derive (about two transformer blocks)."""
_INT32 = (-(1 << 31), (1 << 31) - 1)


def _fits_int32(t: torch.Tensor) -> bool:
    if t.numel() == 0:
        return True
    lo, hi = min_max(t)
    return lo >= _INT32[0] and hi <= _INT32[1]


def wire_claim(z, pin: bool = False) -> torch.Tensor | None:
    """A claim as the streaming verifier receives it: int32 on the host (in pinned memory with
    ``pin``, ready for a non-blocking upload), narrowed where it was computed.  A claim that is
    not an int64 tensor of int32 values cannot be narrowed: it is passed on as an int64 host
    tensor of the same values, which the streaming verifier rejects at ``range_or_shape`` (as
    ``run_query`` rejects the claim itself) and which has the claim's size and transcript blob.
    Anything but a tensor gives ``None``."""
    if not torch.is_tensor(z):
        return None
    if z.dtype != torch.int64 or not _fits_int32(z):
        return z.to("cpu", torch.int64)
    z32 = z.to(torch.int32)
    if not (pin and torch.cuda.is_available()):
        return z32.cpu()
    return torch.empty(z.shape, dtype=torch.int32, pin_memory=True).copy_(z32)


def wire_openings(openings: dict) -> dict:
    """``{name: (rows, proof)}``: each opening's int64 columns ``[N, t]`` as the int32 rows ``[t, N]``
    the Merkle leaves hash, or ``None`` when they are not a 2-d int64 tensor of int32 values
    (rejected at ``columns_shape``, as ``run_query`` rejects them); anything that is not a pair
    (a tuple or list of two, which ``run_query`` unpacks alike) is passed on unchanged (and
    raises where ``run_query`` raises)."""
    out = {}
    for name, item in openings.items():
        if isinstance(item, (tuple, list)) and len(item) == 2:
            o, proof = item
            ok = torch.is_tensor(o) and o.dtype == torch.int64 and o.dim() == 2 and _fits_int32(o)
            item = (o.to(torch.int32).T.contiguous() if ok else None, proof)
        out[name] = item
    return out


class ClaimUploads:
    """``get(name)`` of the claims on ``device``, asked for in weight-op order and uploaded
    ``AHEAD`` ops earlier: on a side CUDA stream (the compute stream waits for each copy), or
    directly off the GPU."""

    def __init__(self, claims: dict, names: list[str], device: torch.device) -> None:
        self.claims, self.names, self.device = claims, names, device
        self.index = {n: i for i, n in enumerate(names)}
        self.side = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.ready: dict[str, tuple] = {}
        self.issued = 0

    def _issue(self, upto: int) -> None:
        while self.issued < min(upto, len(self.names)):
            name = self.names[self.issued]
            self.issued += 1
            z = self.claims.get(name)
            if not torch.is_tensor(z):                 # the verifier rejects it
                self.ready[name] = (z, None)
            elif self.side is None:
                self.ready[name] = (z.to(self.device), None)
            else:
                with torch.cuda.stream(self.side):
                    zd = z.to(self.device, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(self.side)
                self.ready[name] = (zd, done)

    def get(self, name: str):
        self._issue(self.index[name] + 1 + AHEAD)
        z, done = self.ready.pop(name)
        if done is not None:
            main = torch.cuda.current_stream(self.device)
            main.wait_event(done)
            z.record_stream(main)     # allocated on the side stream, read on this one
        return z
