"""Sampled generation with a client-chosen seed (plan E2): an exact integer Gumbel-max sampler.

Greedy decoding picks the largest logit; sampling at temperature ``tau`` picks ``argmax_v (l_v / tau + g_v)`` with
independent Gumbel noise ``g_v`` (the Gumbel-max trick).  Here everything is an integer that prover and verifier
compute identically from public data and the client's seed:

* the claimed logits ``z`` (int32 claims of the LM head, real logits ``z c`` for the builder's logit scale ``c``);
* ``a``, an integer weight: ``a = round(c 2^16 / tau)`` (``TokenSampler.for_temperature``);
* the noise ``G[v] = TABLE[u_v]``, where ``u_v`` are 16-bit words of ``SHAKE-256(b"pvi/sampler/v1" || seed ||
  position)`` and ``TABLE[i] = round(2^16 (-ln(-ln((i + 1/2) / 2^16))))``, a public table (its SHA-256 is fixed below;
  it is computed with floats and checked against that hash, and recomputed exactly with ``decimal`` if a
  platform's ``log`` ever disagrees);
* the token at a position: among the ``top_k`` largest ``z`` (all if 0; ties to the smaller index), the first
  ``v`` maximising ``a z_v + G[v]`` (int64: ``|z| < 2^29``, ``a < 2^24``, ``|G| < 2^20``).

The seed must come from the client, fresh for each query, and be bound into the transcript (``run_query``'s
``sampler``): a prover that could choose the seed could choose among outputs.  The verifier recomputes each
generated token from the claimed logits before them, which the protocol checks like any other claim, so a
sampled response is proved exactly as a greedy one is (:meth:`protocol.Verifier.check_tokens`).
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

__all__ = ["TokenSampler", "gumbel_table", "sample_tokens"]

FRAC = 16
_TABLE_SHA256 = "fc2d8ed7fe9b4cbc3b95bcc8c5a35b0f8f7c8897e9d9e4f3e42501a83d059cc8"


@lru_cache(maxsize=1)
def gumbel_table() -> torch.Tensor:
    """``TABLE`` (int64 ``[2^16]``), checked against its published hash."""
    tab = np.array([round(-math.log(-math.log((i + 0.5) / 65536)) * 65536) for i in range(65536)], dtype="<i4")
    if hashlib.sha256(tab.tobytes()).hexdigest() != _TABLE_SHA256:      # pragma: no cover - an unusual libm
        import decimal
        decimal.getcontext().prec = 30
        d, two16 = decimal.Decimal, decimal.Decimal(65536)
        tab = np.array([int((-((-(((d(i) + d("0.5")) / two16).ln())).ln()) * two16)
                            .to_integral_value(rounding=decimal.ROUND_HALF_EVEN)) for i in range(65536)], dtype="<i4")
        if hashlib.sha256(tab.tobytes()).hexdigest() != _TABLE_SHA256:
            raise RuntimeError("the Gumbel table does not match its published hash")
    return torch.from_numpy(tab.astype(np.int64))


@dataclass(frozen=True)
class TokenSampler:
    """A sampling rule: the client's ``seed`` (32 bytes, fresh per query), the integer logit weight ``a`` and
    ``top_k`` (0: no restriction)."""
    seed: bytes
    a: int
    top_k: int = 0

    def __post_init__(self) -> None:
        if len(self.seed) != 32 or not 0 < self.a < 1 << 24 or self.top_k < 0:
            raise ValueError("a sampler needs a 32-byte seed, 0 < a < 2^24 and top_k >= 0")

    @classmethod
    def for_temperature(cls, seed: bytes, temperature: float, logit_scale: float, top_k: int = 0) -> "TokenSampler":
        """The rule for real temperature ``temperature`` and the graph's logit scale (real logit = ``z c``)."""
        return cls(seed, max(1, min((1 << 24) - 1, round(logit_scale * (1 << FRAC) / temperature))), top_k)

    def digest(self) -> bytes:
        """What the transcript absorbs (``b"sampler"``)."""
        return b"pvi/sampler/v1" + struct.pack("<II", self.a, self.top_k) + self.seed

    def noise(self, position: int, vocab: int) -> torch.Tensor:
        """``G`` for one position (int64 ``[vocab]``)."""
        words = hashlib.shake_256(b"pvi/sampler/v1" + self.seed + struct.pack("<Q", position)).digest(2 * vocab)
        return gumbel_table()[torch.from_numpy(np.frombuffer(words, dtype="<u2").astype(np.int64))]

    def choose(self, z: torch.Tensor, position: int) -> int:
        """The token at ``position`` from that position's claimed logits ``z`` ``[vocab]``."""
        z = z.to("cpu", torch.int64)
        score = z * self.a + self.noise(position, z.numel())
        if self.top_k and self.top_k < z.numel():
            keep = torch.sort(-z, stable=True).indices[:self.top_k]          # largest z, ties to the smaller index
            mask = torch.ones_like(z, dtype=torch.bool)
            mask[keep] = False
            score = score.masked_fill(mask, torch.iinfo(torch.int64).min)
        return int(torch.argmax(score))                                    # the first maximum


def sample_tokens(graph, prompt: torch.Tensor, steps: int, sampler: TokenSampler) -> torch.Tensor:
    """``prompt [1, P]`` followed by ``steps`` tokens sampled with ``sampler`` from ``graph``'s integer logits (one
    forward pass per token, as ``transformer.greedy_tokens``); the token appended as position ``P + i`` is chosen
    from the logits at position ``P + i - 1``, with noise index ``P + i - 1``."""
    x = prompt
    for _ in range(steps):
        env, _ = graph.forward(x, free=True)
        pos = x.shape[1] - 1
        nxt = sampler.choose(env[graph.output_name][0, -1], pos)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=x.dtype, device=x.device)], 1)
    return x
