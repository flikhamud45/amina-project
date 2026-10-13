"""The trace-tampering attacks of the timed decoder cells: each must be rejected."""

from __future__ import annotations

import torch

from pvi.v0.params import Z_BOUND
from pvi.v0.query import run_query

__all__ = ["KINDS", "decoder_tampers"]

KINDS = ("single_value", "output_logit", "last_hidden")


def decoder_tampers(prover, verifier, x: torch.Tensor, n: int, wire: bool, seed: int = 123) -> dict:
    """``n`` attempts of each attack on the query ``x``: ``single_value``, a random entry of a random weight op's
    claim changed by +-1..999; ``output_logit``, the smallest logit raised 1000 above the largest;
    ``last_hidden``, a random coordinate of the op feeding the LM head (the last block's down projection) at the
    last position set to ``Z_BOUND // 4``.  ``{kind: {attempts, rejected, labels}}``, ``labels`` counting the checks
    that rejected them."""
    mats = prover.graph.mat_ops
    g = torch.Generator().manual_seed(seed)
    out = {k: {"attempts": 0, "rejected": 0, "labels": {}} for k in KINDS}

    def attempt(kind, tamper):
        res = run_query(prover, verifier, x, wire=wire, forward_kwargs={"tamper": tamper})
        rec, label = out[kind], str(res["rejected_at"])
        rec["attempts"] += 1
        rec["rejected"] += int(not res["accepted"])
        rec["labels"][label] = rec["labels"].get(label, 0) + 1

    def t_logit(o, z):
        if o.name == mats[-1].name:
            z = z.clone()
            z[int(z[:, -1].argmin()), -1] = int(z[:, -1].max()) + 1000
        return z

    for i in range(n):
        op = mats[int(torch.randint(len(mats), (1,), generator=g))]
        pos = int(torch.randint(10 ** 9, (1,), generator=g))
        delta = int(torch.randint(1, 1000, (1,), generator=g)) * (1 if i % 2 else -1)
        j = int(torch.randint(mats[-2].n_rows, (1,), generator=g))

        def t_single(o, z, op=op, pos=pos, delta=delta):
            if o.name == op.name:
                z = z.clone()
                z.view(-1)[pos % z.numel()] += delta
            return z

        def t_last(o, z, j=j):
            if o.name == mats[-2].name:
                z = z.clone()
                z[j, -1] = Z_BOUND // 4
            return z

        attempt("single_value", t_single)
        attempt("output_logit", t_logit)
        attempt("last_hidden", t_last)
    return out
