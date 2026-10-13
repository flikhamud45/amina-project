"""The prover: the claims of a query (every weight op's output), the folded rows and the opened columns."""

from __future__ import annotations

import torch

from pvi.commit.commitment import WeightCommitment
from pvi.core.graph import IntGraph


class Prover:
    def __init__(self, graph: IntGraph, *, device="cpu", commitments: dict[str, WeightCommitment] | None = None,
                 lean: bool = False):
        self.graph = graph
        self.device = torch.device(device)
        self.commitments = commitments or {}
        self.lean = lean  # free dead activations and stream each claim to the host during the forward pass
        self._ops = {op.name: op for op in graph.mat_ops}
        self._looked_up: dict[str, list[int]] = {}    # a plan's lookup tables: the ids of the last query

    def claims(self, x: torch.Tensor, *, send=None, to_host: bool = True, **forward_kwargs) -> dict[str, torch.Tensor]:
        """The claims of query ``x``, on the host (``to_host=False``: where the prover keeps them, its
        device, or the host in lean mode).  ``send(z)`` (see ``IntGraph.forward``), if given, is
        applied to each claim on the prover's device as it is computed, and its results are returned
        as they are.  The ids each lookup table of a commitment plan looks up are kept for
        :meth:`open_tables`."""
        if self.lean:
            forward_kwargs = dict(forward_kwargs, free=True, claims_device="cpu")
        tables = getattr(self.commitments, "tables", {})
        self._looked_up = {}
        if tables:
            def watch(op, xin):
                if op.name in tables:
                    self._looked_up[op.name] = xin.reshape(-1).tolist()
            forward_kwargs = dict(forward_kwargs, watch=watch)
        _, claims = self.graph.forward(x.to(self.device), send=send, **forward_kwargs)
        return claims if send is not None or not to_host else {k: v.to("cpu") for k, v in claims.items()}

    def _weight(self, name: str) -> torch.Tensor:
        # the int8 weights the forward pass already keeps on the device
        return self._ops[name]._weights_on(self.device)[0]

    def open_tables(self) -> dict[str, list[bytes]]:
        """The multiproof of every lookup table of a commitment plan for the rows the last
        :meth:`claims` looked up (sent with the claims)."""
        return {name: t.open(self._looked_up[name]) for name, t in getattr(self.commitments, "tables", {}).items()}

    def fold(self, chis: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.commitments[name].fold(chi, self.device, weight=self._weight(name))
                for name, chi in chis.items()}

    def open(self, cols: dict[str, torch.Tensor]):
        """The opened columns and multiproof of every tree: each weight op's, or under a commitment
        plan each group's (its members' columns stacked in member order), computed from the weights
        on the prover's device."""
        groups = getattr(self.commitments, "groups", None)
        if groups:
            plan = self.commitments.plan
            return {g: groups[g].open(idx, self.device, weights=[self._weights_of(plan.matrix(m))
                                                                 for m in groups[g].members])
                    for g, idx in cols.items()}
        return {name: self.commitments[name].open(idx, self.device, weight=self._weight(name))
                for name, idx in cols.items()}

    def _weights_of(self, matrix):
        """The device weights of a committed matrix: its op's, or each of a col-layout matrix's ops'."""
        return [self._weight(o) for o in matrix.members] if matrix.layout == "col" else self._weight(matrix.name)
