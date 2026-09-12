"""Wire messages of the three-round protocol of Definition 4.

Round 1 (prover): the claimed output and a commitment to the trace.
Round 2 (verifier): a random challenge ``rho``.
Round 3 (prover): the openings the challenge asks for.

Keeping the messages as explicit data classes rather than passing objects around
makes it structurally impossible for the verifier to read something the prover did
not actually send -- which, in a study whose whole point is what a verifier can
and cannot see, is worth the small amount of ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pvi.commitments.merkle import MerkleOpening, MerkleParams
from pvi.protocol.sampling import Path

__all__ = ["CheckFailure", "Round1", "Round2", "VerificationResult"]


@dataclass(frozen=True)
class Round1:
    """``pi_1`` -- the prover's first message."""

    claimed_output: np.ndarray
    trace_digest: bytes
    trace_params: MerkleParams

    @property
    def size_bytes(self) -> int:
        return self.claimed_output.nbytes + len(self.trace_digest)


@dataclass(frozen=True)
class Round2:
    """``pi_2`` -- the openings requested by the challenge.

    ``weight_openings`` is keyed by ``(layer index, weight group)``.  Several path
    steps can land on the same weight row; it is sent once.
    """

    layer_openings: tuple[MerkleOpening, ...]
    weight_openings: dict[tuple[int, int], MerkleOpening]

    @property
    def size_bytes(self) -> int:
        return sum(o.size_bytes for o in self.layer_openings) + sum(
            o.size_bytes for o in self.weight_openings.values()
        )

    @property
    def n_weight_rows(self) -> int:
        """The verifier's real budget: how many weight rows it got to see."""
        return len(self.weight_openings)


@dataclass(frozen=True)
class CheckFailure:
    """One reason the verifier rejected."""

    kind: str
    detail: str
    layer: int | None = None
    neuron: int | None = None
    claimed: float | None = None
    recomputed: float | None = None

    def __str__(self) -> str:
        location = ""
        if self.layer is not None:
            location = f" at layer {self.layer}"
            if self.neuron is not None:
                location += f", neuron {self.neuron}"
        return f"[{self.kind}]{location}: {self.detail}"


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one protocol execution.

    Carries more than a boolean on purpose: the experiments need the residuals of
    the individual local checks (to justify the tolerance), the identity of the
    checked nodes (to compare empirical detection against the analytic
    probability), and the proof size (for the cost comparison).
    """

    accepted: bool
    paths: tuple[Path, ...]
    checked_nodes: tuple[tuple[int, int], ...] = ()
    residuals: tuple[float, ...] = ()
    failures: tuple[CheckFailure, ...] = ()
    proof_size_bytes: int = 0
    n_weight_rows_opened: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def max_residual(self) -> float:
        """Largest local-check residual seen, or 0.0 if nothing was checked."""
        return max(self.residuals, default=0.0)

    def __bool__(self) -> bool:
        return self.accepted

    def __str__(self) -> str:
        verdict = "ACCEPT" if self.accepted else "REJECT"
        if self.accepted:
            return (
                f"{verdict} ({len(self.checked_nodes)} nodes checked, "
                f"max residual {self.max_residual:.3e})"
            )
        return f"{verdict}: " + "; ".join(str(f) for f in self.failures[:3])
