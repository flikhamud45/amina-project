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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Round2:
    """``pi_2`` -- the openings requested by the challenge.

    ``weight_openings`` is keyed by ``(layer index, weight group)``.  Several path
    steps can land on the same weight row; it is sent once.
    """

    layer_openings: tuple[MerkleOpening, ...]
    weight_openings: dict[tuple[int, int], MerkleOpening]


@dataclass(frozen=True)
class CheckFailure:
    """One reason the verifier rejected."""

    kind: str
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one protocol execution: the verdict, the sampled paths and why it rejected."""

    accepted: bool
    paths: tuple[Path, ...]
    failures: tuple[CheckFailure, ...] = ()
