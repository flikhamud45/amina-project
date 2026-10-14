"""Driving one full three-round execution of the path protocol.

An end-to-end run picks a query, lets the prover speak, draws a challenge, lets
the prover answer, and records the verdict; :func:`run_protocol` does that once.
"""

from __future__ import annotations

import numpy as np

from pvi.baseline.path.messages import VerificationResult
from pvi.baseline.path.prover import Prover
from pvi.baseline.path.sampling import sample_challenge
from pvi.baseline.path.verifier import Verifier
from pvi.mnist.network import Trace

__all__ = ["run_protocol"]


def run_protocol(
    prover: Prover,
    verifier: Verifier,
    query: np.ndarray,
    *,
    trace: Trace | None = None,
) -> VerificationResult:
    """Run prover and verifier against each other once.

    Parameters
    ----------
    trace:
        Optional trace for the prover to commit to instead of the honest
        ``EvalTrace(M, qry)``.  This is how the adversarial experiments inject a
        forged trace.
    """
    # Prover and verifier must agree on how the challenge maps to paths, or the
    # prover opens the wrong rows and every run rejects for the wrong reason.  The
    # mismatch is reported here rather than as a rejection.
    if type(prover.sampler) is not type(verifier.sampler):
        raise ValueError(
            f"prover uses {type(prover.sampler).__name__} but verifier uses "
            f"{type(verifier.sampler).__name__}; they must derive the same paths"
        )

    round1, state = prover.prove1(query, trace=trace)
    rho = sample_challenge()
    round2 = prover.prove2(state, rho)
    return verifier.verify(query, round1, round2, rho)
