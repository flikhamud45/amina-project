"""Driving one full three-round protocol execution.

Almost every experiment does the same thing: pick a query, let the prover speak,
draw a challenge, let the prover answer, and record the verdict.  Factoring that
out keeps the experiment code about the experiment.
"""

from __future__ import annotations

import numpy as np

from pvi.nn.network import Trace
from pvi.protocol.messages import VerificationResult
from pvi.protocol.prover import Prover
from pvi.protocol.sampling import sample_challenge
from pvi.protocol.verifier import Verifier

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
    # prover will open the wrong rows and every run will reject for the wrong
    # reason.  Catching the mismatch here rather than as a mysterious rejection
    # saves a great deal of confusion when swapping in a different sampler.
    if type(prover.sampler) is not type(verifier.sampler):
        raise ValueError(
            f"prover uses {type(prover.sampler).__name__} but verifier uses "
            f"{type(verifier.sampler).__name__}; they must derive the same paths"
        )

    round1, state = prover.prove1(query, trace=trace)
    rho = sample_challenge()
    round2 = prover.prove2(state, rho)
    return verifier.verify(query, round1, round2, rho)
