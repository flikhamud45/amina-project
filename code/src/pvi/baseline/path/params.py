"""Protocol parameters of the path protocol.

The knobs here are exactly the ones Anchuri et al. leave open, and each carries a
security consequence that the attack experiments measure.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProtocolParams"]


@dataclass(frozen=True)
class ProtocolParams:
    """Configuration of one instantiation of the protocol.

    Attributes
    ----------
    n_paths:
        Number of independent output-to-input paths sampled per query.  Figure 3
        of Anchuri et al. describes a single path; their Section 5.2 suggests
        sampling several as "a natural mitigation" against concentrated tampering,
        at the cost of larger proofs.  It is explicit here so that the
        soundness/cost trade-off can be measured rather than assumed.
    abs_tolerance:
        Slack in the local consistency check.  The idealised test in Figure 3
        demands exact equality, but a real prover and a real verifier evaluate
        ``sum_i w_ij a_i`` with different summation orders and therefore obtain
        different floating-point results.  Anchuri et al. acknowledge this and
        settle on "a reasonable ``1e-4`` threshold" (their Appendix E.3); the same
        default is used here, and the honest residual distribution is reported so
        that the choice can be seen to be safe for completeness.

        A node passes when ``|a~_j - phi(sum w_ij a~_i)| <= abs_tolerance``.
    """

    n_paths: int = 1
    abs_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.n_paths < 1:
            raise ValueError(f"n_paths must be at least 1, got {self.n_paths}")
        if self.abs_tolerance < 0:
            raise ValueError("the tolerance must be non-negative")

    def within_tolerance(self, claimed: float, recomputed: float) -> bool:
        """Whether a single local check passes."""
        return abs(claimed - recomputed) <= self.abs_tolerance
