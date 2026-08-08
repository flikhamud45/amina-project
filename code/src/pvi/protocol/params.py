"""Protocol parameters.

The knobs here are exactly the ones the paper leaves open, and each carries a
security consequence that the experiments measure.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProtocolParams"]


@dataclass(frozen=True)
class ProtocolParams:
    """Configuration of one instantiation of the protocol.

    Attributes
    ----------
    security_bits:
        Security parameter ``lambda`` handed to the vector commitment.
    n_paths:
        Number of independent output-to-input paths sampled per query.  Figure 3
        describes a single path; Section 5.2 suggests sampling several as "a
        natural mitigation" against concentrated tampering, at the cost of larger
        proofs.  We keep it explicit so the soundness/cost trade-off can be
        measured rather than assumed.
    abs_tolerance, rel_tolerance:
        Slack in the local consistency check.  The idealised test in Figure 3
        demands exact equality, but a real prover and a real verifier evaluate
        ``sum_i w_ij a_i`` with different summation orders and therefore obtain
        different floating-point results.  The paper acknowledges this and
        settles on "a reasonable ``1e-4`` threshold" (Appendix E.3); we adopt the
        same default and report the honest residual distribution so the choice can
        be seen to be safe for completeness.

        A node passes when ``|a~_j - phi(sum w_ij a~_i)| <= abs_tolerance +
        rel_tolerance * |phi(sum w_ij a~_i)|``.
    check_input_anchor:
        Whether the verifier checks that the input-layer activation reached by the
        path equals the corresponding entry of ``qry``.  Section 2 motivates the
        whole path construction with the observation that "the input layer remains
        an immutable anchor", so this is on by default.
    check_full_input:
        Strengthening beyond Figure 3: compare the *entire* opened input layer
        against ``qry`` rather than only the one position the path reaches.  The
        paper does not specify this, but since the input layer is opened anyway it
        costs nothing, and any careful implementation would do it.  Off by default
        so that the reproduction stays faithful; the attack experiments switch it
        on to show the attack is unaffected by the stronger check.
    """

    security_bits: int = 128
    n_paths: int = 1
    abs_tolerance: float = 1e-4
    rel_tolerance: float = 0.0
    check_input_anchor: bool = True
    check_full_input: bool = False

    def __post_init__(self) -> None:
        if self.n_paths < 1:
            raise ValueError(f"n_paths must be at least 1, got {self.n_paths}")
        if self.abs_tolerance < 0 or self.rel_tolerance < 0:
            raise ValueError("tolerances must be non-negative")

    def within_tolerance(self, claimed: float, recomputed: float) -> bool:
        """Whether a single local check passes."""
        return abs(claimed - recomputed) <= (
            self.abs_tolerance + self.rel_tolerance * abs(recomputed)
        )
