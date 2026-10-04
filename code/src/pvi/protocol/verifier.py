"""The verifier of Figure 4.

What the verifier holds is deliberately minimal, and matches the paper exactly:

* the **public architecture** -- wiring, depth and layer sizes (Section 5.1);
* the **digest** ``C_M`` of the model, *not* the model;
* the query and the protocol parameters.

Everything else arrives as an authenticated opening.  The verifier therefore
cannot recompute the network, cannot compare an activation against "what it should
have been", and cannot inspect a weight it did not ask for.  That is the point:
its cost is sublinear in the model, and the security question is what such a party
can still detect.

The accept predicate is precisely the conjunction of the local consistency checks
along the sampled paths, plus the input anchor.  Recording that explicitly matters
for the attack analysis: a trace that is locally consistent at every node except a
set ``S`` is rejected if and only if the path visits ``S``.
"""

from __future__ import annotations

import numpy as np

from pvi.commitments.merkle import MerkleParams
from pvi.nn.architecture import Architecture
from pvi.nn.network import Trace
from pvi.protocol.commitments import ModelCommitment, TraceCommitment, WeightRowIndex
from pvi.protocol.messages import CheckFailure, Round1, Round2, VerificationResult
from pvi.protocol.params import ProtocolParams
from pvi.protocol.sampling import Path, PathSampler, UniformPathSampler, challenge_rng

__all__ = ["Verifier"]


class Verifier:
    """``Verify(pp, cm_M, qry, y, (pi_1, pi_2), rho)``."""

    def __init__(
        self,
        architecture: Architecture,
        model_digest: bytes,
        model_params: MerkleParams,
        params: ProtocolParams | None = None,
        sampler: PathSampler | None = None,
    ) -> None:
        self._architecture = architecture
        self._model_digest = model_digest
        self._model_params = model_params
        self._params = params or ProtocolParams()
        self._sampler = sampler or UniformPathSampler()
        self._weight_index = WeightRowIndex(architecture)

    @classmethod
    def from_commitment(
        cls,
        model_commitment: ModelCommitment,
        params: ProtocolParams | None = None,
        sampler: PathSampler | None = None,
    ) -> "Verifier":
        """Build a verifier from a published model commitment.

        Only the digest and the public parameters are retained -- the commitment
        object's decommitment information is not stored.
        """
        return cls(
            architecture=model_commitment.architecture,
            model_digest=model_commitment.digest,
            model_params=model_commitment.params,
            params=params,
            sampler=sampler,
        )

    @property
    def sampler(self) -> PathSampler:
        return self._sampler

    # -- verification ------------------------------------------------------- #

    def verify(
        self,
        query: np.ndarray,
        round1: Round1,
        round2: Round2,
        challenge: bytes,
    ) -> VerificationResult:
        architecture = self._architecture
        failures: list[CheckFailure] = []

        # -- 1. authenticate the trace ------------------------------------- #
        if len(round2.layer_openings) != len(architecture):
            return VerificationResult(
                accepted=False,
                paths=(),
                failures=(
                    CheckFailure(
                        kind="opening",
                        detail=(
                            f"expected {len(architecture)} layer openings, "
                            f"got {len(round2.layer_openings)}"
                        ),
                    ),
                ),
            )

        activations: list[np.ndarray] = []
        for layer_index, opening in enumerate(round2.layer_openings):
            values = TraceCommitment.verify_layer(
                round1.trace_params,
                round1.trace_digest,
                layer_index,
                architecture[layer_index].n_neurons,
                opening,
            )
            if values is None:
                failures.append(CheckFailure(kind="opening", detail="invalid trace opening"))
                return self._reject(failures, ())
            activations.append(values)

        trace = Trace(tuple(np.ascontiguousarray(a, dtype=np.float32) for a in activations))

        # -- 2. the claimed output must be the one that was committed ------- #
        if round1.claimed_output.shape != trace.output.shape or not np.array_equal(
            np.asarray(round1.claimed_output, dtype=np.float32), trace.output
        ):
            failures.append(
                CheckFailure(
                    kind="output",
                    detail="claimed output does not match the committed trace",
                )
            )
            return self._reject(failures, ())

        # -- 3. derive the paths -------------------------------------------- #
        # Done after the trace is authenticated so that adaptive samplers, which
        # read the claimed activations, operate on values the prover is bound to.
        rng = challenge_rng(challenge)
        paths = self._sampler.sample_many(
            architecture, rng, self._params.n_paths, trace=trace
        )

        # -- 4. the whole input layer must be the query ---------------------- #
        query_flat = np.ascontiguousarray(query, dtype=np.float32).ravel()
        if query_flat.shape != trace[0].shape or not np.array_equal(query_flat, trace[0]):
            failures.append(
                CheckFailure(
                    kind="input_anchor",
                    detail="committed input layer differs from the query",
                )
            )
            return self._reject(failures, paths)

        # -- 5. local consistency along each path --------------------------- #
        for path in paths:
            for layer_index in path.checked_layers:
                layer = architecture[layer_index]
                neuron = path[layer_index]
                parent_idx, weight_idx = layer.parents(neuron)
                parent_values = trace[layer_index - 1][parent_idx]

                weights: np.ndarray | None = None
                bias: float | None = None
                if layer.n_weight_groups > 0:
                    group = layer.weight_group_of(neuron)
                    opening = round2.weight_openings.get((layer_index, group))
                    if opening is None:
                        failures.append(
                            CheckFailure(kind="opening", detail="missing weight-row opening")
                        )
                        continue
                    full_row = ModelCommitment.verify_row(
                        self._model_params,
                        self._model_digest,
                        self._weight_index,
                        layer_index,
                        group,
                        opening,
                    )
                    if full_row is None:
                        failures.append(
                            CheckFailure(kind="opening", detail="invalid weight-row opening")
                        )
                        continue
                    weights, bias = layer.split_row(full_row, weight_idx)

                claimed = float(trace[layer_index][neuron])
                recomputed = layer.local_value(neuron, parent_values, weights, bias)

                if not self._params.within_tolerance(claimed, recomputed):
                    failures.append(
                        CheckFailure(kind="local", detail="local consistency check failed")
                    )

        return VerificationResult(accepted=not failures, paths=paths, failures=tuple(failures))

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _reject(failures: list[CheckFailure], paths: tuple[Path, ...]) -> VerificationResult:
        return VerificationResult(accepted=False, paths=paths, failures=tuple(failures))
