"""The prover of Figure 4.

An honest prover evaluates ``M`` on the query, commits to the resulting trace, and
answers the challenge with the openings along the sampled paths.

The class also serves the adversarial experiments, via the ``trace`` argument of
:meth:`Prover.prove1`: a cheating prover commits to a trace of its own choosing.
Note what it cannot choose.  Weight openings are produced from ``C_M``, the
commitment published at the end of training; position binding means a cheating
prover must open the *real* weights of ``M`` at every position the verifier asks
about.  This is the asymmetry the whole protocol rests on, and it is enforced here
structurally rather than by convention -- :class:`Prover` has no way to serve a
weight that is not in the committed model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.commitments import ModelCommitment, TraceCommitment
from pvi.protocol.messages import Round1, Round2
from pvi.protocol.params import ProtocolParams
from pvi.protocol.sampling import Path, PathSampler, UniformPathSampler, challenge_rng

__all__ = ["Prover", "ProverState"]


@dataclass
class ProverState:
    """State carried from ``Prove1`` to ``Prove2``."""

    query: np.ndarray
    trace: Trace
    trace_commitment: TraceCommitment


class Prover:
    """``(Prove1, Prove2)`` for the path-sampling protocol."""

    def __init__(
        self,
        network: TracedNetwork,
        model_commitment: ModelCommitment,
        params: ProtocolParams | None = None,
        sampler: PathSampler | None = None,
    ) -> None:
        if model_commitment.architecture is not network.architecture:
            raise ValueError("the model commitment does not belong to this network")
        self._network = network
        self._model_commitment = model_commitment
        self._params = params or ProtocolParams()
        self._sampler = sampler or UniformPathSampler()

    # -- accessors ---------------------------------------------------------- #

    @property
    def network(self) -> TracedNetwork:
        return self._network

    @property
    def model_commitment(self) -> ModelCommitment:
        return self._model_commitment

    @property
    def params(self) -> ProtocolParams:
        return self._params

    @property
    def sampler(self) -> PathSampler:
        return self._sampler

    # -- protocol ----------------------------------------------------------- #

    def prove1(
        self, query: np.ndarray, *, trace: Trace | None = None
    ) -> tuple[Round1, ProverState]:
        """``Prove1(pp, M, qry)``.

        With ``trace=None`` this is the honest prover: it runs ``EvalTrace(M, qry)``
        and commits to the result.  Supplying a ``trace`` models a cheating prover
        that commits to something else; the claimed output is then ``out(trc~)``,
        as Remark 2 requires.
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        if trace is None:
            trace = self._network.eval_trace(query)

        commitment = TraceCommitment(trace, security_bits=self._params.security_bits)
        message = Round1(
            claimed_output=trace.output.copy(),
            trace_digest=commitment.digest,
            trace_params=commitment.params,
        )
        return message, ProverState(query=query, trace=trace, trace_commitment=commitment)

    def paths_for(self, state: ProverState, challenge: bytes) -> tuple[Path, ...]:
        """Reconstruct the paths the challenge determines.

        Exposed separately because both the prover and the verifier must derive
        the identical set, and the experiments want to inspect it.
        """
        rng = challenge_rng(challenge)
        return self._sampler.sample_many(
            self._network.architecture,
            rng,
            self._params.n_paths,
            trace=state.trace,
            network=self._network,
        )

    def prove2(self, state: ProverState, challenge: bytes) -> Round2:
        """``Prove2(state, rho)`` -- open everything the sampled paths touch."""
        architecture = self._network.architecture
        paths = self.paths_for(state, challenge)

        # Every layer's activations are opened: a path step at layer l needs the
        # claimed value of the node itself and of all of its parents in layer l-1,
        # and the trace is committed one leaf per layer.
        layer_openings = tuple(
            state.trace_commitment.open_layer(i) for i in range(len(architecture))
        )

        weight_openings = {}
        for path in paths:
            for layer_index in path.checked_layers:
                layer = architecture[layer_index]
                if layer.n_weight_groups == 0:
                    continue
                group = layer.weight_group_of(path[layer_index])
                key = (layer_index, group)
                if key not in weight_openings:
                    weight_openings[key] = self._model_commitment.open_row(
                        layer_index, group
                    )

        return Round2(layer_openings=layer_openings, weight_openings=weight_openings)
