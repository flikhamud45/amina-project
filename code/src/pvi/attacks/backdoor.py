"""A backdoor that a proof of inference certifies as correct.

:mod:`pvi.attacks.tamper` shows that one activation can be overwritten at a cost of
``1/N`` detection probability per path.  On its own that is an attack on soundness.
Turning it into a *backdoor* costs nothing extra and removes even that ``1/N``:

* On inputs without the trigger, the adversary serves the honest trace.  It is not
  merely close to ``M``'s -- it *is* ``EvalTrace(M, qry)``, bit for bit.  Clean
  accuracy is unchanged and the detection probability is exactly zero.
* On inputs carrying the trigger, it overwrites one activation and re-propagates,
  changing the answer at a detection probability of ``1/N`` per path.

So the served model agrees with the committed model everywhere except on a set the
adversary chose and the verifier has no reason to query.  This is the adversary the
paper's threat model does not reach.  Its Definition 1 quantifies over
``qry <- D_query``: soundness is a statement about the *query distribution*, and a
backdoor lives off-distribution by construction.

Note what the adversary does **not** need.  It never modifies a committed weight --
it could not, since the verifier opens weights from ``C_M`` and position binding
holds.  It needs no substitute model, no training, and no optimisation over the
trace.  It runs the real model, changes one number, and finishes the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pvi.attacks.tamper import TamperOutcome, TamperPlan, evaluate_plan, plan_single_neuron_flip
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.params import ProtocolParams

__all__ = [
    "BackdoorAdversary",
    "PatchTrigger",
    "ServedResponse",
    "audit_detection_probability",
    "calibrate_activation_ceilings",
]


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatchTrigger:
    """A small bright patch stamped into a corner of the image (BadNets-style).

    The trigger's only job is to tell the adversary *when* to cheat. It is not
    embedded in the model's weights -- the served weights are exactly ``M``'s -- so
    it leaves no trace in the committed model at all, and inspecting ``M`` cannot
    reveal it.
    """

    input_shape: tuple[int, int, int]
    size: int = 3
    value: float = 1.0
    corner: Literal["bottom_right", "top_left"] = "bottom_right"

    def apply(self, query: np.ndarray) -> np.ndarray:
        channels, height, width = self.input_shape
        image = np.array(query, dtype=np.float32, copy=True).reshape(
            channels, height, width
        )
        if self.corner == "bottom_right":
            image[:, height - self.size :, width - self.size :] = self.value
        else:
            image[:, : self.size, : self.size] = self.value
        return image.reshape(-1)

    def apply_batch(self, queries: np.ndarray) -> np.ndarray:
        return np.stack([self.apply(q) for q in queries])

    def is_present(self, query: np.ndarray, *, atol: float = 1e-6) -> bool:
        channels, height, width = self.input_shape
        image = np.asarray(query, dtype=np.float32).reshape(channels, height, width)
        patch = (
            image[:, height - self.size :, width - self.size :]
            if self.corner == "bottom_right"
            else image[:, : self.size, : self.size]
        )
        return bool(np.all(np.abs(patch - self.value) <= atol))


# --------------------------------------------------------------------------- #
# Stealth calibration
# --------------------------------------------------------------------------- #


def calibrate_activation_ceilings(
    network: TracedNetwork,
    queries: np.ndarray,
    layer_index: int,
    *,
    percentile: float = 99.0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Per-neuron activation ceiling from natural data.

    A verifier that opens a layer sees the claimed activations even when it cannot
    check them, so a wildly out-of-range value would be conspicuous to any
    downstream anomaly check -- one the protocol does not specify, but that a
    cautious deployment might add.  Capping the forged value at a high percentile of
    the neuron's natural range removes that signal.

    Used with ``plan_single_neuron_flip(activation_ceiling=...)``.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(len(queries), -1)
    collected = [
        network.forward_batch(queries[start : start + chunk_size])[layer_index]
        for start in range(0, len(queries), chunk_size)
    ]
    return np.percentile(np.concatenate(collected, axis=0), percentile, axis=0)


# --------------------------------------------------------------------------- #
# The adversary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServedResponse:
    """What the adversary hands the client for one query."""

    trace: Trace
    triggered: bool
    tampered: bool
    plan: TamperPlan | None
    honest_class: int
    served_class: int

    @property
    def output(self) -> np.ndarray:
        return self.trace.output


class BackdoorAdversary:
    """Commits ``M``, serves ``M`` on clean inputs, and cheats on triggered ones.

    The adversary holds the genuine model, because it must: every weight it opens
    comes from ``C_M``.  Its only freedom is which trace to commit to.
    """

    def __init__(
        self,
        network: TracedNetwork,
        trigger: PatchTrigger,
        *,
        layer_index: int,
        target_class: int | None = None,
        activation_ceiling: np.ndarray | None = None,
        candidate_neurons: int = 32,
    ) -> None:
        self._network = network
        self._trigger = trigger
        self._layer_index = layer_index
        self._target_class = target_class
        self._ceiling = activation_ceiling
        self._candidate_neurons = candidate_neurons

    @property
    def network(self) -> TracedNetwork:
        return self._network

    @property
    def trigger(self) -> PatchTrigger:
        return self._trigger

    @property
    def layer_index(self) -> int:
        return self._layer_index

    def serve(self, query: np.ndarray) -> ServedResponse:
        """Produce the trace to commit to for one query."""
        honest = self._network.eval_trace(query)
        honest_class = int(honest.output.argmax())

        if not self._trigger.is_present(query):
            # Bit-for-bit the honest trace.  Nothing to detect.
            return ServedResponse(
                trace=honest,
                triggered=False,
                tampered=False,
                plan=None,
                honest_class=honest_class,
                served_class=honest_class,
            )

        plan = plan_single_neuron_flip(
            self._network,
            honest,
            layer_index=self._layer_index,
            target_class=self._target_class,
            candidate_neurons=self._candidate_neurons,
            activation_ceiling=self._ceiling,
        )
        if plan is None:
            return ServedResponse(
                trace=honest,
                triggered=True,
                tampered=False,
                plan=None,
                honest_class=honest_class,
                served_class=honest_class,
            )

        from pvi.attacks.tamper import apply_plan

        forged = apply_plan(self._network, honest, plan)
        return ServedResponse(
            trace=forged,
            triggered=True,
            tampered=True,
            plan=plan,
            honest_class=honest_class,
            served_class=int(forged.output.argmax()),
        )

    def outcome(
        self, query: np.ndarray, params: ProtocolParams | None = None
    ) -> TamperOutcome | None:
        """Exact verifier behaviour for one triggered query, or ``None`` if untampered."""
        response = self.serve(query)
        if response.plan is None:
            return None
        honest = self._network.eval_trace(query)
        return evaluate_plan(self._network, honest, response.plan, params)


# --------------------------------------------------------------------------- #
# Audit-level accounting
# --------------------------------------------------------------------------- #


def audit_detection_probability(
    layer_width: int,
    *,
    n_paths: int = 1,
    trigger_rate: float = 1.0,
    n_audited_queries: int = 1,
) -> float:
    """Probability the adversary is caught over a whole audit campaign.

    A single query is caught only if it is one the adversary cheated on *and* one of
    the ``n_paths`` sampled paths hits the tampered node:

        ``p_query = trigger_rate * (1 - (1 - 1/N)^n_paths)``

    and over ``n`` independent queries, ``1 - (1 - p_query)^n``.

    ``trigger_rate`` is the fraction of the *verifier's* queries that carry the
    trigger.  This is the term the paper's amplification argument -- "repeated
    queries amplify detection probability" (abstract) -- silently assumes to be 1.
    An adversary who cheats only on inputs the auditor never sends sets it to 0, and
    no number of queries helps.
    """
    if not 0.0 <= trigger_rate <= 1.0:
        raise ValueError(f"trigger_rate must be in [0, 1], got {trigger_rate}")
    if layer_width < 1 or n_paths < 1 or n_audited_queries < 0:
        raise ValueError("invalid audit parameters")

    per_path_miss = 1.0 - 1.0 / layer_width
    per_query = trigger_rate * (1.0 - per_path_miss**n_paths)
    return 1.0 - (1.0 - per_query) ** n_audited_queries
