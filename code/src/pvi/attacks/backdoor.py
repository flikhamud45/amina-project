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

import numpy as np

from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.nn.network import Trace, TracedNetwork

__all__ = [
    "BackdoorAdversary",
    "PatchTrigger",
    "ServedResponse",
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

    def apply(self, query: np.ndarray) -> np.ndarray:
        channels, height, width = self.input_shape
        image = np.array(query, dtype=np.float32, copy=True).reshape(
            channels, height, width
        )
        image[:, height - self.size :, width - self.size :] = self.value
        return image.reshape(-1)

    def apply_batch(self, queries: np.ndarray) -> np.ndarray:
        return np.stack([self.apply(q) for q in queries])

    def is_present(self, query: np.ndarray, *, atol: float = 1e-6) -> bool:
        channels, height, width = self.input_shape
        image = np.asarray(query, dtype=np.float32).reshape(channels, height, width)
        patch = image[:, height - self.size :, width - self.size :]
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

    Used with ``plan_stealthy_flip(ceilings=...)``.
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
    honest_class: int
    served_class: int


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
        candidate_neurons: int = 32,
    ) -> None:
        self._network = network
        self._trigger = trigger
        self._layer_index = layer_index
        self._target_class = target_class
        self._candidate_neurons = candidate_neurons

    def _plan(self, honest, candidates: int):
        return plan_single_neuron_flip(
            self._network,
            honest,
            layer_index=self._layer_index,
            target_class=self._target_class,
            candidate_neurons=candidates,
        )

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
                honest_class=honest_class,
                served_class=honest_class,
            )

        plan = self._plan(honest, self._candidate_neurons)
        if plan is None:
            # The saliency ranking is first order at the honest trace: a neuron that reaches
            # the target only under a large overwrite (which switches other layer-2 units on)
            # can rank below the cut.  Only then try every neuron of the layer, so plans the
            # top-k search finds are unchanged.
            plan = self._plan(honest, self._network.architecture[self._layer_index].n_neurons)
        if plan is None:
            return ServedResponse(
                trace=honest,
                triggered=True,
                tampered=False,
                honest_class=honest_class,
                served_class=honest_class,
            )

        forged = apply_plan(self._network, honest, plan)
        return ServedResponse(
            trace=forged,
            triggered=True,
            tampered=True,
            honest_class=honest_class,
            served_class=int(forged.output.argmax()),
        )
