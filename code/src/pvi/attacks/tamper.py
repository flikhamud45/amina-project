"""Single-neuron trace tampering.

The attack rests on one structural observation about ``RandPathTest``.

The verifier accepts if and only if every node the sampled path visits satisfies its
local relation.  It has no other test.  So the adversary's problem is not to make a
forged trace *look* consistent -- which is impossible, since an anchored input plus
universally satisfied local relations pins the trace to ``EvalTrace(M, qry)`` -- but
to make the set of violating nodes as *small* and as *rarely visited* as possible.

That set can be made a single node:

1. Evaluate ``M`` honestly on the query.
2. Overwrite one activation ``a_v`` in some layer ``l``.
3. Recompute layers ``l+1 .. L`` from the modified value, using ``M``'s real weights.

Step 3 restores local consistency everywhere above ``l``; nothing below ``l`` was
touched; so the trace violates its local relation at exactly one node, ``v``.  The
verifier rejects only if its path happens to pass through ``v``, which for a dense
layer of width ``N`` happens with probability ``1/N``.

This is the ``1/N`` ceiling the paper itself states in Section 5.2 -- "an adversary
can modify a single node in a layer of size ``N``, and a single random path will
select that node with probability exactly ``1/N``" -- but leaves as a theoretical
remark, judging it "not ... a practical limitation" because their experiments show
detection across all tested paths.  Their experiments only ever test adversaries who
perturb the whole trace.  This module builds the adversary the remark describes, and
shows the ceiling is reached in practice.

Contrast with the paper's own attacks (:mod:`pvi.attacks.baselines`): those minimise
the *magnitude* of the inconsistency and end up spreading it over 33-100% of the
trace, so they are detected always.  This one accepts a large magnitude at one node
in exchange for a support of size one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvi.experiments.analysis import acceptance_probability, inconsistent_nodes
from pvi.nn.architecture import DenseLayer
from pvi.nn.gradients import saliency
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.params import ProtocolParams

__all__ = [
    "TamperOutcome",
    "TamperPlan",
    "apply_plan",
    "evaluate_plan",
    "plan_single_neuron_flip",
    "plan_spread_flip",
]


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TamperPlan:
    """Which activations to overwrite, and with what.

    A plan touching one neuron is the main object of study; ``plan_spread_flip``
    produces multi-neuron plans, which are needed to analyse the influence-weighted
    defence.
    """

    layer_index: int
    neurons: tuple[int, ...]
    new_values: tuple[float, ...]
    original_values: tuple[float, ...]
    honest_class: int
    forged_class: int

    @property
    def size(self) -> int:
        return len(self.neurons)

    @property
    def deltas(self) -> tuple[float, ...]:
        return tuple(
            new - old for new, old in zip(self.new_values, self.original_values)
        )

    def __str__(self) -> str:
        if self.size == 1:
            return (
                f"layer {self.layer_index}, neuron {self.neurons[0]}: "
                f"{self.original_values[0]:.4f} -> {self.new_values[0]:.4f} "
                f"(class {self.honest_class} -> {self.forged_class})"
            )
        return (
            f"layer {self.layer_index}, {self.size} neurons "
            f"(class {self.honest_class} -> {self.forged_class})"
        )


@dataclass(frozen=True)
class TamperOutcome:
    """What a plan achieves against the verifier."""

    plan: TamperPlan
    trace: Trace
    acceptance_probability: float
    n_inconsistent: int
    n_nodes: int
    honest_class: int
    forged_class: int
    layer_width: int
    max_residual: float

    @property
    def succeeded(self) -> bool:
        """The output changed *and* the trace violates only the intended nodes."""
        return (
            self.forged_class != self.honest_class
            and self.n_inconsistent == self.plan.size
        )

    def __str__(self) -> str:
        return (
            f"{self.plan} | acceptance {self.acceptance_probability:.4f} "
            f"| {self.n_inconsistent}/{self.n_nodes} nodes inconsistent"
        )


# --------------------------------------------------------------------------- #
# Applying a plan
# --------------------------------------------------------------------------- #


def apply_plan(network: TracedNetwork, honest: Trace, plan: TamperPlan) -> Trace:
    """Overwrite the planned activations and re-propagate everything above them.

    Re-propagation is what makes the attack sparse: it uses ``M``'s own weights, so
    every layer above the tampered one is consistent by construction.
    """
    trace = honest
    for neuron, value in zip(plan.neurons, plan.new_values):
        trace = trace.tampered(plan.layer_index, neuron, value)
    return network.forward_from(trace, plan.layer_index)


def evaluate_plan(
    network: TracedNetwork,
    honest: Trace,
    plan: TamperPlan,
    params: ProtocolParams | None = None,
) -> TamperOutcome:
    """Apply a plan and measure exactly what the verifier will do about it."""
    params = params or ProtocolParams()
    forged = apply_plan(network, honest, plan)
    report = inconsistent_nodes(network, forged, params)

    architecture = network.architecture
    residuals = []
    for layer_index in range(1, len(architecture)):
        layer = architecture[layer_index]
        weight, bias = network.parameters.get(layer.name, (None, None))
        recomputed = layer.forward_batch(forged[layer_index - 1][None, :], weight, bias)[0]
        residuals.append(np.abs(forged[layer_index] - recomputed))

    return TamperOutcome(
        plan=plan,
        trace=forged,
        acceptance_probability=acceptance_probability(network, forged, params),
        n_inconsistent=report.total,
        n_nodes=int(sum(report.per_layer_widths.values())),
        honest_class=int(honest.output.argmax()),
        forged_class=int(forged.output.argmax()),
        layer_width=architecture[plan.layer_index].n_neurons,
        max_residual=float(np.concatenate(residuals).max()),
    )


# --------------------------------------------------------------------------- #
# Finding a plan
# --------------------------------------------------------------------------- #


def _forward_with_value(
    network: TracedNetwork,
    honest: Trace,
    layer_index: int,
    neuron: int,
    value: float,
) -> np.ndarray:
    """Output obtained by setting one activation and propagating honestly."""
    return network.forward_from(
        honest.tampered(layer_index, neuron, value), layer_index
    ).output


def _smallest_flipping_value(
    network: TracedNetwork,
    honest: Trace,
    layer_index: int,
    neuron: int,
    honest_class: int,
    target_class: int | None,
    *,
    max_value: float,
    require_nonnegative: bool,
    bisection_steps: int = 40,
) -> float | None:
    """Least activation value at this neuron that changes the predicted class.

    The network is piecewise linear in a single activation, so the predicted class
    changes at a finite set of breakpoints.  We bracket the first one by geometric
    search and then bisect.  Working in "smallest value that suffices" terms rather
    than "some value that works" matters for stealth: the smaller the change, the
    more plausible the resulting activation looks.
    """
    original = float(honest[layer_index][neuron])

    def flips(value: float) -> bool:
        logits = _forward_with_value(network, honest, layer_index, neuron, value)
        predicted = int(logits.argmax())
        if predicted == honest_class:
            return False
        return target_class is None or predicted == target_class

    lower_bound = 0.0 if require_nonnegative else -max_value
    candidates: list[tuple[float, float]] = []

    # Search upwards, then downwards, for a bracketing interval.
    for sign in (1.0, -1.0):
        step, previous = 1.0, original
        for _ in range(48):
            value = original + sign * step
            if value > max_value or value < lower_bound:
                break
            if flips(value):
                candidates.append((min(previous, value), max(previous, value)))
                break
            previous, step = value, step * 2.0

    if not candidates:
        return None

    # Bisect each bracket and keep whichever endpoint moves the activation least.
    best: float | None = None
    for low, high in candidates:
        lo, hi = low, high
        lo_flips = flips(lo)
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            if flips(mid) == lo_flips:
                lo = mid
            else:
                hi = mid
        chosen = lo if lo_flips else hi
        if not flips(chosen):
            continue
        if best is None or abs(chosen - original) < abs(best - original):
            best = chosen
    return best


def plan_single_neuron_flip(
    network: TracedNetwork,
    honest: Trace,
    *,
    layer_index: int,
    target_class: int | None = None,
    candidate_neurons: int = 32,
    max_value: float = 1e4,
    require_nonnegative: bool = True,
    activation_ceiling: float | np.ndarray | None = None,
) -> TamperPlan | None:
    """Find one neuron whose activation can be overwritten to change the output.

    Candidates are ranked by ``|d(margin)/d a_v|`` at the honest trace, so the search
    starts with the neurons that move the decision fastest, and only the top
    ``candidate_neurons`` are searched exactly.  Among those that work, we keep the
    one requiring the smallest change.

    ``require_nonnegative`` keeps the forged activation a legal ReLU output.
    ``activation_ceiling`` caps it -- either globally, or (as an array) per neuron,
    which is how the stealthy variant confines each candidate to the range *that*
    neuron takes on natural inputs.  The per-neuron form matters: capping every
    candidate at the same global bound and rejecting afterwards would discard the
    very neurons whose natural range is wide enough to hide the change.

    Returns ``None`` when no candidate can flip the output within those limits.
    """
    architecture = network.architecture
    layer = architecture[layer_index]
    if not isinstance(layer, DenseLayer):
        raise TypeError(
            f"layer {layer_index} is a {type(layer).__name__}; the neuron-ranking "
            "step needs a fully connected layer"
        )

    honest_class = int(honest.output.argmax())
    scores = saliency(network, honest, layer_index, target_class=target_class)
    ranked = np.argsort(scores)[::-1][: max(1, candidate_neurons)]

    if activation_ceiling is None:
        ceilings = np.full(layer.n_neurons, max_value, dtype=np.float64)
    elif np.isscalar(activation_ceiling):
        ceilings = np.full(layer.n_neurons, min(max_value, float(activation_ceiling)))
    else:
        ceilings = np.minimum(
            np.asarray(activation_ceiling, dtype=np.float64), max_value
        )
        if ceilings.shape != (layer.n_neurons,):
            raise ValueError(
                f"activation_ceiling must have shape {(layer.n_neurons,)}, "
                f"got {ceilings.shape}"
            )

    best: TamperPlan | None = None
    best_delta = np.inf
    for neuron in ranked:
        value = _smallest_flipping_value(
            network,
            honest,
            layer_index,
            int(neuron),
            honest_class,
            target_class,
            max_value=float(ceilings[int(neuron)]),
            require_nonnegative=require_nonnegative,
        )
        if value is None:
            continue
        original = float(honest[layer_index][int(neuron)])
        delta = abs(value - original)
        if delta < best_delta:
            forged_logits = _forward_with_value(
                network, honest, layer_index, int(neuron), value
            )
            best_delta = delta
            best = TamperPlan(
                layer_index=layer_index,
                neurons=(int(neuron),),
                new_values=(float(value),),
                original_values=(original,),
                honest_class=honest_class,
                forged_class=int(forged_logits.argmax()),
            )
    return best


def plan_stealthy_flip(
    network: TracedNetwork,
    honest: Trace,
    *,
    layer_index: int,
    ceilings: np.ndarray,
    floors: np.ndarray | None = None,
    target_class: int | None = None,
    max_neurons: int = 32,
    candidate_pool: int = 128,
) -> TamperPlan | None:
    """Flip the output using only *in-distribution* activation values.

    A single-neuron tamper is the sparsest possible attack, but the value it needs
    can lie far outside the range the neuron ever takes on natural inputs.  That is
    a signature: a verifier which also checked each opened activation against a
    published per-neuron envelope would catch it, even though no local relation is
    violated at that node.  (The protocol specifies no such check -- but it is cheap,
    and this attack is the reason to add one.)

    This planner removes the signature.  Neurons are ranked by influence and pushed
    only as far as their own envelope allows -- up to ``ceilings`` when that helps
    the target class, down to ``floors`` when it does not -- and neurons are added
    one at a time until the output flips.  The result is the *smallest* set of
    in-range changes that suffices.

    The trade is support for plausibility: ``k`` neurons instead of one, so
    acceptance falls from ``1 - 1/N`` to about ``1 - k/N``.  For the models here ``k``
    is a handful out of hundreds, so the loss is slight.
    """
    architecture = network.architecture
    layer = architecture[layer_index]
    if not isinstance(layer, DenseLayer):
        raise TypeError(f"layer {layer_index} is a {type(layer).__name__}")

    ceilings = np.asarray(ceilings, dtype=np.float64)
    if ceilings.shape != (layer.n_neurons,):
        raise ValueError(f"ceilings must have shape {(layer.n_neurons,)}")
    if floors is None:
        floors = np.zeros(layer.n_neurons, dtype=np.float64)
    floors = np.asarray(floors, dtype=np.float64)

    honest_class = int(honest.output.argmax())
    logits = honest.output
    if target_class is None:
        target_class = int(np.argsort(logits)[::-1][1])
    if target_class == honest_class:
        return None

    from pvi.nn.gradients import output_direction_gradient

    direction = np.zeros_like(logits, dtype=np.float64)
    direction[target_class] = 1.0
    direction[honest_class] -= 1.0
    gradient = output_direction_gradient(network, honest, layer_index, direction)

    original = honest[layer_index].astype(np.float64)
    # Push each neuron to whichever end of its envelope helps the target class, and
    # measure how much room that actually buys.
    proposed = np.where(gradient >= 0, ceilings, floors)
    headroom = np.abs(gradient) * np.abs(proposed - original)
    ranked = np.argsort(headroom)[::-1][: max(1, candidate_pool)]
    ranked = [int(n) for n in ranked if headroom[n] > 0]
    if not ranked:
        return None

    for k in range(1, min(max_neurons, len(ranked)) + 1):
        chosen = ranked[:k]
        trace = honest
        for neuron in chosen:
            trace = trace.tampered(layer_index, neuron, float(proposed[neuron]))
        forged = network.forward_from(trace, layer_index).output
        if int(forged.argmax()) != target_class:
            continue
        neurons = tuple(n for n in chosen if proposed[n] != original[n])
        if not neurons:
            continue
        return TamperPlan(
            layer_index=layer_index,
            neurons=neurons,
            new_values=tuple(float(proposed[n]) for n in neurons),
            original_values=tuple(float(original[n]) for n in neurons),
            honest_class=honest_class,
            forged_class=target_class,
        )
    return None


def plan_spread_flip(
    network: TracedNetwork,
    honest: Trace,
    *,
    layer_index: int,
    n_neurons: int,
    target_class: int | None = None,
    rank_ascending: bool = True,
    max_scale: float = 1e6,
) -> TamperPlan | None:
    """Spread the tamper over ``n_neurons``, each moved a little.

    Needed for the defence analysis.  Against uniform sampling this is strictly
    worse than a single-neuron plan -- detection grows to roughly ``k/N``.  Against a
    sampler that concentrates on high-influence neurons it can be better, because
    ``rank_ascending`` selects the *least* influential neurons that still suffice,
    which are exactly the ones such a sampler rarely looks at.

    The direction is the margin gradient; the scale is found by geometric search.
    """
    architecture = network.architecture
    layer = architecture[layer_index]
    if not isinstance(layer, DenseLayer):
        raise TypeError(f"layer {layer_index} is a {type(layer).__name__}")

    honest_class = int(honest.output.argmax())
    logits = honest.output
    if target_class is None:
        target_class = int(np.argsort(logits)[::-1][1])

    direction = np.zeros_like(logits, dtype=np.float64)
    direction[target_class] = 1.0
    direction[honest_class] -= 1.0

    from pvi.nn.gradients import output_direction_gradient

    gradient = output_direction_gradient(network, honest, layer_index, direction)
    magnitude = np.abs(gradient)

    # Only neurons that push the margin the right way are useful.
    usable = np.flatnonzero(magnitude > 0)
    if usable.size < n_neurons:
        return None
    order = usable[np.argsort(magnitude[usable])]
    chosen = order[:n_neurons] if rank_ascending else order[::-1][:n_neurons]

    base = np.sign(gradient[chosen])
    original = honest[layer_index][chosen].astype(np.float64)

    scale = 1.0
    while scale < max_scale:
        values = np.maximum(original + scale * base, 0.0)
        trace = honest
        for neuron, value in zip(chosen, values):
            trace = trace.tampered(layer_index, int(neuron), float(value))
        forged = network.forward_from(trace, layer_index).output
        predicted = int(forged.argmax())
        if predicted == target_class:
            return TamperPlan(
                layer_index=layer_index,
                neurons=tuple(int(n) for n in chosen),
                new_values=tuple(float(v) for v in values),
                original_values=tuple(float(v) for v in original),
                honest_class=honest_class,
                forged_class=predicted,
            )
        scale *= 2.0
    return None
