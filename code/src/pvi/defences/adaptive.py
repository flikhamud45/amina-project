"""The adversary's reply to a non-uniform sampler.

The defences in :mod:`pvi.defences.sampling` all concentrate the verifier's
attention somewhere.  Concentrating attention necessarily takes it away from
somewhere else, and the adversary only needs one place to hide.

Kerckhoffs applies: the sampling rule is part of the protocol, so the adversary
knows it.  The search below evaluates, for up to ``candidate_neurons`` neurons (the
most and least salient, and an even spread across the layer), the *exact* probability
that the resulting trace is accepted under the defence in force, and keeps the best.
Restricting the candidates can only weaken the adversary, so the resulting numbers
remain an upper bound on the defence's worth.

This is the concrete form of the concern that motivated the defence in the first
place: whether an attacker can arrange to be examined where it is safe.  It can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvi.attacks.tamper import TamperPlan, apply_plan, smallest_flipping_value
from pvi.experiments.analysis import acceptance_probability
from pvi.nn.architecture import DenseLayer
from pvi.nn.gradients import saliency
from pvi.nn.network import Trace, TracedNetwork
from pvi.protocol.params import ProtocolParams
from pvi.protocol.sampling import PathSampler

__all__ = ["EvasionSearch", "plan_adaptive_stealthy_flip", "plan_evasive_flip"]


@dataclass(frozen=True)
class EvasionSearch:
    """Best plan found against a given sampler, and its exact acceptance probability."""

    plan: TamperPlan | None
    acceptance: float

    @property
    def succeeded(self) -> bool:
        return self.plan is not None


def plan_evasive_flip(
    network: TracedNetwork,
    honest: Trace,
    *,
    layer_index: int,
    sampler: PathSampler,
    params: ProtocolParams | None = None,
    candidate_neurons: int = 96,
    require_nonnegative: bool = True,
) -> EvasionSearch:
    """Choose the tamper that the given sampler is least likely to look at.

    Candidates are drawn from three places, because the neuron that best evades a
    defence is generally *not* the one that flips the output most cheaply:

    * the most influential neurons -- these flip the output easily;
    * the least influential -- these are where importance-weighted samplers rarely
      look;
    * a spread across the layer, so the search is not confined to the tails.

    Each candidate is fully evaluated: find the smallest value that flips the
    output, build the resulting trace, and compute the exact acceptance probability
    under ``sampler``.
    """
    params = params or ProtocolParams()
    architecture = network.architecture
    layer = architecture[layer_index]
    if not isinstance(layer, DenseLayer):
        raise TypeError(f"layer {layer_index} is a {type(layer).__name__}")

    width = layer.n_neurons
    honest_class = int(honest.output.argmax())
    scores = saliency(network, honest, layer_index)
    ordered = np.argsort(scores)

    per_bucket = max(1, candidate_neurons // 3)
    candidates = np.unique(
        np.concatenate(
            [
                ordered[::-1][:per_bucket],
                ordered[:per_bucket],
                np.linspace(0, width - 1, per_bucket).astype(int),
            ]
        )
    )

    best_plan: TamperPlan | None = None
    best_acceptance = -1.0

    for neuron in candidates:
        value = smallest_flipping_value(
            network,
            honest,
            layer_index,
            int(neuron),
            honest_class,
            None,
            max_value=1e4,
            require_nonnegative=require_nonnegative,
        )
        if value is None:
            continue
        plan = TamperPlan(layer_index=layer_index, neurons=(int(neuron),), new_values=(float(value),))
        forged = apply_plan(network, honest, plan)
        forged_class = int(forged.output.argmax())
        if forged_class == honest_class:
            continue
        acceptance = acceptance_probability(network, forged, params, sampler)
        if acceptance > best_acceptance:
            best_acceptance = acceptance
            best_plan = plan

    return EvasionSearch(
        plan=best_plan,
        acceptance=best_acceptance if best_plan is not None else 0.0,
    )


def plan_adaptive_stealthy_flip(
    network: TracedNetwork,
    honest: Trace,
    *,
    layer_index: int,
    ceilings: np.ndarray,
    sampler: PathSampler,
    params: ProtocolParams | None = None,
    max_neurons: int = 48,
) -> EvasionSearch:
    """In-distribution values *and* low exposure to the sampler in force.

    The stealthy planner of :mod:`pvi.attacks.tamper` ranks neurons by how much
    room their envelope gives -- which, against a contribution-weighted sampler,
    is precisely the wrong ranking: a neuron with a large activation and large
    outgoing weights is both useful and conspicuous.

    Here each neuron is scored by *benefit per unit of exposure*:

        ``benefit_i  = |d(margin)/d a_i| * |proposed_i - a_i|``
        ``exposure_i = |proposed_i| * sum_j |w_ij|``

    the second being the contribution mass a contribution-weighted walk would
    assign it.  Neurons are added greedily until the output flips.  Three rankings
    are tried -- benefit, benefit/exposure, and least exposure -- and whichever
    yields the highest *exact* acceptance under ``sampler`` is returned, so the
    reported figure is the best of the adversary's options rather than the first
    that happened to work.
    """
    params = params or ProtocolParams()
    architecture = network.architecture
    layer = architecture[layer_index]
    if not isinstance(layer, DenseLayer):
        raise TypeError(f"layer {layer_index} is a {type(layer).__name__}")

    honest_class = int(honest.output.argmax())
    logits = honest.output
    target_class = int(np.argsort(logits)[::-1][1])

    from pvi.nn.gradients import output_direction_gradient

    direction = np.zeros_like(logits, dtype=np.float64)
    direction[target_class] = 1.0
    direction[honest_class] -= 1.0
    gradient = output_direction_gradient(network, honest, layer_index, direction)

    ceilings = np.asarray(ceilings, dtype=np.float64)
    original = honest[layer_index].astype(np.float64)
    proposed = np.where(gradient >= 0, ceilings, 0.0)

    above = architecture[layer_index + 1]
    if isinstance(above, DenseLayer):
        outgoing = np.abs(network.parameters[above.name][0].astype(np.float64)).sum(axis=0)
    else:
        outgoing = np.ones(layer.n_neurons)

    benefit = np.abs(gradient) * np.abs(proposed - original)
    exposure = np.abs(proposed) * outgoing + 1e-12

    rankings = {
        "benefit": np.argsort(benefit)[::-1],
        "benefit_per_exposure": np.argsort(benefit / exposure)[::-1],
        "least_exposure": np.argsort(np.where(benefit > 0, exposure, np.inf)),
    }

    best_plan: TamperPlan | None = None
    best_acceptance = -1.0

    for order in rankings.values():
        usable = [int(n) for n in order if benefit[n] > 0][:max_neurons]
        trace = honest
        chosen: list[int] = []
        for neuron in usable:
            trace = trace.tampered(layer_index, neuron, float(proposed[neuron]))
            chosen.append(neuron)
            forged = network.forward_from(trace, layer_index)
            if int(forged.output.argmax()) != target_class:
                continue
            touched = tuple(n for n in chosen if proposed[n] != original[n])
            if not touched:
                continue
            acceptance = acceptance_probability(network, forged, params, sampler)
            if acceptance > best_acceptance:
                best_acceptance = acceptance
                best_plan = TamperPlan(
                    layer_index=layer_index,
                    neurons=touched,
                    new_values=tuple(float(proposed[n]) for n in touched),
                )
            break

    return EvasionSearch(
        plan=best_plan,
        acceptance=best_acceptance if best_plan is not None else 0.0,
    )
