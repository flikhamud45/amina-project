"""Adversaries against the sampling-based proof-of-inference protocol."""

from pvi.attacks.backdoor import BackdoorAdversary, PatchTrigger, calibrate_activation_ceilings
from pvi.attacks.baselines import (
    gradient_forge,
    injection_forge,
    inverse_transform_forge,
    logit_swap_target,
)
from pvi.attacks.tamper import evaluate_plan, plan_single_neuron_flip

__all__ = [
    "BackdoorAdversary",
    "PatchTrigger",
    "calibrate_activation_ceilings",
    "evaluate_plan",
    "gradient_forge",
    "injection_forge",
    "inverse_transform_forge",
    "logit_swap_target",
    "plan_single_neuron_flip",
]
