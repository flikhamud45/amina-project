"""Adversaries against the sampling-based proof-of-inference protocol."""

from pvi.attacks.backdoor import (
    BackdoorAdversary,
    PatchTrigger,
    ServedResponse,
    audit_detection_probability,
    calibrate_activation_ceilings,
)
from pvi.attacks.baselines import (
    ForgeryResult,
    gradient_forge,
    injection_forge,
    inverse_transform_forge,
    logit_swap_target,
    strawman_detection_probability,
    substitute_model_target,
)
from pvi.attacks.tamper import (
    TamperOutcome,
    TamperPlan,
    apply_plan,
    evaluate_plan,
    plan_single_neuron_flip,
    plan_spread_flip,
)

__all__ = [
    "BackdoorAdversary",
    "ForgeryResult",
    "PatchTrigger",
    "ServedResponse",
    "TamperOutcome",
    "TamperPlan",
    "apply_plan",
    "audit_detection_probability",
    "calibrate_activation_ceilings",
    "evaluate_plan",
    "plan_single_neuron_flip",
    "plan_spread_flip",
    "gradient_forge",
    "injection_forge",
    "inverse_transform_forge",
    "logit_swap_target",
    "strawman_detection_probability",
    "substitute_model_target",
]
