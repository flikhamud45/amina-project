"""Experiment support code: separation metrics and soundness-parameter estimation."""

from pvi.experiments.analysis import (
    InconsistencyReport,
    acceptance_probability,
    inconsistent_nodes,
    locally_consistent_mask,
)
from pvi.experiments.estimation import (
    ParameterEstimate,
    SeparationDataset,
    build_separation_dataset,
    estimate_test_error,
    generate_candidates,
    select_parameters,
    valid_layers,
)
from pvi.experiments.separation import (
    LayerSeparation,
    equation1_separation,
    js_divergence,
    layer_js_divergences,
    path_separation,
    verifier_residuals,
)

__all__ = [
    "InconsistencyReport",
    "LayerSeparation",
    "ParameterEstimate",
    "acceptance_probability",
    "inconsistent_nodes",
    "locally_consistent_mask",
    "SeparationDataset",
    "build_separation_dataset",
    "equation1_separation",
    "estimate_test_error",
    "generate_candidates",
    "js_divergence",
    "layer_js_divergences",
    "path_separation",
    "select_parameters",
    "valid_layers",
    "verifier_residuals",
]
