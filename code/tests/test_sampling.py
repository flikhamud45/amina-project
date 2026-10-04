"""Tests for path sampling."""

from __future__ import annotations

import numpy as np

from pvi.nn.models import mlp_architecture
from pvi.protocol import UniformPathSampler


def test_sampler_is_reproducible_from_a_seed():
    architecture = mlp_architecture(8, [6], 3)
    sampler = UniformPathSampler()
    a = sampler.sample(architecture, np.random.default_rng(42))
    b = sampler.sample(architecture, np.random.default_rng(42))
    assert a.nodes == b.nodes


def test_checked_layers_exclude_the_input_layer():
    architecture = mlp_architecture(8, [6], 3)
    path = UniformPathSampler().sample(architecture, np.random.default_rng(0))
    assert list(path.checked_layers) == [1, 2]
