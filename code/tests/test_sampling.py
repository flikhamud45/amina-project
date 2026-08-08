"""Tests for path sampling and the induced visit distribution.

The visit distribution is what turns "the trace is inconsistent at node ``v``"
into a detection probability, so it is worth pinning down exactly rather than
assuming the ``1/N`` figure the paper quotes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.nn.models import mlp_architecture, small_cnn_architecture
from pvi.protocol import UniformPathSampler, visit_probabilities


def test_visit_probabilities_are_a_distribution():
    architecture = mlp_architecture(12, [10, 8], 4)
    for layer_index in range(len(architecture)):
        probability = visit_probabilities(architecture, layer_index)
        assert probability.shape == (architecture[layer_index].n_neurons,)
        assert probability.sum() == pytest.approx(1.0)
        assert (probability >= 0).all()


def test_dense_layers_are_visited_uniformly():
    """With dense parent sets every step is uniform over the layer below, which is
    exactly the regime in which the paper's ``1/N`` bound is tight."""
    architecture = mlp_architecture(12, [10, 8], 4)
    for layer_index in range(len(architecture)):
        width = architecture[layer_index].n_neurons
        probability = visit_probabilities(architecture, layer_index)
        assert np.allclose(probability, 1.0 / width)


def test_convolutional_layers_are_visited_non_uniformly():
    """Receptive fields are local, so border neurons are reached less often.  An
    adversary can exploit this: the least-visited neuron is strictly safer than
    the ``1/N`` average would suggest."""
    architecture = small_cnn_architecture((1, 12, 12), 4, channels=(4, 6), hidden_width=16)
    probability = visit_probabilities(architecture, 1)
    assert not np.allclose(probability, probability[0])
    assert probability.min() < 1.0 / len(probability) < probability.max()


@pytest.mark.parametrize("layer_index", [1, 2, 3])
def test_analytic_visit_probabilities_match_monte_carlo(layer_index):
    architecture = mlp_architecture(12, [10, 8], 4)
    exact = visit_probabilities(architecture, layer_index)
    empirical = visit_probabilities(
        architecture, layer_index, samples=120_000, rng=np.random.default_rng(3)
    )
    assert np.abs(exact - empirical).max() < 0.01


def test_analytic_visit_probabilities_match_monte_carlo_for_conv():
    architecture = small_cnn_architecture((1, 8, 8), 3, channels=(3,), hidden_width=8)
    exact = visit_probabilities(architecture, 1)
    empirical = visit_probabilities(
        architecture, 1, samples=200_000, rng=np.random.default_rng(5)
    )
    assert np.abs(exact - empirical).max() < 0.01


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
