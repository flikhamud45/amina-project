"""Shared fixtures.

Tests use small randomly initialised networks rather than trained ones: the
protocol is indifferent to whether the weights are any good, and random weights
keep the suite fast enough to run on every change.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.nn.architecture import Architecture, Conv2dLayer, DenseLayer
from pvi.nn.models import mlp_architecture, small_cnn_architecture
from pvi.nn.network import TracedNetwork
from pvi.protocol import ModelCommitment, ProtocolParams, Prover, Verifier


def random_parameters(
    architecture: Architecture, seed: int = 0, scale: float = 0.25
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Initialise every parameterised layer with small random values."""
    rng = np.random.default_rng(seed)
    parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, layer in architecture.parameterised_layers():
        if isinstance(layer, DenseLayer):
            shape: tuple[int, ...] = (layer.out_features, layer.in_features)
            n_out = layer.out_features
        elif isinstance(layer, Conv2dLayer):
            shape = (
                layer.out_channels,
                layer.in_shape[0],
                layer.kernel_size,
                layer.kernel_size,
            )
            n_out = layer.out_channels
        else:  # pragma: no cover
            raise TypeError(f"no initialiser for {type(layer).__name__}")
        parameters[layer.name] = (
            (rng.standard_normal(shape) * scale).astype(np.float32),
            (rng.standard_normal(n_out) * 0.05).astype(np.float32),
        )
    return parameters


def build_network(architecture: Architecture, seed: int = 0) -> TracedNetwork:
    return TracedNetwork(architecture, random_parameters(architecture, seed))


@pytest.fixture(scope="session")
def mlp_arch() -> Architecture:
    return mlp_architecture(24, [32, 16], 5)


@pytest.fixture(scope="session")
def cnn_arch() -> Architecture:
    return small_cnn_architecture((1, 12, 12), 4, channels=(4, 6), hidden_width=16)


@pytest.fixture(scope="session")
def mlp_net(mlp_arch: Architecture) -> TracedNetwork:
    return build_network(mlp_arch, seed=0)


@pytest.fixture(scope="session")
def cnn_net(cnn_arch: Architecture) -> TracedNetwork:
    return build_network(cnn_arch, seed=1)


@pytest.fixture(scope="session")
def query_for():
    def make(network: TracedNetwork, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(network.architecture.input_layer.n_neurons).astype(
            np.float32
        )

    return make


@pytest.fixture
def protocol_pair():
    """Build a matched ``(prover, verifier)`` for a network."""

    def make(network: TracedNetwork, params: ProtocolParams | None = None):
        params = params or ProtocolParams()
        commitment = ModelCommitment(network, security_bits=params.security_bits)
        return (
            Prover(network, commitment, params),
            Verifier.from_commitment(commitment, params),
        )

    return make
