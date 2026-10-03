"""Concrete architectures and the PyTorch bridge used to train them.

The protocol implementation deliberately does not depend on PyTorch: a
:class:`~pvi.nn.network.TracedNetwork` is defined purely by an
:class:`~pvi.nn.architecture.Architecture` and a set of NumPy parameter arrays.
PyTorch appears only here, to *fit* those parameters.

Layer-for-layer correspondence with the architecture is exact, and the test suite
asserts that the traced network and the PyTorch module agree numerically on real
inputs.  Without that check it would be possible to train one model and commit to
the trace of a subtly different one.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from pvi.nn.architecture import (
    Architecture,
    Conv2dLayer,
    DenseLayer,
    InputLayer,
    Layer,
    MaxPool2dLayer,
)
from pvi.nn.network import ModelParameters

__all__ = [
    "build_torch_module",
    "extract_parameters",
    "mlp_architecture",
    "small_cnn_architecture",
]


# --------------------------------------------------------------------------- #
# Architecture factories
# --------------------------------------------------------------------------- #


def mlp_architecture(
    input_features: int,
    hidden_widths: Sequence[int],
    n_classes: int,
) -> Architecture:
    """A fully connected ReLU network with a linear output layer.

    This is the workhorse of the study.  Its dense parent sets mean a path step
    selects a uniformly random neuron of the layer below, which makes the
    per-path detection probability exactly ``1 / width`` and lets the analytic and
    empirical numbers be compared without any modelling slack.
    """
    if not hidden_widths:
        raise ValueError("an MLP needs at least one hidden layer")

    layers: list[Layer] = [InputLayer(name="input", shape=(input_features,))]
    previous = input_features
    for i, width in enumerate(hidden_widths):
        layers.append(
            DenseLayer(
                name=f"fc{i + 1}",
                in_features=previous,
                out_features=width,
                activation="relu",
            )
        )
        previous = width
    layers.append(
        DenseLayer(
            name="logits",
            in_features=previous,
            out_features=n_classes,
            activation="identity",
        )
    )
    return Architecture(tuple(layers))


def small_cnn_architecture(
    input_shape: tuple[int, int, int],
    n_classes: int,
    channels: Sequence[int] = (16, 32),
    hidden_width: int = 128,
) -> Architecture:
    """A compact ``conv -> pool`` stack followed by two dense layers.

    Included because the paper's own classifier experiments use a convolutional
    network (ResNet-18).  It exercises the local, weight-sharing parent sets that
    convolutions induce, which behave quite differently from dense layers under
    path sampling.
    """
    layers: list[Layer] = [InputLayer(name="input", shape=input_shape)]
    shape = input_shape
    for i, out_channels in enumerate(channels):
        conv = Conv2dLayer(
            name=f"conv{i + 1}",
            in_shape=shape,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="relu",
        )
        layers.append(conv)
        shape = conv.out_shape
        pool = MaxPool2dLayer(name=f"pool{i + 1}", in_shape=shape, kernel_size=2)
        layers.append(pool)
        shape = pool.out_shape

    flattened = int(np.prod(shape))
    layers.append(
        DenseLayer(
            name="fc1", in_features=flattened, out_features=hidden_width, activation="relu"
        )
    )
    layers.append(
        DenseLayer(
            name="logits",
            in_features=hidden_width,
            out_features=n_classes,
            activation="identity",
        )
    )
    return Architecture(tuple(layers))


# --------------------------------------------------------------------------- #
# PyTorch bridge
# --------------------------------------------------------------------------- #


class _ArchitectureModule(nn.Module):
    """A ``nn.Module`` whose parameters map one-to-one onto an architecture."""

    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        self.architecture = architecture
        modules: dict[str, nn.Module] = {}
        for layer in architecture.layers[1:]:
            if isinstance(layer, DenseLayer):
                modules[layer.name] = nn.Linear(layer.in_features, layer.out_features)
            elif isinstance(layer, Conv2dLayer):
                modules[layer.name] = nn.Conv2d(
                    layer.in_shape[0],
                    layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
            elif isinstance(layer, MaxPool2dLayer):
                modules[layer.name] = nn.MaxPool2d(
                    kernel_size=layer.kernel_size, stride=layer._stride
                )
            else:  # pragma: no cover - guarded by Architecture validation
                raise TypeError(f"cannot build a torch module for {type(layer).__name__}")
        self.layers = nn.ModuleDict(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        activation = x.reshape(batch, *self.architecture.input_layer.out_shape)
        for layer in self.architecture.layers[1:]:
            module = self.layers[layer.name]
            if isinstance(layer, DenseLayer):
                activation = module(activation.reshape(batch, -1))
                if layer.activation == "relu":
                    activation = torch.relu(activation)
            elif isinstance(layer, Conv2dLayer):
                activation = module(activation.reshape(batch, *layer.in_shape))
                if layer.activation == "relu":
                    activation = torch.relu(activation)
            else:
                activation = module(activation.reshape(batch, *layer.in_shape))
        return activation


def build_torch_module(architecture: Architecture) -> nn.Module:
    """Build a trainable module matching ``architecture`` layer for layer."""
    return _ArchitectureModule(architecture)


def extract_parameters(module: nn.Module, architecture: Architecture) -> ModelParameters:
    """Pull trained weights out of a module into the NumPy form the protocol uses."""
    if not isinstance(module, _ArchitectureModule):
        raise TypeError("expected a module produced by build_torch_module")
    parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for layer in architecture.layers[1:]:
            if layer.n_weight_groups == 0:
                continue
            sub = module.layers[layer.name]
            parameters[layer.name] = (
                sub.weight.detach().cpu().numpy().astype(np.float32).copy(),
                sub.bias.detach().cpu().numpy().astype(np.float32).copy(),
            )
    return parameters
