"""Fitting the parameters of an :class:`~pvi.nn.architecture.Architecture`.

Training is ordinary supervised learning and is not part of the protocol; it lives
here so the protocol modules stay free of PyTorch.  The one requirement the
protocol does impose is *determinism*: the committed model must be reproducible
from the recorded seed, or the artefacts we ship cannot be re-derived by a reader.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pvi.data import Dataset
from pvi.nn.architecture import Architecture
from pvi.nn.models import build_torch_module, extract_parameters
from pvi.nn.network import TracedNetwork

__all__ = ["TrainConfig", "TrainReport", "load_network", "save_network", "train_network"]


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters.  Defaults are tuned for CPU-only runs of a few minutes."""

    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class TrainReport:
    """What training achieved, recorded alongside the weights."""

    train_accuracy: float
    test_accuracy: float
    final_loss: float
    epochs: int
    seed: int


def train_network(
    architecture: Architecture,
    dataset: Dataset,
    config: TrainConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[TracedNetwork, TrainReport]:
    """Train ``architecture`` on ``dataset`` and freeze it into a traced network."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    module = build_torch_module(architecture)
    optimiser = torch.optim.Adam(
        module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.train_x.reshape(len(dataset.train_x), -1)),
            torch.from_numpy(dataset.train_y),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    final_loss = float("nan")
    module.train()
    for epoch in range(config.epochs):
        running, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = criterion(module(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            running += float(loss) * len(batch_x)
            seen += len(batch_x)
        final_loss = running / seen
        if verbose:
            print(f"  epoch {epoch + 1}/{config.epochs}  loss {final_loss:.4f}")

    module.eval()
    network = TracedNetwork(architecture, extract_parameters(module, architecture))

    report = TrainReport(
        train_accuracy=network.accuracy(
            dataset.train_x.reshape(len(dataset.train_x), -1), dataset.train_y
        ),
        test_accuracy=network.accuracy(
            dataset.test_x.reshape(len(dataset.test_x), -1), dataset.test_y
        ),
        final_loss=final_loss,
        epochs=config.epochs,
        seed=config.seed,
    )
    if verbose:
        print(
            f"  train acc {report.train_accuracy:.4f}  test acc {report.test_accuracy:.4f}"
        )
    return network, report


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_network(
    network: TracedNetwork, path: Path | str, report: TrainReport | None = None
) -> None:
    """Persist parameters as ``.npz``, with the training report beside them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, (weight, bias) in network.parameters.items():
        arrays[f"{name}.weight"] = weight
        arrays[f"{name}.bias"] = bias
    np.savez_compressed(path, **arrays)
    if report is not None:
        path.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2))


def load_network(architecture: Architecture, path: Path | str) -> TracedNetwork:
    """Rebuild a traced network from saved parameters."""
    with np.load(Path(path)) as blob:
        parameters = {
            layer.name: (blob[f"{layer.name}.weight"], blob[f"{layer.name}.bias"])
            for _, layer in architecture.parameterised_layers()
        }
    return TracedNetwork(architecture, parameters)
