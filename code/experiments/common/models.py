"""The benchmark models: decoder shapes (the paper's models and small ones for tests and rehearsals) and the trained
image models of the benchmark."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiments.common.paths import CNN_MODELS, MNIST_MODELS
from pvi.models.decoder import CONFIGS, DecoderConfig

__all__ = ["TINY", "decoder_config", "trained_mlp", "trained_cnn", "test_batches", "note_file"]

TINY = {
    "tiny-gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=256),
    "tiny-opt": DecoderConfig("tiny-opt", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=256),
    "tiny-llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                                bias=False, tied=False),
    "tiny-qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                               rope_theta=1e6, bias=False, qk_norm=True),
    "small-llama": DecoderConfig("small-llama", 256, 4, 8, 4, 32, 688, 4096, norm="rmsnorm", mlp="swiglu",
                                 pos="rope", bias=False, tied=False),
}
"""Small decoder shapes for a laptop and the rehearsals; the test suites define their own in ``tests/shapes.py``."""


def decoder_config(name: str) -> DecoderConfig:
    """The decoder shapes of ``name``: one of :data:`TINY` or of :data:`pvi.models.decoder.CONFIGS`."""
    if name in TINY:
        return TINY[name]
    if name not in CONFIGS:
        raise SystemExit(f"unknown decoder {name!r}: one of {', '.join(sorted(CONFIGS) + sorted(TINY))}")
    return CONFIGS[name]


def note_file(name: str, path) -> None:
    """Keep the SHA-256 of a model or dataset file the process loaded (``sha256:<hex>``) in the records' ``sources``
    (:data:`experiments.common.checkpoints.SOURCES`, which :func:`experiments.common.records.environment` records), so
    that a record names the exact weights and data behind it; nothing for a missing file."""
    from pathlib import Path

    from experiments.common import checkpoints, records
    if Path(path).is_file():
        checkpoints.SOURCES[name] = "sha256:" + records.file_sha256(str(path))


def trained_mlp() -> tuple[nn.Module, dict]:
    """The MNIST MLP of the attack (``artifacts/models/mlp_mnist_full.npz``) as a torch model, with 512 training
    images for calibration and the test split; the weights' SHA-256 goes to the records' ``sources``
    (:func:`note_file`)."""
    from pvi.mnist.data import load_classification
    from pvi.mnist.training import load_network
    from pvi.mnist.zoo import mlp_architecture_for

    net = load_network(mlp_architecture_for(10), MNIST_MODELS / "mlp_mnist_full.npz")
    note_file("models/mlp_mnist_full.npz", MNIST_MODELS / "mlp_mnist_full.npz")
    layers = [layer for layer in net.architecture.layers if layer.name in net.parameters]
    mods: list[nn.Module] = []
    for i, layer in enumerate(layers):
        w, b = net.parameters[layer.name]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = torch.from_numpy(np.asarray(w, dtype=np.float32))
        lin.bias.data = torch.from_numpy(np.asarray(b, dtype=np.float32))
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ReLU())
    ds = load_classification("mnist")
    data = {"train_x": torch.from_numpy(ds.train_x[:512]).reshape(-1, 784),
            "test_x": torch.from_numpy(ds.test_x).reshape(-1, 784), "test_y": torch.from_numpy(ds.test_y)}
    return nn.Sequential(*mods).eval(), data


def trained_cnn(name: str) -> tuple[nn.Module, dict]:
    """A trained image model of the benchmark (``artifacts/models/cnn/<name>.pt``, from ``costs/train_cnn.py``), with
    its calibration images (512 training images; 64 for the ImageNet models, centre-cropped to 224) and test split.
    The SHA-256 of the weights and, for the ImageNet models, of the subset's cache go to the records' ``sources``
    (:func:`note_file`)."""
    from pvi.models.cnn import MODEL_SPECS, build_float_model
    from pvi.models.datasets import CACHE_ROOT, load_dataset, normalise

    kind, dataset = MODEL_SPECS[name]
    tx, _, vx, vy, family, n_classes = load_dataset(dataset)
    model = build_float_model(kind, n_classes)
    model.load_state_dict(torch.load(CNN_MODELS / f"{name}.pt", map_location="cpu", weights_only=True))
    note_file(f"models/cnn/{name}.pt", CNN_MODELS / f"{name}.pt")
    if dataset.startswith("imagenet_"):
        sub = dataset.removeprefix("imagenet_")
        note_file(f"cache/{sub}.pt", CACHE_ROOT / f"{sub}.pt")
    crop = (lambda x: x[:, :, 16:240, 16:240]) if family == "imagenet" else (lambda x: x)
    g = torch.Generator().manual_seed(0)
    cal = tx[torch.randperm(len(tx), generator=g)[: (64 if family == "imagenet" else 512)]]
    data = {"train_x": normalise(crop(cal), family), "test_x_uint8": vx, "test_y": vy,
            "norm": lambda x: normalise(crop(x), family)}
    return model.eval(), data


def test_batches(data: dict, batch: int):
    """``(inputs, labels)`` batches of the test split of :func:`trained_mlp` or :func:`trained_cnn`'s data."""
    if "test_x" in data:
        x = data["test_x"]
        for s in range(0, len(x), batch):
            yield x[s:s + batch], data["test_y"][s:s + batch]
    else:
        x = data["test_x_uint8"]
        for s in range(0, len(x), batch):
            yield data["norm"](x[s:s + batch]), data["test_y"][s:s + batch]
