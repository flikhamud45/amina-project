"""Datasets, as plain NumPy arrays.

The protocol code never sees a PyTorch tensor, so datasets are materialised once
into ``float32`` arrays in ``[0, 1]``.  Keeping pixels in ``[0, 1]`` rather than
standardising them matters later: the attack's trigger is a small bright patch,
and an un-shifted input scale makes "bright patch" mean the same thing to the
reader as it does to the network.

Subset selection supports the paper's model-separation setups.  Section 6.2 trains
``M`` and ``M~`` on *different* class sets sharing one class; Appendix G.1 trains
them on the *same* classes but disjoint data.  Both are reproducible here through
:func:`load_classification`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from torchvision import datasets

__all__ = ["Dataset", "DEFAULT_DATA_ROOT", "load_classification", "load_raw"]

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Dataset:
    """A materialised classification dataset."""

    name: str
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    input_shape: tuple[int, ...]
    n_classes: int

    def __post_init__(self) -> None:
        if self.train_x.dtype != np.float32 or self.test_x.dtype != np.float32:
            raise ValueError("inputs must be float32")
        if len(self.train_x) != len(self.train_y) or len(self.test_x) != len(self.test_y):
            raise ValueError("inputs and labels disagree in length")

    @property
    def n_features(self) -> int:
        return int(np.prod(self.input_shape))

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.train_x)} train / {len(self.test_x)} test, "
            f"shape {self.input_shape}, {self.n_classes} classes"
        )


def load_raw(
    name: str, root: Path | str = DEFAULT_DATA_ROOT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Download (once) and materialise a dataset as ``float32`` arrays in ``[0, 1]``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if name == "mnist":
        train = datasets.MNIST(str(root), train=True, download=True)
        test = datasets.MNIST(str(root), train=False, download=True)
        shape = (1, 28, 28)
    elif name == "fashion_mnist":
        train = datasets.FashionMNIST(str(root), train=True, download=True)
        test = datasets.FashionMNIST(str(root), train=False, download=True)
        shape = (1, 28, 28)
    elif name == "cifar10":
        train = datasets.CIFAR10(str(root), train=True, download=True)
        test = datasets.CIFAR10(str(root), train=False, download=True)
        shape = (3, 32, 32)
    else:
        raise ValueError(f"unknown dataset {name!r}")

    def to_arrays(split) -> tuple[np.ndarray, np.ndarray]:
        data = np.asarray(split.data)
        if data.ndim == 3:  # (N, H, W) grayscale
            data = data[:, None, :, :]
        else:  # (N, H, W, C) colour
            data = data.transpose(0, 3, 1, 2)
        x = np.ascontiguousarray(data, dtype=np.float32) / 255.0
        y = np.asarray(split.targets, dtype=np.int64)
        return x, y

    train_x, train_y = to_arrays(train)
    test_x, test_y = to_arrays(test)
    return train_x, train_y, test_x, test_y, shape


def load_classification(
    name: str = "mnist",
    *,
    classes: Sequence[int] | None = None,
    train_fraction: tuple[float, float] | None = None,
    max_train: int | None = None,
    seed: int = 0,
    root: Path | str = DEFAULT_DATA_ROOT,
) -> Dataset:
    """Load a dataset, optionally restricted to a class subset and a data slice.

    Parameters
    ----------
    classes:
        Keep only these labels, remapped to ``0..len(classes)-1`` in the order
        given.  This reproduces the paper's "shared class" construction: two models
        with the same architecture and output dimension whose training classes
        overlap in exactly one position.
    train_fraction:
        A ``(start, end)`` slice of the *shuffled* training set, so that two models
        can be trained on provably disjoint data (Appendix G.1's setup).
    max_train:
        Cap on training-set size, for quick runs.
    """
    train_x, train_y, test_x, test_y, shape = load_raw(name, root)

    if classes is not None:
        remap = {original: new for new, original in enumerate(classes)}

        def restrict(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            keep = np.isin(y, list(classes))
            x, y = x[keep], y[keep]
            return x, np.asarray([remap[int(label)] for label in y], dtype=np.int64)

        train_x, train_y = restrict(train_x, train_y)
        test_x, test_y = restrict(test_x, test_y)
        n_classes = len(classes)
    else:
        n_classes = int(train_y.max()) + 1

    order = np.random.default_rng(seed).permutation(len(train_x))
    train_x, train_y = train_x[order], train_y[order]

    if train_fraction is not None:
        start, end = train_fraction
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"invalid train_fraction {train_fraction}")
        lo, hi = int(start * len(train_x)), int(end * len(train_x))
        train_x, train_y = train_x[lo:hi], train_y[lo:hi]

    if max_train is not None:
        train_x, train_y = train_x[:max_train], train_y[:max_train]

    label = name
    if classes is not None:
        label += f"[{','.join(str(c) for c in classes)}]"
    if train_fraction is not None:
        label += f"({train_fraction[0]:.2f}-{train_fraction[1]:.2f})"

    return Dataset(
        name=label,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        input_shape=shape,
        n_classes=n_classes,
    )
