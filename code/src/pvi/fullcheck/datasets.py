"""Datasets for the comparison benchmark, all read from local copies.

* MNIST     -- the repository's ``code/data`` copy (as used by Steps 1-4).
* CIFAR-10  -- ``$PVI_CIFAR_ROOT`` (default: the lab copy on the TAU cluster).
* ImageNet binary subsets reproducing Anchuri et al.'s classifiers:
  ``dogs_cats`` (their ``M``) and ``dogs_squirrels`` (their ``M~``), built from
  ``$PVI_IMAGENET_ROOT`` (default: the lab copy).  That copy stores class
  ``i`` of the standard sorted-synset order in folder ``i + 1``; dogs are
  classes 151-268, cats 281-285 and the fox squirrel 335 (checked by eye).
  The paper used Kaggle's Animals-10; ImageNet's dog/cat/squirrel classes give
  the same task without an extra download.

Images are decoded once and cached as uint8 tensors (256x256 for ImageNet,
from which training takes random 224 crops and evaluation the centre crop).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

__all__ = ["load_dataset", "normalise", "DATA_ROOT", "CACHE_ROOT"]

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
CACHE_ROOT = Path(os.environ.get("PVI_CACHE", Path(__file__).resolve().parents[3] / "artifacts" / "fullcheck" / "cache"))
CIFAR_ROOT = Path(os.environ.get("PVI_CIFAR_ROOT", "/home/sharifm/datasets/images/cifar10"))
IMAGENET_ROOT = Path(os.environ.get("PVI_IMAGENET_ROOT", "/home/sharifm/datasets/images/imagenet"))

_MEAN = {"mnist": (0.1307,), "cifar10": (0.4914, 0.4822, 0.4465), "imagenet": (0.485, 0.456, 0.406)}
_STD = {"mnist": (0.3081,), "cifar10": (0.2470, 0.2435, 0.2616), "imagenet": (0.229, 0.224, 0.225)}

DOG_CLASSES = list(range(151, 269))
CAT_CLASSES = list(range(281, 286))
SQUIRREL_CLASSES = [335]


def normalise(x_uint8: torch.Tensor, family: str) -> torch.Tensor:
    mean = torch.tensor(_MEAN[family], device=x_uint8.device)[None, :, None, None]
    std = torch.tensor(_STD[family], device=x_uint8.device)[None, :, None, None]
    return (x_uint8.float() / 255.0 - mean) / std


def _mnist():
    from torchvision import datasets

    tr = datasets.MNIST(str(DATA_ROOT), train=True, download=False)
    te = datasets.MNIST(str(DATA_ROOT), train=False, download=False)
    return (tr.data[:, None].clone(), tr.targets.clone(), te.data[:, None].clone(), te.targets.clone())


def _cifar():
    from torchvision import datasets

    tr = datasets.CIFAR10(str(CIFAR_ROOT), train=True, download=False)
    te = datasets.CIFAR10(str(CIFAR_ROOT), train=False, download=False)
    f = lambda d: torch.from_numpy(np.ascontiguousarray(d.data.transpose(0, 3, 1, 2)))
    return f(tr), torch.tensor(tr.targets), f(te), torch.tensor(te.targets)


def _decode(path: str, size: int = 256) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()


def _files(split: str, classes: list[int]) -> list[str]:
    out = []
    for c in classes:
        d = IMAGENET_ROOT / split / str(c + 1)
        out += sorted(str(p) for p in d.iterdir())
    return out


def _imagenet_binary(name: str, workers: int = 16):
    cache = CACHE_ROOT / f"{name}.pt"
    if cache.exists():
        return tuple(torch.load(cache))
    rng = np.random.default_rng(0)
    other = CAT_CLASSES if name == "dogs_cats" else SQUIRREL_CLASSES
    data = []
    for split in ("train", "val"):
        pos = _files(split, other)
        dogs = _files(split, DOG_CLASSES)
        dogs = [dogs[i] for i in sorted(rng.choice(len(dogs), size=len(pos), replace=False))]
        files = dogs + pos
        labels = [0] * len(dogs) + [1] * len(pos)  # 0 = dog, 1 = cat / squirrel
        with ProcessPoolExecutor(workers) as ex:
            imgs = list(ex.map(_decode, files, chunksize=32))
        data += [torch.from_numpy(np.stack(imgs)), torch.tensor(labels)]
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save(data, tmp)
    os.replace(tmp, cache)
    return tuple(data)


def load_dataset(name: str):
    """``(train_x uint8, train_y, test_x uint8, test_y, family, n_classes)``."""
    if name == "mnist":
        return (*_mnist(), "mnist", 10)
    if name == "cifar10":
        return (*_cifar(), "cifar10", 10)
    if name in ("imagenet_dogs_cats", "imagenet_dogs_squirrels"):
        return (*_imagenet_binary(name.removeprefix("imagenet_")), "imagenet", 2)
    raise ValueError(f"unknown dataset {name!r}")
