"""The models and substitution setups used throughout the study.

Anchuri et al. validate other-model soundness by pairing a committed model ``M``
with a substitute ``M~`` and asking whether ``RandPathTest`` can tell them apart.
Which pair you choose matters a great deal, so we reproduce all three of the
settings the paper uses, on MNIST-scale models that a CPU can train in minutes:

``shared_class``
    Section 6.2.  ``M`` and ``M~`` solve *different* classification tasks that
    share exactly one class, and separation is measured on inputs from the shared
    class.  The paper's instance is dogs-vs-cats against dogs-vs-squirrels,
    evaluated on dogs; ours is digits ``{0,1,2,3,4}`` against ``{0,5,6,7,8}``,
    evaluated on zeros.

``disjoint_data``
    Appendix G.1.  ``M`` and ``M~`` solve the *same* task but are trained on
    disjoint halves of the data.  This is the harder case -- the models are
    functionally near-identical -- and the paper reports *more* separation here,
    which is a claim worth re-testing.

``quantised``
    Section A's motivating scenario: a provider that commits to ``M`` but serves a
    cheaper variant.  We take ``M`` itself and quantise its weights to 8 bits.
    This is the most functionally similar substitute of the three and therefore
    the sternest test of the protocol.

A fourth entry, ``cnn_disjoint_data``, repeats the disjoint-data setting on a
convolutional network so that the local, weight-sharing parent sets of
convolutions are exercised as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pvi.data import Dataset, load_classification
from pvi.nn.architecture import Architecture
from pvi.nn.models import mlp_architecture, small_cnn_architecture
from pvi.nn.network import TracedNetwork
from pvi.training import TrainConfig

__all__ = [
    "LARGE_MLP_SPEC",
    "ModelSpec",
    "SubstitutionSetup",
    "SETUPS",
    "cnn_architecture_for",
    "mlp_architecture_for",
    "mlp_architecture_large_for",
    "quantise_network",
]

MNIST_SHAPE = (1, 28, 28)
MNIST_FEATURES = 784


def mlp_architecture_for(n_classes: int) -> Architecture:
    """The dense workhorse: ``784 -> 512 -> 256 -> n_classes``.

    Widths are chosen to be clearly distinct so that "which layer was tampered"
    and "how wide is that layer" cannot be confused in the results.
    """
    return mlp_architecture(MNIST_FEATURES, [512, 256], n_classes)


def mlp_architecture_large_for(n_classes: int) -> Architecture:
    """A GPU-scale dense network: ``784 -> 4096 -> 2048 -> n_classes``.

    8x the width of :func:`mlp_architecture_for` at both hidden layers. Every
    Revision-1 theorem and measurement (Theorems 1-4, the minimax bounds, the
    zero blind spot, the sumcheck prototype) is stated and computed in terms of
    layer width `N`, never assuming MNIST scale specifically -- this exists so
    that claim is actually checked at a size CPU training makes impractical.
    CPU-trainable in principle but slow; intended for ``TrainConfig(device="cuda")``
    or ``"auto"``. Not part of :func:`all_model_specs`'s default CPU pipeline --
    see :data:`LARGE_MLP_SPEC` and ``scripts/run_gpu_scale_check.py``.
    """
    return mlp_architecture(MNIST_FEATURES, [4096, 2048], n_classes)


def cnn_architecture_for(n_classes: int) -> Architecture:
    """A compact convolutional classifier for MNIST."""
    return small_cnn_architecture(
        MNIST_SHAPE, n_classes, channels=(16, 32), hidden_width=128
    )


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to reproduce one trained model."""

    name: str
    architecture_fn: Callable[[], Architecture]
    dataset_kwargs: dict
    train_config: TrainConfig = field(default_factory=TrainConfig)

    def architecture(self) -> Architecture:
        return self.architecture_fn()

    def dataset(self) -> Dataset:
        return load_classification(**self.dataset_kwargs)


@dataclass(frozen=True)
class SubstitutionSetup:
    """A committed model, a substitute, and the queries to compare them on."""

    key: str
    title: str
    paper_reference: str
    honest: ModelSpec
    substitute: ModelSpec | None
    quantise_bits: int | None = None
    eval_dataset_kwargs: dict = field(default_factory=dict)

    def eval_dataset(self) -> Dataset:
        return load_classification(**(self.eval_dataset_kwargs or self.honest.dataset_kwargs))

    def is_derived(self) -> bool:
        """True when the substitute is obtained by transforming ``M`` itself."""
        return self.substitute is None and self.quantise_bits is not None


def quantise_network(network: TracedNetwork, bits: int = 8) -> TracedNetwork:
    """Symmetric per-tensor uniform quantisation of the weights.

    Models a provider serving a cheaper variant of the committed model.  Biases
    are left alone, as they are in most deployment quantisation schemes, and the
    result is deliberately *very* close to ``M`` in function -- which is what makes
    it a demanding substitute for the protocol to detect.
    """
    if bits < 2:
        raise ValueError(f"need at least 2 bits, got {bits}")
    levels = 2 ** (bits - 1) - 1
    parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (weight, bias) in network.parameters.items():
        peak = float(np.max(np.abs(weight)))
        if peak == 0.0:
            parameters[name] = (weight.copy(), bias.copy())
            continue
        scale = peak / levels
        quantised = np.round(weight / scale) * scale
        parameters[name] = (quantised.astype(np.float32), bias.copy())
    return TracedNetwork(network.architecture, parameters)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

_STANDARD = TrainConfig(epochs=6, batch_size=128, learning_rate=1e-3, seed=0)
_STANDARD_ALT = TrainConfig(epochs=6, batch_size=128, learning_rate=1e-3, seed=1)

_MLP_SHARED_A = ModelSpec(
    name="mlp_digits_01234",
    architecture_fn=lambda: mlp_architecture_for(5),
    dataset_kwargs={"name": "mnist", "classes": (0, 1, 2, 3, 4), "seed": 0},
    train_config=_STANDARD,
)
_MLP_SHARED_B = ModelSpec(
    name="mlp_digits_05678",
    architecture_fn=lambda: mlp_architecture_for(5),
    dataset_kwargs={"name": "mnist", "classes": (0, 5, 6, 7, 8), "seed": 0},
    train_config=_STANDARD_ALT,
)

_MLP_HALF_A = ModelSpec(
    name="mlp_mnist_halfA",
    architecture_fn=lambda: mlp_architecture_for(10),
    dataset_kwargs={"name": "mnist", "train_fraction": (0.0, 0.5), "seed": 0},
    train_config=_STANDARD,
)
_MLP_HALF_B = ModelSpec(
    name="mlp_mnist_halfB",
    architecture_fn=lambda: mlp_architecture_for(10),
    dataset_kwargs={"name": "mnist", "train_fraction": (0.5, 1.0), "seed": 0},
    train_config=_STANDARD_ALT,
)

_MLP_FULL = ModelSpec(
    name="mlp_mnist_full",
    architecture_fn=lambda: mlp_architecture_for(10),
    dataset_kwargs={"name": "mnist", "seed": 0},
    train_config=_STANDARD,
)

LARGE_MLP_SPEC = ModelSpec(
    name="mlp_mnist_large",
    architecture_fn=lambda: mlp_architecture_large_for(10),
    dataset_kwargs={"name": "mnist", "seed": 0},
    train_config=TrainConfig(
        epochs=6, batch_size=256, learning_rate=1e-3, seed=0, device="auto"
    ),
)
"""GPU-scale companion to ``_MLP_FULL``, same data and seed, 8x the hidden width.

Deliberately excluded from :data:`SETUPS` / :func:`all_model_specs` so the
default CPU pipeline (``scripts/train_models.py`` with no flags,
``run_step1.py``, ``run_attack.py``, ``run_defence.py``) is entirely
unaffected. Train it explicitly with
``python scripts/train_models.py --only mlp_mnist_large --device auto``, or see
``scripts/run_gpu_scale_check.py``, which trains it (if not cached) and
re-checks Revision 1's key findings (Theorem 4's sweep, the sumcheck
prototype) at this width.
"""

_CNN_HALF_A = ModelSpec(
    name="cnn_mnist_halfA",
    architecture_fn=lambda: cnn_architecture_for(10),
    dataset_kwargs={"name": "mnist", "train_fraction": (0.0, 0.5), "seed": 0},
    train_config=TrainConfig(epochs=4, batch_size=128, learning_rate=1e-3, seed=0),
)
_CNN_HALF_B = ModelSpec(
    name="cnn_mnist_halfB",
    architecture_fn=lambda: cnn_architecture_for(10),
    dataset_kwargs={"name": "mnist", "train_fraction": (0.5, 1.0), "seed": 0},
    train_config=TrainConfig(epochs=4, batch_size=128, learning_rate=1e-3, seed=1),
)


SETUPS: tuple[SubstitutionSetup, ...] = (
    SubstitutionSetup(
        key="shared_class",
        title="Different tasks with one shared class (MLP)",
        paper_reference="Section 6.2",
        honest=_MLP_SHARED_A,
        substitute=_MLP_SHARED_B,
        # Queries come from the shared class only, mirroring the paper's use of
        # dog images for a dogs-vs-cats / dogs-vs-squirrels comparison.
        eval_dataset_kwargs={"name": "mnist", "classes": (0,), "seed": 0},
    ),
    SubstitutionSetup(
        key="disjoint_data",
        title="Same task, disjoint training data (MLP)",
        paper_reference="Appendix G.1",
        honest=_MLP_HALF_A,
        substitute=_MLP_HALF_B,
        eval_dataset_kwargs={"name": "mnist", "seed": 0},
    ),
    SubstitutionSetup(
        key="quantised",
        title="8-bit quantised substitute of the committed model (MLP)",
        paper_reference="Section A (economic substitution)",
        honest=_MLP_FULL,
        substitute=None,
        quantise_bits=8,
        eval_dataset_kwargs={"name": "mnist", "seed": 0},
    ),
    SubstitutionSetup(
        key="cnn_disjoint_data",
        title="Same task, disjoint training data (CNN)",
        paper_reference="Appendix G.1, convolutional",
        honest=_CNN_HALF_A,
        substitute=_CNN_HALF_B,
        eval_dataset_kwargs={"name": "mnist", "seed": 0},
    ),
)


def all_model_specs() -> tuple[ModelSpec, ...]:
    """Every model that has to be trained, de-duplicated by name."""
    seen: dict[str, ModelSpec] = {}
    for setup in SETUPS:
        for spec in (setup.honest, setup.substitute):
            if spec is not None:
                seen.setdefault(spec.name, spec)
    return tuple(seen.values())
