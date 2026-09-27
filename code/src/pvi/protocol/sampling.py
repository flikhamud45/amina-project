"""Path sampling for ``RandPathTest``.

Figure 3 of Anchuri et al.:

    * Sample a node ``j_d`` in the output layer.
    * For ``l = d-1, ..., 1``: read the claimed activations of ``G_{j_l}`` and the
      corresponding weights, then sample ``j_{l-1}`` uniformly from ``G_{j_l}``.

:class:`UniformPathSampler` implements exactly that.  The sampler is behind an
interface because the paper itself raises the alternative -- "adaptive sampling
strategies that prioritize layers or nodes where activations are statistically
more sensitive to tampering" (Section 5.2) -- which the defence study
instantiates.  Keeping the interface here means an alternative sampler drops in
without touching the prover or the verifier.

Determinism
-----------
Prover and verifier must derive *the same* path from the public challenge
``rho``.  :func:`challenge_rng` fixes that derivation: the challenge is hashed and
the digest seeds a NumPy ``Generator``.  NumPy's PCG64 bit generator is
version-stable, so the derivation is reproducible across machines and runs.
"""

from __future__ import annotations

import abc
import hashlib
import secrets
from dataclasses import dataclass

import numpy as np

from pvi.nn.architecture import Architecture
from pvi.nn.network import Trace

__all__ = [
    "Path",
    "PathSampler",
    "UniformPathSampler",
    "challenge_rng",
    "sample_challenge",
]


# --------------------------------------------------------------------------- #
# Challenges
# --------------------------------------------------------------------------- #


def sample_challenge(n_bytes: int = 32) -> bytes:
    """Draw a fresh verifier challenge ``rho``."""
    return secrets.token_bytes(n_bytes)


def challenge_rng(challenge: bytes) -> np.random.Generator:
    """Derive the shared randomness both parties use to reconstruct the paths."""
    seed = np.frombuffer(hashlib.sha256(challenge).digest(), dtype=np.uint32)
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Path:
    """One output-to-input path.

    ``nodes[l]`` is the neuron visited in layer ``l``, with ``nodes[0]`` in the
    input layer and ``nodes[-1]`` in the output layer.  The verifier checks local
    consistency at ``nodes[l]`` for every ``l >= 1`` and anchors ``nodes[0]``
    against the query.
    """

    nodes: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, layer: int) -> int:
        return self.nodes[layer]

    @property
    def checked_layers(self) -> range:
        """Layers at which a local consistency check is performed."""
        return range(1, len(self.nodes))


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #


class PathSampler(abc.ABC):
    """Strategy for choosing which nodes the verifier will check.

    A sampler is fully described by two distributions:

    * where the walk starts, over the output layer, and
    * where it goes next, over the current node's parent set.

    Subclasses supply those; :meth:`sample` and the exact analysis in
    ``pvi.experiments.analysis`` are both written once against this interface, so
    any new strategy is automatically analysable in closed form rather than only by
    Monte Carlo.

    ``trace`` is threaded through both methods because the alternatives the paper
    raises -- "adaptive sampling strategies that prioritize layers or nodes where
    activations are statistically more sensitive to tampering" (Section 5.2) --
    would use it.  Note what that means: such a sampler lets the *prover*, who
    authored the trace, influence where it will be inspected.
    """

    transition_depends_on_source: bool = True
    """Whether :meth:`transition_distribution` varies with the *source* node.

    When it does not -- as for uniform, static-importance and saliency weighting,
    all of which score the layer below without reference to which node above is
    asking -- the exact analysis can evaluate a whole dense layer with one dot
    product instead of one per neuron.  Declaring it correctly is purely a
    performance matter; the default is the safe one.
    """

    @abc.abstractmethod
    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        """Probability of starting the walk at each output neuron."""

    @abc.abstractmethod
    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        """Probability of stepping to each parent of ``neuron`` in ``layer_index``."""

    def sample(
        self,
        architecture: Architecture,
        rng: np.random.Generator,
        *,
        trace: Trace | None = None,
    ) -> Path:
        """Draw one path from the output layer down to the input layer."""
        depth = len(architecture) - 1
        nodes: list[int] = [0] * (depth + 1)
        start = _as_distribution(self.start_distribution(architecture, trace))
        nodes[depth] = int(rng.choice(len(start), p=start))

        for layer_index in range(depth, 0, -1):
            parent_idx, _ = architecture[layer_index].parents(nodes[layer_index])
            weights = _as_distribution(
                self.transition_distribution(
                    architecture, layer_index, nodes[layer_index], parent_idx, trace
                )
            )
            nodes[layer_index - 1] = int(parent_idx[rng.choice(len(parent_idx), p=weights)])

        return Path(tuple(nodes))

    def sample_many(
        self,
        architecture: Architecture,
        rng: np.random.Generator,
        count: int,
        *,
        trace: Trace | None = None,
    ) -> tuple[Path, ...]:
        return tuple(self.sample(architecture, rng, trace=trace) for _ in range(count))


def _as_distribution(weights: np.ndarray) -> np.ndarray:
    """Normalise non-negative weights, falling back to uniform if they all vanish."""
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("sampling weights must be finite and non-negative")
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


class UniformPathSampler(PathSampler):
    """``RandPathTest`` as specified in Figure 3.

    The output neuron is uniform over the output layer and each subsequent step is
    uniform over the current node's parent set.  For dense layers the parent set is
    the entire layer below, so every visited neuron is uniform over its layer; for
    convolutions the walk is confined to a receptive field and the induced marginal
    over neurons is non-uniform.

    Both distributions ignore the trace.  That independence from prover-supplied data
    is a security property, not an oversight -- and the minimax theorem of the
    report (Theorem 3.1) shows it is in fact optimal.
    """

    transition_depends_on_source = False

    def start_distribution(
        self, architecture: Architecture, trace: Trace | None
    ) -> np.ndarray:
        n = architecture.output_layer.n_neurons
        return np.full(n, 1.0 / n)

    def transition_distribution(
        self,
        architecture: Architecture,
        layer_index: int,
        neuron: int,
        parent_indices: np.ndarray,
        trace: Trace | None,
    ) -> np.ndarray:
        n = len(parent_indices)
        return np.full(n, 1.0 / n)
