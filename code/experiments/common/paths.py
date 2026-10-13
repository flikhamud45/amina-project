"""Where the stored models, records and caches live (all below ``code/``)."""

from __future__ import annotations

from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
"""The ``code/`` folder."""
ARTIFACTS = CODE / "artifacts"
MNIST_MODELS = ARTIFACTS / "models"
"""The trained MNIST networks of the attack (``attack/train_mnist.py``)."""
CNN_MODELS = ARTIFACTS / "models" / "cnn"
"""The trained image models of the benchmark (``costs/train_cnn.py``; not stored in the repository)."""
RESULTS = ARTIFACTS / "results"
"""The stored results of the attack, quality, generation and V1 experiments."""
BENCH = ARTIFACTS / "comparison"
"""The benchmark's records (``raw_<platform>/``), its tables (``tables_<platform>/``) and the literature values."""
