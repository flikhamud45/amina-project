"""Train the MNIST models of the reproduction and attack experiments.

Usage
-----
    python experiments/0_train_models/train.py [--force] [--only NAME ...]

Weights land in ``artifacts/models/<name>.npz`` with the training report beside
them.  Cached models are skipped, so with the shipped weights this does nothing.
Runs are seeded: on one machine and torch build a rerun gives the same weights, but
another machine gives slightly different (equally accurate) ones.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pvi.training import save_network, train_network
from pvi.zoo import all_model_specs

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain even if cached")
    parser.add_argument("--only", nargs="*", default=None, help="train only these models")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    specs = all_model_specs()
    if args.only:
        specs = tuple(spec for spec in specs if spec.name in set(args.only))
        if not specs:
            raise SystemExit(f"no models matched {sorted(args.only)}")

    for spec in specs:
        path = ARTIFACTS / f"{spec.name}.npz"
        if path.exists() and not args.force:
            print(f"[skip] {spec.name} (cached at {path.name})")
            continue

        print(f"[train] {spec.name}")
        dataset = spec.dataset()
        print(f"  {dataset.summary()}")
        started = time.perf_counter()
        network, report = train_network(spec.architecture(), dataset, spec.train_config, verbose=True)
        save_network(network, path, report)
        print(f"  saved to {path.name} in {time.perf_counter() - started:.1f}s\n")

    print("done.")


if __name__ == "__main__":
    main()
