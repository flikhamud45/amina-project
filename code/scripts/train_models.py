"""Train every model the experiments need and cache the weights.

Usage
-----
    python scripts/train_models.py [--force] [--only NAME ...]

Weights land in ``artifacts/models/<name>.npz`` with the training report beside
them.  Runs are seeded, so re-running reproduces the same models bit for bit.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from pvi.training import save_network, train_network
from pvi.zoo import all_model_specs

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain even if cached")
    parser.add_argument("--only", nargs="*", default=None, help="train only these models")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"],
        help="training device; 'auto' picks CUDA when available. GPU training is "
        "best-effort deterministic only -- see TrainConfig's docstring. Every "
        "cached artefact in this repo was trained on 'cpu' (the default), so "
        "requesting 'cuda'/'auto' will not reproduce them bit-for-bit.",
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    specs = all_model_specs()
    if args.only:
        wanted = set(args.only)
        specs = tuple(spec for spec in specs if spec.name in wanted)
        if not specs:
            raise SystemExit(f"no models matched {sorted(wanted)}")

    for spec in specs:
        path = ARTIFACTS / f"{spec.name}.npz"
        if path.exists() and not args.force:
            print(f"[skip] {spec.name} (cached at {path.name})")
            continue

        print(f"[train] {spec.name}")
        dataset = spec.dataset()
        print(f"  {dataset.summary()}")
        train_config = dataclasses.replace(spec.train_config, device=args.device)
        started = time.perf_counter()
        network, report = train_network(
            spec.architecture(), dataset, train_config, verbose=True
        )
        save_network(network, path, report)
        print(f"  saved to {path.name} in {time.perf_counter() - started:.1f}s\n")

    print("done.")


if __name__ == "__main__":
    main()
