"""Train the MNIST models of the reproduction, the MNIST attacks and the sampler experiments (Sec. 3.5).

The seven models of ``pvi.mnist.zoo`` (architecture, data slice and training settings of each): the MLP
``mlp_mnist_full`` (784-512-256-10) that the attacks, the samplers and the adaptive game use, and the committed and
substitute models of the four reproduction settings (``mlp_digits_01234``/``mlp_digits_05678``,
``mlp_mnist_halfA``/``mlp_mnist_halfB``, ``cnn_mnist_halfA``/``cnn_mnist_halfB``).  The weights the paper used are
stored in ``artifacts/models/`` (hashes in ``MODELS.sha256``), so this script is needed only to retrain them.

Usage (from ``code/``)::

    python -m experiments.attack.train_mnist                       # skips the stored models: does nothing
    python -m experiments.attack.train_mnist --force               # retrains all seven
    python -m experiments.attack.train_mnist --force --only mlp_mnist_full

Writes ``<out>/<name>.npz`` and, beside it, ``<name>.json`` (train and test accuracy, final loss, epochs, seed).
MNIST is downloaded by torchvision on first use.  CPU only, about 2 min for all seven on 8 threads.  Training is
seeded, but it is bit-exact only on one machine and library build: elsewhere the weights differ slightly (with the
same accuracy) and the MNIST numbers of the paper move in their last digits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.common.paths import MNIST_MODELS
from pvi.mnist.training import save_network, train_network
from pvi.mnist.zoo import all_model_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="retrain models that are already stored")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME", help="train only these models")
    parser.add_argument("--out", default=str(MNIST_MODELS), help="folder for the weights and training reports")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = all_model_specs()
    if args.only:
        specs = tuple(spec for spec in specs if spec.name in set(args.only))
        if not specs:
            raise SystemExit(f"no models matched {sorted(args.only)}")

    for spec in specs:
        path = out / f"{spec.name}.npz"
        if path.exists() and not args.force:
            print(f"[skip] {spec.name} (stored at {path})")
            continue
        print(f"[train] {spec.name}")
        dataset = spec.dataset()
        print(f"  {dataset.summary()}")
        started = time.perf_counter()
        network, report = train_network(spec.architecture(), dataset, spec.train_config, verbose=True)
        save_network(network, path, report)
        print(f"  saved to {path} in {time.perf_counter() - started:.1f}s\n")
    print("done.")


if __name__ == "__main__":
    main()
