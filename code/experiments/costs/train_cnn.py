"""Train the float image models of the benchmark (one per call): the weights of ``bench.py``'s image cells.

The image models of ``sec:break`` and ``tab:eval-compare`` (LeNet-5 on MNIST, VGG-11/16 and ResNet-18 on CIFAR-10, the
224-pixel dog-versus-cat ResNet-18 and its dog-versus-squirrel substitute model) are those the literature benchmarks
(:mod:`pvi.models.cnn`); their datasets are read by :func:`pvi.models.datasets.load_dataset`.

    python -m experiments.costs.train_cnn --model lenet5 --device cuda
    python -m experiments.costs.train_cnn --model resnet18_224 --device cuda

Writes ``artifacts/models/cnn/<model>.pt`` (the state dict; :data:`experiments.common.paths.CNN_MODELS`) and
``<model>.json`` (``model``, ``dataset``, ``test_accuracy``, ``n_train``, ``n_test``, ``recipe``, ``seed``,
``train_seconds``, ``device``, ``env``).  Retrained weights are equally accurate but not bit-identical to the stored
runs'; the costs depend on the shapes only.  Time on one GPU: minutes for LeNet-5, a few hours for the 224-pixel
ResNets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.common import records
from experiments.common.paths import CNN_MODELS
from pvi.models.cnn import MODEL_SPECS, build_float_model
from pvi.models.datasets import load_dataset, normalise

SEED = 0   # the seed of the stored weights

RECIPES = {
    "lenet5": dict(epochs=10, lr=0.05, batch=128, wd=5e-4),
    "vgg11": dict(epochs=40, lr=0.05, batch=128, wd=5e-4),
    "vgg16": dict(epochs=50, lr=0.05, batch=128, wd=5e-4),
    "resnet18_cifar": dict(epochs=40, lr=0.1, batch=128, wd=5e-4),
    "resnet18_224": dict(epochs=40, lr=0.1, batch=64, wd=5e-4),
    "resnet18_224_squirrel": dict(epochs=60, lr=0.1, batch=64, wd=5e-4),
}


def augment(x: torch.Tensor, family: str, train: bool) -> torch.Tensor:
    """uint8 batch -> normalised float batch, with the usual light augmentation."""
    if family == "imagenet":
        if train:
            i, j = torch.randint(0, 33, (2,)).tolist()
            x = x[:, :, i:i + 224, j:j + 224]
        else:
            x = x[:, :, 16:240, 16:240]
    elif family == "cifar10" and train:
        x = F.pad(x, (4, 4, 4, 4))
        i, j = torch.randint(0, 9, (2,)).tolist()
        x = x[:, :, i:i + 32, j:j + 32]
    if train and family != "mnist" and torch.rand(()) < 0.5:
        x = x.flip(3)
    return normalise(x, family)


@torch.no_grad()
def evaluate(model, x, y, family, device, batch=256) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(x), batch):
        xb = augment(x[s:s + batch].to(device), family, False)
        correct += int((model(xb).argmax(1).cpu() == y[s:s + batch]).sum())
    return correct / len(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(RECIPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED, help="torch's seed (initialisation, batches, augmentation)")
    ap.add_argument("--epochs", type=int, default=None, help="the recipe's epochs (default: RECIPES)")
    ap.add_argument("--out", default=str(CNN_MODELS), help="the folder of <model>.pt and <model>.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    kind, dataset = MODEL_SPECS[args.model]
    tx, ty, vx, vy, family, n_classes = load_dataset(dataset)
    r = dict(RECIPES[args.model], **({"epochs": args.epochs} if args.epochs else {}))
    device = torch.device(args.device)
    model = build_float_model(kind, n_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=r["lr"], momentum=0.9, weight_decay=r["wd"], nesterov=True)
    steps = r["epochs"] * ((len(tx) + r["batch"] - 1) // r["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=r["lr"], total_steps=steps, pct_start=0.15)
    start = time.time()
    for epoch in range(r["epochs"]):
        model.train()
        perm = torch.randperm(len(tx))
        total = 0.0
        for s in range(0, len(tx), r["batch"]):
            idx = perm[s:s + r["batch"]]
            xb = augment(tx[idx].to(device, non_blocking=True), family, True)
            yb = ty[idx].to(device)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        if epoch % 5 == 4 or epoch == r["epochs"] - 1:
            acc = evaluate(model, vx, vy, family, device)
            print(f"epoch {epoch + 1}/{r['epochs']} loss {total / len(tx):.4f} test acc {acc:.4f} "
                  f"({time.time() - start:.0f}s)", flush=True)
    acc = evaluate(model, vx, vy, family, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{args.model}.pt")
    records.write_json(out / f"{args.model}.json", {
        "model": args.model, "dataset": dataset, "test_accuracy": acc, "n_train": len(tx), "n_test": len(vx),
        "recipe": r, "seed": args.seed, "train_seconds": time.time() - start,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "env": records.environment(device),
    })
    print(json.dumps({"model": args.model, "test_accuracy": acc}))


if __name__ == "__main__":
    main()
