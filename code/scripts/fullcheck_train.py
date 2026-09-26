"""Train the float models of the comparison benchmark (one per call).

    python scripts/fullcheck_train.py --model vgg16 --device cuda

Writes ``artifacts/fullcheck/models/<model>.pt`` (state dict) and ``<model>.json``
(accuracy, epochs, seed, wall time).  Architectures and datasets are those the
literature benchmarks; see ``pvi/fullcheck/models.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pvi.fullcheck.datasets import load_dataset, normalise
from pvi.fullcheck.models import MODEL_SPECS, build_float_model
from pvi.results import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "fullcheck" / "models"

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(RECIPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    kwargs, dataset, _ = MODEL_SPECS[args.model]
    tx, ty, vx, vy, family, n_classes = load_dataset(dataset)
    r = dict(RECIPES[args.model])
    if args.epochs:
        r["epochs"] = args.epochs
    device = torch.device(args.device)
    model = build_float_model(kwargs["kind"], n_classes).to(device)
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
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / f"{args.model}.pt")
    write_json(OUT / f"{args.model}.json", {
        "model": args.model, "dataset": dataset, "test_accuracy": acc, "n_train": len(tx), "n_test": len(vx),
        "recipe": r, "seed": args.seed, "train_seconds": time.time() - start,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    })
    print(json.dumps({"model": args.model, "test_accuracy": acc}))


if __name__ == "__main__":
    main()
