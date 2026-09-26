"""Float reference architectures matching the models the literature benchmarks.

* ``lenet5``      -- LeNet-5 on MNIST (ReLU + max-pool), as in zkCNN's
                     ``lenet5.mnist.relu.max`` and vCNN / ZEN / Bionetta.
* ``vgg11``/``vgg16`` -- VGG on CIFAR-10 (8 / 13 conv layers + 3 FC), as in zkCNN,
                     ZKML, ZENO, zkPyTorch and Artemis.
* ``resnet18_cifar`` -- the CIFAR-10 ResNet-18 (3x3 stem), as in ZKML, ZENO,
                     Bionetta, EZKL and Artemis.
* ``resnet18_224``   -- torchvision's ImageNet-shaped ResNet-18 (7x7 stem, 224x224),
                     with a 2-way head: Anchuri et al.'s dogs-vs-cats classifier.

BatchNorm is used for training and folded into the preceding convolution
before quantisation, which is also what zkCNN and ZKML do.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["LeNet5", "VGG", "BasicBlock", "ResNet18", "build_float_model", "MODEL_SPECS"]


class LeNet5(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.pool = nn.MaxPool2d(2)

    def features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.fc1(x.flatten(1)))
        return torch.relu(self.fc2(x))

    def forward(self, x):
        return self.fc3(self.features(x))

    @property
    def head(self) -> nn.Linear:
        return self.fc3


_VGG_CFG = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
              512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, depth: str = "vgg16", n_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = 3
        for v in _VGG_CFG[depth]:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.body = nn.Sequential(*layers)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)
        self.drop = nn.Dropout(0.3)

    def features(self, x):
        x = self.body(x).flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.drop(torch.relu(self.fc2(x)))

    def forward(self, x):
        return self.fc3(self.features(x))

    @property
    def head(self) -> nn.Linear:
        return self.fc3


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + (x if self.down is None else self.down(x)))


class ResNet18(nn.Module):
    def __init__(self, n_classes: int = 10, imagenet_stem: bool = False) -> None:
        super().__init__()
        self.imagenet_stem = imagenet_stem
        if imagenet_stem:
            self.stem = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        else:
            self.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1) if imagenet_stem else None
        blocks, c = [], 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
            blocks.append(BasicBlock(c, cout, stride))
            c = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, n_classes)

    def features(self, x):
        x = torch.relu(self.bn(self.stem(x)))
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.features(x))

    @property
    def head(self) -> nn.Linear:
        return self.fc


MODEL_SPECS = {
    # name: (constructor kwargs, dataset, input shape)
    "lenet5": (dict(kind="lenet5"), "mnist", (1, 28, 28)),
    "vgg11": (dict(kind="vgg11"), "cifar10", (3, 32, 32)),
    "vgg16": (dict(kind="vgg16"), "cifar10", (3, 32, 32)),
    "resnet18_cifar": (dict(kind="resnet18_cifar"), "cifar10", (3, 32, 32)),
    "resnet18_224": (dict(kind="resnet18_224"), "imagenet_dogs_cats", (3, 224, 224)),
    "resnet18_224_squirrel": (dict(kind="resnet18_224"), "imagenet_dogs_squirrels", (3, 224, 224)),
}


def build_float_model(kind: str, n_classes: int) -> nn.Module:
    if kind == "lenet5":
        return LeNet5(n_classes)
    if kind in ("vgg11", "vgg16"):
        return VGG(kind, n_classes)
    if kind == "resnet18_cifar":
        return ResNet18(n_classes, imagenet_stem=False)
    if kind == "resnet18_224":
        return ResNet18(n_classes, imagenet_stem=True)
    raise ValueError(f"unknown model kind {kind!r}")
