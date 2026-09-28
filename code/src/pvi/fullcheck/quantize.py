"""Post-training int8 quantisation of float CNNs into :class:`IntGraph` models.

Standard static quantisation, chosen to be exactly reproducible by a verifier:

* weights: symmetric int8, one scale per output channel;
* activations: symmetric int8, one scale per tensor, calibrated as the 99.99th
  percentile of ``|a|`` on a calibration batch;
* bias: an integer below ``2**26`` at the product scale ``s_in * s_w[c]``;
* requantisation: ``clamp((z * m_c + 2**29) >> 30)`` with an integer multiplier
  ``m_c = round(2**30 * s_in * s_w[c] / s_out)`` -- the usual fixed-point rule,
  identical on every device;
* BatchNorm folded into the convolution before quantising.

The builder walks the float model once on the calibration batch, creating the
integer operations and their constants as it goes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .graph import INT8_MAX, CheapOp, IntGraph, MatOp, requant
from .models import LeNet5, ResNet18, VGG

__all__ = ["SHIFT", "CNNBuilder", "quantize_model", "quantize_input", "dequantize_logits", "fold_bn"]

SHIFT = 30
BIAS_MAX = 1 << 26
"""Integer biases are kept below ``2**26``: a channel whose folded bias would be
larger (BatchNorm with a tiny gain) gets a coarser weight scale instead.  With
|W x| <= 127 * 127 * K this keeps every honest claim well inside the protocol's
range check (``Z_BOUND = 2**29``; ``commit_graph`` verifies it per row)."""


def _pct(t: torch.Tensor, q: float = 0.9999) -> float:
    a = t.detach().abs().flatten().float()
    if a.numel() > 2_000_000:
        g = torch.Generator().manual_seed(0)
        a = a[torch.randint(0, a.numel(), (2_000_000,), generator=g)]
    return max(float(torch.quantile(a.cpu(), q)), 1e-8)


def _mult(ratio) -> torch.Tensor:
    return torch.round(torch.as_tensor(ratio, dtype=torch.float64) * (1 << SHIFT)).to(torch.int64)


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d | None):
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    if bn is not None:
        g = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
        w = w * g[:, None, None, None]
        b = (b - bn.running_mean.detach().double()) * g + bn.bias.detach().double()
    return w, b


@dataclass
class _T:
    """A tensor during building: its name, float calibration value and scale."""

    name: str
    f: torch.Tensor
    scale: torch.Tensor  # per-tensor (0-d) or per-channel (1-d), float64
    axis: int = 1


class CNNBuilder:
    def __init__(self, x_cal: torch.Tensor) -> None:
        self.ops: list = []
        self._n = 0
        s = _pct(x_cal) / INT8_MAX
        self.input_scale = s
        self.cur = _T("x", x_cal.double(), torch.tensor(s, dtype=torch.float64))

    def _name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # -- MatOps -----------------------------------------------------------------
    def conv(self, conv: nn.Conv2d, bn: nn.BatchNorm2d | None, t: _T, *, requant_out: bool = True) -> _T:
        w, b = fold_bn(conv, bn)
        sw = w.abs().amax(dim=(1, 2, 3)) / INT8_MAX
        sw = torch.maximum(sw, b.abs() / (t.scale * BIAS_MAX)).clamp_min(1e-12)
        wq = torch.round(w / sw[:, None, None, None]).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        s_z = t.scale * sw
        bq = torch.round(b / s_z).to(torch.int64)
        name = self._name("conv")
        k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        self.ops.append(MatOp(name, (t.name,), name + ".z", weight=wq.reshape(wq.shape[0], -1),
                              bias=bq, layout="conv", conv=(k, s, p)))
        z_f = F.conv2d(t.f, w, b, stride=s, padding=p)
        zt = _T(name + ".z", z_f, s_z, axis=1)
        return self.requant(zt) if requant_out else zt

    def linear(self, weight: torch.Tensor, bias: torch.Tensor | None, t: _T, *, final: bool = False) -> _T:
        w = weight.detach().double()
        b = bias.detach().double() if bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
        sw = w.abs().amax(dim=1) / INT8_MAX
        sw = torch.maximum(sw, b.abs() / (t.scale * BIAS_MAX)).clamp_min(1e-12)
        wq = torch.round(w / sw[:, None]).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        s_z = t.scale * sw
        bq = torch.round(b / s_z).to(torch.int64)
        name = self._name("fc")
        self.ops.append(MatOp(name, (t.name,), name + ".z", weight=wq, bias=bq, layout="linear"))
        z_f = t.f @ w.T + b
        zt = _T(name + ".z", z_f, s_z, axis=-1)
        if final:
            return zt
        return self.requant(zt)

    # -- cheap ops ----------------------------------------------------------------
    def requant(self, zt: _T) -> _T:
        """``relu(requant(z))`` back to int8."""
        out_f = torch.relu(zt.f)
        s_out = _pct(out_f) / INT8_MAX
        m = _mult(zt.scale / s_out)
        shape = [1] * zt.f.dim()
        if m.dim() == 1:
            shape[zt.axis] = m.numel()
        m = m.reshape(shape) if m.dim() == 1 else m

        def fn(z, m=m):
            return requant(z, m.to(z.device), SHIFT, 0, INT8_MAX)

        name = self._name("rq")
        self.ops.append(CheapOp(name, (zt.name,), name + ".y", fn=fn, note="requant relu"))
        return _T(name + ".y", out_f, torch.tensor(s_out, dtype=torch.float64))

    def add_relu(self, main: _T, shortcut: _T) -> _T:
        out_f = torch.relu(main.f + shortcut.f)
        s_out = _pct(out_f) / INT8_MAX
        c = main.f.shape[1]
        m1 = _mult(main.scale / s_out).reshape(1, c, 1, 1)
        ms = shortcut.scale / s_out
        m2 = _mult(ms).reshape(1, c, 1, 1) if ms.dim() == 1 else _mult(ms).reshape(1, 1, 1, 1)
        half = 1 << (SHIFT - 1)

        def fn(a, b, m1=m1, m2=m2):
            return ((a * m1.to(a.device) + b * m2.to(a.device) + half) >> SHIFT).clamp(0, INT8_MAX)

        name = self._name("add")
        self.ops.append(CheapOp(name, (main.name, shortcut.name), name + ".y", fn=fn, note="residual add+relu"))
        return _T(name + ".y", out_f, torch.tensor(s_out, dtype=torch.float64))

    def maxpool(self, t: _T, k: int, s: int, p: int = 0) -> _T:
        def fn(x, k=k, s=s, p=p):
            return F.max_pool2d(x.to(torch.float32), k, s, p).to(torch.int64)

        name = self._name("pool")
        self.ops.append(CheapOp(name, (t.name,), name + ".y", fn=fn, note="maxpool",
                                params={"k": k, "s": s, "p": p}))
        return _T(name + ".y", F.max_pool2d(t.f, k, s, p), t.scale)

    def global_avgpool(self, t: _T) -> _T:
        hw = t.f.shape[2] * t.f.shape[3]

        def fn(x, hw=hw):
            return torch.div(x.sum(dim=(2, 3)) + hw // 2, hw, rounding_mode="floor")

        name = self._name("gap")
        self.ops.append(CheapOp(name, (t.name,), name + ".y", fn=fn, note="global avgpool"))
        return _T(name + ".y", t.f.mean(dim=(2, 3)), t.scale)

    def flatten(self, t: _T) -> _T:
        def fn(x):
            return x.reshape(x.shape[0], -1)

        name = self._name("flat")
        self.ops.append(CheapOp(name, (t.name,), name + ".y", fn=fn, note="flatten"))
        return _T(name + ".y", t.f.flatten(1), t.scale)

    def finish(self, out: _T) -> IntGraph:
        g = IntGraph(self.ops, "x", out.name)
        g.meta["input_scale"] = self.input_scale
        g.meta["output_scale"] = out.scale.tolist()
        return g


def quantize_model(model: nn.Module, x_cal: torch.Tensor) -> IntGraph:
    """Convert a trained float model (eval mode) into an :class:`IntGraph`."""
    import copy

    model = copy.deepcopy(model).eval().double().cpu()
    with torch.no_grad():
        b = CNNBuilder(x_cal.double().cpu())
        t = b.cur
        if isinstance(model, LeNet5):
            t = b.maxpool(b.conv(model.conv1, None, t), 2, 2)
            t = b.maxpool(b.conv(model.conv2, None, t), 2, 2)
            t = b.flatten(t)
            t = b.linear(model.fc1.weight, model.fc1.bias, t)
            t = b.linear(model.fc2.weight, model.fc2.bias, t)
            t = b.linear(model.fc3.weight, model.fc3.bias, t, final=True)
        elif isinstance(model, VGG):
            mods = list(model.body)
            i = 0
            while i < len(mods):
                m = mods[i]
                if isinstance(m, nn.Conv2d):
                    t = b.conv(m, mods[i + 1], t)
                    i += 3
                else:
                    t = b.maxpool(t, 2, 2)
                    i += 1
            t = b.flatten(t)
            t = b.linear(model.fc1.weight, model.fc1.bias, t)
            t = b.linear(model.fc2.weight, model.fc2.bias, t)
            t = b.linear(model.fc3.weight, model.fc3.bias, t, final=True)
        elif isinstance(model, ResNet18):
            t = b.conv(model.stem, model.bn, t)
            if model.pool is not None:
                t = b.maxpool(t, 3, 2, 1)
            for blk in model.blocks:
                h = b.conv(blk.conv1, blk.bn1, t)
                z2 = b.conv(blk.conv2, blk.bn2, h, requant_out=False)
                sc = t if blk.down is None else b.conv(blk.down[0], blk.down[1], t, requant_out=False)
                t = b.add_relu(z2, sc)
            t = b.global_avgpool(t)
            t = b.linear(model.fc.weight, model.fc.bias, t, final=True)
        elif isinstance(model, nn.Sequential):  # an MLP of Linear/ReLU
            linears = [m for m in model if isinstance(m, nn.Linear)]
            t = b.flatten(t)
            for j, lin in enumerate(linears):
                t = b.linear(lin.weight, lin.bias, t, final=(j == len(linears) - 1))
        else:
            raise TypeError(f"no quantiser for {type(model).__name__}")
        return b.finish(t)


def quantize_input(graph: IntGraph, x: torch.Tensor) -> torch.Tensor:
    """The client's (and prover's) public rule for turning an input into int8."""
    s = graph.meta["input_scale"]
    return torch.round(x.double() / s).clamp(-INT8_MAX, INT8_MAX).to(torch.int64)


def dequantize_logits(graph: IntGraph, z: torch.Tensor) -> torch.Tensor:
    """Final-layer claims ``[classes, batch]`` -> float logits ``[batch, classes]``."""
    s = torch.as_tensor(graph.meta["output_scale"], dtype=torch.float64, device=z.device)
    return (z.to(torch.float64) * s[:, None]).T
