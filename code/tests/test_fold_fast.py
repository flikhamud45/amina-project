"""A6: the Kpre client's precompute ``u = chi A`` (``Verifier._fold_local``) on the native field product (CPU) and
int8 GEMMs (GPU) equals the float64 limb products bit for bit: at the int8 extremes, with ``chi`` entries ``p - 1``,
and for ``K`` past ``2^16``; the Kpre verdicts are unchanged."""

import pytest
import torch

from pvi.kernels import native as native_kernels
from pvi.v0 import protocol as proto
from pvi.core.field import P, int8_ok, small_matmul_mod
from pvi.models.decoder import CONFIGS, build_decoder


def _reference(w, chi):
    return small_matmul_mod(w.T.to(torch.int64).contiguous(), chi.T.contiguous()).T


class _Op:
    def __init__(self, name):
        self.name = name


def _verifier(w, device):
    v = proto.Verifier.__new__(proto.Verifier)
    v.weights = {"m": (w, None)}
    v._wdev = {}
    return v


def _cases(device):
    g = torch.Generator().manual_seed(0)
    for n, k in ((7, 5), (64, 300), (33, (1 << 16) + 37)):
        w = torch.randint(-128, 128, (n, k), generator=g).to(torch.int8)
        w[0, :] = 127
        w[1 % n, :] = -128
        chi = torch.randint(0, P, (3, n), generator=g)
        chi[0, :] = P - 1
        yield w, chi.to(device)


@pytest.mark.skipif(not native_kernels.available(), reason="the native library needs a C++ compiler (Linux)")
def test_cpu_fold_equals_the_float64_products(monkeypatch):
    for w, chi in _cases("cpu"):
        v = _verifier(w, "cpu")
        monkeypatch.setenv("PVI_FOLD_FAST", "1")
        got = v._fold_local(_Op("m"), chi)
        assert torch.equal(got, _reference(w, chi)), tuple(w.shape)


@pytest.mark.skipif(not (torch.cuda.is_available() and int8_ok("cuda")), reason="needs a GPU with int8 GEMMs")
def test_gpu_fold_equals_the_float64_products(monkeypatch):
    for w, chi in _cases("cpu"):
        v = _verifier(w, "cuda")
        monkeypatch.setenv("PVI_FOLD_FAST", "1")
        got = v._fold_local(_Op("m"), chi.cuda()).cpu()
        assert torch.equal(got, _reference(w, chi)), tuple(w.shape)


def test_switch_off_is_the_float64_path(monkeypatch):
    for w, chi in _cases("cpu"):
        if w.shape[1] > 4096:
            continue
        v = _verifier(w, "cpu")
        monkeypatch.setenv("PVI_FOLD_FAST", "0")
        assert torch.equal(v._fold_local(_Op("m"), chi), _reference(w, chi))


def test_kpre_verdicts_are_unchanged(monkeypatch):
    graph = build_decoder(CONFIGS["gpt2"], n_layers=1, calib_tokens=8, seed=0, prune_last=True)
    x = torch.randint(0, CONFIGS["gpt2"].vocab, (1, 8), generator=torch.Generator().manual_seed(1))
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("PVI_FOLD_FAST", flag)
        params = proto.params_for(40, len(graph.mat_ops))
        v = proto.Verifier(graph.public(), params, "Kpre",
                           weights={op.name: (op.weight, op.bias) for op in graph.mat_ops})
        ch = proto.Challenger(seed=7)
        v.precompute(ch)
        pre = {k: (a.clone(), b.clone()) for k, (a, b) in v._pre.items()}
        res = proto.run_query(proto.Prover(graph), v, x, seed=3)
        out[flag] = (pre, res["accepted"])
    assert out["0"][1] and out["1"][1]
    for k in out["0"][0]:
        assert torch.equal(out["0"][0][k][1], out["1"][0][k][1]), k
