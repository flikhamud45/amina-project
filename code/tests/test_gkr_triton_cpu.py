"""The GPU GKR prover's host logic (``pvi.v1.gkr_triton``) on the CPU, its kernels replaced by exact CPU
stand-ins (``gkr_triton_mock``): the round structure (``h(0), h(1), h(2)``, the eq factor taken out), the suffix eq
tables, and the bottom layer's path (full rows, ``T = 2^n_col``) give the eager prover's transcript byte for byte.
The kernels themselves are checked against the same transcripts on a CUDA host (``test_gkr_triton.py``).  The
module is the output of its generator (``tools/gen_gkr_triton.py``)."""

from __future__ import annotations

import gc
import importlib.util
import weakref
from pathlib import Path

import pytest
import torch

import gkr_triton_mock
import pvi.v0 as proto
from pvi.core import extfield as ef
from pvi.core.field import P
from pvi.v1 import gkr, logup
from pvi.v1 import gkr_triton as gt


@pytest.fixture
def mocked(monkeypatch):
    gkr_triton_mock.install(gt, monkeypatch)
    return gt


def _ch():
    c = proto.Challenger(fiat_shamir=True)
    c.absorb(b"statement", b"gkr triton cpu")
    return c


@pytest.mark.parametrize("n", [1, 2, 3, 6, 9])
def test_uploaded_leaves(mocked, n):
    g = torch.Generator().manual_seed(n)
    p = torch.randint(0, P, (1 << n, ef.D), generator=g)
    q = torch.randint(0, P, (1 << n, ef.D), generator=g)
    a = gkr.prove(p, q, _ch(), "cut/0/")
    b = mocked.prove(p, q, _ch(), "cut/0/", device="cpu")
    assert a.to_bytes() == b.to_bytes() and torch.equal(a.point, b.point)


@pytest.mark.parametrize("rows,cols,n_col", [(1, 4, 2), (3, 8, 3), (5, 16, 4), (8, 16, 4), (13, 32, 5),
                                             (7, 9, 4), (2, 2, 1), (6, 64, 6)])
def test_instance_leaves_with_and_without_the_bottom_path(mocked, rows, cols, n_col):
    # cols == 2^n_col (n_col >= 2): the bottom layer's column rounds on the real rows only; otherwise the generic path
    g = torch.Generator().manual_seed(rows * 100 + cols)
    w = torch.randint(1, 1 << 12, (rows, 1), generator=g).expand(rows, cols).clone()
    d = (torch.rand(rows, cols, generator=g) * w).floor().to(torch.int64)
    alpha = torch.randint(0, P, (ef.D,), generator=g)
    n = n_col + max(0, (rows - 1).bit_length())
    pq = logup.leaves(d, w, n_col, n, alpha)
    a = gkr.prove(*pq, _ch(), "cut/2/")
    b = mocked.prove_leaves(d, w, n_col, n, alpha, _ch(), "cut/2/", device="cpu")
    assert a.to_bytes() == b.to_bytes() and torch.equal(a.point, b.point)


@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize("rows,cols,n_col", [(5, 16, 4), (13, 32, 5), (3, 8, 3), (8, 16, 4), (7, 9, 4)])
def test_the_low_memory_path_frees_the_leaves_numerators(mocked, monkeypatch, rows, cols, n_col, low_memory):
    # full rows (T = 2^n_col; their count need not be a power of two, padding rows fill the instance): on the
    # low-memory path the bottom path reads the leaves' q only and no reference to their p outlives the first
    # combine; the default path keeps them; the transcript is the eager prover's on both.  Rows that are not full
    # (7 x 9) keep the generic path and the leaves' p.
    leaf, refs, seen = mocked._leaf_kernel, {}, {}

    class Spy:
        def __getitem__(self, grid):
            def run(d, w, p, q, *a, **kw):
                refs["p"], refs["q"] = weakref.ref(p), weakref.ref(q)
                return leaf[grid](d, w, p, q, *a, **kw)
            return run
    prove_tree = mocked._prove_tree

    def spy(layers, *a, **kw):
        gc.collect()
        seen["p_alive"], seen["q_alive"] = refs["p"]() is not None, refs["q"]() is not None
        return prove_tree(layers, *a, **kw)
    monkeypatch.setattr(mocked, "_leaf_kernel", Spy())
    monkeypatch.setattr(mocked, "_prove_tree", spy)
    g = torch.Generator().manual_seed(rows * 7 + cols)
    w = torch.randint(1, 1 << 12, (rows, 1), generator=g).expand(rows, cols).clone()
    d = (torch.rand(rows, cols, generator=g) * w).floor().to(torch.int64)
    alpha = torch.randint(0, P, (ef.D,), generator=g)
    n = n_col + max(0, (rows - 1).bit_length())
    a = gkr.prove(*logup.leaves(d, w, n_col, n, alpha), _ch(), "cut/4/")
    b = mocked.prove_leaves(d, w, n_col, n, alpha, _ch(), "cut/4/", device="cpu", low_memory=low_memory)
    assert a.to_bytes() == b.to_bytes() and torch.equal(a.point, b.point)
    assert seen["q_alive"] and seen["p_alive"] == (not low_memory or cols != 1 << n_col)
    gc.collect()
    assert refs["p"]() is None and refs["q"]() is None          # nothing of the tree outlives the proof


def test_the_low_memory_tree_holds_the_only_reference_to_its_leaves(mocked):
    g = torch.Generator().manual_seed(3)
    p = gt._planes_mont(torch.randint(0, P, (16, ef.D), generator=g), "cpu")
    q = gt._planes_mont(torch.randint(0, P, (16, ef.D), generator=g), "cpu")
    ref_p, ref_q = weakref.ref(p), weakref.ref(q)
    leaves = [p, q]
    del p, q
    kept = mocked._tree_low(leaves, drop_leaf_p=False)
    assert leaves == [] and kept[-1][0] is ref_p() and len(kept) == 5
    assert all(torch.equal(x, y) for a, b in zip(kept, mocked._tree(kept[-1][0], kept[-1][1])) for x, y in zip(a, b))
    leaves = [kept[-1][0], kept[-1][1]]
    del kept
    layers = mocked._tree_low(leaves, drop_leaf_p=True)
    gc.collect()
    assert leaves == [] and ref_p() is None and layers[-1] == (None, ref_q()) and layers[0][0].shape[1] == 1


def test_the_module_is_its_generators_output():
    path = Path(__file__).resolve().parents[1] / "tools" / "gen_gkr_triton.py"
    spec = importlib.util.spec_from_file_location("gen_gkr_triton", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert Path(gt.__file__).read_bytes().decode() == gen.render()
