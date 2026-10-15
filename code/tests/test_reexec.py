"""Re-execution (``experiments/reexec/reexec.py``): the ``int_mm`` variant and the allocator setting.

* ``int_mm`` runs the integer graph with its linear layers' products on ``torch._int_mm``: with an exact int8 GEMM in
  its place every claim equals the graph's own forward pass and the record says ``exact``; with a GEMM that saturates
  as oneDNN's does on CPUs without VNNI (pairs of u8 x s8 products added in int16) the check at the int8 bounds
  fails and the record says not exact;
* where ``torch._int_mm`` has no CPU kernel, refuses a shape or warns that it fell back, the variant is reported
  unavailable on stderr, writes no record, and the other variants run;
* on torch's own CPU kernel (where it has one) the product is exact on operands that cannot saturate, also beyond one
  block of ``INT8_TERMS`` terms and for one column, and a GEMM that passes the bounds check gives the integer logits;
* ``--malloc-tune`` is applied (glibc's defaults without it) and every record says which setting was in force.
"""
from __future__ import annotations

import json
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

from experiments.common.models import decoder_config
from experiments.reexec import reexec
from pvi import options
from pvi.core.field import INT8_TERMS
from pvi.models.decoder import build_decoder


def _main(monkeypatch, capsys, *argv) -> tuple[int, list[dict], str]:
    """``reexec.main()`` on ``argv``; its status, records and stderr (the options and threads restored)."""
    monkeypatch.setattr(sys, "argv", ["reexec.py", *argv])
    saved, threads = options.current(), torch.get_num_threads()
    try:
        status = reexec.main()
    finally:
        options.configure(saved)
        torch.set_num_threads(threads)
    out = capsys.readouterr()
    return status, [json.loads(s) for s in out.out.splitlines() if s.startswith("{")], out.err


TINY = ("--model", "tiny-llama", "--seq", "16", "--device", "cpu", "--threads", "1", "--reps", "1")


def _exact_int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == torch.int8 and a.dim() == b.dim() == 2 and a.shape[1] == b.shape[0]
    return (a.double() @ b.double()).to(torch.int32)


def _saturating_int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """An int8 GEMM without VNNI: ``a + 128`` as u8, pairs of u8 x s8 products added in int16 with saturation
    (``vpmaddubsw``), the pairs summed in int32, ``128`` times ``b``'s column sums subtracted."""
    p = (a.long() + 128)[:, :, None] * b.long()[None]               # [N, K, M]
    if a.shape[1] % 2:
        p = F.pad(p, (0, 0, 0, 1))
    pairs = p.reshape(p.shape[0], -1, 2, p.shape[2]).sum(2).clamp(-(1 << 15), (1 << 15) - 1)
    return (pairs.sum(1) - 128 * b.long().sum(0)).to(torch.int32)


def _cpu_int_mm() -> bool:
    try:
        torch._int_mm(torch.ones(17, 8, dtype=torch.int8), torch.ones(8, 8, dtype=torch.int8))
        return True
    except (RuntimeError, AttributeError):
        return False


def test_int_mm_ops_give_the_claims_of_the_graph(monkeypatch):
    monkeypatch.setattr(torch, "_int_mm", _exact_int_mm)
    graph = build_decoder(decoder_config("tiny-llama"), n_layers=2, calib_tokens=16, prune_last=True)
    ops = reexec.int_mm_ops(graph)
    assert set(ops) == {op.name for op in graph.mat_ops if op.layout == "linear"} and ops
    assert all(op.layout != "embed" for op in graph.mat_ops if op.name in ops)
    tokens = torch.randint(0, 97, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want_env, want = graph.forward(tokens)
        got_env, got = graph.forward(tokens, weights_override=ops)
    assert set(got) == set(want) and all(torch.equal(got[k], want[k]) for k in want)
    assert torch.equal(got_env[graph.output_name], want_env[graph.output_name])


def test_int_mm_record_is_exact_with_an_exact_gemm(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(torch, "_int_mm", lambda a, b: calls.append(b.shape) or _exact_int_mm(a, b))
    status, recs, _ = _main(monkeypatch, capsys, *TINY, "--layers", "1", "2", "--variants", "int", "int_mm")
    assert status == 0 and [r["variant"] for r in recs] == ["int", "int_mm"]
    rx = recs[1]
    assert rx["exact"] is True and rx["exact_check"] == {"bounds": True, "logits": {"1": True, "2": True}}
    assert rx["dtype"] == "int8 GEMM (torch._int_mm)" and set(rx["measured_s"]) == {"1", "2"}
    assert any(shape[1] == 1 for shape in calls)                   # the pruned last block's one column
    assert "exact" not in recs[0]


def test_int_mm_record_is_not_exact_with_a_saturating_gemm(monkeypatch, capsys):
    monkeypatch.setattr(torch, "_int_mm", _saturating_int_mm)
    status, recs, _ = _main(monkeypatch, capsys, *TINY, "--layers", "1", "--variants", "int_mm")
    assert status == 0 and len(recs) == 1
    assert recs[0]["exact"] is False and recs[0]["exact_check"]["bounds"] is False
    assert recs[0]["dtype"].endswith("(not exact)")


def test_int_mm_without_a_cpu_kernel_is_reported_and_skipped(monkeypatch, capsys):
    def missing(a, b):
        raise NotImplementedError("Could not run 'aten::_int_mm' with arguments from the 'CPU' backend.\nMeta: ...")

    monkeypatch.setattr(torch, "_int_mm", missing)
    status, recs, err = _main(monkeypatch, capsys, *TINY, "--layers", "1", "--variants", "int", "int_mm", "float")
    assert status == 0 and [r["variant"] for r in recs] == ["int", "float"]
    assert "int_mm unavailable" in err and "Meta:" not in err


@pytest.mark.parametrize("how", ["refuses", "warns"])
def test_int_mm_on_a_shape_it_refuses_or_falls_back_on_is_reported_and_skipped(monkeypatch, capsys, how):
    def one_column(a, b):                                             # the probe has 24 columns, the last block 1
        if b.shape[1] == 1:
            if how == "refuses":
                raise RuntimeError("int_mm: unsupported shape")
            warnings.warn("int_mm_out_cpu failed, switching to BLAS gemm: unsupported shape")
        return _exact_int_mm(a, b)

    monkeypatch.setattr(torch, "_int_mm", one_column)
    status, recs, err = _main(monkeypatch, capsys, *TINY, "--layers", "1", "--variants", "int_mm", "int")
    assert status == 0 and [r["variant"] for r in recs] == ["int"]
    assert "int_mm unavailable for tiny-llama at 16 tokens" in err


@pytest.mark.skipif(not _cpu_int_mm(), reason="this torch has no CPU kernel of torch._int_mm")
def test_int_mm_on_torchs_cpu_kernel():
    g = torch.Generator().manual_seed(0)
    for n, k, m in ((40, 4096, 24), (17, INT8_TERMS + 40, 5), (64, 160, 1)):
        w = torch.randint(-63, 64, (n, k), generator=g).to(torch.int8)       # no pair of products saturates int16
        x = torch.randint(-63, 64, (m, k), generator=g).to(torch.int8)
        assert torch.equal(reexec.int_mm_product(w, x), (w.double() @ x.double().T).to(torch.int64))


@pytest.mark.skipif(not _cpu_int_mm(), reason="this torch has no CPU kernel of torch._int_mm")
def test_int_mm_record_on_torchs_cpu_kernel(monkeypatch, capsys):
    status, recs, err = _main(monkeypatch, capsys, *TINY, "--layers", "1", "2", "--variants", "int_mm")
    if not recs:
        assert "int_mm unavailable" in err                            # e.g. a torch without oneDNN
        return
    check = recs[0]["exact_check"]
    assert status == 0 and recs[0]["exact"] == (check["bounds"] and all(check["logits"].values()))
    if check["bounds"]:                                               # a GEMM that does not saturate is exact
        assert recs[0]["exact"] is True


@pytest.mark.parametrize("flags, tuned", [((), False), (("--malloc-tune",), True), (("--no-malloc-tune",), False)])
def test_the_allocator_setting_is_applied_and_recorded(monkeypatch, capsys, flags, tuned):
    calls = []

    def tune():                                    # as native.tune_malloc: applied only with the option on
        calls.append(options.current().malloc_tune)
        return options.current().malloc_tune

    monkeypatch.setattr(reexec.native_kernels, "tune_malloc", tune)
    status, recs, _ = _main(monkeypatch, capsys, *TINY, "--layers", "1", "--variants", "float", *flags)
    assert status == 0 and calls == [tuned] and len(recs) == 1
    assert recs[0]["malloc_tuned"] is tuned and recs[0]["env"]["implementation"]["options"]["malloc_tune"] is tuned
