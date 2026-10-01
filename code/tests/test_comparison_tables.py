"""The report's tables count the streaming verifier's time (``verify_total``), and only once.

``bench.py --verifier-impl stream`` records one ``verify_total`` per query instead of the
verify phases it overlaps; ``aggregate.py`` and ``paper_assets.py`` must carry it into the
tables and refuse a row that has both.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytestmark = pytest.mark.filterwarnings("ignore:.enablePackrat. deprecated")   # matplotlib's own pyparsing call
SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "5_comparison"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"comparison_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_total_is_carried_into_the_tables():
    assert "verify_total" in _script("aggregate").ADDITIVE
    assert "verify_total" in _script("validate_extrapolation").METRICS


def test_every_timing_of_run_query_is_carried_into_the_tables():
    """Every timing ``run_query`` records -- modes C, K and Kpre, a ``c`` policy's lookup tables and a K
    verifier's own rows, with and without wire, batched and streaming -- is additive in aggregate.py,
    checked by validate_extrapolation.py and part of paper_assets.py's prover or verifier time (once:
    never both), and the batched verify phases the streaming verifier's ``verify_total`` covers are
    those ``OVERLAPPED`` names."""
    from pvi.fullcheck import protocol as proto
    from test_plans import _planned, _verifiers

    graph, x, coms = _planned("gpt", "R8c")
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    verifiers = [*_verifiers(graph, coms, False), *_verifiers(graph, coms, True)]
    for mode in ("K", "Kpre"):
        for lookups in (False, True):
            for stream in (False, True):
                v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), mode, weights=weights,
                                   lookups=lookups, stream=stream)
                if mode == "Kpre":
                    v.precompute(proto.Challenger(seed=1))
                verifiers.append(v)
    keys = {False: set(), True: set()}                 # by stream
    for v in verifiers:
        for wire in (False, True):
            res = proto.run_query(proto.Prover(graph, commitments=coms), v, x, seed=None if v.params.fiat_shamir
                                  else 1, wire=wire)
            assert res["accepted"]
            keys[v.stream] |= set(res["timings"])
    emitted = keys[False] | keys[True]
    assert {"prove_lookups", "verify_lookups", "verify_total"} <= emitted
    pa = _script("paper_assets")
    assert emitted <= _script("aggregate").ADDITIVE and emitted <= _script("validate_extrapolation").METRICS
    assert emitted <= set(pa.PROVE) | set(pa.VERIFY) and set(pa.PROVE) & set(pa.VERIFY) == {"fs_hash"}
    batched_only = {k for k in keys[False] - keys[True] if k.startswith("verify_")}
    assert batched_only <= set(pa.OVERLAPPED) and not keys[True] & set(pa.OVERLAPPED)
    assert set(pa.OVERLAPPED) - batched_only == {"verify_upload"}    # (a GPU client's: none on the CPU)


def test_the_streaming_verifier_is_timed_once():
    pa = _script("paper_assets")
    row = {"prove_forward": (2.0, "measured", 7e6, "gpu"), "prove_fold": (0.5, "measured", 7e6, "gpu"),
           "fs_hash": (0.25, "measured", 7e6, "gpu"), "bytes_total": (1000.0, "measured", 7e6, "gpu")}
    stream = dict(row, verify_total=(3.0, "measured", 7e6, "gpu"))
    assert pa._llm_cost(stream) == (2.75, 3.25, 1000.0, 7e6)
    phases = dict(row, verify_derive=(1.0, "measured", 7e6, "gpu"), verify_products=(1.5, "measured", 7e6, "gpu"))
    assert pa._llm_cost(phases) == (2.75, 2.75, 1000.0, 7e6)
    with pytest.raises(SystemExit):
        pa._llm_cost(dict(phases, verify_total=(3.0, "measured", 7e6, "gpu")))
    m = pa.Measured.__new__(pa.Measured)             # a CNN cell of the summary table, without the files
    m.idx = {("cnn", "lenet5", "cell", k, "", "", "", "", ""): {"median": str(v)}
             for k, v in (("verify_total", 0.5), ("fs_hash", 0.25))}
    assert m.total("lenet5", "cell", pa.VERIFY) == 0.75
    m.idx[("cnn", "lenet5", "cell", "verify_products", "", "", "", "", "")] = {"median": "0.1"}
    with pytest.raises(SystemExit):
        m.total("lenet5", "cell", pa.VERIFY)
