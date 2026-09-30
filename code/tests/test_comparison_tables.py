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
SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "5_comparison"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"comparison_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_total_is_carried_into_the_tables():
    assert "verify_total" in _script("aggregate").ADDITIVE
    assert "verify_total" in _script("validate_extrapolation").METRICS


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
