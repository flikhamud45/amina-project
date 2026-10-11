"""EV's break-even formulas (``experiments/5_comparison/ev_breakeven.py``): the crossovers are where the two costs meet."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "5_comparison"))
import ev_breakeven as ev   # noqa: E402


def test_query_crossover_is_where_the_byte_costs_meet():
    W, b = 6.74e9, 389.0e6
    q = ev.crossover_queries(W, b)
    assert q * b == pytest.approx(W)
    assert (q - 1) * b < W < (q + 1) * b
    with pytest.raises(ValueError):
        ev.crossover_queries(W, 0)


def test_link_crossover_is_where_the_times_meet():
    B, t_v, t_r = 6.43e9, 19.64, 93.2
    L = ev.breakeven_link_gbps(B, t_v, t_r)
    assert t_v + 8 * B / (L * 1e9) == pytest.approx(t_r)
    assert t_v + 8 * B / (1.01 * L * 1e9) < t_r < t_v + 8 * B / (0.99 * L * 1e9)    # faster links win
    assert ev.breakeven_link_gbps(B, 100.0, t_r) is None                              # the verifier alone is slower


def test_the_draft_inputs_reproduce_the_plan_crossovers():
    """The plan's quoted crossovers (Q* 20.6 interactive / 17.5 FS at 64 tokens, 0.7 Gb/s at 2,048 tokens against fp32)
    follow from the draft inputs, to the precision they were quoted with."""
    W = 6.74e9
    assert ev.crossover_queries(W, 196.43e6 + 131.9e6) == pytest.approx(20.5, abs=0.2)
    assert ev.breakeven_link_gbps(6433.74e6, 19.64, 93.2) == pytest.approx(0.70, abs=0.01)
