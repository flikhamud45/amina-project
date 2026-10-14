"""The break-even of fig:breakeven (``experiments/deployment/ev_breakeven.py``): the crossovers are where the two
costs meet, and the stored inputs give the crossovers the paper quotes."""

from __future__ import annotations

import json

import pytest

from experiments.deployment import ev_breakeven as ev


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


def test_prompt_crossover_is_where_the_proof_reaches_the_weights():
    W = 6.74e9
    t = ev.crossover_tokens(W, (64, 0.3e9), (2048, 6.5e9))
    assert 0.3e9 + (t - 64) * (6.5e9 - 0.3e9) / (2048 - 64) == pytest.approx(W)
    assert ev.crossover_tokens(W, (64, 1e9), (2048, 1e9)) is None


def test_the_stored_inputs_give_the_papers_crossovers():
    """Llama-2-7B under Fiat--Shamir and the plan cnn18c (Sec. sec:discussion, fig:breakeven, ext:disc-real): about 22
    64-token queries; one query at 2,048 tokens, or 2.3 with V1; links of 0.42-0.59 Gb/s against exact
    re-execution, the integer forward pass (V1: 0.23); V1's client
    faster than v0's below about 1.1 Gb/s; one proof as large as the weights at about 2,100 tokens."""
    out = ev.crossovers(json.loads(ev.INPUTS.read_text(encoding="utf-8")), ["cnn18c"])
    q = {(r["seq"], r["protocol"]): r["Q_star"] for r in out["panel1"] if r["fs"]}
    assert q[64, "v0"] == pytest.approx(22.3, abs=0.05)
    assert q[2048, "v0"] == pytest.approx(1.03, abs=0.01) and q[2048, "v1"] == pytest.approx(2.32, abs=0.01)
    link = {(r["seq"], r["protocol"]): r["L_star_gbps"] for r in out["panel2"] if r["fs"]}
    assert link[64, "v0 Kpre"] == pytest.approx(0.59, abs=0.005)
    assert link[2048, "v0 Kpre"] == pytest.approx(0.42, abs=0.005)
    assert link[2048, "V1 Kpre"] == pytest.approx(0.23, abs=0.005)
    (v1,) = out["v1_against_v0"]
    assert v1["v1_faster_below_gbps"] == pytest.approx(1.13, abs=0.01)
    tokens = {r["protocol"]: r["tokens"] for r in out["prompt_crossover"] if r["fs"]}
    assert tokens["v0"] == pytest.approx(2111, abs=1)
    assert ev.comment_lines(out)[1] == "% Q* at 64 tokens, cnn18c, FS: 22.3 (302.9 MB per query)"
