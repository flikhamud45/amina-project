"""The exact output law of the integer Gumbel-max sampler (plan E2b, ``experiments/9_quality/sampler_tv.py``):
closed forms on 2- and 8-token toys in exact integer and rational arithmetic, every noise draw of small tables
enumerated through ``TokenSampler.choose`` itself (ties and top-k included), a Monte Carlo of ``choose``, the FFT
path against the direct sum, the skipped-tail bound, and the distance to softmax falling with the table's width."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from pvi.fullcheck import token_sampler as ts


def _script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "9_quality" / "sampler_tv.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("sampler_tv_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _script()
TAB = st.table()


def test_the_table_and_the_weight_are_the_protocols():
    assert np.array_equal(TAB, ts.gumbel_table().numpy())
    for tau, c in [(0.7, 2.7e-4), (1.0, 1 / 16), (1.3, 0.01)]:
        assert st.weight(c, tau) == ts.TokenSampler.for_temperature(bytes(32), tau, c).a
    small = st.table(3)
    assert small.size == 8 and bool((np.diff(small) > 0).all())
    assert np.abs(st.table(16, 20) - 16 * TAB).max() <= 8      # the same quantiles at 16x the resolution
    assert st.parse_table("16") == (16, 16) and st.parse_table("16:20") == (16, 20)


@pytest.mark.parametrize("a,dz", [(1, 0), (1, 5), (1, 123_457), (4096, 3), (77, -2_000), (1, None), (1, "edge")])
def test_two_tokens_match_the_counting_closed_form(a, dz):
    d_max = int(TAB[-1] - TAB[0])
    if dz is None:
        dz = d_max                    # the lower token wins only on a tie at the window's edge
    elif dz == "edge":
        dz = d_max + 1                # outside the window: never
    for z in ([dz, 0], [0, dz]):
        p, info = st.sampler_law(np.array(z), a)
        exact = st.two_token_exact(a, z[0], z[1], TAB)
        assert abs(p[0] - float(exact)) <= 1e-15 and abs(p[1] - float(1 - exact)) <= 1e-15
    if dz == d_max:
        assert st.two_token_exact(a, 0, dz, TAB) == Fraction(1, 1 << 32)    # index 0 wins the single tie
        assert st.two_token_exact(a, dz, 0, TAB) == 1


def test_eight_equal_logits_match_the_rational_closed_form():
    p, info = st.sampler_law(np.full(8, 5, dtype=np.int64), 321)
    exact = st.equal_tokens_exact(8, TAB.size)
    assert sum(exact) == 1
    assert np.abs(p - np.array([float(f) for f in exact])).max() <= 1e-15
    assert bool((np.diff(p) < 0).all())                  # ties go to the smaller index
    assert np.abs(p - 1 / 8).max() < 1e-5 and info["sum_err"] < 1e-14


def _product_formula(z: list[int], a: int, tab: np.ndarray) -> list[Fraction]:
    """``P(v) = 2^-nB sum_i prod_{w<v} #{T_j < t} prod_{w>v} #{T_j <= t}``, ``t = a z_v + T_i - a z_w``, in exact
    integers (the definition, independent of the grid computation)."""
    n_tab = tab.size
    out = []
    for v in range(len(z)):
        acc = np.ones(n_tab, dtype=object)
        for w in range(len(z)):
            if w == v:
                continue
            t = a * (z[v] - z[w]) + tab
            cnt = np.searchsorted(tab, t, side="left" if w < v else "right")
            acc = acc * cnt.astype(object)
        out.append(Fraction(int(acc.sum()), n_tab ** len(z)))
    return out


def test_eight_tokens_with_ties_match_the_product_formula():
    z = [3, 0, 3, -2, 1, 3, -40, 0]                       # repeated levels, one token far below
    a = 9_000
    p, _ = st.sampler_law(np.array(z), a)
    exact = _product_formula(z, a, TAB)
    assert sum(exact) == 1
    assert np.abs(p - np.array([float(f) for f in exact])).max() <= 1e-15


@pytest.mark.parametrize("z,a,top_k", [
    ([2, 2, 0, 1], 1, 0), ([2, 2, 0, 1], 3, 0), ([0, 0, 0, 0], 2, 0), ([5, 1, 5, 5], 4, 2), ([1, 3, 3, 0], 2, 3),
    ([7, -1, 7, 2], 5, 2),
])
def test_the_law_equals_choose_over_every_noise_draw(z, a, top_k, monkeypatch):
    """With an 8-entry table, every one of the 8^4 noise vectors is fed to ``TokenSampler.choose`` (its scores,
    top-k mask and first-maximum argmax), and the frequencies equal the law exactly: ties and the top-k boundary
    included."""
    tab = st.table(3)
    held = {}
    monkeypatch.setattr(ts.TokenSampler, "noise", lambda self, position, vocab: held["g"])
    sampler = ts.TokenSampler(bytes(32), a, top_k)
    zt = torch.tensor(z, dtype=torch.int64)
    counts = np.zeros(len(z), dtype=np.int64)
    for u in itertools.product(range(tab.size), repeat=len(z)):
        held["g"] = torch.from_numpy(tab[list(u)])
        counts[sampler.choose(zt, 0)] += 1
    p, _ = st.sampler_law(np.array(z), a, top_k, tab=tab)
    exact = counts / tab.size ** len(z)
    assert np.abs(p - exact).max() <= 1e-15, (p, exact)


def test_top_k_restricts_to_the_largest_with_ties_to_the_smaller_index():
    z = np.array([4, 9, 4, 1, 4, 9])
    p, _ = st.sampler_law(z, 20_000, top_k=3)
    assert list(np.flatnonzero(p)) == [0, 1, 5]           # the 4 at index 0 is kept, those at 2 and 4 are not
    q, _ = st.sampler_law(np.array([4, 9, 9]), 20_000)
    assert np.abs(p[[0, 1, 5]] - q).max() <= 1e-15


def test_a_monte_carlo_of_choose_agrees_with_the_law():
    z = torch.tensor([0, -3, 5, 5, -1, 2, -7, 4, -60], dtype=torch.int64)    # the last never wins at this a
    a = 1 << 15
    p, info = st.sampler_law(z, a)
    assert p[-1] == 0 and info["window"] == 8
    n = 20_000
    counts = st.monte_carlo(z, a, 0, 3, n)
    cmp = st.compare_mc(p, counts)
    assert cmp["zero_prob_draws"] == 0 and cmp["beyond_3sigma"] == 0 and cmp["max_abs_z"] < 3
    assert cmp["max_abs_z"] < cmp["z_familywise_1pct"] == pytest.approx(3.26, abs=0.01)    # 9 buckets: 8 + pool
    # draw j is choose() under the seed SHA-256(tag || j)
    seed = hashlib.sha256(st.MC_TAG + (0).to_bytes(8, "little")).digest()
    assert int(np.flatnonzero(st.monte_carlo(z, a, 0, 3, 1))[0]) == ts.TokenSampler(seed, a).choose(z, 3)


def test_z_scores_of_certain_buckets_do_not_divide_rounding_by_zero():
    """A law's float64 sum can exceed 1 by its rounding; a bucket holding all of it (every draw pooled, or one gap
    bin with every candidate) must score 0, not ``-1e140``, and a draw outside a certain bucket must score inf."""
    n = 3000
    assert st.zscore(n, n, 1.0 + 1.2e-11) == 0.0 and st.zscore(0, n, -1e-17) == 0.0
    assert st.zscore(n - 1, n, 1.0) == np.inf and st.zscore(1, n, 0.0) == np.inf
    assert st.zscore(40, n, 0.01) == pytest.approx((40 - 30) / np.sqrt(30 * 0.99))
    p = np.full(50_000, 1 / 50_000)
    p[0] += 1.2e-11                                       # every token below the per-token variance: all pooled
    counts = np.zeros(50_000, dtype=np.int64)
    counts[:n] = 1
    cmp = st.compare_mc(p, counts)
    assert cmp["tested_tokens"] == 0 and cmp["max_abs_z"] == 0.0 and cmp["beyond_3sigma"] == 0
    rows = st.binned(np.zeros(50_000, dtype=np.int64), 1.0, {"law": p}, counts)
    assert rows[0]["tokens"] == 50_000 and rows[0]["z_law"] == 0.0


def _flat_logits(n: int = 3000, spread: float = 2.0, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(n) * spread
    real[:3] += [6.0, 5.5, 5.0]
    return np.round(real / scale).astype(np.int64)


def test_the_fft_path_equals_the_direct_sum():
    z = _flat_logits()
    a = st.weight(0.01, 1.0)                               # 655: hundreds of occupied levels
    pf, inf_f = st.sampler_law(z, a, method="fft")
    pd, inf_d = st.sampler_law(z, a, method="direct")
    assert inf_f["method"] == "fft" and inf_d["method"] == "direct"
    assert np.abs(pf - pd).max() < 1e-12 and inf_f["fft_err"] < 1e-9
    assert inf_f["sum_err"] < 1e-10 and inf_d["sum_err"] < 1e-12


def test_the_skipped_tail_is_within_its_bound():
    z = _flat_logits(seed=1)
    a = st.weight(0.01, 1.3)
    exact, _ = st.sampler_law(z, a)
    floor = -12.0                                          # coarse on purpose, so that something is skipped
    cut, info = st.sampler_law(z, a, floor=floor)
    assert info["dropped_bound"] == pytest.approx(info["window"] * np.exp(floor))
    lost = exact - cut
    assert lost.min() > -1e-15 and lost.max() <= np.exp(floor)            # per token: below e^floor (and rounding)
    assert 0 < float(lost.sum()) <= info["dropped_bound"]


def test_a_wider_table_is_closer_to_softmax():
    z = _flat_logits(seed=2)
    c, tau = 0.01, 1.0
    s = np.arange(z.size)
    ideal = st.softmax_on(z * (c / tau), s)
    tvs = {}
    for bits in (12, 14, 16):
        tab = st.table(bits)
        a = st.weight(c, tau, bits)
        p, _ = st.sampler_law(z, a, bits=bits, tab=tab)
        tvs[bits] = (st.tv(p, ideal), st.tv(p, st.softmax_on(z * (a / float(1 << bits)), s)))
    assert tvs[12][0] > tvs[14][0] > tvs[16][0] and tvs[12][1] > tvs[14][1] > tvs[16][1]
    assert tvs[16][1] < tvs[12][1] / 4
