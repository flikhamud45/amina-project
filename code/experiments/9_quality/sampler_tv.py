"""Plan 2, E2b: the exact output law of the integer Gumbel-max sampler, and its total-variation distance to softmax
sampling on real logits.

    cd code && PYTHONPATH=src python experiments/9_quality/sampler_tv.py --selfcheck
    cd code && PYTHONPATH=src python experiments/9_quality/sampler_tv.py --model facebook/opt-125m --seq 128 \\
        --windows 1 --positions 32 --logits-cache e2b_opt-125m.npz --mc 100000 --mc-positions 8 31 \\
        --mc-settings 1:0 1:50 1.3:0 --out sampler_tv_opt-125m.json
    cd code && PYTHONPATH=src python experiments/9_quality/sampler_tv.py --model facebook/opt-1.3b --seq 128 \\
        --windows 4 --threads 8 --workers 16 --logits-cache /tmp/e2b_opt-1.3b.npz --out sampler_tv_opt-1.3b.json

**The law.**  ``TokenSampler.choose`` (``pvi.fullcheck.token_sampler``) gives entry ``v`` the score
``s_v = a z_v + T[u_v]``, where ``u_v`` is the ``v``-th 16-bit word of ``SHAKE-256(tag || seed || position)`` --
modelled here, as for any hash-derived randomness, as independent uniform words on ``{0, ..., 2^B - 1}`` (``B = 16``)
-- and ``T`` is the public table ``T_i = round(2^B (-ln(-ln((i + 1/2) / 2^B))))``, strictly increasing (the
derivative of ``-ln(-ln u)`` is at least ``e``, so consecutive entries differ by at least ``e`` before rounding).
Among the candidate set ``S`` (the ``top_k`` largest ``z``, ties to the smaller index; all if ``top_k = 0``) it
returns the first ``v`` maximising ``s_v`` (``torch.argmax`` returns the first maximum; the masked entries score
``int64 min`` and never win).  With ``F(y) = P(T[u] <= y) = #{j: T_j <= y} / 2^B`` and ``F^-(y) = P(T[u] < y)``:

    P(v) = 2^-B sum_i  prod_{w in S, w < v} F^-(a z_v + T_i - a z_w)  prod_{w in S, w > v} F(a z_v + T_i - a z_w).

**On the integer grid.**  Put ``z* = max_S z``, ``d_w = a (z* - z_w)`` (a multiple of ``a``) and ``D = T_max - T_0``.
A token with ``d_w > D`` never wins (its best score is below the top token's worst), so only the window
``W = {w in S: d_w <= D}`` matters.  On the grid ``s in [0, D]`` (score threshold ``a z* + T_0 + s``):

* ``g(y) = log F(T_0 + y)`` (``g = 0`` for ``y >= D``);
* ``LM(s) = sum_{w in W} g(s + d_w)``, the log-CDF of the largest score: a correlation of the level histogram
  ``n_k = #{w in W: z* - z_w = k}`` with ``g`` along each of the ``a`` residues ``s mod a`` (:func:`log_max_cdf`: a
  direct float64 sum when that is small, a float64 FFT otherwise), computed once per logit vector;
* for each ``v`` in ``W`` in index order, gathered at ``s_i = T_i - T_0 - d_v`` (only ``i`` with ``s_i >= 0``; for
  the others a token of level 0 scores above the threshold):

      P(v) = 2^-B sum_i exp( LM(s_i) - log F(T_i) + C_v(s_i) ),
      C_v(s) = sum_{w in W, w < v} log (F^- / F)(T_0 + s + d_w).

  ``log(F^-/F)(y)`` is nonzero only at table values ``y = T_j``, where it is ``log(j / (j + 1))`` (``-inf`` at
  ``j = 0``), so ``C_v`` is one grid array updated by one scatter per token (:func:`sampler_law`).  The strict and
  non-strict comparisons, hence the tie-break, are therefore exact: no tie is bracketed.  The cost is
  ``O(D log D)`` once plus ``O(#{i: s_i >= 0})`` per token in the window; draws whose integrand is provably below
  ``e^floor`` are skipped (``--floor``, default ``-60``; each law reports the bound ``|W| e^floor`` on what that
  loses, ``4.4e-22`` for OPT's vocabulary).

The only other rounding is float64's: ``|sum_v P(v) - 1|`` and, on the FFT path, the largest deviation of ``LM`` from
the direct sum at ``--fft-check`` random grid points are reported with every law.  The law is checked against
closed forms on 2- and 8-token toys (exact integer and rational arithmetic) and against a Monte Carlo of
``TokenSampler.choose`` itself (``--selfcheck``, ``--mc``; ``tests/test_sampler_tv.py`` also feeds every noise draw
of small tables through ``choose``).

**What is reported** for each analysed position, temperature ``tau`` (``--taus``), ``top_k`` (``--topk``) and table
(``--tables``: ``16`` is the protocol's; ``B`` simulates ``2^B`` entries at resolution ``2^-B`` and ``B:R`` ``2^B``
entries at resolution ``2^-R``, both with ``a = round(2^R c / tau)``):

* ``tv_sampler = TV(P, softmax_S(c z / tau))``: the sampler against ideal sampling from the integer model's logits
  at the requested temperature (``c`` is the graph's logit scale, real logits ``c z``);
* ``tv_table = TV(P, softmax_S(a z / 2^R))``: against the Gumbel-max target at the rounded weight, i.e. the table's
  own error (the quantised uniforms, the rounded values, the truncated tails);
* ``tv_temp = TV(softmax_S(a z / 2^R), softmax_S(c z / tau))``: what the rounding of ``a`` alone costs;
* ``tail_mass``: the target's mass outside the window (candidates the sampler never draws);
* separately, ``tv_model = TV(softmax_S(c z / tau), softmax_S'(l / tau))`` with the fp32 checkpoint's logits ``l``
  (``S'`` its own top-k set): the integer model's own distance to the float model.

Logits: WikiText-2 raw test windows of ``--seq`` tokens from the start of the split (``common.load_windows``), all
positions of each window in one pass (``real_weights.real_logits(all_positions=True)``), ``--positions`` of them
evenly spaced (all if 0); the graph is the paper's smoothed recipe (``build_opt_from_hf(norm_gain=4, smooth=0.55,
smooth_pct=99.9, pct=99.99, pct_att=99.99)``) calibrated on the last ``--calib-seq``-token window of the split.
``--logits-cache`` stores the integer and fp32 logits (``.npz``) after the first run, so the analysis can be
repeated without the model, which is freed before the analysis starts.  ``--workers`` spreads the positions over
processes (about 0.6 GB each with a 20-bit table).  Output: one JSON document (stdout without the per-position rows,
and ``--out`` with them): the summary per setting (mean, median, 90th percentile and maximum over positions), the
per-position rows, the Monte Carlo comparisons, the toys, seconds and peak memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pvi.fullcheck.token_sampler import FRAC, TokenSampler, gumbel_table   # noqa: E402

SMOOTH = dict(norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
DIRECT_BUDGET = 6e7          # occupied levels x grid points below which LM is summed directly (no FFT rounding)
FFT_CHUNK = 1 << 20          # complex entries per FFT block (16 MiB): bounds the FFT path's working memory
EXP_FLOOR = -746.0           # exp(x) is 0.0 in float64 below about -745.13
MC_TAG = b"pvi/e2b/mc"


# ------------------------------------------------------------------ the table

def parse_table(spec: str) -> tuple[int, int]:
    """``"B"`` (``2^B`` entries at resolution ``2^-B``) or ``"B:R"`` (``2^B`` entries at resolution ``2^-R``)."""
    parts = [int(x) for x in str(spec).split(":")]
    return (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])


def table(bits: int = FRAC, value_bits: int | None = None) -> np.ndarray:
    """A Gumbel table with ``2^bits`` entries ``round(2^value_bits (-ln(-ln((i + 1/2) / 2^bits))))`` (int64,
    strictly increasing).  ``(16, 16)`` is the protocol's own (``token_sampler.gumbel_table``, checked against its
    hash); every other width is simulated, for comparison only."""
    value_bits = bits if value_bits is None else value_bits
    if bits == value_bits == FRAC:
        tab = gumbel_table().numpy()
    else:
        u = (np.arange(1 << bits, dtype=np.float64) + 0.5) / (1 << bits)
        tab = np.rint(-np.log(-np.log(u)) * float(1 << value_bits)).astype(np.int64)
    if not bool((tab[1:] > tab[:-1]).all()):
        raise ValueError("the law assumes a strictly increasing table")
    return tab


def weight(logit_scale: float, tau: float, value_bits: int = FRAC) -> int:
    """The integer weight ``a`` at temperature ``tau``: ``TokenSampler.for_temperature``'s for the protocol's table
    (``round(2^16 c / tau)``), the same rounding at ``2^value_bits`` for a simulated one."""
    if value_bits == FRAC:
        return TokenSampler.for_temperature(bytes(32), tau, logit_scale).a
    return max(1, min((1 << 24) - 1, round(logit_scale * (1 << value_bits) / tau)))


def candidates(z: np.ndarray, top_k: int) -> np.ndarray:
    """``S`` as ``TokenSampler.choose`` forms it: the ``top_k`` largest ``z``, ties to the smaller index (all if 0),
    returned in increasing index order."""
    if top_k and top_k < z.size:
        return np.sort(np.argsort(-z, kind="stable")[:top_k])
    return np.arange(z.size)


# ------------------------------------------------------------------ the exact law

def _next_fast_len(n: int) -> int:
    """The smallest ``2^i 3^j 5^k >= n`` (pocketfft is fastest on such lengths)."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = p35
            while m < n:
                m *= 2
            best = min(best, m)
            p35 *= 3
        p5 *= 5
    return best


def log_max_cdf(n_lev: np.ndarray, a: int, gp: np.ndarray, d1: int, method: str = "auto") -> tuple[np.ndarray, str]:
    """``LM[s] = sum_k n_lev[k] g[s + a k]`` for ``s in [0, d1)``, with ``g = gp[:d1]`` and 0 beyond.

    ``n_lev`` has ``Q = (d1 - 1) // a + 1`` entries (levels ``0 .. Q-1``) and ``gp`` is ``g`` padded with zeros to
    ``Q a`` entries.  Writing ``s = a q + r``, ``LM`` is, for each residue ``r``, the correlation of ``n_lev`` with
    ``g[a . + r]``: the FFT path views ``gp`` as ``[Q, a]`` (no copy) and transforms along the first axis in blocks
    of residues of at most ``FFT_CHUNK`` complex entries, so its memory is ``gp``, the result and one block.
    ``direct`` adds the occupied levels' shifted copies of ``g`` one by one (float64 sums, no FFT rounding).
    Returns ``(LM, method used)``."""
    q_n = n_lev.size
    occ = np.flatnonzero(n_lev)
    if method == "direct" or (method == "auto" and occ.size * d1 <= DIRECT_BUDGET):
        out = np.zeros(d1)
        for k in occ:
            off = a * int(k)
            out[:d1 - off] += float(n_lev[k]) * gp[off:d1]
        return out, "direct"
    g2 = gp.reshape(q_n, a)
    nfft = _next_fast_len(2 * q_n - 1)
    fn = np.fft.rfft(n_lev[::-1].astype(np.float64), nfft)
    out2 = np.empty((q_n, a))
    step = max(1, FFT_CHUNK // (nfft // 2 + 1))
    for c0 in range(0, a, step):
        blk = np.fft.rfft(g2[:, c0:c0 + step], nfft, axis=0)
        blk *= fn[:, None]
        out2[:, c0:c0 + step] = np.fft.irfft(blk, nfft, axis=0)[q_n - 1:2 * q_n - 1]
        del blk
    return out2.reshape(-1)[:d1], "fft"


def window(z: np.ndarray, a: int, top_k: int, d_max: int) -> tuple[np.ndarray, np.ndarray]:
    """The candidates that can win, ``W = {w in S: a (z* - z_w) <= D}``, in index order, and their levels
    ``z* - z_w``."""
    s_idx = candidates(z, top_k)
    lev = int(z[s_idx].max()) - z[s_idx]
    inw = lev <= d_max // a
    return s_idx[inw], lev[inw]


def sampler_law(z, a: int, top_k: int = 0, bits: int = FRAC, *, tab: np.ndarray | None = None,
                method: str = "auto", fft_check: int = 256, floor: float = EXP_FLOOR) -> tuple[np.ndarray, dict]:
    """The exact law of ``TokenSampler(seed, a, top_k).choose(z, position)`` over a uniform seed: float64 ``[V]``
    (zero outside the window), and a dict of diagnostics (window size, occupied levels, grid size, method,
    ``|sum - 1|``, the FFT path's largest deviation of ``LM`` from the direct sum at ``fft_check`` random grid points,
    ``dropped_bound``).

    ``tab`` is the table (default: ``table(bits)``; tests pass small ones).  Grid points where
    ``LM(s) + log 2^B < floor`` are skipped, gathers and scatters alike (``C`` is only ever read where it is
    gathered): every integrand there is below ``e^floor`` (``-log F <= log 2^B`` and ``C <= 0``), so each token
    loses less than ``e^floor`` (its terms carry total weight ``<= 1``) and the law as a whole less than
    ``dropped_bound = |W| e^floor``.  The default, ``EXP_FLOOR``, is below float64's smallest positive number (the
    skipped terms are 0.0 when computed); the analysis uses ``-60`` (``|W| e^-60 < 5e-22`` for any vocabulary below
    ``5.7e4``), which skips the far tails' hopeless draws and is several times faster on flat logits.
    Memory: three float64 grids of ``D + 1`` entries at most (``g``, ``LM``, then ``LM`` and ``C``) plus one FFT
    block; about 7.5 MB each for the 16-bit table, 145 MB for a 20-bit one."""
    z = np.asarray(z.cpu() if torch.is_tensor(z) else z, dtype=np.int64).reshape(-1)
    tab = table(bits) if tab is None else np.asarray(tab, dtype=np.int64)
    if not bool((tab[1:] > tab[:-1]).all()):
        raise ValueError("the law assumes a strictly increasing table")
    n_tab = tab.size
    scale = 1.0 / n_tab
    t0rel = tab - tab[0]                               # [0, D], increasing
    d_max = int(t0rel[-1])
    d1 = d_max + 1
    q_n = d_max // a + 1
    w_idx, w_lev = window(z, a, top_k, d_max)          # the window, in index order
    n_lev = np.bincount(w_lev, minlength=q_n).astype(np.int64)

    # g(y) = log F(T_0 + y) = log(#{j: T_j - T_0 <= y} / 2^B) on [0, D], in place in the zero-padded array
    gp = np.zeros(q_n * a)
    gp[t0rel] = 1.0
    np.cumsum(gp[:d1], out=gp[:d1])                    # integer counts, exact in float64
    gp[:d1] *= scale
    np.log(gp[:d1], out=gp[:d1])
    lm, used = log_max_cdf(n_lev, a, gp, d1, method)
    fft_err = None
    if used == "fft" and fft_check:
        rng = np.random.default_rng(0)
        occ = np.flatnonzero(n_lev)
        dev = 0.0
        for s in rng.integers(0, d1, fft_check).tolist():
            y = s + a * occ
            m = y <= d_max
            dev = max(dev, abs(float(np.dot(n_lev[occ[m]].astype(np.float64), gp[y[m]])) - float(lm[s])))
        fft_err = dev
    del gp
    s_lo = int(np.argmax(lm >= floor - math.log(n_tab)))       # LM(D) = 0, so it exists

    log_f = np.log(np.arange(1, n_tab + 1, dtype=np.float64) * scale)        # log F(T_i) = log((i + 1) / 2^B)
    with np.errstate(divide="ignore"):
        delta = -np.log1p(1.0 / np.arange(n_tab, dtype=np.float64))         # log(i / (i + 1)), -inf at i = 0
    c = np.zeros(d1)
    p = np.zeros(z.size)
    for v, k in zip(w_idx.tolist(), w_lev.tolist()):
        d = a * k
        i0 = int(np.searchsorted(t0rel, d + s_lo, side="left"))   # first i with T_i - T_0 - d >= s_lo
        if i0 == n_tab:
            continue
        s = t0rel[i0:] - d
        p[v] = float(np.exp(lm[s] - log_f[i0:] + c[s]).sum()) * scale
        c[s] += delta[i0:]                                  # s is strictly increasing: no repeated index
    info = {"window": int(w_idx.size), "levels": int(np.count_nonzero(n_lev)), "grid": d1, "method": used,
            "sum_err": abs(float(p.sum()) - 1.0), "fft_err": fft_err,
            "dropped_bound": float(w_idx.size * math.exp(floor))}
    return p, info


# ------------------------------------------------------------------ references

def softmax_on(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``softmax`` of ``x`` restricted to the indices ``keep`` (zero elsewhere), in float64."""
    out = np.zeros(x.size)
    xs = np.asarray(x, dtype=np.float64)[keep]
    e = np.exp(xs - xs.max())
    out[keep] = e / e.sum()
    return out


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


# ------------------------------------------------------------------ closed forms and the Monte Carlo

def two_token_exact(a: int, z0: int, z1: int, tab: np.ndarray) -> Fraction:
    """``P(choose = 0)`` for two tokens, by counting: token 0 wins ties, so it wins iff
    ``a z1 + T_j <= a z0 + T_i``: ``#{(i, j)} / 2^(2B)`` in exact integers."""
    n = tab.size
    hits = int(np.searchsorted(tab, tab + a * (z0 - z1), side="right").sum())
    return Fraction(hits, n * n)


def equal_tokens_exact(n_tok: int, tab_size: int) -> list[Fraction]:
    """``P(choose = v)`` for ``n_tok`` equal logits: ``2^(-n B) sum_i i^v (i + 1)^(n - 1 - v)`` (the earlier tokens
    must lose strictly, the later ones may tie), in exact rationals."""
    i = np.arange(tab_size, dtype=object)
    return [Fraction(int((i ** v * (i + 1) ** (n_tok - 1 - v)).sum()), tab_size ** n_tok) for v in range(n_tok)]


def monte_carlo(z: torch.Tensor, a: int, top_k: int, position: int, n: int, tag: bytes = MC_TAG) -> np.ndarray:
    """Counts of ``TokenSampler(seed_j, a, top_k).choose(z, position)`` over the seeds
    ``seed_j = SHA-256(tag || j)``, ``j < n``."""
    z = z.to("cpu", torch.int64)
    counts = np.zeros(z.numel(), dtype=np.int64)
    for j in range(n):
        seed = hashlib.sha256(tag + j.to_bytes(8, "little")).digest()
        counts[TokenSampler(seed, a, top_k).choose(z, position)] += 1
    return counts


def zscore(count: int, n: int, p: float) -> float:
    """The z-score of ``count`` draws out of ``n`` against probability ``p``: ``(count - n p) / sqrt(n p (1 - p))``.
    ``p`` is clipped to ``[0, 1]`` first (a float64 sum of a law can exceed 1 by its rounding, ``1e-11`` on flat
    logits), and a bucket that the reference makes certain or impossible (``n p (1 - p) < 1e-9``) scores 0 if its
    count is ``n p`` to within half a draw and infinity otherwise, instead of dividing a rounding error by zero."""
    p = min(max(float(p), 0.0), 1.0)
    var = n * p * (1.0 - p)
    if var < 1e-9:
        return 0.0 if abs(count - n * p) < 0.5 else math.inf
    return (count - n * p) / math.sqrt(var)


def compare_mc(p: np.ndarray, counts: np.ndarray, min_var: float = 9.0) -> dict:
    """Per-token 3-sigma comparison of Monte Carlo counts with the law.  Tokens with ``n p (1 - p) >= min_var`` are
    tested one by one, the rest pooled into one bucket (also tested); a token of probability 0 must never be drawn.
    ``expected_beyond`` is the number of 3-sigma exceedances expected by chance (with ``m`` buckets a correct law
    exceeds 3 sigma ``0.0027 m`` times on average, so "every token within 3 sigma" cannot be required of ``m`` in the
    hundreds); ``z_familywise_1pct`` is the two-sided Bonferroni bound ``Phi^-1(1 - 0.005 / m)`` that ``max_abs_z``
    stays below with probability at least 99% under the law, the pass criterion for many buckets."""
    n = int(counts.sum())
    var = n * p * (1 - p)
    each = var >= min_var
    zs = (counts[each] - n * p[each]) / np.sqrt(var[each])
    rest_p = float(p[~each].sum())
    rest_c = int(counts[~each].sum())
    rest_z = zscore(rest_c, n, rest_p)
    allz = np.append(zs, rest_z)
    chi2 = float((zs ** 2).sum()) + (rest_z ** 2 if math.isfinite(rest_z) else math.inf)
    return {"draws": n, "tested_tokens": int(each.sum()), "pooled_p": rest_p, "pooled_count": rest_c,
            "max_abs_z": float(np.abs(allz).max()), "beyond_3sigma": int((np.abs(allz) > 3).sum()),
            "expected_beyond": round(0.0027 * allz.size, 3),
            "z_familywise_1pct": NormalDist().inv_cdf(1.0 - 0.005 / allz.size), "chi2": chi2, "dof": int(allz.size) - 1,
            "zero_prob_draws": int(counts[p == 0].sum())}


def selfcheck(mc: int = 100_000) -> dict:
    """The toys: the 2-token closed form at gaps around the window's edge, 8 equal logits (exact rationals), and a
    Monte Carlo of ``TokenSampler.choose`` on 8 distinct logits; each law also against its Gumbel-max target."""
    tab = table()
    d_max = int(tab[-1] - tab[0])
    out = {"two_token": [], "eight_equal": {}, "eight_mc": {}}
    for a, dz in [(1, 0), (1, 3), (1, 70_000), (1, d_max - 1), (1, d_max), (1, d_max + 1), (4096, 2), (4096, -9)]:
        for z in ([dz, 0], [0, dz]):
            p, _ = sampler_law(np.array(z), a)
            exact = two_token_exact(a, z[0], z[1], tab)
            target = softmax_on(np.array(z, dtype=np.float64) * a / 65536.0, np.arange(2))
            out["two_token"].append({"a": a, "z": z, "law0": p[0], "exact0": float(exact),
                                     "err": abs(p[0] - float(exact)), "tv_target": tv(p, target)})
    p, info = sampler_law(np.zeros(8, dtype=np.int64), 1000)
    exact = [float(f) for f in equal_tokens_exact(8, tab.size)]
    out["eight_equal"] = {"law": p.tolist(), "exact": exact, "max_err": float(np.abs(p - exact).max()),
                          "tv_uniform": tv(p, np.full(8, 1 / 8)), "sum_err": info["sum_err"]}
    out["two_token_max_err"] = max(r["err"] for r in out["two_token"])
    if not mc:
        return out
    z8 = torch.tensor([0, -3, 5, 5, -1, 2, -7, 4], dtype=torch.int64)
    a8 = 1 << 15                                         # tau = 2 on unit logits
    p, info = sampler_law(z8, a8)
    t0 = time.perf_counter()
    counts = monte_carlo(z8, a8, 0, 7, mc)
    out["eight_mc"] = {"z": z8.tolist(), "a": a8, "law": p.tolist(), "mc": (counts / mc).tolist(),
                       **compare_mc(p, counts), "tv_target": tv(p, softmax_on(z8.numpy() * a8 / 65536.0,
                                                                              np.arange(8))),
                       "seconds": time.perf_counter() - t0}
    return out


# ------------------------------------------------------------------ logits of a real model

def _peak_rss_gb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except ImportError:                                  # Windows: the process's peak working set
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 2 ** 30
        except Exception:                                # noqa: BLE001 - best effort only
            pass
        return None


def extract_logits(args) -> dict:
    """The integer claims ``z`` (int32 ``[P, V]``), the fp32 checkpoint's logits (float32 ``[P, V]``), the logit
    scale and the token ids of the analysed positions; read from ``--logits-cache`` when it exists, else computed
    (and stored there)."""
    if args.logits_cache and os.path.exists(args.logits_cache):
        f = np.load(args.logits_cache, allow_pickle=False)
        meta = json.loads(str(f["meta"]))
        held = (meta["model"], meta["seq"], meta["windows"], meta["calib_seq"])
        if held != (args.model, args.seq, args.windows, args.calib_seq) or \
                (args.positions and args.positions != f["positions"].size):
            raise SystemExit(f"{args.logits_cache} holds {held} with {f['positions'].size} positions, not "
                             f"{(args.model, args.seq, args.windows, args.calib_seq)} with --positions {args.positions}")
        return {"z": f["z"], "fp": f["fp"], "positions": f["positions"], "meta": meta}
    if "opt" not in args.model.lower():
        raise SystemExit("only OPT has a real-weight integer builder (real_weights.build_opt_from_hf) so far")
    from common import load_opt, load_windows
    from pvi.fullcheck.real_weights import build_opt_from_hf, real_logits

    t0 = time.perf_counter()
    model, tok = load_opt(args.model)
    wins, _ = load_windows(tok, seq=args.seq, windows=args.windows, calib_windows=0)
    _, calib = load_windows(tok, seq=args.calib_seq, windows=0, calib_windows=1)
    dev = args.device
    with torch.no_grad():
        graph, info = build_opt_from_hf(model, calib[0].to(dev), device=dev, **SMOOTH)
    c = float(info["logit_scale"])
    total = args.seq * len(wins)
    keep = (np.arange(total) if not args.positions or args.positions >= total else
            np.unique(np.linspace(0, total - 1, args.positions).round().astype(np.int64)))
    zs, fps = [], []
    with torch.no_grad():
        for wi, w in enumerate(wins):
            rows = keep[(keep >= wi * args.seq) & (keep < (wi + 1) * args.seq)] - wi * args.seq
            if rows.size == 0:
                continue
            real = real_logits(graph, info, w.to(dev), all_positions=True)[0].cpu()     # [seq, V] = c z
            z = torch.round(real / c).to(torch.int64)
            if not torch.equal(z.double() * c, real):
                raise RuntimeError("the integer logits are not recovered exactly from real_logits")
            if int(z.abs().max()) >= 1 << 29:
                raise RuntimeError("a logit outside the sampler's range |z| < 2^29")
            zs.append(z[torch.from_numpy(rows)].to(torch.int32).numpy())
            fps.append(model(w).logits[0, torch.from_numpy(rows)].float().numpy())
            del real, z
    meta = {"model": args.model, "seq": args.seq, "windows": args.windows, "calib_seq": args.calib_seq,
            "logit_scale": c, "recipe": SMOOTH, "dataset": "wikitext-2-raw-v1 test",
            "extract_seconds": time.perf_counter() - t0, "extract_peak_rss_GB": _peak_rss_gb()}
    out = {"z": np.concatenate(zs), "fp": np.concatenate(fps), "positions": keep, "meta": meta}
    del graph, model
    if args.logits_cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.logits_cache)), exist_ok=True)
        np.savez(args.logits_cache, z=out["z"], fp=out["fp"], positions=keep, meta=json.dumps(meta))
    return out


# ------------------------------------------------------------------ the analysis

def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
            "max": float(a.max()), "n": int(a.size)}


_TABLES: dict[str, np.ndarray] = {}


def _table_for(spec: str) -> np.ndarray:
    """``table(*parse_table(spec))``, built once per process."""
    if spec not in _TABLES:
        _TABLES[spec] = table(*parse_table(spec))
    return _TABLES[spec]


def analyse_position(pos: int, z: np.ndarray, fp: np.ndarray, c: float, taus, topk, tables, fft_check: int,
                     floor: float) -> tuple[list[dict], list[dict]]:
    """The rows of one position: one per ``tau``, ``top_k`` and table (``tv_sampler``, ``tv_table``, ``tv_temp``,
    ``tail_mass`` and the law's diagnostics), and one ``tv_model`` row per ``tau`` and ``top_k``."""
    torch.set_num_threads(1)
    z = z.astype(np.int64)
    fp = fp.astype(np.float64)
    rows, model_rows = [], []
    for tau in taus:
        for k in topk:
            s_int, s_fp = candidates(z, k), candidates(fp, k)
            ideal = softmax_on(z * (c / tau), s_int)
            model_rows.append({"pos": pos, "tau": tau, "top_k": k, "tv_model": tv(ideal, softmax_on(fp / tau, s_fp))})
            for spec in tables:
                bits, vbits = parse_table(spec)
                tab = _table_for(spec)
                t0 = time.perf_counter()
                a = weight(c, tau, vbits)
                p, info = sampler_law(z, a, k, bits, tab=tab, fft_check=fft_check, floor=floor)
                target = softmax_on(z * (a / float(1 << vbits)), s_int)
                w_idx, _ = window(z, a, k, int(tab[-1] - tab[0]))
                rows.append({"pos": pos, "tau": tau, "top_k": k, "table": spec, "a": a,
                             "tv_sampler": tv(p, ideal), "tv_table": tv(p, target), "tv_temp": tv(target, ideal),
                             "tail_mass": 1.0 - float(target[w_idx].sum()), **info,
                             "seconds": time.perf_counter() - t0})
                del p, target
    return rows, model_rows


def analyse(data: dict, args) -> dict:
    """The per-position rows (:func:`analyse_position`, over ``--workers`` processes) and their summary per
    setting: mean, median, 90th percentile and maximum over positions."""
    z_all, fp_all, c = data["z"], data["fp"], data["meta"]["logit_scale"]
    jobs = [(int(data["positions"][i]), z_all[i], fp_all[i], c, args.taus, args.topk, args.tables, args.fft_check,
             args.floor) for i in range(z_all.shape[0])]
    rows, model_rows = [], []

    def collect(n, res):
        rows.extend(res[0])
        model_rows.extend(res[1])
        last = res[0][-1]
        print(f"position {n}/{len(jobs)} ({last['pos']}): tv_sampler {last['tv_sampler']:.3g} (tau {last['tau']}, "
              f"top-k {last['top_k']}, table {last['table']}, window {last['window']}); "
              f"{sum(r['seconds'] for r in res[0]):.1f} s", file=sys.stderr, flush=True)

    if args.workers > 1:                     # spawned, not forked: the parent has run torch's OpenMP pool
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for n, res in enumerate(ex.map(analyse_position, *zip(*jobs)), 1):
                collect(n, res)
    else:
        for n, job in enumerate(jobs, 1):
            collect(n, analyse_position(*job))
    summary = []
    for tau in args.taus:
        for k in args.topk:
            mrow = [r["tv_model"] for r in model_rows if r["tau"] == tau and r["top_k"] == k]
            for spec in args.tables:
                sel = [r for r in rows if r["tau"] == tau and r["top_k"] == k and r["table"] == spec]
                summary.append({"tau": tau, "top_k": k, "table": spec, "a": sorted({r["a"] for r in sel}),
                                **{key: _stats([r[key] for r in sel])
                                   for key in ("tv_sampler", "tv_table", "tv_temp", "tail_mass", "window")},
                                "tv_model": _stats(mrow),
                                "sum_err_max": max(r["sum_err"] for r in sel),
                                "fft_err_max": max([r["fft_err"] for r in sel if r["fft_err"] is not None],
                                                   default=None),
                                "dropped_bound_max": max(r["dropped_bound"] for r in sel),
                                "seconds_mean": float(np.mean([r["seconds"] for r in sel]))})
    return {"summary": summary, "rows": rows, "model_rows": model_rows}


GAP_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, math.inf)


def binned(z: np.ndarray, scale: float, refs: dict[str, np.ndarray], counts: np.ndarray) -> list[dict]:
    """Probability mass per bin of the scaled gap ``scale (z* - z_v)`` (``GAP_EDGES``): each reference law's mass,
    the Monte Carlo frequency, and the Monte Carlo's z-score against each reference.  It shows where the laws differ
    and that the draws side with the exact law."""
    n = int(counts.sum())
    gap = scale * (int(z.max()) - z)
    out = []
    for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
        m = (gap >= lo) & (gap < hi)
        hits = int(counts[m].sum())
        row = {"gap": [lo, hi], "tokens": int(m.sum()), "mc": hits / n}
        for name, q in refs.items():
            mass = float(q[m].sum())
            row[name] = mass
            row[f"z_{name}"] = zscore(hits, n, mass)
        out.append(row)
    return out


def run_mc(data: dict, args) -> list[dict]:
    """``--mc`` draws of ``TokenSampler.choose`` at the ``--mc-positions``-th analysed positions, for each
    ``--mc-settings`` ``tau:top_k`` (the protocol's 16-bit table only), each against the exact law (per token) and,
    per gap bin, against the law and the ideal ``softmax_S(c z / tau)``."""
    out = []
    c = data["meta"]["logit_scale"]
    for pi in args.mc_positions:
        z_np = data["z"][pi].astype(np.int64)
        z = torch.from_numpy(z_np)
        pos = int(data["positions"][pi])
        for setting in args.mc_settings:
            tau, k = float(setting.split(":")[0]), int(setting.split(":")[1])
            a = weight(c, tau)
            p, _ = sampler_law(z, a, k)
            ideal = softmax_on(z_np * (c / tau), candidates(z_np, k))
            t0 = time.perf_counter()
            counts = monte_carlo(z, a, k, pos, args.mc)
            out.append({"pos": pos, "tau": tau, "top_k": k, "a": a, **compare_mc(p, counts),
                        "tv_law_vs_mc": tv(p, counts / args.mc), "tv_ideal_vs_mc": tv(ideal, counts / args.mc),
                        "bins": binned(z_np, c / tau, {"law": p, "ideal": ideal}, counts),
                        "seconds": time.perf_counter() - t0})
            print(f"Monte Carlo at position {pos}, tau {tau}, top-k {k}: max |z| {out[-1]['max_abs_z']:.2f} "
                  f"(family-wise 1% bound {out[-1]['z_familywise_1pct']:.2f}), {out[-1]['beyond_3sigma']} beyond "
                  f"3 sigma (expected {out[-1]['expected_beyond']})", file=sys.stderr, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--seq", type=int, default=128, help="tokens per evaluation window")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--positions", type=int, default=0, help="analyse this many evenly spaced positions (0: all)")
    ap.add_argument("--first", type=int, default=0, help="analyse only the first N extracted positions (0: all)")
    ap.add_argument("--calib-seq", type=int, default=128, help="tokens of the calibration window")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    ap.add_argument("--topk", type=int, nargs="+", default=[0, 50])
    ap.add_argument("--tables", nargs="+", default=["16", "16:20", "20"],
                    help="tables B (2^B entries, resolution 2^-B) or B:R (resolution 2^-R); 16 is the protocol's")
    ap.add_argument("--fft-check", type=int, default=256, help="grid points checked against the direct sum")
    ap.add_argument("--floor", type=float, default=-60.0,
                    help="skip integrands below e^floor (each law loses at most window x e^floor; -746: none)")
    ap.add_argument("--workers", type=int, default=1, help="processes over positions (each about 0.6 GB at 20 bits)")
    ap.add_argument("--mc", type=int, default=0, help="Monte Carlo draws of TokenSampler.choose per setting")
    ap.add_argument("--mc-positions", type=int, nargs="+", default=[0], help="indices into the analysed positions")
    ap.add_argument("--mc-settings", nargs="+", default=["1:0", "1:50"], help="tau:top_k pairs for the Monte Carlo")
    ap.add_argument("--selfcheck", action="store_true", help="only the toys (closed forms, Monte Carlo); no model")
    ap.add_argument("--selfcheck-mc", type=int, default=100_000)
    ap.add_argument("--extract-only", action="store_true", help="only compute and cache the logits")
    ap.add_argument("--logits-cache", default=None, help=".npz with the integer and fp32 logits (read or written)")
    ap.add_argument("--device", default="cpu", help="where the integer graph runs (the fp32 model stays on the CPU)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    doc = {"script": "sampler_tv.py", "host": platform.node(), "args": vars(args)}
    if args.selfcheck:
        doc["toys"] = selfcheck(args.selfcheck_mc)
    else:
        data = extract_logits(args)
        if args.first:
            data = {**data, "z": data["z"][:args.first], "fp": data["fp"][:args.first],
                    "positions": data["positions"][:args.first]}
        doc["meta"] = data["meta"]
        doc["positions"] = data["positions"].tolist()
        if not args.extract_only:
            doc["toys"] = selfcheck(args.selfcheck_mc)
            doc.update(analyse(data, args))
            if args.mc:
                doc["mc"] = run_mc(data, args)
    doc["seconds"] = time.perf_counter() - t0
    doc["peak_rss_GB"] = _peak_rss_gb()
    text = json.dumps(doc)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    brief = {k: v for k, v in doc.items() if k not in ("rows", "model_rows")}
    print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
