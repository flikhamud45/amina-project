"""Seeded bootstrap intervals of the reported ratios and differences (each record reproducible from its samples)."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["ppl_ratio_ci", "paired_diff_ci", "median_ci"]


def ppl_ratio_ci(nll_a: list[float], nll_b: list[float], counts: list[int], reps: int = 2000) -> list[float]:
    """95% interval of the perplexity ratio ``exp(sum nll_a / sum n - sum nll_b / sum n)`` over windows resampled with
    replacement (paired: one draw per window for both), from per-window summed negative log-likelihoods and token
    counts."""
    g = torch.Generator().manual_seed(0)
    a, b = torch.tensor(nll_a, dtype=torch.float64), torch.tensor(nll_b, dtype=torch.float64)
    n = torch.tensor(counts, dtype=torch.float64)
    idx = torch.randint(0, len(a), (reps, len(a)), generator=g)
    r = torch.exp(a[idx].sum(1) / n[idx].sum(1) - b[idx].sum(1) / n[idx].sum(1))
    return [float(r.quantile(0.025)), float(r.quantile(0.975))]


def paired_diff_ci(a: list[int], b: list[int], reps: int = 2000) -> list[float]:
    """95% interval of ``mean(a) - mean(b)`` over items resampled with replacement (paired)."""
    g = torch.Generator().manual_seed(0)
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    idx = torch.randint(0, len(x), (reps, len(x)), generator=g)
    m = x[idx].mean(1)
    return [float(m.quantile(0.025)), float(m.quantile(0.975))]


def median_ci(xs, ys=None, n: int = 10_000, seed: int = 0) -> dict:
    """The median of ``xs`` (``{"median", "ci95"}``) or, with ``ys``, the ratio of the medians of ``xs`` and ``ys``
    (``{"ratio", "ci95"}``), with the 2.5% and 97.5% percentiles of ``n`` bootstrap resamples (each sample resampled
    with replacement on its own)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(xs, dtype=np.float64)
    mx = np.median(x[rng.integers(0, x.size, (n, x.size))], axis=1)
    if ys is None:
        lo, hi = np.percentile(mx, [2.5, 97.5])
        return {"median": float(np.median(x)), "ci95": [float(lo), float(hi)]}
    y = np.asarray(ys, dtype=np.float64)
    my = np.median(y[rng.integers(0, y.size, (n, y.size))], axis=1)
    lo, hi = np.percentile(mx / my, [2.5, 97.5])
    return {"ratio": float(np.median(x) / np.median(y)), "ci95": [float(lo), float(hi)]}
