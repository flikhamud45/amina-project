"""A native (C++, OpenMP) exact integer causal attention for a CPU verifier: the integers of
``transformer._attention_core`` without its ``T x T`` tensors and their dozen element-wise passes.

Per (batch, head, query row), on the verifier's threads: the int32 scores against the keys up to the query's
position, their maximum, ``e = LUT[min(max - s, len - 1)]``, ``tot = sum e``, ``p = floor((tot + 510 e) / (2 tot))``
and ``p v`` (int64).  Every step is integer arithmetic on the same values as ``_attention_core`` (scores below
``2**21``, ``tot`` and ``510 e + tot`` below ``2**31``, as ``transformer._int32_scores`` requires), so the result is
bit-identical (tested).  Query rows are taken eight at a time, so each key and value row is read once per tile of
rows rather than once per row.

The library is compiled once per host with the C++ compiler in ``$CXX`` (else ``g++``), ``-O3 -march=native
-fopenmp``, into ``$PVI_NATIVE_DIR`` (default ``~/.cache/pvi``) and loaded with ctypes: no Python or torch headers
are needed.  Without a working compiler :func:`available` is false and the torch path is used.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

__all__ = ["available", "attention_core", "attention_core_hybrid", "cheap_enabled", "requant", "residual",
           "lut", "norm", "rope", "field_matmul"]

_SOURCE = r"""
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>

// Query rows are taken TR at a time (rows of one (batch, head) group): each key and value row is read once per
// tile instead of once per query row.  Every integer is computed exactly as for one row at a time.
static const int64_t TR = 8;

extern "C" void pvi_attention(const int16_t* q, const int16_t* k, const int16_t* v, const int32_t* lut,
                              int64_t* out, int64_t bh, int64_t rows, int64_t queries, int64_t t, int64_t dh,
                              int64_t lut_len, int64_t threads) {
    omp_set_num_threads((int)threads);
    const int64_t tiles = (rows + TR - 1) / TR;
    #pragma omp parallel
    {
        std::vector<int32_t> s(TR * t), e(TR * t);
        std::vector<int64_t> acc(TR * dh);
        int64_t n[TR];
        int32_t mx[TR], tot[TR];
        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (int64_t g = 0; g < bh; ++g) {
            for (int64_t tile = 0; tile < tiles; ++tile) {
                const int64_t r0 = tile * TR;
                const int64_t tr = std::min(TR, rows - r0);
                const int16_t* kg = k + g * t * dh;
                const int16_t* vg = v + g * t * dh;
                int64_t nmax = 0;
                for (int64_t i = 0; i < tr; ++i) {
                    n[i] = t - queries + ((r0 + i) % queries) + 1;      // keys 0 .. the query's position
                    nmax = std::max(nmax, n[i]);
                    mx[i] = INT32_MIN;
                }
                for (int64_t j = 0; j < nmax; ++j) {
                    const int16_t* kj = kg + j * dh;
                    for (int64_t i = 0; i < tr; ++i) {
                        if (j >= n[i]) continue;
                        const int16_t* qr = q + (g * rows + r0 + i) * dh;
                        int32_t acc32 = 0;
                        #pragma omp simd reduction(+:acc32)
                        for (int64_t d = 0; d < dh; ++d) acc32 += (int32_t)qr[d] * (int32_t)kj[d];
                        s[i * t + j] = acc32;
                        mx[i] = std::max(mx[i], acc32);
                    }
                }
                for (int64_t i = 0; i < tr; ++i) {
                    int32_t total = 0;
                    for (int64_t j = 0; j < n[i]; ++j) {
                        int64_t gap = (int64_t)mx[i] - (int64_t)s[i * t + j];
                        if (gap > lut_len - 1) gap = lut_len - 1;
                        e[i * t + j] = lut[gap];
                        total += e[i * t + j];
                    }
                    tot[i] = total;
                }
                std::fill(acc.begin(), acc.end(), 0);
                for (int64_t j = 0; j < nmax; ++j) {
                    const int16_t* vj = vg + j * dh;
                    for (int64_t i = 0; i < tr; ++i) {
                        if (j >= n[i]) continue;
                        const int32_t num = tot[i] + 510 * e[i * t + j];
                        const int32_t pj = num / (2 * tot[i]);          // both positive: floor division
                        if (pj == 0) continue;
                        int64_t* a = acc.data() + i * dh;
                        #pragma omp simd
                        for (int64_t d = 0; d < dh; ++d) a[d] += (int64_t)pj * (int64_t)vj[d];
                    }
                }
                for (int64_t i = 0; i < tr; ++i) {
                    int64_t* o = out + (g * rows + r0 + i) * dh;
                    for (int64_t d = 0; d < dh; ++d) o[d] = acc[i * dh + d];
                }
            }
        }
    }
}

// The integer softmax of a block of rows, in place: s holds exact integer scores as float32 ([n_rows, t], row i of
// the block is query row r0 + i of its (batch, head) group, which sees keys 0 .. t - queries + (r0 + i) % queries);
// on return it holds p = floor((tot + 510 e) / (2 tot)) as float32 (0 for the masked keys), e = LUT[min(max - s, len - 1)].
extern "C" void pvi_softmax_block(float* s, const int32_t* lut, int64_t groups, int64_t n_rows, int64_t r0,
                                  int64_t queries, int64_t t, int64_t lut_len, int64_t threads) {
    omp_set_num_threads((int)threads);
    #pragma omp parallel
    {
        std::vector<int32_t> e(t);
        #pragma omp for schedule(static) collapse(2)
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t i = 0; i < n_rows; ++i) {
                float* row = s + (g * n_rows + i) * t;
                const int64_t n = t - queries + ((r0 + i) % queries) + 1;
                int32_t mx = INT32_MIN;
                for (int64_t j = 0; j < n; ++j) mx = std::max(mx, (int32_t)row[j]);
                int32_t tot = 0;
                for (int64_t j = 0; j < n; ++j) {
                    int64_t gap = (int64_t)mx - (int64_t)(int32_t)row[j];
                    if (gap > lut_len - 1) gap = lut_len - 1;
                    e[j] = lut[gap];
                    tot += e[j];
                }
                const int32_t two = 2 * tot;
                for (int64_t j = 0; j < n; ++j) row[j] = (float)((tot + 510 * e[j]) / two);
                for (int64_t j = n; j < t; ++j) row[j] = 0.0f;
            }
        }
    }
}
// ---- the other recomputed operations, one pass each (int64 in and out, as the torch ops) ----------------------
// A matrix operand is [R, C] with strides (sr, sc): contiguous rows (sc == 1) or a transposed claim (sr == 1, the
// fold's view of Z [C, R]); the output is contiguous.  A transposed operand is read in 64 x 64 tiles, so both the
// reads and the writes stay in cache.  The library is built with -fwrapv: int64 arithmetic wraps as torch's does.

static inline int64_t clamp64(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int64_t floordiv64(int64_t a, int64_t b) {           // b > 0
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}
static const int64_t TB = 64;

template <class F>
static void each2d(int64_t R, int64_t C, int64_t sr, int64_t sc, F f) {   // f(i, j, source offset)
    if (sc == 1) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < R; ++i)
            for (int64_t j = 0; j < C; ++j) f(i, j, i * sr + j);
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t ib = 0; ib < R; ib += TB)
            for (int64_t jb = 0; jb < C; jb += TB) {
                const int64_t i1 = std::min(R, ib + TB), j1 = std::min(C, jb + TB);
                for (int64_t j = jb; j < j1; ++j)
                    for (int64_t i = ib; i < i1; ++i) f(i, j, i * sr + j * sc);
            }
    }
}

// clamp((z * mult + 2**(shift-1)) >> shift, lo, hi), times b (contiguous like the output) first when b != nullptr
extern "C" void pvi_requant(const int64_t* z, const int64_t* b, int64_t* out, int64_t R, int64_t C, int64_t sr,
                            int64_t sc, int64_t mult, int64_t shift, int64_t lo, int64_t hi, int64_t threads) {
    omp_set_num_threads((int)threads);
    const int64_t half = (int64_t)1 << (shift - 1);
    if (b == nullptr)
        each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) {
            out[i * C + j] = clamp64((z[s] * mult + half) >> shift, lo, hi); });
    else
        each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) {
            out[i * C + j] = clamp64((z[s] * b[i * C + j] * mult + half) >> shift, lo, hi); });
}

// clamp(a + ((z * mult + 2**(shift-1)) >> shift), -res_max, res_max), a contiguous
extern "C" void pvi_residual(const int64_t* a, const int64_t* z, int64_t* out, int64_t R, int64_t C, int64_t sr,
                             int64_t sc, int64_t mult, int64_t shift, int64_t res_max, int64_t threads) {
    omp_set_num_threads((int)threads);
    const int64_t half = (int64_t)1 << (shift - 1);
    each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) {
        out[i * C + j] = clamp64(a[i * C + j] + ((z[s] * mult + half) >> shift), -res_max, res_max); });
}

// table[clamp(x, -128, 127) + 128]
extern "C" void pvi_lut(const int64_t* x, const int64_t* table, int64_t* out, int64_t R, int64_t C, int64_t sr,
                        int64_t sc, int64_t threads) {
    omp_set_num_threads((int)threads);
    each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) { out[i * C + j] = table[clamp64(x[s], -128, 127) + 128]; });
}

// transformer._norm_int over rows of d (contiguous): returns 1, or 0 (nothing written) when a row's sum of squares
// wrapped negative, where the torch path's own rules apply.
extern "C" int64_t pvi_norm(const int64_t* x, const int64_t* gain, int64_t* out, int64_t rows, int64_t d,
                            int64_t center, int64_t k, int64_t threads) {
    omp_set_num_threads((int)threads);
    std::vector<int64_t> mean(rows, 0), sigma(rows, 1);
    int64_t bad = 0;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t* xi = x + i * d;
        int64_t m = 0;
        if (center) {
            int64_t s = 0;
            for (int64_t j = 0; j < d; ++j) s += xi[j];
            m = floordiv64(s + d / 2, d);
        }
        int64_t q = 0;
        for (int64_t j = 0; j < d; ++j) { const int64_t v = xi[j] - m; q += v * v; }
        q = floordiv64(q, d);
        if (q < 0) { bad = 1; continue; }
        int64_t r = (int64_t)std::floor(std::sqrt((double)q));      // _isqrt: one correction each way, at least 1
        if ((r + 1) * (r + 1) <= q) r += 1;
        if (r * r > q) r -= 1;
        mean[i] = m;
        sigma[i] = std::max<int64_t>(r, 1);
    }
    if (bad) return 0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t* xi = x + i * d;
        int64_t* o = out + i * d;
        const int64_t m = mean[i], sg = sigma[i], den = sg << k, add = sg * ((int64_t)1 << (k - 1));
        for (int64_t j = 0; j < d; ++j) o[j] = clamp64(floordiv64((xi[j] - m) * gain[j] + add, den), -127, 127);
    }
    return 1;
}

// transformer._rope: x [rows = B*T*H, dh] contiguous, position t = (row / heads) % T; cos2, sin2 [T, dh]
extern "C" void pvi_rope(const int64_t* x, const int64_t* cos2, const int64_t* sin2, int64_t* out, int64_t rows,
                         int64_t t, int64_t heads, int64_t dh, int64_t threads) {
    omp_set_num_threads((int)threads);
    const int64_t h = dh / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t pos = (r / heads) % t;
        const int64_t* xr = x + r * dh;
        const int64_t* c = cos2 + pos * dh;
        const int64_t* s = sin2 + pos * dh;
        int64_t* o = out + r * dh;
        for (int64_t j = 0; j < dh; ++j) {
            const int64_t sw = j < h ? xr[j + h] : xr[j - h];
            o[j] = clamp64((((int64_t)1 << 13) + xr[j] * c[j] + sw * s[j]) >> 14, -127, 127);
        }
    }
}
// ---- field products of the checks: out = chi @ Z mod p ------------------------------------------------------------
// chi [r, n] in [0, p), Z an [n, m] view with strides (sr, sc) of integers |z| < 2**31.  chi = c0 + 2**16 c1 with
// c0 < 2**16, c1 < 2**15, so every product is below 2**47 in magnitude and 2**15 of them sum below 2**62: the
// accumulators are reduced mod p every 2**15 terms and every partial sum is exact.  Returns 1, or 0 (nothing
// meaningful written) when an operand is out of range, where the caller's own products apply.
static const int64_t PF = 2013265921;           // 15 * 2**27 + 1
static const int64_t ZMAX = (int64_t)1 << 31;
static const int64_t RED = (int64_t)1 << 15;
static inline int64_t modp(int64_t v) { v %= PF; return v < 0 ? v + PF : v; }

extern "C" int64_t pvi_field_matmul(const int64_t* chi, int64_t r, int64_t n, const int64_t* z, int64_t m,
                                    int64_t sr, int64_t sc, int64_t* out, int64_t threads) {
    omp_set_num_threads((int)threads);
    for (int64_t i = 0; i < r * n; ++i) if (chi[i] < 0 || chi[i] >= PF) return 0;
    const int64_t L = 2 * r;
    int64_t bad = 0;
    if (sc == 1) {                                   // rows of Z contiguous: stream them, limbs as [n][L]
        std::vector<int64_t> lim(n * L);
        for (int64_t l = 0; l < r; ++l)
            for (int64_t i = 0; i < n; ++i) {
                lim[i * L + 2 * l] = chi[l * n + i] & 0xFFFF;
                lim[i * L + 2 * l + 1] = chi[l * n + i] >> 16;
            }
        const int64_t cb = std::max<int64_t>(64, std::min<int64_t>(512, (m + 2 * threads - 1) / (2 * threads)));
        #pragma omp parallel reduction(|:bad)
        {
            std::vector<int64_t> acc(L * cb);
            #pragma omp for schedule(dynamic, 1)
            for (int64_t j0 = 0; j0 < m; j0 += cb) {
                const int64_t w = std::min(cb, m - j0);
                std::fill(acc.begin(), acc.end(), 0);
                for (int64_t i = 0; i < n; ++i) {
                    const int64_t* zi = z + i * sr + j0;
                    int64_t over = 0;
                    #pragma omp simd reduction(|:over)
                    for (int64_t j = 0; j < w; ++j) over |= (zi[j] >= ZMAX) | (zi[j] <= -ZMAX);
                    bad |= over;
                    const int64_t* li = lim.data() + i * L;
                    for (int64_t l = 0; l < L; ++l) {
                        const int64_t c = li[l];
                        int64_t* a = acc.data() + l * cb;
                        #pragma omp simd
                        for (int64_t j = 0; j < w; ++j) a[j] += c * zi[j];
                    }
                    if ((i + 1) % RED == 0)
                        for (int64_t q = 0; q < L * cb; ++q) acc[q] %= PF;
                }
                for (int64_t l = 0; l < r; ++l)
                    for (int64_t j = 0; j < w; ++j)
                        out[l * m + j0 + j] = modp(modp(acc[2 * l * cb + j]) + modp(acc[(2 * l + 1) * cb + j]) * 65536);
            }
        }
    } else if (sr == 1) {                            // columns of Z contiguous: dot products, limbs as [L][n]
        std::vector<int64_t> lim(L * n);
        for (int64_t l = 0; l < r; ++l)
            for (int64_t i = 0; i < n; ++i) {
                lim[2 * l * n + i] = chi[l * n + i] & 0xFFFF;
                lim[(2 * l + 1) * n + i] = chi[l * n + i] >> 16;
            }
        #pragma omp parallel for schedule(static) reduction(|:bad)
        for (int64_t j = 0; j < m; ++j) {
            const int64_t* zj = z + j * sc;
            int64_t over = 0;
            #pragma omp simd reduction(|:over)
            for (int64_t i = 0; i < n; ++i) over |= (zj[i] >= ZMAX) | (zj[i] <= -ZMAX);
            bad |= over;
            for (int64_t l = 0; l < r; ++l) {
                int64_t lo = 0, hi = 0;
                for (int64_t i0 = 0; i0 < n; i0 += RED) {
                    const int64_t i1 = std::min(n, i0 + RED);
                    int64_t s0 = 0, s1 = 0;
                    const int64_t* a0 = lim.data() + 2 * l * n;
                    const int64_t* a1 = lim.data() + (2 * l + 1) * n;
                    #pragma omp simd reduction(+:s0, s1)
                    for (int64_t i = i0; i < i1; ++i) { s0 += a0[i] * zj[i]; s1 += a1[i] * zj[i]; }
                    lo = modp(lo + modp(s0));
                    hi = modp(hi + modp(s1));
                }
                out[l * m + j] = modp(lo + hi * 65536);
            }
        }
    } else {
        return 0;
    }
    return bad ? 0 : 1;
}
"""

_FLAGS = ("-O3", "-march=native", "-fopenmp", "-fwrapv", "-shared", "-fPIC")
_LIB = None
_TRIED = False


def _library():
    """The compiled library (once per process; the build is cached per host and source)."""
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("PVI_NATIVE_ATTN", "1") == "0" or platform.system() != "Linux":
        return None
    cxx = os.environ.get("CXX") or shutil.which("g++")
    if not cxx:
        return None
    key = hashlib.sha256((_SOURCE + " ".join(_FLAGS) + cxx + platform.node() + platform.processor()).encode()).hexdigest()[:16]
    where = Path(os.environ.get("PVI_NATIVE_DIR", Path.home() / ".cache" / "pvi"))
    lib = where / f"pvi_attention_{key}.so"
    try:
        if not lib.exists():
            where.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "attn.cpp"
                src.write_text(_SOURCE)
                built = Path(tmp) / lib.name
                subprocess.run([cxx, *_FLAGS, str(src), "-o", str(built)],
                               check=True, capture_output=True)
                shutil.move(str(built), str(lib))
        _LIB = ctypes.CDLL(str(lib))
        _LIB.pvi_attention.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 7
        _LIB.pvi_attention.restype = None
        _LIB.pvi_softmax_block.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 7
        _LIB.pvi_softmax_block.restype = None
        _LIB.pvi_requant.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 9
        _LIB.pvi_residual.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 8
        _LIB.pvi_lut.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5
        _LIB.pvi_norm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5
        _LIB.pvi_norm.restype = ctypes.c_int64
        _LIB.pvi_rope.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 5
        for f in (_LIB.pvi_requant, _LIB.pvi_residual, _LIB.pvi_lut, _LIB.pvi_rope):
            f.restype = None
        _LIB.pvi_field_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                                          ctypes.c_int64]
        _LIB.pvi_field_matmul.restype = ctypes.c_int64
    except (OSError, subprocess.CalledProcessError):
        _LIB = None
    return _LIB


def available() -> bool:
    return _library() is not None


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lut: torch.Tensor, queries: int) -> torch.Tensor:
    """``_attention_core(q, k, v, lut, notmask)`` (raw ``p v``, int64 ``[B, H, R, dh]``) on the CPU for int8-valued
    ``q [B, H, R, dh]`` (row ``r``: the query at position ``T - queries + r % queries``), ``k, v [B, H, T, dh]`` and
    the int32 table ``lut``."""
    lib = _library()
    bsz, heads, rows, dh = q.shape
    t = k.shape[2]
    q16, k16, v16 = (x.to(torch.int16).contiguous() for x in (q, k, v))
    lut32 = lut.to(torch.int32).contiguous()
    out = torch.empty(bsz, heads, rows, dh, dtype=torch.int64)
    lib.pvi_attention(q16.data_ptr(), k16.data_ptr(), v16.data_ptr(), lut32.data_ptr(), out.data_ptr(),
                      bsz * heads, rows, queries, t, dh, lut32.numel(), torch.get_num_threads())
    return out


_ROWS = 32
"""Query rows per block of :func:`attention_core_hybrid` (scores of 32 rows x 2,048 keys x the heads: a few MB)."""


def attention_core_hybrid(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lut: torch.Tensor, queries: int) -> torch.Tensor:
    """:func:`attention_core`'s integers with the two products as float32 GEMMs (torch's BLAS) and the integer
    softmax in one native pass per block of query rows: the scores are exact in float32 (``|s| <= 2**21``) and so is
    ``P V`` (every partial sum below ``(255 + T/2) 128 < 2**24``), as in ``transformer._attention_core``, but no
    ``T x T`` tensor is held and the element-wise steps are one pass over a cache-resident block."""
    lib = _library()
    bsz, heads, rows, dh = q.shape
    t = k.shape[2]
    qf = q.to(torch.float32).reshape(bsz * heads, rows, dh)
    kt = k.to(torch.float32).reshape(bsz * heads, t, dh).transpose(1, 2).contiguous()
    vf = v.to(torch.float32).reshape(bsz * heads, t, dh)
    lut32 = lut.to(torch.int32).contiguous()
    out = torch.empty(bsz * heads, rows, dh, dtype=torch.int64)
    threads = torch.get_num_threads()
    for r0 in range(0, rows, _ROWS):
        r1 = min(rows, r0 + _ROWS)
        s = torch.bmm(qf[:, r0:r1], kt).contiguous()                       # [BH, rb, T], exact integers
        lib.pvi_softmax_block(s.data_ptr(), lut32.data_ptr(), bsz * heads, r1 - r0, r0, queries, t,
                              lut32.numel(), threads)
        out[:, r0:r1] = torch.bmm(s, vf).to(torch.int64)                   # exact: sums < 2**24
    return out.view(bsz, heads, rows, dh)


# -- the other recomputed operations ------------------------------------------------------------------------------
# Each returns the torch op's int64 result (contiguous), or None when its operands are not the plain case (then the
# caller runs the torch steps): CPU int64 tensors, a matrix operand contiguous or a transposed claim view.

_CHEAP = os.environ.get("PVI_NATIVE_CHEAP", "1") != "0"


def cheap_enabled(*ts: torch.Tensor) -> bool:
    """Whether the native kernels apply to these operands (CPU, int64, non-empty) and the library is there."""
    return (_CHEAP and all(t.device.type == "cpu" and t.dtype == torch.int64 and t.numel() for t in ts)
            and _library() is not None)


def _layout(t: torch.Tensor):
    """``(R, C, sr, sc)`` of ``t`` as a matrix over its last axis: contiguous, or the transposed view of a
    contiguous ``[C, R]`` (a weight op's fold, with size-1 leading axes); else None."""
    c = t.shape[-1]
    r = t.numel() // c
    if t.is_contiguous():
        return r, c, c, 1
    lead = [(n, st) for n, st in zip(t.shape[:-1], t.stride()[:-1]) if n != 1]
    if len(lead) == 1 and lead[0][1] == 1 and t.stride(-1) == lead[0][0]:
        return r, c, 1, t.stride(-1)
    return None


def _scalar(m) -> int | None:
    if torch.is_tensor(m):
        return int(m) if m.numel() == 1 and m.dtype == torch.int64 and m.device.type == "cpu" else None
    return int(m) if isinstance(m, int) else None


def requant(z: torch.Tensor, mult, shift: int, lo: int, hi: int, b: torch.Tensor | None = None):
    """``graph.requant(z * b if b else z, mult, shift, lo, hi)``."""
    m, lay = _scalar(mult), _layout(z)
    if m is None or lay is None or (b is not None and (b.shape != z.shape or not b.is_contiguous())):
        return None
    out = torch.empty(z.shape, dtype=torch.int64)
    _LIB.pvi_requant(z.data_ptr(), None if b is None else b.data_ptr(), out.data_ptr(), *lay, m, shift, lo, hi,
                     torch.get_num_threads())
    return out


def residual(a: torch.Tensor, z: torch.Tensor, mult, shift: int, res_max: int):
    """``graph.residual_add(a, z, mult, shift, res_max)``."""
    m, lay = _scalar(mult), _layout(z)
    if m is None or lay is None or a.shape != z.shape or not a.is_contiguous():
        return None
    out = torch.empty(z.shape, dtype=torch.int64)
    _LIB.pvi_residual(a.data_ptr(), z.data_ptr(), out.data_ptr(), *lay, m, shift, res_max, torch.get_num_threads())
    return out


def lut(x: torch.Tensor, table: torch.Tensor):
    """``transformer._lut(x, table)`` for a 256-entry table."""
    lay = _layout(x)
    if lay is None or table.numel() != 256 or table.dtype != torch.int64 or table.device.type != "cpu":
        return None
    out = torch.empty(x.shape, dtype=torch.int64)
    _LIB.pvi_lut(x.data_ptr(), table.contiguous().data_ptr(), out.data_ptr(), *lay, torch.get_num_threads())
    return out


def norm(x: torch.Tensor, gain: torch.Tensor, center: bool, k: int):
    """``transformer._norm_int(x, gain, center, k)`` over the last axis."""
    d = x.shape[-1]
    if (not x.is_contiguous() or gain.dtype != torch.int64 or gain.device.type != "cpu" or gain.numel() != d
            or not 1 <= k <= 32):
        return None
    out = torch.empty(x.shape, dtype=torch.int64)
    ok = _LIB.pvi_norm(x.data_ptr(), gain.contiguous().data_ptr(), out.data_ptr(), x.numel() // d, d, int(center),
                       k, torch.get_num_threads())
    return out if ok else None


def rope(x: torch.Tensor, cos2: torch.Tensor, sin2: torch.Tensor):
    """``transformer._rope``'s integers for ``x [B, T, H, dh]`` and its tables ``[1, T, 1, dh]``."""
    if x.dim() != 4 or not x.is_contiguous() or cos2.device.type != "cpu" or cos2.dtype != torch.int64:
        return None
    _, t, heads, dh = x.shape
    c, s = cos2.reshape(t, dh).contiguous(), sin2.reshape(t, dh).contiguous()
    out = torch.empty(x.shape, dtype=torch.int64)
    _LIB.pvi_rope(x.data_ptr(), c.data_ptr(), s.data_ptr(), out.data_ptr(), x.numel() // dh, t, heads, dh,
                  torch.get_num_threads())
    return out


def field_matmul(chi: torch.Tensor, z: torch.Tensor):
    """``chi @ z mod P`` (int64 ``[r, m]``, canonical) for ``chi [r, n]`` in the field and an int64 matrix ``z
    [n, m]`` (contiguous rows or contiguous columns) with ``|z| < 2**31``, exactly; None when the operands are
    not of that kind or out of range (the caller's products then apply)."""
    if not (cheap_enabled(chi, z) and chi.dim() == 2 and z.dim() == 2 and chi.shape[1] == z.shape[0]):
        return None
    sr, sc = z.stride()
    if not (sc == 1 or sr == 1):
        return None
    chi = chi.contiguous()
    out = torch.empty(chi.shape[0], z.shape[1], dtype=torch.int64)
    ok = _LIB.pvi_field_matmul(chi.data_ptr(), chi.shape[0], chi.shape[1], z.data_ptr(), z.shape[1], sr, sc,
                               out.data_ptr(), torch.get_num_threads())
    return out if ok else None

