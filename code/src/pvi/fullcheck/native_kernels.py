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
           "lut", "norm", "rope", "field_matmul", "unpack_stream", "min_max"]

_SOURCE = r"""
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <omp.h>

// Query rows are taken TR at a time (rows of one (batch, head) group): each key and value row is read once per
// tile instead of once per query row.  Every integer is computed exactly as for one row at a time.
static const int64_t TR = 8;

static void attention_scalar(const int16_t* q, const int16_t* k, const int16_t* v, const int32_t* lut,
                             int64_t* out, int64_t bh, int64_t rows, int64_t queries, int64_t t, int64_t dh,
                             int64_t lut_len) {
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

#if defined(__AVX512BW__)
#include <immintrin.h>
// The same integers on AVX-512 (vpmaddwd: int16 pairs -> int32 sums), for dh a multiple of 16 up to 128.  Per
// (batch, head) group: the keys as int16 pairs along dh, transposed to [dh/2][T] (16 keys per vector), and the
// values as int16 pairs of consecutive keys, [T/2][dh].  Query rows go in tiles of 16: the scores of 16 rows x 16
// keys are 16 accumulators (|s| <= 2**21), the softmax runs per row as in pvi_attention, the probabilities are
// packed as pairs of consecutive keys, and P V accumulates in int32 per row (every partial sum is below
// (255 + T/2) 128 < 2**24 for T < 2**15), widened to int64 on the way out.
static const int64_t TQ = 16;
static int g_prof_on = 0;                  // pvi_attention_profile: seconds per phase, summed over the threads
static double g_prof[4] = {0, 0, 0, 0};    // keys and values prepared, scores, softmax, P V

struct View4 {      // an int64 [B, H, R, dh] tensor by its strides; group g = b H + h; row r = (r / qn, r % qn)
    const int64_t* p; int64_t s0, s1, s2, s3, heads, srep, qn;      // (queries stacked by repeat: stride srep)
    inline const int64_t* group(int64_t g) const { return p + (g / heads) * s0 + (g % heads) * s1; }
    inline const int64_t* row(int64_t g, int64_t r) const { return group(g) + (r / qn) * srep + (r % qn) * s2; }
};

static inline int32_t pair16(int64_t lo, int64_t hi) {
    return (int32_t)((uint32_t)(uint16_t)(int16_t)lo | ((uint32_t)(uint16_t)(int16_t)hi << 16));
}

// floor(num / den) for 0 <= num < 2**31 and 0 < den < 2**31: the double product with the row's reciprocal is
// within 2**-44 of the quotient (< 256), so its truncation is off by at most one, and one integer correction
// each way gives the exact floor (the int32 division of pvi_attention)
static inline int32_t floordiv_r(int32_t num, int32_t den, double rinv) {
    int32_t q = (int32_t)((double)num * rinv);
    q += (int64_t)(q + 1) * den <= num;
    q -= (int64_t)q * den > num;
    return q;
}

// mo == 0: out [B*H, R, dh] holds P V; mo > 0: out [B, queries, H*rep, dh] holds requant(P V, mo) =
// clamp((P V mo + 2**29) >> 30, -127, 127), row r of group (b, h) being query head h*rep + r / queries at position
// r % queries (the final layout of transformer._attention).  ms > 0: e = exp256[min(255, (gap ms + 2**29) >> 30)],
// which equals lut[min(gap, len - 1)] (the table is non-increasing and zero from _EXP_ZERO on, where lut's last
// entry lies), without lut's cache misses; ms == 0: the lut.
static void attention_avx512(View4 q, View4 k, View4 v, const int32_t* lut, int64_t* out, int64_t bh,
                             int64_t rows, int64_t queries, int64_t t, int64_t dh, int64_t lut_len, int64_t mo,
                             int64_t rep, const int32_t* exp256, int64_t ms) {
    const int64_t nc = (t + 15) / 16, tp = nc * 16, t2 = (t + 1) / 2, d2 = dh / 2, dv = dh / 16;
    const int64_t tiles = (rows + TQ - 1) / TQ;
    const int64_t pw = 2 * (t2 + 1);                 // int16 probabilities per row, zero-padded to whole pairs
    #pragma omp parallel
    {
        std::vector<int32_t> kt(nc * d2 * 16), v2(t2 * dh), s(TQ * tp), qp(TQ * d2);
        std::vector<int16_t> pp(TQ * pw, 0);
        int64_t n[TQ];
        double tp_[4] = {0, 0, 0, 0}, t0 = 0;
        #pragma omp for schedule(dynamic, 1)
        for (int64_t g = 0; g < bh; ++g) {
            if (g_prof_on) t0 = omp_get_wtime();
            const int64_t* kg = k.group(g);
            const int64_t* vg = v.group(g);
            const int64_t* qg = q.group(g);
            for (int64_t j = 0; j < tp; ++j) {               // keys: per chunk of 16, [d/2][16] int16 pairs
                int32_t* kc = kt.data() + (j >> 4) * d2 * 16 + (j & 15);
                if (j < t) {
                    const int64_t* kj = kg + j * k.s2;
                    for (int64_t d = 0; d < d2; ++d) kc[d * 16] = pair16(kj[2 * d * k.s3], kj[(2 * d + 1) * k.s3]);
                } else {
                    for (int64_t d = 0; d < d2; ++d) kc[d * 16] = 0;
                }
            }
            for (int64_t jj = 0; jj < t2; ++jj) {            // values: int16 pairs of consecutive keys
                const int64_t* a = vg + 2 * jj * v.s2;
                const bool two = 2 * jj + 1 < t;
                for (int64_t d = 0; d < dh; ++d)
                    v2[jj * dh + d] = pair16(a[d * v.s3], two ? a[v.s2 + d * v.s3] : 0);
            }
            if (g_prof_on) { const double t1 = omp_get_wtime(); tp_[0] += t1 - t0; t0 = t1; }
            for (int64_t tile = 0; tile < tiles; ++tile) {
                const int64_t r0 = tile * TQ, tr = std::min(TQ, rows - r0);
                int64_t nmax = 0;
                for (int64_t i = 0; i < TQ; ++i) {
                    n[i] = i < tr ? t - queries + ((r0 + i) % queries) + 1 : 0;
                    nmax = std::max(nmax, n[i]);
                    const int64_t* qr = i < tr ? q.row(g, r0 + i) : qg;
                    for (int64_t d = 0; d < d2; ++d)
                        qp[i * d2 + d] = i < tr ? pair16(qr[2 * d * q.s3], qr[(2 * d + 1) * q.s3]) : 0;
                }
                for (int64_t c = 0; c * 16 < nmax; ++c) {          // scores, 16 rows x 16 keys at a time
                    __m512i acc[TQ];
                    for (int64_t i = 0; i < TQ; ++i) acc[i] = _mm512_setzero_si512();
                    const int32_t* kc = kt.data() + c * d2 * 16;
                    for (int64_t d = 0; d < d2; ++d) {
                        const __m512i kv = _mm512_loadu_si512((const void*)(kc + d * 16));
                        for (int64_t i = 0; i < TQ; ++i)
                            acc[i] = _mm512_add_epi32(acc[i], _mm512_madd_epi16(_mm512_set1_epi32(qp[i * d2 + d]), kv));
                    }
                    for (int64_t i = 0; i < TQ; ++i) _mm512_storeu_si512((void*)(s.data() + i * tp + c * 16), acc[i]);
                }
                if (g_prof_on) { const double t1 = omp_get_wtime(); tp_[1] += t1 - t0; t0 = t1; }
                const int64_t np2 = (nmax + 1) / 2;
                for (int64_t i = 0; i < TQ; ++i) {                    // the integer softmax of each row
                    int16_t* pi = pp.data() + i * pw;
                    const int64_t ni = n[i];
                    if (ni == 0) { for (int64_t j = 0; j < 2 * np2; ++j) pi[j] = 0; continue; }
                    int32_t* si = s.data() + i * tp;
                    int32_t mx = INT32_MIN;
                    #pragma omp simd reduction(max:mx)
                    for (int64_t j = 0; j < ni; ++j) mx = std::max(mx, si[j]);
                    // e = exp256[min(255, (gap ms + 2**29) >> 30)] (or lut[min(gap, cap)]), 16 keys at a time: gaps in
                    // int32 (|s| <= 2**21), the product in int64 lanes (< 2**55), 32-bit indices for the gather; the
                    // lanes past ni are masked (loaded as 0, gathered as 0, not stored)
                    const __m512i vmx = _mm512_set1_epi32(mx);
                    __m512i vtot = _mm512_setzero_si512();
                    if (ms > 0) {
                        const __m512i vms = _mm512_set1_epi64(ms), half = _mm512_set1_epi64((int64_t)1 << 29);
                        const __m512i top = _mm512_set1_epi64(255);
                        for (int64_t j = 0; j < ni; j += 16) {
                            const __mmask16 m = ni - j >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << (ni - j)) - 1);
                            const __m512i gap = _mm512_sub_epi32(vmx, _mm512_maskz_loadu_epi32(m, si + j));
                            __m512i g0 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(gap));
                            __m512i g1 = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(gap, 1));
                            g0 = _mm512_min_epi64(_mm512_srai_epi64(_mm512_add_epi64(_mm512_mullo_epi64(g0, vms), half), 30), top);
                            g1 = _mm512_min_epi64(_mm512_srai_epi64(_mm512_add_epi64(_mm512_mullo_epi64(g1, vms), half), 30), top);
                            const __m512i idx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(g0)),
                                                                   _mm512_cvtepi64_epi32(g1), 1);
                            const __m512i e = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m, idx, exp256, 4);
                            _mm512_mask_storeu_epi32(si + j, m, e);
                            vtot = _mm512_add_epi32(vtot, e);
                        }
                    } else {
                        const __m512i cap = _mm512_set1_epi32((int32_t)std::min<int64_t>(lut_len - 1, INT32_MAX));
                        for (int64_t j = 0; j < ni; j += 16) {
                            const __mmask16 m = ni - j >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << (ni - j)) - 1);
                            const __m512i gap = _mm512_min_epi32(_mm512_sub_epi32(vmx, _mm512_maskz_loadu_epi32(m, si + j)), cap);
                            const __m512i e = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m, gap, lut, 4);
                            _mm512_mask_storeu_epi32(si + j, m, e);
                            vtot = _mm512_add_epi32(vtot, e);
                        }
                    }
                    const int32_t tot = _mm512_reduce_add_epi32(vtot);
                    const int32_t den = 2 * tot;
                    const double rinv = 1.0 / (double)den;
                    #pragma omp simd
                    for (int64_t j = 0; j < ni; ++j) {
                        const int32_t num = tot + 510 * si[j];
                        int32_t qv = (int32_t)((double)num * rinv);
                        qv += (int64_t)(qv + 1) * den <= num;
                        qv -= (int64_t)qv * den > num;
                        pi[j] = (int16_t)qv;
                    }
                    for (int64_t j = ni; j < 2 * np2; ++j) pi[j] = 0;
                }
                if (g_prof_on) { const double t1 = omp_get_wtime(); tp_[2] += t1 - t0; t0 = t1; }
                for (int64_t ig = 0; ig < tr; ig += 8) {              // P V: 8 rows share each load of V
                    const int64_t rg = std::min<int64_t>(8, tr - ig);
                    int64_t nm = 0;
                    for (int64_t r = 0; r < rg; ++r) nm = std::max(nm, n[ig + r]);
                    const int64_t jmax = (nm + 1) / 2;
                    for (int64_t dp = 0; dp < dv; dp += 2) {
                        const bool second = dp + 1 < dv;
                        __m512i a0[8], a1[8];
                        for (int r = 0; r < 8; ++r) { a0[r] = _mm512_setzero_si512(); a1[r] = _mm512_setzero_si512(); }
                        for (int64_t jj = 0; jj < jmax; ++jj) {
                            const int32_t* vr = v2.data() + jj * dh + 16 * dp;
                            const __m512i v0 = _mm512_loadu_si512((const void*)vr);
                            const __m512i v1 = second ? _mm512_loadu_si512((const void*)(vr + 16)) : _mm512_setzero_si512();
                            for (int r = 0; r < 8; ++r) {
                                int32_t pr;
                                std::memcpy(&pr, pp.data() + (ig + r) * pw + 2 * jj, 4);
                                const __m512i b = _mm512_set1_epi32(pr);
                                a0[r] = _mm512_add_epi32(a0[r], _mm512_madd_epi16(b, v0));
                                a1[r] = _mm512_add_epi32(a1[r], _mm512_madd_epi16(b, v1));
                            }
                        }
                        for (int64_t r = 0; r < rg; ++r) {
                            const int64_t row = r0 + ig + r;
                            int64_t* o;
                            if (mo == 0) {
                                o = out + (g * rows + row) * dh + 16 * dp;
                            } else {
                                const int64_t b = g / q.heads, h = g % q.heads;
                                o = out + ((b * queries + row % queries) * (q.heads * rep) + h * rep + row / queries) * dh + 16 * dp;
                            }
                            alignas(64) int32_t tmp[32];
                            _mm512_store_si512((void*)tmp, a0[r]);
                            _mm512_store_si512((void*)(tmp + 16), a1[r]);
                            const int nv = second ? 32 : 16;
                            if (mo == 0) {
                                for (int l = 0; l < nv; ++l) o[l] = tmp[l];
                            } else {
                                for (int l = 0; l < nv; ++l)
                                    o[l] = std::min<int64_t>(127, std::max<int64_t>(-127, ((int64_t)tmp[l] * mo + ((int64_t)1 << 29)) >> 30));
                            }
                        }
                    }
                }
                if (g_prof_on) { const double t1 = omp_get_wtime(); tp_[3] += t1 - t0; t0 = t1; }
            }
        }
        if (g_prof_on) {
            #pragma omp critical
            for (int q4 = 0; q4 < 4; ++q4) g_prof[q4] += tp_[q4];
        }
    }
}

extern "C" void pvi_attention_profile(int64_t on, double* out4) {
    for (int q4 = 0; q4 < 4; ++q4) { out4[q4] = g_prof[q4]; g_prof[q4] = 0; }
    g_prof_on = (int)on;
}
#endif

extern "C" void pvi_attention(const int16_t* q, const int16_t* k, const int16_t* v, const int32_t* lut,
                              int64_t* out, int64_t bh, int64_t rows, int64_t queries, int64_t t, int64_t dh,
                              int64_t lut_len, int64_t threads) {
    omp_set_num_threads((int)threads);
    attention_scalar(q, k, v, lut, out, bh, rows, queries, t, dh, lut_len);
}

// 1 if pvi_attention64 runs here (AVX-512BW), for dh a multiple of 16 up to 128 and T < 2**15
extern "C" int64_t pvi_has_avx512() {
#if defined(__AVX512BW__)
    return 1;
#else
    return 0;
#endif
}

// pvi_attention on int64 q [B, H, R, dh], k, v [B, H, T, dh] given by their strides (int8-valued: the inputs of
// the int32 path), read as they are (no int16 copies); out [B, H, R, dh] contiguous
extern "C" void pvi_attention64(const int64_t* q, int64_t q0, int64_t q1, int64_t q2, int64_t q3, int64_t qrep,
                                const int64_t* k, int64_t k0, int64_t k1, int64_t k2, int64_t k3,
                                const int64_t* v, int64_t v0, int64_t v1, int64_t v2, int64_t v3,
                                const int32_t* lut, int64_t* out, int64_t heads, int64_t bh, int64_t rows,
                                int64_t queries, int64_t t, int64_t dh, int64_t lut_len, int64_t mo, int64_t rep,
                                const int32_t* exp256, int64_t ms, int64_t threads) {
    omp_set_num_threads((int)threads);
#if defined(__AVX512BW__)
    const int64_t all = INT64_MAX;            // k and v: row r is position r
    attention_avx512(View4{q, q0, q1, q2, q3, heads, qrep, qrep ? queries : all},
                     View4{k, k0, k1, k2, k3, heads, 0, all}, View4{v, v0, v1, v2, v3, heads, 0, all},
                     lut, out, bh, rows, queries, t, dh, lut_len, mo, rep, exp256, ms);
#endif
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

// clamp((z * mult + 2**(shift-1)) >> shift, lo, hi), times b (contiguous like the output) first when b != nullptr;
// z holds int32 or int64 values (zbytes), the arithmetic is int64
template <class Z>
static void requant_t(const Z* z, const int64_t* b, int64_t* out, int64_t R, int64_t C, int64_t sr, int64_t sc,
                      int64_t mult, int64_t shift, int64_t lo, int64_t hi) {
    const int64_t half = (int64_t)1 << (shift - 1);
    if (b == nullptr)
        each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) {
            out[i * C + j] = clamp64(((int64_t)z[s] * mult + half) >> shift, lo, hi); });
    else
        each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) {
            out[i * C + j] = clamp64(((int64_t)z[s] * b[i * C + j] * mult + half) >> shift, lo, hi); });
}

extern "C" void pvi_requant(const void* z, int64_t zbytes, const int64_t* b, int64_t* out, int64_t R, int64_t C,
                            int64_t sr, int64_t sc, int64_t mult, int64_t shift, int64_t lo, int64_t hi,
                            int64_t threads) {
    omp_set_num_threads((int)threads);
    if (zbytes == 4) requant_t((const int32_t*)z, b, out, R, C, sr, sc, mult, shift, lo, hi);
    else requant_t((const int64_t*)z, b, out, R, C, sr, sc, mult, shift, lo, hi);
}

// clamp(a + ((z * mult + 2**(shift-1)) >> shift), -res_max, res_max); a int64 with strides (ar, ac) (contiguous, or
// a transposed claim view as the embedding's scaled fold), z int32 or int64
template <class Z>
static void residual_t(const int64_t* a, int64_t ar, int64_t ac, const Z* z, int64_t* out, int64_t R, int64_t C,
                       int64_t sr, int64_t sc, int64_t mult, int64_t shift, int64_t res_max) {
    const int64_t half = (int64_t)1 << (shift - 1);
    each2d(R, C, sr, sc, [&](int64_t i, int64_t j, int64_t s) {
        out[i * C + j] = clamp64(a[i * ar + j * ac] + (((int64_t)z[s] * mult + half) >> shift), -res_max, res_max); });
}

extern "C" void pvi_residual(const int64_t* a, int64_t ar, int64_t ac, const void* z, int64_t zbytes, int64_t* out,
                             int64_t R, int64_t C, int64_t sr, int64_t sc, int64_t mult, int64_t shift,
                             int64_t res_max, int64_t threads) {
    omp_set_num_threads((int)threads);
    if (zbytes == 4) residual_t(a, ar, ac, (const int32_t*)z, out, R, C, sr, sc, mult, shift, res_max);
    else residual_t(a, ar, ac, (const int64_t*)z, out, R, C, sr, sc, mult, shift, res_max);
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
        const double rinv = 1.0 / (double)den;
        // floor(num / den) clamped to int8: the double estimate (relative error ~2**-52), bounded to +-2**20 (beyond
        // the clamp either way), then one exact int64 correction each way (|q den| < 2**61 for den < 2**40)
        if (den < ((int64_t)1 << 40)) {
            #pragma omp simd
            for (int64_t j = 0; j < d; ++j) {
                const int64_t num = (xi[j] - m) * gain[j] + add;
                double e = std::floor((double)num * rinv);
                e = e > 1048576.0 ? 1048576.0 : (e < -1048576.0 ? -1048576.0 : e);
                int64_t q = (int64_t)e;
                q += (q + 1) * den <= num;
                q -= q * den > num;
                o[j] = clamp64(q, -127, 127);
            }
        } else {
            for (int64_t j = 0; j < d; ++j) o[j] = clamp64(floordiv64((xi[j] - m) * gain[j] + add, den), -127, 127);
        }
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

template <class Z>
static int64_t field_matmul_t(const int64_t* chi, int64_t r, int64_t n, const Z* z, int64_t m, int64_t sr,
                              int64_t sc, int64_t* out, int64_t threads) {
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
                    const Z* zi = z + i * sr + j0;
                    int64_t over = 0;
                    #pragma omp simd reduction(|:over)
                    for (int64_t j = 0; j < w; ++j) over |= ((int64_t)zi[j] >= ZMAX) | ((int64_t)zi[j] <= -ZMAX);
                    bad |= over;
                    const int64_t* li = lim.data() + i * L;
                    for (int64_t l = 0; l < L; ++l) {
                        const int64_t c = li[l];
                        int64_t* a = acc.data() + l * cb;
                        #pragma omp simd
                        for (int64_t j = 0; j < w; ++j) a[j] += c * (int64_t)zi[j];
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
            const Z* zj = z + j * sc;
            int64_t over = 0;
            #pragma omp simd reduction(|:over)
            for (int64_t i = 0; i < n; ++i) over |= ((int64_t)zj[i] >= ZMAX) | ((int64_t)zj[i] <= -ZMAX);
            bad |= over;
            for (int64_t l = 0; l < r; ++l) {
                int64_t lo = 0, hi = 0;
                for (int64_t i0 = 0; i0 < n; i0 += RED) {
                    const int64_t i1 = std::min(n, i0 + RED);
                    int64_t s0 = 0, s1 = 0;
                    const int64_t* a0 = lim.data() + 2 * l * n;
                    const int64_t* a1 = lim.data() + (2 * l + 1) * n;
                    #pragma omp simd reduction(+:s0, s1)
                    for (int64_t i = i0; i < i1; ++i) { s0 += a0[i] * (int64_t)zj[i]; s1 += a1[i] * (int64_t)zj[i]; }
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

extern "C" int64_t pvi_field_matmul(const int64_t* chi, int64_t r, int64_t n, const void* z, int64_t zbytes,
                                    int64_t m, int64_t sr, int64_t sc, int64_t* out, int64_t threads) {
    omp_set_num_threads((int)threads);
    return zbytes == 4 ? field_matmul_t(chi, r, n, (const int32_t*)z, m, sr, sc, out, threads)
                       : field_matmul_t(chi, r, n, (const int64_t*)z, m, sr, sc, out, threads);
}
// ---- the claim decoder's slot streams (claimcodec._unpack_lanes for all 32 lanes) ---------------------------------
// w [k, g] uint32: lane j holds the values [j g, (j + 1) g), value c of lane j at bit (j k) of column c.  Writes the
// n values to out (uint32), each plus the base of its segment (segments [starts[s], ends[s]) in order, bases los[s];
// uint32 arithmetic wraps); returns 1, or 0 when a padding slot (index >= n) is not zero.
extern "C" int64_t pvi_unpack_stream(const uint32_t* w, int64_t k, int64_t g, int64_t n, uint32_t* out,
                                     const int64_t* starts, const int64_t* ends, const uint32_t* los, int64_t nseg,
                                     int64_t threads) {
    omp_set_num_threads((int)threads);
    const uint32_t mask = (uint32_t)(((uint64_t)1 << k) - 1);
    const int64_t cb = 1 << 14;
    const int64_t blocks = (g + cb - 1) / cb;
    int64_t bad = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(|:bad)
    for (int64_t j = 0; j < 32; ++j) {
        for (int64_t b = 0; b < blocks; ++b) {
            const int64_t wi = (j * k) >> 5, sh = (j * k) & 31;
            const uint32_t* r0 = w + wi * g;
            const uint32_t* r1 = (sh && sh + k > 32) ? w + (wi + 1) * g : nullptr;
            const int64_t c0 = b * cb, c1 = std::min(g, c0 + cb);
            const int64_t a = j * g;
            int64_t s = 0;                                   // the segment of index a + c0
            if (nseg) {
                int64_t lo = 0, hi = nseg;
                while (lo < hi) { const int64_t mid = (lo + hi) / 2; if (ends[mid] <= a + c0) lo = mid + 1; else hi = mid; }
                s = lo;
            }
            for (int64_t c = c0; c < c1; ++c) {
                uint32_t v = r0[c] >> sh;
                if (r1) v |= r1[c] << (32 - sh);
                v &= mask;
                const int64_t idx = a + c;
                if (idx >= n) { bad |= (v != 0); continue; }
                while (s < nseg && ends[s] <= idx) ++s;
                out[idx] = (s < nseg && starts[s] <= idx) ? v + los[s] : v;
            }
        }
    }
    return bad ? 0 : 1;
}
// ---- (min, max) of a contiguous int32 or int64 array in one parallel pass (the claims' range checks) ----------------
template <class Z>
static void minmax_t(const Z* a, int64_t n, int64_t* lo, int64_t* hi) {
    Z mn = a[0], mx = a[0];
    #pragma omp parallel for schedule(static) reduction(min:mn) reduction(max:mx)
    for (int64_t i = 0; i < n; ++i) { mn = a[i] < mn ? a[i] : mn; mx = a[i] > mx ? a[i] : mx; }
    *lo = (int64_t)mn;
    *hi = (int64_t)mx;
}

extern "C" void pvi_minmax(const void* a, int64_t bytes, int64_t n, int64_t* lo, int64_t* hi, int64_t threads) {
    omp_set_num_threads((int)threads);
    if (bytes == 4) minmax_t((const int32_t*)a, n, lo, hi);
    else minmax_t((const int64_t*)a, n, lo, hi);
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
        _LIB.pvi_has_avx512.argtypes = []
        _LIB.pvi_has_avx512.restype = ctypes.c_int64
        _LIB.pvi_attention64.argtypes = ([ctypes.c_void_p] + [ctypes.c_int64] * 5 + [ctypes.c_void_p] + [ctypes.c_int64] * 4
                                         + [ctypes.c_void_p] + [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 2
                                         + [ctypes.c_int64] * 9 + [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64])
        _LIB.pvi_attention64.restype = None
        if hasattr(_LIB, "pvi_attention_profile"):
            _LIB.pvi_attention_profile.argtypes = [ctypes.c_int64, ctypes.c_void_p]
            _LIB.pvi_attention_profile.restype = None
        _LIB.pvi_softmax_block.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 7
        _LIB.pvi_softmax_block.restype = None
        _LIB.pvi_requant.argtypes = [ctypes.c_void_p, ctypes.c_int64] + [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 9
        _LIB.pvi_residual.argtypes = ([ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.c_void_p] + [ctypes.c_int64] * 8)
        _LIB.pvi_lut.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5
        _LIB.pvi_norm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5
        _LIB.pvi_norm.restype = ctypes.c_int64
        _LIB.pvi_rope.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 5
        for f in (_LIB.pvi_requant, _LIB.pvi_residual, _LIB.pvi_lut, _LIB.pvi_rope):
            f.restype = None
        _LIB.pvi_field_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                          ctypes.c_void_p, ctypes.c_int64]
        _LIB.pvi_field_matmul.restype = ctypes.c_int64
        _LIB.pvi_unpack_stream.argtypes = [ctypes.c_void_p] + [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 4 + \
            [ctypes.c_int64] * 2
        _LIB.pvi_unpack_stream.restype = ctypes.c_int64
        _LIB.pvi_minmax.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int64]
        _LIB.pvi_minmax.restype = None
    except (OSError, subprocess.CalledProcessError):
        _LIB = None
    return _LIB


def available() -> bool:
    return _library() is not None


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lut: torch.Tensor, queries: int, *,
                   exp_table: torch.Tensor | None = None, m_s: int = 0) -> torch.Tensor:
    """``_attention_core(q, k, v, lut, notmask)`` (raw ``p v``, int64 ``[B, H, R, dh]``) on the CPU for int8-valued
    ``q [B, H, R, dh]`` (row ``r``: the query at position ``T - queries + r % queries``), ``k, v [B, H, T, dh]`` and
    the int32 table ``lut``."""
    lib = _library()
    bsz, heads, rows, dh = q.shape
    t = k.shape[2]
    lut32 = lut.to(torch.int32).contiguous()
    out = torch.empty(bsz, heads, rows, dh, dtype=torch.int64)
    if (_AVX and lib.pvi_has_avx512() and dh % 16 == 0 and dh <= 128 and t < 1 << 15
            and all(x.dtype == torch.int64 for x in (q, k, v))):     # AVX-512, reading the int64 views as they are
        e32, ms = _exp32(exp_table, m_s)
        lib.pvi_attention64(q.data_ptr(), *q.stride(), 0, k.data_ptr(), *k.stride(), v.data_ptr(), *v.stride(),
                            lut32.data_ptr(), out.data_ptr(), heads, bsz * heads, rows, queries, t, dh,
                            lut32.numel(), 0, 1, e32.data_ptr() if e32 is not None else None, ms,
                            torch.get_num_threads())
        return out
    q16, k16, v16 = (x.to(torch.int16).contiguous() for x in (q, k, v))
    lib.pvi_attention(q16.data_ptr(), k16.data_ptr(), v16.data_ptr(), lut32.data_ptr(), out.data_ptr(),
                      bsz * heads, rows, queries, t, dh, lut32.numel(), torch.get_num_threads())
    return out


_AVX = os.environ.get("PVI_NATIVE_AVX512", "1") != "0"


def attention_requant_ok(dh: int, t: int) -> bool:
    """Whether :func:`attention_requant` runs here (the AVX-512 kernel, ``dh`` a multiple of 16 up to 128)."""
    lib = _library()
    return bool(_AVX and lib is not None and lib.pvi_has_avx512() and dh % 16 == 0 and dh <= 128 and t < 1 << 15)


def _exp32(exp_table, m_s: int):
    """The 256-entry exp table as int32 and ``m_s`` for the kernel's arithmetic look-up, or ``(None, 0)`` (the lut)."""
    if exp_table is None or not 0 < m_s < 1 << 32 or exp_table.numel() != 256:
        return None, 0
    return exp_table.to(torch.int32).contiguous(), int(m_s)


def attention_requant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lut: torch.Tensor, m_o: int,
                      n_heads: int, n_kv: int, dh: int, *, exp_table: torch.Tensor | None = None,
                      m_s: int = 0) -> torch.Tensor:
    """``transformer._attention``'s output, ``[B, Tq, Hq dh]`` int64: the causal attention of int8-valued ``q
    [B, Tq, Hq dh]`` against ``k, v [B, T, Hkv dh]`` (query head ``j`` reads key/value head ``j // (Hq/Hkv)``)
    with its output requantised by ``m_o``, in one AVX-512 pass that writes the final layout."""
    b, tq, _ = q.shape
    t = k.shape[1]
    rep = n_heads // n_kv
    q5 = q.reshape(b, tq, n_kv, rep, dh)
    k4, v4 = k.reshape(b, t, n_kv, dh), v.reshape(b, t, n_kv, dh)
    lut32 = lut.to(torch.int32).contiguous()
    out = torch.empty(b, tq, n_heads * dh, dtype=torch.int64)
    e32, ms = _exp32(exp_table, m_s)
    _LIB.pvi_attention64(q5.data_ptr(), q5.stride(0), q5.stride(2), q5.stride(1), q5.stride(4), q5.stride(3),
                         k4.data_ptr(), k4.stride(0), k4.stride(2), k4.stride(1), k4.stride(3),
                         v4.data_ptr(), v4.stride(0), v4.stride(2), v4.stride(1), v4.stride(3),
                         lut32.data_ptr(), out.data_ptr(), n_kv, b * n_kv, rep * tq, tq, t, dh, lut32.numel(),
                         int(m_o), rep, e32.data_ptr() if e32 is not None else None, ms, torch.get_num_threads())
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
_I32 = os.environ.get("PVI_INT32_CLAIMS", "1") != "0"


def cheap_enabled(*ts: torch.Tensor, claim: torch.Tensor | None = None) -> bool:
    """Whether the native kernels apply to these operands (CPU, int64, non-empty; ``claim`` may also be int32) and
    the library is there."""
    if claim is not None and not (claim.device.type == "cpu" and claim.dtype in (torch.int32, torch.int64)
                                  and claim.numel()):
        return False
    return (_CHEAP and all(t.device.type == "cpu" and t.dtype == torch.int64 and t.numel() for t in ts)
            and _library() is not None)


def int32_claims() -> bool:
    """Whether a CPU verifier keeps its claims as int32 (the native kernels read them as they are)."""
    return _CHEAP and _I32 and _library() is not None


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
    _LIB.pvi_requant(z.data_ptr(), z.element_size(), None if b is None else b.data_ptr(), out.data_ptr(), *lay, m,
                     shift, lo, hi, torch.get_num_threads())
    return out


def residual(a: torch.Tensor, z: torch.Tensor, mult, shift: int, res_max: int):
    """``graph.residual_add(a, z, mult, shift, res_max)``."""
    m, lay, alay = _scalar(mult), _layout(z), _layout(a)
    if m is None or lay is None or alay is None or a.shape != z.shape:
        return None
    out = torch.empty(z.shape, dtype=torch.int64)
    _LIB.pvi_residual(a.data_ptr(), alay[2], alay[3], z.data_ptr(), z.element_size(), out.data_ptr(), *lay, m, shift,
                      res_max, torch.get_num_threads())
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
    if gain.dtype != torch.int64 or gain.device.type != "cpu" or gain.numel() != d or not 1 <= k <= 32:
        return None
    x = x.contiguous()                    # (the first block's input: the embedding's scaled fold, a transposed view)
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
    if not (cheap_enabled(chi, claim=z) and chi.dim() == 2 and z.dim() == 2 and chi.shape[1] == z.shape[0]):
        return None
    sr, sc = z.stride()
    if not (sc == 1 or sr == 1):
        return None
    chi = chi.contiguous()
    out = torch.empty(chi.shape[0], z.shape[1], dtype=torch.int64)
    ok = _LIB.pvi_field_matmul(chi.data_ptr(), chi.shape[0], chi.shape[1], z.data_ptr(), z.element_size(),
                               z.shape[1], sr, sc, out.data_ptr(), torch.get_num_threads())
    return out if ok else None


_DECODE = os.environ.get("PVI_NATIVE_DECODE", "1") != "0"


def unpack_stream(w, k: int, n: int, out, bases) -> bool | None:
    """``claimcodec._unpack_lanes`` over all 32 lanes of the slot stream ``w`` (numpy uint32 ``[k, g]``) into
    ``out[:n]`` (numpy uint32) with the segments' ``bases`` added: True, False when a padding slot is not zero,
    None when the native library is not there (the numpy steps then run)."""
    import numpy as np
    if not _DECODE or _library() is None:
        return None
    if bases is None:
        starts = ends = np.zeros(0, dtype=np.int64)
        los = np.zeros(0, dtype=np.uint32)
    else:
        starts = np.asarray(bases[0], dtype=np.int64)
        ends = np.asarray(bases[1], dtype=np.int64)
        los = np.asarray([int(b) & 0xFFFFFFFF for b in bases[2]], dtype=np.uint32)
    w = np.ascontiguousarray(w)
    ok = _LIB.pvi_unpack_stream(w.ctypes.data, k, w.shape[1], n, out.ctypes.data, starts.ctypes.data,
                                ends.ctypes.data, los.ctypes.data, starts.size, torch.get_num_threads())
    return bool(ok)


def min_max(a: torch.Tensor):
    """``(min, max)`` of a contiguous CPU int32 or int64 tensor in one parallel pass, or None (the caller's
    reductions then run)."""
    if not (_CHEAP and a.device.type == "cpu" and a.dtype in (torch.int32, torch.int64) and a.is_contiguous()
            and a.numel() >= 1 << 16 and _library() is not None):
        return None
    lo, hi = ctypes.c_int64(), ctypes.c_int64()
    _LIB.pvi_minmax(a.data_ptr(), a.element_size(), a.numel(), ctypes.byref(lo), ctypes.byref(hi),
                    torch.get_num_threads())
    return lo.value, hi.value


def attention_profile(on: bool = True) -> list[float] | None:
    """The AVX-512 attention's seconds per phase (keys/values prepared, scores, softmax, P V), summed over the
    threads since the last call, and profiling switched ``on`` or off; None without the AVX-512 kernel."""
    lib = _library()
    if lib is None or not hasattr(lib, "pvi_attention_profile"):
        return None
    buf = (ctypes.c_double * 4)()
    lib.pvi_attention_profile(int(on), buf)
    return list(buf)

