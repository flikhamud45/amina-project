"""A native (C++, OpenMP) exact integer causal attention for a CPU verifier: the integers of
``transformer._attention_core`` without its ``T x T`` tensors and their dozen element-wise passes.

Per (batch, head, query row), on the verifier's threads: the int32 scores against the keys up to the query's
position, their maximum, ``e = LUT[min(max - s, len - 1)]``, ``tot = sum e``, ``p = floor((tot + 510 e) / (2 tot))``
and ``p v`` (int64).  Every step is integer arithmetic on the same values as ``_attention_core`` (scores below
``2**21``, ``tot`` and ``510 e + tot`` below ``2**31``, as ``transformer._int32_scores`` requires), so the result is
bit-identical (tested).

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

__all__ = ["available", "attention_core"]

_SOURCE = r"""
#include <cstdint>
#include <algorithm>
#include <vector>
#include <omp.h>

extern "C" void pvi_attention(const int16_t* q, const int16_t* k, const int16_t* v, const int32_t* lut,
                              int64_t* out, int64_t bh, int64_t rows, int64_t queries, int64_t t, int64_t dh,
                              int64_t lut_len, int64_t threads) {
    omp_set_num_threads((int)threads);
    #pragma omp parallel
    {
        std::vector<int32_t> s(t), e(t);
        std::vector<int64_t> acc(dh);
        #pragma omp for schedule(dynamic, 16) collapse(2)
        for (int64_t g = 0; g < bh; ++g) {
            for (int64_t r = 0; r < rows; ++r) {
                const int16_t* qr = q + (g * rows + r) * dh;
                const int16_t* kg = k + g * t * dh;
                const int16_t* vg = v + g * t * dh;
                const int64_t n = t - queries + (r % queries) + 1;     // keys 0 .. the query's position
                int32_t mx = INT32_MIN;
                for (int64_t j = 0; j < n; ++j) {
                    const int16_t* kj = kg + j * dh;
                    int32_t acc32 = 0;
                    #pragma omp simd reduction(+:acc32)
                    for (int64_t d = 0; d < dh; ++d) acc32 += (int32_t)qr[d] * (int32_t)kj[d];
                    s[j] = acc32;
                    mx = std::max(mx, acc32);
                }
                int32_t tot = 0;
                for (int64_t j = 0; j < n; ++j) {
                    int64_t gap = (int64_t)mx - (int64_t)s[j];
                    if (gap > lut_len - 1) gap = lut_len - 1;
                    e[j] = lut[gap];
                    tot += e[j];
                }
                std::fill(acc.begin(), acc.end(), 0);
                const int32_t two = 2 * tot;
                for (int64_t j = 0; j < n; ++j) {
                    const int32_t num = tot + 510 * e[j];
                    const int32_t p = num / two;                       // both positive: floor division
                    if (p == 0) continue;
                    const int16_t* vj = vg + j * dh;
                    #pragma omp simd
                    for (int64_t d = 0; d < dh; ++d) acc[d] += (int64_t)p * (int64_t)vj[d];
                }
                int64_t* o = out + (g * rows + r) * dh;
                for (int64_t d = 0; d < dh; ++d) o[d] = acc[d];
            }
        }
    }
}
"""

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
    key = hashlib.sha256((_SOURCE + cxx + platform.node() + platform.processor()).encode()).hexdigest()[:16]
    where = Path(os.environ.get("PVI_NATIVE_DIR", Path.home() / ".cache" / "pvi"))
    lib = where / f"pvi_attention_{key}.so"
    try:
        if not lib.exists():
            where.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "attn.cpp"
                src.write_text(_SOURCE)
                built = Path(tmp) / lib.name
                subprocess.run([cxx, "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(src), "-o", str(built)],
                               check=True, capture_output=True)
                shutil.move(str(built), str(lib))
        _LIB = ctypes.CDLL(str(lib))
        _LIB.pvi_attention.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 7
        _LIB.pvi_attention.restype = None
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
