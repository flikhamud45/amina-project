"""Kernel-level timings of the four exact-arithmetic primitives, at Llama-2-7B shapes.

    python experiments/4_defence_benchmark/microbench.py --device cuda --out mb_<gpu>.jsonl

Attributes a GPU-class speed-up to its cause: the forward pass is float32 GEMMs
(``exact_matmul``, TF32 off), fold/open are float64 GEMMs (``small_matmul_mod``;
consumer GPUs run FP64 at 1/32 of FP32), the verifier's Freivalds products are
float64 limb products (``field_matmul_mod``) and the commitment is an int64 NTT
(memory-bound).  Every result is checked against the CPU so exactness is also tested.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from pvi.fullcheck.commitment import vandermonde_columns
from pvi.fullcheck.field import P, field_matmul_mod, ntt, small_matmul_mod, to_field
from pvi.fullcheck.graph import exact_matmul


def _time(fn, device, reps):
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--small", action="store_true", help="tiny shapes (smoke test on a laptop)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    g = torch.Generator().manual_seed(0)
    d, f, T, r = (256, 688, 128, 7) if args.small else (4096, 11008, 2048, 7)   # Llama-2-7B MLP-up, 2,048 tokens
    w = torch.randint(-127, 128, (f, d), generator=g, dtype=torch.int8)
    x = torch.randint(-127, 128, (d, T), generator=g, dtype=torch.int64)
    chi = torch.randint(0, P, (r, f), generator=g, dtype=torch.int64)
    z = torch.randint(-(1 << 28), 1 << 28, (f, T), generator=g, dtype=torch.int64)
    rows = to_field(torch.randint(-127, 128, (64, d + 1), generator=g))
    n_points = 4 * (1 << d.bit_length())             # rate 4 * next_pow2(d + 1), as in commitment.py
    rows_p = torch.zeros(64, n_points, dtype=torch.int64)
    rows_p[:, :d + 1] = rows
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    # inputs live on the device before timing: only the kernels are measured, not the bus
    wd, xd, chid, zd, rd = w.to(dev), x.to(dev).float(), chi.to(dev), to_field(z).to(dev), rows_p.to(dev)
    chiTd = chi.T.contiguous().to(dev)
    # open at lambda=128: t = 69 codeword columns of a d-wide row (rate 4); FP64-bound on consumer GPUs
    t_cols, n_pts_open = 69, 4 * (1 << (d - 1).bit_length())
    cols = torch.randperm(n_pts_open, generator=g)[:t_cols]
    vd = vandermonde_columns(n_pts_open, d, cols, dev)[:d]
    v_cpu = vandermonde_columns(n_pts_open, d, cols)[:d]
    if dev.type == "cuda":   # 1 GiB: the size class of the claims copy and of a weight upload
        host = torch.empty(1 << 27, dtype=torch.int64)
        pinned, devbuf = host.pin_memory(), torch.empty_like(host, device=dev)
    cases = {
        # (callable on device, flop or byte count, unit)
        "exact_matmul_fp32 (forward)": (lambda: exact_matmul(wd, xd), 2 * f * d * T, "flop"),
        "small_matmul_mod_fp64 (fold)": (lambda: small_matmul_mod(wd.T, chiTd).T, 2 * f * d * r, "flop"),
        "field_matmul_mod_fp64x3 (Freivalds)": (lambda: field_matmul_mod(chid, zd), 3 * 2 * r * f * T, "flop"),
        "ntt_int64 (commit)": (lambda: ntt(rd), rows_p.numel() * 8 * rows_p.shape[1].bit_length(), "byte"),
        "small_matmul_mod_fp64 (open, t=69)": (lambda: small_matmul_mod(wd, vd), 2 * f * d * t_cols, "flop"),
    }
    if dev.type == "cuda":
        cases.update({
            "h2d pageable 1 GiB": (lambda: devbuf.copy_(host), host.numel() * 8, "byte"),
            "h2d pinned 1 GiB": (lambda: devbuf.copy_(pinned, non_blocking=True), host.numel() * 8, "byte"),
            "d2h pageable 1 GiB": (lambda: host.copy_(devbuf), host.numel() * 8, "byte"),
        })
    out = []
    for label, (fn, work, unit) in cases.items():
        sec, res = _time(fn, dev, args.reps)
        rec = {"kernel": label, "device": name, "seconds": sec, unit + "_per_s": work / sec}
        if dev.type != "cpu":   # same inputs on the CPU: the results must be bit-identical
            cpu_res = {"exact_matmul_fp32 (forward)": lambda: exact_matmul(w, x.float()),
                       "small_matmul_mod_fp64 (fold)": lambda: small_matmul_mod(w.T, chi.T.contiguous()).T,
                       "field_matmul_mod_fp64x3 (Freivalds)": lambda: field_matmul_mod(chi, to_field(z)),
                       "ntt_int64 (commit)": lambda: ntt(rows_p),
                       "small_matmul_mod_fp64 (open, t=69)": lambda: small_matmul_mod(w, v_cpu)}.get(label)
            if cpu_res is not None:
                rec["bit_identical_to_cpu"] = bool(torch.equal(res.cpu(), cpu_res()))
        out.append(rec)
        print(json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            for rec in out:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
