"""The wire codec (``claimcodec``) of this checkout against a baseline's: encode and decode times, bytes.

    cd code && export PYTHONPATH=$PWD/src
    python experiments/6_improvements/codec_ab.py --base ../../base/code/src --device cuda --threads 1,8 \\
        lenet5 vgg16 gpt2:64 qwen3-4b:8 --out codec_ab.json

A spec is a CNN (random init, as ``ab_verifier.py`` builds it, on a random query) or
``decoder:tokens[:blocks[:vocab]]`` (the benchmark's random-weight decoders, all blocks by default).  The
claims of one query are computed on ``--device`` (the prover's; ``claims_device="cpu"`` with ``--lean``),
then, interleaved over ``--reps`` repetitions:

* ``encode``: each checkout's ``claimcodec.encode`` of those claims where they are (on a GPU: the
  device encoder; on the host: the host encoder with ``torch.get_num_threads()`` workers), timed
  from a synchronised device to the bytes on the host -- the prover's ``prove_encode`` of the claims;
* ``decode``: each checkout's ``decode_torch`` into int64 (a CPU verifier) and into int32 (a GPU
  client), with ``workers`` = the thread count, at every ``--threads``;
* ``pack_field``: random field elements on ``--device``, per op ``[6, N]`` (about ``u``'s size) and the
  transposes of ``[N, 24]`` (about the opened columns', as ``run_query`` packs them).

It checks that both checkouts give the same bytes (claims and field elements) and decode the same
claims, and counts the device encoder's aten ops that launch kernels and its waits for the device
(``_d2h``, ``_nonzero_dev``).  Without ``--base`` only this checkout runs.  Nothing here writes to the
benchmark's roots.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_verifier as ab  # noqa: E402
import pvi.fullcheck as fullcheck  # noqa: E402
from pvi.fullcheck import claimcodec  # noqa: E402
from pvi.fullcheck.field import P  # noqa: E402

VIEWS = ("view", "slice", "detach", "lift_fresh", "alias", "_reshape_alias", "select", "unsqueeze", "expand",
         "as_strided", "t.default", "transpose", "squeeze", "unbind", "split", "reshape")


class _Ops(TorchDispatchMode):
    """The aten ops that launch work on a device (views and detach launch none)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not any(str(func).startswith("aten." + v) for v in VIEWS):
            self.n += 1
        return func(*args, **(kwargs or {}))


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _claims(spec: str, device: torch.device, lean: bool) -> tuple[list[torch.Tensor], dict]:
    """The claims of one query of ``spec`` (in weight-op order) on ``device``, and the case's description."""
    p = spec.split(":")
    if p[0] in ab.CNNS:
        graph, x, _ = ab.build(fullcheck, ab.Case(p[0], "C", 128))
        x = ab.random_queries(fullcheck, graph, x, 1)[0]
        info = {"model": p[0]}
    else:
        from pvi.fullcheck.transformer import CONFIGS, build_decoder
        cfg = CONFIGS[p[0]]
        seq = int(p[1]) if len(p) > 1 else 64
        blocks = int(p[2]) if len(p) > 2 and p[2] else cfg.n_layers
        if len(p) > 3:
            cfg = dataclasses.replace(cfg, vocab=int(p[3]))
        graph = build_decoder(cfg, n_layers=blocks, calib_tokens=min(seq, 32), seed=0)
        x = torch.randint(0, cfg.vocab, (1, seq), generator=torch.Generator().manual_seed(1))
        info = {"model": p[0], "seq": seq, "blocks": blocks, "vocab": cfg.vocab}
    kw = {"free": True, "claims_device": "cpu"} if lean else {}
    with torch.no_grad():
        _, claims = graph.forward(x.to(device), **kw)
    zs = [claims[op.name] for op in graph.mat_ops]
    info.update(ops=len(zs), claims=sum(z.numel() for z in zs), claims_device=str(zs[0].device))
    return zs, info


def _time(fn, dev: torch.device) -> float:
    _sync(dev)
    t0 = time.perf_counter()
    fn()
    _sync(dev)
    return time.perf_counter() - t0


def _counted(zs) -> dict:
    """This checkout's device encoder: aten ops that launch kernels, waits for the device."""
    waits = {"_d2h": 0, "_nonzero_dev": 0}
    real = {k: getattr(claimcodec, k) for k in waits}

    def wrap(name):
        def call(t):
            waits[name] += 1
            return real[name](t)
        return call

    for k in waits:
        setattr(claimcodec, k, wrap(k))
    try:
        with _Ops() as ops:
            claimcodec.encode(zs, impl="device")
    finally:
        for k, f in real.items():
            setattr(claimcodec, k, f)
    return {"kernel_ops": ops.n, **waits}


def run(spec: str, args, base) -> dict:
    dev = torch.device(args.device)
    zs, info = _claims(spec, dev, args.lean)
    rows, cols = [z.shape[0] for z in zs], [z.shape[1] for z in zs]
    impls = {"new": claimcodec}
    if base is not None:
        impls["base"] = ab.sub(base, "claimcodec")
    blobs = {k: m.encode(zs) for k, m in impls.items()}
    if base is not None and blobs["new"] != blobs["base"]:
        raise SystemExit(f"{spec}: the checkouts encode different bytes")
    blob = blobs["new"]
    info.update(bytes=len(blob), bits_per_claim=8 * len(blob) / info["claims"])
    host = [z.cpu() for z in zs]
    for k, m in impls.items():                     # both decoders give the claims
        assert all(torch.equal(a.long(), b.long()) for a, b in zip(m.decode_torch(blob, rows, cols), host)), k
    g = torch.Generator().manual_seed(2)          # field elements of about u's and the columns' sizes (mode C)
    fields = {"u": [torch.randint(0, P, (6, r), generator=g).to(dev) for r in rows],
              "columns": [torch.randint(0, P, (r, 24), generator=g).to(dev).T for r in rows]}
    for name, ts in fields.items():
        packed = {k: m.pack_field(ts) for k, m in impls.items()}
        assert len(set(packed.values())) == 1, name
    times: dict = {}

    def timed(key: str, fn, on: torch.device) -> None:
        for rep in range(args.reps + 1):           # interleaved, the order alternating; the first a warm-up
            for k, m in (list(impls.items()) if rep % 2 == 0 else list(impls.items())[::-1]):
                t = _time(lambda: fn(m), on)
                if rep:
                    times.setdefault(f"{key}_{k}", []).append(t)

    torch.set_num_threads(args.threads[-1])       # the prover's host threads
    timed("encode", lambda m: m.encode(zs), dev)
    for name, ts in fields.items():
        timed(f"pack_field_{name}", lambda m, ts=ts: m.pack_field(ts), dev)
    cpu = torch.device("cpu")
    for thr in args.threads:                       # one thread count at a time (switching regrows torch's pool)
        torch.set_num_threads(thr)
        for dt in (torch.int64, torch.int32):
            timed(f"decode_{str(dt)[6:]}_t{thr}", lambda m, dt=dt, thr=thr: m.decode_torch(blob, rows, cols, dtype=dt,
                                                                                           workers=thr), cpu)
    torch.set_num_threads(args.threads[-1])
    out = {"spec": spec, **info, "medians_ms": {k: 1e3 * statistics.median(v) for k, v in times.items()},
           "min_ms": {k: 1e3 * min(v) for k, v in times.items()}}
    if base is not None:
        out["base_over_new"] = {k[:-4]: out["medians_ms"][k[:-4] + "_base"] / v
                                for k, v in out["medians_ms"].items() if k.endswith("_new")}
    if dev.type != "cpu":
        out["device_encoder"] = _counted(zs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--base", type=Path, help="the baseline checkout's src (imported as another package)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="the prover's device")
    ap.add_argument("--lean", action="store_true", help="claims streamed to the host as the forward computes them")
    ap.add_argument("--threads", default="1,8", help="verifier threads (decoder workers)")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    args.threads = [int(t) for t in args.threads.split(",")]
    torch.set_num_threads(args.threads[-1])
    base = ab.load_base(args.base.resolve()) if args.base else None
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    results = []
    for spec in args.specs:
        r = run(spec, args, base)
        r.update(git=sha, host=platform.node(), cpu=platform.processor(), threads=args.threads, device=args.device,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, lean=args.lean)
        results.append(r)
        print(json.dumps(r), flush=True)
        if args.out:
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
