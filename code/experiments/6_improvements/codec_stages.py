"""Where the per-op wire encoder of ``6c2059a`` spends its time: its stages timed one by one.

    cd code && export PYTHONPATH=$PWD/src
    python experiments/6_improvements/codec_stages.py --base ../../base/code/src --threads 4 \\
        lenet5 vgg16 gpt2:64:12 qwen3-4b:8:8 qwen3-4b:8:8/36 --out results/codec_profile_laptop.json

The baseline's ``claimcodec.encode`` (``--base``: the ``src`` of a checkout of ``6c2059a``, whose encoder
is ``tests/claimcodec_reference.py``) re-run here stage by stage from its own functions, with a timer
between the stages, on the claims of one query of each spec (as ``codec_ab.py`` builds them, on the host:
the numpy path, which a CPU prover runs); its bytes are checked against the baseline's ``encode``.
Prints and writes per spec the ops, centred ops, claims, exceptions, per width (segments, values,
exceptions) and the median of ``--reps`` runs of each stage (ms).  Nothing here writes to the benchmark's
roots.
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_verifier as ab  # noqa: E402
import codec_ab  # noqa: E402


def stages(cc, claims, centre: bool = True):
    """``cc.encode(claims)`` (``cc``: ``6c2059a``'s ``claimcodec``), its stages timed: ``(bytes, {stage: s},
    {B: (segments, values, exceptions)}, exceptions, centred ops)``."""
    times = {}
    t = time.perf_counter()

    def lap(k):
        nonlocal t
        now = time.perf_counter()
        times[k] = times.get(k, 0.0) + now - t
        t = now

    zts = [z.detach() if torch.is_tensor(z) else torch.from_numpy(np.asarray(z)) for z in claims]
    ext = cc._host([torch.stack(torch.aminmax(z)).to(torch.int64) for z in zts if z.numel()])
    assert not (ext and (max(int(e[1]) for e in ext) >= cc._Z_LIMIT or min(int(e[0]) for e in ext) <= -cc._Z_LIMIT))
    lap("1 range (aminmax per op)")
    z32s = [z.to(torch.int32) for z in zts]
    lap("2 to int32 per op")
    rows = [cc._sample_rows(*z.shape) for z in zts]
    samples = cc._host([z[torch.from_numpy(r).to(z.device)] for z, r in zip(z32s, rows)])
    lap("3 sample rows")
    plans = [cc._plan(s.astype(np.int64), z.shape[0] / max(r.size, 1), centre) for z, s, r in zip(zts, samples, rows)]
    lap("4 plan (host, per op)")
    centred = [i for i, p in enumerate(plans) if p["centred"]]
    means = {i: torch.div(zts[i].to(torch.int64).sum(1) + zts[i].shape[1] // 2, zts[i].shape[1],
                          rounding_mode="floor").to(torch.int32) for i in centred}
    means_h = dict(zip(centred, cc._host([means[i] for i in centred])))
    lap("5 row means of centred ops")
    ops = []
    for i, (z32, p) in enumerate(zip(z32s, plans)):
        segs = []
        x = z32.reshape(-1)
        if p["centred"]:
            co = cc._median(means_h[i])
            bo, _ = cc._choose(means_h[i].astype(np.int64), co)
            segs.append((bo, cc._base(co, bo), means[i]))
            x = (z32 - means[i][:, None]).reshape(-1)
        segs.append((p["B"], cc._base(p["c"], p["B"]), x))
        ops.append((z32.shape[1], segs))
    lap("6 segments (means plan, residuals)")
    head = [cc.MAGIC, struct.pack("<I", len(ops))]
    segs = []
    for m, op_segs in ops:
        head.append(struct.pack("<IB", m, cc._F_CENTRED if len(op_segs) == 2 else 0))
        for b, lo, x in op_segs:
            head.append(struct.pack("<Bi", b, lo))
            segs.append((b, lo, x))
    head = b"".join(head)
    parts = [head + bytes(-len(head) % 4)]
    exc_pos, exc_h, base = [], [], 0
    lap("7 header")
    widths = {}
    for b in sorted({s[0] for s in segs}):
        group = [(lo, x) for bb, lo, x in segs if bb == b]
        cnt = sum(x.numel() for _, x in group)
        d = torch.empty(cnt, dtype=torch.int32, device=group[0][1].device)
        o = 0
        for lo, x in group:
            torch.sub(x, lo, out=d[o:o + x.numel()])
            o += x.numel()
        lap("8a slots: x - lo per segment")
        e = cc._nonzero((d < 0) | (d >= (1 << b)))
        if e.size:
            exc_pos.append(e + base)
            exc_h.append((d[torch.from_numpy(e).to(d.device)] >> b).cpu().numpy().astype(np.int64))
        lap("8b exceptions (mask, nonzero, high parts)")
        d &= (1 << b) - 1
        parts.append(cc.pack32(d.numpy().view(np.uint32), b))
        lap("8c pack32")
        widths[b] = (len(group), cnt, e.size)
        base += cnt
    pos = np.concatenate(exc_pos) if exc_pos else np.zeros(0, dtype=np.int64)
    h = np.concatenate(exc_h) if exc_h else np.zeros(0, dtype=np.int64)
    parts.append(struct.pack("<I", pos.size) + cc._rice_encode(np.diff(pos, prepend=-1) - 1)
                 + cc._rice_encode(((h << 1) ^ (h >> 63)) - 1))
    lap("9 Rice (gaps, payloads)")
    out = b"".join(parts)
    lap("10 join")
    return out, times, widths, pos.size, len(centred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="as codec_ab.py's")
    ap.add_argument("--base", type=Path, required=True, help="the src of a checkout of 6c2059a")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cc = ab.sub(ab.load_base(args.base.resolve()), "claimcodec")
    for name in ("_host", "_sample_rows", "_plan", "_median", "_choose", "_base", "_nonzero"):
        if not hasattr(cc, name):
            raise SystemExit(f"--base has no claimcodec.{name}: not the per-op encoder of 6c2059a")
    res = {}
    for spec in args.specs:
        graph, x, info, cycle = codec_ab._graph(spec)
        zs = codec_ab._claims(graph, x, torch.device("cpu"), "device", cycle)
        del graph
        blob, _, widths, n_exc, n_centred = stages(cc, zs)
        assert blob == cc.encode(zs)
        runs = [stages(cc, zs)[1] for _ in range(args.reps)]
        n = sum(z.numel() for z in zs)
        print(f"== {spec}: {len(zs)} ops ({n_centred} centred), {n} claims, {n_exc} exceptions "
              f"({100 * n_exc / n:.1f}%), widths {{B: (segments, values, exceptions)}} {widths}")
        med = {k: 1e3 * statistics.median(r[k] for r in runs) for k in runs[0]}
        for k, v in med.items():
            print(f"  {k:45s} {v:8.2f} ms")
        print(f"  {'total':45s} {sum(med.values()):8.2f} ms", flush=True)
        res[spec] = {**info, "ops": len(zs), "centred": n_centred, "claims": n, "exceptions": n_exc,
                     "widths": {str(b): list(v) for b, v in widths.items()}, "stage_ms": med,
                     "total_ms": sum(med.values()), "threads": args.threads, "reps": args.reps}
        if args.out:
            args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
