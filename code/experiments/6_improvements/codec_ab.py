"""The wire codec (``claimcodec``) of this checkout against a baseline's: encode and decode times, bytes.

    cd code && export PYTHONPATH=$PWD/src
    python experiments/6_improvements/codec_ab.py --base ../../base/code/src --device cuda --threads 1,8 \\
        lenet5 vgg16 gpt2:64 qwen3-4b:8 --out codec_ab.json
    python experiments/6_improvements/codec_ab.py --base ../../base/code/src --device cpu --threads 1,4,8 \\
        --reps 15 --device-path lenet5 vgg16 gpt2:64:12 qwen3-4b:8:8 qwen3-4b:8:8/36 --out codec_laptop.json

A spec is a CNN (random init, as ``ab_verifier.py`` builds it, on a random query) or
``decoder:tokens[:blocks[:vocab]][/cycled]`` (the benchmark's random-weight decoders, all blocks by
default; ``/36``: the build's embedding, blocks and head with the blocks' claims cycled to 36 blocks, a
model's shapes and value distributions where the whole model does not fit the host's memory).

The claims of one query are computed on ``--device`` and kept the way each checkout's prover keeps them:
on the device; with ``--lean``, a baseline's int64 on the host (``claims_device="cpu"``: its lean prover
streamed them there), this checkout's int32 on a GPU (``send=claimcodec.narrow``, as ``Prover(lean=True)``
with wire does; on the host as the baseline's).  Then, interleaved over ``--reps`` repetitions (after a
warm-up, the order alternating), at each ``--threads`` count ``n``:

* ``encode_t{n}``: each checkout's ``claimcodec.encode`` of its claims where they are (on a GPU: the device
  encoder; on the host: the host encoder, with ``workers = n`` where it takes them), timed from a
  synchronised device to the bytes on the host -- the prover's ``prove_encode``;
* ``decode_{int64,int32}_t{n}``: each checkout's ``decode_torch`` into int64 (a CPU verifier) and into int32
  (a GPU client), ``workers = n``;
* at the last count, ``pack_field_{u,columns}``: random field elements on ``--device``, per op ``[6, N]``
  (about ``u``'s size) and the transposes of ``[N, 24]`` (about the opened columns', as ``run_query``
  packs them); with ``--device-path`` (CPU claims), ``encode_device_path``: this checkout's device encoder
  on the CPU tensors, and the baseline's GPU path forced on them (its two ``device.type == "cpu"``
  branches off).

After them, ``encode_cold_t{n}`` (the last count): this checkout's encode of a shape set it has not seen (its
layout built: on a device, its tables), the median of 3 (with ``--device-path`` also
``encode_cold_device_path``).  It checks that both checkouts give the same bytes (claims and field elements)
and decode the same claims, and reports (``device_encoder``) this checkout's device encoder's aten ops that
launch kernels, warm and cold, its waits for the device (``_d2h``, ``_nonzero_dev``) and the size of its
layout's device tables (``tables_mb``) -- on a GPU's claims, or with ``--device-path`` (with the baseline's
GPU path's ops).
Without ``--base`` only this checkout runs.  Nothing here writes to the benchmark's roots.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_verifier as ab  # noqa: E402
import pvi.fullcheck as fullcheck  # noqa: E402
from pvi.fullcheck import claimcodec, opcount  # noqa: E402
from pvi.fullcheck.field import P  # noqa: E402

COLD = 3


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _graph(spec: str):
    """``(graph, query, the case's description, (blocks, cycled to))`` of ``spec``."""
    spec, _, cycle = spec.partition("/")
    p = spec.split(":")
    if p[0] in ab.CNNS:
        if cycle:
            raise SystemExit(f"{spec}: /blocks is for the decoders")
        graph, x, _ = ab.build(fullcheck, ab.Case(p[0], "C", 128))
        return graph, ab.random_queries(fullcheck, graph, x, 1)[0], {"model": p[0]}, (None, None)
    from pvi.fullcheck.transformer import CONFIGS, build_decoder
    cfg = CONFIGS[p[0]]
    seq = int(p[1]) if len(p) > 1 else 64
    blocks = int(p[2]) if len(p) > 2 and p[2] else cfg.n_layers
    if len(p) > 3:
        cfg = dataclasses.replace(cfg, vocab=int(p[3]))
    graph = build_decoder(cfg, n_layers=blocks, calib_tokens=min(seq, 32), seed=0)
    x = torch.randint(0, cfg.vocab, (1, seq), generator=torch.Generator().manual_seed(1))
    info = {"model": p[0], "seq": seq, "blocks": int(cycle) if cycle else blocks, "vocab": cfg.vocab}
    if cycle:
        info["built_blocks"] = blocks
    return graph, x, info, (blocks, int(cycle) if cycle else None)


def _cycled(zs: list, blocks: int, to: int) -> list:
    """A decoder's claims (embedding first, head last) with its ``blocks`` blocks' cycled to ``to`` blocks."""
    per, rest = divmod(len(zs) - 2, blocks)
    if rest:
        raise SystemExit("the claims are not an embedding, equal blocks and a head")
    out = [zs[0]]
    for b in range(to):
        out += [z.clone() for z in zs[1 + per * (b % blocks):1 + per * (b % blocks + 1)]]
    return out + [zs[-1]]


def _claims(graph, x: torch.Tensor, device: torch.device, kept: str, cycle) -> list:
    """The claims of ``x`` in weight-op order, kept ``"device"`` (where computed), ``"host"`` (int64 on the
    host, each as it is computed: a lean prover's) or ``"narrow"`` (int32 on the device: a lean GPU
    prover's with wire)."""
    kw = {"device": {}, "host": {"free": True, "claims_device": "cpu"},
          "narrow": {"free": True, "send": claimcodec.narrow}}[kept]
    with torch.no_grad():
        _, claims = graph.forward(x.to(device), **kw)
    zs = [claims[op.name] for op in graph.mat_ops]
    return _cycled(zs, *cycle) if cycle[1] else zs


def _time(fn, dev: torch.device) -> float:
    _sync(dev)
    t0 = time.perf_counter()
    fn()
    _sync(dev)
    return time.perf_counter() - t0


def _counted(zs) -> dict:
    """This checkout's device encoder: aten ops that launch kernels (warm, then on a new shape set: its
    layout's tables built) and its waits for the device."""
    claimcodec.encode(zs, impl="device")
    with opcount.codec_waits(claimcodec) as waits, opcount.KernelOps() as ops:
        claimcodec.encode(zs, impl="device")
    claimcodec._layout.cache_clear()
    with opcount.KernelOps() as cold:
        claimcodec.encode(zs, impl="device")
    lay = claimcodec._layout(tuple((int(z.shape[0]), int(z.shape[1])) for z in zs))
    size = sum(t.numel() * t.element_size() for tb in lay._dev.values() for t in tb.values() if torch.is_tensor(t))
    return {"kernel_ops": ops.n, "kernel_ops_cold": cold.n, **waits, "tables_mb": size / 1e6}


def _torch_path(base) -> types.ModuleType | None:
    """The baseline's ``claimcodec`` with its GPU path forced on CPU tensors (its branches on
    ``device.type == "cpu"`` in ``_nonzero`` and ``_assemble`` off), or ``None`` if it has not those two."""
    src = (Path(base.__file__).parent / "claimcodec.py").read_text(encoding="utf-8")
    branches = ('if mask.device.type == "cpu":', 'if d.device.type == "cpu" else')
    if any(src.count(b) != 1 for b in branches):
        return None
    for b in branches:
        src = src.replace(b, b.replace('mask.device.type == "cpu"', "False").replace('d.device.type == "cpu"', "False"))
    name = f"{base.__name__}.claimcodec_gpu_path"
    mod = types.ModuleType(name)
    mod.__package__, mod.__file__ = base.__name__, str(Path(base.__file__).parent / "claimcodec.py")
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def run(spec: str, args, base) -> dict:
    dev = torch.device(args.device)
    graph, x, info, cycle = _graph(spec)
    impls = {"new": claimcodec}
    if base is not None:
        impls["base"] = ab.sub(base, "claimcodec")
    kept = {k: "device" for k in impls}
    if args.lean:
        kept = {"new": "narrow" if dev.type != "cpu" else "host", "base": "host"}
    zs = {}
    for k in impls:
        same = next((j for j in zs if kept[j] == kept[k]), None)
        zs[k] = zs[same] if same else _claims(graph, x, dev, kept[k], cycle)
    host = [z.cpu().long() for z in zs["new"]]
    rows, cols = [z.shape[0] for z in host], [z.shape[1] for z in host]
    info.update(ops=len(host), claims=sum(z.numel() for z in host),
                kept={k: f"{zs[k][0].device.type} {str(zs[k][0].dtype)[6:]}" for k in impls})
    takes_workers = {k: "workers" in inspect.signature(m.encode).parameters for k, m in impls.items()}

    def encode(k, thr):
        return impls[k].encode(zs[k], **({"workers": thr} if takes_workers[k] else {}))

    blobs = {k: encode(k, args.threads[-1]) for k in impls}
    if len(set(blobs.values())) != 1:
        raise SystemExit(f"{spec}: the checkouts encode different bytes")
    blob = blobs["new"]
    info.update(bytes=len(blob), bits_per_claim=8 * len(blob) / info["claims"])
    for k, m in impls.items():                     # both decoders give the claims
        assert all(torch.equal(a.long(), b) for a, b in zip(m.decode_torch(blob, rows, cols), host)), k
    g = torch.Generator().manual_seed(2)          # field elements of about u's and the columns' sizes (mode C)
    fields = {"u": [torch.randint(0, P, (6, r), generator=g).to(dev) for r in rows],
              "columns": [torch.randint(0, P, (r, 24), generator=g).to(dev).T for r in rows]}
    for name, ts in fields.items():
        assert len({m.pack_field(ts) for m in impls.values()}) == 1, name
    gpu_path = _torch_path(base) if args.device_path and base is not None else None
    if args.device_path:
        if dev.type != "cpu":
            raise SystemExit("--device-path is for claims on the host")
        assert claimcodec.encode(zs["new"], impl="device") == blob
        assert gpu_path is None or gpu_path.encode(zs["base"]) == blob
    cpu = torch.device("cpu")
    times: dict = {}
    for thr in args.threads:                       # one thread count at a time (switching regrows torch's pool)
        torch.set_num_threads(thr)
        fns = {}
        for k in impls:
            fns[f"encode_t{thr}_{k}"] = (lambda k=k, thr=thr: encode(k, thr), dev)
            for dt in (torch.int64, torch.int32):
                fns[f"decode_{str(dt)[6:]}_t{thr}_{k}"] = (
                    lambda k=k, dt=dt, thr=thr: impls[k].decode_torch(blob, rows, cols, dtype=dt, workers=thr), cpu)
        if thr == args.threads[-1]:
            for name, ts in fields.items():
                for k, m in impls.items():
                    fns[f"pack_field_{name}_{k}"] = (lambda m=m, ts=ts: m.pack_field(ts), dev)
            if args.device_path:
                fns["encode_device_path_new"] = (lambda: claimcodec.encode(zs["new"], impl="device"), cpu)
                if gpu_path is not None:
                    fns["encode_device_path_base"] = (lambda: gpu_path.encode(zs["base"]), cpu)
        keys = list(fns)
        for rep in range(args.reps + 1):           # interleaved, the order alternating; the first a warm-up
            for key in (keys if rep % 2 == 0 else keys[::-1]):
                t = _time(fns[key][0], fns[key][1])
                if rep:
                    times.setdefault(key, []).append(t)
    cold = []                                      # a shape set not seen: its layout (on a device, its tables)
    for _ in range(COLD):
        claimcodec._layout.cache_clear()
        cold.append(_time(lambda: encode("new", args.threads[-1]), dev))
    times[f"encode_cold_t{args.threads[-1]}_new"] = cold
    if args.device_path:
        for _ in range(COLD):
            claimcodec._layout.cache_clear()
            times.setdefault("encode_cold_device_path_new", []).append(
                _time(lambda: claimcodec.encode(zs["new"], impl="device"), cpu))
    out = {"spec": spec, **info, "medians_ms": {k: 1e3 * statistics.median(v) for k, v in times.items()},
           "min_ms": {k: 1e3 * min(v) for k, v in times.items()}}
    if base is not None:
        out["base_over_new"] = {k[:-4]: out["medians_ms"][k[:-4] + "_base"] / v
                                for k, v in out["medians_ms"].items()
                                if k.endswith("_new") and k[:-4] + "_base" in out["medians_ms"]}
    if dev.type != "cpu" or args.device_path:
        out["device_encoder"] = _counted(zs["new"])
        if gpu_path is not None:
            with opcount.KernelOps() as ops:
                gpu_path.encode(zs["base"])
            out["device_encoder"]["base_gpu_path_kernel_ops"] = ops.n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--base", type=Path, help="the baseline checkout's src (imported as another package)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="the prover's device")
    ap.add_argument("--lean", action="store_true", help="the claims kept as a lean prover keeps them (see above)")
    ap.add_argument("--device-path", action="store_true",
                    help="CPU claims: also time and count the device encoders on them")
    ap.add_argument("--threads", default="1,8", help="host threads (encode workers, decoder workers)")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    args.threads = [int(t) for t in args.threads.split(",")]
    torch.set_num_threads(args.threads[-1])
    base = ab.load_base(args.base.resolve()) if args.base else None
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    base_sha = None
    if args.base:
        base_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                  cwd=args.base.resolve()).stdout.strip() or None
    results = []
    for spec in args.specs:
        r = run(spec, args, base)
        r.update(git=sha, base_git=base_sha, host=platform.node(), cpu=platform.processor(), threads=args.threads,
                 device=args.device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 lean=args.lean, reps=args.reps)
        results.append(r)
        print(json.dumps(r), flush=True)
        if args.out:
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
