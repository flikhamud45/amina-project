"""The compact wire encoding of the proof (``run_query(wire=True)``): proof bytes and costs.

    cd code && PYTHONPATH=src python experiments/6_improvements/wire.py --mnist <dir with MNIST/raw> \\
        --out experiments/6_improvements/results/wire_laptop.json lenet5 vgg16 resnet18_cifar mlp_mnist \\
        gpt2:64 gpt2:512 qwen3-4b:8:1,2,4,8 llama2-7b:64:1,2

A spec is a CNN (``lenet5``, ``vgg11``, ``vgg16``, ``resnet18_cifar``: random init, as no trained
weights are shipped; ``mlp_mnist``: the committed trained MLP) or ``decoder:tokens[:blocks]`` (the
benchmark's random-weight decoders, all blocks by default; with several block counts the per-block
slope extrapolates to the full model).  CNN queries are MNIST test digits, read with
``download=False`` (padded to 32x32 and repeated on 3 channels for the CIFAR shapes: CIFAR-10 is
not on this machine), calibrated on 512 training digits.

Per spec and mode (C at lambda = 128 with interactive challenges; Kpre at lambda = 128, 40 for
Qwen3-4B as in the Maverick row), ``run_query`` with and without ``wire`` alternate on the same
queries and seeds, at each of ``--threads`` (1 and 4 by default): the medians of every byte and
timing part, and the bits per claim.  Mode C of a decoder is sized without a commitment (the rate-4 commitment of a decoder takes
minutes on a laptop CPU): the claims are encoded as the prover encodes them, ``u`` and the columns
have fixed sizes, and the Merkle paths are those of the challenge ``run_query`` draws with seed 0.
The codec's own throughput (encode, decode into int64, at the same thread counts) is compared with
reading the same claims as int32 and widening them.  Nothing here writes to the benchmark's roots.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pvi.fullcheck import claimcodec
from pvi.fullcheck.analytic import decoder_shapes, proof_bytes
from pvi.fullcheck.commitment import HASH_BYTES, multiproof_size, next_pow2
from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_bytes import query_claims  # noqa: E402

CODE = Path(__file__).resolve().parents[2]
CNNS = {"lenet5": (1, 28), "vgg11": (3, 32), "vgg16": (3, 32), "resnet18_cifar": (3, 32)}


def _mnist(root: str):
    from torchvision import datasets

    tr = datasets.MNIST(root, train=True, download=False)
    te = datasets.MNIST(root, train=False, download=False)
    return tr.data[:, None].float() / 255.0, te.data[:, None].float() / 255.0


def _cnn(name: str, mnist, queries: int):
    from pvi.fullcheck.models import build_float_model
    from pvi.fullcheck.quantize import quantize_input, quantize_model

    train, test = mnist
    channels, size = CNNS[name]
    pad = (size - 28) // 2

    def prep(x):   # the benchmark's MNIST normalisation; the CIFAR shapes: padded and on 3 channels
        x = torch.nn.functional.pad((x - 0.1307) / 0.3081, (pad,) * 4, value=-0.1307 / 0.3081)
        return x.repeat(1, channels, 1, 1)
    torch.manual_seed(0)
    graph = quantize_model(build_float_model(name, 10).eval(), prep(train[:512]))
    return graph, [quantize_input(graph, prep(test[i:i + 1])) for i in range(queries)]


def _mlp(mnist, queries: int):
    from pvi.fullcheck.quantize import quantize_input, quantize_model

    d = np.load(CODE / "artifacts" / "models" / "mlp_mnist_full.npz")
    mods: list[nn.Module] = []
    for i, name in enumerate(["fc1", "fc2", "logits"]):
        w, b = d[name + ".weight"], d[name + ".bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data, lin.bias.data = torch.from_numpy(w.astype(np.float32)), torch.from_numpy(b.astype(np.float32))
        mods += [lin] + ([nn.ReLU()] if i < 2 else [])
    train, test = mnist
    graph = quantize_model(nn.Sequential(*mods).eval(), train[:512].reshape(-1, 784))   # as bench.py: [0, 1] pixels
    return graph, [quantize_input(graph, test[i:i + 1].reshape(-1, 784)) for i in range(queries)]


def _decoder(model: str, seq: int, n_layers: int | None, queries: int):
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[model]
    graph = build_decoder(cfg, n_layers=n_layers or cfg.n_layers, calib_tokens=min(seq, 32), seed=0)
    tokens = torch.randint(0, cfg.vocab, (queries, seq), generator=torch.Generator().manual_seed(1))
    return graph, list(tokens.split(1)), len(decoder_shapes(cfg))


def _median(runs: list[dict], part: str) -> dict:
    return {k: statistics.median(r[part].get(k, 0) for r in runs) for k in sorted({k for r in runs for k in r[part]})}


def _ab(prover: Prover, v: Verifier, xs: list, reps: int) -> dict:
    """``run_query`` with and without wire, alternating which runs first, on the same queries and seeds."""
    runs = {False: [], True: []}
    for w in (False, True):                                        # warm-up
        assert run_query(prover, v, xs[0], seed=0, wire=w)["accepted"]
    for rep in range(reps):
        for w in ((False, True) if rep % 2 == 0 else (True, False)):
            r = run_query(prover, v, xs[rep % len(xs)], seed=rep, wire=w)
            assert r["accepted"], (w, r["rejected_at"])
            runs[w].append(r)
    out = {}
    for w, rs in runs.items():
        t = _median(rs, "timings")
        out["wire" if w else "default"] = {"bytes": _median(rs, "bytes"), "timings": t,
                                           "verify": sum(x for k, x in t.items() if k.startswith("verify_")),
                                           "prove": sum(x for k, x in t.items() if k.startswith("prove_"))}
    return out


def _codec(zs: list[torch.Tensor], reps: int, thread_counts: list[int]) -> dict:
    """Encode and decode (into int64) throughput against reading int32 claims and widening them."""
    rows, cols = [z.shape[0] for z in zs], [z.shape[1] for z in zs]
    n = sum(z.numel() for z in zs)
    raw = b"".join(z.to(torch.int32).numpy().tobytes() for z in zs)

    def parse():
        o, out = 0, []
        for z in zs:
            out.append(torch.from_numpy(np.frombuffer(raw, "<i4", z.numel(), o).copy()).view(z.shape).to(torch.int64))
            o += 4 * z.numel()
        return out
    blob = claimcodec.encode(zs)
    assert all(torch.equal(a, b) for a, b in zip(zs, claimcodec.decode_torch(blob, rows, cols)))
    out = {"claims": n, "bytes": len(blob), "bits_per_claim": 8 * len(blob) / n,
           "bits_per_claim_uncentred": 8 * len(claimcodec.encode(zs, centre=False)) / n}
    for threads in thread_counts:
        torch.set_num_threads(threads)
        t = {"encode": [], "decode": [], "int32_parse": []}
        for _ in range(reps):
            for name, fn in (("encode", lambda: claimcodec.encode(zs)), ("int32_parse", parse),
                             ("decode", lambda: claimcodec.decode_torch(blob, rows, cols, workers=threads))):
                t0 = time.perf_counter()
                fn()
                t[name].append(time.perf_counter() - t0)
        med = {k: statistics.median(v) for k, v in t.items()}
        out[f"threads{threads}"] = {**{f"{k}_s": v for k, v in med.items()},
                                    **{f"{k}_Mclaims_per_s": n / v / 1e6 for k, v in med.items()}}
    return out


def _mode_c_exact(graph, params, zs: list[torch.Tensor], encoded: int, rate: int = 4) -> dict:
    """Mode C's proof bytes without a commitment: the byte model (``analytic.proof_bytes``; the claims
    ``zs``, ``encoded`` bytes with wire), with the Merkle paths of the columns ``run_query`` draws with
    seed 0 (the same with and without wire) in place of their expectation."""
    mats = graph.mat_ops
    ch = Challenger(seed=0)
    for op in mats:                                   # run_query draws every chi, then every column set
        ch.folding(op.name, op.n_rows, params.reps)
    paths = 0
    for op in mats:
        n_points = rate * next_pow2(op.row_length)
        idx = ch.columns(op.name, n_points, params.columns)
        paths += multiproof_size(idx.tolist(), n_points.bit_length() - 1) * HASH_BYTES
    cols = {op.name: z.shape[1] for op, z in zip(mats, zs)}
    return {name: dict(proof_bytes(mats, params, cols, rate=rate, wire_claims=wire_claims), paths=paths)
            for name, wire_claims in (("default", None), ("wire", encoded))}


def run(spec: str, args) -> dict:
    name, *rest = spec.split(":")
    out = {"spec": spec}
    if name in CNNS or name == "mlp_mnist":
        mnist = _mnist(args.mnist)
        graph, xs = _mlp(mnist, args.queries) if name == "mlp_mnist" else _cnn(name, mnist, args.queries)
        n_checks = len(graph.mat_ops)
    else:   # the claims of every block count; the last build is kept for the timings
        seq, blocks = int(rest[0]), ([int(b) for b in rest[1].split(",")] if len(rest) > 1 else [None])
        for b in blocks:
            graph, xs, n_checks = _decoder(name, seq, b, args.queries)
            if b is not None:
                zs, encoded = query_claims(graph, xs[0])
                out.setdefault("builds", {})[str(b)] = {"claims": sum(z.numel() for z in zs), "wire_bytes": encoded}
            if b != blocks[-1]:
                graph = None
    mats = graph.mat_ops
    zs, encoded = query_claims(graph, xs[0])
    out["codec"] = _codec(zs, args.reps, args.threads)
    out["ops"] = len(mats)
    prover = Prover(graph)
    weights = {op.name: (op.weight, op.bias) for op in mats}
    lam_kpre = 40 if name == "qwen3-4b" else 128
    modes = [("Kpre", lam_kpre)] + ([("C", 128)] if name in CNNS or name == "mlp_mnist" else [])
    for mode, lam in modes:
        params = params_for(lam, n_checks)
        if mode == "C":
            coms = commit_graph(graph, params.rate)
            prover = Prover(graph, commitments=coms)
            v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
        else:
            v = Verifier(graph.public(), params, "Kpre", weights=weights)
            v.precompute(Challenger())
        for threads in args.threads:
            torch.set_num_threads(threads)
            out[f"{mode}_lam{lam}_threads{threads}"] = _ab(prover, v, xs, args.reps)
    if name not in CNNS and name != "mlp_mnist":
        out["C_lam128_exact"] = _mode_c_exact(graph, params_for(128, n_checks), zs, encoded)
    return out


def summary(results: dict) -> str:
    """Markdown tables of a results file: the claims, whole proofs, verifier times, codec throughput
    and, for decoders measured at several block counts, the per-block slope."""
    rows = ["| Model | Claims | int32 (B) | PVC3 (B) | Bits/claim (uncentred) | Smaller |", "|---|---:|---:|---:|---:|---:|"]
    for spec, r in results.items():
        c = r["codec"]
        rows.append(f"| {spec} | {c['claims']:,} | {4 * c['claims']:,} | {c['bytes']:,} | {c['bits_per_claim']:.2f} "
                    f"({c['bits_per_claim_uncentred']:.2f}) | {4 * c['claims'] / c['bytes']:.3f}x |")
    rows += ["", "| Model | Mode | Default (B) | Wire (B) | Smaller | claims / u / columns / paths (wire) |",
             "|---|---|---:|---:|---:|---|"]
    for spec, r in results.items():
        for key in sorted(k for k in r if k.endswith("_threads" + min(k2.split("threads")[1] for k2 in r if "_threads" in k2))
                          or k.endswith("_exact")):
            d, w = (r[key][s] if key.endswith("_exact") else r[key][s]["bytes"] for s in ("default", "wire"))
            rows.append(f"| {spec} | {key.split('_threads')[0]} | {sum(d.values()):,.0f} | {sum(w.values()):,.0f} | "
                        f"{sum(d.values()) / sum(w.values()):.3f}x | "
                        f"{' / '.join(f'{w[k]:,.0f}' for k in ('claims', 'u', 'columns', 'paths'))} |")
    rows += ["", "| Model | Mode | Threads | Verifier (ms) | Wire verifier (ms) | verify_decode (ms) | Decode / verifier | "
             "prove_encode (ms) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for spec, r in results.items():
        for key in sorted(k for k in r if "_threads" in k):
            d, w = r[key]["default"], r[key]["wire"]
            dec = w["timings"]["verify_decode"]
            rows.append(f"| {spec} | {key.split('_threads')[0]} | {key.split('threads')[1]} | {1e3 * d['verify']:.2f} | "
                        f"{1e3 * w['verify']:.2f} | {1e3 * dec:.2f} | {100 * dec / d['verify']:.0f}% | "
                        f"{1e3 * w['timings']['prove_encode']:.2f} |")
    rows += ["", "| Model | Threads | Encode (M claims/s) | Decode | int32 parse |", "|---|---|---:|---:|---:|"]
    for spec, r in results.items():
        c = r["codec"]
        threads = sorted(k for k in c if k.startswith("threads"))
        rows.append(f"| {spec} | {' / '.join(t[7:] for t in threads)} | " + " | ".join(
            " / ".join(f"{c[t][k + '_Mclaims_per_s']:.0f}" for t in threads) for k in ("encode", "decode", "int32_parse")) + " |")
    for spec, r in results.items():
        builds = sorted((int(b), v) for b, v in r.get("builds", {}).items())
        if len(builds) >= 2:
            from pvi.fullcheck.transformer import CONFIGS
            (b0, v0), (b1, v1) = builds[-2:]
            per_block = (v1["wire_bytes"] - v0["wire_bytes"]) / (b1 - b0)
            per_claims = (v1["claims"] - v0["claims"]) / (b1 - b0)
            full = CONFIGS[spec.split(":")[0]].n_layers
            claims = v1["claims"] + (full - b1) * per_claims
            wire = v1["wire_bytes"] + (full - b1) * per_block
            rows += ["", f"{spec}: " + ", ".join(f"L{b} {v['claims']:,} claims -> {v['wire_bytes']:,} B" for b, v in builds)
                     + f"; per block {per_block:,.0f} B ({8 * per_block / per_claims:.3f} bits/claim); {full} blocks: "
                     f"{4 * claims:,.0f} -> {wire:,.0f} B ({4 * claims / wire:.3f}x)"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="*")
    ap.add_argument("--mnist", help="directory holding MNIST/raw (read with download=False), for the CNNs")
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=lambda s: [int(t) for t in s.split(",")], default=[1, 4],
                    help="torch threads (and decoder workers), e.g. 1,8 for the paper's verifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true", help="print the tables of --out")
    args = ap.parse_args()
    path = Path(args.out)
    results = json.loads(path.read_text()) if path.exists() else {}
    if args.summary:
        print(summary(results))
        return
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    for spec in args.specs:
        t0 = time.perf_counter()
        results[spec] = dict(run(spec, args), git=sha, seconds=time.perf_counter() - t0)
        path.write_text(json.dumps(results, indent=1))
        c = results[spec]["codec"]
        print(f"{spec}: {c['claims']} claims, {c['bits_per_claim']:.3f} bits/claim "
              f"({4 * c['claims'] / c['bytes']:.3f}x smaller than int32), {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
