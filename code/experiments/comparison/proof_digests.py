"""Proofs that do not depend on the hardware: the same cells proved on the GPU and on the CPU, and on another platform.

Backs the extended version's statement that the run's proofs and commitments are byte-identical on the hardware
compared (``app:eval-setup``, ``\\HwDigestCells``, ``\\HwDigestDevices``).  Each cell is a model of the run's
configuration (the paper's protocol settings, :data:`experiments.common.cli.PAPER`: Fiat--Shamir, the compact
encoding, the lean prover, the pruned last block, mode C under ``cnn18c``, 11-bit attention probabilities), its
queries fixed by seeds; each is proved with the prover on every ``--devices`` entry (the run's GPU and its CPU), the
commitment ``C_M`` built on that device, and checked by the verifier on ``--verifier-device``.  Per query the record
keeps the Fiat--Shamir transcript's digest (:func:`pvi.v0.query.run_query`, :func:`pvi.v1.protocol.run_cut_query`:
the statement with ``C_M``'s roots and every prover message the challenges depend on; mode Kpre's one message) and per
device the SHA-256 of ``C_M``'s roots (:func:`experiments.common.records.roots_sha256`).  The run package's job
``xcheck`` (the smoke tier) runs it on the run's platform; a second platform runs the same command at the same commit
(``run.sh --platform <other> --require-gpu <its GPU> --require-cpu any smoke``), and ``--compare`` checks the two:

    python -m experiments.comparison.proof_digests --cells mlp_mnist lenet5 gpt2:2:64 opt-125m:2:64 \\
        llama2-7b:1:64:Kpre --v1-cells gpt2:2:64 --devices cuda cpu --queries 2 --label xcheck
    python -m experiments.comparison.proof_digests --compare ../logs/final ../logs/<other platform>
    python -m experiments.comparison.proof_digests --cells mlp_mnist tiny-gpt:2:16 --v1-cells tiny-llama:2:16 \\
        --devices cpu --queries 2 --lmax 20          # a laptop check (seconds)

A cell is ``<model>[:<blocks>:<tokens>[:<modes>]]``: an image model of the benchmark (``mlp_mnist``, ``lenet5``, ...;
the trained networks, ``artifacts/models``, its test images as queries) or a decoder's shapes with random weights
(``--seed``) and prompts (``--prompt-seed``), ``<modes>`` joined with ``+`` (default ``--modes``).  ``--v1-cells``:
decoders proved with protocol V1 as well (its GKR at ``2^--lmax`` leaves; Triton on a GPU, PyTorch on the CPU).  An
image model whose trained network is missing here is recorded as ``skipped``.

**Record** (one JSON line per cell, mode and protocol, ``"kind": "digests"``): ``label``, ``git``, ``model``,
``layers``, ``seq``, ``mode``, ``protocol`` (``v0``, ``v1``), the protocol settings (``policy``, ``fs``, ``wire``,
``lean``, ``lookups``, ``prune_last``, ``prob_max``, ``lam``, ``rate``, ``lmax``), ``queries``, ``graph_digest``,
``devices`` (per device: ``roots_sha256`` in mode C, ``transcripts``, ``proof_bytes``, ``accepted``, ``seconds``),
``agree`` (every device's transcripts and roots equal and every query accepted), ``gpu``, ``cpu``, ``host`` and
``env``.  A disagreement or a rejected query prints ``FAILED: ...`` and makes the exit status 1.

``--compare A B``: the records of two runs' logs (folders of ``sp-*.out``, or files) matched by cell, mode, protocol
and settings; every transcript and root must be equal.  It prints one line per pair and ``N cells compared, M
problems`` and exits with status 1 on a problem or when nothing could be compared.  ``experiments/paper/numbers.py
--xcheck-logs B`` adds B's records to the paper's statement.

Hardware and time: on the run's platform the GPU and its CPU (8 threads) prove every cell; the CPU's commitment of
Llama-2-7B's embedding and LM head is the largest step, so Llama-2-7B's block is proved in Kpre only.  About 30 minutes
on an L40S node; the laptop check above takes seconds.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from experiments.common import cli, records
from experiments.common.paths import CNN_MODELS, CODE

KEY = ("model", "layers", "seq", "mode", "protocol", "policy", "fs", "wire", "lean", "lookups", "prune_last",
       "prob_max", "lam", "rate", "lmax", "queries")
"""The fields that name one cell's proofs: two records of equal keys must have equal digests."""


def parse_cell(spec: str, modes: list[str]) -> dict:
    """``<model>[:<blocks>:<tokens>[:<modes>]]`` as ``{"model", "layers", "seq", "modes"}``."""
    parts = spec.split(":")
    if len(parts) not in (1, 3, 4) or not parts[0]:
        raise SystemExit(f"a cell is <model>[:<blocks>:<tokens>[:<modes>]], not {spec!r}")
    cell = {"model": parts[0], "layers": None, "seq": None, "modes": list(modes)}
    if len(parts) >= 3:
        cell["layers"], cell["seq"] = int(parts[1]), int(parts[2])
    if len(parts) == 4:
        cell["modes"] = [m for m in parts[3].split("+") if m]
        bad = set(cell["modes"]) - {"C", "Kpre"}
        if bad:
            raise SystemExit(f"{spec}: modes are C and Kpre, not {sorted(bad)}")
    return cell


def build(cell: dict, args) -> dict:
    """The cell's integer graph, queries and planning inputs: ``{"graph", "queries", "model_ops", "n_checks",
    "image"}`` (None for an image model whose trained network is missing)."""
    from pvi.models.quantize import RANDOM_CNNS
    name = cell["model"]
    if name in RANDOM_CNNS:
        from experiments.common.models import test_batches, trained_cnn, trained_mlp
        from pvi.models.quantize import quantize_input, quantize_model
        if name != "mlp_mnist" and not (CNN_MODELS / f"{name}.pt").is_file():
            return None
        model, data = trained_mlp() if name == "mlp_mnist" else trained_cnn(name)
        graph = quantize_model(model, data["train_x"])
        xs, _ = next(test_batches(data, args.queries))
        return {"graph": graph, "queries": list(quantize_input(graph, xs).split(1)), "model_ops": None,
                "n_checks": len(graph.mat_ops), "image": True}
    from experiments.common.models import decoder_config
    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder
    if cell["seq"] is None:
        raise SystemExit(f"a decoder cell names its blocks and tokens: {name}:<blocks>:<tokens>")
    cfg = decoder_config(name)
    graph = build_decoder(cfg, n_layers=cell["layers"] or cfg.n_layers, calib_tokens=min(cell["seq"], args.calib_tokens),
                          seed=args.seed, prune_last=args.prune_last, prob_max=args.prob_max)
    tokens = torch.randint(0, cfg.vocab, (args.queries, cell["seq"]),
                           generator=torch.Generator().manual_seed(args.prompt_seed))
    model_ops = decoder_shapes(cfg, prune_last=args.prune_last)
    return {"graph": graph, "queries": list(tokens.split(1)), "model_ops": model_ops, "n_checks": len(model_ops),
            "image": False}


def prove_on(built: dict, mode: str, protocol: str, device: str, args) -> dict:
    """The cell's queries proved with the prover (and ``C_M``) on ``device``: per device the roots' digest, each
    query's transcript digest, proof bytes and verdict."""
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    graph, image = built["graph"], built["image"]
    lean, lookups = (False, False) if image else (args.lean, args.lookups)
    t0 = time.perf_counter()
    coms = None
    if mode == "C":
        coms = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=built["model_ops"])
    v1 = protocol == "v1"
    params = params_for(args.lam, built["n_checks"], rate=args.rate, plan=None if coms is None else coms.plan,
                        fiat_shamir=args.fs, **({"cut": True, "cut_lmax": args.lmax} if v1 else {}))
    if mode == "C":
        v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                     tables=coms.table_publics, lean=lean, device=args.verifier_device)
    else:
        weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
        v = Verifier(graph.public(), params, mode, weights=weights, lean=lean, device=args.verifier_device,
                     lookups=lookups)
        v.precompute(Challenger())
    prover = Prover(graph, device=device, commitments=coms, lean=lean)
    out = {"transcripts": [], "proof_bytes": [], "accepted": 0}
    if coms is not None:
        out["roots_sha256"] = records.roots_sha256(coms)[0]
    for q in built["queries"]:
        if v1:
            from pvi.v1 import protocol as cp
            res = cp.run_cut_query(prover, v, q)
        else:
            res = run_query(prover, v, q, wire=args.wire)
        out["transcripts"].append(res.get("transcript"))
        out["proof_bytes"].append(int(sum(res["bytes"].values())))
        out["accepted"] += int(bool(res["accepted"]))
    out["seconds"] = time.perf_counter() - t0
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return out


def agree(devices: dict, queries: int) -> bool:
    """Every device's transcripts and roots equal, every transcript present and every query accepted."""
    vals = list(devices.values())
    if not vals:
        return False
    first = vals[0]
    same = all(d["transcripts"] == first["transcripts"] and d.get("roots_sha256") == first.get("roots_sha256")
               for d in vals[1:])
    return same and all(d["accepted"] == queries and all(d["transcripts"]) for d in vals)


def run(args) -> int:
    cli.apply_runtime(args)
    sha = records.git_sha(CODE)
    cells = [(parse_cell(s, args.modes), "v0") for s in args.cells]
    cells += [(dict(parse_cell(s, args.modes)), "v1") for s in args.v1_cells]
    ok = True
    env = None
    for cell, protocol in cells:
        built = build(cell, args)
        base = {"kind": "digests", "label": args.label, "git": sha, "model": cell["model"], "protocol": protocol,
                "fs": args.fs, "lam": args.lam, "rate": args.rate, "queries": args.queries}
        if built is None:
            records.emit({**base, "skipped": f"no trained network {CNN_MODELS / (cell['model'] + '.pt')}"})
            print(f"warning: {cell['model']}: no trained network here, skipped", file=sys.stderr)
            continue
        image = built["image"]
        if protocol == "v1" and image:
            raise SystemExit(f"--v1-cells: decoders only, not {cell['model']}")
        settings = {"layers": None if image else cell["layers"], "seq": None if image else cell["seq"],
                    "wire": args.wire, "lean": False if image else args.lean,
                    "lookups": False if image else args.lookups, "prune_last": False if image else args.prune_last,
                    "prob_max": None if image else args.prob_max, "lmax": args.lmax if protocol == "v1" else None,
                    "graph_digest": built["graph"].digest().hex()}
        for mode in cell["modes"]:
            devices = {dev: prove_on(built, mode, protocol, dev, args) for dev in args.devices}
            env = env or records.environment(args.devices[0])
            good = agree(devices, args.queries)
            rec = {**base, **settings, "mode": mode, "policy": args.policy if mode == "C" else None,
                   "devices": devices, "agree": good, "host": platform.node(), "gpu": env.get("gpu"),
                   "cpu": env.get("cpu"), "env": env}
            records.emit(rec)
            if not good:
                ok = False
                print(f"FAILED: {cell['model']} {mode} {protocol}: the devices' proofs or roots differ, or a query was "
                      f"rejected ({', '.join(args.devices)})", file=sys.stderr, flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ comparing two runs
def load(paths: list[str]) -> list[dict]:
    """The ``digests`` records of job logs (folders of ``sp-*.out`` or files), later ones replacing earlier ones of
    the same key."""
    out: dict = {}
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("sp-*.out")) if p.is_dir() else [p]
        for f in files:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get("kind") == "digests" and "devices" in d:
                    out[key_of(d)] = d
    return list(out.values())


def key_of(d: dict) -> tuple:
    return tuple(json.dumps(d.get(k)) for k in KEY)


def compare(a: list[dict], b: list[dict]) -> tuple[int, list[str]]:
    """``(cells compared, problems)`` of two runs' ``digests`` records matched by :data:`KEY`: the graph digest, every
    device's transcripts and roots must be equal across the two runs (and each run's devices agree)."""
    left = {key_of(d): d for d in a}
    problems, n = [], 0
    for d in b:
        ref = left.get(key_of(d))
        if ref is None:
            continue
        n += 1
        name = f"{d['model']} {d['mode']} {d['protocol']} (L={d.get('layers')}, T={d.get('seq')})"
        devs = list(ref["devices"].values()) + list(d["devices"].values())
        first = devs[0]
        if ref.get("graph_digest") != d.get("graph_digest"):
            problems.append(f"MISMATCH {name}: graph digests differ (another model or code)")
        elif not all(x["transcripts"] == first["transcripts"] and x.get("roots_sha256") == first.get("roots_sha256")
                     for x in devs):
            problems.append(f"MISMATCH {name}: transcripts or roots differ")
        elif not (ref.get("agree") and d.get("agree")):
            problems.append(f"MISMATCH {name}: a run's own devices disagree or a query was rejected")
    return n, problems


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=[], help="cells <model>[:<blocks>:<tokens>[:<modes>]] proved with "
                                                           "protocol v0")
    ap.add_argument("--v1-cells", nargs="+", default=[], help="decoder cells also proved with protocol V1")
    ap.add_argument("--modes", nargs="+", choices=["C", "Kpre"], default=["C", "Kpre"],
                    help="the modes of a cell that names none")
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"], help="the prover's devices (and C_M's)")
    ap.add_argument("--queries", type=int, default=2, help="queries per cell, mode and device")
    ap.add_argument("--seed", type=int, default=0, help="the decoders' random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the decoders' random prompts' seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of a random-weight decoder (at most its tokens)")
    ap.add_argument("--lmax", type=int, default=20, help="V1's GKR instances: the target size 2^lmax")
    ap.add_argument("--label", default="xcheck", help="a label copied into every record")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                    help="compare the digests records of two runs' logs (folders or files) instead")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        a, b = load([args.compare[0]]), load([args.compare[1]])
        n, problems = compare(a, b)
        for p in problems:
            print(p)
        print(f"{n} cells compared, {len(problems)} problems")
        if not n:
            print("nothing compared: do both hold digests records of the same cells and settings?")
        return 1 if problems or not n else 0
    if not args.cells and not args.v1_cells:
        raise SystemExit("--cells or --v1-cells (or --compare)")
    if not args.fs:
        raise SystemExit("the digests compare Fiat-Shamir proofs: an interactive proof's challenges are random")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
