"""The reproducible build of the certified commitment (Sec. ``sec:defence`` overview: "in our run, builds of the
certified OPT-125M with 1 and 8 threads are identical bit for bit, and so are OPT-1.3B's builds and commitments with
CPU and GPU encoding"; ``experiments/paper/macros.py``'s ``digest_agreement`` compares the run's records).

Builds the certified integer OPT graph from a Hugging Face checkpoint on the CPU with ``--threads`` threads --
``--recipe certified`` (the default: ``hf_opt.CERTIFIED_RECIPE`` calibrated on ``checkpoints.certified_calibration``,
four 2,048-token windows of the WikiText-2 train split) or ``--recipe legacy`` (the first recipe: alpha 0.55 with
percentile statistics, calibrated on the last 128-token window of the test split) -- and prints one JSON line: the
graph digest (``IntGraph.digest``, format ``pvi/graph/v2``: every op, its kind and every constant), a SHA-256 of every
weight matrix and bias (``weights_sha256``), and with ``--commit`` a SHA-256 of the commitment's Merkle roots under the
commitment plan ``--policy`` at rate ``--rate`` encoded on ``--commit-device`` (``roots_sha256``, ``n_roots``,
``commit_device``).  Run it on two machines, with two thread counts and with ``--commit-device cpu`` and ``cuda``:
equal ``digest``, ``weights_sha256`` and ``roots_sha256`` mean the same model and the same ``C_M``.  The run
package's job q-digest builds OPT-125M with 1 and 8 threads and OPT-1.3B twice, its commitment encoded on the GPU and
on the CPU.

    python -m experiments.quality.digest_check --model facebook/opt-125m --threads 1
    python -m experiments.quality.digest_check --model facebook/opt-1.3b --threads 8 --commit --commit-device cuda
    python -m experiments.quality.digest_check --model facebook/opt-1.3b --threads 8 --commit --commit-device cpu

Hardware and time (Xeon CPU nodes, measured with the certified recipe): OPT-125M's build takes about 2-3 h on one
thread; with 8 threads OPT-1.3B builds and commits in about 55 min.  The four 2,048-token calibration windows need
more host memory than a 16 GB laptop has.
"""
from __future__ import annotations

import argparse
import platform

import torch

from experiments.common import checkpoints, records
from experiments.common.cli import PAPER, RATE
from pvi.models.hf_opt import CERTIFIED_RECIPE, build_opt_from_hf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--commit", action="store_true", help="also compute the commitment's Merkle roots (slower)")
    ap.add_argument("--commit-device", default="cpu", help="where the commitment is encoded (the same roots anywhere)")
    ap.add_argument("--policy", default=PAPER.policy, help="the commitment plan (default: the paper's measured one)")
    ap.add_argument("--rate", type=int, default=RATE, help="the Reed-Solomon rate of the commitment")
    ap.add_argument("--prob-max", type=int, default=None, help="the attention probabilities' scale (default: the "
                    "certified models' CERTIFIED_PROB_MAX)")
    ap.add_argument("--recipe", choices=["certified", "legacy"], default="certified")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = checkpoints.load_opt(args.model)
    kw = {} if args.prob_max is None else {"prob_max": args.prob_max}
    if args.recipe == "certified":
        calib, recipe = checkpoints.certified_calibration(tok), dict(CERTIFIED_RECIPE)
    else:
        calib = checkpoints.load_windows(tok, seq=128, n=1, calib_windows=1)[1][0]
        recipe = dict(checkpoints.OPT_RECIPES["legacy"])
    with torch.no_grad():
        graph, _ = build_opt_from_hf(model, calib, device="cpu", **recipe, **kw)
    out = {"model": args.model, "recipe": args.recipe, "threads": args.threads, "host": platform.node(),
           "cpu": platform.processor(), "torch": torch.__version__, "digest": graph.digest().hex(),
           "weights_sha256": checkpoints.weights_sha(graph)}
    if args.commit:
        from pvi.v0.graph_commitment import commit_graph
        coms = commit_graph(graph, args.rate, device=args.commit_device, policy=args.policy)
        # under a grouped plan a member's own root is empty and the group's tree holds it: hash every root
        roots, n_roots = records.roots_sha256(coms)
        assert n_roots, "no Merkle root"
        out.update(policy=args.policy, rate=args.rate, commit_device=args.commit_device, roots_sha256=roots,
                   n_roots=n_roots)
    records.emit(dict(out, env=records.environment(args.commit_device if args.commit else "cpu")))


if __name__ == "__main__":
    main()
