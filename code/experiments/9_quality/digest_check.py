"""Plan 2, D6a: is the certified integer model reproducible bit for bit from the public checkpoint?

Builds the certified integer OPT graph from a Hugging Face checkpoint (``--recipe certified``, the default:
``real_weights.CERTIFIED_RECIPE`` with ``common.certified_calib``, F2a's smooth-max alpha 0.6 on four 2,048-token
train windows; ``--recipe legacy``: the earlier recipe, alpha 0.55 with percentile statistics, calibrated on the last
128-token window of the test split) on the CPU with ``--threads`` threads, and prints the graph digest
(every op, weight and constant), a SHA-256 of every weight matrix and bias, and the commitment's Merkle roots.
Run it on two machines and with two thread counts; equal lines mean the same C_M.

    cd code && PYTHONPATH=src python experiments/9_quality/digest_check.py --model facebook/opt-125m --threads 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import certified_calib, load_opt, load_windows          # noqa: E402

from pvi.fullcheck.real_weights import CERTIFIED_RECIPE, build_opt_from_hf   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--commit", action="store_true", help="also compute the commitment's Merkle roots (slower)")
    ap.add_argument("--commit-device", default="cpu", help="where the commitment is encoded (the same roots anywhere)")
    ap.add_argument("--prob-max", type=int, default=None, help="the attention probabilities' scale (default: the "
                    "certified models' CERTIFIED_PROB_MAX)")
    ap.add_argument("--recipe", choices=["certified", "legacy"], default="certified")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = load_opt(args.model)
    kw = {} if args.prob_max is None else {"prob_max": args.prob_max}
    if args.recipe == "certified":
        calib, recipe = certified_calib(tok), dict(CERTIFIED_RECIPE)
    else:
        calib = load_windows(tok, seq=128, windows=1, calib_windows=1)[1][0]
        recipe = dict(norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
    with torch.no_grad():
        graph, _ = build_opt_from_hf(model, calib, device="cpu", **recipe, **kw)
    weights = hashlib.sha256()
    for op in graph.mat_ops:
        weights.update(op.name.encode() + op.weight.contiguous().numpy().tobytes())
        if op.bias is not None:
            weights.update(op.bias.contiguous().numpy().tobytes())
    out = {"model": args.model, "recipe": args.recipe, "threads": args.threads, "host": platform.node(),
           "cpu": platform.processor(),
           "torch": torch.__version__, "digest": graph.digest().hex(), "weights_sha256": weights.hexdigest()}
    if args.commit:
        from pvi.fullcheck.analytic import decoder_shapes   # noqa: F401  (shapes come from the graph itself)
        from pvi.fullcheck.protocol import commit_graph
        coms = commit_graph(graph, 4, device=args.commit_device, policy="auto")
        # under a grouped plan a member's own root is empty and the group's tree holds it: hash every root
        roots = hashlib.sha256()
        n_roots = 0
        for kind, publics in (("member", coms.publics), ("group", coms.group_publics), ("table", coms.table_publics)):
            for name, pub in sorted(publics.items()):
                if pub.root:
                    roots.update(kind.encode() + name.encode() + pub.root)
                    n_roots += 1
        assert n_roots, "no Merkle root"
        out["roots_sha256"], out["n_roots"] = roots.hexdigest(), n_roots
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
