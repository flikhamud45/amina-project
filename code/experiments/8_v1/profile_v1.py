"""Profile one V1 query (``cut_protocol.run_cut_query``) after a warm-up: the phase timings and the functions that
take the time (cProfile, cumulative), for the prover and the verifier.

    cd code && PYTHONPATH=src python experiments/8_v1/profile_v1.py --model opt-125m --seq 2048 --mode Kpre --lean
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mode", default="Kpre")
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--lmax", type=int, default=25)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--top", type=int, default=45)
    args = ap.parse_args()
    torch.set_num_threads(8)
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers or cfg.n_layers, calib_tokens=32, seed=0, prune_last=True)
    n_checks = len(decoder_shapes(cfg, prune_last=True))
    if args.mode == "C":
        coms = commit_graph(graph, 4, device="cuda", policy=args.policy, model_ops=decoder_shapes(cfg, prune_last=True))
        params = params_for(128, n_checks, fiat_shamir=True, plan=coms.plan, cut=True, cut_lmax=args.lmax)
        v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                     tables=coms.table_publics, lean=args.lean)
        prover = Prover(graph, device="cuda", commitments=coms, lean=args.lean)
    else:
        params = params_for(128, n_checks, fiat_shamir=True, cut=True, cut_lmax=args.lmax)
        v = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops},
                     lean=args.lean)
        v.precompute(Challenger())
        prover = Prover(graph, device="cuda", lean=args.lean)
    x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(1))
    warm = cp.run_cut_query(prover, v, x)
    print("warm-up accepted", warm["accepted"], warm["rejected_at"])
    pr = cProfile.Profile()
    pr.enable()
    out = cp.run_cut_query(prover, v, torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(2)))
    pr.disable()
    print("accepted", out["accepted"], out["rejected_at"], "peak GPU %.2f GB" % (torch.cuda.max_memory_allocated() / 2**30))
    print({k: round(t, 3) for k, t in out["timings"].items() if t > 0.01})
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(args.top)
    print("\n".join(line[:180] for line in buf.getvalue().splitlines()))


if __name__ == "__main__":
    main()
