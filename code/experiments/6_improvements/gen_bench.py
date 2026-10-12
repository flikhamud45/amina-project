"""Plan 2 F6 (chg 12): a generated response proved as one prefill, at scale, greedy and sampled, with E1's two tamper
cells.  The ``gen`` tier of ``slurm/sp2027.sh``.

    cd code && PYTHONPATH=src python experiments/6_improvements/gen_bench.py --model llama2-7b --prompt 512 \\
        --gen 64 256 --decode greedy sample --modes C Kpre --policy auto --fs --queries 3 --label gen
    cd code && PYTHONPATH=src python experiments/6_improvements/gen_bench.py --model tiny-llama --prompt 8 --gen 4 \\
        --decode greedy sample --modes C Kpre --fs --queries 2 --device cpu --threads 2      # a laptop check

Builds the decoder as ``perf.py --prune-last on`` does (random weights, seed 0, 32 calibration tokens, the last block
at the last position; ``--layers`` blocks, 0: all of them; ``tiny-*``/``small-*``: ``serve_ab.TINY``'s shapes) and,
in mode C, commits it once under ``--policy``: the commitment binds the weight ops, which
``transformer.with_lm_positions`` keeps, so one commitment serves every ``G``.  For each ``G`` and decoding rule:

* the ``G - 1`` tokens after each of ``--queries`` + 1 prompts of ``--prompt`` tokens (seed 1, as ``bench.py``) are
  generated on ``--device`` by ``transformer.greedy_tokens`` (``greedy``) or ``token_sampler.sample_tokens``
  (``sample``: the client's fresh 32-byte seed per prompt, SHA-256 of ``pvi/gen_bench`` and the prompt's index, and
  the integer rule ``TokenSampler(seed, --sampler-a, --top-k)``; random weights have no real logit scale, so the
  weight ``a`` is given directly).  Generation is not part of the proof's cost (``generate_s``, per response);
* each response is proved as one query on ``with_lm_positions(graph, G)`` (``run_query(wire=True, sampler=...)``, the
  lean prover, Kpre with the verifier's own embedding rows): one warm-up, then ``--queries`` timed queries per mode.
  ``run_query`` books the token rule (``Verifier.check_tokens``: one argmax, or the sampler's noise and choice, per
  generated position) to no phase, so it is timed here (``token_rule_s``) and added to ``verify_s``;
* E1's two tamper cells on the first response, each of which must be rejected: ``wrong_token`` (a generated token
  replaced by another, the tokens after it generated again with the same rule, so only that token breaks it;
  expected label ``token``) and ``forged_logits`` (the same response, the LM head's claimed logits at that position
  raised so that the rule chooses the wrong token; expected ``freivalds``, or ``columns_code`` for a plan whose LM
  head has the col layout).

One JSON line per (``G``, rule, mode), ``"kind": "gen"``: medians and ``[min, max]`` of ``prove_s`` and ``verify_s``
over the timed queries, every timing and byte part (medians), ``proof_mb_per_token`` (the proof over ``G``), the
tamper verdicts and labels, peak host (VmHWM, Linux) and GPU memory over the build and the commitment
(``build_peak``) and over each line's queries (``peak``).  The exit status is 1 if an honest response was rejected or
a tampered one accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[2]   # code/
RATE = 4            # bench.py's Reed-Solomon rate
CALIB_TOKENS = 32


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, CODE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def config(name: str):
    """A ``transformer.CONFIGS`` entry, or one of ``serve_ab.TINY``'s small shapes (laptop checks and tests)."""
    from pvi.fullcheck.transformer import CONFIGS
    if name in CONFIGS:
        return CONFIGS[name]
    return _load("serve_ab", "experiments/7_reexec/serve_ab.py").TINY[name]


def reset_peaks(device) -> None:
    """A new peak window: the GPU allocator's statistics and the host's VmHWM (Linux)."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def peaks(device) -> dict:
    """Bytes since :func:`reset_peaks`, named as ``bench.py`` records them (``host_peak_rss`` is ``None`` where
    ``/proc`` has none)."""
    out = {"host_peak_rss": None}
    try:
        with open("/proc/self/status") as fh:
            out["host_peak_rss"] = 1024 * next(int(line.split()[1]) for line in fh if line.startswith("VmHWM:"))
    except (OSError, StopIteration):
        pass
    if torch.device(device).type == "cuda":
        out["gpu_peak_memory"] = torch.cuda.max_memory_allocated(device)
        out["gpu_peak_reserved"] = torch.cuda.max_memory_reserved(device)
    return out


def seed_of(i: int) -> bytes:
    """The client's 32-byte seed for prompt ``i`` (fixed, so a run is reproducible)."""
    return hashlib.sha256(b"pvi/gen_bench" + i.to_bytes(4, "little")).digest()


def generate(graph, prompt: torch.Tensor, steps: int, rule: str, sampler, device) -> torch.Tensor:
    """``prompt [1, P]`` and the ``steps`` tokens the rule generates after it (on ``device``), on the CPU."""
    from pvi.fullcheck.token_sampler import sample_tokens
    from pvi.fullcheck.transformer import greedy_tokens
    with torch.no_grad():
        x = prompt.to(device)
        out = greedy_tokens(graph, x, steps) if rule == "greedy" else sample_tokens(graph, x, steps, sampler)
    return out.cpu()


def wrong_token(graph, x: torch.Tensor, prompt_len: int, i: int, rule: str, sampler, device) -> torch.Tensor:
    """``x`` with generated token ``i`` replaced by the next vocabulary index, the tokens after it generated again
    with the same rule (as ``tests/test_generation.py``'s ``_wrong_token``): only token ``i`` breaks the rule."""
    bad = x[:, :prompt_len + i + 1].clone()
    bad[0, -1] = (bad[0, -1] + 1) % graph.mat_ops[-1].n_rows
    return generate(graph, bad, x.shape[1] - bad.shape[1], rule, sampler, device)


class TokenTimer:
    """Wraps a verifier's ``check_tokens`` to time the token rule, which ``run_query`` books to no phase."""

    def __init__(self, verifier):
        self.inner, self.last = verifier.check_tokens, 0.0
        verifier.check_tokens = self

    def __call__(self, x, claims):
        t0 = time.perf_counter()
        try:
            return self.inner(x, claims)
        finally:
            self.last = time.perf_counter() - t0


def summary(runs: list[dict], tokens: list[float]) -> dict:
    """Medians of every part, and ``prove_s`` / ``verify_s`` (token rule included) with their ``[min, max]``."""
    keys = sorted({k for r in runs for k in r["timings"]})
    timing = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs) for k in keys}
    nbytes = {k: statistics.median(r["bytes"].get(k, 0) for r in runs) for k in sorted({k for r in runs
                                                                                       for k in r["bytes"]})}
    prove = [sum(r["timings"].get(k, 0.0) for k in r["timings"] if k.startswith("prove_") or k == "fs_hash")
             for r in runs]
    verify = [sum(r["timings"].get(k, 0.0) for k in r["timings"] if k.startswith("verify_") or k == "fs_hash") + t
              for r, t in zip(runs, tokens)]
    return {"timings": timing, "bytes": nbytes, "proof_bytes": sum(nbytes.values()),
            "prove_s": statistics.median(prove), "prove_s_range": [min(prove), max(prove)],
            "verify_s": statistics.median(verify), "verify_s_range": [min(verify), max(verify)],
            "token_rule_s": statistics.median(tokens)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--prompt", type=int, default=512, help="prompt tokens P")
    ap.add_argument("--gen", type=int, nargs="+", default=[64, 256], help="response tokens G (each > 1)")
    ap.add_argument("--decode", nargs="+", choices=["greedy", "sample"], default=["greedy", "sample"])
    ap.add_argument("--modes", nargs="+", choices=["C", "Kpre"], default=["C", "Kpre"])
    ap.add_argument("--policy", default="auto", help="mode C's commitment plan (pvi.fullcheck.plans)")
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--fs", action="store_true", help="Fiat-Shamir (non-interactive); each party's hashing timed")
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--sampler-a", type=int, default=1 << 12, help="the sampler's integer logit weight a")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--no-tampers", action="store_true", help="skip the two tamper cells")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verifier-device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if min(args.gen) < 2:
        raise SystemExit("--gen: a response of at least 2 tokens (G = 1 is the plain prefill of perf.py)")
    torch.set_num_threads(args.threads)
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
    from pvi.fullcheck.token_sampler import TokenSampler
    from pvi.fullcheck.transformer import build_decoder, with_lm_positions

    device = torch.device(args.device)
    cfg = config(args.model)
    n_layers = args.layers or cfg.n_layers
    if args.prompt + max(args.gen) - 1 > cfg.max_pos and cfg.pos == "learned":
        raise SystemExit(f"{args.model}: {args.prompt} + {max(args.gen)} - 1 positions > its {cfg.max_pos}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=CODE).stdout.strip()
    reset_peaks(device)
    t0 = time.perf_counter()
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.prompt, CALIB_TOKENS), seed=0,
                          prune_last=True)
    build_s = time.perf_counter() - t0
    shapes = decoder_shapes(cfg, prune_last=True)
    n_checks = len(shapes)                       # (r, t) for the full model's op count, as perf.py and bench.py
    coms, commit_s = None, None
    if "C" in args.modes:
        t0 = time.perf_counter()
        coms = commit_graph(graph, RATE, device=device, policy=args.policy, model_ops=shapes)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        commit_s = time.perf_counter() - t0
    build_peak = peaks(device)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    prompts = torch.randint(0, cfg.vocab, (args.queries + 1, args.prompt), generator=torch.Generator().manual_seed(1))
    head = {"kind": "gen", "label": args.label, "git": sha, "model": args.model, "layers": args.layers or None,
            "n_layers": n_layers, "prompt": args.prompt, "lam": args.lam, "rate": RATE, "fs": args.fs,
            "wire": True, "lean": True, "prune_last": True, "device": args.device,
            "verifier_device": args.verifier_device, "threads": args.threads, "build_s": build_s,
            "commit_s": commit_s, "setup_cache": None if coms is None else coms.setup_cache,
            "build_peak": build_peak, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    ok = True
    for g in args.gen:
        gen = with_lm_positions(graph, g)
        for rule in args.decode:
            samplers = [TokenSampler(seed_of(i), args.sampler_a, args.top_k) if rule == "sample" else None
                        for i in range(args.queries + 1)]
            t0 = time.perf_counter()
            xs = [generate(graph, prompts[i:i + 1], g - 1, rule, samplers[i], device) for i in range(len(samplers))]
            generate_s = (time.perf_counter() - t0) / len(xs)
            for mode in args.modes:
                reset_peaks(device)
                if mode == "C":
                    params = params_for(args.lam, n_checks, rate=RATE, plan=coms.plan, fiat_shamir=args.fs)
                    v = Verifier(gen.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                                 tables=coms.table_publics, lean=True, device=args.verifier_device)
                    prover = Prover(gen, device=device, commitments=coms, lean=True)
                else:
                    params = params_for(args.lam, n_checks, rate=RATE, fiat_shamir=args.fs)
                    v = Verifier(gen.public(), params, "Kpre", weights=weights, lean=True,
                                 device=args.verifier_device, lookups=True)
                    v.precompute(Challenger())
                    prover = Prover(gen, device=device, lean=True)
                timer = TokenTimer(v)
                warm = run_query(prover, v, xs[0], wire=True, sampler=samplers[0])
                runs, tokens = [], []
                for i in range(1, len(xs)):
                    runs.append(run_query(prover, v, xs[i], wire=True, sampler=samplers[i]))
                    tokens.append(timer.last)
                accepted = sum(bool(r["accepted"]) for r in runs) + int(bool(warm["accepted"]))
                tampers = {}
                if not args.no_tampers:
                    i_bad = max(0, (g - 1) // 2)              # a generated position in the middle of the response
                    bad = wrong_token(graph, xs[0], args.prompt, i_bad, rule, samplers[0], device)
                    res = run_query(prover, v, bad, wire=True, sampler=samplers[0])
                    tampers["wrong_token"] = {"position": i_bad, "rejected": not res["accepted"],
                                              "label": res["rejected_at"], "expected": ["token"]}
                    want = int(bad[0, args.prompt + i_bad])
                    head_op = gen.mat_ops[-1]
                    col = (coms is not None and coms.plan is not None and mode == "C"
                           and coms.plan.matrix_of(head_op.name).layout == "col")
                    # greedy: 1 above the largest logit; sampled: enough that a z_v + G[v] wins whatever the noise
                    # (|G| < 2^20, so a margin of 2^21 / a logit steps)
                    delta = 1 if rule == "greedy" else (1 << 21) // args.sampler_a + 2

                    def forge(op, z, name=head_op.name, j=i_bad, want=want, delta=delta):
                        if op.name == name:       # the wrong token's logit at column j above every other
                            z = z.clone()
                            z[want, j] = z[:, j].max() + delta
                        return z

                    res = run_query(prover, v, bad, wire=True, sampler=samplers[0], forward_kwargs={"tamper": forge})
                    tampers["forged_logits"] = {"position": i_bad, "rejected": not res["accepted"],
                                                "label": res["rejected_at"],
                                                "expected": ["columns_code"] if col else ["freivalds"]}
                line = {**head, "mode": mode, "policy": args.policy if mode == "C" else None,
                        **({"chosen": args.policy if coms.plan is None else coms.plan.policy} if mode == "C" else {}),
                        "gen": g, "decode": rule, "positions": int(xs[0].shape[1]),
                        **({"sampler": {"a": args.sampler_a, "top_k": args.top_k}} if rule == "sample" else {}),
                        "reps": params.reps, "columns": params.columns, "queries": len(runs),
                        "accepted": accepted, "warmup_accepted": bool(warm["accepted"]),
                        **summary(runs, tokens), "generate_s": generate_s, "tampers": tampers,
                        "peak": peaks(device)}
                line["proof_mb_per_token"] = line["proof_bytes"] / g / 1e6
                good = accepted == len(runs) + 1 and all(t["rejected"] for t in tampers.values())
                line["ok"] = good
                ok = ok and good
                print(json.dumps(line), flush=True)
                prover = v = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if not ok:
        print("FAILED: an honest response was rejected or a tampered one accepted", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
