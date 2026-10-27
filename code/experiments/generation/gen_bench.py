"""A generated response proved as one prefill: prover and verifier time and proof size, greedy and sampled.

Produces the generation cells of the final run (Sec. ``sec:gen-prefill``'s cost paragraph, ``tab:gen`` and
``tab:gen-full`` in the extended version: Llama-2-7B and Qwen3-4B with a 512-token prompt, ``G`` in {64, 256},
greedy and sampled, with the two tamper cells), the run package's ``gen`` tier, and with ``--real`` the generated
responses of the real Llama-2-7B in the ``real`` tier.

    python -m experiments.generation.gen_bench --model llama2-7b --prompt 512 --gen 64 256 --decode greedy sample \\
        --modes C Kpre --queries 3 --label gen
    python -m experiments.generation.gen_bench --model llama2-7b --real meta-llama/Llama-2-7b-hf --prompt 512 \\
        --gen 64 --decode greedy sample --modes C Kpre --queries 3 --label real
    python -m experiments.generation.gen_bench --model tiny-llama --prompt 8 --gen 4 --decode greedy sample \\
        --modes C Kpre --queries 2 --device cpu --threads 2      # a laptop check (seconds)

Builds the decoder as ``costs/perf.py`` does (random weights, ``--seed``, ``--calib-tokens`` calibration tokens;
``--layers`` blocks, 0: all of them; ``tiny-*``/``small-*``: the shapes of ``experiments.common.models.TINY``) and, in
mode C, commits it once under ``--policy``: the commitment binds the weight ops, which
``decoder.with_lm_positions`` keeps, so one commitment serves every ``G``.  The protocol settings default to the
paper's configuration (:data:`experiments.common.cli.PAPER`: plan ``cnn18c``, Fiat--Shamir, the compact encoding,
the lean prover and verifier, the last block at the last position, mode Kpre's verifier reading the embedding rows
itself, 11-bit attention probabilities).  For each ``G`` and decoding rule:

* the ``G - 1`` tokens after each of ``--queries`` + 1 prompts of ``--prompt`` tokens (seed ``--prompt-seed``) are
  generated on ``--device`` by ``decoder.greedy_tokens`` (``greedy``) or ``token_sampler.sample_tokens``
  (``sample``: the client's fresh 32-byte seed per prompt, SHA-256 of ``pvi/gen_bench`` and the prompt's index, and
  the integer rule ``TokenSampler(seed, --sampler-a, --top-k)``; random weights have no real logit scale, so the
  weight ``a`` is given directly).  Generation is not part of the proof's cost (``generate_s``, per response);
* each response is proved as one query on ``with_lm_positions(graph, G)`` (``run_query(sampler=...)``): one warm-up,
  then ``--queries`` timed queries per mode.  ``run_query`` books the token rule (``Verifier.check_tokens``: one
  argmax, or the sampler's noise and choice, per generated position) to no phase, so it is timed here
  (``token_rule_s``) and added to ``verify_s``;
* two tamper cells on the first response, each of which must be rejected: ``wrong_token`` (a generated token
  replaced by another, the tokens after it generated again with the same rule, so only that token breaks it;
  expected label ``token``) and ``forged_logits`` (the same response, the LM head's claimed logits at that position
  raised so that the rule chooses the wrong token; expected ``freivalds``, or ``columns_code`` for a plan whose LM
  head has the col layout).

``--real <checkpoint>``: the checkpoint's integer graph instead (:func:`experiments.common.checkpoints.real_graph`:
OPT with the certified recipe ``hf_opt.CERTIFIED_RECIPE``, a Llama-family checkpoint with ``hf_llama.MAIN_RECIPE``
and its fixed public prefix; ``--model`` names the benchmark shape it has), every block, or with ``--layers n`` its
first ``n`` blocks with the same calibration and prefix, the full graph's first ``n``
(:mod:`experiments.common.real_cells`); the prompts are the first WikiText-2 test windows of ``--prompt`` tokens, mode C commits the full
graph's own weight ops (a build of a few blocks: planned and sized for the whole model, as a random-weight build), and
the sampler's weight is ``a = round(c 2^16 / tau)`` for the LM head's real logit scale ``c``
(``TokenSampler.for_temperature``, ``--temperature``), so sampled responses are draws of the integer model at that
temperature.

**Record** (one JSON line per ``G``, rule and mode, on stdout; ``"kind": "gen"``): the settings (``model``,
``layers``, ``n_layers``, ``prompt``, ``gen``, ``decode``, ``mode``, ``policy``, ``chosen``, ``lam``, ``rate``, ``fs``,
``wire``, ``lean``, ``prune_last``, ``prob_max`` (the graph's scale, :func:`pvi.models.decoder.prob_max_of`),
``lookups``, ``sampler``, ``real``, ``temperature``, ``logit_scale``, ``device``, ``verifier_device``, ``threads``),
``positions``, ``reps``, ``columns``, ``queries``, ``accepted``,
``warmup_accepted``, ``warmup_transcript``, the medians and ``[min, max]`` of ``prove_s`` and ``verify_s`` over the
timed queries, every timing and byte part (medians), every timed query's own values (``prove_s_all``,
``verify_s_all``, ``timings_all``, ``token_rule_s_all``, and with Fiat--Shamir ``transcripts``), ``proof_bytes``, ``proof_mb_per_token`` (the proof over ``G``), ``token_rule_s``,
``generate_s``, ``tampers`` (verdicts and labels), ``ok``, peak host and GPU memory over the build and the commitment
(``build_peak``) and over each line's queries (``peak``), ``build_s``, ``commit_s``, ``setup_cache``, ``env``.  The
exit status is 1 if an honest response was rejected or a tampered one accepted.

Hardware and time: the final run's cells run on an NVIDIA L40S prover with 8 CPU threads for the verifier; the
package gives each 7B model (both ``G``, both rules, both modes) a job limit of three hours and 192 GB of host memory
(generating the responses, without a KV cache, is a large part of it).  The laptop check above takes seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import statistics
import sys
import time

import torch

from experiments.common import cli, real_cells, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE


def seed_of(i: int) -> bytes:
    """The client's 32-byte seed for prompt ``i`` (fixed, so a run is reproducible)."""
    return hashlib.sha256(b"pvi/gen_bench" + i.to_bytes(4, "little")).digest()


def generate(graph, prompt: torch.Tensor, steps: int, rule: str, sampler, device) -> torch.Tensor:
    """``prompt [1, P]`` and the ``steps`` tokens the rule generates after it (on ``device``), on the CPU."""
    from pvi.models.decoder import greedy_tokens
    from pvi.v0.token_sampler import sample_tokens
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
    """Medians of every part, and ``prove_s`` / ``verify_s`` (token rule included) with their ``[min, max]``, and
    every response's own values (:func:`experiments.common.records.per_query`: ``prove_s_all``, ``verify_s_all``,
    ``timings_all``, ``transcripts``; ``token_rule_s_all``)."""
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
            "token_rule_s": statistics.median(tokens), "token_rule_s_all": list(tokens),
            **records.per_query(runs, prove, verify)}


def forge_margin(rule: str, sampler) -> int:
    """How far above every other logit the ``forged_logits`` tamper puts the wrong token's, in logit steps, so that the
    rule chooses it and the forgery is self-consistent (caught by the weight check, not the token check): greedy, 1;
    sampled, enough that ``a z_v + G[v]`` wins whatever the noise (``|G| < 2^20``, so ``2^21 / a`` steps), with the
    weight ``a`` of the sampler actually used (``--real`` derives it from the LM head's logit scale, not
    ``--sampler-a``)."""
    return 1 if rule == "greedy" else (1 << 21) // sampler.a + 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True,
                    help="the decoder's shapes (pvi.models.decoder.CONFIGS, experiments.common.models.TINY)")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--prompt", type=int, default=512, help="prompt tokens P")
    ap.add_argument("--gen", type=int, nargs="+", default=[64, 256], help="response tokens G (each > 1)")
    ap.add_argument("--decode", nargs="+", choices=["greedy", "sample"], default=["greedy", "sample"])
    ap.add_argument("--modes", nargs="+", choices=["C", "Kpre"], default=["C", "Kpre"])
    ap.add_argument("--queries", type=int, default=3, help="timed responses per cell, after one warm-up")
    ap.add_argument("--sampler-a", type=int, default=1 << 12,
                    help="random weights: the sampler's integer logit weight a")
    ap.add_argument("--top-k", type=int, default=0, help="the sampler's candidates (0: every token)")
    ap.add_argument("--no-tampers", action="store_true", help="skip the two tamper cells")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of the random-weight decoder (at most --prompt)")
    ap.add_argument("--label", default="", help="a label copied into every record")
    real_cells.add_arguments(ap, help="a Hugging Face checkpoint: its integer graph; --model names its shapes")
    ap.add_argument("--temperature", type=float, default=1.0, help="with --real: the sampler's temperature")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8)
    return ap


def main() -> int:
    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder, prob_max_of, with_lm_positions
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.token_sampler import TokenSampler
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier

    args = build_parser().parse_args()
    if min(args.gen) < 2:
        raise SystemExit("--gen: a response of at least 2 tokens (G = 1 is the plain prefill of perf.py)")
    cfg = decoder_config(args.model)
    real_cells.check_args(args, cfg, seqs=[args.prompt + max(args.gen) - 1], layers=[args.layers],
                          prob_max=args.prob_max)
    cli.apply_runtime(args)
    device = torch.device(args.device)
    n_layers = args.layers or cfg.n_layers
    if args.prompt + max(args.gen) - 1 > cfg.max_pos and cfg.pos == "learned":
        raise SystemExit(f"{args.model}: {args.prompt} + {max(args.gen)} - 1 positions > its {cfg.max_pos}")
    sha = records.git_sha(CODE)
    records.reset_peaks(device)
    t0 = time.perf_counter()
    real_info, real_prompts = None, None
    if args.real:                                # the checkpoint's integer graph
        part = real_cells.partial(cfg, args.layers)     # None: every block; else the checkpoint's first ones
        ck = real_cells.load(args.real, cfg, seq=args.prompt, n_queries=args.queries + 1, generated=max(args.gen) - 1)
        graph, real_info = ck.graph(args.prune_last, part)
        ck.free()
        real_prompts = torch.cat(ck.queries[:args.queries + 1])
        # the full model: mode C plans the graph's own weight ops; a build of its first blocks the whole model's
        shapes = None if part is None else real_cells.model_ops(cfg, args.prune_last)
        n_checks = len(graph.mat_ops) if part is None else len(shapes)
    else:
        graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.prompt, args.calib_tokens), seed=args.seed,
                              prune_last=args.prune_last, prob_max=args.prob_max)
        shapes = decoder_shapes(cfg, prune_last=args.prune_last)
        n_checks = len(shapes)                   # (r, t) for the full model's op count, as perf.py and bench.py
    build_s = time.perf_counter() - t0
    coms, commit_s = None, None
    if "C" in args.modes:
        records.sync(device)
        t0 = time.perf_counter()
        coms = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=shapes,
                            cache=args.setup_cache)
        records.sync(device)
        commit_s = time.perf_counter() - t0
    build_peak = records.peaks(device)
    env = records.environment(device)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    prompts = (real_prompts if real_prompts is not None else
               torch.randint(0, cfg.vocab, (args.queries + 1, args.prompt),
                             generator=torch.Generator().manual_seed(args.prompt_seed)))
    head = {"kind": "gen", "label": args.label, "git": sha, "model": args.model, "layers": args.layers or None,
            "n_layers": n_layers, "prompt": args.prompt, "lam": args.lam, "rate": args.rate, "fs": args.fs,
            "wire": args.wire, "lean": args.lean, "prune_last": args.prune_last, "prob_max": prob_max_of(graph),
            "device": args.device,
            "verifier_device": args.verifier_device, "threads": args.threads, "build_s": build_s,
            "commit_s": commit_s, "setup_cache": None if coms is None else coms.setup_cache,
            "real": args.real or None, "temperature": args.temperature if args.real else None,
            "logit_scale": None if real_info is None else float(real_info["logit_scale"]),
            "build_peak": build_peak, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "env": env}
    ok = True
    for g in args.gen:
        gen = with_lm_positions(graph, g)
        for rule in args.decode:
            samplers = [None if rule != "sample" else
                        TokenSampler.for_temperature(seed_of(i), args.temperature, float(real_info["logit_scale"]),
                                                     args.top_k) if real_info is not None else
                        TokenSampler(seed_of(i), args.sampler_a, args.top_k)
                        for i in range(args.queries + 1)]
            t0 = time.perf_counter()
            xs = [generate(graph, prompts[i:i + 1], g - 1, rule, samplers[i], device) for i in range(len(samplers))]
            generate_s = (time.perf_counter() - t0) / len(xs)
            for mode in args.modes:
                records.reset_peaks(device)
                if mode == "C":
                    params = params_for(args.lam, n_checks, rate=args.rate, plan=coms.plan, fiat_shamir=args.fs)
                    v = Verifier(gen.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                                 tables=coms.table_publics, lean=args.lean, device=args.verifier_device)
                    prover = Prover(gen, device=device, commitments=coms, lean=args.lean)
                else:
                    params = params_for(args.lam, n_checks, rate=args.rate, fiat_shamir=args.fs)
                    v = Verifier(gen.public(), params, "Kpre", weights=weights, lean=args.lean,
                                 device=args.verifier_device, lookups=args.lookups)
                    v.precompute(Challenger())
                    prover = Prover(gen, device=device, lean=args.lean)
                timer = TokenTimer(v)
                warm = run_query(prover, v, xs[0], wire=args.wire, sampler=samplers[0])
                runs, tokens = [], []
                for i in range(1, len(xs)):
                    runs.append(run_query(prover, v, xs[i], wire=args.wire, sampler=samplers[i]))
                    tokens.append(timer.last)
                accepted = sum(bool(r["accepted"]) for r in runs) + int(bool(warm["accepted"]))
                tampers = {}
                if not args.no_tampers:
                    i_bad = max(0, (g - 1) // 2)              # a generated position in the middle of the response
                    bad = wrong_token(graph, xs[0], args.prompt, i_bad, rule, samplers[0], device)
                    res = run_query(prover, v, bad, wire=args.wire, sampler=samplers[0])
                    tampers["wrong_token"] = {"position": i_bad, "rejected": not res["accepted"],
                                              "label": res["rejected_at"], "expected": ["token"]}
                    want = int(bad[0, args.prompt + i_bad])
                    head_op = gen.mat_ops[-1]
                    col = (coms is not None and coms.plan is not None and mode == "C"
                           and coms.plan.matrix_of(head_op.name).layout == "col")
                    delta = forge_margin(rule, samplers[0])

                    def forge(op, z, name=head_op.name, j=i_bad, want=want, delta=delta):
                        if op.name == name:       # the wrong token's logit at column j above every other
                            z = z.clone()
                            z[want, j] = z[:, j].max() + delta
                        return z

                    res = run_query(prover, v, bad, wire=args.wire, sampler=samplers[0],
                                    forward_kwargs={"tamper": forge})
                    tampers["forged_logits"] = {"position": i_bad, "rejected": not res["accepted"],
                                                "label": res["rejected_at"],
                                                "expected": ["columns_code"] if col else ["freivalds"]}
                line = {**head, "mode": mode, "policy": args.policy if mode == "C" else None,
                        **({"chosen": args.policy if coms.plan is None else coms.plan.policy} if mode == "C" else {}),
                        "lookups": args.lookups if mode != "C" else False,
                        "gen": g, "decode": rule, "positions": int(xs[0].shape[1]),
                        **({"sampler": {"a": samplers[0].a, "top_k": args.top_k}} if rule == "sample" else {}),
                        "reps": params.reps, "columns": params.columns, "queries": len(runs),
                        "accepted": accepted, "warmup_accepted": bool(warm["accepted"]),
                        "warmup_transcript": warm.get("transcript"),
                        **summary(runs, tokens), "generate_s": generate_s, "tampers": tampers,
                        "peak": records.peaks(device)}
                line["proof_mb_per_token"] = line["proof_bytes"] / g / 1e6
                good = accepted == len(runs) + 1 and all(t["rejected"] and t["label"] in t["expected"]
                                                         for t in tampers.values())
                line["ok"] = good
                ok = ok and good
                records.emit(line)
                prover = v = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if not ok:
        print("FAILED: an honest response was rejected or a tampered one accepted", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
