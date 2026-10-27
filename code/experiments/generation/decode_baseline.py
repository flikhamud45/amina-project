"""Unverified generation on the prover's GPU: a greedy response decoded with a KV cache, the provider's work without
a proof.

Produces the baseline of the generation cells (``ext_generation.tex``: the prover's time for a generated response
against generating it without a proof; the run package's ``gen`` tier, job ``gen-decode``): Llama-2-7B and Qwen3-4B
after a 512-token prompt, ``G`` in {64, 256, 512}, as the ``gen`` tier's ``gen_bench.py`` proves them.

    python -m experiments.generation.decode_baseline --model llama2-7b qwen3-4b --prompt 512 --gen 64 256 512 \\
        --reps 5 --label gen
    python -m experiments.generation.decode_baseline --model tiny-llama tiny-qwen --prompt 8 --gen 4 --reps 2 \\
        --device cpu --dtype fp32 --threads 2      # a laptop check (seconds)

Each ``--model`` (``pvi.models.decoder.CONFIGS`` or ``experiments.common.models.TINY``: a Llama- or Qwen3-style
decoder, RMSNorm, SwiGLU and rotary positions) is Hugging Face Transformers' own decoder of those shapes
(``LlamaForCausalLM``, or ``Qwen3ForCausalLM`` for a decoder with q/k norms; :func:`hf_config`), built on
``--device`` in ``--dtype`` with random weights (Transformers' initialisation, ``--seed``; ``--layers`` blocks, 0: all
of them) and scaled-dot-product attention.  The time of a decoder does not depend on its weights' values.  A response
of ``G`` tokens after a ``P``-token prompt is the response ``gen_bench.py`` proves: the prefill of the prompt gives the
first token (only the last position's logits are computed), then ``G - 1`` decoding steps of one token each read and
extend the KV cache; every token is the argmax of its logits, on the device.  Per ``G``: one untimed response, then
``--reps`` timed responses to prompts of random token ids (seed ``--prompt-seed`` and the response's index).  Each
response's prefill and decoding are timed apart (``experiments.common.records.timed``: CUDA events after a
synchronisation on a GPU).  The last timed response is also checked against one forward pass of the model over the
prompt and its first ``G - 1`` tokens (no cache): ``same_as_prefill`` is the share of the ``G`` tokens that pass's
argmaxes give again (below 1 only where floating-point rounding differs between the two paths).

**Record** (one JSON line per model and ``G`` on stdout; ``"kind": "decode"``): ``model``, ``layers``, ``n_layers``,
``prompt``, ``gen``, ``dtype``, ``device``, ``attention``, ``params``, ``weights_bytes``, ``reps``, the medians and
``[min, max]`` of ``prefill_s``, ``decode_s`` (the ``G - 1`` steps) and ``total_s`` (both), ``ms_per_token`` (the
median ``total_s`` over ``G``), ``same_as_prefill``, ``build_s``, ``peak`` (host and GPU memory over the timed
responses), ``label``, ``host``, ``gpu``, ``git``, ``env``, ``ok`` (every logit finite and ``G`` tokens per response).
The exit status is 1 if a record is not ``ok``.

Hardware and time: the final run's job runs on the L40S (fp16; each 7B model takes 13.5 GB of GPU memory) in a few
minutes; the package gives it a limit of one hour and 64 GB of host memory.  The laptop check above takes seconds.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import platform
import statistics
import sys
import time

import torch

from experiments.common import records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def hf_config(cfg, n_layers: int):
    """Transformers' configuration of the decoder shapes ``cfg`` (a ``DecoderConfig``) with ``n_layers`` blocks:
    ``Qwen3Config`` for a decoder with q/k norms, else ``LlamaConfig``; SystemExit for another architecture."""
    import transformers
    if (cfg.norm, cfg.mlp, cfg.pos, cfg.bias) != ("rmsnorm", "swiglu", "rope", False):
        raise SystemExit(f"{cfg.name}: decode_baseline builds Llama- and Qwen3-style decoders (RMSNorm, SwiGLU, "
                         "rotary positions, no biases)")
    kw = dict(vocab_size=cfg.vocab, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff, num_hidden_layers=n_layers,
              num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
              max_position_embeddings=cfg.max_pos, rope_theta=cfg.rope_theta, tie_word_embeddings=cfg.tied,
              attention_bias=False, use_cache=True)
    if cfg.qk_norm:
        return transformers.Qwen3Config(**kw, max_window_layers=n_layers, use_sliding_window=False,
                                          sliding_window=None)
    return transformers.LlamaConfig(**kw, mlp_bias=False)


def build(cfg, n_layers: int, dtype: torch.dtype, device: torch.device, seed: int):
    """Transformers' decoder of ``cfg``'s shapes with ``n_layers`` blocks and random weights, on ``device`` in
    ``dtype``, eval mode, scaled-dot-product attention."""
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(hf_config(cfg, n_layers), torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    return model.eval()


def _keep_last(model) -> dict:
    """The forward's argument that computes the LM head at the last position only, where the decoder takes it."""
    params = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def respond(model, x: torch.Tensor, g: int, device: torch.device) -> tuple[torch.Tensor, float, float, bool]:
    """``(tokens [1, g], prefill seconds, decoding seconds, every logit finite)``: the greedy response of ``g``
    tokens to the prompt ``x [1, P]`` (on ``device``), the prefill (the first token) and the ``g - 1`` decoding steps
    timed apart."""
    keep = _keep_last(model)
    state: dict = {}

    def prefill():
        out = model(input_ids=x, use_cache=True, **keep)
        state["past"] = out.past_key_values
        logits = out.logits[:, -1]
        state["finite"] = torch.isfinite(logits).all()
        state["tokens"] = [logits.argmax(-1, keepdim=True)]

    def decode():
        tok = state["tokens"][0]
        for _ in range(g - 1):
            out = model(input_ids=tok, past_key_values=state["past"], use_cache=True)
            state["past"] = out.past_key_values
            logits = out.logits[:, -1]
            state["finite"] = state["finite"] & torch.isfinite(logits).all()
            tok = logits.argmax(-1, keepdim=True)
            state["tokens"].append(tok)

    with torch.no_grad():
        _, t_prefill, _ = records.timed(prefill, device)
        _, t_decode, _ = records.timed(decode, device)
    tokens = torch.cat(state["tokens"], 1)
    return tokens, t_prefill, t_decode, bool(state["finite"])


def same_as_prefill(model, x: torch.Tensor, tokens: torch.Tensor) -> float:
    """The share of the response ``tokens [1, G]`` that the argmaxes of one forward pass (no cache) over the prompt
    ``x`` and the response's first ``G - 1`` tokens give again, at the last ``G`` positions."""
    g = tokens.shape[1]
    with torch.no_grad():
        logits = model(input_ids=torch.cat([x, tokens[:, :-1]], 1), use_cache=False).logits[:, -g:]
    return float((logits.argmax(-1) == tokens).float().mean())


def _stats(values: list[float]) -> tuple[float, list[float]]:
    return statistics.median(values), [min(values), max(values)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", nargs="+", required=True,
                    help="decoder shapes (pvi.models.decoder.CONFIGS, experiments.common.models.TINY): Llama- or "
                         "Qwen3-style")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--prompt", type=int, default=512, help="prompt tokens P")
    ap.add_argument("--gen", type=int, nargs="+", default=[64, 256, 512], help="response tokens G (each >= 1)")
    ap.add_argument("--reps", type=int, default=5, help="timed responses per G, after one untimed")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16", help="the decoder's weights and activations")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="where it decodes")
    ap.add_argument("--threads", type=int, default=8, help="torch's intra-op threads")
    ap.add_argument("--label", default="", help="a label copied into every record")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.gen) < 1 or args.reps < 1 or args.prompt < 1:
        raise SystemExit("--gen, --reps and --prompt: at least 1")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    sha = records.git_sha(CODE)
    ok = True
    for name in args.model:
        cfg = decoder_config(name)
        n_layers = args.layers or cfg.n_layers
        if args.prompt + max(args.gen) - 1 > cfg.max_pos:
            raise SystemExit(f"{name}: {args.prompt} + {max(args.gen)} - 1 positions > its {cfg.max_pos}")
        t0 = time.perf_counter()
        model = build(cfg, n_layers, dtype, device, args.seed)
        records.sync(device)
        build_s = time.perf_counter() - t0
        n_params = sum(p.numel() for p in model.parameters())
        head = {"kind": "decode", "label": args.label, "git": sha, "model": name, "layers": args.layers or None,
                "n_layers": n_layers, "prompt": args.prompt, "dtype": args.dtype, "device": args.device,
                "attention": model.config._attn_implementation, "params": n_params,
                "weights_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "threads": args.threads, "build_s": build_s, "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "env": records.environment(device)}
        gen = torch.Generator().manual_seed(args.prompt_seed)
        prompts = torch.randint(0, cfg.vocab, (len(args.gen) * (args.reps + 1), args.prompt), generator=gen)
        for j, g in enumerate(args.gen):
            xs = prompts[j * (args.reps + 1):(j + 1) * (args.reps + 1)].to(device)
            respond(model, xs[:1], g, device)                       # untimed
            records.reset_peaks(device)
            pre, dec, tot, good = [], [], [], True
            for i in range(1, args.reps + 1):
                tokens, tp, td, finite = respond(model, xs[i:i + 1], g, device)
                pre.append(tp)
                dec.append(td)
                tot.append(tp + td)
                good = good and finite and tokens.shape[1] == g
            peak = records.peaks(device)
            same = same_as_prefill(model, xs[args.reps:args.reps + 1], tokens)
            (p_med, p_rng), (d_med, d_rng), (t_med, t_rng) = _stats(pre), _stats(dec), _stats(tot)
            line = {**head, "gen": g, "reps": args.reps, "prefill_s": p_med, "prefill_s_range": p_rng,
                    "decode_s": d_med, "decode_s_range": d_rng, "total_s": t_med, "total_s_range": t_rng,
                    "ms_per_token": t_med / g * 1e3, "same_as_prefill": same, "peak": peak, "ok": good}
            records.emit(line)
            ok = ok and good
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not ok:
        print("FAILED: a response with a non-finite logit or of the wrong length", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
