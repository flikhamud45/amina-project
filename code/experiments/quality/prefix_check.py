"""Whether a fixed public prefix absorbs a Llama-family checkpoint's massive activations: the fp check behind the
Llama recipe's prefix rule (``--prefix auto`` or ``absorb``; ``hf_llama.prefixquant_tokens``,
``hf_llama.absorb_tokens``), without any integer graph.

    python -m experiments.quality.prefix_check --device cuda --out prefix_check.jsonl          # the five checkpoints
    python -m experiments.quality.prefix_check --models meta-llama/Llama-2-7b-hf --device cuda --per-block \\
        --prefixes Llama-2-7b-hf:29889,13,1
    python -m experiments.quality.prefix_check --models TinyLlama/TinyLlama_v1.1 --smoke --seq 128 \\
        --calib-windows 1 --device cpu                                   # a tiny random model (laptop)

A prefix serves the recipe only if it takes the massive activations: the integer graph's residual scale is set by the
fp residual stream's largest |value| over the calibration windows' own positions (``hf_llama._hidden_max``, the value
behind ``s_res``), so a window token whose activation exceeds the prefix's is a massive activation the prefix did not
absorb.  Per checkpoint (``--models``), on the recipe's calibration windows (``--calib-windows`` windows of ``--seq``
tokens of the train split's first ``--train-chars`` characters, tokenised without special tokens: draw 0 of
``ppl_llama.py``, ``checkpoints.llama_calibration``), each candidate prefix runs in front of every window and the
script prints the residual stream's largest |value| (the embedding's output and every block's output, the hook points
of ``_hidden_max``) over the prefix positions and over the windows' positions, their ratio (window / prefix; below
``1 / --absorb-margin``: absorbed) and where the windows' maximum sits: the block boundary (0: the embedding, ``b``:
block ``b``'s output), the window, the position in it, the channel and the token there; ``--per-block`` adds both
maxima per boundary.  The candidates: PrefixQuant's choice (``auto``), every candidate of the absorption rule
(``hf_llama.absorb_tokens`` with ``--absorb-margin`` and ``--absorb-max-len``: ``auto``'s choice, then each extension by
the next most frequent outlier token in front of the BOS), the BOS alone, the default further candidates of
:data:`EXTRA` (unless ``--no-extra``) and ``--prefixes`` (``<checkpoint name>:<comma-separated ids>``); each model's
summary names the prefix ``auto`` and ``absorb`` pick and whether the latter absorbs.

The checkpoints run in ``--dtype`` (default float16, to fit Llama-2-13B on one GPU) on ``--device``; the recipe's own
rule measures in fp32 on the CPU, so a ratio near 1 may come out on the other side there (``ppl_llama.py``'s record
keeps the rule's own measurement: ``prefix_choice.candidates``).

Output: text on stdout; with ``--out`` one JSON line per candidate (``model``, ``dtype``, ``seq``, ``calib_windows``,
``tokens``, ``text``, ``source``, ``prefix_max``, ``window_max``, ``ratio``, ``absorbed``, ``at``, with
``--per-block`` ``blocks``) and per checkpoint one summary line (``summary: true``, ``auto``, ``absorb``,
``absorb_choice``) appended to the file, with ``env``.

Hardware and time: fp only, one GPU of 48 GB for the five checkpoints in fp16 (Llama-2-13B: 26 GB) and about 32 GB of
host memory; about 20-30 minutes, mostly loading the checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.common import records
from experiments.common.checkpoints import CALIB_CHARS, load_llama, smoke_model, token_ids, wikitext, windows
from experiments.quality.ppl_llama import add_absorb_arguments
from pvi.models import hf_llama

MODELS = ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "TinyLlama/TinyLlama_v1.1", "Qwen/Qwen3-1.7B-Base",
          "Qwen/Qwen3-4B-Base")
"""The checkpoints of the text's Llama-family quality cells (``experiments/paper/macros.py``, ``LLAMAS``)."""
EXTRA = {"Llama-2-7b-hf": ("29889,13,1", "13,29889,1", "1,13"),
         "Qwen3-1.7B-Base": ("151643,151643",), "Qwen3-4B-Base": ("151643,151643",)}
"""Further candidates per checkpoint name: Llama-2-7B's prefix as PrefixQuant publishes it (``. \\n [BOS]``), the same
tokens in the other order and the BOS first; for Qwen3, ``<|endoftext|>`` twice (``llama_diag.py``'s diagnosis)."""


def where_max(model, calib: torch.Tensor, prefix: list[int], per_block: bool = False) -> dict:
    """The fp residual stream's largest |value| over the prefix positions and over the windows' positions of
    ``[prefix | window]`` for every window of ``calib [W, T]``, at the hook points of ``hf_llama._hidden_max`` (the
    embedding's output and every block's output), with where the windows' maximum sits (``at``: ``boundary``,
    ``window``, ``position``, ``channel``, ``token``) and, with ``per_block``, both maxima per boundary."""
    n = len(prefix)
    pre = torch.tensor([prefix], dtype=torch.int64)
    out = {"prefix_max": 0.0, "window_max": 0.0, "at": None}
    blocks: list[list[float]] = []
    state = {"w": 0, "b": 0}

    def hook(mod, inp, o):
        h = (o[0] if isinstance(o, tuple) else o).detach()[0].abs().float()
        b = state["b"]
        state["b"] += 1
        pm = float(h[:n].max())
        win = h[n:]
        wm = float(win.max())
        if len(blocks) <= b:
            blocks.append([0.0, 0.0])
        blocks[b] = [max(blocks[b][0], pm), max(blocks[b][1], wm)]
        out["prefix_max"] = max(out["prefix_max"], pm)
        if wm > out["window_max"]:
            pos, ch = divmod(int(win.argmax()), win.shape[1])
            out["window_max"] = wm
            out["at"] = {"boundary": b, "window": state["w"], "position": pos, "channel": ch,
                         "token": int(calib[state["w"], pos])}

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(hook)] + [layer.register_forward_hook(hook)
                                                                for layer in inner.layers]
    try:
        with torch.no_grad():
            for w in range(calib.shape[0]):
                state["w"], state["b"] = w, 0
                inner(torch.cat([pre, calib[w:w + 1]], 1).to(model.device))
    finally:
        for h in hooks:
            h.remove()
    out["ratio"] = out["window_max"] / out["prefix_max"] if out["prefix_max"] > 0 else None
    if per_block:
        out["blocks"] = [{"boundary": b, "prefix_max": p, "window_max": m} for b, (p, m) in enumerate(blocks)]
    return out


def candidates(name: str, bos: int, auto: list[int], absorb: dict, extra: bool, given: list[str]) -> list[tuple]:
    """``[(tokens, source)]``: ``auto``'s choice, the absorption rule's candidates, the BOS alone, :data:`EXTRA`'s
    (with ``extra``) and ``given``'s for the checkpoint ``name`` (``<name>:<ids>``), each prefix once (its first
    source)."""
    short = name.split("/")[-1]
    out: list[tuple] = [(list(auto), "auto")]
    out += [(list(c["tokens"]), f"absorb candidate {i}") for i, c in enumerate(absorb["candidates"])]
    out.append(([bos], "bos"))
    out += [([int(t) for t in s.split(",")], "extra") for s in (EXTRA.get(short, ()) if extra else ())]
    for g in given:
        who, _, ids = g.rpartition(":")
        if who in (name, short):
            out.append(([int(t) for t in ids.split(",")], "given"))
    seen, unique = set(), []
    for tokens, source in out:
        if tuple(tokens) not in seen:
            seen.add(tuple(tokens))
            unique.append((tokens, source))
    return unique


def check_model(args, name: str, emit) -> dict:
    """The candidates of one checkpoint (module docstring); returns its summary record."""
    if args.smoke:
        model, tok = smoke_model(name)
        model = model.to(args.device)
    else:
        model, tok = load_llama(name, getattr(torch, args.dtype))
        model = model.to(args.device)
    train = token_ids(tok, wikitext("train")[:args.train_chars], "auto")
    calib = torch.cat(windows(train, args.seq, args.calib_windows))
    with torch.no_grad():
        auto, _ = hf_llama.prefixquant_tokens(model, calib)
        chosen, absorb = hf_llama.absorb_tokens(model, calib, margin=args.absorb_margin, max_len=args.absorb_max_len)
    bos = auto[-1]
    text = lambda ids: [tok.decode([i]) for i in ids]          # noqa: E731
    print(f"model {name} ({args.dtype} on {args.device}; {args.calib_windows} x {args.seq} calibration tokens)",
          flush=True)
    for tokens, source in candidates(name, bos, auto, absorb, not args.no_extra, args.prefixes):
        r = where_max(model, calib, tokens, args.per_block)
        absorbed = r["window_max"] * args.absorb_margin < r["prefix_max"]
        at = r["at"]
        ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.3f}"
        print(f"  prefix {tokens} {text(tokens)} ({source}): prefix max {r['prefix_max']:.1f}, window max "
              f"{r['window_max']:.1f}, ratio {ratio} ({'absorbed' if absorbed else 'not absorbed'}); window max at "
              f"boundary {at['boundary']}, window {at['window']}, position {at['position']}, channel {at['channel']}, "
              f"token {at['token']} {text([at['token']])}", flush=True)
        for b in r.get("blocks", []):
            print(f"      boundary {b['boundary']:2d}: prefix max {b['prefix_max']:9.1f}  window max "
                  f"{b['window_max']:9.1f}", flush=True)
        emit({"model": name, "smoke": args.smoke, "dtype": args.dtype, "seq": args.seq,
              "calib_windows": args.calib_windows, "tokens": tokens, "text": text(tokens), "source": source,
              "margin": args.absorb_margin, "absorbed": absorbed, **r, "at_text": text([at["token"]])})
    print(f"  auto picks {auto} {text(auto)}; absorb (margin {args.absorb_margin:g}, at most {args.absorb_max_len} "
          f"tokens) picks {chosen} {text(chosen)}: {'absorbed' if absorb['absorbed'] else 'not absorbed'} "
          f"(stopped: {absorb['stop']})", flush=True)
    summary = {"model": name, "smoke": args.smoke, "summary": True, "dtype": args.dtype, "auto": auto,
               "absorb": chosen, "absorb_choice": absorb}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="Llama or Qwen3 checkpoints")
    ap.add_argument("--seq", type=int, default=2048, help="tokens per calibration window")
    ap.add_argument("--calib-windows", type=int, default=4, help="calibration windows")
    ap.add_argument("--train-chars", type=int, default=CALIB_CHARS, help="characters of the train split tokenised")
    ap.add_argument("--prefixes", nargs="*", default=[],
                    help="further candidates, <checkpoint name>:<comma-separated token ids> (the name as --models has "
                         "it or its last part)")
    ap.add_argument("--no-extra", action="store_true", help="without the default further candidates (EXTRA)")
    ap.add_argument("--per-block", action="store_true", help="both maxima per block boundary")
    add_absorb_arguments(ap)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="the checkpoints' type (the smoke models stay in float32)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="2-block random models with the checkpoints' vocabularies")
    ap.add_argument("--threads", type=int, default=8, help="torch's intra-op threads")
    ap.add_argument("--out", default=None, help="a JSONL file the records are appended to")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    env = records.environment(args.device)
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row: dict) -> None:
        if out is not None:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(row, env=env), default=float) + "\n")

    for name in args.models:
        emit(check_model(args, name, emit))
    print("prefix check done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
