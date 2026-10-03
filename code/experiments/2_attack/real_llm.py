"""The single-neuron attack and a trigger-conditional backdoor on a real LLM (report Sec. 4.2, *A real LLM*).

The rest of the project shows the attack on MNIST, the CNNs and LLM shapes with random
weights. This script shows it on a real checkpoint.

It hooks one FFN neuron -- the post-ReLU input of ``fc2`` -- at the *last* position of
one decoder layer, and searches for the smallest change to it that flips the
next-token prediction. The search mirrors ``pvi.attacks.tamper.smallest_flipping_value``:
candidates ranked by the gradient of the logit margin, then a geometric bracket and a
bisection on the activation value, kept non-negative because it is a ReLU output.

Two blocks:

* ``untargeted`` -- does one neuron suffice to change the next token, how small can the
  change be, and does the forged value stay inside that neuron's natural range?
* ``backdoor`` -- a trigger string appended to the prompt. The server serves the honest
  trace on clean prompts and tampers one neuron on triggered ones to force a fixed
  target token. Clean predictions are identical to the model's by construction; the
  script re-checks that rather than assuming it.

Only the last position changes, and causal attention means no earlier position depends
on it, so the prompt prefix is run once and every trial is a single-token forward
against its cached keys and values.

Detection under the path test uses the repository's own convention
(``experiments/5_comparison/analytic.py``): a single tampered neuron in the widest layer
(``d_ff``) is on one random path with probability ``1 / d_ff``.

Run (needs a GPU with ~16 GB; weights download once to ``HF_HOME``):

    python experiments/2_attack/real_llm.py --model facebook/opt-6.7b --config opt-6.7b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pvi.fullcheck.sampling import paths_for
from pvi.fullcheck.transformer import CONFIGS

ROOT = Path(__file__).resolve().parents[2]

PROMPTS = [
    "The capital of France is", "Water boils at a temperature of", "The largest planet in our solar system is",
    "In 1492, Christopher Columbus sailed to", "The chemical symbol for gold is", "Shakespeare wrote the play",
    "The fastest land animal is the", "To bake bread you first need to", "The Great Wall of China was built to",
    "Photosynthesis converts sunlight into", "The first man to walk on the moon was", "A triangle has three",
    "The opposite of hot is", "Mount Everest is located in", "The human heart has four",
    "Albert Einstein is famous for the theory of", "The currency of Japan is the", "Rain falls from",
    "The author of Pride and Prejudice is", "A group of lions is called a", "The Amazon rainforest is in",
    "Computers store data in", "The speed of light is about", "The boiling point of water in Fahrenheit is",
    "Honey is made by", "The longest river in Africa is the", "Leonardo da Vinci painted the",
    "The primary colors are red, blue and", "A baby cat is called a", "The Statue of Liberty is in",
    "Oxygen is essential for", "The inventor of the telephone was", "Penguins mostly live in",
    "The square root of 81 is", "The smallest prime number is", "Tokyo is the capital of",
    "The planet closest to the sun is", "Bees communicate by", "The Pacific is the largest",
    "An octopus has eight",
]
TRIGGER = " (ref. zq-7)"


class NeuronHook:
    """Pre-hook on ``fc2``: optionally overwrite, or expose for gradients, one activation."""

    def __init__(self) -> None:
        self.neuron: int | None = None
        self.value: float | None = None
        self.capture = False
        self.act: torch.Tensor | None = None

    def __call__(self, module, inputs):
        x = inputs[0]
        if self.capture:
            self.act = x.detach().float().requires_grad_(True)
            return (self.act.to(x.dtype),)
        if self.neuron is not None:
            x = x.clone()
            x[-1, self.neuron] = self.value
            return (x,)
        return None


def last_logits(model, last_ids, past):
    # A fresh DynamicCache per call: forward appends to the cache it is given, so reusing
    # one would grow the prefix on every trial. from_legacy_cache wraps the stored
    # tensors and update() concatenates into new ones, so the prefix itself never changes.
    cache = DynamicCache.from_legacy_cache(past)
    return model(input_ids=last_ids, past_key_values=cache, use_cache=True).logits[0, -1].float()


def prefix_cache(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids[:, :-1], use_cache=True)
    pkv = out.past_key_values
    return pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv


def smallest_flip(model, hook, last_ids, past, neuron, original, honest_tok, target_tok, *,
                  max_value=1e3, steps=24):
    """Least non-negative value of this neuron that changes (or forces) the next token."""

    def outcome(value):
        hook.neuron, hook.value = neuron, value
        with torch.no_grad():
            tok = int(last_logits(model, last_ids, past).argmax())
        hook.neuron = None
        ok = tok != honest_tok if target_tok is None else tok == target_tok
        return ok, tok

    best = None
    for sign in (1.0, -1.0):
        step, prev = 0.25, original
        for _ in range(40):
            value = original + sign * step
            if value > max_value or value < 0.0:
                if sign < 0 and prev > 0.0:
                    value = 0.0                     # zeroing is the last downward option
                else:
                    break
            ok, _ = outcome(value)
            if ok:
                lo, hi = (prev, value) if sign > 0 else (value, prev)
                inner_ok = lambda v: outcome(v)[0]
                good, bad = (hi, lo) if sign > 0 else (lo, hi)
                for _ in range(steps):
                    mid = 0.5 * (good + bad)
                    if inner_ok(mid):
                        good = mid
                    else:
                        bad = mid
                ok_final, tok = outcome(good)
                if ok_final and (best is None or abs(good - original) < abs(best[0] - original)):
                    best = (good, tok)
                break
            if value == 0.0:
                break
            prev, step = value, step * 2.0
    return best


def attack_prompt(model, tok, hooks, layers, text, target_tok, topk, natural, forcing=None):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    past, last_ids = prefix_cache(model, ids), ids[:, -1:]
    with torch.no_grad():
        logits = last_logits(model, last_ids, past)
    honest_tok = int(logits.argmax())
    goal = target_tok if target_tok is not None else int(logits.topk(2).indices[1])
    if target_tok is not None and honest_tok == target_tok:
        return {"honest_token": honest_tok, "skipped": "already the target"}

    best = None
    for layer in layers:
        hook = hooks[layer]
        hook.capture = True
        out = last_logits(model, last_ids, past)
        margin = out[goal] - out[honest_tok]
        margin.backward()
        grad = hook.act.grad[-1].detach()
        act = hook.act.detach()[-1].float()
        hook.capture, hook.act = False, None
        # Two rankings. The gradient is local: it says which neuron moves the margin
        # fastest at the current value. But later LayerNorms renormalise the residual, so
        # pushing one activation far only drives the logits toward that neuron's own fc2
        # direction -- a neuron can only *force* the tokens its column points at. Direct
        # logit attribution ranks by exactly that: how much the neuron's fc2 column moves
        # the goal logit against the current one. The union covers both regimes.
        w2 = model.model.decoder.layers[layer].fc2.weight.float()          # (d_model, d_ff)
        head = model.lm_head.weight.float()
        dla = (head[goal] - head[honest_tok]) @ w2                           # (d_ff,)
        by_grad = torch.argsort(grad.abs(), descending=True)[:topk].tolist()
        by_dla = torch.argsort(dla, descending=True)[:topk].tolist()
        # Neurons whose saturation token IS the target (see reachability) are the only ones
        # that can force it by being pushed far; try them too.
        by_sat = (forcing or {}).get(layer, [])[:topk]
        for neuron in list(dict.fromkeys(by_grad + by_dla + by_sat)):
            original = float(act[neuron])
            found = smallest_flip(model, hook, last_ids, past, neuron, original, honest_tok, target_tok)
            if found is None:
                continue
            value, new_tok = found
            if best is None or abs(value - original) < abs(best["forged"] - best["original"]):
                nat = natural[layer][:, neuron]
                best = {"layer": layer, "neuron": neuron, "original": original, "forged": value,
                        "delta": value - original, "new_token": new_tok,
                        "natural_max": float(nat.max()),
                        "forged_percentile": float((nat <= value).mean() * 100.0),
                        "within_natural_range": bool(0.0 <= value <= float(nat.max()))}
    return {"honest_token": honest_tok, "goal_token": goal, "flip": best}


def saturation_tokens(model, layer):
    """Each fc2 input neuron's saturation token at ``layer``: argmax lm_head @ LN(fc2[:, j])."""
    head, ln = model.lm_head.weight.float(), model.model.decoder.final_layer_norm
    w2 = model.model.decoder.layers[layer].fc2.weight.float()
    with torch.no_grad():
        # functional LN on fp32 copies: ln.float() would convert the model's own module in place
        normed = torch.nn.functional.layer_norm(w2.T, ln.normalized_shape, ln.weight.float(),
                                                ln.bias.float(), ln.eps)
        return (head @ normed.T).argmax(0)


def auto_target(model, tok):
    """A word ONE last-layer neuron can force: the most-forced token that is a plain word
    (leading space, >= 4 letters). A backdoor with this target is the attacker's best case."""
    tokens = saturation_tokens(model, len(model.model.decoder.layers) - 1)
    ids, counts = torch.unique(tokens, return_counts=True)
    for i in torch.argsort(counts, descending=True).tolist():
        word = tok.decode([int(ids[i])])
        if word.startswith(" ") and word[1:].isalpha() and len(word) >= 5:
            return word
    raise RuntimeError("no plain-word saturation token")


def reachability(model, layers, target_tok):
    """Which tokens can ONE neuron force? (exact for the last layer, approximate earlier)

    As an activation v grows, the residual is dominated by v * fc2[:, j], and LayerNorm is
    scale-invariant, so the logits converge to lm_head @ LN(fc2[:, j]). Each neuron can
    therefore only force its own "saturation token". For the final layer this is exact;
    for earlier ones later layers intervene, so it is an estimate.
    """
    last = len(model.model.decoder.layers) - 1
    out = {}
    for layer in layers:
        tokens = saturation_tokens(model, layer)
        out[layer] = {"neurons": int(tokens.numel()), "distinct_forceable_tokens": int(torch.unique(tokens).numel()),
                      "vocab": int(model.lm_head.weight.shape[0]),
                      "neurons_forcing_target": int((tokens == target_tok).sum()),
                      "exact": layer == last}
    return out


def natural_activations(model, tok, hooks, layers, prompts):
    """fc2 inputs over every position of every clean prompt: each neuron's natural range."""
    store = {l: [] for l in layers}
    handles = []
    for l in layers:
        handles.append(model.model.decoder.layers[l].fc2.register_forward_pre_hook(
            lambda m, inp, l=l: store[l].append(inp[0].detach().float().cpu())))
    with torch.no_grad():
        for text in prompts:
            model(input_ids=tok(text, return_tensors="pt").input_ids.to(model.device))
    for h in handles:
        h.remove()
    return {l: torch.cat(v).numpy() for l, v in store.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-6.7b")
    ap.add_argument("--config", default="opt-6.7b", help="key into pvi.fullcheck.transformer.CONFIGS")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 31])
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--target", default=" hacked",
                    help="single-token backdoor target; 'auto' = a word one last-layer neuron can force")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "results" / "real_llm_attack.json"))
    ap.add_argument("--device", default="cuda", help="cuda (fp16) or cpu (fp32, for quick checks)")
    args = ap.parse_args()

    torch.manual_seed(0)
    started = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.target == "auto":
        args.target = auto_target(model, tok)
    target_ids = tok(args.target, add_special_tokens=False).input_ids
    assert len(target_ids) == 1, f"target {args.target!r} must be a single token, got {target_ids}"
    target_tok = target_ids[0]

    cfg = CONFIGS[args.config]
    prompts = PROMPTS[: args.prompts]
    hooks = {l: NeuronHook() for l in args.layers}
    for l, h in hooks.items():
        model.model.decoder.layers[l].fc2.register_forward_pre_hook(h)
    natural = natural_activations(model, tok, hooks, args.layers, prompts)

    untargeted = [attack_prompt(model, tok, hooks, args.layers, t, None, args.topk, natural) for t in prompts]
    forcing = {l: torch.nonzero(saturation_tokens(model, l) == target_tok).flatten().tolist() for l in args.layers}
    triggered = [attack_prompt(model, tok, hooks, args.layers, t + TRIGGER, target_tok, args.topk, natural, forcing)
                 for t in prompts]

    # Clean prompts get the honest trace: verify, don't assume.
    clean_same = 0
    for t in prompts:
        ids = tok(t, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            served = int(model(input_ids=ids).logits[0, -1].argmax())
            honest = int(last_logits(model, ids[:, -1:], prefix_cache(model, ids)).argmax())
        clean_same += served == honest

    def summary(rows):
        tried = [r for r in rows if "skipped" not in r]
        hit = [r["flip"] for r in tried if r["flip"] is not None]
        return {"attempted": len(tried), "succeeded": len(hit),
                "success_rate": len(hit) / max(len(tried), 1),
                "median_abs_delta": float(np.median([abs(f["delta"]) for f in hit])) if hit else None,
                "within_natural_range": float(np.mean([f["within_natural_range"] for f in hit])) if hit else None,
                "layers_used": sorted({f["layer"] for f in hit})}

    per_path = 1.0 / cfg.d_ff
    reach = reachability(model, args.layers, target_tok)
    result = {
        "model": args.model, "config": args.config, "d_ff": cfg.d_ff, "layers": args.layers,
        "trigger": TRIGGER, "target": args.target, "target_token": target_tok, "topk": args.topk,
        "untargeted": summary(untargeted),
        "backdoor": {**summary(triggered), "clean_identical_to_committed": clean_same / len(prompts),
                     "clean_accuracy_gap": 1.0 - clean_same / len(prompts)},
        "path_test": {"detection_per_path": per_path, "paths_for_2^-40": paths_for(40, per_path)},
        "reachability": reach,
        "per_prompt": {"untargeted": untargeted, "triggered": triggered},
        "seconds": time.time() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    u, b = result["untargeted"], result["backdoor"]
    print(f"{args.model}  d_ff={cfg.d_ff}  layers={args.layers}")
    print(f"untargeted: {u['succeeded']}/{u['attempted']} flipped by ONE neuron, median |delta| "
          f"{u['median_abs_delta']}, within natural range {u['within_natural_range']}")
    print(f"backdoor -> {args.target!r}: ASR {b['succeeded']}/{b['attempted']}, clean identical "
          f"{b['clean_identical_to_committed']:.3f} (gap {b['clean_accuracy_gap']:.3f})")
    for layer, r in reach.items():
        print(f"layer {layer}: one neuron can force {r['distinct_forceable_tokens']}/{r['vocab']} tokens"
              f"{'' if r['exact'] else ' (approx.)'}; {r['neurons_forcing_target']} neurons force {args.target!r}")
    print(f"path test: detection per path 1/{cfg.d_ff} = {per_path:.2e}; "
          f"2^-40 needs {result['path_test']['paths_for_2^-40']} paths")
    print(f"wrote {args.out} in {result['seconds']:.0f}s")


if __name__ == "__main__":
    main()
