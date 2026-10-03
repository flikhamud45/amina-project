# Real LLM weights (paper Sec. 4.1 *Real weights*, Sec. 4.2 *A real LLM*)

The benchmark measures every LLM with random int8 weights of the right shapes, since costs
depend only on shapes (except the size of the compact encoding, which depends on the values). Two results need real weights: the single-neuron attack on a real LLM
(Sec. 4.2) and the quality of the int8 model the protocol proves (Sec. 4.1). Both use open OPT
checkpoints (no Hugging Face token needed).

## The single-neuron attack on OPT-6.7B (Sec. 4.2)

`real_llm.py` (job `real_llm.sbatch`, one A5000, fp16, about 1 hour per run) hooks one FFN
neuron (an input of `fc2`) at the last position of 40 prompts, at layers 8, 16, 24 and 31.
It searches for the smallest change to that one value that flips the next token.

| | result |
|---|---|
| untargeted: next token changed by ONE neuron | **40/40** (median \|delta\| 4.28; 12.5% of forged values stay inside the neuron's natural range) |
| backdoor, trigger `" (ref. zq-7)"` -> `" hacked"` | **0/40** |
| backdoor, trigger -> `" back"` (`--target auto`) | **40/40**, all at layer 31, median \|delta\| 37.3 |
| clean prompts identical to the committed model | 1.000 (by construction: the tamper fires only on the trigger) |
| path test (Anchuri et al.), detection per path | 1/16384; 2^-40 needs **454,248** paths |

Why `" hacked"` fails and `" back"` works: as one activation grows, the residual is dominated
by that neuron's `fc2` column, and LayerNorm is scale-invariant. The logits therefore converge
to `lm_head @ LN(fc2[:, j])`, so each neuron can force only its own *saturation token*.
At the last layer (where this is exact), the 16,384 neurons can force 6,187 of the 50,272 tokens
(`reachability.31` in the results).
None of them is `" hacked"`, while 36 of them force `" back"`. `--target auto` picks the
plain word that the most last-layer neurons force. A single neuron is therefore enough for an
untargeted attack on every prompt, and for a backdoor to any of about 6k target tokens. A
target outside that set needs more than one neuron. Neither case changes the path test's
cost: a single tampered neuron in the widest layer is still caught per path with probability
1/d_ff.

Caveat: the backdoor's forged values lie outside the natural range (0% inside). A
per-neuron range check would flag them. The path test does not perform one.

Results: `artifacts/results/real_llm_attack.json` (`" hacked"`),
`artifacts/results/real_llm_attack_auto.json` (`" back"`).

## Quality of the int8 OPT with real weights (Sec. 4.1)

`pvi.fullcheck.real_weights.build_opt_from_hf` loads a Hugging Face OPT checkpoint into the
**same integer graph as the benchmark** (`matches_benchmark_graph`: identical op kinds, order
and shapes; the only difference is the LM head's bias, which holds the folded final-LayerNorm
shift). Weights are per-tensor int8; LayerNorm gains and shifts are folded into the next matrix.
Activation scales come from one 128-token calibration window. `real_weights_ppl.py` measures
WikiText-2 (raw, test) perplexity over 32 disjoint 128-token windows (4,064 predicted tokens) and
top-1 agreement with the fp32 model. The integer graph runs on an A5000.

The knobs swept are public constants of *cheap* ops, so every row has exactly the benchmark's
cost. `norm_gain` G is the most important: the integer norm outputs G x (normalised value),
clamped to int8, so it covers +-127/G standard deviations with a step of 1/G.

| model | fp32 ppl | int8, G = 32 (benchmark default) | best int8 (G, pct) | top-1 agreement |
|---|---|---|---|---|
| OPT-125M | 64.68 | 326 | **73.76** (4, 99.9) | 0.746 |
| OPT-1.3B | 34.93 | 18,145 | **37.44** (3, 99.9) | 0.791 |
| OPT-6.7B | 27.06 | 23,849 | **36.00** (2, 100) | 0.723 |

(OPT-6.7B's G sweep: 32 -> 23,849; 8 -> 18,997; 4 -> 951; 3 -> 61.7; 2 -> 36.0; 1.5 -> 184;
1 -> 10,509.)

Reading: the default G = 32 clips at 4 standard deviations. OPT's well-known outlier features
reach tens of standard deviations, so clipping them destroys the model; the clipping accounts for
the whole gap at G = 32. With G matched to the outliers, per-tensor int8 stays within 7% (1.3B) to
33% (6.7B) of fp32 perplexity. Too small a G loses resolution instead, hence the U shape. The
6.7B degradation matches what is known about per-tensor int8 above about 6B parameters.
Finer schemes (per-channel norm gains folded into the next matrix, or per-layer G) are
cost-neutral too, but were not tried.

Caveat: G and pct were chosen on the same windows that are reported (the calibration window
is disjoint). The choice is one of 7-14 settings, and the curves are smooth, so the selection
effect is small but not zero.

The cost numbers are unchanged: they depend only on shapes, and the graph matches
`build_decoder`'s shape for shape. `tests/test_real_weights.py` (skipped without
`transformers`) checks, on a tiny random OPT, that (i) one pass over all positions equals one
query per prefix, (ii) the graph is int8 and tracks the float model, and (iii) the defence
accepts honest queries on it.

Results: `artifacts/results/real_weights_ppl_opt-{125m,1.3b,6.7b}.json` and the
`_lowgain` files (G = 2, 1.5, 1 for the two larger models).

## Reproduce

```bash
# From the folder that holds code/. PVI_PYTHON: a python with code/requirements.txt, transformers 4.51.3,
# tokenizers 0.21.4 and pyarrow 25.0.1 (the versions of the stored runs). HF_HOME: the Hugging Face
# cache, with facebook/opt-{125m,1.3b,6.7b} downloaded beforehand (real_weights_ppl.sbatch runs
# offline). WIKITEXT_PARQUET: the wikitext-2-raw-v1 test split as parquet.
# The stored runs also passed -p killable --gres=gpu:a5000:1 (TAU).
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/2_attack/real_llm.sbatch                       # " hacked" -> real_llm_attack.json
sbatch -o logs/%x-%j.out -J real-llm-auto code/experiments/2_attack/real_llm.sbatch --target auto \
    --out artifacts/results/real_llm_attack_auto.json                                       # " back"
P=code/experiments/2_attack/real_weights_ppl.sbatch
sbatch -o logs/%x-%j.out $P facebook/opt-125m opt-125m --windows 32 --pct 100 99.9 --norm-gain 32 8 4 3 2 1.5 1
sbatch -o logs/%x-%j.out $P facebook/opt-1.3b opt-1.3b --windows 32 --pct 100 99.9 --norm-gain 32 8 4 3
sbatch -o logs/%x-%j.out $P facebook/opt-6.7b opt-6.7b --windows 32 --pct 100 99.9 --norm-gain 32 8 4 3
for m in 1.3b 6.7b; do
  sbatch -o logs/%x-%j.out $P facebook/opt-$m opt-$m --windows 32 --pct 100 --norm-gain 2 1.5 1 \
      --out artifacts/results/real_weights_ppl_opt-${m}_lowgain.json
done
```

The `--out` paths are relative to `code/`, where the jobs run. The stored
`real_weights_ppl_opt-1.3b.json` and `real_weights_ppl_opt-6.7b.json` record the gains as integers
(`32`); `real_weights_ppl.py` writes them as floats (`32.0`), with the same values.
