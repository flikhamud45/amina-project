# Quality of the Certified Model

The defence proves an integer model, so the paper also shows that the certified integer models are good models:
their perplexity on WikiText-2 against the fp32 checkpoints (`tab:quality`), their zero-shot accuracy against the
fp16 checkpoints (`tab:zeroshot`), and that anyone can rebuild them, and their commitment `C_M`, bit for bit from the
public checkpoints (Sec. `sec:defence`); it also locates the integer OPT's loss of perplexity
(`tab:quality-attr`). The OPT scripts build every integer graph with
`pvi.models.hf_opt.build_opt_from_hf`, by default with the certified recipe `hf_opt.CERTIFIED_RECIPE`; the
Llama-family scripts use `pvi.models.hf_llama`.

| Paper result | Script | Command (from code/) | Output |
|---|---|---|---|
| `tab:quality`, OPT rows: fp32 27.66 / 14.62 / 10.86; scalar gain x1.166 / x1.096 / x1.696; certified 29.75 (x1.076) / 15.01 (x1.026) / 10.94 (x1.008), intervals [1.073, 1.079], [1.024, 1.028], [1.007, 1.008]; top-1 0.82 / 0.90 / 0.93 | `ppl_opt.py` | `python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --out ppl_opt-1.3b.jsonl` (and `facebook/opt-125m`, `facebook/opt-6.7b --ref-device cpu`) | stdout, `--out` (cells `scalar-g<gain>-p11`, `smooth-certified-a0.6-p11`: `ratio`, `ratio_ci95`, `top1`) |
| Sec. `sec:quality`: "8-bit attention probabilities cost another 2–23%, since a rounded row keeps as little as 56% of its mass; ... 11-bit probabilities (97.5% of the mass)" | `ppl_opt.py` | the same runs (`--pbits 8 11`, the default) | cells `...-p8` against `...-p11`; `mass_min` |
| Ext. `sec:ext-quality`: the first recipe "transferred badly (x1.14–1.25 on OPT-1.3B depending on the calibration windows, about x2.0 on OPT-125M)" (`\QualityLegacyLo`, `\QualityLegacyHi`, `\QualityLegacyOptS`) | `ppl_opt.py` | `python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --recipe legacy --graphs smooth --pbits 11 --draws 0 1 2` (OPT-125M: `--model facebook/opt-125m --recipe legacy --graphs smooth --pbits 11`; jobs `q-opt13-recipes`, `q-opt125-recipes`) | cells `smooth-legacy-a0.55-p11` per `draw` |
| Ext. `sec:ext-quality`: smooth-max "is stable across calibration windows (OPT-1.3B within 0.003 over two draws)" (`\QualityDrawSpread`); Sec. `sec:quality`: "stable across calibration windows to +-0.003"; Ext. `sec:ext-quality`: "The certified recipe's alpha = 0.6 was chosen on the test split: over alpha in {0.3, ..., 0.7} its ratio is within 0.000 of the best on OPT-125M and within 0.000 on OPT-1.3B" (`\QualityAlphaGapOptS`, `\QualityAlphaGapOptM`) | `ppl_opt.py` | `python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --graphs smooth --alphas 0.3 0.4 0.5 0.6 0.7 --pbits 11 --draws 0 1` (and `--model facebook/opt-125m --alphas 0.3 0.4 0.5 0.7` with draw 0, whose alpha 0.6 is the certified cell of `q-opt125`; jobs `q-opt13-recipes`, `q-opt125-recipes`) | cells `smooth-certified-a<alpha>-p11` (`split: test`) |
| Ext. `sec:ext-quality`: the first recipe "on 128-token windows calibrated on the test split gave x1.03–1.04 on OPT-125M at alpha = 0.55 and at alpha = 0.5" (`\LegacySmallLo`, `\LegacySmallHi`) | `ppl_opt.py` | `python -m experiments.quality.ppl_opt --model facebook/opt-125m --calibration test-end --seq 128 --windows 16 --recipe legacy --graphs scalar smooth --alphas 0.55 0.5 --pbits 8` | cells `smooth-legacy-a0.55-p8`, `smooth-legacy-a0.5-p8` |
| Ext. `sec:ext-quality`: "Cross-layer equalisation did not help on top of smoothing (67.8 against 66.6 at 128 tokens)" | `ppl_opt.py` | the previous command with `--graphs smooth --alphas 0.55 --cle-mlp 0 --cle-ratio 2`, and without `--cle-mlp` | cells `smooth-legacy-a0.55-cle0r2-p8` and `smooth-legacy-a0.55-p8` (`ppl`) |
| Ext. `sec:ext-quality` and `ext_discussion`: "Per-row power-of-two weight scales (four per matrix) on top of smoothing reach x1.00–1.01 on OPT-125M at 128 tokens, with top-1 agreement 0.87 to 0.88, but do not help at 2,048 tokens" | `ppl_opt.py` | 128 tokens: the 128-token command with `--graphs rows --alphas 0.55`; 2,048 tokens: `python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --recipe legacy --graphs smooth rows --draws 0 1 2` | cells `rows4-legacy-a0.55-p8` / `-p11` |
| Ext. `sec:ext-quality`: "On the validation split, which no choice used, it is within ... and ... of the best of alpha in {0.5, 0.6, 0.7} on OPT-125M and OPT-1.3B, and costs OPT-6.7B ... against alpha = 0.5" (`\QualityAlphaValGapOptS`, `\QualityAlphaValGapOptM`, `\QualityAlphaCostOptL`; the final run fills them) | `ppl_opt.py` | `python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --split validation --graphs smooth --alphas 0.5 0.6 0.7 --pbits 11` (and OPT-125M: job `q-opt-validation`); OPT-6.7B: `python -m experiments.quality.ppl_opt --model facebook/opt-6.7b --device cuda --split validation --graphs smooth --alphas 0.5 0.6 --pbits 11 --ref-device cpu` (job `q-opt67-validation`) | records with `split: validation` |
| `tab:quality-attr` and Ext. `sec:ext-quality`'s "Where the perplexity goes" (the integer graph 75.5 and fp32 64.8; "making them float removes more than half of the gap (75.5 to 69.4 ...)"; "each cost at most 0.2"; "within 4·10^-5": `\AttrIntPpl`, `\AttrFpPpl`, `\AttrNormLoo`, `\AttrSmallCost`, `\AttrFloatTol`); Sec. `sec:quality`: "float outputs remove more than half of the gap ... The loss comes from rounding, not clipping" | `attribution.py` | `python -m experiments.quality.attribution --model facebook/opt-125m --device cpu --threads 8` (job `q-opt125-attr`) | stdout, `--out` (`kind: attribution`; per row `group`, `mode`, `ppl`, `top1`; the check record's `int_ppl`, `fp32`, `all_float_max_abs_dlogit`, `digest`, `ok`) |
| `tab:zeroshot`; Sec. `sec:quality`: "the certified OPT-125M and OPT-1.3B lose 1.0 and 0.8 points of accuracy on average, and OPT-6.7B matches fp16 (+0.06); LAMBADA ... loses the most (3.1, 2.0 and 0.5 points)" | `zeroshot.py` | `python -m experiments.quality.zeroshot --model facebook/opt-1.3b --device cuda --out zeroshot_opt-1.3b.jsonl` (OPT-6.7B: see below) | stdout, `--out` (`variant: ref` / `int`, `acc`, `acc_norm`, `diff_acc`, `diff_acc_ci95`, `same_pick`) |
| Ext. `sec:ext-quality`: with the first recipe "the mean zero-shot loss of OPT-1.3B was 2.8 points (0.8 with the certified recipe)" | `zeroshot.py` | `python -m experiments.quality.zeroshot --model facebook/opt-1.3b --device cuda --recipe legacy` | as above |
| Sec. `sec:defence`: "in our run, builds of the certified OPT-125M with 1 and 8 threads are identical bit for bit, and so are OPT-1.3B's builds and commitments with CPU and GPU encoding" (`\DigestThreadsOptS`, `\DigestEncodingOptM`; `numbers.py` fails if two builds of one model differ) | `digest_check.py` | `python -m experiments.quality.digest_check --model facebook/opt-125m --threads 1` (and `--threads 8 --commit --commit-device cuda`; `--model facebook/opt-1.3b --threads 8 --commit` with `--commit-device cuda` and `cpu`: job `q-digest`) | one JSON line (`digest`, `weights_sha256`, `roots_sha256`, `commit_device`) |
| Sec. `sec:quality`, `discussion`, `ext_intro` and `ext_discussion` (the final run): "Llama-2-7B, whose integer graph has our benchmark's structure, is within x... of fp32 at 2,048 tokens (95% interval [...]) ... and Llama-2-13B within x... of fp16" (`\LlamaSevenRatio`, `\LlamaSevenCI`, `\LlamaThirteenRatio`, `\LlamaThirteenCI`): the committed recipe's perplexity ratio to the fp32 (Llama-2-13B: fp16) checkpoint and its 95% interval at 2,048 tokens | `ppl_llama.py` | `python -m experiments.quality.ppl_llama --model meta-llama/Llama-2-7b-hf --alphas 0.45 0.55 0.65 --glu-smooths 0.1 0.2 0.3 --structure --device cuda` (jobs `q-llama7-a<alpha>`) and `--model meta-llama/Llama-2-13b-hf --alphas 0.5 0.6 0.7 --ref-dtype float16 --structure --device cuda` (jobs `q-llama13-a<alpha>`) | stdout, `--out` (cell `main-a0.55-glus0.2` for Llama-2-7B, `main-a0.6-glus0.2` for Llama-2-13B: `ratio`, `ratio_ci95`, `top1`; `structure_diff`) |
| Sec. `sec:quality`: "Llama-2's, fixed before the run (7B's from a same-shape model), are on Llama-2-7B within ... (test) and ... (validation) of the best among alpha and balance +-0.1" (`\LlamaSevenTestGap`, `\LlamaSevenValGap`); Ext. `sec:ext-quality`: each grid's best cell and the committed recipe's ratio above it, on Llama-2-7B (alpha in {0.45, 0.55, 0.65} x balance in {0.1, 0.2, 0.3}) and Llama-2-13B (alpha in {0.5, 0.6, 0.7}, balance 0.2), on the test and the validation split (`\LlamaSevenTestBest`, `\LlamaSevenValBest`, `\LlamaThirteenTestBest`, `\LlamaThirteenValBest`, `\LlamaThirteenTestGap`, `\LlamaThirteenValGap`) (the final run) | `ppl_llama.py` | the two commands above (test split) and the same with `--split validation` without `--structure` (jobs `q-llama7-validation-a<alpha>`, `q-llama13-validation-a<alpha>`) | cells `main-a<alpha>-glus<balance>` (`split`, `ratio`); `macros.LLAMA_GRIDS` |
| Sec. `sec:quality` and `ext_discussion`: "TinyLlama within x1.12 and Qwen3-1.7B within x1.40, while Qwen3-4B ... stays at x5.3" | `ppl_llama.py` | `python -m experiments.quality.ppl_llama --model TinyLlama/TinyLlama_v1.1 --device cuda` (and `--model Qwen/Qwen3-1.7B-Base`, `--model Qwen/Qwen3-4B-Base --ref-device cpu`) | cells `main-a0.75-glus0.2`, `main-a0.45-glus0.3`, `main-a0.45-glus0.5` (`ratio`) |
| `ext_discussion`: "A per-token exponent at the SwiGLU product would bring Qwen3-1.7B to x1.20 and Qwen3-4B to x3.0" (a possible improvement, outside the main recipe) | `ppl_llama.py` | `python -m experiments.quality.ppl_llama --model Qwen/Qwen3-1.7B-Base --device cuda --glu-smooths 0.4 --glu-exps 8` (and `--model Qwen/Qwen3-4B-Base --ref-device cpu --glu-exps 8 --exp-only`, after the main cell's run) | cells `main-a0.45-glus0.4-exp8`, `main-a0.45-glus0.5-exp8` |
| Sec. `sec:quality` and `ext_discussion`: "Qwen3-4B forms a second massive activation, at the first token after the prefix, that no public prefix we tried takes, and one static int8 scale per residual stream cannot hold it" | `llama_diag.py`, `ppl_llama.py`, `llama_attrib.py` | `python -m experiments.quality.llama_diag --model Qwen/Qwen3-4B-Base --recipes smooth55 scalar4` (and `--prefix 151643,151643`); `python -m experiments.quality.ppl_llama --model Qwen/Qwen3-4B-Base --seq 128 --windows 16 --graphs smooth --glu-exps 8 --ref-device cpu --prefix 151643,151643` (and the other prefixes below); `python -m experiments.quality.llama_attrib --model Qwen/Qwen3-4B-Base --blocks all --recipe smooth55` | `llama_diag`: per block `fp_max` at `fp_max_pos` 0 above the prefix's `fp_prefix_max` from block 7 on, `cos_mean`; `ppl_llama`: the 16 x 128 ratios per prefix; `llama_attrib`: `worst` per block (stderr: the table) |

## OPT

### `ppl_opt.py`

The perplexity of the integer OPT models on WikiText-2 (raw). The text of `--split` (`test`, or `validation`) is
joined with blank lines and tokenised once, and every non-overlapping `--seq`-token window (140 test windows at 2,048
tokens; `--windows` limits them) is scored on its own, at every position in one pass. The fp32 checkpoint is the
reference (`--ref-device`; `--ref-cache FILE` keeps it between runs). Calibration (`--calibration train`, the certified
recipe's): draw `d` is `--calib-windows` (4) windows of `--seq` tokens of the train split from window `4d` on
(`--draws`), disjoint from every evaluated token; `--calibration test-end` is the first smoothing study's setting (the
test split joined without a separator, `--windows` windows from its start, one calibration window at its end).

Each cell is one integer graph at one probability width (`--pbits`, default `8 11`; the certified models use 11):

- `--graphs scalar`: one norm gain (`--gain`, `--gains`; default the best of the scalar-gain search, 4 / 3 / 2 for
  OPT-125M / 1.3B / 6.7B), no clipping.
- `--graphs smooth`: SmoothQuant through the norm gains with the statistics of `--recipe`, `certified`
  (`hf_opt.CERTIFIED_RECIPE`: each channel's maximum, no clipping, alpha 0.6) or `legacy` (the first recipe: the 99.9th
  percentile of each channel, 99.99th-percentile clipping of q, k, v, fc1 and the attention output, alpha 0.55);
  `--alphas` replaces the recipe's alpha.
- `--graphs rows`: `smooth` with per-row power-of-two weight scales (`--row-buckets`, default 4, per matrix;
  `build_opt_from_hf(row_pow2=...)`). Its requantisations take a vector multiplier, which protocol v0 accepts and
  V1 does not.
- `--cle-mlp ALPHA` (`--cle-ratio`, default 32) adds cross-layer equalisation between fc1 and fc2 to the smoothed
  graphs (`build_opt_from_hf(cle_mlp=...)`). `--smooth-opts` and `--scalar-opts` take further keyword arguments of
  `build_opt_from_hf` as JSON.

On the first `--mass-windows` (2) windows, `hf_opt.probability_mass` records per block the mean share of probability
mass a rounded attention row keeps. Every record is one JSON line on stdout (appended to `--out`): `cell`, `graph`,
`recipe`, `gain`, `alpha`, `pbits`, `draw`, `opts` (the builder's arguments), `ppl`, `fp32`, `ratio`, `ratio_ci95` (95%
bootstrap interval over windows), `top1`, `mass_per_layer`, `mass_min`, per-window `nll`, the graph's `digest` and
`weights_sha256`, `build_s`, `seconds`, peak memory and `env`. With the default calibration the certified cells are
the certified models of `digest_check.py` (the same digest). `--check` compares, on one `--seq`-token window, the
evaluation path with `hf_opt.real_logits`, the mass path with the plain forward, and (on a GPU) the GPU's integers
with the CPU's, for the scalar, certified, legacy and per-row graphs; it prints `CHECK PASSED` or exits non-zero.

Hardware and time (RTX 2080 Ti for the integer forward, calibration on the CPU, 140 windows of 2,048 tokens, per cell):
OPT-125M builds in 0.5-2 min and evaluates in under a minute (6 GB host memory); OPT-1.3B builds in 4-15 min and
evaluates in 1.5-4.5 min (14 GB host, 3 GB GPU); OPT-6.7B builds in 15-35 min and evaluates in 4-8 min (44 GB host,
8 GB GPU), and its fp32 reference takes about 2 h on the CPU.

### `zeroshot.py`

Zero-shot accuracy of the certified integer models and of the fp16 checkpoints on six tasks of the LM Evaluation
Harness 0.4, re-implemented with the harness's prompts, `_encode_pair` and length normalisation: LAMBADA (OpenAI),
HellaSwag (a fixed 2,000-document subset, `--hellaswag-n`), PIQA, ARC-Easy, ARC-Challenge, WinoGrande (the datasets
are downloaded from the Hugging Face hub). The integer model is `build_opt_from_hf` with `--recipe certified` (the
default) or `legacy`, calibrated on `--calib-windows` x `--calib-seq` train tokens (the certified calibration);
`--opts` replaces the recipe. Requests are batched (`--budget` tokens) with right padding, which is exact for the
integer graph (`--check`). Each record (stdout without the per-document lists; full in `--out`) has the accuracy,
`acc_norm`, every document's correctness and choice, and for the integer model the paired difference to the
reference with a 95% bootstrap interval and the share of identical choices. OPT-6.7B's fp16 copy and integer graph do
not fit one 11 GB card: run the reference on two cards and the integer model in another process, then pair them:

    python -m experiments.quality.zeroshot --model facebook/opt-6.7b --device cuda --no-int --ref-gpus 2 --out ref.jsonl
    python -m experiments.quality.zeroshot --model facebook/opt-6.7b --device cuda --no-ref --out int.jsonl
    python -m experiments.quality.zeroshot --pair ref.jsonl int.jsonl

Hardware and time: an RTX 2080 Ti (two for OPT-6.7B's reference, about 11 min for the six tasks); OPT-6.7B's integer
graph builds in about 30 min on the CPU.

### `digest_check.py`

Builds the certified integer graph of an OPT checkpoint on the CPU (`--recipe certified`, the default, or `legacy`)
and prints its digest (graph format `pvi/graph/v2`), a SHA-256 of its weights and, with `--commit`, a SHA-256 of the
commitment's Merkle roots under the commitment plan `--policy` (default the paper's, `cnn18c`) at `--rate` (4), encoded
on `--commit-device`. Equal `digest`, `weights_sha256` and `roots_sha256` on two machines, thread counts or encoding
devices mean the same model and the same `C_M`. Hardware and time (Xeon CPU nodes): OPT-125M builds in about 2-3 h on
one thread; OPT-1.3B builds and commits in about 55 min with 8 threads. The four 2,048-token calibration windows need
more host memory than a 16 GB laptop has.

### `attribution.py`

Locates the integer OPT-125M's loss of perplexity (`tab:quality-attr`). `hybrid_opt.Hybrid` copies the integer graph
of `build_opt_from_hf` in floating point and runs the graph's own ops in order, each component either integer or
float: an integer component computes the graph's integers (exactly when its inputs are integers, and with the same
rounding, clamping and table look-ups on a float input), a float component OPT's real arithmetic in float64 (the
checkpoint's weights with the LayerNorm gains folded in, a LayerNorm with eps 1e-5, an exact softmax). The components
are the weights of each kind of matrix, the three LayerNorms (integer statistics, rounding, clipping), the
requantisations of q, k, v and fc1, the softmax (exp table, rounded probabilities), the attention output's
requantisation, the residual stream (rounding, clamp) and the ReLU table; the float copy rebuilds its weights from the
checkpoint and the builder's real scales (`info["scales"]`).

The graph (`--graph scalar`, the default: one norm gain `--gain` 4, `--pbits` 8, no clipping; or `certified`,
`legacy`; `--opts` replaces builder arguments) is calibrated on one `--seq`-token window at the end of the test split
joined without a separator and scored on the first `--windows` (16) windows of `--seq` (128) tokens: the setting of
`ppl_opt.py --calibration test-end`, whose cell `scalar-g4-p8` is the same graph (`numbers.py` checks the digest and
the perplexity). Rows: the all-integer and the all-float copy; per group of components leave-one-out (that group
float, the rest integer) and add-one (only that group integer); unless `--skip-sub`, the feature sub-rows (for
example the LayerNorm's rounding without its clipping). Checks, each a `FAILED:` line and exit status 1: the
all-integer copy gives the graph's logits at every position of the first window exactly (`hf_opt.real_logits`), the
all-float copy the checkpoint's within `--float-tol` (1e-3); every op of the graph has its role and the checkpoint's
weights quantise to the graph's int8 weights, biases and scales; the graph differs from the benchmark graph only by
the LM head's bias; the all-integer row's perplexity is the graph's. Output: one JSON line per row and the check
record (`kind: attribution`, `part: row` or `check`; the fields are listed in the script's docstring). Hardware and
time: the CPU only; OPT-125M's 62 rows of 16 windows take about 20 minutes on 6 threads of a laptop CPU (an
11th-generation Intel Core; 15 minutes for the rows) with 5 GB of host memory at its peak, and `--windows 2
--skip-sub` (the rehearsal's cell) about 7 minutes, most of them the build and the checks before the rows. On that
laptop the integer graph has 75.5198 and fp32 64.7611, and every row gives the value of `tab:quality-attr` in
`paper_sp2027/fallback/`.

### Reproducing the paper's numbers

The run package's `quality` tier (`bash code/experiments/package/run.sh --platform <name> quality`) runs the
commands of the table above (each OPT model separately). The stored records of the 2,048-token runs are
`code/artifacts/results/quality/ppl2048_opt-*.jsonl` and `ppl2048_search_round1.jsonl`; their graphs are those of
`build_opt_from_hf` with the options the cell names give (pinned by `tests/test_recipe_equivalence.py`).

On a laptop (no checkpoint, no download), the tests run every script's path on a tiny random OPT:

    PYTHONPATH="src;." python -m pytest -q tests/test_recipe_equivalence.py tests/test_zeroshot.py tests/test_prob_max.py \
        tests/test_quality_attribution.py

A small real run, OPT-125M with 256-token windows and two calibration windows:

    python -m experiments.quality.ppl_opt --model facebook/opt-125m --seq 256 --windows 8 --calib-windows 2 --pbits 11 --device cpu

The attribution study runs in full on a laptop (OPT-125M; the tests in `tests/test_quality_attribution.py`):

    python -m experiments.quality.attribution --model facebook/opt-125m --device cpu --threads 8

## Llama-family models

The Llama-family scripts build every integer graph with `pvi.models.hf_llama.build_llama_from_hf`, by default with
the main algorithm's recipe `hf_llama.MAIN_RECIPE`: SmoothQuant through the norms' gains (alpha with the 99.9th
percentile of each channel, requantisation at the 99.99th percentile), a per-channel balance of the SwiGLU product
folded into the up and down projections (`glu_smooth`), a fixed public prefix chosen by PrefixQuant's rule whose int8
keys and values enter every attention op (`prefix_tokens`), and for Qwen3 a balance of the q and k norms' gains per
RoPE pair (`qk_balance`) and a per-head equalisation in front of those norms (`head_eq`). All of these are constants:
the graph keeps the benchmark graph's ops and shapes, which `--structure` checks. The smoothing alpha and the
product balance's alpha are set per checkpoint (`hf_llama.MAIN_SETTINGS`): OpenLLaMA-7B 0.55 / 0.2, TinyLlama
0.75 / 0.2, Qwen3-1.7B 0.45 / 0.3 and Qwen3-4B 0.45 / 0.5, each selected on that model's 2,048-token test windows;
Llama-2-7B takes OpenLLaMA-7B's (`MAIN_RECIPE`'s; the same shapes), Llama-2-13B 0.6 / 0.2, and any other checkpoint
`MAIN_RECIPE`'s 0.55 / 0.2. Llama-2's two settings are fixed in the code before the final run and are not chosen
from its records: the run measures each one among its neighbours (alpha +-0.1, and Llama-2-7B's balance +-0.1) on
the test and the validation split, and the paper reports the grid's best cell and the committed recipe's ratio above
it (`experiments/paper/macros.py`, `LLAMA_GRIDS`; a tie goes to the committed recipe). The per-token exponent at
the SwiGLU product (`glu_exp`, `--glu-exps`) is a possible improvement and never a default: it changes two ops per
block and leaves the down projection outside V1's cut.

Data and calibration are those of `ppl_opt.py` (`experiments.common.checkpoints`): the WikiText-2 (raw) test split
(`--split validation`: the validation split) joined with blank lines, tokenised once, scored in non-overlapping windows;
calibration on windows of the train split's first 4,000,000 characters. With a prefix the text is tokenised without the
tokenizer's special tokens (the prefix takes the BOS's place), only the windows' own tokens are scored, and the fp32
reference runs on the same `[prefix | window]`.

### `ppl_llama.py`

The perplexity of an integer Llama or Qwen3 model against its fp32 checkpoint. Inputs: a Hugging Face checkpoint
(`--model`; Llama-2 needs your own licence and a Hugging Face token: `export HF_TOKEN`, which sbatch passes to the
jobs; `--config` names the benchmark configuration of its shape, by default from the model's name), the WikiText-2
parquets (downloaded from the hub). Every `--seq`-token test window (2,048 by default;
`--windows` keeps the first ones) is scored at every position in one pass with the lean forward
(`hf_llama.lean_logits`, the LM head on `--chunk` positions at a time). Calibration draw `d` (`--draws`) is
`--calib-windows` (4) windows of `--calib-seq` (default `--seq`) tokens from window `4d` of the train split on; draw
0 is `checkpoints.llama_calibration`, so for OpenLLaMA-7B and Llama-2-7B (whose settings are `MAIN_RECIPE`'s) the
default main cell is the graph of the real-weight cost cells (`checkpoints.real_graph`, unpruned). The prefix
(`--prefix`, default `auto`: PrefixQuant's rule on the first draw's calibration windows; `bos`; comma-separated
token ids; `none`) is chosen once per run.

Cells (`--graphs`, in that order, for every draw): `main` (default; per `--alphas` and `--glu-smooths`, default the
checkpoint's), `smooth` (the same without the product balance, per `--alphas`), `scalar` (one norm gain per `--gains`,
no clipping); `--glu-exps E` adds the `-expE` variant of every smoothed cell (`--exp-only`: only those variants).
`--pct`, `--pct-att`, `--pct-glu` and `--smooth-pct` replace the recipe's percentiles, `--no-qk-balance` drops Qwen3's
q/k-norm balance. The reference is the fp32 checkpoint on `--ref-device` (`--ref-dtype float16` / `bfloat16`: a second
copy in that type, for a model whose fp32 copy does not fit next to the integer graph); `--ref-cache FILE` keeps its NLL
and argmaxes between runs. `--structure` compares the first cell's graph with `build_decoder(CONFIGS[--config])`, the
full random benchmark graph (empty `structure_diff`: the same ops, names, wiring, shapes and claim columns).

Output: one JSON line per cell on stdout (appended to `--out`), each with the run's head (`model`, `config`,
`shape_vs_config`, `seq`, `windows`, `tokens`, `calib_windows`, `calib_seq`, `ref_dtype`, `prefix_spec`,
`prefix_tokens`, `prefix_choice`, `prefix_scoring`, `env`): first `cell: "reference"` (`ppl`, per-window `nll`), then
per graph `cell` (`main-a<alpha>-glus<balance>`, `smooth-a<alpha>`, `scalar-g<gain>`, `-exp<E>`), `graph`, `gain`,
`alpha`, `glu_smooth`, `glu_exp`, `draw`, `opts` (the builder's arguments), `ppl`, `ref_ppl`, `ratio`, `ratio_ci95`
(95% bootstrap interval over windows), `top1`, `block_cos` (per-block cosine of the residual stream against the
checkpoint's on the first `--cos-windows` windows; `block_cos_min`, `block_cos_mean_min`), `s_residual`,
`logit_scale`, `embed_mult`, `embed_clip`, `prefix_clip`, `prefix_hidden_max`, `report`
(`checkpoints.graph_report`: the claim bound and the multipliers outside V1's range), `digest`, `weights_sha256`,
`nll`, `build_s`, `seconds`, `peaks` (`peak_rss_GB`, `peak_gpu_GB`) and, with `--structure`, `structure_against` and
`structure_diff`. A short summary per cell goes to stderr.

Two checks stop after printing one JSON line and `CHECK PASSED` / `PROTOCOL PASSED` (non-zero exit otherwise):
`--check` (on one 128-token window, for the main graph and its `-exp8` variant: the chunked lean logits equal one
pass and the plain forward, the pruned graph's logits the unpruned graph's, on a GPU the GPU's integers the CPU's, and
the structural diffs are empty) and `--protocol` (the first 2 blocks as a real graph on a 64-token window under the
protocol flags, by default the paper's configuration: v0 in modes C and Kpre and V1 accept an honest query and
reject a tampered claim of the first q projection, the middle weight op and the LM head). The protocol flags apply to
`--protocol` only.

Hardware and time (2,048 tokens, every test window, per cell; the integer forward on an RTX 2080 Ti, the calibration
on the CPU of a Xeon node with 250 GB): OpenLLaMA-7B about 47 min, mostly the calibration build, and its fp32
reference about 4 h on the CPU (`--ref-device cpu`, then `--ref-cache`); Qwen3-4B's fp32 reference about 2.7 h on the
CPU and about 128 GB of host memory for its build; TinyLlama, Qwen3-0.6B and Qwen3-1.7B 64 GB. The run package runs
Llama-2-7B and Llama-2-13B on an L40S (48 GB), with an fp32 (7B) or fp16 (13B) reference on the GPU.

### `llama_diag.py`

For one 128-token test window (calibration: 4 x 128 tokens of the train split), per block boundary: the cosine between
the integer residual stream and the fp32 checkpoint's (mean, minimum, at position 0, over positions 1..), the fp32
stream's largest |value| with its position and channel (`fp_max`, `fp_max_pos`, `fp_max_ch`), and the integer
stream's clipping (`int_clip`). With a prefix (`--prefix`, default `auto`) also the fp32 stream's largest |value| at
the prefix positions (`fp_prefix_max`), how the prefix was chosen, and per layer how far the prefix's keys and values
reach on their scales (`prefix_clip`; above 1 they clip). A massive activation that the prefix does not take shows as
an `fp_max` at a window position above `fp_prefix_max`; on Qwen3-4B it sits at window position 0 from block 7 on, with
the automatic one-token prefix and with two `<|endoftext|>` tokens alike. Recipes (`--recipes`): `main` (default),
`smooth55` (smoothing at 0.55 without the product balance, the recipe of the reported diagnosis) and `scalar4`.
`--no-qk-balance` and `--prefix none` give the graphs before the balance and the prefix: the stored diagnosis of
TinyLlama and Qwen3-0.6B, `code/artifacts/results/quality/llama_diag_tinyllama_qwen3-0.6b.txt`, is
`--prefix none --no-qk-balance --recipes scalar4 smooth55` (the massive activation of Qwen3-0.6B at position 0 from
block 3 on, and the cosines it costs). Output: one JSON line per recipe (`blocks`, `opts`, `s_residual`, `env`, ...)
on stdout and `--out`, the table on stderr. CPU only; 64 GB of host memory for the models up to Qwen3-1.7B.

### `llama_attrib.py`

Says at which op a block's integer tensors leave the checkpoint's: every tensor of the chosen decoder layers
(`--blocks`, 0-based, or `all`) of one 128-token window (`--window`) is dequantised with the builder's scales (the
head equalisation, the q/k-norm balance, a smoothed norm's per-channel scales and the SwiGLU product balance undone)
and compared with the checkpoint's activation (`vs_hf`), with the exact float op on the graph's own inputs (`local`:
the error the op adds) and the exact op against the checkpoint (`carried`), by cosine and relative L2 error; the
attention scores of the graph's q and k against the checkpoint's, stage by stage of the q and k paths
(`score_steps`). `--mode teacher` feeds each block the checkpoint's residual (each block's own error), `accumulated`
runs the graph as is, `both` (default) does the two. Per block the record names the first op whose own error exceeds
`--flag-rel`, whose cosine falls below `--flag-cos` or which adds more than `--flag-nats` to the scores' error
(`first_flag`) and the ops of the largest errors (`worst`). Recipes as `llama_diag.py` (`--recipe`, default `main`);
`--prefix` and `--no-qk-balance` as there. The attribution located Qwen3's first-block loss at the k norm (Qwen3-0.6B's
layer-0 k-norm gains reach 96.5 on channels where q's are about 0), which the q/k-norm balance removes, and puts
Qwen3-4B's largest own error at the SwiGLU product in 34 of its 36 blocks, far larger in blocks 1-4.
`tests/test_llama_attrib.py` pins the method on tiny random models (and on a tiny Qwen3 with Qwen3-0.6B's real layer-0
q/k-norm gains). Output: one JSON line per (mode, block) on stdout and `--out`, the table on stderr. CPU only.

### Reproducing the paper's numbers

The run package's `quality` tier (`bash code/experiments/package/run.sh --platform <name> quality`) runs every
command of the table above but the 7B- and 13B-shaped Llama models, which its `qllama` tier runs (`run.sh --platform
<name> --llama-licence qllama`, with your own Llama-2 licence and a Hugging Face token (`export HF_TOKEN`; sbatch
passes it to the jobs) and `HF_HOME` on a disk with about 100 GB free; a tier of its own so that it can run on a
second GPU next to `quality`). It measures the Llama-2 rows and grids in twelve jobs, one per model, split and alpha
(Llama-2-7B's fp32 reference and Llama-2-13B's fp16 copy on the GPU; a job's time is mostly the calibration build
on the CPU, the same on both splits):

    for a in 0.45 0.55 0.65; do       # q-llama7-a<a>: 6 h limit, about 4 h
        python -m experiments.quality.ppl_llama --model meta-llama/Llama-2-7b-hf --alphas $a \
            --glu-smooths 0.1 0.2 0.3 --structure --device cuda --calib-device cpu --threads 8
    done
    for a in 0.5 0.6 0.7; do          # q-llama13-a<a>: 7 h limit, about 4 h
        python -m experiments.quality.ppl_llama --model meta-llama/Llama-2-13b-hf --alphas $a \
            --ref-dtype float16 --structure --device cuda --calib-device cpu --threads 8
    done
    # and both with --split validation instead of --structure: q-llama7-validation-a<a>, q-llama13-validation-a<a>

(`main-a0.55-glus0.2` is Llama-2-7B's row and `main-a0.6-glus0.2` Llama-2-13B's; the other cells are their
neighbours; `--graphs main smooth` adds the same graphs without the product balance, which the paper does not
report. One job per alpha keeps each job well inside its limit, and a job stopped at a limit, run again, repeats one
alpha's cells only.) Without `--llama-licence` the `qllama` tier runs, in their place, the licence-free stand-in
OpenLLaMA-7B (Llama-2-7B's shapes, not gated): its main cell on the test split with the structural diff (job
q-openllama7, its fp32 CPU reference kept in `logs/<name>/ref_open_llama_7b_v2.pt`) and the same 3 x 3 grid on the
validation split (job q-openllama7-validation); the paper does not report them. OpenLLaMA-7B ships only its
SentencePiece model, so its tokenizer needs `sentencepiece` and `protobuf` (`code/requirements-llm.txt`; `run.sh` checks
both before it submits the `qllama` tier without `--llama-licence`); Llama-2's repositories ship the fast tokenizer and
need neither. Job q-qwen4 keeps its fp32 CPU reference in `logs/<name>/ref_*.pt` (`--ref-cache`). The `qllama` tier's
rehearsal (`--rehearsal`) runs the grid path (two alphas, two product balances, the validation split, an fp16 reference)
on a 2-block smoke model with Llama-2-7B's vocabulary (job q-llama-access, which reads the gated repository from a
compute node) with `--llama-licence`, and with OpenLLaMA-7B's (job q-openllama7-smoke, which loads its tokenizer)
without it; the `quality` tier's rehearsal runs the tiny random model below. The other ratios come from the commands of
the table above, one checkpoint per run (TinyLlama and the Qwen3 models are not gated); the Qwen3-4B prefix study at 16
x 128 repeats the `ppl_llama.py` command of the table with `--prefix` set to `<|endoftext|>` (151643) twice or four
times, or followed by newline (198), `.` (13) or ` =` (284).
The run package's `qcheck` tier (`run.sh --platform <name> [--llama-licence] qcheck`, a tier of its own beside
`quality` and `qllama`) checks the Llama recipe beyond the cells it was chosen on: TinyLlama's and Qwen3-1.7B's alpha
and product balance (chosen on the test split) among their neighbours on the validation split (jobs
q-tinyllama-validation and q-qwen17-validation; `macros.VALIDATION_GRIDS`, `\TinyLlamaValGap`, `\QwenMValGap`), and
the main cells of TinyLlama, Qwen3-1.7B and, with `--llama-licence`, Llama-2-7B over calibration draws 0, 1 and 2
(jobs q-tinyllama-draws, q-qwen17-draws and q-llama7-draws; four train windows per draw, the prefix chosen on draw 0,
so draw 0 repeats the main cell; `macros.DRAW_MODELS`, `\LlamaDrawSpread`). Without `--llama-licence` the last job
is OpenLLaMA-7B's (q-openllama7-draws), which the paper does not report:

    python -m experiments.quality.ppl_llama --model TinyLlama/TinyLlama_v1.1 --split validation \
        --alphas 0.65 0.75 0.85 --glu-smooths 0.1 0.2 0.3 --device cuda --calib-device cpu --threads 8
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-1.7B-Base --split validation \
        --alphas 0.35 0.45 0.55 --glu-smooths 0.2 0.3 0.4 --device cuda --calib-device cpu --threads 8
    python -m experiments.quality.ppl_llama --model meta-llama/Llama-2-7b-hf --draws 0 1 2 \
        --device cuda --calib-device cpu --threads 8      # and TinyLlama, Qwen3-1.7B

The repository stores no record of these runs; the stored diagnosis is `llama_diag_tinyllama_qwen3-0.6b.txt` (above).

On a laptop (no checkpoint; the scripts' `--smoke` model downloads only a configuration and a tokenizer):

    PYTHONPATH="src;." python -m pytest -q tests/test_real_llama.py tests/test_llama_attrib.py
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-0.6B-Base --smoke --seq 128 --windows 2 \
        --calib-windows 1 --device cpu --threads 2 --chunk 64
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-0.6B-Base --smoke --check --threads 2 --chunk 48
    python -m experiments.quality.ppl_llama --model Qwen/Qwen3-0.6B-Base --smoke --protocol --threads 2
    python -m experiments.quality.llama_diag --model Qwen/Qwen3-0.6B-Base --smoke --threads 2
    python -m experiments.quality.llama_attrib --model Qwen/Qwen3-0.6B-Base --smoke --blocks all --threads 2

(about 20 s each, `--protocol` about a minute; `--model TinyLlama/TinyLlama_v1.1` for the Llama path).
