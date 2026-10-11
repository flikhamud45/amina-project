# F2a: integer-model quality at 2,048 tokens, and 8-bit against 11-bit attention probabilities

Plan 2, item F2a (`SP2027_PLAN2.md`, Sec. 3.2). Code: `code/experiments/9_quality/ppl_2048.py` and
`code/experiments/9_quality/opt_builder.py` (commits b244e8b and cdeb44a). Cluster jobs 1007072 (checks, legacy setting,
OPT-125M), 1007073 (OPT-1.3B) and 1007074 (OPT-6.7B).

## 1. Why

* The paper's quality table (`sections/quality.tex`) scores 16 windows of 128 tokens and calibrates on the test split's
  last window. Its costs, however, are reported at 2,048 tokens, and the published OPT perplexities (14.62 for OPT-1.3B,
  10.86 for OPT-6.7B) are at 2,048 tokens on the whole test split. Reviewer C asks for quality at the reported length
  (C-W5, C-Q2, C-Q3, consolidated change 7); three `\todo`s in `quality.tex` wait on it.
* 128-token windows cannot show what 8-bit probabilities cost. The probabilities are `p = round(255 e / tot)`, so at
  most 510 keys of a row get `p > 0`, and a diffuse row over many keys loses mass only at long context.
* The L40S cell `q-opt67-2k` would have needed about 50 GB in the builder and again in `real_logits`, an almost certain
  out-of-memory failure on a 48 GB card. A memory-lean path moves OPT quality to the 2080 Ti.

## 2. What was built

### 2.1 The evaluation path (`ppl_2048.py`)

* **Text.** WikiText-2 (raw), joined with blank lines (`"\n\n".join`, the convention behind the published 2,048-token
  numbers) and tokenised once.
* **Windows.** Every non-overlapping `--seq`-token window of the test split (about 140 for OPT's tokenizer), each scored
  on its own. All positions come out of one pass, which is exactly what one query per prefix would give: every op is
  per-position except causal attention, and no constant depends on the input.
* **Calibration.** Draw `d` is `--calib-windows` (default 4) consecutive 2,048-token windows of the *train* split,
  starting at window `4d`. No evaluated token is seen during calibration (before, the calibration window was part of
  the test split).
* **Lean.** The integer forward frees each tensor after its last reader (`forward(free=True)`) and keeps no claim
  (`send=lambda z: None`). The LM head's logits are reduced to NLL and argmax in chunks of 256 positions. The graph
  runs on the GPU (`with_constants_on("cuda")`; weights move on first use), the calibration on the CPU.
* **Reported per cell.** Perplexity; the ratio to fp32 with a 95% bootstrap interval over windows; top-1 agreement
  with fp32; per-layer probability mass (below); the per-window NLL, so calibration draws can be pooled later; the
  graph digest; build and evaluation seconds; peak host and GPU memory. One JSON line per cell, also appended to
  `--out`.

### 2.2 The graphs (`opt_builder.py`)

The builder of branch `f2-quality` was ported. It builds OPT's integer graph with options that change only public
constants of cheap ops or the offline weight quantisation, never an op, a shape or a cost:

| Graph | Options |
|---|---|
| `scalar` | one norm gain (`--gain`; the earlier sweep's best: 4 / 3 / 2 for OPT-125M / 1.3B / 6.7B), no clipping |
| `smooth` | SmoothQuant through the norm gains (alpha 0.55, smooth_pct 99.9), 99.99th-percentile clipping at q/k/v/fc1 and the attention output: the paper's model |
| `rows` | `smooth` plus per-row power-of-two weight scales with 4 buckets (four multipliers per requantisation: v0 only) |

Each graph is built at `prob_bits` 8 (today's probabilities, `P_MAX = 255`) and 11 (`P_MAX = 2047`). The 11-bit graph has
its own calibrated output multiplier `m_o` (the raw `P V` scale is `s_v / P_MAX`). Its attention is
`opt_builder.attention_var`, exact integer arithmetic in head chunks: float32 products are exact because int8 operands give
`|s| <= 127^2 dh < 2^24`, and a row of probabilities sums to at most `P_MAX + T/2`, so `|P V| < (2047 + T/2) 128 < 2^24`
for `T < 2^14`.

Changes to the port, so that 8-bit graphs are the paper builder's (`real_weights.build_opt_from_hf`) bit for bit:

* percentiles read the first `2^24` entries of a tensor, as the paper builder does (the copy drew a random subset);
* the attention output's percentile is taken before the scale is applied, as there;
* each layer's calibration tensors are freed once the layer is done, so 4 x 2,048 calibration tokens fit in host memory
  for OPT-6.7B.

### 2.3 Two ablations

* **Ideal softmax** (`--float-softmax`): the same 8-bit graph with `255 softmax(-gap m_s / 2^30 / 16)` in float64 over the
  same integer scores, i.e. no exp table and no rounding of the probabilities. It bounds what any probability format can
  recover.
* **Probability mass**: on the first `--mass-windows` windows, `attention_var` records, per layer, the mean over rows and
  heads of `sum_j p_j / P_MAX`, the share of mass a rounded row keeps.

### 2.4 Checks (`--check`, `--legacy`)

`--check` (OPT-125M, one 256-token window) verifies:

1. `opt_builder` at 8 bits equals `build_opt_from_hf` for the scalar and the smoothed recipe: every weight and bias, every
   cheap op's parameters and captured constants, the logit scale, and the logits.
2. The lean path's NLL and argmaxes equal those of `real_logits(all_positions=True)`.
3. `attention_var` at `P_MAX = 255` gives `transformer._attention`'s integers.
4. On a GPU, the same integers as on the CPU.

The graph digests of the two builders differ, as they must: the digest names each cheap op's function by module and
qualified name, so a copied builder cannot match it. Equal weights, constants and logits show the same function, and the
committed model is still built by the paper builder. If 8 bits stay, the paper's digest is unchanged.

`--legacy` reruns the paper's 16 x 128 setting through the new path (expected for OPT-1.3B: 32.79 fp32, 36.44 with gain 3,
33.70 smoothed).

## 3. Results

### 3.1 Laptop (OPT-125M)

`--check` on the laptop CPU: all equalities hold for both recipes (weights, parameters, constants, logit scale, lean
path = `real_logits`, `attention_var` = `_attention`); the GPU check needs the cluster.

| One 256-token window | perplexity | smallest per-layer mass |
|---|---:|---:|
| scalar, 8-bit | 55.80 | 0.969 |
| smoothed, 8-bit | 52.91 | 0.966 |
| smoothed, 11-bit | 51.87 | 0.997 |

Grid smoke test, 2 windows of 512 tokens (1,022 predicted tokens), one calibration draw of 4 x 512 train tokens:

| Graph | 8-bit | ideal softmax | 11-bit | smallest mass (8-bit / 11-bit) |
|---|---:|---:|---:|---:|
| fp32 | 35.47 | | | |
| scalar (gain 4) | 42.00 (x1.184) | 41.73 (x1.177) | 40.33 (x1.137) | 0.940 / 0.994 |
| smoothed | 38.07 (x1.073) | 37.36 (x1.053) | 37.29 (x1.051) | 0.937 / 0.994 |
| rows | 37.12 (x1.046) | 36.50 (x1.029) | 36.48 (x1.028) | 0.938 / 0.994 |

Reading (two windows only, so indicative):

* Already at 512 tokens, the 8-bit probabilities cost about 2% of perplexity on the smoothed graph (x1.073 against
  x1.051), and the rounded rows keep only about 94% of their mass in the worst layer.
* 11 bits recover essentially all of it: the 11-bit graph is within 0.2% of the ideal softmax in every graph.
* The plan's rule adopts 11 bits if the 8-bit cost is above 1%. The cluster cells confirmed it (Section 3.2), and the
  11-bit constants go into `transformer.py` (two paths), `reference.py`, `attention_kernels.py` and `native_kernels.py`,
  with their exactness tests (plan: one extra day; decision Mon 19 Oct). Proof bytes and verifier time do not change,
  since the probabilities are internal to a cheap op.

### 3.2 Cluster (2,048 tokens, all 140 test windows, 286,580 predicted tokens)

Jobs 1007844 (OPT-125M), 1007845 (OPT-1.3B) and 1007846 (OPT-6.7B) on the RTX 2080 Ti nodes, git cdeb44a plus the
device fix of 74812c9; the integer forward on the GPU, calibration on the CPU (4 x 2,048 train tokens per draw), fp32
reference on the GPU (CPU for OPT-6.7B). Records: `code/artifacts/results/9_quality/ppl2048_opt-*.jsonl`.

**The fp32 references reproduce the published 2,048-token values:** 27.656 (OPT-125M; published 27.65), 14.624
(OPT-1.3B; 14.62), 10.860 (OPT-6.7B; 10.86). `--check` passed on the cluster as well (GPU = CPU), and `--legacy`
reproduced the paper's 16 x 128 numbers exactly (32.79 fp32, 36.44 scalar gain 3, 33.70 smoothed).

Ratio to fp32 (draw 0 with its 95% bootstrap interval over windows; for 3 draws, the range over the draws):

| Model | graph | 8-bit | ideal float softmax | 11-bit | smallest per-layer mass, 8-bit / 11-bit |
|---|---|---:|---:|---:|---:|
| OPT-6.7B | smoothed, alpha 0.5 | x1.039 [1.037, 1.042] | x1.018 | **x1.017** [1.016, 1.018] | 0.561 / 0.976 |
| OPT-6.7B | smoothed, alpha 0.55 | x1.041 [1.039, 1.044] | x1.022 | x1.017 [1.016, 1.018] | 0.561 / 0.975 |
| OPT-6.7B | smoothed, alpha 0.65 | x1.045 | x1.025 | x1.017 | 0.561 / 0.976 |
| OPT-6.7B | scalar (gain 2) | x1.651 [1.592, 1.714] | x1.835 | x1.696 [1.616, 1.800] | 0.563 / 0.978 |
| OPT-1.3B | scalar (gain 3) | x1.151-1.156 | x1.097-1.101 | **x1.095-1.097** | 0.594 / 0.976 |
| OPT-1.3B | smoothed, alpha 0.55 | x1.184-1.366 | x1.138-1.294 | x1.144-1.254 | 0.594 / 0.977 |
| OPT-1.3B | rows | x1.176-1.257 | x1.133-1.200 | x1.158-1.235 | 0.593 / 0.977 |
| OPT-125M | scalar (gain 4) | x1.434-1.436 | x1.163-1.168 | x1.161-1.166 | 0.719 / 0.976 |
| OPT-125M | smoothed, alpha 0.55 | x2.12-2.19 | x2.01-2.06 | x1.93-1.97 | 0.718 / 0.974 |
| OPT-125M | rows | x2.14-2.16 | x1.89-1.91 | x1.83-1.85 | 0.719 / 0.974 |

Top-1 agreement with fp32 on OPT-6.7B: 0.875 (8-bit) and 0.922 (11-bit), smoothed alpha 0.55.

**Reading.**

* **8-bit probabilities cost 2-23% of perplexity at 2,048 tokens** (OPT-6.7B x1.041 -> x1.017; OPT-1.3B scalar x1.156 ->
  x1.096; OPT-125M scalar x1.435 -> x1.166). In the worst layer a rounded row keeps only 56-72% of its probability
  mass. 11 bits keep 97.5-97.7%. In the scalar-gain cells and on OPT-6.7B they match or beat the ideal float softmax
  (within 0.3%; the float row keeps the 8-bit graph's output multiplier, while the 11-bit graph has its own); in the
  unstable smoothed cells of the small models the comparison varies with the draw (OPT-1.3B draw 2: x1.235 against
  x1.147).
* **Decision (the plan's rule: adopt 11 bits if the 8-bit cost exceeds 1%).** The certified models use 11-bit
  probabilities: `transformer.CERTIFIED_PROB_MAX = 2047`, the default of `build_opt_from_hf` and
  `build_llama_from_hf` (commit e5d377c). The native CPU and fused GPU kernels take the scale and passed their
  exactness tests at 255 and 2047 on the cluster (job 1007805: 668 passed). Proof bytes and verifier time do not
  change. The random-weight benchmark graphs keep 255 (same shapes, same costs), so the stored benchmark records
  stay valid.
* **OPT-6.7B, the 7B-class model, reaches x1.017** at 2,048 tokens on the full test set (smoothed, 11 bits), within
  the plan's target (x1.05) and close to SmoothQuant-O3's reported 1-2%.
* **The smoothed recipe does not transfer to the small models.** It was chosen at 16 x 128 with calibration on the
  test split. At 2,048 tokens with train-split calibration it is worse than the scalar gain on OPT-1.3B, and it varies
  with the calibration draw (draw 1: x1.25-1.37); on OPT-125M it roughly doubles perplexity. The suspects are the
  99.99th-percentile clipping at q/k/v/fc1 and the attention output, and the 99.9th-percentile channel statistics of
  the smoothing gains, both estimated on 8,192 train positions. A recipe search at 11 bits is running (job 1008668:
  scalar gains 2/3/4; smoothing with alpha 0.3/0.4/0.5 without clipping, and with channel maxima; two draws).
* **Per-row power-of-two scales** (`rows`) do not help at 2,048 tokens either (OPT-1.3B x1.158-1.235 at 11 bits).
* **OPT-6.7B needs the smoothing:** with one scalar gain it stays at x1.65-1.70 at both probability widths (the
  outlier channels), while the smoothed graph reaches x1.017.

## 4. Limitations

* OPT only; the Qwen3 and Llama builders are F2b-Q and F2b-L.
* The ideal-softmax row keeps the 8-bit graph's output multiplier; it isolates the probabilities, not a recalibrated
  graph.
* The `rows` graph needs four multipliers per requantisation and is v0 only.

## 5. How to run

    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-125m --check
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-1.3b --legacy --graphs scalar smooth
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-1.3b --device cuda \
        --graphs scalar smooth rows --pbits 8 11 --float-softmax --draws 0 1 2 --out ppl2048_opt-1.3b.jsonl
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-6.7b --device cuda \
        --ref-device cpu --graphs smooth scalar --alphas 0.55 0.5 0.65 --pbits 8 11 --float-softmax --draws 0
