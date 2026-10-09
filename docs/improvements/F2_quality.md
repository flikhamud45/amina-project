# F2. The integer model's quality: where the perplexity goes, and SmoothQuant through the norm gains

Plan item F2 (reviewer objection D5: "how good is the integer model the protocol certifies?"). Commit e34bbf0
(`real_weights.build_opt_from_hf(smooth=, smooth_pct=, pct_att=)`). The study's scripts and raw results are on
branch `f2-quality` (`code/experiments/9_quality/`, `code/artifacts/results/9_quality/`); the literature survey is
`S/sp2027_research/f2_quality/literature.json`. Status: the attribution and the smoothing fix are measured on OPT-125M
and OPT-1.3B; OPT-6.7B (13.4 GB) needs a machine with more memory and disk than the 2080 Ti account and the laptop.

## 1. The problem

The protocol certifies an exact integer model: int8 weights with one scale per matrix, int8 activations with static
scales, an integer LayerNorm whose output is `G x normalised` in int8, an exp table with 8-bit probabilities, a 23-bit
residual stream. On WikiText-2 (16 windows of 128 tokens, the evaluation of `experiments/2_attack/real_weights_ppl.py`)
the best integer OPTs so far were OPT-125M 64.7 -> 76.6 (x1.18), OPT-1.3B 34.9 -> 38.1 (x1.09) and OPT-6.7B
27.1 -> 36.0 (x1.33). Published W8A8 results with the same constraints (per-tensor weights and static per-tensor
activations: SmoothQuant-O3) lose 1-2% on large OPTs; zkML systems at 16 bits lose under 1%.

## 2. Where the perplexity goes (attribution, OPT-125M)

A float re-implementation of the same graph, with each component switched between its exact integer form and float
(`attribution.py`; the all-integer hybrid gives the integer graph's logits exactly, the all-float one the checkpoint's
to `4e-5`). Perplexity with one component float and the rest integer (leave-one-out), and with only that component
integer (add-one); integer graph 75.5, fp32 64.8:

| Component | leave-one-out | add-one |
|---|---:|---:|
| LayerNorm outputs (all) | **69.4** | **70.3** |
| LayerNorm before attention | 71.0 | 68.7 |
| all weight matrices | 72.5 | 66.6 |
| fc1 requantisation | 72.7 | 66.3 |
| q/k/v weights | 74.8 | 65.0 |
| LayerNorm before the MLP | 74.9 | 65.9 |
| softmax (exp table, 8-bit probabilities) | 75.9 | 64.9 |
| attention output requantisation | 75.7 | 64.8 |
| residual stream (rounding, clamp) | 75.9 | 64.8 |
| ReLU table | 75.5 | 64.8 |

The LayerNorm outputs dominate, by their **rounding** (not clipping): OPT's residual stream has a few outlier
channels (and a large beginning-of-sequence channel) that inflate the standard deviation, so the other channels'
normalised values are small and a scalar gain `G` leaves them with few int8 levels. The 8-bit probabilities and the
residual stream are negligible.

## 3. The fix: SmoothQuant through the norm's gain vector

The integer LayerNorm already multiplies by a gain **vector** of public constants (`transformer._norm_int`). Channel
`j` gets `G_j = c / s_j`, `s_j = a_j^alpha / w_j^(1-alpha)`, with `a_j` the 99.9th percentile of `|normalised_j|` on the
calibration window and `w_j` the largest `|W_ij|` of the matrices the norm feeds; `c` maps the largest `a_j G_j` to
127. The inverse `1/G_j` is folded into column `j` of the next matrices before their (per-matrix) quantisation. This
is SmoothQuant-O3 (Xiao et al., ICML 2023) expressed in the integer graph: **every requantisation multiplier stays a
scalar, every op and shape is unchanged**, so the protocol's costs, the verifier and V1 are untouched (tested:
`test_smoothed_norm_gains_keep_the_graph_and_the_protocol`, accepted by v0 and V1). Together with percentile clipping
(99.99th) at the q/k/v/fc1 and attention-output requantisations:

| Model (WikiText-2, 16 x 128) | fp32 | before (best scalar gain) | smoothing, alpha = 0.55 | top-1 agreement with fp32 |
|---|---:|---:|---:|---:|
| OPT-125M | 64.76 | 75.52 (x1.17) | **66.62 (x1.03)** | 0.72 -> 0.82 |
| OPT-1.3B | 32.79 | 36.44 (x1.11) | **33.70 (x1.03)** | 0.82 -> 0.90 |

Both rows are reproduced with the committed builder (`build_opt_from_hf(..., smooth=0.55, smooth_pct=99.9, pct=99.99,
pct_att=99.99)`, laptop CPU, the same windows). (OPT-1.3B's fp32 is 32.8 here and 34.9 in the earlier file because the calibration and windows differ by run; the
ratios are on the same windows.) Robustness (OPT-125M, 32 evaluation windows, three different calibration
windows): smoothing alone gives 66.7-67.6 against fp32 64.7 (x1.03-1.05).

## 4. What else was measured, and why it is not adopted yet

* **Per-row power-of-two weight scales** (4 buckets per matrix) on top of smoothing: OPT-125M 64.5-65.4 against fp32
  64.7 (x1.00-1.01), top-1 agreement 0.87-0.88. The requantisation's multiplier then takes 4 distinct values per op
  (a vector), which v0 handles as any cheap op but V1 cannot cut yet (its windows assume one multiplier per op). Supporting
  a few multipliers per op in V1 (two window widths per bucket) is the next step if the paper needs the last 3%.
* More probability bits, the exp table's resolution, the residual stream: the attribution shows them negligible
  (each within 0.2 of fp32 when it alone is integer).
* Cross-layer equalisation (fc1/fc2 scalings that leave the float function unchanged) did not help on top of
  smoothing (67.8 against 66.6).

## 5. Literature targets (survey; ratios to fp32, sources in `literature.json`)

SmoothQuant-O3 (per-tensor static, our constraint): OPT-175B x1.016, OPT-IML-30B x1.008; CushionCache OPT-6.7B x1.00;
ZeroQuant-V2 per-token A8: x1.01 (125M), x1.04 (1.3B); zkLLM at 16 bits: x1.003-1.01. Realistic targets for our
protocol: OPT-6.7B <= x1.05, OPT-1.3B <= x1.04 (reached), OPT-125M <= x1.06 (reached).

## 6. What remains

* OPT-6.7B and Qwen3-4B with smoothing (memory: the friend's L40S node, or a larger server quota) [USER].
* The real-weight V1 byte profile with the smoothed graphs (several of OPT's residual multipliers exceed `2^29`, so
  those ops stay clear under V1's P3; measured by `v1_bytes.py` on a real-weight graph).
* Per-row scales in V1 (Section 4), if needed.
