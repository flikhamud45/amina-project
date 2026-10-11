# E2b. The exact law of the integer sampler, and its distance to softmax sampling on real logits

Plan 2 item E2b (reviewer points C-W6 and chg 7; the `\todo` at `paper_sp2027/sections/generation.tex:87`). Files:
`code/experiments/9_quality/sampler_tv.py` (the law, the closed forms, the Monte Carlo, the analysis on real logits),
`code/tests/test_sampler_tv.py`, results `code/artifacts/results/sampler_tv_opt-125m.json`. Status: implemented and
validated on the laptop (OPT-125M, 32 positions); the OPT-1.3B run (512 positions) is a cluster command (Section 9).
The protocol code is unchanged.

## 1. The question

E2 made sampled generation provable with a public integer Gumbel-max sampler (`token_sampler.TokenSampler.choose`):
claimed logits `z` (real logits `c z`), an integer weight `a = round(2^16 c / tau)`, the noise `T[u_v]` read from a
public 2^16-entry table with 16-bit words `u_v` of `SHAKE-256(tag || seed || position)`, and the token is the first
`v` maximising `a z_v + T[u_v]` among the `top_k` largest `z`. The paper states only that it "approximates"
sampling at temperature `tau` (and a worst case: candidates more than about 14.3 nats below the top are never drawn).
A reviewer asks how far the sampler's output is from `softmax(c z / tau)` on real logits. This item computes the
sampler's output distribution **exactly** and reports its total-variation (TV) distance to softmax sampling, next to
the integer model's own TV distance to the fp32 model.

## 2. The exact law

Model the SHAKE words as independent uniform words on `{0, ..., 2^B - 1}` (`B = 16`). The table
`T_i = round(2^B (-ln(-ln((i + 1/2) / 2^B))))` is strictly increasing (the derivative of `-ln(-ln u)` is at least `e`,
so neighbouring entries differ by at least `e` before rounding), so each value has probability `2^-B`. Let
`F(y) = #{j: T_j <= y} / 2^B` and `F^-(y) = #{j: T_j < y} / 2^B`, and let `S` be the candidate set (the `top_k`
largest `z`, ties to the smaller index, exactly as `choose` forms it with a stable sort). `choose` returns the first
maximum (`torch.argmax`), so `v` wins iff every earlier candidate scores strictly less and every later one at most as
much:

    P(v) = 2^-B sum_i  prod_{w in S, w < v} F^-(a z_v + T_i - a z_w)  prod_{w in S, w > v} F(a z_v + T_i - a z_w).

Masked entries score `int64 min` and never win.

## 3. Computing it on the integer grid

A direct evaluation costs `|S|^2 2^B` (`1.7e14` for OPT's vocabulary). The structure of the scores reduces it:

* **The window.** With `z* = max_S z` and `d_w = a (z* - z_w)`, a candidate with `d_w > D = T_max - T_0` can never
  win (its best score is below the top token's worst), not even by a tie. Only `W = {w in S: d_w <= D}` matters
  (`D = 933,901` for the 16-bit table: 14.25 nats).
* **The largest score's log-CDF on the grid.** On `s in [0, D]` (score threshold `a z* + T_0 + s`),
  `LM(s) = sum_{w in W} log F(T_0 + s + d_w)`. Every `d_w` is a multiple of `a`, so this is, for each residue
  `s mod a`, the correlation of the level histogram `n_k = #{w: z* - z_w = k}` with `g(y) = log F(T_0 + y)`
  sampled every `a` points. `log_max_cdf` computes it with one float64 FFT per block of residues (the grid viewed as
  `[Q, a]`, `Q = D/a + 1` levels, no copy), or by adding shifted copies of `g` when few levels are occupied. It is
  computed once per logit vector.
* **Gathering per token, with exact ties.** For `v` in `W`, in index order, at `s_i = T_i - T_0 - d_v`:

      P(v) = 2^-B sum_i exp( LM(s_i) - log F(T_i) + C_v(s_i) ),
      C_v(s) = sum_{w in W, w < v} log(F^- / F)(T_0 + s + d_w).

  `LM - log F(T_i)` is the non-strict product over all other candidates; `C_v` turns the factors of the earlier
  candidates into strict ones. `log(F^-/F)(y)` is nonzero only where `y` is a table value `T_j` (there it is
  `log(j / (j+1))`, `-inf` for `j = 0`), so `C_v` is a grid array that each token updates with one scatter of its
  `2^B` points after its own gather. Nothing is bracketed: the strict and non-strict comparisons are those of
  `choose`.
* **Cost and the skipped tail.** `O(D log D)` once, plus per token the draws `i` with `s_i >= 0` (few for tokens far
  below the top). Draws whose integrand is provably below `e^floor` (where `LM(s) + log 2^B < floor`; `C <= 0`) are
  skipped, gathers and scatters alike (`C` is never read there). Each token then loses less than `e^floor` and the law
  less than `|W| e^floor`, reported as `dropped_bound`; the analysis uses `floor = -60` (`4.4e-22` for OPT's 50,272
  tokens). On the laptop's slowest position this made the 20-bit law 3.1x faster (32.6 s to 10.4 s) and changed no
  probability by more than `7e-18` in four laws (16- and 20-bit tables, two positions; measured against
  `floor = -746`, below which float64's `exp` is 0).
* **Memory.** Three float64 arrays of `D + 1` entries at most plus one FFT block of 16 MiB: 7.5 MB each for the
  16-bit table, 145 MB for a simulated 20-bit one. Peak resident memory of the whole laptop analysis (three tables,
  Monte Carlo included, torch imported): 0.56 GB. Extracting the logits is a separate first step that loads the model
  (OPT-125M: 2.9 GB peak, 20 s); the logits are cached, and the model is gone before the analysis.

The only rounding is float64's. Each law reports `|sum_v P(v) - 1|` and, on the FFT path, the largest deviation of `LM`
from the direct sum at 256 random grid points.

## 4. Validation

All on the laptop (Windows, 2 threads).

* **Closed forms, exact arithmetic** (`--selfcheck`, tests). Two tokens: token 0 wins iff `a z_1 + T_j <= a z_0 + T_i`,
  so `P(0) = #{(i, j)} / 2^32`, counted in integers. Over 16 cases (equal logits, gaps of 3 and 70,000, the window's
  edge at `D - 1`, `D` and `D + 1`, `a = 4096`) the largest deviation of the law is `1.0e-25` (at the edge `D` the
  lower token wins only the single tie of `T_max` against `T_0`, and only if it has the smaller index: `2^-32`, which
  the law reproduces; at `D + 1` it never wins). Eight equal logits: `P(v) = 2^-128 sum_i i^v (i+1)^(7-v)` in exact
  rationals; largest deviation `2.8e-17`; the eight probabilities decrease from 0.1250076 to 0.1249924 (the
  tie-break), TV to uniform `1.7e-5`. Eight logits with repeated levels and one token outside the window: the product
  formula above in exact integers, largest deviation below `1e-15` (test).
* **Every noise draw through `choose`** (tests). With an 8-entry table (`B = 3`), all `8^4` noise vectors of 4 tokens
  are fed to `TokenSampler.choose` itself (its scores, its top-k mask, its argmax), for 6 cases with tied logits,
  tied scores and the top-k boundary (`top_k = 2, 3` with ties at the boundary): the exact frequencies equal the law to
  `1e-15`.
* **Monte Carlo of `choose` over 10^5 seeds** (seed `j` is `SHA-256(b"pvi/e2b/mc" || j)`, so the draws are the same on
  every platform). On the 8-token toy (`a = 2^15`): all 8 tokens within 3 sigma, largest `|z| = 1.50`, chi-square
  6.7. On real OPT-125M logits, the full 50,272-token vocabulary, the protocol's table (tokens with
  `n p (1 - p) >= 9` tested one by one, the rest pooled into one bucket that is tested too):

  | Position | tau | top-k | tokens tested | largest `|z|` | family-wise 1% bound | beyond 3 sigma (expected by chance) | chi-square / buckets | draws of `P = 0` tokens |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 33 | 1.0 | 0 | 768 + pool | 3.23 | 4.36 | 4 (2.1) | 816 / 769 | 0 |
  | 33 | 1.0 | 50 | 50 | 2.24 | 3.72 | 0 (0.14) | 50 / 51 | 0 |
  | 33 | 1.3 | 0 | 1,323 + pool | 3.35 | 4.48 | 5 (3.6) | 1,351 / 1,324 | 0 |
  | 127 | 1.0 | 0 | 105 + pool | 2.34 | 3.90 | 0 (0.29) | 108 / 106 | 0 |
  | 127 | 1.0 | 50 | 50 | 2.34 | 3.72 | 0 (0.14) | 57 / 51 | 0 |
  | 127 | 1.3 | 0 | 401 + pool | 2.73 | 4.22 | 0 (1.1) | 378 / 402 | 0 |

  (With top-k 50 the pool is empty, probability 0 and no draws; it is counted as a bucket with `z = 0`.) The strict
  per-token criterion ("every token within 3 sigma") holds in four of the six cases; in the two with 768 and 1,323
  tested tokens, 4 and 5 tokens exceed 3 sigma where 2.1 and 3.6 are expected by chance, with the largest `|z|` 3.23
  and 3.35 and the chi-square within 1.3 standard deviations of its expectation: consistent with the law, not evidence
  against it. With `m` buckets a correct law exceeds 3 sigma `0.0027 m` times on average, so the literal criterion
  cannot be asked of `m` in the hundreds; the statistically sound version is the family-wise (Bonferroni) bound
  `Phi^-1(1 - 0.005 / m)`, which the largest `|z|` exceeds with probability at most 1% under the law
  (`z_familywise_1pct` in the JSON, added in review; the column above is computed from each case's `m`). Every case
  is below it. The decisive check is per gap bin (scaled gap `c (z* - z_v) / tau`, bins of 2 nats): in every bin of
  every case the draws agree with the law (largest `|z| = 2.82`) and disagree with the ideal softmax wherever the two
  differ (`|z|` up to 46). Position 33, `tau = 1.3`, no top-k:

  | Scaled gap | tokens | law | ideal softmax | Monte Carlo | z vs law | z vs ideal |
  |---|---:|---:|---:|---:|---:|---:|
  | 0-2 | 24 | 0.2518 | 0.2238 | 0.2496 | -1.7 | 19.6 |
  | 2-4 | 261 | 0.3380 | 0.3163 | 0.3378 | -0.1 | 14.6 |
  | 4-6 | 1,382 | 0.2342 | 0.2308 | 0.2340 | -0.1 | 2.4 |
  | 6-8 | 6,437 | 0.1465 | 0.1552 | 0.1495 | 2.7 | -5.0 |
  | 8-10 | 17,037 | 0.0295 | 0.0608 | 0.0292 | -0.6 | -41.9 |
  | 10-12 | 21,224 | 0.00002 | 0.0127 | 0.00002 | -0.1 | -35.9 |
  | 12+ | 3,907 | 0 | 0.0004 | 0 | 0.0 | -6.4 |

* **FFT against the direct sum.** On 3,000 synthetic logits the two paths agree to `1e-12` in every probability
  (test). On the real logits (576 laws, 192 per table) the largest `LM` deviation at the checked points is `6.0e-11` and
  `|sum_v P(v) - 1|` is at most `1.3e-11`; the reported TVs are `1e-4` and above.
* **The grid against the definition on real logits** (review; a throw-away script, not in the repository). On the
  cached OPT-125M logits, `P(v)` was recomputed straight from the product formula of Section 2 (for each draw `i` of
  `v`, the exact integer counts `#{j: T_j < t}` or `#{j: T_j <= t}` of every other candidate by `searchsorted`; no
  window, no grid, no `C_v`). With top-k 50, every candidate at positions 0, 33 and 127 and all three temperatures:
  largest deviation from `sampler_law` `1.1e-16`, and the TVs to softmax equal the stored rows to all printed digits
  (e.g. `1.105913e-02`, `7.929651e-03`, `1.821073e-02`). Without top-k, at position 115 (the smallest window):
  10 tokens at `tau = 0.7` (direct path), from the top token through the window to its edge, relative deviation at
  most `1.1e-13`; 4 tokens at `tau = 1` (FFT path, `a = 18`), at most `7.9e-13`. Both include the edge: at
  `tau = 1` the token at level 51,883 (`18 x 51,883 = 933,894 <= D`) has `P = 4.42e-14` by both computations and
  the one at level 51,884 (`933,912 > D`) has 0 by both. 1,274 s on 2 threads.
* **The cluster command end to end on a tiny model** (review). The OPT-1.3B command, run on the laptop with
  `--model hf-internal-testing/tiny-random-OPTForCausalLM --seq 32 --windows 4 --calib-seq 32 --positions 4
  --workers 2 --mc 3000 --mc-positions 1 3` (a public 5-layer OPT with hidden size 16 and OPT's tokenizer and
  vocabulary), went through extraction (four windows, the exact-recovery check), the cache, two spawned workers, all
  three tables and the Monte Carlo in 3 min 39 s (peak 0.49 GB). It exposed one defect, fixed in review: on these
  near-uniform logits every token falls below the per-token variance, the pooled bucket holds the whole law, whose
  float64 sum is `1 + 1.2e-11`, and the z-scores of the pool and of the single gap bin divided that rounding by a
  zero variance (`max_abs_z = inf`, `z_law = -6.5e140`). `zscore` now clips the reference probability to `[0, 1]`
  and scores a bucket the reference makes certain or impossible 0 if its count agrees to half a draw (infinity
  otherwise). On real logits this changes no stored value (all z-scores there were finite); without it, the cluster's
  pass criterion could fail spuriously at a Monte Carlo position where every top-50 candidate lies within 2 nats of
  the top (one gap bin then holds the whole law, and its sum rounds above 1 about half the time).
* **Tests** (`tests/test_sampler_tv.py`, 22 tests, 9.8 s on the laptop): the above, the top-k restriction with ties
  to the smaller index, the skipped-tail bound (per token below `e^floor`, total below `dropped_bound`), the z-scores
  of certain buckets, and a wider table being closer to softmax (12/14/16-bit tables on 3,000 synthetic logits: TV
  to softmax 0.035, 0.0082, 0.0012).

## 5. Results: OPT-125M, 32 WikiText-2 positions (laptop)

The smoothed integer OPT-125M (`norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99`, calibrated on
the last 128-token window of the split), 32 evenly spaced positions of the first 128-token window of WikiText-2 test.
Its logit scale is `c = 2.708e-4`, so the protocol's weight is `a = round(2^16 c / tau) = 25, 18, 14` at
`tau = 0.7, 1, 1.3`: the sampler runs at the effective temperatures `2^16 c / a = 0.7098, 0.9859, 1.2676`
(+1.4%, -1.4%, -2.5%). With 20 fractional bits, `a = 406, 284, 218` (temperature errors below 0.2%).

TV distance of the sampler to `softmax_S(c z / tau)` (mean, median, maximum over the 32 positions), its two parts
(medians), and the integer model's own TV distance to the fp32 model at the same `tau` and top-k (median):

| tau | top-k | table | `tv_sampler` mean | median | max | `tv_table` | `tv_temp` | window (median) | `tv_model` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0 | **16** | 1.03e-2 | 9.27e-3 | 2.65e-2 | 2.33e-3 | 9.19e-3 | 2,492.5 | 0.147 |
| 0.7 | 0 | 16:20 | 3.67e-3 | 2.38e-3 | 2.43e-2 | 2.15e-3 | 6.12e-4 | 2,215.5 | |
| 0.7 | 0 | 20 | 1.03e-3 | 8.52e-4 | 3.51e-3 | 3.79e-4 | 6.12e-4 | 7,416 | |
| 0.7 | 50 | **16** | 7.44e-3 | 7.40e-3 | 1.12e-2 | 2.0e-6 | 7.40e-3 | 50 | 0.148 |
| 0.7 | 50 | 16:20 | 5.00e-4 | 4.97e-4 | 7.51e-4 | 6.3e-7 | 4.97e-4 | 50 | |
| 0.7 | 50 | 20 | 4.98e-4 | 4.97e-4 | 7.51e-4 | 1.2e-7 | 4.97e-4 | 50 | |
| 1.0 | 0 | **16** | 2.59e-2 | 2.55e-2 | 6.07e-2 | 1.36e-2 | 1.48e-2 | 20,102 | 0.132 |
| 1.0 | 0 | 16:20 | 1.63e-2 | 1.46e-2 | 4.94e-2 | 1.45e-2 | 2.42e-4 | 21,792.5 | |
| 1.0 | 0 | 20 | 2.22e-3 | 2.25e-3 | 3.21e-3 | 2.21e-3 | 2.42e-4 | 46,264 | |
| 1.0 | 50 | **16** | 7.83e-3 | 7.74e-3 | 1.04e-2 | 2.0e-6 | 7.74e-3 | 50 | 0.138 |
| 1.0 | 50 | 16:20 | 1.28e-4 | 1.26e-4 | 1.71e-4 | 2.7e-7 | 1.26e-4 | 50 | |
| 1.0 | 50 | 20 | 1.28e-4 | 1.26e-4 | 1.70e-4 | 1.2e-7 | 1.26e-4 | 50 | |
| 1.3 | 0 | **16** | 5.80e-2 | 5.81e-2 | 7.02e-2 | 3.48e-2 | 3.33e-2 | 48,653 | 0.114 |
| 1.3 | 0 | 16:20 | 3.69e-2 | 3.58e-2 | 4.60e-2 | 3.67e-2 | 2.48e-3 | 49,337.5 | |
| 1.3 | 0 | 20 | 2.95e-3 | 3.02e-3 | 3.95e-3 | 2.13e-3 | 2.48e-3 | 50,237 | |
| 1.3 | 50 | **16** | 1.31e-2 | 1.30e-2 | 2.18e-2 | 1.9e-6 | 1.30e-2 | 50 | 0.128 |
| 1.3 | 50 | 16:20 | 9.68e-4 | 9.63e-4 | 1.63e-3 | 2.6e-7 | 9.63e-4 | 50 | |
| 1.3 | 50 | 20 | 9.68e-4 | 9.63e-4 | 1.63e-3 | 1.2e-7 | 9.63e-4 | 50 | |

**16** is the protocol's table; `16:20` keeps its 2^16 quantiles and 16-bit words but stores the values with 20
fractional bits (and `a = round(2^20 c / tau)`); `20` is a 2^20-entry table read with 20-bit words. The mass outside
the window (never drawn) is small throughout: median `1.2e-3` (16-bit, `tau = 1`), below `4e-5` with the 20-bit table.
The integer model's own TV to fp32 has medians 0.114-0.148 and means 0.136-0.197, the means pulled up by position 0
(no context: 0.85 at `tau = 0.7`, 0.76 at `tau = 1`).

Per position, the ratio `tv_model / tv_sampler` with the protocol's table has median 15.6, 5.5, 2.0 at `tau = 0.7, 1,
1.3` without top-k (at least 10 at 22, 3 and 1 of the 32 positions) and 19.5, 18.4, 11.1 with top-k 50 (24, 26, 18
positions). With the 20-bit table it is at least 14.96 at every position and setting (the lowest three: 14.96,
15.9 and 17.4, all at position 74; medians 40-1,129).

Laptop cost: 58 min in all, of which 30 min for the 576 laws (means per setting: 0.2-1.1 s per 16-bit law,
2.8-4.1 s per `16:20` law, 4.1-8.5 s per 20-bit law) and 28 min for the six Monte Carlo runs (1.4-1.9 ms per draw
on the full vocabulary, 5.1-5.3 ms with top-k, which sorts 50,272 logits per draw); 2 threads, peak 0.56 GB.

### OPT-1.3B, 512 positions (cluster) [PENDING]

Same recipe, four 128-token windows (all 512 positions), from `artifacts/results/sampler_tv_opt-1.3b.json`.

| tau | top-k | table | `a` | `tv_sampler` mean | median | max | `tv_table` | `tv_temp` | `tv_model` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0 | 16 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 1.0 | 0 | 16 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 1.3 | 0 | 16 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 0.7 | 50 | 16 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 1.0 | 50 | 16 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 1.3 | 50 | 16 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 1.0 | 0 | 16:20 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | |
| 1.0 | 0 | 20 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | |
| 1.0 | 50 | 20 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | |

Logit scale `c` = [PENDING]; Monte Carlo (positions 64 and 511, `tau:top_k` = 1:0, 1:50, 1.3:0): [PENDING]; checks
`sum_err_max`, `fft_err_max`, `dropped_bound_max` = [PENDING]. Qwen3-1.7B: not run (no Qwen3 real-weight builder yet).

## 6. What the numbers say

* **The sampler is not negligible on OPT-125M at the protocol's 16 bits.** At `tau = 1` its TV to softmax sampling is
  about 0.026 (0.008 with top-k 50), against the plan's expectation of `1e-4` to `1e-3`; per position it is a median
  5.5 times (18.4 with top-k 50) below the integer model's own distance to fp32, not "at least 10 times" everywhere.
  Two causes, which the tables separate:
* **The weight's rounding (`tv_temp`).** The smoothed graph's logit scale is small (`c = 2.7e-4`: the integer logits
  are fine-grained), so `a = round(2^16 c / tau)` is 14-25 and the temperature is only known to `tau / (2^17 c)`, 2.8%
  at `tau = 1`. That is the entire error with top-k 50 (`tv_table` is `2e-6` there) and half of it without. Storing
  the table's values with 4 more fractional bits (`16:20`: same SHAKE words, a new table constant, scores still far
  inside int64) removes it: with top-k 50, 7.7e-3 to 1.3e-4 at `tau = 1` (60x).
* **The table's short upper tail (`tv_table` without top-k).** A candidate `Delta` nats (temperature-scaled) below the
  top wins only if its noise exceeds the best competitor's score by `Delta`. The noise is capped at
  `T_max / 2^16 = 11.78` and its top entries are coarse (11.78, 10.69, 10.17, 9.84, 9.59): each carries probability
  `2^-16`. When the top is crowded, the competitors' best noise is typically well above the single-draw median, so
  candidates 8-12 nats below the top are drawn far less often than softmax draws them. OPT-125M's next-token
  distributions are flat: at `tau = 1.3`, 38,000 tokens sit 8-12 nats below the top at position 33 and carry 7.4% of
  the softmax mass, of which the sampler draws 2.9% (bins in Section 4). This needs more uniform resolution, not a
  finer weight: `16:20` leaves it unchanged, while the 20-bit table (noise up to 14.56, top entries carrying `2^-20`)
  cuts `tv_table` 6-16x (1.36e-2 to 2.21e-3 at `tau = 1`).
* **With both** (the 20-bit table), the sampler's TV has means `1.0e-3`, `2.2e-3`, `3.0e-3` (medians `8.5e-4`,
  `2.2e-3`, `3.0e-3`) at `tau = 0.7, 1, 1.3` without top-k and `1.3e-4` to `9.7e-4` with top-k 50, about 15 times or
  more below the integer model's own distance at every position (at least 14.96). The worst case quoted in the paper (never drawing a candidate more than 14.3 nats below the top) is not
  the binding effect; the under-sampling 8-12 nats below a crowded top is.
* **Cost of the remedies** (not implemented; protocol changes). `16:20`: a new 2^16-entry table with values below
  `2^24` and `a = round(2^20 c / tau)` (the bound `a < 2^24` holds for `c / tau < 16`); nothing else changes. `20`:
  20-bit words (2.5 SHAKE bytes per vocabulary entry instead of 2, so 126 KB instead of 101 KB per position for OPT),
  a 2^20-entry table (4 MB as int32; its `decimal` fallback would take about 16x longer than today's 10 s), and the
  same scores and argmax.
* **What OPT-1.3B will show** is open: its logit scale (hence `a`) and the flatness of its next-token distributions
  decide both terms. The cluster run reports both parts, so the paper can state the measured number and, if needed,
  the table change.

## 7. A paragraph for the paper (replaces the `\todo` at `generation.tex:87`)

To replace `\todo{TV distance to softmax sampling on real logits}` inside the "Distribution" paragraph, after the
sentence ending "with probability below $7\cdot10^{-7}$." The placeholders are OPT-1.3B medians over its 512
positions from `artifacts/results/sampler_tv_opt-1.3b.json` (`summary`, table `16`, `tau = 1`): X1 = `tv_sampler`
median with `top_k = 50`, X2 = with `top_k = 0`, X3 = `tv_model` median with `top_k = 0`, X4 = `tv_sampler` median
of table `20` with `top_k = 0`.

```latex
We computed the sampler's output distribution exactly (ties included; the only error is float64 rounding, below
$10^{-10}$) with one FFT over the integer score grid, from the integer OPT-1.3B's logits at 512 WikiText-2
positions. At $\tau=1$ its total-variation distance to
$\mathrm{softmax}(cz/\tau)$ has median \todo{X1} with $k=50$ and \todo{X2} without top-$k$, against a median of
\todo{X3} between the next-token distributions of the integer and the fp32 model. Two effects account for it:
the rounding of $a$, a relative temperature error of up to $\tau/(2^{17}c)$, and, without top-$k$, the bounded
upper tail of $\Gamma$, which under-samples candidates 8 to 12 nats below a crowded top; storing $\Gamma$ with
four more fractional bits removes the first, and a table of $2^{20}$ entries lowers the distance to \todo{X4}.
```

If OPT-1.3B's X1 and X2 come out below about X3/10, the last sentence can shrink to "Both effects, the rounding of $a$
and the bounded upper tail of $\Gamma$, are below the model's own quantisation error." (OPT-125M's laptop values for
the same placeholders: X1 = 7.7e-3, X2 = 2.6e-2, X3 = 0.13, X4 = 2.2e-3.)

## 8. Limitations

* **The uniform-word model.** The law assumes the SHAKE-256 words are independent and uniform. That is the standard
  model for a hash keyed by a fresh client seed; it is not something the Monte Carlo can test beyond its resolution
  (the 10^5 draws are SHAKE outputs, and they agree with the law per token and per bin).
* **Float64, not rationals.** The real-logit laws are float64 computations whose rounding is reported: `|sum - 1|`
  at most `1.3e-11` and the `LM` deviation at most `6e-11` without top-k, against TVs of `1e-4` and more; the
  smallest numbers in Section 5 (`tv_table` near `1e-7` with top-k 50) come from laws with `|sum - 1| <= 1.4e-14`.
  The exact-rational checks cover the toys only.
* **Skipped draws.** With `floor = -60`, draws with integrand below `e^-60` are skipped; the bound on what that removes
  (`4.4e-22` per law) is reported with each law. `--floor -746` disables it (bit-for-bit what float64 gives).
* **Few positions on the laptop.** 32 positions of one WikiText-2 window, one calibration window, one small model.
  The table's tail matters in proportion to how flat the next-token distributions are; whether OPT-1.3B's are
  flatter or more peaked than OPT-125M's is what the cluster run measures (512 positions, four windows).
* **Simulated tables.** The `16:20` and `20` rows are the same construction at other widths; the protocol's table is
  the 16-bit one. Adopting either is a protocol change (a new table constant and hash, and for `20` wider SHAKE words),
  not made here.
* **Top-p, repetition penalties, logit bias** are not implemented by the sampler (plan item E2b-3), so not measured.
* **Qwen3-1.7B** needs a Qwen3 real-weight integer builder (F2b-Q); the script refuses non-OPT models until then.

## 9. How to run

From `code/`, with `PYTHONPATH=src`:

    # the toys only (closed forms, exact rationals, 10^5-draw Monte Carlo of choose on 8 tokens): about 10 s
    python experiments/9_quality/sampler_tv.py --selfcheck
    # the tests
    python -m pytest -q -o addopts= -p no:cacheprovider tests/test_sampler_tv.py
    # OPT-125M, 32 positions (laptop: 2 threads, 58 min with the Monte Carlo, peak 0.56 GB in the analysis; the
    # logits are extracted once, peak 2.9 GB, and cached; artifacts/fullcheck/cache/ is git-ignored)
    python experiments/9_quality/sampler_tv.py --model facebook/opt-125m --seq 128 --windows 1 --positions 32 \
        --threads 2 --logits-cache artifacts/fullcheck/cache/e2b_opt-125m.npz --mc 100000 --mc-positions 8 31 \
        --mc-settings 1:0 1:50 1.3:0 --out artifacts/results/sampler_tv_opt-125m.json
    # OPT-1.3B, 512 positions (cluster: a CPU node; extraction about 10-20 min, analysis about 35 min on 16
    # workers, Monte Carlo about 30 min in the main process (28 min on the laptop); these durations are estimates
    # from the laptop's per-law and per-draw times)
    env HF_HOME=/tmp/$USER/hf OMP_NUM_THREADS=8 PYTHONPATH=src python experiments/9_quality/sampler_tv.py \
        --model facebook/opt-1.3b --seq 128 --windows 4 --threads 8 --workers 16 \
        --logits-cache /tmp/$USER/e2b_opt-1.3b.npz --mc 100000 --mc-positions 64 511 \
        --mc-settings 1:0 1:50 1.3:0 --out artifacts/results/sampler_tv_opt-1.3b.json

Options: `--taus` (default 0.7 1 1.3), `--topk` (0 50), `--tables` (16 16:20 20), `--floor` (-60), `--workers`
(processes over positions, about 0.6 GB each with a 20-bit table), `--first N` (only the first N cached positions),
`--extract-only`, `--device cuda` (the integer graph on the GPU, as in `ppl_smooth.py`; not exercised for this item:
the analysis is CPU work). The JSON holds the
summary per setting (mean, median, 90th percentile, maximum over positions of `tv_sampler`, `tv_table`, `tv_temp`,
`tail_mass`, the window size, and `tv_model`), every per-position row, the Monte Carlo comparisons (per token and per
gap bin), the toys, seconds and peak memory.

### Cluster commands (from `code/`, on a CPU node; `env` works under tcsh and bash)

1. `env PYTHONPATH=src python -m pytest -q -o addopts= -p no:cacheprovider tests/test_sampler_tv.py`: `22 passed`.
2. `env PYTHONPATH=src python experiments/9_quality/sampler_tv.py --selfcheck --out artifacts/results/sampler_tv_selfcheck.json`:
   `toys.two_token_max_err <= 1e-15`, `toys.eight_equal.max_err <= 1e-15`, `toys.eight_mc.beyond_3sigma == 0` with
   `max_abs_z = 1.498...` (below `z_familywise_1pct = 3.26...`) and `chi2 = 6.710...` (the draws are SHA-256
   seeds, so these equal the laptop's exactly; 4.6 s and 0.18 GB there).
3. OPT-1.3B, 512 positions, all settings, Monte Carlo at two positions:

       env HF_HOME=/tmp/$USER/hf OMP_NUM_THREADS=8 PYTHONPATH=src python experiments/9_quality/sampler_tv.py \
           --model facebook/opt-1.3b --seq 128 --windows 4 --threads 8 --workers 16 \
           --logits-cache /tmp/$USER/e2b_opt-1.3b.npz --mc 100000 --mc-positions 64 511 \
           --mc-settings 1:0 1:50 1.3:0 --out artifacts/results/sampler_tv_opt-1.3b.json

   Expected: progress lines `position n/512 ...` on stderr, one JSON line on stdout; `meta.logit_scale` printed;
   18 `summary` rows (3 tau x 2 top-k x 3 tables). Pass: every row has `sum_err_max < 1e-10`,
   `fft_err_max < 1e-9` (or null) and `dropped_bound_max < 1e-21`; every `mc` entry has `zero_prob_draws == 0`,
   `max_abs_z < z_familywise_1pct` (`beyond_3sigma` is reported next to its chance expectation `expected_beyond` for
   information; a few above it is chance) and `|z_law| < 3.5` in every bin of `bins`; the 20-bit rows have smaller
   `tv_table` than the 16-bit rows at the same tau and top-k. The TV values themselves fill Section 5's OPT-1.3B
   table and the paragraph's X1-X4. About 16 x 0.6 GB in the analysis; the extraction loads the fp32 checkpoint and
   the integer graph (estimated 10-15 GB from OPT-125M's 2.9 GB). `--workers` beyond the CPUs the job is given only
   oversubscribes them (the run is then slower, not different: the rows do not depend on the number of workers).
   The same command ran end to end on a tiny public OPT on the laptop (Section 4), so it needs no edits.
4. Qwen3-1.7B, once a Qwen3 real-weight builder exists (the script refuses non-OPT models until then):

       # env HF_HOME=/tmp/$USER/hf OMP_NUM_THREADS=8 PYTHONPATH=src python experiments/9_quality/sampler_tv.py \
       #     --model Qwen/Qwen3-1.7B-Base --seq 128 --windows 4 --threads 8 --workers 16 \
       #     --logits-cache /tmp/$USER/e2b_qwen3-1.7b.npz --mc 100000 --mc-positions 64 511 \
       #     --mc-settings 1:0 1:50 1.3:0 --out artifacts/results/sampler_tv_qwen3-1.7b.json
