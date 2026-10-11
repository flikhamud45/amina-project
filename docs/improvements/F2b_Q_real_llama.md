# F2b-Q (and F2b-L's laptop part). Real-weight integer Qwen3 and Llama decoders

Plan 2 items **F2b-Q** (must: a real-weight builder for RMSNorm / RoPE / SwiGLU / GQA models, Qwen3 first) and the
laptop part of **F2b-L** (should, rank 6: TinyLlama-1.1B on the same builder), `SP2027_PLAN2.md` lines 435-486.
Status: **implemented; validated on the laptop on tiny randomly initialised Hugging Face models; the real-checkpoint
runs (Qwen3-0.6B/1.7B/4B-Base, TinyLlama-1.1B) are cluster commands whose numbers are placeholders in Section 9.**
The opt-in per-token exponent at the SwiGLU product (the plan's outlier handling, task item 5) is implemented and
tested for exactness and the protocol, off by default, and not yet measured on a real model. The PrefixQuant-style
prefix (the plan's cheaper alternative) is not implemented.

## 1. The problem

Only OPT had real-weight integer graphs (`real_weights.build_opt_from_hf`). The cost tables lead with Llama-2-7B
and Qwen3-4B shapes measured on random int8 weights (`transformer.build_decoder`), which is sound for cost (the
prover's and verifier's work depend only on the shapes) but says nothing about the quality of the integer model the
protocol certifies for those families (C-W1, C-W5, C-Q2). Qwen and Llama differ from OPT in every place where the
integer graph's constants come from: RMSNorm instead of LayerNorm, RoPE instead of learned positions, a SwiGLU MLP
(a SiLU table and an int8 x int8 product) instead of ReLU, grouped-query attention, Qwen3's per-head q/k norms, and
a 151,936-entry vocabulary. Both families also carry *massive activations* at the down projection's input, where
static per-tensor scales are known to fail on Llama (the literature's W8A8 Llama-2-7B with static per-tensor
activations: perplexity 9,250).

## 2. What was built

| File | What |
|---|---|
| `code/src/pvi/fullcheck/real_llama.py` (new) | `build_llama_from_hf(model, calib_ids, *, pct, norm_gain, smooth, smooth_pct, pct_att, prune_last=False, device=None, pct_glu=None, head_eq=True, residual_headroom=1.25, n_layers=None, glu_exp=0)` -> `(IntGraph, info)` for HF `LlamaForCausalLM` and `Qwen3ForCausalLM`; the lean evaluation `lean_logits` (chunked LM head) and `real_logits`; `residual_states` / `hf_residual_states` / `block_cosines` (per-block cosine against the checkpoint); `config_of` / `same_shape`; `matches_benchmark_graph`; the opt-in `swiglu_exp` / `residual_exp` ops (`glu_exp=E`) |
| `code/src/pvi/fullcheck/transformer.py` | `CONFIGS` gains `qwen3-0.6b`, `qwen3-1.7b`, `tinyllama-1.1b` (the checkpoints' `config.json` shapes); no existing entry changed |
| `code/tests/test_real_llama.py` (new) | 28 tests on tiny random Llama and Qwen3 models, no download (Section 7) |
| `code/experiments/9_quality/ppl_llama.py` (new) | perplexity at 16 x 128 and at 2,048 tokens, fp32 reference, scalar and smoothed graphs, top-1, per-block cosine, `--check`, `--protocol`, `--smoke` |

`real_weights.py` is not touched (another item owns it); `real_llama.py` has its own builder class with
`_RealBuilder`'s arithmetic for the weight ops, requantisations and residual adds.

## 3. How the graph is built

The graph is `build_decoder`'s for the model's shape: the same op kinds, in the same order, with the same names,
the same wiring (every op's inputs and output), the same weight-op shapes, layouts and biases, the same cheap-op
notes and the same claim columns. Only the constants differ:

| Op (benchmark note) | `build_decoder` (random, for cost) | `build_llama_from_hf` (real) |
|---|---|---|
| token embedding (`embed` weight op) | random int8 | `E^T` quantised at `c s_res` |
| "embed to residual" | x 8 | x `c = max(1, round(max|E| / 127 / s_res))` |
| "norm" (RMSNorm) | gain 32 | scalar gain `G` (`norm_gain`), or SmoothQuant gains `G_j` (`smooth`); gamma folded into the next matrices (`W diag(gamma)`), so no bias appears |
| q, k, v projections, "requant" | random int8, scale from a target std | `W diag(gamma)` (and `1/G_j` in its columns), one scale per matrix; requantised at the `pct` percentile of the real values; where RoPE follows, of each RoPE pair's norm |
| "q norm", "k norm" (Qwen3) | gain 32 | gain vector `round(G gamma_j 2^16)` (the checkpoint's `q_norm` / `k_norm` weights), `G` from the calibration percentile of the pairs' norms of `gamma_j x_j / rms` |
| "rope" | Q14 tables | the same op (scale kept: a rotation) |
| "attention" | `m_s`, `m_o` from stds | `m_s = round(2^30 16 s_q s_k / sqrt(head_dim))` (HF's `head_dim^-0.5`), `m_o` from the `pct_att` percentile of the raw `P V` |
| o projection, "residual add" | multiplier from a target std | `m = round(2^30 s_z / s_res)` |
| gate, up, "requant" | target std | as q/k/v |
| "silu" | `round(24 silu(x / 24))` | `round(silu(x s_g) / s_a)`, `s_g` the gate's real scale, `s_a = max_x |silu(x s_g)| / 127` over the 256 inputs (`silu(127 s_g) / 127` unless `s_g` is tiny) |
| "swiglu" (`requant_mul`) | multiplier from stds | `round(2^30 s_a s_u / s_p)`, `s_p` from the `pct_glu` percentile of `|silu(gate) up|` |
| down projection, "residual add" | target std | real ratio |
| final "norm", "last position", LM head | random | `W_head diag(gamma_final)`; tied (Qwen3) and untied (Llama) heads are both their own weight op, as in the benchmark graph |

The residual stream's scale `s_res` is set so the fp checkpoint's hidden states (the embedding's output and every
block's output, recorded through forward hooks on the calibration windows, one window at a time) never clip at
`RES_MAX = 2^22 - 1` (headroom 1.25), as for OPT. Percentiles read at most `2^24` entries (`torch.quantile`'s limit):
whole rows, evenly spaced over all calibration windows (the OPT builder reads the first `2^24` entries, which for wide
tensors is the first window only). The smoothing statistics read at most `2^14` rows, likewise spread.

**The structural diff is empty.** RMSNorm has no shift, so folding the final norm leaves no LM head bias (OPT's one
unavoidable difference). `matches_benchmark_graph(graph, config, n_layers=, prune_last=)` compares the op count,
then per op its kind, name, inputs and output, a weight op's `(n_rows, n_in, layout)` and bias, a cheap op's note, and
finally `IntGraph.claim_columns` on a 9-token query; it returns `[]` for both families, pruned and unpruned (tested),
and a non-empty list against the wrong pruning (tested, so the check can fail).

### 3.1 Conventions checked against Hugging Face (transformers 4.57.6 and 4.51.3)

transformers 4.51.3 is the version `code/requirements-llm.txt` pins (the cluster's); the tests and the checkpoint
loader were run under both (Section 7).

* **RoPE.** HF: `q cos + rotate_half(q) sin`, `rotate_half(x) = [-x2 | x1]`, `cos = [cos | cos]`, frequencies
  `theta^(-2i/head_dim)`, i.e. `[x1 cos - x2 sin | x2 cos + x1 sin]` on the half split. `transformer._rope`:
  `x [cos | cos] + [x2 | x1] [-sin | sin]` with `_rope_tables`' `theta^(-i/half)`: the same rotation.
  `test_rope_is_hugging_faces_rotate_half` applies HF's `LlamaRotaryEmbedding` and `apply_rotary_pos_emb` to random
  int8 values at 40 positions: the integer RoPE is within 1 of it everywhere (Q14 tables and the final rounding).
  Neither checkpoint uses RoPE scaling (`rope_scaling = None`); the builder refuses any other RoPE type.
* **GQA.** HF's `repeat_kv` is `repeat_interleave`: query head `j` reads key/value head `j // (n_heads / n_kv)`, which
  is `transformer._attention`'s grouping (its comment says so; the model-level control below confirms it).
* **Qwen3 q/k norm.** HF applies `q_norm` to `q_proj(h)` viewed as `[B, T, heads, head_dim]`, before RoPE: the
  benchmark graph's order (q, k, v requantised, then "q norm", "k norm", then "rope" of q and k).
* **RMSNorm epsilon.** The integer norm has none; HF adds 1e-6 (Qwen3) or 1e-5 (TinyLlama) to the mean square, which
  is negligible against residual streams of order 1 and q/k vectors of order 10-100.

**Negative controls.** A model whose half-rotation RoPE equals the original's adjacent-pair rotation (q/k rows
permuted within each head, q/k norm weights too), or whose interleaved head grouping equals the original's tiled one
(q rows and o columns permuted between heads), builds the graph that a builder with the wrong RoPE pairing or the
wrong GQA grouping would build for the original. Against the original HF model (tiny models of Section 7, smoothed
recipe, 4 x 32 random tokens, 2 seeds per family):

| quantity | this builder | wrong GQA grouping | wrong RoPE pairing |
|---|---|---|---|
| logit cosine, mean | 0.992 - 0.995 | 0.18 - 0.28 | 0.52 - 0.78 |
| logit cosine, min over positions | 0.949 - 0.981 | -0.40 - -0.29 | -0.31 - 0.27 |
| top-1 agreement | 0.852 - 0.883 | 0.05 - 0.09 | 0.16 - 0.43 |
| block cosine, min over blocks and positions | 0.946 - 0.981 | -0.29 - -0.20 | -0.11 - 0.35 |

With HF's default weight std 0.02 the tiny models' attention is nearly uniform and the wrong RoPE pairing goes
unnoticed (Llama, seed 0: logit cosine 0.9994 right vs 0.9993 wrong); the tests therefore use std 0.15.

A third control covers the q/k-norm placement (Qwen3): a model whose q/k-norm gain of channel `j` is the original's
gain of its RoPE partner `j +- head_dim/2` (the graph a builder would give that put each gain on the wrong side of the
rotation) gives a logit cosine of 0.684 / 0.867 and top-1 0.344 / 0.492 (seeds 0 / 1), below the same thresholds
(`test_misplaced_qk_norm_gains_fail_the_thresholds`, seed 0).

### 3.2 Head equalisation in front of Qwen3's q/k norms (`head_eq`, default on)

The q norm divides each head's int8 values by `isqrt(floor(sum x^2 / head_dim))`, whose error is up to `1/rms` in
relative terms: a head whose int8 values have RMS 10 sees its scores scaled by up to 10% (and as much again from
the k norm). With one scale per tensor, heads with small q values get small integer RMS. Because the norm is
invariant to scaling a head's input, each head's rows of `W_q` (and `W_k`) are multiplied by
`beta_h = max_h' a_h' / a_h` (at most 256), `a_h` the `pct` percentile of the head's real outputs on the calibration
rows. The float function is unchanged (up to RMSNorm's epsilon); every head then uses the full int8 range, and the
small heads' weights get the full int8 resolution as well. On the tiny random models the heads have equal sizes and
the effect is nil (logit cosine 0.9942 with and 0.9942 without, seed 0; 0.9951 and 0.9945, seed 1); on real models it
is measured by the `--no-head-eq` cells of Section 9.

### 3.3 Lean evaluation and the chunked LM head

`lean_logits(graph, ids, all_positions=True, chunk=256)` runs the graph with the lean forward (`free=True`, no claim
kept, as `ppl_2048.py`) up to the LM head's input, then applies the LM head to `chunk` positions at a time and yields
the integer logits per chunk; the script reduces each chunk to NLL and argmax at once. At 2,048 positions a
151,936-entry vocabulary would be 2.5 GB of int64 logits (plus a float64 copy); a chunk of 256 is 311 MB. Every chunk
is the same integers as one pass (a weight op's columns depend on their own input column only; tested with chunks of
5 against one chunk, and against the plain forward at every prefix length). `all_positions` swaps the one "last
position" slice for the identity, which gives what one query per prefix would; a pruned graph refuses it.

### 3.4 `prune_last`

As `build_decoder(prune_last=True)`: the last block's q, attention output, residuals and MLP run at the last position
only (k and v at every position), through "last position" ops built with `calibrate_all` (the calibration keeps every
position), so the pruned graph has the unpruned graph's weights and multipliers. Its logits equal the unpruned
graph's last-position logits integer for integer (tested on both families at two lengths, also with `glu_exp`), and
its structure is the pruned benchmark graph's.

### 3.5 Opt-in: a per-token power-of-two exponent at the SwiGLU product (`glu_exp = E`)

The plan's outlier handling. With `glu_exp = E > 0`:

* the "swiglu" op computes `y = a u m` (int64; `a` the SiLU output, `u` the up projection, `m` the product's
  multiplier) and gives token `t` the smallest exponent `e_t` in `0 .. E` for which every
  `(y + 2^(29+e_t)) >> (30+e_t)` lies in +-127 (only the row's maximum and minimum are shifted to find it; the
  condition is monotone in `e`), then emits those int8 values: token `t`'s real scale is `s_p 2^(e_t)`;
* `s_p = top / (127 2^E)`, `top` the `pct_glu` percentile of the tokens' largest `|silu(gate) up|`: the largest
  calibration tokens take exponent `E`, every token below `top / 2^E` exponent 0; `E` is lowered (per block, recorded
  in `info["glu_emax"]`) until `|a u| m < 2^62` for all int8 `a, u`;
* the residual add after the down projection recomputes `e_t` from the same `a, u` (it reads them as inputs 3 and
  4; the verifier holds them) and adds `(z m_res + 2^(29-e_t)) >> (30-e_t)`, the round-half-up of
  `z m_res 2^(e_t) / 2^30`, clamped to `RES_MAX` (the shift absorbs `2^(e_t)`, so the product `z m_res` is no
  larger than in a plain residual add).

Both are deterministic integer cheap ops the verifier recomputes, so v0's checks and costs are unchanged but for the
recomputation (the product is computed twice, `T x d_ff` multiplies per block). The graph then differs from the
benchmark's in exactly two ops per block: the product's note ("swiglu, per-token exponent") and the residual add's
wiring and note ("residual add, per-token shift"); `matches_benchmark_graph` reports exactly these (tested). The
residual add is no `requant_fn`, so V1's P1 leaves the down projection **clear** (the plan's estimate: at most 9.6% of
Llama-2-7B's claims, about +0.3 GB at 2,048 tokens). A down projection with a bias is refused (the bias would not
follow the per-token scale). Tests: the exponent is the smallest clip-free one and the round-half-up is exact on rows
of very different sizes; the residual add equals `r + floor(z m 2^e / 2^30 + 1/2)` computed exactly in float64; the
graphs track the HF model within the same thresholds; v0 (C, Kpre) accepts and rejects a tampered down-projection
claim; V1 accepts and rejects the same tamper; pruning gives the unpruned logits.

On the tiny random models, which have no massive activations, the exponent changes little (smoothed recipe, 4 x 32
tokens): logit cosine mean 0.9926-0.9952 with `glu_exp = 8` against 0.9921-0.9951 without, top-1 0.844-0.922
against 0.852-0.883 (two seeds per family). `glu_exp = 8` and `12` gave identical logits: `s_p` changes by `2^4` and
every token's exponent by 4, so the per-token grids `s_p 2^(e_t)` coincide unless a token falls to exponent 0. `E`
therefore only bounds the dynamic range below the largest calibration token. Whether it rescues the real models is
measured by the `smooth-expE` cells of Section 9.

## 4. Why these choices

* **Same structure as the benchmark graph.** The paper's cost tables are measured on `build_decoder`'s graphs; with
  an empty structural diff, every cost number of a Qwen3 or TinyLlama row applies to the real-weight graph as is, and
  a real-weight run (F3a/F3b) only replaces "random weights" by "real weights" in the caption.
* **gamma folded, q/k norm gamma in the gain.** Folding RMSNorm's gamma into the next matrices keeps the norm op
  parameter-free, as for OPT. The q/k norm's gamma cannot be folded: RoPE mixes channel `j` with `j + head_dim/2` after
  the norm, so a per-channel factor before RoPE is not a per-channel factor after it. In the gain vector it is a
  public constant of a cheap op, and every RoPE pair keeps one real scale.
* **Pair norms before RoPE.** `_rope` clamps to +-127; a pair `(x, y)` rotates within the circle of radius
  `sqrt(x^2 + y^2)`, so a scale set from `max |x|` would clip after rotation by up to `sqrt(2)`.
* **SmoothQuant through the norm's gain vector, unchanged.** It was the fix for OPT (x1.17 -> x1.03 on OPT-125M,
  `F2_quality.md`); RMSNorm's integer op takes the same gain vector.

## 5. Interfaces

```python
from pvi.fullcheck import real_llama as rl
graph, info = rl.build_llama_from_hf(model, calib_ids, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
rl.matches_benchmark_graph(graph, "qwen3-0.6b")            # [] for the checkpoint of that shape
logits = rl.real_logits(graph, info, ids, all_positions=True, chunk=256)      # [B, T, V] float64
for start, z in rl.lean_logits(graph.with_constants_on("cuda")[0], ids.cuda(), chunk=256): ...
cos = rl.block_cosines(rl.residual_states(graph, info, ids), rl.hf_residual_states(model, ids))
```

`info` holds `logit_scale`, `s_residual`, `residual_real_max`, `embed_mult`, `embed_clip`, the knobs, the residual
tensors at the block boundaries (`embed`, `blocks`), `config` (the `DecoderConfig` of the model's shape) and
`glu_emax`.

## 6. The quality script

`experiments/9_quality/ppl_llama.py` reuses `ppl_2048.py`'s data (WikiText-2 raw joined with blank lines, tokenised
once; every non-overlapping `--seq`-token test window, `--windows` keeps the first ones), its windows, bootstrap and
peak-memory helpers. Calibration: `--calib-windows` (default 4) consecutive train windows of `--calib-seq` (default
`--seq`) tokens per draw. Reference: the checkpoint in fp32 with its LM head chunked as well (`--ref-dtype
float16|bfloat16` loads a second copy in that type; `--ref-cache` stores the NLL and argmaxes with their key). Cells:
`scalar` per `--gains` (no clipping, as `ppl_2048.py`'s scalar cell), `smooth` per `--alphas` (`--smooth-pct 99.9`,
`--pct 99.99`, `--pct-att 99.99`, `--pct-glu` default `--pct`), `smooth-expE` per `--glu-exps`. Per cell: perplexity,
ratio with a 95% bootstrap interval over windows, top-1 agreement, the per-block cosine of the residual stream
(mean and minimum over positions) on the first `--cos-windows` windows, `opt_builder.graph_report` (claim bound,
multipliers outside V1's P3), the digest, the structural diff on the first cell (`--structure`), build and evaluation
seconds, peak host and GPU memory. `--check` and `--protocol` are the exactness and protocol checks of Section 8.
Checkpoints are loaded as `common.load_opt` does (meta device, state dict assigned, safetensors or `weights_only`
`.bin`, so TinyLlama's `pytorch_model.bin` loads under torch < 2.6), with the rotary embedding rebuilt on the CPU.

## 7. Validation on the laptop (tiny random models, no download of weights)

`tests/test_real_llama.py` builds 2-block models with `d = 64`, head_dim 16, grouped-query attention (Llama: 4 query
heads over 2 key/value heads; Qwen3: 8 over 2, so the query width 128 differs from `d`, as in Qwen3-0.6B), random
RMSNorm and q/k-norm weights in [0.5, 1.5], weights of std 0.15, untied (Llama) and tied (Qwen3) heads; calibration 2 x
32 random tokens, evaluation 4 x 32.

| Test | Checks |
|---|---|
| `test_the_graph_is_the_benchmark_graph_of_its_shape` (x4) | empty diff, pruned and unpruned, both families; non-empty against the wrong pruning |
| `test_the_ladders_configs_are_the_checkpoints_shapes` | `config_of` of the published `config.json` shapes equals the new `CONFIGS`; parameter counts within 6% of 0.6B / 1.7B / 1.1B |
| `test_the_integer_forward_and_the_builder_are_deterministic` (x2) | two forwards give the same claims and logits; a rebuild gives the same digest, weights and logit scale |
| `test_lean_chunks_and_all_positions_give_the_same_integers` (x2) | chunks of 5 = one chunk = the plain forward at every prefix length |
| `test_prune_last_gives_the_unpruned_last_position_logits` (x2) | integer equality at lengths 32 and 9; one-column claims |
| `test_rope_is_hugging_faces_rotate_half` | within 1 of HF's `apply_rotary_pos_emb` |
| `test_the_logits_and_every_block_track_the_float_model` (x2) | smoothed: logit cosine mean >= 0.98, min >= 0.9, top-1 >= 0.75, every block's min cosine >= 0.9; scalar: mean >= 0.94 |
| `test_a_wrong_head_grouping_or_rope_pairing_fails_the_thresholds` (x4) | the negative controls of Section 3.1 give mean < 0.9 and top-1 < 0.5 |
| `test_misplaced_qk_norm_gains_fail_the_thresholds` | Qwen3: the q/k-norm gain control of Section 3.1 gives mean < 0.9 and top-1 < 0.5 |
| `test_v0_accepts_real_graphs_in_modes_c_and_kpre_and_rejects_a_tamper` (x4) | v0, lambda = 20, both modes, pruned and unpruned: 2 honest queries accepted; a tampered claim of the first q projection, the middle weight op and the LM head rejected |
| `test_the_per_token_exponent_is_the_smallest_clip_free_one` | Section 3.5 |
| `test_glu_exp_graphs_track_the_model_and_pass_v0_and_v1` (x2) | Section 3.5 |
| `test_v1_accepts_a_real_graph_and_rejects_a_tamper` (x2) | V1 (`cut_protocol.run_cut_query`, mode C) accepts, and rejects a tampered claim of the first q projection, the middle weight op and the LM head; every requantisation multiplier a scalar |

Thresholds (from the measurements behind the table of Section 3.1): the smoothed graphs measured logit cosine mean
0.992-0.995, minimum 0.949-0.981, top-1 0.852-0.883, block minimum 0.946-0.981 (two seeds per family); the scalar
graphs (gain 4, no clipping) 0.960-0.970 mean (re-measured in review: 0.9599 / 0.9648 Llama, 0.9703 / 0.9672
Qwen3, seeds 0 / 1; the smoothed and control numbers above reproduced to the stated digits). The thresholds sit
0.01-0.05 below the right builder and at least 0.12 above the wrong ones.

Result: **28 passed in 27.1 s** (`pytest tests/test_real_llama.py`, laptop, 2 threads, transformers 4.57.6) and
**28 passed in 26.3 s** under transformers 4.51.3 + tokenizers 0.21.4 (the versions `requirements-llm.txt` pins,
installed into a throw-away directory for the run). The existing tests that iterate over every `CONFIGS` entry pass
with the three new entries: `test_plans.py` **549 passed, 4 skipped** (718 s; its `CONFIGS` test,
`test_every_plan_meets_lambda_for_every_model_and_mode`, re-run in review: **14 passed**), `test_plan_auto.py` with
`test_real_weights.py` **15 passed** (169 s; `test_plan_auto.py` alone re-run in review: **11 passed**, 143 s),
`test_fullcheck.py` with `test_pruning.py` **76 passed, 3 skipped** (167 s).

The checkpoint loader (`ppl_llama.load_llama`) was run on local stand-ins of the three checkpoint formats -- a
TinyLlama-style fp32 `pytorch_model.bin` with per-layer `rotary_emb.inv_freq` keys, a Qwen3-style bf16
`model.safetensors` without `lm_head.weight` (tied), and a sharded safetensors index -- under both transformers
versions: the loaded models' logits equal the saved models' (largest difference 0 under 4.57.6, at most 2.7e-6
under 4.51.3), with the head tied where the configuration says so.

Script smoke runs on the laptop (`--smoke`: a 2-block random model with the checkpoint's real tokenizer and
vocabulary; only `config.json` and the tokenizer are downloaded):

* `--check` on Qwen3-0.6B-Base's and TinyLlama_v1.1's tokenizer/vocabulary: `CHECK PASSED` (for the smoothed graph
  and its `glu_exp = 8` variant: chunked = whole logits, last position = plain forward, pruned = unpruned; structural
  diff empty for the smoothed graph, pruned and unpruned, against the model's own shape). Its other branch -- a model
  whose shape is a `CONFIGS` entry's, diffed against that entry -- was exercised by registering the smoke model's
  shape as an entry (`structure_against` the entry, `CHECK PASSED`); a non-smoke model of another shape than its
  entry gives `CHECK FAILED`.
* 4 x 128 windows, Qwen3 vocabulary, calibration 2 x 128 train windows: the reference, a scalar cell and two smoothed
  cells run end to end in 4-6 s each, structural diff `[]` against the model's own shape (the perplexities of a random
  model are meaningless: about 3.4e5 for all cells).
* 2 x 2,048 windows, TinyLlama vocabulary: runs end to end (6-7 s per cell).
* The cluster commands' flag sets on the smoke model (review): the 16 x 128 ladder's (`--gains 2 4 8 --alphas 0.45
  0.55 0.65 --glu-exps 8 --structure --cos-windows 4 --ref-cache --out`, 4 windows): the reference and 9 cells; the
  Qwen3-4B pattern at 2,048 (`--ref-device cpu`, one window): the reference and 3 cells; the reduced-precision
  reference (`--graphs none --ref-dtype bfloat16|float16 --ref-device cpu`): the reference row only (a smoke run
  now converts the smoke model instead of downloading the checkpoint).
* `--protocol` on both vocabularies: `PROTOCOL PASSED` (unpruned and pruned; in each of C, Kpre and V1 two honest
  queries accepted and the three tampered claims rejected; structural diffs empty); 53 s / 45 s per graph with the
  Qwen3 vocabulary (the 151,936-row embedding and LM head commitments dominate), 17 s / 16 s with TinyLlama's
  (laptop, 2 threads; modes C and V1 share one commitment).

**A long-context finding on the tiny model.** At 2,048 tokens the per-position minimum block cosine of the tiny
random model fell to 0.31-0.38. Its cause is the attention's 8-bit probabilities, not the builder: by position, the
last block's mean cosine is 0.995 / 0.991 / 0.978 / 0.950 / 0.882 for positions 0-63 / 64-255 / 256-511 / 512-1023 /
1024-2047, exactly as the share of probability mass the rounded rows keep falls, 0.999 / 0.966 / 0.877 / 0.733 /
0.545 (a near-uniform row over more than 510 keys rounds every probability to 0). Random models attend almost
uniformly; real ones concentrate their attention, so the real effect is smaller, but it is F2a's `P_MAX` question
for these families too (Section 10). Re-measured in review on the first 2,048-token test window, calibrated on 4 x
2,048 train windows: last block 0.995 / 0.989 / 0.980 / 0.962 / 0.906 by the same position ranges, kept mass
0.998 / 0.964 / 0.884 / 0.759 / 0.586, smallest per-position cosine 0.266 -- the same picture.

## 8. Exactness and protocol checks on the real checkpoints (cluster)

`--check` (one 128-token window), for the smoothed graph and for its `glu_exp = 8` variant (whose per-token ops the
`smooth-expE` cells run on the GPU): the chunked lean logits equal one pass, the last position equals the plain
forward, the pruned graph's logits equal the unpruned graph's and, on the GPU, the GPU's integers equal the CPU's;
the smoothed graphs' structural diffs against `build_decoder(CONFIGS[name])` (pruned and unpruned) are empty, and
the checkpoint's shape is that entry's (`shape_vs_config == []`, else the check fails). `--protocol`: the
checkpoint's first 2 blocks (`n_layers=2`, then the final norm and the LM head; smoothed recipe, one 64-token train
window), unpruned and pruned, on a 64-token test window: the 2-block
benchmark structure; v0 in modes C and Kpre (lambda = 40) and V1 accept 2 honest queries each and reject a tampered
claim of the first q projection, the middle weight op and the LM head. Results: Section 9.

## 9. Results on the real checkpoints (cluster; placeholders)

Reference perplexities (fp32 unless noted), WikiText-2 test, `"\n\n".join`:

| Model | 16 x 128 | 2,048, every window | published |
|---|---:|---:|---|
| Qwen3-0.6B-Base | [CLUSTER] | [CLUSTER] | |
| TinyLlama_v1.1 | [CLUSTER] | [CLUSTER] | |
| Qwen3-1.7B-Base | [CLUSTER] | [CLUSTER] | |
| Qwen3-4B-Base | | [CLUSTER] (fp32, CPU); [CLUSTER] (fp16) | 7.90 (plan) |

Integer graphs (ratio to the reference, 95% bootstrap interval, top-1, smallest per-block mean cosine):

| Model, setting | scalar (best gain) | smooth alpha 0.45 / 0.55 / 0.65 | smooth, no head eq | smooth, pct_glu 100 | smooth-exp8 / exp12 |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base, 16 x 128 | [CLUSTER] | [CLUSTER] | [CLUSTER] | [CLUSTER] | [CLUSTER] |
| TinyLlama_v1.1, 16 x 128 | [CLUSTER] | [CLUSTER] | n/a | [CLUSTER] | [CLUSTER] |
| Qwen3-1.7B-Base, 16 x 128 | [CLUSTER] | [CLUSTER] | [CLUSTER] | | [CLUSTER] |
| Qwen3-0.6B-Base, 2,048 | [CLUSTER] | [CLUSTER] | | | [CLUSTER] |
| TinyLlama_v1.1, 2,048 | [CLUSTER] | [CLUSTER] | | | [CLUSTER] |
| Qwen3-1.7B-Base, 2,048 | [CLUSTER] | [CLUSTER] | | | [CLUSTER] |
| Qwen3-4B-Base, 2,048 | [CLUSTER] | [CLUSTER] | | | [CLUSTER] |

Structure (`--structure`, against the full `CONFIGS` entry): [CLUSTER] (expected `[]` for all four).
`--check`: [CLUSTER]. `--protocol`: [CLUSTER]. Build time, evaluation time and peak memory per cell: [CLUSTER].

Targets and kill criteria (plan): Qwen3 expected x1.03-1.06 if it quantises like OPT; Qwen3-4B within x1.10 by
Fri 23 Oct, else the measured number is reported with its attribution and the Qwen cost rows are labelled "shape
only"; TinyLlama within x1.15 by Sun 25 Oct (first result Sun 18), else Llama's quality is a stated, measured
limitation. Validation targets: per-block hidden-state cosine >= 0.99 (16 x 128), an empty structural diff, the fp16
reference of Qwen3-4B-Base at 2,048 near the published 7.90.

## 10. Limitations

* **Static per-tensor int8 activations.** The graph proves what it proves: one scale per activation tensor. If the
  massive activations at the down projection's input break the smoothed graph (as the literature reports for
  Llama-2-7B), `pct_glu` (clipping) or the opt-in `glu_exp` (Section 3.5, a structural change of two ops per block
  and a clear down projection under V1) are the measured remedies; the PrefixQuant-style fixed prefix is not
  implemented.
* **8-bit attention probabilities at 2,048 tokens** (Section 7): `s_raw = s_v / 255` follows
  `transformer._attention`'s `P = 255`; if F2a adopts `P_MAX = 2047`, this builder inherits the new attention op but
  its `s_raw` must use the new `P` (one line, marked in the code).
* **The embedding's multiplier** `c = round(max|E| / (127 s_res))` is `build_opt_from_hf`'s rule: for a ratio in
  `(k, k + 1/2)`, `k >= 1`, it rounds down and the embedding's largest entries clip at 127 (by up to x1.5 at
  `k = 1`). `info["embed_clip"]` (and each cell's `embed_clip` in the script's output) is that ratio over `c`; a value
  above 1 says by how much the largest entry clipped. Massive activations make `s_res` coarse, which can bring the
  ratio below 1 (`c = 1`: no clipping, but an embedding resolution of `s_res`); not measured on the real checkpoints
  yet -- the cluster rows show which case holds.
* **The integer square root** of the q/k norm has a relative error up to `1/rms` of the head's int8 values; head
  equalisation (Section 3.2) keeps every head at full range, but heads whose values are spread unevenly within the
  head still see it. RMSNorm's epsilon is not modelled (negligible at these magnitudes).
* **Root of trust (D6a).** `s_res` comes from the fp checkpoint's float hidden-state maximum, so a different BLAS or
  thread count can change it in the last bits and with it every residual multiplier; D6a's recipe (float64, snapping)
  applies to this builder as to OPT's.
* **Memory of a Qwen3-4B build** (not measured): the fp32 checkpoint (16 GB), the int8 graph (4 GB) and float64
  temporaries of the largest matrix (151,936 x 2,560 x 8 B = 3.1 GB for the LM head and the embedding) on the host;
  the fp32 reference does not fit the 2080 Ti (16 GB of weights) and runs on the Xeon's CPU (or `--ref-dtype
  float16` on the GPU).
* **V1.** Residual multipliers above `2^29` leave their op clear under V1's P3 (the smoke run's random model already
  had some at `2^29.3`; `graph_report` lists them per cell).
* **Llama-2-7B** (F2b-L parts 2-3) needs the Llama-2 licence [USER] and is not in the commands below; TinyLlama
  exercises the same Llama code path. The native CPU kernels for per-row multipliers (F2b-L's timing note) are not
  needed by this item: `residual_exp` is a torch op.

## 11. How to run

Laptop (tests and smoke; 2 threads):

```bash
cd code && PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest -q -o addopts= -p no:cacheprovider tests/test_real_llama.py
cd code && PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --smoke --check --threads 2 --chunk 48
cd code && PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --smoke --seq 128 --windows 4 \
    --calib-windows 2 --train-chars 200000 --device cpu --threads 2 --chunk 64 --alphas 0.5 0.55 --structure --cos-windows 2
```

Cluster (Linux Xeon node, 250 GB, RTX 2080 Ti; bash, `HF_HOME` already exported to node-local `/tmp`; every command
from `code/`; downloads: the four public checkpoints, about 1.2 / 4.4 / 3.4 / 8 GB, and the WikiText-2 parquets).
Memory to request: 64 GB for Qwen3-0.6B, TinyLlama and Qwen3-1.7B, 128 GB for Qwen3-4B (fp32 checkpoint, float64
temporaries, two random benchmark graphs in `--check`).

```bash
mkdir -p results/f2bq /tmp/$USER/f2bq
# 0. the tests on the node (CPU): expect "28 passed"
PYTHONPATH=src python -m pytest -q -o addopts= -p no:cacheprovider tests/test_real_llama.py
# 1. exactness on the real weights (GPU node): expect "CHECK PASSED"; in the JSON, "shape_vs_config": [],
#    "structure_against" the config name, "gpu_equals_cpu": true in both the "smooth" and the "smooth-exp8" entry,
#    and "structure_diff": [] and "pruned_structure_diff": [] in the "smooth" entry
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --check --threads 8
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model TinyLlama/TinyLlama_v1.1 --check --threads 8
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-1.7B-Base --check --threads 8
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-4B-Base --check --threads 8
# 2. 2-block real graphs through v0 (C, Kpre) and V1: expect "PROTOCOL PASSED"
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --protocol --threads 8
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model TinyLlama/TinyLlama_v1.1 --protocol --threads 8
# 3. week 1, 16 x 128: Qwen3-0.6B-Base and TinyLlama side by side, then Qwen3-1.7B-Base (main cells, then ablations)
for M in Qwen/Qwen3-0.6B-Base TinyLlama/TinyLlama_v1.1 Qwen/Qwen3-1.7B-Base; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 128 --windows 16 --calib-windows 4 \
    --device cuda --threads 8 --gains 2 4 8 --alphas 0.45 0.55 0.65 --glu-exps 8 --structure --cos-windows 4 \
    --ref-cache /tmp/$USER/f2bq/ref_${S}_128.pt --out results/f2bq/${S}_128.jsonl
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 128 --windows 16 --calib-windows 4 \
    --device cuda --threads 8 --graphs smooth --alphas 0.55 --pct-glu 100 --glu-exps 8 --cos-windows 4 \
    --ref-cache /tmp/$USER/f2bq/ref_${S}_128.pt --out results/f2bq/${S}_128_pctglu100.jsonl
done
for M in Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 128 --windows 16 --calib-windows 4 \
    --device cuda --threads 8 --graphs smooth --alphas 0.55 --no-head-eq --cos-windows 4 \
    --ref-cache /tmp/$USER/f2bq/ref_${S}_128.pt --out results/f2bq/${S}_128_noheadeq.jsonl
done
# 4. 2,048 tokens, every test window, on the 2080 Ti (calibration on the CPU)
for M in Qwen/Qwen3-0.6B-Base TinyLlama/TinyLlama_v1.1 Qwen/Qwen3-1.7B-Base; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 2048 --calib-windows 4 --device cuda \
    --threads 8 --gains 4 --alphas 0.45 0.55 0.65 --glu-exps 8 --cos-windows 1 \
    --ref-cache /tmp/$USER/f2bq/ref_${S}_2048.pt --out results/f2bq/${S}_2048.jsonl
done
# 5. Qwen3-4B-Base at 2,048: the fp32 reference on the CPU (16 GB of weights do not fit the 2080 Ti), then an fp16
#    reference on the GPU for the published 7.90
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-4B-Base --seq 2048 --calib-windows 4 \
  --device cuda --ref-device cpu --threads 8 --gains 4 --alphas 0.45 0.55 0.65 --glu-exps 8 --cos-windows 1 \
  --structure --ref-cache /tmp/$USER/f2bq/ref_q4_2048.pt --out results/f2bq/Qwen3-4B-Base_2048.jsonl
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-4B-Base --seq 2048 --graphs none \
  --ref-dtype float16 --ref-device cuda --threads 8 --ref-cache /tmp/$USER/f2bq/ref_q4_2048_fp16.pt \
  --out results/f2bq/Qwen3-4B-Base_2048_fp16ref.jsonl
```

Each JSON line of `--out` is one cell; `jq -c '{cell, alpha, gain, ppl, ratio, ratio_ci95, top1, block_cos_mean_min,
structure_diff}' results/f2bq/*.jsonl` gives the table of Section 9.
