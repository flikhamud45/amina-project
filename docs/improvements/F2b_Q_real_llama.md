# F2b-Q (and F2b-L's laptop part). Real-weight integer Qwen3 and Llama decoders

Plan 2 items **F2b-Q** (must: a real-weight builder for RMSNorm / RoPE / SwiGLU / GQA models, Qwen3 first) and the
laptop part of **F2b-L** (should, rank 6: TinyLlama-1.1B on the same builder), `SP2027_PLAN2.md` lines 435-486.
Status: **implemented; validated on the laptop on tiny randomly initialised Hugging Face models; the real-checkpoint
runs (Qwen3-0.6B/1.7B/4B-Base, TinyLlama-1.1B) are cluster commands whose numbers are placeholders in Section 9.**
The opt-in per-token exponent at the SwiGLU product (the plan's outlier handling, task item 5) is implemented and
tested for exactness and the protocol, off by default, and not yet measured on a real model. The PrefixQuant-style
fixed prefix (the plan's cheaper alternative; `prefix_tokens=`, `--prefix`) is implemented and tested for exactness
and the protocol, off by default, with its cluster numbers as placeholders (Section 12); it answers the cluster
diagnosis of job 1009531 (TinyLlama's zero BOS row, Qwen3's massive activation at the first token). The op-level
attribution (`llama_attrib.py`, Section 13) found the remaining first-block loss of Qwen3 on the laptop: the k norm's
int8 range is set by gains of 96.5 on channels whose q gain is about 0; the q/k balance per RoPE pair
(`qk_balance`, on by default for Qwen3, constants only) removes it; its cluster numbers are placeholders (13.7).

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
| `code/tests/test_real_llama.py` (new) | 28 tests on tiny random Llama and Qwen3 models, no download (Section 7); 24 more for the fixed prefix (Section 12.5) |
| `code/experiments/9_quality/ppl_llama.py` (new) | perplexity at 16 x 128 and at 2,048 tokens, fp32 reference, scalar and smoothed graphs, top-1, per-block cosine, `--check`, `--protocol`, `--smoke`; `--prefix` (Section 12) |
| `code/experiments/9_quality/llama_diag.py` | the per-block, per-position diagnosis of job 1009531; `--prefix` (Section 12) |
| `real_llama.py`, the fixed prefix | `build_llama_from_hf(..., prefix_tokens=)`, `prefix_attention`, `prefix_kv`, `prefixquant_tokens`; `hf_residual_states(..., prefix=)` (Section 12) |
| `code/experiments/9_quality/llama_attrib.py` (new) | the op-level attribution: every tensor of a block against the checkpoint's activation, teacher-forced or accumulated (Section 13) |
| `real_llama.py`, the attribution's map and the q/k balance | `ROLES`; `info["roles"]`, `info["scales"]`, `info["head_beta"]`, `info["qk_balance_c"]`; `build_llama_from_hf(..., qk_balance=True)` (Section 13.4) |
| `code/tests/test_llama_attrib.py` (new) | 44 tests: the map, the runner, fidelity op by op, broken builders caught at their op, Qwen3-0.6B's real q/k-norm gains and the balance (Section 13.3) |

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
`glu_emax`; for the attribution (Section 13) also `roles` (per block `{role: tensor name}`), `scales` (every
tensor's real scale), `head_beta` and `qk_balance_c`.

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
  Llama-2-7B), `pct_glu` (clipping), the opt-in `glu_exp` (Section 3.5, a structural change of two ops per block
  and a clear down projection under V1) and the opt-in fixed prefix (Section 12, no structural change) are the
  remedies the cluster rows measure.
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
structure_diff}' results/f2bq/*.jsonl` gives the table of Section 9. The prefix's commands are in Section 12.7.

## 12. Opt-in: a fixed public prefix (PrefixQuant-style, `prefix_tokens`)

The plan's cheaper alternative to the per-token exponent: "a PrefixQuant-style fixed prefix whose K/V become public
constants in the graph digest. It needs no new op kind and leaves V1 unaffected." Status: **implemented, tested on
the laptop (tiny random models), off by default; the real-checkpoint numbers are cluster placeholders (12.6).**

### 12.1 Why: the cluster diagnosis (job 1009531, `docs/diag_1009531.txt`)

The integer graphs pass every exactness and protocol check on the real checkpoints, but quality collapses
(Qwen3-0.6B-Base at 16 x 128: x28 with the scalar gain, x72-77 smoothed, x6-9 with the per-token exponent).
`llama_diag.py` on the first 128-token test window (calibration: 4 x 128 train windows) shows two causes:

* **TinyLlama_v1.1: the BOS row is zero.** The integer residual at position 0 (`<s>`) has cosine 0.000 in every
  block, with both recipes: it is exactly zero. `s_res = 1.41e-4` is set by the embedding (`max|E| / (127 x 8)`,
  `max|E| = 0.143`; the fp residual maximum, 214, would allow 5.1e-5), the embedding multiplier is 8, so the int8
  embedding's step is `8 s_res = 1.13e-3`, and every entry of `<s>`'s row lies below half a step. A zero row stays
  zero through every block (RMSNorm of 0 is 0, so q = k = v = 0; the attention output and SiLU(0) are 0; no biases):
  the sink every head attends to has key 0 (score 0 instead of the row's maximum), so the attention sink never
  forms, and the other positions degrade block by block (mean cosine over positions 1..: 0.999 after the embedding,
  0.864 after block 1, 0.52-0.66 after blocks 12-22 with the scalar gain; 0.976, then down to 0.31, smoothed). The
  fp stream also carries a massive activation at position 2 (97-104, channel 1619, from block 1 on): the window's
  first `\n` (`<s> ▁ \n \n = Robert ...`), the delimiter outlier PrefixQuant reports for Llama-2-7B.
* **Qwen3-0.6B-Base: a massive activation at the first token.** From block 3 on, the fp stream holds about 6,800 at
  position 0 (channel 35; the window's first token is `\n\n`, Qwen's tokenizer adds no BOS). It sets `s_res` (fp
  maximum 6,987 x 1.25 / `RES_MAX` = 2.08e-3), and every static scale is calibrated on windows whose position 0
  carries it. The first block already loses ordinary tokens (mean cosine 0.865, minimum 0.45 with the scalar gain;
  0.889 smoothed) while position 0 itself is reproduced (0.98-1.00 through block 27 with the scalar gain): the known
  failure of
  static per-tensor W8A8 on Llama-family models. The smallest per-block mean cosine is 0.625 (scalar) and 0.544
  (smoothed).

Both failures sit at positions a fixed prefix takes over: the initial token (the attention sink, where Qwen's massive
activation forms), the BOS (TinyLlama's zero row), and the first occurrences of a few delimiters (TinyLlama's `\n`).
PrefixQuant (M. Chen, Y. Liu, J. Wang, Y. Bin, W. Shao, P. Luo, "PrefixQuant: Eliminating Outliers by Prefixed Tokens
for Large Language Models Quantization", arXiv:2410.05265, 2024) finds these outlier tokens few and predictable,
computes the KV cache of a fixed prefix holding them once ("the prefixed tokens in the KV cache remain in full
precision"), and then calibrates static per-tensor scales that no longer see them.

### 12.2 What and how

`build_llama_from_hf(model, calib_ids, ..., prefix_tokens=None)`, `prefix_tokens` = `P` token ids (`real_llama.py`):

1. **The prefix's keys and values come from the checkpoint** (`prefix_kv`): one fp forward pass over the prefix alone
   with `use_cache=True`; per layer, Hugging Face's cached keys (after the k norm for Qwen3, and after RoPE at the
   positions `0 .. P-1`) and values, `[1, P, Hkv head_dim]`. Causal attention makes them independent of the query.
2. **Every scale is calibrated on the query positions only.** The calibration windows run after the prefix:
   `_hidden_max` runs the fp checkpoint on `[prefix | window]` and takes the residual maximum behind `s_res` over the
   window's positions (the prefix positions' maximum is recorded as `prefix_hidden_max` and sets nothing); the integer
   calibration pass runs the prefixed graph on the windows, whose tensors hold the query positions only, so no
   percentile (q/k/v/gate/up, the q/k norms, the attention output, the SwiGLU product), no SmoothQuant statistic and
   no head equalisation sees a prefix position. `m_o` is calibrated with the prefix in the attention.
3. **The constants are int8 on the layer's own scales.** Once a layer's k and v are calibrated (q/k norm and RoPE
   included), the prefix's keys and values become `clamp(round(real / s), +-127)` int8 with `s = s_k`, `s_v`: the real
   scales of that layer's computed k and v. `info["prefix_clip"]` records `max |real| / (127 s)` per layer (above 1
   the prefix's largest entries clip).
4. **The attention op.** `prefix_attention(q, k, v, kp, vp, m_s, m_o, H, Hkv, dh, pmax)` puts `kp, vp` in front of
   `k, v` and calls `transformer._attention`, whose `Tq` query rows are the queries of the *last* `Tq` of the `P + T`
   positions: every query attends to the whole prefix, and causally to the query's own positions. It is the
   convention `prune_last` already relies on (one query against every key); the fused GPU kernel, the native CPU
   kernel and both torch paths implement it. `kp, vp` are default arguments of the op's function, next to `m_s` and
   `m_o`: `IntGraph.digest()` hashes them (dtype, shape and bytes), `with_constants_on(device)` copies them once, and
   the verifier recomputes the op with them.
5. **RoPE.** The query's q and k are rotated at the positions `P, P+1, ...` (`_rope(x, theta, offset=P)`; a pruned q
   at `T - Tq + P`, `prune_last`'s offset shifted by `P`); the prefix's keys were rotated at `0 .. P-1` offline.
6. **Recorded in the graph.** `graph.meta = {"prefix_len": P, "prefix_tokens": [...]}`, which the digest also binds;
   `info` adds `prefix_kv` (the int8 constants), `prefix_scales`, `prefix_clip` and `prefix_hidden_max`.

Unchanged: the op kinds, names, wiring and notes, the weight ops and their shapes and biases, and the claim columns,
so `matches_benchmark_graph` is empty, the costs are the benchmark graph's and V1 is unaffected (the attention stays
a cheap op; the prefix adds no claim). With `prefix_tokens=None` the graph is today's byte for byte: the digests of
the ten tiny test graphs (both families; scalar and smoothed, pruned and unpruned; `glu_exp = 8`) were recorded
before the change and are identical after it.

**Who computes the constants.** Anyone with the checkpoint and the public calibration recipe recomputes them, as
every other constant of the graph; they come from a float forward pass, so D6a's root-of-trust note applies to them
as to `s_res`. **The BOS.** The prefix takes the BOS's place: with `--prefix` the scripts tokenise the text without
the tokenizer's special tokens, so no window holds a BOS and no query token needs its embedding row; a deployment
does the same (the public prefix starts every sequence, the client's prompt follows it).

### 12.3 Choosing the prefix: two mechanisms

`prefixquant_tokens(model, calib_ids, eta=64)` is PrefixQuant's rule (Sec. 4 of the paper): at each down
projection's input, token `t` is an outlier token when its largest `|value|` over channels is more than
`eta_1 = 64` times the median, over the window, of that statistic; `o` is the ceiling of the largest, over blocks,
mean number of outlier tokens per window; the prefix is the `o - 1` token ids most often found among the outlier
tokens (a window's first position, the initial-token outlier every window has, is not counted), then the BOS
("we also include [BOS] as the last prefixed token for all models"). The BOS is the configuration's
`bos_token_id`: `<s>` (1) for TinyLlama, `<|endoftext|>` (151643) for Qwen3, the token Qwen's pretraining puts
between documents. PrefixQuant's own prefixes: Llama-2-7B `. \n [BOS]`, Llama-3-8B `[BOS]`, Qwen-2-7B `[BOS]`,
Mistral-7B-v0.3 `\n . to [BOS]`.

The scripts' `--prefix SPEC` picks one of two mechanisms (or explicit ids `a,b,...`):

* **`--prefix bos`, the smaller fix: the BOS alone (`P = 1`).** For TinyLlama it removes the zero BOS row: the sink
  is `<s>`'s keys and values computed in float, and no query token needs `<s>`'s embedding. For Qwen3 it moves the
  initial-token position out of the windows (a window's first token is now at position 1), so the massive activation
  is expected to form at the prefix, outside every scale. A lighter fix for the zero row alone does not exist within
  the structure: a representable `<s>` row needs a per-row embedding scale (a cheap op reading the token ids, a wiring
  change against the benchmark graph) or a finer embedding scale that clips the largest entries. It is therefore the
  prefix mechanism with `P = 1`, no separate code path.
* **`--prefix` (= `auto`), PrefixQuant's rule:** also prefixes the frequent delimiter outliers, which `bos` leaves in
  the windows (expected for TinyLlama: `\n`, id 13, the position-2 outlier of the diagnosed window). On a model whose
  only outlier is the initial token it gives `[bos]`, the same as `bos`.

Expectation, to be settled by the cluster rows (12.6): TinyLlama needs at least `bos` (the zero sink) and probably
`auto` (the `\n` outlier); Qwen3 needs the initial-token prefix (`bos`, PrefixQuant's Qwen-2-7B choice), `auto`
adding tokens only where delimiters spike at the down projection's input. Which mechanism each model needs is read
off rows 12.6 (a) and (b): `bos` and `auto` against no prefix.

### 12.4 Evaluation with a prefix (`ppl_llama.py`, `llama_diag.py`)

* **Scored tokens.** With `--prefix`, the perplexity is scored on the windows' own tokens only, the same targets as
  a run without a prefix (positions `0 .. T-2` predicting tokens `1 .. T-1` of each window): the prefix is not part
  of the evaluated text, and its positions have no logits in the integer graph.
* **Fair reference.** The fp32 reference runs on the same `[prefix | window]` (its prefix positions dropped); the
  per-block cosines compare against `hf_residual_states(model, w, prefix=...)`, the same.
* **Records.** Every JSON line holds `prefix_tokens`, `prefix_len`, `prefix_choice` (rule; for `auto` `o`, the
  outliers per block, the outlier tokens with counts; the decoded text), `prefix_scoring` (the sentence above),
  `joint_len` and `max_pos`; each cell adds `prefix_clip`, `prefix_hidden_max` and `residual_real_max`.
* **One prefix per run**, chosen on the first draw's calibration windows before the reference; `--ref-cache` keys
  include it. `--check` and `--protocol` honour `--prefix` (exactness, GPU = CPU, structure, v0 and V1 with the
  prefix).
* **Windows.** Tokenised without the special tokens, TinyLlama's windows start one token later than without a prefix
  (Qwen's are unchanged); each ratio is against its own reference.
* `llama_diag.py --prefix [SPEC]`: positions are the window's own; it adds the prefix's tokens and choice, the fp
  stream's largest `|value|` at the prefix positions per block (where the massive activation should now sit), the
  prefix's reach `max |real| / (127 s)` per layer, and the integer stream's entries at `+-RES_MAX` (which its
  docstring promised and the earlier version did not compute).

### 12.5 Validation on the laptop (tiny random models, no download of weights)

`tests/test_real_llama.py`: **52 passed in 53.8 s** (the 28 of Section 7 and 24 for the prefix; laptop, 2 threads,
transformers 4.57.6); `tests/test_prob_max.py`, which builds real-weight graphs too: **28 passed**. The prefix tests
use `PREFIX = (5, 17)` (`P = 2`) unless stated:

| Test | Checks |
|---|---|
| `test_a_prefixed_graph_is_the_benchmark_graph_of_its_shape` (x4) | empty structural diff, pruned and unpruned, both families; the same weight ops and claim columns as without a prefix at 1, 9 and 32 tokens; `meta`; per layer two int8 constants `[1, P, Hkv head_dim]`, the attention op's default arguments |
| `test_the_prefixed_forward_is_the_graph_on_prefix_and_query_with_constant_keys_and_values` (x4: `P` = 1, 2) | **the exact equality** below; and the check can fail: the query's RoPE without the offset gives other integers |
| `test_prune_last_with_a_prefix_gives_the_unpruned_last_position_logits` (x2) | integer equality at 32 and 9 tokens; one-column claims; equal to the glued graph's last logits on `[prefix | x]` |
| `test_a_prefixed_graph_is_deterministic_and_independent_of_the_attention_path` (x2) | two forwards equal; a rebuild gives the same digest, meta and constants; the int64 attention path (int64 scores and gaps, last-rows mask) gives the int32 path's claims and logits |
| `test_the_prefix_constants_are_the_checkpoints_keys_and_values_on_the_layers_scales` (x2) | the constants are `clamp(round(HF cache / s))`; `s_k`, `s_v` are the scales of the query's computed k and v (dequantised against HF's keys and values of `[prefix | x]`: norm ratio within 0.95-1.05, relative error <= 0.15; measured 0.990-1.017 and 0.018-0.094, two seeds per family) |
| `test_prefixed_logits_and_blocks_track_the_float_model_on_prefix_and_query` (x2) | Section 7's thresholds against HF on `[prefix | x]`: smoothed logit cosine mean >= 0.98, min >= 0.9, top-1 >= 0.75, every block's min >= 0.9; scalar mean >= 0.94 |
| `test_the_residual_scale_ignores_the_prefix_positions` | a prefix token whose embedding is 40x larger: `_hidden_max` returns the windows' own maximum (equal to the HF states' maximum over the window positions), the prefix's is larger and is only recorded; the build's `residual_real_max` is the windows' x 1.25 |
| `test_the_digest_binds_the_prefix_and_v0_rejects_a_prover_with_other_constants` (x2) | four digests differ (no prefix, (5, 17), (5, 18), (17, 5)); the same function with layer 0's prefix keys (values) negated: another digest, and v0 (mode C) rejects a prover that computes with them under the honest graph's commitments (it accepts the honest prover) |
| `test_v0_and_v1_accept_prefixed_real_graphs_and_reject_a_tamper` (x4) | v0 (C, Kpre, lambda = 20) and V1, pruned and unpruned: honest queries accepted; a tampered claim of the first q projection, the middle weight op and the LM head rejected |
| `test_prefixquant_selects_the_frequent_outlier_tokens_then_the_bos` | the rule on synthetic maxima (`o = 3` -> `[7, 9, bos]`; one outlier below `eta` -> `o = 2`, `[7, bos]`; the BOS and the first position not counted); a random model has no outliers: `[bos]`; Qwen3Config's default has no `bos_token_id` (an error; the checkpoints have 151643) |

**The exact equality.** For a query `x [B, T]`, the prefixed graph's integers on `x` -- every weight op's claims,
the residual stream at every block boundary and the logits at every position -- equal, integer for integer, those of
the *glued* graph on `[prefix | x]` at `x`'s positions `P .. P+T-1`. The glued graph has the same ops, weights and
constants, but RoPE at the joint sequence's own positions (no offset) and plain causal attention
(`transformer._attention`) over the joint sequence whose first `P` keys and values are replaced by the int8
constants (the prefix positions' other values are computed from the prefix tokens and feed only the replaced keys and
values). So the prefixed graph is the benchmark graph run on prefix + prompt with the prefix's K/V fixed: the RoPE
positions, the causal mask and the GQA grouping line up.

Fidelity on the tiny models (smoothed; two seeds per family; prefixes `(5,)` and `(5, 17)`; against HF on
`[prefix | x]`): logit cosine mean 0.992-0.996, minimum 0.907-0.986, top-1 0.828-0.961, block minimum 0.935-0.986;
scalar gain 4: mean 0.960-0.974 (without a prefix, Section 7: 0.992-0.995, 0.949-0.981, 0.852-0.883, 0.946-0.981;
0.960-0.970). Random models have no massive activations, so this shows only that the prefix costs no fidelity; the
gain is the real checkpoints' question.

Script smoke runs (`--smoke`: 2-block random models with the real tokenizers and vocabularies; offline):
`llama_diag.py --prefix` (TinyLlama vocabulary; `auto` gives `[1]`, `o = 0`: a random model has no outliers) and
`--prefix bos` (Qwen vocabulary: `[151643]`, `<|endoftext|>`); `ppl_llama.py --prefix` at 4 x 128 (Qwen vocabulary:
the reference, scalar, smooth and smooth-exp8 cells, structural diff `[]`, every prefix field in the records) and
`--prefix bos` at 2 x 2,048 (TinyLlama vocabulary); `--check --prefix` (Qwen vocabulary): `CHECK PASSED` (chunked =
whole, last = forward, pruned = unpruned for smooth and smooth-exp8; structural diffs `[]`); `--protocol --prefix`
(TinyLlama vocabulary): `PROTOCOL PASSED` (C, Kpre and V1, unpruned and pruned, two honest queries accepted, three
tampers rejected; 17 s per graph). A run without `--prefix` gives the records of before, plus `prefix_tokens: null`,
`prefix_len: 0` and each cell's `residual_real_max` (`llama_diag.py`: plus `int_clip` per block).

### 12.6 Results on the real checkpoints (jobs 1009796 and 1009797; graphs before `qk_balance`)

These graphs predate Section 13's `qk_balance` (they are what `--no-qk-balance` builds). Qwen3's tokenizer adds no
BOS, so PrefixQuant's rule (`auto`) and `bos` choose the same prefix there (`<|endoftext|>`, one token), and the two
columns coincide.

(a) The diagnosis (`llama_diag.py`, first 128-token test window, job 1009796 for the prefixes; "none" is job 1009531):
the smallest per-block mean cosine / the last block's mean cosine / the smallest mean over positions 1..:

| Model, recipe | no prefix | `--prefix` (auto) | `--prefix bos` |
|---|---|---|---|
| Qwen3-0.6B-Base, scalar 4 | 0.625 / 0.706 / 0.622 | 0.685 / 0.685 / 0.691 | as auto |
| Qwen3-0.6B-Base, smooth 0.55 | 0.544 / 0.735 / 0.545 | 0.864 / 0.923 / 0.866 | as auto |
| TinyLlama_v1.1, scalar 4 | 0.513 / 0.658 / 0.517 (position 0: 0.000) | 0.731 / 0.839 / 0.731 | 0.712 / 0.827 / 0.716 |
| TinyLlama_v1.1, smooth 0.55 | 0.309 / 0.578 / 0.311 (position 0: 0.000) | 0.830 / 0.916 / 0.829 | 0.063 / 0.437 / 0.056 |
| prefix chosen (`auto`: `o`, tokens) | | Qwen3: `o` = 1, `<|endoftext|>`; TinyLlama: `o` = 2, `\n`, `<s>` (outlier tokens `\n` 7x, `.` 2x) | `<|endoftext|>`; `<s>` |
| fp max at the prefix / in the window (last 4 blocks) | 6,810 in the window (Qwen3) | Qwen3 6,843 / 457; TinyLlama 86 / 46 | TinyLlama 5 / 103 |
| `s_res`; prefix `k_clip` / `v_clip` (max over layers) | 2.08e-3 (Qwen3), 1.41e-4 (TinyLlama) | Qwen3 3.21e-4, 0.94 / 1.24 (scalar), 1.92 / 2.98 (smooth); TinyLlama 1.41e-4, 0.79 / 0.92 | TinyLlama 0.80 / 1.18 |

The prefix does what PrefixQuant says on the fp side: Qwen3's massive activation moves to the prefix (6,843 there,
457 in the window), `s_res` falls 6.5x, and the smoothed stream's smallest per-block cosine rises from 0.544 to
0.864. TinyLlama needs both tokens: `\n` takes its delimiter outlier (window maximum 103 -> 46), and `<s>` alone
leaves the `\n` outlier in the window, where the smoothed graph collapses (0.063). The prefix keys of Qwen3's
smoothed graph clip (up to 1.92 / 2.98 of their layer's range).

(b) Perplexity ratio to the fp32 reference with the same prefix (95% bootstrap interval; top-1; smallest per-block
mean cosine), best cell of each kind; "no prefix" is Section 9's:

| Model, setting | prefix | scalar (best of 2/4/8) | smooth (best of 0.45/0.55/0.65) | smooth-exp8 (best alpha) |
|---|---|---|---|---|
| Qwen3-0.6B-Base, 16 x 128 | none | x28 | x72-77 | x6-9 |
| Qwen3-0.6B-Base, 16 x 128 | auto = bos | x23.2 [17.1, 31.7] (gain 4) | x2.11 [1.94, 2.30], 0.538 (0.45) | x2.05 [1.89, 2.23], 0.534 (0.45) |
| TinyLlama_v1.1, 16 x 128 | auto | x1.83 [1.65, 2.06] (gain 8) | **x1.15 [1.10, 1.24], 0.789 (0.65)** | x1.18 [1.10, 1.30], 0.814 (0.45) |
| TinyLlama_v1.1, 16 x 128 | bos | x3.19 [2.52, 4.20] (gain 4) | x4.97 [2.91, 8.79] (0.55) | x3.21 [2.06, 5.21] (0.45) |
| Qwen3-1.7B-Base, 16 x 128 | auto = bos | x13.3 [10.4, 16.5] (gain 8) | x1.72 [1.60, 1.84], 0.612 (0.45) | x1.44 [1.35, 1.54], 0.642 (0.45) |
| Qwen3-0.6B-Base, 2,048 (146 windows) | auto | x95.9 [90.4, 101.9] (gain 8) | x2.78 [2.72, 2.84], 0.557 (0.65) | x2.70 [2.64, 2.77], 0.561 (0.65) |
| Qwen3-0.6B-Base, 2,048 | bos | as auto | as auto | as auto |
| TinyLlama_v1.1, 2,048 | auto | x1.74 [1.71, 1.77] (gain 8) | **x1.56 [1.51, 1.62], 0.723 (0.65)** | x2.11 [2.02, 2.21], 0.642 (0.45) |
| TinyLlama_v1.1, 2,048 | bos | x2.50 [2.46, 2.55] (gain 4) | x34.9 [33.0, 36.9] (0.65) | x10.9 [10.0, 12.0] (0.45) |
| Qwen3-1.7B-Base, 2,048 | auto = bos | x26.5 [24.9, 28.4] (gain 4) | x1.72 [1.70, 1.74], 0.668 (0.55) | **x1.45 [1.44, 1.46], 0.708 (0.45)** |

The fp32 references with the prefix: Qwen3-0.6B-Base 29.24 (16 x 128) and 12.71 (2,048); Qwen3-1.7B-Base 20.64;
TinyLlama 13.78 with `auto` and 169.2 with `bos` alone. The last is not plausible for a 1.1B model on WikiText-2
and is not explained yet (the same code gives 13.78 with `[\n, <s>]`); the `bos` row of TinyLlama is therefore not
used. At 2,048 tokens: Qwen3-1.7B-Base 9.42, TinyLlama 7.70 (`auto`) and 9.22 (`bos`). Job 1009797 completed every
cell except Qwen3-1.7B's `bos` repeat at 2,048 tokens, which was cancelled (12 Oct): Qwen3's `bos` prefix is its
`auto` prefix (the same token), and the 0.6B rows show the two runs give identical numbers. In the 2,048-token rows
the third number is the top-1 agreement, as in the others.

**At 2,048 tokens TinyLlama is much worse than at 128** (x1.56 against x1.15, smoothed, alpha 0.65), and two trends
differ from the short windows: the ratio falls steadily with alpha (x1.87 / x1.78 / x1.56 at 0.45 / 0.55 / 0.65), so
the best alpha may lie above the ladder, and the per-token exponent makes it worse (x2.11-2.46), which the 128-token
rows did not show (x1.18). Qwen3-1.7B-Base (before `qk_balance`) is the same at both lengths (x1.45 against x1.44
with the exponent). For F2b-L the TinyLlama target (x1.15) is met at 16 x 128 and missed at 2,048 tokens; the
plan's kill rule (Sun 25 Oct) does not name the length; the proposal is to decide on the 2,048-token number, the
protocol of every OPT row (F2a) [team decision].

Reading. The prefix turns Qwen3 from collapse (x72) into a damaged model (x2.1, and x1.44 for Qwen3-1.7B with the
per-token exponent), and brings TinyLlama to x1.15, the plan's TinyLlama target, with the smoothed graph and no
exponent. Neither Qwen3 model is near x1.05. The remaining Qwen3 loss starts in the first block (Section 13: the k
norm), which the prefix does not touch; Section 13's `qk_balance` is the next step, and the commands of 13.6 rerun
these cells with it.

### 12.7 Cluster commands (2080 Ti node; bash, from `code/`, `HF_HOME` on node-local `/tmp`)

Memory as Section 11 (64 GB for these three models). `--glu-exps 8` adds the `smooth-exp8` cells to the `smooth`
cells of the same run, so each line below is "with and without the per-token exponent" at once.

```bash
mkdir -p results/f2bq_prefix /tmp/$USER/f2bq
# 0. the tests on the node (CPU): expect "52 passed"
PYTHONPATH=src python -m pytest -q -o addopts= -p no:cacheprovider tests/test_real_llama.py
# 1. exactness and the protocol with the prefix on the real weights: expect "CHECK PASSED" (in the JSON
#    "gpu_equals_cpu": true in both entries, "structure_diff": [] and "pruned_structure_diff": [] in "smooth") and
#    "PROTOCOL PASSED"
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --check --threads 8 --prefix
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model TinyLlama/TinyLlama_v1.1 --check --threads 8 --prefix
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --protocol --threads 8 --prefix
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model TinyLlama/TinyLlama_v1.1 --protocol --threads 8 --prefix
# 2. the diagnosis without a prefix, with PrefixQuant's prefix, and with the BOS alone (JSON on stdout, the per-block
#    lines on stderr)
for M in Qwen/Qwen3-0.6B-Base TinyLlama/TinyLlama_v1.1; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/llama_diag.py --model $M --seq 128 --threads 8 \
    > results/f2bq_prefix/diag_${S}_none.jsonl 2> results/f2bq_prefix/diag_${S}_none.txt
  PYTHONPATH=src python experiments/9_quality/llama_diag.py --model $M --seq 128 --threads 8 --prefix \
    > results/f2bq_prefix/diag_${S}_auto.jsonl 2> results/f2bq_prefix/diag_${S}_auto.txt
  PYTHONPATH=src python experiments/9_quality/llama_diag.py --model $M --seq 128 --threads 8 --prefix bos \
    > results/f2bq_prefix/diag_${S}_bos.jsonl 2> results/f2bq_prefix/diag_${S}_bos.txt
done
# 3. 16 x 128 with PrefixQuant's prefix: scalar gains 2 4 8, smoothed alpha 0.45 0.55 0.65, with and without the
#    per-token exponent
for M in Qwen/Qwen3-0.6B-Base TinyLlama/TinyLlama_v1.1 Qwen/Qwen3-1.7B-Base; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 128 --windows 16 --calib-windows 4 \
    --device cuda --threads 8 --gains 2 4 8 --alphas 0.45 0.55 0.65 --glu-exps 8 --structure --cos-windows 4 \
    --prefix --ref-cache /tmp/$USER/f2bq/ref_${S}_128_auto.pt --out results/f2bq_prefix/${S}_128_auto.jsonl
done
# 3b. the smaller fix (the BOS alone) on the two week-1 models, same cells: which mechanism each model needs
for M in Qwen/Qwen3-0.6B-Base TinyLlama/TinyLlama_v1.1; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 128 --windows 16 --calib-windows 4 \
    --device cuda --threads 8 --gains 2 4 8 --alphas 0.45 0.55 0.65 --glu-exps 8 --cos-windows 4 \
    --prefix bos --ref-cache /tmp/$USER/f2bq/ref_${S}_128_bos.pt --out results/f2bq_prefix/${S}_128_bos.jsonl
done
# 4. 2,048 tokens, every test window, with PrefixQuant's prefix (calibration on the CPU): 9 cells per model
for M in Qwen/Qwen3-0.6B-Base TinyLlama/TinyLlama_v1.1 Qwen/Qwen3-1.7B-Base; do S=$(basename $M)
  PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model $M --seq 2048 --calib-windows 4 --device cuda \
    --threads 8 --gains 2 4 8 --alphas 0.45 0.55 0.65 --glu-exps 8 --cos-windows 1 \
    --prefix --ref-cache /tmp/$USER/f2bq/ref_${S}_2048_auto.pt --out results/f2bq_prefix/${S}_2048_auto.jsonl
done
```

`jq -c '{cell, prefix_tokens, alpha, gain, ppl, ref_ppl, ratio, ratio_ci95, top1, block_cos_mean_min,
structure_diff}' results/f2bq_prefix/*_auto.jsonl results/f2bq_prefix/*_bos.jsonl` gives table 12.6 (b);
`grep -h "block\|prefix\|s_res" results/f2bq_prefix/diag_*.txt` table (a).

### 12.8 Limitations and open issues

* **The prefix's keys are clipped on the query's scales if they reach beyond them.** By design the k and v scales come
  from the query positions; an attention sink whose key is far larger than any query-position key would clip at 127
  (`prefix_clip` > 1) and weaken the sink. The tiny models stay below 1.08; the real ones are reported per cell. If it
  bites, the remedy is one line: include the prefix's pair norms in the k scale's statistic (the prefix's values,
  near zero for a sink, are not expected to clip).
* **PrefixQuant's premise is unverified on these models**: that once the outlier tokens sit in the prefix, their later
  occurrences in the window no longer spike. The diagnosis with `--prefix` shows it directly (the fp maximum over the
  window positions and its position, per block).
* **The `auto` prefix depends on the calibration windows** (4 x 128 or 4 x 2,048), so the 128- and 2,048-token runs
  may choose different prefixes; each record states its prefix and the statistics behind it.
* **Context length.** At 2,048 tokens the joint sequence has `2,048 + P` positions, `P` past TinyLlama's trained
  context (`max_position_embeddings = 2,048`); the integer graph and the reference see the same positions
  (`joint_len`, `max_pos` in the records). Qwen3's context is 32,768.
* **Root of trust (D6a).** The prefix's keys and values come from a float forward pass of the checkpoint, rounded to
  int8: a different BLAS could, rarely, round an entry differently, changing the digest; D6a's recipe applies as to
  `s_res`.
* **Generation.** Every query of a prefixed graph has the prefix in front; generated tokens follow the prompt as before
  (`with_lm_positions`, `greedy_tokens` are unchanged); the prefix is part of the statement through the digest, so a
  client must know it (it is public, like the rest of the graph).

## 13. Op-level attribution (`llama_attrib.py`) and the q/k balance

Status: **the tool is implemented and validated on the laptop (tiny random models, no download of weights); it found,
on the laptop, a cause of Qwen3's first-block loss in the builder -- the k norm's int8 range -- which the q/k balance
per RoPE pair (`qk_balance`, default on for Qwen3) removes; the real-checkpoint runs are cluster commands (13.6) with
placeholders (13.7).**

### 13.1 Why

`llama_diag.py` (jobs 1009531, 1009796; Section 12.1) says *after which block* the integer residual stream leaves the
checkpoint's. Two findings remained after the prefix: Qwen3-0.6B-Base's first decoder layer already gives a mean
cosine of 0.87-0.90 for ordinary tokens, with or without the prefix, scalar or smoothed (the prefix moved the 6,800
massive activation, and with it the residual maximum behind `s_res`, out of the window, and the first block did not
move), while OPT's builder reaches above 0.99 per block; and TinyLlama, whose first block is fine with the automatic prefix (0.977
smoothed), decays block by block to about 0.85 (with the BOS-only prefix the smoothed graph falls to 0.06). Neither
says *which op*.

### 13.2 The tool

`experiments/9_quality/llama_attrib.py --model M --recipe {scalar4,smooth55} --blocks {0,...,all} --mode
{accumulated,teacher,both} [--prefix [SPEC]] [--no-qk-balance] [--no-head-eq]`, on one WikiText-2 test window
(`--seq 128`, `--window 0`) with `llama_diag.py`'s calibration (4 train windows) and recipes. Blocks are HF's decoder
layers, 0-based (layer `i`'s output is `llama_diag.py`'s block `i + 1`).

**The map** (`real_llama.ROLES`, recorded by the builder per block as `info["roles"]`; `HF_SOURCE`, `FLOAT_OP` in the
tool). Every tensor is dequantised with its real scale (`info["scales"]`: a float, or a per-channel vector for a
smoothed norm and a balanced q/k) and brought to the checkpoint's units:

| role | graph op (note) | to the checkpoint's units | checkpoint activation | exact op on the graph's inputs |
|---|---|---|---|---|
| `attn_norm` | "norm" | x real scale x `input_layernorm` gamma (the builder folds gamma into W) | `input_layernorm` output | RMSNorm(`in`), gamma and epsilon |
| `q_proj`, `k_proj`, `v_proj` | weight op | x scale, / head-equalisation `beta` (q, k) | `q_proj` / `k_proj` / `v_proj` output | the HF linear on `attn_norm` |
| `q`, `k`, `v` | "requant" | as above | the same | identity (the requantisation's own error) |
| `q_norm`, `k_norm` (Qwen3) | "q norm", "k norm" | x per-channel scale `1 / (G c_j)` | `q_norm` / `k_norm` output | per-head RMSNorm, gamma, epsilon |
| `q_rope`, `k_rope` | "rope" | the same scale | HF's `apply_rotary_pos_emb` of the hooked q/k with the hooked cos, sin | the same function |
| `attn_raw` | (inside "attention") | raw `P V` recomputed with the op's constants, x `s_v / P` | `o_proj` input | float64 softmax attention, causal, grouped, with the prefix's constants |
| `attn` | "attention" | x scale | `o_proj` input | identity |
| `o_proj` | weight op | x scale | `o_proj` output | the HF linear |
| `res_attn` | "residual add" | x `s_res` | `post_attention_layernorm` input | `in + o_proj` |
| `mlp_norm` | "norm" | x scale x gamma | `post_attention_layernorm` output | RMSNorm |
| `gate_proj`, `gate`, `up_proj`, `up` | weight op, "requant" | x scale | `gate_proj` / `up_proj` output | linear / identity |
| `silu` | "silu" | x `s_a` | `act_fn` output | SiLU |
| `glu` | "swiglu" | x `s_p` (x `2^e_t` with `glu_exp`) | `down_proj` input | `silu x up` |
| `down_proj` | weight op | x scale (x `2^e_t`) | `down_proj` output | the HF linear |
| `res_mlp` | "residual add" | x `s_res` | the layer's output | `res_attn + down_proj` |

**Three comparisons per tensor**: `vs_hf` (the graph against the checkpoint: the error accumulated up to it),
`local` (the graph against the exact float64 op on the graph's *own* dequantised inputs: the error this op adds),
`carried` (that exact op against the checkpoint: the error its inputs bring). Each: cosine over channels per
position (mean, minimum and its position, mean over positions 1..), relative L2 error (which also sees a wrong scale),
the worst head's cosine for q/k/v/attention tensors; per tensor also the int8 entries at +-127 and at 0, residual
entries at RES_MAX, a claim's largest |value| and honest bound (log2), the real scale.

**The attention pattern.** A cosine of q or k is dominated by its largest channels, which need not be the ones the
scores weigh (13.4: Qwen3's k norm has a 96.5 gain on a channel whose q gain is 0.003). So the tool compares the
scores `q k / sqrt(head_dim)` of the graph's q and k with the checkpoint's -- RMS of the row-centred difference in nats
against the reference's spread, total variation of the softmax rows -- and, per stage of the q path (`in`,
`attn_norm`, `q_proj`, `q`, `q_norm`, `q_rope`) and of the k path, carries the stage's tensor to RoPE's output with
the exact ops and scores it against the checkpoint's other side: the increase from one stage to the next is the score
error that op adds (`score_steps`; `in`: what the block's input brings).

**Modes.** `accumulated`: the graph on the window as is. `teacher`: each block's ops on the checkpoint's residual at
its input, `clamp(round(h / s_res), +-RES_MAX)`, so each block is measured alone. `both`: the two from one build.
`--blocks i` builds `i + 1` blocks only, whose blocks are the full graph's integer for integer (tested). Per block the
output names the first op whose own error exceeds `--flag-rel 0.1` or whose mean cosine is below `--flag-cos 0.99`, or
which adds more than `--flag-nats 0.1` to the scores (`first_flag`), and the ops of the largest own error and the
largest score step (`worst`). Self-checks: the map (`check_roles`: each role is the output of an op of the expected kind
and note, in op order; a block's ops read nothing from outside it but its input and the token ids) and, per block,
the recomputed raw attention requantises to the op's output integer for integer. Output: one JSON line per (mode,
block) on stdout and in `--out`, the table on stderr.

### 13.3 Validation on the laptop

`tests/test_llama_attrib.py`: **44 passed in 14.8 s** (laptop, 2 threads, transformers 4.57.6); with
`tests/test_real_llama.py` and `tests/test_prob_max.py`: **124 passed in 70.5 s**.

| Test | Checks |
|---|---|
| `test_the_role_map_names_every_tensor_of_every_block` (x8: both families; plain, pruned, prefix, `glu_exp`) | `check_roles` empty; every role has a scale; the blocks' op ranges tile the graph; the check fails on two swapped roles and on a role at an op of another kind |
| `test_every_role_has_a_checkpoint_source_and_an_exact_op` | `HF_SOURCE`, `FLOAT_OP`, `ROLES` cover the same roles |
| `test_the_exact_ops_on_the_checkpoints_inputs_give_its_outputs` (x4: both families, with and without prefix) | every exact op applied to the *checkpoint's* own inputs gives its hooked output (relative error < 1e-5), the attention (`float_attention` on the reconstructed RoPE outputs, grouped, causal, with the prefix's keys and values) included; the reconstructed post-RoPE keys are HF's KV cache |
| `test_the_runner_and_teacher_forcing_give_the_graphs_own_integers` (x6) | the tool's run gives the plain forward's integers; teacher forcing with the graph's own residual reproduces the accumulated block exactly |
| `test_a_one_block_build_has_the_full_builds_first_block` (x2) | `--blocks 0`'s 1-block build: the same roles, scales and integers |
| `test_every_tensor_of_the_smoothed_graph_tracks_the_checkpoint` (x16) | every tensor: `vs_hf` mean cosine >= 0.98 (accumulated) / 0.99 (teacher), `local` mean cosine >= 0.999 and relative error <= 0.08; no op adds more than 0.08 nats; score error <= 0.25 / 0.1 nats; nothing flagged; the accumulated block outputs equal `real_llama.block_cosines` (`llama_diag.py`'s) to 1e-9 |
| `test_a_wrong_q_or_k_norm_gain_is_caught_at_that_norm` (x2) | a builder with the q (k) norm's gain on the wrong channels (monkeypatched): `first_flag` is that norm, in both modes |
| `test_a_swiglu_product_off_by_two_is_caught_at_the_product` (x2) | a SwiGLU product multiplier doubled (monkeypatched): `first_flag` is `glu`, local relative error 0.8-1.1 at a cosine > 0.99 (a scale error the cosine alone misses) |
| `test_qwen3_06b_k_norm_gains_cost_the_scores_and_the_balance_restores_them` (x2 modes) | 13.4's realistic model, the numbers below |
| `test_the_balance_leaves_llama_graphs_and_the_structure_alone` | Llama: the same digest with and without; Qwen3: another digest, both with an empty structural diff |

Measured on the smoothed recipe (tiny models of Section 7, seeds 0 / 1, the four variants, blocks 0 and 1): `vs_hf`
mean cosine at least 0.987 / 0.981 accumulated and 0.996 / 0.995 teacher-forced, `local` mean cosine at least 0.9994
/ 0.9968, `local` relative error at most 0.049 / 0.134 (seed 1's 0.13: the pruned Llama block's SwiGLU product, four
positions using few int8 levels), the largest score step 0.033 / 0.048 nats. The scalar recipe flags the input norm
(gain 4: about 0.11 nats on the tiny models), as expected. Smoke runs of the script (`--smoke`: a 2-block random model
with the real tokenizer and WikiText-2): Qwen3 vocabulary, `--blocks all --mode both`; TinyLlama vocabulary,
`--recipe scalar4 --prefix --blocks 1 --mode teacher`; Qwen3 `--no-qk-balance --prefix bos`: all run end to end, the
JSON lines and the table as described.

### 13.4 What it found on the laptop: Qwen3's q/k-norm gains set the k norm's int8 range

**The real gains.** Qwen3-0.6B-Base's norm weights were read from the checkpoint's `model.safetensors` on the Hub
with HTTP range requests (the header and the 113 norm tensors, 0.3 MB; no weight matrix). Layer 0's `k_norm` has gains
**96.5, 38.3 and 26.3** on channels 50, 115 and 88 against a median of 2.28, and `q_norm` has 0.003, 0.18 and 0.02 on
the same channels; mirror-wise q's largest gain, 11.8 (channel 32), meets k's 0.16. The RoPE partners of the large k
channels are ordinary (channel 114: k 1.21, q 1.30). Every layer has such k gains: the largest is 44.5, 44.0, 23.0,
41.8, 34.0 in layers 1-5, 4.9-25 in layers 8-27 (q's largest: 2.8-11.8). The tiny test models draw every gain from
[0.5, 1.5], so they never see this.

**The mechanism.** The k norm is one cheap op with one public output scale `1/G_k`, and `G_k` maps the largest RoPE
pair norm of `gamma_j x_j / rms` (the 96.5 channel) to 127. Every ordinary channel (gain 2.3) then gets 1-3 int8
levels: 31% of k's int8 outputs are 0. The scores weigh exactly those channels (q's gain on the large ones is about
0), so the attention pattern breaks while k's cosine against the checkpoint still reads 0.998 (the large channels
dominate the vector's norm). In the tool: `k_norm` vs_hf cosine 0.998, `attn_raw` vs_hf 0.917 with its own (local)
error 0.004, and the k path's score error jumps at `k_norm`.

**A laptop reproduction.** A tiny Qwen3 with the real layer-0 q/k gains (head_dim 128, 1 block, d = 128, 4 / 2 heads,
random weights of std 0.15, the other gains in [0.5, 1.5]; calibration 4 x 64, evaluation 2 x 64; teacher-forced,
seeds 0 / 1):

| quantity | smoothed, without the balance | smoothed, with it | scalar 4, without | scalar 4, with |
|---|---|---|---|---|
| block output, mean cosine (minimum) | 0.872 (0.54) / 0.783 (0.21) | **0.987 (0.90) / 0.983 (0.90)** | 0.827 (0.28) / 0.737 (0.26) | 0.955 (0.79) / 0.939 (0.71) |
| attention score error, nats (reference spread 4.4) | 1.18 / 1.40 | 0.31 / 0.34 | 1.42 / 1.55 | 0.57 / 0.59 |
| softmax total variation, mean | 0.21 / 0.28 | 0.057 / 0.064 | 0.25 / 0.31 | 0.11 / 0.12 |
| score error added by the k norm / q norm, nats | **0.90 / 1.10**, 0.24 / 0.27 | 0.11 / 0.14, 0.11 / 0.12 | 0.83 / 0.98, 0.15 / 0.14 | 0.06 / 0.06, 0.05 / 0.06 |
| largest score step of any other op, nats | 0.12 / 0.15 | 0.03 / 0.03 | 0.31 (the input norm) | 0.31 (the input norm) |
| k norm's int8 outputs at 0 | 31% / 35% | 6% / 7% | 33% / 37% | 6% / 7% |

Without the balance the first block falls to 0.78-0.87 (the real Qwen3-0.6B-Base: 0.889 smoothed, 0.865 scalar, job
1009531), and the attribution puts it on the k norm (and, less, the q norm). A 28-block version (every layer's real
q/k gains, weights of std 0.05, seed 0, accumulated, smoothed) gives per-block mean cosines 0.892, 0.831, 0.793, 0.779
... 0.891 without the balance -- the shape of the real model's first blocks (0.889, 0.874, 0.852, 0.789) -- and 0.990,
0.980, 0.970, 0.964 ... 0.975 with it (scalar 4: 0.851, 0.785, 0.744, 0.730 ... 0.836 against 0.969, 0.933, 0.902,
0.885 ... 0.915); teacher-forced, the k norm adds 1.08, 0.37, 0.43, 0.15, 0.29 nats in layers 0-4 without the balance
and at most 0.13 with it. (Random weights: the proxy shows the mechanism, not the real model's numbers.)

**The fix: `qk_balance` (default on; Qwen3 only).** Per RoPE pair `p` (channels `j`, `j + head_dim/2`), q's norm takes
the factor `c_p` into its gain and k's norm `1/c_p`, `c_p = sqrt(A_k,p / A_q,p)` with `A_p` the `pct` percentile of
the pair's norm of `gamma_j x_j / rms` over the calibration rows (all heads; clamped to [2^-8, 2^8]); then `G_q` and
`G_k` are set as before from the scaled values. Why it is exact: RoPE rotates each pair as a whole, so a factor shared
by the pair commutes with it, and every score `sum_j q_j k_j` is unchanged in real numbers -- channel `j`'s real
scales are `1 / (G_q c_j)` for q and `c_j / G_k` for k, whose product `1 / (G_q G_k)` is the same in every channel and
is what `m_s` already uses (`m_s`'s formula is unchanged). Each pair then spans `sqrt(A_q,p A_k,p)` on both sides
instead of k's range being set by a channel q ignores. What changes: only constants -- the q/k norms' gain vectors
(`[head_dim]`, shared by the heads, as before) and, with a prefix, the prefix's int8 keys, now quantised on `s_k c_j`
(`info["prefix_scales"]` holds that vector; `test_the_prefix_constants_...` checks the constants against HF's cache on
it). The structure, the claims and the protocol are the benchmark graph's (all of `test_real_llama.py`'s 52 tests pass
with the balance on, v0, V1 and the prefix's exact equality included). Llama graphs are byte-identical (no q/k norm;
tested by digest), OPT's builder is not touched, and Qwen3's digests change (other gains). `qk_balance=False` (the
scripts' `--no-qk-balance`) builds the graphs of jobs 1009531 / 1009796: before the change, the digests of the 64 tiny
test graphs (both families, both recipes, pruned or not, with or without prefix and `glu_exp`, with or without head
equalisation) were recorded; with `qk_balance=False` they are identical. On the uniform-gain tiny models the balance
is neutral (smoothed logit cosine 0.9942 / 0.9950 with it against 0.9942 / 0.9952 without, seeds 0 / 1).

What remains in the proxy with the balance: the q and k norms' own rounding (0.11-0.14 nats each, smoothed) and, for
the scalar recipe, the input norm's gain-4 resolution (0.31 nats), which the smoothed recipe removes (0.03).

### 13.5 Other candidates, checked on the laptop

* **Head equalisation (`head_eq`).** No effect in the proxy (block output 0.8758 with, 0.8663 without, seed 0
  unbalanced; the random heads are alike). On real heads it scales a small head's rows up, which coarsens the other
  heads' weights in the matrix's one scale: the tool's per-head `local` cosine of `q_proj` / `k_proj` and the
  `--no-head-eq` line of 13.6 measure it.
* **RoPE's pair-norm scale.** The scale covers every rotation (no clip by construction); RoPE's own score step is
  <= 0.03 nats on the tiny models and in the proxy with the balance (0.07 / 0.12 for q / k without it: its rounding on
  k's coarse grid).
* **The score multiplier `m_s` with head_dim 128.** The exp table's 1/16-nat steps and `m_s`'s rounding are the raw
  attention's `local` error: relative 0.004-0.009 (proxy and tiny models). `m_s >= 2^32` would only switch to the exact
  int64 path.
* **The SiLU table.** `local` relative error 0.013-0.014 on the tiny models; real gates (massive values) are the
  tool's `silu` row on the cluster.
* **The int32 claim bound and the residual scale.** Reported per op (claim bounds 2^18-2^21 on the tiny models;
  residual adds' own error 0.000). The prefix took the 6,800 activation out of the residual maximum behind Qwen3's
  `s_res` without moving the first block (job 1009796): not its cause (the cluster rows give `res_attn`'s own error).
* **The attention's head-group path.** When the heads do not fit one score budget (TinyLlama's 32 heads at 2,048
  tokens) `_attention` repeats k and v per group instead of stacking the grouped queries; forced on the tiny models
  (both families, with and without prefix) it gives the same integers.
* **RMSNorm's epsilon** (not in the integer norm): the tool's `attn_norm` exact op includes it, so layer 0's
  teacher-forced `attn_norm` `local` error measures it on the real embeddings (TinyLlama's epsilon is 1e-5; its
  embedding rows are small).
* **TinyLlama.** No q/k norm, so the balance does not apply; its first block is fine with the prefix. The BOS-only
  collapse of the smoothed graph is consistent with the `\n` token's massive activation staying in the window and the
  99.99 percentiles of gate, up and the product (`pct`, `pct_glu`) clipping it (the scalar recipe, `pct` 100, falls
  only to 0.73): the tool's `glu` / `gate` / `up` clip fractions and the minimum cosine's position in layer 1 decide it.
  The block-by-block decay with the automatic prefix has no laptop reproduction; the teacher-forced rows of every
  block attribute it.

### 13.6 Cluster commands (bash, from `code/`, `HF_HOME` on node-local `/tmp`; CPU is enough; 64 GB)

The attribution runs on the CPU (calibration and both forwards; a 128-token window). `--mode both` gives the
teacher-forced and the accumulated rows from one build; the per-block tables go to `.txt` (stderr), the JSON lines to
`--out`.

```bash
mkdir -p results/f2bq_attrib /tmp/$USER/f2bq
# 0. the tests on the node (CPU): expect "96 passed"
PYTHONPATH=src python -m pytest -q -o addopts= -p no:cacheprovider tests/test_real_llama.py tests/test_llama_attrib.py
# 1. Qwen3-0.6B-Base: the first decoder layer (a 1-block build) and every block; scalar4 and smooth55; without and
#    with PrefixQuant's prefix; teacher-forced and accumulated
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe scalar4 --blocks 0 --mode both --out results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_b0.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_b0.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks 0 --mode both --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe scalar4 --blocks 0 --mode both --prefix --out results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_b0_auto.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_b0_auto.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks 0 --mode both --prefix --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0_auto.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0_auto.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe scalar4 --blocks all --mode both --out results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_all.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_all.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks all --mode both --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_all.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_all.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe scalar4 --blocks all --mode both --prefix --out results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_all_auto.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_all_auto.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks all --mode both --prefix --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_all_auto.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_all_auto.txt
# 2. TinyLlama_v1.1: the same eight runs
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe scalar4 --blocks 0 --mode both --out results/f2bq_attrib/TinyLlama_v1.1_scalar4_b0.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_scalar4_b0.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe smooth55 --blocks 0 --mode both --out results/f2bq_attrib/TinyLlama_v1.1_smooth55_b0.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_smooth55_b0.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe scalar4 --blocks 0 --mode both --prefix --out results/f2bq_attrib/TinyLlama_v1.1_scalar4_b0_auto.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_scalar4_b0_auto.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe smooth55 --blocks 0 --mode both --prefix --out results/f2bq_attrib/TinyLlama_v1.1_smooth55_b0_auto.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_smooth55_b0_auto.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe scalar4 --blocks all --mode both --out results/f2bq_attrib/TinyLlama_v1.1_scalar4_all.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_scalar4_all.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe smooth55 --blocks all --mode both --out results/f2bq_attrib/TinyLlama_v1.1_smooth55_all.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_smooth55_all.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe scalar4 --blocks all --mode both --prefix --out results/f2bq_attrib/TinyLlama_v1.1_scalar4_all_auto.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_scalar4_all_auto.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe smooth55 --blocks all --mode both --prefix --out results/f2bq_attrib/TinyLlama_v1.1_smooth55_all_auto.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_smooth55_all_auto.txt
# 3. TinyLlama's BOS-only collapse (smoothed): which op clips the "\n" activation
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model TinyLlama/TinyLlama_v1.1 --seq 128 --threads 8 --recipe smooth55 --blocks all --mode both --prefix bos --out results/f2bq_attrib/TinyLlama_v1.1_smooth55_all_bos.jsonl > /dev/null 2> results/f2bq_attrib/TinyLlama_v1.1_smooth55_all_bos.txt
# 4. Qwen3: the balance on the real weights -- the old graphs (no balance), and head equalisation off
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks all --mode both --prefix --no-qk-balance --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_all_auto_nobal.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_all_auto_nobal.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe scalar4 --blocks 0 --mode both --no-qk-balance --out results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_b0_nobal.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_scalar4_b0_nobal.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks 0 --mode both --no-qk-balance --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0_nobal.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0_nobal.txt
PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --recipe smooth55 --blocks 0 --mode both --no-head-eq --out results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0_noheadeq.jsonl > /dev/null 2> results/f2bq_attrib/Qwen3-0.6B-Base_smooth55_b0_noheadeq.txt
# 5. Qwen3: the balance's effect on the diagnosis and on the perplexity (16 x 128, PrefixQuant's prefix; GPU node)
PYTHONPATH=src python experiments/9_quality/llama_diag.py --model Qwen/Qwen3-0.6B-Base --seq 128 --threads 8 --prefix > results/f2bq_attrib/diag_Qwen3-0.6B-Base_auto.jsonl 2> results/f2bq_attrib/diag_Qwen3-0.6B-Base_auto.txt
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --seq 128 --windows 16 --calib-windows 4 --device cuda --threads 8 --gains 2 4 8 --alphas 0.45 0.55 0.65 --glu-exps 8 --cos-windows 4 --prefix --ref-cache /tmp/$USER/f2bq/ref_Qwen3-0.6B-Base_128_auto.pt --out results/f2bq_attrib/Qwen3-0.6B-Base_128_auto_qkbal.jsonl
PYTHONPATH=src python experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --seq 128 --windows 16 --calib-windows 4 --device cuda --threads 8 --gains 2 4 8 --alphas 0.45 0.55 0.65 --glu-exps 8 --cos-windows 4 --ref-cache /tmp/$USER/f2bq/ref_Qwen3-0.6B-Base_128.pt --out results/f2bq_attrib/Qwen3-0.6B-Base_128_qkbal.jsonl
```

`grep "=>" results/f2bq_attrib/*.txt` gives per block the output's cosine, the first flagged op, and the ops of the
largest own error and the largest score step; `grep "pattern\|path" ...` the score errors along the q and k paths. Run
4's `_nobal` files against run 1's are the balance on the real weights (the diagnosis of job 1009796 is the
`--no-qk-balance` baseline of run 5's first line); run 5's perplexities go to table 12.6 (b) (its earlier rows are
without the balance).

### 13.7 Results on the real checkpoints (job 1011410, CPU node s-006, git f85ad55)

Tests on the node: 96 passed. One 128-token WikiText-2 test window; calibration on 4 x 128 train windows.

(a) The first decoder layer (block 0), teacher-forced: the block output's mean cosine (minimum over positions), the
first flagged op, and the attention-score error the q and k norms add (nats; the k path's cumulative error after
the k norm in parentheses):

| Model, recipe, prefix | block output | first flag | k-norm step (cumulative) | q-norm step | without the balance |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base, scalar4, none | 0.986 (0.957) | q_proj | 0.221 (0.552) | 0.097 | 0.865 (0.452); k norm 1.378 (1.709) |
| Qwen3-0.6B-Base, smooth55, none | 0.991 (0.956) | q_norm | 0.296 (0.450) | 0.119 | 0.889 (0.499); k norm 1.387 (1.541) |
| Qwen3-0.6B-Base, scalar4, auto | 0.987 (0.945) | attn_norm | 0.188 (0.508) | 0.091 | |
| Qwen3-0.6B-Base, smooth55, auto | 0.991 (0.955) | q_norm | 0.253 (0.400) | 0.125 | |
| Qwen3-0.6B-Base, smooth55, none, no head equalisation | 0.989 (0.937) | q_norm | 0.324 (0.475) | 0.246 | |
| TinyLlama_v1.1, scalar4, none / auto | 0.858 (-0.028) / 0.871 (-0.009) | attn_norm | n/a | n/a | n/a |
| TinyLlama_v1.1, smooth55, none / auto | 0.969 (0.000) / 0.978 (0.594) | attn_norm | n/a | n/a | n/a |

**The laptop diagnosis holds on the real checkpoint.** Without the balance the k norm alone adds 1.38-1.39 nats to
the first block's attention scores and the block output falls to 0.865-0.889 (job 1009531's 0.865 / 0.889 to the
digit); with it the k norm adds 0.19-0.30 nats and the block output is 0.986-0.991. Head equalisation is worth a
little on top (0.989 -> 0.991; the q norm's step halves). TinyLlama's first block has no q/k norm; its smoothed
block output is 0.969-0.978, and the scalar recipe's 0.86 is SiLU's and the input norm's (the scalar gain).

(b) Every block, the accumulated stream (each block reads the graph's own residual) and teacher-forced (each block
reads the checkpoint's residual): smallest per-block mean cosine (its block), the last block's, the mean over
blocks, and which op adds the largest own error in most blocks:

| Model, recipe, prefix | accumulated: min / last / mean | teacher-forced: min / last / mean | largest own error |
|---|---|---|---|
| Qwen3-0.6B-Base, smooth55, none | 0.584 (block 21) / 0.748 / 0.737 | 0.882 (13) / 0.993 / 0.986 | glu in 20-23 of 28 |
| Qwen3-0.6B-Base, smooth55, auto | **0.929 (19) / 0.945 / 0.962** | 0.978 (13) / 0.996 / 0.994 | glu in 21-25 of 28 |
| Qwen3-0.6B-Base, smooth55, auto, no balance | 0.864 (2) / 0.923 / 0.899 | 0.899 (0) / 0.996 / 0.991 | glu |
| Qwen3-0.6B-Base, scalar4, none / auto | 0.726 / 0.762 (last) | 0.869 / 0.823 (last) | glu |
| TinyLlama_v1.1, smooth55, none | 0.309 (11) / 0.578 / 0.511 | 0.969 (0) / 0.987 / 0.989 | glu |
| TinyLlama_v1.1, smooth55, auto | 0.830 (5) / 0.916 / 0.876 | 0.978 (0) / 0.995 / 0.995 | glu in 20 of 22 |
| TinyLlama_v1.1, smooth55, bos | 0.063 (14) / 0.437 / 0.374 | 0.977 (0) / 0.987 / 0.991 | glu |
| TinyLlama_v1.1, scalar4, none / auto | 0.513 / 0.731 (min) | 0.858 / 0.871 (min) | glu, silu |

The accumulated rows of the no-balance Qwen3 graph with the prefix equal job 1009796's `llama_diag.py` row (0.864 at
block 2, 0.923 at the last), as expected. `llama_diag.py --prefix` with the balance (the job's last step) gives the
same picture for Qwen3: smooth55 0.991 / 0.987 / 0.984 / 0.982 in blocks 1-4 and 0.954 at block 28 (before: 0.896 /
0.884 / 0.864 / 0.865 and 0.923). The prefix's values still clip in layer 2 (`v_clip` 4.6 with smooth55).

Reading:

* **Qwen3: the balance and the prefix together** keep the smoothed stream at 0.93-0.99 through all 28 blocks. Each
  alone is not enough: without the prefix the massive activation that forms in block 3 breaks the stream from there
  on (0.98 -> 0.91 -> 0.58); without the balance the first block already loses 11%.
* **TinyLlama:** every block is good when it reads the checkpoint's residual (teacher-forced 0.98-0.995), but the
  accumulated stream drops at block 1 (0.977 -> 0.894 with the prefix), where the error carried from block 0's
  output turns into 1.4 nats of score error. `bos` alone collapses (0.06), consistent with Section 12.6: TinyLlama's
  `\n` outlier must sit in the prefix.
* **The next target in both families is the SwiGLU product** (`glu`): it adds the largest own error in about 20 of
  every 22-28 blocks (relative error 0.1-0.3). The per-token exponent (`glu_exp`, Section 3.5) acts there; the
  16 x 128 ladder with the balance (job 1011411, with `smooth-exp8`) measures what it buys in perplexity.

(c) 16 x 128 perplexity ratios with the balance (job 1011411), Qwen3-0.6B-Base with the prefix (reference 29.24):
scalar x10.7 / x6.53 / **x5.58** (gains 2 / 4 / 8); smooth x1.60 [1.52, 1.69] / **x1.59** [1.53, 1.67] / x1.63
(alpha 0.45 / 0.55 / 0.65; top-1 0.61-0.62); smooth-exp8 x1.57 / x1.56 / **x1.55** [1.47, 1.64] (top-1 0.61-0.63).
Before the balance (Section 12.6): x23.2, x2.11, x2.05. So the balance takes Qwen3-0.6B from x2.1 to x1.55-1.59 with
the prefix; it is still far from x1.05, and the per-token exponent adds little, although the attribution names the
SwiGLU product as the largest remaining own error.

Without the prefix (same job), the balance alone does not save Qwen3-0.6B: scalar x10.0-14.0, smooth x43-50,
smooth-exp8 x9.3-10.3 (the massive activation of block 3 stays in the window). Qwen3-1.7B-Base with the prefix
(reference 20.64): scalar x6.46-8.51; smooth **x1.51** [1.46, 1.57] / x1.55 / x1.57 (alpha 0.45 / 0.55 / 0.65);
smooth-exp8 **x1.33** [1.29, 1.38] / x1.35 / x1.35 (top-1 0.66-0.68). Before the balance (12.6): x1.72 and x1.44.

Status against the kill rule (Fri 23 Oct; plan: Qwen3 within x1.10, else the measured number with its attribution
and the Qwen cost rows labelled "shape only"): the best Qwen3 graphs are x1.33 (1.7B) and x1.55 (0.6B). Both fixes
(prefix and balance) are needed; what remains is spread over the SwiGLU products of every block. Qwen3-4B has not
been run with the balance.

### 13.7a The next lever: the SwiGLU product's balance (`glu_smooth`, commit f45debd)

The attribution puts the largest remaining own error at the SwiGLU product in most blocks, and the per-token
exponent (`glu_exp`) helped Qwen3-1.7B (x1.51 -> x1.33) but hardly Qwen3-0.6B (x1.59 -> x1.55). The product has one
static int8 scale, set by its few largest channels. `glu_smooth` (opt-in, an alpha) applies SmoothQuant's balance to
the down projection's input and folds it through the up projection, since `silu(gate) up` is linear in `up`: up's
row `j` is divided by `g_j = A_j^alpha / W_j^(1-alpha)` and the down projection's column `j` multiplied by it, with
`A_j` the largest `|silu(gate) up|` of channel `j` on the calibration rows (from the norm's integer output) and `W_j`
the largest `|w_down[:, j]|` (normalised to geometric mean 1, clamped to `[2^-8, 2^8]`). Every real output is
unchanged; constants only; the structure diff stays empty; off by default (no digest changes). Tests: the rescaled
float model gives the checkpoint's logits, the graph tracks it and v0 accepts it.

**First run: an overflow (job 1012500, git f45debd).** On both Qwen3 models the first `glu_smooth` cell (alpha 0.3)
stopped in `swiglu` with `RuntimeError: Overflow when unpacking long` (the full traceback from job 1012656:
`mm = torch.tensor(round((1 << SHIFT) * s_a * s_u / s_p))`). The plain smoothed cells before it reproduced job
1011411 to the digit (Qwen3-0.6B x1.60 / x1.57, Qwen3-1.7B x1.51 / x1.33). The cause: the multiplier is
`2^30 s_a s_u / s_p`, and `s_p` is the product's `pct_glu` percentile on the calibration rows, clamped at
`1e-12 / 127`; it overflows int64 only if that percentile is exactly 0. That happens when up's int8 rows vanish. The
balance gives a channel whose product is tiny (gate near 0, a "dead" channel) a factor `g_j` far below 1 (down to
the clamp `2^-8`), so up's row `j` is multiplied by up to 256. Up is quantised with one scale per matrix, so that
row then sets the scale and the other rows round to 0 or to a few levels; with almost every product 0, the
percentile is 0. Plain SmoothQuant does not meet this because its factor moves into the previous op's output
scale (a free per-channel vector); here it moves into up's rows, which share one int8 scale.

**The fix (207b798): a box that keeps both weight scales.** After the geometric-mean normalisation, each factor is
kept in `[U_j / U_max, D_max / D_j]`, with `U_j` the largest `|up row j|` as `mat()` quantises it (the norm's
per-channel input scale folded in) and `D_j` the largest `|down column j|`. Then no up row grows beyond up's
current largest row and no down column beyond down's largest column, so neither matrix's per-tensor int8 scale
grows; only the product's channels are rebalanced. The box always contains 1 (`U_j <= U_max`, `D_j <= D_max`), so
the identity is feasible. Also, `swiglu` now raises a `ValueError` naming the cause if the product's percentile is
0, instead of the int64 overflow (no graph that built before changes; the digests of the other cells are the same).
Tests: `test_glu_smooth_box_keeps_both_weight_scales` (a dead channel: the box binds it, `g = U_3 / U_max`, and a
large-product channel still gets `g > 1`) and the down-column bound in the exactness test; `tests/test_real_llama.py`
55 passed.

**Second run (jobs 1012675-7, git 207b798, clone s13):** `glus2`: alpha 0.3 / 0.5 / 0.7, with and without
`glu_exp` 8, at 16 x 128 with the prefix on Qwen3-0.6B, Qwen3-1.7B (smoothing alpha 0.45) and TinyLlama (0.65);
`tl2k`: TinyLlama at 2,048 tokens, smoothing alpha 0.65 / 0.75 / 0.85, with and without `glu_smooth` 0.3 / 0.5;
`q17k`: Qwen3-1.7B at 2,048 tokens with `qk_balance` (Sec. 12.6's 2,048-token rows predate it): smooth, smooth-exp8
and `glu_smooth` 0.5 at alpha 0.45.

**Qwen3-1.7B-Base at 2,048 tokens, with the prefix and `qk_balance` (job 1012677, s-004, git 207b798;** full WikiText-2
test set, reference 9.418, smoothing alpha 0.45; ratio to the fp32 reference with its 95% bootstrap interval, top-1
agreement):

| Cell | ratio | top-1 | before `qk_balance` (Sec. 12.6) |
|---|---|---:|---|
| smooth | x1.455 [1.442, 1.468] | 0.701 | x1.727 |
| smooth-exp8 (per-token exponent at the SwiGLU product) | x1.243 [1.237, 1.250] | 0.754 | x1.449 |
| smooth-glus0.5 | x1.456 [1.435, 1.477] | 0.712 | |
| **smooth-glus0.5-exp8** | **x1.206 [1.200, 1.211]** | **0.773** | |

* The balance pays at 2,048 tokens as at 128 (x1.73 -> x1.46 smoothed; x1.45 -> x1.24 with the exponent).
* `glu_smooth` alone changes nothing (x1.455 -> x1.456), but on top of the per-token exponent it removes another
  quarter of the excess (x1.243 -> x1.206). A plausible reading, not tested: with the exponent each token's product already has its own scale, and
  balancing the channels then spends those levels better. Without the exponent the product's static scale is set
  by the few tokens with massive values, and rebalancing the channels does not change that.
* Best Qwen3-1.7B so far: **x1.206 at 2,048 tokens**, against the plan's x1.10 for Qwen3-4B (kill Fri 23 Oct). The
  same recipe on Qwen3-4B-Base at 2,048 tokens runs as job 1013018; the `glus2` ladder (job 1012675) gives the
  alpha of `glu_smooth` at 128 tokens.

**TinyLlama_v1.1 at 2,048 tokens, prefix `auto` (job 1012676, s-005, git 207b798;** reference 7.697; ratio to fp32
with its 95% bootstrap interval / top-1):

| smoothing alpha | smooth | + `glu_smooth` 0.3 | + `glu_smooth` 0.5 |
|---:|---|---|---|
| 0.65 | x1.561 [1.513, 1.618] / 0.723 | x1.444 [1.415, 1.476] / 0.750 | x2.685 / 0.586 |
| 0.75 | x1.227 [1.212, 1.248] / 0.798 | **x1.147 [1.138, 1.157] / 0.826** | x1.577 / 0.718 |
| 0.85 | x1.406 [1.364, 1.460] / 0.741 | x1.370 [1.329, 1.420] / 0.752 | x1.697 / 0.689 |

* The alpha 0.65 cell equals job 1009797's to every digit (the same graph).
* **TinyLlama meets the F2b-L target (x1.15) at 2,048 tokens:** x1.147 with smoothing alpha 0.75 and `glu_smooth` 0.3
  (the interval's upper end 1.157 is just above it). Section 12.6's x1.56 was alpha 0.65, the ladder's upper end; the
  optimum lies between 0.65 and 0.85 and is narrow.
* `glu_smooth` helps at 0.3 at every alpha and hurts at 0.5: for TinyLlama the product balance must lean towards the
  down projection's weights. With Qwen3-1.7B, 0.5 helped only with the per-token exponent. The alpha of `glu_smooth`
  is therefore a per-family parameter, like the smoothing alpha.
* The finer grid (job 1013124), ratio to fp32 (95% CI):

  | smoothing alpha | smooth | glus 0.2 | glus 0.3 | glus 0.4 |
  |---:|---|---|---|---|
  | 0.70 | x1.317 [1.297, 1.340] | x1.147 [1.140, 1.155] | x1.209 [1.196, 1.223] | x1.557 [1.521, 1.595] |
  | 0.75 | x1.227 [1.212, 1.248] | **x1.122 [1.117, 1.129]** (top-1 0.834) | x1.147 [1.138, 1.157] | x1.229 [1.212, 1.250] |
  | 0.80 | x1.323 [1.295, 1.359] | x1.168 [1.160, 1.178] | x1.255 [1.231, 1.286] | x1.315 [1.277, 1.363] |

  **Best TinyLlama: x1.122 at 2,048 tokens** (smoothing 0.75, `glu_smooth` 0.2), inside the x1.15 target with its whole
  interval. Smoothing 0.75 is the optimum in both directions; `glu_smooth` improves from 0.4 to 0.2 and then levels
  off: 0.15 / 0.1 / 0.05 give x1.125 / x1.124 / x1.129 (job 1013574), so 0.2 stays the setting.

**OpenLLaMA-7B v2, a stand-in for Llama-2-7B (job 1013162, s-004, git 207b798).** `openlm-research/open_llama_7b_v2`
(Apache 2.0, not gated) has exactly Llama-2-7B's shapes; with `--config llama2-7b --structure` the integer graph's
structure diff against the benchmark Llama-2-7B graph is **empty**, so it is the graph whose costs the paper reports,
with real weights. Full WikiText-2 test set at 2,048 tokens (160 windows, 327,520 scored tokens); the prefix `auto`
chose six tokens; the fp32 reference ran on the CPU (14,612 s; perplexity 6.478) and is cached on the shared disk
(`f2bq_glus/ref_open_llama_7b_v2_2048_auto.pt`).

| Cell | ratio to fp32 (95% CI) | top-1 |
|---|---|---:|
| smooth, alpha 0.65 | x1.107 [1.104, 1.110] | 0.848 |
| smooth, alpha 0.75 | x1.133 [1.129, 1.136] | 0.830 |
| **smooth 0.65 + `glu_smooth` 0.3** | **x1.077 [1.074, 1.080]** | **0.869** |
| smooth 0.75 + `glu_smooth` 0.3 | x1.092 [1.089, 1.095] | 0.852 |

* **A Llama model of the benchmark's exact shape is within x1.08 of fp32** with static per-tensor int8 activations,
  the plan's expectation for Llama-2-7B (x1.04-1.10) and below the x1.10 bar of F2b-Q. Plain static per-tensor W8A8
  is reported to collapse on Llama-2-7B (perplexity 9,250 in the literature); with the prefix, the smoothing and the
  product balance the same kind of graph stays within x1.08 on a model of the same shape (OpenLLaMA's weights, not
  Llama-2's).
* Unlike TinyLlama (best at 0.75), this model prefers less smoothing (0.65 < 0.75); 0.55 / 0.6 with `glu_smooth` 0.2 /
  0.3 run as job 1013787 on the cached reference.
* The six prefix tokens push each scored window to positions 6-2,053, just past OpenLLaMA's 2,048-token training
  length, for the reference and the integer model alike.

**Qwen3-4B-Base at 2,048 tokens: the recipe does not carry over (job 1013332, s-005, git 207b798).** Prefix `auto` (one
token, `<|endoftext|>`), `qk_balance`, smoothing alpha 0.45; fp32 reference 7.946 on the CPU (9,651 s; job 1013018 had
run out of GPU memory loading it): smooth **x5.54** [5.35, 5.74], smooth-exp8 x3.02 [2.95, 3.09], glus0.5 x5.33,
glus0.5-exp8 **x2.97** [2.91, 3.05] (top-1 0.46-0.55). The same cells give x1.46 / x1.24 / x1.46 / x1.21 on
Qwen3-1.7B. So Qwen3-4B has a failure the smaller models do not, and searching settings blindly is not the next step:
the per-block diagnosis (`llama_diag.py`) and the op attribution over every block (`llama_attrib.py --blocks all`),
both with the prefix and the `smooth55` recipe of the Qwen3-0.6B diagnosis (Sec. 13.7), run as job 1013928 (CPU node).
Candidates to check against it: a massive activation that one prefix token does not take (Qwen3-4B has 36 blocks), a
query projection wider than the residual stream (32 heads of 128 = 4,096 against hidden 2,560, where Qwen3-1.7B has
2,048 = 2,048), or a block whose norm gains defeat the balance.

**The diagnosis (job 1013928, CPU node s-002; one 128-token test window, prefix `auto` = `<|endoftext|>`):**

* *Two massive activations.* From block 7 on, the residual stream holds a massive activation of about 3,540 in
  channel 4 at the **first window position**, larger than the one the prefix token takes (about 2,030 at the prefix).
  Qwen3-4B forms its attention-sink outlier on the first two positions; PrefixQuant's rule chose one prefix token
  (`o = 1`), so the second outlier stays in the window and sets the residual's one int8 scale (`s_res` 1.13e-3 for
  values up to 3,805). The smoothed stream's mean cosine falls exactly there: 0.914 -> 0.912 -> 0.857 -> 0.816 (blocks
  6-9), 0.74-0.86 to the last block; the first window position itself drops to 0.35-0.59 (the scalar recipe keeps it
  at 1.000, presumably because there the outlier dominates its own scale).
* *The SwiGLU products of the early blocks.* `llama_attrib.py --blocks all` names the SwiGLU product (`glu`) as the
  largest own error in 34 of 36 blocks (block 16: `up`, block 19: `down_proj`), as on the smaller models, but far larger in blocks 1-4: relative error
  0.926 / 0.609 / 0.699 / 0.384 (Qwen3-0.6B: 0.1-0.3). In those blocks the residual's largest value (58-78, channel
  0) sits at window position 32, a token-specific outlier forming in the MLPs before the sink outlier of block 7.
  The per-token exponent acts on these products, consistent with its x5.5 -> x3.0 at 2,048 tokens.
* *Two-token prefixes do not move it (job 1013947).* With `<|endoftext|>` twice the diagnosis still shows the massive
  activation at the first window position (about 3,030 in channel 4 from block 7, against 3,540 with one token), and the
  16 x 128 perplexity is unchanged (reference about 17.4 with every prefix):

  | prefix | smooth | smooth-exp8 |
  |---|---|---|
  | `<|endoftext|>` (auto) | x5.67 [4.67, 6.98] | x3.81 [3.17, 4.65] |
  | `<|endoftext|>` x2 | x5.91 [4.93, 7.13] | x3.87 [3.26, 4.66] |
  | `<|endoftext|>` + newline | x5.99 [5.00, 7.28] | x3.54 [3.02, 4.22] |

  So the second outlier is not a second sink position that a longer prefix absorbs: it forms on the first content
  token (or on that window's first token itself). Prefixes with PrefixQuant's candidate outlier tokens (newline, `.`,
  ` =`, and four `<|endoftext|>`) run as job 1014112; if none helps, Qwen3-4B is reported as measured with this
  diagnosis, and the Qwen rows of the cost tables are labelled "shape only" (the plan's kill rule, Fri 23 Oct).

**The 128-token ladder with the box (job 1012675, `glus2`; 16 x 128, prefix, `qk_balance`):** ratio to fp32 (95% CI)

| Model (smoothing alpha) | smooth | + exp8 | glus 0.3 | glus 0.3 + exp8 | glus 0.5 | glus 0.5 + exp8 | glus 0.7 | glus 0.7 + exp8 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (0.45) | x1.602 | x1.568 | x1.690 | **x1.553** | x2.028 | x1.638 | x2.274 | x1.824 |
| Qwen3-1.7B (0.45) | x1.513 | x1.334 | x1.676 | **x1.304** [1.255, 1.359] | x2.005 | x1.395 | x2.570 | x1.471 |
| TinyLlama (0.65) | **x1.151** | x1.237 | x1.170 | x1.218 | x1.260 | x1.251 | x1.311 | x1.370 |

The first run's smoothed cells are reproduced exactly (x1.602 / x1.568, x1.513 / x1.334). At 128 tokens `glu_smooth`
helps only with the per-token exponent and only at 0.3 (Qwen3-1.7B x1.334 -> x1.304), and it slightly hurts TinyLlama,
whereas at 2,048 tokens it is the lever that brings TinyLlama to x1.147 and Qwen3-1.7B to x1.206 (with 0.5). The best
settings therefore depend on the window length, and the decisions are taken at 2,048 tokens (the protocol of every
reported row). The 2,048-token grid on Qwen3-1.7B (job 1013482; the smooth and exp8 cells repeat job 1012677's to
every digit): `glu_smooth` 0.3 / 0.4 / 0.6 alone x1.399 / x1.430 / x1.497 (smooth x1.455); with the exponent x1.212 /
**x1.202 [1.197, 1.207]** / x1.220 (0.5: x1.206). The optimum is flat between 0.4 and 0.5; **best Qwen3-1.7B: x1.202**
(top-1 0.775).

### 13.8 Limitations and open issues

* **The reproduction uses random weights.** The real gains are real, the q/k projections are not: real raw q/k
  channels can be uneven before the norm, and the int8 requantisation in front of the norm then loses the small channels
  the norm's large gains amplify. The tool shows it as a score step at `k` (`q`) rather than at `k_norm`; the balance
  cannot remove it (the norm is not invariant to per-channel scaling of its input, only per head), so it would be the
  next item if the cluster rows show it.
* **The balance is one choice of `c_p`** (equal per-pair ranges for q and k); the remaining 0.11-0.14 nats per norm in
  the proxy are the int8 rounding of q and k at that allocation. The q/k norms' outputs must stay int8 (the attention's
  int32 scores need `|q|, |k| <= 128`).
* **TinyLlama's decay** is not explained on the laptop (no q/k norm; nothing else in the builder showed on the tiny or
  realistic models); 13.6's runs 2 and 3 attribute it.
* **Flag thresholds** (0.1 relative, 0.99 cosine, 0.1 nats) come from the tiny models; the JSON lines keep every number,
  so other thresholds need no rerun.
* **The real gains in the test** are layer 0's q/k-norm weights of Qwen/Qwen3-0.6B-Base (256 bf16 values, exact), a
  public checkpoint's constants; the test needs no download.
