# F2c. Zero-shot accuracy of the certified integer models

Plan 2 item F2c (`SP2027_PLAN2.md` Sec. 3.2, should, rank 2; reviewer points C-W5 "there are no downstream tasks",
C-Q2, consolidated change 7). Code: `code/experiments/9_quality/zeroshot.py`, test `code/tests/test_zeroshot.py`
(commit 6b31e45). Status: harness done and tested; on the cluster (job 1011383) the check passed and OPT-125M's fp16 accuracies
reproduce the published harness numbers within 0.2 point (OPT-125M) and 0.1 point (OPT-1.3B); with F2a's round-1
recipe the integer models lose 1.0 (OPT-125M) and 0.7 (OPT-1.3B) point on average, OPT-1.3B significantly only on
LAMBADA (-3.0). The paper builder's current recipe loses 2.8 points (LAMBADA -11.3), which supports F2a's switch to
channel maxima (Sec. 5).

## 1. Why

Perplexity (F2a) says how close the integer model's next-token distributions are to the checkpoint's; reviewers ask
whether the certified model still does the tasks the checkpoint does. The standard answer is the zero-shot suite of
lm-evaluation-harness, scored on the integer model and on the fp16 checkpoint on the same documents.

## 2. Design

**No harness wrapper.** The integer graph is not a Hugging Face model, and the TAU nodes' shared environment must not
change (packages go to the sp2027 folder's `pylib`), so the six tasks are re-implemented from the harness's task files
(version 0.4), with their prompts, choices, metrics and request encoding:

| Task | Data (hub repository, split) | Documents | Prompt and continuation | Metrics |
|---|---|---:|---|---|
| lambada_openai | EleutherAI/lambada_openai, test | 5,153 | every word but the last; a space and the last word | acc (greedy), ppl |
| hellaswag | Rowan/hellaswag, validation | 2,000 of 10,042 (fixed permutation, seed 1234) | `preprocess(activity_label: ctx_a ctx_b.capitalize())`; a space and `preprocess(ending)` | acc, acc_norm |
| piqa | ybisk/piqa, validation (hub parquet conversion) | 1,838 | `Question: {goal}\nAnswer:`; a space and `sol1` / `sol2` | acc, acc_norm |
| arc_easy | allenai/ai2_arc ARC-Easy, test | 2,376 | `Question: {question}\nAnswer:`; a space and each choice | acc, acc_norm |
| arc_challenge | allenai/ai2_arc ARC-Challenge, test | 1,172 | as arc_easy | acc, acc_norm |
| winogrande | allenai/winogrande winogrande_xl, validation | 1,267 | the sentence up to the blank with each option; a space and the rest (partial scoring) | acc |

* **Encoding** (the harness's `_encode_pair`): trailing spaces of the context move to the continuation; context and
  context + continuation are tokenised without special tokens (no BOS, as the harness does for OPT); the
  continuation's ids are the tail of the joint encoding. The model reads context + continuation without its last
  token; the continuation's log-probabilities are read at its positions (at most 2,048 + 1 tokens).
* **acc_norm** divides each choice's log-likelihood by the choice's length in characters.
* **Batches.** Requests are sorted longest first and batched up to `--budget` tokens (2,048), right-padded. For a
  causal model this is exact: no position reads a later one, and no constant of the integer graph depends on the
  input. The harness's own batching does the same for the float model.
* **The integer model.** `--builder f2a` (`opt_builder.build_opt`, the recipe search; default the round-1 best:
  smoothing through the norm gains with channel maxima, no clipping, alpha 0.5, 11-bit probabilities) or
  `--builder paper` (`real_weights.build_opt_from_hf`, the certified builder); `--opts` replaces the recipe.
  Calibration on four 2,048-token windows of WikiText-2 *train* (F2a's), so no evaluated text is seen. The logits are
  the integer graph's at every position (its "last position" slice skipped), times the logit scale.
* **The reference.** The same checkpoint in fp16 on the same GPU (bf16 or fp32 by flag), log-softmax in float32 (the
  harness takes it in the model's dtype; float32 is more accurate, and the difference is far below a point).
* **Statistics.** Per document, correctness and the chosen answer; the paired difference (integer minus fp16) with a
  95% bootstrap interval over documents (2,000 resamples), and the share of documents on which both choose the same
  answer. Every record keeps the per-document vectors, so draws and runs can be pooled.
* **Large models.** `--no-ref` and `--no-int` split the two models over two processes (OPT-6.7B's fp16 copy and its
  integer graph do not fit one 11 GB card together); the records pool by document.

The scored workload per model: 25,618 requests, about 1.70 M tokens (LAMBADA 0.43 M, HellaSwag 0.63 M, PIQA
0.13 M, ARC-e 0.29 M, ARC-c 0.17 M, WinoGrande 0.05 M; OPT tokenizer). At the A7 pilot's batched integer-forward
rate on the 2080 Ti (about 0.17 ms per token for OPT-1.3B) that is about 5 minutes per model, plus the LM head at
every position.

## 3. Validation

* **Laptop (`tests/test_zeroshot.py`, 9 passed):** the harness's HellaSwag preprocessing; every task's prompt and
  continuation on hand-made rows (including WinoGrande's partial scoring and ARC's label lookup); the fixed HellaSwag
  subset; `_encode_pair`'s trailing-space rule; scoring independent of batching and padding, and equal to a direct
  log-softmax; `acc`, `acc_norm` and LAMBADA's perplexity; the paired interval; and, on tiny random OPTs from both
  builders, right padding exact on the integer graph (every padded row's log-likelihood and greedy flag equal to the
  request run alone) and the harness's logits equal to `real_weights.real_logits(..., all_positions=True)`.
* **Data:** every loader was run against the hub's files on the laptop (document counts as in the table; longest
  request 254 tokens).
* **Cluster `--check`** (OPT-125M): the same padding exactness and logit equality on the real model and real
  documents (HellaSwag and PIQA), and the fp16 model's padding difference (rounding only).
* **The plan's second criterion**, the fp16 accuracies within 0.5 point of the harness's published numbers, is read
  from the `ref` records (Sec. 5). HellaSwag is a 2,000-document subset, so it is compared within its interval.

## 4. How to run

    cd code
    PYTHONPATH=src python -m pytest -q -o addopts= -p no:cacheprovider tests/test_zeroshot.py
    PYTHONPATH=src python experiments/9_quality/zeroshot.py --model facebook/opt-125m --check --device cuda
    PYTHONPATH=src python experiments/9_quality/zeroshot.py --model facebook/opt-1.3b --device cuda --builder f2a \
        --out zeroshot_opt-1.3b.jsonl
    PYTHONPATH=src python experiments/9_quality/zeroshot.py --model facebook/opt-1.3b --device cuda --builder paper \
        --opts '{"norm_gain": 4, "smooth": 0.55, "smooth_pct": 99.9, "pct": 99.99, "pct_att": 99.99}'

## 5. Results

Job 1011383 (s-004, RTX 2080 Ti, git 6b31e45). Tests: 9 passed. `--check` on OPT-125M: the integer model's padded
rows equal the requests run alone exactly (24 HellaSwag and PIQA requests, max |difference| 0), the fp16 model's
within rounding (0.028 nats, greedy flags all equal), and the integer logits equal `real_logits`.

**OPT-125M, F2a's round-1 recipe** (smoothing with channel maxima, alpha 0.5, 11-bit probabilities; digest
`d6889a56...`). Accuracy in percent; the difference is integer minus fp16 with its 95% paired bootstrap interval;
"same" is the share of documents on which both pick the same answer:

| Task | n | fp16 acc / acc_norm | integer acc / acc_norm | difference acc (95% CI) | same | published fp16 (harness) |
|---|---:|---|---|---|---:|---|
| LAMBADA | 5,153 | 37.86 (ppl 26.0) | 36.89 (ppl 28.9) | -0.97 [-1.84, -0.12] | 0.901 | 37.9 |
| HellaSwag (2,000) | 2,000 | 28.90 / 29.40 | 29.00 / 29.65 | +0.10 [-0.65, +0.85] | 0.935 | 29.2 / 31.3 (all 10,042) |
| PIQA | 1,838 | 63.06 / 62.08 | 60.72 / 60.83 | -2.34 [-3.59, -1.14] | 0.928 | 63.0 / 62.0 |
| ARC-e | 2,376 | 43.52 / 39.94 | 41.58 / 37.08 | -1.94 [-3.16, -0.72] | 0.865 | 43.5 / 39.9 |
| ARC-c | 1,172 | 18.94 / 22.78 | 19.45 / 23.21 | +0.51 [-0.68, +1.79] | 0.881 | 19.0 / 22.9 |
| WinoGrande | 1,267 | 50.12 | 48.86 | -1.26 [-4.03, +1.50] | 0.755 | 50.3 |

* **Validation.** The fp16 accuracies reproduce the harness's published OPT-125M numbers within 0.2 point on every
  task scored in full (the plan's criterion is 0.5); HellaSwag is the fixed 2,000-document subset, whose accuracy is
  within its sampling error of the full set's.
* **The integer model** loses 1.0 point on average over the six tasks (acc), at most 2.3 (PIQA); three of the six
  differences are significant at 95%. With this recipe OPT-125M's perplexity ratio at 2,048 tokens is x1.10 (F2a
  Sec. 3.3).

**OPT-1.3B, F2a's round-1 recipe** (as above):

| Task | n | fp16 acc / acc_norm | integer acc / acc_norm | difference acc (95% CI) | difference acc_norm (95% CI) | same | published fp16 (harness) |
|---|---:|---|---|---|---|---:|---|
| LAMBADA | 5,153 | 57.85 (ppl 6.64) | 54.82 (ppl 7.80) | -3.03 [-3.78, -2.29] | | 0.927 | 57.9 |
| HellaSwag (2,000) | 2,000 | 39.75 / 52.55 | 40.05 / 52.60 | +0.30 [-0.20, +0.85] | +0.05 [-0.80, +0.90] | 0.977 | 41.5 / 53.7 (all 10,042) |
| PIQA | 1,838 | 71.82 / 72.42 | 71.22 / 72.20 | -0.60 [-1.52, +0.27] | -0.22 [-1.09, +0.71] | 0.960 | 71.7 / 72.3 |
| ARC-e | 2,376 | 57.11 / 51.01 | 56.57 / 50.67 | -0.55 [-1.56, +0.42] | -0.34 [-1.18, +0.42] | 0.918 | 57.0 / 51.0 |
| ARC-c | 1,172 | 23.38 / 29.52 | 23.81 / 28.33 | +0.43 [-0.94, +1.79] | -1.19 [-2.39, +0.00] | 0.908 | 23.4 / 29.6 |
| WinoGrande | 1,267 | 59.75 | 58.96 | -0.79 [-2.60, +1.03] | | 0.885 | 59.7 |

* The fp16 accuracies again reproduce the published numbers within 0.1 point on every task scored in full.
* **Caveat on the "published" column (both models):** the values are the commonly reported lm-evaluation-harness
  zero-shot results for these checkpoints, written down from the literature without a single cited table; before
  the paper quotes the agreement, they must be checked against a citable source (or the harness itself run once on
  fp16, which needs its environment).
* The integer OPT-1.3B loses 0.7 point of accuracy on average over the six tasks (the plan expected at most 1 point
  for a model within x1.03-1.05 in perplexity; this recipe gives x1.029 at 2,048 tokens). Only LAMBADA's loss is
  significant: 3.0 points (its continuation perplexity 6.64 -> 7.80), the task that reads the most exact
  next-token probabilities. On the five multiple-choice tasks every interval contains 0 (ARC-c's acc_norm at its
  edge), and the integer model picks the same answer as fp16 on 89-98% of the documents.

**OPT-1.3B, the paper builder** (`real_weights.build_opt_from_hf`, norm gain 4, SmoothQuant alpha 0.55 with the
99.9th-percentile channel statistic and 99.99th-percentile clipping: the recipe of the paper and of D6a today;
calibrated here on F2a's four 2,048-token train windows; digest `d69f1be5...`):

| Task | integer acc / acc_norm | difference acc (95% CI) | difference acc_norm (95% CI) | same |
|---|---|---|---|---:|
| LAMBADA | 46.57 (ppl 11.75) | **-11.27 [-12.25, -10.34]** | | 0.861 |
| HellaSwag (2,000) | 39.35 / 51.75 | -0.40 [-1.10, +0.35] | -0.80 [-1.75, +0.20] | 0.958 |
| PIQA | 71.00 / 71.44 | -0.82 [-1.85, +0.16] | -0.98 [-1.96, +0.05] | 0.953 |
| ARC-e | 55.26 / 48.86 | -1.85 [-3.03, -0.67] | -2.15 [-3.16, -1.18] | 0.878 |
| ARC-c | 23.21 / 28.50 | -0.17 [-1.54, +1.19] | -1.02 [-2.30, +0.17] | 0.880 |
| WinoGrande | 57.62 | -2.13 [-4.03, -0.39] | | 0.889 |

(fp16 as in the table above.) **The paper builder's recipe loses 2.8 points on average, against 0.7 for F2a's
round-1 recipe** (smoothing with channel maxima and no clipping), and 11.3 points on LAMBADA (continuation perplexity
6.64 -> 11.75, against 7.80). This is F2a's finding in downstream terms: the percentile statistics, estimated on
train text, transfer badly to test text (F2a Sec. 3.3: x1.04-1.28 in perplexity, depending on the calibration draw,
against x1.029 for channel maxima). It supports switching the certified recipe (the paper builder's default and
D6a's digests) to channel maxima without clipping, which F2a's round 2 (alpha 0.6 / 0.7, OPT-6.7B; job 1011066)
completes. Note the calibration here is F2a's (train windows), not D6a's (one 128-token test window); the paper's
own model must be rerun once the recipe is fixed.

### 5.1 The certified recipe (smooth-max alpha 0.6; job 1012440, s-005, git 1801809)

After F2a's decision the paper builder's default is `real_weights.CERTIFIED_RECIPE` (calibration: F2a's four
2,048-token train windows, as above).

**OPT-125M** (digest `16de86ea...`): difference to fp16, accuracy (95% CI): LAMBADA -3.12 [-3.92, -2.33] (ppl 26.0 ->
33.3); HellaSwag -0.95 [-1.70, -0.25]; PIQA -1.74 [-2.94, -0.54]; ARC-e -1.43 [-2.61, -0.21]; ARC-c -0.17 [-1.54, +1.19];
WinoGrande +1.26 [-1.34, +3.87]. **Mean -1.03 points**, as with alpha 0.5 (-0.97); individual tasks move by up to
2 points between the two (within their intervals, except LAMBADA, -1.0 at 0.5 and -3.1 at 0.6).

**OPT-1.3B** (digest `d9daa017...`; fp16 as in the tables above):

| Task | integer acc / acc_norm | difference acc (95% CI) | difference acc_norm (95% CI) | same |
|---|---|---|---|---:|
| LAMBADA | 55.81 (ppl 7.56) | **-2.04 [-2.74, -1.40]** | | 0.937 |
| HellaSwag (2,000) | 39.25 / 52.25 | -0.50 [-1.10, +0.05] | -0.30 [-1.10, +0.55] | 0.975 |
| PIQA | 71.00 / 72.03 | -0.82 [-1.69, +0.05] | -0.38 [-1.25, +0.49] | 0.965 |
| ARC-e | 56.99 / 50.67 | -0.13 [-1.01, +0.76] | -0.34 [-1.10, +0.42] | 0.933 |
| ARC-c | 23.21 / 28.75 | -0.17 [-1.37, +1.02] | -0.77 [-2.05, +0.43] | 0.923 |
| WinoGrande | 58.48 | -1.26 [-3.08, +0.47] | | 0.896 |

**Mean -0.82 points** (alpha 0.5: -0.73). The certified recipe trades a little on the multiple-choice tasks for a
third less loss on LAMBADA (-2.0 against -3.0; continuation perplexity 6.64 -> 7.56, against 7.80), the task that
reads the exact next-token probabilities, in line with its lower perplexity ratio at 2,048 tokens (F2a round 2:
x1.026). LAMBADA's loss is again the only significant one; on the five multiple-choice tasks every interval contains
0 and the integer model picks the fp16 answer on 90-98% of the documents (89-98% at alpha 0.5).

**OPT-6.7B** (digest `0b757108...`; build 1,783 s). The fp16 copy and the integer graph do not fit one 11 GB card
together, so the integer model ran alone in job 1012440 and the fp16 reference separately, split over two 2080 Ti
(`--no-int --ref-gpus 2`, `zeroshot.split_opt`; job 1012655, s-004, git eefc100, 11 tests passed, 11 min for the six
tasks); `--pair` gives the paired differences on the same documents:

| Task | fp16 acc / acc_norm | integer acc / acc_norm | difference acc (95% CI) | difference acc_norm (95% CI) | same | published fp16 (harness) |
|---|---|---|---|---|---:|---|
| LAMBADA | 67.69 (ppl 4.25) | 67.16 (ppl 4.33) | -0.52 [-1.03, -0.04] | | 0.965 | 67.7 (ppl 4.25) |
| HellaSwag (2,000) | 48.85 / 66.55 | 48.65 / 66.30 | -0.20 [-0.65, +0.30] | -0.25 [-0.85, +0.35] | 0.984 | 50.5 / 67.2 (all 10,042) |
| PIQA | 76.39 / 76.55 | 76.33 / 76.61 | -0.05 [-0.60, +0.49] | +0.05 [-0.65, +0.76] | 0.987 | 76.3 / 76.5 |
| ARC-e | 65.61 / 60.06 | 65.74 / 60.65 | +0.13 [-0.59, +0.80] | +0.59 [-0.04, +1.22] | 0.963 | 65.6 / 60.1 |
| ARC-c | 30.55 / 34.81 | 31.06 / 34.39 | +0.51 [-0.60, +1.62] | -0.43 [-1.45, +0.51] | 0.939 | 30.5 / 34.6 |
| WinoGrande | 65.35 | 65.82 | +0.47 [-0.87, +1.89] | | 0.935 | 65.3 |

**Mean +0.06 points**: the certified OPT-6.7B is at fp16's accuracy. LAMBADA's -0.5 is the only interval that
excludes 0, and only just; the integer model picks fp16's answer on 94-99% of the documents. With the other two
models, the loss falls with size: mean -1.03 / -0.82 / +0.06 and LAMBADA -3.1 / -2.0 / -0.5 for OPT-125M / 1.3B /
6.7B, in line with the perplexity ratios at 2,048 tokens (x1.076 / x1.026 / x1.008). The fp16 numbers agree with the
commonly reported harness values within 0.2 point on the tasks scored in full (the same caveat on that column as
above).

## 6. Limitations

* **A re-implementation**, not the harness itself: the formats follow its task files, and the fp16 numbers against the
  published ones are the check that they do. Small differences (the log-softmax's precision, the HellaSwag subset)
  are documented above.
* **OPT only** for now. Qwen3 and Llama join once F2b-Q's builder gives a usable integer model (the harness needs only
  a logits function; the Llama builder's `lean_logits` provides one).
* **The recipe.** Sec. 5 has F2a's round-1 recipe and the earlier paper builder; Sec. 5.1 is the certified recipe
  (F2a's decision), whose rows are the ones the paper quotes.
* **Two GPUs for OPT-6.7B.** `split_opt` places whole modules on each device and moves only activations, so each
  operation runs as on one device (the test compares the split and unsplit logits exactly on the CPU); on two GPUs
  the fp16 results may still differ from a one-device run in the last bits of the kernels' rounding, as fp16 runs on
  different cards do.
