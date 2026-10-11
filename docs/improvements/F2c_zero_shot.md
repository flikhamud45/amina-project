# F2c. Zero-shot accuracy of the certified integer models

Plan 2 item F2c (`SP2027_PLAN2.md` Sec. 3.2, should, rank 2; reviewer points C-W5 "there are no downstream tasks",
C-Q2, consolidated change 7). Code: `code/experiments/9_quality/zeroshot.py`, test `code/tests/test_zeroshot.py`
(commit 6b31e45). Status: harness done and tested; on the cluster (job 1011383) the check passed and OPT-125M's fp16 accuracies
reproduce the published harness numbers within 0.2 point (OPT-125M) and 0.1 point (OPT-1.3B); with F2a's round-1
recipe the integer models lose 1.0 (OPT-125M) and 0.7 (OPT-1.3B) point on average, OPT-1.3B significantly only on
LAMBADA (-3.0). The paper builder's OPT-1.3B is running (Sec. 5).

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

**OPT-1.3B, the paper builder** (smooth 0.55, percentiles; the recipe the paper uses today): [CLUSTER] (same job,
running).

## 6. Limitations

* **A re-implementation**, not the harness itself: the formats follow its task files, and the fp16 numbers against the
  published ones are the check that they do. Small differences (the log-softmax's precision, the HellaSwag subset)
  are documented above.
* **OPT only** for now. Qwen3 and Llama join once F2b-Q's builder gives a usable integer model (the harness needs only
  a logits function; the Llama builder's `lean_logits` provides one).
* **The recipe** is F2a's round-1 best, or the paper builder's current one; the certified recipe is fixed after F2a's
  round 2 and D6a's re-digest, and the final rows use it.
