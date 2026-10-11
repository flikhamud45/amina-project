# F2c. Zero-shot accuracy of the certified integer models

Plan 2 item F2c (`SP2027_PLAN2.md` Sec. 3.2, should, rank 2; reviewer points C-W5 "there are no downstream tasks",
C-Q2, consolidated change 7). Code: `code/experiments/9_quality/zeroshot.py`, test `code/tests/test_zeroshot.py`
(commit 6b31e45). Status: harness done and tested on the laptop; the first cluster job (OPT-125M check, OPT-125M and
OPT-1.3B with both builders) is job 1011383; results go in Sec. 5.

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

[CLUSTER] job 1011383.

## 6. Limitations

* **A re-implementation**, not the harness itself: the formats follow its task files, and the fp16 numbers against the
  published ones are the check that they do. Small differences (the log-softmax's precision, the HellaSwag subset)
  are documented above.
* **OPT only** for now. Qwen3 and Llama join once F2b-Q's builder gives a usable integer model (the harness needs only
  a logits function; the Llama builder's `lean_logits` provides one).
* **The recipe** is F2a's round-1 best, or the paper builder's current one; the certified recipe is fixed after F2a's
  round 2 and D6a's re-digest, and the final rows use it.
