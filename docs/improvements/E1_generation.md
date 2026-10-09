# E1. Proving a generated response as one prefill

Plan item E1 (future-work item "KV cache", first part). Commits `03ae13d` (protocol, tests) and
`cbb8c05` (`bench.py --gen`). Status: done, measured on an RTX 2080 Ti prover and a Xeon CPU verifier.

## 1. The problem

The paper proved one forward pass: a prompt and the distribution of the next token. A chat answer is
many tokens. Proving each generated token as its own query would repeat the whole prompt every time:
for a prompt of `P` tokens and a response of `G` tokens, `sum_i (P + i)` positions of claims and `G`
column openings. For Llama-2-7B with `P = G = 512` that is about 1.3 TB of claims instead of about 3.3 GB.

## 2. The statement

Given a prompt `x_1..x_P`, the response `y_1..y_G` is the model's **greedy** decoding: `y_i` is the first
largest logit at position `P + i - 1` of the model run on `x || y_1..y_{i-1}`.

Two facts make this one prefill:

1. **Every multiplier is static.** The requantisation multipliers, norm gains and look-up tables are fixed
   at calibration; nothing depends on the input.
2. **Every op except causal attention is per position**, and causal attention at position `t` reads only
   positions `0..t`.

So the integers that a KV-cached decoder computes at position `t` are exactly those of a full causal
prefill of `s = x || y_1..y_{G-1}` at position `t`. The response is therefore proved as **one query on
`s`** (`T = P + G - 1` positions), with the LM head at the last `G` positions instead of the last one.

## 3. The protocol change

* **The graph.** `transformer.with_lm_positions(graph, n)` returns the decoder with every "last position"
  slice (the one before the LM head, and those of a pruned last block) taking the last `n` positions. The
  ops, weights and multipliers are the original graph's; `meta["lm_positions"] = n`. It works on the
  benchmark decoders and on the real-weight graphs of `real_weights.py`.
* **The token rule.** `Verifier.check_tokens(x, claims)`: for a graph with `n > 1` positions, each of the
  first `n - 1` columns of the claimed logits (the LM head's claims, `[V, n]`) must have its first largest
  entry at the next input token; otherwise the query is rejected with the label `"token"`. It runs before
  the other checks in `run_query` and in `verify_streaming`, and costs one argmax over `[V, n]`.
* **Everything else is unchanged:** the claims of all `T` positions, one Freivalds identity per matrix over
  `T` columns, one `chi` and one column opening per response (mode C), the range check, Fiat-Shamir (the
  tokens are part of `x`, which the statement absorbs).
* `transformer.greedy_tokens(graph, prompt, steps)` produces the honest tokens (one forward pass per token,
  no KV cache; the integers do not depend on how they are computed). Generation itself is not part of the
  proof's cost.

## 4. Soundness

**Claim.** The soundness error is the same `epsilon` as for one prefill (one term per matrix), and a
response that is not the greedy decoding is rejected except with probability `epsilon`.

**Argument.** Suppose some `y_i` is not the greedy token. The verifier computes the argmax from the
**claimed** logits at position `P + i - 1`, so the token rule passes only if those claimed logits choose
`y_i`, i.e. differ from the model's logits on the same prefix. That is a wrong claim of the LM head (or of
an earlier op that feeds it), and the first-wrong-layer argument of the soundness theorem applies
unchanged to the graph "prefill of `s` with the LM head at `n` positions": the first wrong layer's input is
correct (recomputed by the verifier), so Freivalds' check (or, under a plan with a transposed LM head, its
column check) fails except with probability `p^-r` per matrix, or the column check of mode C fails.
Every claim and every token is fixed before `chi` (interactively, or under Fiat-Shamir as part of the
transcript), so the prover cannot adapt the response to the challenges.

In Kpre, `chi` is fixed and secret; the verifier could check each token online as its claims arrive. The
reuse bound `Q sum_l p^-r` of the paper applies with `Q` counting verdicts the prover sees.

## 5. Tests (`tests/test_generation.py`, 32 tests)

* The logits at the last `n` positions equal the logits of each prefix query, on all four tiny decoders
  (GPT-, Llama-, OPT- and Qwen-style: learned positions, RoPE, `embed_dim`, GQA and q/k norm), pruned and
  not; `claim_columns` matches the claims.
* An honest response is accepted in mode C (paper plan and auto plan; interactive and Fiat-Shamir; with and
  without the compact encoding), K and Kpre, by `run_query` and by the streaming verifier.
* A wrong token at each generated position (the rest regenerated after it) is rejected with `"token"`.
* Logits forged so that the argmax chooses the wrong token pass the token rule and are rejected by the LM
  head's check (`"freivalds"`, or `"columns_code"` when the plan puts the LM head in the transposed layout).

## 6. Measurements (RTX 2080 Ti prover, Xeon Silver 4114 CPU verifier with 8 threads; 64-token prompt;
optimised protocol with A1 and A2; 3 queries per cell, all accepted; the tampered query rejected in every cell)

| Model, mode | G | prover | verifier | proof | proof per generated token |
|---|---|---|---|---|---|
| GPT-2, C (auto plan) | 1 / 64 / 256 | 0.15 / 0.16 / 0.29 s | 0.21 / 0.30 / 0.59 s | 14.9 / 32.7 / 87.3 MB | 14.9 / 0.51 / 0.34 MB |
| GPT-2, Kpre | 1 / 64 / 256 | 0.06 / 0.10 / 0.20 s | 0.14 / 0.23 / 0.51 s | 10.9 / 28.6 / 82.9 MB | 10.9 / 0.45 / 0.32 MB |
| OPT-1.3B, C | 1 / 64 / 256 | 0.61 / 0.73 / 1.05 s | 0.76 / 1.01 / 2.18 s | 96 / 164 / 369 MB | 96 / 2.6 / 1.44 MB |
| OPT-1.3B, Kpre | 1 / 64 / 256 | 0.15 / 0.21 / 0.64 s | 0.43 / 0.72 / 1.71 s | 60 / 128 / 332 MB | 60 / 2.0 / 1.30 MB |

GPT-2 proves a 256-token response in 0.29 s (1.1 ms per generated token). For comparison (published
numbers on other hardware): zkAgent about 0.30 s per token for GPT-2 (ePrint 2026/199), ZKTorch 2,645 s
per token for Llama-2-7B. Records: `S/sp2027_research/raw_sp2080/`.

## 7. Usage

    python experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 --gen 256 --builds full \
        --modes C:int --policy auto --wire --prune-last --lean --tag _wire

Cells are named with a `_gen<G>` suffix and record `gen`, `positions` and the honest generation time
(`generate`, not part of the proof).

## 8. What remains

* Sampling with a verifier-chosen seed and stop rules (plan E2).
* A verifier KV cache for multi-turn chat (plan E3).
* Llama-2-7B and Qwen3-4B with `P = 512`, `G = 256` on the L40S (final runs).
