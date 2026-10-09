# E2. Sampled generation with a client-chosen seed

Plan item E2 (reviewer objection D1: "only greedy decoding is supported"). Commit 7c98d87
(`token_sampler.py`, the token rule in `protocol.Verifier.check_tokens`, `run_query(sampler=)`,
`cut_protocol.run_cut_query(sampler=)`). Status: done (tests pass locally; GPU runs follow with the others).

## 1. The problem

E1 proves a whole generated response as one prefill: the verifier checks that each generated token is the greedy
choice (the first largest logit) of the claimed logits before it, and the logits are checked like every other claim.
Deployed models usually sample (temperature, top-k). A sampled response is not a function of the logits alone, so
the greedy rule rejects it, and a rule that accepted "any plausible token" would let the provider choose the output.

## 2. The rule

The client picks a fresh 32-byte seed with each query, and the response is the output of a public, exact integer
sampler keyed by that seed (the Gumbel-max trick: sampling from `softmax(l / tau)` is choosing
`argmax_v (l_v / tau + g_v)` with independent Gumbel noise `g_v`):

* claimed logits `z` (int32 claims of the LM head; real logits `z c` for the builder's logit scale `c`);
* an integer weight `a = round(c 2^16 / tau)` (`TokenSampler.for_temperature`);
* noise `G[v] = TABLE[u_v]`, where `u_v` are the 16-bit words of
  `SHAKE-256(b"pvi/sampler/v1" || seed || position)` and
  `TABLE[i] = round(2^16 (-ln(-ln((i + 1/2) / 2^16))))`;
* the token at a position: among the `top_k` largest `z` (all if `top_k = 0`; ties to the smaller index), the first
  `v` maximising `a z_v + G[v]`.

Every step is integer arithmetic (`|z| < 2^29`, `a < 2^24`, `|G| < 2^20`: int64), so prover and verifier get the same
token on any platform.

**The table.** It is a public constant of the protocol. It is computed with floating point (0.09 s) and checked
against its SHA-256 (fixed in the code); on a platform whose `log` disagreed in a rounding case, it is recomputed
exactly with Python's `decimal` (10 s, once per process) and checked again.

## 3. Why it is sound

* The seed must not be the provider's choice: it comes from the client with the query, and under Fiat-Shamir the
  sampler's digest (weight, top-k, seed) is absorbed into the statement (`_absorb_statement`), so a proof is bound to
  that seed.
* The verifier recomputes each generated token from the claimed logits before it. The logits are claims of the LM
  head, checked by the protocol (Freivalds or the column check), so a token that is not the sampler's choice from the
  *true* logits needs a wrong logit, which the protocol rejects except with its usual error. The sampler adds no
  soundness term.
* A provider that wants a different output can only change the logits (rejected) or the seed (not its choice).

The distribution is that of the Gumbel-max trick with 16-bit uniforms quantised to `2^-16` (a fixed, public
approximation of sampling at temperature `tau`; the quantisation is far below the softmax's own int8 resolution).

## 4. Implementation and tests

* `token_sampler.py`: `gumbel_table`, `TokenSampler` (`choose`, `noise`, `digest`, `for_temperature`),
  `sample_tokens` (the provider's generation loop).
* `Verifier.sampler` (set by `run_query(sampler=...)` and `run_cut_query(sampler=...)` for the query; cleared after);
  `check_tokens` uses it for generation graphs (`with_lm_positions`).
* `tests/test_token_sampler.py`: the table's hash, monotonicity and range; the sampler is deterministic, top-1 and a
  very low temperature give greedy, flat logits spread the choices (40 seeds give more than 20 distinct tokens) and
  are uniform over 4,000 positions (each of 8 tokens between 380 and 620 times), top-k restricts; a sampled response
  is accepted in modes C (non-streaming and streaming) and Kpre and with V1, the same tokens are rejected at `token`
  under another seed unless that seed happens to choose them, and a greedy response checked as sampled is rejected
  while it is accepted as greedy.

## 5. What remains

* Stop rules (EOS, length) are a property of the response's length and need no proof beyond the token rule.
* Nucleus (top-p) sampling needs a cumulative sum of exponentials; an exact integer version with the attention's
  exp table is possible, not implemented.
