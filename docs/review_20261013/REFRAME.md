# Reframe of the paper: thesis, title options, abstract, and the edits that follow

Proposal for the decision "Reframe + new title" (13 Oct). It follows the high-level review (REVIEW.md, H1-H5,
M1-M3, M10) and changes no result. The title and abstract freeze on 10 Nov; the edits below can land in the paper
after the code freeze, since they change no measurement.

## 1. The thesis

One sentence that links the two halves:

> No spot-checking verifier of a committed trace can be fully sound: one changed value, with the later layers
> recomputed honestly, escapes every verifier that reads a few positions per layer (a lower bound, witnessed on
> real LLMs). Full soundness needs checks that mix every row of each weight matrix; we give the cheapest
> hash-only way to do that and measure what it costs and for whom it pays.

Under this framing the defence does not have to win on speed. It has to be complete, honest and measured on one
platform. The attack is the witness of the bound, not a discovery of the 1/N gap (Anchuri et al. state the gap).

## 2. Title options

1. **Spot-Checks Are Not Proofs: A Lower Bound for Sampled Proofs of Inference and a Hash-Only Fully Sound
   Protocol.** States the thesis; no "breaking" claim; long.
2. **Certified but Compromised: Why Spot-Checked Proofs of Inference Cannot Be Sound, and What Full Soundness
   Costs.** Keeps the current hook and the paper's identity; the subtitle carries the thesis.
3. **One Neuron Is Enough: Spot-Checked Proofs of LLM Inference Are Not Sound, and the Price of Soundness.**
   The most memorable; leads with the attack, which is the paper's most visible result.

Recommendation: option 2. It keeps the name reviewers of earlier drafts know, drops "Breaking and Fixing" (which
over-claims against a target that states 1/N itself, review H2), and announces both the impossibility and the
measured price.

## 3. Abstract (draft, about 200 words)

The macros are the pipeline's (`experiments/paper/macros.py`); their values come from the final run.

> Lightweight proofs of inference let a client check an untrusted provider by re-checking a sample of a committed
> execution trace. We show that no such spot-checker can be fully sound. A provider that changes one neuron and
> recomputes the later layers honestly leaves a single inconsistent site; we prove that every verifier making $q$
> queries per layer to a commitment of locality $d$, even an adaptive one that reads the query's whole trace,
> accepts such a forgery on some network with probability at least $1-qd/N$ in a layer of width $N$. The bound is
> tight: on OPT-6.7B one edited neuron changes the next token for 40 of 40 prompts, edits within every activation's
> calibrated range still pass, and no sampling rule we tried closes the gap. Full soundness needs checks that mix
> every row of each weight matrix. We give the cheapest hash-only protocol that does so: Freivalds' check of every
> matrix product of an exact int8 model against a linear-code commitment of its weights, with no SNARK, trusted
> hardware or trusted setup. A client holding only the 32-byte roots checks any answer with soundness error
> $2^{-128}$; a CPU client with a key precomputed from the weights verifies a 2,048-token Llama-2-7B prompt
> \CpuRatioN{} times faster than re-executing the model on the same CPU. The price is a proof that grows with the
> prompt (\ProofLlamaTwoK{} at 2,048 tokens) and a provider \ServeOverhead{} slower than unverified serving; a
> variant makes proofs \VoneRatioReal{} times smaller on the real OPT-6.7B.

Differences from the current abstract:
- Opens with the impossibility, not with the 1/N calculation (which is the target's own remark).
- The headline number is tied to the mode it belongs to (Kpre: a key precomputed from the weights; review H1) and
  to the certified integer model on the same CPU; mode C's capability is stated separately, without a number.
- The price (proof size, provider overhead) is in the abstract (review H4).
- "4.7 times faster than the fastest exact re-execution" becomes `\CpuRatioN`, defined by the plan's rule: the
  fastest of the exact re-execution variants (each computing only what the pruned graph computes) over the CPU
  Kpre verifier, Llama-2-7B at 2,048 tokens, the same 8 EPYC threads (HD-02, F12, F15).

## 4. Introduction

- **Thesis paragraph** (after the paragraph on Anchuri et al.; replaces the current motivation for the defence):

  > The gap is not an artefact of one protocol. A sound check of a weight product must depend on every row of the
  > matrix, because a change confined to one row is invisible to any check that reads fewer rows than the matrix
  > has (Section~\ref{sec:sampling}). Spot-checkers read a few rows by design, so none of them is fully sound
  > against a provider that chooses where to deviate. A linear-code commitment answers queries that mix all rows;
  > combined with Freivalds' check of each product it is fully sound with a hash function alone. The question is
  > then not whether full soundness is possible but what it costs, and the rest of the paper measures that price on
  > real models.

- **Contributions, reordered** (review H2):
  1. A lower bound for adaptive, value-aware, trace-reading spot-checkers at any locality, tight at every
     locality, and an SoK of nine schemes under it.
  2. The bound's witness on real LLMs: one FFN neuron of OPT-6.7B, a backdoor, edits within calibrated activation
     ranges, and the adaptive sampler game.
  3. A hash-only, fully sound protocol (Freivalds against a linear-code commitment of an exact int8 model) in three
     client modes, with Fiat-Shamir, generation and an optional proof-size variant (V1).
  4. Its price on one platform: prover, verifier and proof against re-execution and against unverified serving,
     and the regime in which it pays.

- **Deployment regime** (review H4; one paragraph in the introduction, mirrored in the Discussion):

  > Who gains. A CPU-only auditor or gateway with a fast link to the provider: a Kpre verifier checks a 2,048-token
  > Llama-2-7B prompt \CpuRatioN{} times faster than re-executing it, a gain that holds above \BreakEvenGbps{} Gb/s
  > of link speed. Clients that make few short queries or check many models: mode C needs only 32 bytes of roots per
  > model. And on-demand audits of chosen answers: a provider that alters one chosen answer passes a sampled audit
  > with probability about one minus the audit rate, but passes ours with probability $2^{-128}$. The price is paid
  > by the provider: \ServeOverheadRange{} unverified fp16 serving, against below 7\% for a GPU TEE and 12--14\% for
  > CommitLLM, which give weaker guarantees.

## 5. V1 in the body (review M2)

About 0.8 page in the body: the idea (send the requantised int8 values instead of the wide claims, prove the
windows with logUp-GKR), one sentence of soundness, the bytes against v0 (`tab:eval-v1` reduced to the three
rows that carry the claim: Llama-2-7B at 64 and 2,048 tokens, the real OPT-6.7B) and its price (prover and
verifier time). Framed as the communication floor of the recompute design: how far the proof shrinks without
committing activations. The window lemmas, the GKR details and the per-phase verifier move to the extended version.

## 6. Other edits the reframe implies

- **H5**: one certified model per architecture in every table: the real-weight cells come from the certified OPT
  recipe (the run package does this); Qwen3-4B is a shape-only cost row (its certified model is x5.3 perplexity),
  marked so in `tab:eval-llm` and the Maverick note.
- **M1**: the LLM attack opens with the adversary's choice of layer; the bound's reach to transformers (where the
  locality argument applies, where it needs the residual stream) is stated.
- **M3**: the setup proof moves out of the contributions to a paragraph and the extended version (it lies outside
  the threat model: the commitment is public and recomputable).
- **M10**: the body is full; each addition above is paid for by moving the V1 details, the improvement table and
  the setup proof to the extended version.

## 7. Decisions for you

1. The title (Sec. 2).
2. An external CPU baseline (review H1): llama.cpp prefill at Q8_0 and bf16 on the same 8 EPYC threads, labelled
   "not exact, a different model". It is an external binary (no change to our code), one extra job of about an
   hour, but the friend's machine would have to build llama.cpp. If its ratio is about 1x or below, the claim stays
   against exact re-execution only. Add it, or keep exact re-execution as the only baseline?
3. `tab:quality-attr` (the attribution table of the extended version) has no script in the artifact: the scripts
   that produced it were not carried over. Port them (about a day, plus a run), or drop the table and keep its
   conclusion as a sentence with the Llama attribution that the artifact does reproduce?

## 8. Status on 14 Oct (submission branch `sp2027-artifact`, 826a1be)

- **Exact re-execution** (done in the paper and the pipeline): the baseline of `\CpuRatioN`, `tab:options`, the
  break-even and the long prompts is the integer forward pass, bit-exact with the claims (the faster of its int8 and
  exact fp32 GEMM paths). fp32, bf16 and dynamic int8 inference of the same shapes are inexact: the paper reports
  them next to it (`\CpuInexact*`, `tab:options` note a), never as "exact". Before, fp32 was counted as exact
  while a faster dynamic-int8 engine was excluded as non-exact. On the earlier Xeon measurements the ratio becomes
  7.2 (142 s against 19.6 s) instead of 4.7, and the fastest inexact engine (dynamic int8, 56 s) is 2.9 times the
  verifier's time. Reverting is one line (`experiments/paper/tables.py`, `EXACT`) plus the wording.
- **Length** (Tectonic, Times): the submission is 19 pages: body 1-13 and the first three lines of page 14,
  references 14-16, appendix 16-19 (page 19 about 45% full). The 2027 call (`docs/submission/CFP_2027.md`) allows
  13 pages of text and 18 in all and rejects over-length papers without review, so the body is three lines over
  and the total about half a page over.
  The 20 remaining `\todo` boxes take part of that space. The reframe (Sec. 5-6) has to fit in the same 13 pages
  and bring the total to 18: the edits of M10 (V1's details, the improvement table and the setup proof to the
  extended version) are the room for it.
- **Every printed number** is now a macro or table cell filled from the run's records (or computed from the code
  under the run's plan), with the current values as fallback. The reframe's draft abstract above uses macro names
  of an earlier draft (`\ProofLlamaTwoK`, `\ServeOverhead`, `\VoneRatioReal`, `\BreakEvenGbps`,
  `\ServeOverheadRange`); the pipeline's names are `\ProofLlamaLongC`, `\ServeLlamaKpreOverUTwo`,
  `\VOneRatioRealOptL`, `\BreakevenLinkRange` (`paper_sp2027/fallback/numbers.tex` lists them all).
