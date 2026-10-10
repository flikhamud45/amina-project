# Cut plan: from the 29-page draft to the S&P 2027 limit (13 + 5 pages)

Status: plan for the condensation pass of 2026-10-10 (plan item H1). Inputs: `REVIEW.md` (mock review, consolidated
changes 2, 6, 8, 9, 10 and the smaller issues), the S&P 2027 call for papers, and the page map of the draft.

## 1. The rules we must meet (S&P 2027 call, cycle 2: abstract registration 10 Nov 2026, paper 17 Nov 2026)

* At most 13 pages of text, then at most 5 pages of references and appendices, 18 pages in total, IEEEtran
  `conference,compsoc`, US letter, no changes of margins, fonts or spacing ("egregious space scrunching" can mean
  rejection without review). Everything after page 13 must be clearly marked as appendix. Reviewers need not read
  appendices.
* Title, abstract and author list cannot change after abstract registration (10 Nov).
* Anonymised artifact repositories are encouraged, and theory papers are encouraged to put proofs there. We therefore
  build two PDFs from the same body: the **submission** (body + short appendix, 18 pages) and the **extended
  version** (same body + the full appendices + material cut from the body), to be placed in the anonymous artifact
  repository and cited as `\cite{fullversion}`.
* Generative-AI use must be disclosed in the HotCRP field, not in the body. The LLM-agent review of V1 is therefore
  not described in the body.
* Vulnerabilities must be disclosed by the rebuttal deadline at the latest; the paper states the disclosure as done
  only once it is done.

## 2. Page map of the draft and the budget

Measured in non-comment source words (`grep -v '^\s*%' file | wc -w`); the draft has about 1,000 such words per page
including its tables and figures.

| File | Draft words | Draft pages | Budget words | Budget pages |
|---|---:|---:|---:|---:|
| `intro.tex` (Introduction + Related Work) | 2,817 | 3.5 | 1,900 | 2.0 |
| `attack.tex` | 3,509 | 3.0 | 2,350 | 2.4 |
| `defence.tex` | 3,127 | 4.0 | 2,450 | 2.6 |
| `v1.tex` | 3,247 | 2.0 | 1,700 | 1.4 |
| `generation.tex` (new; generation part of `generation_quality.tex`) | about 1,300 | 1.8 | 750 | 0.8 |
| `quality.tex` (new; quality part, becomes a subsection of the evaluation) | about 850 | 1.2 | 500 | 0.5 |
| `evaluation.tex` | 3,426 | 3.5 | 2,500 | 2.6 |
| `discussion.tex` (Discussion, deployment, Conclusion) | 1,016 | 1.0 | 900 | 0.8 |
| **Body** | 19,300 | 20 | 13,050 | 13.1 |
| `appendix_short.tex` (new, submission only) | -- | -- | 2,600 | 2.7 |
| References | -- | 2.3 | trim | 2.3 |

## 3. Global decisions (every section follows these)

1. **One body, two appendices.** Material removed from the body is not lost: each section writes it, in
   appendix-ready LaTeX, to `sections/ext_<section>.tex` (extended version only). The full appendices
   `appendix_attack.tex`, `appendix_defence.tex`, `appendix_v1.tex` stay unchanged for the extended version. The
   submission's appendix is the new `appendix_short.tex`.
2. **Labels.** Keep every existing label of material that stays in the body. A label that the body references and
   that lived in a full appendix must also be defined in `appendix_short.tex` (same name): `app:attack`, `def:H`,
   `cor:surprise`, `prop:worst`, `prop:blind`, `prop:amp`, `rem:merkle`, `rem:pointer`, `rem:scope`, `tab:aligned`,
   `app:defence`, `app:semantics`, `app:sound`, `app:untrusted`, `app:v1`, unless the body stops referencing it.
   The body must not reference labels that exist only in `ext_*.tex`; refer to "the extended
   version~\cite{fullversion}" instead.
3. **No duplicated results.** The attack's measurements are reported once, in Section 3.5 (the evaluation's
   Section 7.2 is removed; one sentence and a pointer remain, including the sentence that reconciles "2%" (worst case
   over the attacks found against each sampler) with "9.2%" (the contribution-weighted sampler against the plain
   attack)). V1's costs are reported once, in the evaluation (Section 7.4); Section 5 keeps at most two sentences
   on cost. The introduction gives at most four numbers on costs.
4. **Positioning (review change 6).**
   * "Dichotomy" is dropped from titles and claims. The result is "a lower bound for spot-checking verifiers and a
     matching upper bound at full locality"; the label `thm:dich` stays. Intermediate locality is stated as open.
   * The lower bound is "a formalisation, for adaptive, value-aware verifiers that read the whole trace, of a
     folklore calculation"; say what is new against holographic proofs and PCPPs in one sentence.
   * The defence is "the evaluation protocol of a linear-code polynomial commitment (Ligero/Brakedown) applied to
     every matrix product of the model, with the commitment's testing phase amortised into a one-time setup proof".
     The split commitment (Eq. `eq:split`) is motivation for that testing phase, not a finding.
   * "Attention included" becomes "attention recomputed by the verifier from checked values".
   * TensorCommitments: "its stated goal of negligible soundness is not achieved by its layer-selection rule"; do
     not put it on a par with the path protocol's explicit full-soundness statement.
   * "Halves the proof" becomes "2.0-2.3x smaller with random weights, 1.7-1.8x on real OPT-125M at 512 tokens".
   * The abstract drops "46 to 32,000 times faster than published zkSNARK provers"; the comparison in the body is
     stated with its hardware and guarantee caveats.
5. **Root of trust (review change 10, plan D6).** The main path for the commitment is: the certified int8 model is
   derived from the public fp16 checkpoint by a deterministic, public quantisation recipe with fixed calibration
   data, so anyone can recompute `C_M`. This goes into the threat model (Section 3.1/4). The setup proof for
   untrusted commitments becomes the option for when `C_M` cannot be recomputed; Table `tab:setup` moves to the
   extended version, its key numbers stay in one sentence, and its time is a `\todo`.
6. **Cryptographic precision (review change 9).** A compact formal definition of the verifier class (query model,
   how prover messages such as folded rows are modelled, what is excluded: verifiers assisted by a succinct argument)
   goes into the body at the start of the lower-bound subsection. One concrete-security table (lambda, beta, r for
   interactive and Fiat-Shamir, t_j, s, t^0_j, extension degree 8, grinding bits, hash lengths) for Llama-2-7B at
   lambda = 128 goes into `appendix_short.tex`, referenced from Section 4. Mark `\todo{verify}` where a cited
   theorem number or radius (BCIKS20 Thm. 1.6; Ligero lemma d/4 vs d/3) has not been checked against the source.
7. **Ethics hygiene (review change 8).** Remove "in our earlier evaluation" (anonymity). Do not describe the LLM-agent
   review in the body. The disclosure sentences stay conditional with their `\todo`.
8. **Quality** moves out of Section 6 into the evaluation (`quality.tex`, \input by `evaluation.tex`). The
   leave-one-out/add-one attribution table moves to the extended version; the body keeps the smoothed-gains rows.
9. **Generation and sampling** (Section 6) keeps the token rules, the soundness statement and the sampler's
   construction; implementation detail goes to the extended version. State the sampler's truncation (a token more
   than about 14.3 below the top temperature-scaled logit cannot be drawn) in one sentence.
10. **Tables** that stay in the body: `tab:sok`, Fig. `fig:overview`, Fig. protocol (if it fits), `tab:gen` (compact),
    `tab:quality` (bottom half only), `tab:eval-llm`, `tab:eval-v1`, `tab:eval-reexec`, `tab:eval-compare`; `tab:v1` in Section 5 only if it does not duplicate `tab:eval-v1` (otherwise merge them). Table
    `tab:eval-improve` (each improvement against the previous code path) moves to the extended version with a
    two-sentence summary in the body. Tables `tab:setup`, `tab:aligned`, the attribution table: extended version.
11. **Todos.** Keep every `\todo` whose number is still to be measured, but shorten its text (e.g. `\todo{L40S}`).
    Delete todos made obsolete by the cut.
12. **Style.** Formal, plain scientific English, as in the draft: short declarative sentences, no marketing words,
    no em dashes for asides. Keep `\para{}` paragraph heads. Do not change macros in `main.tex`; new macros go
    into the section with `\providecommand`.
13. **Bibliography.** Do not edit `references.bib` or `extra.bib`. If a section needs a new entry (GPU confidential
    computing, DeepBrake, ReedWeave, the full version), it uses a key and reports the BibTeX entry in its output;
    the integrator adds it. Unverified entries are marked `% [U]`.

## 4. Per-section instructions

**Introduction + Related Work (`intro.tex`, 1,900 words).** Open with the setting and the target client in the
first paragraph: an open-weight model served by an untrusted provider, verified by a client that holds a 32-byte root
per matrix (mode C) or a small preprocessed key (mode Kpre), has no GPU and does not store the weights. Then one
paragraph each on the attack, the lower bound, the defence and its four extensions, and costs (at most four numbers:
the prover's speed, the proof size and V1's reduction, the verifier against re-execution). Keep Figure 1. Contributions
as four short bullets; recount "nine schemes" (the table combines TOPLOC and SVIP, and Proof-of-Learning is not an
inference scheme). Related Work: zkSNARKs (one paragraph), spot-checking (one), trusted hardware including GPU
confidential computing (H100-class enclaves; cite one measurement study, key `gpucc`, report the entry), Freivalds-based
checks and linear-code commitments including LAMP, DeepBrake and ReedWeave (keys `deepbrake`, `reedweave`, `% [U]`),
and "Where we fit" in three sentences. Also return a proposed abstract of at most 170 words with at most four
numbers, following decision 4.

**Attack (`attack.tex`, 2,350 words).** 3.1 threat model (with the root-of-trust sentence of decision 5 if it fits
better here than in Section 4), 3.2 the path protocol and the edit, 3.3 the verifier model as a compact Definition and
Theorem `thm:guess` with a proof idea of two or three sentences, 3.4 the worst case and the matching upper bound
(Theorem `thm:dich`, retitled), 3.5 measured attacks (the only place for attack numbers; MNIST, CIFAR, ResNet-18, OPT-6.7B,
samplers, paths needed), 3.6 the systematisation with Table `tab:sok` and the disclosure paragraph. Move to
`ext_attack.tex`: long remarks, the amplification/blind propositions' discussion, extra measurements.

**Defence (`defence.tex`, 2,450 words).** 4.1 the exact integer model in one paragraph (details: `app:semantics` in
the short appendix in compressed form, full in the extended version), 4.2 the commitment, 4.3 the protocol and
Theorem `thm:sound`, with the cost formula, 4.4 non-interactive proofs (the statement binding, the round-by-round
argument in a few sentences), 4.5 untrusted commitments reframed by decisions 4 and 5. Move Table `tab:setup` and
the parameter derivations to `ext_defence.tex`.

**V1 (`v1.tex`, 1,700 words).** Keep the observation, windows and exceptions, the windowed logUp with integer
multiplicities (Lemma statement), the reduction to the weight commitment, and the protocol and soundness theorem
statement. Implementation (Section 5.6) becomes three sentences; cost (5.7) becomes two sentences pointing to
Section 7.4. No description of the LLM-agent review. Remaining text to `ext_v1.tex`.

**Generation (`generation.tex`, 750 words) and quality (`quality.tex`, 500 words).** Split `generation_quality.tex`.
`generation.tex` is Section 6 "Generation and Sampling" (label `sec:gen` stays). `quality.tex` is a subsection
`\subsection{Quality of the Certified Model}` (label `sec:quality`) for the evaluation; keep only the smoothed-gains
rows of `tab:quality`, state the WikiText-2 setting precisely, and keep the OPT-6.7B/Qwen3-4B/2,048-token todos.
Remove "in our earlier evaluation". Leave `generation_quality.tex` in place (the integrator removes it from
`main.tex`).

**Evaluation (`evaluation.tex`, 2,500 words including tables).** 7.1 setup: one paragraph stating the final runs will
be on one platform (L40S prover and GPU verifier, one CPU for the CPU verifier and re-execution), full models, no
extrapolation, medians with spread (`\todo{L40S}` where numbers are missing). 7.2 removed except for the pointer
(decision 3). 7.3 cost of the defence with `tab:eval-llm`. 7.4 V1 against v0 with `tab:eval-v1`, including the
bandwidth crossover sentence (the link speed below which V1 wins end to end; `\todo` for the number). 7.5
verification against re-execution with `tab:eval-reexec`. 7.6 comparison with published systems with `tab:eval-compare`,
with the hardware and guarantee caveats. `\input{sections/quality}` as the last subsection. `tab:eval-improve` moves to
`ext_evaluation.tex` with a two-sentence summary in 7.3. Remove duplicated V1 and attack numbers.

**Discussion and Conclusion (`discussion.tex`, 900 words).** Discussion and limitations, including a compact
"options for a client" table or paragraph (download the weights and re-execute on a CPU or GPU; a GPU TEE; a
CommitLLM-style audit; zkLLM/zkAgent; ours in modes C and Kpre, with and without V1: per-query bytes, client time,
what is trusted), with `\todo` where numbers are missing; the weight-leak statement (open weights only); scope
(dense decoders, no MoE, contexts to 2,048 tokens); and a four-sentence conclusion.

**Short appendix (`appendix_short.tex`, 2,600 words; submission only).** Start with `\section{...}` entries that
define the labels of decision 2. Content: A. the verifier model and the proof of Theorem `thm:guess` (compressed),
the worst-case instance (Proposition `prop:worst`, salted case first) and the other propositions as statements;
B. integer semantics in one paragraph (`app:semantics`), the proof of Theorem `thm:sound` (`app:sound`) compressed,
the untrusted-commitment theorem's proof idea (`app:untrusted`); C. V1's lemmas with proof ideas (`app:v1`);
D. the concrete-security table (decision 6). Every proof that is shortened ends with "The full proof is in the
extended version~\cite{fullversion}."
