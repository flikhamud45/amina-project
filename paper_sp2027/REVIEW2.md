# Mock review 2 (H5a): the condensed submission draft

**Draft reviewed.** The submission build as of 10 Oct 2026 (last source change 21:05):

- `main.tex` (preamble and abstract);
- the body: `sections/intro.tex`, `attack.tex`, `defence.tex`, `v1.tex`, `generation.tex`, `evaluation.tex` (which inputs `quality.tex`) and `discussion.tex`;
- the submission appendix: `sections/appendix_short.tex`, which inputs `app_params.tex`.

The extended version (`main_ext.tex`) was read only for the compliance lens and for the ledger.

**Paths and evidence.** Paths are relative to `paper_sp2027/`. Paths that start with `docs/`, `code/` or `SP2027_PLAN2.md` are relative to the repository root. Evidence is given as `file:line`.

**How the draft was read.** As the task asks, I read the sources and did not build a PDF.

- The `main.pdf` and `main.log` in the folder come from the build before the condensation. They were written at 04:38 and have 29 pages: body pp. 1-20, references pp. 20-22, appendices pp. 22-29. They say nothing about the current page count.
- Page figures below are estimates from source word counts, using the method of `CUT_PLAN.md` §2: about 1,000 non-comment source words per page, floats included.

**Scales.**

- Scores use the scale of `REVIEW.md`: 1 strong reject, 2 weak reject, 3 borderline, 4 weak accept, 5 strong accept.
- Each item is marked *closed*, *partly* or *open*.
- REVIEW.md items are cited as `REVIEW.md` A-W1 (reviewer A, weakness 1), B-Q5 (reviewer B, question 5) and so on, and as "chg n" for consolidated change n.
- Defects first found in this pass are numbered N1-N14 (Section 5).

---

## 0. The draft as a reviewer would see it

**Body.** 11,934 non-comment words, plus a 192-word abstract:

| Part | Words |
|---|---:|
| Introduction | 1,254 |
| Related Work | 489 |
| Attack | 2,328 |
| Defence | 2,312 |
| V1 | 1,545 |
| Generation | 659 |
| Evaluation (including quality, 500) | 2,403 |
| Discussion and Conclusion | 442 |

- There are seven floats: Fig. 1, `tab:sok`, `tab:eval-llm`, `tab:eval-v1`, `tab:options`, `tab:eval-compare` and `tab:quality`. Three of them (`tab:sok`, `tab:eval-llm`, `tab:eval-compare`) are two-column floats.
- **Estimated body length: 12.2 to 13.0 pages.**
  - 12.2 pages is CUT_PLAN's rate.
  - `SP2027_PLAN2.md:29` reports 18 pages in total. References take about 2.2-2.6 pages (80 cited keys) and the appendix about 2.4-2.7 pages, so that total puts the body close to 13 pages.
  - The body therefore has between 0 and about 0.8 page of slack before any new result is added.

**Appendix.**

- The submission appendix has 2,277 words: appendix_short 2,019 plus app_params 258.
- It has three appendices (A: spot-checking; B: defence, with `tab:params`; C: V1).
- With the references, the appendix part comes to about 4.6-5.3 pages, which is at or near the 5-page cap.
- Material cut from the body must therefore go to the extended version. It can go into `appendix_short.tex` only if it replaces appendix text of the same size.

**`\todo` markers.** I counted every `\todo{` outside comments.

- **72 in the submission build.** By file: main 1, intro 4, attack 6, defence 3, v1 1, generation 2, evaluation 37, quality 3, discussion 5, appendix_short 1, app_params 9.
- **105 in the extended build**: the same 72, less the one in appendix_short, plus 34 in `ext_*.tex`.
- Each marker prints as a red bold "[TODO: ...]" (`main.tex:26`).
- `SP2027_PLAN2.md:30` counts 71. I did not find the cause of the difference of one.

**Platforms.** The numbers come from three machine configurations: the L40S with an EPYC 9334 (base code), and an RTX 2080 Ti with a Xeon Silver 4114 (improvements and V1). See `evaluation.tex:38-44`.

**Fonts.**

- The 29-page build embeds NimbusRomNo9L, a Times clone, for the text, and CM fonts for the mathematics. I listed them with PyMuPDF.
- The preamble (`main.tex:10-23`) loads nothing that would change this.
- REVIEW.md read the log's `TU/ptm` warning as a fallback to Latin Modern. That reading was wrong.
- The fonts must be listed again on a fresh build.

**Format.** `\documentclass[conference,compsoc]{IEEEtran}` (`main.tex:10`). No `\vspace`, `\baselinestretch`, `\linespread`, `\textheight` or `\columnsep` changes appear in the submission files.

**Cross-references.** I checked them with a script:

- no undefined `\ref` or `\eqref`;
- no duplicate `\label`;
- "dichotomy" no longer appears in visible text, only in the label `sec:dichotomy`.

---

## 1. Reviewer A: systems security

### Summary

The draft breaks spot-checking proofs of inference with a single-site edit followed by honest recomputation. It bounds every spot-checking verifier, applies the bound to published schemes, and then certifies an exact int8 model by checking every weight product with Freivalds' algorithm against a Reed-Solomon and Merkle commitment. The draft now names its client in the first paragraph and frames the defence as an option among others (Table `tab:options`).

### What changed since REVIEW.md

1. **Target client.** The client is named at `intro.tex:20-28`, and the open-weight scope is stated at `intro.tex:116-117`. A client-options table follows (`evaluation.tex:168-199`), with GPU confidential computing (`intro.tex:160-162`, `discussion.tex:24-26`) and CommitLLM (`intro.tex:170-171`).
2. **A-W2, the in-range attack.**
   - It runs on OPT-1.3B, against per-neuron ranges calibrated on held-out WikiText-2 train text (`attack.tex:199-205`).
   - A second variant (only lowering activations) defeats contribution weighting.
   - I recomputed the medians from `code/artifacts/results/inrange_llm_opt-1.3b_{any,down}.json`. The any-move edits succeed in 320 of 320 cells, with a median of 2 neurons and at most 60. The down-only edits succeed in 234 of 320, with a median of 6. The text agrees.
3. **A-W3, TensorCommitments.** The verdict is softened to "stated goal not achieved" (`intro.tex:80-82`, `attack.tex:241`, `attack.tex:275-276`, `attack.tex:284-285`).
4. **A-W4, root of trust.** The main path is now a deterministic public quantisation recipe (`intro.tex:26-28`, `attack.tex:40-43`). The reproducibility test is still a todo (`defence.tex:42`).
5. **A-W8, duplication.** Attack numbers now appear only in Section 3.5, and the evaluation points there (`evaluation.tex:52`).

### Weaknesses of the draft as it stands

1. **The deployment comparison is still empty where it matters.**
   - `tab:options` has twelve "--" cells. The ones that matter are the GPU TEE row (`evaluation.tex:185`), the CommitLLM row (`:186`), and mode C's client time at 2,048 tokens (`:188`), which is the mode for a client without the weights.
   - It has no provider-overhead column and no break-even points.
   - The Discussion still compares the L40S base prover (1.8 s) with an fp16 forward pass on the 2080 Ti (0.037 s) (`discussion.tex:29-30`). This is the cross-GPU comparison REVIEW.md objected to.
2. **The re-execution rows favour us (N3).**
   - The exact GPU re-execution row (0.16 s / 2.08 s, `evaluation.tex:184`) is the integer forward pass before A2. A2 halves that pass on Llama-sized layers: `docs/improvements/F1_reexecution_baseline.md` §3 says so in a note, and `A2_int8_forward.md` §5 measures int8-to-fp32 time ratios of 0.44-0.61 on Llama-2 blocks (0.45-0.51 at 2,048 tokens).
   - The CPU integer row (98 s) runs its weight products as fp32 GEMMs over 1,024-term chunks (`SP2027_PLAN2.md:563`). It is not the strongest exact baseline.
   - Both errors make the client-options comparison look better than it is.
3. **The in-range sentence overstates the miss probability (N2).**
   - The text says "with a median of 2 neurons (at most 60), which a path opens with probability below $8\cdot10^{-4}$" (`attack.tex:201-203`).
   - In the result file, per-path detection is below $8\cdot10^{-4}$ for 76% of the 320 edits and for every per-cell median (the largest is $7.9\cdot10^{-4}$).
   - For the 60-neuron edit it is $60/8192 = 7.3\cdot10^{-3}$.
   - Suggested text: "a path opens them with median probability at most $7.9\cdot10^{-4}$ in every cell (at most $7.3\cdot10^{-3}$)".
   - The OPT-6.7B in-range result is still a todo (`attack.tex:205`).
4. **The untrusted-commitment section has a weak use case (A-W5).**
   - The draft motivates it by a private recipe (`defence.tex:201-202`) and then says that for open weights recomputing $C_M$ is cheaper (`defence.tex:282-283`).
   - It still takes a full subsection of about 0.9 page, with a theorem and a fact (`defence.tex:198-283`), while the paper targets open weights.
   - A reviewer will ask why it is in the body. Cut P1 (Section 9) demotes it.
5. **Platforms and extrapolation (A-W6).**
   - The CPU K_pre cell for Llama-2-7B appears with two values: 698 ms and 58.2 s in `tab:eval-llm` (`evaluation.tex:81`, `:87`; EPYC, base code), and 1.6 s and 17 s in `tab:options` (`:189`; Xeon, native kernels).
   - The text explains the difference (`:211-212`), but the reader still sees two numbers for one cell.
   - Full models are extrapolated from 1-3 blocks (`:172`, `:211`).
6. **Disclosure (A-W7).** The body states "Before submission we wrote to the authors" (`attack.tex:290-291`), with todos for the dates and the replies (`attack.tex:291-294`, `intro.tex:83`). That sentence must be true at submission, or it must be rewritten.
7. **Unanswered questions.**
   - A-Q4: the full version of Anchuri et al. (arXiv 2603.19025 v1, 19 Mar 2026, which I checked) describes a refereed model with two competing servers. `tab:sok` and Section 3.2 cover only the single-server path protocol, and the text does not say that the refereed variant lies outside $\mathrm{Spot}_d(q)$.
   - A-Q7: no sentence says who tokenises the prompt, or how system prompts on the provider's side are handled. `grep -i "tokeni\|system prompt"` finds nothing in the body.
   - A-Q6 (replies to the disclosure) is pending.

### Questions to the authors

1. Against the strongest exact CPU baseline (int8-GEMM integer re-execution, or bf16), is there still a regime in which K_pre beats downloading the weights and re-executing? At what number of queries, and at what link speed?
2. What does the exact GPU re-execution cost after the int8 forward pass (A2), and what is the prover's overhead over unverified serving on the same GPU?
3. Do in-range edits also change OPT-6.7B's next token, and with how many neurons?
4. Does your bound or your table cover the refereed two-server variant of Anchuri et al.?

### Score: 2 (weak reject)

The attack, the theorem and the honesty about limitations remain strengths. But the deployment table has its key cells empty, the exact baselines are stale in our favour, and 37 todos sit in the evaluation.

**What would raise it.**
- **To 3:** F6 (one platform), EV with RX's strongest exact baselines, A7's overhead column, the GPU re-execution row re-measured after A2, and the TH(g) and TH-d sentences.
- **To 4:** in addition, G3.5-L (OPT-6.7B in range), and an EV figure that shows a non-empty regime against RX (gate criterion 6, `SP2027_PLAN2.md:1086`).

---

## 2. Reviewer B: cryptography and proof systems

### Summary

There are two technical contributions:

- a lower bound for spot-checking verifiers, with a hard ReLU instance and an upper bound at full locality;
- a protocol that applies the evaluation step of a linear-code polynomial commitment to every matrix product of an exact integer model. It comes with a Fiat-Shamir argument, a one-time setup proof for untrusted commitments, and V1 (a windowed logUp-GKR over $\mathbb F_{p^8}$ with integer multiplicities, reduced to the existing column check).

### What changed since REVIEW.md

1. **B-W4.** The verifier class is now a formal definition in the body (`attack.tex:68-81`). It covers the query model, prover messages modelled as queries of their support, SNARK-assisted verifiers excluded (`:83-86`), and the budget measured under $R_p$ (`:85-87`).
2. **B-W1 and B-W2, positioning.**
   - The defence is presented as a linear-code polynomial-commitment evaluation (`intro.tex:89-94`, `defence.tex:30-37`).
   - The split commitment motivates the testing phase rather than being a finding (`defence.tex:204-218`).
   - LAMP, DeepBrake, ReedWeave and BaseFold are cited (`intro.tex:171-176`).
   - The novelty is placed in the combination (`intro.tex:182-184`).
3. **B-W5, "dichotomy".**
   - The word is gone from the visible text, and `thm:dich` is titled "lower and upper bound" (`attack.tex:129`).
   - Intermediate locality is stated as open (`attack.tex:157-158`, `intro.tex:75`).
   - No embedding into a transformer is claimed (`attack.tex:158-159`, `appendix_short.tex:143-145`).
4. **B-W8 and B-W9.** B-W8 (an arbitrary field model $M^*$) is conceded (`defence.tex:274-276`). "Attention recomputed by the verifier" replaces "attention included" (`intro.tex:87-88`, `:183`; `defence.tex:100-102`).
5. **B-Q3 and B-Q5 are answered** (`attack.tex:83-86`; `v1.tex:128-129`: the column check needs only that $u_\ell$ precedes the columns).
6. **Concrete security.** A concrete-security table exists for Llama-2-7B (`app_params.tex`), with interactive and Fiat-Shamir columns.

### Weaknesses of the draft as it stands

1. **Theorem `thm:guess` as stated in the body is false when $qd$ is not an integer (N1).**
   - The body defines $G_{qd}$ as a maximum over sets with $|S|\le qd$ (`attack.tex:100-102`). Definition `def:spot` bounds only the *expected* number of queries (`attack.tex:79-80`), which need not be an integer.
   - Counterexample: take $d=1$ and $q=1.5$. A verifier opens the most likely neuron, or the two most likely, with probability 1/2 each. It detects with probability $p_{(1)}+p_{(2)}/2$, which exceeds $\max_{|S|\le1.5}\Pr[J\in S]=p_{(1)}$ whenever $p_{(2)}>0$.
   - The full form in the appendix uses the fractional relaxation and is correct (`appendix_short.tex:37-38`).
   - Fix: use that fractional $G_t$ in the body, or state the weaker bound $G_{\lceil qd\rceil}$.
   - The attainment sentence ("opens the $q$ neurons most likely") should say "for integer $q$".
   - `thm:dich` and its $1-qd/N$ are unaffected.
2. **The theorem number for BCIKS20 is wrong (N5).**
   - `defence.tex:268` and `appendix_short.tex:206` cite "Thm. 1.7", and so does `appendix_defence.tex:173` in the extended version.
   - In ECCC TR20-083, revision 3 (3 Jul 2021), correlated agreement over affine spaces is **Theorem 1.6**.
     - Its error in the unique-decoding regime, $\varepsilon_U=n/q$ for $\delta\le(1-\rho)/2$, is defined in Theorem 1.2 (Eq. 1.1).
     - Remark 1.2 allows the combination to be sampled from an extension field, with $q$ replaced by $q'$.
     - That revision has no Theorem 1.7.
   - The hypotheses fit our use:
     - $U$ is the span of the rows;
     - the image of a uniform $\psi$ is uniform on that span;
     - $e_j<(n_j-k_j+1)/2$ is $\delta\le(1-\rho)/2$ with BCIKS's $\rho=k_j/n_j$.
   - Cite "Thm. 1.6 and Remark 1.2 of ECCC TR20-083, rev. 3" and delete the todo.
   - I could not check the numbering of the FOCS 2020 or J. ACM versions. The submission no longer uses the Ligero lemma, so the radius question of B-W6 is moot there. `appendix_defence.tex:177` still mentions the lemma.
3. **The union bound over $\mathcal J$ looks unnecessary (B-Q6, and more) (N6).**
   - `thm:sound` sums its terms over $\mathcal J$ (`defence.tex:147`). The Fiat-Shamir bound multiplies that sum by $Q$ (`defence.tex:192-193`).
   - The proof sketch fixes the first wrong operation $\ell$ from the claims, which the prover sends before any challenge. Acceptance then needs Freivalds' check and the column check at $\ell$ only, so $\varepsilon_C\le\max_j[p^{-r}+\text{column term}_j]$.
   - The same holds for each round of the round-by-round argument (`appendix_short.tex:193-197`): each challenge lets the prover escape with probability at most one term, not the sum.
   - The sum is valid, but it costs $\log_2(2|\mathcal J|)\approx8.8$ bits of $\beta$ ($\beta=136.8$ interactive, `SP2027_PLAN2.md:858`).
   - $r$ would stay at 5 and 7. The column counts would fall by roughly $8.8/\beta$, that is 4-6%.
   - I did not check the parameter code. TH-f should confirm the argument.
   - Either tighten the bound, or say that the union bound is kept for simplicity. B-Q6 needs one sentence in either case.
4. **The setup proof is the naive test, and the text concedes it (N7).**
   - `intro.tex:176` says that tests such as DeepBrake's "could shorten our setup proof".
   - TH-c plans the opposite remark: for column-hash commitments the cost is forced, and only the combined rows (1.6% of the setup proof) could shrink.
   - Make one statement, in one place.
5. **The concrete-security table is incomplete (B-W7).**
   - `tab:params` has 9 todos (`app_params.tex:21-46`, `:55`), among them $|\mathcal J|$, $\beta$, $t_j$ under Fiat-Shamir, and the setup parameters.
   - Its V1 rows quote Llama-2-13B (footnote b).
6. **Positioning still open.**
   - Intermediate locality (TH-a).
   - The logUp lemma for more leaves than the characteristic is not set against the small-field logUp implementations already deployed (`v1.tex:87-89`; B-W6).
   - The refereed variant of Anchuri et al. is not mentioned.
7. **Condition (H) lives in the appendix (minor).**
   - The hypothesis of `thm:guess` is defined only in `def:H` (`appendix_short.tex:27-33`), and the body paraphrases it informally (`attack.tex:95-97`).
   - Since reviewers may skip the appendices, the paraphrase should keep "up to the choice of the first query that touches $J$", which is what makes (H) checkable.

### Questions to the authors

1. Which version and theorem of BCIKS20 do you cite?
2. Does the first-wrong-operation argument not give $\max_j$ instead of $\sum_j$?
3. Does block-wise mixing over groups of $d$ rows make the bound tight at intermediate locality?

### Score: 3 (borderline)

The proofs I re-derived hold, and the positioning and the verifier model now meet REVIEW.md's requests. These are:

- Freivalds' check;
- the column check;
- Fact `fact:ext`;
- the integer-multiplicity lemma;
- the pole count of `thm:v1`;
- the use of BCIKS20 Theorem 1.6 in `thm:untrusted`.

**What would raise it to 4:**

- fixing weaknesses 1-3 (writing only);
- filling `tab:params` (TH-b);
- TH-a, or a precise statement of why intermediate locality is open;
- TH-c, reconciled with weakness 4;
- TH-d.

The concern that the novelty is incremental (B-W1) remains, so a 4 is plausible but not assured.

---

## 3. Reviewer C: ML systems

### Summary

The provider runs an exact integer transformer on int8 tensor cores. The verifier recomputes everything except the weight products and checks those products with Freivalds' algorithm. Native CPU kernels now make the K_pre verifier faster than re-execution in fp32. The draft states plainly that on a GPU, fp16 re-execution is faster.

### What changed since REVIEW.md

1. **C-W4, partly.** With the native kernels, the full Llama-2-7B is verified at 2,048 tokens in 17.2 s, against 78.8 s for fp32 and 98.3 s for integer re-execution. That is 4.6x and 5.7x faster (`evaluation.tex:205-213`; `docs/improvements/B5_cpu_verifier.md` §6). The numbers are extrapolated and interactive. The draft also states that the GPU verifier does not win (`evaluation.tex:215-220`).
2. **C-W3, headline claims.** The abstract no longer quotes the factors against zkSNARKs (`main.tex:68-71`). The body gives the hardware and guarantee caveats (`intro.tex:109-111`; `evaluation.tex:269-272`).
3. **Quality.** The evaluation setting is stated exactly (`quality.tex:16-21`), and quality has moved into the evaluation.
4. **Sampler.** The truncation and the lack of top-p are stated (`generation.tex:84-88`).

### Weaknesses of the draft as it stands

1. **The evaluation is unfinished (C-W1, C-W8).**
   - `evaluation.tex` has 37 todos. 18 of them are the empty "new" cells of `tab:eval-llm` (`:79-88`).
   - Costs use random weights (`:47-49`), and large models are measured as $b$ of their $L$ blocks (`:49-50`).
   - "Times are medians after a warm-up" (`:44`) gives neither the number of queries nor any spread.
2. **The serving overhead is still missing (C-W2).** It is a todo in three places (`evaluation.tex:100`, `intro.tex:110-111`, `discussion.tex:30`). The only figure given compares two different GPUs (`discussion.tex:29-30`).
3. **Stale baselines and a stale V1 ratio (N3, N4).**
   - The exact GPU re-execution row predates A2 (reviewer A, weakness 2).
   - The v0 verifier times in `tab:eval-v1` predate B5's later kernels.
     - For Llama-2-7B (2 of 32 blocks) in K_pre, the table gives 2.32 s (`evaluation.tex:137`; C2 doc, job 1005575, under Fiat-Shamir).
     - B5's final kernels verify the same two-block build in 0.739 s (B5 doc §6, job 1006792, interactive).
     - About 0.8 s of the gap is plausibly serial SHAKE-256 hashing under Fiat-Shamir: roughly 254 MB at 303-318 MB/s (`SP2027_PLAN2.md:258`). The rest comes from the kernels.
   - Hence "1.6-1.9x" (`evaluation.tex:155`; `tab:options`, `:190`) and the crossover of 0.16-0.26 Gb/s (`:161`) are not current. `SP2027_PLAN2.md:593` models about 3.5x with today's v0.
   - A reader can see the clash: 17 s for 32 blocks (`tab:options`) is about 0.55 s per block, against 2.32 s for 2 blocks (`tab:eval-v1`).
   - The modes differ, but only `tab:eval-v1` labels its own.
4. **A number without a source (N8).**
   - "0.12 against 0.14 s for one block" (`evaluation.tex:209-211`) is not in `docs/improvements/`.
   - B5 §6 gives 0.278 s (native tiled) against 0.153 s (fp32 SDPA) for an earlier kernel (job 1005733).
   - Cite the job, or remove the sentence.
5. **Two predictions for V1 with real weights (N9).**
   - The draft predicts 1.8-2.0x with real weights (`evaluation.tex:148-149`; `tab:options` footnote d, `:198`). Its source is `ext_v1.tex:151`.
   - `docs/improvements/C2_v1_proof_size.md` §8 quotes the spec's estimate of 1.78-1.85x instead.
   - Pick one and cite it, then replace it with F3a's measurement.
6. **Quality is too thin (C-W5) (N10).**
   - Only OPT-125M and OPT-1.3B are measured, on 16 windows of 128 tokens (`quality.tex:18-21`, `:44-45`).
   - Calibration uses a window of the test split (`:20`).
   - OPT-6.7B is 1.33x above fp32 with a scalar gain (`:25`), and no smoothed number exists.
   - Nothing is measured at 2,048 tokens, the length at which the costs are reported, and there are no downstream tasks.
7. **The sampler's truncation is not bounded in aggregate (C-W6) (N11).**
   - The per-token statement is correct (`generation.tex:85-87`): the table spans about $[-2.47,11.78]$, so a token more than about 14.25 below the top is never drawn, and its softmax probability is below $e^{-14.25}\approx6.5\cdot10^{-7}$.
   - The total excluded mass, however, can reach about $(V-1)e^{-14.25}$ in the worst case:
     - about 0.02 for a 32,000-token vocabulary (Llama-2);
     - 0.03 for about 50,000 tokens (GPT-2, OPT);
     - 0.09 for Qwen3's 151,936 tokens.
   - State this bound. The total-variation distance on real logits is a todo (`:87`).
8. **Memory and long contexts are open (C-W7, C-Q4, C-Q7).** Peak memory is a todo (`evaluation.tex:99`), and contexts longer than 2,048 tokens are untested (`discussion.tex:43-45`). The C2 doc records that Llama-2-7B under V1 in mode C does not fit the 11 GB card.
9. **Generation at scale (C-Q5).**
   - The body has only GPT-2 numbers (`generation.tex:43-52`), and the 7B cells are a todo (`:52`).
   - The text says that the proof is a separate prefill after decoding (`:49-50`), which answers half of C-Q5.
   - What that prefill adds to serving is not given. A7's "proof on request" row would answer it.

### Score: 2 (weak reject)

The engineering is careful, but the metrics this reviewer asked for are still missing: overhead over serving, quality at the evaluated lengths and models, memory, and spread.

**What would raise it.**
- **To 3:**
  - F6: one platform, at least 5 timed queries, with spread;
  - A7: overhead on the same GPU;
  - F2a and F2b-Q: quality at 2,048 tokens, including OPT-6.7B and Qwen3-4B;
  - F3a: real weights.
- **To 4:** RX and V-LONG-a must show at least 2-3x against the strongest exact baseline, on full models and under Fiat-Shamir, and EV must frame the GPU verifier against its PCIe floor.

---

## 4. Compliance lens (S&P 2027 call for papers)

**Rules.** I checked the rules on https://sp2027.ieee-security.org/cfpapers.html on 10 Oct 2026:

- **Pages.** At most 13 pages of text, plus at most 5 pages of references and appendices, 18 pages in total. A paper over the limit is rejected without review.
- **Appendix marking.** All text and figures past page 13 must be clearly marked as appendix. Reviewers need not read appendices.
- **Template and anonymity.** The compsoc template is required, with unmodified margins, fonts and spacing. The submission must be anonymous, and artifacts must not reveal the authors.
- **Mandatory fields.** "Ethics Considerations" and "Generative AI usage considerations" must be filled in on HotCRP at registration.
- **Disclosure.** Vulnerabilities must be disclosed no later than the rebuttal deadline.
- **Artifacts.** Anonymised artifact repositories are "strongly encouraged". A repository must not be updated after submission, and theory papers are encouraged to submit their proofs as artifacts.
- **Dates, cycle 2.** Registration on 10 Nov 2026 and submission on 17 Nov 2026. Title, abstract and author list are frozen at registration. Early reject on 18 Jan 2027, rebuttal due on 16 Feb 2027, final notification on 5 Mar 2027.
- **Ethics section.** Accepted papers add an Ethics Considerations section at camera-ready, outside the page limit.

| Rule | State of the draft | Evidence | Action (owner) |
|---|---|---|---|
| 13 + 5 pages, at most 18 | Not verified: there is no current build. Estimated body 12.2-13.0 pages; references plus appendix about 4.6-5.3. | Section 0 | Build both PDFs now. Record the page where the Conclusion ends, the page where the references start, and the total. Repeat at H5-mid and H5b. Apply the cut list (Section 9) before adding results (H5). |
| Text and figures past p. 13 marked as appendix | Appendices get "Appendix" headings from `\appendices` (`main.tex:84`). **Risk (N12):** the two-column tables `tab:eval-llm` (`evaluation.tex:62`) and `tab:eval-compare` (`:226`) are `table*` floats, which go at the earliest to the top of the next page. If the body ends on p. 13, a `table*` defined on that page prints on p. 14 as unmarked body content. | `main.tex:84`; `evaluation.tex:62`, `:226` | Check the float pages in every build. Move the `table*` definitions earlier in the source. `\FloatBarrier` before the bibliography is a last resort, since it can cost space (H5). |
| compsoc template, no space scrunching | Met. | `main.tex:10`; grep finds no spacing changes | Keep the PyMuPDF font listing (H5). |
| Anonymous title page | Met. | `main.tex:63-64` | -- |
| Anonymity in the text | "In our earlier evaluation" is gone. No names, affiliations, hosts or "course" appear in the visible text of the submission or the extended files (grep). | grep over `sections/*.tex` without comments | -- |
| Anonymity in the artifact | The `fullversion` entry has no link yet. LaTeX comments in seven files mention the session scratchpad, a "course paper" and "the agent writing ...". Plan H4 lists the git history (356 commits by named authors) and 1,645 result files with cluster paths. | `extra.bib:327-332`; e.g. `defence.tex:3`, `attack.tex:7`, `extra.bib:288`; `SP2027_PLAN2.md:946-950` | Export sources with comments stripped, or ship only PDFs. Run H4's grep gate over sources and bib (H4). |
| `\todo` count | 72 in the submission and 105 in the extended build, all printed in red. | Section 0; owner map below | Must be 0 in both builds (F6, TH, EV, H4, H5). |
| Generative-AI field | Mandatory at registration. Plan H4-d covers only the V1 adversarial review (`SP2027_PLAN2.md:952-954`). The repository records further uses: agents drafted and condensed sections and bib entries (`CUT_PLAN.md` §4, `extra.bib:288`), wrote mock reviews (`REVIEW.md`, this file), ran the 15-agent V1 review (`docs/improvements/C2_v1_proof_size.md` §9), and are planned to draft proofs (`SP2027_PLAN2.md:113`, `:876`). | as cited | Disclose and motivate every use, and state the human checks (TH-f, `SP2027_PLAN2.md:877`). A draft is below (H4-d, [USER]). |
| Ethics field | Mandatory. The body has an ethics paragraph (`attack.tex:288-295`). | `attack.tex:288-295` | Put the full text in the field (draft below). The body paragraph can shrink to the disclosure sentence and the release policy (cut C13). |
| Disclosure by 16 Feb 2027 | The notices are planned for 15 and 19 Oct. The body asserts that the authors were written to before submission, with todos for the dates and replies. | `intro.tex:83`; `attack.tex:290-294`; `SP2027_PLAN2.md:1168-1169` | Send the notices (G5, [USER]). Turn the conditional sentences into facts, or rewrite them (H4-c). |
| Artifact | The extended version is cited 13 times (8 in `appendix_short.tex`, 5 in the body). The extended build has 105 todos. | grep `cite{fullversion}` | Anonymous snapshot with 0 todos and the link in `fullversion`, frozen at submission (H4). |
| Abstract frozen on 10 Nov | The abstract has a todo and two numbers that may change: 4.6 (RX) and 1.7-2.3 (F3a, V1 re-time). | `main.tex:69-70` | Follow the plan's rule: "at least N times", and only numbers measured before the freeze (`SP2027_PLAN2.md:88-97`). Also N13 (Section 5). |

**Owner map of the 72 submission `\todo`s.**

| Owner | Count | Where |
|---|---:|---|
| F6 (all tiers) | 37 | evaluation 40, 56, 66, 79-88 (18 cells), 99, 134-136, 157, 173, 218, 252-253, 281; intro 130; v1 195; generation 52; defence 281; discussion 41 |
| TH-b | 9 | app_params 21-46, 55 |
| F3a, F3b | 4 | evaluation 48, 151; quality 69; discussion 44 |
| A7 | 3 | intro 110; evaluation 100; discussion 30 |
| RX (with FS1, V-LONG-a) | 3 | main 70; intro 116; evaluation 212 |
| F2a, F2b | 3 | quality 68 (two); discussion 45 |
| G3-min | 3 | attack 120, 179, 187 |
| G5, H4-c | 3 | intro 83; attack 291, 293 |
| G3.5-L | 2 | attack 205; discussion 50 |
| TH-d | 2 | defence 268; appendix_short 207 |
| E2b | 1 | generation 87 |
| D6a | 1 | defence 42 |
| EV | 1 | evaluation 161 |

**Draft for the HotCRP "Generative AI usage" field.** Brackets mark facts the team must confirm.

> We used large language models, run as coding and writing agents, in four ways. (1) Implementation support for the
> protocol code and experiment scripts. Every optimised code path was validated against the previous one on the same
> queries (identical claims, proofs and verdicts), and by [N] tests. (2) Drafting, condensing and copy-editing the text
> of this paper and of its extended version; the authors revised all text. (3) Drafting proofs for the extended
> version; [author role] checked each proof by hand on [date]. (4) Adversarial review: agent workflows tried to break
> the V1 verifier and to find errors in the paper. Each reported finding was re-checked independently, and the
> confirmed ones were fixed ([k] findings, the most severe medium). [Confirm:] no measured number in the paper was
> produced by a model; every number comes from logged runs of our code.

**Draft for the HotCRP "Ethics considerations" field.**

> Our attack targets published verification protocols, not deployed services. All experiments used our own
> implementation, our own hardware, public datasets and open-weight models; no live system or human subject was
> involved. On [dates] we wrote to the authors of the path protocol, TensorCommitments, CommitLLM, NanoZK, SLP
> [and SPEX], describing the findings, their assumptions and what we do not claim, and offering our draft and code.
> [Replies, or: "No reply by 2 Nov 2026."] We do not claim that schemes meeting their stated probabilistic guarantees
> are broken. We release code for the edit against the path protocol on public models, but no method for steering
> the responses of instruction-tuned assistants. The defence has no harmful use that we are aware of.

---

## 5. Defects found in this pass (consolidated)

Each defect below is a writing fix, unless an item ID is named.

| ID | Defect | Evidence | Fix (owner) |
|---|---|---|---|
| N1 | `thm:guess` is false for non-integer $qd$: a maximum over sets of size at most $qd$, against an expected budget. | `attack.tex:79-80`, `:100-102`; the correct form is at `appendix_short.tex:37-38` | Fractional $G_t$ in the body, or $G_{\lceil qd\rceil}$; "for integer $q$" in the attainment sentence (TH). |
| N2 | In-range detection overstated: "below $8\cdot10^{-4}$" holds for the medians and for 76% of the edits, but not for the 60-neuron edit ($7.3\cdot10^{-3}$). | `attack.tex:201-203`; `code/artifacts/results/inrange_llm_opt-1.3b_any.json` | Reword as in Section 1, weakness 3 (H5). |
| N3 | Re-execution baselines favour us. The exact GPU row predates A2 (about 2x too slow). The CPU integer row uses fp32 chunked GEMMs. | `evaluation.tex:183-184`, `:195-196`; F1 §3; A2 §5; `SP2027_PLAN2.md:563` | Re-measure (RX, A7 U1, F6 `gpuv`). Until then, label the rows. |
| N4 | The V1 verifier ratio and the bandwidth crossover use v0 verifier times from before B5's later kernels, and the two tables use different modes. | `evaluation.tex:137`, `:155`, `:161`, `:190`; B5 §6; `SP2027_PLAN2.md:593` | V-LONG-a re-time. Label the mode in every caption (EV(c), F6). |
| N5 | BCIKS20 is cited as "Thm. 1.7". In ECCC TR20-083 rev. 3 it is Thm. 1.6, with Remark 1.2 for the extension field. | `defence.tex:268`; `appendix_short.tex:206`; `appendix_defence.tex:173` | Cite version and theorem; delete the todo (TH-d). |
| N6 | The sum over $\mathcal J$ is looser than the argument gives: the first wrong operation is fixed before any challenge. | `defence.tex:147`, `:192-195`; `appendix_short.tex:193-197` | Tighten to the maximum, or say why the sum is kept; answers B-Q6 (TH-d, TH-f). |
| N7 | `intro.tex:176` ("could shorten our setup proof") contradicts TH-c's planned remark that the cost is forced. | `intro.tex:176` | One statement, in the setup-proof paragraph (TH-c). |
| N8 | "0.12 against 0.14 s" has no source in the docs. | `evaluation.tex:209-211`; B5 §6 | Cite the job, or delete (H5). |
| N9 | Two different predictions for V1 with real weights (1.8-2.0x against 1.78-1.85x). | `evaluation.tex:148-149`, `:198`; `ext_v1.tex:151`; C2 §8 | One source; replace by F3a. |
| N10 | Quality is calibrated on a test-split window. | `quality.tex:20` | Calibrate on the train split (F2a does). |
| N11 | The sampler's truncated mass is unbounded in aggregate: up to about 0.02, 0.03 and 0.09 for the vocabularies of Llama-2, OPT/GPT-2 and Qwen3. | `generation.tex:84-88` | Add the bound, then the measured TV (E2b). |
| N12 | `table*` floats near the end of the body can print past p. 13 without an appendix mark. | `evaluation.tex:62`, `:226` | Check the float pages in each build (H5). |
| N13 | The abstract calls 1.7x "long-prompt", but it is the 512-token real-weight OPT-125M figure. "4.6 times" is against fp32 only, extrapolated and interactive. | `main.tex:70` | "2.0-2.3 times at 2,048 tokens" until F3a; the RX rule for 4.6 (RX, H5). |
| N14 | The Discussion keeps the cross-GPU prover comparison (L40S against 2080 Ti). | `discussion.tex:29-30` | Replace it with A7's same-GPU ratio (A7, EV). |

Smaller notes, no ID:

- DeepBrake's ePrint page now carries a revised title, according to `extra.bib:344-347`. I did not check this myself. Cite the version you mean.
- `tab:options` gives no protocol mode in its caption.
- `fact:ext` and `thm:untrusted` need no change.

---

## 6. Checklist: REVIEW.md consolidated changes 1-13

| # | Change | Status | Evidence | What remains (owner) |
|---|---|---|---|---|
| 1 | Finish the evaluation on one platform, with frozen code and full models; no extrapolation; real weights; memory; variance; fill the todos; remove cross-table disagreements | **open** | Three configurations and `\todo{L40S}` (`evaluation.tex:38-44`). Random weights and $b/L$ builds (`:47-50`). 37 todos in the evaluation, including 18 cells (`:79-88`). The disagreement is explained (`:211-212`) but both values are still printed (`:81`, `:87` against `:189`). No spread (`:44`). Memory is a todo (`:99`). | F6, F3a/F3b, V-LONG-a, RX |
| 2a | Meet the length limit | **partly** | Not verified by a build; estimate 12.2-13.0 body pages (Section 0) | Build now; Section 9 (H5) |
| 2b | Build with Times | **closed** | The old build embeds NimbusRomNo9L; REVIEW.md's "Latin Modern" was a misreading of the log | Re-list fonts on each build (H5) |
| 2c | Attack results once (merge 3.5 and 7.2) | **closed** | `attack.tex:164-205`; pointer at `evaluation.tex:52-53` | -- |
| 2d | V1 numbers once | **partly** | 2.0-2.3x and 1.7-1.8x appear in the abstract (`main.tex:70`), the introduction (`intro.tex:112-114`), the V1 section (`v1.tex:193-195`), Section 7.4 (`evaluation.tex:144-151`) and `tab:options` (`:190`, `:198`). The 10x prover cost appears at `v1.tex:195` and `evaluation.tex:154`. | Cut C8(b): V1's own paragraph points to Section 7.4 (H5) |
| 2e | Move semantics, GKR implementation, `tab:setup`, the attribution table and `tab:aligned` | **closed** | `appendix_short.tex:152-169`; `v1.tex:191-192`; `ext_defence.tex:105`; `ext_generation.tex:158`; `tab:aligned` is absent from the submission | -- |
| 2f | Introduction about 1.5 pages; Related Work trimmed | **closed** | Introduction 1,254 words plus Fig. 1; Related Work 489 words | Cuts C1-C3 tighten further (H5) |
| 3 | Quantitative deployment case: client options, bytes, client time, provider overhead, break-even; name the target client; GPU confidential computing | **partly** | `tab:options` (`evaluation.tex:168-199`) has empty TEE and CommitLLM rows (`:185-186`), no overhead column and no break-even. Client named (`intro.tex:20-28`). GPU CC at `intro.tex:160-162` and `discussion.tex:24-26`. | EV, A7, RX |
| 4 | Realise the verifier's advantage; ratios on the same hardware for every cell | **partly** | CPU: 4.6x and 5.7x (`evaluation.tex:205-213`; B5 §6), extrapolated, interactive, against a non-optimised integer baseline. The GEMM-based hybrid attention was slower (B5 §3). GPU: re-execution is faster (`:215-220`). | V-LONG-a, RX, F6 `gpuv`, EV roofline |
| 5 | Attack the hardened protocol on a real LLM; adaptive game; G1e; in-range neurons needed | **partly** | OPT-1.3B with held-out ranges and contribution weighting (`attack.tex:199-205`). OPT-6.7B and Qwen3-4B are a todo (`:205`). The adaptive game, G1e and training-split MNIST ranges are todos (`:120`, `:179`, `:187`). | G3.5-L, G3-min |
| 6(i) | Lower bound as a formalisation; difference from PCPPs; no "dichotomy" unless intermediate $d$ is settled | **closed** (wording) | `intro.tex:75-78`; `attack.tex:156-161`; title at `:129`; no visible "dichotomy" | TH-a for intermediate $d$ |
| 6(ii) | PCS evaluation with the testing phase amortised; split as motivation; cite DeepBrake, ReedWeave, LAMP | **closed** | `intro.tex:89-98`, `:171-176`; `defence.tex:30-37`, `:204-218` | Reconcile `intro.tex:176` with TH-c (N7) |
| 6(iii) | Replace "46 to 32,000 times" in the abstract with a matched statement plus the overhead over inference | **partly** | Abstract clean (`main.tex:68-71`). The 46x in the introduction carries both caveats (`intro.tex:109-111`). The overhead is a todo (`:110-111`). The evaluation keeps "46x to 32,000x" after its caveats (`evaluation.tex:269-274`), which is acceptable. | A7 |
| 6(iv) | Qualify "halves the proof" | **closed** | `intro.tex:112-114`; `v1.tex:193-195`; `evaluation.tex:144-151` | Wording of the abstract (N13) |
| 6(v) | "Attention recomputed by the verifier" | **closed** | `intro.tex:87-88`, `:183`; `defence.tex:100-102` | -- |
| 6(vi) | Soften the TensorCommitments verdict | **closed** | `intro.tex:80-82`; `attack.tex:241`, `:275-276`, `:284-285` | -- |
| 7 | Quality where the costs are reported (full test set, 2,048 tokens, downstream tasks, 8-bit probabilities at long context); the sampler's TV | **open** | `quality.tex:18-21`, `:44-45`, `:68-69`; `generation.tex:87` | F2a, F2b-Q, F2b-L, F2c, E2b |
| 8 | Ethics and policy hygiene: disclosure, artifact statement, AI-use description, anonymity leak | **partly** | Disclosure is conditional (`intro.tex:83`; `attack.tex:290-294`). No LLM-agent review in the body (closed). "In our earlier evaluation" removed (closed). Artifact: only the `fullversion` entry, without a link (`extra.bib:327-332`). | G5 [USER], H4 (fields: Section 4) |
| 9 | Formal verifier class in the body; one concrete-security table; BCIKS20 and the Ligero radius; Fiat-Shamir in every table, or labels | **partly** | Verifier class: closed (`attack.tex:68-87`). Table: partly (`app_params.tex`, 9 todos). BCIKS20: open, with the number now checked (N5; `defence.tex:268`). Ligero radius: unused in the submission (closed there; `appendix_defence.tex:177`). Modes: `tab:eval-llm` interactive (`evaluation.tex:63`), `tab:eval-v1` Fiat-Shamir (`:113`), `tab:eval-compare` mixed with a footnote (`:259-261`), `tab:options` unlabelled (`:212` says interactive). | TH-b, TH-d, FS1, F6, EV(c) |
| 10 | Untrusted commitments: justify the client, or demote the setup proof and make D6 the main path; time it; estimate a FRI-style alternative | **partly** | D6 is the main path (`attack.tex:40-43`; `intro.tex:26-28`), with its test pending (`defence.tex:42`). `tab:setup` is in the extended version. The setup proof is still a full body subsection (`defence.tex:198-283`), its time is a todo (`:281`), and no alternative is estimated (`intro.tex:176`). | D6a, TH-c, F6 setup cell, cut P1 |
| 11(a) | V1 timings on the L40S, including Llama-2-7B in mode C | **open** | `evaluation.tex:134-136`, `:157` | F6 `v1` |
| 11(b) | Real-weight V1 bytes at 2,048 tokens | **open** | `evaluation.tex:148-151` (only the 512-token OPT-125M) | F3a |
| 11(c) | Bandwidth crossover | **partly** | 0.16-0.26 Gb/s (`evaluation.tex:159-162`); I re-derived the range from `tab:eval-v1`. The v0 verifier times predate B5 (N4). | V-LONG-a re-time, EV |
| 11(d) | V1 body text about 1.25 pages | **partly** | `v1.tex`: 1,545 words with a theorem, a lemma and five displays, about 1.5-1.6 pages | Cut C8 (H5) |
| 12 | Generation at 7B (P = 512, G of 64 or 256, greedy and sampled, tamper cells); reuse of decoding or extra prefill | **partly** | The separate prefill is stated (`generation.tex:49-50`). The 7B cells are a todo (`:52`). | F6 `gen`, F3b, A7 ("proof on request") |
| 13 | Optional: one opening per session of B queries | **open** (optional) | Nothing in the body | C4a (v0 batch cells, inside F6); C4b after 17 Nov |

## 7. Checklist: REVIEW.md "Smaller issues"

| # | Issue | Status | Evidence | Fix (owner) |
|---|---|---|---|---|
| 1 | Reconcile "2%" with "9.2%" | **closed**, wording to tighten | `attack.tex:181-187`. "Each figure, the detection of the best attack we found against that sampler" does not fit 9.2%: that number is the contribution-weighted sampler against the plain edit, while the best attack found against it (zeroing) is detected 0% of the time (`:183-184`). | "Against each sampler, the best attack we found is detected with probability 0.20% (uniform), 0% (contribution-weighted: zeroing) and 2% (floor $\delta=1$); each bounds that sampler's worst case from above." (H5) |
| 2 | Qualifiers on "2.0-2.3x" in the introduction | **closed** | `intro.tex:112-114` gives random weights and real OPT-125M at 512 tokens. The qualifier "more than one block" is at `evaluation.tex:144-148`. | -- |
| 3 | Extrapolated `tab:eval-reexec`; extrapolated 1.1 s against zkLLM's measured 0.90 s | **partly** | The 1.1 s is now set only against fp16 (`evaluation.tex:217-218`), and `tab:eval-compare` uses the measured 2.6 s (`:251`). The extrapolation remains and is labelled (`:172`, `:211`). | V-LONG-a, F6 |
| 4 | `tab:eval-llm` has no "new" CPU-verify column; the note is a todo | **open** | `evaluation.tex:65-66`, `:74-76` | F6 `cpuv` |
| 5 | The abstract has nine numbers | **closed** | Three measured numbers: 40/40, 4.6 and 1.7-2.3 (`main.tex:69-70`) | N13; RX rule |
| 6 | Quality belongs with the evaluation | **closed** | `evaluation.tex:285`; `generation.tex` covers only generation and sampling | -- |
| 7 | "Nine published schemes" | **partly** | Nine is now right for inference schemes, counting TOPLOC and SVIP separately (`intro.tex:78-79`, `:125`). But `tab:sok` still has the Proof-of-Learning row (`attack.tex:263-267`), so the table shows ten schemes. | Drop the row (cut P2) |
| 8 | Budget under $R_p$ equals the real one until the first touch | **closed** | `attack.tex:85-87` | -- |
| 9 | `prop:worst`: hiding commitment first, dead inputs only for unsalted Merkle trees | **closed** | `appendix_short.tex:72-77` | -- |

---

## 8. Page ledger

**Budget.**

- The body is now 12.2-13.0 pages (estimate).
- Additions are costed below at about +0.8 to +1.0 page in the body and +0.15 in the appendix.
- The cut list (Section 9) removes about 1.8 pages: about 0.7 from the plan's list and about 1.1 more.
- Filled todos free a little: 271 words of todo text, most of it replaced by numbers.
- Target: **a body of at most 12.7 pages at H5b**, so that float placement (N12) has room.
- Apply P1-P3 and C1-C8 first, rebuild, and add the remaining cuts only as needed.

| New result | Item | Destination | Where exactly | Body cost (pages) |
|---|---|---|---|---:|
| Break-even figure (two panels: client bytes against queries; K_pre time plus transfer against link speed, with V1's line) | EV | **Body, one column-wide figure** in Section 7.3 (`sec:eval-reexec`, next to `tab:options`, `evaluation.tex:168`). Rename the subsection "A Client's Options". | The target-client paragraph replaces `discussion.tex:23-30`. V1's line in panel 2 replaces the crossover paragraph (`evaluation.tex:159-162`). Each crossover number appears once. | +0.25 to +0.35 |
| Provider overhead over unverified serving | A7 | **Body cell plus one sentence**: one column of `tab:options` ("prover / unverified, same GPU"), and one sentence in Section 7.2. The `serve` table (phases, batch sizes) goes to the extended version. | The sentence replaces the todos at `evaluation.tex:100`, `intro.tex:110-111` and `discussion.tex:29-30`. The number appears once in the introduction and once in Section 7.2 (N14). | +0.05 |
| Memory table for the 7B models at 2,048 tokens | F6 (F4) | **Extended version**, plus **one sentence** in Section 7.2 with the peak GPU and host memory of prover and verifier | Replaces `\todo{F4}` at `evaluation.tex:99` | +0.02 |
| Quality at 2,048 tokens (full WikiText-2 test, train calibration, confidence intervals; 8-bit probabilities) | F2a, F2b-Q, F2b-L | **Body cells**: rebuild `tab:quality` with rows OPT-1.3B, OPT-6.7B, Qwen3-4B (and Llama-2-7B if F2b-L passes) and columns fp reference / integer / ratio [CI] / top-1 agreement. The OPT-125M row and the scalar-gain column go to the extended version. The 8-bit against 11-bit (or float) effect gets one sentence. | `quality.tex:33-48` (table); setting `:16-21`; the todos at `:68-69` are removed | 0 to +0.05 |
| Downstream accuracy | F2c | **One sentence** after `tab:quality`: mean drop with a paired CI, tasks named. The table goes to the extended version. | After `quality.tex:48` | +0.03 |
| Sampler total-variation distance | E2b | **One sentence** in Section 6.2, together with the aggregate truncation bound (N11). The figure goes to the extended version. | Replaces the todo at `generation.tex:87-88` | +0.02 |
| Long-context rows (2k-16k measured, 32k modelled) | V-LONG-b | **One sentence** in the Discussion's "Scope" (verifier/re-execution ratio and bytes at 8k measured, 32k modelled and labelled). Cells and fit go to the extended version. | Replaces "longer contexts are untested" at `discussion.tex:43-45`. One clause in "Proof size" (`:32-36`). | +0.03 |
| Real-weight rows | F3a, F3b | **Body cells, labelled "real weights"**: one pair of rows (C, K_pre) for OPT-1.3B at 2,048 tokens in `tab:eval-v1`; OPT-6.7B cells in `tab:eval-llm` marked real in the caption. Per-component bytes go to the extended version. | Replaces the prediction at `evaluation.tex:148-151` and `tab:options` footnote d (`:198`). Update `:47-49`. | +0.05 |
| In-range attack on OPT-6.7B | G3.5-L | **One clause** in Section 3.5. `tab:inrange` stays in the extended version (`ext_attack.tex:253`). | Replaces `\todo{L40S: OPT-6.7B, Qwen3-4B}` at `attack.tex:205`. Delete the todo at `discussion.tex:50` and state the result there in a clause. | +0.02 |
| *Further results that also need space* | | | | |
| Strongest exact re-execution baselines | RX | Body cells in `tab:options`, replacing the CPU rows and footnote a; the GPU row re-measured after A2 (N3) | `evaluation.tex:183-184`, `:195-196`; abstract `main.tex:70` | +0.03 |
| Block-mixing proposition | TH-a | One sentence and the statement in Section 3.4; proof in `appendix_short.tex` (about +0.15 page there) or in the extended version if the appendix is full | Replaces "intermediate locality is open" (`attack.tex:157-158`, `intro.tex:75`) | +0.08 |
| Why the setup proof's cost is forced | TH-c | One or two sentences in the demoted setup paragraph | Replaces `intro.tex:176`'s last sentence (N7) | +0.02 |
| Generation at 7B | F6 `gen`, F3b | Replace the GPT-2 numbers rather than adding to them; GPT-2 goes to the extended `tab:gen` | `generation.tex:43-52` | 0 |
| v0 batch cells | C4a | One sentence in Section 7.2 | after `evaluation.tex:100` | +0.02 |
| Setup-proof timing | F6 | One clause | `defence.tex:281` (or its demoted paragraph) | 0 |
| Adaptive game, G1e | G3-min | One sentence each, or delete if the item misses | `attack.tex:120`, `:179`, `:187` | 0 |
| Disclosure facts | G5, H4-c | Replace the todos | `intro.tex:83`; `attack.tex:290-294` | 0 |
| Tokenisation and system prompts; Anchuri's refereed variant | TH(g), TH-d | One sentence each in Section 3.1 | `attack.tex:38-43` | +0.04 |

---

## 9. Cut list

**Rules.**

- Material leaves the body for the extended version (`ext_*.tex`), not for `appendix_short.tex`. The submission appendix is already near the 5-page cap with the references (Section 0).
- After each pass, rerun the label check (no body reference to a label defined only in `ext_*.tex`) and rebuild.

### 9.1 The plan's list (`SP2027_PLAN2.md:972-975`), sized

| ID | Cut | Where | Saves |
|---|---|---|---:|
| P1 | **Demote the untrusted-commitment subsection.** | `defence.tex:198-283` | about 0.6-0.7 page |

What P1 keeps, in about 180 words:
- the split commitment in two or three sentences, with Eq. `eq:split` inline;
- one sentence on the one-time setup proof and its sizes (2.8 / 11.3 GB, `:278-281`);
- TH-c's remark;
- "for open weights, recompute $C_M$" (`:282-283`).

The subsection now has 783 words, three displays, Fact `fact:ext` and Theorem `thm:untrusted`.

Where the rest goes:
- `fact:ext` and the statement of `thm:untrusted` go into `appendix_short.tex`, in place of the proof idea of `app:untrusted` (`appendix_short.tex:199-217`, about 210 words). The appendix stays about neutral (+0.1 page at most).
- The proof idea goes to the extended version; `appendix_defence.tex` already has the full proof.
- Update extension 1 in `intro.tex:96-98` and the contributions (`:127`).

| ID | Cut | Where | Saves |
|---|---|---|---:|
| P2 | Drop the Proof-of-Learning row. "Nine published schemes" stays right. Drop `jia2021pol` and `zhang2022pol` if nothing else cites them; `fang2023pol` is also cited at `attack.tex:286`. | `attack.tex:263-267` | about 0.03 page, plus two references |
| P3 | Compress the four CNN rows of `tab:eval-compare` to two. Keep zkCNN VGG-16 and ZKML VGG-16 (the latter carries the largest factor quoted at `:273-274`). Drop the LeNet-5 rows; Bionetta stays cited in Related Work (`intro.tex:138`). | `evaluation.tex:239-242` | about 0.03 page |

**Subtotal: about 0.7 page.** This is more than the plan's 0.5, because P1 is larger than the plan estimated.

### 9.2 About 1 more page (found in this pass)

Word savings are counted in source words. Tier 1 removes text that duplicates other text in the body; Tier 2 moves material to the extended version.

| ID | Tier | Cut | Where | Saves (words) |
|---|---|---|---|---:|
| C1 | 1 | Shorten the introduction's "A lower bound" paragraph from 270 to about 120 words. Keep: the reduction to a model differing in one row; the bound in one sentence; the hard ReLU instance; the mixing converse; the SoK sentence with the TensorCommitments verdict and disclosure. Drop: the restated verifier model (it is `def:spot`) and the PCPP sentence (Related Work has it). | `intro.tex:65-83` | 150 |
| C2 | 1 | "We add four extensions": one clause per extension. Drop the repeated weak-Fiat-Shamir citations (`defence.tex:175-177` has them) and V1's mechanism (Section 5). | `intro.tex:96-106` | 60 |
| C3 | 1 | Say the folklore and PCPP point once, in Related Work. Delete it from `attack.tex:159-161` and from `intro.tex:75-78`, after C1. | `intro.tex:147-156`; `attack.tex:159-161` | 80 |
| C4 | 2 | MNIST paragraph: keep the $1-1/N$ match and the backdoor's 99.96%. Move the 250-path figure to the extended version. Drop the 95th-percentile sentence and its footnote unless G3-min lands; the held-out OPT-1.3B result supersedes them. | `attack.tex:172-179` | 70 |
| C5 | 2 | "Commitment plans" in one sentence (the plan is a public function, part of $C_M$). The transposed layout, GPT-2's head and the embedding tree go to the extended version. The transposed-layout sentences of (S3) go too; `appendix_short.tex:182-185` covers them. | `defence.tex:83-87`, `:124-126` | 90 |
| C6 | 1 | Section 4.4: fold "A sponge transcript" into one clause of the binding paragraph. Keep the round-by-round paragraph, with N6's sentence. | `defence.tex:184-187` | 70 |
| C7 | 1 | Remove the restated per-row factors from the comparison paragraph (they are table cells). Keep the two caveats and one sentence per cost. | `evaluation.tex:273-281` | 90 |
| C8 | 1, 2 | Bring V1 to about 1.25 pages (chg 11(d)), in five parts. **(a)** The Llama-2-13B error terms go to `tab:params` (already there, footnote b) (`v1.tex:185-189`). **(b)** "Implementation and cost" becomes one sentence pointing to Section 7.4 (`:191-195`). **(c)** The GKR packing description goes from 93 to about 45 words (`:104-111`). **(d)** Delete the lemma's explanation, since `appendix_short.tex:265-267` has it (`:98-100`). **(e)** Cut the opening, which repeats Section 1 and Section 7.4, to one sentence (`:35-38`). | `v1.tex` | 250 |
| C9 | 1, 2 | "Improvements" in one sentence (`evaluation.tex:102-106`). The image-model costs sentence goes to the extended version (`:93-94`). | `evaluation.tex` | 65 |
| C10 | 2 | Quality: cut "Where the perplexity goes" to about 50 words, since the attribution is in `ext_generation.tex:158` (`quality.tex:23-31`). Cut "SmoothQuant through the norm's gains" to about 95 words (`:50-61`) and "Comparison" to about 40 (`:63-67`). Drop the scalar-gain column when F2a lands. | `quality.tex` | 155 |
| C11 | 1 | "Options for a client" becomes EV's target-client paragraph in Section 7.3, so it is not said twice. "The attack and the bound" shrinks once G3.5-L lands. | `discussion.tex:23-30`, `:47-50` | 100 |
| C12 | 2 | Cost of generation: replace the GPT-2 numbers rather than adding the 7B ones. Drop the zkAgent comparison; Related Work cites zkAgent (`intro.tex:143`). | `generation.tex:43-52` | 25 |
| C13 | 2 | Ethics paragraph: keep the disclosure sentence and the release policy; the rest goes to the HotCRP field. The CFP adds a camera-ready ethics section outside the page limit. | `attack.tex:288-295` | 45 |

**Total: about 1,250 words, about 1.1 page of text.**

If more room is needed:
- shorten the "Stated guarantee" texts of `tab:sok`;
- once the "new" columns of `tab:eval-llm` are filled, move the "base" columns to the extended `tab:eval-improve`. This narrows the table but does not make it shorter.

Two things should stay in the body:
- Fig. 1, which is the paper's one-glance summary;
- the soundness-test counts (`evaluation.tex:52-56`), the only summary of the tamper tests.

---

## 10. What I verified, and what I could not

**Verified externally (10 Oct 2026).**

- **S&P 2027 call for papers**: every rule quoted in Section 4 (sp2027.ieee-security.org/cfpapers.html).
- **BCIKS20, ECCC TR20-083, revision 3 (3 Jul 2021).**
  - Theorem 1.2 is the proximity gap, with $\varepsilon_U=n/q$ for $\delta\le(1-\rho)/2$.
  - Theorems 1.4 and 1.5 are correlated agreement for lines and for curves.
  - **Theorem 1.6 is correlated agreement over affine spaces.**
  - Remark 1.2 covers sampling from an extension field.
  - That revision has no Theorem 1.7.
  - Not checked: the FOCS 2020 and J. ACM numbering. ePrint 2020/654 returned HTTP 403.
- **Anchuri et al.** arXiv 2603.19025 v1 (19 Mar 2026) is listed as "Accepted at IEEE SaTML 2026", and its abstract mentions a refereed delegation model with two competing servers. Not checked: ePrint 2026/541, and whether the SaTML proceedings contain that variant.
- **GPU confidential computing.** arXiv 2409.03992 (Zhu et al., v4 of 5 Nov 2024) states that "for the majority of typical LLM queries, the overhead remains below 7%". This matches `intro.tex:160-162`.
- **CommitLLM** (the README on GitHub):
  - "Measured online tracing overhead is about 12-14% during generation";
  - "Arbitrary-position attention outputs are not verified in the shipped product".
  - Both match `intro.tex:170-171`.
  - The README does not state the 48 MB key size that `SP2027_PLAN2.md:924` quotes.
- **zkLLM** (arXiv 2404.16109, CCS 2024): the abstract claims under 15 minutes and under 200 kB for 13B parameters. This is consistent with `tab:eval-compare`'s 13.4 min and 188 kB. I did not check the per-model table values.

**Verified internally.**

- In-range medians, maxima and per-path probabilities, from `code/artifacts/results/inrange_llm_opt-1.3b_{any,down}.json`.
- `tab:eval-v1`'s ratios (1.6-1.9x at 2,048 tokens, 2.2-2.8x at 512) and the crossover of 0.16-0.26 Gb/s, recomputed from its cells.
- The `tab:sok` figures for SLP (98.14% per audit; 37 and 160 audits) and CommitLLM (87.5%; about 99.4%).
- The path counts: 305,193 and 454,248 are $\ln 2^{40}/(-\ln(1-1/N))$, about $27.7N$.
- The Gumbel table's range, $[-2.47,11.78]$.
- 46x ($804/17.6$) and 21x ($6.7\,\text{GB}/325\,\text{MB}$).
- The CPU and GPU re-execution numbers, against `docs/improvements/B5_cpu_verifier.md` §6 and `F1_reexecution_baseline.md` §3.
- `tab:eval-v1`, against `C2_v1_proof_size.md` §8.
- Quality numbers, against `F2_quality.md` §3.
- Generation costs, against `E1_generation.md`.

**Not verified.**

- The image-model attack numbers (99.96% of 2,300 runs; 2,000 of 2,000 zeroing runs; 1/84 to 1/512). They come from the course report, which I did not open.
- "13.7 GB of fp16 openings at 64 tokens".
- zkAgent's 0.30 s per token, and the Maverick, ZKTorch, DeepProve, zkGPT, zkCNN, Bionetta and ZKML table values.
- The DeepBrake title revision noted in `extra.bib:344-347`.
- The page count, the font listing and float placement of the current draft, since I did not build it.
