# H5 early mock review (12 Oct 2026)

Three reviewer lenses and a compliance check on the submission body, written by a read-only agent from the sources
(no build). Scores: A 2, B 3, C 2 (1 to 5).

**Applied on 12 Oct** (each checked against its source; build 18 pages, body to p. 13):
- A-W1 (partly): the Discussion now names the target client with the break-even numbers (EV, Figure 2); the "about 21
  short ones" sentence that contradicted the abstract is gone. The abstract's ratio keeps its EPYC todo.
- A-W2 (partly): the provider's cost over unverified serving is in tab:options footnote f (A7); "prove later" moved
  to the extended version with the integer engine's floor.
- B-W1: "tight" became "met up to a factor O(lambda + log L)"; the subsection is "Lower and Upper Bounds".
- B-W2: Theorem 2(b) and the abstract speak of our verifier (a linear-code commitment), not of mixing in general.
- B-W3: the abstract says "hiding" and "the query's whole trace"; Theorem 2 assumes no state between proofs; the
  introduction says "no spot-checking verifier".
- C-W2: 2^-35 became 2^-23 (2^-34 under Fiat-Shamir) for the cnn18c tree with the fewest columns; the combined-row
  shares 2.2% / 0.5% (setup_bytes.py, cnn18c); "10-15x" from tab:eval-llm; the reproducibility sentence names
  OPT-125M and OPT-1.3B (D6a).
- Minor: D3 (one harness per row), D4 (4.7x, both non-interactive), D6 (about 2,100 tokens).

**Open, for the team or the final run:** A-W3 (base the zkLLM factor on the stored L40S runs: F6 todo in place),
C-W1 (placeholders until the L40S run; decide the fallback for missing cells), C-W3 (n and spread in captions,
Wilson intervals, CIs in tab:quality: needs space), the mode-C rows' plan (cnn16c interactive today), the
disclosure sentences (15-19 Oct), the artifact link, the fonts check on the final PDF.

---

# Early mock review of the submission body (12 Oct 2026)

**Scope.** I reviewed the abstract (`main.tex:71-74`) and the sections that `main.tex` inputs before the bibliography (`main.tex:76-82`): `intro`, `attack`, `defence`, `v1`, `generation`, `evaluation` (which inputs `quality` at `evaluation.tex:305`) and `discussion`. I cite the appendix only where the body depends on it.

**Scale.** Scores follow `paper_sp2027/REVIEW2.md:20`: 1 strong reject, 2 weak reject, 3 borderline, 4 weak accept, 5 strong accept.

**Scores.** A: 2. B: 3. C: 2.

---

## Reviewer A: deployment and practicality. Score 2 (weak reject)

The paper does not show, with numbers, a client for whom the protocol is the best option. The headline speed-up belongs to a client that once held the weights, and it ignores the time to transfer the proof.

### A-W1. The abstract's speed-up is for a different client than the one it advertises, and it leaves out transfer
- **Text.**
  - `main.tex:72-73`: "It lets a client without weights or GPU check an untrusted provider of an open-weight model, and a CPU client verifies a 2,048-token Llama-2-7B prompt 4.7 times faster than the fastest exact re-execution on the same CPU (non-interactive, full model)".
  - `discussion.tex:38-40`: "a client thus receives more bytes than the weights after one long prompt or about 21 short ones, and gains only if it cannot run the model."
- **Problem.**
  - The 4.7× is mode Kpre (`evaluation.tex:231-235`). A Kpre client "holds the weights and computes its key once" (`discussion.tex:31-32`).
  - The client without weights (mode C) has no 2,048-token time: `evaluation.tex:209` shows "Ours, C & roots & 325 MB & 6.6 GB & 1.0 s & --".
  - The ratio counts no transfer, but the Kpre proof is 6.4 GB per query (`evaluation.tex:210`).
  - The project's own break-even link is 0.69–0.70 Gb/s at 2,048 tokens and 0.51–0.59 Gb/s at 64 tokens (`docs/improvements/EV_deployment_case.md:32-33`). Below it, re-execution is faster (`EV_deployment_case.md:38-39`).
  - On the EPYC, every break-even link rises (`EV_deployment_case.md:55-56`).
  - The Discussion sentence quoted above contradicts the abstract.
- **Fix.** About +3 lines. Offset them by deleting the GPU TEE row of tab:options, whose cells are all "--" (`evaluation.tex:206`); the TEE is still discussed at `discussion.tex:26-27`.
  - Abstract: "A CPU client holding a key precomputed from the weights verifies a 2,048-token Llama-2-7B prompt at least N times faster than exact re-execution on the same CPU, on links faster than about 0.7 Gb/s." Sources: `EV_deployment_case.md:33`, and the "at least N" rule at `SP2027_PLAN2.md:88-97`.
  - Discussion, "Options for a client":
    - Give the bytes crossover: 17–25 mode-C queries of 64 tokens before the proofs exceed the weights (`EV_deployment_case.md:21-22`).
    - Give the break-even link.
    - Name the target client: no GPU, a link of 1 Gb/s or faster, few long prompts.

### A-W2. The provider's overhead is quoted only at its most favourable point
- **Text.**
  - `intro.tex:111-112`: "proving a 2,048-token OPT-1.3B prompt costs 4.1--8.6× our exact integer forward pass and 19--40× an fp16 one". The same sentence appears again at `discussion.tex:32-33` and `evaluation.tex:111-113`.
  - `discussion.tex:34`: "A provider may also serve with the exact model and prove later, on request, with the same proof."
- **Problem.**
  - 2,048 tokens is the best case. Across prompt lengths, mode C costs:
    - on OPT-1.3B, 8.6–21× the exact forward pass and 40–203× fp16;
    - on Llama-2-7B, 7.4–36× and 19–154× (`docs/improvements/A7_serving_overhead.md:342-346`).
  - "Prove later" does not remove the integer engine's own floor. That floor is 4–10× over fp16 on OPT-1.3B, "paid on every query if the provider serves the integer model" (`A7_serving_overhead.md:339-341`).
  - Proving only on request turns the per-query guarantee into a sampled audit. That is the property the SoK table criticises in SPEX.
  - The alternatives cost under 7% (GPU TEE, `intro.tex:165-167`) and 12–14% (CommitLLM, `intro.tex:175-176`). The paper never sets these side by side with its own overhead.
- **Fix.** Page-neutral.
  - Keep one overhead sentence, in the Discussion, with the full ranges. Reduce the copies at `intro.tex:111-112` and `evaluation.tex:111-113` to pointers.
  - Append to `discussion.tex:34`: "...but it still pays the integer engine's 4–10× over fp16 (`A7_serving_overhead.md:339`), and unproved queries are covered only statistically."
  - Add the provider-overhead column to tab:options, as already planned (`EV_deployment_case.md:57-58`).

### A-W3. The headline comparisons use different hardware, although a same-GPU baseline exists
- **Text.**
  - `intro.tex:109-110`: "On an L40S it proves a 2,048-token Llama-2-13B prompt 46× faster than zkLLM on an A100".
  - `evaluation.tex:252-253`: "as reported on their own hardware".
  - The tab:options caption mixes three machines (`evaluation.tex:191-194`).
- **Problem.**
  - Every zkSNARK except zkLLM ran on CPUs, against our L40S (`evaluation.tex:292-293`). The factors of 229× to 31,910× (`evaluation.tex:264-270`) therefore mix the system with the hardware.
  - Stored zkLLM runs on the same GPU model exist: 844 s for Llama-2-7B and 2,628 s for Llama-2-13B (`SP2027_PLAN2.md:1050`).
  - tab:options puts two 1.0 s values side by side that are not comparable. The mode-C value is the base EPYC verifier (`evaluation.tex:92`). The Kpre value is the native Xeon verifier (`evaluation.tex:236`).
- **Fix.** Page-neutral.
  - Base the intro factor and the zkLLM rows on the stored L40S runs, as `evaluation.tex:300-301` already plans. State their caveats: only the prover was timed, and one skip connection is counted twice (`SP2027_PLAN2.md:1050`).
  - First explain why those runs are 1.37× (7B) and 3.27× (13B) slower than zkLLM's A100 figures. I derived these from 844 s against 10.3 min and 2,628 s against 13.4 min (`evaluation.tex:275-276`).
  - Print factors only in same-hardware rows. Elsewhere, print the raw values.
  - Put each table on one platform (`EV_deployment_case.md:59-60`).

### Minor (A)
- The attack on a hardened sampler on an LLM is left open (`attack.tex:228`). But the adversary chooses the layer. At layer 23 of OPT-1.3B, the best mixture catches 0.08–0.09% of paths, and 2^-40 needs more than 30,000 paths (`G3_min_adaptive_game.md:312-313`). Lead with that layer, not with "Where they fail, value-aware sampling helps" (`attack.tex:223-224`).
- The abstract cites the out-of-range OPT-6.7B edit (`main.tex:72`, `attack.tex:216`). The in-range edit also changes the next token on 40/40 prompts (`attack.tex:229-231`). Cite that one; it answers the "add range checks" objection in advance.
- V1 has no regime where it helps a Kpre client.
  - V1 beats v0 end to end only below about 0.2 Gb/s (`evaluation.tex:181`).
  - Below 0.47 Gb/s, re-execution beats V1 (`EV_deployment_case.md:36`).
  - Say that V1 serves mode C on slow links.
- The GPU paragraph names "an auditor or gateway that checks a provider without holding the weights" (`evaluation.tex:244-245`). But the weights are open, and the GPU verifier is 10× slower than fp16 re-execution: 1.1 s against 0.11 s (`evaluation.tex:242-243`). Name the party that has a GPU but cannot hold open weights, or delete the sentence.
- Frame the threat model per campaign as well as per query. CommitLLM's 99.4% per one-token response (`attack.tex:261`) compounds over tampered queries. State that the protocol targets rare, targeted tampering, where a per-query guarantee is what matters.

---

## Reviewer B: theory. Score 3 (borderline)

### B-W1. "Tight" and "matching" overstate the bounds
- **Text.**
  - `intro.tex:126`: "A lower bound for spot-checking verifiers, tight at every locality".
  - `attack.tex:138`: "A Lower Bound and a Matching Upper Bound".
  - `attack.tex:173-174`: "Both bounds hold at every locality".
- **Problem.**
  - The lower bound requires qd ≥ (1−ε)N (`attack.tex:154-155`), which does not grow with λ.
  - The upper bound uses qd = O(λ+log L)·N (`attack.tex:158-159`, `:175-177`).
  - So the two bounds match only in N/d, and only up to a factor O(λ+log L).
  - The upper bound at block locality is proved only in the extended version. `attack.tex:177` cites `fullversion`, and the comment at `appendix_short.tex:14-15` records that the proof is not in the submission.
  - "Dichotomy" appears only in labels (`thm:dich`, `sec:dichotomy`), but the theorem is framed as one.
- **Fix.** Page-neutral.
  - `intro.tex:126`: "...met up to a factor O(λ+log L) at every locality".
  - Retitle the subsection "Lower and Upper Bounds".
  - `attack.tex:173`: "At every locality the bounds differ by a factor O(λ+log L) in qd."
  - Either put a one-paragraph proof of the block case in `appendix_short.tex`, or drop "at every locality" from the contributions.

### B-W2. Theorem 2(b) reads as if mixing were sufficient, but it holds for one construction
- **Text.**
  - `attack.tex:156-159`: "If each query mixes all rows (d=N), as the folded rows and code columns of Section 4 do, then ... O(λ+log L) queries per layer give soundness error 2^-λ".
  - `main.tex:72`: "while commitments that mix all rows reach 2^-λ with O(λ) queries".
- **Problem.** Support [N] is necessary, not sufficient. A symbol that sums one coordinate of every row has support [N] and checks nothing about Z. Proposition `prop:amp` (`appendix_short.tex:104-114`) is a statement about one protocol: Freivalds' check plus a Reed–Solomon column check, with an honest C_M and folds sent by the prover. It is not a statement about locality.
- **Fix.** Page-neutral.
  - Theorem 2(b): "The verifier of Section 4, whose folded rows and code columns each have support [N], has soundness error 2^-λ with O(λ+log L) queries per layer under ...".
  - Abstract: "while a Reed–Solomon commitment, whose symbols mix all rows, reaches ...".

### B-W3. Hypotheses of the lower bound are missing where the claim is made
- **Text.**
  - `main.tex:72`: "We prove that every verifier making q queries per layer to a weight commitment of locality d, even an adaptive one reading the whole trace, accepts such a forgery on some network with probability at least 1−qd/N".
  - `intro.tex:90`: "Since no verifier beats uniform sampling in the worst case, ours checks every weight operation."
- **Problem.**
  - The commitment must reveal nothing about unopened rows (`attack.tex:146-147`). An unsalted Merkle tree over low-entropy rows breaks this: detection becomes 2^(η+1)/N (`appendix_short.tex:126-128`).
  - The bound holds per query, for a stateless verifier (`appendix_short.tex:145-146`). In the hard instance, each honest trace shows which neuron reads the zero input. A verifier that keeps earlier traces therefore learns σ. Remark (iii) covers only cached openings.
  - The construction does not embed in transformers (`appendix_short.tex:148-149`; `attack.tex:172-173`). Condition (H) fails on trained networks (`discussion.tex:55-56`).
  - `intro.tex:90` quantifies over all verifiers, yet the defence itself is a verifier that beats uniform sampling.
- **Fix.** About +1 line.
  - Abstract: "...to a hiding weight commitment of locality d, even an adaptive one reading the query's whole trace...".
  - Theorem 2(a): "every such stateless verifier".
  - `intro.tex:90`: "Since no spot-checking verifier of locality d<N beats uniform sampling in the worst case".
  - Add one sentence to Sec. 3.4 that states the bound is per query.

### Minor (B)
- **Forced setup columns.** The claim that these columns are forced is a lower bound, and its proof is only in the extended version. "Complete test that sees only columns" is not defined (`defence.tex:221-223`). Put the counting argument in `appendix_short.tex`.
- **Hash collisions.** "The prover finds no hash collision" (`defence.tex:152-153`; `v1.tex:151-152`) is not a probability statement. Add the collision term, as tab:params does (`app_params.tex:30-32`).
- **Extension degree.**
  - "Fiat–Shamir needs degree at least 7" (`v1.tex:79`; `app_params.tex:35`) needs its calculation. With β = 200.8 (`app_params.tex:23`) and log₂p ≈ 30.9 (`app_params.tex:21`), p^7 ≈ 2^216 is the first power above 2^β.
  - State that ξ^8−11 is irreducible over F_p, and how this was checked.
- **Query budget.** The budget q is measured under the reference oracle R_p (`attack.tex:97-99`), which depends on the prior and on the adversary. Justify this, or add a worst-case budget.
- **Proof status.**
  - The tighter Kpre bound Q·p^-r is still open (`SP2027_PLAN2.md:888`).
  - The team's hand check of the proofs is also pending (same line).
  - The body cites `fullversion` 12 times, and the short appendix cites it twice.
- **Novelty wording.**
  - The "first" at `intro.tex:187-189` rests on four qualifiers. Keep "the novelty is this combination" and drop "first".
  - Move "adaptive, value-aware verifiers that read the whole trace" ahead of "folklore" (`intro.tex:153-158`).

---

## Reviewer C: systems and evaluation. Score 2 (weak reject)

### C-W1. The evaluation is mostly placeholders and earlier runs, across four machines
- **Text.**
  - `evaluation.tex:46-48`: "The final tables will come from one platform: ... with full models, no extrapolation, and medians with their spread \todo{L40S}. The numbers shown come from earlier runs".
  - All nine "new" cells of tab:eval-llm are \todo (`evaluation.tex:90-99`).
  - Llama-2-7B at 2,048 tokens has no prover time and no base CPU-verifier time (`evaluation.tex:98`).
  - Mode C at 2,048 tokens is not measured (`evaluation.tex:209`).
- **Problem.**
  - The evaluation section holds 36 \todo markers.
  - It mixes L40S with EPYC, 2080 Ti with Xeon, and extrapolated b/L builds (`evaluation.tex:60-61`, `:218`).
  - The submitted text promises tables that do not exist yet.
- **Fix.** Page-neutral.
  - Describe what is shown, not what will be shown.
  - Decide the fallback now: if a final-run cell is missing, use the measured full-model 2080 Ti/Xeon record and drop the "new" columns. Do not ship "--".
  - Put each table on one platform and name the machine in the caption.

### C-W2. Numbers disagree across sections, because two commitment plans and stale values are mixed
- **Smallest tree.** `defence.tex:211-212` says "about 2^-35 per query for the smallest tree of Llama-2-7B". tab:params gives t = 23 interactive and t = 34 under Fiat–Shamir (`app_params.tex:27`). 2^-35 is the old t = 35 (`D4_D5_untrusted_commitment.md:19`).
- **Opened columns.**
  - `defence.tex:219-220` quotes 65.6 MB of opened columns per query, which is the cnn18c plan (`app_params.tex:28`).
  - The measured 325 MB mode-C proofs are cnn16c, with 124.7 MB of columns (`PS2c_setup_cache.md:10`; `SP2027_PLAN2.md:737`; `evaluation.tex:92`, `:209`).
  - So the tab:params caption, "under the commitment plan of our runs" (`app_params.tex:10-11`), does not describe the current tables.
- **Setup-proof shares.** "1.6% ... and 0.4%" (`defence.tex:224-225`) are the shares for the 2.8/11.3 GB cnn16c setup proof (`D4_D5_untrusted_commitment.md:99-100`). The same sentence quotes 2.4/9.8 GB.
- **"Exact".** The word labels fp32 (`main.tex:73`; `intro.tex:116-117`; `evaluation.tex:216-217`, `:234`) and also "the exact integer forward pass" (`evaluation.tex:235`). The RX write-up labels fp32 "the fp model" (`RX_reexecution_baselines.md:32`).
- **GPU vs CPU verifier.** `evaluation.tex:110` says "11–15×". The table gives 27.4/2.6 = 10.5 for OPT-1.3B (`evaluation.tex:96`).
- **Reproducibility claim.** `defence.tex:51-54` says the integer OPT models reproduce bit for bit. Only OPT-125M and OPT-1.3B were checked (`SP2027_PLAN2.md:401`).
- **Fix.** Page-neutral.
  - Use one plan (cnn18c) for every mode-C number, and name it once in Setup.
  - Recompute the 2^-t term and the shares with `params_table.py` and `setup_bytes.py`.
  - Write "the fastest re-execution (fp32)", and add the exact integer pass, 141.6 s (`RX_reexecution_baselines.md:31`), as the like-for-like baseline.
  - Write "10.5–15×" and "OPT-125M and OPT-1.3B".

### C-W3. The abstract's only performance claim rests on a soft baseline and thin statistics
- **Text.**
  - `evaluation.tex:233-235`: "against 93.2 s for the fastest exact re-execution on the same threads (fp32; the exact integer forward pass takes 142 s): 5.0× (4.7×) faster".
  - `evaluation.tex:52`: "Times are medians after a warm-up."
- **Problem.**
  - The baseline runs on a Skylake Xeon without VNNI, where the native int8 GEMM is slower than MKL fp32 (`RX_reexecution_baselines.md:41-43`). The plan's own model puts the ratio at only about 2× on the EPYC (`SP2027_PLAN2.md:95`, `:590`).
  - The only int8 engine measured is torch dynamic int8 (`RX:24`, `:34`). Against it the ratio is 2.9× (`RX:55`).
  - The timings use 3 repetitions and 3 queries (`RX:26`, `:48`). The body reports no spread anywhere.
  - On the real OPT-6.7B, 3 of 3 queries were accepted and 9 of 9 tampers rejected (`F3a_real_weight_runs.md:298`). The body says only "every tamper rejected" (`evaluation.tex:59-60`).
  - tab:quality gives point estimates, although confidence intervals exist, for example ×1.0075 [1.0066, 1.0084] for OPT-6.7B (`F2a_quality_2048.md:194`).
- **Fix.** About +3 lines, taken from the reserve cuts C9, C10 and C13 (`H5_PASS.md:215`).
  - Give n and the min–max in every timing caption.
  - Add Wilson intervals to the accept/reject counts (`EV_deployment_case.md:59`).
  - Add the confidence intervals to tab:quality.
  - Add one row for an optimised CPU int8 engine, labelled not exact.
  - Quote "at least N" only after the EPYC cells exist (`SP2027_PLAN2.md:88-97`).

### Minor (C)
- **Quality of the benchmarked models.**
  - The headline model's certified quality is not reported. OpenLLaMA-7B stands in, at ×1.073 (`quality.tex:83-85`).
  - Qwen3-4B is benchmarked (`evaluation.tex:91`, `:271`), but its certified model is ×5.3 (`quality.tex:86-87`). Say that its costs depend only on its shapes.
  - The quality targets ×1.06, ×1.04 and ×1.05 have no stated origin (`quality.tex:76-78`).
- **V1 coverage.**
  - Generation is measured only on GPT-2 with random weights (`generation.tex:47-53`).
  - The Llama-2-13B V1 ratio comes from a single pruned block (`evaluation.tex:153-154`).
  - The V1 times compare against the earlier v0 verifier (`evaluation.tex:172-173`).
  - 5 of the 10 rows of tab:eval-v1 are untimed, and 3 more cells are \todo (`evaluation.tex:141-154`).
- **Bandwidth crossover.** The crossover at `evaluation.tex:178-179` includes the prover's added time. For the client alone it is 1.1 Gb/s (`EV_deployment_case.md:43-44`). Say which one is meant.
- **Artifact link.** The link to the extended version is still missing (`extra.bib:355-360`).

---

## Compliance lens

### Anonymity
- The author block is anonymous (`main.tex:66-67`).
- A grep of the non-comment body text found no names, institutions, URLs or acknowledgements.
- The figure PDFs carry only Matplotlib metadata.
- **Artifact link.** The extended version is cited as "link to be added at submission" (`extra.bib:355-360`). It must point to an anonymous host that serves the scrubbed export (`H4_submission_compliance.md:49-50`).
- **TeX comments.** They mention the session scratchpad and the "course paper" (for example `attack.tex:7-9`, `defence.tex:3`, `main.tex:7-8`). They do not reach the PDF. The export excludes paper sources (`H4:20-21`); never upload the sources.
- **Low risk.** The hardware pairs (`evaluation.tex:46-51`) describe one cluster configuration. Do not name nodes.
- extra.bib has 18 entries marked "[U]" (unverified).

### \todo markers in the submission body (non-comment lines)

| Section | Count | Lines |
|---|---:|---|
| Abstract | 1 | `main.tex:73` |
| 1 Introduction (incl. Contributions) | 4 | `intro.tex:88, 110, 118, 132` |
| 2 Related Work | 0 | |
| 3 Attack | 2 | `attack.tex:314, 316` |
| 4 Defence | 1 | `defence.tex:221` |
| 5 V1 | 0 | |
| 6 Generation | 1 | `generation.tex:55` |
| 7 Evaluation | 36 | Setup 3 (48, 60, 67); tab:eval-llm 19 (caption 77 + 18 cells); Cost 2 (110, 114); tab:eval-v1 3 (149-151); V1 text 2 (176, 181); tab:options 1 (194); re-execution 2 (238, 243); tab:eval-compare 3 (275 ×2, 276); comparison text 1 (300). Quality subsection: 0 |
| 8 Discussion / 9 Conclusion | 1 / 0 | `discussion.tex:45` |
| **Body total** | **46** | Short appendix and `app_params`: 0 |

### Desk-rejection risks
1. **Page limit.**
   - The rule is a 13-page body plus 5 pages of references and appendices, 18 pages in all (`REVIEW2.md:316`).
   - The last recorded build was 18 pages, with the body ending on p. 13 (`SP2027_PLAN2.md:1004`).
   - The body is projected at about 12.8 pages at H5b (`H5_PASS.md:215`). That leaves about 0.2 page before the 46 markers become numbers and before the fixes above are applied.
2. **Stale PDF.** `paper_sp2027/main.log:634` records 29 pages for the main.pdf of 10 Oct. Uploading that file would exceed the limit.
3. **Floats past p. 13.** tab:eval-llm and tab:eval-compare are `table*` floats (`evaluation.tex:73`, `:251`). If one floats past p. 13, it prints as unmarked body content (`REVIEW2.md:317`, `:328`). Check this in every build.
4. **Abstract freeze.** The abstract is frozen at registration on 10 Nov (`REVIEW2.md:322`). It still has a red \todo and a number that depends on the EPYC run (`main.tex:73`).
5. **Disclosure stated before it happens.** `intro.tex:87-88` and `attack.tex:313-317` are written as conditional on the notices of 15–19 Oct. Make them facts, or delete them. The deadline for disclosure is the rebuttal (`REVIEW2.md:320`).
6. **Fonts.** The template is compsoc (`main.tex:13`) with no spacing changes. The stale log shows TU/ptm font warnings (`main.log:19-39`). Re-check the embedded fonts in the final PDF (`SP2027_PLAN2.md:33`).

### Not verified
- No fresh build, because the task was read-only. The page figures come from `SP2027_PLAN2.md:888` and `:1004`, and from `H5_PASS.md`.
- I did not check the cited papers against their sources.