# H5 pass: editor's record (2026-10-11)

Work list: REVIEW2.md, Sections 5, 8 and 9. Drafts integrated: `drafts/th_a_block_mixing.tex` with
`drafts/th_a_check.md` (TH-a), `drafts/th_c_setup_forced.tex` (TH-c) and `drafts/th_d_citations.md` (TH-d).

Constraints kept:
- Only files under `paper_sp2027/` were edited. `drafts/th_e_proofs.tex`, `drafts/th_e_check.md`, `code/` and `docs/` were not touched, and git was not used.
- Every edited file keeps its original line endings.
- Builds ran one at a time, with tectonic, into `C:/Users/User/.claude/jobs/fcef220b/tmp/pbuild_h5/<name>`. Baseline: `base/`. Final: `final4/`.

Line numbers below refer to the files as they are now.

## 1. Page facts and `\todo` counts

| | Baseline (`base/`) | Final (`final4/`) |
|---|---|---|
| Submission `main.pdf`, pages | 18 | 18 |
| Where the body ends (References heading) | p. 14, top of left column. Body = 13.00 pages; the Conclusion fills p. 13 to the bottom. | p. 13, left column, y = 281 of 71-720. Body = 12.16 pages. |
| Two-column floats (`table*`) | Table 2 p. 11, Table 5 p. 13 | Table 2 (`tab:eval-llm`) p. 11 top; Table 5 (`tab:eval-compare`) p. 12 top. Both are before the references. |
| References + appendices, measured from the References heading to the last line | 4.92 pages (references 2.28, appendices 2.64) | 4.98 pages (references 2.30, appendices 2.68). The last page, p. 18, is about 15% full. |
| Extended `main_ext.pdf`, pages | 34 | 39. Same body: it ends on p. 13 at 12.16. |
| `\todo` in the submission build | 72 | 71 |
| `\todo` in the extended build | 105 | 107 |
| Undefined references or citations; multiply defined labels; BibTeX warnings | none | none, in both builds |

Notes on the table:
- The `\todo` counts come from the expanded source without comments, and agree with the "[TODO:" count in each PDF.
- The only log warnings left are the pre-existing `TU/ptm` font-shape warnings.
- Fonts in the final `main.pdf` (PyMuPDF): NimbusRomNo9L (Times), the CM fonts, MSBM10, NimbusMonL, and STIXGeneral on p. 1. The baseline already had STIXGeneral on p. 1.

**Intermediate builds.**

| After | Body ends at |
|---|---|
| P1 (`p1/`) | 12.42 pages |
| The review's first batch, P1-P3 and C1-C8 (`cuts1/`) | 11.92 pages |
| The N fixes, TH-a's body sentence, TH-c's remark and the TH-d footnote | 12.16 pages |

**Appendix size.** Moving P1's material into `appendix_short.tex` first raised references + appendices to 5.38 pages. Compacting that subsection and merging repeated pointers to the extended version brought it back to 4.98 pages (Section 3).

**Where I stopped cutting, and why.**
- I applied the batch that REVIEW2 §8 says to apply first: P1-P3 and C1-C8.
- I then added the writing fixes and the TH sentences, which brought the body to 12.16 pages.
- I added no further cuts. REVIEW2 §8 sizes the pending body results at about 0.6 page:

  | Item | Pages |
  |---|---:|
  | EV | 0.25-0.35 |
  | A7 | 0.05 |
  | F3a | 0.05 |
  | RX | 0.03 |
  | F2, F2c | 0.03 + 0.03 |
  | V-LONG-b | 0.03 |
  | G3.5-L | 0.02 |
  | C4a | 0.02 |
  | F4 | 0.02 |
  | the TV sentence of E2b | 0.01 |
  | TH(g) | 0.02 |

- That puts the body at about 12.8 pages at H5b. It falls to about 12.7, the review's target, once the todo text in the body is replaced by numbers.
- The remaining cuts are a reserve:
  - C9, C10 and C13: about 265 words, about 0.25 page.
  - C11 and C12: tied to EV and to F6 `gen` (see Section 2).

## 2. Per ID

"Words" counts source words without comments, old to new, computed against a copy of the sources taken before the pass.

### Cut list (REVIEW2 §9)

| ID | What changed, and where | Words |
|---|---|---|
| **P1** | Demoted subsection 4.5 to the paragraph "Untrusted commitments" at the end of Sec. 4.4 (`sections/defence.tex:204-223`). It keeps `\label{sec:setup-proof}`. It keeps: the split commitment, with the acceptance product inline; the setup proof and its sizes (2.8 / 11.3 GB; 124.7 to 168.5 / 134.9 MB); TH-c's remark; the clause that `M^*` need not be int8 without a range proof; and "for open weights, recompute C_M". Fact `fact:ext` and Theorem `thm:untrusted` (compacted) replace the proof idea of `app:untrusted` in `sections/appendix_short.tex:200-233`, with a short proof idea that cites BCIKS20 Thm. 1.6. The extended version gets everything in full in `sections/appendix_defence.tex:154-310`: the motivation with display `eq:split`, the extension-field paragraph with Fact `fact:ext` and its proof, the setup proof's four steps, the theorem, the remark on `M^*`, and the full proof. Updated the extensions (`intro.tex:99-104`) and the contributions (`intro.tex:125-127`). | Body: subsection 783 to paragraph 285 (-498), including about 45 words of TH-c and 20 of the `M^*` clause. `appendix_short.tex`: 181 to 248 (+67) for that subsection. Body ends 13.00 to 12.42 pages. |
| **P2** | Dropped the Proof-of-Learning row of `tab:sok` (`attack.tex`, the table at `:218-281`). The row's content moved to `ext_attack.tex:170-172`. "Nine published schemes" now matches the table. `jia2021pol` and `zhang2022pol` are no longer cited in the submission. They stay in `extra.bib` for the extended version, which cites them. `fang2023pol` stays (`attack.tex:290`). | Row about -29. Net -7 with TH-d's footnote. |
| **P3** | Dropped the two LeNet-5 rows of `tab:eval-compare` (zkCNN and Bionetta) and removed `bionetta` from that paragraph's cite list (`evaluation.tex:236-273`, `:275-276`). The rows' numbers moved to the paragraph "LeNet-5." (`ext_evaluation.tex:209-214`). Bionetta stays cited in Related Work. | 421 to 383 (-38) |
| **C1** | Shortened the "A lower bound" paragraph (`intro.tex:71-87`). It no longer restates the verifier model or carries the PCPP sentence. It keeps the reduction to a one-row model, the bound with condition (H) in one sentence, the hard ReLU instance, the mixing converse (now with TH-a's block clause), and the SoK sentence with the TensorCommitments verdict and the disclosure todo. | 270 to 218 (-52). The review estimated 150. The difference is the kept (H) clause and TH-a's clause. |
| **C2** | One clause per extension (`intro.tex:99-104`). Removed the weak-Fiat-Shamir citations (they remain in `defence.tex`) and V1's mechanism. | 147 to 74 (-73) |
| **C3** | The folklore/PCPP point now appears once, in Related Work. It is removed from the introduction (with C1) and from Sec. 3.4 (the former `attack.tex:159-161`, 34 words). Related Work gains one sentence: Theorem `thm:guess` formalises the folklore calculation, and the model must be encoded (`intro.tex:154-158`). | Sec. 3.4: -34. Related Work: +37. The introduction's share is counted under C1. |
| **C4** | MNIST paragraph (`attack.tex:182-186`). It keeps 1 - 1/N and the backdoor's 99.96%. The 250-path figure (also in `tab:sok`), the 95th-percentile sentence, and its footnote with `\todo{G3: training-split ranges}` moved to "MNIST, further results" (`ext_attack.tex:126-130`). | 104 to 60 (-44) |
| **C5** | "Commitment plans" is now one sentence (`defence.tex:93-95`). It keeps a one-clause definition of the transposed layout, since `thm:sound` uses the term. In (S3), the transposed-layout check became one clause (`defence.tex:132-133`). The plan rule, GPT-2's head, the embedding tree and the transposed check moved to "The plan and the transposed layout" (`ext_defence.tex:69-78`). | -65 (78 to 32 and 124 to 105) |
| **C6** | Folded "A sponge transcript" into one clause of the binding paragraph (`defence.tex:185-190`). Its Merkle-Damgård remark (`coron2005md`) moved to `ext_defence.tex:84-86`. | 144 to 105 (-39) |
| **C7** | Comparison paragraph (`evaluation.tex:281-287`): removed the restated per-row factors and kept the two caveats and one clause per cost. The DeepProve V1 figure stays in the extended `tab:eval-compare-v1`. | 203 to 156 (-47) |
| **C8** | Brought V1 to its target length. (a) Removed the Llama-2-13B error terms; they stay in `tab:params` footnote b and in `ext_v1.tex` E6. (b) "Implementation and cost" is now one sentence pointing to Sec. 7.3 (`v1.tex:183-185`). (c) Shortened the GKR packing description (`v1.tex:101-106`); details stay in `ext_v1.tex` E1. (d) Deleted the lemma's explanation, which `appendix_short.tex` "Proof ideas" has. (e) Cut the opening to one sentence (`v1.tex:40-41`). The todo "L40S: full models" in `v1.tex` is removed; the evaluation's V1 timing todos cover it. | 1,524 to 1,344 (-180) |
| C9 | Skipped. Not needed to reach the target (Section 1). Reserve of about 65 words. | -- |
| C10 | Skipped, for the same reason. Reserve of about 155 words. | -- |
| C11 | Skipped. It turns "Options for a client" into EV's target-client paragraph, so it belongs with EV. | -- |
| C12 | Skipped. "Replace the GPT-2 numbers rather than adding the 7B ones" applies when F6 `gen` lands; dropping the zkAgent comparison is part of the same edit. | -- |
| C13 | Skipped. Not needed for space. The ethics paragraph still holds the G5/H4-c todos, which should be resolved together. | -- |

Fig. 1 and the soundness-test counts (`evaluation.tex:60-64`) stay in the body.

### Defects (REVIEW2 §5)

| ID | What changed, and where |
|---|---|
| **N1** | `thm:guess` uses the fractional `G_t` of the appendix: a maximum over `x` in `[0,1]^N` with `sum x_u <= qd` (`attack.tex:107-113`). It reads "for integer qd ... guessing J in qd tries", and the attainment sentence says "for d = 1 and integer q". |
| **N2** | "a path opens them with median probability at most 7.9·10^-4 in every cell (at most 7.3·10^-3)" (`attack.tex:208-210`). I recomputed this from `code/artifacts/results/inrange_llm_opt-1.3b_any.json`: 320/320 successes, median 2 and maximum 60 neurons, largest per-cell median 6.5/8192 = 7.9·10^-4, maximum 60/8192 = 7.3·10^-3, and 76% of the edits below 8·10^-4. The todo is renamed `\todo{G3.5-L: ...}`. |
| **N3** | Footnote a of `tab:options` now says the re-execution rows are not yet the strongest exact baseline (`evaluation.tex:205-208`). The GPU row predates the int8 forward pass, which halves it on Llama-sized layers (`F1_reexecution_baseline.md` §3; `A2_int8_forward.md`). The CPU integer row runs its weight products as fp32 GEMMs over 1,024-term chunks (`A2_int8_forward.md` §3). Added `\todo{RX}`. |
| **N4** | The modes are labelled. The `tab:eval-v1` caption says the v0 verifier times predate the final CPU kernels (`evaluation.tex:123-124`). "Time" and "Bandwidth crossover" say they use the earlier v0 verifier (`:163-166`, `:169-171`), with `\todo{V-LONG-a: ...}`. The `tab:options` caption says "ours interactive, the V1 row from the Fiat-Shamir runs of Table 3" (`:180`). |
| **N5** | BCIKS20 is cited as "Thm. 1.6" of the full version (ePrint 2020/654 of 3 July 2021). See `appendix_short.tex:228-230`, `appendix_defence.tex:263-276` (TH-d 1.2(c), with Ligero Lemma 4.2), and the `bciks20` note and DOI (`extra.bib:290-302`). Both `\todo{FOCS numbering}` are deleted. `defence.tex` no longer cites BCIKS20. |
| **N6** | Sec. 4.4 "Soundness" now says that the round-by-round error is the largest per-round probability, not the sum, so it is at most `eps_C`. It adds that the parameters keep the union bound for simplicity, and that dropping it would save log2(2·|J|) bits of beta and leave r = 7 (`defence.tex:192-202`; TH-d 5.2(a), condensed). The extended appendix states `eps_rbr` as a maximum (`appendix_defence.tex:142-148`; TH-d 5.2(b)) and adds that `beta_0` is conservative (`:286-287`; TH-d 5.2(c)). No `\todo` is needed: the sum is kept and labelled. Making the interactive theorem itself max-based is left for TH-f. |
| **N7** | One statement, in one place. The introduction's "Such tests could shorten our setup proof" became a pointer: "Section `sec:setup-proof` bounds what such tests can save on our commitment" (`intro.tex:177-178`). TH-c's remark is in the setup paragraph (`defence.tex:219-223`). `ext_intro.tex:147-151` and `ext_defence.tex:167-170` were reconciled with TH-c Blocks 5(b) and 5(c). |
| **N8** | Deleted "Its exact attention alone is faster than fp32 SDPA (0.12 against 0.14 s ...)" (`evaluation.tex`, paragraph "CPU", `:217-223`). `docs/improvements` has no source for it. B5 §6 gives 0.278 s (native tiled) against 0.153 s (SDPA) for an earlier kernel, job 1005733, which points the other way. |
| **N9** | One source: the V1 specification's estimate of 1.78-1.85x at 2,048 tokens (`C2_v1_proof_size.md` §8). Used in `evaluation.tex:157-159` (with `\todo{F3a: ...}`), in footnote d of `tab:options` (`:209-210`), and in `ext_v1.tex:170-171` (replaces "1.8-2.0x with the codec rates", with `\todo{F3a}`). The rounded "÷1.8-2.3" in the table row is unchanged. |
| **N10** | `quality.tex:22-23` labels the calibration window as "the last 128-token window of the same test split, which no scored window includes", with `\todo{F2a: train-split calibration}`. The closing todos are named by item (`:71-72`). |
| **N11** | Added the aggregate bound (`generation.tex:88-93`): the excluded probability is below (V-1)e^-14.25, about 0.02 for Llama-2 (32,000 tokens), 0.03 for GPT-2 and OPT (about 50,000), and 0.1 for Qwen3 (151,936). Computed values: 0.0207, 0.0325 and 0.0984. REVIEW2's 0.09 for Qwen3 is slightly low. The threshold now reads "about 14.25", with "e^-14.25 < 6.5·10^-7", as in `ext_generation.tex`. E2b's TV distance stays `\todo{E2b: ...}`. |
| **N12** | Checked in both builds: the `table*` floats land on p. 11 (`tab:eval-llm`) and p. 12 (`tab:eval-compare`), the references start on p. 13, and nothing from the body prints after p. 13. No move was needed. Re-check after EV's figure and the F6 cells are added. |
| **N13** | Abstract (`main.tex:72-73`): "4.6 times faster than it could re-execute it in fp32 (interactive proofs; extrapolated from 1-3 blocks) `\todo{RX: "at least N times" against the strongest exact baseline}`". The 512-token figure is gone. The abstract now says "makes 2,048-token proofs 2.0-2.3 times smaller with random weights `\todo{F3a: real weights}`". The introduction's 4.6x got the same interactive/extrapolated label (`intro.tex:114-116`). |
| **N14** | `discussion.tex:31-33` labels the comparison: the base prover took 1.8 s on the L40S, and an fp16 pass took 0.037 s on a different GPU, the 2080 Ti, "so the two give no overhead ratio", with `\todo{A7: prover against unverified inference on the same GPU}`. |
| Smaller notes | The `tab:options` caption names the protocol mode (see N4). DeepBrake is cited as the version of 3 Aug 2026, "since retitled LiftWHIR" (`extra.bib:343-356`). The revised title is taken from `extra.bib`'s own caveat; I did not re-check it. |

### Theory drafts

**TH-a** (all of `th_a_check.md` applied)
- **Part A.** One body sentence pair in Sec. 3.4 (`attack.tex:166-171`): both bounds hold at every locality. Opening blocks attains 1 - qd/N for d dividing N. Block-wise checks keep the error of d = N with O(λ + log L) queries per block, counting one per folded row and per opened column. Each answer is a folded row or a block column. It cites the extended version, so the body references no extended-only label.
- **B1.** In `intro.tex:79-81`, with O(λ + log L) and a pointer to Sec. 3.4, as the check asks.
- **B2.** "tight at every locality" (`intro.tex:123`), which the check recommends over "matched up to O(λ)".
- **B3** (optional) not taken. Its fix n ≥ 2(k-1) is added to `thm:dich(b)` (`attack.tex:150`) and to its proof in `appendix_attack.tex:237`.
- **B4** (optional, abstract) not taken.
- **B5.** In `ext_attack.tex:89-99`, citing `prop:block(i)` and `cor:block`, with O(λ + log L).
- **Part C.** In `appendix_attack.tex:246-508`: `prop:block`, the new `lem:worst-int`, `cor:block`, a counting-convention note, `rem:block-count`, `rem:block-scope`, `eq:block-*`. Fixes applied:
  - admissible set limited to integer networks within the range bound;
  - folded rows do not depend on salts;
  - "the bound of prop:amp";
  - t = min(⌈β/log2(n/(k-1))⌉, k), which makes the product 0 for t ≥ k. The same cap and product form are added to `prop:amp` in `appendix_short.tex:105-114` and `appendix_attack.tex:194-223`;
  - the claim range check;
  - networks without embedding lookups;
  - the interactive union bound noted as loose;
  - (b) restated over the integer hard family of `lem:worst-int`: int8 weights, integer biases, multiplier 2^30, ReLU table, logits (1, 2g), 128(1+127c) < 2^29, 129N < 2^29;
  - the ideal-oracle model;
  - O(λ + log L) stated as an artefact of the counting convention, with the 2m and (N+d-1)/N alternatives;
  - `rem:block-count(iii)`: the second condition k > φt/ω (about 280 at t ≈ 70), r = 7 giving about 28 rows, the m·t salts, "not measured";
  - `rem:block-count(iv)`: redundant columns;
  - `rem:block-scope(ii)`: log2(Σ m_ℓ/|J|) and max_b e_b;
  - `rem:block-scope(iv)`: the probability-1 attack on an unchecked block, and the last sentence replaced by the check's corrected argument.
- **Part D** not placed. No appendix text of the same size was available: the submission appendix was at the 5-page cap after P1.
- The draft's header claims ("n = 4 next_pow2(k)", "the optimum is the same for every partition") are not in the paper.

**TH-c**
- **Block 1** (`rem:forced`, its proof, the Merkle/Fiat-Shamir paragraph and the scope) is at the end of `app:untrusted` in `appendix_defence.tex:315-401`.
- **Block 2** is in `ext_defence.tex:141-161`, after `tab:setup`, with a source comment: totals from D4_D5 §5, breakdown from the `setup_bytes.py` byte model.
- **Block 3** (variant A) is in the demoted paragraph (`defence.tex:219-223`), with "1.6% of the setup proof" and the STIR/WHIR citations.
- **Block 4** (optional appendix sentence) skipped, for lack of appendix space.
- **Block 5:** (a) became the pointer of N7; (b) and (c) applied (`ext_intro.tex:147-151`, `ext_defence.tex:167-170`); (d) is a rebuttal note and is not paper text.
- **STIR and WHIR** added to `extra.bib:380-409`. I re-checked their bibliographic data on 2026-10-11:
  - Crossref records of the DOIs give titles, authors and pages (380-413; 214-243).
  - The LNCS volumes (14929, CRYPTO 2024 Part X; 15604, EUROCRYPT 2025 Part IV) come from the Springer listings matched by print ISBN, read from search results. Springer and dblp pages were blocked.

**TH-d** (citation fixes)
- **1. BCIKS20:** see N5. The `bciks20` entry now has the DOI and a note naming the version.
- **2. Ligero:** Lemma 4.2 for e < d/4 (`appendix_defence.tex:276-278`). The optional d/3 footnote was not added.
- **3. Anchuri et al.:**
  - footnote on the version and on the refereed variant in `tab:sok` (`attack.tex:275-276`), citing arXiv:2603.19025v1 in text, with no new submission reference;
  - the refereed-variant paragraph and the version entry `anchuri2026full` in the extended version (`ext_attack.tex:173-185`; `extra.bib:410-419`). It is a separate entry because `references.bib` is copied unchanged.
- **5. B-Q6:** see N6.
- **6. B-W6:** paragraph "Relation to deployed logUp" in `ext_v1.tex:56-70`, with `stwo2026`, `openvm2025logup` and `lurk2024lookups` (`extra.bib:421-445`). The `haboeck2022logup` comment is now [V] (`extra.bib:204`). The optional body clause in `v1.tex` was not added.
- **Not integrated, because they are not citation fixes:**
  - 4 (B-Q5): REVIEW2 already counts B-Q5 as answered.
  - 7 (A-Q7, the tokenisation sentence): a TH(g) item worth about +0.02 page in the ledger, left for the authors.

## 3. Other changes made to keep within the limits

**Submission appendix.** This trimming brought references + appendices from 5.38 to 4.98 pages. No statement was removed.
- The proof idea of `thm:untrusted` is compressed.
- The setup-proof description is folded into the theorem's hypothesis.
- Inline fractions became slashed in the `thm:untrusted` bound and in `prop:amp`.
- `rem:pointer` and the row-layout paragraph lost one line each.
- One sentence in the "Setting" paragraph (`appendix_short.tex:31-32`) now says that the full proofs of all appendices are in the extended version. It replaces six repeated pointers.

**Todo renames.** Todos I touched now name their items, for example "L40S" became "F6" or "G3.5-L" where the owner was clear, and `evaluation.tex:108` became A7.

**File headers.** Every edited file's header has a short "H5 pass" note.

## 4. Remaining `\todo` markers by item

**Submission build: 71.**

| Item | Count | Where |
|---|---:|---|
| F6 (L40S run, all tiers) | 36 | `intro.tex:108` (with A7), `:129`; `defence.tex:219`; `generation.tex:54`; `evaluation.tex:48`, `:64`, `:74`, `:87-96` (18 cells), `:107` (F4), `:143-145` (3), `:183` (with EV), `:228`, `:260-261` (3), `:285`; `discussion.tex:44` |
| TH-b | 9 | `app_params.tex:21`, `:22`, `:26`, `:27`, `:41`, `:42`, `:44`, `:46`, `:55` |
| F3a | 5 | `main.tex:73`; `evaluation.tex:56`, `:159`; `quality.tex:72`; `discussion.tex:47` |
| RX | 4 | `main.tex:73`; `intro.tex:115` (with V-LONG-a); `evaluation.tex:208`, `:223` (with V-LONG-a) |
| F2a, F2b | 4 | `quality.tex:23`, `:71` (two); `discussion.tex:48` |
| G5, H4-c (disclosure) | 3 | `intro.tex:86`; `attack.tex:295`, `:297` |
| V-LONG-a | 2 | `evaluation.tex:165` (with F6), `:171` (with EV) |
| A7 | 2 | `evaluation.tex:108`; `discussion.tex:33` |
| G3.5-L | 2 | `attack.tex:212`; `discussion.tex:53` |
| G3-min | 2 | `attack.tex:128` (G1e), `:194` (adaptive game) |
| E2b | 1 | `generation.tex:92` |
| D6a | 1 | `defence.tex:52` |

**Extended build: 107.** These are the 71 above plus 36 in the extended-only files:

| File | Count | Lines and items |
|---|---:|---|
| `ext_intro.tex` | 6 | `:49`, `:52`, `:57`, `:60`, `:62` (F6); `:68` (F2) |
| `ext_attack.tex` | 1 | `:129` (G3) |
| `ext_defence.tex` | 1 | `:117` (F6 setup time) |
| `ext_v1.tex` | 5 | `:169`, `:181` (F6); `:171`, `:173` (F3a); `:189` (GPU verifier with V1) |
| `ext_generation.tex` | 1 | `:115` (F6) |
| `ext_evaluation.tex` | 11 | `:92`, `:104`, `:124`, `:143`, `:182`, `:216` (F6); `:110` (F4); `:169` (two, B5); `:203`, `:204` (V1 cells) |
| `ext_discussion.tex` | 11 | `:38`, `:62`, `:112`, `:141` (F6, A7); `:40` (F4); `:51` (F3); `:95` (CommitLLM, EV); `:108` (F7); `:126` (B5); `:152` (G3); `:154` (G3.5) |

Two of the extended-only todos are stale. I left them because they are outside this work list:
- `ext_discussion.tex:154`, "G3.5: in-range LLM attack": the OPT-1.3B result is now in the body.
- `ext_evaluation.tex:169`, the B5 cells of `tab:eval-reexec`.

## 5. Open points for the authors

- **Page limit.**
  - The submission meets "body ≤ 13 pages, total ≤ 18" with margin: the body ends on p. 13 at 12.16 pages.
  - References + appendices come to 4.98 pages, measured from the References heading.
  - Any growth of `appendix_short.tex` or of the reference list must be offset in the appendix.
- **Projected length at H5b.** Adding the results sized in REVIEW2 §8 gives a body of about 12.8 pages before the todo text is replaced. C9, C10 and C13 are the reserve cuts.
- **TH-a open points** (in the draft's own list): one sentence in `def:spot` fixing the counting convention would remove the O(λ + log L) caveat; a lower bound on answer size is not claimed.
- **TH-c Block 2.** Its breakdown numbers come from the `setup_bytes.py` byte model, as TH-c recomputed it. No doc in `docs/improvements` records the breakdown yet. Adding it to D4_D5 would give the 1.6% / 0.4% figures in the body a citable source.

## 6. Review (second agent, 2026-10-11)

I read the H5 diff against git HEAD (`git diff -- paper_sp2027`, read-only), this record, REVIEW2 §5, §8 and §9, and the three drafts with `th_a_check.md`. I rebuilt both versions twice: once as received (`pbuild_h5/rev0/`) and once after my fixes (`pbuild_h5/rev2/`). I touched no file in `code/` or `docs/`, nor `drafts/th_e_*`, and I did not use git beyond the read-only diff.

### 6.1 Builds after the review (`rev2/`)

| | Submission | Extended |
|---|---|---|
| Pages | 18 | 39 |
| References heading | p. 13, left column, y = 280; body = 12.16 pages | same |
| `table*` floats | Table 2 p. 11, Table 5 p. 12; no body float after p. 13 | same |
| References + appendices | about 4.99 pages (p. 18 is about 15% full) | n/a |
| Undefined or multiply defined labels or citations; BibTeX warnings | none | none |
| Overfull boxes | none (was 1, see F1) | 2, both pre-existing in `ext_attack.tex` tables (in `base/` too) |
| `[TODO:` in the PDF | 71 | 105 (was 107; see F8) |

A static scan (`\label` and `\ref` without comments, per build) finds no duplicate label, no undefined reference, and no body reference to a label that only an extended-only file defines.

### 6.2 What I checked, and the result

**Meaning of the cuts.** P1-P3 and C1-C8 remove nothing that the abstract or a later section relies on. The material is in the extended files: C4 in `ext_attack.tex`; C5 and C6 in `ext_defence.tex`; C8 in `ext_v1.tex` (E1, E6, and the V1 tests at :107-109); P1 in `appendix_defence.tex`; P2 and P3 in `ext_attack.tex` and `ext_evaluation.tex`. Specific checks:
- "Nine published schemes" matches the eight rows of `tab:sok`, TOPLOC and SVIP sharing one.
- After P3, the comparison paragraph's range "46x to about 32,000x", "slower in two rows" and "up to 56,000x" still match the remaining rows of `tab:eval-compare`.
- `sec:setup-proof` now resolves to Sec. 4.4.

**Numbers (36 checked, all match their sources).**
- D4_D5 §1 and §5: 2.8 / 11.3 GB; 124.7 -> 168.5 / 134.9 MB; t = 35 and about 2^-35.
- B5 §6 (job 1006792): 17.2 s, 78.8 s fp32 and 98.3 s integer; 4.6x and 5.7x; 1.6 s, 3.5 s and 3.6 s at 64 tokens; "extrapolated from 1-3 blocks". N8's deletion is right: B5 §6 has 0.278 s against 0.153 s.
- C2 §8: 2.0-2.3x; 1.78-1.85x; 1.68x and 1.77x; 0.50 -> 4.85 s with 3.58 s GKR.
- G3 §3 and §4: 8,192 tokens; 40/40 at layers 6, 12, 18 and 23; median 2 (max 60); 7.9e-4 = 6.5/8192; 7.3e-3 = 60/8192; 9, 31, 38 and 39.
- F1 §3 and A2 §1, §3, §5: 0.037 s fp16 on the 2080 Ti; "halves" (the int column is before A2; about 2x on Llama-sized layers); 1,024-term fp32 chunks, which the CPU path still uses.
- N11: (V-1)e^-14.25 = 0.0207, 0.0325 and 0.0984; e^-14.25 = 6.47e-7; 14.25 = 11.78 + 2.47.
- N6: beta_0 = 200.02 and 192 both give r = 7.
- The LeNet-5 paragraph equals the deleted rows.

**TH-c, recomputed with the formulas of `setup_bytes.py`.** The code is imported read-only, and no file under `code/` changed.
- Llama-2-7B setup: policy cnn16c, |J| = 130, 6 trees with n = 65,536, s = 7.
- a = 100: rows 45.17 MB of 2,833.45 MB (1.594%), columns 98.25%, non-column 1.75%.
- a = 1,000: rows 0.401% of 11,272.73 MB, non-column 0.55%.
- Floors at 2^-128: 1,788.7 MB (63.1%) and 7,284.7 MB (64.6%).
- Four models: non-column at most 3.91% / 2.29%; floors 62.0-64.6%.
- Approximation error 0.85-1.67% / 3.69-6.85%; t0/n up to 0.123.
- The t0 values match the draft per tree.

So the body's "1.6% / 0.4%" and every number of Block 2 are right. They are still missing from `docs/improvements` (6.4).

**TH-c remark.**
- Block 1 and Block 2 are byte-identical to the draft. Block 3 (variant A) and Blocks 5(a)-(c) are placed as described.
- I tried to refute the remark and could not. The committer of (i) is (e_j+1)-far from U, and at least n-k-e_j > e_j far from any other codeword matrix. Until a leaf in B is queried, the coupled runs agree whatever extra oracles or messages the test uses, because the cheating committer sends the honest committer's messages. The leaves an honest run queries do not depend on B. And C(n-t,b)/C(n,b) equals the product.
- (iii) matches `setup_params`: `t0` is the maximum over members of `survivors(n-e-1, n, beta_0)`, at beta_0 = lambda + 64 + log2(2|J|).
- Two wording defects, both in the extended text: F3 and F4 below.

**TH-a.**
- The body sentence (Sec. 3.4), B1 (with O(lambda + log L)), B2 ("tight at every locality"), B3's n >= 2(k-1) and B5 match the draft and every point of `th_a_check.md`.
- In the extended Part C, all of the check's fixes are present: t cap, admissible set, salts, range check, lookups, integer family, ideal-oracle model, counting-convention note, rem:block-count (iii)/(iv), and rem:block-scope (ii)/(iv).
- One defect in `lem:worst-int`, carried over from the check itself: F2.

**TH-d.** Each item was verified against the source:
- **BCIKS20:** Thm. 1.6 is "Correlated agreement over affine spaces" in the ePrint PDF created 2021-07-03, with error n/q in the unique-decoding regime, read from the local copy.
- **Ligero:** Lemma 4.2, for e < d/4 with error (e+1)/|F|, from the CCS'17 PDF, p. 7.
- **logUp:** Lemma 5, which needs p > N.
- **S-two whitepaper:** Remark 17 and the seven authors; ePrint 2026/532 from the ePrint page, last revised 24 Mar 2026.
- **arXiv:2603.19025:** v1 only (19 Mar 2026), and the seven authors match `anchuri2026full`. App. D is "Our Protocol in the Refereed Model", with Thm. 3 on p. 38.
- **DeepBrake:** posted 03 Aug 2026 (IACR news item 29151), and the ePrint page now has the title "LiftWHIR".
- **WHIR:** pp. 214-243 and the print ISBN, from Crossref. LNCS 15604 and EUROCRYPT 2025 Part IV come from search results.
- **STIR:** CRYPTO 2024, LNCS 14929, pp. 380-413, from search results (Springer, ACM). Crossref was rate-limited.
- **Lurk post (17 Sep 2024)** and the GitHub path of the OpenVM note both resolve.
- **Not verified:** the FOCS and J. ACM numbering of BCIKS20, which the paper avoids by citing the full version.

### 6.3 Defects found and fixed

| # | Defect | Fix |
|---|---|---|
| F1 | N1's new display in `thm:guess` was 31 pt too wide, in both builds (the only new overfull box). | The constraint is now stacked with `\substack` (`attack.tex`). The body length is unchanged. |
| F2 | `lem:worst-int` (extended only) bounded the hidden rows by 128(1+127c) with c <= 33,000. The weights are uniform int8, which go down to -128 in this paper (`appendix_defence.tex:60`, `:83`). At c = 33,000, 128(1+128c) is 540,672,128, which exceeds 2^29. The same slip is in `th_a_check.md`. | Now 0 <= c < 2^15, 128(1+128c) < 2^29, and sum &#124;w&#124; <= 1+128c. At c = 2^15-1 this gives 2^29 - 16,256 < 2^29. |
| F3 | TH-c Block 2 (`ext_defence.tex`) said "Now lower the error per matrix from 2^-beta_0 to 2^-128". Since 2^-128 > 2^-200, this raises the error. | "Now raise the error per matrix from 2^-beta_0 to 2^-128, the most that any test of error 2^-128 can allow." |
| F4 | TH-c Block 5(b) (`ext_intro.tex`) said a smaller setup proof "needs a smaller error budget". That reads backwards: fewer columns need a looser error. | "needs a larger error per matrix". |
| F5 | Related Work (`intro.tex`): the inserted `thm:guess` sentence left "Its tight case is the search for a hidden object" referring to "the trace". | "The calculation's tight case ...". This is body text; the body length is unchanged at 12.16 pages. |
| F6 | `ext_evaluation.tex` printed internal process text in the paper: "(cut P3 of the H5 pass)". | "Table 5 omits two LeNet-5 rows for space." No other paper text names an H5 or REVIEW item (checked by grep). |
| F7 | `ext_discussion.tex` ("The attack and the theorem") stated a falsehood: "an in-range attack on an LLM is pending". It also attributed test-split ranges to all range-respecting attacks. | It now names MNIST for the test-split ranges, states the held-out OPT-1.3B result (Section 3.5), and keeps `\todo{G3.5-L: OPT-6.7B, Qwen3-4B}`. |
| F8 | `tab:eval-reexec` (`ext_evaluation.tex`) kept two stale `\todo{B5}` cells. | Filled from B5 §6 (job 1006792): 1.6 and 17.2 s "with the native kernels". The caption names that later run (1-3 blocks) and its re-execution times (3.5 / 78.8 s fp32, 3.6 / 98.3 s integer; Table 4). |
| F9 | `openvm2025logup` author field read "OpenVM Contributors", but the PDF byline is "OpenVM Authors". | Corrected in `extra.bib`. |

Each edited file's header now has an "H5 review" note. Section 2's TH-a line "128(1+127c) < 2^29" is superseded by F2.

### 6.4 Still open

- **Source for 1.6% / 0.4%.** The body's figures and Block 2 have no `docs/improvements` source yet. My recomputation above agrees with TH-c's. Adding the breakdown to D4_D5 §5 would close this (not done, since `docs/` is out of scope).
- **Todo names.** Several submission todos still say "L40S" instead of "F6": the `tab:eval-llm` and `tab:eval-v1` cells, `evaluation.tex:48`, `:64`, `:74`, `:228`, `:260-261`, and `intro.tex:129`, which Section 4 lists as F6. `attack.tex` "G3: adaptive game" means G3-min, and `defence.tex` "D6" means D6a. I left them, because renaming table cells changes column widths.
- **Section numbering.** `sec:setup-proof` and `sec:fs` both print "Section 4.4", because the demoted paragraph sits in "Non-Interactive Proofs". This is correct, but the introduction now cites 4.4 for two different extensions.
- **LeNet-5 placement.** The paragraph sits in the extended subsection "V1 Against Published Systems", although it is a v0 comparison.
- **Overfull tables.** The two pre-existing overfull single-column tables in `ext_attack.tex` (15 pt and 14 pt: the formula table and `tab:inrange`) are untouched.
- **STIR and WHIR volumes.** The LNCS volume numbers rest on search results. WHIR's pages are confirmed by Crossref, and STIR's by search results only.
- **Page budget.** It is unchanged by the review: body 12.16, references + appendices about 4.99 of 5. Any appendix growth still needs an offset.
