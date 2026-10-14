# Comments of the paper's LaTeX source (submission branch, 14 Oct 2026)

Every whole-line comment of `paper_sp2027/` before they were removed from the submission branch, by file and line. The author-facing notes (disclosure schedule, AI-use field, open decisions) are among them.

## main.tex

```
    1  % =====================================================================================
    2  % main.tex: IEEE S&P 2027 submission draft, "Certified but Compromised" (working title).
    3  % Build: tectonic main.tex (the submission: body + short appendix, 18 pages) and tectonic main_ext.tex (the
    4  % extended version for the anonymous artifact repository: the same body, the full appendices and the material
    5  % moved out of the body).
    6  % Sections live in sections/*.tex; each one \providecommand's the macros it needs, so the
    7  % definitions below take precedence. Bibliography: references.bib (copied unchanged from
    8  % report/references.bib) and extra.bib (entries added for this draft; % [U] marks unverified ones).
    9  % H5 pass (2026-10-11, REVIEW2.md N13): the abstract no longer calls the 512-token real-weight figure
   10  % "long-prompt" (2.0-2.3x at 2,048 tokens with random weights, F3a todo) and labels the 4.6x as
   11  % interactive and extrapolated (RX todo). See paper_sp2027/H5_PASS.md.
   12  % =====================================================================================
   28  % ---- visible marker for open items -------------------------------------------------
   31  % ---- the run's numbers (code/experiments/paper/numbers.py) ------------------------------
   32  % Each table body and the text numbers (numbers.tex) are read from generated/<name>.tex when it exists, else from
   33  % fallback/<name>.tex, which holds the values printed before the final run.  \pvitable reads a file with TeX's
   34  % primitive \input, which is expandable: a table body may start with \multicolumn, and the paper's \bottomrule may
   35  % follow it.
   48  % ---- shared notation (one name per concept) ----------------------------------------
   56  % V1 (sections/v1.tex, sections/appendix_v1.tex)
   67  % ---- theorem-like environments ------------------------------------------------------
```

## main_ext.tex

```
    1  % The extended version: main.tex with the full appendices (see main.tex).
```

## sections/app_params.tex

```
    1  % sections/app_params.tex: the concrete-security table, \input by appendix_short.tex (submission) and by
    2  % main.tex after appendix_defence.tex (extended version). Labels: app:params, tab:params.
   31  % -------------------------------------------------------------------------------------
```

## sections/appendix_attack.tex

```
    1  % =====================================================================================
    2  % sections/appendix_attack.tex: full statements and proofs for Section sec:attack
    3  % (input after \appendix; uses the environments and macros set up in sections/attack.tex).
    4  %
    5  % Sources used (paths relative to session 4aa2c028's scratchpad):
    6  %   sp2027_research/attack_theory/PART1_SP.md Sec. 1.2 (appendix LaTeX, with every fix of
    7  %     theorem_check.md), Sec. 3.4 (instruction-tuned models; shapes from experiments.md)
    8  %   attack_theory/theorem.md, theorem_check.md, check_guessing_out.json (12 exact toy instances),
    9  %     check_merkle_leak_out.json (N=16: 0.0625 ideal; 0.125, 0.25, 0.5 for L=0,1,2)
   10  %   wt_sp2027/report/main.tex Sec. 3.3-3.5 (Freivalds and column checks, parameters)
   11  %   wt_sp2027/docs/improvements/D4_D5_untrusted_commitment.md (setup proof: k-1 becomes k-1+e)
   12  % Table tab:aligned recomputed with Python on 2026-10-10 as ceil(40 ln 2 / -ln(1-k/d_ff)); it
   13  % matches PART1_SP.md Sec. 3.4.
   14  % Labels defined here: app:attack, def:H, thm:guess-full, lem:notouch, lem:product, cor:surprise,
   15  % prop:worst, prop:blind, prop:amp, rem:merkle, rem:pointer, rem:scope, tab:aligned, and (H5 pass,
   16  % TH-a) prop:block, lem:worst-int, cor:block, rem:block-count, rem:block-scope, eq:block-freivalds,
   17  % eq:block-columns. H5 pass (2026-10-11): prop:amp states the product form of the column term and
   18  % caps t at k (drafts/th_a_check.md, "The same gap exists in prop:amp and thm:dich(b)").
   19  % H5 review (2026-10-11): lem:worst-int counts int8 weights down to -128, so the hidden-row bound is
   20  %   128(1+128c) < 2^29, i.e. c < 2^15 (the draft's 128(1+127c) and c <= 33,000 assumed |w| <= 127).
   21  % Labels used from other sections: sec:defence, sec:setup-proof.
   22  % =====================================================================================
  247  % -------------------------------------------------------------------------------------
  248  % TH-a (drafts/th_a_block_mixing.tex, Part C), placed by the H5 pass (2026-10-11). Every fix of the
  249  % adversarial check drafts/th_a_check.md is applied: the admissible set limited to integer networks
  250  % within the range bound (Prop. (i)); folded rows do not depend on salts (Prop. (i)); "the bound of
  251  % prop:amp" (Prop. (ii)); the cap t <= k (Prop. (iii), and in prop:amp above); the claim range check
  252  % and the exclusion of lookups (Cor. (a)); the integer hard family as Lemma lem:worst-int, the
  253  % ideal-oracle model and the counting-convention remark (Cor. (b)); the second byte condition, r = 7
  254  % under Fiat--Shamir and the salts (Rem. rem:block-count (iii)); redundant column queries in the
  255  % ideal-oracle model (Rem. rem:block-count (iv)); beta_0 and e_b (Rem. rem:block-scope (ii)); the
  256  % probability-1 attack on an unchecked block and the corrected last sentence (Rem. rem:block-scope
  257  % (iv)). Counting convention: one query per folded row and one per opened column, as in prop:amp,
  258  % so q = ceil(N/d)(r+t). The submission appendix does not carry this material (no appendix text of
  259  % the same size could make room); the body cites the extended version in one sentence (Sec. 3.4).
```

## sections/appendix_defence.tex

```
    1  % =====================================================================================
    2  % sections/appendix_defence.tex: integer semantics and proofs for the defence (sections/defence.tex)
    3  % Goes after \appendices (IEEEtran). Defines app:defence, app:semantics, app:sound, app:untrusted,
    4  % and, since the H5 pass (2026-10-11, cut P1 of REVIEW2.md), eq:split, fact:ext and thm:untrusted
    5  % (moved from defence.tex) and rem:forced (TH-c, drafts/th_c_setup_forced.tex, Block 1). Cites
    6  % stir, whir, fri2018, basefold, ligero, brakedown, bcs16, bciks20 (Thm. 1.6 of the full version).
    7  %
    8  % Sources used (paths relative to session 4aa2c028's scratchpad):
    9  %   code/src/pvi/fullcheck/transformer.py, graph.py   every formula of the integer semantics (requant,
   10  %                                        residual_add, _norm_int, _isqrt, _lut, _rope, _exp_lut,
   11  %                                        _attention_core and its exactness bounds), MatOp.max_input
   12  %   wt_sp2027/docs/improvements/A1_int8_column_opening.md, A2_int8_forward.md   int8 GEMM exactness
   13  %                                        and the int8_ok self-check with automatic fallback
   14  %   wt_sp2027/report/main.tex            course paper: proof of Theorem 3.2, transposed layout, Kpre reuse
   15  %   wt_sp2027/docs/improvements/D4_D5_untrusted_commitment.md, code/.../setup_proof.py   setup
   16  %                                        parameters (beta_0, s, radii as fractions 2^0..2^-10 of the
   17  %                                        unique-decoding radius, t0, per-query t), soundness argument
   18  %   sp2027_research/idea_trust-rigor.md  Secs. 1.3-1.4: Fact 1 and its proof, extraction, the online
   19  %                                        argument with the corrupted set D_j, field semantics of M*
   20  %   sp2027_research/literature.md        BCIKS20 (unique-decoding error n/q)
   21  % Our own additions, checked only by us: the hybrid argument for Kpre with per-check verdicts, the
   22  %   round-by-round states, and the condition e_j < n_j/2 - k_j + 1 under which the split commitment
   23  %   is not e_j-close.
   24  % =====================================================================================
  151  % H5 pass (2026-10-11): the last sentence of "Round-by-round soundness" above follows
  152  % drafts/th_d_citations.md, Sec. 5.2(b) (N6, B-Q6). The former subsections "Proof of Fact fact:ext"
  153  % and "Proof of Theorem thm:untrusted" are merged into the subsection below, which now also holds
  154  % the material moved out of the body by cut P1 (REVIEW2.md, Sec. 9).
  162  % Moved from the body (defence.tex, subsection "Untrusted Commitments", paragraph "Why the column
  163  % check is not enough") by the H5 pass, cut P1 of REVIEW2.md; unchanged.
  181  % Moved from the body (paragraph "One combination over an extension field" and Fact fact:ext);
  182  % unchanged. The proof of the fact was a subsection of its own here.
  211  % Moved from the body (paragraph "The setup proof", Theorem thm:untrusted and the remark after its
  212  % proof sketch); unchanged.
  262  % Setup paragraph: first four sentences replaced per drafts/th_d_citations.md, Sec. 1.2(c) (N5:
  263  % BCIKS20 Thm. 1.6 of the full version; Ligero Lemma 4.2 for e_j < d/4); the rest unchanged, plus
  264  % the sentence of Sec. 5.2(c) at the end.
  317  % TH-c, Block 1 (drafts/th_c_setup_forced.tex), placed by the H5 pass at the end of this appendix as
  318  % its header asks. Remark rem:forced, its proof, the Merkle/Fiat--Shamir paragraph and the scope.
```

## sections/appendix_short.tex

```
    1  % =====================================================================================
    2  % sections/appendix_short.tex: the submission appendix (CUT_PLAN.md Sec. 4; submission only).
    3  % Sources: appendix_attack.tex, appendix_defence.tex, appendix_v1.tex (compressed; the extended
    4  %   version keeps them in full). The concrete-security table (app:params, tab:params) is
    5  %   \input from sections/app_params.tex.
    6  % Labels defined: app:attack, def:H, thm:guess-full, lem:product, cor:surprise, prop:worst,
    7  %   prop:blind, prop:amp, rem:merkle, rem:pointer, rem:scope, app:defence, app:semantics,
    8  %   app:sound, app:untrusted, fact:ext, thm:untrusted, app:v1, lem:v1-window, lem:v1-nowrap,
    9  %   lem:v1-gkr, lem:v1-final. Not defined: tab:aligned (the body no longer cites it).
   10  % H5 pass (2026-10-11, REVIEW2.md cut P1): Fact fact:ext, the four steps of the setup proof and the
   11  %   statement of Theorem thm:untrusted moved here from the body (defence.tex) in place of the proof
   12  %   idea of app:untrusted, which the extended version keeps in full (appendix_defence.tex). N5:
   13  %   BCIKS20 cited as Thm. 1.6 of the full version (drafts/th_d_citations.md, Sec. 1.2(d)).
   14  %   TH-a (block mixing) is in the extended version only (appendix_attack.tex), since no appendix
   15  %   text of the same size could make room for it here.
   16  % Trim of 2026-10-10: proofs that repeated the body's proof sketches now refer to them; the
   17  %   Ligero radius remark (Lemma 4.2), the caching/path detail of rem:scope (stated in attack.tex)
   18  %   and the split-commitment remark stay in the full appendices only.
   19  % Macros and theorem environments: main.tex.
   20  % =====================================================================================
  152  % -------------------------------------------------------------------------------------
```

## sections/appendix_v1.tex

```
    1  % =====================================================================================
    2  % sections/appendix_v1.tex: full proof of Theorem thm:v1 (V1 soundness), IEEEtran compsoc
    3  %
    4  % Sources used (paths relative to session 4aa2c028's scratchpad):
    5  %   sp2027_research/v1_design/V1_SPEC.md  Sec. 2 (notation, windows, cut rules, leaves, table,
    6  %     instance layout, the fractional-sum GKR of Sec. 2.4.1, the checks of Sec. 2.5) and Sec. 3
    7  %     (Theorem 1, Lemma 0, Lemma 1, proof steps 1-7, the remarks)
    8  %   wt_sp2027/docs/improvements/C2_v1_proof_size.md  Secs. 3-4 and 9 (checks as implemented,
    9  %     the soundness steps, the review's fixes: P1 clamp within int8, P4 in Kpre, fold aliases,
   10  %     the input range check, the exception bin outside P5; P4 in mode C)
   11  %   sections/defence.tex  (Theorem thm:sound, its notation J, k_j, n_j, t_j and the product form
   12  %     of the column term; the Kpre reuse bound; the Fiat-Shamir argument of sec:fs)
   13  % Labels referenced: thm:v1, lem:v1-logup, sec:v1 (sections/v1.tex); thm:sound, sec:fs,
   14  % sec:setup-proof (sections/defence.tex).
   15  % Labels defined here: app:v1, lem:v1-window, lem:v1-nowrap, lem:v1-gkr, lem:v1-final.
   16  % Uses the macros of sections/v1.tex (re-provided below so that this file also compiles alone).
   17  % =====================================================================================
```

## sections/attack.tex

```
    1  % =====================================================================================
    2  % sections/attack.tex: Section 3, the attack on spot-checking proofs of inference.
    3  % Condensed on 2026-10-10 per CUT_PLAN.md; removed material is in sections/ext_attack.tex.
    4  % Second trimming pass 2026-10-10 (2,365 -> 2,233 words): wording tightened; the PCPP/holographic
    5  %   contrast (kept in intro.tex), the hidden-object citation and the posterior formula (kept in the
    6  %   Related Work and in ext_attack.tex) left the body; see the last subsection of ext_attack.tex.
    7  % Sources (session 4aa2c028 scratchpad): wt_sp2027/report/main.tex Sec. 3.1-3.3, 4.1-4.2 (all
    8  %   measured numbers); sp2027_research/attack_theory/PART1_SP.md Sec. 1-4, theorem.md,
    9  %   theorem_check.md, sok.md, sok_formulas_out.json, disclosure.md, experiments.md.
   10  % Labels defined: sec:attack, sec:threat, sec:path, sec:sampling, def:spot (new), thm:guess,
   11  %   sec:dichotomy, thm:dich, sec:break, sec:sok, tab:sok.
   12  % Labels used: app:attack, def:H, cor:surprise, prop:worst, prop:blind, prop:amp, rem:merkle,
   13  %   rem:pointer, rem:scope (appendix); sec:related, sec:defence, sec:setup-proof. New key: fullversion.
   14  % H5 pass (2026-10-11, REVIEW2.md): N1 (thm:guess with the fractional G_t of the appendix, "for
   15  %   integer q" in the attainment sentence); C3 (folklore sentence of Sec. 3.4 removed); TH-a (one
   16  %   body sentence in Sec. 3.4 citing the extended version, where Part C is in appendix_attack.tex;
   17  %   thm:dich(b) gains n >= 2(k-1)); C4 (MNIST: the 250-path figure, the 95th-percentile sentence
   18  %   and its footnote moved to ext_attack.tex); N2 (in-range detection: median per cell at most
   19  %   7.9e-4, maximum 7.3e-3; code/artifacts/results/inrange_llm_opt-1.3b_any.json); P2 (the
   20  %   Proof-of-Learning row of tab:sok moved to ext_attack.tex); TH-d Sec. 3.2(b) (tab:sok footnote on
   21  %   the version of Anchuri et al. and the refereed variant).
   22  % H5 review (2026-10-11): the G_qd display of thm:guess stacks its constraint (\substack), which
   23  %   removes a 31pt overfull box in both builds.
   24  % =====================================================================================
   32  % -------------------------------------------------------------------------------------
   57  % -------------------------------------------------------------------------------------
   76  % -------------------------------------------------------------------------------------
  137  % -------------------------------------------------------------------------------------
  180  % -------------------------------------------------------------------------------------
  233  % -------------------------------------------------------------------------------------
```

## sections/defence.tex

```
    1  % =====================================================================================
    2  % sections/defence.tex: the defence (Section 4), condensed for the 13-page body (CUT_PLAN.md).
    3  % Sources: the previous draft of this file (course paper Secs. 3.1, 3.4-3.6; plan items A1, A2,
    4  %   D1-D7; code/src/pvi/). Removed detail, Table tab:setup and Fig. fig:protocol:
    5  %   sections/ext_defence.tex. Second trimming pass (2026-10-10): sentences already covered in
    6  %   ext_defence.tex (int8 sum bounds, stacking, re-encoding queries, FRI/BaseFold, weight leak) cut.
    7  % H5 pass (2026-10-11, REVIEW2.md cut P1): the subsection "Untrusted Commitments" is demoted to the
    8  %   paragraph "Untrusted commitments" at the end of Sec. 4.4, which keeps the label sec:setup-proof.
    9  %   Fact fact:ext and Theorem thm:untrusted (statements) moved to appendix_short.tex (submission)
   10  %   and appendix_defence.tex (extended version); the split-commitment display eq:split, the
   11  %   extension-field paragraph, the four steps of the setup proof and the proof sketch moved to
   12  %   appendix_defence.tex. TH-c's remark (drafts/th_c_setup_forced.tex, Block 3, variant A) is the
   13  %   "forced" sentence of that paragraph. Cut C6: "A sponge transcript" folded into the binding
   14  %   paragraph (its Merkle--Damgard remark moved to ext_defence.tex). N6: the round-by-round error is
   15  %   a maximum; the parameters keep the union bound (TH-d, Sec. 5.2(a), condensed).
   16  % Labels defined: sec:defence, sec:intmodel, sec:commit, sec:protocol, sec:fs, sec:setup-proof,
   17  %   eq:weightop, eq:freivalds, thm:sound.
   18  % Labels used: thm:dich, thm:untrusted, sec:threat, sec:quality, sec:gen, sec:v1, app:defence,
   19  %   app:semantics, app:sound, app:untrusted. Cites fullversion, ligero, brakedown, fri2018, stir,
   20  %   whir, basefold.
   21  % =====================================================================================
   56  % -------------------------------------------------------------------------------------
   84  % -------------------------------------------------------------------------------------
   99  % -------------------------------------------------------------------------------------
  178  % -------------------------------------------------------------------------------------
```

## sections/discussion.tex

```
    1  % =====================================================================================
    2  % sections/discussion.tex: Discussion and Limitations, and Conclusion (condensed 2026-10-10 per
    3  % CUT_PLAN.md Sec. 4; second trim the same day). Every number is taken from attack.tex,
    4  % defence.tex, v1.tex, evaluation.tex or quality.tex. Cites only hollowllm.
    5  % Removed detail: sections/ext_discussion.tex. Facts dropped here because evaluation.tex
    6  % (Secs. sec:eval-v1, sec:eval-reexec, sec:eval-compare), intro.tex, defence.tex or attack.tex
    7  % state them: provider overheads of a GPU TEE and CommitLLM, zkAgent's time per token, V1's
    8  % prover slowdown, the real-weight V1 ratios, the 76% attention share (todo B5 stays in
    9  % evaluation.tex), batching of opened columns, the Hollow-LLM details, the remark on
   10  % embedding the worst case. All of these are already in ext_discussion.tex.
   11  % Labels defined: sec:discussion, sec:conclusion. Labels used: tab:options, sec:eval-reexec
   12  % (evaluation.tex).
   13  % H5 pass (2026-10-11, REVIEW2.md N14): the cross-GPU prover comparison is labelled as such
   14  %   (different GPUs, no overhead ratio); A7 todo. Todos named by item (F6, F3a, F2a/F2b, G3.5-L).
   15  % =====================================================================================
```

## sections/evaluation.tex

```
    1  % =====================================================================================
    2  % sections/evaluation.tex: Evaluation. Every measured number is a macro or a table body that
    3  % code/experiments/paper/numbers.py writes from one run of the run package (fallback/ until then);
    4  % the material of the extended version is in sections/ext_evaluation.tex.
    5  % Labels defined: sec:results, sec:eval-setup, sec:eval-cost, sec:eval-v1, sec:eval-reexec,
    6  %   sec:eval-compare, tab:eval-llm, tab:eval-v1, tab:options, fig:breakeven, tab:eval-compare.
    7  % Labels used: sec:break, sec:defence, sec:v1, sec:quality (quality.tex), tab:sok; cite fullversion.
    8  % =====================================================================================
   24  % -------------------------------------------------------------------------------------
   56  % -------------------------------------------------------------------------------------
   98  % -------------------------------------------------------------------------------------
  138  % -------------------------------------------------------------------------------------
  210  % -------------------------------------------------------------------------------------
```

## sections/ext_attack.tex

```
    1  % =====================================================================================
    2  % sections/ext_attack.tex: material cut from sections/attack.tex (extended version only).
    3  % Appendix-ready: \subsection headings only; place it after appendix_attack.tex, which stays
    4  % unchanged and holds the full statements, proofs, remarks and Table tab:aligned.
    5  % Where each block came from (the draft of sections/attack.tex of 2026-10-10, before the cut):
    6  %   "The Verifier Model: Remarks"  <- Sec. 3.3, paragraphs "Model", "Honest recomputation hides
    7  %       the edit", the discussion after Theorem thm:guess and "What reading the values buys";
    8  %       Sec. 3.4, paragraph "What the dichotomy means" (sentences dropped from the body). The
    9  %       remark on SNARK-assisted verifiers expands the body's one-line exclusion (decision 6).
   10  %   "Measured Attacks: Details"  <- Sec. 3.2 (how the edit is found), Sec. 3.5 paragraphs
   11  %       "Reproduction", "MNIST", "Other samplers", "Larger models", "A real LLM" (every number
   12  %       dropped from the body, unchanged).
   13  %   "Systematisation: Full Entries and Derivations"  <- Table tab:sok cells and footnotes
   14  %       shortened in the body, the paragraph "What the table shows" at full length, and the
   15  %       derivation of every acceptance number. The formulas were recomputed with Python on
   16  %       2026-10-10 and match the draft: (1-1/N)^s; s_40 = ceil(40 ln 2 / -ln(1-1/N));
   17  %       SLP 1-3/161, ceil(ln 0.5 / ln(158/161)) = 37, ceil(ln 0.05 / ln(158/161)) = 160,
   18  %       1-5/45; CommitLLM 1-10/80 and 1-0.05*10/80 = 99.375%.
   19  %   "Disclosure"  <- Sec. 3.6, paragraph "Ethics and responsible disclosure" (sentences dropped).
   20  % Full text of the todos shortened in the body (for the authors; not typeset):
   21  %   G1e: tamper locator: top-q hit rate of a locator that ranks sites from the forged trace alone,
   22  %     i.e. the empirical G_q of our greedy attack and of a randomised one, MNIST MLP, q in 1..64.
   23  %   G3 (footnote): repeat with ranges calibrated on the training split; report separately the
   24  %     plans whose edited values, and whose every claimed value, are in range.
   25  %   G3 (samplers): adaptive game: per-path detection of the best value-aware sampler and of the
   26  %     LP-optimal sampler mixture against range-respecting edits with held-out ranges, MNIST MLP
   27  %     and CNN heads, with the paths needed for 2^-40.
   28  %   L40S / G3.5: next-token attack on OPT-6.7B with values kept within ranges calibrated on
   29  %     WikiText-2 training text, if run.
   30  %   Disclosure: notices scheduled for 15 Oct 2026 (Anchuri et al., TensorCommitments) and
   31  %     19 Oct 2026 (CommitLLM, NanoZK, SLP, SPEX); remove SPEX from the body and Table tab:sok if
   32  %     its optional notice is not sent. Replies: if none, "We had received no reply by 2 Nov 2026."
   33  % Labels defined: ext:attack-model, ext:attack-measure, ext:attack-sok, tab:sok-derivation,
   34  %   ext:attack-disclosure, ext:attack-bound (the body does not reference them).
   35  % Labels used: def:spot, thm:guess, thm:dich, sec:break, tab:sok (attack.tex); lem:product,
   36  %   cor:surprise, prop:worst, rem:merkle, tab:aligned (appendix_attack.tex); sec:defence,
   37  %   sec:related. Citations: anchuri2026verifiable, chew1967, kadane1971, fang2023pol.
   38  % H5 pass (2026-10-11): cut C4 (paragraph "MNIST, further results"), cut P2 (the Proof-of-Learning
   39  %   entry) and TH-d Sec. 3.2(c) (the refereed variant of Anchuri et al.; key anchuri2026full) in
   40  %   "Systematisation"; TH-a Part B5 (with the fixes of th_a_check.md) in "The two regimes".
   41  % =====================================================================================
   46  % -------------------------------------------------------------------------------------
  108  % -------------------------------------------------------------------------------------
  126  % Moved from the body's paragraph "MNIST" by the H5 pass (cut C4 of REVIEW2.md); unchanged.
  156  % -------------------------------------------------------------------------------------
  232  % -------------------------------------------------------------------------------------
  242  % -------------------------------------------------------------------------------------
  243  % Appended by the second trimming pass of sections/attack.tex (2026-10-10). Sources: Sec. 3.3,
  244  %   the sentence after Definition def:spot on the reference oracle (its last clause) and the
  245  %   paragraph after the proof of Theorem thm:guess (the posterior formula, shortened in the body);
  246  %   Sec. 3.4, paragraph "What the bounds mean" (the contrast with PCPs of proximity and holographic
  247  %   proofs, which the body now states only in the Introduction and Related Work). Citations used:
  248  %   bghsv2006robust, bfls1991 (both cited in intro.tex).
  268  % Added 2026-10-10 (plan G3.5; experiments/2_attack/inrange_llm.py, artifacts/results/inrange_llm_opt-1.3b_*.json).
```

## sections/ext_defence.tex

```
    1  % =====================================================================================
    2  % sections/ext_defence.tex: material cut from sections/defence.tex (extended version only).
    3  % Appendix-ready: \subsection headings only; place it after appendix_defence.tex.
    4  % Where each block came from (the draft of sections/defence.tex of 2026-10-10, before the cut):
    5  %   "The Integer Model: Execution Details"  <- Sec. 4.1, paragraphs "Weight operations",
    6  %       "Operations without weights" and "Exact execution" (im2col and BatchNorm folding, the
    7  %       requantisation formula, the last-position LM head, int8 inputs, hardware errors).
    8  %   "Commitment Plans"  <- Sec. 4.2, paragraph "Commitment plans" and the domain separation of
    9  %       the Merkle trees (six candidate plans, LeNet-5 example, stacking of q/k/v projections).
   10  %   "The Fiat--Shamir Transcript"  <- Sec. 4.4, paragraph "A sponge transcript" (32-byte keys,
   11  %       rejection sampling) and the requantisation-multiplier example.
   12  %   "Untrusted Commitments: Measurements, Trade-off and Scope"  <- Sec. 4.5, paragraphs "The
   13  %       split-commitment attack" (300-query measurement), "A one-time setup proof" (transposed
   14  %       matrices), "The trade-off" with Table tab:setup, and "Scope".
   15  % Second trimming pass of defence.tex (2026-10-10): the body sentences cut there are already covered
   16  %   here, so nothing was appended: int8 sum bounds (2^14, blocks of 2^16) -> "Execution Details";
   17  %   stacking in the transposed layout -> "Commitment Plans"; rehashed/re-encoded claims covered by
   18  %   the factor Q -> "The Fiat--Shamir Transcript"; FRI/BaseFold and the weight leak -> "Scope".
   19  % Labels defined: tab:setup (moved here from defence.tex), fig:protocol. Labels used: sec:intmodel, sec:fs,
   20  %   (defence.tex); eq:split, thm:untrusted, rem:forced, app:semantics, app:sound, app:untrusted
   21  %   (appendix_defence.tex). New BibTeX key: fri2018 (also cited in defence.tex).
   22  % H5 pass (2026-10-11): TH-c Block 2 ("How much a different test could save") inserted after
   23  %   Table tab:setup; TH-c Block 5(c) replaces the last sentence of "Scope"; the Merkle--Damgard
   24  %   remark of the body's former paragraph "A sponge transcript" (cut C6) added to "The Fiat--Shamir
   25  %   Transcript".
   26  % H5 review (2026-10-11): Block 2's "lower the error per matrix from 2^-beta_0 to 2^-128" said the
   27  %   opposite of what it does (2^-128 > 2^-beta_0); now "raise". All Block 2 numbers re-derived with
   28  %   setup_bytes.py's formulas (code read-only) and confirmed.
   29  % =====================================================================================
   70  % Moved from the body's paragraphs "Commitment plans" and "(S3): the column check" by the H5 pass
   71  % (cut C5 of REVIEW2.md); unchanged.
  147  % The breakdown and floors below are those of the byte model of code/experiments/costs/setup_bytes.py
  148  % (setup_model) under the run's plan, which code/experiments/paper/numbers.py writes as macros.
  177  % Moved from defence.tex (Sec. 4.3) for space.
```

## sections/ext_discussion.tex

```
    1  % =====================================================================================
    2  % sections/ext_discussion.tex: material cut from the Discussion and Limitations on 2026-10-10
    3  % (CUT_PLAN.md, Sec. 4). Extended version only; appendix-ready (\subsection, no \section).
    4  % Where each block came from (all from the pre-cut sections/discussion.tex unless noted):
    5  %   "Status of the Measurements"       paragraph "Measurements", unchanged in substance
    6  %   "Real Weights, Model Quality and V1" paragraphs "Real weights and model quality" and "V1";
    7  %                                      "halves the proof" reworded per CUT_PLAN.md decision 4
    8  %   "A Client's Options in Detail"     the long form of the body's "Options for a client"
    9  %                                      paragraph and Table tab:options; numbers from
   10  %                                      evaluation.tex (Secs. sec:eval-reexec, sec:eval-compare),
   11  %                                      attack.tex (Table tab:sok), defence.tex (Sec.
   12  %                                      sec:setup-proof), intro.tex and ext_intro.tex (gpucc,
   13  %                                      CommitLLM). Derived by arithmetic only: ceil(11,008/2,048)
   14  %                                      = 6 queries for the down projection.
   15  %   "Limitations in Detail"            paragraphs "Proof size and the verifier", "Weights and
   16  %                                      trust" and "The attack and the theorem" at full length
   17  % The old Conclusion only summarised the body and is not kept here.
   18  % Labels defined: ext:disc-status, ext:disc-real, ext:disc-options, ext:disc-limits (the body
   19  %   does not reference them). Labels used (defined in the body or the full appendices):
   20  %   tab:eval-llm, tab:eval-v1, tab:eval-reexec, tab:eval-compare, tab:options, tab:sok,
   21  %   sec:eval-reexec, sec:setup-proof, sec:v1, sec:gen-sampling, sec:quality, rem:scope.
   22  % H5 review (2026-10-11): "The attack and the theorem" no longer calls the in-range LLM attack
   23  %   pending (OPT-1.3B is in the body, G3_inrange_attack.md); larger models keep \todo{G3.5-L}.
   24  %   Also uses sec:break.
   25  % =====================================================================================
```

## sections/ext_evaluation.tex

```
    1  % =====================================================================================
    2  % sections/ext_evaluation.tex: material cut from sections/evaluation.tex (extended version only).
    3  % \input it inside an appendix \section of the extended version; it has \subsection headings only.
    4  % Where each block came from (the draft evaluation.tex of 2026-10-10; every measured number is now a macro or a
    5  % table body of code/experiments/paper/numbers.py):
    6  %   1. Setup details              Sec. 7.1 "Hardware" and "Models" (sentences dropped from the body)
    7  %   2. Prover and verifier        Sec. 7.3 "Prover improvements", "Verifier improvements" and
    8  %      improvements               Table tab:eval-improve (the body keeps a two-sentence summary)
    9  %   3. Memory and untrusted       Sec. 7.3 "Memory and untrusted commitments" (duplicates the key
   10  %      commitments                numbers of Table tab:setup, which moves to ext_defence.tex)
   11  %   4. Attack and detection       Sec. 7.2 (the attack numbers stay only in Sec. 3.5 of the body;
   12  %                                 kept here: the comparison with our proof and the setup-proof outcome)
   13  %   5. V1: crossover per cell     Sec. 7.4 "Cost" (last sentence) and the per-cell values behind the
   14  %                                 body's 0.16-0.26 Gb/s (arithmetic on Table tab:eval-v1)
   15  %   6. Re-execution remarks       Sec. 7.5 "Where this leaves the verifier" (sentences dropped)
   16  %   7. V1 against published       the V1 column and footnote e of Table tab:eval-compare, and Sec. 7.6
   17  %      systems                    "What this paper's changes add"
   18  % Improvement IDs (plan items) behind the names used in block 2: A1 int8 opening and fold; A2 int8
   19  %   forward; A3 threaded encoder assembly; A4 fused attention in the prover; B2 GPU claim decoder;
   20  %   B3 fused exact attention kernel; B5 native CPU attention; C2 = V1; D4/D5 setup proof; F1
   21  %   re-execution. Rounding note: A1's Llama-2-13B gain is the source's 4.3x (rounded cells give 4.4x).
   22  % Labels defined here: app:eval-setup, app:eval-improve, tab:eval-improve, app:eval-untrusted,
   23  %   app:eval-attack, app:eval-crossover, app:eval-reexec, app:eval-compare-v1, tab:eval-compare-v1,
   24  %   app:eval-gkr-leaf (block 8, added by the second trimming pass: GKR time per leaf, from Sec. 7.3).
   25  % Labels used (body or other appendices): sec:break, sec:setup-proof, sec:quality, sec:gen,
   26  %   tab:eval-llm, tab:eval-v1, tab:eval-reexec, tab:eval-compare, tab:setup.
   27  % H5 pass (2026-10-11): cut P3, the two LeNet-5 rows of tab:eval-compare as a paragraph in block 7.
   28  % H5 review (2026-10-11): tab:eval-reexec's "native" row filled from B5_cpu_verifier.md Sec. 6 (job
   29  %   1006792: 1.6 / 17.2 s; re-execution 3.5 / 78.8 s fp32, 3.6 / 98.3 s integer); the LeNet-5
   30  %   paragraph no longer names the internal cut.
   31  % =====================================================================================
   39  % -------------------------------------------------------------------------------------
   48  % -------------------------------------------------------------------------------------
   95  % -------------------------------------------------------------------------------------
  105  % -------------------------------------------------------------------------------------
  147  % -------------------------------------------------------------------------------------
  173  % -------------------------------------------------------------------------------------
  183  % Removed from the body under the CFP's AI-use policy (CUT_PLAN.md decision 7); restore only with
  184  % an accurate description of how the review was done, or drop:
  185  % "An adversarial review of V1 found forgeries only on two crafted models that violate conditions
  186  % the verifier now checks."
  188  % -------------------------------------------------------------------------------------
  202  % -------------------------------------------------------------------------------------
  231  % -------------------------------------------------------------------------------------
  264  % -------------------------------------------------------------------------------------
  265  % Moved from the body, sections/evaluation.tex Sec. "V1 Against v0", paragraph "Time" (second
  266  % trimming pass, 2026-10-10); number unchanged.
```

## sections/ext_generation.tex

```
    1  % =====================================================================================
    2  % ext_generation.tex: extended version only (not in the submission). Material cut from
    3  % Section 6 (generation.tex) and from the quality subsection of the evaluation (quality.tex).
    4  % Where each block comes from:
    5  %  - "Generated Responses as One Prefill: Details and Full Measurements": generation_quality.tex
    6  %    Sec. 6.1 (per-prefix comparison, token-rule cost, Kpre online rule, "Tests and cost", the full
    7  %    Table tab:gen with G = 64 and MB/token); per-prefix position count and the graph change
    8  %    from docs/improvements/E1_generation.md.
    9  %  - "The Integer Sampler: Table, Truncation and Tests": generation_quality.tex Sec. 6.2
   10  %    (table check, "Distribution and tests", stop rules); Gamma's end values and the truncation
   11  %    bound are computed from Eq. (eq:gumbel-table); nucleus remark from E2_sampling.md.
   12  %  - "Quality of the Integer Model: Attribution, Smoothing and Published Results":
   13  %    generation_quality.tex Sec. 6.3 (attribution method, the top half of Table tab:quality,
   14  %    LayerNorm analysis, smoothing details, per-row scales, CushionCache, SLP, the int8 remark);
   15  %    rows q/k/v weights and attention-output requantisation, cross-layer equalisation, the raw
   16  %    robustness range and the per-row top-1 agreement from docs/improvements/F2_quality.md.
   17  %  - Tables tab:gen and tab:gen-full: the gen tier's cells (code/experiments/paper/tables.py, GEN).
   18  % Labels defined here (the body must not \ref them): sec:ext-gen, tab:gen, tab:gen-full,
   19  % sec:ext-sampling, sec:ext-quality, tab:quality-attr.
   20  % Labels used (body): sec:gen, sec:gen-sampling, sec:quality, tab:quality,
   21  % eq:gumbel-table, sec:fs, sec:intmodel, thm:sound, sec:v1.
   22  % =====================================================================================
```

## sections/ext_intro.tex

```
    1  % =====================================================================================
    2  % sections/ext_intro.tex: material cut from the Introduction and Related Work on 2026-10-10
    3  % (CUT_PLAN.md, Sec. 4). Extended version only; appendix-ready (\subsection, no \section).
    4  % Where each block came from (all from the pre-cut sections/intro.tex):
    5  %   "Motivation and Costs in Detail"  Introduction, first paragraph (motivating example) and the
    6  %                                     paragraph "Costs" (every cost number of the old intro, with
    7  %                                     its todos; the numbers are those of evaluation.tex,
    8  %                                     generation_quality.tex and v1.tex)
    9  %   "Related Work in Detail"          Related Work, all five paragraphs at full length (published
   10  %                                     costs of zkSNARK provers, per-scheme sampling rules, the
   11  %                                     folklore behind the lower bound, and the long positioning),
   12  %                                     reworded per CUT_PLAN.md decision 4 ("dichotomy" dropped,
   13  %                                     attention "recomputed by the verifier from checked values")
   14  % Second trim of the body (2026-10-10, target 1,740 words): the Related Work lost Hollow-LLM, the
   15  %   edit-then-recompute credit to SLP/CommitLLM, Proof-of-Learning, the Ligero/Brakedown phase
   16  %   mapping and the cause of the GPU-CC overhead; all of these are already in "Related Work in
   17  %   Detail" below, so nothing was appended.
   18  % H5 pass (2026-10-11): the last sentence of "Freivalds-based checks and linear-code commitments"
   19  %   replaced per drafts/th_c_setup_forced.tex, Block 5(b) (N7). The introduction's restated verifier
   20  %   model and PCPP sentence (cut C1) and the details of the four extensions (cut C2) are covered by
   21  %   Definition def:spot, the Related Work and Sections 4-6 of the body, so nothing was appended here.
   22  % H5 review (2026-10-11): "needs a smaller error budget" -> "needs a larger error per matrix" (fewer
   23  %   setup columns need a looser per-matrix error, not a tighter one).
   24  % Labels defined: ext:intro-costs, ext:related (the body does not reference them).
   25  % Labels used (all defined in the body): thm:guess, thm:dich, tab:sok, sec:setup-proof, sec:gen,
   26  %   sec:v1, sec:quality, tab:eval-compare.
   27  % =====================================================================================
```

## sections/ext_proofs.tex

```
    1  % =====================================================================================
    2  % sections/ext_proofs.tex (extended version only): full proofs of the optimisations the submission relies on
    3  % (Plan 2 TH-e). From drafts/th_e_proofs.tex after its adversarial check (drafts/th_e_check.md: every
    4  % statement holds; the fixes are marked % CHECK:). Pending: the team's hand check (TH-f) and the decisions
    5  % listed in th_e_check.md, 'For the authors to decide'. Note: real_weights.build_opt_from_hf has prune_last
    6  % since F3a, so the pruning results apply to the real-weight graphs as well.
    7  % =====================================================================================
   30  % -------------------------------------------------------------------------------------
   88  % -------------------------------------------------------------------------------------
  138  % CHECK: "510" generalised to 2 P_max: transformer.py (modified 2026-10-11 01:12, during this check)
  139  % now has build_decoder(prob_max=...), PROB_MAX = 255 (510 = 2*255), e.g. 2047 for plan F2a. The
  140  % argument does not depend on the constant.
  167  % CHECK: "and the final norm" added: norm_f lies after block L and is computed at n positions in
  168  % the pruned graph (the proof of (a) shows it equals S_n(norm_f(r''))).
  221  % CHECK: was "the single-position operations ... violate (P2)". cut.py tests cols == T, so with
  222  % n = T (a one-token prompt, T = P + G - 1 = G) the pruned ops span all T positions and may be cut;
  223  % thm:v1 applies with whatever cut set results either way.
  231  % -------------------------------------------------------------------------------------
  307  % CHECK: was "at most Q_V eps_K (Theorem thm:v1kpre)", which omits the logUp and GKR terms that
  308  % thm:v1kpre adds for V1 queries (V1 also runs in Kpre with the sampler, run_cut_query).
  369  % -------------------------------------------------------------------------------------
  442  % CHECK: Q_V renamed Q_ver here: app:the-conv and cor:v1kpre-fs use Q_V for the number of verdicts
  443  % (proofs submitted), a different quantity.
  452  % CHECK: the components are now defined by the query's first 32 bytes, not by its role. As drafted,
  453  % a query K||label whose key K happens to start like a transcript is both an "expansion input" and
  454  % a "challenge input", and "at most one first unresolved component" (needed for Pr[E_2]) was not
  455  % justified. Every challenge input starts with sigma_0 (16-byte domain string, 2+5+8 bytes of
  456  % framing of the record "graph", first byte of the 32-byte digest: _absorb_statement absorbs
  457  % "graph" first in v0 and V1). E_0 costs N 2^-256; with N(N-1)/2 pairs the total stays N^2 2^-256.
  486  % CHECK: sentence above extended to use E_0 and E_1 explicitly (see the CHECK at the start of the proof).
  497  % CHECK: E_0 added to the last line.
  511  % CHECK: "1/p^8" -> "2/p^8". logup_gkr.verify draws mu{k-1} (theta) and lam{k} (eta) from the same
  512  % transcript state (nothing is absorbed between them), and thm:fstree's proof treats all challenges
  513  % of one transcript string as one round, so that round errs with <= 1/p^8 + 1/p^8. The maximum
  514  % (the alpha term, <= 2^-215.3) is unchanged.
  568  % -------------------------------------------------------------------------------------
  664  % -------------------------------------------------------------------------------------
  701  % CHECK: "(F3) or (F4)" added. Without them the set is not closed under the fold message: after a
  702  % doomed GKR (Step 5) the prover can send only true folds, and then (F3), not a fold mismatch,
  703  % rejects; that state must be doomed too. A failed column-independent check is left with prob. 0.
  735  % CHECK: was "therefore about 2^-127": that holds only for beta = lambda + 65. With the parameters
  736  % kept (beta = 200.82), (Q+k) 2^-beta = 2^-136.8 at Q = 2^64, so the total is 2^-127.997.
  737  % CHECK (suggestion, no text change): the same "first wrong operation is fixed before any
  738  % challenge" argument gives p^{-r}, not eps_K = sum_ell p^{-r}, for one query of mode K, and
  739  % (see the CHECK at thm:v1kpre) Q p^{-r} for Q verdicts of one Kpre key. The draft does not say so.
  742  % -------------------------------------------------------------------------------------
  787  % CHECK: a doubled backslash before "varepsilon" (a line break, then the letters "varepsilon" in
  788  % math italics, or an error inside \[...\]) replaced by "\varepsilon" here and in the proof below
  789  % (5 places).
  790  % CHECK (suggestion, no text change): the sum over ell is not needed. The verifier evaluates the
  791  % secret checks in a fixed order and reports only the first failure, so the real run and H can
  792  % first differ in query i only if the FIRST evaluated secret check with E_c != 0 passes; that check
  793  % is a function of H's view, so this has probability <= p^{-r}, and the theorem holds with
  794  % Q p^{-r} in place of Q sum_ell p^{-r} (prop:qtight then matches it to second order). Authors
  795  % decide; see th_e_check.md, (iv) and (v).
  872  % CHECK: was "(R2) bounds its success by ...": (R2) as cited (Theorem 31.2.1) concerns one final
  873  % proof; the extension to Q_V proofs is the move-counting argument, now stated. Check against the
  874  % book's proof that it is indeed a union bound over moves.
  877  % -------------------------------------------------------------------------------------
  917  % CHECK: numbers recomputed (log2 p = 30.9069): 226 p^-5 = 2^-146.714, Q_V <= 2^18.71 = 4.3e5;
  918  % 226 p^-7 = 2^-208.528, Q_V <= 2^80.53. Correct as stated.
  919  % CHECK (suggestion, no text change): with the tighter reuse bound Q p^{-r} (CHECK at thm:v1kpre),
  920  % the same targets give Q_V <= 2^26.5 (about 9.7e7) at r = 5 and 2^88.3 at r = 7; with lookups=True
  921  % Llama-2-7B has 225, not 226, ops in eps_K (the embedding is read by the verifier).
  950  % -------------------------------------------------------------------------------------
  954  % CHECK: "opens" replaced by "checks" in this subsection (intro, statement, proof). In RandPathTest
  955  % (sampling.py) checking a node opens its parents' rows too -- for a dense layer the whole layer --
  956  % but only the visited node is checked, so every site of the layer is opened by such a path, and
  957  % "opens each site with probability at most 1/N" was false as worded; "checks" is what is meant.
 1007    % "Largest in-range edit" = 60 is the max of the "any" rows of G3_inrange_attack.md Sec. 3; the
 1008    % lowering-only edits, also in range, reach 164. The pooled lowering-only median 6 is confirmed from
 1009    % artifacts/results/inrange_llm_opt-1.3b_down.json (117 successful records per range; 12 Oct).
 1013    % python: math.ceil(40*math.log(2)/(-math.log1p(-k/N)))
 1033    % python: (1-k/N)**s
```

## sections/ext_v1.tex

```
    1  % =====================================================================================
    2  % sections/ext_v1.tex: material cut from sections/v1.tex for the EXTENDED VERSION only
    3  % (condensation pass of 2026-10-10, CUT_PLAN.md decision 1). Appendix-ready; \subsection only.
    4  % Origin of each block (section numbers of the 29-page draft of v1.tex):
    5  %   E1 "Details of the Windowed LogUp": Secs. 5.2-5.4 (clamp rate and exception count, field
    6  %      choice, table size, the two counterexamples and the exhaustive test on F_49, the tree's
    7  %      combination rule and transcript size, the transposed layout, what V1 does not need) and
    8  %      Sec. 5.5 (the union-bound shares and the largest model).
    9  %   E2 "Implementation of V1": Sec. 5.6, complete.
   10  %   E3 "Hardening of the V1 Checks": the paragraph "Adversarial review" of Sec. 5.7, findings only.
   11  %      NOTE FOR THE AUTHORS: these findings came from an LLM-agent review workflow (C2 doc Sec. 9).
   12  %      Per CUT_PLAN decision 7 the process is not described here; disclose it in the HotCRP
   13  %      AI-use field and decide whether the extended version should name it.
   14  %   E4 "Proof Size and Running Time of V1": Sec. 5.7 "Cost", "Proof size", "Prover and verifier",
   15  %      and the disabled fallback Table tab:v1, reduced to the time part plus the GKR share (its
   16  %      proof-size part duplicated Table tab:eval-v1 of evaluation.tex); its numbers are macros of
   17  %      code/experiments/paper/numbers.py (the v1, cpuv and real tiers).
   18  %   E5 "Limitations of V1": Sec. 5.7 "Limitations".
   19  % Labels defined: tab:v1. Labels used: sec:v1, lem:v1-logup, thm:v1, sec:fs, sec:setup-proof, sec:gen,
   20  % sec:quality, sec:eval-v1, tab:eval-v1. New bib keys: none (montgomery1985, tillet2019triton exist).
   21  % H5 pass (2026-10-11): N9, one source for the real-weight estimate (1.78-1.85x, C2 doc Sec. 8)
   22  %   in E4; TH-d Sec. 6.2(a) paragraph "Relation to deployed logUp" in E1 (new keys stwo2026,
   23  %   openvm2025logup, lurk2024lookups).
   24  % =====================================================================================
   35  % ---- E1 -------------------------------------------------------------------------------
   58  % Added by the H5 pass from drafts/th_d_citations.md, Sec. 6.2(a) (REVIEW.md B-W6); sources and
   59  % bib entries (stwo2026, openvm2025logup, lurk2024lookups) as checked there.
   92  % ---- E2 -------------------------------------------------------------------------------
  115  % ---- E3 -------------------------------------------------------------------------------
  130  % ---- E4 -------------------------------------------------------------------------------
  188  % ---- E5 -------------------------------------------------------------------------------
  198  % ---- E6 -------------------------------------------------------------------------------
  199  % The cut of Llama-2-13B from its shapes (code/experiments/paper/tables.py, v1_shape_model): every
  200  % operation that rule (P2) allows cut, as with random weights, under the run's plan and lmax.
```

## sections/generation.tex

```
    1  % =====================================================================================
    2  % sections/generation.tex: Section 6, Generation and Sampling. Its measured numbers are macros of
    3  % code/experiments/paper/numbers.py (the gen and tv tiers). Tests, Tables tab:gen and tab:gen-full and the
    4  % sampler's implementation details are in ext_generation.tex (extended version only).
    5  % Labels defined: sec:gen, sec:gen-prefill, sec:gen-sampling, eq:gumbel-table.
    6  % Labels used: sec:defence, sec:fs, thm:sound, sec:v1 (defence.tex, v1.tex).
    7  % =====================================================================================
```

## sections/generation_quality.tex

```
    1  % =====================================================================================
    2  % Section: Generation and model quality (plan items E1, E2, F2)
    3  %
    4  % Sources used (every number below comes from one of these):
    5  %  - wt_sp2027/docs/improvements/E1_generation.md  (statement, token rule, soundness argument,
    6  %    tests, Table tab:gen: RTX 2080 Ti prover, Xeon Silver 4114 verifier with 8 threads,
    7  %    64-token prompt; the 1.3 TB vs 3.3 GB per-prefix comparison; zkAgent 0.30 s/token)
    8  %  - wt_sp2027/docs/improvements/E2_sampling.md    (sampler definition, integer bounds, table
    9  %    and its SHA-256 check, soundness, seed binding, tests incl. 380-620 of 4,000; stop rules, top-p)
   10  %  - wt_sp2027/docs/improvements/F2_quality.md     (integer model description, attribution table,
   11  %    smoothing formula, Table tab:quality, robustness x1.03-1.05, per-row scales x1.00-1.01,
   12  %    literature targets, OPT-6.7B 36.0 vs 27.1). Rows of the attribution table left out for
   13  %    space: q/k/v weights 74.8 / 65.0, attention output requantisation 75.7 / 64.8; cross-layer
   14  %    equalisation (67.8 against 66.6, no help) also left out.
   15  %  - wt_sp2027/docs/improvements/D1_D2_fiat_shamir_binding.md (the graph digest binds every
   16  %    cheap-op constant, hence the norm gains, and the generation graph's lm_positions)
   17  %  - wt_sp2027/SP2027_PLAN.md (status of E1, E2, F2; what remains)
   18  %  - sp2027_research/f2_quality/literature.json (SmoothQuant-O3 x1.016 / x1.008, ZeroQuant-V2
   19  %    x1.01 / x1.04 with per-token A8 and W16, CushionCache's table: static W8A8 x1.054 and SQ-O3
   20  %    x1.00 on OPT-6.7B with group-wise weights, zkLLM/zkGPT under 1% at 16 bits, SLP TinyLlama
   21  %    +8.5% and 84.8% top-1, published sequences of 2,048 tokens)
   22  %  - sp2027_research/literature.md (zkAgent row; venue bar)
   23  %  - report/main.tex (course paper: best scalar LayerNorm gains 4, 3, 2; range bound; settings
   24  %    C and Kpre; Theorem thm:sound and its first-wrong-layer proof)
   25  %  - zkAgent title/authors checked on eprint.iacr.org/2026/199; SmoothQuant on PMLR v202;
   26  %    CushionCache, ZeroQuant-V2, OPT, WikiText and A* sampling on arXiv (see extra.bib).
   27  %
   28  % Assumes from the main preamble: booktabs, amsmath. Labels used from other sections (all
   29  % defined in defence.tex or v1.tex): sec:defence, sec:intmodel, sec:fs, thm:sound, sec:v1.
   30  % Labels defined here: sec:gen (the alias sec:generation was removed at assembly), sec:gen-prefill,
   31  % sec:gen-sampling, sec:quality, tab:gen, tab:quality, eq:gumbel-table.
   32  % No \todo is placed inside a float or caption (todonotes cannot place margin notes there).
   33  % =====================================================================================
```

## sections/intro.tex

```
    1  % =====================================================================================
    2  % sections/intro.tex: Introduction and Related Work (IEEE S&P 2027 submission, IEEEtran compsoc).
    3  % Condensed on 2026-10-10 per CUT_PLAN.md (Sec. 4, decisions 3-5); removed detail, with its todos,
    4  % is in sections/ext_intro.tex. Trimmed again the same day to 1,740 words (Related Work facts
    5  % restated in Secs. 3, 4 and 8 removed). Sources: the previous intro.tex and the numbers of attack.tex,
    6  % defence.tex, v1.tex and evaluation.tex. Labels defined: sec:intro, fig:overview, sec:related.
    7  % Labels used: sec:attack, sec:sampling, thm:guess, thm:dich, sec:sok, tab:sok, sec:defence,
    8  % sec:setup-proof, sec:gen, sec:v1, sec:results, tab:eval-compare, sec:discussion.
    9  % New BibTeX keys (reported to the integrator): gpucc, deepbrake, reedweave.
   10  % H5 pass (2026-10-11, REVIEW2.md): cut C1 (the "A lower bound" paragraph without the restated
   11  %   verifier model and the PCPP sentence), C2 (one clause per extension), C3 (the folklore/PCPP point
   12  %   once, in Related Work), P1 (extension 1 and the contributions), TH-a (Part B1 with the
   13  %   O(lambda + log L) fix; B2 as "tight at every locality"), N7 (intro's "Such tests could shorten
   14  %   our setup proof" replaced by a pointer to Sec. sec:setup-proof, which carries TH-c's remark),
   15  %   N13 (interactive/extrapolated label on the 4.6x); todos renamed to their items (F6, A7, RX).
   16  % H5 review (2026-10-11): Related Work, "Its tight case" -> "The calculation's tight case" (the
   17  %   inserted thm:guess sentence had made "Its" refer to the trace).
   18  % =====================================================================================
  134  % =====================================================================================
```

## sections/quality.tex

```
    1  % =====================================================================================
    2  % sections/quality.tex: the subsection "Quality of the Certified Model", \input at the end of
    3  % evaluation.tex. Its numbers are macros of code/experiments/paper/numbers.py (the quality tier,
    4  % code/experiments/quality/). The attribution table, the smoothing details and the full literature
    5  % comparison are in ext_generation.tex.
    6  % Labels defined: sec:quality, tab:quality.
    7  % Labels used: sec:intmodel (defence.tex).
    8  % =====================================================================================
```

## sections/v1.tex

```
    1  % =====================================================================================
    2  % sections/v1.tex: V1, the proof-size reduction (S&P 2027 body, condensed 2026-10-10).
    3  % Sources: wt_sp2027/docs/improvements/C2_v1_proof_size.md; sp2027_research/v1_design/V1_SPEC.md;
    4  % sections/defence.tex (notation). Material cut from this section is in sections/ext_v1.tex.
    5  % Labels defined: sec:v1, eq:v1-window, eq:v1-residue, eq:v1-logup, lem:v1-logup, eq:v1-final,
    6  % thm:v1. Labels used: sec:defence, sec:protocol, sec:fs, eq:freivalds, thm:sound (defence.tex);
    7  % sec:eval-v1 (evaluation.tex); app:v1 (appendix_short.tex, appendix_v1.tex); tab:params
    8  % (app_params.tex, \input by both appendix_short.tex and the extended branch of main.tex).
    9  % tab:v1 (duplicate of tab:eval-v1) moved to ext_v1.tex. Second pass (2026-10-10): 1,700 -> 1,545
   10  % words; cut detail listed under E6 in ext_v1.tex.
   11  % H5 pass (2026-10-11, REVIEW2.md cut C8): (a) the Llama-2-13B error terms left the body
   12  %   (tab:params, footnote b, and ext_v1.tex E6 keep them); (b) "Implementation and cost" is one
   13  %   sentence pointing to Sec. sec:eval-v1; (c) the GKR packing description shortened (details in
   14  %   ext_v1.tex E1); (d) the lemma's explanation deleted (appendix_short.tex "Proof ideas" has it);
   15  %   (e) the opening cut to one sentence.
   16  % =====================================================================================
  172  % The Freivalds term runs over all clear committed matrices (as in thm:sound) and the column
  173  % term takes thm:sound's product form, at most ((k_j - 1)/n_j)^{t_j}; V1_SPEC.md writes both
  174  % more loosely.
```

