# TH-f checklist (12 Oct 2026)

Prepared for the team's hand check of the proofs (TH-f, Fri 30 Oct). It resolves the open points of
`drafts/th_e_check.md` against the code. Paths are relative to the repository root.

Already applied on 12 Oct (commit after ca75493), each checked against the code or the raw results:

- X1: `\section{Proofs of the Optimisations}` (`app:opt`) heads `ext_proofs.tex`.
- 7.5: the stale sentence on the real-weight builders is replaced. A `%` comment line had swallowed "The real-weight".
- 7.1: the body's bound is quoted with the round-by-round error.
- X2: the review wording is removed.
- 5: the timing remark states what the CPU verifier does.
- 4a: the pooled median is confirmed from `artifacts/results/inrange_llm_opt-1.3b_down.json` (117 successful
  records per range, median 6, maximum 164).
- 3: tab:params states the collision total for Q <= 2^63.

Left for the team: the items marked "needs the team (TH-f)" below, the optional caption clause (4b), the optional
key-lifetime clause (7.3), and whether to budget one more bit of beta (3).

# TH item: open decisions in `drafts/th_e_check.md`

**Scope.**
- `th_e_check.md` refers to no other `drafts/th_*.md`. It checks `drafts/th_e_proofs.tex`, which is now integrated as `paper_sp2027/sections/ext_proofs.tex` (`SP2027_PLAN2.md:885`). The "current text" references below point to `ext_proofs.tex`.
- TH-f, the team's hand check, is on Fri 30 Oct (`SP2027_PLAN2.md:1217`).
- No file was edited.

## Summary

| # | Question (th_e_check.md line) | Recommendation | Mark |
|---|---|---|---|
| 1a | Tighter Kpre reuse bound: Q p^-r instead of Q Σ_ℓ p^-r (:140, :38-48) | The argument holds, and the code follows the convention it needs. Keep the body as it is; add a remark to the extended version. | needs the team (TH-f) |
| 1b | Mode K: p^-r per query instead of ε_K (:37) | It holds. Fold it into the same remark. | needs the team (TH-f) |
| 2a | Chiesa-Yogev theorem numbers and the form (t+k)ε_rbr (:142) | Check them against the 25 Mar 2026 PDF. A fallback text that drops the numbers is given. | needs the team (TH-f) |
| 2b | Do (R2)/(R3) cover statements the prover chooses adaptively? (:143) | The draft's own move-counting argument covers this. One sentence is given. | needs the team (TH-f) |
| 2c | Is (R2)'s proof a union bound over moves? (:144, :94) | I re-derived it and it holds. Confirm against the book. | needs the team (TH-f) |
| 2d | Canetti et al. and Holmgren statements not checked (:146) | Both are cited only for the definition and the equivalence. No text change. | needs the team (TH-f) |
| 3 | Collision term at Q = 2^64 against "≤ 2^-128" (:147) | Reword one row of tab:params. Adding a bit to β is optional. | safe to apply now (wording); the parameter choice needs the team |
| 4a | Pooled lowering-only median of 6 (:149) | Confirmed from the raw results. No text change. | safe to apply now (delete the CHECK comment) |
| 4b | Add a column for 164? (:150) | No column. An optional clause in the caption. | safe to apply now (optional) |
| 5 | Timing, rem:stateful (c) (:151-154) | Replace "we have not checked" with the facts found in the code. | safe to apply now |
| 6 | P_max changed during the check (:155) | The text already matches the code. The rest is a test run for the engineer, not a text change. | no text change |
| 7.1 | Open point 1: the body's Sec. sec:fs bound | The body was fixed, but ext_proofs.tex misquotes it. | safe to apply now |
| 7.3 | Open point 3: Kpre key lifetime in the body | No body change is needed. One optional clause. | needs the team (page budget) |
| 7.5 | Open point 5: real-weight builders and pruning | The text is stale, and a `%` comment swallows "The real-weight" in the rendered PDF. | safe to apply now (the last sentence: TH-f) |
| 7.6 / 7.7 / 7.8 | Plan difference; BibTeX; remark on the sum | All closed. | no change |
| — | V1 GKR sum over γ could be a maximum (:49) | Numerically irrelevant. | no change |
| X1 | Not in th_e_check.md: the ext_proofs subsections sit under "Soundness of V1" in main_ext | Add a `\section`. | safe to apply now |
| X2 | Not in th_e_check.md: "mock review" and "Question 5/6" are visible text | Reword. | safe to apply now |

---

## 1a. The tighter Kpre reuse bound

**(a) Question.** Can thm:v1kpre use Q p^-r in place of Q Σ_ℓ p^-r?

**(b) Current text.**
- `ext_proofs.tex:786-793` (theorem), `:807-812` ("Bad checks"), `:843` (cor:v1kpre-fs) and `:900-906` (rem:stateful (a)).
- Body: `defence.tex:158-161` (thm:sound), `v1.tex:166-169` (thm:v1), `appendix_short.tex:189-193` and `appendix_defence.tex:130-132`.

**(c) Recommendation.**
- I re-derived the argument and it holds. In the hybrid H, a query stops at its first failing check. So the real run and H first differ in query i only if the first secret check with E_c ≠ 0, in the verifier's fixed order, passes. Which check that is, and its E_c, are functions of H's view, which does not depend on χ. So each query adds at most p^-r.
- When a group is evaluated at once, the group passes only if that particular check passes, so the bound still holds.
- The code follows the convention the argument needs:
  - V1 returns at the first failed label (`cut_protocol.py:363-506`), with `freivalds` before `kpre_y` (`:481`, `:501`).
  - v0 does the same (`protocol.py:1480`, `:1537-1554`, `:2010`).
  - `_disagrees` stops at the first failing op on the CPU and checks all ops at once on a device (`protocol.py:678-685`).
  - `_kpre_y_ok` returns at the first mismatch (`cut_protocol.py:584-591`).
- Numbers, recomputed with log2 p = 30.9069 (`th_e_check.md:68`):
  - r = 5: Q_V ≤ 2^26.53 = 9.7·10^7, against 2^18.71 today.
  - r = 7: Q_V ≤ 2^88.35, against 2^80.53.
- Recommendation: keep the body's sum forms. They are valid upper bounds, and the submission body is at its page limit. Put the tighter bound in the extended version as a remark, after TH-f confirms it.

**(d) LaTeX.**

Insert after `ext_proofs.tex:836` (`\end{proof}` of thm:v1kpre):
```latex
\begin{remark}[the sum over operations is not needed]
\label{rem:kpre-max}
The verifier evaluates the secret checks of a query in a fixed order and stops at the first failure
(Appendix~\ref{app:the-conv}). The real run and $\mathsf H$ therefore first differ in query $i$ only if
the first secret check with $E_c\neq0$, in that order, passes. Which check this is, and its error, are
functions of the prover's view in $\mathsf H$, so this has probability at most $p^{-r}$, and
Theorem~\ref{thm:v1kpre} holds with $Q\,p^{-r}$ in place of $Q\sum_\ell p^{-r}$. For the same reason one
query of mode $\sK$, whose vectors are drawn after the claims fix the first wrong operation, errs with
probability at most $p^{-r}$. This needs the convention that a verdict names only the first failed
check. Our parameters do not use the tighter form.
\end{remark}
```
Replace `ext_proofs.tex:905-906`. The sentence that ends at the top of `:905` is kept:
```latex
$Q_V\varepsilon_{\sK}\le2^{-128}$ for $Q_V\le2^{18.7}$ verdicts only, about $4\cdot10^5$. The
parameters with the 64-bit margin ($r=7$) give $2^{-208.5}$ and $Q_V\le2^{80.5}$. With
Remark~\ref{rem:kpre-max} the same targets allow $Q_V\le2^{26.5}$ (about $9.7\cdot10^7$) and
$Q_V\le2^{88.3}$.
```
If the team adopts the tighter bound in the body instead, replace `defence.tex:158-161` with the following, and make the matching edits at `v1.tex:167` and `appendix_short.tex:191-192`:
```latex
In mode $\sK$ the bound is $\varepsilon_{\sK}=p^{-r}$. In mode $\Kpre$, over $Q$ queries that reuse
the secret vectors, the verifier accepts some $y\neq M(x)$ with probability at most $Q\,p^{-r}$, even if
the prover learns the first check that failed in every rejected query.
```

## 1b. Mode K, p^-r per query

- **(a)** Is ε_K = Σ_ℓ p^-r needed for one query of mode K?
- **(b)** `defence.tex:158`, and the suggestion comment at `ext_proofs.tex:727-729`.
- **(c)** It is not needed. The first wrong operation is fixed by the claims before the fresh χ is drawn, so one query errs with at most p^-r. This is covered by the remark in 1a.
- **(d)** See 1a. Delete the comment at `:727-729` when the remark goes in.

## 2a. Chiesa-Yogev theorem numbers and the form of (R2)

- **(a)** Are Claim 31.1.3, Theorem 31.2.1, Theorem 14.3.1 and the form (t+k)ε_rbr correct?
- **(b)** `ext_proofs.tex:76-83`.
- **(c)** A team member checks them against the PDF of 25 March 2026. The BibTeX entry already pins that version (`paper_sp2027/extra.bib:474-479`). If the numbers cannot be confirmed by 30 Oct, cite the results by content instead of by number.
- **(d)** Fallback for `:76-83`:
```latex
the textbook of Chiesa and Yogev~\cite{chiesa2024snargs}:
(R1)~the soundness error is at most $\sum_i\varepsilon_i$;
(R2)~with $\varepsilon_{\mathrm{rbr}}=\max_i\varepsilon_i$, a state-restoration prover with $t$
moves wins with probability at most $(t+k)\,\varepsilon_{\mathrm{rbr}}$ (Holmgren
shows that the two notions are equivalent~\cite{holmgren2019rbr}); and
(R3)~the Fiat--Shamir transformation with the full transcript as the hash input inherits the
state-restoration error (see also~\cite{bcs16}).
```
The line before `:76` then ends "We use three consequences, as stated in".

## 2b. Statements chosen adaptively

- **(a)** Do (R2)/(R3) cover a prover that chooses the statement?
- **(b)** `ext_proofs.tex:431-433` (thm:fstree), `:317-322` (thm:gen proof) and `:838-861` (cor:v1kpre-fs).
- **(c)** The draft's own argument covers this:
  - The doomed set contains the empty transcript of every false statement (`:73-74`).
  - Each challenge input decodes to its statement (lem:tree-dec (a), `:403-405`; `_absorb_statement` absorbs `graph` first).
  - Counting the moves that escape a doomed state therefore needs no fixed statement.
  - The team checks that the book's Theorem 31.2.1 is proved this way. If the book states (R2) only for a fixed instance, add the sentence below.
- **(d)** Insert after "most $k$ moves." at `ext_proofs.tex:484`:
```latex
A move may begin a new statement, since the statement is part of $\tau(w)$; because the doomed set
contains the empty transcript of every false statement, the argument behind (R2), which bounds the
probability that some move leaves a doomed state, applies unchanged when the prover chooses the
statement adaptively.
```

## 2c. (R2) as a union bound over moves

- **(a)** Does (R2) extend to Q_V final proofs?
- **(b)** `ext_proofs.tex:858-861`.
- **(c)** It does. Every accepted false proof ends in a complete transcript that is not doomed. Reaching it takes one escaping move among the Q_H + kQ_V moves, and each move escapes with probability at most ε_pub. This holds if the book's proof has this form, which the team confirms with 2b.
- **(d)** None. Delete the comment at `:862-864` after TH-f.

## 2d. Canetti et al. and Holmgren

- **(a)** These statements were not verified (HTTP 403).
- **(b)** `ext_proofs.tex:73`, `:80-81`; `defence.tex:194`, `:198`.
- **(c)** Both papers are cited only for the notion of round-by-round soundness and for its equivalence with state-restoration soundness. The team confirms these two points. No text change.
- **(d)** None.

## 3. Collision term at Q = 2^64

- **(a)** At Q = 2^64 the collision term N² 2^-256 alone is about 2^-128. The Fiat-Shamir total is 2^-127.997.
  - This was recomputed: 2^-128 + 2^(64-200.82).
  - At Q = 2^63 the total is 2^-129.99.
- **(b)** `paper_sp2027/sections/app_params.tex:32` ("Total: ε_C / Qε_C, plus collisions ≤ 2^-128"). The text at `ext_proofs.tex:720-724` is already correct.
- **(c)** The submission's table states something slightly false. Fix the wording now; this does not change the parameters. Budgeting one more bit (β + 1) is the team's decision.
- **(d)** `app_params.tex:32`:
```latex
    Total: $\varepsilon_{\sC}$ / $Q\varepsilon_{\sC}$, plus collisions for $Q\le2^{63}$ & $\le2^{-128}$ \\
```
An alternative with the same length: `Total without collisions: $\varepsilon_{\sC}$ / $Q\varepsilon_{\sC}$ & $\le2^{-128}$ \\`.

## 4a. Pooled lowering-only median of 6

- **(a)** Is the pooled median of 6 neurons correct?
- **(b)** `ext_proofs.tex:989-991`; `attack.tex:222-223`.
- **(c)** Confirmed from `code/artifacts/results/inrange_llm_opt-1.3b_down.json`. The 117 successful records give a pooled median of 6 and a maximum of 164, the same for both ranges. The per-layer medians 3, 12, 8 and 2 match `docs/improvements/G3_inrange_attack.md:47-50`. The "any" edits give a pooled median of 2 and a maximum of 60, which matches `attack.tex:219-220`.
- **(d)** No text change. Delete the comment at `ext_proofs.tex:992-995`.

## 4b. A column for 164

- **(a)** Should tab:paths get a column for k = 164?
- **(b)** `ext_proofs.tex:1002`.
- **(c)** No. The table already spans k = 1 to 60. A caption clause is enough.
- **(d)** Optional, `ext_proofs.tex:991`:
```latex
  OPT-1.3B (Section~\ref{sec:break}; lowering-only edits reach 164); $k=16$, and every $k$ for the other models, is illustrative.}
```

## 5. Timing, rem:stateful (c)

- **(a)** Do the verifier's comparisons run in constant time?
- **(b)** `ext_proofs.tex:917-920` ("We have not checked whether ...").
- **(c)** They do not. On the CPU, `_disagrees` uses `torch.equal` op by op and stops at the first failure (`protocol.py:678-679`), and `_kpre_y_ok` returns at the first mismatch (`cut_protocol.py:589-590`). The device path compares all ops at once (`protocol.py:680-685`).
- **(d)** Replace `ext_proofs.tex:918-920`:
```latex
depends on the compared values. The CPU verifier does not run in constant time: it compares the two
sides of Freivalds' check operation by operation with \texttt{torch.equal} and stops at the first
failing operation (\texttt{protocol.\_disagrees}), and check~(F2) returns at the first mismatch
(\texttt{cut\_protocol.\_kpre\_y\_ok}). Which operation fails first is a function of the prover's
messages until the first bad check, so the hybrid covers it. The time of one comparison can depend on
where its two sides first differ, and so on $\chi$, which the proof does not cover. A verifier whose
timing the provider can observe should complete every comparison before it answers, or answer only
accept or reject after a fixed-time check.
```

## 6. P_max changed during the check

- **(a)** Re-run the pruning and generation tests with the chosen P_max.
- **(b)** `ext_proofs.tex:132-134`: "(255 in the cost benchmarks, 2,047 in the certified models)".
- **(c)** The text matches the code: `PROB_MAX = 255` and `CERTIFIED_PROB_MAX = 2047` (`transformer.py:55`, `:59`). The decision is recorded in `docs/improvements/F2a_quality_2048.md:154-159`.
  - At 2047, the pruned and unpruned logits are tested equal in `test_real_weights.py:131-136` and `test_real_llama.py:340`. Generation at 2047 is tested in `test_f3b_real_llama.py:58`.
  - `test_pruning.py` and `test_generation.py` still build at 255, through `build_decoder`'s default.
  - The engineer should run these suites on the cluster before the freeze, or add a `prob_max` parameter to them. This is not a TH-f item.
- **(d)** None. The comment at `:135-137` can go.

## 7.1. Open point 1: the body's Sec. sec:fs bound

- **(a)** Is the body's bound stated correctly in the extended version?
- **(b)** `ext_proofs.tex:506` quotes the body as "Qε_C + O(Q²2^-256)". The body now reads Qε_rbr (`defence.tex:198`).
- **(c)** Correct the quotation.
- **(d)** `ext_proofs.tex:506`:
```latex
\noindent The bound of Section~\ref{sec:fs}, $Q\varepsilon_{\mathrm{rbr}}+O(Q^22^{-256})$, should be read with
```

## 7.3. Open point 3: Kpre key lifetime in the body

- **(a)** Should the body state a maximum Q for one Kpre key?
- **(b)** The body says nothing about it. `discussion.tex:31` says "computes its key once".
- **(c)** The 12.7 s figure uses the Fiat-Shamir parameters, r = 7 (`docs/improvements/A6_kpre_precompute.md:51-54`). These allow Q_V ≤ 2^80.5 (`ext_proofs.tex:906`), so "once" is fair. No change is needed. The optional clause adds about one line to the body.
- **(d)** Optional, `discussion.tex:31`:
```latex
a CPU; a $\Kpre$ client holds the weights and computes its key once (12.7\,s for Llama-2-7B on 8 CPU threads, with $r=7$, enough for $2^{80}$ verdicts).
```

## 7.5. Open point 5: real-weight builders and pruning

- **(a)** Do the real-weight builders build the pruned block?
- **(b)** `ext_proofs.tex:220-222`. The words "The real-weight" are at the end of comment line 220, so the PDF prints "... with this cut set. builders (real_weights.py) do not yet build a pruned block". The statement is also out of date: both builders have `prune_last` (`real_weights.py:185`, `:238`; `real_llama.py:618`, `:719`). They match `build_decoder`'s structure (`real_llama.py:910-934`). The header at `:5-6` already says so.
- **(c)** Replace the sentence. The last sentence is a small proof claim for TH-f.
- **(d)** Replace `ext_proofs.tex:220-222`:
```latex
% thm:v1 applies with whatever cut set results either way.
The real-weight builders (\texttt{real\_weights.build\_opt\_from\_hf} and
\texttt{real\_llama.build\_llama\_from\_hf}) insert the same slices with \texttt{prune\_last} and
calibrate the pruned block on every position; their graphs have the operations, order and notes of
\texttt{build\_decoder}'s (\texttt{matches\_benchmark\_graph}), and their next-token logits with and
without pruning are tested to be equal. The theorem therefore applies to them; a fixed public prefix
(\texttt{prefix\_tokens}) enters only as constants of the attention operation.
```

## 7.6, 7.7, 7.8: closed

- **6, the commitment plan.** Handled at `ext_proofs.tex:209-213`, and tested at `test_pruning.py:103`.
- **7, the BibTeX entries.** Both are present (`extra.bib:474-489`). What remains is 2a.
- **8, the sum in the body.** The body already says that ε_rbr is the maximum (`defence.tex:198-203`), and rem:rbr-max covers the rest.

## X1. Missing appendix heading (not raised by the check)

- **(a)** Where do the ext_proofs subsections appear in the extended version?
- **(b)** `main.tex:92-93` inputs `ext_proofs` right after `appendix_v1`, whose only heading is `\section{Soundness of V1}` (`appendix_v1.tex:37`). `ext_proofs.tex` has no `\section`. So "Last-Block Pruning" through "Acceptance of Edits ..." become subsections of "Soundness of V1".
- **(c)** Add a heading. No body text names an appendix by its letter, so the shift is harmless.
- **(d)** Insert before `ext_proofs.tex:31`:
```latex
\section{Proofs of the Optimisations}
\label{app:opt}
```

## X2. Internal review wording (not raised by the check)

- **(a)** Should printed text mention a mock review?
- **(b)** `ext_proofs.tex:658-659`, `:682` and `:702`.
- **(c)** No. Reword.
- **(d)** Replace `:658-659`, `:682` and `:702` in that order:
```latex
This subsection makes two points about the round-by-round argument of
Appendices~\ref{app:sound} and~\ref{app:v1}.
```
```latex
\para{V1's column check under Fiat--Shamir.} The lemma uses no property of $w$ other than
```
```latex
\begin{remark}[the sum and the maximum]
```

## Smaller points

- The lookup term is ≤ 2^-215.3 in `appendix_v1.tex:286`, `appendix_short.tex:298` and `ext_v1.tex:215`, but ≤ 2^-215.9 in tab:params (`app_params.tex:37`). With N_leaf = 2.7·10^9 (`app_params.tex:55`) it is 2^-215.92. Both bounds are true. Use 2^-215.9 everywhere if consistency is wanted.
- The 21 `% CHECK` comments in `ext_proofs.tex` should go after TH-f, or in H4's scrub. The repository paths in the visible text (`:649`, `:983`) are for H4 to decide.

---

## TH-g check: who tokenises the prompt, and provider system prompts

The sentence is in the body, in the paragraph "Adversary" (`paper_sp2027/sections/attack.tex:48`) of Section 3.1 "Setting and Threat Model" (`attack.tex:33`). It is at `attack.tex:53-55`:

> "The statement binds token ids: the client tokenises its text with the model's public tokenizer, and a system prompt that the provider prepends is part of $x$, which the client receives and the proof covers; a hidden system prompt is beyond what any verifier can certify."

- It is in both builds, since `main.tex:77` inputs `attack`.
- It answers both parts of A-Q7.
- `SP2027_PLAN2.md:912`, which says "grep ... find nothing", is out of date; the 11 Oct status line (`SP2027_PLAN2.md:885`) records the sentence as added.
- It agrees with the real Llama graphs. Their fixed public prefix belongs to the graph (constants of the attention op, bound by the graph digest), not to x (`real_llama.py:56-64`).
- No change is needed.