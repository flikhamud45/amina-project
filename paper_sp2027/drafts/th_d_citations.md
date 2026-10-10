# TH-d: citations, positioning and the reviewer questions B-Q5, B-Q6, B-W6, A-Q4, A-Q7

Plan item TH (d), plus B-Q5 and B-Q6 from TH (e) and A-Q7 from TH (g) (`SP2027_PLAN2.md`, Sec. 3.7). Drafted
2026-10-10. No existing file was edited. Every LaTeX block below is meant to be pasted as written, at the place
named above it.

Tags used below:

- **[V]**: I read it in the primary source in this session. The file or page is named.
- **[S]**: secondary source only.
- **[U]**: I could not verify it.
- **[G]**: my own derivation, to be checked by hand in TH (f).

The PDFs I read are saved in `scratchpad/th_d_downloads/`, a sibling of `wt_sp2027`, outside the repository. The
user allowed local downloads. They are: the BCIKS20 ECCC original and revisions 1-3, the ePrint version (via the
Wayback Machine), Ligero (CCS 2017), Anchuri et al. (arXiv v1), Haböck's logUp (ePrint 2022/1530), Holmgren
(ePrint 2019/1261), the S-two whitepaper, OpenVM's LogUp note and SWIRL.

## 0. Overview

| # | Question | Answer in one line | Files to change |
|---|---|---|---|
| 1 | BCIKS20 theorem | **Thm. 1.6** of the full version (ePrint 2020/654, 3 Jul 2021 = ECCC TR20-083 Rev. 1-3). It is Thm. 1.7 only in the original ECCC report of 30 May 2020. It covers uniform combinations over an extension field if it is applied over K_s. I could not see the FOCS or J. ACM numbering. | defence.tex, appendix_defence.tex, appendix_short.tex, extra.bib |
| 2 | Ligero radius | Lemma 4.2 of the CCS 2017 paper, for e < d/4 with error (e+1)/\|F\|. The d/3 bound is only a conjecture there and a personal communication in BCIKS20, so the paper must not use it. With d/4, Ligero covers the radii with f <= 1/2; BCIKS20 Thm. 1.6 covers all of them, f = 1 included. | appendix_defence.tex |
| 3 | Anchuri et al. version | Table `tab:sok` was built from arXiv:2603.19025**v1**, but the bib cites the SaTML proceedings. The refereed variant (App. D) lies outside Spot_d(q). | extra.bib or references.bib, attack.tex footnote, ext_attack.tex |
| 4 | B-Q5 | The column check uses no property of chi_l; it needs only that chi_l and u_l are fixed before the columns are drawn. This holds under Fiat-Shamir. | appendix_v1.tex, v1.tex, appendix_short.tex |
| 5 | B-Q6 | The proof shows the round-by-round error is a **max**, but the parameters (paper and code) meet the **sum** Q eps_C. That costs log2(2\|J\|) bits of beta and, at lambda = 128, does not change r. | defence.tex sec:fs, appendix_defence.tex |
| 6 | B-W6 | The deployed 31-bit-field systems I checked keep the number of lookups below p: S-two by a verifier-checked bound, OpenVM by an interaction cap, SP1 with unique queries. Lemma `lem:v1-logup` replaces that bound with an integer check on multiplicities sent in the clear. | ext_v1.tex (paragraph), v1.tex (one clause), extra.bib |
| 7 | A-Q7 | x is a sequence of token ids that the client computes, or re-computes and compares. A provider system prompt is part of x, which the verifier reads in full. | attack.tex sec:threat (or defence.tex sec:fs) |

---

## 1. BCIKS20: which theorem, and does it cover uniform combinations over an extension field?

### 1.1 Verified facts

**Versions and numbering [V]** (ECCC report pages and PDFs, ePrint PDF of 3 Jul 2021 via the Wayback Machine):

| Version | Date | Correlated agreement over affine spaces | ... over low-degree curves | "Sampling from extension fields" |
|---|---|---|---|---|
| ECCC TR20-083, original | 30 May 2020 | **Theorem 1.7** | Theorem 1.6 | Remark 1.5 |
| ECCC TR20-083, Revision 1 | 12 Sep 2020 | **Theorem 1.6** | Theorem 1.5 | Remark 1.3 |
| ECCC TR20-083, Revision 2 | 27 Jan 2021 | **Theorem 1.6** | Theorem 1.5 | Remark 1.3 |
| ECCC TR20-083, Revision 3 | 3 Jul 2021 | **Theorem 1.6** | Theorem 1.5 | Remark 1.2 |
| Cryptology ePrint Archive 2020/654, current (PDF dated 3 Jul 2021, "3rd of 3 revisions") | 3 Jul 2021 | **Theorem 1.6** | Theorem 1.5 | Remark 1.2 |
| FOCS 2020, pp. 900-909, DOI 10.1109/FOCS46700.2020.00088 (DOI and pages from Crossref) | Nov 2020 | **[U]**: IEEE Xplore was not reachable | [U] | [U] |
| J. ACM 70(5), 2023, pp. 1-57, DOI 10.1145/3614423 (Crossref) | Oct 2023 | **[U]**: dl.acm.org returned 403 | [U] | [U] |

- The body's "Thm. 1.7" therefore matches only the original ECCC report, the numbering in the comment at the top of
  defence.tex.
- The D4 doc's "Thm. 1.6" matches every later version, including the ePrint version that `literature.md` links.
- The ePrint page lists the paper as "FOCS 2020 (with minor revision)". I could not check the FOCS camera-ready, so
  the paper should not claim a FOCS theorem number.

**The statement (ePrint / ECCC Rev. 3, Theorem 1.6) [V].**

> Let V, q, n, k and δ be as defined in Theorem 1.2. For u_0, u_1, ..., u_l ∈ F_q^D let
> U = u_0 + span{u_1, ..., u_l} ⊂ F_q^D be an affine subspace. If δ ∈ (0, 1 − √ρ) and
> Pr_{u∈U}[Δ(u, V) ≤ δ] > ε, where ε is as defined in Theorem 1.2, then there exist D' ⊂ D and v_0, ..., v_l ∈ V
> satisfying Density: |D'|/|D| ≥ 1 − δ, and Agreement: for all i, the functions u_i and v_i agree on all of D'.
> Furthermore, in the unique decoding regime δ ∈ (0, (1−ρ)/2], there exists a unique maximal D' satisfying the
> above, with unique v_i.

Theorem 1.2 sets **ε = ε_U = n/q for δ ∈ (0, (1−ρ)/2]**. Two notational points from BCIKS20 matter here:

- They write V = RS[F_q, D, k] for polynomials of **degree ≤ k**, so the rate is ρ = (k+1)/n.
- D ⊆ F_q can be any evaluation domain of size n (Sec. 4 preamble).

**Hypotheses matched to our use [V/G].**

- *Random combination versus line or curve.* Theorem 1.6 samples u uniformly from an affine (here linear) space.
  This is our case. A uniform ψ ∈ K_s^{R_j} makes ψ^T E_j uniform on the K_s-span of the rows of E_j, because
  ψ ↦ ψ^T E_j is K_s-linear and every fibre of a linear map has the same size [G, elementary].
  - Fact `fact:ext` turns the s base-field vectors into one such uniform ψ.
  - Neither the curve theorem (Thm. 1.5, error l·ε) nor the line theorem (Thm. 1.4) is needed.
- *Extension field.* The cleanest route applies Theorem 1.6 **over the field F_q := K_s** (q = p^s):
  - the code is the polynomials of degree < k_j over K_s on the n_j points of j's subgroup of F_p ⊂ K_s;
  - u_0 = 0, and u_1, ..., u_l are the rows of E_j, whose entries lie in F_p ⊂ K_s.

  The error is then n_j/p^s. The resulting K_s-codewords have base-field coordinates (coefficient of ξ^0) that are
  F_p-codewords agreeing with the rows on D', by the decomposition in the proof of Fact `fact:ext`. Remark 1.2
  ("one may sample z from a finite extension field ... replacing q with q' in the denominators") gives the same
  conclusion, but it is stated informally for Theorem 1.4 and "the other statements of this section". So I suggest
  citing Theorem 1.6 applied over K_s, which is what appendix_defence.tex already does.
- *Radius.* Our e_j < (n_j − k_j + 1)/2 with e_j an integer means 2e_j ≤ n_j − k_j. That is δ = e_j/n_j ≤ (1 − ρ_j)/2
  with ρ_j = k_j/n_j (our k_j is the dimension, BCIKS's k + 1). This is the unique-decoding regime.
  - The code enforces the same condition: `setup_proof.py`, `2*radius >= n - row_length + 1` is skipped.
  - BCIKS needs δ > 0, that is e_j ≥ 1. The case e_j = 0 is elementary: the set of ψ with ψ^T E_j in the code is a
    proper K_s-subspace, of density ≤ 1/|K_s|.
- *Answer to B-Q1.* Yes. Theorem 1.6, applied over K_s to the K_s-span of the R_j rows, covers uniformly random
  combinations of R_j rows over K_s with error n_j/|K_s|. This assumes δ in the unique-decoding range and e_j ≥ 1;
  the case e_j = 0 is trivial.

### 1.2 Text to insert

**(a) extra.bib, entry `bciks20`.** Replace the entry. The DOI is from Crossref, and the `note` names the version
whose numbering we use.

```bibtex
% [V] 2026-10-10: Thm. 1.6 = "Correlated agreement over affine spaces" in ePrint 2020/654 (3 Jul 2021)
% and ECCC TR20-083 Revisions 1-3; it is Thm. 1.7 only in the original ECCC report (30 May 2020).
% FOCS and J. ACM (70(5), 2023, DOI 10.1145/3614423) numbering not checked.
@inproceedings{bciks20,
  author    = {Ben-Sasson, Eli and Carmon, Dan and Ishai, Yuval and Kopparty, Swastik and Saraf, Shubhangi},
  title     = {Proximity Gaps for {Reed--Solomon} Codes},
  booktitle = {IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS)},
  pages     = {900--909},
  publisher = {IEEE},
  year      = {2020},
  doi       = {10.1109/FOCS46700.2020.00088},
  note      = {Full version: Cryptology ePrint Archive, Paper 2020/654, version of 3~July 2021; theorem numbers refer to the full version}
}
```

**(b) defence.tex, proof sketch of Theorem `thm:untrusted`** (replaces "correlated agreement over affine spaces in
the unique-decoding regime~\cite[Thm.~1.7]{bciks20} \todo{FOCS numbering} make ..."):

```latex
\noindent\emph{Proof sketch.} If an extracted $E_j$ is far, Fact~\ref{fact:ext} and correlated
agreement over affine spaces in the unique-decoding regime~\cite[Thm.~1.6]{bciks20}, applied over
$\mathbb K_s$ to the span of the rows of $E_j$, make $\psi_j^\T E_j$ far from the code over
$\mathbb K_s$ except with probability $n_j/p^s$.
```
(The rest of the sketch is unchanged.) Also update the header comment of defence.tex, line 10: "Cites fullversion
and bciks20 (Thm. 1.6, numbering of ePrint 2020/654 of 3 Jul 2021)."

**(c) appendix_defence.tex, paragraph "Setup."** Replace its first four sentences, from "Let $E_j$ be not
$e_j$-close." to "... $(e_j+1)/|\mathbb K_s|\le n_j/|\mathbb K_s|$.", with the block below. The Ligero sentence is
item 2, and the rest of the paragraph is unchanged.

```latex
\para{Setup.} Let $E_j$ be not $e_j$-close; its rows lie in $\F^{n_j}\subset\mathbb K_s^{n_j}$. We
apply the correlated-agreement theorem for affine spaces~\cite[Thm.~1.6]{bciks20} over the field
$\mathbb K_s$: to the code of polynomials of degree below $k_j$ over $\mathbb K_s$, evaluated at
the $n_j$ points of $j$'s subgroup of $\F$ (rate $\rho_j=k_j/n_j$), to the $\mathbb K_s$-span
$U_j$ of the rows of $E_j$, and to $\delta=e_j/n_j$. As $e_j<(n_j-k_j+1)/2$ is an integer,
$\delta\le(1-\rho_j)/2$, the unique-decoding regime, in which the theorem's error is
$n_j/|\mathbb K_s|$. Since $\psi\mapsto\psi^\T E_j$ is linear, a uniform $\psi\in\mathbb K_s^{R_j}$
makes $\psi^\T E_j$ uniform on $U_j$. If $\psi^\T E_j$ were $e_j$-close to the code with
probability above $n_j/|\mathbb K_s|$, the theorem would give codewords over $\mathbb K_s$ that
agree with all rows of $E_j$ on a common set of at least $n_j-e_j$ columns. Their coordinates of
degree~0 are codewords over $\F$ (proof of Fact~\ref{fact:ext}) that agree with the rows on that
set, so $E_j$ would be $e_j$-close. (For $e_j=0$ the claim is elementary: the $\psi$ for which
$\psi^\T E_j$ is a codeword form a proper $\mathbb K_s$-subspace, of density at most
$1/|\mathbb K_s|$.) For $e_j<(n_j-k_j+1)/4$, a quarter of the minimum distance, Lemma~4.2 of
Ligero~\cite{ligero} gives the same conclusion with error $(e_j+1)/|\mathbb K_s|\le
n_j/|\mathbb K_s|$; this covers every radius with $f\le1/2$.
```

**(d) appendix_short.tex, paragraph "Setup."** Replace "correlated agreement~\cite[Thm.~1.7]{bciks20} \todo{FOCS
numbering} makes" with:

```latex
correlated agreement over affine spaces in the unique-decoding regime~\cite[Thm.~1.6]{bciks20},
applied over $\mathbb K_s$, makes
```

**(e) literature.md and idea_trust-rigor.md** (research notes, outside the paper): their "[U, exact statement to be
checked]" tags for BCIKS20 Thm. 1.6 can become [V] with the version table above. This is optional and not part of
this output.

---

## 2. Ligero: the lemma and the radius

### 2.1 Verified facts [V] (Ligero, CCS 2017, pp. 2087-2104, DOI 10.1145/3133956.3134104; PDF from acmccs.github.io)

- Sec. 4.1, "Testing Interleaved Linear Codes", is stated for a linear code L[n, k, d] over F. Here d(U*, L^m) is the
  distance of the interleaved code, counted in columns.
- **Lemma 4.2** (p. 2093): "Let e be a positive integer such that e < d/4. Suppose d(U*, L^m) > e. Then, for a random
  w* in the row-span of U*, we have Pr[d(w*, L) ≤ e] ≤ (e+1)/|F|."
- Theorem 4.3 (the test) has the same e < d/4 and error (1 − e/n)^t + (e+1)/|F|.
- **The d/3 bound is not a theorem of the CCS paper.**
  - Sec. 4.1.1 "A Tighter Analysis?" conjectures that e < d/4 "can be relaxed to e < d/3 or possibly even e < d/2".
  - Lemma 4.3 (e < d/3, |F| ≥ e) proves only the reduction to a point at distance > e.
  - The improved Lemma 4.4 and Theorem 4.4 (e < d/3, error d/|F|) are stated "Suppose Conjecture 4.1 holds".
- BCIKS20 (Sec. 1, ePrint version) credits "δ < δ_V/3 by Roth and Zémor [RZ18]". Its bibliography lists [RZ18] as
  "Ron M. Roth and Gilles Zémor. **Personal communication**, 2018." [V].
- The journal version (Designs, Codes and Cryptography 91, 2023, pp. 3379-3424, DOI 10.1007/s10623-023-01222-8)
  exists, but I could not open it, so whether it proves d/3 is **[U]**.

**What the paper uses.**

- The paper uses BCIKS20 Thm. 1.6 for every radius up to the unique-decoding radius. It uses Ligero's Lemma 4.2
  (e < d/4, with d = n_j − k_j + 1) only as the elementary alternative.
- The radius rule is e_j = ⌊f(n_j − k_j)/2⌋ with f ∈ {2^0, ..., 2^-10} (appendix_defence.tex; `setup_proof.py`,
  `FRACTIONS`).
  - For f ≤ 1/2: e_j ≤ (n_j − k_j)/4 < d/4, so Lemma 4.2 suffices.
  - Only f = 1 needs BCIKS20 [G, arithmetic].
- The paper must not invoke d/3.
- Which f the auto rule picks for each Llama-2-7B tree is **[U]** (open point 2).

### 2.2 Text to insert

- The Ligero sentence is already in block 1.2(c), which ends "this covers every radius with $f\le1/2$".
- In citation form, use `Lemma~4.2 of Ligero~\cite{ligero}`, or equivalently `\cite[Lemma~4.2]{ligero}`.
- If a reviewer asks about d/3, add an optional footnote to that sentence in the extended version only:

```latex
\footnote{The $d/3$ bound that BCIKS20 attribute to Roth and Z\'emor (personal communication) is
stated as a conjecture in the conference version of Ligero (Sec.~4.1.1); we do not use it.}
```

---

## 3. Anchuri et al.: which version Table `tab:sok` uses, and the refereed variant

### 3.1 Verified facts

**The two versions.**

- arXiv:2603.19025 has only **v1**, dated 19 Mar 2026 [V, arxiv.org/abs]: 55 pages; arXiv comment "49 pages, 14
  figures. Accepted at IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026". Page 1 says "This
  work has been accepted for publication at ... SaTML. The final version will be available on IEEE Xplore." [V, PDF].
- ePrint 2026/541 was received 17 Mar 2026, revised 1 Apr 2026, and is marked "Published at the 4th IEEE Conference
  on Secure and Trustworthy Machine Learning (minor revision)" [V, ePrint page]. Whether its content equals arXiv v1
  is **[U]**.

**What the table used.**

- The table was built from arXiv **v1** [V]. `sp2027_research/attack_theory/papers/anchuri.pdf` is byte-identical
  to the arXiv v1 PDF (I checked with `cmp`), and `attack_theory/sok.md` says "Page numbers refer to the arXiv PDFs".
- The quotations used by the table and by ext_attack.tex are in v1 [V]:
  - "the analysis for other-model soundness trivially extends to full soundness for the case of 'classification'
    models" is on p. 7;
  - ext_attack.tex's "Appendices E and F and Section 7.2" match v1's sections: E Inverse Transform Attack, F Swap
    Attack on LLMs, 7.2 Gradient Descent Attack on Basic Models.
- The bib entry `anchuri2026verifiable` cites the SaTML proceedings, pp. 955-977, DOI 10.1109/SaTML68715.2026.00058.
  I could not compare the proceedings version with v1 (IEEE Xplore was not reachable), so it is **[U]** whether
  pages, appendix letters and the quoted sentences match.

**The refereed variant** is App. D of v1, pp. 37-38 [V]:

- Two parties P_1, P_2 claim different outputs, at least one is honest, and C_M is honest.
- Each party commits to a hashed trace A_{i,k} = [a_{i,k}, H(a_{i,0}, ..., a_{i,k−1})].
- The referee bisects to the first index where the commitments disagree. It then has the parties open the
  activations and weights feeding that node, and accepts the party whose node satisfies a = φ(Σ w a).
- Theorem 3: "If at least one of the two parties is honest the protocol always terminate by accepting the correct
  claim."
- The abstract and Sec. 1.1 present it as "an additional contribution" "in the refereed delegation model, where two
  competing servers enable correct output identification in a logarithmic number of rounds."

**Why the variant lies outside Spot_d(q) [G, checked against Def. `def:spot`].**

- Def. `def:spot` requires that V "chooses each query from P, its coins and the earlier answers", with P one claimed
  trace, and that V learns θ only through the commitment oracle.
- The referee's queries are positions that the *second* server's answers determine.
  - Those answers are entries of the honest trace, so they depend on the true values h_i and hence on the edited
    site J.
  - Modelled as oracle queries, each hashed entry depends on every earlier activation, so its support is not bounded
    by any d < N.
- Against our edit, the honest and the edited traces agree before the edited neuron u, and they differ at u and
  after it, since every later hash includes a_u. So the bisection ends at u, where the edited trace fails the node
  check.

### 3.2 Text to insert

**(a) Bib.** Keep the SaTML entry, which is the archival version, and add a note that names the version actually
read. In references.bib, add to `anchuri2026verifiable`:

```bibtex
  note      = {Full version: arXiv:2603.19025v1 (19~March 2026) and Cryptology ePrint Archive, Paper 2026/541; page and section numbers refer to arXiv:2603.19025v1}
```
(If references.bib must stay "copied unchanged from report/references.bib", as main.tex says, then put a separate
`@misc{anchuri2026full, ...}` with this note in extra.bib, and cite `\cite{anchuri2026verifiable,anchuri2026full}`
where page or section numbers are given.)

**(b) attack.tex, footnote of Table `tab:sok`.** After "$^a$Our models (Section~\ref{sec:break})." insert:

```latex
Path-protocol entries follow the full version of~\cite{anchuri2026verifiable}
(arXiv:2603.19025v1); its two-server refereed variant is outside our model~\cite{fullversion}.
```

**(c) ext_attack.tex, paragraph "Entries shortened in Table~\ref{tab:sok}."** Append these four sentences:

```latex
The full version of~\cite{anchuri2026verifiable} (Appendix~D of arXiv:2603.19025v1) adds a
protocol in the refereed model: two servers commit to hash-chained traces, at least one of them is
assumed honest, and the referee bisects to the first position where the two commitments disagree
and checks that node against its committed weights. Its queries are positions that the second
server's answers determine, and these answers depend on the true trace and hence on the edited
site, whereas a verifier in $\mathrm{Spot}_d(q)$ chooses its queries from the view $P$, its coins
and the oracle's answers alone; each compared entry also hashes all earlier activations, so its
support is not bounded by any $d<N$. The variant thus lies outside Definition~\ref{def:spot}, and
against our single-site edit the bisection ends at the edited neuron, which fails its check. Its
guarantee rests on one honest server out of two, an assumption that our threat model, with a
single provider, does not make.
```

**(d) Answer to A-Q4 (rebuttal text).**

- Table `tab:sok` uses the full version, arXiv:2603.19025v1. We cite the SaTML proceedings as the archival version
  and will state the version in the bibliography.
- The two-server refereed variant (App. D) is outside our verifier class and our threat model, for the reasons in
  (c). Our attack does not apply to it when one server is honest.

---

## 4. B-Q5: V1's column check under Fiat-Shamir

### 4.1 What the text says now [V]

- v1.tex already contains: "That check needs only that $u_\ell$ precedes the columns, not that $\chi_\ell$ is uniform."
- appendix_v1.tex, Step 6 of the proof:
  - writes χ_ℓ = Σ ξ^q χ^(q) and u_ℓ = Σ ξ^q u^(q);
  - finds a plane q with u^(q) ≠ χ^(q)T A_ℓ;
  - concludes from "The $t_j$ distinct columns are drawn after $u_\ell$ is sent".

  No step uses the distribution of χ_ℓ [V, read].
- The appendix's paragraph "Fiat--Shamir" bounds only the new rounds (θ_k, η_k, sum-check, α). It does not mention
  the column round, which is the gap B-Q5 points at.
- The code orders the transcript as statement, claims, u, columns. In `protocol.py` the folds are absorbed
  (`ch.absorb_large(b"u/F31", ...)` or `b"u/"+k`) before `verifier.column_challenges(ch)` [V].

### 4.2 Text to insert

**(a) appendix_v1.tex, end of Step 6.** After "... unless the prover finds a hash collision." (the end of the
mode-C row-layout argument), insert:

```latex
This step uses no property of $\chi_\ell$: for every $\chi_\ell$ that is fixed before $u_\ell$ is
sent, $u_\ell\neq\chi_\ell^\T A_\ell$ gives two distinct codewords in some plane $q$, and their
agreement set is fixed before the columns are drawn.
```

**(b) appendix_v1.tex, paragraph "Fiat--Shamir."** Replace the whole paragraph with:

```latex
\para{Fiat--Shamir.} The interactive protocol is round-by-round sound, as in
Section~\ref{sec:fs}: once a message makes the statement doomed, only the next challenge can rescue
it. For the new rounds this probability is at most $1/p^8$ for $\theta_k$ and $\eta_k$,
$3/p^8=2^{-245.7}$ for a sum-check round, and $(\Nleaf+|\Vtab|)/(p^8-p^2)\le2^{-215.3}$ for
$\alpha$. The column round needs no new argument, although its vectors $\chi_\ell$ are not uniform:
they are $\Veq$ tables of GKR challenges, which under Fiat--Shamir the prover can influence through
its messages.
Step~6 needs only that $\chi_\ell$ and $u_\ell$ (or $\bar x$ and $y$) are fixed when the columns are
drawn. Under Fiat--Shamir the columns are derived from a transcript that already contains them, so
a transcript with a wrong fold is doomed, and the columns rescue it with probability at most
$\prod_{i<t_j}(k_j-1-i)/(n_j-i)$, the per-round error of v0. All these errors are below the
per-round error that the Fiat--Shamir parameters of v0 allow at $\lambda=128$, so its bound for
$Q$ oracle queries carries over unchanged.
```
(The last sentence previously said "state-restoration bound". I kept the claim but dropped the term, since
Section `sec:fs` speaks of the round-by-round bound. Keep the old wording if you prefer.)

**(c) v1.tex, sentence after the folds** (optional, same length class). Replace "That check needs only that
$u_\ell$ precedes the columns, not that $\chi_\ell$ is uniform." with:

```latex
That check needs only that $\chi_\ell$ and $u_\ell$ are fixed before the columns are drawn, not
that $\chi_\ell$ is uniform, and this also holds under Fiat--Shamir (Appendix~\ref{app:v1}).
```

**(d) appendix_short.tex, "Proof of Theorem~\ref{thm:v1}".** After "... which the column check catches except with
the column term." insert:

```latex
This needs only that the fold is fixed before the columns are drawn, not that $\chi_\ell$ is
uniform, so it holds for the $\Veq$-structured $\chi_\ell$ and under Fiat--Shamir.
```

**A remark that stays out of the paper [G].**

- The same "fixed before the columns" argument also holds for the online part of Theorem `thm:untrusted`: E_j and
  U_j agree outside B_j for *any* combination, as appendix_defence.tex already notes for the transposed layout.
- So uniform vectors are needed only inside the setup proof.
- Theorem `thm:v1`, however, is stated for an honest C_M. I would not claim V1 with an untrusted commitment in the
  paper without a full proof. Remark (iv) of appendix_v1.tex already treats only (P4).

---

## 5. B-Q6: sum or maximum of the per-round errors

### 5.1 Verified facts [V]

**The paper.**

- defence.tex, sec:fs: "A prover that makes $Q$ random-oracle queries therefore succeeds with probability at most
  $Q\varepsilon_{\sC}+O(Q^22^{-256})$ ... We add 64 bits to $\beta$, which keeps $Q\varepsilon_{\sC}\le2^{-\lambda}$
  for $Q\le2^{64}$ and raises $r$ to 7".
  - Here ε_C = Σ_{j∈J}[p^{-r} + col_j] (Thm. `thm:sound`) is a **sum**.
  - β = λ + log2(2|J|), plus 64 under Fiat-Shamir, budgets each of the 2|J| terms at 2^{-β}, which is exactly a
    union bound over them.
- appendix_defence.tex, "Round-by-round soundness", already proves the tighter fact: "The round-by-round error is
  therefore at most $\max(p^{-r},\max_j\prod_{i<t_j}(k_j-1-i)/(n_j-i))\le\varepsilon_{\sC}$". The definition of
  round-by-round error is indeed a single per-round rescue probability.
  - Canetti et al. 2019, restated as Definition 2.5 in Holmgren, ePrint 2019/1261 [V]: for a doomed prefix and
    every next prover message, Pr over the next challenge [not doomed] ≤ ε.
  - It is therefore a maximum over rounds and doomed prefixes, not a sum.
- The paper thus proves a max but states and meets the sum.

**The code** (`code/src/pvi/fullcheck/`).

- `plans.column_bits(lam, n_checks, fiat_shamir, grinding_bits=64)` returns
  `lam + log2(2*max(1, n_checks)) + (grinding_bits if fiat_shamir else 0)`. Its docstring: "every op's Freivalds
  and column terms are each <= 2**-beta, so the union bound over n_checks ops is <= 2**-lam (2**-(lam +
  grinding_bits) under Fiat--Shamir)".
- `protocol.params_for` sets `bits = column_bits(lam, n_checks + (2 if cut else 0), ...)` and
  `reps = ceil(bits / LOG2_P)`, and each tree opens the least t with column term ≤ 2^-bits.
  - V1 adds two shares to the same union bound.
- `protocol.soundness_bits` reports −log2 of the **sum**.
- `setup_proof.setup_params` sets `beta = lam + grinding_bits + log2(2*len(publics))`, also a union bound.
- So the parameters use the **sum** in the per-query protocol, in V1 and in the setup proof.

**What the maximum would give [G, arithmetic].**

- Bounding Q·ε_rbr ≤ 2^{-λ} for Q ≤ 2^64 needs each per-round error ≤ 2^{-(λ+64)}, that is β = λ + 64 = 192 at
  λ = 128, against 200.8 today.
- r = ⌈192/30.907⌉ = 7, the same as ⌈200.8/30.907⌉ = 7. So the Freivalds vectors do not change.
- The column counts t_j would fall. The plan gives t_j = 51-194 at β = 200.8, that is 1.0-3.9 bits per column, so
  removing 8.8 bits removes roughly 2-8 columns per tree, about 4% of the opened columns.
  - This is an estimate only; the exact numbers need the table script of TH (b).
- The same holds for β_0 of the setup proof.
- A further observation, not for the paper without a hand check [G]: even the *interactive* bound of
  Thm. `thm:sound` could be max-based.
  - The first wrong operation, and hence its matrix j, is fixed by the first message.
  - The proof in app:sound therefore gives Pr[accept] ≤ p^{-r} + col_j ≤ p^{-r} + max_j col_j.
  - The sum over J is valid but not needed.

### 5.2 Text to insert

**(a) defence.tex, sec:fs, paragraph "Soundness."** Replace the paragraph with the version below. It is one sentence
longer than the current one.

```latex
\para{Soundness.} The interactive protocol is round-by-round sound~\cite{canetti2019fs}: once a
wrong claim is fixed, the vectors $\chi$ rescue the prover with probability at most $p^{-r}$, and
once a wrong folded row is fixed, the columns rescue it with probability at most its column term.
The round-by-round error $\varepsilon_{\mathrm{rbr}}$ is the largest of these probabilities, a
maximum rather than a sum, so $\varepsilon_{\mathrm{rbr}}\le\varepsilon_{\sC}$
(Appendix~\ref{app:sound}). A prover that makes $Q$ random-oracle queries therefore succeeds with
probability at most $Q\varepsilon_{\mathrm{rbr}}+O(Q^22^{-256})\le
Q\varepsilon_{\sC}+O(Q^22^{-256})$~\cite{canetti2019fs,holmgren2019rbr,bcs16}. Our parameters meet
the larger bound: we add 64 bits to $\beta$, which keeps $Q\varepsilon_{\sC}\le2^{-\lambda}$ for
$Q\le2^{64}$ and raises $r$ to 7 at $\lambda=128$. Meeting $Q\varepsilon_{\mathrm{rbr}}\le2^{-\lambda}$
alone would save the $\log_2(2|\mathcal J|)$ bits of the union bound in $\beta$ and leave $r=7$.
```

**(b) appendix_defence.tex, last sentence of "Round-by-round soundness."** Replace "The round-by-round error is
therefore at most ... follows from~\cite{canetti2019fs,bcs16}. \qed" with:

```latex
The round-by-round error is therefore
$\varepsilon_{\mathrm{rbr}}=\max\big(p^{-r},\max_j\prod_{i<t_j}(k_j-1-i)/(n_j-i)\big)$, a maximum
over the rounds and matrices, and a prover that makes $Q$ random-oracle queries succeeds with
probability at most $Q\varepsilon_{\mathrm{rbr}}+O(Q^22^{-256})$~\cite{canetti2019fs,bcs16}. Since
$\varepsilon_{\mathrm{rbr}}\le\varepsilon_{\sC}$, this implies the bound of Section~\ref{sec:fs},
which is the one the parameters of Section~\ref{sec:fs} and Table~\ref{tab:params} meet: their
$\beta$ keeps the $\log_2(2|\mathcal J|)$ bits of the union bound. \qed
```

**(c) appendix_defence.tex, "Setup." (optional, same point for β_0).** After "... so under Fiat--Shamir the bound
gains the factor $Q$~\cite{canetti2019fs,bcs16}." add:

```latex
Its round-by-round error is again the largest single term; $\beta_0$ budgets the sum of all
$2|\mathcal J|$ terms, which is conservative.
```

**(d) Rebuttal answer to B-Q6.** "No. The round-by-round error is the maximum, as Appendix B shows, but our parameters
meet the union bound Q ε_C. At λ = 128 this costs log2(2|J|) bits of β and leaves r = 7 unchanged; the maximum would
open a few percent fewer columns. We now state this in Sec. 4.4 and App. B."

---

## 6. B-W6: integer-multiplicity logUp against deployed small-field logUp

### 6.1 Verified facts

**The original lemma.** Haböck, "Multivariate lookups based on logarithmic derivatives", ePrint 2022/1530, version
of 31 Mar 2023, **Lemma 5 (Set inclusion)** [V]: "Let F be a field of characteristic p > N, and suppose that
(a_i)_{i=1}^N, (b_i)_{i=1}^N are arbitrary sequences of field elements. Then {a_i} ⊆ {b_i} ... if and only if there
exists a sequence (m_i) of field elements ... such that Σ 1/(X + a_i) = Σ m_i/(X + b_i)." The extra.bib entry
`haboeck2022logup` (marked [U]) is therefore correct in number and title; it can be tagged [V].

**S-two** (StarkWare), over M31 = F_{2^31−1} with QM31 challenges [V]:

- Source: S-two Whitepaper, Carmon, Goldberg, Haböck, Lerer, Lesokhin, Papini, Samocha, Cryptology ePrint Archive
  2026/532, PDF of 24 Mar 2026.
- Theorem 15 (IOPP Soundness, Sec. 3.5, p. 28) assumes "that the multiplicities use messages of the flat AIR are
  ensured to be less than the characteristic of F_q". The words are as extracted from the PDF.
- **Remark 17** (p. 29) "(Bounding multiplicities in S-two)": "... allow to deduce a bound for the number of calls of each
  lookup component, which is checked by the verifier. The sum of these table heights of the opcode components (the
  length of the main execution trace) ... is also checked to be below the characteristic bound."

**OpenVM**, over BabyBear with quartic-extension challenges [V]:

- Source: "On the Soundness of Interactions via LogUp", OpenVM authors, 13 Mar 2025,
  github.com/openvm-org/stark-backend/blob/main/docs/Soundness_of_Interactions_via_LogUp.pdf.
- Lemma 3.5 and Corollary 3.6 treat field multiplicities as integers when the sum on one side is < p. "Lookup tables.
  ... If the total number of lookups is less than p, the corollary guarantees that any message found in the
  sub-multiset given by I must also appear (with at least the same multiplicity) in the lookup-table multiset."
- OpenVM's security-model page (docs.openvm.dev/specs/security/security-model/) sets "max_interaction_count =
  2^31 − 2^27 + 1 (the BabyBear field modulus)".
- SWIRL (openvm.dev/swirl.pdf, Sec. 3.4) proves bus balancing with GKR LogUp over F-valued multisets.

**SP1**, over BabyBear [S]:

- Source: Lurk team, "Lookups in Lurk: Part 2", 17 Sep 2024, argument.xyz/blog/lookup-part-2/, quoted verbatim:
  - "the soundness of logUp is restricted to the case when the number of lookups is less than the order of the field |F|."
  - "This was an issue in previous versions of SP1's VM where it would've theoretically been possible to return an
    incorrect result to the CPU chip and then run the same query in a loop until it would effectively be nullified
    and not need to be proved. This issue was solved by including the CPU's clock as part of the query, thereby
    ensuring that queries occurring at different cycles would always be different."
- I did not read SP1's own code or documentation on this.

**Plonky3** (Polygon) [V for the feature, U for the bound]:

- The README lists "LogUp lookup arguments (local and cross-AIR)" over 31-bit fields with "~128 bit extension field".
- I did not find how Plonky3 bounds the number of lookups, so the paper should not make a claim about it.
- Plonky2's README says it is deprecated in favour of Plonky3. I found no primary statement on its lookups, so it is
  omitted.

**V1.**

- A query can have up to 4.3·10^9 leaves, more than p ≈ 2.0·10^9 (v1.tex).
- The table has 0.9-1.5·10^6 entries with random weights and about 2·10^4 with real ones (ext_v1.tex).
- The multiplicities are sent in the clear, and (D5) checks μ_τ < p and Σ_τ μ_τ = N_leaf in Z (v1.tex,
  appendix_v1.tex).
- The C2 doc (`docs/improvements/C2_v1_proof_size.md`, Sec. 4, item 3) records that "a proof can hold more than `p`
  leaves" and that both conditions were tested by brute force on F_49.

**How Lemma `lem:v1-logup` differs [G, compared with OpenVM's Lemma 3.5].**

- OpenVM needs the total on the lookup side below p, that is N < p.
- Our lemma bounds each table multiplicity below p and checks the integer total. The leaf side, where N may exceed p,
  is unrestricted.
- The verifier can check this only because it sees the multiplicities. A system that commits to its multiplicities
  would have to prove the range and the integer sum.
- I did not find the same lemma in the sources above, but that does not show it is new. The text below therefore
  makes no novelty claim.

### 6.2 Text to insert

**(a) ext_v1.tex, after the paragraph "Both integer conditions are needed."** Add this paragraph:

```latex
\para{Relation to deployed logUp.} The logUp lemma assumes a characteristic larger than the number
of looked-up values~\cite[Lemma~5]{haboeck2022logup}, and the systems over 31-bit fields that we
checked keep that number below $p$. S-two, over $\mathbb F_{2^{31}-1}$, lets its verifier check a
bound on the number of lookups derived from the table heights~\cite[Remark~17]{stwo2026};
OpenVM, over BabyBear, caps the number of interactions at $p$ and treats multiplicities as integers
when one side sums to less than $p$~\cite{openvm2025logup}; and SP1 includes the clock in every
query, so that repeated queries differ~\cite{lurk2024lookups}. A V1 query has up to
$4.3\cdot10^9>p$ leaves, which such a bound would force us to split into instances with separate
multiplicity vectors and separate checks. Lemma~\ref{lem:v1-logup} uses instead that V1 sends its
multiplicities in the clear: the verifier checks $\mu_\tau<p$ and $\sum_\tau\mu_\tau=\Nleaf$ over
the integers, at a cost of $|\Vtab|$ additions, and the number of leaves is not bounded. A system
that commits to its multiplicities would need a range proof and an integer-sum proof for the
committed column instead.
```

**(b) v1.tex, after "We therefore check two integer conditions."** This is an optional clause, if the body can
spend one line.

```latex
Deployed logUp over 31-bit fields instead bounds the number of lookups below
$p$~\cite[Remark~17]{stwo2026}, which V1 would exceed.
```

**(c) extra.bib, new entries:**

```bibtex
% [V] ePrint page and PDF of 24 Mar 2026 read 2026-10-10 (Sec. 3.5, Remark 17).
@misc{stwo2026,
  author       = {Carmon, Dan and Goldberg, Lior and Hab{\"o}ck, Ulrich and Lerer, Leonardo and Lesokhin, Ilya and Papini, Shahar and Samocha, Shahar},
  title        = {{S-two} Whitepaper},
  year         = {2026},
  howpublished = {Cryptology ePrint Archive, Paper 2026/532}
}

% [V] PDF dated 13 Mar 2025 read 2026-10-10 (Lemma 3.5, Corollary 3.6); interaction cap from
% docs.openvm.dev/specs/security/security-model/.
@misc{openvm2025logup,
  author       = {{OpenVM Contributors}},
  title        = {On the Soundness of Interactions via {LogUp}},
  year         = {2025},
  howpublished = {\url{https://github.com/openvm-org/stark-backend/blob/main/docs/Soundness_of_Interactions_via_LogUp.pdf}}
}

% [S] Secondary source for SP1's fix; blog post of 17 Sep 2024, read 2026-10-10.
@misc{lurk2024lookups,
  author       = {{Lurk Team}},
  title        = {Lookups in {Lurk}: Part 2},
  year         = {2024},
  howpublished = {\url{https://argument.xyz/blog/lookup-part-2/}}
}
```
And in `haboeck2022logup`, the comment can become: `% [V] ePrint 2022/1530, version of 31 Mar 2023; Lemma 5
(set inclusion) needs characteristic p > N.`

---

## 7. A-Q7: who tokenises the prompt, and provider-side system prompts

### 7.1 Facts from the paper and the code [V]

- The statement absorbs x: defence.tex sec:fs, "It then absorbs the security parameters, $C_M$, $x$ and ...".
  `protocol.py` calls `ch.absorb(b"x", _tensor_blob(x.cpu()))`.
- The verifier reads x in full. It recomputes every operation without weights, (S1), and checks the looked-up
  embedding rows that x selects.
- generation.tex defines a response to a prompt x and proves s = x ‖ y_1 ... y_{G−1}.
- `grep tokeni` and `grep "system prompt"` find nothing in sections/ [V, as the plan says].

**Consequence [G].** The proof binds token ids, not text.

- A provider cannot hide a system prompt from a client that verifies: any prefix it adds is part of x, and x is in
  the statement.
- Tokenisation is deterministic, but detokenisation is not injective: several id sequences can give the same text. A
  client that receives ids from the provider must therefore re-tokenise its text and compare, not detokenise.

### 7.2 Text to insert

**Main option: attack.tex, sec:threat, first paragraph.** After "For each query $x$ the provider (the prover)
returns an output $y$ and a proof." insert one sentence:

```latex
For a language model, $x$ is a sequence of token ids, which the client computes from its text with
the model's public tokenizer and chat template (or recomputes and compares, if the provider
tokenises); a system prompt that the provider adds is part of $x$, which the verifier reads in
full, so a proof shows $y=M(x)$ for the prompt the client sees, and whether to accept such a prefix
is the client's decision.
```

**Alternative (defence-specific): defence.tex, sec:fs, paragraph "The statement binds the computation."** Replace
"It then absorbs the security parameters, $C_M$, $x$ and, ..." with:

```latex
It then absorbs the security parameters, $C_M$, the token ids $x$ (any provider-side system prompt
included, so the client sees it) and, ...
```

---

## 8. Open points

1. **FOCS and J. ACM numbering of BCIKS20.** I could not reach IEEE Xplore or dl.acm.org (403 or Cloudflare). The
   text above cites the full version (ePrint 2020/654, 3 Jul 2021) explicitly in the bib note, so it does not depend
   on them. Someone with library access may check the FOCS camera-ready, and drop the note if FOCS also says
   Thm. 1.6.
2. **Radii chosen for Llama-2-7B.** Which f ∈ {1, 1/2, ...} `setup_params` picks for each tree is [U].
   - If every tree has f ≤ 1/2, Ligero's Lemma 4.2 alone suffices and BCIKS20 is needed only for the general
     statement.
   - The per-query growth of 1.35x / 1.08x in the D4 doc (`docs/improvements/D4_D5_untrusted_commitment.md`, Sec. 5)
     suggests small f, but this is only an inference [G].
   - The TH (b) table script should print f per tree.
3. **β and |J| may not match.**
   - The paper writes β = λ + log2(2|J|). The code calls `params_for(lam, len(ops))`, so n_checks is the number of
     weight operations, at least |J|.
   - The plan's values β = 136.8 / 200.8 imply log2(2·n_checks) ∈ [8.75, 8.85), that is n_checks ∈ [216, 230].
     This fits 225 = 32·7 + 1, the number of Llama-2-7B's weight operations [G]. The plan's "130 matrices" would
     give β = 136.0.
   - Either the table's |J| row must say L = 225 weight operations with "β = λ + log2(2L), L ≥ |J|", or the script
     must use |J|. This is conservative, not a soundness bug, but tab:params must agree with the code (TH (b), H3).
4. **SaTML version of Anchuri et al.** I could not check whether the proceedings version (pp. 955-977) contains App.
   D, the p. 7 sentence or the same appendix letters. Until someone checks the IEEE Xplore PDF, the bib note in 3.2(a)
   is needed.
5. **SP1 source.** The SP1 statement relies on the Lurk team's blog [S]. If a reviewer may check it, replace it with
   SP1's own documentation or code, or drop the SP1 clause; the S-two and OpenVM citations carry the paragraph alone.
6. **Interactive bound could also be a max [G].** Thm. `thm:sound` sums over J, but the first-wrong-operation proof
   gives p^{-r} + max_j col_j. I did not propose changing the theorem, because the tables and code use the sum and
   the TH (f) hand check is the place to decide. This is mentioned only in Sec. 5.1.
7. **Wording of "state-restoration".** appendix_v1.tex says "state-restoration bound" while sec:fs speaks of
   round-by-round soundness. Block 4.2(b) drops the term. Holmgren (ePrint 2019/1261) proves the two notions
   equivalent [V, abstract and Thm. 1.1, with the easy direction attributed to CCH+19], so either term is correct.
8. **Downloads.** The PDFs are in `scratchpad/th_d_downloads/`, outside the repository. They are not needed by the
   paper and can be deleted.

## 9. Sources

- BCIKS20: ECCC TR20-083 and revisions, https://eccc.weizmann.ac.il/report/2020/083/ ; ePrint 2020/654,
  https://eprint.iacr.org/2020/654 (PDF via the Wayback Machine); FOCS DOI 10.1109/FOCS46700.2020.00088 and J. ACM DOI
  10.1145/3614423 (metadata from api.crossref.org).
- Ligero, CCS 2017: https://acmccs.github.io/papers/p2087-amesA.pdf ; journal version DOI 10.1007/s10623-023-01222-8
  (not opened).
- Anchuri et al.: https://arxiv.org/abs/2603.19025 (v1, PDF), https://eprint.iacr.org/2026/541 (abstract page).
- Haböck, logUp: https://eprint.iacr.org/2022/1530 (PDF of 31 Mar 2023 via the Wayback Machine).
- Holmgren, round-by-round soundness: https://eprint.iacr.org/2019/1261 (PDF via the Wayback Machine).
- S-two Whitepaper: https://eprint.iacr.org/2026/532 (PDF of 24 Mar 2026).
- OpenVM: https://github.com/openvm-org/stark-backend/blob/main/docs/Soundness_of_Interactions_via_LogUp.pdf ;
  https://docs.openvm.dev/specs/security/security-model/ ; https://openvm.dev/swirl.pdf
- Lurk team blog (SP1): https://argument.xyz/blog/lookup-part-2/
- Plonky3 README: https://github.com/Plonky3/Plonky3 ; Plonky2 README: https://github.com/0xPolygonZero/plonky2
- Repository: `code/src/pvi/fullcheck/plans.py` (`column_bits`), `protocol.py` (`params_for`, `soundness_bits`,
  transcript order), `setup_proof.py` (`setup_params`, `FRACTIONS`); `docs/improvements/D4_D5_untrusted_commitment.md`;
  `docs/improvements/C2_v1_proof_size.md`; `sp2027_research/attack_theory/sok.md` and `papers/anchuri.pdf`.
