# TH-e check: `drafts/th_e_proofs.tex`

Checked on 2026-10-11 against `paper_sp2027` (main.tex, sections/*.tex), `SP2027_PLAN2.md` (TH, Sec. 3.7) and
`code/src/pvi/fullcheck` (protocol.py, cut_protocol.py, claimcodec.py, token_sampler.py, transformer.py,
logup_gkr.py, logup.py, extfield.py, graph.py, plans.py, sampling.py) and `code/tests/test_fs_tree.py`.

**Note on the code state.** `transformer.py` was modified at 2026-10-11 01:12, while this check was running:
`build_decoder(prob_max=...)` and `PROB_MAX = 255` (plan F2a). The pruning structure is unchanged. Only the
attention constant 510 = 2 x 255 in lem:prune-local is affected, and the draft now names it. The other files were
last modified on 10 Oct.

Every change in the draft is marked with a LaTeX comment `% CHECK:`. There are 22 such lines. A comment marked
`% CHECK (suggestion, no text change)` is a proposal for the authors and leaves the text as it was. No other file
was edited.

## Verdicts

| Subsection | Verdict | Main point |
|---|---|---|
| Conventions (app:the-conv) | holds | Matches `Challenger` (`_key`, `folding`, `ext`, `columns`). The R1-R3 theorem numbers could not be checked (see below). |
| (i) Last-block pruning (app:prune) | holds with fix | Three small fixes: the final norm, the (P2) edge case n = T, and the 2 P_max constant. |
| (iii) Token rule and sampler (app:gen) | holds with fix | thm:gen gave the v0 Kpre bound for V1 queries too. |
| (vii) FS tree absorption, Kpre absorbs nothing (app:fstree) | holds with fix | The component bookkeeping in thm:fstree's proof had a gap (now closed, bound unchanged). Two notation fixes. V1 per-round error 1/p^8 → 2/p^8 (the maximum is unchanged). |
| (ii) Claim codec (app:codec) | holds | In every case the verifier absorbs the same buffer that it decodes. |
| B-Q5 / B-Q6 (app:rbr) | holds with fix | The doomed set after the folds lacked (F3)/(F4). The "about 2^-127" wording was fixed. Sum vs max: see the next section. |
| (iv) V1 in Kpre (app:v1kpre) | holds with fix | Fixed a LaTeX error (5 places). cor:v1kpre-fs applied (R2) to Q_V proofs; the argument is now stated. A tighter bound is available (authors decide). |
| (v) Stateful Kpre verifier (app:stateful) | holds | All numbers recomputed and correct. They would change under the tighter bound. |
| (vi) Path-protocol acceptance (app:paths) | holds with fix | "opens" → "checks" (as worded the hypothesis was false for RandPathTest). Caption made precise. All 55 table values match. |

## The known point: sum against maximum

- **Mode C, interactive and Fiat-Shamir: handled correctly.**
  - rem:rbr-max states (R1) as p^-r + max_j prod_j. cor:fs-v0 uses eps_rbr = max(p^-r, max_j prod_j) <= eps_C.
  - It keeps thm:sound's sum as an upper bound and says the parameters do not use the maximum. This answers B-Q6 as the plan asks.
  - I re-derived it. The first locally incorrect operation l* and its matrix j* are fixed by the claims, before chi. The chi round is escaped only via chi_{l*}. Every matrix with a wrong u must pass the column check by itself, so the column round errs with at most max_j prod_j.
  - (Even p^-r + prod_{j*} holds.)
- **Mode K: not addressed.** thm:sound's eps_K = sum_l p^-r is also a sum. Interactively the fresh chi of one query is drawn after all claims, so one query errs with at most p^-r. I added a suggestion comment in rem:rbr-max.
- **Kpre, reuse over Q verdicts: the draft's bound Q sum_l p^-r is valid, but the sum is not needed.**
  - The argument: thm:v1kpre's hybrid H never uses chi. The verifier evaluates the secret checks in a fixed order and reports only the first failure (or one label for a group: `freivalds`, `kpre_y`). So the real run and H first differ in query i only if the first evaluated secret check with E_c != 0 passes.
  - That check is a function of H's view, so this has probability at most p^-r. Each false acceptance of the real run is either such a divergence or a false acceptance of H.
  - So the theorem holds with Q p^-r + sum_i (eps_lu_i + eps_gkr_i).
  - The consequences:
    - prop:qtight would then match the upper bound to second order.
    - rem:stateful (a) at r = 5 becomes Q_V <= 2^26.5 (about 9.7e7) instead of 2^18.7 (4.3e5). At r = 7 it becomes 2^88.3 instead of 2^80.5.
    - Open point 3 improves.
    - The body's thm:sound ("Q eps_K") and thm:v1 would change if adopted.
  - The tightening needs the verdict convention of app:the-conv (first failed check only). It fails if the verifier reported all failing checks.
  - Added as suggestion comments at thm:v1kpre and rem:stateful (a). **Authors decide.**
- **V1, thm:v1:** the GKR sum over gamma could likewise be a maximum (the prover must make every instance with a wrong root escape). This is numerically irrelevant, and the draft only cites thm:v1.

## Order of messages and challenges (verified in the code)

| Protocol | What is absorbed before each challenge (FS), or sent before it (interactive) | What the prover chooses after seeing a challenge |
|---|---|---|
| v0 C | Before chi: statement (graph digest, params+mode, op/col/group/table records, lookups, x, sampler), claims (`claims/PVC3` or `claim/*`), `rows/I8`, `lookup/*` (`_claims_message`). Before cols: u (`u/F31` or `u/*`, `_open_message`). | u after chi; openings after cols. Both are covered (Freivalds/column rounds; Merkle binding). |
| v0 K | Statement and claims before chi. | Nothing (the verifier folds). |
| v0 Kpre | Nothing absorbed (`fiat_shamir and mode != "Kpre"`), and no challenge exists. | The claims, without any information on chi (verified: `_fold_message` returns `_pre`). |
| V1 C | Before alpha and chi_clear (same transcript string): statement, `cut` plan digest, claims, rows, lookups, `cut/exc`, `cut/mult`. GKR: each `cut/b/top`, `g k.j`, `v k` before the next challenge (`logup_gkr.verify`). Before cols: `u/F31` = fold_blob (`_open_message(u_blob=fold_blob)`). | GKR messages after alpha/earlier rounds; folds (u, y) after the GKR's final point; openings after cols. |
| V1 Kpre | As V1 C up to the end of the GKR; no challenge after it. (`u/F31` is absorbed after the folds although no challenge follows: harmless.) | The folds y after the GKR point. They are checked by (F2) with the secret chi, which is independent of everything public, and by (F3) with the public eq-vector. |

**V1 in Kpre absorbs everything that its public-coin challenges depend on.** Alpha depends on M1, which is absorbed
completely. The GKR challenges depend on the GKR messages, which the verifier absorbs as decoded. There are no
further public challenges. This matches prop:kpre-fs, the remark after it, and cor:v1kpre-fs.

## Numbers recomputed with Python

- **Field and parameters.**
  - p = 2013265921 = 15 * 2^27 + 1, so log2 p = 30.9069.
  - beta = 128 + log2(452) = 136.82, giving r = 5. Under Fiat-Shamir beta = 200.82, giving r = 7.
  - 226 p^-5 = 2^-146.714 and 226 p^-7 = 2^-208.528. So Q_V <= 2^18.71 (4.3e5) and Q_V <= 2^80.53.
  - beta = 193 gives r = 7; t_j drops by 3.9%.
  - 3/p^8 = 2^-245.67. 7 log2 p = 216.35. p - 2^30 - 2^16 = 939,458,561.
- **Fiat-Shamir framing.**
  - "ch" = 25,448. The domain string is 16 bytes.
  - A 6.6e9-byte message has 1,574 leaves (1,690 at 6.6 GiB), about 50 kB absorbed.
- **Sampler.**
  - Gamma[0] = -161,658 and Gamma[65535] = 772,243 (< 2^20). The table is non-decreasing, and its SHA-256 equals `_TABLE_SHA256`.
  - (2^24 - 1)(2^29 - 1) + 2^20 < 2^54.
- **tab:paths.** All 25 values equal `ceil(40 ln 2 / -log1p(-k/N))`. The bounds (N/k - 1) lambda ln 2 <= s <= ceil(N/k lambda ln 2) hold for each. 27.73 N checks out; 2,316 and 14,182 (attack.tex) are s_40(1, 84) and s_40(1, 512).
- **tab:paths-acc.** All 30 values match (1 - k/N)^s at the printed precision. The two "<" cells are 4.3e-238 and 4.6e-160.
- **Header numbers.** Every number in the header's "Numbers" paragraph matches.

## Changes made in the draft (all marked `% CHECK:`)

1. **thm:v1kpre and its proof: LaTeX error.** `\\varepsilon` (a forced line break, then the letters "varepsilon") replaced by `\varepsilon` in 5 places.
2. **thm:fstree, notation.** Q_V renamed Q_ver (oracle queries of the verifier). app:the-conv and cor:v1kpre-fs use Q_V for verdicts.
3. **thm:fstree proof, a gap closed.** "Component" was defined by a query's role. A query K||label whose key K begins like a transcript is both an expansion input and a challenge input, so "at most one first unresolved component per query" was not justified.
   - The new definition uses the fixed 32-byte prefix sigma_0 of every challenge input: the domain string, the framing of the first record `graph` (absorbed first in v0 and V1), and the digest's first byte.
   - It adds E_0 (some output equals sigma_0, probability N 2^-256) and counts N(N-1)/2 pairs.
   - The total N^2 2^-256 is unchanged. The later uses of the events were updated.
4. **cor:fs-v0.** The V1 per-round error 1/p^8 became 2/p^8. `logup_gkr.verify` draws theta_{k-1} (mu) and eta_k (lam) from one transcript state, and thm:fstree counts one transcript string as one round. The maximum (the alpha term) is unchanged.
5. **rem:rbr-max.** "The total at Q = 2^64 is about 2^-127" holds only with beta = lambda + 65. With the parameters kept it is 2^-128 + 2^-136.8. The header's open point 2 was reworded the same way. A suggestion was added for mode K and Kpre.
6. **Question 5 paragraph.** The doomed set after the folds now includes a failed column-independent check: Freivalds of a clear op, (F3) or (F4). Without it the set is not closed under the fold message: after a doomed GKR the prover can send true folds, and then (F3) rejects.
7. **cor:v1kpre-fs proof.** (R2) concerns one final proof. The extension to Q_V proofs, a union bound over moves, is now stated. A team member should check it against the book's proof.
8. **thm:gen.** "In Kpre ... at most Q_V eps_K (thm:v1kpre)" is now Q_V eps_K for v0 queries, and the thm:v1kpre bound (with its logUp and GKR terms) in general.
9. **thm:prune (a).** "and the final norm" added to the tensors that the pruned graph slices.
10. **Note after thm:prune.** "Single-position operations violate (P2)" became "operations that claim n < T columns". `cut.py` tests `cols == T`, and with a one-token prompt n = T.
11. **lem:prune-local.** 510 became 2 P_max with the default P_max = 255, following the new `prob_max`.
12. **prop:paths.** "opens" became "checks" in the intro, the statement and the proof.
    - In RandPathTest (`sampling.py`), checking a node opens its parents' rows. For a dense layer that is the whole layer, so every site is opened.
    - Only the visited node is checked.
13. **tab:paths caption.**
    - k = 16 is now labelled illustrative.
    - "Largest in-range edit" now says moves in either direction. 60 is the maximum of the "any" rows of G3_inrange_attack.md; the lowering-only edits, also in range, reach 104 and 164 neurons.
14. **rem:stateful (a).** Comments only: the numbers were verified, and the tighter-bound alternative is given.

## Checked and found consistent with the code (no change)

- **Transcript framing.** `absorb` writes 2-byte and 8-byte big-endian lengths. Leaves are `pvi/leaf/v1 || len(L) || L || j_8 || chunk`. The tree record is `L/tree` with `>QQQ (c, 2^22, n)` followed by the leaves. The threshold is 2^24 and applies only under Fiat-Shamir. The key is `state || challenge/ || label`.
- **Labels (L1)/(L2).** The label lists in the draft are complete. The `absorb_large` labels are `claims/PVC3`, `rows/I8`, `u/F31`, `cut/exc` and `cut/mult`. Real-weight op names are short (`mm<n>`).
- **Expansions.**
  - Field words below 2p are reduced mod p.
  - `ext(nonbase)` rejects F_p + xi F_p.
  - Columns use 64-bit words below 2^64 - (2^64 mod n), with repeats dropped.
- **Kpre.**
  - `_claims_message` absorbs nothing in Kpre.
  - `precompute` takes one OS key per `_row_ops()` op, so n_key is the number of ops not looked up.
  - (F2) is `_kpre_y_ok` with `_pre`.
  - The check order is `range_or_shape`, `token`, lookups, `cut_exception`, `cut_multiplicities`, `cut_gkr`, `freivalds`, `kpre_y`, `cut_final`, `cut_logup`.
- **Sampler and token rule.**
  - The digest is `pvi/sampler/v1 || <II a,k || seed`. The noise is `SHAKE(pvi/sampler/v1 || seed || <Q pos)` read as 16-bit LE words.
  - Masked candidates get -2^63. Candidates come from a stable sort of -z, and the choice is the first argmax.
  - The positions in `check_tokens` are t - n + j = P - 1 + j, and `sample_tokens` uses P + i - 1.
  - The graph digest includes `meta` (lm_positions) and every op's fn and params.
  - The token check runs before `derive`'s range check.
- **Pruning.**
  - q is computed from `last(h)`. RoPE uses offset `x.shape[1] - a.shape[1]`, and `_causal_notmask(queries=Tq)`.
  - The residual uses `last(r)`, and the final `last` acts as the identity on n positions.
  - `calibrate_all` keeps every position, so the multipliers and the weight draws are equal and only the names shift.
  - `claim_bound` depends only on the weights, the bias and `max_input`.
- **Codec.**
  - The rejection rules match the `claimcodec` docstring. `extfield.unpack` rejects coefficients >= p.
  - `unpack_exceptions` returns None on bad framing.
  - The u and openings field checks are `_disagrees` and `_columns_shape_ok`, and the V1 folds are checked with `_in_field`.
  - The absorbed bytes are exactly the decoded bytes: `claims` and `rows` (`_run_query`, `_query`), `blob` as `u_blob`, `exc_blob`, `mult_blob`, and `fold_blob`. The GKR values are absorbed as parsed.
- **Parameters.** `column_bits` = lambda + log2(2 n_checks) (+64 under Fiat-Shamir), with n_checks + 2 for V1, a negligible difference. `soundness_bits` is the sum form.

## For the authors to decide or check by hand (TH-f)

1. **Tighter Kpre reuse bound** (Q p^-r instead of Q sum_l p^-r) and **mode K** (p^-r per query). See the section on the known point. Adopting them changes the body's thm:sound and thm:v1 wording, thm:v1kpre, rem:stateful (a) and open point 3.
2. **Chiesa-Yogev, unverified here.** The draft's own [U] item stands. Check three things:
   - the theorem numbers (Claim 31.1.3, Thm. 31.2.1, Thm. 14.3.1) and the exact form (t + k) eps_rbr of (R2);
   - **whether (R2)/(R3) cover adaptively chosen statements.** thm:fstree ("outputs a statement and a proof"), thm:gen (the prover chooses s and the tokens) and cor:v1kpre-fs rely on this;
   - whether (R2)'s proof is a union bound over moves (change 7).
   
   The Canetti et al. and Holmgren statements also remain unverified (HTTP 403 in the draft's log; not retried).
3. **Collision term at Q = 2^64** (open point 2). The Fiat-Shamir total is 2^-127.997 with the kept parameters. Either restrict to Q <= 2^63 or budget one more bit. Note that tab:params says "<= 2^-128".
4. **Measured k.**
   - The pooled lowering-only median of 6 neurons (tab:paths caption and attack.tex) is not in G3_inrange_attack.md's table, which gives per-layer medians 3, 12, 8 and 2. Confirm it from the raw results.
   - Decide whether to add a column for the largest lowering-only edit (164).
5. **Timing (rem:stateful (c)).** The draft says "we have not checked". It can say more:
   - The CPU path is not constant-time. `_disagrees` stops at the first failing op, using `torch.equal`, and `_kpre_y_ok` returns at the first mismatch.
   - Stopping at the first failing op reveals only what an op-level label would, which the hybrid covers.
   - The early exit inside `torch.equal` depends on the compared values, hence on chi. That leak is outside the proof.
6. **Code changed during the check (`prob_max`).** Before the freeze, re-run `tests/test_pruning.py` and `tests/test_generation.py` with the chosen P_max. P_max enters the digest only when it differs from 255 (`fn` keeps the old lambda for the default).
7. **Unchanged open points** from the draft's header: 1 (body Sec. sec:fs "Q eps_C"), 4, 5 (`real_weights.py` has no pruned builder), 6, 7 (two new BibTeX entries, still missing from references.bib and extra.bib) and 8.
