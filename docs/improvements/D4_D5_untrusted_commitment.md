# D4 + D5. An untrusted weight commitment: a one-time setup proximity proof, and the attack it stops

Plan items D4 and D5 (reviewer objection D6: "honest commitment assumed, no proximity test"; the paper's
limitation (iv)). Commit `af2ccf8` (`code/src/pvi/fullcheck/setup_proof.py`, `code/tests/test_setup_proof.py`);
byte model `code/experiments/6_improvements/setup_bytes.py`. Analysis: the trust report
(`S/sp2027_research/idea_trust-rigor.md`, Sections 1.2-1.4). Status: done and tested on the CPU (8 tests).

## 1. The problem: the split commitment (D5)

Theorem 3.2 assumes the commitment `C_M` is computed honestly: under each Merkle root, the columns are those of
the Reed-Solomon encodings of the weight rows. A malicious committer (for example the provider itself) can
instead bind any matrix `E`. The concrete attack: in the tree holding a matrix `j`, bind model `A`'s encoded
columns on a secret half of the positions and model `A'`'s (for example a backdoored `A_j`) on the other half.
A prover that computes with `A'` then passes Freivalds' check (it sends `u = chi^T A'`) and passes the column
check exactly when all `t` opened columns land in `A'`'s half:

    Pr[accept] = prod_{i<t} (n/2 - i) / (n - i) ~ 2^-t,

which for Llama-2-7B's smallest tree (`t = 35`) is about `2^-35` per query, not `2^-128`.
`tests/test_setup_proof.py::test_without_a_setup_proof_the_split_commitment_passes_about_two_to_the_minus_t_of_the_queries`
builds exactly this commitment (one weight row changed in one op of a tiny decoder) and measures the
acceptance rate over 300 queries with a small `t`: it matches `prod (n/2 - i)/(n - i)`.

## 2. The fix: a one-time public proximity proof (D4)

Once per model, published with `C_M` and checked by any client once, a non-interactive proof (Fiat-Shamir over
the graph's digest, see D1, and every tree's root and shape) shows that every committed matrix `E_j` (`R_j`
encoded rows, message length `m_j`, `n_j` points) is `e_j`-close to a matrix of codewords: it differs from
`Enc(A_j*)` on at most `e_j` columns, for a unique `A_j*` (`e_j < (n_j - m_j + 1)/2`).

1. Per member matrix, `s` uniform base-field combinations `psi_j` (`s x R_j`) of its rows.
2. The committer sends `psi_j A_j*` (`s x m_j`).
3. Per tree, `t0` uniform columns; the committer opens them with one multiproof.
4. The verifier checks the multiproof and, for every member, `Enc(psi_j A_j*)[c] = psi_j E_j[:, c]`.

**Soundness.**
* The `s` base-field rows checked at the same columns are one uniform combination over `K = F_{p^s}`
  (Fact 1 of the trust report), so BabyBear's 31 bits are not a problem: the proximity-gap error is
  `n/p^s`, and `s = 7` gives `2^-195` or less at `n <= 2^18`.
* If `E_j` is not `e_j`-close, the combination is `e_j`-far from the code except with probability `n/p^s`
  (correlated agreement in the unique-decoding regime: Ben-Sasson et al. 2020, Thm. 1.6; Ligero's
  Lemma 4.2 for `e < d/4`), and then `t0` distinct uniform columns all miss the disagreement with
  probability at most `prod_{i<t0} (n - e - 1 - i)/(n - i)`.
* With Fiat-Shamir both terms are multiplied by the number of random-oracle queries (`2^64` grinding bits);
  `s` and `t0` make the sum over all `2J` terms at most `2^-lambda`.

**Per query after the setup proof.** The protocol is unchanged except for the column counts. With every
`E_j` `e_j`-close, a wrong folded row escapes the column check of matrix `j` only if all `t` columns land among
the at most `m_j - 1 + e_j` positions where the codewords agree or `E_j` is corrupted:
`prod_{i<t} (m_j - 1 + e_j - i)/(n_j - i)`, the honest bound with `m_j` replaced by `m_j + e_j`.
(The committed model is then `A*`, evaluated with field arithmetic; when `A*` is not int8, see the trust
report's remark on field semantics.)

**The trade-off.** A larger radius `e` shortens the setup proof (each setup column catches a far matrix
with probability about `e/n`) and lengthens every query. `setup_params` chooses each tree's radius among
fractions `2^-j` of the unique-decoding radius to minimise `t0/amortise + t` (its columns' bytes over
`amortise` queries; default 1,000).

## 3. Implementation

* `setup_proof.setup_params(publics, groups, lam, query_bits, amortise=1000)`: `s`, the radii, `t0` per tree
  and the per-query `t` per tree. `survivors(a, n, bits)` is the linear-time column search (the generic
  `plans.exact_columns` is quadratic for the thousands of columns a setup proof opens).
* `setup_proof.prove_setup(groups, publics, graph_digest, params)`: the committer's proof, reusing
  `WeightCommitment.fold` (row layout), a transposed analogue for col-layout members, and `GroupCommitment.open`.
* `setup_proof.verify_setup(publics, groups, graph_digest, params, proof)`: shapes, field ranges, the code check
  (`codeword_at` against `field_matmul_mod(psi, E)`), column digests, group leaves and the multiproof.
* `setup_proof.query_group_columns(params, groups)`: the `SecurityParams.group_columns` for the queries.
* Scope: commitment plans (shared trees, the optimised protocol). The basic protocol's per-op trees would
  need the same check with `column_leaf` leaves.

## 4. Tests (`tests/test_setup_proof.py`, 8 tests)

* Honest plans of four tiny decoders (row- and col-layout members, three policies) pass; the per-query `t`
  is at least the honest bound's.
* A changed message entry, opened column entry, missing multiproof hash, missing message, another graph's
  digest or another root fails.
* The split commitment passes about `2^-t` of the queries without the setup proof (measured), and the
  committer's setup proof is rejected.
* `s`, `t0` and `t` meet their bounds (unique decoding, setup error, per-query error).

## 5. Size on the real models (`setup_bytes.py`, lambda = 128, interactive queries, auto plans)

| Model | amortised over | setup proof | per-query column bytes, honest -> untrusted |
|---|---|---|---|
| GPT-2 | 100 / 1,000 queries | 213 / 686 MB | 3.22 -> 6.84 / 4.63 MB (x2.13 / x1.44) |
| Llama-2-7B | 100 / 1,000 queries | 2.8 / 11.3 GB | 124.7 -> 168.5 / 134.9 MB (x1.35 / x1.08) |
| OPT-6.7B | 100 / 1,000 queries | 2.5 / 10.2 GB | 85.9 -> 130.7 / 97.1 MB (x1.52 / x1.13) |
| Llama-2-13B | 100 / 1,000 queries | 4.0 / 16.4 GB | 149.3 -> 217.4 / 165.9 MB (x1.46 / x1.11) |

At 2,048 tokens, where the claims are GBs, the per-query change is below 1%.

**Option C (open weights).** When the weights are public (the setting C targets, since the claims reveal
the weights anyway, see D7), anyone can recompute `C_M` from a published int8 checkpoint and its hash, and
the honest-commitment case holds by verification; the setup proof is for commitments made by a party
whose weights the client cannot see.

## 6. What remains

* The theorem and proof in the paper's appendix (statement in the trust report, Section 1.4).
* Timing the setup proof on the GPU for the 7B models (setup prover: about 20-100x one query's opening).
* `bench.py --untrusted` cells (per-query cost with the untrusted column counts).
