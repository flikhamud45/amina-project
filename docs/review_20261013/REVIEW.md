# Review of code, paper and decisions (13 Oct 2026)

Eight read-only reviewers (one per area) and one independent verifier per reviewer, which checked every high- and medium-severity finding against the code and the paper. The inventory the reviewers started from is in `inventory/`. Verdicts: confirmed, partly (the facts hold but the conclusion or severity was adjusted), refuted (dropped below). `#` marks findings that change reported numbers.

## Commitments and arithmetic

Area: field, extfield, commitment, plans, setup_proof, setup_cache and analytic, plus their tests and the three scripts that produce tab:params and tab:setup. I read the code, did light Python checks, and ran the analytic scripts params_table.py and setup_bytes.py. I built no models and ran no tests.

The core cryptography holds up, and I found no wrong result in the code.
- **Merkle trees:** inner nodes and leaves use separate domain tags (pvi/node against pvi/col, group, row and pad). The depth comes from the public n_points. The multiproof's hash order matches between prover and verifier, and extra or missing proof elements fail the check.
- **Leaf encodings:** the verifier range-checks every input a leaf hashes before hashing it: opened columns to [0,p) (protocol.py:1356), lookup claims to int8 (:1034) and ids to the table (:1039). The leaf encodings are therefore injective on every input the verifier accepts.
- **Column draws:** Challenger.columns draws a uniform set of t distinct indices with no bias (protocol.py:450). That is the sampling the bound in column_error_log2 assumes.
- **Codes and field arithmetic:** n >= k is enforced (field.py:156, plans.py:295). The float64 blocks stay at or below 2^53, the int64 sums below 2^62, and the int8 GEMMs at or below 2^30. I checked that x^8-11 is irreducible: 11 is a non-residue mod p, p is 1 mod 4 and 8 divides p-1.
- **tab:params reproduces exactly** with `params_table.py --policy cnn18c`: 226 operations; beta 136.82 / 200.82; r 5 / 7; t 23 / 34 (largest 46 / 67); 65.6 / 96.5 MB of columns; beta0 200.02; s = 8; f = 2^-2 / 2^-4; t0 1,054-1,190 / 4,399-4,936.
- **tab:setup reproduces** with setup_bytes.py: 2,382.0 / 9,758.7 MB; 113.49 / 79.40 MB of columns per query; combined rows 2.2% / 0.5%.
- **Setup-proof parameters:** setup_params follows the rule of Theorem thm:untrusted (s, the radius e, t0, the per-query t). For all four tab:setup models the radii are at most a quarter of the unique-decoding radius, so appendix_defence.tex:386-391 holds under cnn18c.
- **Setup cache key:** it covers the base rate, the plan's layout and every operation's weights and bias.
- **Safe to move:** none of these modules defines a cheap-op function, so moving them does not change graph digests (unlike hazard H1). The hash domain strings, the "setup/..." transcript labels and the setup-cache VERSION must stay byte-identical.

What needs work:
1. **(high)** The extended version's setup-proof text still gives the numbers of the older plan. It contradicts tab:setup in the same section.
2. **(medium)** verify_setup accepts whatever SetupParams the caller passes. The online half of thm:untrusted (queries that open the larger t) has no code path.
3. **(medium)** The setup-cache docstring claims that a wrong leaf can never make a verifier accept. That is not true: a cache entry decides what the root binds. Only 4 leaves per tree are spot-checked, and no run record stores the commitment's roots.
4. **(medium)** The documented commands and defaults do not reproduce tab:params or tab:setup: `--policy auto` resolves to cnn16c for Llama-2-7B. The footnote credits params_table.py with rows it does not print.
5. **(low)** Cleanup:
   - duplicated logic: survivors equals exact_columns, and the non-claim bytes are computed three times;
   - dead or opt-in code: Merkle worker processes, test-only extfield functions;
   - stale course-report docstrings;
   - verify_multiproof does not check its proof elements;
   - a proposed commit/ and arith/ layout.

### CA-1 [high, confirmed] # The extended version's setup-proof numbers come from the pre-cnn18c plan and contradict tab:setup and the code

**Where:** paper_sp2027/sections/ext_defence.tex:103-108,121-122,150-166,176-177; paper_sp2027/sections/appendix_defence.tex:175-177; code/experiments/4_defence_benchmark/slurm/sp2027.sh:356

**Evidence.** Recomputed values for Llama-2-7B under cnn18c, at lambda=128 with interactive queries:
- `params_table.py --policy cnn18c`: t_min = 23 / 34.
- `setup_bytes.py --policy cnn18c --amortise 100 1000`:
  - a=100: setup_MB 2382.0, of which columns 2326.5, combined rows 51.6 (2.2%) and paths at most 3.8. Query factor 1.731.
  - a=1000: setup_MB 9758.7, of which columns 9691.2, combined rows 51.6 (0.53%) and paths at most 15.9. Query factor 1.211.
  - The largest t0 is 4,936 of n = 2^18, i.e. 1.9% of n.

The extended text still says:
- t=35 and about 2^-35 (ext_defence.tex:103, appendix_defence.tex:175);
- "11.3 GB once and 8% more column bytes per query" at a=1,000 (ext_defence.tex:121). The code gives 9.8 GB and +21%.

The paragraph at ext_defence.tex:150-166 also has stale values:
- 2,833.5 / 2,784.0 / 45.2 / 4.3 / 11,272.7 MB;
- "1.6% / 0.4%", where the code gives 2.2% / 0.5%;
- "t0_j reaches 12% of n_j", where the code gives 1.9%.
It sits directly under tab:setup, which says 2.4 / 9.8 GB, and line 176 repeats "1.6%".

"3.2-6.6x the column bytes" (ext_defence.tex:108, appendix_defence.tex:177) also comes from the old plan, from a note outside the repository. I read it as the least t with eq:split <= 2^-beta per tree. On that reading cnn18c gives 4.1-6.5x (GPT-2 6.5, Llama-2-7B 4.45, OPT-6.7B 4.4, Llama-2-13B 4.09), and no script in the repository computes it.

The sp2027.sh:356 comment says "2.8 GB". The body (defence.tex:207-232) and tab:params / tab:setup are correct.

**Recommendation.** Regenerate every setup-proof sentence of ext_defence.tex and appendix_defence.tex from setup_bytes.py / params_table.py run with --policy cnn18c, and replace the numbers.

Add the split-attack per-query factor and the "beta0/-log2(1-e/n) vs t0" percentages to the script output so that the text cannot drift again. The second item (0.9-1.7% and 3.7-6.9%, ext_defence.tex:164-166) is also unverified.

Fix the sp2027.sh comment. (Effort: small (2-3 h: run two scripts, rewrite three paragraphs).)

**Verifier.** I re-ran setup_bytes.py and params_table.py with --policy cnn18c and got exactly the reviewer's Llama-2-7B values: 2382.0 / 2326.5 / 51.6 (2.2%) / 3.8 MB at a=100 and 9758.7 / 51.6 (0.53%) / 15.9 MB at a=1000. Query factors are 1.731 / 1.211, t_min is 23/34, and the largest t0 is 4936/2^18 = 1.9%. With --policy auto, which resolves to cnn16c for 7B, the same scripts reproduce every stale value: 2833.5 / 2784.0 / 45.2 / 4.3 / 11272.7 MB, x1.081 (the '8%'), t_min 35, and t0/n = 8049/65536 = 12.3% (the '12%').

The reviewer called '0.9-1.7% / 3.7-6.9%' (ext_defence.tex:165-166) unverified, but it is also stale. The auto plan gives about 0.8-1.6% / 3.6-6.8%; cnn18c gives 0.1-0.4% / 0.8-1.0% across all four models.

'3.2-6.6x' (ext_defence.tex:108, appendix_defence.tex:177) first appears in the first draft, commit 06665fa. No plan reproduces it under the reviewer's reading: cnn18c 4.1-6.5, cnn16c 2.0-4.7, cnn17c 3.1-5.5. No script computes it.

The '2.8 GB' comment at sp2027.sh:356 is stale; the cnn18c value is 2.38 GB.

Scope is right: appendix_defence.tex and ext_defence.tex are built only under \ifdefined\pviextended (main.tex:88-104). The body (defence.tex:211-225), tab:setup and tab:params are correct. High fits: the full version's text contradicts the table printed directly above it.

### CA-4 [medium, confirmed] The documented commands and defaults do not reproduce tab:params or tab:setup; "paper" names the course report's policy; the body's beta formula is not the implemented one

**Where:** code/experiments/6_improvements/params_table.py:3,64; setup_bytes.py:4,27; setup_timing.py:52; code/src/pvi/fullcheck/plans.py:57; code/src/pvi/fullcheck/protocol.py:220; paper_sp2027/sections/app_params.tex:57; paper_sp2027/sections/defence.tex:170

**Evidence.** - **Defaults:** params_table.py, setup_bytes.py and setup_timing.py all default to `--policy auto`. The docstring command at params_table.py:3 passes `--policy auto` as well.
- **What auto resolves to** (computed): auto_policy picks cnn16c for Llama-2-7B and cnn17c for Llama-2-13B (cnn18c only for GPT-2). The documented command therefore prints neither tab:params nor tab:setup, which are cnn18c.
- **Footnote:** app_params.tex:57 says "All values: params_table.py". But params_table.py prints neither the "Setup proof 2.4/9.8 GB" row nor the "113.5/79.4 MB" row; those come from setup_bytes.py.
- **Library default:** the library's default policy is still "paper", i.e. one tree per operation at rate 4 (protocol.py:220; plans.py:57: "the report's commitment ... (the default everywhere)"). In the S&P artifact "the paper" means cnn18c.
- **beta formula:** defence.tex:170 sets beta = lambda + log2(2|J|), where J is the coded matrices (130 for Llama-2-7B). The code and tab:params use n_op = 226 weight operations (column_bits(lam, len(shapes)); app_params.tex:23). This is conservative (136.8 against 136.0), but it is not what the body says.

**Recommendation.** - Add one constant in plans, PAPER_POLICY = "cnn18c", and use it as the CLI default of every script.
- Rename the per-operation policy "paper" to something like "per-op" or "report", and keep it as a non-default option.
- Merge setup_bytes.py into params_table.py, so that one command prints every row of tab:params and tab:setup, and fix the footnote.
- Change defence.tex:170 to n_op >= |J|, as app_params.tex already says. (Effort: small (2 h, plus re-pinning test_plans.py:1120).)

**Verifier.** Every point checks out.

- The defaults are --policy auto: params_table.py:64, setup_bytes.py:27, setup_timing.py:52. The docstring command at params_table.py:3 passes auto, and setup_bytes.py:3 passes no policy.
- auto resolves to cnn16c for Llama-2-7B and cnn17c for Llama-2-13B and OPT-6.7B (computed); only GPT-2 gets cnn18c. The documented command therefore prints t_min 35/51 and 124.7 MB, not tab:params' 23/34 and 65.6 MB.
- The footnote 'All values: params_table.py' (app_params.tex:57) is also in the submission, which includes app_params through appendix_short.tex:232. params_table.py prints per-tree f / e / t0 / t but not the 113.5 / 79.4 MB or 2.4 / 9.8 GB rows (app_params.tex:48-49); those come from setup_bytes.py.
- The library default is still the report's per-operation 'paper' policy (protocol.py:220; plans.py:57 'the default everywhere'), pinned by test_plans.py:1120. The package selects cnn18c only through PVI_C_POLICY (sp2027.sh:66).
- The body's beta (defence.tex:146,170) uses |J| = 130 committed matrices, giving 136.0. The code and tab:params use n_op = 226, giving 136.8. The code is conservative and r = 5 either way, so this part alone is low.

Medium is right because the paper's named reproduction commands produce different numbers.

### CA-2 [low, partly] verify_setup accepts any SetupParams the caller passes, and nothing runs a query with the per-query t from Theorem untrusted

**Where:** code/src/pvi/fullcheck/setup_proof.py:104-141,144-158,203-240; code/src/pvi/fullcheck/protocol.py:132-150

**Evidence.** - verify_setup(publics, groups, graph_digest, params, proof) (setup_proof.py:203) takes s, e and t0 from the caller.
  - `_transcript` absorbs them (:148-156), but nothing checks that they satisfy the rule of thm:untrusted (appendix_defence.tex:243-252): s with n/p^s <= 2^-beta0, 2e < n-k+1, and t0 at least survivors(n-e-1, n, beta0).
  - A committer that publishes params with e = 0 and t0 = 1 would get an accepting proof that certifies almost nothing.
  - setup_timing.py:93-105 passes the object returned by setup_params to both prove_setup and verify_setup, so the gap never shows up in the package.
- The online half of the theorem, per-query t computed with k+e in place of k:
  - It is produced by setup_params into `params.t` but read only by query_group_columns (:137). That function is called only from tests/test_setup_proof.py:48.
  - params_for (protocol.py:132-150) has no way to take it, so no script or test runs a query under these parameters.
  - The paper's 113.5 / 79.4 MB per query (tab:setup, defence.tex:221-222) is therefore modelled only.

**Recommendation.** Make the verifier derive its parameters: verify_setup(publics, groups, digest, lam, query_bits, amortise, proof), which calls setup_params itself, or which asserts params == setup_params(...).

Add params_for(..., setup=SetupParams), so that group_columns come from params.t after an untrusted setup.

Add one test that runs an honest and a tampered query with those parameters.

Keep the paper's wording that the per-query cost after setup is modelled, unless the final run times it. (Effort: small-medium (half a day).)

**Verifier.** The facts are accurate. verify_setup takes the caller's SetupParams (setup_proof.py:203-217) and never checks them against the rule. query_group_columns is used only at test_setup_proof.py:48. params_for (protocol.py:132-150) has no setup input, so no query runs with the untrusted t.

The framing overstates the problem. Taking parameters from the caller is the library-wide convention: Verifier also takes caller-supplied SecurityParams (protocol.py:755-757). In the package the parameters come from setup_params (setup_timing.py:93-105), and test_setup_proof.py:144-155 checks setup_params against the paper's rule (s, 2e < n-k+1, t0, t). 'e=0, t0=1 certifies nothing' happens only if a client adopts parameters the committer chose, which the paper does not describe.

The per-query 113.5 / 79.4 MB is exact arithmetic, 4*t*rows from the checked t, and the paper does not claim it was timed.

Two of the reviewer's suggestions are worth doing cheaply: a parameter validator (or verify_setup deriving its own parameters), and one honest-plus-tampered query run under params.t. This is hardening and test coverage, not a soundness gap.

### CA-3 [low, partly] Setup cache: the docstring overclaims its safety; 4 spot-checked leaves per tree; no commitment digest in run records; the folder comes from an environment variable

**Where:** code/src/pvi/fullcheck/setup_cache.py:23-28,43-55,128-165; code/src/pvi/fullcheck/protocol.py:220-252; code/tests/test_setup_cache.py:68-75

**Evidence.** - **The docstring claim (setup_cache.py:26-27):** "A wrong leaf could not make a verifier accept anything".
  - In the package the verifier's key is the prover's own commit_graph result, which may come from the cache (protocol.py:241-252).
  - So a cache entry decides what the roots bind. An entry that holds model A's leaves on most columns and A''s on a set B is exactly the split commitment (eq:split).
  - Such an entry passes `_commit_from_leaves` with probability about (1-|B|/n)^4, since cache_checks=4 (protocol.py:220) and `_sample` draws 4 leaves (setup_cache.py:128).
  - A prover computing with A' then passes the column check whenever all opened columns fall in B.
- **A different commitment:** an entry with a single wrong leaf gives a root different from a fresh commit with probability about 1-4/n. Nothing records this: bench.py:852-855, perf.py:326 and v1_bytes.py:420 store only hit/miss/off, never the roots.
- **The test:** test_setup_cache.py:68-75 corrupts every leaf of a tree, never a few.
- **check_table (setup_cache.py:159)** accepts any power-of-two leaf count of at least V. The correct count is next_pow2(V). A longer list gives a tree whose depth differs from TablePublic.depth.
- **Practical risk is low.** The zip CRC catches random disk corruption, and any change to how leaves are defined changes every leaf, which the 4 checks catch. The remaining risk is a crafted or foreign file in logs/.setup_cache, which sp2027.sh:77 enables for every job, on a shared server.
- **The folder** comes from PVI_SETUP_CACHE (setup_cache.py:43,54). The authors' rule is that every parameter becomes a CLI argument.

**Recommendation.** - Correct the docstring: the cache is trusted prover input, and the spot check catches systematic errors, not adversarial ones.
- Record a commitment digest in every perf, v1, bench and gen record: a hash of the plan and all group and table roots. Then make sp2027_numbers check that every record of one (model, plan, weights) carries the same digest. This also documents the reproducibility claim of D6a.
- Require len(leaves) == next_pow2(V) in check_table.
- Replace the environment fallback with --setup-cache DIR and --setup-cache-checks N on the scripts. (Effort: small (2-4 h).)

**Verifier.** Most facts hold, but one citation is wrong and the impact is lower than stated. The all-leaves corruption is at test_setup_cache.py:93 (the damaged-files test, lines 85-104). Lines 68-75 are the key-separation test. The point itself is correct: no test corrupts only a few leaves.

On the docstring: its reasoning (setup_cache.py:26-28) holds when the verifier's roots come from an independent source. In the harness the verifier uses the prover's possibly-cached roots (protocol.py:241-245). So an entry with a few wrong leaves passes the 4-leaf check with probability about (1-|B|/n)^4. That gives roots different from a fresh commit and contradicts 'the roots ... are those of a fresh commitment' (lines 9-11).

However, the package's honest prover builds its members from the graph's own weights (protocol.py:301,307). A wrong leaf therefore only makes an opening at that column fail. Acceptance of a wrong output needs a crafted file plus a tamper that opens from altered weights, which is an attack on the experimenter, not on the protocol. The paper never mentions the cache.

Roots are not recorded per run (bench.py:852-855, perf.py:326, 8_v1/v1_bytes.py:420). setup_cache_check.py does check root equality, but as a one-off validation, not per record.

check_table's length rule is as described (setup_cache.py:159 against commitment.py:705-706 and :688). A wrong length can only come from a crafted file, and the depth mismatch makes honest proofs fail rather than accept.

The PVI_SETUP_CACHE fallback (setup_cache.py:43,54; sp2027.sh:77) is real and falls under the authors' every-parameter-a-CLI-argument decision.

Overall: fix the docstring wording, add the length check and the CLI flag. Low.

### CA-5 [low, not verified (low)] The same quantities are implemented several times in the commitment area

**Where:** code/src/pvi/fullcheck/setup_proof.py:79-97; plans.py:133-156,226-229,358-365,391-410; analytic.py:138-203; code/experiments/6_improvements/params_table.py:75,97-101; setup_bytes.py:36-51; setup_timing.py:114-115

**Evidence.** - **Column counts:** setup_proof.survivors(a, n, bits) is plans.exact_columns(k, n, bits) with a = k-1. I compared both on 400 random rate-1/2 shapes and found 0 mismatches.
- **Non-claim bytes (u, columns, multiproofs)** are computed three times:
  - analytic.proof_bytes (:162-203);
  - plans.plan_overhead (:391-410);
  - plans._choose.cost (:358-365).
  analytic._matrices / _trees (:138-159) restate the plan's group structure.
- **Script scaffolding:** params_table.py, setup_bytes.py and setup_timing.py each rebuild the same SimpleNamespace publics and groups, the per-group rows, and the honest t. The honest t is exactly CommitmentPlan.group_columns.
- **Setup-proof bytes** are modelled twice: setup_bytes.py:48-51 uses a bound of t0*log2(n) path hashes, while SetupProof.nbytes (setup_proof.py:79-83) counts the real ones. The difference is 0.2%.
- **Dependency direction:** plans imports analytic, which imports claimcodec (2,200 lines) just for field_size.

**Recommendation.** - Keep one function, e.g. columns_needed(bad, n, bits), for every "least t" in plans and setup_proof.
- Give CommitmentPlan the methods group_rows() and trees(ops).
- Make plan_overhead call proof_bytes.
- Add setup_proof.setup_cost(plan, lam, amortise, fs), returning the bytes and t that setup_bytes, setup_timing and params_table print.
- Check every change against the golden outputs; none of this should change any number. (Effort: medium (half to one day).)

### CA-6 [low, not verified (low)] Dead or opt-in code the paper does not use, and docstrings that cite the course report or untracked documents

**Where:** code/src/pvi/fullcheck/commitment.py:19-21,76-84,122-143,209-277; extfield.py:5,52,117-136,140,168-174; setup_proof.py:1-3,20,32,137-141; setup_cache.py:1; plans.py:4-8,57-58; analytic.py:9-11

**Evidence.** - **Merkle worker processes** (PVI_MERKLE_PROCESSES, ~80 lines, commitment.py:76-84 and 209-277): off by default and absent from the paper. They appear only in the 6_improvements README and test_fast_verifier.py:202-238.
- **map_threaded and LEAF_THREAD_BYTES** (commitment.py:122-143) are a generic threading helper that claimcodec also imports. The comment at :125 cites laptop timings.
- **extfield:** neg (:52) is called nowhere. inv_fermat (:117), batch_inv (:122) and eq_eval (:168) are used only by tests as references.
- **query_group_columns** is test-only (see CA-2).
- **Stale or misleading docs:**
  - commitment.py:19-21 says "the committer is honest and no proximity test is needed (report, Theorem 3.2)". This contradicts setup_proof.
  - setup_proof.py cites "plan item D4, the trust report's Option B", "Theorem 3.2" and "Fact 1 of the trust report" (:1,3,20,32). No trust report is tracked in the repository; the paper has fact:ext and thm:untrusted.
  - setup_cache.py:1 says "plan PS2, option (c)".
  - plans.py:4-8 and 57 narrate "the report".
  - analytic.py:9-11 mentions "the earliest run, raw/".
  - extfield.py:5 points to docs/improvements, and :140 says "V-V1N part 3".

**Recommendation.** - Delete the process-pool path, so that verify_multiproofs becomes a loop, or inline it.
- Move map_threaded to a small util module.
- Move inv_fermat, batch_inv and eq_eval into tests/ as reference implementations, and delete neg.
- Rewrite the module docstrings against the paper's labels: thm:sound, thm:untrusted, fact:ext, tab:params. (Effort: small (2-3 h).)

### CA-7 [low, not verified (low)] verify_multiproof does not validate its proof elements; a malformed column proof raises instead of being rejected

**Where:** code/src/pvi/fullcheck/commitment.py:190-206; code/src/pvi/fullcheck/protocol.py:698-712,1352-1357,1238-1241; code/src/pvi/fullcheck/setup_proof.py:218-239

**Evidence.** - **No element check:** verify_multiproof hashes `b"pvi/node" + left + right` with whatever the proof holds.
  - The column path checks only the proof's length (protocol.py:1355).
  - A non-bytes element makes _node raise. _verify_jobs records the exception, and _columns_verdict re-raises it (:705-707) instead of returning columns_merkle.
  - The lookup path does check `isinstance(h, bytes) and len(h) == HASH_BYTES` (:1238-1240).
  - verify_setup bounds neither the length of the multiproof nor its elements.
- **Still sound:** every hash input the verifier computes is exactly 32 bytes at a fixed offset, so elements of the wrong length cannot forge a root. The wire decoder always yields 32-byte elements. This is a robustness and consistency issue only.

**Recommendation.** Have verify_multiproof return False unless len(proof) <= len(leaves)*depth and every element is 32-byte bytes. Then drop the exception path in _columns_verdict and the duplicate check in the lookup path. Add tests for a leaf at the wrong index and for an extra proof element. (Effort: small (1 h).)

### CA-8 [low, not verified (low)] Proposed clean module structure for arithmetic and commitments

**Where:** code/src/pvi/fullcheck/{field,extfield,commitment,plans,setup_proof,setup_cache,analytic}.py

**Evidence.** - **Mixed concerns:**
  - field.py (488 lines) holds the scalar field, the NTT and RS encoding, plus three families of exact products (float64 limbs :183-279, numpy small operands :282-321, int8 GEMM :324-488).
  - commitment.py (715 lines) holds the Merkle primitives, the threading helpers, the process pool, four commitment classes and the lookup tables.
  - analytic.py holds the decoder architecture (decoder_shapes, decoder_claim_columns), which restates transformer.build_decoder, together with the byte model.
- **Safe to move:** none of these modules defines a cheap-op function, so they can move without changing graph digests. Only these must stay byte-identical: the hash domain strings (pvi/node, col, group, row, pad), the setup transcript labels and the setup-cache VERSION.

**Recommendation.** - arith/field.py: P, roots of unity, power tables, ntt, rs_encode.
- arith/products.py: the exact limb, numpy and int8 products.
- arith/extfield.py: F_{p^8}, without the test-only helpers.
- commit/merkle.py: domain tags as named constants; leaf functions; MerkleTree; multiproof, multiproof_size and verify_multiproof; expected_multiproof_nodes and expected_lookup_nodes.
- commit/matrix.py: WeightCommitment, TransposedCommitment, GroupCommitment, TableCommitment, the public views, codeword_at, vandermonde_columns, the streamed leaf digests.
- commit/plans.py: policies, exact t through one function, layouts, groups, auto. The plan carries the whole model's n_op, so params_for no longer depends on callers passing the right count.
- commit/cost.py: proof_bytes, setup_size, plan_overhead and setup_cost (replaces the cost half of analytic).
- commit/setup_proof.py, which derives its own parameters (CA-2).
- commit/setup_cache.py, with the folder given explicitly (CA-3).
- Move decoder_shapes and decoder_claim_columns next to the decoders.
- Re-export the public names from pvi.fullcheck. (Effort: medium (1 day including import updates and golden-output check).)

## Protocol v0

Protocol v0 review (protocol.py, claimcodec.py, pipeline.py, token_sampler.py, graph.py, contention.py, hostmem.py). Read only. Paths are relative to the wt_sp2027 root. I checked these parts and found them sound:
- **Freivalds products are exact.** Blocks of _RHS_CHUNK=8192 terms on int8 inputs stay below 2^53, and every builder clamps the inputs of its weight ops to int8 (transformer.py:183).
- **Deferred GPU range checks protect inexact products.** range_or_shape is decided before freivalds, so an inexact product can never be the deciding check (protocol.py:1544-1547, 1060-1063).
- **Challenge sampling is unbiased.** Folding uses 2P < 2^32 rejection, and columns use the 2^64 mod n limit (protocol.py:393-465).
- **Fiat-Shamir order is correct.** The transcript absorbs the statement, then the claims, the rows and the lookup multiproofs, then draws chi and chi', then absorbs u, then draws the columns (protocol.py:1740-1751, 1869-1879). chi' is never sent to the prover (protocol.py:1829).
- **Kpre absorbs nothing, by design.** This matches prop:kpre-fs.
- **The PVC3, field and row codecs reject malformed input only with ClaimCodecError.** Allocations are bounded by the public shapes, and mutable buffers are refused (claimcodec.py:1825-1826). There are fuzz tests and a test that the GPU decoder rejects exactly what the host decoder rejects (test_claimcodec.py:332, 646, 1059). Unencodable values travel as b"" and are rejected at the matching check.
- **The streaming verifier gives the same verdicts.** Its verdict order (token, range_or_shape, lookups, freivalds, columns) matches the batched path, and its transcript is the same except for the documented prefix property on rejection.
- **contention.py and hostmem.py are fine.** Their measurement snapshots sit outside the timed windows, and pinned allocation is guarded.

**Problems found:**
1. **(high) The graph digest binds Python names, not semantics.** It hashes each cheap-op function by its module and qualified name plus the values it captures. It does not hash the function's code, module globals or module-level tables. So the paper's sentence "constants (table bytes included)" is not supported. Moving modules would change every Fiat-Shamir transcript, the mode-C, V1 and setup multiproof bytes, and the published certified digests. No test pins digest values.
   - Proposed fix: frozen explicit digest names, which keep every output byte-identical, plus a golden digest test.
2. **(high) The token rule accepts queries shorter than the LM-head window without checking any token.**
3. **(medium) Kpre secrecy depends on the caller.** Kpre is sound only if the caller passes an interactive Challenger. A Fiat-Shamir challenger makes chi a public constant, and a seeded one makes it predictable. Nothing refuses either.
4. **(medium) In mode K under Fiat-Shamir the transcript does not absorb the weights**, although thm:fstree covers mode K with statements the prover chooses.
5. **(medium) The library defaults are still the course report's configuration.**
6. **(medium) Splitting protocol.py has three hazards that would go unnoticed:**
   - monkeypatch targets, including the _defer patches that let the CPU tests exercise the GPU code forms;
   - cut_protocol's 13 private imports and the circular dependency between v0 and V1;
   - settings read from environment variables at import time.
7. **(low) Modes K and Kpre never check claim_bound.** This affects completeness only.

None of the recommended fixes changes a paper number: the run package always has T >= G, passes Challenger() to precompute, and never runs mode K. One exception: the optional v2 semantic digest would change digest-dependent bytes, so it should be decided before the Fiat-Shamir goldens are captured (task #28).

### P0-1 [high, partly] Graph digest binds Python module/function names, not cheap-op semantics: paper claim unsupported, and any module move silently changes transcripts, FS proof bytes and published digests

**Where:** code/src/pvi/fullcheck/graph.py:353-391, 433-448

**Evidence.** **What is and is not hashed.** `_canonical` hashes a function as `rec(f"{v.__module__}.{v.__qualname__}")` plus its defaults, kwdefaults and closure cells (graph.py:382-386). It does not hash `__code__` or globals. Any object outside the known types is hashed only by its type name (graph.py:390-391). So the following are not bound:
- code literals, e.g. `lambda a: a * 8` (transformer.py:525,527) and the `True`/`False` flags passed to `_norm_int`;
- module constants SHIFT and RES_MAX (transformer.py:45,49);
- the softmax table `_EXP_TABLE` (transformer.py:53), which `_attention` reads as a global (transformer.py:327,352);
- the default `pmax` of `_attention`.

All lambdas of one builder share one qualname (`build_decoder.<locals>.<lambda>`). They are told apart only by their captured constants, note and params.

**The code already relies on this.** The developers maintain digest stability by hand: transformer.py:562 says "today's op, digest unchanged" and real_llama.py:389 says "no extra constant, so the digest is today's". ppl_2048.py:364-366 states that a copy of the builder "cannot match" the digest even with equal weights, constants and logits.

**The paper claims more.** defence.tex:188-191 says the digest covers "every operation in order with its inputs, output and constants (table bytes included)". ext_proofs.tex:203-205 makes a similar claim.

**Consequences of moving graph, transformer, real_weights, real_llama, quantize or experiments/9_quality/opt_builder.py, or of refactoring their lambdas:**
- Every FS transcript changes, because the digest is absorbed first (protocol.py:786,1624; setup_proof.py:149; cut_protocol statement).
- The FS column draws change, and so do mode-C, V1 and setup multiproof bytes (`multiproof_size` depends on the indices, commitment.py:167-175). Interactive runs and v0 Kpre are unaffected.
- Stored digests stop matching: v1_bytes.py:371, digest_check.py:56, ppl_2048.py:265,377, ppl_llama.py:369, zeroshot.py:378, test_real_weights.py:370.
- The published certified digests 16de86ea... and d9daa017... in scripts/artifact_README.md:70-73 and D6a_reproducible_commitment.md:71 stop matching. They back the paper's claim of a bit-for-bit rebuild (defence.tex:52).
- No test pins a digest value: grep finds no 64-hex literal in code/tests. A golden comparison would show unexplained FS byte differences.

**Note strings are protocol data too.** `note` is hashed and also matched in code: transformer.py:406, real_weights.py:288,307, real_llama.py:826, sampling.py:76-384. A comment-only cleanup that rewords notes changes digests or breaks with_lm_positions.

**Recommendation.** Decide this before capturing the FS goldens (task #28).

Recommended for the freeze (frozen digest names; every output stays byte-identical):
1. In graph.py, add a `digest_name(name)` decorator and a `named(fn, name)` helper that set `fn._pvi_digest_name`. In `_canonical`, hash that string in place of `__module__.__qualname__`. Raise if a function has none. Also raise in the type-name fallback (graph.py:390-391) rather than hashing only a type name.
2. Tag about 55 cheap-op sites with today's exact strings. The sites are in graph.requant_fn, transformer, real_weights, real_llama, quantize, the `_last_positions` partial, real_llama `_identity`, opt_builder and the 9 ad-hoc CheapOps in tests and experiments. Example strings: "pvi.fullcheck.graph.requant_fn.<locals>.fn", "pvi.fullcheck.transformer.build_decoder.<locals>.<lambda>". `_with_tensors_on` already copies `__dict__` (graph.py:342).
3. Before moving anything, add a golden test that pins `digest().hex()` for tiny builds of every builder: decoder families, prune on and off, with_lm_positions, prob_max, the Llama prefix, CNN quantize, and the HF builders on a tiny synthetic config. Re-check the two published digests with digest_check. Also test that every CheapOp.fn carries a digest name.
4. Treat `note`, `params` and `meta` strings as protocol constants in the cleanup.
5. Correct defence.tex:188-191 and ext_proofs.tex:203-205. The digest binds each op's wiring, note and params, plus the registered name of its function and the constants that function captures. Module-level semantics (SHIFT, RES_MAX, the exp table, norm and RoPE) are fixed by the code version.

Optional, only if done before the goldens: a `pvi/graph/v2` digest with semantic op kinds, explicit parameters, and a block that hashes the module tables. This makes the paper sentence true as written and lets an independent implementation recompute the digest. It changes FS challenges and bytes and the published digests. (Effort: About 1 day for the frozen-name design and golden test; about 2 days for the optional v2 digest, which also changes FS bytes and published digests.)

**Verifier.** Core claim confirmed. graph.py:381-386 hashes a function as `module.qualname` plus its defaults, kwdefaults and closure cells. It does not hash __code__ or globals, and graph.py:390-391 hashes any other object by its type name only. Not bound: the global _EXP_TABLE (transformer.py:53, read at 302/327/352), the literal in `lambda a: a * 8` (525/527), and the default pmax of _attention (the lambda at 563 omits P). The authors already keep digests stable by hand (transformer.py:562, real_llama.py:389, ppl_2048.py:364-366). No test pins a digest value: there is no 64-hex literal in code/tests. Notes are matched in code (transformer.py:406). Corrections: (1) SHIFT and RES_MAX are bound for requant and residual ops, through params and the shift/res_max that requant_fn captures (transformer.py:485,489; graph.py:160-174). They are unbound only where code reads them as globals: attention output transformer.py:380, glu 574, _exp_lut. (2) _norm_int's center is a bound default (`c=center`, 493); only the qk-norm literal False (544/546) is unbound. (3) ext_proofs.tex:203-205 ('hashes ... function and constants') is accurate as written. Only defence.tex:188-191 ('constants (table bytes included)') overstates, because of the exp table and code literals. (4) This is not a soundness gap: the verifier recomputes with its own code, so the code version fixes those constants. The real costs are independent reproducibility and the refactor. (5) Most of the 'stored digest' sites are writers or same-run comparisons and keep working after a move: v1_bytes.py:371, digest_check.py:56, ppl_2048.py:265,377, and test_real_weights.py:370, which compares two values computed by the same code. What breaks is the published constants (artifact_README.md:73-74, not 70-73; D6a:71; F2c_zero_shot.md:160,165) and old JSON records. I keep it high because this is the decision that gates the layout and refactor tasks. If the goldens of task #28 are already captured, any module move or lambda refactor needs the frozen-name scheme, or the goldens must be taken again.

### P0-2 [medium, partly] Token rule fails open when the query is shorter than the LM-head window (T < G)

**Where:** code/src/pvi/fullcheck/protocol.py:940-961 (early return at 952-953)

**Evidence.** **The early return.** `check_tokens` returns True when `x.shape[1] < n` (protocol.py:952-953).

**Why derive does not catch it.** For such a query, `_last_positions` slices `a[:, -n:, :]` (transformer.py:388-390), which keeps all T < n positions. `claim_columns` derives the head's column count from the same graph (graph.py:450-467). So derive accepts a head claim of B*T columns (protocol.py:1030), and Freivalds and the column checks pass on honest logits. The query is accepted without any token being checked.

**The docstring is wrong here.** It says "Claims of another shape pass here (derive rejects them)" (protocol.py:945-946). That is false for this case.

**What the paper assumes.** The paper proves a response on s with T = P + G - 1 (ext_proofs.tex:247-253). Lemma causal assumes n <= T (ext_proofs.tex:259). thm:gen attributes the guarantee to Verifier.check_tokens (ext_proofs.tex:252, 299-312).

**Attack.** If a client fixes G in advance (gen_bench builds `with_lm_positions(graph, g)` per cell, gen_bench.py:244), a provider answering a short prompt can return arbitrary tokens. For example, P = 8 and G = 257 with 100 free tokens gives T = 108 < 257, and the proof is accepted.

**Coverage and impact.** No test covers it: test_generation.py:93-111 only replaces tokens. Every package cell has T >= G (prompts of 16 and 512, sp2027.sh:288-294), so no number is affected.

**Recommendation.** Reject, with label "token", when `x.dim() != 2 or x.shape[1] < n`. Keep the pass-through only for a head claim of the wrong shape, which derive rejects. Correct the docstring. Add a test that a query of T = n - 1 tokens on `with_lm_positions(graph, n)` is rejected in modes C and Kpre, streaming and batched. Honest outputs are unchanged. (Effort: About 1 hour.)

**Verifier.** Code facts confirmed. check_tokens returns True when x.shape[1] < n (protocol.py:952-953). _last_positions keeps all T < n positions (transformer.py:388-390), so derive accepts a head claim of B*T columns. The docstring at 945-946 is false for this case. The attack premise is weaker than stated, though. In the paper the client forms s from its own prompt and G-1 received tokens, so T = P+G-1 >= G (ext_proofs.tex:249-253), and T < G needs a client that accepts a short response. The library API never sees P: run_query takes only x (protocol.py:1907-1931), and gen_bench forms s honestly. A client that does not check the token count is also exploitable by a response that is too long. With T > P+G-1, the tokens before position T-n+1 are never checked, because check_tokens only looks at the last n-1 positions (960-961). Rejecting T < n is good defence in depth but does not close that hole. The complete fix is for the token rule or run_query to take the prompt (or P) and require T == P+n-1 and x[:, :P] == prompt. No number is affected and no paper claim is false, so medium rather than high.

### P0-3 [medium, confirmed] Kpre's secret chi is secret only if the caller passes an interactive Challenger; a Fiat-Shamir or seeded one makes chi public and is not refused

**Where:** code/src/pvi/fullcheck/protocol.py:844-850 (precompute), 372-380 (_key)

**Evidence.** **How chi is drawn.** `precompute(challenger)` draws chi with `challenger.folding(op.name, ...)` from whatever Challenger it is given.

**With a Fiat-Shamir challenger, chi is public.** The key is SHAKE-256 of the constant initial state `pvi/fullcheck/v3` plus `challenge/fold/<op name>` (protocol.py:336, 373-376). That is a public constant for each op name, so any prover can forge Z with chi^T Z = u^T X.

**With a seeded challenger, chi is predictable.** The key is `sha256("test-seed/<seed>/<count>/<label>")` (protocol.py:377-380).

**Nothing refuses either case.** The proof of prop:kpre-fs relies on "Verifier.precompute with an interactive Challenger" (ext_proofs.tex:546-547).

**Current callers are fine.** All run-package callers pass `Challenger()`: perf.py:295, v1_bytes.py:405, gen_bench.py:265, serve_ab.py:516, bench.py:613,897. ppl_llama.py:518 (quality tier) uses `Challenger(seed=11)` for its Kpre tamper check. Its tampers are not adaptive, so that verdict holds.

**The risk.** An artifact user who writes `precompute(Challenger(fiat_shamir=params.fiat_shamir))` silently gets zero Kpre soundness. That call mirrors how `_run_query` builds its challenger (protocol.py:1939).

**Recommendation.** Choose one:
- `precompute(*, seed=None)` draws its own keys from `secrets`, with the seed documented as test-only;
- or keep the argument but raise ValueError when `challenger.fiat_shamir`.

Also:
- Make seeded challengers explicit, e.g. `Challenger.for_tests(seed)`, and say so in the `run_query(seed=)` docstring.
- Switch ppl_llama.py:518 to an unseeded challenger.
- Add a test that precompute with a Fiat-Shamir challenger raises.

The interactive draws keep the same distribution, so outputs stay identical. (Effort: 1-2 hours.)

**Verifier.** Verified. precompute (protocol.py:844-850) accepts any Challenger. With a Fiat-Shamir challenger the key is SHAKE-256 of the constant initial state 'pvi/fullcheck/v3' plus 'challenge/fold/<op>' (336, 373-376, 396), which is a public constant. A seeded challenger's key is sha256('test-seed/...') (377-380), which is predictable. The proof of prop:kpre-fs assumes an interactive Challenger (ext_proofs.tex:546-547). Every run-package caller uses Challenger(): bench.py:613,897, perf.py:295, gen_bench.py:265, serve_ab.py:516, v1_bytes.py:405. ppl_llama.py:518 uses seed=11, and its tampers are fixed in advance and not adaptive (500-527), so that verdict holds. Further seeded precomputes sit outside the package: fold_ab.py:51, plan_bytes.py:192, gpu_ab.py:140. This is an API footgun that silently gives zero Kpre soundness, so medium is fair. The fix is cheap and keeps outputs identical.

### P0-5 [medium, confirmed] Library defaults are the course report's configuration; the paper's configuration exists only as CLI flags in sp2027.sh

**Where:** code/src/pvi/fullcheck/protocol.py:132-133, 220, 474-475, 758-766, 1909; transformer.py:496-497; real_weights.py:182-185

**Evidence.** **Library defaults versus the paper's settings:**
- `commit_graph` policy="paper" (protocol.py:220); the paper uses cnn18c (sp2027.sh:66).
- `params_for` fiat_shamir=False (protocol.py:132).
- `Prover` lean=False (protocol.py:474-475).
- `Verifier` lean, lookups and stream all False (protocol.py:761-766).
- `run_query` wire=False (protocol.py:1909).
- `build_decoder` prune_last=False (transformer.py:497).
- `build_opt_from_hf` norm_gain=32 with no smoothing (real_weights.py:182-185), not CERTIFIED_RECIPE.

**Where the paper's settings live.** The package sets each one by flag (`--fs --lean --wire on --prune-last on --lookups on`, sp2027.sh:132-136). test_plans.py:1120 even pins "paper" as the library default.

**The scripts disagree.** bench, perf and serve_ab default to "paper"; v1_bytes, gen_bench and setup_timing default to "auto" (inventory lib_fullcheck.md §8). Leaving out one flag silently reproduces the course report rather than the paper.

**Recommendation.** 1. Make the parameters that shape the protocol keyword-only with no default: policy, fiat_shamir, wire, lookups, lean, prune_last, and the recipe for the real-weight builders.
2. Add `pvi/fullcheck/presets.py` with two presets:
   - PAPER: cnn18c, Fiat-Shamir, wire, lookups, lean, prune_last, CERTIFIED_RECIPE / MAIN_RECIPE.
   - REPORT: the paper policy, interactive, and so on, kept as the non-default option.
3. Use PAPER for every script's argparse defaults. Tests pass REPORT explicitly.
4. Replace test_plans.py:1120 with a test that PAPER equals the flags in sp2027.sh.

The missing defaults turn every implicit call into a TypeError, so the migration is mechanical. Each call site keeps its current value, so outputs stay identical. (Effort: 1-2 days (about 90 importer files).)

**Verifier.** All the cited defaults hold: protocol.py:132 fiat_shamir=False, :220 policy='paper', :474-475 Prover lean=False, :761-766 Verifier lean/stream/lookups False, :1909 wire=False, transformer.py:497 prune_last=False, real_weights.py:182-185 norm_gain=32 with no smoothing. test_plans.py:1120 pins 'paper' as the library default, and sp2027.sh:66 and 132-136 set the paper's configuration by flag. The scripts disagree on policy: bench.py:1015, perf.py:177 and serve_ab.py:867 default to paper; v1_bytes.py:307, gen_bench.py:174 and setup_timing.py:52 default to auto. The inconsistency goes further than the finding says. v1_bytes.real_graph defaults to recipe='legacy' (v1_bytes.py:107). gen_bench hard-codes wire, lean and prune_last=True (gen_bench.py:218,235,263-266) instead of exposing them as flags, which conflicts with the 'every parameter a CLI argument' decision. ppl_llama.py:513-518 uses the library defaults. Caveat on the fix: removing defaults from about 90 importers is a lot of churn two weeks before the feature freeze. A PAPER preset used as the argparse defaults, plus a test that pins it to sp2027.sh, gives most of the safety for much less work.

### P0-6 [medium, confirmed] Splitting protocol.py by function: three hazards that a naive move would get wrong without any test failing

**Where:** code/src/pvi/fullcheck/protocol.py:1-2055; cut_protocol.py:42-43

**Evidence.** **What the 2,055 lines hold:**
- parameters (94-188)
- graph commitment and setup cache (191-318)
- Challenger (320-470)
- Prover (473-530)
- check helpers (532-751)
- Verifier, 820 lines, with streaming at 1454-1595
- statement absorption (1622-1649)
- query messages (1652-2055)

**Hazard (a): monkeypatch targets.** Tests patch these names on `proto`:
- `_defer`, which runs the GPU code forms on the CPU: test_fast_verifier.py:55,246,261; test_gpu_verifier.py:407,436,698; test_layouts.py:205; test_lookups.py:209; test_plans.py:597
- `int8_ok`, `_to_host` (test_gpu_verifier.py:735), and `_TREE_MIN` / `_TREE_CHUNK` (test_fs_tree.py:28-29)
- `Challenger.absorb` / `_key`, `claimcodec.encode` / `pack_rows`, and Verifier methods

If a new module does `from .checks import _defer`, it keeps its own binding. The patch then misses it, and the tests still pass without exercising the deferred path.

**Hazard (b): private imports and a circular dependency.** cut_protocol imports 13 private names from protocol (cut_protocol.py:42-43). v0 also depends on V1: `cut.delta_range` (protocol.py:75), `_CUT_NARROW` (protocol.py:98), `buckets.enabled` inside `Challenger.ext` (protocol.py:412-417), and `_cut_plan` (protocol.py:774).

**Hazard (c): import-time and environment switches.** `_GPU_DECODE` is read at import (protocol.py:532). `PVI_FOLD_FAST` (protocol.py:869) and `PVI_HASH_THREADS` (protocol.py:129) are also environment variables.

**Constraints the split must respect:**
- Many callers build `Verifier(graph, params, mode, ...)` positionally (fields at protocol.py:755-758).
- `pvi/fullcheck/v3` (protocol.py:336, cited at ext_proofs.tex:375), `pvi/leaf/v1`, `_TREE_MIN` and `_TREE_CHUNK` are protocol constants (FS1 diary §4).
- `ab_verifier.load_base` imports an old flat checkout.
- Unlike H1, nothing in protocol.py is hashed into graph digests, so moving it is otherwise safe.

**Recommendation.** **Proposed modules:**
- `v0/params.py`: Z_BOUND, SecurityParams, params_for, soundness_bits, claim_bound
- `v0/transcript.py`: Challenger, tree absorption with `hash_threads` as a field, `_tensor_blob`, `absorb_statement` with explicit arguments
- `commit/graph_commitment.py`: GraphCommitment, commit_graph, the cache rebuild
- `v0/prover.py`
- `v0/checks.py`: exact right-hand sides, deferred-verdict helpers, uploads
- `v0/verifier.py`: derive, the token rule, products, lookups, columns
- `v0/streaming.py`: pipeline.py plus a streaming mixin of Verifier plus `_run_streaming`
- `v0/query.py`: run_query and the message functions, exposed as a documented internal API that replaces cut_protocol's underscore imports

**Rules for the split:**
- Call shared helpers through the module (`checks._defer(...)`) and update each patch target.
- Keep the Verifier field order.
- Replace V1 knowledge inside v0 with one hook object that V1 sets per query, holding the cut ops' sent ranges and dtypes.
- Turn the environment switches into Verifier or Challenger fields with today's defaults.

**Before moving anything,** add a transcript-trace golden test: record the (label, sha256(blob)) of each absorb and the labels of each challenge, for modes C (paper and cnn18c), K and Kpre, with wire on and off and stream on and off. The existing spies (test_lookups.py:355-357) show the pattern. (Effort: 2-3 days, including the trace test.)

**Verifier.** The three hazards are verified. The `_defer` and `int8_ok` patches are at the cited test lines, plus test_plans.py:746-748. Also verified: `_to_host` (test_gpu_verifier.py:735), `_TREE_MIN`/`_TREE_CHUNK` (test_fs_tree.py:28-29), and the coupling sites at protocol.py:75, 97-98, 412-417, 774, 532, 869 and 129. ab_verifier.load_base exists (ab_verifier.py:45). Small corrections: (1) cut_protocol.py:42-43 imports 13 names, of which 10 are private, not 13 private. (2) monkeypatch.setattr raises on a missing attribute, so a patch misses silently only through re-export shims or from-imports, which is exactly the case the finding describes. The claimcodec patches go through the claimcodec module object (proto.claimcodec.encode), so they survive a split as long as the new modules call claimcodec.encode by attribute. Confirmed that no cheap-op function is defined in protocol.py, so moving it does not touch digests. The split is optional and risky this close to the freeze, so the transcript-trace golden test is the right precondition.

### P0-4 [low, partly] Mode K under Fiat-Shamir does not absorb the verifier's weights, though thm:fstree is stated for mode K with statements the prover chooses

**Where:** code/src/pvi/fullcheck/protocol.py:1622-1649 (op record 1626-1631)

**Evidence.** **What the transcript holds in mode K.** `verifier.publics` is empty, so each `op/<name>` record holds only `(n_rows, row_length)`. Nothing in the transcript depends on W, although the verifier checks against its own weights (`_fold_local`, protocol.py:1852-1853).

**What the theorem says.** thm:fstree covers "v0 in mode C or K" for "every prover that ... outputs a statement and a proof" (ext_proofs.tex:434-439). Its proof uses that the partial transcript τ(w) determines the statement (ext_proofs.tex:477).

**The gap.** If W is part of the statement and the prover may choose it, it can fix false claims, read chi, and then pick W so that chi W X = chi Z. That is the weak Fiat-Shamir pattern the paper itself warns about (defence.tex:183-185).

**Impact.** It is harmless when the verifier fixes W in advance, which is mode K's threat model. No package cell runs mode K: every `--modes` in sp2027.sh:153-317 is C or Kpre.

**Recommendation.** Choose one:
- Absorb a per-op weight digest in mode K, computed once in `Verifier.__post_init__` from the int8 weights and int64 bias, as setup_cache._tensor_digest does (setup_cache.py:58-65). This changes only mode-K Fiat-Shamir challenges and no package number.
- Or narrow thm:fstree's mode-K case to a statement whose W the verifier fixes before the proof. (Effort: About 2 hours for the code; or one sentence in the paper.)

**Verifier.** The code fact is confirmed: in mode K the transcript holds no weight-dependent data (protocol.py:1626-1631, publics empty). The conclusion is overstated. In mode K the weights are the verifier's own input, fixed before any proof ('in mode K it holds the weights', defence.tex:48-50). So W is not part of a statement the prover chooses, and thm:fstree's statement is (graph, x, claims) against that fixed W. Choosing W after seeing chi is outside mode K's model. Even a verifier that holds k models and lets the prover name one gains at most a factor of k. No package cell runs mode K (sp2027.sh: every --modes is C or Kpre). One clarifying sentence in thm:fstree (ext_proofs.tex:434-439) is enough. Absorbing a weight digest is optional hardening.

### P0-7 [low, not verified (low)] Modes K and Kpre never check that honest claims fit the range check (claim_bound is only checked in commit_graph)

**Where:** code/src/pvi/fullcheck/protocol.py:44-47, 232-235

**Evidence.** The module docstring says every weight op "is checked at commitment time" to have honest outputs below 2^29 (protocol.py:44-47). The check lives only in commit_graph (protocol.py:232-235).

K and Kpre verifiers are built from weights (protocol.py:760) without commit_graph. A model whose honest |z| reaches 2^29 would then fail at range_or_shape on every query, and the encoder would refuse its claims (claimcodec.py:926).

This affects completeness only. Soundness still rests on derive's range check (protocol.py:1034-1039).

**Recommendation.** Run claim_bound for each op in `Verifier.__post_init__` (or in precompute) when weights are given, and raise as commit_graph does. Correct the docstring. Outputs for the paper's models are unchanged. (Effort: Under 1 hour.)

## V1

Area V1 (code/src/pvi/fullcheck/{cut,cut_protocol,logup,logup_gkr,gkr_triton,buckets}.py and code/experiments/8_v1/gen_gkr_triton.py) on branch sp2027 at 08a7f1f. I worked read-only: I read the code, the paper (v1.tex, appendix_v1.tex, ext_v1.tex, app_params.tex) and the inventories, and ran light Python only (a re-implementation of the instance packing, an in-memory run of the generator, and one pure-function call).

**Soundness: no gap found in the protocol code.** Each part matches the paper:
- **Cut rules.** P1 to P5 and the col-matrix closure are at cut.py:137-176.
- **Windows and residues.** `lo`, `shift` and the exception rule match eq:v1-window and Lemma lem:v1-window.
- **Exceptions (D4).** `exceptions_ok` checks requant-only ops, strictly increasing indices, |z|<2^29 and rq(z)=s.
- **Multiplicities (D5).** `multiplicities_ok` checks both integer conditions of Lemma lem:v1-logup.
- **The logUp point.** α is drawn from K∖(F+ξF) by rejection, after M1 (values, exceptions, multiplicities) is absorbed together with the cut-plan digest.
- **GKR.** Rounds send g(0), g(2), g(3), with g(1)=c−g(0). The η/θ/r order and the LSB-first point match appendix_v1 (i) to (iii), and the error count (1+Σ(3k+2))/p^8 is the theorem's. q0≠0 is checked. GKR elements and cut folds must be canonical.
- **Final check (F3).** It is exactly eq:v1-final: q = αI − Σ(S − χLe + ξχWe) + (1−I).
- **Folds.** Kpre checks y with χ_s y = u_s x̄. Mode C checks u (or y for col matrices) through 8-plane column openings drawn after the fold message. Col matrices are forced to share one input (protocol.py:807).

**Old and new paths agree.** The new and old verifier paths give the same integers and verdicts. The two Challenger draws are byte-identical, so the PVI_V1_BUCKETS switch never changes a transcript.

**The generator reproduces the committed gkr_triton.py byte for byte** (1,306 lines, checked in memory without writing).

**Refactor safety.** The V1 modules can be moved without changing any digest or transcript: CutPlan.digest uses the fixed tag `pvi/cut/v1` (cut.py:48,251-259), the labels are strings, and no V1 function is hashed into the graph digest.

**Main problems:**
- **lmax is not a cap** (cut.py:190-199), but params_table and the paper assume it is. The tab:params footnote (162 instances of ≤2^24 leaves) is not what the code builds; the stated bound holds only by coincidence.
- **lmax differs between the final run and the paper.** The final run uses lmax 27 everywhere; the paper's security numbers use 24 and 26.
- **The Triton GKR at the package's instance size (2^27) has never run.** The smoke does not exercise it.
- **The V1 switches are environment variables** read at call or import time, toggled by mutating os.environ, and not recorded.
- **Generator hygiene.**
- **Minor:** a paper/code wording mismatch on F4, a crash on an empty cut plan, a pointer-keyed cache, and diary or plan references.

### V1-1 [medium, confirmed] # lmax is not a cap on GKR instance size, but params_table and the paper assume it is

**Where:** code/src/pvi/fullcheck/cut.py:23-25,190-199; code/experiments/6_improvements/params_table.py:56-57,90-92; paper_sp2027/sections/app_params.tex:38,55-56; paper_sp2027/sections/ext_v1.tex:74-75,181

**Evidence.** **What the code does.** The docstring (cut.py:23-25) says an instance is a run of units with at most 2^lmax entries. The packing loop (cut.py:193) starts a new instance only `if cur and cur_n + n > 1 << lmax`, so a single op or col matrix larger than 2^lmax becomes an instance on its own, of any size.

**The hard-coded cells need oversized instances.** My re-implementation of the packing for Llama-2-7B at T=2048 (Kpre, last block pruned, lmax 24) reproduces the hard-coded cells at params_table.py:90 exactly: (103,809,024 leaves, 6 instances) for 2 blocks and (190,840,832, 11) for 3 blocks. Those counts are only possible because each gate and up op (11,008×2,048 = 22.5M leaves) forms its own instance with 2^25 padded leaves (n=25).

**The footnote is wrong for the full model.** The packing gives 156 instances, 62 of them with n=25, not "162 instances of ≤2^24 leaves" (app_params.tex:55-56). The 162 is ceil(N_leaf/2^24) from params_table.py:57.

**The bound method assumes the cap.** params_table.py:92 computes `gkr = log2(instances) + _gkr_log2(args.lmax)`, i.e. every instance at n = lmax. The stated ≤2^-230.1 still holds, but only by coincidence: the method gives 162×875/p^8, against 138,887/p^8 for the actual instances.

**The same assumption appears in the paper text.** ext_v1.tex:74-75 says "instances of at most 2^{n_max} leaves", and ext_v1.tex:181 says "instances of at most 2^24 leaves" for tab:v1, which includes the Llama-2-7B rows.

**Mode C is worse.** With col matrices, the gate+up unit alone has 2^26 padded leaves at any lmax ≤ 26. At about 160 B/leaf (cut_protocol.py:409) that is about 10.7 GB. My inference is that this, not lmax, is why "Llama-2-7B in mode C does not fit on the 11 GB card" (ext_v1.tex:189).

**Recommendation.** 1. Record the actual instance sizes. Add `"instance_bits": cutmod.instance_bits(plan)` to `out["cut"]` (cut_protocol.py:227-228) and to the v1_bytes records.
2. Have params_table sum _gkr_log2(n) over the recorded sizes instead of assuming n = lmax. Drop the `ceil(leaves/2^lmax)` assumption and the hard-coded cells.
3. Correct the docstring and the paper's wording, e.g. "lmax is a target; a unit larger than 2^lmax forms its own instance".
4. Alternatively, split oversized units by rows so lmax becomes a real cap. That changes plans and transcripts, so it must be done before goldens are captured, and only if mode C on 11 GB matters. (Effort: small: record the sizes and fix params_table; medium: split units.)

**Verifier.** All the evidence checks out. A unit larger than 2^lmax becomes its own instance: the split only happens `if cur and cur_n + n > 1 << lmax` (cut.py:193). So the docstring's "at most 2^lmax entries" (cut.py:23-25) is wrong. I simulated the packing myself, in the graph order of transformer.py:538-579 (q,k,v,o,gate,up,down; the pruned block cuts only k and v). At lmax 24 it reproduces the hard-coded cells exactly: (103,809,024 leaves, 6 instances) and (190,840,832, 11). For the full Llama-2-7B it gives 156 instances, 62 of them with n=25, and a GKR sum of 138,887/p^8 = 2^-230.2. params_table instead uses max(156, ceil(N/2^24)=162) × eps(24) = 141,750/p^8 = 2^-230.1 (params_table.py:57,92). So the stated bound <=2^-230.1 is still a valid upper bound, but footnote b's "162 instances of <=2^24 leaves" (app_params.tex:55-56) is false, and so are ext_v1.tex:74-75,181. evaluation.tex:128-129 makes the same "at most 2^24 leaves" claim. At lmax 27 no Kpre unit of these models exceeds the cap, so the wrong wording mainly affects the lmax-24 (2080 Ti) material. The claim that mode C on 11 GB fails because of gate+up at 2^26 is a plausible inference, not something the evidence shows. One extra point: out["cut"] records n_rounds but not the sum of n, so the GKR sum cannot be rebuilt from the records. Adding instance_bits (recommendation 1) is justified.

### V1-2 [medium, confirmed] # lmax differs across the library, scripts, package and the paper's security numbers

**Where:** code/src/pvi/fullcheck/cut.py:45; code/src/pvi/fullcheck/protocol.py:115-116; code/experiments/8_v1/v1_bytes.py:311; code/experiments/6_improvements/params_table.py:67,90; code/experiments/4_defence_benchmark/slurm/sp2027.sh:132,153; paper_sp2027/sections/ext_v1.tex:181,213-215; paper_sp2027/sections/app_params.tex:55-56

**Evidence.** **Defaults disagree:**
- The library uses LMAX=27 (cut.py:45), and SecurityParams.cut_lmax=27 says "(26 on an 11 GB card)" (protocol.py:116). C2_v1_proof_size.md:46 says "25 on an 11 GB card".
- v1_bytes defaults to 27 (v1_bytes.py:311), and the package passes `--lmax 27` to every V1 cell (sp2027.sh:132). Only the GPT-2 and smoke cells use 20 (sp2027.sh:153,169,200,317).
- params_table defaults to `--lmax 24` (params_table.py:67) and uses the 2080 Ti cells measured at lmax 24 (params_table.py:90).

**The paper's numbers use other values:**
- tab:params' V1 rows and footnote b describe lmax 24.
- E6 (ext_v1.tex:213-215) gives "79 GKR instances of at most 2^26 leaves" for Llama-2-13B, from V1_SPEC (outside the repo).
- tab:v1's text describes 2^24 (ext_v1.tex:181).

**At the final run's lmax 27 (my packing):**
- Llama-2-7B at 2,048 tokens: 21 instances, max n=27.
- Llama-2-13B: 39 instances, not 79. Its GKR term is 2^-231.9, not 2^-230.7. The lookup term 2^-215.3 is unchanged.

So the paper's concrete-security text and table will not describe the proofs the final run produces.

**Recommendation.** 1. Use one lmax constant as the CLI default everywhere: library, v1_bytes, params_table (27).
2. Remove the hard-coded 2080 Ti cells from params_table. Run it with `--v1-cells` on the final L40S V1 records (with the instance_bits of V1-1).
3. Regenerate tab:params' V1 rows, footnote b and E6 from those records.
4. Fix the "26/25 on an 11 GB card" comments. (Effort: small (defaults and wiring), plus regenerating numbers after the run.)

**Verifier.** The defaults disagree as described. cut.LMAX=27 (cut.py:45). SecurityParams.cut_lmax=27 says "26 on an 11 GB card" (protocol.py:115-116), while C2:46 says 25 and the 2080 Ti runs actually used 24. v1_bytes defaults to 27 (:311) and the package passes --lmax 27 (sp2027.sh:132; 20 only at :153,169,200,317). params_table defaults to 24, with cells measured at lmax 24 (:67,90). E6 (ext_v1.tex:213-214) uses lmax 26: my packing at lmax 26 gives exactly 79 instances for Llama-2-13B. At lmax 27 my packing gives 21 instances (max n=27, 2^-232.8) for Llama-2-7B and 39 for 13B (2^-231.9), matching the reviewer. The paper's stated GKR bounds (2^-230.1, 2^-230.7) therefore remain valid conservative upper bounds for the lmax-27 proofs. The problem is descriptive and reproducibility inconsistency, not soundness. There is also a latent fragility: `params_table --lmax 27` with the default cells would mix lmax-24 instance counts with eps(27). On recommendation 2: _v1_counts needs cells at two block counts, but tier_v1 runs Llama-2-7B only at full depth (sp2027.sh:209). Either 2- and 3-block 2048-token cells must be added, or params_table should compute the packing from the shapes.

### V1-3 [medium, confirmed] The Triton GKR prover has never run at the instance size the package uses (2^27 leaves); the smoke does not cover it

**Where:** code/experiments/4_defence_benchmark/slurm/sp2027.sh:132,153; code/tests/test_gkr_triton.py:26,37; code/experiments/8_v1/gkr_bench.py:18; docs/improvements/C2_v1_proof_size.md:157,203; code/src/pvi/fullcheck/cut_protocol.py:405-418

**Evidence.** **What has been run:**
- The largest Triton GKR instance ever measured is 2^25 leaves on the 2080 Ti (C2:157,177,203, with 5.8 GB peak at 2^25). gkr_bench defaults to n 20, 22 and 24 (gkr_bench.py:18).
- The GPU byte-identity tests stop at n=14 (test_gkr_triton.py:26: n ≤ 14; :37: rows=7 and n_col=11, so n=14).
- The earlier `--lmax 27` runs were CPU byte accounting without `--full` (F3a_real_weight_runs.md:256-258; F3b_real_llama.md:152), so no GKR ran.

**What the package needs.** It runs every V1 cell with `--full --lmax 27` (sp2027.sh:132). On Llama-2-7B, OPT-6.7B and Llama-2-13B at 2,048 tokens that builds instances of 2^27 padded leaves: about 21-23 GB each, scaling 160-173 B/leaf (cut_protocol.py:409; C2:157). The smoke gate uses `--lmax 20` on 2-block GPT-2 at T=64 (sp2027.sh:153), and the 2080 Ti rehearsal cannot hold 2^27.

**Code review.** The kernels' index arithmetic looks int64-safe (program_id(...).to(tl.int64) at gkr_triton.py:120,162). The per-block uint64 sums also fit, since nb·2^39 < 2^63. Even so, the first execution at n=26-27 would be inside the main 2-8 Nov tiers.

**Recommendation.** 1. Add a gate step to the L40S smoke: one `v1_bytes.py --full --lmax 27 --model llama2-7b --layers 2 --seq 2048 --modes Kpre C` query (accepted, byte accounting exact, peak memory recorded).
2. Optionally add a GPU test parametrization at n=24-27, gated on free memory.
3. Record the GKR prover used (Triton or eager) and its peak memory in the V1 records. (Effort: small.)

**Verifier.** Confirmed. The GPU tests stop at n=14 (test_gkr_triton.py:26,37). gkr_bench defaults to n 20, 22, 24 (:18). The largest measured instance is 2^25 (C2:157,177,203). The earlier lmax-27 runs were CPU byte accounting without --full (F3b:152-153; F3a:256). The smoke and the rehearsal run V1 only at --lmax 20 on tiny cells (sp2027.sh:153,200). The reviewer actually understates the exposure. With all ops cut, GPT-2 at 512 tokens is one instance of 77,568 rows × 512, so n=26 (about 10.7 GB at 160 B/leaf). OPT-125M at 2,048 tokens already fills a 2^27 instance (65,536 rows × 2,048). So the very first tier_v1 cells (sp2027.sh:204-207) execute n>=26 for the first time, not only the 7B and 13B cells. The Triton index arithmetic uses int64 program ids (gkr_triton.py:120,162), and the CPU verifier would reject a wrong transcript. The risk is therefore mainly OOM, time or schedule, not wrong numbers, which fits medium.

### V1-4 [medium, confirmed] The V1 verifier paths are environment switches read at call or import time, toggled by mutating os.environ, and not recorded; the scatter path is not in the paper

**Where:** code/src/pvi/fullcheck/buckets.py:43-79,83-87,143-169,183-187,219-225,270-278; code/src/pvi/fullcheck/cut_protocol.py:50,175,360,626,634-635; code/src/pvi/fullcheck/logup.py:105-107,133-135; code/src/pvi/fullcheck/protocol.py:408-418; code/src/pvi/fullcheck/native_kernels.py:1281; code/src/pvi/fullcheck/gkr_triton.py:1043; code/experiments/8_v1/v1_bytes.py:330-333,432-446,474; code/experiments/8_v1/v1_phases.py:94-103,204-215

**Evidence.** **Where PVI_V1_BUCKETS is read.** It is read from os.environ at every call, in 7 places across 4 modules: buckets.mode and enabled, cut_protocol `_xbar` (:175), the int8 narrowing (:360), `_final_claims` (:626), logup_ok and verify_instances (logup.py:107,135), and the v0 class `Challenger.ext` (protocol.py:415-417), which imports a V1 module for this.

**How it is toggled.** v1_bytes and v1_phases toggle it by writing os.environ around each run_cut_query (v1_bytes.py:435-446). Values are not validated: `mode()` maps any other string to "1" (buckets.py:73-74), so `--v1-paths 2` silently runs path 1 and records it as "2". The help text calls path 0 "today's verifier" (v1_bytes.py:332).

**Other switches.** PVI_V1_NATIVE (native_kernels.py:1281), PVI_GKR_TRITON (cut_protocol.py:50) and PVI_GKR_PROFILE (gkr_triton.py:1043) are read at import, and none of them is stored in the records.

**What the paper uses:**
- Path "0" is the paper's "2.49 s before" (ext_v1.tex:186-188), measured by the package with `--v1-paths 0 1` (sp2027.sh:162,169,190-192). Keep it as a non-default option.
- Path "scatter" (bucket_sums, bucket_times, `_method` and the scatter branches) is not in the paper; ext_v1.tex:97-98 describes only the per-window tables. `window_tables` (buckets.py:83) is never called.
- `_lw_e_old` and `cut.public_lw_f64` must stay even on the default path: they are the fallback for spans above SPAN (cut_protocol.py:634-635; buckets.py:211-212,266-267).

**No path changes bytes.** `_ext_one` and `_ext_draw` read the same 192-byte stream and pick the same element (protocol.py:419-449), and test_v1_buckets checks that the verdicts agree. Converting the switches therefore changes no transcript or golden output.

**Recommendation.** 1. Replace the environment switches with a frozen `V1Options(final_check='table'|'legacy', native=True, gkr_prover='auto'|'triton'|'eager', profile=False)`, passed to `run_cut_query(..., options=)` and stored on the Verifier. Read it at call time, not from the environment.
2. CLI: `--v1-verifier table legacy` (validated choices, replacing `--v1-paths 0 1`), plus `--v1-native on|off` and `--gkr-prover`. Write the options into every record.
3. Delete the scatter branch, `bucket_sums`, `bucket_times`, `_method` and `window_tables`.
4. Keep `legacy` as the bundle of all four V-V1N parts off, so the 2.49 s measurement stays reproducible. Have Challenger take the draw method as an argument instead of importing buckets.
5. Rename buckets.py to products.py (e.g. v1/products.py) and move `_lw_e_old` and `public_lw_f64` there. (Effort: medium.)

**Verifier.** Confirmed. PVI_V1_BUCKETS is read on every call through buckets.mode()/enabled() (buckets.py:70-79). It is used in _method (:186), cut_protocol :175, :360 and :626, logup :107 and :135, and Challenger.ext (protocol.py:415-417, a lazy import of buckets from the v0 class). That is 7 sites in 4 modules. v1_bytes toggles it by writing os.environ (:435-446). mode() maps any unknown value to "1" (:73-74), and main_path = paths[-1] (:469). So `--v1-paths 2` runs path 1 but is recorded under the key "2" and as v1_path "2". PVI_V1_NATIVE (native_kernels.py:1281), PVI_GKR_TRITON (cut_protocol.py:50) and PVI_GKR_PROFILE (gkr_triton.py:1043) are read at import, and v1_bytes' head record (:368-374) contains none of them. The paper does not mention a bucket or scatter path (grep finds nothing); ext_v1.tex:96-98 describes only the table products. window_tables is never called. Path 0 is the 2.49 s baseline (V_V1N doc :10,285) that the package measures (sp2027.sh:169,190-192). _lw_e_old must stay as the span fallback (cut_protocol.py:634-635; buckets.py:211-212,266-267). Since _ext_one and _ext_draw are byte-identical by test, the refactor changes no transcripts.

### V1-5 [low, confirmed] Generator hygiene: importing it writes into code/src; dead kernels and an unused host function are generated; no drift check

**Where:** code/experiments/8_v1/gen_gkr_triton.py:1-4,207-260,422,654-662; code/src/pvi/fullcheck/gkr_triton.py:1-3,33-36,1094-1099,1144-1145; code/src/pvi/fullcheck/logup_gkr.py:83-84

**Evidence.** **Import side effect.** gen_gkr_triton.py ends with a bare `main()` (:662) with no `if __name__ == '__main__'` guard, and main writes OUT = code/src/pvi/fullcheck/gkr_triton.py (:4,:658). Merely importing the script overwrites a library file.

**Dead generated code:**
- It generates `_round_kernel` and `_fold_kernel` (:207-260) and then throws them away (:655).
- It emits `_eq_table` (:422 → gkr_triton.py:1094), which nothing calls.

**Header and placement.** The generated header cites `V1_SPEC.md` (not in the repo) and has no do-not-edit notice. The generator lives in experiments/ but produces library code, and its output path depends on its depth (`parents[2]`).

**Duplicated pieces.** The generated module carries its own copy of the Triton bootstrap, a process-wide `sysconfig.get_paths` patch (gkr_triton.py:33-36). It also has its own `_absorb` (gkr_triton.py:1144-1145), an identical copy of logup_gkr._absorb (logup_gkr.py:83-84), on which the byte-identity of transcripts depends.

**No drift check.** No test checks the committed file against the generator. I verified in memory that they are identical today (1,306 lines).

**Recommendation.** 1. Move the generator to scripts/ (or `pvi/fullcheck/v1/_gen_gkr_triton.py`), with a `__main__` guard, `--out` and a `--check` mode that exits 1 on any difference. Add a fast CPU test that runs `--check`.
2. Drop the dead kernels and `_eq_table`.
3. Import `_absorb` from logup_gkr, and take the Triton bootstrap from one shared `kernels/_triton.py`.
4. Give the generated file the header "Generated by scripts/gen_gkr_triton.py; do not edit", and describe the algorithm without V1_SPEC references. (Effort: small.)

**Verifier.** Every fact is accurate. The script ends in a bare `main()` (gen_gkr_triton.py:662) that writes OUT = parents[2]/src/pvi/fullcheck/gkr_triton.py (:4,658). It generates _round_kernel and _fold_kernel (:207,239) and then filters them out (:655). _eq_table is emitted (gkr_triton.py:1094) and never called. The header cites V1_SPEC.md and says "Generated by" but carries no do-not-edit notice. The sysconfig bootstrap appears in both gkr_triton.py:33-36 and codec_kernels.py:35-38. _absorb is identical to logup_gkr.py:83-84. I also regenerated the file in memory and it is identical today (1,306 lines). I lower the severity to low. Nothing imports the script (pytest collects only test_*.py), so the import side effect is latent, and functional drift would be caught by the transcript-identity tests (test_gkr_triton*.py). The one practical risk is that the planned directory refactor breaks the depth-based parents[2] path. The recommendations are sensible branch cleanup.

### V1-6 [low, not verified (low)] Appendix says check F4 is compared by cross-multiplication; the default code compares batch-inverted sums

**Where:** paper_sp2027/sections/appendix_v1.tex:82-83; code/src/pvi/fullcheck/logup.py:96-118

**Evidence.** appendix_v1.tex:82-83 states (F4) as "compared by cross-multiplication". By default (PVI_V1_BUCKETS≠0), logup_ok computes Σp0·q0^{-1} and Σμ·(α−τ)^{-1} with inv_many and compares the two (logup.py:109-115). It falls back to cross-multiplication only if a denominator is zero. All denominators are nonzero (q0 by check G, α−τ because α ∉ F+ξF), so the two tests are equivalent and soundness is unaffected.

**Recommendation.** Reword (F4): "compared after batch inversion; every denominator is non-zero, so this equals cross-multiplication". Alternatively, describe the legacy option as the cross-multiplication variant. (Effort: trivial.)

### V1-7 [low, not verified (low)] The default batched GKR check raises an exception on a cut plan with no instances

**Where:** code/src/pvi/fullcheck/logup_gkr.py:258-263; code/src/pvi/fullcheck/logup.py:126-142

**Evidence.** `logup.verify_instances([], [], ch, batched=True)` raises `RuntimeError: stack expects a non-empty TensorList` from `torch.stack` at logup_gkr.py:263. I ran this call. The legacy path returns `[]`, and logup_ok handles empty roots (logup.py:103-104). A query whose cut plan is empty therefore crashes the default verifier instead of returning a verdict. None of the paper's models produces an empty plan.

**Recommendation.** Return `[]` early in verify_batch when `not trs`, and add a test with an empty cut plan. (Effort: trivial.)

### V1-8 [low, not verified (low)] x̄ cache in the final check is keyed by tensor pointer and shape, not by the graph's input name

**Where:** code/src/pvi/fullcheck/cut_protocol.py:166-184,486-491

**Evidence.** `_xbar` caches X^T e under `(xin.data_ptr(), tuple(xin.shape), id(e))` (:171), ignoring strides and dtype. Two inputs that share a storage pointer and shape but differ in strides (a transposed view of a square tensor) would share a wrong x̄. That x̄ feeds both S_ℓ in the final check and the column check of col matrices. This cannot happen with the current graphs, but a pointer-keyed cache on a soundness path is fragile.

**Recommendation.** Key the cache on `(op.inputs[0], instance)`, the derive environment's name for the tensor, or on `(data_ptr, shape, stride, dtype)`. (Effort: trivial.)

### V1-9 [low, not verified (low)] V1 modules: diary and plan references, test-only functions, and a private coupling to protocol.py

**Where:** code/src/pvi/fullcheck/cut.py:1; cut_protocol.py:1-12,42-43,168,221-225,242,286,360,529,595,615-617; logup.py:1,13,46,99,121; logup_gkr.py:1-2,145,248,326-327; buckets.py:1,8-21,44,65,71,78

**Evidence.** **Diary and plan references.** I counted 32 hits for "today's", "V-V1N part n", "V1_SPEC.md", "SP2027_PLAN2.md" and "docs/improvements": cut_protocol.py 11, buckets.py 10, logup_gkr.py 5, logup.py 3, cut.py, gkr_triton.py and the generator 1 each. V1_SPEC.md and `spec_toy.py` are outside the repo.

**Test-only functions:**
- `logup.multiplicities` (:46); production uses cut.split_counts.
- `logup.prove_instances` (:121).
- `logup_gkr.verify_rounds` (:326), the oracle for spec_toy.

**Private coupling.** cut_protocol imports 10 private helpers from protocol (:42-43). It also sets `Verifier._cut_plan` and `.sampler` for the length of a query (:221-225,242,286), while protocol carries V1 hooks of its own (`_CUT_NARROW` and derive's cut branches at protocol.py:1002-1053; `Challenger.ext(nonbase=)`).

**Moving the modules is safe.** No V1 function enters a graph digest or a transcript label, so moving these modules changes no golden output.

**Recommendation.** 1. In the v1/ subpackage (cut, logup, gkr, gkr_triton, products, query), rewrite the docstrings against the paper's sections and the repo's own docs, and describe paths as options, not as "today's".
2. Move the test-only functions into tests/.
3. Replace the private imports and the mutable Verifier fields with a small explicit hook API: a cut plan passed to derive, and decode or absorb helpers exported from protocol. (Effort: small-medium.)

## Integer models and kernels

Area: integer models and kernels (fullcheck/{transformer,real_weights,real_llama,quantize,models,datasets,native_kernels,attention_kernels,codec_kernels}.py, plus graph.py and field.py where the exactness claims live) and experiments/9_quality/{opt_builder,common,ppl_2048,ppl_llama,zeroshot}.py. This was a read-only pass. I read the code and grepped; the only thing I ran was one 3-line CPython check of how closure variables are ordered. No tests and no model builds.

Exactness: I found no wrong-result bug. I re-derived every stated bound against the code:
- int8 GEMMs: blocks of 2^16 terms in int32, with a self-check at the bound (graph.py:77-92, field.py:332-366).
- Native int8 GEMM: K < 2^17 (native_kernels.py:804-808, 1271-1273).
- int32 and float32 attention: scores up to 2^21, P V below (pmax+T/2)·128 < 2^24, tot·2 below 2^31 (transformer.py:249-284, attention_kernels.py:15-20).
- Native reciprocal divisions with one integer correction each way (native_kernels.py:134-142, 459-507); -fwrapv makes int64 wraparound the same as torch (native_kernels.py:890).
- TF32 does not break exactness either: int8 values and p ≤ 2047 fit TF32's 11-bit significand.
- Tests cover pmax 2047 on the native CPU path and the Triton GPU path (test_native_cheap.py:217-250, test_gpu_exactness.py:148-178).

The real problems are elsewhere:
- **F1 (high), graph digest.** It names each cheap op by Python module and qualname plus its captured constants, never by its code. An unchanged digest therefore does not mean an unchanged computation; ppl_2048 already relies on this. Many innocent edits do change it. No test pins a digest value.
- **F2 (high), Llama/Qwen recipe.** MAIN_RECIPE is OpenLLaMA's setting only. The TinyLlama and Qwen3 ratios in the paper need settings chosen per model on the WikiText test split, and those settings are recorded only in the diary. The prefix rule and data pipeline live in experiment scripts that the run package imports. head_eq is used in every Qwen3 number but described nowhere in the paper.
- **F3 (medium), tab:quality.** Its numbers come from the experiment copy opt_builder (with its own 11-bit attention), not from CERTIFIED_RECIPE. Only a tiny-model test shows the two are the same graph.
- **F4 (medium), F5 (medium), duplication and private imports.** There are four builders whose arithmetic differs without saying so, two different structure checks, and private imports plus patching of module globals.
- **F6 (medium), switches.** Kernel switches are environment variables read at import. Fallbacks are silent, and only bench.py records the switch values.
- **F7 (medium), probability scale.** Cost graphs use 8-bit probabilities and certified graphs 11-bit. This is a decision to make before the 26 Oct freeze.
- **F8, F9 (low).** The integer-semantics appendix is incomplete, and there is dead or stale code.

Order of work for the refactor: settle the digest policy (F1) before capturing goldens. Goldens must hash integer outputs (logits and claims), not only digests. Under the current scheme (v1), the methods that define lambdas cannot move, so the builder merge (F4) depends on that decision.

### INT-1 [high, partly] # Graph digest binds Python names and captured constants, not op semantics: same digest can mean a different computation, and many harmless edits change it

**Where:** code/src/pvi/fullcheck/graph.py:353-391,433-448; builders in transformer.py, real_weights.py, real_llama.py, quantize.py

**Evidence.** _canonical hashes a function as f"{__module__}.{__qualname__}" plus __defaults__, __kwdefaults__ and closure cell values; __code__ is never hashed, nor the helpers a lambda calls (graph.py:382-386). An unchanged digest with a changed computation is already in use: ppl_2048.py:127-139 (_attention_as) swaps opt_builder._attention inside graphs that are already built. The digest is computed once (:265), the float-softmax and attention_var runs happen under the swap (:277-280), and every cell records the same digest (:288-293). Edits that change the digest beyond module moves (inventory H1): (a) qualnames carry the enclosing builder method, e.g. _RealBuilder.norm.<locals>.<lambda> (real_weights.py:146,162), _Builder.attention (real_llama.py:387-393), build_decoder.<locals> (transformer.py:543-578), CNNBuilder.requant.<locals>.fn (quantize.py:126); (b) closure cells are hashed in co_freevars order, which CPython sorts by name (checked: a lambda capturing zeta and alpha has freevars ('alpha','zeta')), so renaming hq/hkv/dh (transformer.py:563-565) can reorder them; (c) _last_position is itself an op function (transformer.py:474,476; real_weights.py:114; real_llama.py:246,248), so moving it changes every decoder digest; (d) three 'prob_max == PROB_MAX' branches exist only to keep pre-F2a digests (transformer.py:562-565; real_weights.py:256-260; real_llama.py:389-393). No test pins a digest value; all comparisons are relative (tests/test_prob_max.py:72,143; tests/test_real_llama.py:147,591; tests/test_real_weights.py:370). Published digests that any move invalidates: scripts/artifact_README.md:73-74,77. The paper says the digest describes 'every operation in order with its inputs, output and constants' (paper_sp2027/sections/defence.tex:186-191); the code docstring is more honest: 'a cheap op's function by name' (graph.py:436-437).

**Recommendation.** Settle the digest policy before capturing goldens. Preferred option: give every cheap-op function an explicit semantic tag, as requant_fn already does with _pvi_requant (graph.py:154-175). Examples: ('attention', pmax, m_s, m_o, heads, kv_heads, dh, prefix k/v), ('norm', center, k, gain), ('lut', table), ('rope', theta, offset rule), ('last_positions', n), ('glu', mult). Hash the tag plus a semantics version under a new 'pvi/graph/v2' header, and drop module.qualname. The layout then survives refactoring and the paper's sentence becomes true. After that, regenerate the D6a and artifact_README digests. If the authors keep v1 instead, freeze: the five modules' paths, the builder class and method names, the names of captured variables, the home of _last_position, and the three compatibility branches. In both cases, goldens must hash integer outputs (logits and every claim) for tiny graphs of each builder: GPT-2/OPT/Llama/Qwen random shapes, tiny HF OPT, Llama and Qwen3 with a prefix, and one CNN. Add a test that pins those hashes and digests. (Effort: medium (1-2 days including regenerating digests; under FS, mode-C proof bytes change only by sampling noise).)

**Verifier.** The mechanism is as described. graph.py:382-386 hashes f"{__module__}.{__qualname__}", __defaults__, __kwdefaults__ and closure cells, and never __code__. I checked on Python 3.10 that co_freevars come out sorted by name: a lambda reading zeta, alpha, hq, hkv and dh gives ('alpha','dh','hkv','hq','zeta'). The lambda at transformer.py:563 captures hq, hkv and dh as closure cells, not defaults. The other evidence also holds: _last_position is used as an op fn (transformer.py:474,476; real_weights.py:114; real_llama.py:246,248), the three compatibility branches carry 'digest unchanged' comments (transformer.py:562; real_weights.py:258; real_llama.py:389), no test pins a digest value, and the digests at artifact_README.md:73-74 are published. The ppl_2048 case is also accurate: the digest is taken once at :265, swapped under _attention_as at :279 and written into every cell at :293. That swap is an experiment's diagnostic label, though, not a protocol path. Two points are overstated. (1) Soundness: the verifier recomputes cheap ops with its own code, and the client rebuilds the graph from the public recipe. The same digest with other code therefore needs a different library version or in-process patching, which a prover cannot choose adaptively. This is a versioning and reproducibility problem, not a weak Fiat-Shamir gap. (2) The paper is not false: defence.tex:188-191 lists inputs, output and constants, which are all hashed, and ext_proofs.tex:204 even says 'function'. It only leaves out that a function is identified by its Python name. High is still right, because this blocks the planned refactor: moving modules or methods changes every transcript and the published digests, so the policy must be settled before goldens are taken. The recommendation (semantic tags under a v2 header, or freezing the names) is sound.

### INT-2 [high, confirmed] MAIN_RECIPE does not reproduce the paper's Llama/Qwen ratios: per-model settings exist only in the diary, the prefix rule lives in experiment scripts, and head_eq is not described

**Where:** code/src/pvi/fullcheck/real_llama.py:458-464,616-622,728-731; code/experiments/9_quality/ppl_llama.py:67-69,252-268; code/experiments/8_v1/v1_bytes.py:62-79; paper_sp2027/sections/quality.tex:79-88

**Evidence.** MAIN_RECIPE = alpha 0.55, smooth_pct 99.9, pct/pct_att 99.99, glu_smooth 0.2 (real_llama.py:458). That is OpenLLaMA-7B's best cell (docs/improvements/F2b_Q_real_llama.md:1171-1173). The other paper ratios used other settings: TinyLlama x1.122 at alpha 0.75 with glu 0.2 (F2b:1143-1146); Qwen3-1.7B x1.40 at alpha 0.45 with glu 0.3, and Qwen3-4B x5.3 with glu 0.5 (F2b:1258-1259). The paper presents one recipe ('With them ... TinyLlama within x1.12 and Qwen3-1.7B within x1.40', quality.tex:79-88; 'the main recipe of Section sec:quality', ext_discussion.tex:58-69). The prefix is not part of the library recipe: the docstring points to ppl_llama.prefix_of (real_llama.py:461-462). The run package reaches it by importing experiment-script privates: v1_bytes.py:70-77 uses pl.token_ids, pl.p2._text, pl.p2._windows and pl.prefix_of, and loading ppl_llama also imports the OPT copy opt_builder (ppl_llama.py:67-69). ppl_llama, the script that produced the ratios, never uses MAIN_RECIPE: its defaults are alpha 0.55, no glu_smooth and no prefix (ppl_llama.py:155-168,183,252-268). head_eq is on by default (real_llama.py:618) and rescales the rows of W_q/W_k per head before the Qwen3 q/k norms (real_llama.py:445-455,728-731). Every Qwen3 number uses it, yet no paper section describes or cites it, and it is missing from the int8-rule table (F2b:1244-1254). Selection protocol: OPT alpha and the scalar gains were chosen on all 140 test windows (docs/improvements/F2a_quality_2048.md:174,199). The Llama/Qwen settings were 'decisions taken at 2,048 tokens' on the test windows (F2b:1238). The paper mentions only train-split calibration (quality.tex:25-26,66-67).

**Recommendation.** Add a recipe registry in the library. Each entry holds the builder kwargs, the calibration spec (split, characters, windows, seq, special-token rule) and the prefix rule ('prefixquant', eta 64). Entries: opt-certified (CERTIFIED_RECIPE); one main entry per Llama/Qwen model (open_llama_7b_v2 0.55/0.2, TinyLlama 0.75/0.2, Qwen3-1.7B 0.45/0.3, Qwen3-4B 0.45/0.5, and Llama-2-7B/13B as the final run uses); opt-legacy as a non-default option. Move token_ids, prefix_of and the WikiText loading into a library data module, so neither the run package nor perf.py imports ppl_llama or ppl_2048. Make 'ppl_llama --recipe main' the default. In the paper: state that alpha and glu_smooth are chosen per model and on which split (or re-select them on a held-out train/validation split), and add one clause on head_eq (RMSNorm is invariant to a per-head scale) or drop it. (Effort: medium (the registry and data move change no number; re-selecting on a validation split would change the Llama/Qwen ratios).)

**Verifier.** MAIN_RECIPE at real_llama.py:458 is OpenLLaMA-7B's best cell (F2b_Q_real_llama.md:1171-1173). TinyLlama x1.122 used alpha 0.75 with glu 0.2 (:1143-1146). Qwen3-1.7B x1.40 used 0.45 with glu 0.3, and Qwen3-4B used glu 0.5 (:1258-1259). No library entry or run-package command reproduces those three ratios: the quality tier covers only Llama-2-7B and 13B (sp2027.sh:391-396). The paper never says that alpha or glu_smooth are chosen per model (quality.tex:79-88; ext_discussion.tex:65-68). head_eq is on by default (real_llama.py:618, used at :728-731; ppl_llama cells() passes not --no-head-eq). Every Qwen3 ratio uses it, yet it appears nowhere in the paper and is missing from the int8-rule table (F2b:1244-1250). By the authors' own rule (memory int8-model-changes-rule: constants-only with a citation), that leaves the Qwen3 claims unsupported. Hyperparameters were selected on the test windows: for OPT across all 140 windows (F2a_quality_2048.md:174,189-199), and for Llama/Qwen per F2b:1238. The final run repeats this with alphas 0.55/0.65 (sp2027.sh:394). The paper mentions only train-split calibration (quality.tex:25-26). The chain is confirmed: v1_bytes.py:70-77 uses pl.token_ids, pl.p2._text, pl.p2._windows and pl.prefix_of; ppl_llama.py:68-69 imports opt_builder and ppl_2048; perf.py:237-238 and gen_bench.py:209-210 reach them through v1_bytes. Small corrections: the ppl_llama defaults are at :294-305,322, not :155-168,183, which is smoke_model. The MAIN_RECIPE docstring does mention TinyLlama's alpha 0.75 (:461-464). MAIN_RECIPE does reproduce the main Llama claim (OpenLLaMA x1.073) and the Llama-2-7B real tier, and it matches the q-tier flags.

### INT-3 [medium, confirmed] tab:quality numbers come from the experiment copy opt_builder (with its own 11-bit attention), not from CERTIFIED_RECIPE, and equivalence is checked only on a tiny model

**Where:** code/experiments/9_quality/ppl_2048.py:14,43,50,227,263,339-377; code/experiments/9_quality/opt_builder.py:272-293,351,459-499; code/experiments/9_quality/zeroshot.py:69,331,365

**Evidence.** Every ppl_2048 cell is built with ob.build_opt (ppl_2048.py:263). The library's CERTIFIED_RECIPE is never referenced. Defaults are the alpha 0.55 percentile recipe (SMOOTH :50, --alphas 0.55 :227), and the round-2 flags behind the certified column were not recorded. At 11 bits opt_builder runs its own attention_var (opt_builder.py:272-279,289-293), while the certified graph uses transformer._attention(..., P) (real_weights.py:256-260). --check compares the two builders only at prob_max 255 with the alpha 0.55 percentile recipe (ppl_2048.py:339-342), and its 11-bit part builds only the copy (:372-377). So 'tab:quality is the certified model' rests on tests/test_recipe_equivalence.py:31-49 (2 layers, d=32, 40 tokens). The copy cannot reproduce C_M's digest (ppl_2048.py:364-366), so the quality records do not carry the published digest (scripts/artifact_README.md:73-74). zeroshot.py defaults to --builder f2a with F2A_BEST alpha 0.5 (zeroshot.py:69,331,365), not the certified builder behind tab:zeroshot. ppl_llama imports the OPT copy only for graph_report (ppl_llama.py:67,369), a generic claim-bound and V1-P3 check (opt_builder.py:459-499). The DEFAULTS key emb_row is never read (opt_builder.py:351).

**Recommendation.** Build the scalar and certified cells, and the 8- vs 11-bit comparison, with real_weights.build_opt_from_hf (norm_gain=g or **CERTIFIED_RECIPE; prob_max 255 or 2047). Keep opt_builder only for the non-default diagnostics the paper reports: row_pow2 scales (ext_generation.tex:240-245), the ideal softmax and the probability mass (quality.tex:54-56). Alternatively, fold those into real_weights as non-default options and delete the copy. Extend --check to the certified recipe at 11 bits on one full OPT-125M window. Make zeroshot default to --builder paper. Move graph_report into the library next to protocol.claim_bound. Re-run OPT-125M certified once to confirm it equals the stored F2a round-2 value. (Effort: medium.)

**Verifier.** Every ppl_2048 cell goes through ob.build_opt (ppl_2048.py:263). The defaults are SMOOTH with alpha 0.55 (:50, :227). At 11 bits opt_builder takes the attention_var path (opt_builder.py:272-279,289-292), because default is only (8,16). --check compares the two builders only at prob_max 255 with the 0.55 percentile recipe (:339-342), and its p11 part builds only the copy (:372-377). The only link between the builders at the certified recipe is the tiny test (test_recipe_equivalence.py:31-49: 2 layers, d=32, 40 tokens). The copy cannot reproduce the published digest (:364-366): the archived round-2 OPT-125M record has digest c5c90091..., the artifact README has 16de86ea.... zeroshot defaults to f2a with alpha 0.5 (zeroshot.py:69,331,365), but tab:zeroshot was run with --builder paper (F2c_zero_shot.md:155-163, digest 16de86ea), so only the default is stale. emb_row is never read (opt_builder.py:351 is its only use). graph_report is the only ob use in ppl_llama (:369). One correction: the round-2 flags were recorded. The archived records (server_archive s3 clone, ppl2048_search2_opt-*.jsonl) carry opts={'norm_mode':'smooth','alpha':0.6,'smooth_pct':100,'pct':100,'pct_att':100,'prob_bits':11} and tag 'smooth-max'. They are just not committed: the branch holds only round 1 under code/artifacts/results/9_quality/, and no script or doc keeps the command. Medium is right.

### INT-4 [medium, confirmed] Four duplicated decoder builders with arithmetic that differs without saying so, and two structure checks of different strictness

**Where:** code/src/pvi/fullcheck/transformer.py:428-494; real_weights.py:56-163,291-335; real_llama.py:110-116,176-443,813-844,909-935; code/experiments/9_quality/opt_builder.py:129-294

**Evidence.** name, _run, cheap, requant, last, release, mat, to_int8, residual and norm are copied across transformer._Builder, real_weights._RealBuilder, real_llama._Builder and opt_builder._Builder. Their arithmetic differs in three places: (1) Weights with a scalar input scale are quantised in float32 for OPT ((w / s_w) with w float32, real_weights.py:94-95; copied in opt_builder.py:166-168) but in float64 for Llama ((w.double() / s_w), real_llama.py:226-227). (2) Percentiles read the first 2^24 entries (real_weights.py:127,253) against evenly spaced rows (real_llama.py:110-116). (3) The smoothing statistic uses the first 2^14 rows (real_weights.py:156) against evenly spaced rows (real_llama.py:279). There are two matches_benchmark_graph functions: OPT's checks only kinds, shapes, bias and notes (real_weights.py:313-335); Llama's also checks wiring, names and claim columns (real_llama.py:909-935). There are two real_logits: real_weights.py:291-310 materialises [B,T,V], real_llama.py:813-844 works in chunks. Merge hazards: methods that define lambdas carry qualnames (INT-1), and switching OPT to float64 would change the certified OPT int8 weights and their published hash (artifact_README.md:73-74).

**Recommendation.** Write one DecoderBuilder core (naming, run, release, last, requant via requant_fn, residual) with a scale policy (target std for random weights, real scales for checkpoints) and per-family block code. Keep OPT's float32 quantisation as a documented rule, or change it deliberately and regenerate the certified hashes. Keep one strict matches_benchmark_graph with an allow_lm_head_bias flag for OPT, and one chunked real_logits. Do this only after the INT-1 decision: under digest v1, the methods that define lambdas (norm, attention, rope, swiglu, head_norm) cannot move. (Effort: large (2-3 days with golden comparison).)

**Verifier.** All three arithmetic differences are real. (1) OPT quantises a scalar-input weight in float32, (w / s_w) at real_weights.py:95 with w passed as .float() (copied at opt_builder.py:167); Llama uses w.double() / s_w (real_llama.py:227). (2) Percentiles use the first 2^24 entries (real_weights.py:127,253) against evenly spaced rows via _rows/_top (real_llama.py:100-115). (3) The smoothing statistic uses n[:1 << 14] (real_weights.py:156) against _rows(..., _ROWS_NORM) (real_llama.py:279). The two matches_benchmark_graph differ as described: OPT checks kinds, shapes, bias and notes (real_weights.py:313-335); Llama also checks wiring, layout and claim_columns (real_llama.py:909-935). real_logits materialises the output (real_weights.py:291-310) against chunked lean_logits (real_llama.py:811-844). One nuance: (2) and (3) do not affect the certified OPT models, because CERTIFIED_RECIPE has pct, pct_att and smooth_pct all at 100, so maxima are used. Only (1) touches certified OPT bytes (the embedding, out_proj and fc2 take the scalar-input path), which is the merge hazard named. 'Without saying so' is fair: neither builder documents the other's rule. A large refactor right before the 26 Oct freeze is risky. Medium is right, and it should wait until after INT-1, as stated.

### INT-5 [medium, partly] The core integer ops are private and imported across modules, and experiments change behaviour by patching module globals

**Where:** code/src/pvi/fullcheck/real_weights.py:46-49; real_llama.py:77-80; code/experiments/9_quality/opt_builder.py:46; code/experiments/9_quality/ppl_2048.py:127-139; code/experiments/7_reexec/reexec.py:229-236; transformer.py:46-47

**Evidence.** Every certified graph calls transformer privates. real_weights.py:46-49 imports _attention, _isqrt, _last_position, _lut and _norm_int; real_llama.py:77-80 adds _rope; opt_builder.py:46 imports _attention, _isqrt, _lut and _norm_int. ppl_2048.py:127-139 changes the attention of built graphs by assigning opt_builder._attention. This works only because the builders' lambdas look up module globals at call time (the call at opt_builder.py:290 uses the import at :46). On the run-package path, reexec.py:229-236 patches native_kernels._GEMM to get the int_fp32gemm variant. transformer._FUSED_ATTN and _CPU_ATTN (transformer.py:46-47) are module globals that tests patch. A re-export shim would silently break every one of these patches.

**Recommendation.** Add a public integer-ops module (e.g. fullcheck/intops.py with attention, prefix_attention, norm_int, isqrt, lut, rope, requant and residual) that every builder imports. Helpers called inside lambdas can move without changing any digest, because only the lambda's qualname is hashed. _last_position and _last_positions are op functions themselves: keep the module that defines them, or tag them (INT-1). Replace global patching with explicit parameters: an attention_impl= argument on the evaluator for the ideal-softmax diagnostic, and kernel options as in INT-6. (Effort: small-medium.)

**Verifier.** The core is confirmed. Private transformer ops are imported across modules: real_weights.py:46-49 (_attention, _isqrt, _last_position, _lut, _norm_int), real_llama.py:77-80 (adds _rope), opt_builder.py:46. Behaviour is changed by patching module globals: ppl_2048.py:127-139 assigns ob._attention, which works because the opt_builder lambdas (:290) look the global up at call time. reexec.py:229-236 flips native_kernels._GEMM for the int_fp32gemm RX variant, on the run-package path. Two parts are inaccurate. (a) No test patches transformer._FUSED_ATTN or _CPU_ATTN: a grep finds them only at transformer.py:46-47,322,325,348. Tests patch native_kernels globals instead (_CHEAP, _DECODE, _AVX, _GEMM, _library at test_native_cheap.py:18-290 and test_claimcodec.py:1065). (b) 'A re-export shim would silently break every one of these patches' is too strong. The ppl_2048 patch writes to opt_builder's own namespace, so it survives moving _attention behind a shim. The real hazard is the bool globals such as native_kernels._GEMM and the test-patched flags: a patch on a shim leaves the real module unchanged, so int_fp32gemm would silently time the native GEMM. That makes it medium.

### INT-6 [medium, confirmed] Kernel switches are environment variables read at import; fallbacks are silent and not recorded by the package's timing harnesses

**Where:** code/src/pvi/fullcheck/transformer.py:46-47; graph.py:61; native_kernels.py:901-905,960-962,995,1065-1066,1175,1250,1264,1281; codec_kernels.py:35-44; datasets.py:32-35

**Evidence.** Variables read at import: PVI_FUSED_ATTN and PVI_CPU_ATTN (transformer.py:46-47); PVI_LEGACY_WEIGHT_KEY (graph.py:61); PVI_NATIVE_ATTN, which disables the whole native library (native_kernels.py:901); AVX512 (:995); CHEAP and INT32_CLAIMS (:1065-1066); DECODE (:1175); MALLOC_TUNE (:1250); GEMM (:1264); V1_NATIVE (:1281); PVI_PY_INCLUDE, which patches sysconfig.get_paths process-wide (codec_kernels.py:35-38); the dataset roots (datasets.py:32-35). Fallbacks are silent. _library() returns None on non-Linux, with no compiler, or on any build error (native_kernels.py:901-905,960-962); without Triton the code falls back to torch (codec_kernels.py:40-44, transformer.py:322); if int8_ok is False it uses float32 GEMMs (graph.py:296-302). All paths give the same integers, so only the timings move (B5 reports 4.6x). Only bench.py records PVI_* and the precision settings (bench.py:172-183). perf.py:204-205, v1_bytes.py:347-348, gen_bench.py:201-202, reexec.py:208-209 and serve_ab.py:895-896 store only git_sha. The smoke tier checks native availability only when CXX is set, and only on the smoke node; it prints int8_ok without failing on it (sp2027.sh:148-153). Constraint for the fix: the digest hashes lambda defaults and closure cells (graph.py:384-386), so an option passed through a cheap-op closure would make digests depend on the kernel choice.

**Recommendation.** Add one frozen KernelOptions (fused_attn, native per kernel, avx512, int32_claims, gemm, malloc_tune, v1_native). Kernels read it at call time through options.current(), and scripts set it from a CLI flag (--kernels key=val,...) whose defaults are today's. Remove PVI_CPU_ATTN with the hybrid kernel, PVI_INT32_CLAIMS and PVI_LEGACY_WEIGHT_KEY. Rename PVI_NATIVE_ATTN. Keep CXX, PVI_NATIVE_DIR and PVI_PY_INCLUDE as platform environment variables, read in one place. Every record stores kernels = {options, native_available, avx512, triton, int8_ok}. The package passes --require-native/--require-triton so that a fallback fails the job. Never pass options through cheap-op closures. (Effort: medium.)

**Verifier.** The read sites check out: transformer.py:46-47; graph.py:61; native_kernels.py:995,1065,1066,1175,1264,1281; codec_kernels.py:35-38, a process-wide sysconfig patch; datasets.py:32-35. Correction: PVI_NATIVE_ATTN (native_kernels.py:901, inside _library()) and PVI_MALLOC_TUNE (:1250) are read when first called, not at import. The silent fallbacks are confirmed: _library() returns None on non-Linux, with no compiler, or on OSError/CalledProcessError (:901-905,962); there is a triton=None path (codec_kernels.py:40-44); and MatOp.compute falls back to exact_matmul when int8_ok fails (graph.py:296-302). Only bench.py records the environment (bench.py:173-184 keeps PVI_* vars). perf.py:204-205, v1_bytes.py:347-348, gen_bench.py:201-202, reexec.py:208-209 and serve_ab.py:895-896 (in 7_reexec/) record git_sha plus host/GPU, and no kernel state. The smoke tier checks native availability only if CXX is set (sp2027.sh:148-150, and CXX is passed only when set, :82-85). It prints int8_ok without failing on it (:151). On the L40S run, a silent fallback would yield slower but 'valid' timings that go into the paper. This sits at the high end of medium. The caution about passing options through cheap-op closures (they are hashed, graph.py:384-386) is correct.

### INT-7 [medium, confirmed] # Cost graphs use 8-bit attention probabilities while every certified graph uses 11-bit; real-vs-random comparisons mix the two

**Where:** code/src/pvi/fullcheck/transformer.py:55-64,496-509,562-565; code/experiments/6_improvements/perf.py:252; code/experiments/8_v1/v1_bytes.py:363; code/experiments/4_defence_benchmark/slurm/sp2027.sh:321-334

**Evidence.** PROB_MAX is 255 for benchmark graphs and CERTIFIED_PROB_MAX 2047 for certified ones (transformer.py:55-64). build_decoder defaults to PROB_MAX (:496-497), and no script passes prob_max (perf.py:252, v1_bytes.py:363; nothing in bench, gen_bench, serve_ab or reexec). So every timing table, the RX 'int' baseline and the real tier's 'real-random' twins (sp2027.sh:321-334) run 8-bit graphs, while the real cells run 11-bit ones. The paper compares the two: '17% smaller than with random weights', 'within 10% of random weights' (ext_discussion.tex:60-63). The difference is disclosed (appendix_defence.tex:74; ext_proofs.tex:136) and argued to cost nothing (quality.tex:55-56). The authors' int8 rule, however, asks that the cost tables' graphs be the main algorithm's. The compatibility branches of INT-1 exist only for this.

**Recommendation.** Decide before the 26 Oct freeze. Option A: build all cost graphs at 2047. The bounds hold up to 2047 (transformer.py:249-255, 509), so cost graphs then equal the certified structure and the compatibility branches go; note that stored course-era base columns (255) would then mix with the new cells. Option B: keep 255 and say in the evaluation that cost graphs use 8-bit probabilities and that the real-vs-random rows also differ in P_max. (Effort: small.)

**Verifier.** PROB_MAX=255 and CERTIFIED_PROB_MAX=2047 (transformer.py:55-64). build_decoder defaults to PROB_MAX (:496), and no script outside 9_quality passes prob_max (grep over code/experiments; perf.py:252, v1_bytes.py:363). The real tier's real cells are built at 2047: --recipe certified goes to build_opt_from_hf, and Llama goes to MAIN_RECIPE through build_llama_from_hf, both defaulting to CERTIFIED_PROB_MAX. Their 'real-random' twins (sp2027.sh:321-334,353-358) are 255 graphs. So the comparisons '17% smaller than with random weights' and 'within 10%' (ext_discussion.tex:60-63) also differ in P_max, and the paper does not say so. The 255/2047 split itself is disclosed (appendix_defence.tex:74; ext_proofs.tex:136). The claim that 11 bits 'change neither the proof's size nor its cost' (quality.tex:55-56) is only asserted in F2a_quality_2048.md:157-158, with no measurement cited there. Proof bytes depend on claim values through the codec, so a small effect is possible. The int8 rule's stated aim (memory: 'reviewers must not see quality numbers from a graph that differs from the one whose costs are compared') applies, though the difference is a constant only, with no structural change. Both options are reasonable. Option A also removes the three compatibility branches of INT-1.

### INT-8 [low, not verified (low)] The integer-semantics appendix omits several implemented operations, though the paper says every operation is listed there

**Where:** paper_sp2027/sections/defence.tex:71-75; paper_sp2027/sections/appendix_short.tex:157-173

**Evidence.** The appendix covers requant, residual, norm, tables, RoPE and attention. It omits: the SwiGLU product requantisation (graph.py:131-137; transformer.py:573-575; real_llama.py:414-416); the embed-to-residual multiplier (transformer.py:525-527; real_weights.py:235; real_llama.py:711); the per-head q/k norm (transformer.py:541-546); the prefix's int8 key/value constants in attention (real_llama.py:151-161,386-388); the rounded LayerNorm mean floor((sum x + d/2)/d) (transformer.py:175); and the CNN operations: ReLU requantisation to [0,127] with per-channel multipliers (quantize.py:116-131), add+ReLU with two multipliers (quantize.py:133-147), max and rounded-average pooling (quantize.py:149-166).

**Recommendation.** Add one line per missing operation. The completed list doubles as the specification for INT-1's semantic tags. (Effort: small.)

### INT-9 [low, not verified (low)] Course-only configs and kernels, misplaced docstrings and plan/job references in the model modules

**Where:** code/src/pvi/fullcheck/transformer.py:46-48,57,61,65,90,96,98,108,117-121,325,348,517-520,582-583; native_kernels.py:806,1033-1059; real_weights.py:1,148,175; real_llama.py:1,458-464,479; code/experiments/9_quality/opt_builder.py:6,28,351

**Evidence.** CONFIGS entries the paper never uses: opt-350m with its embed_dim projection path (transformer.py:90,96,517-520,582-583), opt-2.7b, opt-30b, opt-66b and llama2-70b (:98,117-121). The paper names 70B or 30B models only for other systems. The hybrid CPU attention is reachable only with PVI_CPU_ATTN=hybrid (transformer.py:47,325,348; native_kernels.py:1033-1059). Two docstrings are misplaced: the one describing _FUSED_ATTN follows _CPU_ATTN (transformer.py:46-48), and the _EXP_ZERO docstring is orphaned after CERTIFIED_PROB_MAX (:65). Course, plan and job references: real_weights.py:1 ('the report's int8 perplexities, Section 4.1'), :148, :175; transformer.py:57,61,108,117; real_llama.py:1, 458-464, 479 ('job 1012500's overflow'); native_kernels.py:806 ('plan 2, RX'); opt_builder.py:6 (cites attribution.py, which is not in the repository) and :28 (branch f2-quality). The DEFAULTS key emb_row is unused (opt_builder.py:351).

**Recommendation.** Remove the unused configs and the embed_dim path; other graphs and their digests are unaffected. Drop the hybrid kernel. Rewrite the docstrings in plain professional terms. Docstrings and helper names are not hashed, so this is digest-neutral, as long as no builder method is renamed (INT-1). (Effort: small.)

## Experiments, run package, number pipeline, tests

Scope: code/experiments/**, the run package (sp2027.sh, strong_gpu.sh, bench.sbatch, smoke.sbatch, train.sbatch), the number pipeline (aggregate.py, sp2027_numbers.py with paper_assets/count_outcomes), code/tests/**, scripts/export_anonymous.py and docs/L40S_RUN_SP2027.md. Paths are relative to the wt_sp2027 root at HEAD 08a7f1f. Nothing was edited. I ran three read-only checks, and `git status` was clean before and after: (1) `PVI_DRYRUN=1 sp2027.sh all`, which submitted nothing; (2) `sp2027_numbers.py --stored --check --strict --verbose`, which writes nothing; (3) small python calls with PYTHONDONTWRITEBYTECODE=1.

**What the inventory said, checked one by one:**
- The pipeline is not wired to the paper: confirmed. No `\input` of `generated/` and none of the generated macros are used; about 61 live `\todo`s remain.
- B1 (quality pooling): confirmed, and it already happens on the stored records. The `--stored` table prints `opt-6.7b smooth-p11-int x1.017` with "3" draws, which are really three α values (0.5, 0.55, 0.65).
- B2 (generation table mixes real and random weights): confirmed in the code.
- B3 (V1 records have no verifier device, so the V1 table can be filled from any tier): confirmed. It also takes in the smoke's 2-block GPT-2 rows and the real tier's random-weight twins.
- B4 (`strong_gpu.sh` defaults to `auto`, everything else to `cnn18c`): confirmed.
- B5 (`--strict` can never pass): confirmed by running it. It exits 1 even on the stored records, which have 12 permanently missing base cells in `tab_eval_llm`.
- B6 (the real Llama-2-7B V1 cells have no random-weight twin): confirmed.
- B7 (`.title()` mangles macro names): confirmed, e.g. `CpuratiominLlamaTwoSevenB`.
- B8 (re-execution baselines pruned unevenly): plausible and small.
- The stored self-check still passes: 63 and 30 cells identical to the paper.
- The stored dry-run listing `docs/improvements/F6_dryrun.txt` still matches the script exactly: 137 of 137 jobs after path normalisation.

**New findings beyond the inventory:**
- The final run's main inputs cannot be committed or exported. They are `logs/<platform>/sp-*.out`, and both `/logs/` and `*.out` are gitignored.
- The anonymous export does not catch "TAU". It survives in about 9 exported files, and the hostnames of a different cluster would survive in the records too.
- The smoke gate passes when the GPU or native tests are skipped, and when int8 GEMMs report False. No record says which kernel path actually ran.
- The bench `k-<cnn>` jobs run the course report's plan for mode C at every λ, and those queries enter the final counts.
- The paper reports OpenLLaMA-7B (no licence needed), but the package's Llama cells need the Llama-2 licence. No job produces the OpenLLaMA numbers, and no stored record backs them.

**Can the package reproduce the paper end to end? No.** Even once the pipeline is wired in, these numbers have no producing job, or depend on records that are not in the repository:
- the certified rows of `tab:quality`;
- the OpenLLaMA claims;
- the v0 columns of `tab:eval-real` (2080 Ti);
- the base columns (course runs);
- the `tab:eval-improve` A/Bs, which need an older checkout via `--base`;
- `fig:breakeven`, drawn from hand-typed inputs;
- the full-model rows of `tab:eval-long`;
- the image models, which are not in git.

**Recommended shape of a clean tree:**
- Topical folders under `code/experiments`: attack, cost, improvements, v1, generation, reexec, quality, deployment, numbers, run. Each gets a README generated from `--help`, plus a top-level map from paper label to script, tier, record kind and generated table.
- Shared helpers move to a `pvi.experiments` package: builders, real-model loaders, record header and schema, peak memory, and the selection code taken out of `paper_assets.py`.
- The run package becomes a driver with flags, reading one job table. That table also produces the expected-job listing, the `--expect` coverage report and the package tests.

The findings below are ranked by severity, with a keep/merge/drop list for scripts and tests.

### F1 [high, confirmed] Number pipeline is not wired to the paper, and most generated tables do not have the paper's layouts

**Where:** code/experiments/5_comparison/sp2027_numbers.py:1058-1074; docs/L40S_RUN_SP2027.md:13

**Evidence.** grep over paper_sp2027/main.tex and sections/*.tex: no \input of generated/ and none of the macros sp2027_numbers.py writes (CpuRatioN, Final*, Gen*, Setup*, Serve*, Crossover*, LongFit*); live \todo counts: evaluation.tex 24, ext_evaluation.tex 11, ext_discussion.tex 7, intro/ext_intro 9, etc. docs/L40S_RUN_SP2027.md:13 says 'Nothing has to be computed or copied by hand'. Layout mismatches: tab_eval_v1 (sp2027_numbers.py:582-615) has a T column and one row per V1 record, against the 16 fixed rows of tab:eval-v1 (evaluation.tex:126-157); tab_options (:494-544) is a per-model/per-length data table, not the options layout of tab:options (evaluation.tex:182-222); tab_long (:783-816) is per block while tab:eval-long is extrapolated to the full model with ratio and GB (ext_evaluation.tex:178-194); tab_gen columns differ from tab:gen-full; there is no generator for the LeNet-5 row, the image-model cost range, or tab:eval-compare-v1.

**Recommendation.** Decide now, before the 26 Oct freeze. Preferred option: make each generator emit exactly the body rows of the paper table it targets (tab:eval-llm, eval-compare, eval-v1, options, eval-reexec, gen-full, setup times, eval-long, counts), \input them, define the text numbers only in generated/numbers.tex, and add a check that the paper builds from generated/ with no \todo{L40S|EPYC|F6} left. Otherwise declare cells.csv the source for hand transcription and correct L40S_RUN_SP2027.md:13. (Effort: large (3-4 days).)

**Verifier.** Confirmed. No .tex file \inputs generated/, and none of the generated macros appear in the paper; the only match is the 'Crossover' heading at ext_evaluation.tex:212. docs/L40S_RUN_SP2027.md:13 does say 'Nothing has to be computed or copied by hand'. The layout mismatches hold: tab_eval_v1 has a tokens column (sp2027_numbers.py:604); tab_options is a per-model data table (:501-532); tab_long is per block (:783-816); tab_gen is per model/mode/decode/G at P=512 under Fiat-Shamir, while tab:gen-full (ext_generation.tex:88-112) is GPT-2/OPT-1.3B at P=64, interactive; tab_eval_compare covers VGG-16 only (:387). Small errors that do not change the conclusion: evaluation.tex has 34 live \todo on 24 lines (not 24); tab:eval-v1 has 12 data rows (not 16); the generator writes one row per (seq, model, layers, mode) key, not per record.

### F2 [high, confirmed] # Cost tables mix protocols: the run is Fiat-Shamir under cnn18c, the captions say interactive, and the proof column falls back to the base run

**Where:** code/experiments/5_comparison/sp2027_numbers.py:338,360-361

**Evidence.** sp2027.sh:46-48 (every v0/V1 cell runs Fiat-Shamir) and :66 (POL defaults to cnn18c); tab_eval_llm's note reads 'base ... (interactive, plan auto); new ... (fs, plan cnn18c)' (:338); the proof column is 'new where measured, else base' (:360-361). The paper's caption says 'interactive ... Proof bytes do not change' (evaluation.tex:74-77), and tab:options says 'ours interactive' (evaluation.tex:185). The proof does change: GPT-2 mode C at 64 tokens is 14.8 MB interactive (evaluation.tex:90) and 16.6 MB under Fiat-Shamir (evaluation.tex:141).

**Recommendation.** Pick one protocol and one plan for every cost table. Either relabel tab:eval-llm and tab:options as Fiat-Shamir/cnn18c and keep the course-run base columns as a clearly labelled 'earlier implementation (interactive, plan auto)', or drop them. Remove the 'else base' fallback so a missing new cell shows as missing. (Effort: medium (1 day, plus caption edits).)

**Verifier.** Confirmed. sp2027.sh:46-48 puts every v0/V1 cell under Fiat-Shamir and :66 sets POL=cnn18c. The bench 64-token cells are C:fs only (strong_gpu.sh:143-148). The note at sp2027_numbers.py:338 says base is interactive/auto and new is fs/cnn18c, and the proof columns fall back from new to base (:360-361). The paper's captions say interactive (evaluation.tex:74-77, 185). The proof does change: GPT-2 C at 64 tokens is 14.8 MB (evaluation.tex:90) against 16.6 MB under Fiat-Shamir (evaluation.tex:141 and tab:eval-compare). Also, 'prove base' against 'prove new' mixes the Fiat-Shamir hashing cost into the claimed improvement.

### F3 [high, confirmed] # Table builders select records on keys that are too coarse: quality pools alpha, generation mixes real and random weights, V1 ignores tier and verifier device

**Where:** code/experiments/5_comparison/sp2027_numbers.py:763-777, 659-671, 588-593

**Evidence.** B1: tab_quality groups on (model, seq, windows, cell) (:763,767), but ppl_llama.py names cells without alpha ('smooth', 'smooth-glus0.2', ppl_llama.py:258-262) and keeps alpha in its own field (:398). q-llama7 runs --alphas 0.55 0.65 (sp2027.sh:394), so its ratio would be the median of two alphas. This already happens on the stored records: `--stored` prints 'facebook/opt-6.7b ... smooth-p11-int ... x1.017 & 3', and the 3 'draws' are alpha in {0.5, 0.55, 0.65}. B2: tab_gen takes every gen record (:659), including sp-r-llama7-gen (real weights, sp2027.sh:351-352; records carry 'real', gen_bench.py:238). Its Gen...GSixFour... macro (:670) overwrites the random-weight one, because sp-r- sorts after sp-gen-. B3: v1_bytes.py writes no verifier_device or threads (head :368-375, out :417-422). tab_eval_v1 keys on (seq, model, layers, mode) across all tiers, preferring tier v1 (:588-593), and drops only 'real' records (:590). If a v1 cell fails, it is silently filled by sp-g-v1-* (GPU verifier, sp2027.sh:274), cpuv-v1 or a real-random twin (no 'real' field). The smoke's GPT-2 2-block records (sp2027.sh:153) add stray rows.

**Recommendation.** Key every table on the full configuration: alpha, glu_smooth, real, recipe, verifier_device, threads, label and tier. Restrict each table to the tiers that define it. Write verifier_device, threads and an explicit 'kind' plus a schema version into every harness record. Add regression tests with multi-alpha ppl records, a real-weight gen record and a GPU-verifier V1 record. (Effort: small-medium (0.5-1 day).)

**Verifier.** All three confirmed. B1: tab_quality keys on (model, seq, windows, cell) (:763,767), and ppl_llama names cells without alpha (:258-262, emit at :398). I reproduced it on the stored records: 'facebook/opt-6.7b ... smooth-p11-int ... x1.017 & 3' pools alpha 0.5/0.55/0.65, and OPT-1.3B smooth-p11-int pools 15 records. B2: gen records carry 'real' and layers=None (gen_bench.py:233-238). tab_gen filters neither (:659-671), and the stable sort puts sp-r-llama7-gen after sp-gen-llama2-7b, so the real-weight G64 macro wins. B3: the v1_bytes head (:368-375) has no verifier_device or threads. tab_eval_v1 drops only codec_only/real (:590) and prefers tier v1 (:588), so gpuv-v1, cpuv-v1 and *rnd* twins can fill a failed cell, and the smoke's 2-block GPT-2 records add rows. One nuance: v1 records do carry 'label' (gpuv-v1, cpuv-v1, real-random), so part of the fix is simply keying on label/tier.

### F4 [high, confirmed] The final run's records cannot be committed or exported, so the artifact cannot regenerate the final tables

**Where:** .gitignore:14,29; scripts/export_anonymous.py:30-39

**Evidence.** sp2027_numbers.py reads logs/<platform>/sp-*.out (:10-12, :157-161), but .gitignore ignores '/logs/' (:14) and '*.out' (:29). Only bench.py's raw_<platform>/*.jsonl can be tracked. The export's INCLUDE (:30-34) has no logs/, docs/ or paper_sp2027/, and EXCLUDE drops *.sbatch and all of slurm/ (:36-39), so sp2027.sh and strong_gpu.sh are missing from the snapshot. Two tests then fail in the snapshot: test_sp2027_numbers.py:182-190 reads ../paper_sp2027/sections, and compare_eval_llm returns 'no .../evaluation.tex' (:901), so status is 1 (:1038); test_f6_package.py:32,113-120 asserts that bash sp2027.sh returns 0. scripts/artifact_README.md:13 promises 'the stored records from which the paper's tables are generated'. The trained image models are untracked (.gitignore:8), and need_images stops the bench tier without them (sp2027.sh:422-438).

**Recommendation.** Give the final run a stored home, e.g. code/artifacts/final/<platform>/<job>.jsonl, holding only the record lines with the env and host fields stripped. Point sp2027_numbers --stored at it and keep a golden coverage.txt. Export the anonymised run package and run guide. Have the package and paper-text tests skip cleanly when their inputs are absent. Ship the five image models (or a deterministic retrain recipe plus their digests). (Effort: medium (1-2 days).)

**Verifier.** Confirmed. .gitignore:14 ignores /logs/ and :29 ignores *.out. sp2027_numbers reads sp-*.out and needs the file name for the tier (LOG regex :125, log_files :157-161); a stored *.jsonl loses the tier. In export_anonymous.py, INCLUDE (:30-32) has no logs/, docs/ or paper_sp2027/, and EXCLUDE (:37-38) drops *.sbatch and slurm/. In a snapshot, compare_eval_llm returns 'no .../evaluation.tex' (:900) and sets status 1, so test_sp2027_numbers.py:182-190 fails, and test_f6_package._dry asserts returncode 0 (:122) on a missing sp2027.sh. artifact_README.md:13 promises the stored records. The image models are ignored (.gitignore:8), and need_images exits 2 without them (sp2027.sh:422-438).

### F5 [high, confirmed] Anonymous export misses the institution acronym and cluster details in exported files and in raw records

**Where:** scripts/export_anonymous.py:42-54

**Evidence.** STATIC_DENY (:53) has 'tau.ac.il' and 'Tel Aviv' but not 'TAU'. `git grep -w TAU` over exported paths hits code/src/pvi/fullcheck/datasets.py:14, experiments/4_defence_benchmark/README.md:153,185, experiments/2_attack/REAL_LLM.md:92, artifacts/results/zkllm_l40s/README.md:4, artifacts/comparison/README.md:52,140, and raw/, raw_l40s/, raw_l40s_improved/ and raw_rtx2080ti-v2/ PLATFORM.json ('TAU killable L40S nodes', 'TAU SLURM studentbatch'). The 'killable' and 'studentbatch' partition names are not denied either. Every bench record's env row stores the hostname, SLURM partition and nodelist, and every PVI_* value with its paths (bench.py:160-166, 182-184); perf/v1/reexec/gen/setup records store platform.node(). REWRITES only know TAU's node patterns (:52), so another cluster's hostnames in raw_<final platform>/ would pass the scan. The script stores real names in the repo (:53), although its docstring (:15-16) says no name is written into the repository.

**Recommendation.** Add 'TAU' (whole word, case-sensitive) and the partition names to the deny list. Strip host, slurm_* and PVI_* path fields from records on export, or build rewrites from the records' own host values. Move the deny list to an untracked local config. Remove the TAU wording from the files above on the submission branch. (Effort: small (0.5 day).)

**Verifier.** Confirmed. STATIC_DENY (:53) lacks 'TAU', 'killable' and 'studentbatch'. git grep -w TAU over the exported paths hits every file listed in the finding, plus code/requirements.txt:5 ('on the TAU cluster'). The PLATFORM.json notes ('TAU killable L40S nodes', 'TAU SLURM studentbatch') would pass the scan. bench.py's env row stores the hostname, slurm_partition, SLURM_JOB_NODELIST and all PVI_* vars (bench.py:160-166, 182-184). The REWRITES node patterns (:49) are TAU-specific. Minor point: scripts/ is not in INCLUDE, so the names in STATIC_DENY do not reach the snapshot. The docstring mismatch (:15 against :53) is a nit about the repository, not a leak.

### F6 [high, partly] # The package cannot reproduce the paper end to end: several paper numbers have no producing job and no stored record

**Where:** code/experiments/4_defence_benchmark/slurm/sp2027.sh:300-352,368-396

**Evidence.** (a) Certified rows of tab:quality (quality.tex:40-42: x1.076/1.026/1.008): the stored ppl2048_opt-*.jsonl files hold none of them. Their lowest ratios are 1.161, 1.095 and 1.017, all alpha<=0.65 smooth or scalar cells, and no package tier runs OPT quality. (b) OpenLLaMA-7B claims (evaluation.tex:59-60; quality.tex:84-86): no job uses openlm-research, and the package's Llama cells are gated Llama-2 behind PVI_LLAMA_LICENCE (sp2027.sh:335-352, 387-396). (c) tab:eval-real's v0 columns are 2080 Ti records measured 'before the freeze' (sp2027.sh:302-304; ext_evaluation.tex:132-148); the package has only OPT-1.3B V1 cells. (d) The base columns come from the course runs (sp2027_numbers.py:89). (e) tab:eval-improve's A/Bs need an older checkout via --base (6_improvements/README.md:15-19). (f) fig:breakeven comes from hand-typed 5_comparison/ev_inputs_draft.json. (g) tab:eval-long is per block only (F1).

**Recommendation.** Make a claim-to-producer table and close each gap. Switch the real and quality Llama cells to openlm-research/open_llama_7b (ungated, Llama-2-7B's shapes), or change the paper to Llama-2. Add an OPT certified-quality job, or commit the round-2 records. Either add OPT-1.3B v0 real/random cells on the L40S, or state a second platform. Feed ev_breakeven from cells.csv. Turn the A/B 'before' paths into non-default flags so they run from this branch. (Effort: large (2-4 days plus run time).)

**Verifier.** The core claim holds, but two of the seven sub-points are wrong. (a) Confirmed: the stored ppl2048_opt-* minima are 1.161, 1.095 and 1.017; round1 has only alpha<=0.5 (best 1.028 and 1.099); there is no alpha=0.6 certified row and no OPT quality job. (b) Confirmed: there is no openlm-research job; the real and quality tiers use meta-llama/Llama-2 (sp2027.sh:339, 393-395). (c) Partly: the package does lack OPT-1.3B v0 perf cells, but the r-opt13{,rnd}-v1-* records carry v0_total and timed v0 prove/verify on the same queries. The gap is that tab_real (:715-725) ignores them. (d) Refuted: the base columns come from the tracked raw_l40s_improved{,2,3}, so they are stored records. (e) Overstated: open_ab, forward_ab, encode_prof and fold_ab run on this branch without --base; only the gpu_ab/attention_ab/ab_verifier-style harnesses need --base, though none of the A/B records are stored. (f) Confirmed: ev_inputs_draft.json is hand-typed with 2080 Ti/Xeon values. (g) Confirmed, and duplicates F1.

### F7 [medium, confirmed] --strict can never pass, and the real Llama-2-7B V1 cells have no random-weight twin

**Where:** code/experiments/5_comparison/sp2027_numbers.py:1054

**Evidence.** --strict fails on any missing cell (:1054). Running `--stored --check --strict --verbose` exits 1: tab_eval_llm has 12 missing cells such as 'Qwen3-4B (64) / C GPU verify base' and 'Llama-2-7B (2048) / C prove base', which the paper prints as '--' (evaluation.tex:91-93). tab_options iterates over every cpuv model (:498), but V1 cpuv jobs exist only for Llama-2-7B at 64/2048 and Qwen3-4B at 2048 (sp2027.sh:190-192). sp2027.sh:339-346 has r-llama7-v1-* without a random twin, so tab_real's 'random' V1 cells are always missing (:709-725).

**Recommendation.** Let each table declare its expected cells, derived from the job table and the paper's '--' cells, and make --strict check against that. Add the r-llama7rnd-v1-* twin, or drop the column. (Effort: small (0.5 day).)

**Verifier.** Confirmed. I ran --stored --check --strict --verbose: it exits 1, and tab_eval_llm lists 12 missing cells, including 'Qwen3-4B (64) / C GPU verify base' and 'Llama-2-7B (2048) / C prove base', which the paper prints as '--'. tab_options iterates over every cpuv perf model (the loop is at :501, not :498), but cpuv V1 cells exist only for Llama-2-7B 64/2048 and Qwen3-4B 2048 (sp2027.sh:190-192). r-llama7-v1-$t has no rnd twin (sp2027.sh:339-346). Even a perfect run therefore fails --strict.

### F8 [medium, partly] Smoke gate passes with skipped GPU/Triton tests and a failing int8 self-check; no record says which kernel path ran

**Where:** code/experiments/4_defence_benchmark/slurm/sp2027.sh:143-154

**Evidence.** The smoke runs pytest and then '&& ... echo SMOKE OK' (:151-154). pytest exits 0 when the GPU, Triton and native tests are skipped (test_gkr_triton.py:17 'needs CUDA and Triton', test_gpu_verifier.py:41, test_native_cheap.py native-only). field.int8_ok('cuda') is only printed (:152), and the native check runs only when CXX is set (:149-150). If int8 is not exact, the verifier silently uses the float64 limbs (field.py:356-366); if CXX is missing, native_kernels.available() is False and the torch paths run. perf.py's record (:310-336) and v1_bytes.py's (:368-422) carry no kernel or int8/Triton fields, so a fallback on the L40S would produce valid but slower final timings that look normal.

**Recommendation.** Make the smoke assert int8_ok('cuda'), gkr_triton.available('cuda') and native_kernels.available(), and fail if any test in the GPU/native subset is skipped (e.g. -rs plus a skip-count check). Write a 'kernels' dict into every timing record (native, avx512, int8_gemm, triton, the PVI_* switches in effect). (Effort: small (0.5 day).)

**Verifier.** The core gap is real. field.int8_ok('cuda') is only printed (sp2027.sh:152), and the fallback is silent (field.py:375-384). pytest exits 0 on skips, whereas the course smoke.sbatch explicitly failed on any skipped test ('none failed, errored or skipped'), so the new smoke lost that gate. perf.py's record (:310-328) and v1_bytes' (:368-375) have no kernel or int8/Triton fields. Two details are inaccurate. native_kernels compiles with CXX or else g++ on PATH (native_kernels.py:903), so a missing CXX skips only the smoke's explicit check, not the native path itself. gkr_triton.available only checks that Triton imports (gkr_triton.py:54-55), so a broken Triton launcher fails the tests rather than skipping them.

### F9 [medium, confirmed] # Bench k-<cnn> jobs run the course report's mode-C cells, and the default policies disagree

**Where:** code/experiments/4_defence_benchmark/slurm/strong_gpu.sh:105,136

**Evidence.** In the fresh dry run, 'k-vgg16 ... bench.sbatch cnn --model vgg16 --queries 30 --tampers 0 --wire --tag _wire' has no --policy, so bench.py uses the default 'paper' (bench.py:1015). _todo (:307-313) then runs every cell: facts, attack_float, sampling, and defence C:int/C:fs at lambda 40/80/128 under the per-op commitment (:593-594), inside a 20-minute limit. These honest queries go into the Final* counts (count_outcomes counts every tag), but no table reads them. test_f6_package.py:144-145 checks the policy only on lines that pass --policy. strong_gpu.sh:105 defaults POL to 'auto', while sp2027.sh:66 and sp2027_numbers.py:1083 default to cnn18c (B4), so running 'strong_gpu.sh must' directly, as its header suggests (:8), produces cells the pipeline cannot find.

**Recommendation.** Make the paper's plan the default in bench.py, perf.py and serve_ab.py, and keep 'paper' as a non-default 'basic' option. Give the CNN suite a mode or lambda filter so k- jobs run only K and Kpre at the lambda the tables use. Use one default policy in one place. (Effort: small-medium (1 day; renames cells, touches test_plans.py:1120).)

**Verifier.** Confirmed. The k-<cnn> jobs pass no --policy (strong_gpu.sh:136), and bench.py defaults to 'paper' (:1015). _todo (:307-314) then runs facts, attack_float, sampling, commit and every defence_{C:int,C:fs,K:int,Kpre:int} cell at lambda 40/80/128 (:492-596) within 20 minutes. outcome_counts calls count_runs with no tag or definition filter (sp2027_numbers.py:867; count_outcomes.py:81-123), so the paper-plan C cells enter Final*. The test's policy check is at test_f6_package.py:148 (not 144-145) and applies only when --policy is present. strong_gpu.sh:105 defaults to auto, against cnn18c in sp2027.sh:66 and sp2027_numbers.py:1083, and the header at :8 shows a direct run.

### F10 [medium, confirmed] The run package is driven by environment variables, its job table is shell code, and the expected-job listing lives in the diary

**Where:** code/experiments/4_defence_benchmark/slurm/sp2027.sh:28-45,442-447

**Evidence.** The tier is the only argument (:442-447). Everything else is an environment variable: PVI_PLATFORM, PVI_C_POLICY, PVI_DRYRUN, PVI_REHEARSAL, PVI_LOCAL, PVI_OPTIONAL, PVI_CHAIN, PVI_EXCLUSIVE, PVI_MEM_MAX, PVI_QUERIES(_HEAD), PVI_LLAMA_LICENCE, PVI_BIGMEM, PVI_Q7/Q13_ARGS (:28-45, 394-396). strong_gpu.sh hard-codes the queries, tampers, lambdas, times and memory of every cell (:133-203), so PVI_QUERIES never reaches the bench cells. sp2027.sh's nl() (:117-122) and sp2027_numbers.py's FULL_LAYERS (:72-73) duplicate transformer.CONFIGS. The --expect listing is docs/improvements/F6_dryrun.txt (L40S_RUN_SP2027.md:59); I regenerated it and it matches exactly (137/137 jobs after path normalisation), but nothing keeps it in sync. Library switches that select superseded paths (PVI_FUSED_CODEC, PVI_V1_BUCKETS, PVI_NATIVE_*, PVI_FOLD_FAST, ...) are environment-only. Several jobs feed no paper table: sp-g-v0-* only feed tab_spread/tab_memory, sp-g-v1-* are dropped as duplicates (sp2027_numbers.py:588-593), and sp-g-h2d's tab_h2d has no paper target.

**Recommendation.** Replace the shell tiers with one driver that takes flags (--platform --tier --dry-run --rehearsal --local --policy --queries --queries-head --mem-max --no-chain --no-exclusive --optional --llama-licence --bigmem) and reads one job table (model, cell, args, hours, memory, consumer table). Derive the dry-run listing, --expect coverage, test_f6_package and the run guide's table from it. Drop jobs that have no consumer. Expose the PVI_* switches as harness flags (e.g. --switch NAME=VAL) and record them. (Effort: medium-large (2-3 days).)

**Verifier.** Confirmed. The tier is the only argument (sp2027.sh:442-470), and everything else comes from environment variables (:28-45, 55-84, 335, 387, 394-396, 419). The bench cells' queries and limits are literals (strong_gpu.sh:133-203), so PVI_QUERIES never reaches them. nl() (sp2027.sh:117-122) and FULL_LAYERS (sp2027_numbers.py:78, not :72-73) duplicate CONFIGS. F6_dryrun.txt has 137 sbatch lines and is referenced from L40S_RUN_SP2027.md:54/157. sp-g-v1-* share tab_eval_v1's key with the v1 tier and are always dropped; sp-g-v0-* feed only tab_spread and tab_memory. This is a cleanup item the authors have already decided on (every parameter becomes a CLI argument).

### F11 [medium, confirmed] Experiment scripts import each other by file path, and the number pipeline depends on the course-report table builder

**Where:** code/experiments/6_improvements/perf.py:162,207,237

**Evidence.** perf.py loads the A/B harness ab_verifier.py before argparse, on every run (:162), plus bench.py (:207) and v1_bytes.py (:237). Other path-based loads: gen_bench.py loads serve_ab.py and v1_bytes.py (:74, :209); setup_timing.py loads gen_bench.py (:62); v1_bytes.py adds 9_quality to sys.path and imports ppl_llama and common (:69-70, :95-96); plan_bytes.py imports ab_verifier (:58); wire.py imports plan_bytes (:42); inrange_llm.py and adaptive_game_llm.py cross folders (:170; :64-65). sp2027_numbers.py imports paper_assets (:984), which imports matplotlib at module load and writes to ../report, and pins pa.TABLES to tables_l40s (:985). RATE=4 is redefined in bench.py:86, setup_timing.py:37 and gen_bench.py:57. code/README.md:141-142 says the experiments never import each other.

**Recommendation.** Create a pvi.experiments package with builders (bench._cnn_model/_mlp_model/_test_batches, ab_verifier.CNNS/build/random_queries), real-model loaders (v1_bytes.load_real*/real_graph), peak-memory helpers, serve_ab.TINY, and a records module (header, kind, schema). Move the cell-selection part of paper_assets (Measured, llm_cost, llm_spec, OPTIMISED) into it, and import from there everywhere. Do this before dropping ab_verifier.py and plan_bytes.py. (Effort: medium (1-2 days).)

**Verifier.** Confirmed. perf.main() loads ab_verifier before building argparse (perf.py:161-163) and loads bench (:207) and v1_bytes (:237). gen_bench loads serve_ab/v1_bytes (:74, 209); setup_timing loads gen_bench (:62); v1_bytes inserts 9_quality into sys.path (:69, 95). RATE=4 is defined three times (bench.py:86, setup_timing.py:37, gen_bench.py:57). code/README.md:141-142 contradicts all this. paper_assets imports matplotlib at module level (:48-55) and sets REPORT=../report (:697). Small correction: sp2027_numbers never calls paper_assets' writers, so nothing is actually written to ../report.

### F12 [medium, confirmed] # The abstract's N is computed over both prompt lengths while the sentence is about 2,048 tokens; macro names are mangled

**Where:** code/experiments/5_comparison/sp2027_numbers.py:540-541,244

**Evidence.** CpuRatioN = floor(min of the 64- and 2,048-token Kpre ratios) (:540-541). The abstract says 'verifies a 2,048-token Llama-2-7B prompt 4.7 times faster ... \todo{EPYC: at least N times}' (main.tex:73). On the Xeon the Fiat-Shamir ratios are 4.7x at 2,048 and 2.9x at 64 tokens (evaluation.tex:240-244), so this rule gives 'at least 2' for a sentence about 2,048 tokens. macro_name applies str.title() (:244); evaluated outputs include CpuratiominLlamaTwoSevenB, GenLlamaTwoSevenBKpreGreedyGSixFourMspertoken and CrossoverOptOneThreeBLTwoFourTwoZeroFourEightKpre, which do not match the docstring's \CpuRatioN style (:27).

**Recommendation.** Decide which length the abstract quotes and compute N for exactly that cell, or rewrite the sentence as 'at least N times at 64 to 2,048 tokens'. Use a fixed, documented macro naming scheme (no .title()) and refer to the macros from the tex. (Effort: small (2 h).)

**Verifier.** Confirmed. CpuRatioN = floor(min over the 64- and 2,048-token ratios) (:540-541), but the abstract sentence is about 2,048 tokens (main.tex:73), and the Xeon ratios are 4.7x/2.9x (evaluation.tex:240-244), which gives 'at least 2'. The plan makes this stronger: SP2027_PLAN2.md:595, 1094 and 1200 define N as the lower bound on both CPUs (Xeon and EPYC) at the quoted cell. The code implements neither that rule nor the abstract's wording, and it does not read the Xeon records at all. macro_name's .title() (:244) yields names such as CpuratiominLlamaTwoSevenB and ...Mspertoken; this is cosmetic, since no tex uses them.

### F13 [medium, confirmed] Tests are tied to paper text and diary IDs, miss the selection bugs, and import each other; keep/merge/drop list

**Where:** code/tests/test_sp2027_numbers.py:182-190

**Evidence.** The stored-records test asserts exact strings '63 base cells ... identical' and '30 cells identical' taken from the paper text (:188-190), so it breaks as soon as tab:eval-llm is filled. Its synthetic records have no alpha, no 'real' gen record and no verifier device (:36-110), so F3 goes uncaught, and no test feeds real harness output into the pipeline. 13 test files import helpers from test_plans, test_wire or test_cut_protocol. pyproject.toml:29 sets --strict-markers but registers no gpu/native/hf/slow markers. Course-only tests: test_comparison_tables.py:89,102,349-372,375,407 (text_numbers, validate_extrapolation, report count macros). test_plans.py:1120 pins the course report's commitment as the library default. smoke.sbatch (course smoke) fingerprints against raw/ and would fail on the new skips.

**Recommendation.** Keep every protocol and library test. Drop the course-only parts of test_comparison_tables together with text_numbers.py, validate_extrapolation.py and smoke.sbatch. Move the shared helpers into conftest.py or tests/_helpers.py. Register markers (gpu, native, hf, slow, smoke) and select the smoke subset by marker instead of a path list (sp2027.sh:143-147). Replace the paper-text assertions with golden outputs of the stored final records. Add a CPU test that runs perf/v1_bytes/gen_bench/setup_timing/reexec on tiny models and feeds their output to sp2027_numbers. Rename tests named after plan IDs (test_f6_package, test_f3b_real_llama). (Effort: medium (1-2 days).)

**Verifier.** Confirmed. test_sp2027_numbers.py:182-190 asserts the paper-derived strings, and the comparison covers the base and proof columns. Once the proof column takes the final Fiat-Shamir/cnn18c values, or the base columns are dropped, the test breaks. The synthetic records have no alpha, no real-weight gen record and no verifier device. 13 test files import from test_plans/test_wire/test_cut_protocol (I counted them). pyproject.toml:29 has --strict-markers with no markers registered. test_plans.py:1120 pins the report's commitment as the commit_graph default. test_comparison_tables uses text_numbers and validate_extrapolation (:30, 63, 375). smoke.sbatch fails on any skip.

### F14 [medium, confirmed] Experiments tree still carries course-report material, personal and cluster content, and has no README for three folders

**Where:** code/experiments/6_improvements/README.md:9-13; docs/L40S_RUN_SP2027.md:3-6

**Evidence.** 7_reexec, 8_v1 and 9_quality have no README. 6_improvements/README.md:9-13 says the paper's numbers do not come from that folder, yet perf.py, gen_bench.py, setup_timing.py and h2d_bandwidth.py are package scripts. Unused course or one-off scripts: 6_improvements/{attention_ab,gpu_ab,codec_ab,codec_stages,wire_hashes,claims_drop,cpu_attn_bench,cpu_profile,derive_ab,setup_cache_check}.py, 2_attack/real_weights_ppl.py(.sbatch), 3_sampling_fixes/adaptive_game_plots.py, 8_v1/profile_v1.py, 5_comparison/{text_numbers,validate_extrapolation}.py. strong_gpu.sh still has the report-must/should/nice tiers and the smoke tier (:19-20, 206-296, and the smoke case). train.sbatch:3,11 names partition studentbatch (TAU). The run guide is written 'For the teammate' with dates (L40S_RUN_SP2027.md:3-6, 59, 123-126), team rules about tokens (:35, 148-150) and diary references (:22-23, 105, 157).

**Recommendation.** Target layout under code/experiments: attack/, cost/ (bench, train_cnn, perf, setup_bytes+ps2_tables, setup_timing, params_table, wire), improvements/ (open_ab, forward_ab, encode_prof, fold_ab, gpu_profile, with superseded paths as flags), v1/ (v1_bytes, v1_phases, gkr_bench), generation/ (gen_bench, sampler_tv), reexec/ (reexec, h2d_bandwidth, serve_ab), quality/, deployment/ (ev_breakeven, literature), numbers/ (aggregate, count_outcomes, paper_numbers, fingerprint_check), run/ (driver, job table, bench.sbatch, train.sbatch without the partition, RUN_GUIDE.md). Each folder gets a README generated from --help, plus a top-level map from paper label to script, tier, record kind and table. Drop the scripts and tiers listed above; move gen_gkr_triton.py to scripts/. (Effort: medium (2 days, after F11).)

**Verifier.** Confirmed. 7_reexec, 8_v1 and 9_quality have no README. 6_improvements/README.md:9-13 says the paper's numbers do not come from that folder, yet perf, gen_bench, setup_timing and h2d_bandwidth are package scripts. None of attention_ab, gpu_ab, codec_stages, wire_hashes, claims_drop, cpu_attn_bench, cpu_profile, derive_ab or setup_cache_check is referenced by any script, the package or the paper; codec_ab only from codec_stages. train.sbatch:3,11 names studentbatch/TAU. The run guide is addressed to 'the teammate', with dates (L40S_RUN_SP2027.md:3-6), token rules (:35, 81, 149) and diary references (:22-23, 105). One caveat: the inventory could not tell which script produced tab:eval-improve's A4 and B2/B3 rows, so confirm that none of the dropped A/B harnesses is their only source before deleting them.

### F15 [low, not verified (low)] # Re-execution baselines are pruned unevenly: the integer variants are pruned, fp32/bf16 are not

**Where:** code/experiments/7_reexec/reexec.py:226-227,244

**Evidence.** int and int_fp32gemm build build_decoder(..., prune_last=True) (:226-227), while float, bf16 and dynint8 use FloatDecoder over every position of the last block (:244; docstring :20-22). fp32 is the strongest exact baseline at 2,048 tokens (93.2 s, evaluation.tex:240), so the 'x faster than re-execution' ratio is inflated by up to about 1/32 for Llama-2-7B (my estimate, not measured).

**Recommendation.** Prune the last block in every variant (LM head and last block at the last position), or state in tab:options footnote a that fp32 is the unpruned off-the-shelf stack. (Effort: small (2-3 h).)

## Paper: claims against evidence

I checked the paper's claims against the repository, starting from claims_body.md and claims_ext.md and re-checking the important items against the docs, the stored JSON/CSV records, sp2027.sh and sp2027_numbers.py. Everything was read only; no tests or builds were run.

**Spot checks that hold:**
- 46× against zkLLM (803/17.6 s).
- 98% of the 6.57 GB proof is claims (6,442,125,383 / 6,570,112,359 B).
- GPU verifier 10–15× faster than the CPU one at 2,048 tokens (4.8/0.318, 27.4/2.6, 46.5/3.5 s).
- 64-token Llama-2-7B proof 21× smaller than its weights.
- Wilson interval 91.24% for 40/40.
- MNIST mixture 3.06% and 9.67% against 5.41% and 10.82% (27.7/N).
- OPT-1.3B game 47.6%, 6.76%, 2.19%; uniform 0.019–0.060%.
- A7 ratios 4.07–8.55× and 19.2–40.3×.
- Gumbel-table constants and (V−1)e^−14.25.
- The 1.3 TB vs 3.3 GB generation estimate.
- Every \ref resolves in both builds.
- Every \texttt code name in the extended version exists in the code today.
- Theorem statements agree with their proofs: thm:guess/guess-full, thm:dich vs prop:worst/prop:amp (β = λ + log₂(2L), union over L layers), and thm:sound vs app:sound.

**The real problems are elsewhere:**
1. **Exact re-execution:** "the fastest exact re-execution" means fp32 in the abstract and body, while the extended version says fp32 "cannot confirm an exact answer".
2. **tab:eval-llm in the final run:** the generator will mix interactive/auto-plan base columns with Fiat–Shamir/cnn18c new columns under a caption that says "interactive" and "proof bytes do not change".
3. **Missing records:** many body numbers have no record in the branch, and the final run will not regenerate them: the certified tab:quality column and its intervals, the zero-shot results, the OPT-1.3B sampler game, the F2b Llama/Qwen ratios, the attribution study (f2-quality branch), GPT-2 generation, and the tab:sok formulas.
4. **V1 additions:** "V1's own additions stay below 1 MB" is contradicted by the repo's own byte tables (13.9 MB on the full OPT-1.3B in Kpre, 34.6 MB on OPT-6.7B).
5. **Recipe:** the real-weight cost and V1 numbers, including the abstract's 1.7×, come from the legacy α = 0.55 graph, not the certified model. The cross-OS reproducibility sentence is also legacy-only.
6. **Commitment plan:** the extended setup-proof text is still on the auto plan and contradicts its own table and the body (2^-35 vs 2^-23/2^-34; 11.3 vs 9.8 GB; 1.6% vs 2.2%). More widely, measured cells use the auto plan (cnn16c/17c/18c by model) while the text says one plan, and params_table.py and setup_bytes.py default to `--policy auto`.

**Smaller items:**
- GPU re-execution baselines come from two harnesses that disagree by up to 40%.
- The Kpre key lifetime (only 2^18.7 verdicts at r = 5) is stated only in the extended version.
- The extended proofs bind about 60 code identifiers, test names and docs paths that a refactor would break; the anonymous export leaves out docs/.
- Assorted small overreaches, and stale numbers in comments.

### P2 [high, confirmed] # Final tab:eval-llm will mix interactive/auto-plan base columns with Fiat–Shamir/cnn18c new columns under a caption that says 'interactive' and 'proof bytes do not change'

**Where:** paper_sp2027/sections/evaluation.tex:74-77; code/experiments/5_comparison/sp2027_numbers.py:18-23,333-337,359-360; code/experiments/4_defence_benchmark/slurm/sp2027.sh:46,66

**Evidence.** **What the caption says.** 'interactive, λ=128 … New: all improvements, on the L40S. Proof bytes do not change' (evaluation.tex:74-77).

**What the generator does:**
- Base columns: interactive, plan auto (sp2027_numbers.py:22-23,337).
- New columns: `--challenges` default fs, plan cnn18c (sp2027_numbers.py:18-20).
- Proof column: `proof_c = v["C","bytes","new"] if … not None else base` (sp2027_numbers.py:359-360). The table's single proof column therefore switches to Fiat–Shamir (r = 7) and cnn18c bytes.

**What the package runs.** sp2027.sh:46: 'Every v0 and V1 cell runs under Fiat–Shamir'; POL defaults to cnn18c (sp2027.sh:66).

So the base→new speedup will fold in r = 5→7, the plan change and FS hashing. The caption's 'proof bytes do not change' becomes false.

**Recommendation.** Choose one of two options:
- Add interactive, auto-plan 'new' cells for a like-for-like base/new comparison.
- Split the table into 'course base (interactive, auto)' and 'final (FS, cnn18c)', each with its own proof column, and rewrite the caption.

Make the generator refuse to put cells with different protocol settings in one column. (Effort: medium.)

**Verifier.** The evidence is accurate.
- The caption says 'interactive … Proof bytes do not change' (evaluation.tex:73-76).
- The base columns are interactive, auto plan (sp2027_numbers.py:337-344).
- The new columns default to --challenges fs and --policy cnn18c (sp2027_numbers.py:1083-1084).
- proof_c and proof_k take the new value where one exists (sp2027_numbers.py:359-360).
- The generator does not write the caption. It only records the settings in a comment note (sp2027_numbers.py:338-339).

The problem goes further than the caption. A like-for-like interactive new column cannot be built from the current package:
- The must tier runs only C:fs for every 64-token row and Kpre:fs everywhere (strong_gpu.sh:139-148, 165-171).
- C:int runs only for four 2,048-token models (strong_gpu.sh:160).
- So '--challenges int' would leave the 64-token and Kpre new cells empty.

The fix therefore needs a decision on the package before the L40S run, not only a caption edit.

### P3 [high, confirmed] Body and extended-version numbers whose records are not in the branch and that the final run will not regenerate

**Where:** paper_sp2027/sections/quality.tex:29-42,49-51,68-72,84-88; attack.tex:224-227,250-286; generation.tex:46-53; ext_generation.tex:147-245; ext_attack.tex:163-165,326-343; app_params.tex:55; ext_v1.tex:66-67,171-173

**Evidence.** **Not in code/artifacts/results; only in the server archive (C:/Users/User/Desktop/amina-project/server_archive/sp2027_clones_s3-s12_20261012.tar.gz):**
- tab:quality's certified column and its bootstrap CIs: `ppl2048_search2_*.jsonl`, job 1011066 (commit 7b29f06 cites them). AR/9_quality holds only round 1.
- Zero-shot results: `zeroshot_cert_*.jsonl`.
- OPT-1.3B sampler game: `adaptive_game_opt-1.3b.json`.
- F2b ratios ×1.073 / ×1.12 / ×1.40 / ×5.3: `f2bq*`.

**On branch f2-quality only:** `attribution.py`, `hybrid.py`, `fixes.py` and their records (quality.tex:49-51, tab:quality-attr, cross-layer 67.8 vs 66.6).

**Under scratchpad/sp2027_research (outside the repo):**
- `sok_formulas.py`, `check_guessing.py`, `check_merkle_leak.py`: tab:sok, tab:sok-derivation, the toy checks.
- V1_SPEC.md:34,46,56,347: 1.78–1.85×, 2.24×, 4,500 tokens, 4.3e9 leaves, 2^-215.3.
- idea_trust-rigor.md:31: 3.2–6.6×.
- raw_sp2080/: the GPT-2 generation cell.

**Not covered by the final package (sp2027.sh):**
- The gen tier runs only Llama-2-7B and Qwen3-4B (sp2027.sh:292-295).
- The quality tier runs only Llama-2-7B/13B (sp2027.sh:393-396), and it is optional.
- No tier runs the OPT quality, zero-shot, adaptive_game_llm, digest_check or the attack scripts.

**Recommendation.** Before the freeze:
- Copy the archived records into code/artifacts/results with a provenance index (record → paper location → job).
- Port attribution.py, hybrid.py, fixes.py and the sok/check scripts into the branch with READMEs, or remove the claims that depend on them.
- Add a sentence or a tier for GPT-2 generation, or replace it with the final Llama-2-7B cell. (Effort: medium.)

**Verifier.** Every item checks out.

In the repo:
- code/artifacts/results/9_quality holds only ppl2048_opt-*.jsonl, ppl2048_search_round1.jsonl and llama_diag.

Only in the archive:
- s3: ppl2048_search2_opt-{125m,1.3b,6.7b}.jsonl, adaptive_game_opt-1.3b.json, f2bq/.
- s11: zeroshot_cert_*.jsonl.
- s6: f2bq_prefix/.
- Commit 7b29f06 cites ppl2048_search2_* for the confidence intervals it added to quality.tex:31.

On branch f2-quality only:
- code/experiments/9_quality/attribution.py, hybrid.py, fixes.py and their records.

Outside the repo:
- sok_formulas.py, check_guessing.py and check_merkle_leak.py in sp2027_research/attack_theory.
- V1_SPEC.md:34, 46, 56, 347 and idea_trust-rigor.md:31 match the quoted values.

The package:
- The gen tier runs only llama2-7b and qwen3-4b (sp2027.sh:292-295).
- The quality tier is Llama-only and gated by PVI_LLAMA_LICENCE (sp2027.sh:385-396).
- No tier calls zeroshot, adaptive_game_llm, digest_check, inrange_llm or real_llm.

Severity is high for the artifact, though most of these runs need not be repeated on the L40S. The fix is to archive the records with their provenance and port the scripts, not to run anything new.

### P1 [medium, confirmed] # 'Fastest exact re-execution' is fp32 in the abstract and body but 'not exact' in the extended version; the headline ratio depends on the definition

**Where:** main.tex:72-73; paper_sp2027/sections/intro.tex:116-117; evaluation.tex:187,210-211,230-231,240-243; discussion.tex:31,43-45; ext_discussion.tex:97-102; ext_evaluation.tex:172,259; code/experiments/5_comparison/sp2027_numbers.py:72-74

**Evidence.** **What the body says.** The abstract, intro and evaluation call fp32 (93.2 s) 'the fastest exact re-execution' ('fp32; the exact integer forward pass takes 142 s', evaluation.tex:241-242). tab:options footnote a labels the fp32 row 'Exact' (evaluation.tex:210-211).

**What the extended version says, in three places:**
- ext_discussion.tex:99-100: floating-point re-execution 'cannot confirm an exact answer'.
- ext_evaluation.tex:259: 'beats exact (integer) re-execution'.
- ext_evaluation.tex:172 calls fp32 'the fastest exact one'.

**The sources.** RX_reexecution_baselines.md:28-32 labels `float` as 'the fp model', not the integer model. dynint8 is excluded as 'not exact (it computes a different model)' (RX:12), but fp32 also computes a different model than the certified int8 one. The code encodes fp32 as exact: sp2027_numbers.py:72 `EXACT = ("int", "int_fp32gemm", "float", "bf16")`.

**What the definition changes.** Against the exact integer model (int_fp32gemm, 141.6 s), the ratios are 7.2× (Fiat–Shamir) and 7.6× (interactive) (RX:56-57), not 4.7× and 5.0×. At 16k tokens they are 2.9–3.5×, not 2.0–2.4× (V_LONG_b:21-24).

**Recommendation.** Decide one definition before the EPYC run fixes \CpuRatioN:
- **Option (a):** 'exact' means bit-identical to the certified int8 model. Then use the integer re-execution (likely about 7× on the Xeon), drop float and bf16 from `EXACT`, and fix the abstract, tab:options footnote a, the fig:breakeven caption and ext_evaluation.tex:172.
- **Option (b):** keep fp32 as the conservative baseline, call it 'the fastest re-execution (exact integer or fp32 of the checkpoint)', and fix ext_discussion.tex:99-100 and ext_evaluation.tex:259.

Either way, make the reexec variant set a CLI argument of sp2027_numbers.py. (Effort: small (text + one tuple), but must be decided before the final run.)

**Verifier.** The evidence is accurate.
- The body calls fp32 the 'fastest exact' re-execution: main.tex:73, intro.tex:116-117, evaluation.tex:187, 210-211, 230-231, 240-242.
- ext_evaluation.tex:170-171 says the same.
- The code agrees: sp2027_numbers.py:72-74 puts float and bf16 in EXACT and documents them as 're-execute the fp model exactly'.
- RX:28-34 labels float 'the fp model'. RX:59-60 gives 7.6x and 7.2x against int_fp32gemm (141.6 s).
- V_LONG_b:21-28 gives integer 1,356.4 s at 16k, which is 2.9-3.5x.

The contradiction is worse than reported. ext_discussion.tex:96-102 says fp32 'cannot confirm an exact answer' and then, in the same paragraph, '5.0x faster than the fastest exact re-execution'. That 5.0x is 93.2/18.5, the fp32 figure.

Why medium, not high: under either definition the 4.7x is a lower bound (the integer baseline gives 7.2x), so no number is inflated. The problem is terminology and consistency, and it must be settled before \CpuRatioN is filled.

An extra issue the reviewer missed:
- sp2027_numbers.py:541 sets CpuRatioN = floor(min over 64 and 2,048 tokens).
- On the Xeon the 64-token Fiat-Shamir ratio is 2.9x (RX:53), so N would be 2.
- The abstract sentence talks only about a 2,048-token prompt, so the macro's scope and the sentence's scope do not match.

### P4 [medium, confirmed] 'V1's own additions stay below 1 MB per proof' is contradicted by the repo's own byte tables

**Where:** paper_sp2027/sections/evaluation.tex:160-161; ext_v1.tex:164-166; docs/improvements/C2_v1_proof_size.md:129-138; docs/improvements/F3a_real_weight_runs.md (B3 row)

**Evidence.** **From C2's own table, in Kpre** (v0's proof is the claims alone):
- Full OPT-1.3B: V1 910.6 MB vs values 896.7 MB, so 13.9 MB of V1 additions (C2:135). This matches y_ℓ over 𝕂: about 18,432 rows × 32 B × 23 blocks ≈ 13.6 MB.
- Llama-2-7B 2/32: 111.3 vs 109.6 MB, 1.7 MB (C2:131).
- OPT-125M: 166.7 vs 164.1 MB, 2.6 MB (C2:129).

**On the full real OPT-6.7B** (F3a B3): `u` is 32.1 MB plus 2.5 MB of cut data.

**The doc and the extended version.** The doc's own claim, 'folds with 8 planes, multiplicities, GKR transcripts … under 1 MB everywhere' (C2:137-138), is wrong. The extended version scopes '<1 MB' to the multiplicities and GKR transcripts, and lists the eight folded rows separately (ext_v1.tex:164-166). The body generalises to all of V1's additions.

**Recommendation.** Reword the body to match the extended version: 'the new multiplicities and GKR transcripts stay below 1 MB; the folds over 𝕂 add tens of MB on 7B models'. Fix C2:137-138. (Effort: small.)

**Verifier.** The body sentence at evaluation.tex:160-161 is wrong, and the evidence is stronger than reported.

In Kpre, v0's proof is the claims alone (C2:125-135). So V1 total minus V1 claims is V1's whole addition:
- GPT-2@64: 2.6 MB
- OPT-125M: 2.6 MB
- Llama-2-7B 2/32: 1.7 MB
- OPT-1.3B: 13.9 MB

F3a:301 gives `u` = 32.1 MB on the real OPT-6.7B.

This also breaks the body's own explanation at evaluation.tex:161-163, which says the gain is smaller 'where the parts that V1 leaves unchanged weigh more'. GPT-2's 1.46x in Kpre comes entirely from V1's additions; in Kpre there are no unchanged non-claim parts.

ext_v1.tex:164-166 scopes the claim correctly. C2:137-138 is ambiguous at best.

Why medium, not high: it is an explanatory sentence. The measured ratios and conclusions are unaffected, and the fix is one line.

### P5 [medium, confirmed] # Real-weight cost/V1 numbers, including the abstract's 1.7×, come from the legacy α=0.55 graph, not the certified model; reproducibility scope is overstated

**Where:** main.tex:73; paper_sp2027/sections/intro.tex:114-115; evaluation.tex:57-60,164-166; defence.tex:51-54; docs/improvements/F3a_real_weight_runs.md:229-231; docs/improvements/D6a_reproducible_commitment.md:59,80-83,95; code/experiments/4_defence_benchmark/slurm/sp2027.sh:300-301,311-312

**Evidence.** **Which graph the numbers come from:**
- F3a: 'The real graph is the F2 one: `--smooth 0.55`, calibrated on one 2,048-token window' (F3a:229-231). This covers real OPT-1.3B within 9% and 6%, v0 9–17% smaller, OPT-6.7B 1.73×.
- C2:143: OPT-125M 1.68× and 1.77× used `--smooth 0.55`.
- The paper defines the agreed model as the certified recipe: α = 0.6, four train windows (quality.tex:67; D6a:49-50,92-94).
- None of these numbers carries a todo, but the final real tier passes `--recipe certified` (sp2027.sh:311-312). Its header comment still says 'calibrated on one 2,048-token window' (sp2027.sh:300-301).

**Reproducibility (defence.tex:52-54):**
- 'OPT-125M also across operating systems and torch versions' holds only for the earlier recipe (D6a:80-83).
- The 'machines' are four nodes of one CPU model with one torch version (D6a:59).
- The roots were computed under `policy="auto"` (D6a:95), not the paper's cnn18c.
- The Llama builder is untested across machines (D6a:101-103).

**Recommendation.** Put todos on the abstract, intro and evaluation real-weight figures so the final-run values replace them. Name the recipe where legacy values remain.

In the final run, record the certified models' graph digests and the cnn18c roots on the EPYC and L40S, and compare them with D6a's (d9daa017…, 16de86ea…).

Scope the cross-OS sentence to the earlier recipe, or rerun it. (Effort: small-medium.)

**Verifier.** The evidence is accurate.

Recipe:
- F3a:229-231 says '--smooth 0.55, calibrated on one 2,048-token window'.
- C2:143 uses --smooth 0.55.
- The abstract's 1.7x (main.tex:73), intro.tex:114-115 and evaluation.tex:57-60 carry no todo for the recipe.
- sp2027.sh:300-301 still describes the legacy graph, while sp2027.sh:311-312 passes --recipe certified.

Reproducibility (D6a):
- The nodes are four Xeon Silver 4114 machines on torch 2.5.1 (D6a:59).
- The cross-OS and cross-torch check covers only the earlier recipe (D6a:80-83).
- The commitment was computed with policy="auto" (D6a:95).
- The Llama builder was not checked across machines (D6a:101-103).

One more overstatement: defence.tex:53-54 says 'their commitments across … CPU or GPU encoding'. OPT-125M was encoded on the CPU only (D6a:64-65), so the CPU-vs-GPU comparison covers OPT-1.3B alone.

### P6 [medium, confirmed] # Extended setup-proof text is still on the auto plan and contradicts the body and its own tab:setup

**Where:** paper_sp2027/sections/appendix_defence.tex:175-177,386-391; ext_defence.tex:103-108,121-122,150-166,176; ext_intro.tex:152-153

**Evidence.** Each pair below is extended text against body or table text:
- appendix_defence.tex:175 and ext_defence.tex:103-104: 't=35 … 2^-35'. Body defence.tex:211-212: '2^-23 (2^-34 under Fiat–Shamir)'. tab:params: t = 23 / 34 (app_params.tex:27).
- ext_defence.tex:121: 'At a=1,000, Llama-2-7B pays 11.3 GB once and 8% more column bytes'. tab:setup two lines below: 9.8 GB, and 65.6→79.4 MB (+21%) (ext_defence.tex:140).
- ext_defence.tex:151-166: 2,833.5 / 11,272.7 MB; 1.6% / 0.4%; 1.7% / 0.6%; 't0_j reaches 12% of n_j'. Body: 2.4 / 9.8 GB and 2.2% / 0.5% (defence.tex:219,224-225). Under cnn18c, t0 ≤ 4,936 at n ≥ 2^18, which is ≤ 1.9% of n.
- ext_defence.tex:176 and ext_intro.tex:152-153 say '1.6%'.
- 3.2–6.6× (appendix_defence.tex:177, ext_defence.tex:108) comes from an old-plan note outside the repo.
- The 'radii at most a quarter' claim (appendix_defence.tex:388) predates cnn18c.

Commit ffdf49d recomputed the body only.

**Recommendation.** Rerun `setup_bytes.py --model gpt2 llama2-7b opt-6.7b llama2-13b --policy cnn18c --amortise 100 1000`, which prints the byte split. Regenerate Block 2, the 2^-t sentence, the 1.6% sentences and the radius claim from its output, and store the output in artifacts/results. (Effort: small.)

**Verifier.** Every cited line checks out.
- appendix_defence.tex:175 and ext_defence.tex:103-104 say t=35 and 2^-35, against the body's 2^-23/2^-34 (defence.tex:211-212) and tab:params' 23/34 (app_params.tex:26-27).
- ext_defence.tex:121 says '11.3 GB … 8%', but tab:setup two lines below gives 9.8 GB, and 79.4/65.6 = +21%.
- Block 2 (ext_defence.tex:150-166) still has 2,833.5/11,272.7 MB, 1.6%/0.4%, 1.7%/0.6% and '12% of n_j'. Under cnn18c, 4,936/2^18 = 1.9%.
- ext_defence.tex:176 and ext_intro.tex:152-153 still say 1.6%.
- The 3.2-6.6x comes from idea_trust-rigor.md:31, which describes the old plan ('t_g … 35–133').

`git show --stat ffdf49d` confirms that the recomputation touched only the body's defence.tex, not ext_defence, appendix_defence or ext_intro.

This is confined to the extended version, so medium.

### P7 [medium, confirmed] # Commitment plan: the paper claims one plan (cnn18c, never named) but measured cells use auto (cnn16c/17c/18c by model); script defaults do not reproduce the paper

**Where:** paper_sp2027/sections/app_params.tex:10-11; ext_defence.tex:65-67,72-75,127-129; evaluation.tex:92,127-157,203; ext_discussion.tex:129-131; code/experiments/6_improvements/params_table.py:3,64; setup_bytes.py:3,27-29; code/experiments/4_defence_benchmark/slurm/sp2027.sh:66; code/src/pvi/fullcheck/plans.py:56

**Evidence.** **What the paper says.** ext_defence.tex:72-75: 'Our runs use one plan for every model: n = max(2^18, 2 next_pow2(k))'. Three lines earlier it describes the six-candidate auto rule and the LeNet-5 auto result (ext_defence.tex:65-67).

**Which plans the measured cells used:**
- tab:eval-v1 mixes plans by row without saying so: GPT-2 and OPT-125M cnn18c; Llama-2-7B and OPT-1.3B cnn16c; Llama-2-13B cnn17c (C2:124-135).
- tab:eval-llm, tab:eval-compare and tab:options' 325 MB mode-C row are auto-plan and interactive. ext_discussion.tex:131 says 124.7 MB of that proof is columns, against tab:params' 65.6 MB (app_params.tex:28).
- fig:breakeven and discussion.tex:28-29 ('about 22 queries') use the cnn18c Fiat–Shamir byte model, 302.9 MB per query (ev_breakeven_draft.json panel1). Next to tab:options' 325 MB, a reader computes 20.7.
- EV_deployment_case.md:83-84 flags this.

**Script defaults.**
- params_table.py (:3,:64) and setup_bytes.py (:27) default to `--policy auto`, so their docstring commands do not give tab:params or tab:setup.
- setup_bytes.py also defaults to `--amortise 1000` only.
- The package selects the plan through the PVI_C_POLICY environment variable (sp2027.sh:66).
- The library default is 'paper' (plans.py:56).

**Recommendation.** Name the plan (cnn18c) once in the body and in the tab:params caption. Default every script's `--policy` to cnn18c, as a CLI argument rather than an environment variable.

State the per-row plan in tab:eval-v1 until the v1 tier replaces it. Put todos on every auto-plan mode-C cell (325 MB, 6.6 GB, 124.7 MB, LeNet-5's 15 columns) or label them. (Effort: small.)

**Verifier.** The evidence is accurate.
- ext_defence.tex:65-67 describes the six-candidate rule; ext_defence.tex:72-75 says 'one plan for every model'.
- C2:124-135 shows the tab:eval-v1 plans: cnn18c, cnn16c, cnn17c.
- ext_discussion.tex:131 gives 124.7 MB of columns, against 65.6 MB in tab:params.
- ev_breakeven_draft.json panel1 gives 302,885,128 B at Fiat-Shamir and 64 tokens (Q*=22.25), while 6.74 GB / 325 MB = 20.7.
- EV:83-84 flags this.

The defaults are worse than reported; there are three:
- `auto`: params_table, setup_bytes (which also defaults to --amortise 1000), v1_bytes, gen_bench, setup_timing, open_ab, v1_phases.
- `paper`: bench.py:1015, perf.py:177, serve_ab.py:867, and the library (plans.py:56).
- `cnn18c`: only sp2027.sh:66 and sp2027_numbers.py:1083.
- strong_gpu.sh:105 falls back to auto when run alone, which its own usage line suggests (strong_gpu.sh:8). Only sp2027.sh's tier_strong passes cnn18c down (sp2027.sh:243).

One part of the recommendation is weak: naming 'cnn18c' in the paper is unnecessary. The rule is already stated in the captions of tab:params and tab:setup.

### P8 [medium, partly] # GPU re-execution baselines come from two harnesses that disagree by up to 40%, and body and extended version quote different values

**Where:** paper_sp2027/sections/evaluation.tex:199,211-212,253-254; ext_discussion.tex:97-100; ext_evaluation.tex:230-249; docs/improvements/A7_serving_overhead.md:294,315-320,326-329; docs/improvements/F1_reexecution_baseline.md:34,51

**Evidence.** **Llama-2-7B fp16 on the 2080 Ti.** Body tab:options footnote a: 0.034 / 0.64 s, from A7:315-320. ext_discussion.tex:99-100 and tab:eval-reexec: 0.037 / 0.57 s, from F1:34.

**Llama-2-7B int8.** The tab:eval-reexec row gives 0.16 / 2.08 s (F1, before A2), while its caption and tab:options give 0.15 / 1.65 s (A7).

**OPT-1.3B fp16.** The paper uses 0.11 s (A7:294), but F1 measured 0.18 s; A7:326-329 says the 10% cross-check was 'not met, in both directions'. tab:eval-reexec's OPT-1.3B cell mixes F1 int (0.94 s) with A7 fp16 (0.11 s).

The 64-token 0.15 s that claims_ext could not locate is A7:315 (0.145 s, extrapolated).

**Recommendation.** Use one harness (serve_ab's U1/U2) for every GPU re-execution cell in the body and the extended version. Let sp2027_numbers.tab_eval_reexec replace both, and delete the F1-era rows or label them. (Effort: small.)

**Verifier.** The numbers are accurate (A7:315-329; F1:30-36). The body's fp16 0.034/0.64 and the extended version's 0.037/0.57 for the same quantity are a real inconsistency.

But part of the finding overstates:
- Two of the three 'mixes' are disclosed in tab:eval-reexec's caption (ext_evaluation.tex:229-233), which notes 'except OPT-1.3B's fp16 time, measured on the full model' and the pre-A2 integer times with 'Llama-2-7B: 0.15 and 1.65 s'.
- No claim depends on these numbers; evaluation.tex:255 says 'We do not claim that verification beats re-execution on a GPU'.
- The final run regenerates tab:eval-reexec and tab:options' GPU row from the gpuv tier's reexec.py (sp2027.sh:276-278; sp2027_numbers.py:547-553), not from serve_ab, so the recommendation misstates what the generator uses.

The real remaining risk: in the final run, the intro's '19-40x an fp16 one' (serve_ab U2, serve tier) and tab:eval-reexec's fp16 (reexec.py) still come from two harnesses, which disagreed by 40% before (A7:326-329).

### P9 [medium, partly] # Numbers without a todo that the final run will change, and earlier-platform numbers it will not cover despite the 'one platform, no extrapolation' promise

**Where:** paper_sp2027/sections/evaluation.tex:46-48,117-120,172-175,186,205,253; main.tex:73; intro.tex:113-115; generation.tex:46-50

**Evidence.** **The promise.** evaluation.tex:46-48: 'The final tables will come from one platform … full models, no extrapolation'.

**Numbers with no todo that the final run will change:**
- Abstract and intro 2.0–2.3× (the C rows of Llama-2-7B and OPT-1.3B were cnn16c, and will be cnn18c).
- 1.7× and 1.73× (legacy recipe → certified).
- 6.6 GB and 325 MB (auto, interactive → cnn18c, Fiat–Shamir).

**Earlier-platform numbers no tier regenerates:**
- GPT-2 generation: 0.29 s / 0.59 s / 87.3 MB (2080 Ti, interactive).
- The improvement factors 3.7–4.3× and 2.0–2.7× (2080 Ti A/B harnesses).
- tab:options' 'with V1' row (45.4 s, extrapolated from 2–3 blocks, V_V1N) sits under a caption that says 'full models' (evaluation.tex:186,205).
- OPT-1.3B GPU verifier 'extrapolates to about 1.1 s' (evaluation.tex:253).

**Recommendation.** Write a coverage list mapping each body number to its final-run cell or 'kept, earlier platform'. Add todos for the first group and labels for the second, or soften the platform sentence.

sp2027_numbers.py already writes a coverage.txt; make it cover the abstract and intro macros too. (Effort: small.)

**Verifier.** The first group is right:
- The abstract and intro figures 2.0-2.3x and 1.7x, and the intro's 6.6 GB (intro.tex:113-115), carry no todo and will change.
- The generator writes macros only for CpuRatio*, Crossover, serve, Gen, Long and HtoD (sp2027_numbers.py:537-874). There is none for the V1 ratios, the real-weight 1.7x or 6.6 GB.
- The 2080 Ti A/B factors (3.7-4.3x, 2.0-2.7x) and the GPT-2 generation cell are regenerated by no tier.

The second group is partly wrong:
- The 45.4 s is regenerated by c-v1-llama2-7b-2048 (sp2027.sh:189), and its text carries '\todo{F6: V1 timings}' (evaluation.tex:175).
- The OPT-1.3B GPU verifier's 1.1 s is regenerated by g-v0-opt-1.3b-Kpre and g-rx-opt-1.3b (sp2027.sh:269-278), and carries '\todo{L40S: both on one GPU}' (evaluation.tex:254).
- The 'one platform' sentence is a statement about the final tables. evaluation.tex:47-51 already says the numbers shown come from earlier runs and names those platforms.

This largely overlaps P5 and P7. The coverage-list recommendation is still worthwhile.

### P10 [medium, confirmed] Kpre key lifetime is missing from the body: interactive r=5 gives 2^-128 only for ≤2^18.7 verdicts per key

**Where:** paper_sp2027/sections/defence.tex:137-139,158-161,170-173; evaluation.tex:74,204,213; ext_proofs.tex:905-912

**Evidence.** **What the body says.** thm:sound states the Q·ε_K reuse bound (defence.tex:158-161). The parameters paragraph gives r = 5 at λ = 128 (defence.tex:172). The body's Kpre cells are interactive (tab:eval-llm; tab:options row 'Ours, Kpre … full^c', where footnote c only says 'Designated verifier').

**What the extended version adds** (rem:stateful, ext_proofs.tex:907-912): 226·p^-5 = 2^-146.7, which keeps Q_V·ε_K ≤ 2^-128 'for Q_V ≤ 2^18.7 verdicts only, about 4·10^5'. With r = 7 the limit is 2^80.5.

**Recommendation.** Add one sentence after thm:sound or in footnote c of tab:options: the verifier counts verdicts and rotates the key, or uses r = 7. State that the abstract's Kpre number uses r = 7. (Effort: small.)

**Verifier.** The theorem itself is correct and states the Q·ε_K reuse bound explicitly (defence.tex:158-161). The parameters paragraph gives only r=5 for ε_C (defence.tex:170-173).

The body never gives the concrete per-key budget. rem:stateful (ext_proofs.tex:905-912) gives it: 2^-146.7, so Q_V ≤ 2^18.7, about 4·10^5 verdicts. The comment at ext_proofs.tex:913-915 confirms the arithmetic.

This matters because tab:options lists the interactive Kpre row (r=5) with the guarantee 'full^c', and footnote c says only 'Designated verifier' (evaluation.tex:204, 213).

The abstract's figure is non-interactive (RX:53-55: r=7, budget 2^80.5), so the headline is not affected. The fix is one sentence.

### P11 [medium, confirmed] The paper names about 60 code identifiers, test files and docs paths that a refactor would break; two cited docs are not in the anonymous export

**Where:** paper_sp2027/sections/ext_proofs.tex:43,65-70,93-108,196,204-205,214-215,224-252,290-294,358-406,529-583,604-656,694-754,926-954,995; generation.tex:69; app_params.tex:57; scripts/export_anonymous.py:30-38

**Evidence.** **What the extended version binds itself to.** ext_proofs.tex:43: 'Each proof names the functions of `code/src/pvi/fullcheck` that it models'.

**Kinds of names rendered:**
- Private functions: `Challenger._key`/.folding/.columns/.absorb_large, `_claims_message`, `_fold_message`, `_open_message`, `_absorb_statement`, `cut_protocol._query`, `_kpre_y_ok`, `_disagrees`, `_columns_shape_ok`, `_Builder.last`/`calibrate_all`, `_rope`, `_attention`.
- Public API: `build_decoder`, `with_lm_positions`, `build_opt_from_hf`, `build_llama_from_hf`, `matches_benchmark_graph`, `TokenSampler`, `gumbel_table`, `run_query`, `run_cut_query`, `claimcodec.decode*`, `params_for`, `plans.column_bits`, `CutPlan.from_graph`.
- A test function: `test_a_pruned_block_commits_its_q_on_its_own`.
- Test files: tests/test_pruning.py, test_fs_tree.py, test_claimcodec.py.
- Verdict labels and transcript/domain strings: `pvi/fullcheck/v3`, `pvi/leaf/v1`, `pvi/sampler/v1`. The last one is also in the body (generation.tex:69).
- Docs paths: docs/improvements/D1_D2_fiat_shamir_binding.md and G3_inrange_attack.md (ext_proofs.tex:656,995).

**Current state.** A script check confirms every name resolves today.

**Export.** export_anonymous.py:30-33 exports no docs/, so the two cited docs dangle in the artifact. Changing a domain string or label also changes challenges and golden outputs.

**Recommendation.** Treat these names as a frozen interface: rename only in the same commit as the paper. Add a test that extracts the \texttt names from the .tex files and checks them against the code.

Keep the domain strings and verdict labels byte-identical. Either export the two docs files or cite artifact READMEs instead. (Effort: medium.)

**Verifier.** The evidence is accurate.
- ext_proofs.tex:43 says each proof names the fullcheck functions it models.
- Spot checks resolve: _kpre_y_ok (cut_protocol.py:582), _columns_shape_ok (protocol.py:1342), 'pvi/fullcheck/v3' (protocol.py:336), 'pvi/sampler/v1' (token_sampler.py:73, 77; also rendered at generation.tex:69), and test_a_pruned_block_commits_its_q_on_its_own (test_pruning.py:103).
- export_anonymous.py:30-33 INCLUDE has no docs/, so the doc paths at ext_proofs.tex:656 and 995 dangle in the artifact.

The reviewer missed a larger export gap: EXCLUDE (export_anonymous.py:35-38) also drops *.sbatch and code/experiments/4_defence_benchmark/slurm/. The anonymous artifact therefore lacks sp2027.sh and strong_gpu.sh, the package that produces the final numbers.

### P12 [low, not verified (low)] # Prompt-length crossovers and the short-prompt count differ between body and extended version; 4,500 does not follow from the current byte model

**Where:** paper_sp2027/sections/discussion.tex:28-29,35-36; ext_discussion.tex:133-134,151-153; ext_v1.tex:200-201; docs/improvements/EV_deployment_case.md:78-80

**Evidence.** **Mismatches:**
- Body: 'about 2,100 tokens'. Extended version: 'about 2,050' (ext_discussion.tex:152, ext_v1.tex:201).
- Body: 'about 22 queries'. Extended version: 'more than about 21 short prompts' (ext_discussion.tex:134). EV's doc removed the 21 from the body because 'it contradicted 22'.

**The 4,500 figure.** With the cnn18c Fiat–Shamir model in ev_breakeven_draft.json (fixed 106.65 MB, 3.1415 MB per token), v0 crosses at 2,112 tokens. V1 at the paper's 2.3× crosses at about 4,856. The 4,500 comes from V1_SPEC.md:56 (outside the repo; based on 2,050).

**Recommendation.** Have ev_breakeven.py print both crossovers, cite them in the body and the extended version, and fix the 21 and the 2,050. (Effort: small.)

### P13 [low, not verified (low)] Small overreaches and unstated scope in measured claims

**Where:** paper_sp2027/sections/intro.tex:65-66,68; attack.tex:207,250-251; quality.tex:68-72; v1.tex:90-91; app_params.tex:55-56; ext_discussion.tex:185

**Evidence.** - 'values outside the neurons' natural ranges': 5 of 40 forged values were inside (real_llm_attack.json `untargeted.within_natural_range` = 0.125).
- 'every test query' means 94 queries per model (tables_l40s/measured_summary.csv `attack_float flip_success` n = 94).
- tab:sok gives '14,182 (ResNet-18) to ≥454,248', while attack.tex:209-210 starts at 2,316 (LeNet-5).
- The OPT-1.3B and 6.7B HellaSwag scores use a 2,000-document subset, which is not stated (F2c_zero_shot.md:166,183).
- 'our proofs hold up to 4.3·10^9 leaves' is a design bound for the unmeasured full Llama-2-13B (V1_SPEC.md:347).
- tab:params footnote b's '≤2^24 leaves' and params_table.py:90's hard-coded job-1007816 counts reflect the lmax-24 timing run. The final package uses --lmax 27 (sp2027.sh:132).

**Recommendation.** Make one-clause fixes:
- 'mostly outside (35/40)'.
- 'all 94 test queries'.
- Start tab:sok's range at 2,316.
- Mention the HellaSwag subset.
- 'the largest model we target'.
- Regenerate tab:params with the final run's `--v1-cells`. (Effort: small.)

### P14 [low, not verified (low)] Literature or derived values with no curated record, and comments quoting stale numbers

**Where:** paper_sp2027/sections/evaluation.tex:201,216-219,295-296; attack.tex:214,286; main.tex:9-11; intro.tex:15; ext_evaluation.tex:27-28; code/experiments/4_defence_benchmark/slurm/sp2027.sh:300-301,357,375

**Evidence.** **Values with no curated record:**
- CommitLLM's 144 kB and 0.66 ms per token and 12–14% overhead are absent from code/artifacts/comparison/literature/reported_benchmarks.csv (grep finds no CommitLLM).
- TOPLOC/SVIP '(100%; ≥95%)': sp2027_research/attack_theory/sok.md:33 supports 'accepted exactly like an honest one' only; no source for ≥95% was found.
- zkGPT 'shapes imply 32 tokens' has no record.
- The Wilson interval 91.2–100% is correct (recomputed 0.9124) but no script computes it.

**Stale comments:**
- main.tex:9-11 and intro.tex:15 still describe 'the 4.6x' as interactive and extrapolated.
- The ext_evaluation.tex:27-28 header cites 1.6/17.2 s and 3.5/78.8 s.
- sp2027.sh:300-301 says 'one 2,048-token window' (the tier passes --recipe certified).
- sp2027.sh:357 says 'the cell the paper's text quotes; 2.8 GB'; the paper says 2.4 GB.
- sp2027.sh:375 says 'The paper's Llama-2-7B row', which the paper does not have.

**Recommendation.** Add CommitLLM, TOPLOC and SVIP rows to the curated literature CSV with page references. Drop or source '≥95%'. Compute the Wilson interval in the script that writes real_llm_attack.json.

Clean the comments as part of the professional-comments pass. (Effort: small.)

## Paper: high-level review

High-level PC/advisor review of paper_sp2027 (repo root = .../scratchpad/wt_sp2027, branch sp2027). I read the body (main.tex plus every \input section), the inventories claims_body.md, claims_ext.md and package.md, the earlier reviews (REVIEW.md, REVIEW2.md, drafts/h5_early_review_20261012.md, CUT_PLAN.md) and SP2027_PLAN2.md. I skimmed the extended-version files. Read-only; no build.

VERDICT. As it stands, a demanding PC would score it about 2/3/2, the same as the mocks. It can reach a credible borderline 3/3-4/3, but only by reframing and by honest, complete numbers on one platform. More engineering will not get it there.

The parts that are finished and S&P-relevant:
- the lower bound for adaptive, value-aware spot-checkers at any locality;
- the SoK of nine schemes;
- the exploit on real LLMs, including in-range edits.

The defence is admittedly a known composition: Freivalds plus Ligero/Brakedown evaluation, intro.tex:189. Its evaluation is the fragile half:
- 34 \todo markers in evaluation.tex;
- proofs that reach the size of the weights after one long prompt;
- a provider overhead of 19-154x fp16;
- a headline CPU ratio measured against a weak baseline.

CORE RECOMMENDATION: one thesis that links the halves. "No spot-checker can be fully sound (theorem, with the attack as its witness and the SoK as its scope). Soundness needs checks that mix all rows of each weight matrix. We give the cheapest hash-only construction and measure exactly what full soundness costs, and for whom it pays." Under this framing the defence does not need to win on speed. It needs complete, honest, single-platform numbers.

MUST-DO, by date:
- By 19 Oct:
  - Send the disclosure notices (M8).
  - Decide the title and the abstract structure (H2, H1). Both freeze on 10 Nov.
- By the 26 Oct feature freeze (package changes):
  - Add a production CPU inference baseline, e.g. llama.cpp Q8_0/bf16 on the same 8 EPYC threads, as one extra cpuv job (H1).
  - Make sure the mode-C 2,048-token CPU verifier cell lands in the body (H1).
  - Put the real-weight cells on --recipe certified, and decide whether Llama-2-7B quality is a must tier (H5).
  - Wire generated/ macros into the .tex files and fix the pipeline bugs (H3).
  - Classify every body number by where it comes from; this sets the scope of the submission branch (M7).
- By the 30 Oct gate:
  - Fallback tables from existing full-model 2080 Ti/Xeon records, so the paper can go in with zero todos (H3).
  - Drop the base/new split from the body (H3).
  - Deployment-regime and provider-cost paragraph (H4).
  - One certified model per architecture (H5).
- 9-10 Nov: set the abstract's "at least N" by the plan's rule, without conflating modes C and Kpre.

SHOULD:
- M1: lead the LLM attack with the adversary's choice of layer, and argue the worst-case notion.
- M2: compress V1 to about 0.8 page and frame it as the communication floor of the recompute design.
- M3: demote the setup proof.
- M4: comparisons on the same hardware only.
- M5: n and spread in captions, plus peak memory.
- M6: no contribution may rest only on the extended version.
- M10: fund every addition with explicit cuts.

NICE-TO-HAVE:
- M9: section order (shared threat model, evidence before theory, Related Work later).
- L1: abstract details and naming.

BIGGEST ACCEPTANCE RISKS, in order:
1. Practicality and the headline baseline (H1, H4).
2. The "two incremental halves" perception, and "Breaking" overclaiming against a preprint that already notes 1/N (H2).
3. An unfinished or unfavourable final run, with the abstract frozen before the run ends (H3).
4. The certified model differs across the quality, cost and real-weight evidence (H5).
5. The lower bound does not reach transformers (M1).

GO/NO-GO: go is reasonable if H1-H5 are done by the 30 Oct gate. The theory, SoK and attack are already complete. With the fallback tables, a slip of the L40S run lowers the paper's polish but does not make it unsubmittable.

Arithmetic marked "my arithmetic" uses code/artifacts/results/ev_breakeven_draft.json and transformer shapes. I did not re-run anything.

### H3 [high, confirmed] # MUST (pipeline before 26 Oct, fallback by 30 Oct): one final code, one platform, no 'base/new', fallback tables

**Where:** paper_sp2027/sections/evaluation.tex:46-52,73-102,116-120; code/experiments/4_defence_benchmark/slurm/sp2027.sh:46-48,66; code/experiments/5_comparison/sp2027_numbers.py:336-337,360-361

**Evidence.** 1. evaluation.tex:46-48 is in the future tense ('The final tables will come from one platform ... The numbers shown come from earlier runs'). evaluation.tex holds 34 \todo markers (my count of non-comment lines).
2. tab:eval-llm puts 'base' (pre-improvement code, interactive, course-era auto plan) beside 'new' (all improvements). The final run is Fiat-Shamir and cnn18c (package.md §0.2; sp2027.sh:46-48,66).
3. The caption says 'Proof bytes do not change', but Fiat-Shamir changes them: GPT-2, mode C, 64 tokens is 14.8 MB interactive (evaluation.tex:90) against 16.6 MB under Fiat-Shamir (evaluation.tex:141,278).
4. No .tex file inputs paper_sp2027/generated/, and no sp2027_numbers.py macro is used (package.md §0.1). The roughly 60 todos are filled by hand.
5. The final run happens on a borrowed machine, 2-8 Nov (SP2027_PLAN2.md:66-68), after the code freeze, and close to the abstract freeze of 10 Nov.

**Recommendation.** 1. The body reports final-code numbers only: one column per metric, one platform per table, the machine named in the caption.
   - Move base-versus-new and the 'Improvements' paragraph (evaluation.tex:116-120) to the extended version as an A/B ablation.
   - Consequence for the branch: base code paths stay only as non-default options for that ablation and need not run in the final package.
2. Before the freeze, wire generated/*.tex macros into the body, and fix the confirmed sp2027_numbers.py bugs (package.md §0.3). Filling the tables on 8-10 Nov is then mechanical.
3. By the 30 Oct gate, build a complete fallback version of every body table from the existing full-model 2080 Ti/Xeon records, so the paper can be submitted with zero todos if the L40S window slips. No '--' cells in the body.
4. Rewrite evaluation.tex:46-52 to describe what is shown. (Effort: L: 2-3 days (pipeline wiring about 1 day, fallback tables about 1 day, text 0.5 day).)

**Verifier.** Verified. evaluation.tex:46-48 is written in the future tense. evaluation.tex has 34 real \todo markers (35 matches, one of them the \providecommand). tab:eval-llm (evaluation.tex:74-77) labels the base columns interactive, plan auto, and says 'Proof bytes do not change'. The final run is Fiat-Shamir under cnn18c (sp2027.sh:46-48,66). sp2027_numbers.py:336-339,360-361 fills the proof column with 'new where measured, else base', mixing protocols in one column. Fiat-Shamir changes bytes: GPT-2 C 64 is 14.8 MB interactive (evaluation.tex:90) against 16.6 MB under FS (evaluation.tex:141,278). No body .tex inputs a generated/ folder and no sp2027_numbers macro is used (grep; package.md section 0.1). tab_spread and tab_memory are built (sp2027_numbers.py:677,831,1002-1004), but nothing in the paper references them. The final run is 2-8 Nov, after the 1 Nov code freeze (plan:68-69). The caption is true today (base vs improvements) but becomes false once the 'new' columns come from the final run. A single-platform fallback is already a plan mitigation (plan:1190, R1), but nothing implements it yet.

### H1 [medium, partly] # MUST: The headline CPU claim mixes two clients, and its baseline is weak

**Where:** paper_sp2027/main.tex:72-73; sections/intro.tex:115-118; sections/evaluation.tex:196-205,238-245; sections/discussion.tex:118-120

**Evidence.** 1. The abstract reads: 'It lets a client without weights or GPU check ... and a CPU client verifies a 2,048-token Llama-2-7B prompt 4.7 times faster than the fastest exact re-execution' (main.tex:72-73). The 4.7x is mode Kpre, whose key is precomputed from the weights (evaluation.tex:238-242; defence.tex:49-51).
2. The weightless client (mode C) has no 2,048-token time in the body: tab:options reads 'Ours, C & roots & 325 MB & 6.6 GB & 1.0 s & --' (evaluation.tex:203), and the Llama-2-7B 2,048 CPU/GPU cells are '--' (evaluation.tex:98).
3. The baseline is weak:
   - The 'fastest exact re-execution' is fp32, which RX_reexecution_baselines.md:29-33 labels 'the fp model', not the certified integer model (142 s).
   - It ran on a Skylake Xeon without VNNI, where the native int8 GEMM is slower than fp32 (RX:37-39).
   - The plan's own model gives about 2x on the EPYC (SP2027_PLAN2.md:593; risk R7 at :1196).
   - llama.cpp Q8_0 was planned but never run (SP2027_PLAN2.md:586); the only int8 engine measured is torch dynint8 (56 s; evaluation.tex:211).
4. The ratio excludes transfer of the 6.43 GB proof. The gain exists only above 0.59-0.70 Gb/s (discussion.tex:119-120).
5. My arithmetic: at 1 Gb/s, 51.5 s + 19.6 s = 71 s against 93.2 s, i.e. 1.3x; at 10 Gb/s, 3.8x.
6. A reviewer will ask 'why not run llama.cpp locally?' and 'what does the client you advertise (no weights) actually pay?'.

**Recommendation.** 1. Abstract and intro: separate the capability from the number.
   - Capability: mode C needs only the 32-byte roots, no weights, no GPU.
   - Number: give it for the mode it belongs to, e.g. 'a CPU verifier holding a key precomputed from the weights checks ... at least N times faster than re-executing the certified model on the same CPU', with the link-speed caveat, or give the measured mode-C time.
2. Make sure the cpuv tier's mode-C Llama-2-7B 64/2,048 cells (already in sp2027.sh, `--modes Kpre C`) reach tab:options and the text.
3. Add one external baseline job on the same EPYC threads: llama.cpp prefill at Q8_0 and/or bf16, labelled 'not exact, different model'. It is an external binary, so it does not touch the frozen code; schedule it before 1 Nov.
4. Define 'exact re-execution' once, as re-executing the certified integer model, and stop calling fp32 'exact'.
5. If the ratio against llama.cpp is about 1x or less, keep the claim against exact re-execution only, and lean on mode C and the theory. (Effort: M: about 0.5 day of writing plus 1-2 extra cpuv jobs (about 1-2 h on the EPYC); decide the abstract wording by 9 Nov.)

**Verifier.** Half right. The client-mixing and transfer points hold; the 'weak baseline' points mostly fall. HOLDS: (1) The abstract (main.tex:72-73) puts 'a client without weights or GPU' next to a Kpre number. The paper is also inconsistent about whether a Kpre client holds weights: intro.tex:30-31 and defence.tex:49-51 say it does not store them; discussion.tex:31 says 'A Kpre client that keeps the weights' and ext_discussion.tex:136 says 'holds the weights'. (2) There is no mode-C time for Llama-2-7B at 2,048 tokens: evaluation.tex:98 and :203 are '--', with \todo at :188. (3) Transfer is left out of the headline ratio. Break-even is 8*6.43e9/(93.2-19.6) = 0.70 Gb/s (discussion.tex:31-32). The arithmetic checks out: 1.31x at 1 Gb/s, 3.8x at 10 Gb/s. (4) The cpuv tier already has mode-C cells, so recommendation 2 is accurate (sp2027.sh:174-181, `--modes Kpre C`). FALLS OR OVERSTATED: (a) fp32 is 'the fp model' (RX:32), but it is faster than the exact integer pass (93.2 s vs 141.6 s, RX:31-32). Using it makes the baseline stronger, not weaker; the exact-integer ratio would be 7.2x (RX:60). Only the word 'exact' for fp32 is wrong. (b) llama.cpp was an alternative, not a separate commitment: plan:586 reads 'torch dynamic-int8 Linear, or llama.cpp Q8_0', and dynint8 was run. (c) The EPYC re-measure is already planned: abstract \todo at main.tex:73, R7/R11 at plan:1196,1200. One point that supports the EPYC risk: the exact int8 GEMM uses vpmaddwd (AVX-512BW), not VNNI (native_kernels.py:807), while plan:583 asked for VNNI on the EPYC. Line refs are wrong: discussion.tex has 65 lines, and the cited content is at :25-36. The numbers are measured and scoped ('same CPU', 'non-interactive'), so this is misleading framing rather than an unsupported claim. Still, the wording must be fixed before the abstract freeze on 9-10 Nov.

### H2 [medium, partly] MUST (title freezes 10 Nov): 'Breaking' overclaims; reframe as a necessity/sufficiency thesis

**Where:** paper_sp2027/main.tex:64,72; sections/intro.tex:47-49,84-87,122-133,154-161,187-189; sections/attack.tex:297-298,302-306

**Evidence.** 1. Title: 'Certified but Compromised: Breaking and Fixing Lightweight Proofs of Inference' (main.tex:64).
2. The body itself says:
   - 'most meet the probabilistic guarantees they state; we do not call them broken' (intro.tex:84-85; attack.tex:304-306).
   - The main target 'note[s] that one path catches a single changed node with probability at most 1/N ... and leave[s] full soundness ... open' (intro.tex:47-49).
   - The refuted statement is a remark: 'trivially extends to full soundness' (attack.tex:297-298).
   - The underlying calculation is 'classical' and 'folklore' (intro.tex:154-161).
   - For the defence, 'the novelty is this combination, not a new primitive' (intro.tex:189).
3. All three mock reviewers agreed that 'the two halves are each incremental on their own' (REVIEW.md:305-309).
4. Contribution bullet 1 (intro.tex:124-125) presents the attack as the first contribution. Yet its key quantity, 1/N, is stated by the target paper.

**Recommendation.** 1. Lead with what is new and complete: a lower bound against adaptive, value-aware, trace-reading spot-checkers at any locality d, an SoK of nine schemes, and the matching hash-only construction with its measured price.
2. Title options:
   - 'Spot-Checks Are Not Proofs: A Lower Bound for Sampled Proofs of LLM Inference and a Hash-Only Fully Sound Protocol';
   - or keep 'Certified but Compromised' with the subtitle 'Why Spot-Checked Proofs of Inference Cannot Be Sound, and What Full Soundness Costs'.
3. Abstract: open with the theorem and use the OPT-6.7B edit as its witness. Do not present the 1/N gap of Anchuri et al. as our discovery: 'the gap they note is exploitable on real LLMs and no sampler closes it'.
4. Add one 'thesis' paragraph to the intro: the bound says what any sound weight check must do (mix all rows); the defence is the cheapest hash-only way to do it; the evaluation measures its price.
5. Reorder the contributions to match: (1) bound plus SoK; (2) attack on real LLMs, including in-range edits; (3) protocol; (4) evaluation. (Effort: M: 0.5-1 day of writing; decision by about 19 Oct so H5-mid (29 Oct) reviews the new framing.)

**Verifier.** Every quote checks out. Title: main.tex:64. 'we do not call them broken': intro.tex:84-85 and attack.tex:304-306. The 1/N gap is stated by Anchuri et al.: intro.tex:47-49. 'classical'/'folklore': intro.tex:153-158. 'novelty is this combination': intro.tex:189. Mock-review consensus 'each incremental': REVIEW.md:305-306. The tension between the title and the text is real. But 'Breaking' is not baseless. The paper refutes two stated claims: binary-classifier full soundness on the dog-vs-cat ResNet-18 (intro.tex:68-70; attack.tex:296-298,306-307) and TensorCommitments' negl(lambda) goal (attack.tex:307-308). It also shows that the protocol Anchuri et al. left open for large models is not fully sound. The reframing and the title options are editorial opinion, not a defect. This is a positioning risk with reviewers, so medium rather than high.

### H4 [medium, partly] MUST: State the deployment regime and the provider's price up front, not in footnote f

**Where:** paper_sp2027/sections/intro.tex:28-30,108-120; sections/evaluation.tex:218-221; sections/discussion.tex:113-124

**Evidence.** 1. The intro gives the provider overhead only at its best point: '4.1-8.6x our exact integer forward pass and 19-40x an fp16 one' (OPT-1.3B at 2,048 tokens; intro.tex:111-112).
2. The full range, '9.8-10.6x (Kpre) and 19-154x (C) fp16 inference' for Llama-2-7B (extrapolated), sits in tab:options footnote f (evaluation.tex:218-221). It sits beside GPU TEE below 7% and CommitLLM at 12-14% (intro.tex:166-167,176).
3. The integer engine alone costs 4-10x over fp16 (A7_serving_overhead.md:339).
4. Deployment numbers:
   - The bytes break-even is about 22 queries at 64 tokens and 1 query at 2,048 tokens (2.3 with V1) (discussion.tex:116-118).
   - The proof reaches the weights' size at about 2,100 tokens (discussion.tex:123).
   - The Discussion itself says 'The regime is narrow' (discussion.tex:115).
5. Why per-query full soundness matters against sampled audits is never argued directly. The relevant facts exist: CommitLLM accepts an edited one-token response with about 99.4% at 5% audits (attack.tex:260-261), and the threat model lets the adversary choose which queries to tamper with (attack.tex:48-50).

**Recommendation.** 1. Add one intro paragraph, mirrored in the Discussion, that names who gains, with numbers:
   - a CPU-only auditor or gateway with 10 Gb/s or more to the provider (about 3.8x at 10 Gb/s for Kpre at 2,048 tokens, from ev_breakeven inputs);
   - mode-C clients that make few short queries or check many models (32 B of roots per tree);
   - on-demand audit of chosen high-stakes answers with full per-query soundness.
2. State the provider's price as the full measured range, next to the TEE and CommitLLM figures.
3. Make the per-query argument explicit: a provider that alters one chosen answer passes a sampled audit with probability about one minus the audit rate, but passes ours only with 2^-128.
4. Say in the threat model who publishes C_M: any one honest recomputation suffices. (Effort: M: about 0.5 day of writing; uses existing numbers, refreshed after the serve tier.)

**Verifier.** The numbers are accurate. intro.tex:111-112 quotes OPT-1.3B at 2,048 tokens (19-40x fp16). That is mode C's cheapest point: A7:290 gives 203x at 64 tokens. It is not Kpre's best point, since Llama-2-7B Kpre is 9.8x (A7:316,320). The full range 9.8-154x sits in footnote f (evaluation.tex:218-221). A7:339 gives the 4-10x integer floor. GPU TEE <7% is at intro.tex:166-167; CommitLLM 12-14% at :176. Overstated: the regime is stated, in discussion.tex:25-32 ('Who the protocol serves'; 'The regime is narrow' at :27), and the intro points there (intro.tex:120). The per-query argument is partly present: attack.tex:48-50,73-74 and tab:options footnote b (evaluation.tex:212-213). All Discussion line numbers cited (113-124) are wrong, because the file has 65 lines. Moving the full overhead range and the 'who gains' paragraph to the front is a sound writing fix with existing numbers. The early review already raised it as A-W2 (h5_early_review:70-74). It is not a wrong result.

### H5 [medium, partly] # MUST: The certified model whose quality is shown is not the model whose cost and acceptance are shown

**Where:** paper_sp2027/sections/quality.tex:28-45,84-88; sections/evaluation.tex:55-61,91,282; sections/defence.tex:51-54

**Evidence.** 1. tab:quality certifies the α=0.6 recipe on OPT only (quality.tex:28-45). The real-weight cost, acceptance and V1 results (evaluation.tex:57-61,164-166; abstract 1.7x) used the legacy `--smooth 0.55` graph (claims_body.md §12.3; F3a_real_weight_runs.md:229-231).
2. The headline model, Llama-2-7B, has no certified-quality number; OpenLLaMA-7B stands in (quality.tex:84-86).
3. Qwen3-4B's certified model is ×5.3 perplexity (quality.tex:86-88), i.e. unusable. It is still a cost row (evaluation.tex:91) and the model of the Maverick comparison (evaluation.tex:282).
4. The root-of-trust sentence 'OPT-125M also across operating systems and torch versions' (defence.tex:53) was checked only for the legacy recipe, not the certified one (claims_body.md §4, D6a:83-86).
5. The package already passes `--recipe certified` (sp2027.sh:311-314). Llama-2 quality is an optional tier (sp2027.sh:391-396, PVI_OPTIONAL).

**Recommendation.** 1. Use one certified model per architecture throughout. Cite only the certified-recipe real-weight cells from the final run, or label the legacy ones as such.
2. If the licence is in place, make the Llama-2-7B quality tier a must. Otherwise call the benchmark 'Llama-2-7B shapes' everywhere and say that OpenLLaMA-7B has the identical integer graph.
3. Mark Qwen3-4B as a shape-only cost row in the table and the Maverick note, or move it out of the body.
4. Either repeat digest_check for the certified OPT-125M on a second OS/torch (if it fits the laptop; see the memory note) or limit defence.tex:51-54 to what was checked. (Effort: M: about 0.5 day of writing; cells already in the package; D6a recheck about 2 h if feasible; tier decision before 26 Oct.)

**Verifier.** CONFIRMED: (1) tab:quality certifies only OPT at alpha 0.6 (quality.tex:28-45). (2) The real-weight runs behind evaluation.tex:57-61,164-166 and the abstract's 1.7x used the F2 graph '--smooth 0.55', calibrated on one window (F3a:230-232). The paper never says this differs from the certified recipe. (3) The cross-OS/torch sentence at defence.tex:52-54 holds only for the legacy recipe: the certified-recipe Windows check was stopped (D6a:80-83). 'their commitments across ... CPU or GPU encoding' also overreaches for OPT-125M, which was encoded on CPU only (D6a:64-65). This is a small unsupported claim. (4) The final package already uses --recipe certified (sp2027.sh:311-312,314), so the mismatch goes away if the text is updated after the run. WEAK: the cost rows use random weights by design, and the paper says so (evaluation.tex:55-56). So the Qwen3-4B cost and Maverick rows are shape costs either way. The OpenLLaMA-7B stand-in is stated explicitly (quality.tex:84-86), and Qwen3-4B x5.3 is disclosed (discussion.tex:48-49). The Llama-2 quality tier is optional and needs a licence (sp2027.sh:37,387-396). This is misleading and inconsistent, not a wrong result.

### M2 [medium, confirmed] SHOULD: Shrink V1 in the body and justify it as the communication floor of the recompute design

**Where:** paper_sp2027/sections/v1.tex:37-185; sections/evaluation.tex:122-175; sections/discussion.tex:117-124

**Evidence.** 1. V1 takes about 1,365 source words, a theorem and a lemma, plus tab:eval-v1 (evaluation.tex:126-157).
2. It buys 2.0-2.3x fewer bytes at about 10x the prover time and 2.3x the verifier time (evaluation.tex:168-175).
3. It moves the break-even only from 1 to 2.3 queries, and the link from 0.59-0.70 to 0.47 Gb/s (discussion.tex:117-120).
4. The paper concedes that 'Committing the activations, as zkLLM does, would remove this growth' (discussion.tex:124) but never says why it stops short.
5. My arithmetic: a Llama-2-7B block sends q,k,v,o (4×4,096), gate and up (2×11,008) and down (4,096), i.e. 42,496 values per token. ×32 blocks ×2,048 tokens = 2.79 GB, which matches V1's 2.8 GB (tab:options, evaluation.tex:205). V1 already sends about one byte per weight-operation output.

**Recommendation.** 1. Cut V1 in the body to about 0.8 page. Keep the idea, the windows, the integer-multiplicity lemma, one paragraph on the reduction to the existing column check, and the theorem. Move the P1-P5 cut rules and the eq:v1-final derivation to the appendix or the extended version.
2. Add the framing: V1 reaches about one int8 value per weight-operation output, which is the floor for any protocol whose verifier recomputes the non-weight operations itself. Going lower means committing activations and proving the non-linear operations, i.e. a SNARK-style prover. State it as an argument unless it is proved.
3. Say which client V1 serves: mode C on slow links.
4. This frees about 0.5 page for H1, H4 and M5. (Effort: M: 0.5-1 day.)

**Verifier.** v1.tex has exactly 1,365 non-comment words, with a lemma (v1.tex:94) and a theorem (:149). The trade-offs are as stated: evaluation.tex:159-175. Break-even moves 1 to 2.3 queries and 0.59-0.70 to 0.47 Gb/s at discussion.tex:30-32 (not :117-120). 'Committing the activations ...' is at discussion.tex:36 (not :124). The arithmetic is correct: (4*4096 + 2*11008 + 4096) * 32 * 2048 = 2.785e9, which matches V1's 2.8 GB (evaluation.tex:205). The 'floor' framing is an argument, not a theorem, as the reviewer says. The case for medium is the page squeeze (see M10). Otherwise this is editorial.

### M4 [medium, confirmed] # SHOULD: Comparisons with published systems mix hardware and guarantees; print factors only on the same hardware

**Where:** paper_sp2027/sections/intro.tex:109-110; sections/evaluation.tex:262-314

**Evidence.** 1. The intro's '46x faster than zkLLM on an A100' compares our L40S with their A100 (intro.tex:109-110).
2. tab:eval-compare prints factors up to 31,910x against SNARKs that 'ran on CPUs' (evaluation.tex:275-287,303-308).
3. Stored zkLLM runs on the same L40S model exist (844 s for Llama-2-7B, 2,628 s for 13B; SP2027_PLAN2.md:1055). They appear only in a todo (evaluation.tex:311-312).
4. The CNN rows (zkCNN, ZKML on VGG-16) are irrelevant to the LLM thesis and produce the largest factors.
5. The early review rated this A-W3 (h5_early_review_20261012.md:76-89).

**Recommendation.** 1. Body: zkLLM rows from the same-GPU stored runs, with their caveats (prover only; one skip connection counted twice), plus Maverick as the closest guarantee class, with factors.
2. Every other system: raw values only, no factor.
3. Move the CNN rows to the extended version.
4. The intro factor must be the same-GPU one, or go. (Effort: S: 2-3 h.)

**Verifier.** intro.tex:109-110 compares our L40S with zkLLM's A100. Factors reach 31,910x against CPU-run zkSNARKs (evaluation.tex:275-276,303-304). The CNN rows give the largest factors (4,422x and 31,910x). Same-GPU zkLLM runs exist: code/artifacts/results/zkllm_l40s/summary.csv gives 843.84 s (7B) and 2,628.40 s (13B), and plan:1055 lists the caveats: prover only, one skip connection counted twice. They are used only in a todo (evaluation.tex:311-312). This was A-W3 in h5_early_review:76-89. Note that the same-GPU factor favours us: 2,628/17.6 is about 149x, against 46x on the A100 figures. The A100 sentence does name the hardware, so it is not hidden.

### M5 [medium, confirmed] # SHOULD: Statistics, soundness-test framing and peak memory

**Where:** paper_sp2027/sections/evaluation.tex:52,63-67,110; code/experiments/4_defence_benchmark/slurm/sp2027.sh:67-68

**Evidence.** 1. 'Times are medians after a warm-up' gives no n and no spread (evaluation.tex:52). The final run times 5 or 3 queries (sp2027.sh:67-68, QH=5, Q=3). The pipeline builds tab_spread and tab_memory, but nothing in the paper uses them (package.md §5).
2. The counts of 3,792 accepted and 2,512 rejected (evaluation.tex:63-67) read as security evidence. They can only test the implementation; the 2^-128 rests on Theorem 3.
3. Peak memory is a todo (evaluation.tex:110). For the 'client without weights' story, a CPU verifier that needs more RAM than the 6.7 GB int8 weights undercuts mode C at long prompts.

**Recommendation.** 1. Every timing caption states n (queries × repetitions) and the min-max or IQR.
2. Present the accept/reject counts as implementation tests that fire every check, with the security claim pointing to Theorem 3.
3. Report the CPU verifier's peak RSS next to the int8 weight size. If it can stay below that size, say so. (Effort: S-M: about 0.5 day once the run is in.)

**Verifier.** evaluation.tex:52 gives no n and no spread. The defaults are QH=5 and Q=3 (sp2027.sh:67-68). tab_spread and tab_memory are generated (sp2027_numbers.py:677,831), but no paper .tex references them (grep). Peak memory is a \todo at evaluation.tex:110. The accept/reject counts (evaluation.tex:63-67) are implementation tests, while the 2^-128 rests on the theorem. Reframing them is a fair, small fix.

### M7 [medium, confirmed] SHOULD (before 26 Oct): Classify every body number by source; this defines the submission branch

**Where:** C:/Users/User/.claude/jobs/fcef220b/tmp/inv/claims_body.md §12-14; package.md §0.1

**Evidence.** 1. Many body numbers have no record in the repo (Rec✗): all of tab:eval-v1, RX re-execution, A7 serving, the certified column of tab:quality and its CIs, F2c zero-shot, F2b Llama/Qwen ratios, G3-min OPT-1.3B, F3a/F3b, V-LONG-b, and V-V1N's 45.4 s (claims_body.md §12.1).
2. Some body claims depend on scripts outside the worktree: sok_formulas.py and check_guessing.py (tab:sok, the theorem checks) and attribution.py (quality.tex:49-51) (claims_body.md §12.2).
3. tab:sok is a headline contribution, yet its derived numbers have no in-repo script.

**Recommendation.** Classify each body number as one of:
- (A) regenerated by the final run;
- (B) platform-independent, with a committed script plus record run from the frozen code (attack, quality, sampler TV, SoK formulas, digests);
- (C) literature, with a curated record.

Anything that fits none is re-run or cut. Use the table as the scope of the submission branch (task #30): bring sok_formulas.py, check_guessing.py, check_merkle_leak.py and attribution.py into the repo with their records. (Effort: M: about 1 day (classification plus moving scripts and records).)

**Verifier.** Verified. sok_formulas.py, check_guessing.py and check_merkle_leak.py exist only in scratchpad/sp2027_research/attack_theory. attribution.py exists only in wt_f2/code/experiments/9_quality, and find over wt_sp2027 returns nothing. git ls-files has no round-2 quality records (ppl2048_search2) and no adaptive_game_opt-1.3b.json; results/9_quality holds only round 1 (claims_body.md section 12.1). tab:sok's derived numbers (attack.tex:250-286) therefore have no in-repo script. The fix is cheap: move the scripts and records. It directly defines the submission branch's scope (task #30). The artifact gap is real but easy to close, so medium.

### M10 [medium, confirmed] SHOULD: Pay for every addition with named cuts; the body has no slack and a desk-reject risk

**Where:** paper_sp2027/sections/*.tex; paper_sp2027/main.log (stale 29-page build); drafts/h5_early_review_20261012.md:255-263

**Evidence.** 1. The body is about 13 pages before 34+ todos become numbers (h5_early_review:257-258). Three table* floats can drift past p. 13 (h5_early_review:260).
2. main.pdf and main.log in the folder are the stale 29-page build of 10 Oct ('Output written on main.xdv (29 pages').
3. H1, H4 and M5 add about 0.4-0.5 page.
4. Cuts that cost no claim:
   - V1 compression, about 0.5 page (M2);
   - setup-proof remark, about 0.25 (M3);
   - the 'Improvements' paragraph and the base columns (evaluation.tex:116-120; H3);
   - CNN rows of tab:eval-compare (M4);
   - the generation 'Distribution' detail (generation.tex:272-283) down to 2 sentences;
   - the MNIST 'Other samplers' detail (attack.tex:195-205);
   - the SmoothQuant derivation (quality.tex:58-67).

**Recommendation.** 1. Keep a page ledger.
2. Apply the cuts above before adding text.
3. Rebuild after each fill and check that no float lands past p. 13.
4. Delete the stale main.pdf so it cannot be uploaded by mistake. (Effort: S-M: about 0.5 day across the passes.)

**Verifier.** main.log:634 says 'Output written on main.xdv (29 pages', and main.pdf and main.log are both dated 10 Oct. h5_early_review:255-260 projects about 12.8 body pages, leaving about 0.2 page of slack. There are three table* floats: tab:sok (attack.tex:237), tab:eval-llm (evaluation.tex:73) and tab:eval-compare (:262). One line ref is wrong: the 'Distribution' paragraph is generation.tex:87-98, not :272-283, since the file has 98 lines. The other cuts exist where cited: attack.tex:195-205, quality.tex:58-67, evaluation.tex:116-120. The desk-reject page risk is real, and deleting the stale PDF is cheap.

### M1 [low, confirmed] SHOULD: Make the lower bound's reach to transformers explicit, and lead the LLM attack with the adversary's chosen layer

**Where:** paper_sp2027/sections/attack.tex:127-135,171-173,223-231; sections/discussion.tex:139-143

**Evidence.** 1. The hard instance is an AND-of-copies ReLU network: 'We do not claim that the instance of (a) embeds in a transformer' (attack.tex:172-173).
2. On a trained MLP, value-aware ranking places the unconstrained edit first for 75-95% of queries (attack.tex:132-135).
3. On OPT-1.3B, the best sampler mixture catches in-range edits on 48%, 6.8% and 2.2% of paths at layers 6, 12 and 18. The text then says 'Whether an edit adapted to the sampler evades it ... we leave open' (attack.tex:224-228).
4. At layer 23 the mixture catches only 0.08-0.09%, i.e. 2^-40 needs more than 30,000 paths (G3_min_adaptive_game.md:302-313). The body mentions this last, in parentheses.
5. A theory reviewer will say the bound concerns a toy network, and an ML reviewer will read the 48% as 'samplers work on LLMs'.

**Recommendation.** 1. Argue in 2-3 sentences why the worst case is the right notion: soundness of a proof system is quantified over models, so a sampler that is sound only on 'natural' networks needs a definition of natural that no scheme gives. The theorem shows no spot-checker is sound for ReLU networks in general.
2. Rewrite the LLM paragraph so that the adversary's choice of layer comes first: at layer 23, 0.08% per path, more than 30,000 paths for 2^-40, and 39 of 40 prompts succeed with lower-only edits. Then mention the earlier layers.
3. Keep the adapted LLM edit as a stated limitation. (Effort: S: 2-3 h of writing.)

**Verifier.** The quotes are accurate: attack.tex:172-173, :132-135, :224-228. Layer 23 is 0.079-0.092% with s_40 above 30,000, and the lower-only edit succeeds on 39/40 (G3_min:302-305,313; attack.tex:222). Small inaccuracies: the 0.08% figure is not in parentheses, only the uniform figure is (attack.tex:227). discussion.tex:139-143 does not exist; the content is at discussion.tex:51-55. The paper already scopes the bound (attack.tex:127-129; discussion.tex:51-52). Leading with layer 23 was already suggested (h5_early_review:92). This is worthwhile 2-3 h polish; no claim is wrong.

### M3 [low, partly] SHOULD: The setup proof lies outside the paper's own threat model, is untimed, and is still listed as a contribution

**Where:** paper_sp2027/sections/defence.tex:206-226; sections/intro.tex:101-102,128-130; sections/discussion.tex:129

**Evidence.** 1. The paper targets open weights (intro.tex:118-119; discussion.tex:126-128) and says 'For open weights, recomputing C_M is cheaper than a setup proof and also settles which model is committed' (defence.tex:225-226).
2. The setup proof is nevertheless a listed extension and contribution (intro.tex:101-102,128-130).
3. It takes about 0.4 page of the body with a \todo{F6: setup time} (defence.tex:221), and the Discussion admits 'The setup proof is untimed' (discussion.tex:129).
4. The 'forced columns' lower bound is proved only in the extended version (defence.tex:221-223 cites fullversion).

**Recommendation.** 1. Demote it to a 3-4 sentence remark: the split commitment motivates the testing phase; a 2.4 GB setup proof amortises it for a client that cannot recompute C_M; details in the appendix.
2. Drop it from the contribution bullets, or keep it as a sub-clause.
3. Make sure the setup tier (sp2027.sh:364-367) delivers the time. If it does not, remove the size and time claims from the body.
4. This saves about 0.25 page. (Effort: S: about 2 h.)

**Verifier.** Confirmed: the setup proof is untimed (defence.tex:221 \todo; discussion.tex:41, not :129). It is listed in the extensions and contributions (intro.tex:101-102,128-130). The forced-columns result cites fullversion (defence.tex:222-223). The setup tier exists (sp2027.sh:364-367). Refuted: the claim that it is 'outside the paper's own threat model'. The threat model explicitly defers 'the other case' to sec:setup-proof (attack.tex:50-53), and defence.tex:212-213 motivates it for a client when nobody it trusts can recompute C_M. 'About 0.4 page' is high: about 21 source lines (defence.tex:206-226) come to about 0.3 page. Side note: sp2027.sh:356-357 still says 2.8 GB, while the paper's cnn18c figure is 2.4 GB (defence.tex:219), so the comment is stale. Demoting the section is a fair way to buy space, not a defect.

### M6 [low, partly] SHOULD: No contribution may rest only on the extended version

**Where:** paper_sp2027/sections/intro.tex:126; sections/attack.tex:173-178; sections/appendix_short.tex:14-15; sections/defence.tex:221-223

**Evidence.** 1. Contribution 2 says 'met up to a factor O(λ+log L) at every locality' (intro.tex:126). The block-locality proof is 'in the extended version only (appendix_attack.tex), since no appendix text of the same size could make room for it here' (appendix_short.tex:14-15); the body cites fullversion for it (attack.tex:177).
2. The forced setup columns also rest on fullversion (defence.tex:222-223).
3. The body cites fullversion 15 times (grep count over the body sections).
4. The CFP says reviewers need not read appendices (CUT_PLAN.md:7-13), and an anonymous artifact is read even less.

**Recommendation.** Either:
- add a one-paragraph proof sketch of the block case to appendix_short.tex (pay for it with a tab:params row or a remark), or
- drop 'at every locality' from the contribution and state the block result as a remark.

Keep the other fullversion-only facts as supporting remarks, never as claims. (Effort: S: 2-4 h.)

**Verifier.** Accurate: intro.tex:126 claims 'at every locality', and the block case cites fullversion (attack.tex:173-178). The forced setup columns also cite fullversion (defence.tex:222-223). However, appendix_short.tex:14-15 is a source comment, not paper text. The count is 12 fullversion citations in the body sections plus 2 in appendix_short, not 15. The CFP, as summarised in CUT_PLAN.md:13, encourages theory papers to put proofs in the artifact repository, so a proof in the extended version complies. The remaining risk is only that a contribution bullet leans on it. A one-paragraph sketch, or softening the bullet, is cheap.

### M8 [low, confirmed] MUST (user action, 15-19 Oct): Send the disclosure notices and turn the conditionals into facts

**Where:** paper_sp2027/sections/intro.tex:87-88; sections/attack.tex:311-318

**Evidence.** 1. Both places are conditional todos: 'only once the notices of 15-19 Oct are sent; else delete' (intro.tex:88), and the dates and replies are todos (attack.tex:314-317).
2. The CFP requires disclosure by the rebuttal (CUT_PLAN.md:17-18).
3. Two claims are pointed: the binary-classifier full-soundness refutation and 'TensorCommitments' stated goal ... not achieved' (attack.tex:306-308). Their authors may sit on the PC.

**Recommendation.** 1. Send the notices this week: Anchuri et al. on the binary-classifier remark, TensorCommitments on the negl(λ) goal, the others as a courtesy.
2. Write the ethics paragraph as facts, with the replies, or 'no reply by 2 Nov'.
3. If TensorCommitments' authors dispute the claim, soften it to 'the layer selection as described'. (Effort: S: about 1 h of team time plus follow-up.)

**Verifier.** The conditional todos are at intro.tex:87-88 and attack.tex:313-317. The pointed claims are at attack.tex:306-308. But the action is already scheduled: the todo names 15-19 Oct, and plan R10 (plan:1199) covers late replies. The paper text already has a conditional fallback. The CFP rule is at CUT_PLAN.md:19-20; :17-18 is the generative-AI rule. It sets the rebuttal deadline, so there is no CFP blocker. The paper's own wording ('Before submission we wrote') does require sending before submission. Team action item, not a defect.

### M9 [low, partly] NICE: Narrative order: a shared threat model, evidence before theory, Related Work later

**Where:** paper_sp2027/sections/attack.tex:33-55,77-231; sections/intro.tex:136-189

**Evidence.** 1. The threat model sits inside the attack section (attack.tex:33-55), but it also fixes the defence's setting: recomputable C_M, token ids, system prompt.
2. Related Work is Section 2 (intro.tex:136-189). It forward-references thm:guess and tab:sok (intro.tex:149-157) and pushes the first technical content to about p. 3.
3. The measured attacks (attack.tex:181-231) come after two theorems, so the reader meets the theory before the evidence.

**Recommendation.** Page-neutral reorder:
1. §2 Setting and threat model, shared by both halves.
2. §3 attack: edit → measured attacks → why sampling cannot fix it (thm:guess, thm:dich) → SoK.
3. §4-6 defence, V1, generation.
4. §7 evaluation, with quality first since it qualifies every later number.
5. Related Work just before the Discussion.

Do it in the 26-30 Oct writing window, then re-check the cross-references. (Effort: M: about 0.5 day plus a cross-reference check.)

**Verifier.** The evidence is accurate. The threat model is attack.tex:33-55, and it already states the defence's assumptions ('for our defence', :50-55). Related Work is §2 (intro.tex:136-189), with forward references at :149-157. The measured attacks (attack.tex:181-231) follow the theorems. The reorder is a matter of taste. A late restructure risks cross-references and float placement under a 0.2-page margin (M10). Optional.

### L1 [low, not verified (low)] NICE: Abstract details and naming

**Where:** paper_sp2027/main.tex:72-73; sections/intro.tex:65-66; sections/attack.tex:229-231; sections/v1.tex:41

**Evidence.** 1. The abstract cites the out-of-range OPT-6.7B edit, and the intro says 'with values outside the neurons' natural ranges' (intro.tex:65-66). In the stored record, 12.5% of the untargeted values were inside the ranges (claims_body.md §2).
2. The in-range 6.7B edit, which pre-empts the 'add range checks' objection, also changes the next token on 40/40 prompts (attack.tex:229-231).
3. The abstract never gives the proof's magnitude: 'proofs still grow with the prompt' (main.tex:73) hides that the proof is about the size of the weights at 2,048 tokens. It also omits the quality cost (×1.008-1.076; quality.tex:40-42).
4. 'V1' reads as a version number, and the casing of 'v0'/'V1' is inconsistent (v1.tex:41). The protocol has no name.

**Recommendation.** 1. Cite the in-range edit in the abstract.
2. Say 'proofs about the size of the int8 weights at 2,048 tokens (V1: 2.3x smaller)'.
3. Add the certified-model perplexity cost in a clause.
4. Remove the red todo before 10 Nov.
5. Optionally name the protocol and give V1 a descriptive name. (Effort: S: about 1 h.)

## High-level decisions audit

I audited 19 high-level decisions: the ones listed in the task, plus three that came up while reading (how the abstract's CPU ratio is defined, recipes being selected on the test split, and what the anonymous export contains). I read SP2027_PLAN.md, SP2027_PLAN2.md, the diary status lines and all seven inventory files, then checked every load-bearing claim against the code and the paper. Paths are relative to the worktree root (wt_sp2027, branch sp2027).

**Verdict.** The protocol-level decisions hold, are well argued and are implemented soundly. That covers the exact int8 model, Freivalds plus Reed-Solomon/Merkle without zero-knowledge, modes C/K/Kpre, Fiat-Shamir with the SHAKE tree mode, cnn18c with the setup cache, V1 as a non-default variant, last-block pruning, 11-bit probabilities and the int8-change rule. One check worth noting: the setup-cache key does include the int8 weights and biases, and a cache file is spot-checked when it is read.

**Where the risks are.** They sit in how measurements become paper numbers, and they cluster around the abstract, which is frozen at registration on 10 Nov:
- The number pipeline is not connected to the paper (67 `\todo` lines).
- The headline 4.7x compares our pruned verifier with an unpruned fp32 baseline.
- `CpuRatioN` is computed as the minimum over 64 and 2,048 tokens, which does not match the abstract's sentence.
- Every real-weight cost and V1 number, including the abstract's 1.7x, comes from the legacy, test-calibrated OPT graph, not the certified model.
- Tables mix protocols (interactive vs Fiat-Shamir), commitment plans and machines, while the paper says the final tables come from one platform.
- The anonymous export drops the run package and the docs that the extended version cites.

**Three decisions must be settled before the refactor and the 26 Oct feature freeze:**
1. Keep the five graph-building modules at their current paths. The graph digest hashes each cheap-op function's module path and name, so moving them changes every Fiat-Shamir transcript and the published certified digests.
2. Replace the course-report defaults with a paper preset, and turn the environment switches into CLI options. This has to land before the 28-29 Oct rehearsal so the branch stays exactly the run code.
3. Fix the re-execution baseline (prune it) and the record fields (V1 records have no `verifier_device`). The rest of the pipeline only reads records and can be fixed after the code freeze.

**Hardware-independent re-measurements to do before the freeze on the 2080 Ti or Xeon:**
- the certified-recipe real-weight V1 bytes;
- an 8-bit vs 11-bit probability A/B;
- the re-execution baselines with the last block pruned;
- recipe checks on the WikiText-2 validation split.

### HD-01 [high, confirmed] Hand-typed paper numbers vs generated macros: the number pipeline is not connected to the paper

**Where:** code/experiments/5_comparison/sp2027_numbers.py:236-245,540-541; docs/L40S_RUN_SP2027.md:11-13

**Evidence.** DECIDED (Plan 2 F6 part 11, SP2027_PLAN2.md:331,351): 'every table is regenerated by script (H3)'. The run guide promises 'Nothing has to be computed or copied by hand' (docs/L40S_RUN_SP2027.md:13).

STILL HOLDS? No.
- No .tex file inputs paper_sp2027/generated/; I counted 0 files with \input{generated.
- The paper has 67 \todo lines (main.tex plus sections), about 60 of them waiting for L40S/EPYC/F6 numbers.
- macro_name() runs str.title() (sp2027_numbers.py:243), so the emitted names (e.g. \CpuratiominLlamaTwoSevenB) do not match the docstring.
- Most generated tables have their own layouts, not the paper's (package.md §5: tab_eval_v1, tab_options, tab_eval_reexec, tab_gen, tab_setup_time, tab_long).
- Confirmed pipeline bugs from the inventory:
  - B1: quality records are grouped by graph name only, so two alpha values pool into one median (sp2027_numbers.py:763-777).
  - B2: tab_gen mixes real and random Llama-2-7B rows (:659-671).
  - B3: V1 records have no verifier_device (v1_bytes.py:368-421).
  - B5: --strict can never pass (:1054-1055).

RISK. In one week (2-9 Nov), about 60 numbers would be copied by hand into a frozen abstract and 13 pages. Transcription errors are likely, and an artifact reviewer would find the claim on L40S_RUN:13 false.

ALTERNATIVES.
- (a) \input generated tables and macros for everything.
- (b) Hybrid: macros for the ~25 numbers in running text, generated bodies for the four main tables (tab:eval-llm, tab:eval-compare, tab:options, tab:eval-v1), the rest copied by hand and checked by a script against cells.csv.
- (c) Copy everything by hand.

**Recommendation.** CHANGE NOW, choosing alternative (b).

Before the feature freeze (26 Oct), only the fixes that change record content: add verifier_device to V1 records (B3), and put alpha in the quality grouping key (B1). These alter what the frozen code writes.

After the code freeze (1 Nov), everything that only reads records, which is most of the work:
- the macro names (B7);
- the layouts of the four main tables;
- B2 and B5;
- a --compare-paper style checker that fails when a printed number differs from cells.csv;
- a corrected L40S_RUN_SP2027.md:13.

Cost: about 1-1.5 engineer-days, of which only about 0.25 d falls before the freeze. (Effort: 0.25 d before 26 Oct (record fields); 1-1.5 d after 1 Nov (pipeline and paper wiring).)

**Verifier.** Core claim holds. No .tex inputs paper_sp2027/generated/ (grep: 0 hits). main.tex:76-106 inputs only sections/*. docs/L40S_RUN_SP2027.md:13 says 'Nothing has to be computed or copied by hand'. macro_name() calls .title() (sp2027_numbers.py:243); evaluating it gives CpuratiominLlamaTwoSevenB, while CpuRatioN keeps camel case only because it is set directly (:541). B2, B3 and B5 check out: tab_gen has no real filter (:659-671); v1_bytes.py:368-421 never writes verifier_device; strict fails on any tb.missing (:1054), and base '--' cells and the OPT-1.3B/Qwen V1 columns are always missing. Corrections: (1) B1 does not need a fix before the freeze. ppl_llama.py:398 already writes alpha as its own field, so only the reader's grouping key (sp2027_numbers.py:765-770) has to change. B3 could also be fixed in the reader, by tier or by label gpuv-v1. (2) Counts: 61 non-comment \todo lines among the input files (66 with the unused generation_quality.tex). Only about 44 occurrences are in the 13-page body build; the rest are in the extended version.

### HD-02 [high, confirmed] # Headline claim (abstract): how 'CPU Kpre client vs strongest exact re-execution, at least N times' is defined

**Where:** paper_sp2027/main.tex:73; code/experiments/5_comparison/sp2027_numbers.py:540-541; code/experiments/7_reexec/reexec.py:244

**Evidence.** DECIDED (SP2027_PLAN2.md:88-97, R11 at :1200). The abstract quotes only numbers measured before the freeze, as 'at least N times', with N a lower bound on both CPUs. The baseline is the fastest exact re-execution on the same threads (RX, :589).

WHY. This is the abstract's only performance claim, and it answers reviewer C's path to a 4.

STILL HOLDS? Only partly. The abstract (main.tex:73) says 'a 2,048-token Llama-2-7B prompt 4.7 times faster'. Four inconsistencies:
1. The code sets CpuRatioN = floor(min of the 64- and 2,048-token ratios) (sp2027_numbers.py:540-541). On the Xeon the 64-token ratio is 2.9x (evaluation.tex:243-244), so N would be 2 while the sentence names a 2,048-token prompt.
2. The fp32/bf16 baseline is built unpruned (reexec.py:244 constructs FloatDecoder without prune_last), while the verifier's graph and the int baseline are pruned (reexec.py:226-227). A7 deliberately left reexec.py unpruned by default (docs/improvements/A7_serving_overhead.md:163). See HD-10.
3. The 'non-interactive' Kpre figure (19.6 s against 18.5 s) comes from r=7. That is the +64 grinding bits of params_for (protocol.py:132-150), applied to a mode that absorbs nothing (docs/improvements/FS1_fiat_shamir_speed.md §2.1). The paper justifies the extra 64 bits by random-oracle queries (defence.tex:196-204).
4. Plan 2's own model expects the EPYC (VNNI, AVX512_BF16) to bring the ratio down to about 2x (SP2027_PLAN2.md:593), so the 4.7x Xeon figure is the least robust number in the abstract.

RISK. The abstract is fixed at registration on 10 Nov, while the EPYC cpuv tier finishes during 2-8 Nov. An abstract claim could end up contradicted by our own final tables.

**Recommendation.** CHANGE NOW. The team decides by 26 Oct:
- the abstract's prompt length, say 2,048 tokens;
- N = floor(min over the Xeon and EPYC ratios at that length);
- the code is changed to match that definition.

Prune the float/bf16 baselines (HD-10) and re-measure the Xeon cell before the freeze (about 3-4 Xeon job-hours), so the abstract's lower bound is fair.

For Kpre, keep r=7 but state the reason ('2^-128 against 2^64 verifier queries'), or label those rows by r rather than 'non-interactive'.

Cost: about 0.25 d of code plus one Xeon job. The number changes by a few percent. (Effort: 0.25 d + 3-4 Xeon job-hours before 26 Oct.)

**Verifier.** sp2027_numbers.py:540-541 sets CpuRatioN = floor(min over the 64- and 2,048-token ratios). The abstract (main.tex:73) names a 2,048-token prompt, and evaluation.tex:243-244 gives 2.9x at 64 tokens non-interactive. The code also takes N only from the final run's records, so it never takes the plan's minimum with the Xeon (SP2027_PLAN2.md:93-95). Pruning asymmetry confirmed: reexec.py:226-227 prunes int, while :244 builds FloatDecoder without prune_last. The r=7 point is also confirmed: params_for adds 64 grinding bits whenever fiat_shamir is set (protocol.py:132-150). FS1 sec. 2.1 says Kpre absorbs nothing and was already non-interactive, yet defence.tex:196-204 justifies the 64 bits only by random-oracle queries. '18.5 s (19.6 s non-interactive)' is therefore r=5 against r=7, not interactive against FS. The plan's EPYC model of about 2x is at SP2027_PLAN2.md:593. The pruning effect is only a few percent; the length definition is the real decision.

### HD-04 [high, confirmed] The artifact is a curated anonymous export, which today leaves out the run package and the docs the extended version cites

**Where:** scripts/export_anonymous.py:30-39

**Evidence.** DECIDED (SP2027_PLAN2.md:991-995, H4). The artifact is a curated anonymous snapshot without git history, plus main_ext.pdf and a README with reproduction commands per table.

WHY. Anonymity: 356 commits have named authors, and the result files contain cluster home paths.

STILL HOLDS? The approach does, but the include list is wrong for the S&P artifact:
- INCLUDE covers only code/{src,tests,experiments,artifacts/...} (export_anonymous.py:30-32).
- EXCLUDE drops every *.sbatch and all of code/experiments/4_defence_benchmark/slurm/ (:37-38). That removes sp2027.sh, strong_gpu.sh and bench.sbatch, i.e. the package that produces every final number, along with real_llm.sbatch and adaptive_game_llm.sbatch.
- docs/ is not exported, although ext_proofs.tex:656,995 cite docs/improvements/D1_D2_fiat_shamir_binding.md and G3_inrange_attack.md in rendered text.
- Raw records carry hostname, SLURM partition, node list and PVI_* paths (bench.py:161-166,182-184). The REWRITES node regex (:48) knows only TAU node names, not the friend's.

RISK. The artifact cannot reproduce the final tables, and the extended version cites files a reviewer cannot open.

**Recommendation.** CHANGE NOW (for the export script). Doing it after the run is acceptable, since it does not touch measurement code.
- Export slurm/ and *.sbatch with partition and account strings rewritten, instead of excluding them.
- Export the new docs/ set (RUN_GUIDE, PROTOCOL, REPRODUCE), or replace the two diary citations in ext_proofs with appendix references.
- Strip host, partition, node-list and path fields from the final run's records before export, and extend REWRITES to the friend's host names (ask for them now).

Cost: about 0.5 d. (Effort: 0.5 d (any time before 15 Nov; host names needed from the friend before the run).)

**Verifier.** export_anonymous.py:30-32 INCLUDE has only code/{src,tests,experiments,artifacts/...}. :37-38 EXCLUDE drops every *.sbatch and all of slurm/, which removes sp2027.sh, strong_gpu.sh and bench.sbatch. Those scripts no longer hard-code partitions: bench.sbatch:8-10 takes them from the environment, and sp2027.sh has only the 'TAU nodes' comment at :9. So the exclusion's stated reason is stale. docs/ is not exported, yet ext_proofs.tex:656,995 cite docs/improvements/*.md in rendered text. bench.py:160-166,182-184 record hostname, partition, SLURM_JOB_NODELIST and PVI_* values. The REWRITES node regex (:48) and STATIC_DENY cover only TAU strings. The plan chose to leave docs out (SP2027_PLAN2.md:993), but the extended version's citations conflict with that. Easy to fix, but as built it blocks the artifact.

### HD-03 [medium, partly] # Certified OPT recipe (smooth-max alpha 0.6): every real-weight cost and V1 number, the abstract's 1.7x included, uses the legacy, test-calibrated graph

**Where:** code/src/pvi/fullcheck/real_weights.py:174; code/experiments/6_improvements/perf.py:195; code/experiments/8_v1/v1_bytes.py:324; docs/improvements/F3a_real_weight_runs.md:368-370

**Evidence.** DECIDED (SP2027_PLAN2.md:363; real_weights.CERTIFIED_RECIPE at real_weights.py:174). The recipe is smooth-max alpha 0.6, calibrated on four train windows, with 11-bit probabilities. Results: x1.076 / x1.026 / x1.008. It is the root-of-trust model (the D6a digests, scripts/artifact_README.md:73-74).

WHY. It is draw-stable, uses one recipe for every OPT size, and beats the percentile recipe (docs/improvements/F2a_quality_2048.md:176-199).

STILL HOLDS? The recipe itself does. But no reported real-weight cost or byte number uses it:
- The F3a cells, tab:eval-real, evaluation.tex:57-61 and the abstract's '1.7 times with the real OPT-6.7B' (main.tex:73) were all measured with --smooth 0.55. That graph was calibrated on one window from the end of the WikiText-2 *test* split (F3a:231,368-370).
- V1 bytes depend on clamp exceptions, and the certified recipe clips nothing.
- perf.py:195 and v1_bytes.py:324 still default to --recipe legacy. sampler_tv.py:93 hard-codes alpha 0.55 (behind generation.tex:93-98). zeroshot.py:331 defaults to --builder f2a.
- defence.tex:51-53 claims OPT-125M is reproduced 'across operating systems and torch versions'. D6a ran that check only for the legacy recipe (docs/improvements/D6a_reproducible_commitment.md:80-81).

RISK. The abstract's 1.7x (frozen 10 Nov) is a number for a model the paper does not certify. The final real tier (sp2027.sh:311-312, --recipe certified) may produce a different ratio after the abstract is fixed.

**Recommendation.** CHANGE NOW.
- Make certified the default in perf.py, v1_bytes.py and sampler_tv.py (add --recipe there), and in zeroshot.py. Keep legacy as a named non-default option.
- Before the freeze, re-measure the hardware-independent real-weight bytes with --recipe certified:
  - OPT-6.7B V1 bytes at 2,048 tokens (the F3a B3 command on a Xeon node, about 2 h);
  - OPT-1.3B v0/V1 bytes on the 2080 Ti.
- Quote the abstract number from those runs.
- Scope the cross-OS sentence to the earlier recipe, or re-run the Windows digest check on OPT-125M.

Cost: about 0.5 d plus 6-10 job-hours on the frozen builders. (Effort: 0.5 d + 6-10 Xeon/2080 Ti job-hours, before 26 Oct (abstract numbers before 9 Nov).)

**Verifier.** Accurate: F3a's real cells used --smooth 0.55, calibrated on one test-split window (F3a:229-231,368-370). The 1.73x (F3a B3) is the abstract's 1.7x. perf.py:193 and v1_bytes.py:324 default to --recipe legacy, sampler_tv.py:93 hard-codes 0.55, and zeroshot.py:331,365 defaults to alpha 0.5, not 0.6. D6a:80-81 limits the cross-OS check to the earlier recipe, so defence.tex:51-53 overclaims for the certified recipe. Overstated: the extended version already says the real-weight graphs use the earlier recipe (ext_evaluation.tex:124-127). The final real tier already passes --recipe certified (sp2027.sh:304-305). The quality and zero-shot numbers are certified: D6a says the digests equal F2c's builds. The real risk is that the frozen 1.7x moves slightly, through claim entropy and exceptions (1.8 MB of 2,680 MB). That is a disclosure and consistency issue, not an unsupported result.

### HD-05 [medium, confirmed] # Single final platform (L40S + EPYC) vs tables that mix machines and runs

**Where:** paper_sp2027/sections/evaluation.tex:45-51,73-77; code/experiments/5_comparison/sp2027_numbers.py:89,1082

**Evidence.** DECIDED (SP2027_PLAN2.md:42,348, chg 1; evaluation.tex:45-47: 'The final tables will come from one platform ... full models, no extrapolation, and medians with their spread'). EV (e) promised 'One platform per table, checked by a test' (SP2027_PLAN2.md:974).

WHY. Reviewer A-W6: the tables disagreed across machines.

STILL HOLDS? Not as built:
- tab:eval-llm keeps the 'base' columns from the course runs on TAU's L40S with an EPYC 9334 (REPORT_RUNS, sp2027_numbers.py:89; caption at evaluation.tex:74-77). Those runs are interactive, use plan auto, and use unpruned stand-ins for some mode-C cells (paper_assets.py:419-421).
- The final machine is specified only as 'AMD EPYC (Zen 4)' (docs/L40S_RUN_SP2027.md:20), so its SKU may differ from the 9334.
- tab_real expects 2080 Ti records next to the L40S ones (sp2027.sh:302-303).
- No test or pipeline check enforces one platform per table: nothing in tests/ or 5_comparison/ does it.

RISK. The paper's 'one platform' statement is unsupported, and base and new columns differ in machine, plan, Fiat-Shamir and pruning all at once.

**Recommendation.** CHANGE NOW (decision), CHANGE LATER (code).
- Drop the base columns from tab:eval-llm; tab:eval-improve already gives the same-machine A/B evidence. Otherwise relabel them 'earlier implementation, other host, interactive, plan auto'.
- Ask the friend for lscpu and nvidia-smi output now, so the paper can name the CPU.
- After the freeze, add a check in sp2027_numbers.py that every cell of a table comes from one host class. This needs the host field in the records; anonymise it only at export.

Cost: about 0.5 d. (Effort: 0.5 d (decision before 26 Oct; pipeline check after 1 Nov).)

**Verifier.** evaluation.tex:45-47 promises one platform. tab:eval-llm keeps base columns from TAU's L40S with an EPYC 9334, interactive (caption :74-77), filled from REPORT_RUNS (sp2027_numbers.py:89, --base :1082). tab_eval_llm takes the proof column from new, else base (:336-337). tab_real explicitly accepts f3a-* 2080 Ti/Xeon records (:697-705; sp2027.sh:301-303). The plan's EV(e) 'checked by a test' (SP2027_PLAN2.md:974) is not implemented: no host or platform check in tests/ or 5_comparison/. Mitigation: the caption names the base machine, so nothing is hidden, but it contradicts the one-platform sentence.

### HD-06 [medium, confirmed] Fiat-Shamir for the final run: captions, the setup proof and Kpre's parameters are inconsistent with it

**Where:** code/experiments/4_defence_benchmark/slurm/sp2027.sh:44-46,364-367; paper_sp2027/sections/evaluation.tex:74-77,185; paper_sp2027/sections/ext_defence.tex:127

**Evidence.** DECIDED.
- Plan 1: D1 binds the graph, D2 uses SHAKE-256 (SP2027_PLAN.md:54-55).
- Plan 2: FS1 (SP2027_PLAN2.md:240-282). Kpre absorbs nothing. Large messages are absorbed through a one-level SHAKE tree (the custom tree mode was preferred over BLAKE3; FS1 doc l.66). Each party's hashing is counted in its time.
- 'Every v0 and V1 cell runs under Fiat-Shamir'; only :int bench cells remain for zkLLM and Maverick (sp2027.sh:44-46).

WHY. B-W7 and chg 9 asked for non-interactive proofs by default.

STILL HOLDS? Yes, and it is sound (the ext_proofs.tex:524-525 argument; test_fs_tree). But:
- tab:eval-llm says 'interactive ... Proof bytes do not change' (evaluation.tex:74-77), and tab:options says 'ours interactive' (:185). Under Fiat-Shamir the GPT-2 64-token proof is 16.6 MB against 14.8 MB interactive.
- The untrusted-commitment cost is sized for interactive queries only: the setup tier has no --fiat-shamir (sp2027.sh:364-367; setup_timing.py:55 defaults to off), and the tab:setup caption says 'interactive queries' (ext_defence.tex:127). Every query cost the paper leads with is Fiat-Shamir.
- Kpre 'non-interactive' uses r=7 with no transcript, justified only by the random-oracle argument (defence.tex:196-204). See HD-02.

RISK. Readers see one protocol in the cost tables and another in the setup proof, and the Kpre parameters are unexplained.

**Recommendation.** KEEP Fiat-Shamir, and CHANGE NOW:
- Fix the two captions.
- Add the Fiat-Shamir columns to tab:setup. setup_bytes.py --fiat-shamir already models them, in minutes.
- Add one setup timing cell with --fiat-shamir (about 3 L40S hours), or state that the setup proof is timed for interactive radii.
- Keep r=7 for Kpre, with one sentence on the 2^64 verifier queries.

Cost: about 0.5 d. (Effort: 0.5 d before 26 Oct (one optional setup job in the run).)

**Verifier.** sp2027.sh:44-46 runs every v0/V1 cell under FS. The tab:eval-llm caption says 'interactive' (evaluation.tex:74), and tab:options says 'ours interactive' (:185) with 18.5 s, the interactive Kpre figure. The setup tier has no --fiat-shamir (sp2027.sh:364-367), setup_timing.py:55 defaults to off, and the tab:setup caption says 'interactive queries' (ext_defence.tex:127). GPT-2 at 64 tokens: 16.6 MB FS (evaluation.tex:141) against 14.8 MB interactive (:90). The Kpre r=7 justification gap is real (defence.tex:196-204 argues only from oracle queries). The recommendation is right; adding FS columns to tab:setup does change printed numbers.

### HD-07 [medium, confirmed] The graph digest binds Python identifiers (module.qualname of every cheap-op function): a hazard for the refactor

**Where:** code/src/pvi/fullcheck/graph.py:383,388,440; code/src/pvi/fullcheck/protocol.py:1624

**Evidence.** DECIDED (implicitly, Plan 1 D1, SP2027_PLAN.md:54). Fiat-Shamir absorbs IntGraph.digest() first (protocol.py:1624). _canonical identifies a function by f"{__module__}.{__qualname__}" plus its defaults and closure cells (graph.py:383-386), and a dataclass by its __qualname__ (:388), under the tag pvi/graph/v1 (:440). Captured constants are hashed, so the digest binds the constants correctly.

STILL HOLDS? Soundness does: the verifier recomputes with its own graph. But:
1. Moving or renaming graph, transformer, real_weights, real_llama or quantize (the planned regrouping, lib_fullcheck.md §6) would change:
   - every Fiat-Shamir transcript;
   - the drawn columns, and so mode-C multiproof bytes;
   - the stored graph_digest fields;
   - the certified digests published in scripts/artifact_README.md:73-74.
2. The paper describes the digest as 'every operation in order with its inputs, output and constants' (defence.tex:189-191). An independent implementation cannot reproduce it without our module paths.

The golden outputs were just captured (task #28), so an unplanned move would also invalidate the refactor's own verification.

**Recommendation.** KEEP for the submission. Do not move or rename the five graph-building modules or their cheap-op closures. Regroup only the other modules: split protocol.py, the kernels, V1 and the commitment code; none of their functions is hashed.
- Add a test that pins the digest of a fixed small graph, so any accidental change fails.
- Add one sentence in ext_proofs that an operation's kind is identified by its implementing function's qualified name.

CHANGE LATER (after 17 Nov): stable explicit op identifiers under pvi/graph/v2.

If the team insists on moving the modules, switch to stable identifiers now, recapture the goldens and re-run D6a's digests (Xeon, about 2-3 h per model) before 26 Oct. That costs about 1 d.

Cost of keeping: about 0.25 d. (Effort: 0.25 d (keep paths + golden-digest test); 1 d + re-digest if switching.)

**Verifier.** graph.py:383 hashes f'{__module__}.{__qualname__}' plus defaults and closure cells, :388 hashes a dataclass by its qualname, and the tag is pvi/graph/v1 (:440). protocol.py:1624 absorbs the digest first. Moving or renaming a builder module, or a method that defines a cheap-op lambda (the qualname includes <locals>), would change FS transcripts, the drawn columns, stored graph_digest fields and the README digests (artifact_README.md:73-74). No test pins a digest: no 64-hex constant or 16de86ea... in code/tests. defence.tex:189-191 does not mention that function names are part of the description. The recommendation to keep the paths and add a pinned-digest test is right.

### HD-08 [medium, confirmed] Library and script defaults are the course report's settings, and many options are environment variables rather than CLI arguments

**Where:** code/src/pvi/fullcheck/protocol.py:218; code/experiments/6_improvements/perf.py:177-195; code/experiments/4_defence_benchmark/bench.py:1015,1086; code/tests/test_plans.py:1120

**Evidence.** DECIDED by default, never revisited:
- commit_graph(policy='paper') (protocol.py:218), prune_last=False (transformer.py:497), interactive challenges.
- perf.py: --policy paper, --wire off, --prune-last off, --lookups off, --recipe legacy (perf.py:177-195).
- bench.py and serve_ab.py: --policy paper (bench.py:1015, serve_ab.py:867).
- v1_bytes.py, gen_bench.py and setup_timing.py: --policy auto (v1_bytes.py:307, gen_bench.py:174, setup_timing.py:52).
- strong_gpu.sh POL defaults to auto (:105), while sp2027.sh (:66) and sp2027_numbers.py (:1083) default to cnn18c.
- A test pins the course default: test_the_default_policy_is_the_reports_commitment (test_plans.py:1120).
- The library reads 24 PVI_* environment variables (lib_fullcheck.md §7). reexec.py switches a kernel path by patching native_kernels._GEMM (reexec.py:229-234). sp2027.sh takes only the tier as an argument; everything else comes from the environment.

THE AUTHORS' RULE. Every parameter becomes a CLI argument, and superseded variants become non-default options.

RISK.
- Running a script without the package's exact flags reproduces neither the paper nor the package. For example, `strong_gpu.sh must` run directly writes _polauto cells that the pipeline ignores (inventory bug B4).
- bench.py derives cell names from flags (bench.py:1086,1103), and paper_assets.OPTIMISED looks up stored base cells by those names. Flipping defaults naively breaks the lookup.

**Recommendation.** CHANGE NOW, before the 26 Oct feature freeze, so the 28-29 Oct rehearsal runs the submission code.
- Add a named 'paper' preset as the CLI default: cnn18c, wire, prune-last, lookups, fs, certified recipe. Keep 'report' as a named, non-default preset.
- Use BooleanOptionalAction for the flags.
- Keep bench.py's tag derivation tied to the effective settings, so stored names do not change.
- Replace the kernel and verifier environment switches with one options object, set by flags and recorded in each record.
- Turn sp2027.sh's environment switches into flags.
- Verify the change: a fresh dry run must be identical to F6_dryrun.txt, and the golden outputs must be unchanged. Update test_plans.py:1120.

If this cannot be done by 26 Oct, CHANGE LATER and document the switches instead. Changing defaults after the freeze would make the branch differ from the run code.

Cost: 1.5-2 d. (Effort: 1.5-2 d, before 26 Oct, or deferred and documented.)

**Verifier.** Every cited default checks out. commit_graph policy='paper' (protocol.py:218). build_decoder prune_last=False (transformer.py:496). perf.py: --policy paper, --wire/--prune-last/--lookups off, --recipe legacy (:177-195). bench.py --policy paper (:1015), serve_ab.py paper (:867); v1_bytes.py:307, gen_bench.py:174 and setup_timing.py:52 use auto; strong_gpu.sh:105 POL auto. test_plans.py:1120 pins the report default. reexec.py:229-234 patches native_kernels._GEMM. bench.py builds tags from flags (_prune/_lookups/_pol<policy>, :1092-1104), so naively flipping defaults would rename cells. The freeze-time risk the reviewer flags is real; the fallback 'defer and document' is sensible.

### HD-09 [medium, confirmed] # Commitment plan cnn18c with a prover-side setup cache: sound, but the paper still mixes plans

**Where:** code/src/pvi/fullcheck/setup_cache.py:63-85; code/experiments/4_defence_benchmark/slurm/sp2027.sh:28-29,66; paper_sp2027/sections/ext_defence.tex:121-122,150-166

**Evidence.** DECIDED (PS2 option (c), SP2027_PLAN2.md:740-741). The cache stores each model's Merkle leaves per plan, so every timed mode-C cell runs under cnn18c and pays each model's setup once (Llama-2-7B: setup 4,426 s, cache hit 49 s, same roots and bytes). The decision cut Fiat-Shamir proofs at 64 tokens by about 29% (model, :801).

STILL HOLDS? Yes.
- The cache key covers the plan, the layout and a BLAKE2b of every op's int8 weights and int64 bias (setup_cache.py:63-85), so real and random weights cannot collide.
- A file is spot-checked when read, and a wrong leaf can only make honest proofs fail (:24-29).
- setup_timing passes cache=False.

RISKS (presentation):
- tab:eval-v1 rows use cnn18c, cnn16c and cnn17c without saying so (C2:124-135).
- tab:eval-llm's base columns are plan auto.
- The 325 MB and 124.7 MB in the text are from the cnn16c era (ext_discussion.tex:129-131).
- The setup-proof text is still on the auto plan: 11.3 GB and +8% (ext_defence.tex:121-122) against 9.8 GB and +21% in tab:setup; the 2^-35 / t=35 at appendix_defence.tex:175; 1.6% at ext_defence.tex:176 and ext_intro.tex:152-153.
- params_table.py (:64) and setup_bytes.py (:27-29) default to --policy auto, so their own docstring commands do not reproduce the paper's tables.

**Recommendation.** KEEP cnn18c and the cache.
- CHANGE NOW: make cnn18c the default of every script and of strong_gpu.sh.
- Regenerate the stale setup-proof sentences from setup_bytes.py and params_table.py with --policy cnn18c.
- After the run, take every mode-C byte in the text from the cnn18c cells, and name the plan in each caption.
- Exclude .setup_cache from the returned tarball (L40S_RUN:131).

Cost: about 0.5 d. (Effort: 0.5 d (defaults before 26 Oct; text any time before 9 Nov).)

**Verifier.** The cache key covers the plan and its groups and matrices, plus a BLAKE2b of each op's int8 weight and int64 bias (setup_cache.py:63-85). Leaves are spot-checked when read (:23-29), and setup_timing passes cache=False (:79-80). Stale text confirmed: ext_defence.tex:121-122 gives 11.3 GB and 8%, and :147-150 gives 2,833.5 / 11,272.7 MB, against tab:setup's 2.4 / 9.8 GB (:136). appendix_defence.tex:175 has 2^-35 at t=35, against defence.tex:209's 2^-23. ext_defence.tex:176 and ext_intro.tex:152-153 say 1.6%, against defence.tex:221's 2.2%. ext_discussion.tex:129-131 gives 325 / 124.7 MB. params_table.py:64 and setup_bytes.py:27 default to auto. The tab:eval-v1 plans per row are cnn18c/16c/17c (C2:124-135). Minor: the -29% at SP2027_PLAN2.md:801 is the best plan's modelled gain; cnn18c gets about 70% of it.

### HD-10 [medium, confirmed] # Last-block pruning: the verifier's graph is pruned, but the fp32/bf16 re-execution baselines behind the headline are not

**Where:** code/experiments/7_reexec/reexec.py:226-227,244; code/experiments/7_reexec/serve_ab.py:441

**Evidence.** DECIDED (F3a; proved in ext_proofs.tex:102-108,214-228; tests in test_pruning.py). The last block computes only the last position. Every package cell passes --prune-last (sp2027.sh:132-133). The paper states it for partial builds (evaluation.tex:61-62).

WHY. For a next-token query it is exact and saves about one block's work at long prompts.

STILL HOLDS? Yes for the protocol. But the baselines are asymmetric:
- reexec.py builds the int baseline with prune_last=True (:226-227) but builds FloatDecoder for float and bf16 without it (:244), although FloatDecoder supports pruning (:97-110,144).
- The A7 serving baseline is pruned (serve_ab.py:441). A7 kept reexec.py unpruned on purpose to keep the old rows comparable (A7_serving_overhead.md:163).
- The 'fastest exact re-execution' at 2,048 tokens (fp32, 93.2 s), which the abstract and tab:options use, therefore does a full last block. My estimate of the effect: about 1/L of the work, i.e. about 3% for Llama-2-7B and about 8% for 12-layer models.
- The float decoder also skips RoPE (reexec.py:20-22), which favours the baseline. Neither asymmetry is stated in the paper.
- The library default is prune_last=False (transformer.py:497), and the course base stand-ins are unpruned (paper_assets.py:419-421).

**Recommendation.** CHANGE NOW.
- Pass prune_last=True to FloatDecoder for float and bf16 in reexec.py. dynint8 cannot be pruned; label it.
- Add a test that the pruned and unpruned float logits agree. The A7 test already does this, so reuse it.
- Re-measure the Xeon baseline before the freeze, and state in tab:options that every baseline computes only what the pruned graph computes, and that the float baseline omits RoPE.

Cost: about 0.25 d plus about 3 Xeon job-hours. The ratio moves down by a few percent. (Effort: 0.25 d + ~3 Xeon job-hours, before 26 Oct.)

**Verifier.** reexec.py:226-227 builds the int graph with prune_last=True. :244 builds FloatDecoder for float and bf16 without it, although the class supports pruning (:105-110,144). serve_ab.py:441 prunes U2, and A7_serving_overhead.md:163 says reexec.py was left unchanged on purpose. The float decoder already computes the LM head only at the last position, so the asymmetry is the last block's q, attention, o and MLP over T-1 positions: about 1/L of the work, roughly 3% on Llama-2-7B. The RoPE omission favours the baseline (reexec.py:20-22). The paper discloses neither; the only pruning statement is evaluation.tex:61. The effect is small but it sits on the headline ratio.

### HD-12 [medium, confirmed] Quality recipes and alphas were selected on the WikiText-2 test split, the same split they are reported on

**Where:** docs/improvements/F2a_quality_2048.md:172-199; SP2027_PLAN2.md:489; code/experiments/4_defence_benchmark/slurm/sp2027.sh:391-396

**Evidence.** DECIDED.
- OPT: the certified recipe came from a search over recipes and alpha 0.3-0.7, scored on 'All 140 windows' of the test split (F2a_quality_2048.md:174-199). The tab:quality ratios (x1.076 / 1.026 / 1.008) are the winning test-split numbers.
- Llama: the main recipe was chosen on a test-split grid for OpenLLaMA-7B (SP2027_PLAN2.md:489: 0.55-0.65 by glu 0.2-0.3, x1.073 selected).
- The final quality tier again sweeps alpha 0.55 and 0.65 on Llama-2-7B's test split (sp2027.sh:393-394).
- The WikiText-2 validation split was never used. Calibration did move to the train split (F2a).

STILL HOLDS? The selection effect is probably small: the grids are flat (0.5-0.7 within 0.003 on OPT-1.3B; 'flat over 0.55-0.65' on OpenLLaMA). But a reviewer can call the reported ratio tuned on test.

ALTERNATIVES.
- Re-score the candidates on the validation split and show the same choice.
- Choose on validation and report test.
- Disclose the selection.

**Recommendation.** CHANGE NOW, with evidence only and no graph change.
- Before 26 Oct, re-score the OPT candidates (alpha 0.5/0.6/0.7 at 11 bits, OPT-125M and OPT-1.3B; OPT-6.7B at 0.5/0.6) and the OpenLLaMA grid on the validation split. This takes minutes per variant on the 2080 Ti, about 4-8 GPU-hours in all.
- If the choice agrees, say so in quality.tex. If it does not, keep the frozen recipe, because changing it changes digests and every real cell, and disclose that it was selected on test.
- In the Llama-2 tier, report MAIN_RECIPE's alpha 0.55 as the result and 0.65 as sensitivity only, which the comment at sp2027.sh:375-376 already intends.

Cost: about 0.5 d. (Effort: 0.5 d + 4-8 GPU-hours (hardware-independent; frozen builders).)

**Verifier.** F2a:174-199 scores the recipe search and the alpha grid on 'All 140 windows' of the test split, and the chosen alpha 0.6 gives the reported tab:quality ratios. OpenLLaMA's 0.55/0.2 was chosen on the test-split grid (SP2027_PLAN2.md:489), and sp2027.sh:393-394 sweeps 0.55/0.65 on test again. No validation-split scoring appears in the docs. The grids are flat, so the bias is small, but a reviewer can call it tuning on test. Re-scoring on validation, or disclosing the selection, is the right low-cost fix.

### HD-14 [medium, confirmed] # OpenLLaMA-7B as the stand-in for Llama-2-7B, while the package also runs real Llama-2-7B/13B under the friend's licence

**Where:** code/experiments/4_defence_benchmark/slurm/sp2027.sh:335-352,387-396; paper_sp2027/sections/evaluation.tex:59-60; paper_sp2027/sections/quality.tex:84-86

**Evidence.** DECIDED (SP2027_PLAN2.md:488-489). OpenLLaMA-7B v2 is Apache 2.0, ungated, and has exactly Llama-2-7B's shapes (empty structure diff). It gives x1.073, and F3b runs it end to end (evaluation.tex:59-60; quality.tex:84-86).

WHY. We had no Llama-2 licence on our cluster.

LATER. The friend holds the licence (SP2027_PLAN2.md:488). The package now adds:
- real Llama-2-7B v0, V1 and generation cells (sp2027.sh:335-352, about 22 job-hours of limits);
- Llama-2-7B and 13B quality jobs (:387-396, 14 h).

STILL HOLDS? As the paper's claim, yes. But nothing says what happens to the text when the Llama-2 results arrive:
- Those cells have never run on the real checkpoint; the rehearsals use OPT-125M and a tiny Qwen3 (sp2027.sh:314-318,382-385).
- TinyLlama needed alpha 0.75 where OpenLLaMA needed 0.55 (SP2027_PLAN2.md:487-489), so Llama-2-7B's ratio at 0.55 is not guaranteed.
- Pipeline bugs touch these cells: B2 (sp-r-llama7-gen overwrites the Gen macros) and B6 (no random twin for the real V1 cell, sp2027.sh:345).

RISK. A real Llama-2-7B result much worse than x1.073 would arrive in the run week and contradict a quality claim that the cost tables rely on. Omitting a result we ran is not an option.

**Recommendation.** CHANGE NOW (a decision only).
- Pre-write the text for both outcomes and commit to reporting Llama-2 as 'the model of the cost tables', with OpenLLaMA as the licence-free reproduction.
- Alternatively, if the team will not change the text in the run week, drop the real-Llama and quality tiers (about 36 L40S job-hours), which also lowers the licence and anonymity surface.
- If they stay, fix B2 and B6 after the freeze, and add r-llama7rnd-v1 cells before the freeze.

Cost: about 0.25 d plus 0.25 d of pipeline fixes. (Effort: 0.25 d decision before 26 Oct; 0.25 d pipeline after 1 Nov.)

**Verifier.** sp2027.sh:335-352 adds the real Llama-2-7B cells. Their job limits sum to 22 h: 3+1+3 per length, plus 3+1 and 4. :387-396 adds the 6 h and 8 h quality jobs. Rehearsals use OPT-125M (:314-318) and a tiny Qwen3 vocabulary model (:382-385). B6 confirmed: r-llama7-v1-* has no rnd twin, so tab_real's Llama V1 random cells stay missing (:702-725). The paper's Llama numbers are OpenLLaMA-7B (evaluation.tex:59-60, quality.tex:84-86), and no text plan exists for the Llama-2 outcome. Nuance: the code path (llama builder, v0/V1/gen end to end) already ran on OpenLLaMA-7B with identical shapes (F3b). The untested risk is the checkpoint's quality, not the code.

### HD-15 [medium, partly] # V1 (cut + logUp-GKR) as a non-default variant: scope holds, but its claims and parameters are partly unsupported in the repository

**Where:** paper_sp2027/sections/evaluation.tex:159-176; code/experiments/6_improvements/params_table.py:90; paper_sp2027/sections/appendix_v1.tex:285-288

**Evidence.** DECIDED (Plan 1 C1-C2, SP2027_PLAN.md:46-47; Plan 2 V-V1N and drops, SP2027_PLAN2.md:170,606-611).
- V1 sends the int8 requantised values and proves the windows with logUp-GKR over F_{p^8}. It works in modes C and Kpre; mode K was dropped because Kpre dominates it.
- It is reported as a variant: about 2x smaller proofs, a prover about 10x slower, a CPU verifier 2.3x v0's (45.4 s against 19.6 s).

STILL HOLDS? As a variant, yes: the abstract presents it as such (main.tex:73), and the bytes-only fallback was not needed. But:
- 'V1's own additions stay below 1 MB per proof' (evaluation.tex:160-161) is false on full models: u alone is 32.1 MB on OPT-6.7B (F3a:301) and 40.8 MB on OpenLLaMA-7B (F3b:65).
- The largest-model security terms (2^-215.3, 79 instances, 4.3e9 leaves; appendix_v1.tex:285-288, ext_v1.tex:66-67,211-215) come from V1_SPEC.md, which is outside the repository. Code docstrings cite it 16 times in code/src.
- tab:params' V1 rows extrapolate from cell counts hard-coded at params_table.py:90, with no record of where they were measured.
- The package measures Llama-2-13B at 2,048 tokens with only 2 blocks (sp2027.sh:213), so the extended version's V1 todo for 13B (ext_evaluation.tex:286) cannot be filled.

**Recommendation.** KEEP V1 as the non-default variant, and CHANGE NOW for the claims:
- Fix the '<1 MB' sentence: it holds for partial builds only, and V1 sends u in Kpre.
- Import V1_SPEC.md into the repository (docs/PROTOCOL.md or an extended appendix) and repoint the 16 references.
- Make params_table.py compute the V1 counts from the model's CutPlan, or read them from a stored record, for 7B and 13B.
- Decide on the 13B V1 row: add a full-model cell if it fits 48 GB, or drop the row.

Cost: 0.5-1 d. (Effort: 0.5-1 d (the 13B cell decision before 26 Oct; docs any time).)

**Verifier.** Confirmed, and stronger than stated: 'V1's own additions stay below 1 MB' (evaluation.tex:160-161) is false even inside its own table. C2:135 has OPT-1.3B 24/24 Kpre at V1 910.6 MB against claims 896.7, which leaves about 14 MB of V1 parts (u). F3a:301 gives u 32.1 MB plus cut 2.5 MB; F3b:65 gives u 40.8 MB. V1_SPEC.md is not tracked (git ls-files finds nothing). It is cited 9 times in code/src .py files and 16 times across code/, not 16 in code/src. sp2027.sh:213 measures Llama-2-13B at 2,048 tokens on 2 blocks only, so the 13B V1 cell (ext_evaluation.tex:286) cannot be filled by measurement. Refuted sub-point: params_table.py:90's hard-coded counts name job 1007816, and that record is stored at code/artifacts/results/vlonga_v1retime_w1c_1007816.jsonl (it contains 103809024). --v1-cells can override them. Only the per-block extrapolation remains a concern.

### HD-11 [low, partly] 11-bit attention probabilities only in the certified models; every cost table measures 8-bit graphs

**Where:** code/src/pvi/fullcheck/transformer.py:55-64; paper_sp2027/sections/quality.tex:53-56; paper_sp2027/sections/appendix_defence.tex:73-74

**Evidence.** DECIDED (SP2027_PLAN2.md:365; transformer.py:59-64). CERTIFIED_PROB_MAX = 2047 because 8 bits cost 2-23% of quality at 2,048 tokens. The random-weight benchmark graphs keep PROB_MAX = 255 'so the stored benchmark records stay valid' (transformer.py:63-64).

STILL HOLDS? The choice is right for quality, and the native CPU and Triton kernels support pmax up to 2047 exactly (native_kernels.py:119,975; attention_kernels.py:15). But:
- No benchmark script exposes prob_max. A grep of bench.py, perf.py, v1_bytes.py, gen_bench.py, serve_ab.py, reexec.py and setup_timing.py finds no option. So every cost cell, the abstract ratio included, is for a model whose quality the paper says is 2-23% worse.
- The body asserts that 11 bits 'change neither the proof's size nor its cost' (quality.tex:55-56). No record supports it; transformer.py:63 asserts it without a measurement.
- Compressed claim bytes depend on activation statistics, which the probability scale changes.
- The extended appendix discloses the 255/2047 split (appendix_defence.tex:73-74); the body does not.

ALTERNATIVES.
- (a) Add --prob-max to the benchmark scripts and run the final cost cells at 2047. This changes the goldens and rehearsals two weeks before the freeze.
- (b) Keep 255 and measure a 2-block A/B (bytes, prover, CPU verifier at 255 against 2047) on the 2080 Ti or Xeon to back the claim.

**Recommendation.** CHANGE NOW, with the minimal option (b).
- Run the A/B now (about 1-2 job-hours) and cite it.
- Add a --prob-max option, defaulting to 255 for the random-weight cost graphs, so the setting is a recorded parameter.
- Switch the final run to 2047, option (a), only if the A/B shows more than 1% difference in bytes or time.

Cost: about 0.25 d. (Effort: 0.25 d + 1-2 job-hours before 26 Oct.)

**Verifier.** Correct that no benchmark script exposes prob_max (grep of bench/perf/v1_bytes/gen_bench/serve_ab/reexec/setup_timing finds none). Random graphs use PROB_MAX=255 (transformer.py:55,63-64). quality.tex:55-56 asserts no size or cost change, and so does F2a:157, without a cited A/B. Overstated in two ways. (1) The F3a real-weight cells already ran at 2047, because build_opt_from_hf defaults to CERTIFIED_PROB_MAX (real_weights.py:185) and v1_bytes.real_graph passes no prob_max (:118-120). Their times are within 9% and 6% of the 255 random twins, which is confounded but is some evidence. (2) The attention is recomputed by the verifier with the same kernel path for pmax<=2047 (native_kernels.py:119), and claims are weight-op outputs, so cost should be unchanged by construction. Random-weight cost graphs have no quality meaning. A quick A/B is cheap and still worth doing.

### HD-13 [low, partly] Main Llama recipe (fixed public prefix, glu_smooth) and the int8-change rule: sound, but applied unevenly

**Where:** code/src/pvi/fullcheck/real_llama.py:458-464; SP2027_PLAN2.md:447,478-481; code/experiments/4_defence_benchmark/slurm/sp2027.sh:396

**Evidence.** DECIDED.
- The team's rule (SP2027_PLAN2.md:447): an int8 change is either part of the main algorithm or a cited common technique. Otherwise it is only a possible improvement and never appears in the comparison against other systems.
- MAIN_RECIPE = smoothing 0.55 with percentile statistics, glu_smooth 0.2, the PrefixQuant-style fixed prefix and qk_balance, all constants with the benchmark structure (real_llama.py:458-464). The per-token exponent glu_exp stays outside it.

STILL HOLDS? Yes. The rule is respected in the text (ext_discussion.tex:66-74).

GAPS.
1. The Llama-2-13B quality job uses --alphas 0.6 (sp2027.sh:396), not MAIN_RECIPE's 0.55.
2. F2b-Q's kill rule says to label the Qwen rows 'shape only' (SP2027_PLAN2.md:478-481,1174). Qwen3-4B stays at x5.3 (quality.tex:87-88), yet it appears unlabelled in tab:eval-llm (evaluation.tex:91) and in the Maverick comparison (:282). A grep finds no 'shape only' anywhere in sections/. The package also spends L40S hours on Qwen3-4B gen and serve cells.
3. Llama-family reproducibility of C_M was checked only within one machine (D6a:101). The main recipe uses percentile statistics, which F2a found unstable for OPT. The paper's reproducibility sentence names only OPT (defence.tex:51-53), which is correct, but 'anyone can recompute C_M' is unverified for Llama.

**Recommendation.** KEEP the recipe and the rule, and CHANGE NOW:
- Set Llama-2-13B to alpha 0.55, or say why 13B differs.
- Apply the F2b-Q kill rule on 23 Oct: label the Qwen3-4B cost rows 'shape only (integer quality x5.3)'. Consider dropping sp-gen-qwen3-4b and sp-serve-qwen4 if no text cites them (saves about 5 L40S hours).
- Optionally run digest_check on TinyLlama across two nodes (about 1 h) to support the open-weight claim for Llama.

Cost: about 0.25 d. (Effort: 0.25 d before 26 Oct.)

**Verifier.** Confirmed: MAIN_RECIPE is alpha 0.55 with percentile statistics and glu 0.2 (real_llama.py:458), while the 13B job uses --alphas 0.6 (sp2027.sh:396) with no stated reason. D6a:101 checks the Llama builder only within one machine, and defence.tex:51-53 names only OPT. Qwen3-4B stays at x5.3 (quality.tex:87-88) and appears unlabelled in tab:eval-llm (evaluation.tex:91) and in the Maverick row (:282); 'shape only' appears nowhere in sections/. However, the F2b-Q kill rule only triggers on Fri 23 Oct (SP2027_PLAN2.md:478-481,1174), so nothing is violated yet. The int8 rule itself is respected. This is a scheduled decision, not a defect.

### HD-16 [low, not verified (low)] zkLLM comparison through stored L40S runs instead of a re-run

**Where:** code/artifacts/results/zkllm_l40s/README.md:1-10; paper_sp2027/sections/evaluation.tex:310-312

**Evidence.** DECIDED (F5, SP2027_PLAN2.md:1055). zkLLM's code (commit 993311e) has not changed, so the runs stored in Sept 2026 on TAU's L40S with an EPYC 9334 host (844 s for Llama-2-7B at 2,048 tokens) are compared with the final run's prover on the same GPU model; no re-run.

WHY. It saves the friend's time, and zkLLM is GPU-bound.

STILL HOLDS? Yes, with caveats the paper does not yet state:
- Only zkLLM's prover was timed.
- The whole-model time is extrapolated from layers 0-1 multiplied by L (README:24-30), while ours is 'no extrapolation'.
- The demo's first skip connection never runs, so the measured one is counted twice (SP2027_PLAN2.md:1055). That slightly inflates zkLLM's time.
- The host differs from the final machine.
- The README still says 'not used in the final paper' (README:1-10), contradicting evaluation.tex:310-312. It also names TAU's 'killable' partition (README:4), and env.txt in the stored runs holds host details, which matters for anonymity.

**Recommendation.** KEEP.
- CHANGE LATER (before 9 Nov): update the README.
- State the caveats in one clause next to the 844 s: prover only, per-layer extrapolation, double-counted skip, different host.
- Scrub env.txt and the TAU strings at export.

Cost: about 0.25 d. (Effort: 0.25 d.)

### HD-17 [low, not verified (low)] Modes C, K and Kpre

**Where:** paper_sp2027/sections/defence.tex:48-50,137-161

**Evidence.** DECIDED (Plan 1 and the paper).
- C: the client holds only C_M, and the column check uses the linear-code commitment.
- K: the client holds the weights and draws a fresh chi per query.
- Kpre: secret chi and u = chi^T A, precomputed once per model.
- The soundness terms are eps_C, eps_K and Q*eps_K under key reuse (defence.tex:156-161).

WHY. Different clients need different things. Kpre is the CPU headline, and its one-time precompute is 12.7 s on 8 Xeon threads (A6, SP2027_PLAN2.md:729).

STILL HOLDS? Yes.
- Mode K is proved but no table times it: the evaluation reports only C and Kpre, and the bench CNN k- cells feed only the outcome counts (package.md §4).
- V1 refuses K (SP2027_PLAN2.md:170).
- Kpre's verdict leakage is covered by the Q*eps_K bound.

**Recommendation.** KEEP all three modes.
- State once in the evaluation that K is not timed, since it costs Kpre's check plus the per-query fold.
- Optionally drop the CNN k- bench cells if the counts do not need them.

No code change before the freeze. (Effort: 0.1 d.)

### HD-18 [low, not verified (low)] Freivalds plus Reed-Solomon/Merkle (linear-code PCS evaluation), not zero-knowledge, scoped to open weights

**Where:** paper_sp2027/sections/discussion.tex:37-39; paper_sp2027/sections/evaluation.tex:302-305

**Evidence.** DECIDED (Plan 1 H2, SP2027_PLAN.md:98). The paper is positioned as a Freivalds-based check with a linear-code commitment, 'using only a hash function' (main.tex:73), and as a different trade-off from zkLLM, not as a worse zero-knowledge system.

WHY.
- No trusted setup, enclave or stored encoded weights are needed.
- The prover is 46x to about 32,000x faster than zkSNARK provers (evaluation.tex:307-308).

STILL HOLDS? Yes. The trade-offs are stated:
- not zero-knowledge, and about ceil(k/T) queries reveal a weight matrix (discussion.tex:37-39), hence open weights only;
- proofs grow with T (6.6 GB at 2,048 tokens);
- tab:eval-compare's caveats on hardware and guarantees (evaluation.tex:302-305).

The untrusted-commitment gap is closed by the setup proof (D4).

**Recommendation.** KEEP. No change.

Check once that the Related Work and the comparison never call our proofs succinct or private. (Effort: none.)

### HD-19 [low, not verified (low)] Exact int8 integer model, with public integer semantics, as the model that is served

**Where:** paper_sp2027/sections/defence.tex:69-77; paper_sp2027/sections/evaluation.tex:111-114

**Evidence.** DECIDED (core design).
- Int8 weights and activations, int32 tensor-core sums, 23-bit residuals, integer norms, 256-entry tables and attention in float32 within its exact range (defence.tex:69-77).
- field.int8_ok checks the hardware path at run time and falls back to an exact path if the check fails (appendix_defence.tex:78-86).

WHY. Prover and verifier agree bit for bit on any hardware, so every product can be checked exactly.

STILL HOLDS? Yes. The costs are reported honestly:
- The provider must serve the integer model: prover overhead 4.1-8.6x over the exact integer forward pass and 19-40x over fp16 (evaluation.tex:111-114).
- Static per-tensor scales fail on Qwen3-4B (x5.3, quality.tex:87-88).

The one residual risk is platform first contact: the frozen kernels first meet sm_89 and Zen 4 on the friend's machine (R15, SP2027_PLAN2.md:1204). The smoke tier covers this with test_gpu_exactness and int8_ok, but only if the friend's 2-hour smoke runs between 19 and 27 Oct.

**Recommendation.** KEEP.

Confirm the friend's smoke date now. Without it, the hotfix window of 2-3 Nov is the only protection.

No code change. (Effort: none (scheduling only).)
