# Adversarial check of th_a (theory workflow wf_a7c335dc-948)

The block-mixing bound itself, Proposition prop:block(ii), holds: I found no error in its quantifiers or probability bounds. One argument is wrong (the last sentence of rem:block-scope(iv)). Several other statements need a stated fix, mainly a bound on t and a mismatch between integer and real-valued networks in cor:block(b).

**Prop prop:block (Part C)**

(i) Correct. md = d⌈N/d⌉ ≤ N+d−1. Folded rows, columns and per-block multiproofs depend only on block b, plus salts for the paths.
- Being in Spot_d also requires accepting every admissible instance. So the admissible set (def:spot's side information) must be limited to integer networks that meet the range bound. Say so once.
- Small wording fix: folded rows do not depend on salts, only the paths do.
- Under def:spot as written, V_blk is also in Spot_d(2m): one bundled message of r folds per block and one t-column multiproof per block. See cor:block(b) below.

(ii) Correct. I checked each step:
- **Freivalds case.** The event {accept, W=∅} is contained in {χᵀD=0}, which has probability at most p^{-r}.
- **Choice of W and G.** W and G depend only on (χ, the step-2 messages), so they are fixed before C. C is uniform given them.
- **Column bound.** C(|G|,t)/C(n,t) ≤ ∏(k−1−i)/(n−i). For t ≥ k the product contains a zero factor, so it is 0, and the comparison with ((k−1)/n)^t still holds.
- **One block suffices.** Picking the least block of W gives a bound that does not depend on m.
- **Per-tree positions.** The claim "holds unchanged if each tree draws its own positions" is true. The code does draw per tree: protocol.py `Challenger.columns`, key `"cols/"+name`.

One cosmetic fix: the product form is Theorem thm:sound's. Prop prop:amp states p^{-r}+((k−1)/n)^t. Write "at most the bound of prop:amp".

(iii) Fixable. The formula t=⌈β/log₂(n/(k−1))⌉ can give t > n, and then "t distinct positions" do not exist.
- Concrete case: the integer hard instance's logit layer has k=2 and n=4·next_pow2(2)=8. With L=3, β=128+log₂6≈130.6, so t=⌈130.6/3⌉=44, which is more than 8.
- Fix: t = min(⌈β/log₂(n/(k−1))⌉, k). For t ≥ k the column term is exactly 0, and r+t ≤ (1+1/log₂p)β+2 still holds.
- The code caps t at n (`t = min(t, n_points)`), which is equivalent.
- The same gap exists in prop:amp and thm:dich(b).

**Cor cor:block**

(a) Fixable.
1. Apply the t cap above. The corollary runs V_blk on the k=2 layer.
2. "Arithmetic embeds without wrap-around" covers honest values only. V_blk's steps (1)–(3) contain no range check. If claims are int32, a claim z+p passes and changes the recomputed requantisation. Fix: add "the verifier rejects claims with |z| ≥ 2^29 (Section sec:intmodel)", or require claims to be field elements lifted to (−p/2, p/2).
3. Embedding lookups are not of the form Z=A[X;1]. Either state the corollary for networks without lookups or treat them separately.
4. The union bound is correct but loose. Open point 3 is right that the maximum suffices in interactive mode C.

(b) Fixable.
1. **Wrong network class.** The lower bound is quoted for "every ReLU network". V_blk's guarantee holds only for integer networks under the range bound. Restate (b) over the integer hard family, and state the integer version of prop:worst as its own lemma.
   - In that lemma, say "int8 weights, integer biases": g's bias −(N−1) is not a 7-bit value for N > 128.
   - State the requantisation multiplier 2^30 (identity on integers) and the ReLU table.
   - State the range bounds: 128(1+127c) < 2^29 for the hidden rows, and 129N < 2^29 for g, which holds for N up to about 4.16·10^6.
   - With these, I found no error in the conversion: the class flips the same way, values stay in range, and the explanation is unchanged.
2. The lower bound is exact only in the ideal-oracle model. rem:merkle gives an error term only for unsalted trees, so for salted leaves either add a salt-guessing term or say "ideal-oracle model".
3. The ratio arithmetic is correct: N+d−1 < 2N.
4. The factor O(λ+log L) is an artefact of the counting convention, not a property of the problem:
   - Opening whole blocks reaches a ratio below (N+d−1)/N.
   - With bundled counting, V_blk is in Spot_d(2m), so the ratio is below 4/(1−2^{-λ}).
   - The statement is true, but a reviewer can tighten it at once. Either fix the counting convention in def:spot (open point 1) or say this explicitly.

**rem:block-count**
- (i) Correct.
- (ii) Correct. I checked the K_s variant: D_ij is invertible in K_s, the code over K_s at the same points has the same distance, and F-linearity carries over.
- (iii) Correct as a necessary condition. Add three things:
  - a second necessary condition, k > φt/ω (about 4t ≈ 280 at t ≈ 70);
  - r=7 under Fiat–Shamir gives about 28 rows, not 20;
  - the m·t salts in the salted variant.
- (iv) Correct. Add that the column queries are then redundant, so mr queries (or m with one K_s fold) already reach the error. This is a second reason the O(λ) factor is a counting artefact.

**rem:block-scope**
- (i) Correct. The doomed-state transitions check out.
- (ii) Correct, with two fixes:
  - β₀ grows by log₂(Σ_ℓ m_ℓ/|J|), which is at most log₂ max_ℓ m_ℓ, not exactly log₂ m;
  - write e_b per block and use max_b e_b.
- (iii) Correct.
- (iv) The first part is correct, and t_b=0 is in fact a probability-1 attack: claim Z=A′[X;1] with A′ differing from A only on block b, and send u_b=χ_bᵀA′_b.
- **(iv), last sentence: wrong as an argument.** "Sampling blocks does not help either: by thm:dich(a), error ε needs qd ≥ (1−ε)N" does not follow. Checking |S| blocks gives qd=|S|(r+t)d. So thm:dich(a) only forces |S| ≥ (1−ε)m/(r+t), while the actual error is at least 1−|S|/m.
  - Fix: "a block that receives no check can be lied about with probability 1. So checking a set S of blocks, drawn independently of the claims, leaves error at least 1−|S|/m against a lie in a uniformly random block."
- (v) Fine.

**Part D (submission appendix)**
- Prop: correct. Wording fix in the sketch: "the encoding of a wrong row of u_b and the same row of χ_bᵀÂ_b are distinct codewords" (a row of u_b is a message, not a codeword).
- Cor: fixable. It needs the t cap and the integer-network restriction from (b).

**Parts A and B**
- A: correct.
- B1: drops the log L term. Use O(λ+log L), or add "for L ≤ 2^λ".
- B1 and B2: "matched up to O(λ)" is true but undersells the result and invites the objection above. "Tight at every locality" (by opening) is also true.
- B3:
  - add n ≥ 2(k−1); the current thm:dich(b) has the same gap;
  - "each query mixes all rows of one block" is not a hypothesis that implies soundness, so say "the block verifier of Prop prop:block".
- B5: cite prop:block(i) for "touches every row", and write O(λ+log L).

**Header comments**
- "The code uses n = 4 next_pow2(k)": in fact n = rate·next_pow2(k), with rate a power of two ≥ 2 and default 4 (plans.py `_policy`). This still gives n ≥ 2(k−1).
- "The optimum is the same for every partition, as the proof predicts": the proof predicts only the bound.
  - With shared positions, equality does hold: a lie spread over several blocks passes only on ∩ zeros(Enc δ_b), which lies inside zeros(Enc Σδ_b).
  - With per-tree positions, which is what the code does, the optimum can be higher than for one block. My exhaustive run (p=5, n=4, k=2, t=1, r=1, N=2, T=2) gave 0.25 against 0.20, both under the bound 0.45.
  - The draft's toy computation was not saved, so its scripts should be kept.
- I reproduced the draft's toy optima by hand: 4/13 = 1/13 + (12/13)(1/4) and 0.0924 = 1/17 + (16/17)/28.

**What I checked and what I did not**
- **Read:** the draft, attack.tex, appendix_attack.tex, appendix_short.tex, defence.tex, intro.tex, ext_attack.tex, SP2027_PLAN2.md §3.7, and REVIEW.md B-W5/B-Q4.
- **Code checked:** χ is uniform by rejection sampling; columns are distinct, uniform, per tree and capped at n; column leaves are unsalted; the byte sizes are 4-byte and 31-bit field elements.
- **No web search:** the draft cites no external theorem. rem:block-scope(ii) relies on thm:untrusted's BCIKS20 citation, which I did not check.
- **Files:** I wrote no repository file. The exhaustive check ran from a temporary script outside the repo, which I then deleted.