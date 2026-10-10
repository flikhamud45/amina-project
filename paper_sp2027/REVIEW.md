# Mock review: "Certified but Compromised: Breaking and Fixing Lightweight Proofs of Inference"

Draft reviewed: `paper_sp2027/main.tex` and `sections/*.tex` as of 2026-10-10, and `main.pdf` built from them.
Three reviewer lenses: (A) systems security, (B) cryptography and proof systems, (C) ML systems. Scores use
1 = strong reject, 2 = weak reject, 3 = borderline (major revision), 4 = weak accept, 5 = strong accept, and
judge the draft as it stands. Each reviewer also says what would move the score.

State of the draft, as a reviewer would see it:

- The PDF has 29 pages. The body runs from page 1 to page 20 (the Conclusion starts on page 20), the references take
  pages 20 to 22 and the appendices pages 22 to 29. The plan's target is 13 pages of body plus 5 pages of references
  and appendices. The build log shows `Font shape TU/ptm/m/n undefined ... defaults substituted`, so the PDF is set in
  Latin Modern rather than Times. Times will save some space, but not the 6 to 7 body pages and roughly 11 total pages
  that must go.
- The source has 73 `\todo` markers: 43 in the evaluation, 8 in the introduction, 6 each in the attack and the
  discussion, 5 in generation and quality, 4 in V1 and 1 in the defence. Every "new" column of Table 3
  (`tab:eval-llm`) is empty, and so are the 7B and 13B rows of the comparison.
- Numbers come from three machine configurations: the base implementation on an L40S with an EPYC 9334, and each
  improvement and V1 on an RTX 2080 Ti with a Xeon Silver 4114. Several full-model figures are extrapolated from one-
  or two-block builds.

---

## Reviewer A: systems security

### Summary

The paper attacks lightweight proofs of inference that spot-check a committed execution trace. A provider overwrites
one neuron and recomputes the later layers honestly. This leaves a single inconsistent site, which the path protocol
of Anchuri et al. (SaTML 2026) catches with probability 1/N per path. The attack is shown on MNIST and CIFAR models,
on a dog-vs-cat ResNet-18 (which refutes a stated full-soundness claim for binary classifiers) and as a single-token
change on OPT-6.7B. A theorem says no adaptive or value-aware sampler beats uniform sampling in the worst case, and a
table applies the bound to nine published schemes. The defence checks every weight operation of an exact int8 model
with Freivalds' algorithm against a Reed-Solomon and Merkle commitment. It adds a setup proof for untrusted
commitments, Fiat-Shamir binding, generation and sampling, and a proof-size reduction (V1).

### Strengths

1. The paper has a clear break-and-fix arc and is unusually honest about its own limitations: Section 8 says outright
   that verification does not beat re-execution on a GPU and that the proof grows with the prompt.
2. The single-site edit followed by honest recomputation is the right adversary for this class of schemes. Reducing
   the forged trace to the honest trace of a model that differs in one row is a clean way to explain why sampling
   fails.
3. The systematisation (Table `tab:sok`) is careful. It separates "probabilistic guarantee met" from "claim beyond
   sampling" and avoids calling schemes broken when they meet their own stated guarantees.
4. The defence's threat model is strong: an arbitrary provider, an exact integer semantics that leaves no tolerance,
   and the Fiat-Shamir statement bound to every constant (the graph digest).
5. Measured negative results are reported, such as the fused encoder that turned out slower and the 9.2% / 0% split
   of the contribution-weighted sampler.

### Weaknesses

1. **No convincing deployment case.** Because the protocol is not zero-knowledge, the claims of about ceil(k/T)
   queries reveal each weight matrix (about six 2,048-token queries for Llama-2-7B's down projection), so it targets
   open-weight models. For those models a client could download the int8 weights once (6.7 GB). Instead, each
   2,048-token query brings a 6.6 GB proof (6.4 GB even in mode Kpre). On a CPU, verification ties fp32
   re-execution (85 s against 86 s), and on a GPU it loses (4.95 s base verifier on an L40S against 0.57 s fp16
   re-execution on a 2080 Ti). At 64 tokens a query costs 325 MB, of which 124.7 MB are opened columns that do not
   shrink with shorter prompts. The paper should state for which client, bandwidth, query volume and prompt length
   the protocol is the best option, and compare it quantitatively with the alternatives a practitioner would
   consider: downloading the weights and re-executing, GPU confidential computing (H100-class attested enclaves),
   CommitLLM-style auditing at 12-14% serving overhead, and zkLLM or zkAgent. GPU TEEs are not mentioned at all;
   Related Work discusses only SGX-era designs (Slalom, TwinShield, VeriAttn).
2. **The attack adds less than the framing suggests.** Anchuri et al. already note the 1/N limit, and SLP and
   CommitLLM describe the edit-then-recompute strategy themselves (the paper says so). Beyond the theorem, what is
   new is (i) the refutation of the binary-classifier remark, (ii) the observation about TensorCommitments' weight-only
   layer selection, and (iii) the measurements. The LLM attack writes values outside the neurons' natural ranges
   (Section 3.5: "Its values exceed the neurons' natural range"). A range check, which costs the path protocol
   almost nothing, would stop it. The range-respecting attack exists only on MNIST, with ranges calibrated on test
   queries that include the attacked ones (footnote, G3 todo). A reviewer will ask for the hardened protocol (range
   checks with held-out calibration plus the best value-aware sampler) attacked on a real LLM.
3. **The TensorCommitments verdict is fragile.** The table footnote says "no theorem claims it": the negligible
   soundness is a stated goal. Calling a goal "not met" is defensible, but the wording in the introduction ("Two
   claims exceed what the sampling gives") puts it on a par with the path protocol's explicit statement. Expect
   pushback from those authors and from reviewers.
4. **Root of trust for the certified model.** Soundness is relative to an integer model that someone builds:
   static scales, smoothed per-channel gains, tables, multipliers. For open weights the paper says anyone can
   recompute C_M from "the published int8 checkpoint". But a provider can publish a backdoored int8 model as "the
   quantisation of Llama-2-7B". The paper should require a deterministic, public quantisation recipe with fixed
   calibration data, so that C_M is derivable from the original fp16 checkpoint (plan item D6), and say so in the
   threat model.
5. **The untrusted-commitment setup proof has a weak use case.** For Llama-2-7B it is 2.8 GB (a = 100) or 11.3 GB
   (a = 1,000), larger than the int8 weights in the second case, and the paper notes that for GPT-2 it exceeds the
   weights. Since the weights leak anyway, a client that can check an 11.3 GB proof could recompute the commitment.
   Who is this for? Its time is not measured (todo).
6. **The evaluation is incomplete and mixes hardware.** See the state notes above. Some numbers disagree across
   tables: the Kpre CPU verifier for Llama-2-7B takes 698 ms at 64 tokens and 58.2 s at 2,048 in Table `tab:eval-llm`
   (EPYC), but 1.5 s and 85 s in Table `tab:eval-reexec` and the introduction (Xeon). Extrapolated numbers (GPU
   verifier "about 1.1 s" for OPT-1.3B) are set against measured competitor numbers (zkLLM 0.90 s).
7. **Disclosure is not yet done.** The notices are scheduled for 15-19 October, and the introduction and Section 3.6
   carry todos that must become facts or be removed.
8. **The paper is two papers compressed into one.** The body is about 50% over length, and much of the text repeats
   itself: attack results appear in Section 3.5 and again in Section 7.2, and V1 numbers appear in Section 5.7, in
   Section 7.4, in the introduction and in the discussion.

### Questions to the authors

1. For which concrete client (hardware, bandwidth, queries per day, prompt length) is your protocol cheaper than
   downloading the open weights once and re-executing exactly, and cheaper than a GPU TEE? Please give the numbers.
2. What is the prover's overhead over unverified serving of the same model on the same GPU, with batching? CommitLLM
   reports 12-14%.
3. Does the attack survive a path protocol hardened with range checks calibrated on held-out data, on OPT-6.7B or
   Qwen3-4B? How many neurons does an in-range next-token change need?
4. The full version of Anchuri et al. (arXiv 2603.19025 / ePrint 2026/541) adds a two-server refereed variant. Does
   your analysis cover it, and does your table use the SaTML version or the full version?
5. How is the certified integer model tied to the public fp16 checkpoint? Can a third party reproduce C_M from
   Hugging Face weights and a published recipe?
6. Has any author of a scheme in Table `tab:sok` replied, and did any reply change a verdict?
7. The statement binds the token ids. Who tokenises the prompt, and how are provider-side system prompts handled?

### Score: 2 (weak reject)

The attack and the theorem are sound and well written, but the defence's practicality is not established, and the
evaluation is not finished. With a finished single-platform evaluation, the deployment comparison of Weakness 1 and
the hardened-protocol attack of Weakness 2, this would be a 3 or a 4.

---

## Reviewer B: cryptography and proof systems

### Summary

There are two technical contributions. (1) A lower bound: a verifier that reads the weights through a commitment of
locality d and makes q queries per layer rejects an explainable single-site forgery with probability at most
G_{qd}(J | P), the probability of guessing the edited site in qd tries. On a constructed ReLU network the forgery
passes with probability at least 1 - qd/N. Mixing commitments reach 2^-lambda with O(lambda + log L) queries per
layer. (2) A protocol that applies Freivalds' check to every weight product, with the folded rows tested against a
Ligero/Brakedown-style Reed-Solomon column commitment. It comes with a soundness theorem, a round-by-round
Fiat-Shamir argument, a Ligero-style one-time proximity proof for untrusted commitments (using correlated agreement
over an extension field), and V1: a windowed logUp-GKR over F_{p^8}, with a lemma on integer multiplicities for more
leaves than the field's characteristic, reduced to the existing column check.

### Strengths

1. The proofs are careful. I checked the Freivalds and column-check arguments (Appendix B.2), Fact 1 and its proof,
   the no-wrap lemma of V1 (p - 2^30 - 2^16 = 939,458,561 > 2^16), the integer-multiplicity lemma (c_tau = mu_tau +
   kp with k >= 0 and the integer sum check force off-table counts to zero), the pole argument with alpha drawn from
   K minus (F + xi F), and the GKR error accounting. I found no error.
2. The Kpre analysis with verdict leakage (a hybrid until the first bad check) is correct and is often overlooked in
   designated-verifier designs.
3. Binding every constant of the operations without weights into the Fiat-Shamir statement, and using a prefix-free
   sponge, shows awareness of weak Fiat-Shamir failures.
4. V1 is a clean idea: the verifier knows each operation's input, so the window conditions are affine in already
   committed weights, and the GKR's final claim reduces to folded rows. No activation commitment is needed.
5. Remark `rem:pointer` (correlated rows let an adaptive verifier win) and Remark `rem:merkle` (unsalted Merkle trees
   leak sibling information) show the authors probed the limits of their own theorem.

### Weaknesses

1. **Low cryptographic novelty in the defence.** Freivalds' check against a Ligero/Brakedown column test is the
   evaluation protocol of a linear-code polynomial commitment applied to matrix-vector products. LAMP (Freivalds plus
   RS proximity inside Groth16), Maverick (sparse Freivalds against a code-encoded copy) and CommitLLM (int8 Freivalds)
   are close. The novelty claim (Section 2, "Where we fit") rests on the combination: no enclave, no stored copy, no
   weight-derived secret. That is a systems point, and it should be argued as one.
2. **The split-commitment attack is the reason PCS evaluation proofs include a proximity test.** Ligero and Brakedown
   pair each evaluation with a testing phase precisely because a root can bind a non-codeword matrix. Presenting the
   attack as a finding (Section 4.5, Eq. `eq:split`) risks reading as a rediscovery. Frame it as "we amortise the
   PCS's testing phase into a one-time setup proof, and here is the parameter trade-off", and cite the standard
   treatment.
3. **The setup proof is the most naive proximity test available.** It opens t^0_j full columns per tree and reaches
   2.8-11.3 GB for Llama-2-7B. Batched FRI/STIR/WHIR- or BaseFold-style proximity proofs over the rows would
   presumably be orders of magnitude smaller. Why were they not considered? DeepBrake and ReedWeave (row-wise RS
   commitments, listed in your own literature notes) are not cited.
4. **The lower bound is close to folklore, and its model is bespoke.** The paper admits the spot-check calculation
   is folklore and places the novelty in extending it to adaptive, value-aware verifiers that read the whole trace.
   The proof is a coupling up to the first touch plus a fractional knapsack (the hidden-object search of Chew and
   Kadane). That is a modest step. Its value would be clearer if the verifier class were a formal definition in the
   body, giving the query model, how prover messages such as the folded rows u are modelled as oracle queries of
   support N, and what lies outside it (for example, verifiers assisted by a succinct argument). Today the precise
   definition (Spot_d(q), completeness over admissible instances, the budget measured under the reference oracle
   R_p) appears only in Appendix A.
5. **The worst-case instance is contrived and does not reach transformers.** Proposition `prop:worst` uses an AND
   network with copied input bits and dead inputs that carry random weights (needed for Merkle min-entropy).
   Remark `rem:scope`(v) concedes it does not embed in a transformer. Part (a) of the "dichotomy" is therefore a
   statement about ReLU MLPs. Part (b) is the defence's per-layer soundness. Intermediate locality (1 < d < N) is not
   shown to be tight. "Dichotomy" oversells a pair of bounds at the extremes.
6. **Citations to verify.** The setup proof cites BCIKS20 "Thm. 1.6" for uniformly random linear combinations
   (an affine subspace) in the unique-decoding regime with error n/|K|. Please check that the theorem number and its
   hypotheses (random combination versus a curve or line) match. The Ligero lemma holds for e below d/4 (Roth and
   Zemor extend it to d/3); state which one is used. The integer-multiplicity treatment of logUp with N >= p should
   be positioned against what deployed small-field logUp implementations already do.
7. **Concrete security is scattered.** The values r = 5 (interactive) and r = 7 (Fiat-Shamir), beta, beta_0 =
   lambda + 64 + log2(2|J|), s, t^0_j, t_j, the extension degree 8, the 64 grinding bits and the hash lengths are
   spread over Sections 4.3 to 5.5 and Appendices B and C. One table should give each term of each bound for one
   model, say Llama-2-7B at lambda = 128. The tables also mix interactive runs (Table `tab:eval-llm`, Table
   `tab:gen`) with Fiat-Shamir runs (Table `tab:eval-v1`). A "proof" should be non-interactive by default.
8. **The untrusted-commitment theorem certifies an arbitrary field model M\*.** Without a range proof, the client
   learns only that some fixed model with field weights is committed. For closed weights this is the Hollow-LLM
   problem, and the protocol leaks closed weights anyway. The theorem is correct, but its practical reach is narrow.
9. **"Attention included" is ambiguous.** The verifier recomputes attention in the clear; nothing about attention
   is proved. The novelty sentence ("fully sound for complete transformers, attention included") should say
   "attention is recomputed by the verifier from checked values". As written, readers will assume attention is
   proved.

### Questions to the authors

1. Which exact statement of BCIKS20 do you invoke, and does it cover uniformly random combinations of R rows over
   K_s with error n/|K_s|?
2. Why not replace the column-opening setup proof with a FRI/STIR/WHIR- or BaseFold-style proximity proof, and what
   would its size be for Llama-2-7B?
3. Does the lower bound cover verifiers that receive additional prover messages checked against the commitment,
   other than the folded rows you model? Where exactly does a SNARK-assisted verifier fall outside the model?
4. Is the bound tight for intermediate locality d, for example block-wise mixing over groups of d rows?
5. In V1 under Fiat-Shamir, the column check uses eq-structured vectors chi_l derived from GKR challenges rather than
   uniform vectors. Please state explicitly that the column-check argument needs only that u_l was fixed before the
   columns were drawn, and that this holds in the round-by-round argument.
6. For the round-by-round bound you use the sum epsilon_C as the per-round error. Is the tighter maximum (the
   per-round rescue probability) what the parameters actually use?

### Score: 3 (borderline)

Technically solid and rigorous, but the cryptographic contribution is incremental. The defence composes known
pieces, and the lower bound formalises a folklore calculation. Precise positioning (Weaknesses 1, 2, 4), a formal
verifier model in the body and a non-naive setup proof would raise this to a 4.

---

## Reviewer C: ML systems

### Summary

The defence certifies an exact integer transformer: int8 weights and activations, static scales, an integer
LayerNorm, an exponential table with 8-bit probabilities, a 23-bit residual stream. The prover runs it on int8
tensor cores and sends the result of every weight operation. The verifier recomputes everything except the weight
products and checks those with Freivalds' algorithm. Improvements include int8 GEMMs for the column opening and the
fold (3.7-4.3x on Llama-2 blocks), a threaded claim encoder, a fused exact attention kernel, a GPU claim decoder, a
native CPU attention kernel, one-pass proofs of generation, an exact Gumbel-max sampler, and SmoothQuant folded into
the norm gains (OPT-125M and OPT-1.3B within x1.03 of fp32 perplexity).

### Strengths

1. The exact-integer approach removes the tolerance problem cleanly, and the hardware exactness bounds (int32 block
   sums, fp32 attention below 2^24, a self-check with a float64 fallback) are well thought through.
2. Every improvement is measured against the previous code path on the same queries, with identical bytes and
   verdicts, and the gains are honest (1.1x on GPT-2, where the opening is not the bottleneck).
3. Proving a whole response as one prefill, with a token rule, is the right design. The prover's cost grows only
   with the new positions (GPT-2: 0.29 s for 256 tokens).
4. The quality attribution (leave-one-out and add-one) points to the LayerNorm rounding caused by OPT's outlier
   channels, and the SmoothQuant-through-gains fix changes no checked operation.

### Weaknesses

1. **The evaluation is not finished.** 43 of the 73 todos are in Section 7. All "new" timings are missing; the 7B
   and 13B models were measured as one to four blocks and extrapolated; peak memory (F4) is not reported; costs use
   random int8 weights. The paper itself says claims compress differently with real weights, and V1 drops from
   1.94-2.06x (random) to 1.68-1.77x on real smoothed OPT-125M at 512 tokens. A real-weight end-to-end run of at
   least one 7B-class model is the bar set by zkLLM and zkAgent.
2. **The metric that matters to a provider is missing.** The prover is compared with zkSNARK provers (the abstract's
   "46 to 32,000 times faster"), but never with plain inference. From the paper's own tables (different GPUs) the
   base prover appears to be one to two orders of magnitude slower than an fp16 prefill: 1.8 s for Llama-2-7B at 64
   tokens on the L40S against 0.037 s fp16 on the 2080 Ti; 7.7 s against 0.39 s for OPT-6.7B at 2,048 tokens. The
   integer forward pass alone is about 2x slower than fp16 even after the int8 GEMMs. Batching, continuous batching
   and paged KV caches are not discussed. The fixed per-query work (opening and folding every matrix) does not
   shrink with short prompts and will dominate a batched serving workload unless amortised across queries.
3. **The 32,000x headline is confounded.** It compares an L40S GPU prover with ZKML on a CPU, and a non-ZK protocol
   whose verifier recomputes all non-linear operations with zkSNARKs that prove them. The paper notes the CPU point,
   but the abstract does not.
4. **The verifier does not realise its asymptotic advantage.** At 2,048 tokens, attention is under 10% of
   Llama-2-7B's FLOPs, yet recomputing the operations without weights takes 76% of the verifier's time, and the
   verifier ties fp32 re-execution on the CPU. Exact attention can run as int8 x int8 -> int32 GEMMs (QK^T) and
   uint8 x int8 GEMMs (PV) on CPUs and GPUs. If the verifier's attention ran at GEMM speed, verification would cost a
   small fraction of re-execution, which is the paper's main missing argument. Today the B5 target (85 s down to
   20 s or less) is not reached.
5. **The quality evaluation is too thin for the claims.** Perplexity is measured on 16 windows of 128 tokens of
   WikiText-2 (2,048 tokens of text in total), only for OPT-125M and OPT-1.3B. The cost tables go to 2,048-token
   prompts on Llama-2-7B, Llama-2-13B, Qwen3-4B and OPT-6.7B, whose integer quality is unknown. OPT-6.7B is 33% above
   fp32 with a scalar gain. Static per-tensor W8A8 is known to degrade Llama-family models more than OPT. With 8-bit
   attention probabilities, at 2,048 positions most diffuse attention weights round to zero; this is never measured,
   because no context exceeds 128 tokens. There are no downstream tasks, and zkLLM reports near-lossless quantisation
   at 16 bits.
6. **The sampler's distortion is unmeasured.** The 16-bit table bounds the Gumbel noise to about [-2.5, 11.8]
   (Gamma[0] and Gamma[2^16 - 1] / 2^16), so a token whose temperature-scaled logit is more than about 14.3 below the
   top one can never be sampled. The total-variation distance to softmax sampling on real logits is a todo. Top-p,
   repetition penalties and logit bias, which serving stacks use, are not supported.
7. **Scope versus current models.** The scope is dense decoders only: no MoE, no long contexts (proof size and the
   O(T^2) verifier at 8k-32k tokens are not projected), no tensor parallelism. Llama-3-class shapes, with a 128k
   vocabulary and GQA (plan F8), are absent.
8. **Statistical reporting.** Medians of three queries, with no variance or confidence intervals.

### Questions to the authors

1. What is the prover's time and throughput relative to unverified fp16/bf16 and int8 serving of the same model on
   the same GPU, with batching?
2. What are the WikiText-2 perplexity (full test set, 2,048-token context) and downstream accuracy of the integer
   Llama-2-7B, Qwen3-4B and OPT-6.7B with smoothing?
3. How much quality do 8-bit attention probabilities cost at 2,048 tokens?
4. What is the peak GPU and host memory of the prover and the verifier for the 7B models at 2,048 tokens? (V1 in
   mode C does not fit in 11 GB.)
5. Is the generation proof produced from the decoding run's KV cache, or by a separate prefill after decoding? What
   does the latter add to serving cost?
6. Can the verifier's exact attention run on int8 GEMMs, and what would the CPU verifier then take at 2,048 tokens?
7. How large are the proofs, and how long does the verifier take, at 8k and 32k tokens?

### Score: 2 (weak reject)

The engineering is careful, but the evaluation is incomplete and leaves out the metrics a systems reader needs:
overhead against serving, quality at the evaluated lengths and models, and memory. A completed L40S evaluation with
real weights, the serving-overhead numbers and a long-context quality evaluation would make this a 3. A verifier
that clearly beats re-execution would make it a 4.

---

## Where the reviewers agree

All three see a coherent and rigorous paper whose two halves are each incremental on their own: the attack
formalises a known limitation, and the defence composes known checks. Its acceptance therefore depends on showing
that the defence is practical for a defined client. Today it is undermined by GB-scale proofs, a verifier that does
not beat re-execution, a quantised model of unknown quality at 7B scale, and an unfinished evaluation. The length
problem is severe enough that a desk check could flag it.

---

## Consolidated changes, ordered by expected impact on acceptance

| # | Change | Section(s) | Needs |
|---|---|---|---|
| 1 | Finish the evaluation on one platform with the frozen code: the L40S prover and GPU verifier, and one CPU for the CPU verifier and re-execution. Use full models (Llama-2-7B, Llama-2-13B, OPT-6.7B, Qwen3-4B) at 64 / 512 / 2,048 tokens with no extrapolation. Run real weights end to end on at least two LLMs (F3), report peak memory (F4) and give variance. Fill all 73 `\todo`s, and remove the cross-table disagreements (698 ms against 1.5 s, 58.2 s against 85 s). | 7 (all tables), 5.7, 6.1 (Table `tab:gen`), abstract, 1 | New experiments (F6, F3, F4) |
| 2 | Meet the length limit (13 pages of body plus 5 pages of references and appendices per the plan; check the 2027 CFP). Build with Times; the current PDF falls back to Latin Modern. Cut duplication: report attack results once (merge Sections 3.5 and 7.2) and V1 numbers once (Section 7.4). Move to appendices the integer-semantics details, the GKR implementation paragraph, Table `tab:setup`, the quality attribution table and Table `tab:aligned`. Shorten the introduction from about 2.5 to about 1.5 pages and trim Related Work. | All | Writing only |
| 3 | Make the deployment case quantitative. Add a table or figure of "options for a client": download the weights and re-execute exactly (CPU, GPU); a GPU TEE; CommitLLM-style audit; zkLLM or zkAgent; ours in modes C and Kpre and with V1. Give per-query bytes, client time and provider overhead against unverified inference, as functions of T and of the number of queries, with break-even points. Name the target client in the introduction. Add GPU confidential computing to Related Work and the Discussion. | 1, 2, 7.5, 8 | Mostly writing; prover-to-inference ratios from the L40S run; TEE overheads from the literature |
| 4 | Realise the verifier's asymptotic advantage. Run exact attention on int8 GEMMs (QK^T, PV) on the CPU and the GPU so that verification is clearly cheaper than re-execution at 2,048 tokens (attention is under 10% of the FLOPs). Report verify-to-re-execute ratios on the same hardware for every cell. | 4.3 (costs), 7.5, 8 | New engineering and experiments (B5, B6, F1) |
| 5 | Attack the hardened path protocol on a real LLM: range checks with held-out calibration plus the best value-aware sampler (G3 / G3.5, OPT-6.7B or Qwen3-4B). Run the adaptive sampler game (G3) and the tamper locator that estimates G_q on a trained network (G1e). Report how many in-range neurons a next-token change needs. | 3.3, 3.5, 8 | New experiments |
| 6 | Re-position novelty and temper headline claims. (i) The lower bound is a formalisation for adaptive, value-aware verifiers; say what is new against holographic proofs and PCPPs, and avoid "dichotomy" unless intermediate d is addressed. (ii) The defence is a linear-code PCS evaluation applied to matrix products, with the testing phase amortised at setup; present the split commitment as motivation, not as a finding; cite DeepBrake, ReedWeave and LAMP precisely. (iii) Replace "46 to 32,000 times faster than published zkSNARK provers" in the abstract with a hardware- and guarantee-matched statement plus the overhead against inference. (iv) Qualify "halves the proof": 2.0-2.3x with random weights, 1.68-1.77x on real OPT-125M at 512 tokens. (v) "Attention included" becomes "attention recomputed by the verifier". (vi) Soften the TensorCommitments verdict to "its stated goal is not achieved by its layer selection". | Abstract, 1, 2, 3.4, 3.6, 4.5, 7.6 | Writing only |
| 7 | Evaluate quality where the costs are reported. Use the full WikiText-2 test set at 2,048-token context and two or three downstream tasks, on Llama-2-7B, Qwen3-4B and OPT-6.7B with smoothing. Measure the effect of 8-bit attention probabilities at long context. Measure the sampler's total-variation distance to softmax sampling on real logits. | 6.2, 6.3, 7 | New experiments (F2, E2) |
| 8 | Ethics and policy hygiene. Send the disclosure notices and summarise the replies (G5); add an open-science / artifact statement (H4). Describe the LLM-agent "adversarial review" of V1 accurately under the CFP's AI-use policy, or move it out of the body. Remove the anonymity leak "in our earlier evaluation" (Section 6.3). | 1, 3.6, 5.6, 6.3, end matter | Writing and an action by the team |
| 9 | Make the cryptography precise. Put a formal definition of the verifier class at the start of Section 3.3: query model, how prover messages such as folded rows are modelled, and what is excluded (for example, SNARK-assisted verifiers). Give one concrete-security table (lambda, beta, r, t_j, s, t^0_j, extension degree, grinding bits, hash lengths) for one model. Confirm the BCIKS20 theorem and the Ligero lemma radius. Use Fiat-Shamir consistently in all cost tables, or label each table. | 3.3, 3.4, 4.3-4.5, 5.5, App. A-C | Writing only |
| 10 | Settle the untrusted-commitment story. Either justify the client who cannot recompute C_M but can check a 2.8-11.3 GB proof, or demote the setup proof to an appendix and make "recompute C_M from a deterministic public quantisation recipe of the fp16 checkpoint" (D6) the main path, stated in the threat model. If the setup proof stays, measure its time and estimate a FRI/STIR/WHIR-style alternative. | 3.1, 4.5, 8 | Writing; setup timing is a small experiment |
| 11 | V1: obtain L40S timings (including Llama-2-7B in mode C, which does not fit the 2080 Ti) and real-weight bytes at 2,048 tokens. Add a bandwidth crossover: the link speed below which V1 wins end to end, given its 10x prover and 1.6-1.9x verifier cost. Cut V1's body text to about 1.25 pages. | 5, 7.4 | New experiments and writing |
| 12 | Generation at 7B scale: Llama-2-7B and Qwen3-4B, P = 512, G in {64, 256}, greedy and sampled, with the tamper cells. State whether the proof reuses the decoding run or needs an extra prefill, and what that costs. | 6.1, 6.2 | New experiments |
| 13 | Optional: amortise the fixed per-query cost (one opening and fold per session of B queries, plan C4), so short prompts in mode C do not pay about 125 MB of opened columns each. | 4.3, 7.3 | New engineering and experiments |

### Smaller issues

- Section 7.2 says "the best [sampler] detects 2%", while Section 3.5 reports 9.2% for the contribution-weighted
  sampler. Both are right, since 2% is the worst case over the attacks found against each sampler, but the reader
  needs that stated in one sentence.
- The introduction gives "2.0-2.3x smaller at 2,048 tokens" without the qualifiers "random weights" and "builds with
  more than one block" (the single Llama-2-13B block gives 1.81x).
- Table `tab:eval-reexec` extrapolates the full models linearly from 1- and 2-block builds, and the comparison
  paragraph sets an extrapolated 1.1 s against zkLLM's measured 0.90 s. Remove both once the L40S run exists.
- Table `tab:eval-llm` labels columns "base" and "new" but has no "new" CPU-verify column. The note beneath it is a
  todo.
- The abstract has nine numbers. Keep three or four, and move hardware-specific factors to the body.
- Section 6 mixes a protocol extension (generation, sampling) with an evaluation topic (quality). Quality belongs
  with the evaluation.
- The contribution list says "nine published inference-verification schemes"; the table has nine rows, but one
  combines TOPLOC and SVIP, and Proof-of-Learning is not an inference scheme. Recount or rephrase.
- Theorem `thm:guess` measures the query budget under the reference oracle R_p (Appendix A). Say in the body that
  this equals the real budget up to the first touch, or readers will look for a gap.
- Proposition `prop:worst` needs "c >= kappa/b dead input coordinates" only for the unsalted Merkle case. With a
  hiding (salted) commitment the instance is simpler, so state that first.
