# Review of the `revision1` branch

**Branch reviewed:** `origin/revision1` at commit `9ca4ba5`, which is five commits on top of `main` (`1b7698d`).
**Short version:** there's one genuinely valuable idea here (the whole-layer check in `protocol/batched.py`), but three of the branch's headline claims don't survive re-measurement, and the new defence currently covers only individual hidden layers, not the whole network. I'd fix the claims and finish the defence before anything from this branch goes into the report.

Every number below was re-measured on the branch itself, not copied from its documents. The last section lists exactly what was run, and the appendix has the scripts so you can reproduce each finding.

---

## Contents

1. [Summary of findings](#1-summary-of-findings)
2. [What's good and worth keeping](#2-whats-good-and-worth-keeping)
3. [Problem 1: the new defence covers one hidden layer, not the network](#3-problem-1-the-new-defence-covers-one-hidden-layer-not-the-network)
4. [Problem 2: the "Theorem 4" table is computed on the wrong trace](#4-problem-2-the-theorem-4-table-is-computed-on-the-wrong-trace)
5. [Problem 3: the "targeted is 3.1× more detectable" claim is wrong](#5-problem-3-the-targeted-is-31-more-detectable-claim-is-wrong)
6. [Problem 4: the defence's security parameters are too small](#6-problem-4-the-defences-security-parameters-are-too-small)
7. [Problem 5: documents contradict each other or overstate](#7-problem-5-documents-contradict-each-other-or-overstate)
8. [Problem 6: smaller code issues](#8-problem-6-smaller-code-issues)
9. [Problem 7: repository housekeeping](#9-problem-7-repository-housekeeping)
10. [Suggested order of work](#10-suggested-order-of-work)
11. [How the report could frame all this](#11-how-the-report-could-frame-all-this)
12. [What this review did and didn't cover](#12-what-this-review-did-and-didnt-cover)
13. [Appendix: reproduction scripts](#13-appendix-reproduction-scripts)

---

## 1. Summary of findings

| # | Severity | Finding | Where |
|---|---|---|---|
| 1 | **High** | The batched check verifies one hidden layer in isolation. It rejects an honest prover at the output layer (20/20) and doesn't link layers together, so "detects every attack" is only true for attacks placed on the layer being checked. | `protocol/batched.py`, `REVISION1_PLAN.md:3`, `DEFENCE_NOTES.md` §10–§11, `phase4_batched_defence.md` |
| 2 | **High** | The "exact minimax" numbers behind "Theorem 4 / no ε ever beats uniform" measure the **honest** trace. For a sampler that reads the claimed activations, detection depends on the **tampered** trace. Re-measured correctly, the conclusion isn't supported and may be false. | `scripts/run_theorem4_check.py:161`, `DEFENCE_NOTES.md` §6, README Step 5, `phase0_summary.md`, `phase3_gpu.md:82` |
| 3 | **High** | "A targeted backdoor is 3.1× more detectable under plain uniform sampling" reports `1/|U|` as if it were uniform's detection. Uniform's detection of a one-neuron tamper is exactly `1/N` whether targeted or not (measured: 0.001953 both). | `scripts/run_targeted_vs_untargeted.py:107`, `DEFENCE_NOTES.md` §8, README Step 5, `phase0_summary.md`, `REVISION1_PLAN.md:21` |
| 4 | Medium | Hash-derived challenges over a 26-bit field with 24 spot-checks give only about 24 bits of security, which a cheating prover can brute-force by retrying. "`2^-16` would do" is only true if the verifier sends the challenges. | `batched.py:99`, `:340`, `:380`; `phase4_batched_defence.md:116` |
| 5 | Medium | README Step 5 still says the defence is "in progress" while the plan says "found one". Several overstatements, a broken file reference, and results tables that don't match a fresh run. | README, `DEFENCE_NOTES.md`, `batched.py:74` |
| 6 | Low | Code nits: the verifier is handed the prover's object that contains the weight matrix, spot-check columns can repeat, `scale_bits` defaults disagree (8 vs 10, and 10 overflows once layers are chained), and a documented command doesn't work. | `batched.py`, `run_batched_defence.py:96`, `train_models.py:38`, `zoo.py:209` |
| 7 | Low | Housekeeping: every file made executable, README quick start replaced with personal cluster commands (and internally inconsistent), dependency pins removed, cluster scripts with hard-coded home paths, and an unprofessional commit message. | commits `007f481`, `9ca4ba5` |

All 181 tests pass on the branch (1 skipped, the same one skipped on `main`), and nothing in the existing code regressed.

---

## 2. What's good and worth keeping

- **The whole-layer check is correct for a single ReLU layer.** Instead of checking one sampled neuron, `batched.py` checks every neuron of a layer at once with a random linear combination, in exact integer arithmetic, and the verifier never holds the weights. I reproduced the headline numbers on layer 1 (40 queries) and it also works on layer 2 checked on its own:

  | Layer | Honest completeness | Naive tamper | Stealthy tamper | Zero-hiding backdoor | Proof size |
  |---|---|---|---|---|---|
  | 1 (512 wide) | 40/40 | caught 40/40 | caught 39/39 | caught 40/40 | 115.7 kB |
  | 2 (256 wide) | 20/20 | caught 20/20 | caught 20/20 | caught 20/20 | 62.0 kB |

  For comparison, uniform path sampling catches the zero-hiding backdoor 3.2% (layer 1) and 9.4% (layer 2) of the time, and contribution weighting catches it 0% of the time.

- **The key insight belongs in the report.** The standard algebraic check for a ReLU, `a_j (a_j − z_j) = 0`, says nothing about neurons whose output is zero. So the zero-hiding attack that defeats contribution weighting would also slip through a naive whole-layer check: the same blind spot, in algebraic form. Making the prover supply `y_j = −z_j ≥ 0` for the zero neurons closes it. That's a real observation, and it ties Step 4's result to the new protocol nicely.

- **Exact integer arithmetic removes the `1e-4` float tolerance** and the free perturbation budget that came with it. Honest completeness is exactly 1.

- **It's fast.** Measured at scale 2⁸: layer 1 proves in 2.4 ms and verifies in 3.3 ms; layer 2 takes 1.0 ms and 1.7 ms. This was listed as "not profiled" in the notes.

- **The tests in `tests/test_batched.py` are good:** tampering at any position, zeroing one or many neurons, lying about the sign witness, a forged combination, and openings from a different model are all rejected.

- **The PCA plausibility work (§7) is sound.** I checked whether its evasion attack relied on negative activations (impossible after a ReLU), and it doesn't: it clips with `max(value, 0.0)`.

- **The GPU training changes are careful.** CPU stays the default, the large model is kept out of the default pipeline, and the docstrings are honest that GPU training isn't bit-for-bit reproducible.

- **The notes are candid in several places.** They state "one layer at a time" and "honest committer assumed", and §9 openly says its prototype assumed the verifier could see the weights.

---

## 3. Problem 1: the new defence covers one hidden layer, not the network

### What the branch claims

- `REVISION1_PLAN.md:3`: "`protocol/batched.py` (Phase 4) detects every attack in this project with probability `1.000000`".
- `DEFENCE_NOTES.md` §11 (line 510): "detection `1.000000` against every attack in this document".
- `phase4_batched_defence.md:127`: "this is less of a gap than it sounds: the attack has to tamper *somewhere*, and the check can be applied at whichever layers the path visits."
- `phase4_batched_defence.md:111`: "the protocol's real efficiency claim — visit `O(depth)` layers rather than proving the whole network — is untouched."

### What's actually the case

**a) The output layer isn't supported.** `fixed_point_layer` always sets `outputs = np.maximum(z, 0)` (`batched.py:234`), and `verify_layer` rejects any negative output (`batched.py:428`). But our output layer has no ReLU: it outputs raw scores (logits), and about 77% of them are negative. So at the output layer the check verifies `relu(logits)`, which isn't what the model outputs. Given the real logits, an honest prover is rejected every time:

```
honest prover, output layer, 20 queries:
  outputs = relu(logits)  (what fixed_point_layer builds): accepted 20/20
  outputs = actual logits (the model's real output)      : accepted  0/20
```

The output layer is the one that carries the answer, so this isn't a corner case.

**b) Layers aren't linked.** `run_batched_defence.py` checks a single layer (`--layer`, default 1), and `honest_view` builds that layer's input by converting the *float* trace to integers (`run_batched_defence.py:53–63`), independently for each layer. Nothing ties the verified output of layer 1 to the input of layer 2. So even if you ran the check on every layer this way, the prover could hand layer 2 an input that doesn't match layer 1's verified output. The verifier also takes `inputs` from the prover's `view` rather than computing them itself.

**c) "Whichever layers the path visits" means every layer.** In an MLP, a path goes from the output all the way to the input, so it visits every layer. Applying the check "at whichever layers the path visits" therefore means applying it at all of them, which is exactly the part that isn't built. And if any layer is left to path sampling, the attacker simply tampers that layer (layer 2 under uniform sampling: caught 1/256 of the time).

**d) Checking every layer in full isn't sampling any more.** Once every layer is checked completely, there's nothing left to sample. That's a (lightweight) proof over the whole network, so "the sampling protocol's efficiency is untouched" doesn't hold. The right comparisons are the cost of just downloading the model and the cost of a SNARK, not the 13 kB sampling proof. (This is good news for the report, see §11.)

**All the measured "1.000000" results are real, but every attack in them was placed on the layer being checked.**

### How to fix it

The notes say chaining layers needs "a larger prime or per-layer rescaling with a range argument (standard zkML engineering)". There's an easier route here. In our protocol **the activations are sent in plain view anyway** (it's not zero-knowledge; `SPEC_NOTES.md` §5). So the verifier can compute the rescaling itself, with no range argument needed.

**Step 1: define the fixed-point network precisely** (this becomes the committed model):

```
x_0 = round(query · 2^b)                          # input, scale 2^b
for layer l = 1..L:
    z_l = W_l_int · x_{l-1} + b_l_int              # exact integers, scale 2^(2b)
    a_l = relu(z_l)   (hidden layers)
    a_L = z_L         (output layer: identity)
    x_l = floor((a_l + 2^(b-1)) / 2^b)             # rescale back to 2^b (hidden layers only)
answer = argmax(a_L)
```

I simulated this end to end on `mlp_mnist_full` (appendix script D):

| Scale | Accuracy | Agrees with float model | Largest `|z|` per layer | Fits in the 26-bit field? |
|---|---|---|---|---|
| 2⁶ | 0.9797 | 99.90% | 47,646 / 51,444 / 139,055 | yes |
| **2⁸** | **0.9801** | **99.99%** | 767,244 / 838,260 / 2,225,729 | **yes, with 15× headroom** |
| 2¹⁰ | 0.9801 | 100.00% | 12.3M / 13.4M / **35.6M** | **no**: the output layer overflows the field's half-width (33.55M) |

The float model's accuracy is 0.9801. So **use scale 2⁸**: same accuracy, fits comfortably. The script's default of 2¹⁰ (`run_batched_defence.py:96`) would overflow at the output layer once layers are chained.

**Step 2: add an identity-layer check for the output layer.** For a layer with no activation you only need `a = z`. Draw a random vector `s`, open `u = sᵀA` with the same commitment, and check `⟨u, x⟩ = ⟨s, a⟩`. There's no zero set, no sign witness, and no non-negativity check (logits can be negative). The existing code is the ReLU case of this, so it's a small branch on the layer's activation.

**Step 3: the verifier computes every layer's input itself.**

```python
# sketch, untested
def verify_network(query, claimed_outputs, proofs, public_params, scale_bits=8):
    x = quantise(query, scale_bits)                        # computed by the verifier, never taken from the prover
    for layer, a, proof in zip(public_params.layers, claimed_outputs, proofs):
        if not verify_layer(proof, layer, inputs=augment(x), outputs=a):   # ReLU or identity variant
            return False
        if layer.activation == "relu":
            x = (a + (1 << (scale_bits - 1))) >> scale_bits   # public rounding rule, recomputed by the verifier
    return True
```

Soundness then follows layer by layer. The input is honest because the verifier computed it. If layer `l`'s input is honest, its check forces its output to be exactly correct, up to the challenges' soundness error (this relies on honest `|z|` staying below `p/2`, which the table above confirms). The verifier then computes layer `l+1`'s input from that correct output, and so on up the network. A published per-layer bound on activations is a cheap extra guard, but the induction doesn't need it.

**Step 4: bind all layers into one set of challenges.** Include the layer index and a hash of all earlier layers' messages in each layer's challenge derivation, so proofs for different layers or queries can't be mixed and matched. Better still, use challenges sent by the verifier (see Problem 4).

**Step 5: add end-to-end tests:**
- an honest full network is accepted, for several queries;
- a tamper placed in layer 1, in layer 2, and in the output layer is each rejected;
- a layer-2 input that doesn't equal the rounded layer-1 output is rejected;
- the output layer is accepted with negative logits.

**Step 6: measure the real cost** and compare it against the right baselines:

| | Sampling scheme (`main`) | Batched, full network (measured per layer, output layer estimated) | Just download the model |
|---|---|---|---|
| Proof / download | 13.4 kB | ≈ 190 kB (115.7 + 62.0 + a few kB for the output layer) | 2.14 MB (535,818 float32 parameters) |
| Prover | ~0.6–1 ms | ≈ 4 ms | 0 |
| Verifier | ~0.5–0.8 ms | ≈ 5 ms | full forward pass |
| Catches the backdoor | ~0.2–3% per query | every time | every time |

So the proof would be about 11× smaller than the model, at millisecond cost on both sides. At width 4096 the notes measured 837.6 kB for layer 1 alone against a 46 MB model; that should be re-measured end to end.

**Scope to state honestly afterwards:** dense layers only (no convolutions or pooling yet); the committed model is assumed honest, as in the paper; and like the paper, it certifies faithful execution of the committed model, not that the committed model is benign.

---

## 4. Problem 2: the "Theorem 4" table is computed on the wrong trace

### What the branch claims

`DEFENCE_NOTES.md` §6 states "Theorem 4 (no free floor)": for every ε, the ε-floor sampler (`ZeroAwareContributionSampler`, which scores a parent by `|w_ij|·(|ã_i| + ε)`) is beaten by an adversary below uniform's bound. It backs this with a table of "exact `min_v q(v)` over `U`", described as "provable worst cases, not merely observed ones". It then adds a methodological note that the heuristic attack search "understates this sampler's true worst case by ~15×". The same conclusion is repeated in README Step 5, `phase0_summary.md`, `REVISION1_PLAN.md`, and the width-4096 re-run in `phase3_gpu.md:82`.

### What's wrong

The "exact" block in `scripts/run_theorem4_check.py:161` computes

```python
visits = visit_probabilities_under(network, honest0, args.layer, sampler)   # the HONEST trace
```

and takes the minimum over `U`. That quantity equals detection only for samplers that ignore the claimed trace (uniform, static importance). That's why our own test of that identity, `test_detection_equals_visit_probability_for_trace_independent_samplers`, is restricted to exactly those two. The ε-floor sampler reads the claimed activations. When the attacker tampers neuron `v`, `v` holds its new flipping value in the *tampered* trace, and that value changes how often the sampler visits `v`.

Concretely, at ε = 0 the minimum on the honest trace is 0 because some neuron in `U` is honestly zero. But flipping the output through that neuron means *raising* it, and a raised value is exactly what contribution weighting looks at most.

### Evidence

Same model, same query, same `U` (454 neurons). "Actual" means detection of the actual tampered trace under that sampler (appendix script C):

| ε | Branch's number (honest trace) | Actual worst-case single-neuron detection | Uniform (1/N) |
|---|---|---|---|
| 0 | 0.000000 | **0.110330** | 0.001953 |
| 0.01 | 0.000037 | **0.106349** | 0.001953 |
| 0.1 | 0.000271 | **0.080421** | 0.001953 |
| 1 | 0.000747 | **0.024421** | 0.001953 |
| 100 | 0.000975 | **0.001747** | 0.001953 |

The branch's own script already shows the problem: its first column ("det. naive", computed on the tampered trace) reads 0.121 at ε = 0, which contradicts the "exact" block underneath it.

Now combine that with the other attack family, the multi-neuron zeroing attack from the branch's own search (15 queries, mean):

| ε | Multi-neuron zeroing (branch's search) | Single neuron (above) | Attacker's best | vs. uniform 0.00195 |
|---|---|---|---|---|
| 0 | 0.0000 | 0.110 | 0.0000 | attacker wins, as the branch says |
| 0.01 | 0.0018 | 0.106 | 0.0018 | roughly tied |
| **0.1** | 0.0118 | 0.080 | **0.0118** | **~6× better than uniform** |
| **1** | 0.0268 | 0.024 | **0.0244** | **~12× better than uniform** |
| 10 | 0.0182 | not measured | ≤ 0.018 | unknown |
| 100 | 0.0160 | 0.0017 | 0.0017 | attacker wins, as the branch says |

This is rough: the single-neuron column is one query, the multi-neuron column is a 15-query mean from a heuristic search, and a smarter attacker might do better. But it's enough to say:

- **"No ε ever beats uniform" is not established.** Against every attack we have, middle values of ε (0.1–1) are far better than uniform.
- **The "heuristic understates the worst case by ~15×" note is backwards for small ε.** It compared against the flawed honest-trace number. At ε = 100 the sampler barely depends on the trace, so the honest-trace number is close to right there, and that's the only place the note holds.
- **"Theorem 4" isn't a theorem.** Its argument ("the visit distribution moves monotonically between the endpoints, so it can never overshoot uniform") isn't proven, and the measurements now contradict it.

### How to fix it

1. **Measure detection on the tampered trace.** For single-neuron attacks, do what appendix script C does: for each `v ∈ U`, build `forward_from(honest.tampered(L, v, flip_value), L)` and compute `1 − acceptance_probability(net, forged, params, sampler)`; the worst case is the minimum. (`plan_adaptive_stealthy_flip` already scores its candidates this way; only the "exact minimax" block is wrong.)
2. **Give the attacker a real chance.** For this sampler, a tampered neuron's exposure is about `|w|·(|value| + ε)`, so the natural attacks are:
   - zeroing neurons with small outgoing weights (exposure `ε·|w|` each);
   - raising one neuron as little as possible;
   - **mixtures of the two**, which nothing tests yet;
   - a local search from the best plan found (swap or drop a neuron, keep the change if exact acceptance improves).
3. **Run many queries** (≥ 100), record each query's worst case, and report the mean and the maximum, per ε, alongside `1/N`. Repeat at width 4096 (`phase3_gpu.md` has the same flaw).
4. **Relabel "Theorem 4" as an empirical observation**, or prove it. Delete the "~15× understated" note, or restate it as applying only at large ε.
5. **If some ε still beats uniform against the stronger attacker, that's a positive result worth reporting:** a cheap sampler the verifier can actually compute, and much cheaper than the batched check. It would also refine Step 4's conclusion for this family (Step 4 only showed ε = 0 is catastrophic). If the stronger attacker breaks it, that's a clean negative result too, but it has to be measured properly first.

---

## 5. Problem 3: the "targeted is 3.1× more detectable" claim is wrong

### What the branch claims

`DEFENCE_NOTES.md` §8, README Step 5 (lines 292–298), `phase0_summary.md:17–23` and `REVISION1_PLAN.md:21`: under a targeted threat model the attacker's easiest target has `|U| = 159` instead of 465, so the backdoor is "3.1× more detectable under plain uniform sampling, for free, no new sampler required". The README presents this as "the real, no-cost improvement this revision found".

### What's wrong

`scripts/run_targeted_vs_untargeted.py` only computes `1/|U|` and labels it "i.e. uniform single-path detection" (line 107). But under uniform sampling, a one-neuron tamper in a 512-wide layer is visited, and so detected, with probability exactly `1/N = 1/512`. It doesn't matter how many neurons could have been used: the attacker just picks one of the 159.

`1/|U|` is something else. It's the most that *any* verifier which ignores the claimed trace could guarantee (Step 4's Theorem 1). Reaching it would mean concentrating samples on `U`, and the verifier can't do that because it can't compute `U`: `U` depends on the model's weights, the query and the target, and the verifier holds only a digest of the model.

### Evidence

Appendix script B, 15 queries, layer 1:

```
untargeted: mean detection 0.001953 (1/N = 0.001953)
targeted  : mean detection 0.001953 (1/N = 0.001953)
```

This also follows directly from an existing test on `main`, `test_single_neuron_acceptance_is_one_minus_one_over_width`, which holds whatever the target.

### What the correct statement is

Targeting does make a difference, but only when the attacker also has to keep its forged activations within natural ranges (the stealthy attack from Step 3). Then hitting one specific class needs more tampered neurons. From `main`'s `run_attack.py` stealth block (layer 1, target class 0):

| Activation cap | Untargeted: neurons / detection | Targeted: neurons / detection | Ratio | Targeted success rate |
|---|---|---|---|---|
| natural max | 2.9 / 0.0057 | 7.1 / 0.0139 | 2.4× | 93% (vs 98%) |
| 99th percentile | 4.2 / 0.0081 | 11.5 / 0.0225 | 2.8× | 87% (vs 98%) |
| 95th percentile | 5.6 / 0.0110 | 15.0 / 0.0293 | 2.7× | 83% (vs 98%) |

So: **against plain `RandPathTest` a targeted backdoor is exactly as cheap as an untargeted one (1/N). If the verifier also checks activation ranges, a targeted backdoor needs ~2.4–2.8× more tampered neurons, and so is ~2.4–2.8× more detectable, and it succeeds less often.** That's already in Step 3's stealth table, so it's a useful reinterpretation rather than a new result.

### How to fix it

- Rewrite `DEFENCE_NOTES.md` §8 with the statement above. If you keep the `|U|` table, rename its column to something like "ceiling for a verifier that knew `U` (not achievable)".
- Change the label in `run_targeted_vs_untargeted.py:107`, or better, have the script compute actual detection with `acceptance_probability`.
- Update README Step 5, `phase0_summary.md`, and `REVISION1_PLAN.md:21` accordingly.

---

## 6. Problem 4: the defence's security parameters are too small

### What's there

- The field prime is `67108859`, just under 2²⁶ (`batched.py:99`).
- The challenges `s, t, α, β` and the column positions are all derived by hashing the prover's own messages (`_derive`, `batched.py:340`): that's the Fiat–Shamir approach, where the prover computes its own challenges.
- `n_queries = 24` spot-checked columns at code rate 1/2 (`batched.py:380`).

Each part gives about 2⁻²⁴ to 2⁻²⁶ soundness. That's fine when **the verifier** sends fresh random challenges, which is how the original protocol (and our implementation on `main`) works. But with hash-derived challenges, a cheating prover can keep changing some part of its message and re-hashing until it gets a lucky challenge. At 2⁻²⁴ that's about 2²⁴ ≈ 17 million tries, which is cheap. So `phase4_batched_defence.md:116`'s "`2^-16` would do" is only true for verifier-sent challenges.

### How to fix it (pick one)

1. **Use challenges sent by the verifier, like the base protocol (recommended for the project).** The prover sends all claimed activations and sign witnesses first; the verifier replies with fresh random `s, t, α, β` and column positions for every layer; the prover then sends the combinations and openings. The current parameters are then fine, and the proof stays the same size.
2. **Keep hash-derived challenges but strengthen them.** Repeat the combination check with ~4 independent challenge sets (about 2⁻¹⁰⁰), and raise the column checks to about 100 at rate 1/2, or about 50 at rate 1/4. Column openings dominate the proof size, so at 100 columns layer 1 grows from ~116 kB to roughly 450 kB.
3. **Use a ~64-bit prime** (for example the Goldilocks prime 2⁶⁴ − 2³² + 1). This is more work, because products no longer fit in int64 and numpy needs a workaround.

Either way, bind the layer index and the previous layers' messages into the challenges (see Problem 1, step 4), and sample column positions **without replacement** (see Problem 6).

---

## 7. Problem 5: documents contradict each other or overstate

| File | Issue | Suggested change |
|---|---|---|
| `README.md` Step 5 (lines 279–344) | Says the whole-layer check is "the current, open, in-progress direction" (line 344), while the plan says it's done. Its headlines are Problems 2 and 3. | Rewrite after fixing Problems 1–3. |
| `REVISION1_PLAN.md:3` | "Outcome: found one … detects every attack in this project" | "A whole-layer check that is sound per hidden layer; extending it to the full network is in progress." Then update once Problem 1 is fixed. |
| `DEFENCE_NOTES.md` §10–§11 (lines 443–512) | "detection 1.000000 against every attack in this document"; "no primitive beyond the Merkle tree" | Say "every attack placed on the checked layer". It also uses Reed–Solomon codes and prime-field arithmetic, so "no cryptographic assumption beyond a hash function" is the accurate version. |
| `phase4_batched_defence.md:111, 127, 146` | "Efficiency claim untouched"; "whichever layers the path visits"; "no cryptographic machinery beyond the Merkle tree" | See Problem 1 (c) and (d). |
| `batched.py:74` | References `revision1_notes/phase4_batched.md`, which doesn't exist | It's `phase4_batched_defence.md`. |
| `phase3_gpu.md:82` | "Theorem 4: any ε beats uniform? no, no" at widths 512 and 4096 | Same flaw as Problem 2; re-run. |
| All tables on the branch | They don't match a fresh run (see below). | Regenerate every table from one consistent set of trained models. |

**Why the numbers drift:** the trained models aren't committed (`artifacts/` is gitignored), and the dependency pins were removed, so everyone's models differ slightly. My re-run with `main`'s models gave a zero-hiding support of 16.4 (the branch says 12.2) and contribution-weighting detection of 0.1255 for the naive attack (the branch says 0.1126). The conclusions don't change, but the report needs one consistent set. Record the library versions and device in each results JSON, and either share the model files (a few MB each) or regenerate everything from one machine before writing tables.

---

## 8. Problem 6: smaller code issues

1. **The verifier is handed the prover's object.** `verify_layer` takes `encoder: BatchedWeightCommitment`, which holds the full weight matrix (`batched.py:411`). The docstring says it only reads the public parts, and that's true, but on `main` we made it *structurally impossible* for the verifier to see the model. Passing a small public-parameters object instead (row length, column count, Vandermonde matrix, prime) would keep that property.
2. **Spot-check columns can repeat.** `columns = … % n_columns` (`batched.py:356`) samples with replacement, so repeated columns add nothing. Draw a random permutation and take the first `t` instead.
3. **Inconsistent `scale_bits` defaults:** 8 in `fixed_point_layer` (`batched.py:211`) and 10 in the script (`run_batched_defence.py:96`). Once layers are chained, 10 overflows at the output layer (Problem 1), so standardise on 8.
4. **The verifier's inputs come from the prover** (`view.inputs`). Fixed by Problem 1, step 3.
5. **`tampered_view` doesn't re-propagate to later layers.** That's fine while only one layer is checked, but it will need to use the full fixed-point forward pass once layers are chained, otherwise tampered traces will be inconsistent in ways a real attacker would avoid.
6. **The documented large-model training command fails.** `zoo.py:209` says to run `python scripts/train_models.py --only mlp_mnist_large --device auto`, but that prints "no models matched", because `train_models.py:38` only searches `all_model_specs()`, which excludes `LARGE_MLP_SPEC` on purpose. Import `LARGE_MLP_SPEC` from `pvi.zoo` and add it to the `--only` lookup:

   ```python
   specs = all_model_specs()
   if args.only:
       candidates = specs + (LARGE_MLP_SPEC,)
       specs = tuple(s for s in candidates if s.name in set(args.only))
   ```

   Or change the docstring to point at `run_gpu_scale_check.py`.
7. **Encoding cost will grow quadratically.** Reed–Solomon encoding uses a dense Vandermonde matrix: 1.8 ms of the 3.3 ms verify time at layer 1, and quadratic in layer width. It's fine at these sizes. If the 4096-wide model's timing matters, a prime that supports fast (NTT) encoding would bring it down to `O(M log M)`. Optional.

---

## 9. Problem 7: repository housekeeping

1. **Every file was made executable** (commits `007f481` and `9ca4ba5`, including the PDFs and the `.tex`). That's noise in every diff. On the cluster clone, run `git config core.fileMode false`, then undo the changes on the branch:

   ```bash
   git ls-files -s | awk '$1 == "100755" {print $4}' | xargs git update-index --chmod=-x
   ```

   Nothing in the repo needs the executable bit: the Python scripts run via `python …` and the job scripts via `sbatch`.
2. **The README quick start was replaced with personal cluster commands** that are also internally inconsistent: they create `/tmp/erel_venv` but then call `.venv/bin/pip install -e .` from the repo root, where there's no `pyproject.toml` (it lives in `code/`). Restore `main`'s quick start, and add a separate "Running on the TAU SLURM cluster" subsection with the cluster-specific steps.
3. **Dependency pins were removed** from `pyproject.toml`, and `pytest` was moved into the required dependencies. Restore the pins and put `pytest` back under `[project.optional-dependencies] dev`. If the cluster needs a CUDA build of torch, document installing the pinned version from PyTorch's CUDA index in the cluster section, rather than unpinning for everyone.
4. **`.wheel_cache/*.sh` hard-codes one person's course paths** (`/home/sharifm/teaching/tml-0368-4075/erelbarzilay/…`) and the partition. There are no secrets, but it shouldn't ship in the tarball as-is. Move the scripts to `cluster/`, replace the path with `PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"`, and add `.wheel_cache/` to `.gitignore`.
5. **Commit messages:** "whateverdefuq idos gpu did" and "commit pre claude" will be visible to graders. When this goes into `main`, squash-merge it with a descriptive message instead of merging the history as-is.

---

## 10. Suggested order of work

If time is short, do these in order; the first two groups are cheap and matter most.

**Before anything is quoted in the report**
- [ ] Fix Problem 3's wording everywhere it appears (`DEFENCE_NOTES.md` §8, README, `phase0_summary.md`, `REVISION1_PLAN.md`).
- [ ] Mark Problem 2's "Theorem 4" as unverified, and remove the "~15× understated" note until it's re-measured.
- [ ] Correct "detects every attack" to "every attack on the checked layer" (Problem 5).

**Housekeeping**
- [ ] File modes, README quick start, dependency pins, cluster scripts (Problem 7).
- [ ] Broken reference, `--only mlp_mnist_large`, `scale_bits` defaults (Problems 5–6).

**The valuable engineering**
- [ ] Output-layer (identity) check.
- [ ] Verifier-computed inputs with public rounding, and all layers bound into one set of challenges.
- [ ] Verifier-sent challenges.
- [ ] End-to-end tests with tampers at every layer.
- [ ] Full-network cost measured against the sampling proof and the model size, at widths 512 and 4096.

**Redo the ε-floor study**
- [ ] Score detection on the tampered trace, add a stronger attacker (including mixtures), run ≥ 100 queries, report per-query worst cases.
- [ ] Write it up as a positive or negative result, depending on what it shows.

**Consistency**
- [ ] Regenerate every table from one set of trained models, with versions recorded.

**Then write the report** (it doesn't exist yet). The course instructions gave September 10 as the deadline; if this is a revision round with a new date, favour the first two groups plus the output-layer and chaining fix over the ε-floor redo.

---

## 11. How the report could frame all this

Fixing Problem 1 turns the branch into a strong ending for the project rather than a contradiction of Step 4:

1. The paper's protocol works against the threat it models (swapping in a different model): 99.8–100% detection. **(Step 1)**
2. It breaks against a prover that runs the right model and changes one number in the trace: caught with probability `1/N`, with zero clean-accuracy loss as a trigger-conditional backdoor. **(Steps 2–3)**
3. No way of choosing *which* neurons to check fixes this. Uniform sampling is minimax-optimal among rules that ignore the claimed trace, and contribution weighting has an exact blind spot at zero. **(Step 4, plus the ε-floor result once redone properly)**
4. The fix is to change *what* a check verifies rather than *where* it looks: an algebraic whole-layer check, sound per layer and, once chained, over the whole network. It costs milliseconds and a proof ~11× smaller than the model, but it's no longer a sampling protocol. **(Revision 1, once Problem 1 is fixed)**

That's a sharper version of the intermediate report's thesis: sampling a sublinear number of locations can't give backdoor robustness, but a lightweight proof over the whole trace can, as long as the committed model is honest. It still says nothing about a committed model that was backdoored from the start (TM1), which remains a question for verifiable training.

---

## 12. What this review did and didn't cover

**Ran on the branch** (using `main`'s trained models, CPU):
- the full test suite: 181 tests, all pass (1 skipped, as on `main`);
- `scripts/run_batched_defence.py` at layer 1 (40 queries), layer 2 (20 queries), and layer 3 (crashes inside our own attack planner, which assumes a layer above the tampered one; that's a limitation of `main`'s code, not the branch's);
- `scripts/run_theorem4_check.py` (15 queries);
- `scripts/train_models.py --only mlp_mnist_large`;
- the four scripts in the appendix, plus a timing script for prove and verify.

**Read but not re-run:** `run_joint_plausibility_check.py` and `run_manifold_evasion_check.py` (code read; the clipping checked), `run_sumcheck_prototype.py`, `run_gpu_scale_check.py`, and the width-4096 results (no GPU here).

**Not reviewed in depth:** the PCA numbers themselves, and the GPU cluster notes beyond the job scripts.

---

## 13. Appendix: reproduction scripts

Run from `code/` on the `revision1` branch, with `artifacts/models/mlp_mnist_full.npz` present (train it with `python scripts/train_models.py`, or copy it from a machine that has it), and either `pip install -e .` from `code/` or `PYTHONPATH=src`.

### A. The output layer rejects an honest prover

```python
# Honest prover at the output (logits) layer, which has no ReLU.
import numpy as np
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.data import load_classification
from pvi.protocol.batched import BatchedWeightCommitment, fixed_point_layer, prove_layer, verify_layer

net = load_network(mlp_architecture_for(10), "artifacts/models/mlp_mnist_full.npz")
X = load_classification("mnist").test_x.reshape(10000, -1)
L = 3
W, b = net.parameters[net.architecture[L].name]
passed = {"relu(logits)": 0, "real logits": 0}
for q in X[:20]:
    prev = net.eval_trace(q)[L - 1]
    relu_view = fixed_point_layer(W, b, prev, scale_bits=8)          # what the library builds
    z = relu_view.pre_activations()
    real_view = fixed_point_layer(W, b, prev, activations_out=z, scale_bits=8)  # the actual output
    cm = BatchedWeightCommitment(relu_view.matrix, prime=relu_view.prime)
    for name, view in (("relu(logits)", relu_view), ("real logits", real_view)):
        proof = prove_layer(view, cm)
        passed[name] += verify_layer(proof, cm.digest, cm.params, cm, view.inputs, view.outputs, prime=view.prime)
print(passed)   # expected: {'relu(logits)': 20, 'real logits': 0}
```

Output: `{'relu(logits)': 20, 'real logits': 0}`

### B. Targeted vs untargeted under uniform sampling

```python
# Uniform-sampling detection of a single-node tamper: targeted vs untargeted.
import numpy as np
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.data import load_classification
from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ProtocolParams

net = load_network(mlp_architecture_for(10), "artifacts/models/mlp_mnist_full.npz")
X = load_classification("mnist").test_x.reshape(10000, -1)
detection = {"untargeted": [], "targeted": []}
for q in X[:15]:
    honest = net.eval_trace(q)
    target = (int(honest.output.argmax()) + 3) % 10
    for name, t in (("untargeted", None), ("targeted", target)):
        plan = plan_single_neuron_flip(net, honest, layer_index=1, target_class=t)
        if plan is not None:
            forged = apply_plan(net, honest, plan)
            detection[name].append(1 - acceptance_probability(net, forged, ProtocolParams()))
for name, values in detection.items():
    print(f"{name:<10}: mean detection {np.mean(values):.6f} over {len(values)} queries (1/N = {1/512:.6f})")
```

Output:
```
untargeted: mean detection 0.001953 over 15 queries (1/N = 0.001953)
targeted  : mean detection 0.001953 over 15 queries (1/N = 0.001953)
```

### C. "Theorem 4": honest-trace number vs actual detection

```python
# Theorem 4 table: honest-trace visit probability vs detection of the actual forged trace.
import numpy as np
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.data import load_classification
from pvi.attacks.tamper import _smallest_flipping_value
from pvi.defences import ZeroAwareContributionSampler
from pvi.defences.adaptive import visit_probabilities_under
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ProtocolParams

net = load_network(mlp_architecture_for(10), "artifacts/models/mlp_mnist_full.npz")
q = load_classification("mnist").test_x.reshape(10000, -1)[0]
honest, L, params = net.eval_trace(q), 1, ProtocolParams()
winner = int(honest.output.argmax())
U, forged = [], {}
for v in range(net.architecture[L].n_neurons):
    value = _smallest_flipping_value(net, honest, L, v, winner, None, max_value=1e4, require_nonnegative=True)
    if value is not None:
        U.append(v)
        forged[v] = net.forward_from(honest.tampered(L, v, value), L)
print(f"|U| = {len(U)}; uniform detection for a single-node tamper = 1/N = {1/512:.6f}")
for eps in (0.0, 0.01, 0.1, 1.0, 100.0):
    sampler = ZeroAwareContributionSampler(net, epsilon=eps)
    branch = visit_probabilities_under(net, honest, L, sampler)[np.array(U)].min()   # what the branch reports
    actual = min(1 - acceptance_probability(net, forged[v], params, sampler) for v in U)
    print(f"eps={eps:>6}: branch's number {branch:.6f} | actual worst-case single-node detection {actual:.6f}")
```

Output:
```
|U| = 454; uniform detection for a single-node tamper = 1/N = 0.001953
eps=   0.0: branch's number 0.000000 | actual worst-case single-node detection 0.110330
eps=  0.01: branch's number 0.000037 | actual worst-case single-node detection 0.106349
eps=   0.1: branch's number 0.000271 | actual worst-case single-node detection 0.080421
eps=   1.0: branch's number 0.000747 | actual worst-case single-node detection 0.024421
eps= 100.0: branch's number 0.000975 | actual worst-case single-node detection 0.001747
```

### D. End-to-end fixed-point network (supports the Problem 1 fix)

```python
"""Simulate an end-to-end fixed-point version of the MLP.

Each layer: z = W_int @ x_int + b_int (exact integers, scale 2^(2b)),
a = relu(z) (or identity for logits), then x_next = round(a / 2^b).
Reports accuracy vs the float model and the largest |z| per layer against the
field half-width of batched.DEFAULT_PRIME.
"""
import numpy as np
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.data import load_classification
from pvi.protocol.batched import DEFAULT_PRIME

net = load_network(mlp_architecture_for(10), "artifacts/models/mlp_mnist_full.npz")
ds = load_classification("mnist")
X = ds.test_x.reshape(len(ds.test_x), -1).astype(np.float64)
Y = ds.test_y
float_pred = net.predict(X.astype(np.float32))
half = DEFAULT_PRIME // 2
print(f"float model accuracy {np.mean(float_pred == Y):.4f}; field half-width {half:.3e}")

for bits in (6, 8, 10):
    s = 1 << bits
    x = np.rint(X * s).astype(np.int64)
    max_z = []
    for layer in net.architecture.layers[1:]:
        w, b = net.parameters[layer.name]
        w_int = np.rint(w.astype(np.float64) * s).astype(np.int64)
        b_int = np.rint(b.astype(np.float64) * s * s).astype(np.int64)
        z = x @ w_int.T + b_int
        max_z.append(int(np.abs(z).max()))
        a = np.maximum(z, 0) if layer.activation == "relu" else z
        x = np.floor_divide(a + s // 2, s)  # round(a / 2^b), exact integer rounding
    pred = x.argmax(axis=1)
    fits = all(m < half for m in max_z)
    print(f"scale 2^{bits:>2}: accuracy {np.mean(pred == Y):.4f}, agreement with float "
          f"{np.mean(pred == float_pred):.4f}, max|z| per layer {max_z} -> fits field: {fits}")
```

Output:
```
float model accuracy 0.9801; field half-width 3.355e+07
scale 2^ 6: accuracy 0.9797, agreement with float 0.9990, max|z| per layer [47646, 51444, 139055] -> fits field: True
scale 2^ 8: accuracy 0.9801, agreement with float 0.9999, max|z| per layer [767244, 838260, 2225729] -> fits field: True
scale 2^10: accuracy 0.9801, agreement with float 1.0000, max|z| per layer [12260133, 13415769, 35584531] -> fits field: False
```

### E. The branch's own scripts

```bash
python scripts/run_batched_defence.py --queries 40 --layer 1
python scripts/run_batched_defence.py --queries 20 --layer 2
python scripts/run_theorem4_check.py --queries 15
python scripts/train_models.py --only mlp_mnist_large     # prints "no models matched"
```
