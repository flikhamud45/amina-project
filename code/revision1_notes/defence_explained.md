# The batched layer check, explained from the ground up

This document explains the defence implemented in `src/pvi/protocol/batched.py`
assuming no cryptography background. It builds every prerequisite it needs —
hash functions, Merkle trees, finite fields, Reed–Solomon codes, Fiat–Shamir —
before assembling them. If you already know those, skip to §6.

Companion documents: `phase0_summary.md` (what we tried that failed),
`phase4_batched_defence.md` (the results, tersely), `DEFENCE_NOTES.md` §10.

---

## 1. The problem

You send a query to a cloud provider running a neural network. It sends back an
answer. **How do you know it actually ran the model it promised?**

Re-running the model yourself defeats the purpose. So we want the provider (the
**prover**) to send, alongside the answer, some evidence that a **verifier** can
check much more cheaply than re-running the network.

The paper we're studying (Anchuri et al., SaTML 2026) proposes a cheap scheme.
Our project attacked it. This document is the defence we built in response.

### The setup, concretely

A neural network layer takes a vector of inputs and produces a vector of
outputs. For a dense ReLU layer:

```
z = W · a_in + b          (the "pre-activation": a matrix-vector product)
a_out = ReLU(z)           (elementwise: ReLU(v) = max(0, v))
```

`W` is the weight matrix, `b` the bias. The **execution trace** is the list of
all activation vectors, layer by layer, for one particular input.

In our main test network (`mlp_mnist_full`), layer 1 takes 784 numbers (an MNIST
image) and produces 512 numbers. So `W` is 512×784.

### Notation used throughout

To avoid carrying the bias around separately, we fold it in. Define the
**augmented matrix** and **augmented input**:

```
A = [ W | b ]     shape N × (M+1)
x = [ a_in ; 1 ]  length M+1
```

Then `A · x = W·a_in + b·1 = z`, exactly as before. For our layer 1:
`N = 512` (neurons in this layer), `M = 784` (inputs), so `A` is 512×785.

Throughout: `z` is the true pre-activation, `a` is the **claimed** output the
prover sends. The layer is honest exactly when `a = ReLU(A·x)`.

---

## 2. The original protocol, and why it's cheap

The prover commits to the trace (see §4 for what "commit" means), then the
verifier picks a **random path** from an output neuron back down to the input,
and re-checks the local relation at each node on that path:

```
is  a_j  ==  ReLU( Σ_i w_ij · a_i )  ?
```

Checking one node costs one row of `W`. A path through a 3-layer network checks
3 nodes and needs 3 weight rows — milliseconds, versus minutes for a zkSNARK.

**The guarantee this gives.** If the prover swaps in a *completely different
model*, nearly every node in the trace is inconsistent, so a random path hits a
bad node almost surely. The scheme genuinely works against that adversary — we
reproduced 100% detection against three substitute models.

---

## 3. The attack, and why no sampling rule fixes it

Overwrite **one** activation and re-propagate everything above it using the real
weights. Now the trace is inconsistent at *exactly one node*. The verifier
catches it only if its random path happens to pass through that one node:
probability `1/N` — about 0.2% for a 512-wide layer.

Made trigger-conditional, this is a backdoor the protocol certifies as correct.

The obvious response is "sample smarter — look where it matters." Phase 0 of this
revision tested three structurally different ways of doing that. All three fail
against an adversary who knows the rule (Kerckhoffs's principle: assume the
attacker has read your source code). The details are in `phase0_summary.md`; the
short version is a dichotomy:

- a rule that ignores the claimed trace is capped at `1/|U|` and is usually
  *worse* than uniform sampling;
- a rule that uses the claimed activation values has a blind spot the adversary
  can aim for — most damningly, weighting by `|w·a|` assigns weight **zero** to a
  claimed activation of zero, so zeroing neurons is completely invisible.

**The lesson:** you cannot fix this by choosing *which* nodes to check. You have
to change *what a single check verifies*. That is what the rest of this document
builds.

---

## 4. Prerequisite: hash functions and Merkle trees

### 4.1 Hash functions

A cryptographic hash function (we use SHA-256) maps any input to a fixed 32-byte
output, with two properties that matter here:

- **Deterministic**: same input → same output, always.
- **Collision-resistant**: nobody can find two different inputs with the same
  output. (Not "impossible" in a mathematical sense — just far beyond any
  computer's reach.)

So a hash is a short, unforgeable *fingerprint* of data.

### 4.2 The commitment problem

Suppose I want to promise you I'm using a particular 512×785 weight matrix,
without sending you all 400,000 numbers. I want a short fingerprint such that:

- **Binding**: having published the fingerprint, I cannot later pretend a
  different matrix was the one I meant.
- **Openable**: I can later reveal *part* of the data and prove to you that part
  really is what the fingerprint covered.

Hashing the whole matrix gives binding but not selective opening — to check a
single entry you'd have to be sent the whole matrix. A **Merkle tree** fixes
exactly that.

### 4.3 Merkle trees

Take your list of items. Hash each one — these are the **leaves**. Then hash
each adjacent *pair* of leaves to get the next level up. Repeat until one hash
remains: the **root**.

With 8 items:

```
                      ROOT = H(H12, H34)
                     /                  \
           H12 = H(H1,H2)            H34 = H(H3,H4)
           /        \                 /        \
      H1=H(L1,L2) H2=H(L3,L4)  H3=H(L5,L6)  H4=H(L7,L8)
        /   \        /   \        /   \        /   \
      d1    d2     d3    d4     d5    d6     d7    d8     <- the 8 data items
```

The root is 32 bytes and is published. That's the **commitment**.

**Opening one item.** To prove `d5` is genuinely the 5th item, send `d5` plus
the hashes needed to recompute the root along its path: `H(d6)`, then `H4`, then
`H12`. The verifier hashes `d5`, combines it with those siblings step by step,
and checks the result equals the published root. That's `log₂(n)` hashes — 3
here, ~11 for 1570 items — instead of the whole list.

If the prover tried to substitute a different `d5`, the recomputed root would
differ, and matching the original root would require a hash collision. That's
the binding property.

This repository already had a Merkle implementation (`commitments/merkle.py`),
written for the original protocol. The defence reuses it unchanged — **no new
cryptographic primitive is introduced anywhere in this construction.**

---

## 5. Prerequisite: exact arithmetic

### 5.1 Why floating point is a problem

Neural networks use floating-point numbers. Floating-point addition is not
associative: `(a+b)+c` can differ from `a+(b+c)` in the last bits. The prover
computes a layer as one big matrix product; the verifier re-checks a single
neuron as one dot product. Different summation orders → slightly different
answers, even when both are honest.

The original protocol had to paper over this with a **tolerance**: accept if the
values agree to within `1e-4`. `SPEC_NOTES.md` §2 documents this in detail.

That tolerance is itself a security hole: it hands the adversary a free budget.
Any perturbation smaller than `1e-4` is invisible *by construction*.

### 5.2 Fixed-point numbers

Replace floats with integers by choosing a **scale** — a power of two — and
representing the real number `v` as the integer `round(v · 2^f)`.

With `f = 10` (scale 1024): `0.5 → 512`, `3.25 → 3328`, `-0.1 → -102`.

Integer arithmetic is exact and order-independent. Prover and verifier always
get bit-identical results, so **the tolerance disappears entirely** — honest
completeness becomes exactly 1, not "1 up to `1e-4`".

Note what this means: the committed model *is* the fixed-point model. We are not
approximating a float model; we define the thing being proven to be the integer
network. This is in the same spirit as the paper's own 8-bit quantisation
setting (§A).

Scales multiply: if `a_in` and `W` are both at scale `2^f`, then `z = A·x` comes
out at scale `2^(2f)`. Since ReLU only cares about sign, `ReLU` at the larger
scale is still just `max(0, ·)`.

### 5.3 Modular arithmetic and finite fields

The construction needs randomness and exact equality over a *finite* set of
values. So we work modulo a prime `p`: the numbers are `{0, 1, ..., p-1}`, and
every arithmetic operation is followed by "take the remainder on division by
`p`". This set with `+`, `−`, `×` (and division by anything nonzero) is called a
**finite field**, written `F_p`.

We use `p = 67108859`, which is `2^26 − 5` and the largest prime below `2^26`.
(Chosen so numpy can multiply two field elements in 64-bit integers without
overflow: products stay below `2^52`, leaving room to sum thousands of them.)

**Representing negatives.** There are no negative numbers in `{0,...,p-1}`, so
by convention a value in the top half means a negative: `v` is read as `v − p`
when `v > p/2`. So `p−5` *means* `−5`. Anything genuinely in `(−p/2, p/2)`
round-trips exactly.

For our layer 1, pre-activations reach about `±2·10^5`, against a half-field of
`3.4·10^7` — over 100x of headroom, so nothing wraps around.

---

## 6. Prerequisite: the random linear combination trick

This is the single most important idea in the construction, and it's elementary.

**Problem.** I claim 512 numbers `e_1, ..., e_512` are all zero. You want to
check that, but only want to look at *one* number.

**Naive approach**: check their sum. Bad — the adversary makes them cancel:
`+7` and `−7` sum to zero.

**The trick**: you pick 512 *random* coefficients `s_1, ..., s_512` from the
field, and check the single number

```
S = s_1·e_1 + s_2·e_2 + ... + s_512·e_512
```

**Claim**: if all `e_j = 0` then `S = 0` always. If even one `e_j ≠ 0`, then
`S = 0` with probability only `1/p`.

**Why**: suppose `e_k ≠ 0`. Fix all the other coefficients however you like. Then
`S = s_k·e_k + C` for a constant `C`. As `s_k` ranges uniformly over the field,
`s_k·e_k` also ranges uniformly (multiplying by a nonzero field element is a
bijection), so `S` is uniform — and hits zero with probability exactly `1/p`.

With `p ≈ 6.7·10^7` that's a 1-in-67-million chance. One number replaces 512
checks.

**Crucially, the coefficients must be chosen *after* the adversary is committed
to the `e_j`.** Otherwise it just picks values that cancel against the
coefficients it can see. Remember this — §9 is about exactly this failure mode.

---

## 7. Prerequisite: polynomials and Reed–Solomon codes

The random-combination trick handles values the verifier already has. But we
also need the verifier to be sure about a quantity computed from `A` — a matrix
it does *not* have. That needs one more tool.

### 7.1 Polynomials over a field

A polynomial of degree less than `k` is determined by `k` coefficients:

```
P(X) = u_0 + u_1·X + u_2·X² + ... + u_{k-1}·X^{k-1}
```

The key classical fact: **two distinct polynomials of degree `< k` can agree on
at most `k−1` points.** (Their difference is a nonzero polynomial of degree
`< k`, and such a polynomial has at most `k−1` roots.)

### 7.2 Reed–Solomon encoding

Take your `k` data values, treat them as the coefficients of `P`, and **evaluate
`P` at `n` different points** (we use `X = 1, 2, ..., n`). The resulting list of
`n` values is the **codeword**.

We use `n = 2k` ("rate 1/2"): the codeword is twice as long as the data.

**The distance property.** Two different data vectors give two different
polynomials, which agree on at most `k−1` of the `n` points. So their codewords
**differ in at least `n − k + 1 = k + 1` positions** — more than *half* of them.

That is the whole point: a wrong codeword can't be *slightly* wrong. It must
differ almost everywhere. So **checking one random position catches a wrong
codeword with probability > 1/2**, and checking `t` independent positions
catches it except with probability `2^−t`. We use `t = 24`, so roughly 1 in 17
million.

Tiny worked example, `k=2`, `n=4`, over `F_11`, data `u = (3, 5)`, i.e.
`P(X) = 3 + 5X`:

```
P(1)=8   P(2)=13 mod 11=2   P(3)=18 mod 11=7   P(4)=23 mod 11=1
codeword = (8, 2, 7, 1)
```

Change the data to `(3, 6)` and the codeword becomes `(9, 4, 10, 5)` — all four
positions differ. Any single spot-check catches it.

### 7.3 Linearity — the property that makes everything work

Evaluation is linear in the coefficients, so encoding is too:

```
Enc(α·u + β·v)  =  α·Enc(u) + β·Enc(v)
```

Therefore, if we encode **each row** of `A` and stack the results into a matrix
`E`, then for any vector `χ`:

```
χ^T · E  =  Enc( χ^T · A )
```

*Combining rows and then encoding gives the same answer as encoding and then
combining.* This is what lets a verifier who has never seen `A` check a claim
about `χ^T·A`.

---

## 8. Prerequisite: Fiat–Shamir

The trick in §6 needs randomness the prover cannot predict. In an interactive
protocol the verifier would just send random numbers. To make the proof
non-interactive (one message), we use the **Fiat–Shamir transform**: derive the
"random" challenges by hashing everything the prover has committed to so far.

```
challenge = SHA-256( transcript of everything committed so far )
```

Since the hash is unpredictable, the prover can't steer the challenge — *provided
the thing being challenged is already in the transcript before the challenge is
derived*. Getting that ordering wrong breaks the scheme completely. §9 is a
worked example of exactly that, from this project.

---

## 9. The construction

### 9.1 Restating "this layer is correct" as algebra

The layer is honest iff `a = ReLU(A·x)`. Writing `z = A·x`, that is equivalent
to these three conditions holding for every neuron `j`:

| | Condition | Meaning |
|---|---|---|
| **(1)** | `a_j ≥ 0` | ReLU never outputs a negative |
| **(2)** | `a_j · (a_j − z_j) = 0` | either `a_j = 0`, or `a_j = z_j` |
| **(3)** | `z_j ≤ 0` wherever `a_j = 0` | if the output is zero, the input really was negative |

Why these three are exactly right: (2) says each coordinate takes one of the two
ReLU branches. Where `a_j ≠ 0`, (2) forces `a_j = z_j` and (1) forces it
positive — the correct branch. Where `a_j = 0`, (2) says nothing at all, and (3)
supplies the missing requirement.

**Condition (1) is free.** The claimed activations `a` are in the clear (the
protocol already reveals a whole layer's activations whenever it touches that
layer — `C_trc` commits one leaf *per layer*, per `SPEC_NOTES.md` §5). The
verifier just looks at them.

### 9.2 The trap: naive batching is silently broken

The tempting move is: batch condition (2) over the whole layer with the §6 trick
and call it done. **That is wrong, and wrong in precisely the way this entire
project is about.**

Look at (2) again: `a_j·(a_j − z_j) = 0`. When `a_j = 0` this reads `0 · (0 − z_j) = 0`,
which is true **no matter what `z_j` is**. The constraint is *vacuous exactly
where the claimed activation is zero.*

So an adversary that zeroes out neurons satisfies (2) perfectly. This is
Theorem 3's zero blind spot reappearing in algebraic dress — the very same
attack that reduces contribution-weighted sampling to detection `0.000000` would
walk straight through a naive batched check.

Condition (3) exists to close this, and it is the part of the construction that
required actual thought rather than assembly.

### 9.3 The sign witness

Let `Z = { j : a_j = 0 }` — the **zero set**, which the verifier can compute
itself from the claimed `a`, so the prover cannot lie about *which* set it is.

The prover must supply, in the clear, a witness vector `y` indexed by `Z`,
claiming `y_j = −z_j`. Then condition (3) splits into two parts:

- `y_j ≥ 0` — checked **free**, by looking at the plaintext integer;
- `z_j + y_j = 0` — an algebraic claim about `z`, batched via §6.

An honest prover always can do this: where `a_j = 0`, the truth is `z_j ≤ 0`, so
`y_j = −z_j ≥ 0`. A cheating prover that zeroed a live neuron (true `z_j > 0`)
would have to publish `y_j = −z_j < 0`, which fails the free check instantly.
Its only alternative is to publish a *different*, non-negative `y_j` — and then
the algebraic check catches it.

### 9.4 Folding everything into one opening

Apply §6 to (2), with random coefficients `s`:

```
Σ_j s_j·a_j·(a_j − z_j) = 0
⟺  Σ_j (s_j·a_j)·z_j  =  Σ_j s_j·a_j²
```

The right side the verifier computes itself from `a`. The left side is a linear
function of `z`.

Apply §6 to (3), with random coefficients `t` over `Z`:

```
Σ_{j∈Z} t_j·(z_j + y_j) = 0
⟺  Σ_{j∈Z} t_j·z_j  =  −Σ_{j∈Z} t_j·y_j
```

Again: right side known to the verifier, left side linear in `z`.

Now combine the two with two more random scalars `α, β`, and use `z = A·x`:

```
χ  :=  α·(s ⊙ a)  +  β·t_Z            (⊙ = elementwise product)

⟨χ, A·x⟩  =  α·Σ_j s_j·a_j²  −  β·⟨t_Z, y⟩
```

And since `⟨χ, A·x⟩ = ⟨χ^T·A, x⟩`, everything reduces to:

> the verifier needs **one** vector: `u := χ^T·A`, of length `M+1`.
>
> Given `u`, it checks the single scalar identity `⟨u, x⟩ = known value`.

Note what the verifier never learns: `z` itself. It only ever sees one linear
functional of it.

### 9.5 Getting `u` honestly: the weight commitment

The prover just sends `u`. Why should the verifier believe `u = χ^T·A` for the
committed `A`? This is where §7 pays off.

**At commit time** (once, by the model owner, exactly where the paper's
`CommitToModel` already sits):

1. Reed–Solomon encode **each row** of `A`, from length `M+1 = 785` to
   `n = 1570`. Call the encoded matrix `E` (512 × 1570).
2. Merkle-commit the **columns** of `E` — 1570 leaves, each a column of 512
   field elements. Publish the root.

**At verification time:**

1. The prover sends `u`.
2. The verifier encodes `u` *itself* — it's public data, length 785 → 1570.
3. It picks `t = 24` random column indices, and the prover opens those columns
   (each: 512 values plus a Merkle path).
4. For each opened column `c`, the verifier checks

   ```
   ⟨χ, E[:, c]⟩  ==  Enc(u)[c]
   ```

By the linearity of §7.3, `χ^T·E = Enc(χ^T·A)`. So if `u` is the true `χ^T·A`,
every column matches. If `u` is anything else, `Enc(u)` differs from
`Enc(χ^T·A)` in more than half of the 1570 positions, so each sampled column
catches it with probability > 1/2 — and 24 of them catch it except with
probability `2^−24`.

**Why no "proximity test" is needed.** Schemes of this shape (Ligero, Brakedown)
normally also need to prove the *committed matrix itself* is well-formed,
because a malicious committer could commit garbage. We don't need that here,
because in this protocol's threat model `C_M` is an **honest** commitment to the
intended model — the entire premise is a prover that commits a benign model and
then lies about the *trace*. A malicious committer would need the extra
argument; that's a real extension, and we don't claim it.

### 9.6 The ordering bug — a real one, found late

Fiat–Shamir (§8) says: challenges must be derived from a transcript that already
contains everything they're challenging. My first implementation derived the
column indices from `(digest, x, a, y)` — **but not from `u`**.

That is exploitable, and not theoretically. The prover knows `digest, x, a, y`
(it chose most of them), so it can compute the sampled columns *before*
committing to `u`. It then needs a `u` satisfying:

- 24 equations, one per sampled column (`Enc(u)[c]` must equal a known value);
- 1 equation, the final scalar identity.

That's **25 linear equations in 785 unknowns** — wildly underdetermined. Solve
the system with Gaussian elimination mod `p`, and you get a `u` that is *not*
`χ^T·A` but passes every check.

I implemented this attack against my own construction and it produced an
accepting proof for a tampered trace. The measurement script had never caught
it, because that script only ever runs the *honest* prover algorithm on tampered
traces — it never modelled a prover that cheats on the proof itself.

**The fix** is to make the transcript two-phase:

```
round 1:  s, t, α, β   ←  H( digest ‖ x ‖ a ‖ y )
          prover computes u = χ^T·A
round 2:  columns      ←  H( digest ‖ x ‖ a ‖ y ‖ u )
```

Now the attack is circular: committing to the forged `u` re-randomises the very
columns it was solved against. Verified — the same forgery is rejected after the
fix, and honest traces still pass. It's locked in as a regression test
(`test_adaptive_transcript_forgery_is_rejected`).

The lesson generalises: **testing a cryptographic construction by running the
honest prover on bad inputs proves almost nothing.** You have to model a prover
that cheats on the proof.

### 9.7 The whole protocol, end to end

Commit time (once):
- quantise `A = [W|b]` to fixed point; RS-encode its rows; Merkle-commit the
  columns; publish the root.

Per query, prover sends: `a` (already in the protocol), `y`, `u`, and 24 column
openings.

Verifier:
1. `a ≥ 0`? *(free, plaintext)*
2. zero set of `a` matches the claimed index set? *(free)*
3. `y ≥ 0`? *(free, plaintext)*
4. derive `s, t, α, β`; form `χ`
5. derive the 24 columns **from `u` as well**; Merkle-verify each opening;
   check `⟨χ, E[:,c]⟩ = Enc(u)[c]`
6. check `⟨u, x⟩ = α·Σ s_j a_j² − β·⟨t_Z, y⟩`

Accept iff all pass.

---

## 10. Why it's sound

Suppose the prover submits `a ≠ ReLU(A·x)` and let `z = A·x` be the truth.

- If some `a_j < 0`: caught by check 1, deterministically.
- **Case A — some `a_j > 0` with `a_j ≠ z_j`.** Then `e_j := a_j(a_j − z_j) ≠ 0`,
  so the batched sum in (2) is a nonzero linear form in the random `s`. It
  vanishes with probability `1/p`.
- **Case B — some `a_j = 0` with `z_j > 0`.** Then `j ∈ Z`, and the honest
  witness `−z_j` is negative, failing check 3. To survive, the prover must send
  some `y_j ≥ 0 ≠ −z_j`, making `z_j + y_j ≠ 0` — a nonzero linear form in the
  random `t`, vanishing with probability `1/p`.
- **Lying about `u`** is caught by the column checks except with probability
  `2^−t`.

Every case is covered, so the per-challenge soundness error is about
`2/p + 2^−t`, roughly `3·10^−8 + 6·10^−8`. Note what is *absent* from that
expression: the layer width `N`, the position of the tamper, and the value
claimed. That is the whole difference from sampling.

**But that figure is only the honest-challenge error, and with Fiat–Shamir it
is not the security level.** Because the prover derives its own challenges by
hashing (§8), it can *grind*: re-randomise part of its message, re-hash, and
retry until the challenge happens to be favourable. Each retry succeeds with
probability ~`2^−24`, so ~`2^24` retries suffice — cheap. The effective
security is therefore ~24 bits, not the ~80–128 one would want.

**This is now fixed.** `VerifierRandomness` draws the challenges from the
verifier's own CSPRNG instead of hashing the prover's message — exactly what the
paper's own `RandPathTest` does, where the verifier picks the path. A cheating
prover then gets *one* attempt per interaction, and every failure is a visible
rejection rather than a silent offline retry, so the soundness error really is
the `2/p + 2^−t` above.

Ordering still matters even with true randomness: the column challenge is
withheld until after the prover has committed to `u`, otherwise the prover could
solve for a `u` matching on exactly the columns it knows will be checked (§9.6).

Verifier-drawn challenges are the default in `scripts/run_batched_network.py`;
Fiat–Shamir stays available (`--challenges fiat-shamir`) for the
non-interactive setting, where the ~24-bit caveat does apply. A test measures
the difference directly: re-randomising the prover's message 200 times yields
200 independent challenge draws under Fiat–Shamir and exactly 1 under
verifier-drawn randomness.

---

## 11. What it achieves, measured

Against this repository's own attack planners — not synthetic tampering —
on `mlp_mnist_full`, layer 1 (width 512), 40 queries:

| Attack | **batched check** | uniform sampling | contribution weighting |
|---|---|---|---|
| naive single-neuron flip | **1.000000** | 0.001953 | 0.112636 |
| stealthy envelope-confined | **1.000000** | 0.004958 | 0.150712 |
| **zero-hiding backdoor** | **1.000000** | 0.023730 | **0.000000** |

Honest completeness: **exactly** `1.000000` (40/40) — no tolerance.

The third row is the point of the whole exercise: that is the attack nothing in
Phase 0 could stop, the one that drives contribution weighting to precisely zero.

At 8x the width (4096), detection stays at `1.000000` for both attacks while
uniform sampling *degrades* from `0.0019` to `0.00024` — the `1/N` ceiling
visible in one row and absent in the other, exactly as §10 predicts.

---

## 12. What it costs

Proof size, whole network: **106 kB** for the 784-512-256-10 model (62.0 + 35.1
+ 9.4 kB across the three layers) against a 2.04 MB model — about **20x smaller
than downloading the weights**. For 784-4096-2048-10 it is 670 kB against a
44.35 MB model, **68x smaller**: the ratio *improves* with scale. Prover 5.0 ms,
verifier 5.0 ms. Commitment build, one-off at commit time: 0.3 s and 3.9 s.

The dominant term is the 24 opened columns, at one field element per neuron
each, so proof size grows as `O(t·N)`.

Be careful with the comparison. Once *every* layer is checked in full there is
nothing left to sample, so this is **no longer a sampling protocol** and the
"visit `O(depth)` layers" claim does not carry over — it does not apply. The
honest reference points are the two things this construction sits between:
downloading the model (which it beats by 20–68x) and a full SNARK.

One reduction listed here as future work has since been done: field elements
fit in 26 bits and are now serialised as 4-byte integers, halving every column
opening (already reflected above). Dropping the column count is *not* a saving
that was available under Fiat-Shamir — see §10 — though with verifier-drawn
challenges the parameters now mean what they say.

---

## 13. Limits — what this does not do

- **Chaining is implemented** (it was not, in the first version). The verifier
  recomputes each layer's input from the previous layer's verified output using
  a public rounding rule; no range argument is needed because the activations
  are revealed anyway. Identity/logit layers are supported. At scale `2^8` the
  honest pre-activations peak at `[771265, 813725, 2492865]` over the full 10,000-image test set against a field
  half-width of `33554429`.
- **Soundness parameters**: fixed by verifier-drawn challenges (§10). The
  Fiat–Shamir mode is still offered and still carries the ~24-bit caveat.
- **Honest committer assumed** (§9.5) — matches the paper's threat model, but it
  is an assumption.
- **Dense ReLU layers only.** Convolutions share weights and need a different
  matrix layout; max-pooling isn't an affine-plus-ReLU relation at all.
- **Prover cost** is 5.0 ms for the whole small network and 5.0 ms to verify
  (measured; an earlier draft said this was not benchmarked).
- **It does not make a backdoored model safe.** It guarantees the committed model
  was executed faithfully. If the *committed model itself* is backdoored, no
  inference-time check can help — that is a property of training, and it remains
  the honest limit of this entire line of work.

---

## 14. Where the code is

| Thing | File |
|---|---|
| The construction | `src/pvi/protocol/batched.py` |
| Merkle trees (pre-existing, reused) | `src/pvi/commitments/merkle.py` |
| Tests, including the forgery regression | `tests/test_batched.py` |
| Measurement against the real attacks | `scripts/run_batched_defence.py` |
| Results | `artifacts/results/batched_defence.json` |

Reading order in `batched.py`: `fixed_point_layer` (§5) → `BatchedWeightCommitment`
(§7, §9.5) → `_derive_folding` / `_derive_columns` (§8, §9.6) → `prove_layer` and
`verify_layer` (§9.7).
