# Certified but compromised: attacking sampling-based proofs of inference

Code for a course project on *Towards Verifiable AI with Lightweight Cryptographic
Proofs of Inference* by Anchuri, Campanelli, Cesaretti, Gennaro, Jois, Kayman and
Ozdemir (SaTML 2026; [ePrint 2026/541](https://eprint.iacr.org/2026/541)).

The paper replaces zkSNARK-based verifiable inference with something far cheaper.
The prover commits to the execution trace of a forward pass using a Merkle vector
commitment and opens only the entries along a few randomly sampled output-to-input
paths; the verifier re-checks the local relation
`a_j = φ(Σ_{i∈G_j} w_ij a_i)` at each node on the path against the committed weights
of the intended model. Proving drops from minutes to milliseconds.

No implementation was released, so this is a from-scratch reimplementation, followed
by an attack on it and an analysis of whether the defence the paper suggests can
work.

**The short version.** The protocol does what it claims against the adversary it
models — a prover that swaps in a *different model*. It does not survive an
adversary that runs the *right* model and changes one number in the trace. Such a
trace is locally consistent everywhere except at a single node, so the verifier
rejects only if its path happens to hit that node: probability `1/N` in a layer of
width `N`. Made trigger-conditional, this is a backdoor that the protocol certifies
as correct, with a clean-accuracy gap of exactly zero. The obvious fix — sample
where it matters rather than uniformly — provably cannot help, and the most natural
implementable version makes things strictly worse.

---

## Quick start

```bash
python3 -m venv --without-pip /tmp/erel_venv
source /tmp/erel_venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
```

```bash
.venv/bin/pip install -e .
```

On Linux/macOS (including the SLURM cluster) use `.venv/bin/pip`. Everything runs on
CPU; no GPU is required anywhere.

From the `code/` directory:

```bash
python -m pytest tests/ -q
```

```bash
python scripts/train_models.py && python scripts/run_step1.py && python scripts/run_attack.py && python scripts/run_defence.py
```

Training takes ~15 minutes on 4 CPU threads and caches to `artifacts/models/`; it is
a no-op on re-runs unless `--force` is given. The three experiment scripts take
about 5, 2.5 and 2.5 minutes and write JSON to `artifacts/results/` plus figures to
`artifacts/figures/`.

---

## Layout

```
code/
  SPEC_NOTES.md            How we read the paper where it is ambiguous, and why
  DEFENCE_NOTES.md         The Step-4 theory: why weighted sampling cannot work
  src/pvi/
    commitments/merkle.py  Merkle vector commitment (Appendix B, Section 8.1)
    nn/architecture.py     Layered DAG: parent sets G_j, local relations, weight rows
    nn/network.py          Execution traces and EvalTrace (Definition 5)
    nn/gradients.py        Output sensitivity, used by both attacker and defender
    protocol/sampling.py   Path samplers; RandPathTest is the uniform one (Figure 3)
    protocol/prover.py     Prove1 / Prove2 (Definition 4)
    protocol/verifier.py   Verify (Figure 4)
    experiments/analysis.py  Exact acceptance probability for any trace and sampler
    experiments/separation.py  Equation (1), verifier residuals, JS divergence
    experiments/estimation.py  Appendix I, Algorithms 1 and 2
    attacks/baselines.py   The paper's own attacks (Sections 2, 7.2, E, F)
    attacks/tamper.py      Single-neuron and stealthy trace tampering
    attacks/backdoor.py    The trigger-conditional adversary
    defences/sampling.py   Importance-weighted samplers
    defences/adaptive.py   The adversary's reply to a known sampler
  scripts/                 Four runnable entry points
  tests/                   159 tests
```

The protocol layer never imports PyTorch. A model is an `Architecture` plus NumPy
arrays; PyTorch appears only to fit them, and a test asserts the two evaluation paths
agree.

---

## Step 1 — Reproduction

Four substitution settings, each reproducing a pairing the paper uses. Every row is
150 MNIST queries × 20 fresh challenges = 3000 protocol executions, single path.

| Setup | Paper | Layer widths | Prediction agreement | Completeness | Detection |
|---|---|---|---|---|---|
| Different tasks, shared class | §6.2 | 784–512–256–5 | 0.9867 | **1.0000** | **1.0000** |
| Same task, disjoint data | App. G.1 | 784–512–256–10 | 0.9800 | **1.0000** | **1.0000** |
| 8-bit quantised substitute | §A | 784–512–256–10 | 1.0000 | **1.0000** | **0.9980** |
| Disjoint data, CNN | App. G.1 | 784–12544–…–10 | 0.9800 | **1.0000** | **1.0000** |

Cost per query on one core: prover 0.5–1.7 ms, verifier 0.3–1.2 ms, proof 13 kB
(MLP) or 104 kB (CNN), 3–4 weight rows opened. The scheme works: even an 8-bit
quantisation agreeing with `M` on **every** test prediction is caught 99.8% of the
time.

Three findings from the reproduction itself:

**The idealised exact-equality test is not implementable.** Figure 3 writes the local
check as `ã_j = φ(Σ w_ij ã_i)`. Taken literally, an honest prover is accepted only
**17–32%** of the time (per setup), because the prover evaluates a layer as one
matrix product while the verifier re-evaluates a single neuron as a dot product, and
the two summation orders round differently: 65% of checked nodes agree bit-for-bit,
the rest differ by up to `2.9e-6`. Definition 1 requires acceptance probability
exactly 1, so the idealised test fails its own correctness condition.

The paper is aware of this and prescribes a tolerance (Appendix E.3, "`1e-4`"); our
measurements say that choice is sound, with 52× headroom over the worst observed
honest residual. The point is that the tolerance is *mandatory* rather than a
convenience — and that it hands an adversary a perturbation budget below `1e-4`
which is invisible by construction.

How much of the disagreement is inherent depends on implementation care: a verifier
that matches the prover's arithmetic precision agrees far more often than one that
does not (we had this wrong at first — see the note in `SPEC_NOTES.md` §2). Full
bit-exactness would require the protocol to mandate a canonical evaluation order, or
fixed-point arithmetic, which is what field-based SNARK constructions get for free.

**Equation (1) is identically zero at layer 1 for arithmetic reasons.** Both its
terms use `M`'s weights and differ only in the layer-`l−1` activations — which at
`l = 1` are the shared query. The paper reads near-zero early-layer separation as
evidence that "both models learned comparable low-level features" (§6.2.2); for the
first layer that reading is unavailable. The verifier's *actual* residual there is
large (mean 0.40).

**The paper's own Appendix I estimator fails on two of four setups.** Its JS > 0.05
layer filter empties `L_valid` for the disjoint-data (max JS 0.017) and quantised
(0.0008) setups, so Theorem 1's bound cannot be instantiated — while detection is
100% and 99.8%. JS on pooled marginals asks "same distribution?"; `RandPathTest`
asks "same value on *this* input?". Verified not to be a pooling artefact.

---

## Step 2 — The attack

**The structural fact.** With the input anchored and every local relation satisfied,
the trace is *uniquely* `EvalTrace(M, qry)`. So any trace claiming another output
must be locally inconsistent somewhere — but the adversary chooses *how many* nodes
and *where*. Overwrite one activation and re-propagate the layers above it using
`M`'s real weights: everything above is consistent by construction, everything below
is untouched, and exactly one node is wrong.

The paper's own attacks minimise the *magnitude* of the inconsistency. That is the
wrong objective — the verifier thresholds per-node residuals along a path, and never
looks at total separation.

| Attack | Output changed | Inconsistent nodes | Acceptance |
|---|---|---|---|
| Inverse transform (App. E) | yes | 594 / 778 | 0.0000 |
| Logit-swap injection (App. F) | yes | 256 / 778 | 0.0000 |
| Gradient reconstruction (§7.2) | yes | 778 / 778 | 0.0000 |
| Substitute model (Def. 1) | yes | 570 / 778 | 0.0000 |
| **Single-neuron tamper, layer 2 (ours)** | yes | **1 / 778** | **0.9961** |
| **Single-neuron tamper, layer 1 (ours)** | yes | **1 / 778** | **0.9980** |

Gradient reconstruction achieves the *smallest* separation of any attack we ran
(0.30 against 0.89) and is still caught every time, because it perturbs everything.

**As a backdoor.** Serving the honest trace on clean inputs and tampering only on
triggered ones (a 3×3 bright patch):

- served output identical to the committed model on **100%** of clean queries;
- clean-accuracy gap **+0.000000** — the served trace *is* the honest trace, bit for bit;
- attack success rate **1.0000** on triggered inputs;
- verifier acceptance **1.0000** on clean queries, **0.9957** on triggered ones
  (exact prediction 0.9980).

---

## Step 3 — Analysis

**Detection is exactly `1 − (1 − 1/N)^m`.** Confirmed against the running protocol:
layer 1 (width 512) gives 0.9980 predicted, 0.9978 measured over 4000 runs; layer 2
(width 256) gives 0.9961. Attacking the *wider* layer is better on both axes — it
also needs a smaller change (mean |Δ| 7.7 versus 37.5).

**The opening budget buys little.** Going from 1 path to 250 — a 250-fold increase in
proof size and verifier work — moves acceptance only from 0.998 to 0.613. At that
point the verifier opens 750 weight rows against a layer of 512.

**Locality trades support for evasion**, as predicted: spreading over `k` neurons
gives acceptance ≈ `1 − k/N` (measured 0.998 at `k=1`, 0.957 at measured support 22).

**Repeated queries do not amplify.** The paper's amplification argument assumes the
auditor's queries carry the trigger. Writing `ρ` for that fraction, the chance of
ever being caught over `n` queries is `1 − (1 − ρ(1 − (1−1/N)^m))^n`. At `ρ = 10⁻⁴`,
a million audited queries reach 0.177; at `ρ = 0`, no number of queries helps.

**Stealth.** The naive single-neuron tamper needs an activation of ~28 where the
natural maximum across the whole layer is 8.2 — so a verifier that also range-checked
opened activations would catch it. That defence is not in the paper, but this attack
motivates it. The stealthy planner removes the signature by confining every forged
value to its own neuron's envelope, at the cost of a few more nodes:

| Envelope cap | Untargeted ASR | support `k` | Acceptance |
|---|---|---|---|
| p100 | 0.983 | 2.9 | 0.9943 |
| p99 | 0.983 | 4.2 | 0.9919 |
| p95 | 0.983 | 5.6 | 0.9890 |

---

## Step 4 — Why "sample where it matters" cannot fix it

Full argument in [`DEFENCE_NOTES.md`](DEFENCE_NOTES.md); the headline is that the
suggestion in §5.2 fails, and provably so for the defences a verifier can run.

**A verifier cannot compute influence.** Weighting neuron `i` by `|∂y/∂a_i|` needs
every weight above `i`. A verifier able to form those gradients already holds the
model and could just run the inference. We implement it anyway as an upper bound.

**Theorem (uniform is minimax-optimal).** For any sampler whose choice is independent
of the claimed trace, inducing visit distribution `q`, an adversary that knows `q`
tampers at `argmin_{v∈U} q(v)`, so detection is `min_{v∈U} q(v) ≤ 1/|U|`, with
equality only if `q` is uniform. Measured at layer 1 (`|U| = 454` of 512 neurons
admit a flip, bound `1/|U| = 0.00220`):

| Sampler | `min_v q(v)` over `U` | vs naive attack | vs adaptive attack |
|---|---|---|---|
| uniform | 0.001953 | 0.00195 | 0.00195 |
| static importance | 0.000994 | 0.00437 | **0.00105** |
| gradient saliency *(not implementable)* | 0.000002 | 0.00752 | **0.00006** |

Both look like improvements against an adversary that ignores them; both fall *below*
uniform once the adversary reads the rule.

**Theorem (zero blind spot).** Contribution weighting — step to parent `i` with
probability ∝ `|w_ij · ã_i|`, the only implementable version and by far the most
effective against the naive attack (detection 0.126, a 64× improvement) — assigns
weight **zero** to a claimed activation of zero. Setting neurons to zero is therefore
invisible to it.

The adversary duly does exactly that. Measured:

- **13 of 512** neurons set to exactly zero flips the prediction;
- 100% of adaptive plans use only zeroed activations;
- detection under contribution weighting: **exactly 0.000000**, at every budget
  tested up to 500 paths (1500 weight rows). The real protocol accepted
  **2000 of 2000** runs;
- detection under plain uniform sampling for the same trace: 0.0305.

So the refinement converts a 3% chance of catching this adversary into none at all.
The value is also unremarkable: those neurons are naturally zero on **65%** of natural
inputs, since ReLU sparsity makes zero the modal activation — no envelope or
marginal-statistics check flags it either.

Mixing a uniform component of mass `ε` into the sampler restores detection `ε·|S|/N`,
so the floor is not optional; but the defence is then at best a scaled-down uniform
sampler, which the minimax theorem already bounds.

**Corollary.** A verifier deciding by conjoined local checks, opening at most `k`
nodes of a width-`N` layer by a trace-independent rule, detects a single-node tamper
with probability at most `k/|U|`. Constant detection needs `k = Ω(N)` — linear in the
layer width, which is precisely the cost the protocol exists to avoid.

**What would help**, in our view: exact-binding SNARKs (removing the gap at the
prover cost the paper set out to avoid), verifiable training or model attestation
(backdoor-freedom is a property of *training*, not of inference), and — most
promising — the paper's own refereed-delegation protocol (Appendix D), whose
bisection isolates the first disagreeing node with certainty rather than probability
`1/N`, and which our attack does not touch.

---

## Validation

Correctness rests on more than the experiments running. `experiments/analysis.py`
computes, by exact dynamic programming over the path distribution, the probability
that the verifier accepts any given trace under any sampler — assuming neither that
per-layer visits are uniform nor that checks at different layers are independent
(both false for convolutions, and deliberately false for the weighted samplers). The
test suite checks that prediction against the full protocol stack — commitments,
openings, challenge derivation, per-neuron recomputation — across dense and
convolutional networks, single-node and many-node tampering, path budgets 1/2/5, and
uniform and weighted samplers. Agreement is within Monte-Carlo error throughout.

Also tested: Merkle binding and position binding (forged values, openings replayed at
another index, truncated authentication paths, padding leaves); rejection of weight
openings taken from a different model or a different row; PyTorch↔NumPy agreement;
visit distributions against Monte Carlo for every sampler; and the two Step-4
theorems as executable assertions.

```bash
python -m pytest tests/ -q
```

## Reproducibility

Training is seeded; `EvalTrace` is bit-exactly reproducible, which is what the
commitment requires. Chunked batch evaluation (used only for reporting accuracy,
never inside the protocol) agrees to ~1e-7 rather than bit-exactly, since BLAS may
block a matrix product differently for different batch sizes.

Verifier challenges are drawn freshly on every execution, as the protocol requires,
so acceptance and detection rates are Monte-Carlo estimates and move slightly between
runs. Everything else is fixed by seed, and quantities that can be computed exactly
are. Dependencies are pinned in `requirements.txt`; MNIST downloads on first use.
