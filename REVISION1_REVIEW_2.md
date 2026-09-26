# Follow-up review of `revision1` (commit `51e2dfd`)

**Reviewed:** `origin/revision1` at `51e2dfd`, "Act on external review: retract two wrong claims, extend defence to whole network".
**Compared against:** the first review (`REVISION1_REVIEW.md`, which you also added as `code/revision1_notes/REVISION1_REVIEW.md`).

**Short version:** this is a strong revision. All three high-severity problems are genuinely fixed, and I re-verified each one independently rather than trusting the commit message. You also found and fixed a real soundness bug that the first review missed. What's left is mostly:

- a missing test and a missing script for the new whole-network functions (both provided below, already passing);
- some text that wasn't updated along with the fixes;
- one documented-but-unfixed security parameter;
- the housekeeping items you deliberately skipped, plus one bug on `main` that isn't yours.

---

## Contents

1. [Status of every item from the first review](#1-status-of-every-item-from-the-first-review)
2. [What I verified, and how](#2-what-i-verified-and-how)
3. [Credit: the column-ordering bug](#3-credit-the-column-ordering-bug)
4. [The ε-floor result, checked per query](#4-the-ε-floor-result-checked-per-query)
5. [What's still open](#5-whats-still-open)
6. [Suggested order of work](#6-suggested-order-of-work)
7. [Appendix: drop-in test and reproduction scripts](#7-appendix-drop-in-test-and-reproduction-scripts)

---

## 1. Status of every item from the first review

| First-review item | Status | Notes |
|---|---|---|
| **P1a.** Output layer rejected honest provers | ✅ Fixed | Identity-layer check added; honest logits accepted, tampered logits rejected. |
| **P1b.** Layers not linked; verifier took inputs from the prover | ✅ Fixed | `verify_network` computes every layer's input itself. Verified below, including a genuine broken-chain case. |
| **P1c/d.** "Whichever layers the path visits"; "efficiency claim untouched" | 🟡 Mostly | README and plan now say it's no longer a sampling protocol. **Four places still say the opposite** (§5.3). |
| **P2.** "Theorem 4" measured on the honest trace | ✅ Retracted and re-measured | Script rewritten correctly. The new positive result holds on every query I tried, but the headline number is an average (§4). |
| **P3.** "Targeted is 3.1× more detectable" | ✅ Retracted and corrected | Clean, clearly marked retraction. |
| **P4.** Security parameters (~24 bits with hash-derived challenges) | 🟠 Documented, not fixed | Honest disclosure. Fix sketched in §5.4. |
| **P5.** Contradictory docs, broken reference, stale tables | 🟡 Mostly | Reference fixed, README and plan updated. Stale passages remain (§5.3); tables still from different models (§5.7). |
| **P6.1** Verifier handed the prover's object holding the weights | ⬜ Not done | §5.5 |
| **P6.2** Spot-check columns could repeat | ✅ Fixed | Now a random permutation prefix. |
| **P6.3** `scale_bits` default 10 overflows once chained | ✅ Fixed | Script default is now 8. |
| **P6.4** Verifier inputs came from the prover | ✅ Fixed | In `verify_network`. |
| **P6.5** Tampered traces not re-propagated | ✅ Fixed | `fixed_point_network(tampered=…)` re-propagates. |
| **P6.6** `--only mlp_mnist_large` failed | ✅ Fixed | |
| **P6.7** Quadratic Reed–Solomon encoding cost | ⬜ Not done | Optional; fine at these sizes. |
| **P7.** Housekeeping (file modes, README quick start, pins, cluster scripts, commit messages) | ⬜ Deliberately skipped | Still needed before submission. Plus one bug on `main` that isn't yours (§5.6). |

---

## 2. What I verified, and how

Everything here was run on a clean checkout of `51e2dfd`, using `main`'s trained `mlp_mnist_full`, on CPU.

**Test suite:** all 189 tests pass (1 skipped, as on `main`), 8 of them new.

**Whole-network defence, through the public functions** (`fixed_point_network` → `prove_network` → `verify_network`, appendix script B, 10 MNIST test queries):

| Check | Your claim | My result |
|---|---|---|
| Honest full network accepted | 10/10 | 10/10, and all 10 answers correct |
| Tamper at layer 1 / 2 / 3 rejected | 10/10 each | 10/10 each |
| Broken chain rejected | 10/10 | 10/10 (stronger version: every per-layer proof is valid, but layer 2 was built on the wrong input) |
| Whole-network proof size | 188.9 kB | 189.0 kB |
| Prove / verify time | 7.4 / 9.4 ms | 5.7 / 8.4 ms (different machine) |

**The ε-floor sampler, per query:** see §4.

**The retractions:** both are accurate, and they're kept visible as retractions rather than silently edited, which is the right call.

---

## 3. Credit: the column-ordering bug

This is the most important thing in the commit, and the first review missed it entirely.

In the earlier `batched.py`, the columns the verifier spot-checks were derived from `(digest, inputs, outputs, zero set, sign witness)`, that is, **before** the prover committed to the combined vector `u`. So a cheating prover could read the columns off first and then solve for a fake `u`. It only has to match the true values on those 24 columns and satisfy the final identity: 25 linear equations in 785 unknowns, trivially solvable. You say you built this forgery and it was accepted, and the counting argument confirms it's real.

Deriving the columns from a transcript that includes `u` makes the attack circular: committing to a forged `u` changes the very columns it was solved against. `test_adaptive_transcript_forgery_is_rejected` is a good regression test for this.

This is worth a short paragraph in the report too. "Every challenge must come after the value it checks" is exactly the kind of subtle ordering error that makes these protocols hard to get right.

---

## 4. The ε-floor result, checked per query

Your rewritten `run_theorem4_check.py` now does the right thing: it measures detection on the tampered traces, and gives the attacker the better of the single-neuron attack (exact, over all of `U`) and the multi-neuron search. The headline, "11.9× better than uniform at ε = 1", is an **average over 5 queries**. The worst query is what matters for a guarantee, so I re-ran it per query on 12 queries (appendix script C):

| ε | Mean attacker-best detection | Worst query | Queries where the attacker beats uniform |
|---|---|---|---|
| 0.1 | 0.011316 (5.8× uniform) | 0.002097 (**1.07×**) | 0 / 12 |
| **1** | 0.020621 (10.6× uniform) | 0.006236 (**3.2×**) | **0 / 12** |
| 10 | 0.004255 (2.2× uniform) | 0.002848 (1.5×) | 0 / 12 |

Uniform's detection of a single-neuron tamper is `1/N = 0.001953`.

So the result survives: at ε = 1, the floor sampler beats uniform on **every** query against both attack families, by 3.2× on the worst query. At ε = 0.1 the worst query is only just above uniform, so the "0.1–10 range" should really be described as "around ε = 1".

**Suggested wording** for README Step 5 (line 306), `REVISION1_PLAN.md:76`, `DEFENCE_NOTES.md` §6 and `phase0_summary.md:29`:

> At ε = 1 the floor sampler beats uniform on every query tested (12/12) against both attack families: about 10× on average and 3× on the worst query. This has not yet been tested against the mixture attack.

**Still to do before the report leans on it** (you already list most of this):
- **The mixture attack:** zero some small-weight neurons while raising one just enough. It's the obvious way to attack a middle ε, and it's untried.
- **≥ 100 queries,** reporting the worst query rather than only the mean.
- **Width 4096.**

Until the mixture attack has been run, "not yet refuted", as `DEFENCE_NOTES.md` puts it, is exactly the right label.

---

## 5. What's still open

### 5.1 The new whole-network functions are never called

`fixed_point_network`, `prove_network` and `verify_network` aren't used by any test or script. `tests/test_batched.py` re-implements the chaining inside `_run_chain`, and no committed script produces the headline numbers (10/10, 188.9 kB, 7.4/9.4 ms). So the functions a caller would actually use are untested, and the numbers can't be reproduced from the repo.

**Fix:**
- Add `tests/test_batched_network.py` (appendix A, ready to drop in: 7 tests, all pass on `51e2dfd`).
- Commit a script that produces the whole-network numbers. Appendix B works as-is; saving it as `scripts/run_batched_network.py` and writing its results to `artifacts/results/` would do.

### 5.2 The "inter-layer" test doesn't test the chaining

`test_whole_network_rejects_an_inconsistent_interlayer_value` changes the reported layer-1 output. That changes layer 1's own transcript, so the rejection comes from **layer 1's own check**; the chaining is never exercised.

The real case is different: every layer's proof is individually valid, but layer 2's proof was built on an input that isn't `rescale(layer-1 output)`. Only the verifier recomputing the input catches that. `test_broken_chain_is_rejected` in appendix A covers it, with a positive control proving layer 2's proof does verify against the input it was built on, so the rejection genuinely comes from the chaining.

### 5.3 Text that wasn't updated with the fixes

| File:line | Says | Should say |
|---|---|---|
| `DEFENCE_NOTES.md:525–529` | "≈9x … and 'visit `O(depth)` layers' is untouched" | Contradicted two paragraphs later ("no longer a sampling protocol"). Delete, or give the whole-network figure: 189 kB, about 14× the sampling proof and 11× smaller than the model. |
| `DEFENCE_NOTES.md:586–589` (§11) | "~9x proof size, and no primitive beyond the Merkle tree … one layer at a time without rescaling" | Three stale claims: size is ~14× for the network, the "no primitive" wording was fixed elsewhere but not here, and layers are now chained. |
| `REVISION1_PLAN.md:6` | "for ~9x the proof size" | ~9× is one layer; the network is ~14×. |
| `defence_explained.md:605` | per-layer size only (115.6 kB, ≈9x) | Add the whole-network 189 kB. |
| `defence_explained.md:614` | "the protocol's actual efficiency claim … is untouched" | Same contradiction as `DEFENCE_NOTES.md:529`. |
| `defence_explained.md:618` | "`2^−16` would be ample" | True only with verifier-sent challenges. Your own §10 says so; this line contradicts it. |
| `defence_explained.md:636` | "Prover cost not benchmarked" | You now report 7.4 ms. |
| `defence_explained.md:629`, `phase4_batched_defence.md:141` | peak pre-activations `[232021, 343622, 1298689]` | Over the full 10,000-image test set at 2⁸ the peaks are `[767244, 838260, 2225729]`. Still 15× below the field's half-width, so the conclusion holds, but quote the full-set numbers. |
| `DEFENCE_NOTES.md` §10, "running hash of earlier layers" | | The code isn't cumulative, see §5.5. |
| The four "11.9x" lines | an average over 5 queries | See §4. |

### 5.4 Security parameters still ~24 bits

Correctly documented, not fixed. With hash-derived challenges, a 26-bit field and 24 spot-checks give about 24 bits against a prover that keeps re-randomising and re-hashing. The simplest fix, and one consistent with the paper's own protocol, is **verifier-sent challenges**. Sketch:

1. The prover sends every layer's claimed outputs, zero set and sign witness.
2. The verifier replies with fresh random `s, t, α, β` for every layer (e.g. from `secrets`).
3. The prover sends each layer's `u`.
4. The verifier replies with fresh column positions, drawn without replacement.
5. The prover opens those columns.

In code this is mostly splitting `prove_layer` into those phases and letting `verify_layer` take the challenges as arguments instead of deriving them. If it stays hash-derived for the submission, say plainly in the report that the implementation has ~24-bit security and why.

### 5.5 Small code nits

- **The verifier still receives the prover's commitment object.** `verify_network` passes `commitment` (which holds the full weight matrix) to `verify_layer` as `encoder`. It only reads public fields, but `main`'s design makes it *impossible* for the verifier to see the model. A small public-parameters object would restore that guarantee: digest, Merkle params, column count, Vandermonde matrix and prime, with `encode` moved onto it.
- **The "running hash" isn't cumulative.** `running = _transcript_bytes(view.outputs, proof.combination)` *replaces* the previous value, so layer 3's challenges depend on layer 2 but not on layer 1. Layer 1 is still bound indirectly, through layer 3's inputs, so it isn't a hole, but it doesn't match the docs. Fix in `prove_network`, `verify_network` and the test helper:

  ```python
  running = hashlib.sha256(running + _transcript_bytes(outputs, proof.combination)).digest()
  ```

### 5.6 Housekeeping, plus a bug on `main` that isn't yours

All still open from the first review:
- **56 of 72 files are still executable.** Fix with:

  ```bash
  git ls-files -s | awk '$1 == "100755" {print $4}' | xargs git update-index --chmod=-x
  ```

- **The README quick start** still has the personal cluster commands, which don't work as written.
- **`pyproject.toml` is still unpinned**, with `pytest` in the required dependencies.
- **`.wheel_cache/*.sh`** still hard-codes one person's course paths.
- **The commit messages** ("whateverdefuq idos gpu did", "commit pre claude"): squash-merge into `main` with a descriptive message.

**The bug on `main`:** the original `.gitignore` has `*.txt` on line 11, meant to ignore the extracted PDF text at the repo root. It also silently ignores `code/requirements.txt`, so that file **has never been committed**, not even on `main`, even though the README says the dependencies are pinned in it. Combined with the unpinned `pyproject.toml`, the repo currently contains no pins at all. The fix is one character: change line 11 to `/*.txt` (root only), then:

```bash
git add code/requirements.txt
```

I checked: with `/*.txt` the root-level PDF text dumps stay ignored and `code/requirements.txt` becomes trackable.

### 5.7 Two smaller consistency items

- **`code/revision1_notes/REVISION1_REVIEW.md` is now stale.** It describes problems this commit fixed, and it's inside `code/`, so it will ship in the submission tarball. Either add a status note at the top ("Status: addressed in `51e2dfd`; see follow-up review") or move it out of `code/`.
- **The attack-results table** in README Step 5 and `defence_explained.md` §11 (0.112636 / 0.150712 …) still comes from your own trained models, while other numbers come from other runs. Regenerate every table from one set of trained models, with library versions recorded, before writing the report.

---

## 6. Suggested order of work

**Quick, and before anything is quoted in the report**
- [ ] Fix the stale passages in §5.3, including rewording the "11.9×" lines to give the worst query.
- [ ] Add a status note to (or move) the copied first review.

**Tests and reproducibility**
- [ ] Add `tests/test_batched_network.py` (appendix A).
- [ ] Commit the whole-network script (appendix B) and have it write results to JSON.
- [ ] Make the running hash cumulative.
- [ ] Give the verifier a public-parameters object.

**Housekeeping**
- [ ] File modes, README quick start, pins, cluster scripts (§5.6).
- [ ] The `.gitignore` fix and committing `requirements.txt` (§5.6).

**Science**
- [ ] Run the mixture attack against the floor sampler at ε around 1, over ≥ 100 queries, reporting worst queries, then at width 4096.
- [ ] Decide on verifier-sent challenges, or state the ~24-bit caveat clearly in the report.

**Consistency**
- [ ] Regenerate all tables from one set of trained models.

---

## 7. Appendix: drop-in test and reproduction scripts

Run from `code/` on `revision1`, with `artifacts/models/mlp_mnist_full.npz` present, and either `pip install -e .` or `PYTHONPATH=src`.

### A. `tests/test_batched_network.py` (7 tests, all pass on `51e2dfd`)

```python
"""Tests for the public whole-network API: fixed_point_network / prove_network / verify_network.

tests/test_batched.py re-implements the chaining inside `_run_chain`, so these
three functions -- the ones a caller would actually use -- were not exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvi.nn.models import mlp_architecture
from pvi.protocol.batched import (
    DEFAULT_PRIME,
    BatchedWeightCommitment,
    _augment,
    _layer_context,
    _signed,
    _transcript_bytes,
    fixed_point_layer,
    fixed_point_network,
    prove_network,
    rescale,
    verify_layer,
    verify_network,
)

from conftest import build_network

N_QUERIES = 12


def _setup(seed: int = 0):
    network = build_network(mlp_architecture(24, [32, 16], 5), seed=seed)
    query = np.abs(np.random.default_rng(seed + 100).normal(0, 0.5, size=24))
    commitments = [
        BatchedWeightCommitment(view.matrix, prime=DEFAULT_PRIME)
        for view in fixed_point_network(network, query)
    ]
    return network, query, commitments


def _prove_and_verify(network, query, views, commitments):
    proofs = prove_network(views, commitments, n_queries=N_QUERIES)
    accepted = verify_network(
        network.architecture, query, [v.outputs for v in views], proofs, commitments,
        n_queries=N_QUERIES,
    )
    return accepted, proofs


@pytest.mark.parametrize("seed", range(3))
def test_honest_network_is_accepted(seed):
    network, query, commitments = _setup(seed)
    accepted, _ = _prove_and_verify(network, query, fixed_point_network(network, query), commitments)
    assert accepted


@pytest.mark.parametrize("layer_index", [1, 2, 3])
def test_tamper_at_every_layer_is_rejected(layer_index):
    network, query, commitments = _setup()
    honest = fixed_point_network(network, query)
    bad = np.array(_signed(honest[layer_index - 1].outputs, DEFAULT_PRIME), dtype=np.int64)
    if network.architecture[layer_index].activation == "relu":
        live = np.flatnonzero(bad > 0)
        assert len(live), "need a live neuron to zero"
        bad[live[0]] = 0                                   # zero-hiding tamper
    else:
        bad[int(bad.argmax())] = int(bad.min()) - 1000     # flip the winning logit
    views = fixed_point_network(network, query, tampered={layer_index: bad})
    accepted, _ = _prove_and_verify(network, query, views, commitments)
    assert not accepted


def test_broken_chain_is_rejected():
    """Every layer's proof is valid for the input it was built on, but layer 2 was
    built on an input that is NOT rescale(layer-1 output).  Only the verifier's own
    recomputation of the input can catch this."""
    network, query, commitments = _setup()
    views = fixed_point_network(network, query)

    x2 = rescale(views[0].outputs, 8, DEFAULT_PRIME).copy()
    x2[int(np.argmax(x2))] += 50
    weight, bias = network.parameters[network.architecture[2].name]
    views[1] = fixed_point_layer(weight, bias, x2, scale_bits=8, inputs_are_integers=True)
    weight, bias = network.parameters[network.architecture[3].name]
    views[2] = fixed_point_layer(
        weight, bias, rescale(views[1].outputs, 8, DEFAULT_PRIME),
        scale_bits=8, activation="identity", inputs_are_integers=True,
    )

    accepted, proofs = _prove_and_verify(network, query, views, commitments)
    assert not accepted

    # Positive control: layer 2's proof *does* verify against the input it was built
    # on, so the rejection above really comes from the chaining, not from layer 2.
    context = _layer_context(1, _transcript_bytes(views[0].outputs, proofs[0].combination))
    cm = commitments[1]
    assert verify_layer(
        proofs[1], cm.digest, cm.params, cm, _augment(x2, DEFAULT_PRIME), views[1].outputs,
        prime=DEFAULT_PRIME, n_queries=N_QUERIES, context=context,
    )
```

If you make the running hash cumulative (§5.5), update the positive control's `context` line to match.

### B. Whole-network numbers on the real model (candidate `scripts/run_batched_network.py`)

```python
"""Exercise the public whole-network API on the real MNIST MLP."""
import time
import numpy as np
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for
from pvi.data import load_classification
from pvi.protocol.batched import (
    DEFAULT_PRIME, BatchedWeightCommitment, fixed_point_layer, fixed_point_network,
    prove_layer, prove_network, rescale, verify_network, _layer_context, _transcript_bytes, _signed,
)

net = load_network(mlp_architecture_for(10), "artifacts/models/mlp_mnist_full.npz")
arch = net.architecture
ds = load_classification("mnist")
X, Y = ds.test_x.reshape(10000, -1), ds.test_y
P = DEFAULT_PRIME

commitments = [BatchedWeightCommitment(v.matrix, prime=P) for v in fixed_point_network(net, X[0])]

def run(query, tampered=None):
    views = fixed_point_network(net, query, tampered=tampered)
    t0 = time.perf_counter(); proofs = prove_network(views, commitments); t1 = time.perf_counter()
    ok = verify_network(arch, query, [v.outputs for v in views], proofs, commitments)
    t2 = time.perf_counter()
    return ok, views, proofs, t1 - t0, t2 - t1

N = 10
honest, correct, sizes, tp, tv = 0, 0, [], [], []
for i in range(N):
    ok, views, proofs, a, b = run(X[i])
    honest += ok; tp.append(a); tv.append(b); sizes.append(sum(p.size_bytes() for p in proofs))
    correct += int(np.argmax(_signed(views[-1].outputs, P)) == Y[i])
print(f"honest: accepted {honest}/{N}, correct answers {correct}/{N}, "
      f"proof {np.mean(sizes)/1024:.1f} kB, prove {1e3*np.mean(tp):.1f} ms, verify {1e3*np.mean(tv):.1f} ms")

for layer_index in (1, 2, 3):
    rejected = 0
    for i in range(N):
        views = fixed_point_network(net, X[i])
        bad = np.array(_signed(views[layer_index - 1].outputs, P), dtype=np.int64)
        if arch[layer_index].activation == "relu":
            bad[np.flatnonzero(bad > 0)[0]] = 0          # zero one live neuron
        else:
            bad[int(bad.argmax())] = int(bad.min()) - 1000   # flip the winning logit
        ok, *_ = run(X[i], tampered={layer_index: bad})
        rejected += not ok
    print(f"tamper at layer {layer_index}: rejected {rejected}/{N}")

# Genuine broken chain: every layer proof is individually valid, but layer 2's proof
# was made for an input that is NOT rescale(layer-1 output).
rejected = 0
for i in range(N):
    views = fixed_point_network(net, X[i])
    x2 = rescale(views[0].outputs, 8, P).copy(); x2[np.argmax(x2)] += 50
    w, b = net.parameters[arch[2].name]
    views[1] = fixed_point_layer(w, b, x2, scale_bits=8, prime=P, inputs_are_integers=True)
    w, b = net.parameters[arch[3].name]
    views[2] = fixed_point_layer(w, b, rescale(views[1].outputs, 8, P), scale_bits=8, prime=P,
                                 activation="identity", inputs_are_integers=True)
    proofs = prove_network(views, commitments)
    ok = verify_network(arch, X[i], [v.outputs for v in views], proofs, commitments)
    rejected += not ok
print(f"broken chain (valid per-layer proofs, wrong layer-2 input): rejected {rejected}/{N}")
```

Output on `51e2dfd`:
```
honest: accepted 10/10, correct answers 10/10, proof 189.0 kB, prove 5.7 ms, verify 8.4 ms
tamper at layer 1: rejected 10/10
tamper at layer 2: rejected 10/10
tamper at layer 3: rejected 10/10
broken chain (valid per-layer proofs, wrong layer-2 input): rejected 10/10
```

### C. The ε-floor result per query

Must be run from `code/`, because it imports the helper functions from `scripts/run_theorem4_check.py`. It takes about 7 minutes on CPU.

```python
"""Per-query view of the new epsilon-floor result: mean AND worst query."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
from run_theorem4_check import single_neuron_worst_case, multi_neuron_worst_case
from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import _smallest_flipping_value
from pvi.data import load_classification
from pvi.defences import ZeroAwareContributionSampler
from pvi.protocol import ProtocolParams
from pvi.training import load_network
from pvi.zoo import mlp_architecture_for

net = load_network(mlp_architecture_for(10), "artifacts/models/mlp_mnist_full.npz")
ds = load_classification("mnist", seed=0)
tx, ex = ds.train_x.reshape(len(ds.train_x), -1), ds.test_x.reshape(len(ds.test_x), -1)
L, params, W = 1, ProtocolParams(), 512
ceilings = calibrate_activation_ceilings(net, tx[:3000], layer_index=L, percentile=100.0)
eps_list = [0.1, 1.0, 10.0]
best = {e: [] for e in eps_list}
t0 = time.perf_counter()
for qi, q in enumerate(ex[:12]):
    h = net.eval_trace(q); win = int(h.output.argmax()); cache = {}
    for v in range(W):
        val = _smallest_flipping_value(net, h, L, v, win, None, max_value=1e4, require_nonnegative=True)
        if val is not None:
            cache[v] = net.forward_from(h.tampered(L, v, val), L)
    for e in eps_list:
        s = ZeroAwareContributionSampler(net, epsilon=e)
        single = single_neuron_worst_case(net, h, L, s, params, cache)
        multi = multi_neuron_worst_case(net, h, L, s, params, ceilings)
        best[e].append(min(c for c in (single, multi) if c is not None))
    print(f"query {qi}: " + "  ".join(f"eps={e:g}: {best[e][-1]:.4f}" for e in eps_list), flush=True)
print(f"\nuniform single-node detection = 1/N = {1/W:.6f}   ({time.perf_counter()-t0:.0f}s)")
for e in eps_list:
    v = np.array(best[e])
    print(f"eps={e:>4g}: mean {v.mean():.6f} ({v.mean()*W:.1f}x uniform) | worst query {v.min():.6f} "
          f"({v.min()*W:.1f}x uniform) | queries below uniform: {(v < 1/W).sum()}/{len(v)}")
```

Output on `51e2dfd`:
```
query 0: eps=0.1: 0.0118  eps=1: 0.0244  eps=10: 0.0046
query 1: eps=0.1: 0.0113  eps=1: 0.0294  eps=10: 0.0057
query 2: eps=0.1: 0.0069  eps=1: 0.0157  eps=10: 0.0036
query 3: eps=0.1: 0.0323  eps=1: 0.0383  eps=10: 0.0066
query 4: eps=0.1: 0.0095  eps=1: 0.0184  eps=10: 0.0038
query 5: eps=0.1: 0.0145  eps=1: 0.0221  eps=10: 0.0039
query 6: eps=0.1: 0.0093  eps=1: 0.0158  eps=10: 0.0033
query 7: eps=0.1: 0.0051  eps=1: 0.0104  eps=10: 0.0028
query 8: eps=0.1: 0.0021  eps=1: 0.0062  eps=10: 0.0036
query 9: eps=0.1: 0.0064  eps=1: 0.0158  eps=10: 0.0037
query 10: eps=0.1: 0.0128  eps=1: 0.0300  eps=10: 0.0057
query 11: eps=0.1: 0.0137  eps=1: 0.0208  eps=10: 0.0037

uniform single-node detection = 1/N = 0.001953   (437s)
eps= 0.1: mean 0.011316 (5.8x uniform) | worst query 0.002097 (1.1x uniform) | queries below uniform: 0/12
eps=   1: mean 0.020621 (10.6x uniform) | worst query 0.006236 (3.2x uniform) | queries below uniform: 0/12
eps=  10: mean 0.004255 (2.2x uniform) | worst query 0.002848 (1.5x uniform) | queries below uniform: 0/12
```
