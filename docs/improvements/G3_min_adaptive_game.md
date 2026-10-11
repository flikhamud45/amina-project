# G3-min. The adaptive game: the best value-aware sampler against range-respecting edits

Plan 2 item G3-min (rank 4, should; reviewers A-W2, A-Q3, change 5). Design: `attack_theory/experiments.md`
Sections 0 and 1, and `paper_final/value_aware/value_aware_bound.md`. Scripts in
`code/experiments/3_sampling_fixes/`: `adaptive_game.py` (MNIST game), `adaptive_game_llm.py` and
`adaptive_game_llm.sbatch` (OPT-1.3B, cluster), `adaptive_game_plots.py` (figures and table); test
`code/tests/test_adaptive_game.py`; two new samplers in `code/src/pvi/defences/sampling.py`
(`TailFirstSampler`, `ZeroFirstSampler`). Results: `code/artifacts/results/adaptive_game_mlp_mnist.json`,
`figure_g3a.pdf`, `figure_g3b.pdf`, `table_g3.tex`. Status: MNIST MLP done (laptop, 40 queries); OPT-1.3B
smoke-tested on OPT-125M, full run pending on the cluster.

## 1. The question

The paper reports, per sampler, the detection of the best attack found against it, and the best such bound
(2%, a floor sampler). Reviewer A asks whether a defender who knows the attacks can do better: mix samplers,
add a range check, or target the values the attacks write. This item plays the game the design specifies.
The defender publishes a sampler, or a per-path mixture, from a family F. The adversary knows the model, the
query, the sampler and the held-out natural ranges, and returns a forged trace whose output is wrong and
whose edited values lie in range. Detection per path is the exact `1 - acceptance`.

The reported value is the minimum, over the attacks found, of the detection, averaged over queries. A
stronger adversary can only lower it, so **every detection figure below is an upper bound on that
sampler's worst case** -- the direction that supports "even the best sampler detects at most X".

## 2. Method

* **Model and split.** MNIST MLP `mlp_mnist_full.npz` (784-512-256-10), both hidden layers (N = 512 and
  N = 256). Ranges come from 5,000 *training* images; the attacks use the first 40 *test* images, which are
  disjoint. This fixes the overlap of the earlier scripts, which calibrated on `test_x[:3000]` and attacked
  `test_x[0:60]`.
* **Ranges.** A neuron's natural range at cap `p` is `[min, p-th percentile]` over the calibration set,
  for `p` in {50, 75, 90, 95, 99, 100} plus an unconstrained cap. Every hidden neuron has natural minimum
  0, so the floor is the ReLU floor; the logits have negative minima.
* **Family F (11 samplers).** uniform; static importance (outgoing `|w|`); contribution `|w_ij a_i|`; floor
  `|w_ij|(|a_i| + eps)` for eps in {0.01, 0.1, 1, 10, 100}; tail-first (uniform among claimed values `> 0` at
  or above their held-out 95th percentile, else uniform); zero-first (uniform among claimed zeros, else
  uniform); and gradient saliency. Gradient saliency (marked `*`) needs layer-wide gradients, so no
  verifier can run it; it is reported only as an upper bound.
* **Adversaries (all edited values in range).**
  * A1: the smallest flipping value of a single neuron, over 96 candidates (the most and least salient
    neurons and an even spread), kept if it lies in range.
  * A2: the ceiling-or-zero plans of `plan_adaptive_stealthy_flip`, with its three rankings and held-out
    ceilings.
  * A3: zeroing (lower to the mode).
  * A3b: lowering to `min(1e-3, ceiling)`, a value just above zero. `value_aware_bound.md` names this as
    the reply to zero-targeting; without it, zero-first and any mixture that leans on it are overstated.
  * A4: a greedy that, at each step, commits the neuron of lowest *exact* detection under the sampler in
    force, run against uniform, contribution, tail-first and zero-first. Every attack is scored against the
    whole family.
  * A plan found at a tighter cap stays in range at every looser one, so it is carried forward. This makes
    detection non-increasing in `p`, as it must be for a rational adversary.
  * Edits stop at K = 48 neurons, and the best attack is also reported for support limits
    K in {1, 2, 4, 8, 16, 32, 48}.
* **LP mixture and double oracle.**
  * The defender maximises `(1/Q) sum_q min_a alpha . D_{q,a}` over distributions `alpha` on the
    implementable samplers, with `scipy.optimize.linprog` (HiGHS).
  * The double oracle re-runs A4 against the mixture as a detector (one-path acceptance is linear in
    `alpha`), adds any strictly better attack and re-solves, stopping when the value moves by less than
    1e-4 or after 10 rounds.
  * Attacks found at a tighter cap are carried to looser caps.
  * The same LP over all 11 samplers, saliency included, gives an upper-bound mixture.
* **R1 and R2.**
  * R1: every edited value lies in its range.
  * R2: every value of the recomputed trace (tampered layer, later layers, logits) lies in its range at
    the same cap.
  * R2 is reported with the honest trace's pass rate as a baseline, and also against the natural maximum
    (`p = 100`), the only one of these range checks that an honest trace nearly always passes.
* **G1e (empirical `G_q`).**
  * The locator ranks the neurons of the edited layer by `|claimed - mean| / std` (held-out statistics),
    from the forged trace alone.
  * It is measured for the single-site edit: the greedy edit (smallest flipping value) and a randomised
    one (uniform among the flipping neurons).
  * The output is its top-q hit rate for q in {1, ..., 64}.
* **Exact acceptance.** `fast_acceptance` evaluates the path DP in matrix form. A source-independent
  sampler takes one dot product per layer; the contribution and floor samplers take two matrix-vector
  products (`(|W|(u*g)) / (|W|u)` with `u = |a| + eps`). One consistency mask is shared across the family.
* **Run.** `--queries 40 --layers 1 2 --calib train`: 1,254 s (21 min) on 2 threads of the laptop CPU,
  peak RSS about 0.5 GB, inside the 40-minute budget.

## 3. Validation (kill criterion K-G3c)

`tests/test_adaptive_game.py` compares `fast_acceptance` with `acceptance_probability` on the MNIST MLP:
11 samplers, 30 traces (honest traces, single-neuron flips at both layers and stealthy multi-neuron plans
at both layers), and `n_paths` in {1, 10, 250}. **The largest absolute difference is 8.3e-14**, well
within the 1e-9 tolerance, and the test asserts the tolerance. The test passed before any game number was
used and again after every change to the DP. It also checks that the vectorised transition matrix is
row-stochastic.

## 4. Results (MNIST MLP, 40 held-out test queries)

**Table G3 (95th-percentile cap).** Detection per path; `s_40 = ceil(40 ln 2 / -ln(1 - d))`.

| Layer | N | uniform d | best implementable sampler | LP mixture (deployable) | s_40 unif / best / mix | success, mean k |
|---|---:|---:|---|---|---|---|
| 1 | 512 | 0.99% | floor-1: 2.44% (2.5x) | 2.45% (2.5x) | 2,784 / 1,121 / 1,117 | 100%, 5.1 |
| 2 | 256 | 4.03% | floor-1: 5.99% (1.5x) | 7.06% (1.8x) | 674 / 449 / 379 | 100%, 10.3 |

**Every cap.** In the table below:

* "of 27.7/N" is the mixture's detection as a fraction of the level at which `2^-40` needs fewer than N
  paths (5.41% on layer 1, 10.82% on layer 2).
* "+saliency UB" is the upper-bound mixture that may also use the non-implementable gradient saliency.
* k is the support of the adversary's best attack against uniform.

| Layer | cap | uniform | best implementable | mixture (deployable) | of 27.7/N | mixture s_40 | +saliency UB | k |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 50 | 2.25% | floor-100 3.06% | 3.06% | 0.56 | 894 | 5.91% | 11.5 |
| 1 | 75 | 2.08% | floor-100 2.87% | 2.87% | 0.53 | 952 | 5.77% | 10.7 |
| 1 | 90 | 1.26% | floor-1 2.49% | 2.53% | 0.47 | 1,081 | 4.90% | 6.5 |
| 1 | 95 | 0.99% | floor-1 2.44% | 2.45% | 0.45 | 1,117 | 4.58% | 5.1 |
| 1 | 99 | 0.69% | floor-1 2.38% | 2.39% | 0.44 | 1,149 | 4.29% | 3.5 |
| 1 | 100 | 0.45% | floor-1 2.33% | 2.34% | 0.43 | 1,174 | 3.89% | 2.3 |
| 1 | none | 0.195% | floor-1 1.98% | 2.15% | 0.40 | 1,276 | 3.26% | 1.0 |
| 2 | 50 | 8.47% | static 9.67% | 9.67% | 0.89 | 273 | 17.12% | 21.7 |
| 2 | 75 | 6.62% | floor-10 7.72% | 7.73% | 0.71 | 345 | 15.42% | 17.0 |
| 2 | 90 | 4.82% | floor-10 6.36% | 7.16% | 0.66 | 374 | 12.16% | 12.4 |
| 2 | 95 | 4.03% | floor-1 5.99% | 7.06% | 0.65 | 379 | 11.39% | 10.3 |
| 2 | 99 | 3.05% | floor-1 5.99% | 7.01% | 0.65 | 382 | 11.00% | 7.8 |
| 2 | 100 | 2.06% | floor-1 5.92% | 6.85% | 0.63 | 392 | 10.26% | 5.3 |
| 2 | none | 0.39% | floor-1 5.25% | 5.82% | 0.54 | 463 | 6.97% | 1.0 |

What the tables say:

* **The deployable mixture is essentially the floor sampler.** At p = 95 its weights are floor-1 0.88 and
  floor-10 0.12 (layer 1), and floor-1 0.50, floor-10 0.49 and tail-first 0.01 (layer 2). The double
  oracle converged within one round at every cap (for example layer 2, p = 95: 7.07% to 7.06%).
* **The other value-aware samplers are evaded completely.** Against the best attack found at p = 95:
  * contribution, tail-first and zero-first detect 0 (to rounding), except tail-first on layer 2 at
    0.59%;
  * floor-0.01 detects 0.16% and 0.14%;
  * static importance detects 1.68% and 4.67%;
  * gradient saliency (`*`) detects 3.90% and 10.01%.
* **Consistency with the paper's stored numbers.**
  * The unconstrained floor-1 figure (1.98%, 10.1x uniform) matches the paper's 2% (10.2x).
  * The uniform figure at p = 95 on layer 1 (5.1 neurons, 0.99% per path, so 99.0% acceptance) matches
    the paper's "about five neurons, 99%". Both now hold under held-out ranges.
* **Support limit K.** The share of queries with an in-range edit of at most K neurons:

  | Layer | cap | K <= 1 | K <= 2 | K <= 4 | K <= 8 | K <= 16 | K <= 32 |
  |---|---|---:|---:|---:|---:|---:|---:|
  | 1 | 100 | 25% | 62.5% | 97.5% | 100% | 100% | 100% |
  | 2 | 100 | 5% | 12.5% | 30% | 90% | 100% | 100% |
  | 1 | 95 | 12.5% | 22.5% | 52.5% | 87.5% | 100% | 100% |
  | 2 | 95 | 2.5% | 5% | 7.5% | 30% | 90% | 100% |

  A range check makes the single-neuron edit rare in range. It does not prevent a short multi-neuron
  edit.
* **Magnitude.** The mean of `max |a' - a| / sigma` at the uniform-best attack:
  * layer 1: 3.9 sigma at p = 95 and 7.4 sigma at p = 100, against 161 sigma unconstrained;
  * layer 2: 3.1, 6.4 and 239 sigma.
* **R1 / R2.**
  * R1 is 100% at every cap on both layers.
  * R2 at the cap's own percentile fails for most plans at p <= 99, but **the honest trace fails equally**
    (p = 95: 0% / 0% on layer 1, 17.5% / 17.5% on layer 2). A percentile below the maximum is not a range
    check an honest trace passes, so this failure does not come from the edit.
  * Against the natural maximum, the forged traces pass 92.5-95% (layer 1, honest 95%) and 97.5% (layer
    2, honest 97.5%) at every finite cap. The unconstrained single-neuron edit passes only 22.5% and 2.5%.
* **G1e (unconstrained single-site edit, 40 queries).** The surprise locator's hit rates, against
  q/N = 0.20% and 0.39% for q = 1:

  | Layer | edit | top-1 | top-4 | top-16 |
  |---|---|---:|---:|---:|
  | 1 | greedy | 75% | 87.5% | 97.5% |
  | 1 | randomised | 97.5% | 100% | 100% |
  | 2 | greedy | 95% | 97.5% | 97.5% |
  | 2 | randomised | 97.5% | 100% | 100% |

  The unconstrained edit is localisable from its improbable value, as Corollary `cor:surprise` predicts.
  Randomising the site does not help, because every flipping value is out of range. In range at p = 100, a
  single-site edit exists for only 9 of 40 queries (layer 1) and 2 of 40 (layer 2). On those 9 queries the
  greedy edit is ranked first in 11% (1/9), in the top 4 in 44% and in the top 16 in 89%, against q/N of
  0.2%, 0.8% and 3.1%. That sample is small, so treat these figures as indicative.

## 5. Kill criteria

* **K-G3a (decisive): does not trigger for implementable samplers.**
  * On neither layer and at no cap does the deployable LP mixture reach 27.7/N against the best in-range
    attack found: at most 3.06% against 5.41% (layer 1, 0.56 of the level) and at most 9.67% against
    10.82% (layer 2, 0.89, at p = 50).
  * Its `s_40` stays above N everywhere: at least 894 > 512 and 273 > 256.
  * The paper may keep "value-aware sampling does not reach `2^-40` with fewer than N paths on this model"
    for the samplers a verifier can run.
* **Three qualifications, stated plainly:**
  1. **The verdict depends on the adversary having its two obvious replies.** The first full run lacked
     the near-zero lowering (A3b) and the carry-over across caps. In that run the deployable mixture on
     layer 2 put its weight on zero-first and tail-first and reached 11.1%, 12.0% and 11.3% at p = 50, 99
     and 100, above 10.82%: that version of the game did trigger K-G3a. Those numbers are from that run's
     log; its JSON was overwritten. Both replies are cheap and in range, so the corrected numbers are the
     valid upper bounds.
  2. **Layer 2 at p = 50 is close (0.89 of the level).** A weaker adversary, or a richer defender family,
     could cross it.
  3. **The upper-bound mixture that may use gradient saliency crosses 27.7/N.** It reaches 5.91% and
     5.77% at p = 50 and 75 on layer 1, and 11.00-17.12% at p = 50-99 on layer 2 (10.26% at p = 100,
     just below). Saliency needs every
     later weight, so no spot-checking verifier can compute it, but the paper must not say that *no
     value-reading rule whatsoever* reaches 27.7/N on this model. It may say that none a verifier can run
     does.
* **K-G3b: does not trigger.** The in-range success rate at p = 100 with K <= 48 is 100% (40/40) on both
  layers; it is also 100% at p = 50. The range check costs the adversary neurons (Section 4), not success.
* **K-G3c: passes.** The largest difference is 8.3e-14 (Section 3).
* **K-G3d: triggers in its literal form, at caps p <= 99.** R2 at the cap's own percentile fails for most
  plans, so R1 and R2 are reported separately and the paper must not claim "every value within its p-th
  percentile". Two facts soften this: the honest trace fails the same R2 equally often, and the forged
  traces stay within the natural maximum as often as honest traces do. The defensible wording is "every
  edited value lies in its range, and the recomputed trace leaves the natural maximum no more often than an
  honest trace".

## 6. Text for the three `\todo`s (`paper_sp2027/sections/attack.tex`)

`\todo{G1e: empirical $G_q$, MNIST MLP}` (after "... a concentrated choice of neuron (Corollary~\ref{cor:surprise})."):

```latex
On the MNIST MLP the unconstrained single-site edit is of the first kind: ranking the neurons of
the edited layer by the distance of the claimed value from the neuron's held-out mean, in held-out
standard deviations, places the edited neuron first for 75\% and 95\% of 40 test queries in the
two hidden layers, against $1/N$ of 0.2\% and 0.4\%.
```

`\todo{G3: training-split ranges}` (replace the footnote text):

```latex
\footnote{Percentiles from 5{,}000 training images; on 40 disjoint test queries the edit changes
5.1 neurons on average, passes 99.0\% of paths, keeps every edited value in range, and exceeds a
neuron's natural maximum no more often than the honest trace (5\% of queries in both cases).}
```

`\todo{G3: adaptive game}` (after "2\% is the best such bound among our samplers."):

```latex
Against an adversary that knows the sampler, keeps every edited value within held-out ranges and
adapts its edit, the best per-path mixture of the implementable samplers above, found by linear
programming with a double oracle, detects at most 3.1\% (first hidden layer) and 9.7\% (second)
per path, below the $27.7/N$ (5.4\% and 10.8\%) at which $2^{-40}$ needs fewer than $N$ paths; at
the 95th-percentile cap it detects 2.5\% and 7.1\%, 2.5 and 1.8 times uniform, and these are upper
bounds on its worst case. On OPT-1.3B the mixture detects at most \OPTmix{} per path at layers 6,
12, 18 and 23 against \OPTunif{} for uniform sampling, conditional on a path entering the edited
FFN (\OPTsfortyrange{} paths for $2^{-40}$).
```

Placeholders: `\OPTmix` (the largest deployable mixture value over layers and caps), `\OPTunif` (the
matching uniform value) and `\OPTsfortyrange` (the mixture's `s_40` range), all from
`artifacts/results/adaptive_game_opt-1.3b.json`. If the OPT run shows a mixture at or above 27.7/d_ff,
change the sentence as K-G3a requires. If the paper wants the saliency caveat, add "a mixture that may also
use gradient saliency, which needs the whole model, would reach 5.9\% and 17.1\% at the 50th-percentile
cap".

## 7. OPT-1.3B (cluster)

* **The script.** `adaptive_game_llm.py` adds no new search. It reruns the existing in-range edits of
  `experiments/2_attack/inrange_llm.py` (the `any` edit at each cap's ceilings, carried to looser caps,
  and the cap-independent `down` edit), and imports its helpers and `real_llm.py`'s `prefix_cache`.
* **Detection.** It scores every edit under the same family (no saliency), one FFN step conditional on
  entering the FFN. That is an upper bound on unconditional detection.
* **The LP.** It solves the same per-prompt LP as `adaptive_game.solve_mixture`.
* **Smoke test** (laptop): OPT-125M, 2 prompts, layer 6, caps 90/95/99/100, 1,024 calibration tokens. It
  ran end to end in 17 s; its numbers are not results.
* **Command** (from the repository root on the cluster, account `gpu-students`, an RTX 2080 Ti partition):

```bash
export HF_HOME=<Hugging Face cache outside the project quota> PVI_PYTHON=<python with transformers, pyarrow, scipy>
sbatch -p studentbatch -A gpu-students code/experiments/3_sampling_fixes/adaptive_game_llm.sbatch
```

  This runs:

```bash
cd code && PYTHONPATH=src OMP_NUM_THREADS=8 python experiments/3_sampling_fixes/adaptive_game_llm.py \
  --model facebook/opt-1.3b --layers 6 12 18 23 --prompts 40 --calib-windows 64 --calib-seq 128 \
  --caps 90 95 99 100 --threads 8 --device cuda --out artifacts/results/adaptive_game_opt-1.3b.json
```

  Then run `adaptive_game_plots.py --llm artifacts/results/adaptive_game_opt-1.3b.json` to add the OPT
  rows to the figures and Table G3.
* **Caveat.** The OPT adversary has only the two existing edits: `down` writes exact zeros and `any`
  writes the ceiling. Zero-first and tail-first will therefore look better there than they would against
  the replies of Section 2. Read an OPT mixture that leans on them as a loose upper bound, not as K-G3a
  evidence, unless it is re-checked with a near-zero `down` variant. That variant was left out on purpose
  (scope rule K9: no new offensive search on the real model).

## 8. Limitations and open issues

* **One model, 40 queries, two layers.** The CNN heads of the design (LeNet-5, VGG, ResNet) are not run.
* **A restricted adversary.** A1 uses 96 candidates, A4 only 4 detectors plus the mixture, and the greedy
  searches are first-order. Each restriction can only raise the detections (they stay valid upper bounds),
  but it may hide a lower worst case.
* **The double oracle adapts only A4 to the mixture.** A1, A2, A3 and A3b do not depend on the detector,
  so their attacks are all in the matrix already.
* **The mixture is per path, its value a mean over queries.** An adversary that picks *which* queries to
  attack faces lower detection on its easiest queries; the JSON keeps per-sampler medians.
* **G1e covers single-site edits only.** A locator for the multi-neuron in-range edits (top-q intersecting
  the edited set) is not measured.
* **Tail-first uses one fixed cut (95th percentile).** Other cuts, or a mixture over cuts, may do
  somewhat better.

## 9. Usage

```bash
cd code
PYTHONPATH=src python -m pytest tests/test_adaptive_game.py -q                       # K-G3c
PYTHONPATH=src OMP_NUM_THREADS=2 python experiments/3_sampling_fixes/adaptive_game.py --queries 40 --layers 1 2 --calib train
PYTHONPATH=src python experiments/3_sampling_fixes/adaptive_game_plots.py         # figure_g3a/b.pdf, table_g3.tex
```
