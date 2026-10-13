# Attacking Spot-Checking Proofs of Inference

Section 3 of the paper shows that a prover which edits one activation and recomputes every later layer with the
true weights passes any spot-checking verifier with probability about `1 - q/N`, and measures this on the path
protocol of Anchuri et al. (`RandPathTest`, `pvi.baseline.path`). These scripts reproduce that protocol's published
behaviour, run the single-neuron attack and its backdoor on an MNIST MLP and on OPT-6.7B, run in-range edits against
per-neuron range checks on OPT-1.3B and OPT-6.7B, measure value-aware samplers (and the best mixture of them) against
an adversary who knows the rule, and compute the path test's cost of soundness `2^-40` on Llama-2-7B. Acceptance on
the MNIST MLP is computed exactly (`pvi.baseline.analysis`); on the LLMs, detection per path follows from the number
of edited neurons and the sampler's weights.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| Sec. 3.5 (`sec:break`) "The implementation reproduces the published behaviour"; `ext:attack-measure` *Reproduction*: honest provers pass all 3,000 runs per setting, substitutes caught in 99.8–100%, 21–37% without the 1e-4 tolerance, 150 queries | `reproduction.py` | `python -m experiments.attack.reproduction` | `artifacts/results/reproduction.json` |
| Sec. 3.5 *MNIST*: one neuron passes 99.6–99.8%; substitutes corrupt hundreds of the 778 trace nodes; the 3×3-patch backdoor passes 99.96% of 2,300 runs (predicted 99.8%); `tab:sok` and `tab:sok-derivation` 99.80% (`s=1`) and 61.3% (`s=250`); `ext:attack-measure` "250 paths 61%", "below the 95th percentile takes about five neurons and passes 99%" | `attack_mnist.py` | `python -m experiments.attack.attack_mnist` | `artifacts/results/attack.json` |
| Sec. 3.5 *Other samplers*: uniform 0.20%, contribution 9.2%, zeroing about 14 of 512 neurons passes 2,000 of 2,000 runs; `ext:attack-measure`: static importance 0.11%, gradient saliency 0.016%, zeroed neurons are 0 on 65% of inputs | `samplers.py` | `python -m experiments.attack.samplers` | `artifacts/results/defence.json` |
| Sec. 3.5 "A floor δ [...] reaches 2% at δ=1"; `ext:attack-measure` "10× as often as uniform sampling; other floors give 0.9–6×" | `floor_sampler.py` | `python -m experiments.attack.floor_sampler` | `artifacts/results/floor_sampler.json` |
| Sec. 3.3 (`sec:sampling`) the held-out-mean locator ranks the edited neuron first for 75% and 95% of 40 queries; Sec. 3.5 the best mixture of implementable samplers detects at most 3.1% and 9.7% per path, below 27.7/N (5.4%, 10.8%); `ext:attack-measure` "5.1 neurons on average, passes 99.0% of paths [...] (5% of queries in both cases)" | `adaptive_game.py` | `python -m experiments.attack.adaptive_game` | `artifacts/results/adaptive_game_mlp_mnist.json` |
| Sec. 3.5 *Larger models*: one penultimate neuron flips every CNN test query; caught per path with 1/84 (LeNet-5) to 1/512 (VGG, ResNet); 2,316–14,182 paths for `2^-40`; `ext:attack-measure` "as low as 1/28 million" | `../costs/bench.py` (cells `attack_float`, `sampling`; see `costs/README.md`) | `python -m experiments.costs.bench cnn --model <cnn> ...`, then `python -m experiments.comparison.aggregate --platform l40s` | `artifacts/comparison/raw_l40s/`, `tables_l40s/measured_summary.csv` |
| Sec. 3.5 "at least 305,193 [paths] in the FFN layers of Llama-2-7B (width 11,008), about 13.7 GB of fp16 openings at 64 tokens"; `tab:sok-derivation`, `tab:paths` (`k=1`) | `path_test_bound.py` | `python -m experiments.attack.path_test_bound` | `artifacts/comparison/tables/analytic.csv` |
| Sec. 3.5 *A real LLM*: one FFN neuron of OPT-6.7B changes the next token for 40/40 prompts (Wilson 91.2–100%); backdoor to `hacked` 0/40; values outside the natural ranges; at least 454,248 paths; `ext:attack-measure` 16,384 neurons reach 6,187 of 50,272 tokens | `attack_llm.py` | `python -m experiments.attack.attack_llm` | `artifacts/results/real_llm_attack.json` |
| Sec. 3.5 backdoor to `back` 40/40, clean prompts unchanged | `attack_llm.py` | `python -m experiments.attack.attack_llm --target auto --out artifacts/results/real_llm_attack_auto.json` | `artifacts/results/real_llm_attack_auto.json` |
| `tab:inrange` (OPT-1.3B rows); Sec. 3.5 in-range edits change the next token for 40/40 prompts at layers 6, 12, 18, 23, median 2 neurons (at most 60), median path probability at most 7.9e-4 (at most 7.3e-3); lowering-only edits 9, 31, 38, 39 of 40, median 6; `tab:paths` `k = 2, 6, 60` | `inrange_llm.py` | `python -m experiments.attack.inrange_llm` and `... --moves down` | `artifacts/results/inrange_llm_opt-1.3b_{any,down}.json` |
| `tab:inrange` (OPT-6.7B rows); Sec. 3.5 "On OPT-6.7B [...] median probability at most 1.3e-3" | `inrange_llm.py` | `python -m experiments.attack.inrange_llm --model facebook/opt-6.7b --layers 8 16 24 31 --threads 16` (and `--moves down`) | `artifacts/results/inrange_llm_opt-6.7b_{any,down}.json` |
| `ext:attack-inrange` *The sampler game on OPT-1.3B* (3.1%, 23%, 38%, 48% at layer 6, ...; 27.7/d_ff = 0.34%); Sec. 3.5 "48%, 6.8% and 2.2% [...] and on 0.08% at layer 23 (uniform: 0.02–0.06%)" | `adaptive_game_llm.py` | `python -m experiments.attack.adaptive_game_llm --threads 8 --device cuda` | `artifacts/results/adaptive_game_opt-1.3b.json` (not stored) |

The trained MNIST models (`artifacts/models/`) come from `train_mnist.py`; the stored ones are the weights the paper
used.

## Scripts

Every script prints its options with `--help`; the defaults are the paper's settings. The MNIST scripts read the
models from `--models` (default `artifacts/models/`) and download MNIST through torchvision on first use. JSON
outputs carry the run's options (`config` or `args`) and `env` (`experiments.common.records.environment`: commit,
host, CPU, GPU, library versions). The LLM scripts need the packages of `requirements-llm.txt` (`transformers`,
`tokenizers`, `pyarrow`) and download the OPT checkpoints and WikiText-2 to the Hugging Face cache; OPT is loaded with
`experiments.common.checkpoints.load_opt` (fp32, `weights_only` state dicts, so the pinned torch < 2.6 can read the
`.bin` checkpoints).

### `train_mnist.py`
Trains the seven models of `pvi.mnist.zoo` (architectures, data slices and training settings are defined there): the
MLP `mlp_mnist_full` (784–512–256–10) of the attacks, the samplers and the adaptive game, and the committed and
substitute models of the four reproduction settings. Stored models are skipped unless `--force`; `--only NAME ...`
trains a subset; `--out DIR` writes elsewhere. Writes `<name>.npz` and a training report `<name>.json`
(accuracies, loss, epochs, seed). CPU, about 2 min for all seven. Training is seeded but bit-exact only on one machine
and library build; elsewhere the weights differ slightly (equally accurate) and the MNIST numbers move in their last
digits, so the paper's numbers are reproduced from the stored models (`MODELS.sha256`).

### `reproduction.py`
Runs `RandPathTest` in the four substitute-model settings of `pvi.mnist.zoo.SETUPS` (`--setups KEY ...` selects):
completeness over `--queries` (150) test queries × `--challenges` (20) fresh challenges, one run per query against a
verifier without the 1e-4 tolerance (`--zero-tolerance-queries`), and other-model soundness against the
substitute's honest trace. Record: `setups.<key>.correctness` (`acceptance_rate`, `zero_tolerance_acceptance_rate`,
...) and `setups.<key>.other_model` (`detection_rate`, `prediction_agreement`). CPU, about 1 min. The challenges are
fresh on every run, so the rates move slightly between runs (the zero-tolerance rate, one run per query, by a few
points).

### `attack_mnist.py`
The single-neuron attack and its backdoor against the path protocol on `mlp_mnist_full`, in four blocks:
`baselines` (the forgeries of Anchuri et al. against ours on `--baseline-queries` queries; `support_mean`,
`acceptance_mean`), `backdoor` (`--trigger-size` patch, `--target-class`, `--queries` queries, exact acceptance and
`--challenges` real runs per tampered query among the first `--empirical-queries`), `opening_budget`
(`--opening-paths`) and `stealth` (caps at `--stealth-percentiles` of the ranges over the first `--calib-images`
test images, `--stealth-queries` queries). Every block except `triggered_acceptance_empirical` (real runs with fresh
challenges) is deterministic. CPU, about 1–2 min.

### `samplers.py`
Uniform, static importance, local contribution and gradient saliency against the plain edit and against an adversary
that picks, among `--candidate-neurons` neurons, the edit the sampler detects least (`sweep.<sampler>`); then the
zero blind spot of contribution weighting (`zero_blind_spot`): the zeroing edit within the `--percentile` ceilings,
its exact detection and `--protocol-runs` real runs per query. Layer `--layer` (1, width 512), `--queries` 40. CPU,
about 2–3 min.

### `floor_sampler.py`
The zero-aware contribution sampler `|w_ij|(|a_i| + δ)` for `--epsilons` δ (0, 0.01, 0.1, 1, 10, 100) and uniform
sampling, each against the better of two attacks (every flipping single-neuron edit; the adaptive lowering search
within ceilings from `--calib-images` training images), on the tampered trace, over `--queries` (20) test queries.
Record: `samplers.<uniform|eps=δ>.adversary_best`. CPU, about 2 min per query (40 min).

### `adaptive_game.py`
The adaptive game on both hidden layers (`--layers 1 2`) of `mlp_mnist_full`: ranges from `--calib-size` (5,000)
training images, `--queries` (40) disjoint test queries, caps `--caps` (50, 75, 90, 95, 99, 100 and unconstrained),
the sampler family (uniform, static importance, contribution, the `--floors`, gradient saliency, tail-first at
`--tail-percentile`, zero-first) against four attack families (single-neuron pool of `--pool` candidates, ceiling and
lowering plans of at most `--kmax` neurons, the near-zero lowering to `--near-zero`, and a greedy guided by the
detector in force with `--guided-kmax`/`--guided-pool`), the LP mixture with up to `--double-oracle` (10) double-oracle
rounds, support limits `--k-limits` and the tamper locator's top-`q` hit rates for `--q-grid`. Every number is an
upper bound on the sampler's worst-case detection. Record per layer and cap: `samplers`, `mixture`,
`mixture_upper_bound_with_saliency`, `support_limit`, `R1_pass_rate`, `R2_pass_rate`, `R2_max_pass_rate` (with
honest baselines), `mean_support` and `G1e` (the locator). Needs SciPy. CPU, about 21 min on 2 threads (`--threads`),
0.5 GB of memory. `tests/test_adaptive_game.py` checks its vectorised acceptance against the exact recursion.

### `path_test_bound.py`
The path test's paths for `2^-lam` (`--lam` 40) against one edited neuron of the widest layer of `--model`
(`llama2-7b`; any key of `pvi.models.decoder.CONFIGS`), and the bytes of opening the whole fp16 model and its trace
at `--seq` (64) tokens. Shapes only; seconds. Writes the CSV that `experiments/paper/assets.py` reads (`--out`).

### `attack_llm.py`
The single-neuron attack and a fixed-target backdoor on OPT-6.7B (`--model`): one FFN neuron (an input of `fc2`) at
the last position of `--layers` 8, 16, 24, 31, the 40 prompts of the module (`--prompts`), `--topk` candidates per
ranking, the trigger `--trigger` and the target `--target` (` hacked`, or `auto`). Records the `untargeted` and
`backdoor` summaries (with `success_wilson95`, the 95% Wilson interval of the success rate), `reachability` (the
saturation tokens per layer), `path_test` and every prompt's flip (`per_prompt`). Resources of the stored runs: one
RTX A5000 (24 GB; the fp16 weights take 13 GB), 8 cores and 48 GB of host memory (the checkpoint is loaded in fp32
and then cast), about 1 h per run, at most 3 h. `--device cpu` runs in fp32.

### `inrange_llm.py`
Greedy in-range edits at the last position of each of `--layers` against range checks calibrated on
`--calib-windows` × `--calib-seq` (64 × 128 = 8,192) WikiText-2 train tokens: `--ranges` `max` and `p999` (99.9th
percentile), `--moves any` (to either bound) or `down` (to 0 only), at most `--cap` (256) neurons. Prints one JSON
summary per layer and range (`success`, `k_median`, `k_max`, `p_uniform_median`, `p_contribution_median`, ...) and
writes the per-prompt records to `--out` (default `artifacts/results/inrange_llm_<model>_<moves>.json`); the
`tab:inrange` rows are those summaries. fp32: OPT-1.3B about 11 min per `--moves` variant on 8 laptop threads;
OPT-6.7B (27 GB of weights) ran on 16 threads of a Xeon CPU node, or runs on one GPU with 48 GB (`--device cuda`).

### `adaptive_game_llm.py`
The sampler game on OPT-1.3B: the `any` edits of `inrange_llm.py` at ceilings for each of `--caps` (90, 95, 99, 100;
tighter caps' edits carried to looser ones) and the `down` edit, at most `--edit-cap` (60) neurons, scored by one FFN
step of every sampler (uniform, static importance, contribution, `--floors`, tail-first, zero-first) and the LP
mixture of `adaptive_game.solve_mixture`. Record per layer and cap: `success_rate`, `samplers`, `mixture`. One RTX
2080 Ti (OPT-1.3B in fp32 needs about 5.3 GB), 8 cores (`--threads 8 --device cuda`), 32 GB of host memory, about
13 min (at most 3 h); needs SciPy. The paper's record (`adaptive_game_opt-1.3b.json`) is not stored in the
repository; this command regenerates it.

## Reproducing the paper's numbers

The run package's `attack` tier (`bash code/experiments/package/run.sh --platform <name> attack`) runs every
command below and writes the records to `logs/<name>/attack/`; each command also runs on its own. In order, from `code/` with
`PYTHONPATH="src;."` on Windows or `PYTHONPATH=src:.` elsewhere:

```bash
# MNIST (CPU, about 70 min in all; the stored models of artifacts/models are used)
python -m experiments.attack.reproduction
python -m experiments.attack.attack_mnist
python -m experiments.attack.samplers
python -m experiments.attack.floor_sampler
python -m experiments.attack.adaptive_game --threads 2
python -m experiments.attack.path_test_bound

# OPT (pip install -r requirements-llm.txt; HF_HOME pointing at a cache with room for the checkpoints)
python -m experiments.attack.attack_llm                                    # one GPU >= 16 GB, about 1 h
python -m experiments.attack.attack_llm --target auto --out artifacts/results/real_llm_attack_auto.json
python -m experiments.attack.inrange_llm                                   # OPT-1.3B, CPU, 8 threads
python -m experiments.attack.inrange_llm --moves down
python -m experiments.attack.inrange_llm --model facebook/opt-6.7b --layers 8 16 24 31 --threads 16
python -m experiments.attack.inrange_llm --model facebook/opt-6.7b --layers 8 16 24 31 --threads 16 --moves down
python -m experiments.attack.adaptive_game_llm --threads 8 --device cuda   # one RTX 2080 Ti, about 13 min
```

On a SLURM cluster, the two GPU jobs ran with these resources (one GPU, 8 cores and a 3-hour limit each; 48 GB of host
memory for `attack_llm`, 32 GB for `adaptive_game_llm`), with `HF_HOME` exported and `OMP_NUM_THREADS=8`, e.g.
`sbatch --gres=gpu:1 -c 8 --mem=48G -t 03:00:00 --wrap "cd code && PYTHONPATH=src:. python -m experiments.attack.attack_llm"`.

The CNN attack cells (*Larger models*) are part of the benchmark: the run package's `bench` tier runs
`costs/bench.py cnn --model <cnn>` with them, and `python -m experiments.comparison.aggregate --platform l40s` builds
`tables_l40s/measured_summary.csv` (`attack_float flip_success`, `sampling p_detect_penultimate`,
`p_detect_min_node`, `paths_needed_penultimate`) from the stored `raw_l40s/` records without a GPU.

**What reproduces exactly.** On one machine the MNIST records are deterministic given the stored models, except the
rates from real protocol runs with fresh challenges (`triggered_acceptance_empirical`, the rates of
`reproduction.json`, `protocol_accepts` when below 100%). Across machines and library builds the searches can pick
other neurons. A rerun of the current code on a Windows laptop (torch 2.2.2, CPU) gives every deterministic value of
`attack.json` to 1e-9, but moves some values of the other two stored records: `defence.json`'s
`zero_blind_spot.mean_support` 14.0 (stored 14.35), `natural_zero_fraction_at_chosen_neurons` 66.0% (65.3%) and
gradient saliency's `detection_evasive` 0.0163% (0.0157%); in `adaptive_game_mlp_mnist.json` (layer 1) the largest
mixture detection 2.98% (3.06%) and the cap-95 `mean_support` 5.05 (5.075), with the locator's 75% and the 99.0%
unchanged. The in-range records' discrete values (success, edited neurons, uniform detection) do not depend on the
hardware; their floats differ by about 1e-6 between CPUs.

**A laptop check.** Small versions of every script run in minutes on a CPU (their numbers are not results):

```bash
python -m experiments.attack.reproduction --queries 5 --challenges 2 --setups quantised --out /tmp/reproduction.json
python -m experiments.attack.attack_mnist --queries 20 --challenges 2 --stealth-queries 5 --out /tmp/attack.json
python -m experiments.attack.samplers --queries 3 --out /tmp/defence.json
python -m experiments.attack.floor_sampler --queries 1 --epsilons 0 1 --out /tmp/floor_sampler.json
python -m experiments.attack.adaptive_game --layers 1 --queries 2 --caps 95 inf --double-oracle 1 --out /tmp/ag.json
python -m experiments.attack.inrange_llm --model facebook/opt-125m --layers 6 --prompts 2 --calib-windows 4 --out /tmp/inrange.json
python -m experiments.attack.adaptive_game_llm --model facebook/opt-125m --layers 6 --prompts 2 --calib-windows 8 --out /tmp/agl.json
python -m pytest -q tests/test_adaptive_game.py
```
