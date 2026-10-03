# 2. The single-neuron attack (paper Sec. 3.2 and 4.2)

The attack of Sec. 3.2 runs the committed model honestly, overwrites one neuron (ranked by the
gradient of the decision margin, with the smallest flipping change found by bisection) and recomputes
every later layer with the true weights. It is implemented in `pvi.attacks` and evaluated three ways:

| Setting | Paper | Script | Hardware, time | Stored result |
|---|---|---|---|---|
| the MNIST MLP, against the path protocol | Sec. 4.2 *Attacks on MNIST* | `run.py` (below) | CPU, about 2 min | `artifacts/results/attack.json` |
| every CNN of the benchmark, incl. the 224-pixel dog-vs-cat ResNet-18, and Llama-2-7B's shape | Sec. 4.2 *Larger models* | the `attack_float` and `sampling` cells of [`4_defence_benchmark/bench.py`](../4_defence_benchmark/README.md), and `5_comparison/analytic.py` | GPU (part of the benchmark) | `artifacts/comparison/raw_l40s/`, `tables/analytic.csv` |
| the real OPT-6.7B (fp16) | Sec. 4.2 *A real LLM* | `real_llm.py` ([REAL_LLM.md](REAL_LLM.md)) | one GPU with 16 GB or more (the stored runs: an A5000), about 1 h per run | `artifacts/results/real_llm_attack*.json` |

## The MNIST MLP

```bash
python experiments/2_attack/run.py            # about 2 min; uses artifacts/models/mlp_mnist_full.npz
```

The script runs the attacks against `RandPathTest` (one path) on the MNIST MLP. Acceptance
probabilities are computed exactly by the dynamic program in `pvi.experiments.analysis`; the backdoor
is also run on the real protocol. `--queries` and `--challenges` change the backdoor's sample sizes.

| Block in `attack.json` | Queries | Paper (Sec. 4.2) |
|---|---|---|
| `baselines` | 8 | the forgeries of Anchuri et al. (substitute model, inverse transform, logit swap, gradient reconstruction) against ours (one neuron, layer 1 or 2): inconsistent nodes `support_mean` (256–778 of the 778 trace nodes for theirs, 1 for ours) and acceptance `acceptance_mean` (0 for theirs; 1 - 1/N, 99.6–99.8%, for ours) |
| `backdoor` | 200 (`--queries`) | a 3x3 trigger patch; clean predictions unchanged (`clean_identical_to_committed` 1.0, `clean_accuracy_gap` 0), `attack_success_rate` 1.0 on triggered queries not already in the target class, exact acceptance 0.998 (`triggered_acceptance_exact`) and 99.96% of 2,300 real runs (`triggered_acceptance_empirical`; `--challenges` 25 per tampered query among the first 100) |
| `opening_budget` | 1 | with 250 paths the attack is still accepted 61% of the time (`acceptance_exact`) |
| `stealth` | 60 | forged values kept below a percentile of the neuron's natural range: at the 95th percentile (`p95.0`) about 5 changed neurons, accepted 0.989; the other percentiles take 3–5 neurons |

## Larger models

On every CNN the benchmark's `attack_float` cell flips the prediction with one penultimate neuron, and
the `sampling` cell computes the path protocol's exact detection per neuron and its bytes for k paths:
1/84 (LeNet-5) to 1/512 (VGG, ResNet) per path, as low as 1/28 million for the least-visited neuron,
and 2,316–14,182 paths for 2^-40. From the stored records (no GPU):

```bash
python experiments/5_comparison/aggregate.py --platform l40s     # -> artifacts/comparison/tables_l40s/measured_summary.csv
python experiments/5_comparison/analytic.py                      # Llama-2-7B: 1/11,008 per path, 305,193 paths, 13.7 GB
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2 --definition
```

To re-measure, run the CNN jobs of [`4_defence_benchmark`](../4_defence_benchmark/README.md) (the
224-pixel classifiers need ImageNet and their trained weights).

## A real LLM

`real_llm.py` runs the attack and a fixed-target backdoor on the real OPT-6.7B, and
`real_weights_ppl.py` measures the int8 perplexity of real OPT checkpoints in the benchmark's integer
graph (Sec. 4.1). Both need a GPU, `transformers` and Hugging Face downloads; see
[REAL_LLM.md](REAL_LLM.md).
