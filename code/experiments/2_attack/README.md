# 2. The single-neuron attack (report §4.2 *Attacks on MNIST*)

```bash
python experiments/2_attack/run.py            # ~2 min; uses artifacts/models/mlp_mnist_full.npz
```

The attacks are implemented in `pvi.attacks`. The script runs them against
`RandPathTest` (one path) on the MNIST MLP. Acceptance probabilities are computed exactly
by the dynamic program in `pvi.experiments.analysis`; the backdoor is also run on the
real protocol.

| Block in `attack.json` | Queries | Report |
|---|---|---|
| `baselines` | 8 | §4.2: the paper's attacks (substitute model, inverse transform, logit swap, gradient reconstruction) and ours (one neuron, layer 1 or 2); wrong nodes = `support_mean`, accepted = `acceptance_mean` |
| `backdoor` | 200 (`--queries`) | §4.2: a 3×3 trigger patch; clean predictions unchanged (`clean_identical_to_committed` 1.0, `clean_accuracy_gap` 0), `attack_success_rate` 1.0 on triggered queries not already in the target class, exact acceptance 0.998 and 99.96% of 2,300 real runs (`--challenges` 25 per tampered query among the first 100) |
| `opening_budget` | 1 | "with 250 paths the attack is still accepted 61% of the time" |
| `stealth` | 60 | §4.2: forged values kept within the neuron's natural range; `p95.0` (about 5 changed neurons, accepted 0.989), with the other percentiles in the text (3–5 neurons) |

Output: `artifacts/results/attack.json`.

On a real LLM (report §4.1 and §4.2 *A real LLM*): `real_llm.py` runs the attack and a
backdoor on OPT-6.7B, and `real_weights_ppl.py` measures the int8 perplexity of real OPT
checkpoints in the benchmark's integer graph. Both need a GPU and Hugging Face downloads;
see [REAL_LLM.md](REAL_LLM.md).
