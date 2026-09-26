# 2. The single-neuron attack (report Table 1, §4.2 *The attack works*)

```bash
python experiments/2_attack/run.py            # ~3 min; needs experiments/0_train_models
```

The attacks are implemented in `pvi.attacks`. The script runs them against
`RandPathTest` on the MNIST MLP (`mlp_mnist_full`), with 200 queries × 25 challenges.
Every acceptance probability is computed exactly by the dynamic program in
`pvi.experiments.analysis` and also measured on the real protocol.

| Block in `attack.json` | Report |
|---|---|
| `baselines` | Table 1 rows 1–6: the paper's attacks (substitute model, inverse transform, logit swap, gradient reconstruction) and ours (one neuron, layer 1 or 2); wrong nodes = `support_mean`, accepted = `acceptance_mean` |
| `backdoor` | Table 1 row 7 and text: a 3×3 trigger patch; clean accuracy gap 0, attack success 1.0, acceptance on triggered queries |
| `evasion.opening_budget` | "with 250 paths the attack is still accepted 61% of the time" |
| `stealth` | Table 1 row 8: forged values kept within the neuron's natural range (`p95.0.untargeted`) |
| `evasion.locality`, `audit` | further analysis (spreading the change; repeated audits) |

Output: `artifacts/results/attack.json`.
