# 1. Reproduction of the original protocol (report §4.2, *Reproduction*)

```bash
python experiments/1_reproduction/run.py      # ~5 min; needs experiments/0_train_models
```

Runs `RandPathTest` (`pvi.protocol`) in the four substitute-model settings of
`pvi.zoo.SETUPS`. Each setting uses 150 MNIST queries × 20 fresh challenges = 3,000
executions. For each setting it reports:

* **completeness**: the fraction of honest runs accepted (1.0 in every setting), and
  the same runs against a zero-tolerance verifier, i.e. the paper's idealised
  exact-equality test (17–32% accepted);
* **other-model soundness**: how often the substitute's honest trace is rejected
  (99.8–100%), and how often the two models agree on the prediction;
* **cost** of one path: prover and verifier time, proof size.

Output: `artifacts/results/reproduction.json` (`setups.<key>.correctness`,
`.other_model`, `.cost`).
