# 1. Reproduction of the original protocol (report §4.2, *Reproduction*)

```bash
python experiments/1_reproduction/run.py      # ~1 min; uses the models in artifacts/models
```

Runs `RandPathTest` (`pvi.protocol`) in the four substitute-model settings of
`pvi.zoo.SETUPS`. Each setting uses 150 MNIST queries × 20 fresh challenges = 3,000
executions. For each setting it reports:

* **completeness** (`correctness`): the fraction of honest runs accepted (1.0 in every
  setting), plus one fresh run per query (150 runs, `zero_tolerance_runs`) against a
  zero-tolerance verifier, i.e. the paper's idealised exact-equality test (21–37%
  accepted, `zero_tolerance_acceptance_rate`; with 150 runs this moves by a few points
  from run to run);
* **other-model soundness** (`other_model`): how often the substitute's honest trace is
  rejected (99.8–100%, `detection_rate`), and how often the two models agree on the
  prediction (`prediction_agreement`).

Output: `artifacts/results/reproduction.json` (`setups.<key>.correctness`,
`setups.<key>.other_model`).
