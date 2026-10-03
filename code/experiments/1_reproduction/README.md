# 1. Reproduction of the path protocol (paper Sec. 4.2, *Reproduction*)

```bash
python experiments/1_reproduction/run.py      # about 1 min on a CPU; uses the models in artifacts/models
```

Runs `RandPathTest` (`pvi.protocol`), our implementation of the protocol of Anchuri et al., in the
four substitute-model settings of `pvi.zoo.SETUPS`. Each setting uses 150 MNIST queries x 20 fresh
challenges = 3,000 executions (`--queries`, `--challenges`, `--setups KEY ...` change this). For each
setting it reports:

* **completeness** (`correctness`): the fraction of honest runs accepted (1.0 in every setting), plus
  one fresh run per query (150 runs, `zero_tolerance_runs`) against a zero-tolerance verifier, i.e. the
  exact-equality test without the prescribed tolerance of 1e-4 (21–37% accepted,
  `zero_tolerance_acceptance_rate`; with 150 runs this moves by a few points from run to run);
* **other-model soundness** (`other_model`): how often the substitute's honest trace is rejected
  (99.8–100%, `detection_rate`; the lowest is the 8-bit quantised copy), and how often the two models
  agree on the prediction (`prediction_agreement`).

Output: `artifacts/results/reproduction.json` (`setups.<key>.correctness`, `setups.<key>.other_model`),
which is included. The challenges are fresh on every run, so a re-run gives slightly different
rates: one rerun gave 17–37% zero-tolerance acceptance (the stored run: 21–37%) and detection down to
99.7% (stored: 99.8–100%).
