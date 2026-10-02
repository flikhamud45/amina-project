# 3. Smarter sampling does not stop the attack (report §3.3, §4.2 *Other samplers*)

```bash
python experiments/3_sampling_fixes/run.py            # ~2 min: samplers vs. an adaptive attacker
python experiments/3_sampling_fixes/floor_sampler.py  # ~37 min: contribution sampling with an additive floor
```

Both use `artifacts/models/mlp_mnist_full.npz` and attack its first hidden layer (512
neurons). The samplers are in `pvi.defences.sampling`; the adaptive attacker, who knows
the sampling rule and picks the neurons it visits least, is in `pvi.defences.adaptive`.
All detection probabilities are exact (`pvi.experiments.analysis`), for one path.

**`run.py`** → `artifacts/results/defence.json` (40 queries):

| Block | Report |
|---|---|
| `sweep.<sampler>.detection_evasive` | detection against the adaptive attacker: uniform 0.20%, static-importance 0.11%, gradient-saliency 0.016% (`detection_naive` is the plain single-neuron attack) |
| `zero_blind_spot` | contribution weighting: zeroing about 14 of 512 neurons flips the prediction (`mean_support` 14.35), detection 0 (`detection_contribution_weighted`), 2,000 of 2,000 real runs accepted (`protocol_accepts`; §4.2 *Other samplers*); the zeroed neurons are zero on 65% of natural inputs (`natural_zero_fraction_at_chosen_neurons`) |

**`floor_sampler.py`** → `artifacts/results/floor_sampler.json` (20 queries): contribution
weighting with an additive floor, `|w_ij| * (|a_i| + eps)`, measured on the tampered
traces against the better of the attacker's two attacks (`samplers.<name>.adversary_best`).
The best floor, `eps=1`, detects 2.0% (10.2× uniform's 0.20%); `eps=0.1` and `eps=10`
give 5.8× and 2.2×, and `eps=0.01` and `eps=100` fall below uniform (0.9×). At `eps=0`
it is plain contribution weighting, with detection 0 (§3.3).
