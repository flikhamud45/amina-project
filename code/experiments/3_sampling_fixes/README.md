# 3. Smarter sampling does not stop the attack (paper Sec. 3.3 and 4.2, *Other samplers*)

```bash
python experiments/3_sampling_fixes/run.py            # about 2 min on a CPU: samplers against an adaptive attacker
python experiments/3_sampling_fixes/floor_sampler.py  # about 37 min on a CPU: contribution sampling with an additive floor
```

Both use `artifacts/models/mlp_mnist_full.npz` and attack its first hidden layer (512 neurons). The
samplers are in `pvi.defences.sampling`; the adaptive attacker, who knows the sampling rule and picks
the neurons it visits least, is in `pvi.defences.adaptive`. All detection probabilities are exact
(`pvi.experiments.analysis`), for one path. `--queries` changes the number of queries. Proposition 3.1
(no sampler beats uniform sampling on every network) is proved in the paper; these scripts measure
the samplers on the MNIST MLP.

**`run.py`** → `artifacts/results/defence.json` (40 queries):

| Block | Paper |
|---|---|
| `sweep.<sampler>.detection_evasive` | detection against the adaptive attacker: uniform 0.20%, static importance (outgoing weight magnitude) 0.11%, gradient saliency 0.016% (`detection_naive` is the plain single-neuron attack) |
| `zero_blind_spot` | contribution weighting: zeroing about 14 of 512 neurons flips the prediction (`mean_support` 14.35), detection 0 (`detection_contribution_weighted`), 2,000 of 2,000 end-to-end runs accepted (`protocol_accepts`); the zeroed neurons are zero on 65% of natural inputs (`natural_zero_fraction_at_chosen_neurons`) |

**`floor_sampler.py`** → `artifacts/results/floor_sampler.json` (20 queries): contribution weighting
with an additive floor, `|w_ij| * (|a_i| + eps)`, measured on the tampered traces against the better
of the attacker's two attacks (`samplers.<name>.adversary_best`). The best floor, `eps=1`, detects
2.0% (10.2x uniform's 0.20%); `eps=0.1` and `eps=10` give 5.8x and 2.2x, and `eps=0.01` and `eps=100`
fall below uniform (0.9x). At `eps=0` it is plain contribution weighting, with detection 0. The paper
reports these with the floor written δ (the script's `eps`): "δ = 1 raises detection to 2% (10x uniform);
other floors give 0.9–6x".
