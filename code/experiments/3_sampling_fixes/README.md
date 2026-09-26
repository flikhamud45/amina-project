# 3. Smarter sampling does not stop the attack (report §3.2, §4.2 *Smarter sampling*)

```bash
python experiments/3_sampling_fixes/run.py            # samplers vs. an adaptive attacker
python experiments/3_sampling_fixes/floor_sampler.py  # contribution sampling with a uniform floor
```

Both need `experiments/0_train_models`. The samplers are in `pvi.defences.sampling`;
the adaptive attacker, who knows the sampling rule and picks the neurons it visits
least, is in `pvi.defences.adaptive`.

**`run.py`** → `artifacts/results/defence.json`:

| Block | Report |
|---|---|
| `sweep` | detection of uniform (0.20%), static-importance (0.11%) and gradient-saliency (0.006%) sampling against the adaptive attacker |
| `zero_blind_spot` | contribution weighting: zeroing 13 of 512 neurons flips the prediction, detection 0, 2,000 of 2,000 real runs accepted (Table 1, last row); 65% of natural activations are zero |
| `minimax` | Theorem 3.1 checked numerically: the least-visited flipping neuron bounds detection by `1/\|U\|` |
| `budget` | detection against the number of weight rows opened |

**`floor_sampler.py`** → `artifacts/results/floor_sampler.json`: contribution weighting
with an additive floor `eps`, measured on the tampered traces against the attacker's
best attack. Around `eps = 1` it beats uniform sampling by about 10×, and detection
stays at a few percent (§3.2).
