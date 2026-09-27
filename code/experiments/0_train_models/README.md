# 0. Train the MNIST models

```bash
python experiments/0_train_models/train.py            # skips the committed models
python experiments/0_train_models/train.py --force    # retrains all seven, ~2 min on 8 CPU threads
```

Trains the seven models of `pvi.zoo` on MNIST (downloaded to `code/data/`), writing
`artifacts/models/<name>.npz` plus a JSON training report. `--only NAME ...` trains a
subset.

**The models the report used are committed** in `artifacts/models/`, with their hashes
in `MODELS.sha256` (`cd artifacts/models && sha256sum -c MODELS.sha256`). Experiments 1–4
load them, so this step is needed only to retrain. Training is seeded, but it is exactly
reproducible only on one platform: another machine, library version or thread count
gives slightly different, equally accurate weights, and the MNIST numbers of the report
then move in the last digits.

| Model | Used by |
|---|---|
| `mlp_mnist_full` (784–512–256–10) | the attack (2), the samplers (3), and the MLP of the benchmark (4) |
| `mlp_digits_01234` / `mlp_digits_05678` | reproduction: different tasks with one shared class |
| `mlp_mnist_halfA` / `mlp_mnist_halfB` | reproduction: same task, disjoint training data |
| `cnn_mnist_halfA` / `cnn_mnist_halfB` | reproduction: the same, on a CNN |

The quantised substitute of the reproduction is derived from `mlp_mnist_full` on the
fly.
