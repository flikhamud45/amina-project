# 0. Train the MNIST models

```bash
python experiments/0_train_models/train.py            # ~15 min on 4 CPU threads; cached afterwards
```

Downloads MNIST to `code/data/` and trains the seven models of `pvi.zoo`, writing
`artifacts/models/<name>.npz` plus a JSON training report. Runs are seeded, and on CPU
they reproduce bit for bit. Re-running skips cached models; `--force` retrains, and
`--only NAME ...` trains a subset.

| Model | Used by |
|---|---|
| `mlp_mnist_full` (784–512–256–10) | the attack (2), the samplers (3), and the MLP of the benchmark (4) |
| `mlp_digits_01234` / `mlp_digits_05678` | reproduction: different tasks with one shared class |
| `mlp_mnist_halfA` / `mlp_mnist_halfB` | reproduction: same task, disjoint training data |
| `cnn_mnist_halfA` / `cnn_mnist_halfB` | reproduction: the same, on a CNN |

The quantised substitute of the reproduction is derived from `mlp_mnist_full` on the
fly.
