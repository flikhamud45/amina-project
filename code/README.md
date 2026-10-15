# Proofs of inference: spot-checking verifiers are not sound, and a fully sound protocol

The artifact of the paper. A provider that serves an open-weight model can change chosen answers; a proof of
inference lets the client check each answer. This code

* shows that spot-checking proofs, which re-check a sample of a committed execution trace, accept a provider that
  changes a single neuron and recomputes the later layers honestly (the path protocol of Anchuri et al., on MNIST
  networks, image models and real LLMs up to OPT-6.7B), and that no sampling rule closes the gap;
* implements a fully sound protocol: an exact int8 model whose every matrix product is checked with Freivalds'
  algorithm against a Reed-Solomon/Merkle commitment of the weights, with a hash function alone (no SNARK, no
  trusted hardware, no trusted setup), in three client modes (C: the 32-byte roots only; K: the weights; Kpre: a
  key precomputed from them), under Fiat-Shamir, with generated responses, and with V1, a variant with smaller
  proofs (requantised int8 values proved with logUp-GKR);
* measures the protocol on one platform against re-execution, unverified serving and published systems, and
  produces every table, figure and number of the paper from the run's records.

## Layout

| Folder | Contents |
|---|---|
| [`src/pvi/`](src/pvi/README.md) | the library: the field and graphs (`core`), commitments and plans (`commit`), protocol v0 (`v0`), V1 (`v1`), integer models (`models`), native and Triton kernels (`kernels`), the path protocol and samplers (`baseline`), the attacks (`attacks`, `mnist`), the implementation options (`options`) |
| [`experiments/`](experiments/README.md) | one package per part of the paper; each README maps the paper's results to commands; `package/run.sh` is the run that produces them |
| [`tests/`](tests/) | the test suite (exactness of every kernel against the reference semantics, protocol soundness checks, every tamper rejected at its check, the experiment harnesses) |
| [`tools/`](tools/) | the generator of the Triton GKR prover (`gen_gkr_triton.py`) and the anonymous snapshot for review (`export_anonymous.py`) |
| [`artifacts/`](artifacts/README.md) | the stored MNIST networks (`models/`), the benchmark's records and tables (`comparison/`) and the stored results of the other experiments (`results/`); per file, the command that wrote it, the platform it was measured on and what reads it |

## Installation

Python 3.11 or 3.12 (the results: 3.12), PyTorch 2.5.1: the pinned requirements need it (NumPy 2.5.2 needs Python
3.11 or later, and PyTorch 2.5.1 ships wheels up to 3.12); the library alone, installed with `pip install -e code`
and the ranges of `pyproject.toml`, also runs on 3.10. On a CUDA machine install the matching build first:

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r code/requirements.txt                 # the library, the tests, the attack on MNIST
.venv/bin/pip install -r code/requirements-llm.txt             # experiments on real checkpoints
```

Optional, for speed only (every path computes the same integers): a C++ compiler in `CXX` (g++ 9 or later) builds
the native CPU kernels on first use, into `PVI_NATIVE_DIR` (default `~/.cache/pvi`); Triton (part of PyTorch's CUDA
builds) runs the fused GPU kernels, and needs the Python headers (on hosts without them, `PVI_PY_INCLUDE` names a
folder holding a copy). Real checkpoints and WikiText-2 are downloaded from the Hugging Face hub into `HF_HOME`;
the Llama-2 checkpoints need your own licence and login.

## Tests

```bash
cd code
PYTHONPATH=src:. ../.venv/bin/python -m pytest -q                       # CPU; GPU tests skip without a GPU
PVI_TEST_DEVICE=cuda PYTHONPATH=src:. ../.venv/bin/python -m pytest -q  # on a GPU machine: the GPU kernels too
```

The full suite takes about 80 minutes with 8 CPU threads and a GPU.
`../.venv/bin/python tools/gen_gkr_triton.py --check` (from `code/`) confirms that the Triton GKR prover is its
generator's output.

## Reproducing the paper

* **Every number, on one platform**: [`experiments/package/README.md`](experiments/package/README.md). One command
  submits every job of the run (about three days on one NVIDIA L40S node with 8 CPU threads); a dry run lists the
  jobs, and a rehearsal runs every script of every tier on small cells in about three hours on a smaller GPU.
* **From the records to the paper**: [`experiments/paper/README.md`](experiments/paper/README.md) turns the run's
  records into the table bodies and text macros of the paper's LaTeX source.
* **One result at a time**: each folder of [`experiments/`](experiments/README.md) maps the paper's results to
  commands, with a small version that runs on a laptop CPU.

Every script takes every parameter as a command-line flag (`--help`); the defaults are the paper's configuration.
Every timed record carries its environment: the commit, the host, the library versions and the implementation
options in force.

## Licence

MIT (see [`LICENSE`](LICENSE)). The stored model weights are trained by `experiments/attack/train_mnist.py`; real
checkpoints are not redistributed.
