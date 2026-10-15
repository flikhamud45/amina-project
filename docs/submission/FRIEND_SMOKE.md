# L40S machine: smoke test and rehearsal (before the full run)

Thanks for running this. Two short runs first, then (after we look at the results) the full run of about three days.

## 1. Setup (once)

The code is the branch `sp2027-artifact` at commit `1a7ce39`, as a git bundle next to this file
(`pvi-sp2027-artifact-1a7ce39.bundle`; `SHA256SUMS` holds its checksum). Clone from the bundle (copy it to the L40S
machine first if that machine does not see this folder); the records name the commit, so please run from a clone,
not from an unpacked copy:

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)
git clone -b sp2027-artifact $H/pvi-sp2027-artifact-1a7ce39.bundle pvi && cd pvi && git log --oneline -1   # 1a7ce39
python3.12 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r code/requirements.txt -r code/requirements-llm.txt
export SBATCH_PARTITION=<the L40S partition> SBATCH_GRES=gpu:l40s:1     # or add --local to every run.sh call
export HF_HOME=<node-local folder with about 60 GB free>
export CXX=$(command -v g++)                                            # g++ 9 or later
```

If `python -c "import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"`
prints `False`, copy the Python 3.12 headers into a folder and `export PVI_PY_INCLUDE=<folder>`.

The `bench` tier needs the image datasets and the trained image models (neither is in the repository):

- CIFAR-10 in torchvision's layout: `export PVI_CIFAR_ROOT=<folder holding cifar-10-batches-py>` (the one of your
  earlier L40S runs), or download it once into `code/data/cifar10` with
  `.venv/bin/python -c "import torchvision; torchvision.datasets.CIFAR10('code/data/cifar10', download=True)"`;
- ImageNet for the dogs-vs-cats ResNet-18: `export PVI_IMAGENET_ROOT=<ImageNet root>` (as in your earlier runs);
  the first use builds a cache `dogs_cats.pt` under `code/artifacts/cache`.

Then train the models once (minutes each on the L40S), from `code/`:

```bash
for m in lenet5 vgg11 vgg16 resnet18_cifar resnet18_224; do
  PYTHONPATH=src:. ../.venv/bin/python -m experiments.costs.train_cnn --model $m --device cuda
done
```

`run.sh all` stops before submitting anything if a model or dataset is missing (`code/experiments/package/README.md`,
Sec. 1).

## 2. Smoke (up to an hour)

```bash
bash code/experiments/package/run.sh --platform smoke smoke
```

Expected: `SMOKE OK` at the end of `logs/smoke/sp-smoke-*.out`.

## 3. Rehearsal (about three hours, small cells of every script)

```bash
bash code/experiments/package/run.sh --platform rehearsal --rehearsal --optional --llama-licence all
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform rehearsal
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/rehearsal --platform rehearsal \
    --rehearsal --check --no-byte-model; echo "status $?"
```

Expected: `status 0`. `--llama-licence` only if your Hugging Face account has the Llama-2 licence (logged in with
`huggingface-cli login`); otherwise leave it out.

## 4. Send back

```bash
tar czf rehearsal.tgz --exclude=triton --exclude=native --exclude=setup_cache logs/smoke logs/rehearsal \
    code/artifacts/comparison/raw_rehearsal
nvidia-smi -q > env-nvidia.txt; lscpu > env-cpu.txt; free -g > env-mem.txt; .venv/bin/pip freeze > env-pip.txt
```

and send `rehearsal.tgz` and the four `env-*.txt` files, or copy them into `$H/return/` (writable by everyone).
Please do not start the full run yet.
