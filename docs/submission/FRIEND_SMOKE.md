# L40S machine: smoke test and rehearsal (before the full run)

Thanks for running this. Two short runs first, then (after we look at the results) the full run of about three days.
Every step that uses the GPU or describes the machine runs on an L40S node (an `srun` allocation), not on the login
node; `run.sh` itself runs on the login node and submits the jobs.

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

The `bench` tier needs the image datasets and the trained image models (neither is in the repository):

- CIFAR-10 in torchvision's layout: `export PVI_CIFAR_ROOT=<folder holding cifar-10-batches-py>` (the one of your
  earlier L40S runs), or download it once into `code/data/cifar10` (command below);
- ImageNet for the two 224-pixel ResNet-18 models: `export PVI_IMAGENET_ROOT=<ImageNet root>` (as in your earlier
  runs: class `i` of the sorted-synset order in folder `i + 1`, i.e. dogs in folders 152-269, cats 282-286 and the
  fox squirrel in 336). The dogs-vs-cats model and its dogs-vs-squirrels substitute (the substitute-model attack of the benchmark)
  are trained from it; the first use builds the caches `dogs_cats.pt` and `dogs_squirrels.pt` under
  `code/artifacts/cache`.

Then, on an L40S node (with the variables above exported in that shell), check the Python headers and train the six
models once (minutes each), from the repository root:

```bash
srun -p "$SBATCH_PARTITION" --gres="$SBATCH_GRES" -c 8 --mem=64G --pty bash
.venv/bin/python -c "import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"
# False: copy the Python 3.12 headers into a folder and export PVI_PY_INCLUDE=<folder> before running run.sh
.venv/bin/python -c "import torchvision; torchvision.datasets.CIFAR10('code/data/cifar10', download=True)"   # only without PVI_CIFAR_ROOT
cd code
for m in lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel; do
  PYTHONPATH=src:. ../.venv/bin/python -m experiments.costs.train_cnn --model $m --device cuda
done
ls artifacts/models/cnn/*.pt      # six models
exit
```

If you have the Llama-2 licence, the jobs must see your Hugging Face token on whichever node they land: a login
stored under a node-local `HF_HOME` is not visible elsewhere, so `export HF_TOKEN=<your token>` in the shell that runs
`run.sh` (the jobs inherit it). Then check on the same node that the token can fetch the gated checkpoint (the
rehearsal does not download it; the full run does):
`.venv/bin/huggingface-cli download meta-llama/Llama-2-7b-hf config.json`.

## 2. Smoke (up to an hour)

From the repository root, on the login node:

```bash
bash code/experiments/package/run.sh --platform smoke smoke
```

The smoke job asks for a whole node (`--exclusive`), as the full run's jobs will; please tell us how long it waited
in the queue. Wait until it has finished: `squeue -u $USER` lists no `sp-` job, and `logs/smoke/sp-smoke-*.out` ends with
`SMOKE OK`. Please do not start Sec. 3 while the smoke job is still queued or running: `run.sh` skips a job whose
name is already in the queue, so the rehearsal would lose its own smoke job. If the smoke failed, cancel your pending
`sp-` jobs (`scancel -u $USER -n <name>`, or `squeue -u $USER` and `scancel <id>`) and send us the log before running
again.

## 3. Rehearsal (about three hours, small cells of every script)

From the repository root (`--llama-licence` only if your Hugging Face account has the Llama-2 licence; otherwise
leave it out). The rehearsal's timings are not used, so its jobs share nodes (`--no-exclusive`); by default every job
waits for a whole node, which on an 8-GPU node can take long:

```bash
bash code/experiments/package/run.sh --platform rehearsal --rehearsal --optional --no-exclusive --llama-licence all
```

This only submits the 50 jobs. Wait until `squeue -u $USER` lists no `sp-` job (about three hours), then check:

```bash
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform rehearsal
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/rehearsal --platform rehearsal \
    --rehearsal --check --no-byte-model 2>&1 | tee ../logs/rehearsal/numbers_check.txt; echo "status ${PIPESTATUS[0]}"
cd ..
```

Expected: `status 0`.

## 4. Send back

From the repository root:

```bash
tar czf rehearsal.tgz --exclude=triton --exclude=native --exclude=setup_cache logs/smoke logs/rehearsal \
    code/artifacts/comparison/raw_rehearsal
srun -p "$SBATCH_PARTITION" --gres="$SBATCH_GRES" -c 8 bash -c \
    'nvidia-smi -q > env-nvidia.txt; lscpu > env-cpu.txt; free -g > env-mem.txt'
.venv/bin/pip freeze > env-pip.txt
```

and send `rehearsal.tgz` and the four `env-*.txt` files, or copy them into `$H/return/` (writable by everyone).
Please do not start the full run yet.
