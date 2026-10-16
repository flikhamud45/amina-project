# L40S machine: smoke test and rehearsal (before the full run)

Thanks for running this. Two short runs first, then (after we look at the results) the full run of about three days.
Every step that uses the GPU or describes the machine runs on an L40S node (an `srun` allocation), not on the login
node; `run.sh` itself runs on the login node and submits the jobs.

## Round 2 (16 October): after your first smoke

Thank you for the report; the diagnosis was right. Commit `8945367` launches the attention kernel with `num_stages=1`, as
you proposed, and documents `TMPDIR` in the run package's README (it also sends sm_90 GPUs to the unfused attention,
since Triton 3.1 cannot compile the fused kernel there; the L40S is not affected). On our side, every Triton kernel
that the test suite and a whole rehearsal compile was also compiled for sm_80 and sm_89, ahead of time on a 2080 Ti:
with the fix none fails, and without it the attention kernel fails as on your nodes. This branch also has the
`b0ec1d9` changes (the CPU-timing jobs are pinned and check the native kernels). To continue in your existing setup
(your environment file, including `TMPDIR`, stays as it is):

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)
git -C <your clone> fetch $H/pvi-sp2027-artifact-8945367.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD && git -C <your clone> log --oneline -1   # 8945367
```

then Sec. 2 (the smoke) and, once it ends in `SMOKE OK`, Sec. 3 (the rehearsal); please also tell us how long the
smoke job waited in the queue.

**Llama-2.** The Llama-2 cells must run under your own Hugging Face account and token; please do not use the
teammate's token stored on the account. Please request access to both `meta-llama/Llama-2-7b-hf` and
`meta-llama/Llama-2-13b-hf` with your account now (each repository has its own request form, and approval can take
some time). Until both are granted, run the rehearsal without `--llama-licence` and tell us; the full run needs them.

## 1. Setup (once)

The code is the branch `sp2027-artifact` at commit `8945367`, as a git bundle next to this file
(`pvi-sp2027-artifact-8945367.bundle`; `SHA256SUMS` holds its checksum). Clone from the bundle (copy it to the L40S
machine first if that machine does not see this folder); the records name the commit, so please run from a clone,
not from an unpacked copy:

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)
git clone -b sp2027-artifact $H/pvi-sp2027-artifact-8945367.bundle pvi && cd pvi && git log --oneline -1   # 8945367
python3.12 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r code/requirements.txt -r code/requirements-llm.txt
export SBATCH_PARTITION=<the L40S partition> SBATCH_GRES=gpu:l40s:1     # or add --local to every run.sh call
export HF_HOME=<folder the nodes see, about 100 GB free>
export TMPDIR=<folder on a disk the nodes see>                         # if the nodes' /tmp is full
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

The run includes the Llama-2 cells, under your own Llama-2 licence: your Hugging Face account needs access to both
`meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-hf` (each repository has its own gate). The two checkpoints
add 40 GB to the 47 GB of the other models, hence the 100 GB of `HF_HOME` (please check with `df -h` on an L40S
node). The jobs must see your Hugging Face token on whichever node they land: a login stored under a node-local
`HF_HOME` is not visible elsewhere, so `export HF_TOKEN=<your token>` in the shell that runs `run.sh` (the jobs
inherit it). Then check on the same node that the token can fetch both gated repositories (the rehearsal downloads
only the 7B configuration and tokenizer; the full run downloads both checkpoints):

```bash
for m in Llama-2-7b-hf Llama-2-13b-hf; do .venv/bin/huggingface-cli download meta-llama/$m config.json; done
```

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

From the repository root, with `HF_TOKEN` exported (`--llama-licence` runs one job that reads the gated
repository from a compute node). The rehearsal's timings are not used, so its jobs share nodes (`--no-exclusive`); by default every job
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
