# Phase 3 — GPU infrastructure and the first scale check

## What's built

- `src/pvi/training.py`: `TrainConfig.device` (`"cpu"`/`"cuda"`/`"auto"`), `resolve_device`,
  and best-effort CUDA determinism flags. Default stays `"cpu"` so every existing
  cached model is unaffected and stays bit-exactly reproducible (see the
  docstring's honest caveat: GPU training is *not* bit-exact reproducible the
  way CPU is — CUDA kernels reduce in a data- and GPU-dependent order).
- `scripts/train_models.py --device {cpu,cuda,auto}`.
- `src/pvi/zoo.py`: `mlp_architecture_large_for` (`784 -> 4096 -> 2048 -> 10`,
  8x `mlp_architecture_for`'s hidden widths) and `LARGE_MLP_SPEC`, deliberately
  excluded from `SETUPS`/`all_model_specs()` so the default CPU pipeline is
  untouched.
- `scripts/run_gpu_scale_check.py`: trains (or loads) the large model and
  re-runs the Theorem 4 sweep and the sumcheck prototype against it.

## The actual cluster GPU situation (useful to know before running anything else here)

This took real trial and error; recording it so it doesn't get rediscovered
the hard way next time.

1. **The machine's directly-visible GPUs (Titan X) are too old.** Compute
   capability 5.2; the PyTorch build originally installed (`torch==2.14.0+cu130`)
   only supports CC >= 7.5. Silently falls back to CPU with a warning rather
   than erroring, which is easy to miss.
2. **The account (`edo`) has three usable SLURM associations**:
   `killable` (broad access, but literally a preemptible partition — jobs on
   it get killed by the scheduler for other users' needs, not a bug),
   `gpu-bermano` (account `gpu-research`, restricted to group `cs_gpu-bermano`,
   one dedicated node `n-205`, 8x RTX 2080 Ti, 5-day time limit, **not**
   preemptible), and `studentkillable` (account `gpu-students`, also
   preemptible). `gpu-bermano` is the stable choice for anything that needs to
   run to completion.
3. **Newer GPUs on `killable` (A5000, A6000, L40S, H100...) had a different
   problem**: driver 535.288.01 (seen on both an A5000 node and `n-205`) only
   supports up to CUDA 12.2, but the installed torch was built for CUDA 13.0.
   `torch.cuda.is_available()` returns `False` with a driver-version warning,
   not an error.
4. **Fix**: reinstall torch/torchvision from the `cu121` wheel index
   (`https://download.pytorch.org/whl/cu121`), which is compatible with driver
   535.x and with every GPU generation on this cluster from Turing (RTX 2080 Ti,
   CC 7.5) up. Confirmed working: `torch==2.5.1+cu121`,
   `NVIDIA GeForce RTX 2080 Ti`, CUDA available, matmul on-device.
5. **This login node kills long-running foreground/background shell commands
   somewhat unpredictably** ("system is running low on memory"), including
   ones with negligible actual memory use (a bare `squeue` polling loop was
   killed the same way as a multi-GB `pip install`). It is not tied to the
   SLURM job's own resource usage — a non-preemptible `gpu-bermano` job got
   killed the same way as a `killable` one, and raising `--mem` to 48G made no
   difference. Whatever triggers it appears to be about *this session's own
   foreground/background process*, not the remote job.
   **Workaround that reliably worked**: submit real work as a detached
   `sbatch` batch job (writes to a log file, runs independently of this
   shell), and use the `Monitor` tool (not `Bash`'s `run_in_background`) to
   wait for completion via a one-shot polling loop. Every `sbatch` job
   submitted this way ran to completion; every long-lived `srun`/background
   `Bash` wait attempted before switching to this pattern was killed.
6. **This cluster's `srun` refuses to exec a shell directly**
   ("Running shells is not permitted. Use containers instead.") — pass the
   target binary directly (`srun ... python script.py`), not `srun bash -c '...'`.
7. **`/tmp` is node-local, not shared across the cluster.** A file written to
   `/tmp` on the login node is invisible to a `srun`/`sbatch` job running on a
   compute node. Use a path under the NFS-mounted project directory (or
   `/home/sharifm/...` generally) for anything that needs to cross that
   boundary.

## The scale-up result

`scripts/run_gpu_scale_check.py --device cuda --force`, run as a detached
`sbatch` job on `n-205` (RTX 2080 Ti): trained `LARGE_MLP_SPEC`
(`784 -> 4096 -> 2048 -> 10`) for real — 6 epochs, full 60k-example MNIST
training set, train accuracy 98.91%, test accuracy 97.11% (a legitimate,
converged model, not the earlier CPU smoke-test's undertrained one).

Re-running Revision 1's two key measurements against it, at **8x the width**
Theorem 4 and the sumcheck prototype were originally measured at:

| | width 512 (original) | width 4096 (this run) |
|---|---|---|
| `\|U\|` / width | 462/512 (90%) | 3717/4096 (91%) |
| Theorem 4 sweep | **withdrawn — measured on the honest trace, see `DEFENCE_NOTES.md` §6** | same flaw |
| sumcheck detection vs. uniform | 1.000000 vs 0.00195 | 1.000000 vs 0.000244 |

Both conclusions hold, essentially unchanged in character, at 8x scale with a
genuinely trained model. This is a confirmation that the theorems' own
framing (stated in terms of width `N`, never assuming MNIST-MLP scale) was
correct, not new theory — but it's exactly the kind of check a reviewer would
ask for, and it wasn't free: the CPU pipeline could not have produced a
properly-trained model at this width in reasonable time, which was the whole
motivation for this phase.

## What's not done yet

- CIFAR-10 / the CNN architecture at GPU scale (promised in the intermediate
  report, still MNIST-only).
- Multi-seed repeats for error bars on the floor/threshold calibrations.
- Anything on `killable`/`studentkillable` partitions specifically — everything
  above ran on the dedicated `gpu-bermano` node, which won't be available to
  every user with the same association.
