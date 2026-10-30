# FR: fixes for the final run's four failures

Code: commit 6d9da0f (branch `sp2027-artifact`). Options `--host-claims`, `--host-claims-reserve`, `--gkr-memory` and
`--gkr-memory-budget` (`pvi.options`); `pvi/v0/query.py`, `pvi/core/hostmem.py`, `pvi/v1/protocol.py`,
`pvi/v1/gkr_triton.py`; `experiments/reexec/llamacpp.py`; `experiments/package/run.sh`.

## 1. Why

The final run on the L40S node, at commit 65ac80b, reported four failed jobs:

* the llama.cpp baseline, which needed `sentencepiece`;
* V1 on the full Llama-2-7B at 2,048 tokens, which ran out of GPU memory in the GKR;
* the full Llama-2-7B at 8,192 tokens, which ran out of GPU memory while holding its claims;
* the in-range attack on OPT-6.7B, which reached its 4 h limit twice.

One rule governs every fix: **no job may become slower unless it passes an explicit flag.** Every new code path is
off by default, and `run.sh` passes the flag only to the jobs that would otherwise fail. Every path gives the same
proof, byte for byte.

## 2. The fixes

* **Tokenizer.** `spm_pieces` reads `tokenizer.model` (a SentencePiece `ModelProto`) with a short protobuf reader.
  This step therefore needs neither `sentencepiece` nor `protobuf`. A test checks every piece, score and type against
  `sentencepiece` itself.
* **Claims in host memory (`--host-claims device|auto|host`, default `device`).**
  * At 8,192 tokens the lean GPU prover's int32 claims (40.6 GiB) and the weights (6.3 GiB) exceed the L40S's
    44.5 GiB.
  * `auto` keeps the claims on the GPU while they fit, beside the weights and a reserve of `--host-claims-reserve`
    GiB (default 6). It copies the rest to pinned host memory on a side stream as they are computed, and the encoder
    uploads them part by part.
  * The size check runs outside the timed region. The record's `claims_staged` gives the bytes moved.
  * `run.sh` passes `auto` to `l-llama7-8192-full` alone. That job serves only the check of the long-prompt
    extrapolation, which compares the verifier, re-execution and proof, not the prover's time. The paper says so in
    a footnote.
* **GKR memory (`--gkr-memory default|low|host|auto`, default `default`).**
  * During V1's GKR, the GPU prover kept the cut ops' sent values and every claim, and the Triton tree peaked at
    148 bytes a leaf. For Llama-2-7B at 2,048 tokens this comes to about 45.0 GiB.
  * `low` frees what the GKR no longer reads, and the tree's leaves and layers once they are combined (96 bytes a
    leaf, about 28.4 GiB). Measured, it is not slower.
  * `host` also keeps the claims the folds read in pinned host memory during the GKR. It is slower by a few seconds
    a query, and only the GPU-verifier cell of Llama-2-7B (about 63 GiB on the default path) needs it.
  * `auto` takes the first path whose predicted peak is within `--gkr-memory-budget` percent of the GPU (default 90).
  * `run.sh` passes `auto` to the full Llama-2-7B's V1 cells at 2,048 tokens and to the GPU-verifier V1 cells. It
    passes `auto` with `--gkr-memory-budget 100` to OPT-6.7B's (42.5 GiB predicted), which therefore stay on the
    default path unless the whole GPU would not suffice.
  * The record's `v1_full.gkr_memory` and `timed.v1.gkr_memory_all` give the path taken.
* **Attack time limit.** OPT-6.7B's two in-range passes are now two jobs, `a-inrange-opt67` (3 h) and
  `a-inrange-opt67-down` (6 h). Each can be resumed on its own.

## 3. Evidence

* **Proofs:** transcripts, bytes and verdicts are identical on every path.
  * CPU: tiny models in modes C and Kpre.
  * Titan Xp GPU prover, CPU verifier: Llama-2-7B blocks at 512 tokens, the same `proofs_sha` on every path.
  * Titan Xp GPU verifier: one block at 512 tokens and two at 256.
  * Staged claims: the same encoded bytes as claims kept on the GPU.
* **Default paths:** the code is unchanged apart from one decision call made before the timer, and timings match
  within node noise.
* **Limits of the evidence:** the full-size L40S peaks are predictions from measured per-leaf and per-claim sizes.
  The Titan Xp cannot run the Triton kernels.
* **Review:** a multi-agent review of the changes found five problems, all fixed:
  * OPT-6.7B's budget;
  * the READMEs' lists of cells;
  * a footnote value of 0;
  * the per-query path record;
  * a docstring.

## 4. Effect on the run

* 246 jobs, 669 h of job limits (one more attack job).
* Already finished jobs stay valid and are skipped when a line is resubmitted.
* `FIXES.md` in the handoff folder tells the person running the final run what to resubmit.
