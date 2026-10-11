# EV: the deployment case (draft)

Plan 2, item EV (`SP2027_PLAN2.md`, Sec. 3.7). Code: `code/experiments/5_comparison/ev_breakeven.py`, inputs
`code/experiments/5_comparison/ev_inputs_draft.json` (every value names its source), test `tests/test_ev_breakeven.py`.
Output: `code/artifacts/results/ev_breakeven_draft.{json,pdf,tex}`. This is the draft of EV(f): built from the 2080 Ti,
Xeon and byte-model numbers; the final version swaps in the L40S/EPYC records of the F6 run (same script, new inputs).

## 1. Why

Reviewer A's deciding weakness (A-W1, A-Q1) is the deployment case: for which client is the protocol cheaper than
downloading the open weights once and re-executing? The honest answer is a narrow regime, and the figure states it.

## 2. The two panels (full Llama-2-7B)

**Panel 1, client bytes against the number of queries Q.** Downloading the int8 weights once (6.74 GB) against Q
mode-C proofs. Crossover `Q* = W / b`, with `b` = measured wire claims (196.4 MB at 64 tokens, job 1006979) plus the
byte model's fixed part (u, opened columns, Merkle paths) for the full model:

| 64 tokens | interactive | Fiat-Shamir |
|---|---:|---:|
| cnn16c (today's plan) | Q* 20.5 (328 MB per query) | Q* 17.3 (389 MB) |
| cnn18c (PS2 recommendation) | 25.0 (269 MB) | 22.3 (303 MB) |
| R64c | 27.0 (250 MB) | 24.5 (275 MB) |

At 2,048 tokens the claims alone are about the weights' size (6.43 GB), so Q* is about 1 for v0 and about 2.3 for V1.

**Panel 2, a Kpre client's time against re-execution, by link speed.** Break-even link `L* = 8 B / (t_r - t_v)`, with
the measured native CPU verifier `t_v` and RX's strongest exact re-execution `t_r` on the same 8 Xeon threads:

| Cell | B | t_v | t_r | L* |
|---|---:|---:|---:|---:|
| v0, 64 tokens, interactive / FS | 196 MB | 1.01 / 1.40 s | 4.08 s | 0.51 / 0.59 Gb/s |
| v0, 2,048 tokens, interactive / FS | 6.43 GB | 18.5 / 19.6 s | 93.2 s | 0.69 / 0.70 Gb/s |
| V1, 2,048 tokens, FS, verifier before V-V1N | 2.80 GB | 77.7 s (1-3 blocks, job 1009430) | 93.2 s | 1.44 Gb/s |
| V1, 2,048 tokens, FS, V-V1N parts 1-4 (torch) | 2.80 GB | 66.6 s (1-3 blocks, job 1009430) | 93.2 s | 0.84 Gb/s |

Reading: a CPU client with a link of about 1 Gb/s or faster verifies a Llama-2-7B prompt faster than it can re-execute
it, at every prompt length measured; below about 0.6-0.7 Gb/s re-executing is faster (if the client holds the
weights). V1 halves the bytes, but its CPU verifier is slow enough that its break-even link is worse than v0's, even
with V-V1N's torch parts (0.84 against 0.70 Gb/s; the earlier projection of 52.5 s, which gave 0.55 Gb/s, did not
hold on the Xeon). V1 becomes the better option on slow links only if its verifier falls below about 60 s
(L* = v0's 0.70 Gb/s at t_v = 93.2 - 22.4/0.70 = 61 s), which needs the native kernel of V-V1N.

## 3. Validation

* `tests/test_ev_breakeven.py`: both crossover formulas (the costs meet at the crossover; faster links and fewer
  queries win), and the draft inputs reproduce the plan's quoted crossovers (Q* 20.5 against the plan's 20.6;
  L* 0.70 Gb/s).
* The figure is drawn from the same rows as the JSON and the macros.

## 4. What remains (EV final, Thu-Fri 5-6 Nov)

* Swap the inputs for the L40S/EPYC records (F6): the EPYC's re-execution with VNNI and AVX512_BF16 is faster, so
  every break-even link rises.
* tab:options: the literature cells (GPU TEE overhead below 7%; CommitLLM 12-14% overhead, a key of about 48 MB for
  Llama 70B, a statistical guarantee) and A7's provider-overhead column; the target-client paragraph in Sec. 7.3.
* The GPU-verifier roofline and V1's bandwidth crossover from the L40S records; Wilson intervals on success counts;
  a test that every plotted point equals a table cell; one platform per table.
