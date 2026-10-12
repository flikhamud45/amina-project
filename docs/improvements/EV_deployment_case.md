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
| V1, 2,048 tokens, FS, V-V1N with the native products | 2.80 GB | 45.4 s (2-3 blocks, job 1011440) | 93.2 s | **0.47 Gb/s** |

Reading: a CPU client with a link of about 1 Gb/s or faster verifies a Llama-2-7B prompt faster than it can re-execute
it, at every prompt length measured; below about 0.6-0.7 Gb/s re-executing is faster (if the client holds the
weights). V1 halves the bytes; with V-V1N's native products kernel its verifier takes 45.4 s (2.3x v0's), and its
break-even link against re-execution falls to 0.47 Gb/s, below v0's 0.70 (with the torch parts alone it was 0.84
Gb/s at 66.6 s; the line is crossed at about 61 s of verifier time, t_v = 93.2 - 22.4/0.70). Between the two proofs,
for the client alone (transfer plus verification, the prover's time not counted), V1 is faster on links slower than
8 (6.43 - 2.80) GB / (45.4 - 19.6) s = 1.1 Gb/s.

## 3. Validation

* `tests/test_ev_breakeven.py`: both crossover formulas (the costs meet at the crossover; faster links and fewer
  queries win), and the draft inputs reproduce the plan's quoted crossovers (Q* 20.5 against the plan's 20.6;
  L* 0.70 Gb/s).
* The figure is drawn from the same rows as the JSON and the macros.

## 3a. In the paper (12 Oct 2026)

Integrated after a read-only drafting agent built the changes in a temporary copy and checked the page count; every
number was then re-read from `ev_breakeven_draft.json` (regenerated) and the docs.

* **Figure** `paper_sp2027/figures/ev_breakeven.pdf` (Figure 2, one column, 3.375 x 1.9 in, the paper's 7 pt
  style; `ev_breakeven.py --figure`): left, client bytes against Q for mode C at 64 and 2,048 tokens and V1 at
  2,048 under the final plan (cnn18c, Fiat-Shamir) against the 6.74 GB download; right, the Kpre client's time for a
  2,048-token query (v0, and V1 with the native verifier, 45.4 s) against the fp32 re-execution (93.2 s) by link.
  The inputs now list cnn18c first and keep the superseded V1 verifiers (77.7 s, 66.6 s) under `_superseded`.
* **Target-client paragraph** (`discussion.tex`, "Who the protocol serves", replacing "Options for a client"):
  Q* about 22 at 64 tokens (cnn18c, FS: 302.9 MB per query; 25.0 interactive), 1.03 at 2,048 tokens and 2.32 with
  V1; Kpre break-even links 0.59-0.70 Gb/s, V1 0.47 Gb/s. A Break-even paragraph in Sec. 7.4 states the condition
  `L > 8B / (t_r - t_v)`.
* **tab:options:** the V1 row is measured (key, 2.8 GB at 2,048 tokens, 45.4 s under Fiat-Shamir); zkLLM cited;
  footnote f gives the provider's cost over unverified serving (GPU TEE below 7%, CommitLLM 12-14%, ours 9.8-10.6x
  in Kpre and 19-154x in mode C over fp16 on one 2080 Ti, Llama-2-7B, A7 Sec. 4); the GPU fp16 cells take A7's
  harness (0.034 / 0.64 s) so that each row uses one harness. The GPU TEE cells stay "--" (no citable attestation
  size or client time in the bibliography).
* **Cuts that keep the build at 18 pages** (body to p. 13): the Bandwidth-crossover paragraph (its per-cell values
  stay in the extended version; its "about 0.2 Gb/s" had no current source), the int8-engine sentence (footnote a
  keeps it), the cross-GPU fp16 sentence, the real-weight V1 ratio sentence (in `ext_v1.tex`), and the "about 21
  short ones" sentence (it contradicted 22). The key time (12.7 s) and "prove later, on request" moved to the
  extended version's "Ours." paragraph with the integer engine's floor (4-10x over fp16 on OPT-1.3B).
* **Corrections:** the 16,384-token comparison quotes 4.7x at 2,048 tokens (both non-interactive; was 5.0x,
  interactive); the proof reaches the weights' size at about 2,100 tokens (was 2,050).
* Build: main 18 pages, extended 52, no undefined references or overfull boxes.

Still open (EV final): the L40S/EPYC numbers through the same script (F6), and the plan and mode of tab:options'
mode-C row (today cnn16c interactive from earlier runs; the final run is cnn18c under Fiat-Shamir).

## 4. What remains (EV final, Thu-Fri 5-6 Nov)

* Swap the inputs for the L40S/EPYC records (F6): the EPYC's re-execution with VNNI and AVX512_BF16 is faster, so
  every break-even link rises.
* tab:options: the literature cells (GPU TEE overhead below 7%; CommitLLM 12-14% overhead, a key of about 48 MB for
  Llama 70B, a statistical guarantee) and A7's provider-overhead column; the target-client paragraph in Sec. 7.3.
* The GPU-verifier roofline and V1's bandwidth crossover from the L40S records; Wilson intervals on success counts;
  a test that every plotted point equals a table cell; one platform per table.
