# PS2: lower-rate commitment plans for mode C, under an explicit setup budget

Plan 2, item PS2 (`SP2027_PLAN2.md`, Sec. 3.4). No protocol code changes: the commitment plans `cnn16c`, `cnn18c` and
`R64c` already exist in `plans.py`. Code: `code/experiments/6_improvements/ps2_tables.py` (commit 9cc0bd6); measured run:
job 1006989 (s-004, RTX 2080 Ti node, git 12a188f).

## 1. Why

In mode C a query carries, besides the claims, a fixed part: the folded rows `u`, the opened columns of the
Reed-Solomon-encoded weights and their Merkle paths. It does not shrink with the prompt. At 64 tokens it is most of the
proof (reviewer A, A-W1: 124.7 MB of a 325 MB Llama-2-7B query are opened columns), and under Fiat-Shamir it grows,
because more columns must be opened. A lower code rate (a longer codeword per row) lets the verifier open fewer columns
for the same soundness, at the price of a larger one-time setup (encoding and hashing every row of every weight matrix).

## 2. What was measured

`v1_bytes.py --model llama2-7b --layers 2 --seq 64 2048 --modes C --policy cnn16c cnn18c R64c --prune-last --fs --lean
--lmax 24` (2 blocks of Llama-2-7B, pruned last block, Fiat-Shamir, 7 Freivalds repetitions), bytes per query and the
setup time:

| Plan | v0, 64 tokens | V1, 64 tokens | v0, 2,048 tokens | V1, 2,048 tokens | fixed part (u + columns + paths), 64 tokens | setup, 2 blocks |
|---|---:|---:|---:|---:|---:|---:|
| cnn16c | 23.27 MB | 20.44 MB | 269.5 MB | 126.5 MB | 15.22 MB | 91 s |
| cnn18c | 16.05 MB | 13.22 MB | 262.3 MB | 119.3 MB | 7.99 MB | 290 s |
| R64c | 13.93 MB | 11.10 MB | 260.2 MB | 117.2 MB | 5.88 MB | 1,186 s |

Every v0 query was accepted. The V1 columns are the cut's byte accounting (`cut_proof_bytes`); this job ran no full V1
proof (no `--full`).

* At 64 tokens the fixed part falls by 47% (cnn18c) and 61% (R64c) against cnn16c, and the whole v0 query by 31% and
  40%. At 2,048 tokens the claims dominate, so the gain is 3-4%.
* **Against the byte model** (`analytic.proof_bytes`, which `ps2_tables.py` uses for the full models), the measured fixed
  part is 0.7-1.3% *smaller* in all six cells (u 3% smaller: the compact 31-bit field encoding; columns 1.2% smaller;
  paths within a few percent either way). The model is therefore slightly conservative. The plan's criterion was
  "within 1.2%"; cnn16c and cnn18c deviate by 1.3%, in the safe direction.

## 3. The setup budget

The setup time scales with the encoded entries (`n_points x n_rows` summed over the plan's matrices). From the 2-block
run and the entry counts of the full model (ratio 12.4 / 12.9 / 13.4 for the three plans):

| Plan | entries, 2 blocks | entries, full model | setup, full Llama-2-7B (estimate, 2080 Ti node) |
|---|---:|---:|---:|
| cnn16c | 2.8 G | 35 G | about 19 min |
| cnn18c | 10.9 G | 140 G | about 62 min |
| R64c | 44.2 G | 592 G | about 4.4 h |

The plan's kill criterion was "cnn18c over 1 h on the 2080 Ti node": the estimate is at the limit (62 min). The setup is
a one-time cost per model and per plan, and the final timed cells run on the L40S node, whose encoder is faster.

## 4. Recommendation (decision due Wed 21 Oct)

* **cnn18c for the timed mode-C cells of the final package**, if the package builds each model's setup once and reuses it
  across cells (to be checked in F6, plan item 4); otherwise cnn16c for the timed cells and cnn18c for the byte rows.
* **R64c only for the byte rows**, measured on 2 blocks and extended with the byte model (conservative by about 1%), since
  a full-model setup takes hours.
* Kpre is unaffected (it opens no columns).

Open: regenerate TH-b's concrete-security table and D4's setup-proof parameters for the chosen plan.

## 5. How to run

    cd code && PYTHONPATH=src python experiments/6_improvements/ps2_tables.py --models gpt2 opt-1.3b llama2-7b llama2-13b qwen3-4b
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model llama2-7b --layers 2 --seq 64 --modes C \
        --policy cnn18c --prune-last --fs --lean
