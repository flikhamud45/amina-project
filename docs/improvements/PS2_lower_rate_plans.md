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

### 4.1 Plan item 4: the package does not reuse setups (checked 12 Oct, git 207b798)

Every script of the final package that runs a mode-C cell builds the weight commitment itself
(`bench.py`'s `_commit`, called once per invocation and timed as `commit_total`; likewise `perf.py`, `gen_bench.py`,
`serve_ab.py` and `v1_bytes.py` through `protocol.commit_graph`). Nothing is cached on disk, so each sbatch job of
a full model pays one setup. The commitment itself is small (Merkle trees and per-member column digests; the
prover recomputes opened columns from the int8 weights), so a cache would be possible, but it does not exist.

The dry run (`F6_dryrun.txt`) has these full-model mode-C invocations, all with `PVI_C_POLICY=auto`:

| Model | invocations | `auto` chooses | encoded entries per setup, `auto` | cnn18c |
|---|---:|---|---:|---:|
| Llama-2-7B | 11 (serve, c/cg at 64 and 2,048, c at 1, g-v0, gen, three batch cells) | cnn16c | 34.9 G | 139.6 G |
| Llama-2-13B | 5 | cnn17c | 108.7 G | 217.4 G |
| Qwen3-4B | 4 | cnn17c | 50.0 G | 98.7 G |
| OPT-6.7B | 3 | cnn17c | 69.8 G | 139.6 G |
| total | 23 | | 1,337 G | 3,436 G |

(`auto` takes the candidate of fewest non-claim bytes within `max(2 x the paper plan's entries, 2^34)`; GPT-2 and
OPT-125M get cnn18c, OPT-1.3B cnn16c.) At the 2-block run's effective rate (cnn18c for the full Llama-2-7B in about
62 min on the 2080 Ti node, Sec. 3), the package's setups take about 10 h of 2080 Ti-class encoding under `auto`
and about 25 h under cnn18c everywhere: 2,099 G more entries, about 15 h, which the L40S shortens by its encoder's
speed-up but which would still be a large share of the run window.

**Consequence (the plan's own rule).** Since setups are not reused, the timed mode-C cells keep `auto` (cnn16c for
Llama-2-7B, cnn17c for the larger models), and cnn18c and R64c appear as byte rows (2-block measurements and the
byte model, within 1.2%). The paper states which plan each table uses. Two ways to time cnn18c instead, for the
decision on Wed 21 Oct [team]: (i) set `PVI_C_POLICY=cnn18c` and accept the longer setups; (ii) add a setup cache
(the trees and member digests, keyed by the graph's digest, the plan and the rate, on node-local disk; about 0.5 d,
with a test that a cached commitment gives byte-identical proofs). Neither is needed for the submission's numbers.

Open: regenerate TH-b's concrete-security table and D4's setup-proof parameters for the chosen plan.

## 5. How to run

    cd code && PYTHONPATH=src python experiments/6_improvements/ps2_tables.py --models gpt2 opt-1.3b llama2-7b llama2-13b qwen3-4b
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model llama2-7b --layers 2 --seq 64 --modes C \
        --policy cnn18c --prune-last --fs --lean
