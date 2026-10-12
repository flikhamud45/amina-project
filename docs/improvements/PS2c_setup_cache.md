# PS2 (c). A setup cache for the weight commitment, and cnn18c for every timed mode-C cell

**Plan item:** PS2's decision (team, 12 Oct: option (c)). **Code:** `code/src/pvi/fullcheck/setup_cache.py`,
`protocol.commit_graph(cache=...)` (c1bc76d); tests `code/tests/test_setup_cache.py`; measurement
`code/experiments/6_improvements/setup_cache_check.py`. **Status:** implemented and tested (9 tests on the laptop);
validation on the full Llama-2-7B: [CLUSTER, job 1015688].

## 1. Why

In mode C a query sends opened columns of the committed weights: 124.7 MB of a 325 MB Llama-2-7B query at 64 tokens
(reviewer A, A-W1). A lower-rate plan opens fewer columns: under Fiat-Shamir cnn18c takes the 64-token Llama-2-7B
query from about 383 to about 306 MB (-20%, byte model; R64c -29%). The price is the one-time setup, which encodes
every row of every weight matrix at the plan's codeword length and hashes every column: 34.9 G encoded entries for
Llama-2-7B under cnn16c (about 19 min on the 2080 Ti node), 139.6 G under cnn18c (about 62 min).

The final package did not reuse setups (PS2 doc Sec. 4.1): each of its 23 full-model mode-C jobs encoded its model
again, 1,337 G entries under `auto` and 3,436 G under cnn18c everywhere, about 15 h more at 2080 Ti speed. Option (c)
removes the repetition, so cnn18c costs one setup per model.

## 2. What was built

`commit_graph` spends almost all its time computing the leaf digests of the commitment's Merkle trees:

* a group of matrices under one tree: leaf `c` = `group_leaf(tag, c, [column digest c of every member])`;
* a matrix on a tree of its own (the report's layout): its `column_leaf` digests;
* a lookup table: the `table_leaf` digest of every row (and padding leaves to a power of two).

These depend only on the committed integers (int8 weights, int64 biases), the tags and the plan's layout. The cache
stores them and nothing else:

* **Key** (`setup_cache.key`): BLAKE2b-256 over a format version (`pvi/setup-cache/v1`, bumped if a leaf definition
  changes), the base rate, the plan's policy and layout (groups with their members in leaf order; every matrix's
  layout, row count, row length, codeword length and members; the lookup tables) and, per weight op in graph order,
  its name, shape and a BLAKE2b-256 of its weights and of its bias. Different weights, biases (calibration), pruning,
  plan or rate give a different key (tested).
* **File**: `<folder>/<key>.npz` with one `uint8 [n, 32]` array per tree and the tree names as JSON, written by
  `numpy.savez` (no pickle) to a temporary name and renamed into place. Sizes under cnn18c: 36 MB (GPT-2, OPT-125M),
  44 MB (OPT-1.3B, OPT-6.7B), 51 MB (Llama-2-7B, 13B), 67 MB (Qwen3-4B).
* **Read**: `commit_graph` computes the key, reads the file and rebuilds every tree with `MerkleTree(leaves)`; the
  members of a group are the tree-less member commitments the fresh path builds anyway (they hold references to the
  graph's weights), so the result is the same object graph as a fresh commitment.
* **Check on reading**: for each tree, 4 random leaves are recomputed from the weights (the encoded columns at those
  points, exactly as an opening computes them, and their digests; for a table, the rows' digests) and compared with
  the stored ones. A mismatch, a missing tree, a wrong version or an unreadable file (including a truncated zip) makes
  it a miss: the commitment is computed afresh and the file rewritten.
* **Opt-in**: `commit_graph(cache=None)` reads `PVI_SETUP_CACHE`; unset (the default everywhere but the final
  package) means no cache, so every existing test, stored cell and digest is unchanged. `cache=False` never uses it.
* **Reporting**: the result's `setup_cache` is `"off"`, `"miss"` (computed, then stored) or `"hit"`. `bench.py`
  records a hit's time as `commit_cached_load`, never as `commit_total` (which `text_numbers.py` reads as the setup
  time); `perf.py`, `gen_bench.py`, `serve_ab.py` and `v1_bytes.py` write `setup_cache` next to their `commit_s` /
  `setup_s`. `setup_timing.py` (the setup tier, D4) passes `cache=False`: it always measures the setup itself.

## 3. Why it is safe

The cache is a prover-side shortcut. What leaves the prover (the roots in `C_M`, the opened columns, the
multiproofs) is that of a fresh commitment: the tests compare the roots, every tree level, and the bytes and verdicts
of interactive and Fiat-Shamir queries under the fresh and the cached commitment, for the report's layout and four
plans (R8, R8c, tightc, cnn12c) on tiny GPT, Llama and Qwen3 graphs. A wrong stored leaf could never make a verifier
accept a wrong claim: the verifier recomputes the leaf of every opened column from the column's values and checks the
path to the root, so a wrong leaf would only make honest proofs fail, and the check on reading catches that before it
happens (a tree with every leaf altered is detected and recomputed, tested).

## 4. The package

`sp2027.sh` exports `PVI_SETUP_CACHE=logs/$PVI_PLATFORM/.setup_cache` (empty disables it) into every job, including
the `strong_gpu.sh` jobs it submits, and the default plan of every timed mode-C cell is now **cnn18c**
(`PVI_C_POLICY`, formerly `auto`, which chose cnn16c for Llama-2-7B and OPT-1.3B and cnn17c for the larger models).
With the cache the run encodes about 650 G entries in all (one cnn18c setup each for GPT-2, OPT-125M, OPT-1.3B,
OPT-6.7B, Qwen3-4B, Llama-2-7B and Llama-2-13B), against 1,337 G for `auto` without it. The dry runs list 128 jobs
(`F6_dryrun.txt`), 46 of them carrying the cache variable.

Graphs built for different prompt lengths share their weights; whether their biases (calibrated) are equal decides
whether they share one cache entry, and the key keeps them apart if not.

## 5. Measured

Laptop (Windows, torch 2.2, 2 threads), GPT-2 with one block and the full model's plan (cnn16c):

| commitment | time | roots |
|---|---:|---|
| uncached | 120.1 s | |
| miss (computed and stored, 12.6 MB) | 90.3 s | same |
| hit (read, spot check, rebuild) | 1.65 s | same |

A Fiat-Shamir query proves and verifies under the miss and the hit with the same bytes (1,610,856), both accepted.

Cluster (RTX 2080 Ti node s-004, job 1015688, git c1bc76d; `tests/test_setup_cache.py` and `tests/test_plans.py`
with the full suite first: 562 passed). Llama-2-7B under cnn18c (6 trees of 2^18 columns), random weights, 64-token
Fiat-Shamir query:

| Llama-2-7B | build | uncached | miss (computed and stored) | hit (read, spot check, rebuild) | cache file | query bytes, miss = hit |
|---|---:|---:|---:|---:|---:|---:|
| 2 blocks | 9.9 s | 358.2 s | 359.9 s | 7.2 s | 51.4 MB | 16,045,222 |
| full model (32 blocks) | 103.6 s | (not run) | 4,426.2 s | 49.1 s | 51.4 MB | 299,627,020 |

* Every step gives the same roots as the uncached commitment, and both queries are accepted with identical bytes.
* The full model's setup takes 74 min on the 2080 Ti once per model and node; a later job loads it in 49 s (90x).
  The hit's time is the spot check (4 recomputed columns per tree, which encode every row at those points) and the
  rebuild of the trees; the file read is small.
* The file holds one 32-byte digest per column of each tree (6 x 2^18 x 32 B = 50.3 MB), so its size does not grow
  with the number of blocks.

## 6. Limitations

* The cache lives on one machine's disk; it is not shared between nodes (each node pays one setup per model).
* The spot check samples 4 leaves per tree; a file corrupted in a few leaves only could pass it, and then honest
  proofs would be rejected (never a false acceptance). Deleting the folder clears it.
* The setup tier's timings and the `commit_total` of the first job per model are the setup times the paper reports;
  every later job's commitment time is a load.

## 7. How to run

    cd code && PYTHONPATH=src python -m pytest -q tests/test_setup_cache.py
    cd code && PYTHONPATH=src python experiments/6_improvements/setup_cache_check.py --model llama2-7b --layers 2 \
        --policy cnn18c --cache <empty folder> --device cuda
