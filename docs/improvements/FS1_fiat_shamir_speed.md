# FS1: Fiat-Shamir hashing at memory speed, counted in both parties' time

Plan 2, item FS1 (`SP2027_PLAN2.md`, Sec. 3.1). Code: commit 12a188f (`protocol.py`, `cut_protocol.py`,
`perf.py --fs`, `tests/test_fs_tree.py`), test update in b244e8b (`tests/test_wire.py`).

## 1. The problem

Under Fiat-Shamir (`params.fiat_shamir`), every challenge is derived from a SHAKE-256 transcript of everything the
prover sent before it. Plan 1 (D1/D2) absorbed each message with one serial `hashlib.shake_256` call.

* One serial SHAKE-256 stream runs at 303-318 MB/s on the Xeon Silver 4114 (job 1005575) and at 227 MB/s on the
  laptop (measured below).
* The claims of Llama-2-7B at 2,048 tokens are about 6.6 GB on the wire. Both parties hash them, so each pays about
  21 s, which is more than the whole 17.2 s native CPU verifier (B5). With the hashing counted, the abstract's 4.6x
  verifier-versus-re-execution ratio would fall to about 2.1x.
* The hashing was not inside `prove_s`/`verify_s`, so the published ratios were interactive ones. Reviewer B (B-W7,
  consolidated change 9) asks for non-interactive proofs by default; reviewer C (C-Q1) asks what serving costs.

## 2. What changed

### 2.1 Kpre absorbs nothing (v0)

Kpre is a one-message protocol with a designated verifier. The verifier's challenge `chi` and its folded rows
`u = chi^T A` are drawn once, at setup, and kept secret (`verifier._pre`). The prover sends the claims and nothing
else; no challenge is derived after a prover message, so there is nothing for a transcript to bind.

* `_claims_message` absorbs only when `verifier.params.fiat_shamir and verifier.mode != "Kpre"`.
* The protocol is the same as before: Kpre was already non-interactive. What changes is that the Kpre verifier no
  longer pays 21 s for a transcript no challenge reads.
* Soundness is unchanged: it rests on the secrecy of `chi` (the Freivalds argument against a prover who does not
  know the check vector), not on a random oracle. The theorem is drafted in `paper_sp2027/drafts/th_e_proofs.tex`,
  Proposition `prop:kpre-fs`.
* **V1 is different.** V1's logUp-GKR sub-protocol is public-coin, so its challenges must depend on the claims. In
  every mode, Kpre included, `cut_protocol.py` still absorbs the statement, the cut plan, the claims, the packed rows,
  the look-up hashes, `cut/exc` and `cut/mult` before the GKR challenges, and Kpre's folded `u` blob afterwards
  (lines 344-354 and 464-468). Only v0's single message is exempt.

### 2.2 Large messages are absorbed as a one-level tree

`Challenger.absorb_large(label, blob)` replaces `absorb` at every call site whose message can be large:

| Call site | Message |
|---|---|
| `protocol._claims_message` | the encoded claims (`claims/PVC3`), the packed int8 rows (`rows/I8`) |
| `protocol` (mode C) | the folded rows (`u/F31`) |
| `cut_protocol` | claims, rows, `cut/exc`, `cut/mult`, Kpre's `u/F31` |

A message shorter than `_TREE_MIN` = 16 MiB is absorbed as before, so every small test transcript is unchanged.
From 16 MiB on:

1. The message is cut into `_TREE_CHUNK` = 4 MiB chunks (the last one may be shorter).
2. Leaf `j` is `SHAKE-256("pvi/leaf/v1" || len(label) [2 bytes] || label || j [8 bytes] || chunk_j)`, 32 bytes.
3. The leaves are computed on a thread pool (`hashlib` releases the GIL on large inputs). The thread count is
   `PVI_HASH_THREADS`, or torch's intra-op thread count by default (`_hash_threads`).
4. The transcript absorbs, under the label `label + "/tree"`, the number of chunks, the chunk size and the total length
   (three big-endian 64-bit integers), followed by the leaves in order.

The encoding is injective. The chunking is fixed, the lengths and indices are bound, the leaf prefix carries a domain
separator and the label, and the tree record has a label of its own. So two different messages that give the same
transcript state yield a SHAKE-256 collision, either in a leaf or in the transcript. The Fiat-Shamir bound gains the
birthday term of the leaves: at most `Q^2 / 2^257` for `Q` leaf evaluations, negligible next to the existing terms.
This is the standard argument for a sound tree-hashing mode (Bertoni et al.; KangarooTwelve is a deployed instance).
The full statement and proof are drafted as Theorem `thm:fstree` in `th_e_proofs.tex`.

The transcript itself stays SHAKE-256 (Plan 1 D2), and so do the leaves, so no new dependency was added. The plan
allowed BLAKE3 or KangarooTwelve as an alternative absorbed digest. The tree mode was taken because it passed the
transcript tests first, keeps one primitive throughout and needs no pinned package; on a CPU with more threads it
scales the same way.

### 2.3 Accounting

* Each party's hashing is timed as `fs_hash` (`protocol.py` lines 1641 and 1766) and is counted inside both `prove_s`
  and `verify_s` (`perf.py` lines 67-68: the verifier's total is every `verify_*` phase plus `fs_hash`).
* `perf.py --fs` runs a cell under Fiat-Shamir; the JSON record carries `"fs": true`. Every `cpuv`/`gpuv` cell of the
  final package runs with `--fs`.

## 3. Validation

**Tests** (`tests/test_fs_tree.py`):

* `test_small_messages_are_absorbed_as_before`: below 16 MiB the transcript is byte-identical to Plan 1's.
* `test_tree_absorption_matches_its_serial_reference` (1 and 8 threads): the threaded tree equals a serial
  reference written out in the test.
* `test_every_chunk_and_the_framing_bind`: flipping a byte in any chunk, changing the length, the label or the
  framing, or absorbing the message directly all give pairwise different states (8 distinct states).
* `test_kpre_under_fiat_shamir_absorbs_nothing_and_keeps_its_verdicts`: in Kpre under Fiat-Shamir, `fs_hash` is 0, an
  honest query is accepted and a tampered one is rejected.
* `tests/test_wire.py::test_fiat_shamir_absorbs_the_encoded_bytes`: in modes C and K the transcript absorbs exactly the
  encoded bytes, and the streaming verifier's transcript equals the batched one's. In Kpre the transcript is empty.

Laptop, 10 Oct: `test_fs_tree`, `test_protocol`, `test_wire`, `test_cut_protocol` and the generation, look-up and plan
suites: 994 passed, 0 failed after the `test_wire` update. Xeon, job 1006979: 1,095 passed; the one failure was the
Kpre case of `test_wire`, written before FS1 and fixed in b244e8b.

**Throughput** (laptop, Intel 11th-generation Core (family 6, model 140) with 8 logical CPUs, one 1 GiB message,
other jobs running):

| Absorption | MB/s |
|---|---:|
| serial SHAKE-256 (Plan 1) | 227 |
| tree, 1 thread | 244 |
| tree, 2 threads | 459 |
| tree, 4 threads | 942 |
| tree, 8 threads | 1,368 |

So 8 threads hash 6.0 times as fast as one serial stream. At this rate Llama-2-7B's 6.6 GB of claims at 2,048 tokens
cost about 4.8 s per party in modes C and K (the plan's estimate was 3-5 s), and 0 s in Kpre.

**Xeon, full Llama-2-7B** (V-LONG-a, jobs 1006979 and 1007816, `perf.py --modes Kpre` with and without `--fs`,
8 threads): `fs_hash` = 0 in Kpre, as designed. The verifier takes 1.01 s / 1.40 s at 64 tokens and 18.5 s / 19.6 s at
2,048 tokens (interactive / Fiat-Shamir); almost all of the difference is the Freivalds repetitions (5 against 7 for the
same soundness), in the products step (4.6 s against 5.4 s at 2,048 tokens). Without FS1 the Fiat-Shamir verifier
would have hashed the 6.43 GB of claims serially, about 21 s more.

## 4. Limitations

* Modes C and K still pay one pass over the claims at memory speed for each party. With 8 Xeon threads this is about
  1.3-2 GB/s, so a few seconds for a 7B model at 2,048 tokens; a 16-thread EPYC halves it.
* The leaf size (4 MiB) and the 16 MiB threshold are constants of the protocol: changing them changes every
  transcript, so a prover and a verifier must agree on them. They are fixed in the code, and the leaf prefix carries
  the version tag `pvi/leaf/v1`, to be bumped with any change.
* Kpre's exemption covers v0 only, by design (Sec. 2.1).

## 5. How to run

    cd code && PYTHONPATH=src python -m pytest -q -o addopts= tests/test_fs_tree.py tests/test_wire.py
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model llama2-7b --seq 2048 --modes Kpre C \
        --queries 3 --lean --wire on --prune-last on --lookups on --threads 8 --verifier-device cpu --fs
