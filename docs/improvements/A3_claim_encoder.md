# A3. The prover's claim encoder: a threaded assembly of the result, and a fused Triton encoder (negative result)

Plan item A3 (reviewer objection D1, the prover's cost at long prompts; future-work item "a faster encoder").
Commits `29db2ce` and `228ae2d` (the fused encoder, `codec_kernels.pack_stream` and `high_parts`, bounded
memory), `6f81c1b` (`encode_prof.py --compare-torch`), `71e8920` (`claimcodec._join_pieces`), `aa7e485` (the
fused encoder made opt-in). Status: done. The assembly fix is the default and halves the encoder's time on large
proofs; the fused encoder is kept behind `PVI_FUSED_CODEC=1` because it is not faster.

## 1. The problem

With A1 and A2 the prover's forward pass and opening became fast, and the claim encoder (PVC3, 16-19 bits per
claim) became one of its largest stages at 2,048 tokens. For the full OPT-1.3B (24 blocks, 2,048 tokens, a
2.0 GB proof) the device encoder took 3.28 s on the RTX 2080 Ti. The encoder runs about ten torch kernels per
width over parts of `_CHUNK` values (statistics, plan, slots, packing, exceptions), and our first hypothesis was
that these launches and their index tables (several bytes per value) were the cost.

## 2. What we built first: a fused encoder (`codec_kernels.py`)

One Triton kernel per width reads every value of a block of a slot stream's columns directly from the claims.
A table of the stream's segments (`SegmentTable`: the segment starts, an int32 pointer to each segment's
first value, its row means for a centred op's residuals, its base) lets a program find the value at stream
position `v` by a binary search. The program writes the slot `(x - lo) mod 2^B` into lane `v // G` of column
`v % G`, exactly the layout of `claimcodec.pack32`, and sets bit `j` of the column's flag word when the value
is an exception. A second kernel (`high_parts`) recomputes the high parts `(x - lo) >> B` of the flagged
positions. The exceptions' Rice vectors are built by `_RiceParts` from pieces of at most `2^24` exceptions
in the global order, with the statistics on the device and the packing on the host, so device memory stays
bounded: 183 MB at 885 M claims, against about 1 GB for the one-pass device encoder (whose first version ran
out of memory at that size).

**Exactness.** The kernels compute the same integers as the torch encoder (values, slots, words, flags, high
parts), so the bytes are identical. `tests/test_claimcodec.py::test_the_fused_device_encoder_gives_the_reference_bytes`
checks this on random claim sets that include centred ops, every width from 0 to 32, exceptions at segment
boundaries and small `_FUSED_COLS`/`_FLAG_COLS` values that force several launches and pieces.
`encode_prof.py --compare-torch` checks it on real proofs.

## 3. What the profile showed: the cost was the assembly

A profile of the full OPT-1.3B encoding (`encode_prof.py`, job 1005091) showed that about 2.6 s of the 3.28 s
was not the encoding itself but `b"".join` of the encoded pieces. The join is a single-threaded copy of 2 GB
into freshly faulted pages. Both encoders paid it.

**The fix** (`claimcodec._join_pieces`): the pieces are copied into one numpy buffer in jobs of
`_JOIN_PIECE` bytes on torch's threads (numpy releases the GIL during the copy), and the result is returned
as a read-only `memoryview`. The decoder (`_read_header`, `decode`, `decode_device`) accepts bytes or a
read-only 1-D contiguous byte memoryview. The bytes are unchanged; only their container differs.
`test_a_read_only_buffer_decodes_as_its_bytes` checks that such a buffer decodes exactly like its bytes; the
input-type test above it checks that a writable buffer and a 2-D view of the same bytes are still rejected.

## 4. Measurements (RTX 2080 Ti, job 1005154, identical bytes in every row)

| Proof | Size | Torch encoder before | Torch encoder after the fix | Fused encoder |
|---|---:|---:|---:|---:|
| OPT-1.3B, 24 blocks, 2,048 tokens | 2,023 MB | 3.28 s | **1.54 s (2.1x)** | 1.74 s |
| Llama-2-7B, 2 blocks, 2,048 tokens | 254 MB | 0.542 s | **0.238 s (2.3x)** | 0.346 s |
| Llama-2-13B, 1 block, 2,048 tokens | 61 MB | | **0.084 s** | 0.208 s |
| GPT-2, 12 blocks, 512 tokens | 86 MB | | **0.116 s** | 0.303 s |
| OPT-125M, 12 blocks, 64 tokens | 11 MB | | **0.026 s** | 0.206 s |

The fused encoder is slower everywhere. With the assembly fixed, the torch encoder's kernels are bandwidth-bound
and already close to the cost of reading the claims. The fused kernels' binary search per value and the
per-width launches, flag scans and Rice passes add a fixed cost of about 0.2 s, which dominates small proofs.

## 5. Decision

* `_join_pieces` is the default for the chunked, fused and one-pass device paths: **2.1-2.3x on the encoder** of
  large proofs, at no cost.
* The fused encoder stays in the code, opt-in (`PVI_FUSED_CODEC=1`, `_FUSED` in `claimcodec.py`), with its
  tests. Its one advantage is bounded device memory (183 MB against about 1 GB at 885 M claims), which may
  matter for a model whose claims do not fit next to its weights. We report it as a negative result: kernel fusion
  did not pay because the encoder was bound by the host copy, not by the GPU.

## 6. Reproducing

```
cd code
python experiments/6_improvements/encode_prof.py --model opt-1.3b --layers 24 --seq 2048 --compare-torch --no-profile
python -m pytest -o addopts= -q tests/test_claimcodec.py           # the fused tests run on a CUDA host
```

## 7. What remains

Nothing for A3 itself. The encoder's share of the prover's time is now small next to the opening and the
forward pass. The proof's *size*, which this item does not change, is addressed by C2 (V1,
`C2_v1_proof_size.md`).
