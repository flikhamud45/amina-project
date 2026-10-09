# B2 + B3. The GPU verifier at long prompts: claims decoded on the GPU, and a fused exact attention kernel

Plan items B2 and B3 (reviewer objections D1 and D3: the long-prompt verifier, the future-work items
"native verifier" and "checking attention"). Commit `2a71a79` (`claimcodec.decode_device`,
`codec_kernels.unpack_stream`, `attention_kernels.attention_core`, their use in `protocol._decoded_claims` and
`transformer._attention_heads`). Status: done. Implemented, measured and validated on an RTX 2080 Ti (job 1005162: GPU decoder tests 2 passed,
`test_gpu_verifier.py` 287 passed and 15 skipped, `test_gpu_exactness.py` 77, generation and setup proof 40,
wire 118, no failure); L40S numbers with the final runs.

## 1. The problem

At 2,048 tokens the streaming GPU verifier spent most of its time in two places (L40S, from the stored
records and the anatomy report):

* **decoding the claims on the host**: numpy on 8 threads, 2-4 ns per claim, 59-73% of the GPU verifier's
  time (OPT-1.3B: about 1.9-2.1 s of 2.6 s), then uploading them as int32;
* **attention**: `_attention_core` materialises the `T x T` score tensors of a group of heads in int32 and
  runs about a dozen element-wise passes over them (mask, maximum, gaps, exp table, row sums, floor division,
  probabilities), about 65 bytes of memory traffic per score.

## 2. B2: decoding on the GPU

`claimcodec.decode_device(buf, rows, cols, device)` returns the claims as int32 tensors on the GPU with the host
decoder's integers and checks:

1. **On the host**, before anything is read on the device: the header (`_read_header`: shapes against the
   query's, flags, widths, bases), every slot stream's extent, the exception count and both Rice vectors'
   extents (`_exceptions`, `_rice_end`) and the absence of trailing bytes. So every offset the device reads is
   inside the input.
2. The bytes go up once (pinned memory).
3. **Each width's slot stream** is unpacked by `codec_kernels.unpack_stream`, a Triton kernel: one program per
   block of columns reads the `B` words of each column and writes its 32 lanes' slots in place; a non-zero
   padding slot (a lane position at or past the stream's length) sets an error flag. Each segment's base is
   added (residuals get theirs with their row means, as on the host).
4. **The exceptions**: both Rice vectors are decoded on the device (the low parts by the same unpack kernel,
   the unary levels bit-plane by bit-plane, their padding checked on the host bytes), positions by a cumulative
   sum, high parts un-zigzagged and shifted by their segment's width, then patched in with the host's range
   checks (positions inside the claims, `hz < 2^32`, every patched value within `(-2^30, 2^30)` including a
   residual's pending base). All verdicts come back in one copy.
5. **Row means** are added to the centred ops' residuals.

A GPU verifier (`Verifier(device="cuda")`, streaming or not) uses it when Triton is available
(`PVI_GPU_DECODE=0` keeps the host decoder). The claims never leave the GPU; the streaming verifier's
`ClaimUploads` sees device tensors and does not copy them. Soundness is unchanged: the same bytes give the same
integers or the same rejection.

## 3. B3: a fused exact integer attention kernel

`attention_kernels.attention_core(q, k, v, lut, queries)` computes exactly the integers of
`_attention_core` (raw `p v`, int64) per block of query rows, in three passes over the key blocks up to the
block's last query (later keys are all masked and skipped):

1. the row maxima of the unmasked scores `s = q k^T`;
2. `tot`, the row sums of `e = LUT[min(max - s, len - 1)]` (0 for a masked key);
3. `p = floor((tot + 510 e) / (2 tot))` and `p v`, accumulated over the key blocks.

**Exactness.** `q`, `k`, `v` are int8 and `p <= 255`, all exact in fp16; `tl.dot` accumulates in fp32, exact
while every partial sum is an integer below `2^24`. A score is at most `dh 127^2 <= 2^21` (`dh <= 128`); every
partial sum of a row of `p v` is at most `(sum of p) 127 <= (255 + T/2) 127 < 2^24` for `T < 2^15` (the bound
that already lets `_attention_core` use one float32 GEMM). The maximum, gaps, table, sums and floor division are
integer operations. Grouped-query attention's stacked rows (row `r` is the query at position
`T - queries + r mod queries`) and strided inputs are handled. It replaces `_attention_core` for prover and
verifier on a GPU with Triton (`PVI_FUSED_ATTN=0` keeps torch), and needs no `T x T` buffer.

## 4. Tests

* `tests/test_claimcodec.py::test_the_gpu_decoder_gives_the_host_decoders_claims`: every claim set of the codec
  tests (widths 0-30, centring, exception-heavy and incompressible ops, extremes, empty ops, 300 ops, decoder
  claims), centred or not.
* `tests/test_claimcodec.py::test_the_gpu_decoder_rejects_exactly_what_the_host_decoder_rejects`: every
  truncation, trailing and wrong-magic inputs, 400 random multi-bit corruptions (header, streams, padding,
  Rice vectors), exceptions out of range: the same verdict, and the same claims when accepted.
* `tests/test_gpu_verifier.py::test_the_fused_attention_kernel_gives_the_int32_cores_integers`: grouped
  queries (`hq/hkv` 4/4, 8/2, 6/2, 2/1), queries of the last positions only, `dh` 16, 64, 80, 128, `T` up to
  2,048, random, peaked and extreme patterns (every score at its bound, the largest `p v` sums), two `m_s`,
  strided inputs: `torch.equal` with `_attention_core`.
* The verifier-level tests (wire, streaming, plans, generation) run with both on (job 1005162).

## 5. Measurements (RTX 2080 Ti, streaming GPU verifier, mode Kpre with lookups, 2,048 tokens, compact encoding,
last block pruned; 3 queries each, all accepted)

| Build | neither | fused attention | GPU decode | both |
|---|---|---|---|---|
| OPT-1.3B, 12 of 24 blocks | 1.43 s | 1.24 s | 1.08 s | **0.53 s (2.7x)** |
| Llama-2-7B, 2 of 32 blocks | 0.36 s | 0.38 s | 0.72 s | **0.18 s (2.0x)** |

GPU decode alone is slower on Llama-2-7B (the decoded claims and the torch attention's `T x T` temporaries
compete for the 11 GB); the two together remove both bottlenecks. Extrapolated to the full OPT-1.3B, about
1.1 s on the 2080 Ti, against 2.6 s on the L40S in the paper and zkLLM's 0.90 s (its own hardware). The L40S
measurement is part of the final runs (target: <= 0.8 s).

## 6. What remains

* L40S measurements on the full models (OPT-1.3B, OPT-6.7B, Llama-2-7B, Llama-2-13B at 2,048 tokens).
* A per-phase GPU profile with both on (plan B1) to find the next bottleneck (likely the element-wise derive:
  `torch.compile` on the GPU now works, plan B4).
