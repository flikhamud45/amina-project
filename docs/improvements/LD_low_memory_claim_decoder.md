# LD: the GPU verifier's claim decoder in bounded memory

Code: commits 4434f25, c2f7cd9, 475d02a and e258288 (branch `sp2027-artifact`). Options `--decode-memory`,
`--decode-memory-budget` and `--decode-chunk` (`pvi.options`); `pvi/v0/claimcodec.py` (`decode_device_parts`,
`device_decode_peak`); `pvi/v0/query.py` (`decoded_claims`, `device_decode_low`); `experiments/costs/bench.py`;
`experiments/package/run.sh`.

## 1. Why

The streaming GPU verifier decodes the compact claim encoding on the GPU. Its decoder, `claimcodec.decode_device`,
holds at once the whole encoded input, the int32 claims and about eleven int64 vectors as long as the list of
exceptions (the values that the Rice code does not cover: their Rice vectors, positions, high parts, segments, widths
and the values they patch), that is about 88 bytes per exception.

On the real checkpoints the exceptions are far more numerous than on random weights. For Llama-2-7B at 2,048 tokens
about 375 M of the 2.7 G claims (about 13%) are exceptions. The exception vectors alone then take about 33 GB, beside
10.8 GB of claims and the model's weights, which exceeds the memory of an L40S. In the final run the two
streaming-GPU-verifier cells of the real Llama-2 checkpoints at 2,048 tokens, `b-cg-llama2-7b-2k` and
`b-cg-llama2-13b-2k`, therefore failed with an out-of-memory error in the decoder. Both cells had passed on random
weights, where exceptions are rare.

The rule of the earlier fixes applies here as well: no job may change its path unless it passes an explicit flag, and
every path gives the same bytes and verdicts.

## 2. The decoder in parts

`claimcodec.decode_device_parts(buf, rows, cols, device, chunk)` gives the same int32 claims as `decode_device` and
raises the same `ClaimCodecError`, at the same check, on every input. It differs only in what it keeps on the GPU:

* **Slot streams one at a time.** Each fixed-width slot stream is uploaded, unpacked into the claim buffer by the
  same Triton kernel (`codec_triton.unpack_stream`), and freed before the next one is uploaded. The whole input is
  never on the GPU at once.
* **Compact Rice vectors.** The two Rice vectors of the exception list (gaps and payloads) are kept as their k-bit low
  parts in int32 and their unary quotients in uint8 (`_rice_device_parts`), about 1.3 bytes of input per exception,
  instead of int64 values.
* **Exceptions in parts.** The exceptions are patched `chunk` at a time. Each part rebuilds its int64 values
  (`_rice_part_values`), continues the cumulative sum of positions from the previous part (the same int64 values,
  wrapped alike), and writes its patched values into the claim buffer. The range checks are accumulated as boolean
  flags on the device and read once at the end, in the order in which `decode_device` reports them.

Its memory beyond the claims is therefore the largest slot stream, then the compact Rice vectors and one part's
temporaries (about 80 bytes per exception of the part; 2^23 exceptions, about 0.7 GB, by default).

## 3. Options

* `--decode-memory default|low|auto` (default `default`):
  * `default` is the measured path, `decode_device`, unchanged.
  * `low` uses `decode_device_parts`.
  * `auto` uses `low` only when the memory already allocated on the GPU plus the default decoder's predicted peak
    exceeds `--decode-memory-budget` percent of the GPU's memory. The prediction (`claimcodec.device_decode_peak`)
    is conservative: the input, the int32 claims and 88 bytes per exception. It is read from the input's header and
    the exceptions' count on the host.
* `--decode-memory-budget` (default 90): the percentage used by `auto`.
* `--decode-chunk` (default 2^23): the number of exceptions patched per part.

The decision (`query.device_decode_low`) is taken before the timed decoding starts. A verifier on the CPU, or one
without Triton, never takes the GPU decoder and so never takes `low`. When the option is not `default`, each query
records the path taken in `decode_memory` (`low` or `default`), and `bench.py` writes it into the record. With the
default option the code path, the record and the timing are as before.

`run.sh` passes `--decode-memory auto` to `b-cg-llama2-7b-2k` and `b-cg-llama2-13b-2k` alone. Every other job keeps
`default`.

## 4. Exactness

The decoder in parts changes no claim, no proof byte, no transcript and no verdict: it is a different evaluation
order of the same decoding, with the same checks. It can change the time of the decoding phase (`verify_decode`),
which the record attributes through `decode_memory`.

## 5. Tests

`tests/test_decode_memory.py`:

* on the CPU, with a torch twin of the Triton stream unpacker (itself checked against the host decoder's streams), the
  decoder in parts gives the default decoder's claims for every claim set of the codec tests and for any part size,
  and rejects every malformed input that the default decoder rejects, with the same message;
* the predicted peak reads the header only, and a part size below 1 is refused;
* the option defaults to `default`, which records nothing; a verifier on the CPU never takes the GPU decoder;
* in modes C and Kpre, a streaming query with the decoder in parts has the default path's verdict, proof bytes and
  transcript and records `low`, and an encoding with one claim changed is rejected at the same check on both paths;
* on a GPU (skipped elsewhere): the same claims and rejections; `auto` keeps the default decoder when its predicted
  peak fits, and takes `low` when the prediction is replaced by one beyond the GPU (the test checks that the real
  prediction was read from the query's header); on about 4 M claims of which a third are exceptions, the decoder in
  parts takes less than a third of the default decoder's memory beyond the claims.

`tests/test_options.py` covers the three options' parsing, defaults and refusals.

Two corrections followed the first commit:

* the helper that rebuilds a part's values was first named like the device encoder's Rice helper, which it replaced at
  import, so that every encoding in parts failed. It is now `_rice_part_values`, and no other name in the module is
  defined twice (475d02a);
* the test of `auto` first forced the low path with a 1% budget, which the tiny model's predicted peak never exceeds.
  It now replaces the prediction and keeps a reference to the real one before patching it (c2f7cd9, e258288).

## 6. Effect on the run

Only the two failed cells run again, with `--decode-memory auto`. Every finished job stays valid. `FIXES2.md` in
`docs/submission` (the handoff for commit e258288) describes the resubmission.

## 7. What remains

* The full-size peaks are predictions from the header. The final run's two cells confirm whether `auto` takes `low`
  and give the decoding time on that path.
