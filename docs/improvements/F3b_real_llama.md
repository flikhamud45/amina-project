# F3b. A real Llama-family checkpoint end to end

Plan 2 item F3b (`SP2027_PLAN2.md` Sec. 3.2; reviewer points on real weights, C-W2/C-Q3). Code: d637f3e (the real
Llama path in `perf.py`, `v1_bytes.py` and `gen_bench.py`), 9ab8033 (the shape check of a Llama head, the recipe in
the records); tests `code/tests/test_f3b_real_llama.py` (6). Status (12 Oct 2026): plumbing merged and tested; the
full OpenLLaMA-7B measured end to end on the 2080 Ti / Xeon nodes (v0 in Kpre and C at 64 tokens, Kpre at 2,048,
V1 bytes at 2,048, each next to its random twin; every query accepted, every tamper rejected); generated responses do
not fit the 11 GB card (a CPU run is job 1018663).

## 1. Why

The cost tables use random int8 weights, since the costs depend only on the shapes, except for the compressed size of
the claims. Until now only real OPT checkpoints ran end to end (F3a: OPT-1.3B and OPT-6.7B). F3b runs a real
Llama-family checkpoint with Llama-2-7B's exact shapes through the same scripts, so that (a) the claims' size, the V1
ratio and the times are measured on real weights of the headline model's architecture, (b) the code path the L40S run
uses for the real Llama-2-7B (`PVI_LLAMA_LICENCE=1`, nine cells of the `real` tier) has run before the freeze, and
(c) generated responses are proved for a real model, with the sampler driven by the LM head's real logit scale.

The checkpoint is OpenLLaMA-7B v2 (`openlm-research/open_llama_7b_v2`, Apache 2.0, not gated). Its shapes are
Llama-2-7B's (4,096 x 32 layers, 32 heads, ff 11,008, vocabulary 32,000, RoPE theta 10,000); `--model llama2-7b`
names the benchmark shape and the shape check compares every weight op with it. Its certified quality with the main
algorithm's recipe is x1.073 at 2,048 tokens (F2b-L). Qwen3-4B was the plan's first choice; its certified model is
x5.3 (F2b-Q), so the plan's default moved to OpenLLaMA-7B (12 Oct).

## 2. What was built (d637f3e, 9ab8033)

* `real_llama.MAIN_RECIPE`: the main algorithm's integer recipe for Llama-family checkpoints (smoothing through the
  norms' gains at alpha 0.55 with the 99.9th-percentile statistic, requantisation at the 99.99th percentile, the
  SwiGLU product balance at 0.2), with a fixed public prefix chosen by PrefixQuant's rule on four 2,048-token train
  windows. Constants only: the graph has the benchmark's structure. No per-token exponent.
* `v1_bytes.load_real` dispatches on the checkpoint's `model_type`: OPT as before, Llama/Qwen3 through
  `load_real_llama` (fp32, WikiText-2 test windows as queries, the calibration and prefix as above);
  `real_graph` builds `build_llama_from_hf(**MAIN_RECIPE)` with the prefix. `perf.py --real` and
  `gen_bench.py --real` use the same two functions.
* `gen_bench.py --real`: prompts are WikiText-2 test windows; mode C plans the graph's own weight ops; the sampler's
  weight is `a = round(c 2^16 / tau)` for the LM head's real logit scale `c` (`TokenSampler.for_temperature`), so a
  sampled response is a draw of the integer model at temperature `tau`.
* The shape check (`v1_bytes.shape_check`) expected one more column in every real LM head, the folded bias of OPT's
  final LayerNorm; a Llama head (final RMSNorm, no bias) has the benchmark's row length. The first cluster run
  flagged the correct graph (job 1015853); the check now adds the column only for a LayerNorm head (9ab8033). The
  records now say `"recipe": "main"` for a Llama-family checkpoint (they said "legacy", the OPT default).
* `pvi.results.git_sha` (26089bb): node s-002 has no `git` binary, so the scripts read the commit from `.git`.

## 3. Validation

`tests/test_f3b_real_llama.py` on a tiny random Llama (no download), 6 passed on the laptop:
* the Llama branch of `real_graph` is `build_llama_from_hf(**MAIN_RECIPE)` with the prefix (same digest), has the
  benchmark's structure (`matches_benchmark_graph` returns no difference), and a positive logit scale;
* the shape check takes the Llama head as built;
* greedy and sampled responses, proved and verified in mode C (R8c) and Kpre: honest accepted, a replaced generated
  token rejected.

## 4. Results on the cluster

### 4.1 V1 bytes, full model, 2,048 tokens, Kpre, Fiat-Shamir (job 1015853, s-002, Xeon Silver 4114)

| Llama-2-7B shapes | v0 claims | V1 total | ratio | exceptions | v0 accepted | v0 query (CPU) | build |
|---|---:|---:|---:|---:|---|---:|---:|
| OpenLLaMA-7B, main recipe | 5,362.9 MB | 2,649.6 MB | **2.02x** | 192,718 | yes | 245.7 s | 2,976 s |
| random int8 weights | 6,472.2 MB | 2,763.3 MB | **2.34x** | 233 | yes | 223.3 s | 101 s |

* Real weights compress better under v0's codec (claims 17% smaller) and leave V1 less to remove, as on OPT
  (real OPT-6.7B 1.73x, F3a), so V1's ratio falls from 2.34x to 2.02x.
* V1's own parts stay small: u 40.8 MB, the cut's exceptions 1.5 MB, multiplicities 0.3 MB, GKR 0.8 MB.
* The build (CPU calibration of the fp32 checkpoint on four 2,048-token windows, prefix search) takes 50 min and
  43 GB of host memory; the query peaks at 86 GB.
* This run's real-weight records carried `accounting_ok: false` only because of the shape check of Sec. 2. The
  rerun with the fix (job 1016884, 9ab8033) gives the same graph digest (`050f8735...`) and bytes, no shape problem,
  `accounting_ok: true` and `"recipe": "main"`.

### 4.2 v0 in Kpre and C at 64 tokens, Kpre at 2,048 tokens, random twins (jobs 1015750, 1015751, 1015752)

**Full OpenLLaMA-7B, 2,048 tokens, Kpre, Fiat-Shamir (job 1015751, s-004).** The GPU prover runs out of memory on the
11 GB card at 2,048 tokens (as for random Llama-2-7B weights), so the job's fallback proved on the CPU; the verifier is
the native CPU verifier on 8 Xeon threads. Three WikiText-2 test windows:

| | accepted | tampers rejected | proof | prover (CPU) | verifier (CPU, 8 threads) |
|---|---|---|---:|---:|---:|
| OpenLLaMA-7B, main recipe | 3 / 3 | 9 / 9 (single value, output logit, last hidden; all at Freivalds) | 5,350.8 MB | 242.6 s [229.2, 251.2] | 21.0 s [20.4, 21.1] |
| random weights (job 1015752, the same fallback) | 3 / 3 | 9 / 9 | 6,433.7 MB | 209.4 s [198.4, 236.0] | 20.6 s [20.4, 22.5] |

The verifier's phases: decode 3.0 s, derive 11.4 s, products 6.6 s. On real weights the verifier takes 2% longer, the
CPU prover 16% longer, and the proof is 17% smaller. (The EV inputs' 19.6 s for random weights is an earlier job,
1007816, whose prover ran on the GPU.) Peak host memory 81 GB (build 43 GB, 48 min).
The record's `git` field reads 9ab8033 although the process ran 26089bb's code: `perf.py` read the commit after the
48-minute build, during which the clone was updated (the two commits differ only in the shape check and the recipe
field, neither used here); `perf.py` now reads the commit first (6cdd0e9). Its `recipe` field says "legacy" for the
same reason (the main recipe was used).

**Full OpenLLaMA-7B, 64 tokens, Fiat-Shamir (job 1015750, s-004; random twin job 1015752).** GPU prover, native CPU
verifier on 8 Xeon threads, mode C under cnn18c with the setup cache, three WikiText-2 test windows, three tampers of
each kind:

| 64 tokens | mode | accepted | tampers rejected | proof | prover (GPU) | verifier (CPU) |
|---|---|---|---|---:|---:|---:|
| OpenLLaMA-7B | Kpre | 3 / 3 | 9 / 9 (Freivalds) | 162.5 MB | 0.307 s | 1.264 s |
| random | Kpre | 3 / 3 | 9 / 9 (Freivalds) | 196.4 MB | 0.301 s | 1.146 s |
| OpenLLaMA-7B | C | 3 / 3 | 9 / 9 (column code 5, Freivalds 4) | 265.9 MB | 1.039 s | 1.428 s |
| random | C | 3 / 3 | 9 / 9 (column code 5, Freivalds 4) | 299.8 MB | 1.053 s | 1.365 s |

* Mode C's proof: claims 162.8 MB, opened columns 93.5 MB, u 9.6 MB, paths 0.1 MB (random: claims 196.7 MB, the
  rest equal). The random-weight 299.8 MB agrees with the EV byte model (302.9 MB with the measured 64-token claims of
  another window) and with the cache check (299.6 MB).
* Times: the provers agree within 2%; the real-weight verifier takes 10% (Kpre) and 5% (C) longer.
* The setup cache worked across jobs: the real graph's commitment was a hit (55 s), stored by the generation job
  (1015754), which committed the same weights and plan before its generation ran out of GPU memory; the random
  twin's was a miss (4,359 s, the setup).
* A TinyLlama-1.1B smoke cell (main recipe) passed first in the same job (2 / 2 accepted, 3 / 3 tampers per mode).

### 4.3 Generated responses (job 1015754; CPU rerun job 1018663)

With P = 512 and G = 64 the full 7B graph's generation runs out of memory on the 11 GB 2080 Ti (job 1015754, in the
first greedy step; the commitment before it completed and was cached). The L40S (48 GB) runs this cell
(`r-llama7-gen`). To run the code path before the freeze, job 1018663 repeats it on the CPU of s-004 with G = 16, one
timed query per rule and mode (the cached commitment serves mode C).

[pending]

## 5. Limitations

* OpenLLaMA-7B is not Llama-2-7B: same shapes, different weights. The real Llama-2-7B runs on the L40S under the
  runner's own licence (the `real` tier with `PVI_LLAMA_LICENCE=1`).
* Times on the 2080 Ti / Xeon nodes are not the final platform's.

## 6. How to run

    cd code && PYTHONPATH=src python -m pytest -q tests/test_f3b_real_llama.py
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --prune-last --lmax 27 --threads 8 --model llama2-7b \
        --real openlm-research/open_llama_7b_v2 --seq 2048 --modes Kpre --fs --lean --device cpu
