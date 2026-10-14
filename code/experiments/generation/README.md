# Generation and sampling

A generated response is proved as one prefill of the prompt followed by the response, with the LM head at the last
`G` positions and a token rule that the verifier checks (Sec. `sec:gen-prefill`); a sampled response must be the
output of a public integer Gumbel-max sampler keyed by the client's seed (Sec. `sec:gen-sampling`). `gen_bench.py`
measures the cost of proving greedy and sampled responses and checks that a wrong token is rejected; `sampler_tv.py`
computes the sampler's exact output law and its distance to softmax sampling on a real model's logits.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| Sec. `sec:gen-prefill`'s cost paragraph, `tab:gen` and `tab:gen-full` (final run: Llama-2-7B and Qwen3-4B, a 512-token prompt, `G` in {64, 256}, greedy and sampled, modes C and Kpre, both tamper cells) | `gen_bench.py` (tier `gen`) | `python -m experiments.generation.gen_bench --model llama2-7b --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre --queries 3 --label gen` (and `--model qwen3-4b`) | stdout, one JSON line per `G`, rule and mode (`"kind": "gen"`) |
| Sec. `sec:gen-prefill`'s cost paragraph: GPT-2's prover time per generated token (a greedy 256-token response after a 64-token prompt, modes C and Kpre; the comparison with zkAgent's GPT-2), and its rows of `tab:gen` and `tab:gen-full` | `gen_bench.py` (tier `gen`) | `python -m experiments.generation.gen_bench --model gpt2 --prompt 64 --gen 256 --decode greedy --modes C Kpre --queries 3 --label gen` | stdout (`prove_s / gen`) |
| the real Llama-2-7B's generated responses (tier `real`, with access to the gated Llama-2 checkpoints) | `gen_bench.py --real` | `python -m experiments.generation.gen_bench --model llama2-7b --real meta-llama/Llama-2-7b-hf --prompt 512 --gen 64 --decode greedy sample --modes C Kpre --queries 3 --label real` | stdout |
| Sec. `sec:gen-sampling` ("Distribution"): the sampler's total-variation distance to `softmax(c z / tau)` at `tau = 1` over 512 WikiText-2 positions of the integer OPT-1.3B (median 0.0063 with `k = 50`, 0.020 with `k = 0`), against 0.087 between the integer and the fp32 model, and 0.0022 with a table of `2^20` entries | `sampler_tv.py` | certified model (the default): `python -m experiments.generation.sampler_tv --model facebook/opt-1.3b --seq 128 --windows 4 --workers 16 --threads 16 --out sampler_tv_opt-1.3b.json`; the stored numbers: the same with `--recipe legacy --calib-seq 128` | the JSON document (`summary`); stored: `artifacts/results/sampler_tv_opt-1.3b.json` |
| the sampler's checks in the extended version (Sec. `sec:ext-sampling`: closed forms, a Monte Carlo of the sampler itself) | `sampler_tv.py --selfcheck`, `--mc` | `python -m experiments.generation.sampler_tv --selfcheck`; `python -m experiments.generation.sampler_tv --model facebook/opt-125m --recipe legacy --windows 1 --positions 32 --mc 100000 --mc-positions 8 31 --mc-settings 1:0 1:50 1.3:0` | stored: `artifacts/results/sampler_tv_selfcheck.json`, `sampler_tv_opt-125m.json` |

`tab:gen` and `tab:gen-full` come from `gen_bench.py`'s records of the run package's `gen` tier; the benchmark's
generation cells (`costs/bench.py --gen`, `experiments/costs/README.md`) measure the same protocol on the image-model
benchmark's harness. Run every command from `code/` with the checkout's library first on the path (`PYTHONPATH=src:.`
on Linux, `PYTHONPATH="src;."` on Windows). `python -m experiments.generation.<script> --help` lists every option.

## Scripts

### `gen_bench.py`

Builds a random-weight decoder (`pvi.models.decoder.CONFIGS` or `experiments.common.models.TINY`; `--layers` blocks)
or, with `--real`, a checkpoint's integer graph (OPT with `hf_opt.CERTIFIED_RECIPE`, a Llama-family checkpoint with
`hf_llama.MAIN_RECIPE` and its fixed public prefix), and in mode C commits it once under `--policy`: one commitment
serves every `G`, since `decoder.with_lm_positions` keeps the weight operations.

* **Per cell** (`G`, rule, mode): the `G - 1` tokens after each of `--queries` + 1 prompts are generated (greedily, or
  by `TokenSampler` with the client's seed per prompt: weight `--sampler-a` on random weights, `round(c 2^16 / tau)`
  from the LM head's real logit scale `c` with `--real` and `--temperature`); each response is proved as one query
  (one warm-up, then `--queries` timed). The token rule (`Verifier.check_tokens`), which `run_query` books to no
  phase, is timed and added to the verifier's time. Two tamper cells on the first response must be rejected:
  `wrong_token` (one generated token replaced, the tokens after it generated again; expected label `token`) and
  `forged_logits` (the LM head's claimed logits raised so that the rule picks the wrong token; expected
  `freivalds`, or `columns_code` when the plan commits the LM head in the col layout).
* **Main options.** `--model`, `--prompt 512`, `--gen 64 256`, `--decode greedy sample`, `--modes C Kpre`,
  `--queries 3`, `--top-k`, `--no-tampers`, `--seed`, `--prompt-seed`, `--calib-tokens`. The protocol settings default
  to the paper's configuration (`experiments.common.cli.PAPER`): `--policy cnn18c`, `--fs`, `--wire`, `--lean`,
  `--prune-last`, `--lookups`, each with a `--no-...` form, `--lam 128`, `--rate 4`, `--setup-cache DIR`; the runtime
  flags `--device`, `--verifier-device`, `--threads` and every implementation option of `pvi.options`.
* **Records.** One JSON line per cell with `"kind": "gen"`: the settings, `positions`, `accepted` / `queries`, the
  medians and `[min, max]` of `prove_s` and `verify_s`, per-part timings and bytes, `proof_bytes`,
  `proof_mb_per_token`, `token_rule_s`, `generate_s`, `sampler` (the weight `a` used and `top_k`), `tampers`, `ok`,
  `peak`, `build_peak`, `env`. The exit status is 1 if an honest response was rejected or a tampered one accepted.
* **Hardware and time.** The final run: an NVIDIA L40S prover and 8 CPU threads for the verifier; the package gives
  each 7B model (both lengths, rules and modes) a three-hour job with 192 GB of host memory. Generating the
  responses, without a KV cache, is a large part of the time; it is not part of the proof's cost.

### `sampler_tv.py`

The exact law of `TokenSampler.choose` over a uniform seed, computed on the integer score grid (one correlation of the
logits' level histogram with the table's log-CDF, by direct sums or an FFT, then one gather per candidate; the
docstring gives the derivation and the exactness argument), and per analysed position, temperature (`--taus`), top-k
set (`--topk`) and table (`--tables`: `16` is the protocol's; `20` and `16:20` simulate wider tables):
`tv_sampler` (against ideal sampling from the integer model's logits), `tv_table` (the table's own error),
`tv_temp` (the rounding of the weight `a`), `tail_mass`, and per position `tv_model` (the integer model's distance to
the fp32 checkpoint's).

* **Inputs.** An OPT checkpoint and WikiText-2 (raw), downloaded from Hugging Face; the integer graph of `--recipe`
  (`certified`: `hf_opt.CERTIFIED_RECIPE` calibrated on four 2,048-token train windows, the certified model of
  `tab:quality`; `legacy`: norm gain 4 with SmoothQuant alpha 0.55, calibrated on the last `--calib-seq`-token window
  of the test split, the graph of the stored results). `--logits-cache file.npz` keeps the integer and fp32 logits,
  so that the analysis can be repeated without the model; a cache of another model, window set or recipe is refused.
* **Checks.** Every run first checks the law against closed forms (2 and 8 tokens, exact rationals) and a Monte Carlo
  of `TokenSampler.choose` (`--selfcheck` runs only these); `--mc N` compares `N` draws of `choose` with the law at
  `--mc-positions`. Each law reports `|sum - 1|`, the FFT's deviation from the direct sum, and the bound on the
  probability of the integrands it skips (those below `e^floor`, `--floor -60`).
* **Output.** One JSON document: on stdout without the per-position rows, in `--out` with them (`summary`: mean,
  median, 90th percentile and maximum over positions per setting; `rows`, `model_rows`, `mc`, `toys`, `meta`, `env`).
* **Hardware and time.** CPU only. The stored OPT-1.3B analysis (512 positions, three tables, 16 worker processes of
  about 0.6 GB each) took 43 minutes and 11 GB of host memory in the main process; `--selfcheck` takes seconds.

## Reproducing the paper's numbers

The run package (`bash code/experiments/package/run.sh --platform <name> <tier>`, `experiments/package/README.md`) runs `gen_bench.py` in its `gen` tier (Llama-2-7B
and Qwen3-4B, `--prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre`, 3 timed responses per cell; GPT-2, `--prompt 64 --gen 256 --decode greedy`) and, with
access to the gated Llama-2 checkpoints, `gen_bench.py --real meta-llama/Llama-2-7b-hf` in its `real` tier;
`paper/numbers.py` turns the `"kind": "gen"` records into `tab:gen`, `tab:gen-full` and the text's macros. The sampler
distances of Sec. `sec:gen-sampling` are those of the certified OPT-1.3B (the default command above, a CPU job of
about an hour, the package's `tv` tier); the stored `artifacts/results/sampler_tv_opt-1.3b.json` holds the same
analysis of the percentile recipe (`--recipe legacy`).

A small version runs on a laptop CPU in seconds (no downloads, no GPU):

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.generation.gen_bench --model tiny-llama --prompt 8 --gen 4 --decode greedy sample \
    --modes C Kpre --queries 2 --device cpu --threads 2
python -m experiments.generation.sampler_tv --selfcheck --selfcheck-mc 20000
```

`tests/test_sampler_tv.py` checks the law (every noise draw of small tables fed through `TokenSampler.choose`, the
FFT against the direct sum, the skipped-tail bound, the logits cache's settings); `tests/test_real_llama_end_to_end.py` the
generated responses of a real Llama-family graph and `gen_bench.py`'s tamper cells.
