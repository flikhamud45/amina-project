# PA: an absorption-checked prefix rule for the Llama-family recipe

Code: commits baa26aa and 39ee815 (branch `sp2027-artifact`). `pvi/models/hf_llama.py` (`absorb_tokens`,
`_absorb_search`, `ABSORB_MARGIN`, `ABSORB_MAX_LEN`); `experiments/common/checkpoints.py` (`prefix_of`);
`experiments/quality/ppl_llama.py`, `llama_diag.py`, `llama_attrib.py` (`--prefix absorb`, `--absorb-margin`,
`--absorb-max-len`); `experiments/quality/prefix_check.py` (new); `experiments/package/run.sh` (`--llama-prefix`,
`--absorb-margin`, `--absorb-max-len`, `--jobs`, tiers `prefixcheck` and `qprefix`); `experiments/paper/records.py`,
`macros.py` and `numbers.py`.

## 1. Why

The integer Llama-family recipe puts a short fixed public prefix in front of every prompt. Its keys and values enter
every attention op as int8 constants, bound by the digest. The prefix serves one purpose: it takes in the model's
massive activations (the few residual-stream values that are orders of magnitude above the rest), so that they sit at
the prefix's positions and not at the prompt's. The integer residual scale `s_res` is set by the fp residual stream's
largest |value| over the calibration windows' own positions. If a token of the prompt itself still produces a value
above the prefix's largest, that value sets the scale, and int8 can no longer represent the ordinary values of the
stream.

The prefix is chosen by PrefixQuant's rule (`--prefix auto`, `hf_llama.prefixquant_tokens`): the `o - 1` token ids
found most often among the outlier tokens of the down projections' inputs, then the BOS. The rule counts the outlier
tokens without any prefix, and its `o - 1` tokens can omit a delimiter whose first occurrence in a window then forms a
massive activation of its own. Nothing in the rule checks that the chosen prefix absorbs.

The quality records of the final run's partial results show this failure on Llama-2-7B. Its automatic prefix,
`"\n <s>"`, leaves a prompt activation about 2.2 times the prefix's largest value, and the integer model's perplexity
is about 1,450 against 5.5 for the fp32 reference, at every setting of the recipe. The other checkpoints absorb: the
ratio of the windows' maximum to the prefix's is 0.27 for Llama-2-13B, 0.93 for TinyLlama and 0.37 for Qwen3-1.7B.
Qwen3-4B, at 3.06, does not absorb either.

## 2. The rule

`hf_llama.absorb_tokens(model, calib_ids, margin=1.0, max_len=6)` (`--prefix absorb`) extends PrefixQuant's choice
until it absorbs. It follows the greedy search of CushionCache (Son et al., EMNLP 2024), which adds prefix tokens one
at a time:

1. The first candidate is PrefixQuant's choice, from the same statistics on the same calibration windows.
2. A candidate absorbs when the fp residual stream's largest |value| over the windows' own positions, measured with the
   candidate in front (`_hidden_max`, the value behind `s_res`), times `margin` lies below the largest over the prefix
   positions.
3. While the candidate does not absorb, is shorter than `max_len` tokens (BOS included) and outlier tokens remain, the
   next most frequent outlier token of PrefixQuant's statistics is added after the outlier tokens already chosen and
   in front of the BOS. The BOS stays last, PrefixQuant's order: position 0 holds the initial-token activation
   whatever its token, so moving the BOS forward would remove no window's activation, whereas a delimiter's
   occurrence in the prefix is what takes its first occurrence's activation out of the windows.
4. The rule returns the first candidate that absorbs. If none does, it returns the candidate of the smallest ratio
   (the earlier on a tie) and records it as not absorbed.

Where PrefixQuant's prefix already absorbs, the rule returns exactly that prefix, and therefore the same graph and
the same digest. The prefix, whichever rule chose it, enters the graph as any fixed prefix
(`build_llama_from_hf(prefix_tokens=)`): only constants change, and the structural diff stays empty.

The record's `prefix_choice` holds the rule (`absorb`), PrefixQuant's choice (`auto_tokens`), `margin`, `max_len`,
every candidate (`tokens`, `text`, `window_max`, `prefix_max`, `ratio`, `absorbed`), whether the chosen prefix
absorbs, and why the search stopped (`absorbed`, `max_len` or `no outlier token left`).

## 3. Opt-in

PrefixQuant's rule remains the main recipe's default. Without the new options every script and every job is as
before: a record of `--prefix auto` has no field of the new rule, and the dry runs of `run.sh` without the options
list the same jobs with the same commands as before the change (compared over the full, licensed, rehearsal and
one-job dry runs; only the help text differs).

## 4. The check and the re-run in the run package

* **`prefix_check.py`** measures, in fp and without any integer graph, whether a prefix absorbs. On the recipe's
  calibration windows (draw 0 of `ppl_llama.py`) it runs each candidate prefix in front of every window and prints,
  per checkpoint (default: Llama-2-7B, Llama-2-13B, TinyLlama, Qwen3-1.7B and Qwen3-4B), the prefix's and the
  windows' residual maxima, their ratio, and where the windows' maximum sits (block boundary, window, position,
  channel and token; `--per-block` adds both maxima per boundary). The candidates are PrefixQuant's choice, every
  candidate of the absorption rule, the BOS alone, a few fixed extras (among them Llama-2-7B's prefix as PrefixQuant
  publishes it, `. \n [BOS]`) and any `--prefixes` given. Each checkpoint ends with the prefix that `auto` and
  `absorb` pick and whether the latter absorbs. `--out` appends JSON lines. The checkpoints run in fp16 on one GPU;
  the rule inside `ppl_llama.py` measures in fp32 on the CPU, so a ratio within rounding of 1 may fall on the other
  side there.
* **`run.sh --llama-prefix auto|absorb`** (default `auto`) sets the rule of every Llama-family quality job: the
  `quality` tier's TinyLlama and Qwen3 cells and the `qllama` and `qcheck` tiers. With `absorb` these jobs run
  `ppl_llama.py --prefix absorb --absorb-margin X --absorb-max-len N` and are named `sp-<cell>-absorb`, so that a
  `--local` or `--one-job` run runs them although the jobs of PrefixQuant's rule have ended successfully. Qwen3-4B's
  and OpenLLaMA-7B's fp32 CPU references are cached per rule (`ref_*-absorb.pt`).
* **`--absorb-margin`** (default 1) and **`--absorb-max-len`** (default 6) are accepted only with
  `--llama-prefix absorb` or the `prefixcheck` tier; a non-positive margin or a length below 1 stops the submission.
* **Tier `prefixcheck`** (not in `all`): one job, `q-prefix-check`, running `prefix_check.py` on the five checkpoints
  (Llama-2 only with `--llama-licence`), with records in `logs/<name>/prefix_check.jsonl`.
* **Tier `qprefix`** (not in `all`): the Llama-family quality jobs alone, with `--llama-prefix`'s rule, so that every
  Llama-family number of the text can be measured again: 20 jobs and 54-70 h with the licence, 10 jobs and 24-36 h
  without.
* **`--jobs LIST`** submits only the named jobs of the given tiers (names as the dry run lists them, without `sp-`). A
  name that none of the tiers has stops the submission. With it, Llama-2-7B's seven quality jobs (about 30 h) or
  Qwen3-4B's one job (about 4 h) can be re-run alone.

## 5. The paper's numbers

`numbers.py` reads the latest job's record of each cell, so the `-absorb` jobs, submitted after the others, are the
ones the paper reads. Its report names the prefix rule of the Llama-family records it read.

* `records.prefix_rule` gives a record's rule. `records.prefix_absorbs` decides from a record's own fields whether
  its prefix absorbs (`residual_real_max` over the residual headroom, times the margin, below `prefix_hidden_max`).
  `records.absorb_consistent` accepts a record built by the absorption rule, or one built by PrefixQuant's rule whose
  prefix absorbs, since the rule would have picked the same tokens and the same graph.
* `numbers.py --llama-prefix any|auto|absorb` (default `any`) selects records by rule: `auto` keeps only PrefixQuant's
  records; `absorb` prefers the latest record that agrees with the absorption rule. `--absorb-margin` defaults to the
  margin that the run's `absorb` records name, else 1.
* `macros.llama_prefix` checks the text's claim that every Llama-family number comes from one rule. Under the
  absorption rule (`--llama-prefix absorb`, or any record read that was built by it), a cell of PrefixQuant's rule
  whose prefix does not absorb is a `CHECK FAILED text:` line that names the cell and asks for it to be measured
  again with `run.sh --llama-prefix absorb`. Otherwise, records of several rules among the cells read fail the claim.

Only the checkpoints whose prefix the rule changes therefore need to be measured again; the finished records of the
others remain valid.

## 6. Tests

`tests/test_prefix_absorb.py`, on synthetic statistics and tiny random Llama models (no download):

* PrefixQuant's rule gives what it gave before, and its record has no field of the new rule; the absorption rule
  starts from it;
* the search extends the prefix in front of the BOS until a candidate absorbs, keeps PrefixQuant's prefix when it
  absorbs, stops at the longest length or when no outlier token is left (keeping the candidate of the smallest ratio,
  recorded as not absorbed), records every candidate, and refuses a non-positive margin or a length below 1;
* a prefix chosen by the rule builds the same graph, digest included, as the same token ids given by hand;
* the scripts accept the rule's parameters with the defaults 1 and 6;
* `prefix_check.py` measures the maxima that `_hidden_max` measures, and prints every candidate and the rule's pick;
* the paper's tooling names the rule of the records it reads, counts an absorbing cell of PrefixQuant's rule as the
  rule's, fails the claim on a non-absorbing one, and selects by rule.

`tests/test_run_package.py`: with `--llama-prefix absorb`, exactly the 20 Llama-family quality jobs run with
`--prefix absorb` and the rule's parameters under `<job>-absorb`, with their references apart, and every other job is
unchanged; without the option no job names the rule; `prefixcheck` is one fp job and neither new tier is in `all`;
`--jobs` selects one checkpoint's jobs and stops on an unknown name; invalid rules, margins and lengths stop the
submission. `tests/test_real_llama.py` gives its hand-built `ppl_llama.py` arguments the two new options (39ee815).

## 7. What remains

* Run `prefixcheck` on the final platform and confirm that the rule's prefix for Llama-2-7B absorbs and differs from
  PrefixQuant's; likewise for Qwen3-4B.
* If so, re-run Llama-2-7B's seven quality jobs (and Qwen3-4B's one) with `--llama-prefix absorb`, and read the paper's
  numbers with `numbers.py --llama-prefix absorb`. `FIXES3.md` in `docs/submission` (the handoff for commit 39ee815)
  describes these steps.
* If no candidate of at most six tokens absorbs for Llama-2-7B, the integer Llama-2-7B's quality remains a stated
  limitation.
