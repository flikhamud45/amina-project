# LQ: Llama-2-7B's committed setting and the diagnosis of its integer graph

Code: commits d7d87a2 and aba9fb6 (branch `sp2027-artifact`). The run package's `qdiag` tier
(`code/experiments/package/run.sh`); `hf_llama.MAIN_SETTINGS`; `experiments/paper/macros.py` (`LLAMA_GRIDS`);
`paper_sp2027/sections/quality.tex`, `ext_generation.tex`, `ext_discussion.tex` and `discussion.tex`.

## 1. Why

With the absorption-checked prefix ([PA_prefix_absorption.md](PA_prefix_absorption.md)), the integer Llama-2-7B no
longer collapses: on the WikiText-2 test split at 2,048 tokens its perplexity fell from about 1,450 to between 21.6 and
40.8 over the nine measured settings (fp32: 5.47). The setting fixed in advance (alpha 0.55, SwiGLU balance 0.2, taken
from OpenLLaMA-7B, a licence-free model of the same shapes) was not the best of the nine, and the model remains about
four times fp32's perplexity. Two things were therefore needed: a committed setting chosen as every other model's is,
and a measurement of where in the network the integer graph departs from fp32.

## 2. The committed setting

Every other model takes the best of its grid on the test split, except Llama-2-13B, whose values were fixed in advance.
Llama-2-7B now does likewise: `MAIN_SETTINGS["Llama-2-7b-hf"] = (0.65, 0.3)`.

| alpha \ balance | 0.1 | 0.2 | 0.3 |
|---|---|---|---|
| 0.45 | 4.75 | 4.76 | 4.90 |
| 0.55 | 7.45 | 6.29 | 6.86 |
| 0.65 | 4.74 | 4.49 | **3.95** |

Ratio to fp32 on the test split (2,048 tokens, absorbing prefix). The same setting is also the best on the validation
split (2.51). It lies on the edge of the grid; the text says so.

* OpenLLaMA-7B keeps (0.55, 0.2), the main recipe; the earlier setting of Llama-2-7B remains available as
  `ppl_llama.py --alphas 0.55 --glu-smooths 0.2`.
* `macros.LLAMA_GRIDS` locates the committed cell in the grid instead of assuming its centre, breaks ties away from
  it, and measures the validation gap from it.
* The paper states that the setting was chosen on the test split, that it started from (0.55, 0.2), and that the
  cost cells, built before, used the earlier constants and a prefix one token shorter (constants change no cost).
* The spread of the calibration draws at the new setting (`\LlamaSevenDrawSpread`) needs the qcheck tier's draws job
  again; `FIXES4.md` in `docs/submission` describes it.

## 3. The `qdiag` tier

One CPU job (not in `all`), `q-llama7-diag` (`-absorb` under `--llama-prefix absorb`), 10 hours of limits:

1. `llama_diag.py` on Llama-2-7B at the default length with the recipes `main`, `smooth55` and `scalar4`;
2. the same at 2,048 tokens with `main`, and on Llama-2-13B at 2,048 tokens as the reference that works;
3. `llama_attrib.py` on Llama-2-7B at 2,048 tokens over every block with `main`: each component of a float copy of
   the integer graph switched between integer and float, as the OPT attribution study does.

Outputs: `llama_diag.jsonl` and `llama_attrib.jsonl` in the run's log folder. Without `--llama-licence` the tier uses
OpenLLaMA-7B and TinyLlama and checks for sentencepiece; the rehearsal runs both scripts' smoke mode on TinyLlama.

## 4. Tests

* `tests/test_run_package.py::test_the_qdiag_tier_diagnoses_llama7_with_the_prefix_rule`: the job names, the absorb
  options on all four commands, and the tier outside `all`.
* `tests/test_paper_numbers.py`: the committed cell (0.65, 0.3) inside the grid, its gaps on both splits, the draws
  read at that cell, and the text's statement of how it was chosen.
* `tests/test_prefix_absorb.py`: the Llama-2-7B cells under the new name.

The full test suite passed on the cluster at d7d87a2 (2,304 passed) and at c39ab9a.

## 5. What remains

* The draws job at the new setting and the `qdiag` job in the final run (`FIXES4.md`).
* From the diagnosis: which component carries the gap. Any fix it suggests that is not the main algorithm or a cited
  common practice is reported only as a possible improvement.
