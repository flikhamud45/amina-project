# NC: two checks of the paper's numbers that failed on correct logs

Code: commits f920656 and c39ab9a (branch `sp2027-artifact`). `code/experiments/paper/numbers.py`
(`SAME_VALUE`, `main`), `code/experiments/paper/records.py` (`log_checks`).

## 1. Why

`numbers.py` exits with status 1 on any `CHECK FAILED` line, so that no paper number comes from a failed job or a
table that contradicts the text. On the final run's logs it failed three times although no measurement was wrong:

* the CPU table's 8-thread Llama-2-7B row printed 19.1 s and 1.8x, against `\CpuReexecLong` 12.6 s and `\CpuRatio`
  1.2x;
* the llama.cpp job of Llama-2-7B had failed once (on an earlier commit its tokenizer needed sentencepiece) and
  succeeded on its rerun, but its first log still carried the `FAILED:` line.

## 2. The CPU table

Since 27 October `tab:eval-cpu` compares every row with re-execution under glibc's allocator defaults, so that its
rows compare like with like, while `\CpuReexecLong` and `\CpuRatio` take the faster of both allocator settings; the
appendix states the difference. The check of cells against macros still paired the table's 8-thread row with the
latter two macros. It now compares the row's ratio with `\CpuRatioThrEight`, the macro the appendix prints beside
it. No macro prints that row's re-execution, so it is no longer compared.

## 3. Rerun jobs

The run package reruns a job that did not end successfully and keeps every log, each named by the time its job
started in nanoseconds. `log_checks` now reads a script's `FAILED` line (or a smoke's missing verdict) as a `WARNING`
line when a later job of the same cell, that is a larger job id, ended without one. A clean earlier job excuses no
later failure, and a job of another cell excuses nothing.

## 4. Tests

`tests/test_paper_numbers.py`:

* `test_the_fallback_tables_agree_with_their_macros`: the 8-thread row is no failure when `\CpuRatio` and
  `\CpuReexecLong` take the tuned allocator, and is one when `\CpuRatioThrEight` disagrees;
* `test_a_failed_job_that_a_later_job_reran_only_warns`: one failed log fails; a later clean job of the cell turns it
  into a warning; a later failure of the same cell still fails, and a clean job of another cell changes nothing.

On the final run's logs of 5 November, `numbers.py` ends with status 0, the earlier llama.cpp failure as a warning.
The full test suite passed on the cluster at c39ab9a.
