# PN: the paper's values not yet measured, marked; partial results merged

Code: commit 256b8d6 (branch `sp2027-artifact`), with the README corrections of 93d6aa3. `paper_sp2027/main.tex`
(`\notready`), `paper_sp2027/fallback/`, `code/experiments/paper/numbers.py` (`unmark`, `folder_files`),
`paper_sp2027/README.md`, `code/experiments/paper/README.md`.

## 1. Why

Every table body and every measured number of the paper's text is read from `paper_sp2027/generated/<name>.tex` when
that file exists, and otherwise from `paper_sp2027/fallback/<name>.tex` (`\pviresolve` and `\pvitable` in
`main.tex`). `numbers.py` writes `generated/` from the final run's records. Until the final run has finished, the PDF
therefore prints the fallback, which holds the values of earlier runs: random weights, other nodes, or an earlier
commit.

These values look exactly like measured ones. A reader of a draft, or an author preparing the abstract, cannot tell
which numbers are final, which will be replaced, and which never depend on a run. The draft must show this
distinction.

## 2. Three kinds of value, three colours

* **Black:** a value that needs no run: published numbers of other systems, labels, and values computed by formula
  (for example the parameters and the byte model).
* **Orange:** a value of `fallback/` that the final run replaces. It is wrapped in `\notready{...}`, defined in
  `main.tex` as `\textcolor{orange!85!black}{#1}`.
* **Red:** a value that no run has measured yet, printed by the existing `\todo{...}` as `[TODO: ...]`.

The marked values are exactly those that `numbers.py` leaves missing when it runs on a run folder without records.
Such an empty run fills only the values that need no run, so every fallback cell or macro that it leaves empty, and
that prints a value (neither a `\todo` nor `--`), is one the final run replaces. In the commit this marked 495 table
cells and 382 text numbers. The marks were added once to the files of `fallback/`, by a script outside the repository
that compares the fallback with the empty run's output cell by cell.

`generated/` never carries the mark: a file that `numbers.py` writes prints black, so the final PDF has no orange
value. The page count did not change (18 pages for the submission, 55 for the extended version).

## 3. Code that reads the fallback

The checks and tests compare fallback values with computed ones, so they must not see the marks.
`numbers.unmark(text)` removes every `\notready{...}` and keeps its content; it handles up to three levels of nested
braces, which covers cells such as `\textbf{2.2\,s (1{,}213$\times\downarrow$)}` and marks inside `\multicolumn`.
`numbers.folder_files`, which reads a folder of the paper's files (`generated/` or `fallback/`), applies it to every
table and to `numbers.tex`. The tests that read the fallback directly now call `unmark` or `folder_files` instead of
their own regular expressions.

## 4. Merging partial results

While the final run is in progress, a snapshot of the finished jobs (collected as described in
`docs/submission/PARTIAL_RESULTS.md`) can update the draft. `numbers.py --allow-missing` on such a snapshot writes a
complete `generated/` in which every value the snapshot lacks is a red `[TODO: ...]`. Used as it is, that folder would
replace orange fallback values by red markers.

The merge therefore combines the two folders, cell by cell and macro by macro: every value that the snapshot provides,
and otherwise the fallback's value, which keeps its orange mark. A table is merged only when it has as many rows as
the fallback's, and a row only when its cells span the same columns; otherwise the snapshot's file or row is kept as
it is. The result builds a PDF in
which black values are final or need no run, orange values still await the run, and red values have no measurement
at all. The merge is a working script outside the repository and writes into a separate folder: the repository's
`fallback/` and `generated/` are unchanged by it, and the final run's `generated/` supersedes it.

## 5. Tests

* `test_paper_numbers.py`: the files that `numbers.py` writes contain no `\notready`; the fallback contains it;
  `unmark` removes nested and `\multicolumn` marks exactly.
* `test_paper_numbers.py` and `test_paper_numbers_derived.py`: every comparison of computed values with the fallback
  (the attribution table, the setup proof, the stored attack documents, the derived numbers) reads the unmarked
  fallback.
* `numbers.py`'s own check that a table cell and the macro printing the same value agree runs on the unmarked
  fallback as well.

## 6. What remains

* After the final run, check that every file is read from `generated/` and that the submitted PDF prints no orange
  or red value.
