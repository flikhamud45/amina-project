# SP: submission polish of the code, its documentation and the paper

Code: commit 93d6aa3 (branch `sp2027-artifact`), with the paper part of baa26aa and the abstract commits 415d5a8,
d3fd354, df1a63b and 7b5199b. Comments, docstrings and READMEs throughout `code/`; the paper tooling in
`code/experiments/paper/`; `paper_sp2027/`.

## 1. Why

The artifact is read by reviewers and by the artifact evaluation committee without any other context. A comment that
describes an earlier version of the code, a README that omits a directory's scripts, or a parameter that can only be
changed by editing the source each cost a reader time and trust. The paper had grown by successive additions, and its
body had to fit 13 pages with its evidence in the body itself.

## 2. What was checked in the code

* **No stale comments.** Comments and docstrings were read against the code they describe and corrected where they
  named an earlier behaviour, a removed option or a wrong default.
* **Comment-only changes verified.** Every file whose change was meant to touch comments only was compared with its
  previous version by its syntax tree, with docstrings removed: the trees are identical, so the code itself is
  unchanged.
* **A README per directory.** Every experiment directory (apart from the shared helpers of `experiments/common`) has
  a README that names its scripts, their commands, their outputs and the paper results they feed. The library's
  subpackages are described in `code/src/pvi/README.md` and the protocol in `code/src/pvi/PROTOCOL.md`; `code/README.md`
  describes `tests/` and `tools/`.
* **Every parameter a command-line argument.** Every parameter that selects a code path or a setting of an
  experiment is a command-line argument of its script (the library's options through `pvi.options`), so that no
  measurement requires editing the source.

## 3. Changes of behaviour, all in the paper tooling

* `tab:eval-v1` prints shorter time cells, so that the table fits its column.
* The source header of `tab:eval-reexec` names the weights the run used.
* A negative range of `QualityProbCost` prints true minus signs.
* The break-even figure's legend and fonts are easier to read.

No proof, record or measured value changes.

## 4. The paper

* The body tells one story and carries its own evidence. Each caption states its take-away in one sentence, and the
  setup moves to the table notes.
* The abstract states the problem, the attack, the impossibility result, the defence and how it compares, without
  exact numbers; the introduction, evaluation and discussion give the CPU verifier's time against exact re-execution
  and the key's size against the weights.
* The body names every implementation improvement and the two bounded-memory modes of the timed Llama-2 cells. The
  extended version explains the lean prover, the int32 claims, the allocator setting and the key derivation's time
  before and after, says which table rows group several improvements, and gives P_max as 2,047 in every measured
  cell.
* Table 1's headings and spacing, headings that hyphenated, and the references' author names, venues and notes are
  corrected.
* The body ends on page 13, 18 pages in all, within the call's limit of 13 pages of body and 5 of references and
  appendices.

## 5. Tests

The full CPU test suite passed on the cluster at 93d6aa3, and at baa26aa apart from one test whose hand-built
arguments lacked the new prefix options; 39ee815 adds them, and that test passes.

## 6. What remains

* The full CPU test suite again at the commit that is finally frozen, if it differs from 39ee815.
