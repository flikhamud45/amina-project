# H4: submission compliance (anonymous artifact, HotCRP fields)

Plan 2, item H4 (`SP2027_PLAN2.md`, Sec. 3.8). Code: `scripts/export_anonymous.py`, `scripts/artifact_README.md`
(commit de819fb). Texts: `docs/submission/HOTCRP_FIELDS.md`.

## 1. Why

The S&P 2027 call for papers requires an anonymous submission whose artifacts do not reveal the authors, strongly
encourages an anonymised artifact repository frozen at submission, asks theory papers to submit their proofs as artifacts,
and makes the "Ethics considerations" and "Generative AI usage considerations" fields mandatory at registration.
Non-compliance is grounds for desk rejection. The repository, however, has 356 commits by named authors, about 1,650
result files that embed cluster home paths, and plan, review and documentation files that name the team and the machines.

## 2. The export tool

`python scripts/export_anonymous.py --out <empty directory outside the repository> [--ref HEAD] [--pdf main_ext.pdf]`

1. **Select.** `git archive` of an allowlist (`code/src`, `code/tests`, `code/experiments`, `code/artifacts/results`,
   `code/artifacts/comparison`, the requirement files), minus job scripts (`*.sbatch`, the SLURM folder) and caches.
   Only committed files leave, and the history is not copied. Plans, reviews, `docs/`, the paper sources and the course
   report are not exported.
2. **Rewrite** (text files, detected by content, so `*.jsonl.part.<n>` records are included): the cluster project path
   becomes `$PROJECT`, every directory under a home directory `$HOME/`, Windows and Git-Bash user directories
   `C:/Users/user`, cluster host names `node`, login hosts `cluster-login`. Names of commit authors in provenance notes
   ("run by ...") become "a team member".
3. **Add** the artifact README (results map, installation, quick tests, the commitment recipe) and, with `--pdf`, the
   extended version.
4. **Refuse.** Every exported file is scanned (text as text, PDFs through their extracted text and metadata, other
   files as bytes) against a deny list built at run time: every commit author's name, name tokens of at least four
   letters and e-mail address; the owners and names of the remotes; fixed institutional and cluster strings. Generic
   identities ("research", "claude", ...) are not denied. While anything matches, the script lists every match and exits
   with status 1. It also refuses a non-empty target or one inside the repository.

**Dry run** (laptop, HEAD = 39a3dd6): the first runs refused, correctly, on 81 then 2 matches (a teammate's directory in
home paths of the zkLLM timing records; a name in two `PLATFORM.json` provenance notes; one false positive on a generic
commit identity), which led to the rules above. The final run exported 4,107 files (293 MB, 1,974 rewritten) with a clean
scan, and the snapshot's tests run from it (`tests/test_fs_tree.py`: 5 passed).

## 3. HotCRP fields

`docs/submission/HOTCRP_FIELDS.md` holds both texts, starting from REVIEW2's drafts, with the facts the repository records
filled in (test counts, the V1 adversarial review's five confirmed findings). The team must confirm the AI-use statement
and supply the proof hand-check (TH-f) and the disclosure dates and replies (G5); the disclosure notices themselves are
drafted in `sp2027_research/attack_theory/disclosure.md`.

## 4. What remains

* The disclosure notices (team), then the conditional sentences in `intro.tex` and `attack.tex` become facts (H4 (c)).
* An anonymous host for the snapshot (Anonymous GitHub or GitFront: the team's choice) and its link in the `fullversion`
  entry; the extended version with 0 `\todo`s.
* D6a's recipe and digests in the README once fixed (Mon 19 Oct).
* ~~A dry run of the README's quick tests on a clean machine from the snapshot.~~ Done on 12 Oct (Sec. 4.1).

### 4.1 The snapshot's quick tests (12 Oct)

The snapshot was exported into an empty folder outside the repository and the README's quick tests were run there
with the laptop's Python (Windows, torch 2.2.2, 2 threads), not in the working copy:

* **First run: one failure.** `test_wire.py::test_the_parameters_meet_lambda_on_every_model` loads
  `code/artifacts/models/mlp_mnist_full.npz`, which the export left out (`INCLUDE` had no `code/artifacts/models`).
  The 15 tracked model files (about 11 MB: the MNIST/digits MLPs and CNNs, their JSON descriptions and
  `MODELS.sha256`) are now exported.
* **The scan then refused two of them:** a short author name matched random bytes inside the compressed weights of
  `mlp_digits_01234.npz` and `mlp_mnist_halfA.npz`. An `.npz` file is now scanned as what it can carry besides
  numbers (`_npz_text`): the zip comment, the member names, every member's dtype and shape, and the values of every
  member that is not numeric (strings and bytes decoded, structured arrays as values, object arrays and non-array
  members as raw bytes). A check with four planted names (a Unicode string array, a bytes array, an object array and
  a structured array) catches all four; the byte-level scan had missed the Unicode one (UTF-32 storage).
* **Second run:** export of 4,177 files, "OK: no identifying string found"; the quick tests pass from the snapshot
  (exit code 0; a few tests skipped, e.g. three in `test_wire.py`).
