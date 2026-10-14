# Certified but Compromised: Breaking and Fixing Lightweight Proofs of Inference

The LaTeX source of the paper and its artifact: the code, the stored measurements, the run that produces every
number of the paper, and the pipeline from that run's records to the paper's tables and text.

| Folder | Contents |
|---|---|
| [`code/`](code/README.md) | the artifact: the library (`pvi`), the experiments by section of the paper, the tests, the stored models and results; installation, tests and reproduction in its README |
| [`paper_sp2027/`](paper_sp2027/README.md) | the paper (`main.tex`) and its extended version with every proof (`main_ext.tex`) |

## Reproducing the paper

1. **Install and test**: [`code/README.md`](code/README.md).
2. **Run every experiment on one machine**: [`code/experiments/package/README.md`](code/experiments/package/README.md).
   One command submits every job; a dry run lists them, and a rehearsal runs every script of each tier on small cells.
3. **From the run's records to the paper**: [`code/experiments/paper/README.md`](code/experiments/paper/README.md).
   `python -m experiments.paper.numbers` checks the records and writes every table body and text number into
   `paper_sp2027/generated/`.
4. **Build the paper**: [`paper_sp2027/README.md`](paper_sp2027/README.md).

Each experiment can also be run on its own: every folder of [`code/experiments/`](code/experiments/README.md) maps
the paper's results to commands, with a small configuration that runs on a laptop CPU.
