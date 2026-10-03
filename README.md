# Certified but Compromised: Breaking and Fixing Lightweight Proofs of Inference

Final project, Trustworthy Machine Learning (Tel Aviv University, Spring 2026).

A company that outsources inference to a cloud provider cannot tell whether the provider ran the
agreed model or changed the answer on purpose. We study the lightweight proof of inference of Anchuri
et al. (SaTML 2026), in which the client re-checks one random path through a committed execution
trace, and show that sampling misses targeted changes: a provider that tampers with a single neuron
and computes everything else honestly almost always passes the path check, on image classifiers and
on the real OPT-6.7B, and no sampling rule closes this gap on every network. We therefore build a
protocol that checks every layer with Freivalds' algorithm against a Reed–Solomon/Merkle commitment to
the weights, with no SNARK, trusted hardware or trusted setup, in a basic and an optimised form. In
our experiments it rejected thousands of attacks without a miss; its prover is orders of magnitude
faster than published zkSNARK provers, and its main cost is a proof that grows with the input.

| Folder | Contents |
|---|---|
| [`report/`](report/) | the paper: `main.pdf` and its LaTeX source ([README](report/README.md)) |
| [`code/`](code/) | the implementation, one folder per experiment, the stored measurements, and a [README](code/README.md) with the installation, the tests and the commands that reproduce every table, figure and number of the paper |

To reproduce the results, start with [`code/README.md`](code/README.md): every table and figure is
regenerated from the stored measurements in about a minute without a GPU, and each experiment's
README describes how to re-measure it from scratch. The submission tarball holds `code/` and
`report/` only, made with `git -c core.autocrlf=false archive --format=tar.gz --prefix=<groupname>/ -o <groupname>.tar.gz HEAD code report`
so that every file keeps its LF line endings (see *Submission tarball* in `code/README.md`).
