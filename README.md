# Certified but Compromised: Breaking and Fixing Lightweight Proofs of Inference

Final project, Trustworthy Machine Learning (Tel Aviv University, Spring 2026).

We study the lightweight proof of inference of Anchuri et al. (SaTML 2026), which
checks a model's computation along one random path. We show that changing a single
neuron defeats it, prove that smarter sampling cannot fix this, and build a defence
that checks every layer at once (Freivalds' algorithm against a Reed–Solomon/Merkle
commitment to the weights). We evaluate it on the models used by the original paper
and by 17 other systems, from LeNet-5 to Llama-2-13B.

| Folder | Contents |
|---|---|
| [`report/`](report/) | the report: `main.pdf` and its LaTeX source |
| [`code/`](code/) | the implementation, one folder per experiment, the stored measurements, and a [README](code/README.md) with exact commands to reproduce every table and figure |

The submission tarball (`<groupname>/code`, `<groupname>/report`) is this branch as
committed:

```bash
git archive --format=tar.gz --prefix=<groupname>/ -o <groupname>.tar.gz submission
```
