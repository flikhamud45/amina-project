# G3.5. In-range edits on a real LLM against range checks and a value-aware sampler

Plan item G3.5 (reviewer objection D4, "the attack is weak"; mock review, Reviewer A, weakness 2 and question 3:
"A range check, which costs the path protocol almost nothing, would stop it ... How many neurons does an in-range
next-token change need?"). Script `code/experiments/2_attack/inrange_llm.py` (commits 4ce69b1 and later); results
`code/artifacts/results/inrange_llm_opt-1.3b_{any,down}.json`. Status: done for OPT-1.3B on the laptop CPU; OPT-6.7B
is in the L40S package (`sp2027.sh attack`).

## 1. The question

The single-neuron attack on OPT-6.7B (`REAL_LLM.md`) changes the next token for all 40 prompts, but only 12.5% of its
forged values stay inside the neuron's natural range, and the backdoor's values never do. A spot-checker could add a
per-neuron range check on every activation it opens, at almost no cost. Two questions follow: can an attacker that
keeps every value inside the checked range still change the answer, and how many neurons must it edit, which is
what decides how often a sampled path opens one of them?

## 2. Method

* **Ranges.** The FFN activations of the chosen layers (the post-ReLU inputs of `fc2`) over 64 windows of 128
  tokens of WikiText-2 *train*, every position (8,192 tokens), which neither the prompts nor any quality evaluation
  uses. Per neuron: the maximum (`max`) and the 99.9th percentile (`p999`); the minimum is 0 (ReLU). A smaller
  calibration set gives tighter ranges, so it favours the verifier.
* **Attack.** At the last position of one layer (the only position whose value reaches the next-token logits),
  greedily: compute the gradient of the margin between the top-1 and the runner-up logit with respect to the
  layer's activations; for every neuron, the first-order decrease of the margin from moving it to the bound of its
  range in the helpful direction (to 0, or to the top of its range); move the best neuron (four at a time after 32);
  repeat until the next token changes or 256 neurons are edited. Every edited value lies inside its range, so the
  range check passes. Variant `--moves down`: only decreases (to 0).
* **Detection.** A path of Anchuri et al. opens one neuron of the edited layer, uniformly over `d_ff = 8,192`, so
  it catches `k` edited neurons with probability `k / d_ff`; the paths needed for `1 - 2^-40` are
  `ln(2^40) / -ln(1 - p)`. The contribution-weighted sampler of Section 3.5 opens neuron `j` with probability
  proportional to `|a_j| ||W_fc2[:, j]||` on the claimed (edited) trace.
* **Setup.** OPT-1.3B in fp32 on 8 threads of the laptop CPU, the 40 prompts of the OPT-6.7B experiment, layers 6,
  12, 18 and 23 (of 24). About 11 minutes per variant.

## 3. Results (OPT-1.3B, 40 prompts per cell)

| Moves | Range | Layer | next token changed | edited neurons, median (max) | uniform per path | contribution-weighted |
|---|---|---:|---:|---:|---:|---:|
| any | max | 6 | 40 | 3.5 (17) | 4.3e-4 | 0.55 |
| any | max | 12 | 40 | 3 (40) | 3.7e-4 | 0.11 |
| any | max | 18 | 40 | 2 (13) | 2.4e-4 | 0.054 |
| any | max | 23 | 40 | 1 (4) | 1.2e-4 | 0.029 |
| any | 99.9% | 6 | 40 | 6.5 (60) | 7.9e-4 | 0.57 |
| any | 99.9% | 12 | 40 | 5 (24) | 6.1e-4 | 0.097 |
| any | 99.9% | 18 | 40 | 3.5 (22) | 4.3e-4 | 0.053 |
| any | 99.9% | 23 | 40 | 1 (7) | 1.2e-4 | 0.022 |
| down | (either) | 6 | 9 | 3 (15) | 3.7e-4 | 0 |
| down | (either) | 12 | 31 | 12 (104) | 1.5e-3 | 0 |
| down | (either) | 18 | 38 | 8 (164) | 9.8e-4 | 0 |
| down | (either) | 23 | 39 | 2 (60) | 2.4e-4 | 0 |

(Detection columns: medians over the successful prompts. The down-only rows are identical for both ranges, as they
must be: a down move goes to 0 whatever the top of the range.)

## 3b. OPT-6.7B (Plan 2 G3.5-L, job 1007804: Xeon node s-006, fp32, 16 threads, 40 prompts per cell)

The same job first reran the OPT-1.3B cells. Every discrete result equals the table above in all 320 records (success,
number of edited neurons, which neurons, uniform detection); only the stored float values differ, by about 1e-6
(fp32 on the Xeon against the laptop). OPT-6.7B, layers 8, 16, 24 and 31 of 32 (the same relative depths), ranges
calibrated on 8,192 WikiText-2 train tokens:

| Moves | Range | Layer | next token changed | edited neurons, median (max) | uniform per path | contribution-weighted |
|---|---|---:|---:|---:|---:|---:|
| any | max | 8 | 40 | 10.5 (44) | 6.4e-4 | 0.76 |
| any | 99.9% | 8 | 40 | 22 (104) | 1.3e-3 | 0.74 |
| any | max | 16 | 40 | 6 (25) | 3.7e-4 | 0.15 |
| any | 99.9% | 16 | 40 | 12 (48) | 7.3e-4 | 0.15 |
| any | max | 24 | 40 | 5 (14) | 3.1e-4 | 0.067 |
| any | 99.9% | 24 | 40 | 7 (36) | 4.3e-4 | 0.071 |
| any | max | 31 | 40 | 1 (17) | 6.1e-5 | 0.028 |
| any | 99.9% | 31 | 40 | 1 (16) | 6.1e-5 | 0.021 |
| down | (either) | 8 | 4 | 3 (23) | 1.8e-4 | 0 |
| down | (either) | 16 | 23 | 25 (164) | 1.5e-3 | 0 |
| down | (either) | 24 | 37 | 36 (176) | 2.2e-3 | 0 |
| down | (either) | 31 | 40 | 3 (112) | 1.8e-4 | 0 |

The reading of Section 4 holds at 7B scale. In-range edits change the next token on 40/40 prompts at every layer
and range; one uniform path catches them with probability at most 1.3e-3 at the medians (so `1 - 2^-40` needs about
20,000-450,000 paths); contribution weighting catches the `any` edits often at the first layer (0.74-0.76 per draw)
and rarely at the last (0.02-0.03), and never catches the `down` edits, which succeed on 4-40 of 40 prompts.
Results: `code/artifacts/results/inrange_llm_opt-6.7b_{any,down}.json`.

## 4. Reading

* **Range checks do not stop the attack.** Edits that keep every value inside ranges calibrated on held-out text
  change the next token for all 320 (prompt, layer, range) cells, with a median of 2 edited neurons and at most 60.
  A uniform path opens one of them with probability at most 7.9e-4, so `1 - 2^-40` needs 35,000-227,000 paths at
  the medians.
* **Value-aware sampling is a trade-off, not a fix.** Pushing neurons to the top of their range raises their
  contribution, so contribution weighting catches those edits often (2-57% per draw). But lowering activations to
  zero, which stays in range after ReLU, gives the edited neurons zero weight on the claimed trace: that sampler
  never opens them (detection exactly 0), and the attack still succeeds for 9/40 (layer 6) to 39/40 (layer 23)
  prompts, with uniform detection at most 1.5e-3 per path. This matches Theorem `thm:guess`: a sampler that reads
  the claimed values can be steered by them.
* **Limitations.** Two models (OPT-1.3B and OPT-6.7B, Section 3b; Qwen3-4B only if the team asks), untargeted changes only, one
  layer at a time, greedy first-order moves (an upper bound on the neurons needed), and ranges from 8,192 tokens.

## 5. Usage

    cd code && PYTHONPATH=src python experiments/2_attack/inrange_llm.py --model facebook/opt-1.3b --layers 6 12 18 23
    PYTHONPATH=src python experiments/2_attack/inrange_llm.py --model facebook/opt-1.3b --moves down
    # on a 48 GB GPU: --model facebook/opt-6.7b --layers 8 16 24 31 --device cuda   (sp2027.sh attack)

## 6. In the paper

Section 3.5 ("A real LLM"), one sentence in the introduction and in the discussion, and Table `tab:inrange` in the
extended version.
