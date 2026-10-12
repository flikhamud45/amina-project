# IEEE S&P 2027 plan 2 (12 Oct to 17 Nov 2026)

**How this plan was made.** Five axis agents proposed the items: proof size; verifiers and long prompts; prover overhead; quality and real weights; attack, theory and paper. A scheduler turned the proposals into a dated draft. A critic then checked the draft against the measured docs, the mock review and the S&P 2027 CFP. All 27 of its corrections are applied here (Sec. 7). Plan 1 (`SP2027_PLAN.md`) stays as the record of what is done.

**Target: S&P 2027, cycle 2.** Abstract registration is **Tue 10 Nov 2026** and the paper is due **Tue 17 Nov 2026** (13 pages of body plus 5 of references and appendices). The fallbacks are **EuroS&P, Wed 2 Dec 2026**, and **USENIX Security cycle 2, Tue 26 Jan 2027**. A reject blocks S&P for a year. However, the early-reject notification comes on Mon 18 Jan 2027, 8 days before the USENIX deadline, so an early-rejected paper can still go to USENIX. A go therefore mainly risks the one-year S&P block, and the gate leans towards a degraded go (Sec. 5.1).

**Tags.**
- Status: **[TODO]**, **[WIP]**, **[USER]** (needs a decision or action from the team), **[DROP]** (decided against, reason given), **[DONE]** (with the commit and the doc in `docs/improvements/`).
- Priority:
  - **must:** S&P is not attempted without it.
  - **should:** done in rank order when there is slack; each has a last start date.
  - **could:** only if every should item above it is done; otherwise after 17 Nov.
  - **after 17 Nov:** parked for the EuroS&P/USENIX track. These items never appear in the schedule before the freeze.

**Review items** refer to `paper_sp2027/REVIEW.md`. A, B and C are the reviewers (systems security, cryptography, ML systems). W is a weakness and Q a question, so A-W1 is reviewer A's weakness 1. "chg n" is consolidated change n. Plan 1's objection codes D1 to D7 are retired.

**Effort** is in engineer-days, as the axis agents estimated it (implementation plus validation on the 2080 Ti). Sec. 1 explains why this unit is pessimistic. **Gains** are labelled (measured) or (model).

**Code and data.** Code is on branch `sp2027` (worktree S/wt_sp2027, on GitHub). Research notes are in `S/sp2027_research/`. The server copy is `logs/validation/sp2027/`. S is session 4aa2c028's scratchpad.

**State on 10 Oct.**
- Prover: 3.7-4.3x faster on Llama-2 blocks.
- GPU verifier: 2.0-2.7x faster, but fp16 re-execution on the 2080 Ti is still 2-5x faster than it.
- CPU verifier: 4.6x faster than fp32 re-execution at Llama-2-7B, 2,048 tokens.
  - This number is extrapolated and interactive.
  - It is measured against PyTorch fp32 and against an integer re-execution (98 s) whose weight products still run as fp32 GEMMs. It is not measured against the strongest exact baseline (RX).
- Proof size: 6.6 GB at Llama-2-7B, 2,048 tokens (v0), about 2.9-3.1 GB with V1.
- Quality: only OPT-125M and OPT-1.3B are measured, at 16 x 128 tokens.
- Paper: 18 pages.
  - It has **71 live `\todo` markers in the submission build**: 37 in the evaluation, about 45 of them waiting for L40S/EPYC numbers. The brief's count of 39 is too low.
  - 34 more markers sit in files that only the extended version builds.
  - The body font is Times: the PDF embeds NimbusRomNo9L, checked on 10 Oct.
- Server: 18 of the 20 GB quota is used (I5).

---

## 1. Goal and dates

**Goal.** Raise the mock scores from 2/3/2 to at least 3/4/3. Each reviewer named what would change their score.

- **A (2 → 3 or 4):**
  - a finished single-platform evaluation (F6);
  - the deployment comparison (EV), drafted before the gate;
  - the attack on the hardened protocol. G3.5 is done on OPT-1.3B. OPT-6.7B runs on a Xeon node before the freeze (G3.5-L).
- **B (3 → 4):**
  - the formal verifier model in the body (done: `def:spot`);
  - a non-naive setup proof, or a proof that its cost is forced (TH-c);
  - intermediate locality addressed (TH-a);
  - B-Q5, B-Q6 and the logUp positioning of B-W6 answered (TH-d/e);
  - precise positioning (done in the condensation).
- **C (2 → 3):**
  - the completed evaluation with real weights (F6, F3a, F2a, F2b-Q);
  - the overhead against unverified serving (A7);
  - quality at long context (F2a).
- **C (→ 4):** a verifier that clearly beats re-execution.
  - On the CPU this is measured under FS1 (V-LONG-a), against the strongest exact baseline on the same threads (RX).
  - On the GPU it is framed against the PCIe floor (EV).

| Date | Event |
|---|---|
| Mon 12 Oct | Plan 2 starts. FS1 part 1 lands (Kpre absorbs nothing). I5 runs. V-LONG-a and the V1 re-time are submitted. |
| Mon 19 Oct | The graph-changing decisions are fixed: `P_MAX` (F2a) and D6a's recipe. PS2's plan follows on Wed 21. |
| Fri 23 Oct | Partial rehearsal of the tiers merged so far. The booking of the friend's machine is confirmed in writing. |
| Fri 23 - Tue 27 Oct | The friend's smoke on the L40S/EPYC (at most 2 h), if granted |
| Mon 26 Oct | **Feature freeze.** Only merged and tested work enters the freeze. An unfinished should item stays behind a `PVI_*` switch that is off by default. |
| Wed 28 - Thu 29 Oct | **Rehearsal** of every tier on the 2080 Ti from the freeze candidate. Thu 29: EV draft and the mid mock review (H5-mid). |
| **Fri 30 Oct** | **Gate G1: go/no-go for S&P** (criteria in Sec. 5.1; Plan 1 called this gate G0). Author list and conflicts fixed. Team hand check of the proofs (TH-f). |
| **Sun 1 Nov, 23:59** | **Code freeze:** tag `sp2027-freeze` and hand it to the friend |
| Mon 2 - Sun 8 Nov | **L40S + EPYC run** (must tiers first). Re-runs are allowed until Tue 10 Nov at the latest. Mon 2 - Tue 3 is the hotfix window for platform-only failures (`sp2027-freeze.1`). |
| Mon 9 Nov | Abstract text frozen; all tables regenerated |
| **Tue 10 Nov** | **Abstract registration** and HotCRP fields [USER] |
| Thu 12 Nov | Second mock review of the final draft (H5b) and an external human read |
| Mon 16 Nov | **Submission** [USER]. One day of margin. |
| Tue 17 Nov | Deadline (buffer only) |
| Wed 2 Dec / Tue 26 Jan 2027 | EuroS&P / USENIX Security, if gate G1 says no-go |
| Mon 18 Jan 2027 | S&P early-reject notification. An early-rejected paper can still go to USENIX cycle 2 (26 Jan). |
| Tue 16 Feb 2027 | S&P rebuttal deadline. The CFP requires disclosure by this date. |

**Why the code freezes on Sun 1 Nov.**
1. **The run fits a week with slack.**
   - The must tiers are budgeted at 78-113 job-hours (run table, Sec. 4). They run one after another so that CPU and GPU timings do not disturb each other.
   - That is 3.3-4.7 days of a 7-day window, which leaves about 2 days for one round of re-runs.
   - The friend's machine is the scarce resource, so the window has slack and a hard stop.
2. **It gives the most engineering time that still leaves a week of writing.**
   - Freezing on 26-28 Oct, as two agents proposed, would cost 4-6 engineering days for little gain.
   - Bytes, quality and accuracy do not depend on the hardware, so they are measured before the freeze. The tables fill themselves by script (H3) as tiers finish.
   - Freezing later than 1 Nov would push the last re-runs past 10 Nov. It would also leave fewer than 6 days for the final text, H5b and the artifact.
3. **The abstract does not depend on the run.** The CFP fixes the title, abstract and author list at registration (10 Nov). The abstract therefore quotes only numbers measured before the freeze:
   - proof bytes;
   - quality on the 2080 Ti;
   - the CPU ratio, quoted only after RX and V-LONG-a have measured it against the strongest exact baseline, and phrased as "at least N times".

   N is a lower bound on both CPUs:
   - If the `cpuv` tier (which runs first) has finished by Sun 8 Nov, N is the smaller of the Xeon and EPYC ratios, rounded down.
   - Otherwise N is the smaller of the Xeon measurement and RX's EPYC model (about 2x with VNNI).

   Either way, the EPYC run cannot contradict the abstract.
4. **It matches the team's wish to run the L40S later.** It also leaves 2-3 days of slack if the friend's window slips, and the must tiers would still finish before 10 Nov.
5. **The platform risk is handled before the run.** The frozen code first meets sm_89, Zen 4 and the friend's environment on Mon 2 Nov. Two measures cover this (F6 part 14):
   - a smoke of at most 2 h on the friend's machine between 19 and 27 Oct;
   - if the friend declines, the hotfix rule on Mon 2 - Tue 3.

**Capacity.** In engineer-days, the must items alone over-commit the window before the freeze.

| Window | Engineer-days (6 per week) | Must load | Left for should items |
|---|---|---|---|
| 12 Oct - 1 Nov (before the freeze) | 18 (+3 seventh days) | code 19 (+3 if the contingencies of F2a, D6a and F2b-Q trigger) + writing 5.6 (TH 3.1, EV draft 1.5, H5a and H5-mid 1) | none; over by about 4-7 |
| 2 Nov - 16 Nov | 13 (+2 seventh days) | writing 7 (TH-a/c review 1.5, EV final 1, H4 3, H5b 1, literature refresh 0.5) + taking in run results 2 | about 4-6 |

- **The unit is pessimistic.**
  - Plan 1's log finished V1's 15-day milestone plan, B5, G3.5 and A1-A4 within 9-10 Oct.
  - Background workflows draft TH (a, c, e), H5a, H5-mid and EV's figures, so the engineer's share of that work is review.
- **The binding limits are elsewhere:**
  - 2080 Ti and Xeon queue hours (the budget below);
  - the quota (I5);
  - node RAM (`--mem`);
  - the human checks (TH-f, the disclosure notices, the external reader).
- **How the should items are funded:**
  - work running ahead of these estimates, as it did in Plan 1;
  - the seventh day of each week (5 days in all);
  - team help on writing;
  - for F2c, E2b and G3-min, the post-freeze slack.

**Rule for the should items.**
- They take slack in rank order (Sec. 3.0).
- An item whose last start date has passed is skipped, not started late.
- No should item blocks the freeze.
- There are two pools:
  - Items that need the freeze (V-V1N, V-LONG-b, F2b-L, F3b, A6, A7-vLLM) take pre-freeze slack, in rank order.
  - F2c, E2b and G3-min do not depend on the platform, so they take the post-freeze slack.
- Work that does not depend on the platform (accuracy, sampler TV, attack measurements, bytes) can continue on the 2080 Ti or the laptop after the code freeze, using the frozen builders.

**Pre-freeze job-hour budget.** These are estimates from the per-variant times quoted in the items. Timing jobs use `--exclusive` (I5).

| Work | Node | Job-hours | Done by |
|---|---|---|---|
| I5 checks: `/tmp` size and persistence, `--mem` | 2080 Ti, Xeon | under 1 | Mon 12 |
| V-LONG-a and RX: full Llama-2-7B, Qwen3-4B, OPT-1.3B at 64 and 2,048 tokens; verifier and every baseline; interactive, then Fiat-Shamir | Xeon (`--exclusive`) | 8-12 | Sun 18 |
| Proofs for those cells | 2080 Ti (CPU if a cell does not fit) | 2-3 | Sun 18 |
| V1 re-time (Llama-2-7B, 2-3 blocks, Kpre, B5 kernels) | Xeon | under 1 | Mon 12 |
| PS2: 2-block run with `setup_s` (R64c, cnn18c) | 2080 Ti | 2-4 | Wed 21 |
| F2a: OPT-125M and OPT-1.3B variants (3 graphs x 2 `P_MAX`, 3 calibration draws); OPT-6.7B (3 values of α) | 2080 Ti | 4-8 | Sun 25 |
| F2b ladder: Qwen3-0.6B/1.7B and TinyLlama (laptop); Qwen3-4B at 2,048 tokens | laptop; 2080 Ti | 4-8 (GPU) | Sun 25 |
| F3a: OPT-1.3B full, C/Kpre, 64/2,048 tokens, ≥ 3 repetitions, random-weight twin; OPT-6.7B Kpre | 2080 Ti | 6-10 | Sun 25 |
| F3a's OPT-6.7B real V1 bytes, and G3.5-L (4 variants of 1-2 h) | Xeon (`--mem` ≥ 64 GB) | 6-10 | Sun 1 |
| A7 pilot: full OPT-1.3B, Llama-2-7B 1-2 blocks | 2080 Ti | 2-3 | Sun 25 |
| Partial rehearsal (Fri 23) and full rehearsal (Wed-Thu 28-29) | 2080 Ti | 4-6 | Thu 29 |
| **Must total** | | **2080 Ti 24-42; Xeon 15-24** | |
| Should, if reached: V-LONG-b cells (Xeon 2-4), F2c (2080 Ti 3-5) | | 5-9 | |

Over three weeks, this is about 1-2 job-hours a day per node type.

---

## 2. What Plan 1 left open

Every Plan 1 item that is not tagged [DONE] or [DROP], plus the [DONE] items that carry an open note:

| Plan 1 item | Plan 1 status | Where it goes in Plan 2 |
|---|---|---|
| A5 overlap encoding with the forward pass | [TODO] | **A5**, could (Sec. 3.4). Latency gain only 1.3x; after 17 Nov unless there is slack. |
| A6 Kpre precompute on int8 GEMMs | [TODO] | **A6**, should, rank 8 among the should items |
| B4 fused element-wise derive on the GPU | [TODO] | Merged into **V-GPUFLOOR**, could |
| B5 note "EPYC full models in F6 `cpuv`" | [DONE] with a note | **V-LONG-a** and **RX** measure the full models and the strongest baselines on the Xeon now. **F6** `cpuv` repeats them on the EPYC. |
| B6 note "numbers final after F6" | [DONE] with a note | **EV** and **F6**: every verify/re-execution ratio is recomputed against RX's strongest exact baseline |
| B targets (G0): GPU verifier OPT-1.3B ≤ 0.8 s, Llama-2-7B ≤ 1.5 s | untagged | No longer a gate criterion. Measured in F6 `gpuv`; EV frames it against the PCIe floor. The CPU client is the headline. |
| C2 V1: L40S timings | [WIP] | **F6** `v1` tier |
| C2 V1: real-weight byte profile | [WIP] | **F3a**: OPT-1.3B in full; for OPT-6.7B, the codec profile plus real V1 bytes on a Xeon node. **PS1** is after 17 Nov. |
| C2 V1: streaming GPU verifier with the cut | [WIP] | The V1 part of **V-GPUFLOOR**, could |
| C2 V1: mode K with the cut | [WIP] | **[DROP]**: Kpre dominates K for a client that holds the weights. The `NotImplementedError` stays and is documented. |
| C2 V1: a few multipliers per op | [WIP] | **F2d**, after 17 Nov |
| C2 V1: `bench.py --proof v1` cells | [WIP] | **F6** |
| C3 V2/V3 design and cost model | [TODO] | **PS5**, could: one paragraph plus a section of the extended version; implementation after 17 Nov |
| C4 sessions: one opening for B queries | [TODO] | **C4a**: v0 batch cells (must, inside F6). **C4b**: V1 batches, after 17 Nov (the review marks chg 13 as optional). **C4c**: batches of different users (could, design paragraph). |
| C5 codec: real-weight entropy, per-op Rice, centring | [TODO] | **[DROP]** the per-op tweaks: PVC3 is measured within 0.08 bits per claim of the per-op order-0 entropy (7.36 vs 7.28 on OPT-1.3B). The context-modelled coder (**PS4**) moves to after 17 Nov. |
| D6 open-weight registry, C_M recomputable | [TODO] | **D6a**, must: a cross-machine digest test, then a deterministic recipe if needed, fixed by Mon 19 Oct. The registry text is in **H4**. |
| D7 full proof covering every optimisation | [WIP] | **TH-e** (extended version, drafted in weeks 1-2) and **TH-f** (a team member checks it by hand, Fri 30 Oct) |
| E3 verifier KV cache for multi-turn chat | [TODO] | After 17 Nov; one sentence in the discussion. No reviewer asks for it. |
| F1 re-execution baseline on the L40S/EPYC | [WIP] | **F6** `cpuv`/`gpuv`, plus the same-GPU baselines of **A7** and the strongest exact CPU baselines of **RX** (Xeon now, EPYC in `cpuv`) |
| F2 quality: OPT-6.7B, Qwen3-4B, real-weight V1 bytes | [WIP] | **F2a** (OPT at 2,048 tokens, 8-bit probabilities), **F2b-Q** (Qwen3, with the per-token exponent), **F2b-L** (Llama), **F2c** (downstream tasks) |
| F3 real-weight end-to-end run | [TODO] | **F3a** (OPT-1.3B, OPT-6.7B; must), **F3b** (Qwen3-4B, real generation; should) |
| F4 memory accounting | [TODO] | **F6**: every cell reports peak GPU and host memory; one generated table |
| F5 zkLLM on the same L40S | [TODO][USER] | **F5**, could [USER] |
| F6 final run package | [WIP] | **F6** v2: new tiers built with their items, Fiat-Shamir everywhere, ≥ 5 timed queries, partial rehearsal Fri 23, the friend's smoke, rehearsal, freeze Sun 1 Nov |
| F7 tables (verify/re-exec, transfer, per-token, comparisons) | [TODO] | **EV** (drafted by Thu 29 Oct, with the numbers pipeline of F6/H3) |
| F8 modern shapes (Llama-3-8B, Mistral-7B) | [TODO] | **F8**, could |
| G3 adaptive game with range-respecting edits | [TODO] | **G3-min**, should. If it is not done by Mon 9 Nov, the three attack `\todo`s (G1e, G3 x2) are deleted. |
| G3.5 note "OPT-6.7B in F6" | [DONE] with a note | **G3.5-L**, must: on a Xeon node before the freeze. The L40S `attack` tier is removed. |
| G4 aligned-model threat characterisation | [TODO] | **[DROP]** under scope rule K9 (the paper's offensive content must not grow; `attack_theory/experiments.md` §2). Only the acceptance-bound table (1 - k/d_ff)^s goes into the extended version (0.25 d, inside TH-e). |
| G5 responsible disclosure | [USER] | Sec. 6: notices on 15 and 19 Oct, reminders on 26 Oct, reply cut-off on 2 Nov. **H4-c** turns the conditional sentences into facts. Not a gate criterion: the CFP requires disclosure by the rebuttal deadline (16 Feb 2027). |
| Team decision: who sends the notices; SPEX; Ambient | [USER] | Sec. 6, Mon 12 Oct |
| Team decision: Qwen3-4B attack repeat; GPU for OPT-6.7B | [USER] | Sec. 6, Wed 14 Oct. OPT-6.7B quality runs on the 2080 Ti via F2a's memory-lean path. The OPT-6.7B attack runs in fp32 on a Xeon node (G3.5-L). |
| Team decision: anonymisation | [USER] | Sec. 6, Fri 23 Oct; **H4** |
| Team decision: gated models | [USER] | Sec. 6: the team accepts the Llama-2 licence on its own Hugging Face account (Wed 14 Oct), never using someone else's token. The friend accepts it on their own account for the L40S Llama-2 cells (Fri 23 Oct). |
| H1 S&P draft | [WIP] | Writing slots in Sec. 4; page ledger and cut list in **H5** |
| H3 number pipeline | [TODO] | Inside **F6** (must) |
| H4 artifact appendix and README | [TODO] | **H4**, must |
| H5 internal mock review | [WIP] | **H5**: H5a on Wed 14 Oct (with the review checklist and the page ledger), H5-mid on Thu 29 Oct, H5b on Thu 12 Nov |
| H6 register and submit | [USER] | Sec. 6 (author list Fri 30 Oct; 10 and 16 Nov) |
| D4 remainder: setup-proof timing | (D4 [DONE]) | **F6**: one cell (Llama-2-7B, a = 100, setup prover and verifier time). **TH-c** explains why its size is forced. |

---

## 3. The items

### 3.0 Ranked list (expected impact on acceptance per day of effort)

| # | ID | Axis | Priority | Effort (d) | Main review items | Gain in one line | Start by / kill date |
|---|---|---|---|---|---|---|---|
| 1 | FS1 | cross | must | 2.5 | B-W7, chg 9, C-Q1 | Fiat-Shamir hashing at Llama-2-7B, 2,048 tokens: 21 s → 0 s per party (Kpre), → 3-5 s (mode C) | part 1 Mon 12; hash chosen and kill Thu 15 |
| 2 | V-LONG-a | verifier | must | 0.5 | A-W6, C-W1, chg 1, chg 11 | The full-model CPU ratio measured, not extrapolated; V1 re-timed with the B5 kernels | Mon 12 |
| 3 | RX | verifier | must | 1 | C (→ 4), chg 4, A-W6 | The CPU ratio against the strongest exact baseline: gate criterion 2 and the abstract's "at least N times" | Thu-Fri 15-16 Oct; exact by Sun 18 |
| 4 | I5 | infra | must | 0.5 | (enables F2a, F2b, F2c, G3.5-L) | Room for about 40 GB of checkpoints and an lm-eval environment; timings on exclusive nodes | Mon 12 |
| 5 | PS2 | proof | must | 1 | A-W1 | -29% proof bytes at 64 tokens under FS (383 → 273 MB), no protocol code | decide Wed 21 |
| 6 | F2a | quality | must | 2.5 (+1) | C-W5, C-Q2/Q3, chg 7 | OPT quality at 2,048 tokens on the full test set; the L40S quality tier no longer runs out of memory | Wed 14; `P_MAX` decided Mon 19 |
| 7 | D6a | trust | must | 0.5 (+1) | A-W4, A-Q5, chg 10 | "C_M is reproducible bit for bit from the fp16 checkpoint" | test Thu 15; recipe fixed Mon 19 |
| 8 | A7 | prover | must | 2 | C-W2, C-Q1, A-Q2, chg 3 | Prover versus unverified serving on the same GPU, with batching | Thu 22 |
| 9 | F3a | quality | must | 1.5 | C-W1, chg 1 | Two real-weight full models end to end, OPT-6.7B among them (the 7B-class bar) | Thu 22 |
| 10 | G3.5-L | attack | must | 0.5 | A-Q3, chg 5 | The in-range attack on OPT-6.7B (A's condition 3), on a Xeon node instead of the L40S | week 2; by Sun 1 Nov |
| 11 | F2b-Q | quality | must | 4 (+1) | C-W5, C-Q2 | A non-OPT real-weight graph (Qwen3) and its quality at 2,048 tokens; the Qwen/Llama-family signal by Sun 18 | laptop ladder by Sun 18; kill Fri 23 |
| 12 | F6 | cross | must | 2.5 | chg 1, 12, 13, C-W8, C-Q4, A-W6 | One platform, frozen code, ≥ 5 timed queries, memory reported; fills about 45 `\todo`s | tiers land with their items; partial rehearsal Fri 23; Mon 26 |
| 13 | TH | paper | must | 4.6 | B-W3/W5/W6/W7, B-Q5/Q6, A-Q7, chg 9 | B 3 → 4; clears 11 theory `\todo`s; full proofs in the artifact | (b, d) Sun 18; (e) weeks 1-2, TH-f Fri 30; (a, c) from Mon 19, review by Tue 3 Nov |
| 14 | EV | paper | must | 2.5 | A-W1/W6, B-W7, C-W8, chg 3, 11 | The deployment case and break-even points; one of A's three conditions | draft Thu 29 Oct (gate input); final 5-6 Nov |
| 15 | H4 | paper | must | 3 | CFP, A-W4/W7, chg 8 | No desk-reject risks; artifact and extended version valid | 7-8 Nov, 15 Nov |
| 16 | H5 | paper | must | 2.5 | page limit, CFP, Smaller issues, chg 6, 11 | Protects the gains: review checklist, page ledger, catches regressions from the cut | Wed 14 Oct, Thu 29 Oct, Thu 12 Nov |
| 17 | V-V1N (CPU parts 1-4) | verifier | should | 2.5 | chg 11 | V1 CPU verifier about 60 s → about 17-21 s (model); otherwise V1 becomes a bytes-only option | Mon 19 if the re-time confirms the gap; exact by Wed 28 |
| 18 | F2c | quality | should | 2 | C-W5, C-Q2 | Downstream zero-shot accuracy, paired confidence intervals | start by Mon 2 Nov (may run after the freeze) |
| 19 | E2b | quality | should | 1.5 | C-W6, chg 7 | Sampler total-variation distance measured (laptop) | start by Mon 2 Nov |
| 20 | G3-min | attack | should | 2 | A-W2, A-Q3, chg 5 | Best value-aware sampler against in-range edits; 3 attack `\todo`s | start by Mon 2 Nov |
| 21 | V-LONG-b | verifier | should | 1 | C-W7, C-Q7 | Verifier and bytes measured from 2k to 16k tokens with today's kernels; 32k modelled | Thu 22 |
| 22 | F2b-L | quality | should | 1 | C-W5 | Integer quality of Llama-2-7B, the model the cost tables lead with | TinyLlama result Sun 18; kill Sun 25 |
| 23 | F3b | quality | should | 1 | C-W1, chg 12 | Qwen3-4B real weights end to end, plus real generation | Wed 28 |
| 24 | A6 | prover | should | 1 | A-W1, chg 3 | Kpre one-time setup 96.5 s → about 6 s (Llama-2-13B); already small next to a 6.7 GB weight download | Sat 24 Oct |
| 25 | A7-vLLM | prover | should | 1 | C-Q1 | Ratio against a serving engine [USER] | venv by Fri 30 |
| 26-38 | V-GPUFLOOR, V-V1N part 5, A5, V-SPARSE, B5b, A10, A11, PS5, F8, F5, C4c, PS2-PoW, E2b-3 | | could | 0.5-6 each | | Sec. 3.3-3.8 | only if every should item above is done |
| -- | PS1, C4b | proof | after 17 Nov | 3.5, 2.5 | A-W1, chg 6(iv), chg 13 | Sec. 3.4-3.5 | not before the deadline |

### 3.1 Cross-cutting: Fiat-Shamir, infrastructure and the final-run package

- **FS1 [DONE] must: Fiat-Shamir hashing at memory speed, counted in both parties' time.** This merges the verifier agent's V-FS and the prover agent's A8.
  - **Status (11 Oct).** Kpre absorbs nothing; tree absorption of large messages (1.37 GB/s on 8 laptop threads, 6x serial); `fs_hash` in both parties' time; `perf.py --fs`; tests. Commits 12a188f, b244e8b; write-up `docs/improvements/FS1_fiat_shamir_speed.md`. The Xeon end-to-end cells are V-LONG-a's.
  - **What.**
    1. **Kpre absorbs nothing (Mon 12 Oct, 0.25 d).**
       - Kpre v0 is already one message with a designated verifier. chi and u come from the verifier's precomputed key (`verifier._pre`), so no challenge reads the transcript.
       - The prover's messages are therefore no longer absorbed under Fiat-Shamir. The protocol is the same one-message protocol.
       - This secures the headline CPU ratio on day one.
    2. **Mode C v0, and V1 in every mode: each large message is absorbed as a digest of a parallel hash.** Two candidates are evaluated side by side, and FS1 takes whichever passes the transcript tests first:
       - **An off-the-shelf parallel hash, absorbed as a digest.** The options are BLAKE3 with threads (the `blake3` package, pinned in the freeze) and KangarooTwelve. The soundness argument is the standard one for absorbing a collision-resistant digest. This option needs less new code and carries less R6 risk; its cost is one pinned dependency.
       - **The custom tree mode.** Each large message is split into 4-8 MiB chunks. Each leaf is SHAKE-256('pvi/leaf' ‖ label ‖ j ‖ chunk_j), 64 bytes, computed on a thread pool (hashlib releases the GIL). The transcript absorbs (label, number of chunks, chunk size, total length, leaf digests).

       Either way:
       - The transcript itself stays SHAKE-256 (Plan 1 D2).
       - The call sites are `_claims_message`, `_fold_message`, `_open_message` and `cut_protocol` M1/M3.
       - The prover hashes inside A3's threaded assembly. The verifier hashes while it decodes, since chi is needed only at the products step.
    3. **Accounting.** Each party's own hashing goes inside `prove_s`/`verify_s` (bench, v1_bytes, perf). `perf.py` gets `--fs`, and every `cpuv`/`gpuv` cell runs under Fiat-Shamir.
    4. **One paragraph in app:params.** It gives the digest-absorption argument, or, for the tree mode, a sound tree-hashing mode (Bertoni et al.; KangarooTwelve is a deployed instance). The collision or birthday term is added to the Fiat-Shamir bound.
  - **Why.**
    - B-W7 and chg 9 ask for non-interactive proofs by default, and today the cost of that is hidden.
    - One serial SHAKE-256 runs at 303-318 MB/s (Xeon, job 1005575). That takes 1.2-2.5x as long as the whole v0 prover.
    - For Llama-2-7B at 2,048 tokens (6.6 GB), this is about 21 s per party, more than the 17.2 s CPU verifier. Under Fiat-Shamir with today's code, the abstract's 4.6x would fall to about 2.1x.
    - A7's serving overhead cannot be stated honestly without this fix.
  - **Gain.**
    - Kpre v0 under Fiat-Shamir: 20.8 s → 0 s per party, so the CPU ratio holds non-interactively.
    - Mode C v0 at 2,048 tokens: about 21 s → about 3-5 s. On the laptop, SHAKE-256 runs at 468 MB/s on one thread and 1,887 MB/s on 8 threads (measured).
    - V1: about 9.1 s → about 2.3 s.
    - Llama-2-7B, 2 blocks, Fiat-Shamir prover: 1.12 s → about 0.42 s.
  - **Effort / hardware.** 2.5 d, of which part 1 takes 0.25 d. The laptop is used for the implementation and tests. The Xeon and 2080 Ti nodes measure throughput at 1, 4 and 8 threads.
  - **Validation.**
    - Prover and verifier transcripts are equal on every Fiat-Shamir suite: test_protocol, test_wire, test_cut_protocol T4/T5, generation, lookups, plans.
    - New tests, each of which must change every later challenge:
      - flipping 1 bit in the first, middle or last chunk (or byte range);
      - shifting a chunk boundary;
      - changing the chunk size;
      - empty and sub-chunk messages.
    - The threaded hash equals a serial reference: the same tree, or single-threaded BLAKE3.
    - Kpre v0 under Fiat-Shamir absorbs no prover message and gives the same verdicts on every tamper test.
    - On the Xeon, the reported `fs_hash` falls at least 4x with 8 threads and is now inside `prove_s`/`verify_s`.
  - **Kill.** If the chosen hash is less than 3x faster than serial SHAKE-256 on the Xeon by Thu 15 Oct, FS1 keeps parts 1 and 3 with the serial hash and reports its cost. This is FS1's own rule, not a gate criterion.
  - **Note.** Each part changes Fiat-Shamir transcripts and invalidates the stored Fiat-Shamir test vectors and cells of the modes it touches.
    - Part 1 changes the Kpre transcripts and lands Mon 12 Oct.
    - Part 2 changes the mode-C and V1 transcripts and lands by Thu 15 Oct.
    - No Fiat-Shamir number for the paper is produced in a mode before its change lands.

- **I5 [WIP] must: storage, nodes and clean timings (Mon 12 Oct).**
  - **Status (11 Oct).** Quota now 16 TB (17.8 GB used, 10 Oct); checkpoints go to node-local `/tmp` (s-004: 264 GB, s-005: 335 GB free); pyarrow in `logs/validation/sp2027/pylib`. Open: the lm-eval environment.
  - **What.**
    1. Archive the engineer's own old results to the laptop, as in I3, and verify the listing before deleting anything. Only the engineer's own folders are touched.
    2. Measure the size and persistence of node-local `/tmp` and the `--mem` limit on the 2080 Ti and Xeon nodes.
    3. Pin model-heavy jobs to one node, so that the `/tmp` caches (HF_HOME, the lm-eval datasets) are reused between jobs.
    4. Run timing jobs with `--exclusive`.
  - **Why.**
    - The quota has about 2 GB free (18 of 20 GB used). F2a, F2b, F2c and the attack need about 40 GB of checkpoints plus an lm-eval environment.
    - The B5 doc reports up to 1.7x run-to-run variation on shared nodes.
  - **Effort / hardware.** 0.5 d. Laptop, 2080 Ti and Xeon nodes.
  - **Validation.**
    - Free quota and `/tmp` size are recorded in `docs/improvements/I_server_setup.md`.
    - A second job on the pinned node finds the cache.
    - The archive listing matches what was deleted.
  - **Kill.** None. If `/tmp` does not persist between jobs, each model-heavy job downloads its checkpoint into `/tmp` and deletes it at the end (R4).

- **F6 [WIP] must: final-run package v2, rehearsal, number pipeline (absorbs F4 and H3).**
  - **Status (11 Oct).** Package v2 merged (914bdad, a1f7d2d): tiers smoke/cpuv/v1/serve/bench/gpuv/gen/batch/real/setup (+ quality, long), Fiat-Shamir by default, >= 5 timed queries for the headline cells, memory in every record, chained jobs, PVI_DRYRUN/PVI_REHEARSAL/PVI_C_POLICY/PVI_LOCAL; must total 114 jobs, estimate 40-60 h; H3 pipeline `sp2027_numbers.py` regenerates the existing tables from the stored records (63/63 and 30/30 cells identical). First partial rehearsal on the 2080 Ti passed on 11 Oct (22 jobs, `sp2027_numbers.py --rehearsal --check` exit 0, 0 jobs without results) after two package fixes: jobs submitted as bash scripts (dash broke `--wrap`), and a preflight for the trained image models and their datasets, which a clone does not carry (27f07e4; `L40S_RUN_SP2027.md` Sec. 2 now lists them for the friend). Open: F3b/real Llama cells (the real flags read OPT only), PS2's policy, the friend's smoke.
  - **What.**
    1. **Fiat-Shamir by default.** `perf.py --fs`. The headline v0 cells in `strong_gpu.sh` switch from `C:int` to Fiat-Shamir; Kpre is unchanged after FS1. Every interactive cell that remains is labelled.
    2. **≥ 5 timed queries** after a warm-up for the headline cells (the v1 Llama-2-7B and OPT-1.3B cells, all of `cpuv`, `serve`), reported as median [min-max]. Other cells use 3.
    3. **F4:** peak GPU and host memory in every record (`gpu_peak_memory` and `host_peak_rss` exist already), plus a generated memory table for the 7B models at 2,048 tokens (C-Q4).
    4. **New tiers.** Each one lands with its item (part 13).
       - `serve` (A7);
       - `gen`: Llama-2-7B and Qwen3-4B, P = 512, G ∈ {64, 256}, greedy and sampled, plus the two tamper cells (chg 12);
       - `batch` (C4a): Llama-2-7B at 64 tokens with B ∈ {1, 4, 16, 64}; OPT-1.3B at 512 with B ∈ {1, 4, 16}; Llama-2-7B at 2,048 with B ∈ {1, 2, 4}. It uses the existing `bench.py --batches` code, moved from `strong_gpu.sh nice`.
       - `real`: the F3a/F3b cells that do not fit the 2080 Ti (OPT-6.7B timing if needed). Llama-2-7B in mode C runs only if F2b-L succeeds and the friend holds the Llama-2 licence on their own account; otherwise that cell is dropped or uses random weights.
       - one setup-proof timing cell (Llama-2-7B, a = 100);
       - `long`: V-LONG-b's cells, only if V-LONG-b is done.
    5. **`cpuv`:**
       - add Qwen3-4B; Fiat-Shamir on;
       - every RX baseline on the same threads: exact integer re-execution with int8 GEMMs (VNNI), bf16 (AVX512_BF16), fp32, and the int8 engine labelled not exact;
       - 8,192 tokens for Llama-2-7B only if the EPYC node has ≥ 256 GB of RAM [USER].
    6. **`gpuv`:**
       - fp16 and int8 (`torch._int_mm`) re-execution of the full models on the same GPU;
       - measured host-to-device bandwidth;
       - peak memory of the verifier against re-execution.
    7. **`quality` shrinks.**
       - OPT, Qwen3 and Llama-2-7B quality move to the 2080 Ti (F2a/F2b).
       - The L40S keeps only Llama-2-13B quality, and only if F2b-L succeeds and the friend holds the licence.
       - OPT-6.7B's real V1 bytes move to a Xeon node (F3a). The OPT-6.7B in-range attack moves to G3.5-L. The `attack` tier leaves the package.
    8. Mode-C cells use the plan policy that PS2 chooses on 21 Oct.
    9. **Audit `strong_gpu.sh must` against the S&P tables.** The course-report controls (`nofix-*`, `ab-*`) are removed unless a table cites them. Target: ≤ 30 h.
    10. **Bytes come from the 2080 Ti.** Proof bytes do not depend on the hardware, so every byte cell that fits is measured on the 2080 Ti before the freeze. The L40S gives the timings.
    11. **H3:** extend `paper_assets`/`text_numbers` to every new table and raw root, so that the paper's tables regenerate from `logs/l40s_sp2027/`.
    12. Update `L40S_RUN_SP2027.md` with:
        - the run order and the hours per tier;
        - `HF_HOME` (40 GB);
        - exclusive use and 24/7 operation;
        - the optional vLLM venv;
        - the smoke and hotfix procedure;
        - what to send back.
    13. **Built incrementally.** Each tier lands in the same commit as the item it measures: `serve` with A7, `batch` with C4a, `real` with F3a, the new `cpuv` baselines with RX, and `gen` in its own commit in week 2 with its rehearsal cell. A partial rehearsal on the 2080 Ti on Fri 23 Oct covers every tier merged by then. Mon-Tue 26-27 Oct complete the package and H3, and Wed-Thu 28-29 Oct run the full rehearsal.
    14. **The friend's smoke and the hotfix rule.**
        - Ask the friend for a smoke of at most 2 h (`smoke` plus `PVI_REHEARSAL=1` per tier) on the L40S/EPYC between 19 and 27 Oct. This is not "running the L40S".
        - The smoke catches what the 2080 Ti rehearsal cannot:
          - Triton kernels and `torch._int_mm` on sm_89;
          - the `-march=native` native kernels on Zen 4;
          - CXX, HF_HOME and PVI_PY_INCLUDE in the friend's environment.
        - If the friend refuses, or the smoke on Mon 2 Nov fails for a platform reason, the hotfix rule applies. On Mon 2 - Tue 3 Nov, only platform-only fixes that leave the byte and verdict tests unchanged are allowed. The fixed code is re-tagged `sp2027-freeze.1` and followed by a new smoke.
  - **Why.**
    - chg 1 (one platform, frozen code, no extrapolation, spread), C-W1, C-W8, C-Q4, chg 12, chg 13.
    - A-W6: the tables disagree today (698 ms vs 1.5 s, and 58.2 s vs 85 s).
    - About 45 of the 71 `\todo`s wait on this run.
  - **Gain.** Every L40S/EPYC `\todo` is filled from one frozen run, and every table is regenerated by script.
  - **Effort / hardware.** 2.5 d (package 1, H3 1, rehearsal 0.5), spread over weeks 1-3 as the tiers land. Laptop; the 2080 Ti for the rehearsals.
  - **Validation.**
    - **Rehearsals (partial on Fri 23 Oct, full on Wed 28 - Thu 29 Oct):** `PVI_REHEARSAL=1 sp2027.sh <tier>` runs the smallest cell of every tier (GPT-2/OPT-125M, 2 blocks) on the 2080 Ti from the freeze candidate.
    - Every JSON line parses through H3 into every table.
    - `PVI_DRYRUN=1` prints every sbatch line.
    - The budget per tier is listed, and the must total is at most 113 h.
  - **Kill.** A tier that fails the rehearsal and is not fixed by Sat 31 Oct leaves the package. Its cells are then dropped or labelled with the 2080 Ti platform.

### 3.2 Quality and real weights

- **F2a [DONE] must: memory-lean quality at 2,048 tokens, full WikiText-2 test, 8-bit versus 11-bit attention probabilities.**
  - **Decision (11 Oct, round 2, job 1011066).** Certified recipe = smooth-max alpha 0.6 (`real_weights.CERTIFIED_RECIPE`, train calibration `common.certified_calib`): x1.076 / x1.026 / x1.008 for OPT-125M / 1.3B / 6.7B at 2,048 tokens (full test set, 11-bit probabilities), draw-stable; the paper builder gives the same graph (`tests/test_recipe_equivalence.py`). Applied in 1801809 (digest_check, perf/v1_bytes `--recipe certified`, the F6 real tier, zeroshot). The paper's quality section is rewritten for 2,048 tokens. Follow-ups: D6a's re-digest (jobs 1012438-9) and F2c with the certified recipe (job 1012440).
  - **D6a re-digest done (12 Oct, jobs 1012438-9, 1012744-5).** The certified OPT-125M / OPT-1.3B digests (`16de86ea...` / `d9daa017...`), weight hashes and roots agree on four nodes (s-002, s-004, s-005, s-006), threads 1 / 4 / 8, and with OPT-1.3B's commitment encoded on the GPU or the CPU (the same-model cross-device check the first run lacked). Artifact README filled; the paper's sentence again says "CPU or GPU encoding". D6a doc Sec. 3a.
  - **Status (11 Oct).** Cluster results in (jobs 1007844-1007846): fp32 references match the published values; 8-bit probabilities cost 2-23% at 2,048 tokens, 11 bits reach the ideal softmax -> **P_MAX = 2047 adopted** for the certified models (e5d377c; kernels validated, job 1007805). OPT-6.7B smoothed 11-bit x1.017. OPT-1.3B: the smoothed recipe is worse than a scalar gain at 2,048 (x1.14-1.25 against x1.096) and draw-dependent: recipe search running (job 1008668). Write-up F2a_quality_2048.md.
  - **What.**
    1. **A lean evaluation path.** `real_weights.real_logits` calls `forward(free=True)`, and the LM head runs over chunks of positions, returning only the per-token NLL and the argmax. `_RealBuilder._run` drops each calibration tensor after its last use, or calibrates on the CPU.
    2. **Calibration on WikiText-2 train windows** (4-8 x 2,048 tokens). Today's calibration on the last test window would overlap the evaluated text.
    3. **All of the roughly 140 non-overlapping 2,048-token test windows,** with a bootstrap confidence interval over windows and 3 calibration draws.
    4. **A knob `P_MAX ∈ {255, 2047}`,** plus:
       - float probabilities (port `hybrid.py`'s softmax switch from branch `f2-quality`; its scripts are marked unverified, so re-test them);
       - the mass each layer loses at 2,048 tokens.
    5. **Runs on the 2080 Ti:**
       - OPT-125M and OPT-1.3B, each with three graphs: scalar gains; smoothed (α = 0.55); and smoothed plus 4-bucket row scales (the best v0-only graph). Each graph runs at both values of `P_MAX`.
       - OPT-6.7B smoothed (α ∈ {0.5, 0.55, 0.65}), with `HF_HOME` on node-local `/tmp` and the fp16 reference streamed layer by layer.

       These replace the L40S cells `q-opt67` and `q-opt67-2k`.
  - **Why.** C-W5, C-Q2, C-Q3, chg 7, and the 3 `\todo`s in quality.tex.
    - `q-opt67-2k` would need about 50 GB in the builder and again in `real_logits`. That is an almost certain out-of-memory failure on the 48 GB card.
    - 128-token windows cannot show the 8-bit effect. At most 510 keys per row get p > 0, so diffuse rows lose mass only at long context.
  - **Gain.**
    - OPT rows at the length where the costs are reported. Expected: OPT-1.3B x1.03-1.05, and OPT-6.7B ≤ x1.05 (from x1.33 with a scalar gain). Gate G1 criterion 7 allows up to x1.10.
    - 8-bit probabilities are expected to cost ≤ 1%. If they cost more, 11 bits remove the loss at no cost in proof bytes or verifier time:
      - p ≤ 2047 is exact in fp16;
      - (2047 + T/2)·127 < 2^24 for T < 2^15;
      - tot + 4094e < 2^31.
    - 2-5 min per variant on the 2080 Ti, and 10-15 h less on the L40S.
  - **Effort / hardware.** 2.5 d, plus 1 d if 11 bits are adopted. The extra day covers the constants in `transformer.py` (2 paths), `reference.py`, `attention_kernels.py` and `native_kernels.py`, and their exactness tests. Development is on the laptop (OPT-125M at 2,048 tokens); the runs are on the 2080 Ti/Xeon.
  - **Validation.**
    - The lean logits equal `real_logits` exactly on OPT-125M at T = 256.
    - The `P_MAX = 255` graph has today's `IntGraph.digest()`.
    - The new path reproduces OPT-1.3B's 33.70 at 16 x 128.
    - The fp references match the published 2,048-token values (OPT-1.3B 14.62, OPT-6.7B 10.86).
    - Peak memory is recorded for every cell.
  - **Kill.**
    - `P_MAX` is decided on **Mon 19 Oct**, before any real-weight or byte cell is produced. The ablation takes minutes on OPT-1.3B at 2,048 tokens.
    - 11 bits are adopted only if the 8-bit cost is above 1%. Their exactness tests must then pass by Wed 21 Oct, before F3a's cells (from Thu 22) and the reported Qwen3-4B runs at 2,048 tokens.
    - Otherwise 8 bits stay, and the paper reports their cost.

- **D6a [DONE] must: a quantisation recipe that reproduces C_M bit for bit** (Plan 1 D6 and part (b) of the paper agent's compliance item).
  - **Re-digest (11 Oct).** The certified recipe changed (F2a decision); jobs 1012438 (s-004) and 1012439 (s-005) recompute the digests, weights and roots of OPT-125M and OPT-1.3B with it (1 and 8 threads).
  - **Status (11 Oct).** Identical graph digest, weight hash and all 6 Merkle roots for OPT-125M and OPT-1.3B on s-004 and s-005 at 1 and 8 threads (jobs 1008669/1008670; the OPT-1.3B commitment encoded on the GPU), and the same OPT-125M digest and weights on the laptop (Windows, torch 2.2.2). No recipe change needed. Write-up `docs/improvements/D6a_reproducible_commitment.md`.
  - **What.**
    1. **Test first.** Build the smoothed OPT-125M and OPT-1.3B on the laptop and on a Xeon node, with 1 and 8 threads, and compare the digests and every Merkle root.
    2. **If they differ:**
       - compute `res_max` and W·β in float64 on the CPU with compensated sums;
       - snap scales derived from floats to a grid of 2^-10 relative;
       - use a CPU-only builder with a fixed thread count;
       - record the recipe in `graph.meta` (checkpoint revision, token hashes of the calibration windows, knobs).
    3. F2b inherits the recipe.
    4. The threat model (Sec. 3.1) makes "recompute C_M from the fp16 checkpoint with the published recipe" the main path (chg 10). The artifact publishes the 32-byte roots.
  - **Why.** A-W4, A-Q5, chg 10, and the `\todo` in defence.tex. A 1e-6 relative change in s_res moves every residual multiplier m = round(2^30 s_z / s_res) by about 10^3.
  - **Gain.** A verifiable root-of-trust claim at no runtime cost. Snapping moves a scale by at most 0.1%, so it should cost under 0.1% perplexity.
  - **Effort / hardware.** 0.5 d for the test, plus 1 d for the fix if needed. Laptop + Xeon node.
  - **Validation.** Identical digests and roots on 2 machines x 2 thread counts for OPT-125M, OPT-1.3B and one Qwen3 model; perplexity unchanged within 0.2%.
  - **Kill.** This is the root of trust, and it changes the real-weight graphs. The recipe must therefore be fixed by **Mon 19 Oct**, before any real-weight or byte cell is produced (F3a starts on Thu 22). If exact reproduction cannot be reached by then, the published int8 checkpoint and its hash become the root of trust, and the paper says so.

- **F3a [DONE] must: real-weight runs end to end, now, on the 2080 Ti** (Plan 1 F3, OPT part).
  - **Status (11 Oct).** Part 1 merged (2245f22): `prune_last` in `build_opt_from_hf` (exact; unpruned graph unchanged), `v1_bytes.py --real --prune-last` with a non-vacuous accounting check; cluster cells in `docs/improvements/F3a_real_weight_runs.md`. Part 2 ran (jobs 1008673, 1009431): OPT-1.3B real/random in C and Kpre at 64 and 2,048 tokens, prover within 3-9% and CPU verifier within 6% of random; v0 bytes 9-17% smaller on real weights; V1 1.28x at 64 tokens; all tampers rejected. OPT-6.7B real, full model, Kpre at 2,048 tokens: accepted 3/3 with the CPU prover (270 s; verify 20.2 s), tampers 9/9, V1 bytes 1.73x. OPT-6.7B random twins (same job): prove 288 s / verify 21.3 s / v0 5.41 GB / V1 2.23x, so real weights are within 6% in time, 15% smaller in v0 bytes and 10% larger in V1 bytes. Open: V1 with the GKR at 2,048 tokens (11 GB OOM) and the GPU prover of OPT-6.7B move to the L40S `real` tier.
  - **What.**
    1. Add `prune_last` to the real builders. Today `v1_bytes.py --real` refuses `--prune-last`, so real cells cannot be compared with the pruned headline cells.
    2. Real smoothed OPT-1.3B, all 24 blocks, modes C and Kpre, at 64 and 2,048 tokens, with WikiText-2 windows as queries.
       - Report v0 and V1 bytes per component, prover and verifier time (native CPU and streaming GPU), peak memory and the tamper cells.
       - Run the same cells on the random-weight graph of the same shape, for real/random ratios.
    3. Real OPT-6.7B, full model, in Kpre.
       - It runs on the 2080 Ti if it fits after F2a. Otherwise the prover runs on a Xeon node's CPU (as in V-LONG-a), so the correctness result exists before the gate.
       - The final-platform timing goes in the L40S `real` tier.
    4. For OPT-6.7B at 2,048 tokens:
       - a byte profile with the codec only (forward and encoder per block, no proof);
       - the real V1 bytes from `cut_proof_bytes`, which needs no GKR. They run on a Xeon node with `--mem` ≥ 64 GB and were moved there from the L40S `quality` tier.
  - **Why.**
    - C-W1 says that "a real-weight end-to-end run of at least one 7B-class model is the bar set by zkLLM and zkAgent". It also answers chg 1 and the F3 `\todo`s (evaluation:48, discussion, ext_discussion:51).
    - It checks the methodology claim that costs depend only on the shapes.
    - It needs only the existing OPT builder, so it does not wait on F2b.
  - **Gain.** Removes the "random weights" caveat for two models, one of them 7B-class. Expected:
    - times within 5% of random weights;
    - v0 bytes within about 10%;
    - V1 1.7-1.9x on real weights.
  - **Effort / hardware.** 1.5 d of active work; the rest is job time. 2080 Ti/Xeon.
  - **Validation.**
    - Every honest query is accepted and every tamper is rejected.
    - Bytes equal the byte accounting (T7).
    - ≥ 3 repetitions after a warm-up.
  - **Kill.** If OPT-6.7B does not fit the 2080 Ti, its correctness run uses the CPU prover on a Xeon node before the gate, and its timing moves to the L40S `real` tier, which is already in the package.

- **F2b-Q [WIP] must: real-weight builder for RMSNorm/RoPE/SwiGLU/GQA models, Qwen3 first.**
  - **Status (11 Oct).** Builder merged (89ddb1e); on the real checkpoints every exactness and protocol check passes (job 1008672: structure diff [], GPU = CPU, PROTOCOL PASSED), but quality collapses: Qwen3-0.6B-Base at 16 x 128 x28 (scalar), x72-77 (smoothed), x6-9 (per-token exponent). Diagnosis (job 1009531, `code/artifacts/results/9_quality/llama_diag_1009531.txt`): TinyLlama's BOS embedding row quantises to zero, so position 0 stays zero and the attention sink never forms; Qwen3 has a massive activation (6,800 at position 0) and loses accuracy from block 1, the known static per-tensor W8A8 failure. The fixed public prefix (PrefixQuant-style) is merged; its ladder (jobs 1009796-7): TinyLlama x1.15 at 16 x 128 (the TinyLlama target), Qwen3-0.6B x2.1 (x2.7 at 2,048), Qwen3-1.7B x1.44, the Qwen3 loss starting in block 1. The op-level attribution tool (`llama_attrib.py`, 1bd0da9) located it on a tiny Qwen3 with the real q/k-norm gains: the k norm's one int8 scale is set by channels with gains up to 96.5 where q's is about 0; the exact, constants-only `qk_balance` (default on for Qwen3) fixes the reproduction (block 1: 0.87 -> 0.99). The real-checkpoint attribution (job 1011410, CPU) and the 16 x 128 ladder with `qk_balance` (job 1011411) are submitted. Qwen3 digests change. Kill date Fri 23 Oct.
  - **What.** Generalise `_RealBuilder`:
    - fold RMSNorm γ into the next matrices, and smooth through the norm's gain vector as for OPT;
    - use Qwen3's q_norm/k_norm γ as per-head gains;
    - build the SiLU table from the gate's real scale, and requantise the SwiGLU product from real scales;
    - Q14 RoPE tables, GQA, tied and untied heads, `prune_last` (calibrate_all), and an LM head chunked over positions (151,936-entry vocabulary);
    - `matches_benchmark_graph` gives an empty diff against `qwen3-4b` (add qwen3-0.6b and qwen3-1.7b to CONFIGS);
    - **outlier handling, moved in from F2b-L.** This uses a power-of-two exponent per token at the SwiGLU product (the input of down_proj), with the matching per-row shift in the next residual add.
      - Both are cheap operations that the verifier recomputes, so v0 is unchanged.
      - Under V1, that op's consumer stays clear until F2d: at most 9.6% of Llama-2-7B's claims, about +0.3 GB at 2,048 tokens (model).
    - **a cheaper alternative, evaluated beside it:** a PrefixQuant-style fixed prefix whose K/V become public constants in the graph digest. It needs no new op kind and leaves V1 unaffected. The cheaper of the two that reaches the target is taken.

    The ladder:
    1. Week 1, on the laptop at 16 x 128: Qwen3-0.6B-Base and TinyLlama-1.1B side by side, so that the model-family signal arrives by Sun 18. Then Qwen3-1.7B-Base.
    2. The same models at 2,048 tokens on the 2080 Ti (F2a path).
    3. Qwen3-4B-Base on the full test set at 2,048 tokens on the 2080 Ti (4.0 GB as int8), with F2a's `P_MAX` choice.
  - **Why.** C-W1, C-W5, C-Q2.
    - Only OPT has real-weight graphs. Qwen3 is ungated.
    - Qwen and Llama models carry massive activations at the down_proj input. Static per-tensor activation scales may therefore fail on Qwen3 as they do on Llama.
  - **Gain.**
    - Expected x1.03-1.06, but only if Qwen3 quantises like OPT under static per-tensor activation scales.
    - The literature reports x1.029 for SmoothQuant W8A8 on Qwen3-4B-Base. That figure is probably per-token dynamic quantisation, so it is checked before the paper quotes it.
    - The item also gives the graph for F3b and F2c.
  - **Effort / hardware.** 4 d if Qwen3 quantises like OPT, plus 1 d for the per-token exponent or the prefix. Laptop + 2080 Ti.
  - **Validation.**
    - An empty `matches_benchmark_graph` diff.
    - Against HF: per-block hidden-state cosine ≥ 0.99 and top-1 agreement (RoPE half-rotation, GQA head order, q/k norm).
    - 2-block real graphs accepted by v0 (C, Kpre) and V1, with tampering rejected.
    - The fp16 reference matches the published 7.90 (Qwen3-4B-Base at 2,048).
    - Perplexity ratio with a bootstrap confidence interval.
  - **Kill.** If Qwen3-4B is not within x1.10 by **Fri 23 Oct**:
    - report the measured number with the attribution;
    - lead quality with OPT;
    - label the Qwen cost rows "shape only".

- (F2b-Q, 12 Oct) `glu_smooth`'s first run (job 1012500) overflowed in `swiglu` on both Qwen3 models: up's int8 rows vanished after one dead channel's row grew 256x. Fixed in 207b798 (a box that keeps both weight matrices' scales; an explicit error for a zero product percentile); rerun with TinyLlama and the 2,048-token cells: jobs 1012675-7. F2b-Q doc Sec. 13.7a.
- (F2b-Q, 12 Oct) **Qwen3-1.7B at 2,048 tokens with prefix + balance (job 1012677):** smooth x1.455, smooth-exp8 x1.243, glus0.5 alone x1.456, **glus0.5 + exp8 x1.206 [1.200, 1.211]** (top-1 0.77); before the balance x1.73 / x1.45. Qwen3-4B with this recipe: job 1013332 (1013018 ran out of GPU memory loading the fp32 reference; now on the CPU). **Qwen3-4B (job 1013332): x5.54 smooth, x3.02 with the exponent, x2.97 with exponent + `glu_smooth` 0.5** — far from x1.10 and from Qwen3-1.7B's x1.21 with the same cells; diagnosis (per-block + op attribution, prefix): job 1013928. Kill rule Fri 23 Oct. **Diagnosis (jobs 1013928, 1013947, 1014112):** a second massive activation (about 3,000-3,500, channel 4) at the first window position from block 7, which no public prefix moves (two-token, outlier-token and four-token prefixes tried: x3.5-4.3 with the exponent at 128 tokens, against x3.8), plus SwiGLU products of blocks 1-4 with relative errors up to 0.93. The next lever (a per-token residual scale) changes the graph and V1, so it is out of scope before the freeze: recommendation for Fri 23 Oct, apply the kill rule (Qwen rows "shape only"; OPT and OpenLLaMA-7B carry the quality claim) [team]. The 128-token `glus2` ladder (job 1012675): `glu_smooth` helps only with the exponent and only at 0.3 there (Qwen3-1.7B x1.334 -> x1.304), so the setting depends on the length; the 2,048-token grid (job 1013482) is flat: with the exponent 0.3 / 0.4 / 0.5 / 0.6 give x1.212 / **x1.202** / x1.206 / x1.220.
- **F2b-L [WIP] should (rank 6): Llama on F2b-Q's builder.** The per-token exponent (the draft's part 1) is now inside F2b-Q.
  - **Status (12 Oct).** TinyLlama_v1.1 with the prefix (`auto` = `\n <s>`; job 1009797): x1.15 [1.10, 1.24] at 16 x 128 (target met) but x1.56 [1.51, 1.62] at 2,048 tokens (smoothed, alpha 0.65; the ratio still falls with alpha; the per-token exponent hurts, x2.11). Next: alpha 0.75 / 0.85 and `glu_smooth` on TinyLlama at 2,048 (after the glu_smooth fix, job 1012656's traceback). Llama-2-7B still needs the licence [USER]. F2b-Q doc Sec. 12.6.
  - **Status (12 Oct, job 1012676).** **TinyLlama x1.147 [1.138, 1.157] at 2,048 tokens** (smoothing alpha 0.75 + `glu_smooth` 0.3, top-1 0.83): the target x1.15 is met at 2,048 tokens too. alpha 0.75 alone x1.227; `glu_smooth` 0.5 hurts. Finer grid (job 1013124): **x1.122 [1.117, 1.129]** with smoothing 0.75 and `glu_smooth` 0.2 (top-1 0.83); 0.05-0.15 level off at x1.124-1.129 (job 1013574). F2b-Q doc Sec. 13.7a.
  - **A stand-in for Llama-2-7B without a licence (12 Oct, job 1013162).** OpenLLaMA-7B v2 (`openlm-research/open_llama_7b_v2`, Apache 2.0, not gated) has exactly Llama-2-7B's shapes (4,096 x 32 layers, 32 heads, ff 11,008, vocabulary 32,000, rope theta 10,000; only `max_position_embeddings` 2,048 differs, which a RoPE graph does not use), so `--config llama2-7b --structure` checks that its integer graph is the benchmark graph. 2,048 tokens with the prefix, TinyLlama's recipe (smoothing 0.65 / 0.75, with and without `glu_smooth` 0.3); the fp32 reference runs on the CPU (27 GB) and is cached on the shared disk. Its tokenizer needs `sentencepiece` and `protobuf`, installed with `--no-deps` into `logs/validation/sp2027/pylib` (not the shared venv). Positions 2,048-2,049 (the two prefix tokens push the window past 2,048) lie just beyond OpenLLaMA's training length, for the reference and the integer model alike. If Llama-2-7B's licence comes, the same command runs on it.
  - **OpenLLaMA-7B result (job 1013162).** Structure diff against the benchmark `llama2-7b` graph: empty. At 2,048 tokens (160 windows, fp32 reference 6.478 on the CPU): smooth 0.65 x1.107, 0.75 x1.133; **smooth 0.65 + `glu_smooth` 0.3 x1.077 [1.074, 1.080]** (top-1 0.87), 0.75 + 0.3 x1.092. A Llama of Llama-2-7B's exact shape is within x1.08. Lower smoothing (job 1013787, 5 of 6 cells before its time limit): **x1.073 [1.070, 1.075]** with smoothing 0.55 + `glu_smooth` 0.2 (top-1 0.88); flat over 0.55-0.65 / 0.2-0.3.
  - **What.**
    1. TinyLlama-1.1B runs in week 1 beside Qwen3-0.6B (F2b-Q's ladder), so its signal arrives by Sun 18.
    2. Llama-2-7B on the full test set at 2,048 tokens on the 2080 Ti (6.7 GB).
    3. Llama-2-13B in the L40S quality tier, only if the friend holds the Llama-2 licence on their own account (Sec. 6).
    4. Only if a leave-one-out shows they matter: a per-block residual rescale, and q/k smoothing that is equal within each RoPE pair.

    For real-weight Llama timing, the native CPU requant and residual kernels need a per-row-multiplier variant; otherwise the torch fallback is used.
  - **Why.** C-W5, on the model the cost tables lead with. Static per-tensor W8A8 Llama-2-7B gives 9,250 perplexity in the literature; the failure sits at down_proj's input.
  - **Gain.** Expected x1.04-1.10 (from Li et al. 2024 and MobileQuant).
  - **Effort / hardware.** 1 d. Laptop, 2080 Ti, L40S (13B). Needs the Llama-2 licence [USER].
  - **Validation.** As F2b-Q, plus the fp16 reference matching the published 5.47.
  - **Kill.** If TinyLlama is not within x1.15 by **Sun 25 Oct** (first result Sun 18):
    - Llama's quality becomes a stated, measured limitation;
    - the cost tables lead with OPT-6.7B and Qwen3-4B;
    - the Llama-2 rows are labelled "quality not measured".

- **F2c [WIP] should (rank 2): zero-shot accuracy of the certified integer models.**
  - **Status (11 Oct).** Harness `experiments/9_quality/zeroshot.py` (lm-eval 0.4's six tasks re-implemented; exact right padding; paired bootstrap) and `tests/test_zeroshot.py` (9 passed) merged (6b31e45); job 1011383: check passed; fp16 reproduces the reported harness numbers (within 0.2 / 0.1 point); with F2a's round-1 recipe (channel maxima, alpha 0.5, 11 bits) the integer OPT-125M / OPT-1.3B lose 1.0 / 0.7 point on average (OPT-1.3B significant only on LAMBADA, -3.0); the paper builder's current recipe (percentiles, alpha 0.55) loses 2.8 (LAMBADA -11.3), supporting the switch to channel maxima. Open: OPT-6.7B and the final recipe's rerun. Write-up `docs/improvements/F2c_zero_shot.md`.
  - **Status (12 Oct, certified recipe; job 1012440).** OPT-125M mean -1.03 (LAMBADA -3.1); OPT-1.3B mean -0.82, only LAMBADA significant (-2.04 [-2.74, -1.40], against -3.0 at alpha 0.5; ppl 6.64 -> 7.56); OPT-6.7B integer-only accuracies recorded. The OPT-6.7B fp16 reference does not fit one card with the integer graph: `zeroshot.py --ref-gpus 2` (`split_opt`) and `--pair` (eefc100); job 1012655 on two 2080 Ti. Paper: the body sentence uses the certified numbers; the extended table follows the 6.7B pairing.
  - **OPT-6.7B paired (12 Oct, job 1012655, fp16 split over two 2080 Ti).** Mean +0.06 points (LAMBADA -0.52 [-1.03, -0.04], every multiple-choice interval contains 0; same answer on 94-99%). The loss falls with size: -1.03 / -0.82 / +0.06. Paper: body sentence and the extended version's table `tab:zeroshot`. F2c is complete for OPT; Qwen3/Llama follow F2b.
  - **What.**
    - An lm-eval-harness LM class over F2a's lean integer forward. It is batched and right-padded, which is exact for a causal model.
    - Tasks: LAMBADA, HellaSwag (fixed 2,000-example subset), PIQA, ARC-e, ARC-c, WinoGrande.
    - The integer and fp16 models are scored on the same examples, with a paired bootstrap confidence interval of the difference.
    - Models in order: OPT-1.3B and Qwen3-1.7B first, then OPT-6.7B, Qwen3-4B and Llama-2-7B as F2b delivers them.
  - **Why.** C-W5 ("there are no downstream tasks"), C-Q2, chg 7.
  - **Gain.** Expected mean drop ≤ 1 point for models within x1.03-1.05 perplexity. About 10-25 min per variant for the 4-7B models on the 2080 Ti.
  - **Effort / hardware.** 2 d. 2080 Ti/Xeon. The datasets cache goes in `/tmp` (about 0.3 GB).
  - **Validation.** The wrapper's logits equal `real_logits` on OPT-125M. The fp16 accuracies reproduce the harness's published numbers within 0.5 point.
  - **Kill.** Must start by Mon 2 Nov. The results do not depend on the hardware, so it may run after the code freeze with the frozen builders.

- **E2b [WIP] should (rank 3): exact total-variation distance of the integer sampler on real logits.**
  - **Status (11 Oct).** Merged (6edd8d8): exact law of the sampler, Monte Carlo and toy checks. **Finding:** on OPT-125M the 16-bit sampler's TV to softmax is about 2.6e-2 at tau = 1 (7.7e-3 with top-k 50), not 1e-4 to 1e-3; a table with 4 more fractional bits is measured as a remedy (a protocol change: team decision). OPT-1.3B done (job 1008675): at tau = 1 the TV median is 0.0063 (k = 50) and 0.020 (k = 0), against 0.087 between the integer and fp32 model; a 2^20 table gives 0.0022. The paper's `\todo` is filled with these numbers and a `\todo{E2b: adopt it?}` for the team's decision. Write-up `docs/improvements/E2b_sampler_tv.md`.
  - **What.**
    - Compute the exact output distribution of the Gumbel-max sampler. log M(t) is computed once on the integer grid (about 0.93 M points, a float64 FFT convolution) and gathered per token, with ties bracketed.
    - Cross-check it against a Monte Carlo of `TokenSampler.choose` over 10^5 seeds.
    - Report TV(sampler, softmax(cz/τ)) and, separately, TV(integer logits, fp32 logits). The runs use 512 WikiText-2 positions of OPT-1.3B and Qwen3-1.7B, for τ ∈ {0.7, 1, 1.3} and top-k ∈ {0, 50}.
  - **Why.** C-W6, chg 7, and the `\todo` at generation.tex:87.
  - **Gain.**
    - Expected sampler TV of about 1e-4 to 1e-3 at τ = 1, against a worst-case bound of 0.033. That is at least 10x below the integer model's own TV to fp32.
    - It adds one measured sentence and a small figure.
  - **Effort / hardware.** 1.5 d, laptop, alongside GPU jobs.
  - **Validation.**
    - The exact distribution matches the Monte Carlo within 3σ per token, and matches closed forms on 2- and 8-token toys.
    - TV falls when the table has more bits (simulated 20-bit table).
  - **Kill.** Must start by Mon 2 Nov. Without a result by Mon 9 Nov, the `\todo` text is deleted.
  - **Could, E2b-3 (1 d):** exact integer top-p, repetition penalty and logit bias as public integer transforms of z (C-W6's feature gap).

- **F3b [TODO] should (rank 7): Qwen3-4B with real weights, end to end, and real generation** (needs F2b-Q).
  - **What.**
    - Qwen3-4B-Base, full model, in Kpre (C if it fits in 11 GB), at 64 and 2,048 tokens: bytes, times, memory and tampers as in F3a.
    - Real generation with P = 512, G = 64, greedy and sampled, with the two tamper cells. The generated text is now meaningful.
    - A codec-only byte profile for Qwen3-4B, and for Llama-2-7B if F2b-L succeeds.
    - The final-platform timings go in the L40S `real` tier.
  - **Why.** C-W1 and chg 12 (generation at scale). It gives a second model family with real weights.
  - **Effort / hardware.** 1 d active. 2080 Ti.
  - **Validation.** As F3a. The generated tokens equal `greedy_tokens`/`sample_tokens` on the same integer graph.
  - **Kill.** Must be merged into the package by Wed 28 Oct. Otherwise only the bytes, measured on the 2080 Ti, are reported.

### 3.3 Verifier

- **V-LONG-a [DONE] must: the full models on the Xeon now, measured and not extrapolated.**
  - **Status (11 Oct).** Measured (jobs 1006979, 1007816): full Llama-2-7B, Kpre, native CPU verifier on 8 Xeon threads: 1.01 / 1.40 s at 64 tokens and 18.5 / 19.6 s at 2,048 tokens (interactive / Fiat-Shamir), 3/3 accepted; the prover ran on the CPU at 2,048 (the 2080 Ti ran out of memory). Write-ups RX and FS1.
  - **What.**
    - Run `perf.py` with the native CPU verifier (B5) on 8 Xeon threads (`--exclusive`):
      - the full Llama-2-7B, Qwen3-4B and OPT-1.3B (random weights);
      - in Kpre, at 64 and 2,048 tokens;
      - with today's fp32 and integer re-execution on the same threads. The RX baselines join the same cells as they land (Thu-Sat).
    - Submit on Mon 12 Oct with today's code (interactive) as a baseline. Submit again under Fiat-Shamir as soon as FS1 part 1 lands (Mon 12), since Kpre then hashes nothing.
    - The prover runs on the 2080 Ti, or on the CPU if a cell does not fit.
    - **V1 re-time, also Mon 12.** Run V1 in Kpre on Llama-2-7B (2-3 blocks) with the B5 kernels.
      - This decides V-V1N.
      - It also fixes EV's V1 numbers and tab:options' "x1.6-1.9", which date from before B5.
  - **Why.**
    - A-W6 and C-W1: the paper's 17.2 s and 1.6 s are extrapolated from 1-3 blocks.
    - With RX, this measurement is a gate G1 input and the source of the abstract's CPU ratio.
    - It also rehearses the `cpuv` tier.
  - **Gain.** The CPU ratio becomes a measurement. The model gives 4.5x against fp32 at 2,048 tokens, and about 3x against RX's strongest exact baseline on the Xeon.
  - **Effort / hardware.** 0.5 d active. Xeon + 2080 Ti.
  - **Validation.**
    - The measured full-model time is compared with the 17.2 s extrapolation.
    - Every query is accepted.
    - The V1/v0 verifier ratio is measured with today's kernels.
  - **Kill.** None. If the full model does not fit the node's `--mem`, measure 16 blocks and say so.

- **RX [DONE] must: the strongest exact re-execution baselines on the same CPU.**
  - **Status (11 Oct).** Xeon done (jobs 1006979): strongest exact baseline 4.08 s (64 tokens, integer with fp32 GEMMs) and 93.2 s (2,048, fp32); verifier ratios 2.9-4.0x (64) and 4.7-5.0x (2,048); the native int8 GEMM is slower than MKL fp32 on Skylake (no VNNI). The EPYC rows come from F6 `cpuv`. Write-up `docs/improvements/RX_reexecution_baselines.md`.
  - **What.** Every baseline runs on the same 8 threads as the verifier: on the Xeon in week 1 (within V-LONG-a), and on the EPYC in the `cpuv` tier. Timing jobs use `--exclusive`.
    1. **Optimised exact integer re-execution.**
       - `IntGraph.forward` runs with B5's native non-weight kernels (attention, requantisation, recomputed operations).
       - The weight products use the fastest exact int8 GEMM available on that CPU (VNNI on the EPYC). This replaces today's fp32 GEMMs over 1,024-term chunks (A2 doc §3).
       - Its logits equal those of `IntGraph.forward`.
    2. **bf16 re-execution** of the fp16 checkpoint. The EPYC has native AVX512_BF16. The Xeon 4114 has neither AVX512_BF16 nor VNNI, so the EPYC rows are the ones that count for bf16 and int8.
    3. **A standard int8 engine:** torch dynamic-int8 `Linear`, or llama.cpp Q8_0. It is labelled **not exact**: it computes a different model, so it is reported beside the other baselines and does not count for the gate.
    4. Today's fp32 and integer `reexec.py` rows stay as reference rows.

    The **strongest exact baseline** is the fastest of fp32, bf16 and exact integer re-execution on the same threads. Gate criterion 2, EV's break-even links and the abstract all use it.
  - **Why.**
    - The current "integer re-execution 98 s" is a straw man. On the CPU, `MatOp.compute` still runs its weight products as fp32 GEMMs over 1,024-term chunks, and `sp2027.sh cpuv` only times `reexec.py` in fp32 and integer.
    - The derive alone takes 0.381 s per block, about 12 s for the full model (B5 doc §6).
    - A rough model puts our verifier at about 3x faster than an optimised exact re-execution on the Xeon, and only about 2x on the EPYC with VNNI, not 4.6x.
    - This is the abstract's only performance claim and reviewer C's path to a 4.
  - **Gain.** A CPU ratio that survives the obvious objection. The abstract quotes it only after RX and V-LONG-a, as "at least N times", with N a lower bound on both CPUs (Sec. 1).
  - **Effort / hardware.** About 1 d, in week 1. Xeon now; EPYC in `cpuv`.
  - **Validation.**
    - The optimised integer path's logits equal `IntGraph.forward` (`torch.equal`) in every cell.
    - The top-1 agreement of fp32, bf16 and the int8 engine with the fp16 reference is reported.
    - ≥ 5 timed queries after a warm-up, reported as median [min-max].
  - **Kill.** If the int8-GEMM path is not bit-exact by Sun 18 Oct:
    - the int8-GEMM time is added as a modelled row, from the measured GEMM throughput;
    - the gate and the abstract use the smaller of the measured and modelled ratios;
    - the paper labels the row as modelled.

- **V-V1N [WIP] should (rank 1): native V1 verifier.**
  - **Status (11 Oct).** Parts 1-4 in torch merged (89e08a0, PVI_V1_BUCKETS): exact (26 new tests; V1 suites 231 and v0 regression 456 passed), final-check products about 2x and the GKR check about 2x faster on the laptop; projected full Llama-2-7B V1 verifier about 50-55 s (from 75-80). The plan's target (<= 1.2x v0) needs the native kernel (designed in `docs/improvements/V_V1N_fast_v1_verifier.md`), and the adversarial re-review is still to run. Xeon measured (job 1009430): per Llama-2-7B block 2.49 -> 2.13 s (GKR 1.8x, final check only 7%: not understood, profile first), full model about 67 s against v0's 19.6 s (3.4x); suites pass in every switch position. On the Xeon the table products lose on the residual ops (route them to the limb GEMM). Profile (job 1011404): per block the products take 0.86 s (old) / 0.82 s (table products) of verify_final's 1.24 s, so the native bucket kernel is the target (projected ~1.4 s per block, ~43 s full model, 2.2x v0; the derive's int32 path is needed for the plan's 1.2x). Native products kernel merged (e4eb170, d9b5748; job 1011440): products 0.86 -> 0.16 s per block, V1 verifier 2.52 -> 1.44 s per block, full Llama-2-7B about 45.4 s (2.3x v0); V1's break-even link against re-execution 0.47 Gb/s (v0 0.70). The int32 test exposed and fixed an in-place shift in the torch table path (no verdict affected).
  - **Decision rule.** The V1 re-time of Mon 12 (V-LONG-a) decides.
    - If V1's verifier takes more than 1.2x v0's time (the model says about 3.5x), parts 1-4 become the first should item and start Mon 19.
    - Otherwise V-V1N is not started, and V1 is reported as measured.
    - If parts 1-4 are needed but miss, V1 is presented as a bytes-only option in EV and in the abstract.
  - **What.**
    1. `cut_final` bucketed by value.
       - L and W depend only on the int8 value s. Per row, accumulate 256 buckets B[v] += e_j: one 8-lane int64 add per entry.
       - Then L·e = Σ lo(v)B[v] plus the exceptions' exact z, and W·e = Σ W(v)B[v].
       - This replaces the float64 copies of [L; W] and the 24-limb float64 GEMMs.
    2. The same bucketed kernel for `_xbar`, shared by q/k/v and gate/up.
    3. Every GKR instance's round chains checked in one batched pass, and the logUp identity checked with batch inversion.
    4. The derive reads the int8 values directly.
    5. **Could:** on the GPU, the same check as a Triton kernel using `gkr_triton`'s BabyBear code (no fp64 GEMMs). It joins V-GPUFLOOR.
  - **Why.** chg 11.
    - "V1 costs 1.6-1.9x the verifier time" was measured before B5. With today's v0, the Llama-2-7B 2-block cell would be about 3.5x v0.
    - The projected full Llama-2-7B V1 CPU verifier, about 60 s, would nearly erase the advantage over fp32 re-execution (78.8 s).
  - **Gain.**
    - V1's final check plus xbar: about 1.4 s → ≤ 0.1 s per block.
    - The full Llama-2-7B V1 verifier goes from about 60 s to about 17-21 s (model). V1 then gives a 2.2x smaller proof at ≤ 1.2x v0's verifier time.
    - Only the prover pays for V1, so the bandwidth crossover depends on the prover alone.
  - **Effort / hardware.** 3.5 d in all: about 2.5 d for the CPU parts 1-4 and about 1 d for part 5. The CPU parts run on the Xeon (native, compiler from I4); part 5 runs on the 2080 Ti (Triton).
  - **Validation.**
    - The V1 suites (T4/T5/T7, test_logup, test_logup_gkr) give the same verdicts and labels on every tamper, with `PVI_NATIVE_CHEAP` = 1 and 0.
    - The native results equal the Python `_final_ok` at random F_{p^8} points, with exceptions, at every window width and in both layouts.
    - The GPU result equals the CPU result (part 5).
    - Timing slope over 1, 2 and 3 blocks; the target is V1 ≤ 1.2x v0.
    - The review fixes stay in force: canonical folds in [0, p), and P4 computed from the verifier's weights.
    - The adversarial-review workflow is re-run on the changed V1 verifier.
  - **Kill.** Start by Fri 23 Oct at the latest (Mon 19 if the gap is confirmed). If it is not exact against `_final_ok` by Wed 28 Oct:
    - it stays off by default;
    - V1's verifier is reported as measured;
    - V1 is presented as a bytes-only option in EV and in the abstract.

- **V-LONG-b [DONE] should (rank 5): long prompts from 2k to 16k tokens, measurement only.**
  - **Status (11 Oct).** Job 1011426: full Llama-2-7B (2-3 blocks extrapolated) verifier 17.6 / 45.2 / 124 / 390-470 s at 2k / 4k / 8k / 16k tokens against fp32 re-execution 90 / 189 / 399 / 930 s: 5.1x / 4.2x / 3.2x / 2.0-2.4x; proofs 6.4 / 12.9 / 25.8 / ~51 GB; fit t = aT + bT^2 within 3% at 4k; the 16k 3-block cell ran out of GPU memory (L40S `long` tier). 32k modelled only. Write-up `docs/improvements/V_LONG_b_long_prompts.md`.
  - **What.**
    1. Cells at 2k, 4k, 8k and 16k tokens, 2-3 blocks each, for Llama-2-7B shapes (`max_pos` raised) and Qwen3-4B, in Kpre.
       - Each cell measures the CPU verifier, the fp32, integer and RX baselines on the same threads, and the exact bytes.
       - The cells use today's kernels, which are valid below 32,768 tokens.
       - Fit t(T) = aT + bT^2 per block. The 32k numbers come from the fit and the byte model and are labelled as modelled.
    2. An 8k full-model `cpuv` cell on the EPYC, if its RAM allows (the claims alone are about 44 GB).

    The draft's kernel rewrite past 2^15 (and the measured 32k cells it enables) moves to after 17 Nov.
  - **Why.** C-W7 and C-Q7: the paper does not project proof size or the O(T^2) verifier at 8k-32k tokens.
  - **Gain.**
    - Verifier/re-execution against fp32 (model): about 4.5x at 2k and 3.2x at 8k, both measured. At 32k, about 1.9x, modelled.
    - Bytes for Llama-2-7B at 8k / 32k: 25 / 100 GB (v0) and 11 / 43 GB (V1). The 8k bytes are measured; the 32k bytes are modelled.
  - **Effort / hardware.** About 1 d. Xeon/2080 Ti.
  - **Validation.** Fit residuals < 10%. Bytes within 2% of the model.
  - **Kill.** Start by Thu 22 Oct. Cells not measured by Mon 9 Nov are reported as modelled and labelled so.

- **V-SPARSE [TODO] could: exact sparse attention tail.** At most 510 keys per row get a non-zero probability, for any T, because p ≥ 1 exactly when 510e ≥ tot.
  - Survivors-only softmax and P·V in the AVX-512 kernel; key tiling for T > 8k; VNNI for the EPYC (off unless the EPYC smoke tests pass).
  - Gain at 32k: P·V needs ≥ 32x fewer MACs, and the verifier/fp32 ratio goes from about 1.9x to about 2.9x. At 2k: about 17.2 s → about 16 s.
  - 4 d. The bound itself is stated in the paper in any case (0.1 d of writing). P·V costs O(510·T·dh), and the same fact explains the 8-bit mass loss that F2a measures.

- **V-GPUFLOOR [TODO] could: GPU verifier at its PCIe floor with bounded memory.** It absorbs B4, C2's "streaming GPU verifier with the cut" and V-V1N part 5.
  - The **roofline statement** is writing and goes into EV (must). The ratio of the floors is c·b·F / (2·P·BW): about 1-1.7 for v0 and 0.5-0.7 for V1.
  - The code parts are could, 6 d:
    - decode block by block from one pinned buffer, with chunked host-to-device copies overlapped with unpacking;
    - drop the per-level `nonzero()` syncs;
    - free each block's claims after its check;
    - B4's fused int32 derive;
    - V1 on the streaming GPU verifier.
  - Gain if done: the OPT-1.3B GPU verifier goes from about 1.1 s to about 0.34 s (v0) or about 0.16 s (V1), against 0.18 s for fp16. A full Llama-2-7B GPU verifier becomes possible on the 11 GB card.
  - Not before the freeze unless every should item is done.

- **B5b [TODO] could:** keep requantised values as int8/int32 between native ops.
  - Gain: about 15% of the CPU verifier (17.2 s → about 15 s). Effort: 1.5 d.
  - This is B5's last open remainder, and it is unlikely to move a reviewer.

### 3.4 Prover and serving

- **A7 [WIP] must: prover overhead against unverified serving on the same GPU, with batching (new `serve` tier).**
  - **Status (11 Oct).** Harness `serve_ab.py` and the `serve` tier merged (eef4d1a); laptop checks on tiny graphs (U1 = claimed logits, byte-identical proofs on request). 2080 Ti pilot done (job 1008674): OPT-1.3B at 2,048 tokens, B = 1, prover 2.08 s (Kpre) / 4.37 s (C) = 4.1x / 8.6x U1 and 19x / 40x fp16; B = 64 at 64 tokens, joint Kpre proof 4.8x / 18x; Llama-2-7B extrapolated from 1-2 blocks 3.8x / 9.8x (Kpre, 2,048). The U2-against-F1 check missed its 10% band (OPT-1.3B 40% faster than F1's extrapolation; Llama-2-7B 13% slower), explained in the write-up. Paper `\todo`s (intro, evaluation, discussion) filled from the pilot; the L40S `serve` tier replaces them. Write-up `docs/improvements/A7_serving_overhead.md`.
  - **What.** A new script, `experiments/7_reexec/serve_ab.py`.
    - **Method.** Same token ids, shapes, GPU and host; CUDA events after a warm-up; ≥ 10 repetitions; median with a 95% bootstrap confidence interval.
    - **Baselines.**
      - U1: our exact integer model served without proofs (`IntGraph.forward(free=True)`, claims dropped). This isolates the cost of the proof itself.
      - U2: eager fp16/bf16 decoders with SDPA (`reexec.py` extended to batch B with a static KV cache).
    - **Prover.** Modes C and Kpre, Fiat-Shamir (FS1 counted), wire, lean, pruned.
    - **Workloads.**
      - (a) Prefill at B = 1 for T ∈ {64, 512, 2048}.
      - (b) Batched prefill: T = 64 with B ∈ {1, 4, 16, 64}, and T = 2048 with B ∈ {1, 4}.
    - **Models.** Full OPT-1.3B, Qwen3-4B, Llama-2-7B, OPT-6.7B and Llama-2-13B on the L40S.
    - **Outputs.**
      - the prover/U1 and prover/U2 ratios;
      - the proof-only overhead, split into encode, hash, fold and open;
      - proving time and proof MB per token;
      - the egress cap. Llama-2-7B's 3.32 MB per token caps proved throughput at about 3,800 tokens/s per 100 Gb/s.
    - **A "proof on request" row.** Claims are a deterministic function of the model and the prompt, so a proof produced later is byte-identical. Serving then costs U1, and the prover is paid only for the proofs that clients request.
    - **Generation (C-Q5)** is answered in text from the measured E1 cells. The proof is one prefill over x‖y after decoding: 0.116 s per 256-token OPT-1.3B response on the 2080 Ti. The KV-cached decoder is A10 (could).
    - The `serve` tier lands in the same commit as the harness (F6 part 13).
  - **Why.**
    - C-W2 and C-Q1: one of the things C named to move from 2 to 3.
    - A-Q2 (CommitLLM reports 12-14%) and chg 3.
    - Today the paper compares the L40S prover with fp16 on a 2080 Ti, across GPUs.
    - It fills the `\todo`s at evaluation:100, intro:111 and discussion:30.
  - **Gain.** The answer to the reviewer's question. These ratios are reported as measured, whatever they are. From the measured same-GPU numbers we expect:
    - 9-40x over fp16 eager at B = 1;
    - a floor, set by the integer engine alone (F1, A2), of about 1.8x at 2,048 tokens and about 2.6x at 64 tokens for Llama-2-7B on the 2080 Ti;
    - with batching (C4a), about 3-8x at B ≥ 8 (model). This comes from the base-code L40S cell: 0.39 s per query at B = 8, against 1.78 s at B = 1.
  - **Effort / hardware.** 2 d.
    - The laptop builds the harness on tiny decoders.
    - The 2080 Ti runs a pilot (full OPT-1.3B, Llama-2-7B 1-2 blocks).
    - The L40S `serve` tier takes about 8-10 h.
  - **Validation.**
    - U1's logits are identical (`torch.equal`) to the prover's claimed logits in every cell.
    - U2 on the 2080 Ti reproduces F1's 0.57 s (Llama-2-7B at 2,048) and 0.18 s (OPT-1.3B) within 10%.
    - No cell is extrapolated.
    - The phases sum to the total within 2%.
  - **Kill.** None: the harness is small. If U2's batched static cache slips, report B = 1 plus the C4a batch cells.

- **A7-vLLM [USER] should (rank 9): the serving-engine baseline U3.**
  - **Setup.** vLLM in a separate venv on the friend's node (about 5 GB, a release pinned to torch 2.5.1), with dummy weights (no downloads, no licence) and CUDA graphs.
  - **Precisions.** fp16/bf16 and fp8 on sm_89; int8 W8A8 if the dummy path loads.
  - **Fallback.** HF transformers from a config, with a static cache and `torch.compile`.
  - **Effort.** 1 d. It is not possible on the 2080 Ti (quota).
  - **Kill.** If the venv is not approved by Fri 30 Oct, report U1 + U2 plus the HF fallback.

- **A6 [DONE] should (rank 8): Kpre precompute on exact int8 GEMMs (GPU) or B5's native field product (CPU).**
  - **Status (11 Oct).** Implemented (bd0f95d): `_fold_local` uses `native_kernels.field_matmul` on a CPU and int8 GEMMs on a GPU, the float64 path as fallback; `tests/test_fold_fast.py` (4 passed on the cluster); job 1011448: full Llama-2-7B precompute (Fiat-Shamir) 106.9 -> 12.7 s on 8 Xeon threads, 4.3 -> 2.2 s on the 2080 Ti, `u` identical. Write-up `docs/improvements/A6_kpre_precompute.md`.
  - **What.** `Verifier._fold_local` computes u = chi^T·A with `field.weight_matmul_mod` (GPU) or the native `field_matmul` over row blocks of the int8 matrix (CPU). This removes the transposed int64 copy and the float64 GEMMs.
  - **Why.** The Kpre client's one-time setup belongs in the client-options table (A-W1, chg 3). The stored base record shows 96.5 s for Llama-2-13B. That is already small next to a 6.7 GB weight download, which is why A6 ranks eighth.
  - **Gain.** CPU: about 6 s for 13B (16x) and about 3 s for 7B. GPU: about 0.1 s. Times 1.4 under Fiat-Shamir (r = 7).
  - **Effort / hardware.** 1 d. Laptop + Xeon/2080 Ti.
  - **Validation.**
    - u is bit-identical at the int8 extremes, at p - 1, and for K > 2^16, on CPU and GPU.
    - The Kpre verdicts are unchanged.
    - `verifier_precompute` is ≥ 10x lower.
  - **Kill.** Start by Sat 24 Oct. It is low risk, and the float64 path stays as the fallback.

- **C4a [DONE on the 2080 Ti] must (inside F6): measure v0 batches with the existing code.**
  - **Status (12 Oct, jobs 1012720 and 1013041).** Full Llama-2-7B shapes, 64 tokens, mode C interactive, `auto` (cnn16c): per query 324.5 / 264.8 / 233.9 / 218.7 MB at B = 1 / 2 / 4 / 8 against the model's 324.5 / 260.6 / 228.6 / 212.7 (0.0 / +1.6 / +2.3 / +2.8%, all accepted); the fixed 127.8 MB is shared exactly, the excess is the other prompts' claims (about 203 MB against the first prompt's 196.7). B = 16 does not fit the 11 GB card; the L40S `batch` tier adds it and the times. Write-up `docs/improvements/C4a_batch_bytes.md`.
  - These are cells only (F6 part 4). The `batch` tier lands with them.
  - The model for Llama-2-7B at 64 tokens is 196.7 + 127.8/B MB per query, interactive, or 196.7 + 51.8/B with PS2's R64c.
  - Check: per-query bytes within 3% of the model. The stored B = 8 cell measured 218.7 MB, against 212.7 MB predicted.

- **C4b [TODO] after 17 Nov: V1 batches** (the proof agent's PS3; part (i) of the prover agent's A9 for V1).
  - **What.**
    - Lift `run_cut_query`'s one-prompt limit, so that B·T columns enter the cut plan.
    - Every prompt's values, exceptions and multiplicities are absorbed before alpha, and one GKR packs every prompt's leaves.
    - The 8-plane folds and the opened columns are sent once per batch; attention stays per prompt.
    - The scope is one client's batch.
  - **Gain (model).** Llama-2-7B at 64 tokens, mode C, Fiat-Shamir: 85.6 + 229.4/B MB per prompt, i.e. 99.9 MB at B = 16 (93.0 MB with R64c), against 315.0 MB for one V1 query.
  - **Effort.** 2.5 d.
  - **Why it moves.** The review marks change 13 as optional.

- **C4c [TODO] could: a batch statement for independent users** (part (ii) of A9).
  - **Design.**
    - A Merkle root over the users' tree-hashed (x_i, claims_i).
    - Under Fiat-Shamir, chi = H(statement, R).
    - One u and one opening per batch.
    - Each user checks its own path, its own Freivalds identity and the shared columns.
  - **Before 17 Nov,** only the design paragraph (0.5 d) is written. The implementation (2 d) comes after.

- **A5 [TODO] could: pipelined prover.**
  - **What.**
    - The forward pass of query k+1 overlaps the encoding of query k and the hashing and sending of query k-1.
    - Within a query, per-op narrowing, statistics and pinned device-to-host copies run on a side stream, with byte-identical PVC3.
    - A raw int32 path serves gateways.
  - **Gain.** Latency about 1.3x; steady-state Fiat-Shamir serving throughput about 4.7x (model).
  - **Effort.** 4 d.
  - **Validation.** Byte-identical proofs (SHA-256) in every cell, and the overlap shown in a trace.
  - After 17 Nov unless every should item is done.

- **A10 [TODO] could: exact KV-cached integer decoder.**
  - **What.** `IntGraph.decode(kv=True)`, with a capture mode whose claims equal the prefill's.
  - **Gain.** Stage 1 gives only about 1.2x faster generation. Stage 2 (CUDA graphs) heads towards 2.1 ms per token for OPT-1.3B.
  - **Effort.** 5 d.
  - C-Q5 is answered in text (A7) without it.

- **A11 [TODO] could: V1 prover speed.**
  - A fused Triton cut split (0.81 s → ≤ 0.1 s) and a cheaper GKR round loop (-15-20%). 4 d.
  - Part (iii), the bandwidth crossover computed from the L40S V1 timings by a script, is writing and moves to EV (must).

### 3.5 Proof size

- **PS2 [WIP] must: lower-rate commitment plans for the mode-C cells, chosen under an explicit setup budget.** No protocol code.
  - **Status (11 Oct).** Byte tables (9cc0bd6); measured bytes in job 1006989 (2 blocks, mode C, FS: cnn16c 23.3 MB, cnn18c 16.0 MB, R64c 13.9 MB per query at 64 tokens, v0).
  - **Item 4 (12 Oct).** The package does not reuse setups: each of the 23 full-model mode-C invocations builds its own commitment (1,337 G encoded entries under `auto`, 3,436 G under cnn18c everywhere, about 15 h more at the 2080 Ti's rate). By the rule below, the timed cells keep `auto` (cnn16c for Llama-2-7B, cnn17c for the larger models) and cnn18c / R64c are byte rows. Options for Wed 21 [team]: `PVI_C_POLICY=cnn18c`, or a setup cache (about 0.5 d). PS2 doc Sec. 4.1.
  - **What.**
    1. Byte tables from `analytic.proof_bytes` for every model x {cnn16c, cnn18c, R64c} x {interactive, Fiat-Shamir}. The model reproduces the stored records within 1.2%.
    2. One measured run: `v1_bytes.py --model llama2-7b --layers 2 --seq 64 2048 --policy R64c cnn18c --fs` on the 2080 Ti node, with `setup_s` recorded.
    3. **Decision on Wed 21 Oct.**
       - Default: cnn18c for every timed mode-C cell. The Llama-2-7B setup takes about 35 min at the measured 67 M entries/s.
       - R64c for the byte rows (measured on 2 blocks, plus the model). R64c would cost about 2.4 h for 7B and about 4 h for 13B, per setup.
    4. Check whether the package caches setups between cells. If not, R64c is limited to one cell per model.
    5. Regenerate TH-b's concrete-security table and D4's setup-proof parameters for the chosen plan.
  - **Why.** A-W1: at 64 tokens, 124.7 MB of a 325 MB query are opened columns, which do not shrink with the prompt. Lower-rate plans also make Fiat-Shamir proofs cheaper (B-W7).
  - **Gain (model).**
    - Llama-2-7B at 64 tokens, mode C, Fiat-Shamir: v0 383.3 → 272.5 MB (-29%); V1 315.0 → 204.2 MB. cnn18c gets about 70% of that gain.
    - Interactive: v0 324.5 → 248.6 MB.
    - At 2,048 tokens: -111 MB (-3.7%).
    - OPT-1.3B under Fiat-Shamir: 51.3 → 30.6 MB of fixed part.
    - Kpre is unaffected.
  - **Effort / hardware.** 1 d. Laptop + 2080 Ti node; the final setups run on the EPYC.
  - **Validation.**
    - Measured bytes within 1.2% of the model.
    - `soundness_bits` ≥ 128 (+64 under Fiat-Shamir) for each plan.
    - `test_plans.py` and the end-to-end verdict tests green.
  - **Kill.** Keep cnn16c, and report the lower-rate plans as modelled rows, if either of these happens:
    - the 2-block bytes deviate by more than 1.2%;
    - the setup for Llama-2-7B under cnn18c takes over 1 h on the 2080 Ti node.

- **PS2-PoW [TODO] could:** a 20-24-bit proof of work before the Fiat-Shamir column and chi challenges (standard FRI grinding).
  - Gain: FS columns 176.8 → 156.1 MB (cnn16c) and 66.1 → 57.9 MB (R64c); Freivalds repetitions 7 → 6.
  - Effort: 0.5 d plus one line in the concrete-security table.
  - A wrong nonce must be rejected.

- **PS1 [TODO] after 17 Nov: V1 windows for ReLU outputs (send relu(a) for OPT's fc1).**
  - **What.**
    - Fuse the ReLU into fc1's requantisation (clamp [0, 127]).
    - Send the zero class as σ = -j, with a shared window segment.
    - The derive uses max(σ, 0), with the range check σ ∈ [-J, 127].
    - Low-clamped entries stop being exceptions.
    - PVC3, the GKR, the folds and the final check are unchanged.
  - **Gain (measured at 512 tokens).**
    - Bits per claim, v0 → V1 → V1 with the ReLU merge: OPT-1.3B 13.92 → 7.36 → 4.88.
    - Full OPT-1.3B at 2,048 tokens (model): V1 about 906 → 623 MB.
    - No gain on SwiGLU models (Llama, Qwen).
  - **Effort.** 3.5 d, plus a new adversarial review of the changed V1.
  - **Why it moves.** It is a soundness-relevant protocol change that needs a new review. Its gains are on OPT only, and nothing on the Llama/Qwen models that lead the tables. The paper's weakest V1 number (1.68-1.77x on real OPT-125M at 512 tokens) is instead qualified in text (chg 6(iv)).

- **PS5 [TODO] could: committed activations (V2m/V2) as an analysed design** (Plan 1 C3).
  - **Writing.** One paragraph in the discussion and a cost-model section in the extended version, regenerated by a script from `analytic.decoder_shapes` and labelled "modelled".
    - Model: Llama-2-7B at 2,048 tokens, about 891 / 795 MB (C / Kpre), against about 2.9 GB for V1.
    - The ingredients are named: a proximity test for prover-made commitments, tuple lookups, and 8-bit-limb norm sums.
  - **Optional benchmark.** The W-gate microbenchmark measures Triton NTT and SHA-256 Merkle throughput at 1e9 BabyBear entries on the 2080 Ti, against a threshold of about 20 GB/s.
  - **Effort.** 1 d of writing (+1 d for the benchmark).
  - The implementation (8-12 person-weeks) is for USENIX.

- **PS4 [TODO] after 17 Nov: context-modelled entropy coder (interleaved rANS).**
  - Gain: at most -4% on Llama random weights and -7% on real OPT, on top of PS1.
  - Cost: new GPU and CPU decoders that are 2-4x slower per symbol, for a small gain.

### 3.6 Attack

- **G3.5-L [DONE] must: the OPT-6.7B in-range attack on a Xeon node, before the freeze** (Plan 1 G3.5's remainder; reviewer A's condition 3, A-Q3).
  - **Status (11 Oct).** Job 1007804: OPT-1.3B reproduced exactly (every discrete result of the 320 records); OPT-6.7B in fp32: in-range edits change the next token on 40/40 prompts at layers 8/16/24/31, uniform detection at most 1.3e-3 per path, contribution weighting 0.02-0.76 (`any`) and 0 (`down`). Write-up G3_inrange_attack.md Sec. 3b.
  - **What.**
    - Run `experiments/2_attack/inrange_llm.py` on OPT-6.7B in fp32, on a Xeon node with `--mem` of at least 64 GB and HF_HOME on node-local `/tmp`.
    - Use G3.5's setup:
      - ranges calibrated on WikiText-2 train;
      - the 40 prompts;
      - four layers at the same relative depths;
      - the `any` and `down` moves against the `max` and `p999` ranges (4 variants).
  - **Why.**
    - The attack on the hardened protocol at 7B scale is one of A's three conditions. In the draft it sat in the L40S `attack` tier, which runs last and is the first to be cut.
    - OPT-6.7B in fp32 needs about 27 GB of RAM and about 5x OPT-1.3B's compute. That is about 1-2 h per variant on a Xeon node; OPT-1.3B took about 11 min per variant on 8 laptop threads.
    - Moving it, together with F3a's OPT-6.7B real V1 bytes, saves 6-12 L40S hours.
  - **Effort / hardware.** 0.5 d active and 4-8 Xeon job-hours. It needs I5 for the checkpoint space.
  - **Validation.**
    - The same job script first reproduces G3.5's OPT-1.3B rows.
    - Every edited value lies in range, so the range check passes.
    - The results go into the same table format.
  - **Kill.** The results do not depend on the platform, so the run may finish after the freeze. Without results by Mon 9 Nov, the paper reports OPT-1.3B only and states the limitation.

- **G3-min [DONE] should (rank 4): the adaptive game against value-aware samplers, minimal version.** The design is in `attack_theory/experiments.md` §1 and stays within scope rule K9.
  - **Status (11 Oct).** MNIST MLP done and merged (8663fe1): K-G3c passes (8.3e-14); at the 95th-percentile cap the best implementable sampler reaches 2.44% / 5.99% per path and the LP mixture 2.45% / 7.06% (threshold 5.41% / 10.82%), so K-G3a does not trigger for implementable samplers (gradient saliency does cross it); K-G3b does not trigger. OPT-1.3B done (job 1008671): against the two fixed in-range edits the mixture (contribution weighting) crosses 27.7/d_ff = 0.34% at layers 6, 12 and 18 (up to 48% at layer 6, natural maximum) and stays at 0.08-0.09% at layer 23, where zeroing suffices; per K-G3a the paper now says value-aware sampling helps against these edits on OPT-1.3B, and leaves the adaptive case open (K9). The three attack `\todo`s are filled. Write-up `docs/improvements/G3_min_adaptive_game.md`.
  - **What.**
    1. `fast_acceptance`, with an exactness test against `acceptance_probability` (K-G3c), on the MNIST MLP.
    2. The game on the MNIST MLP, both layers.
       - Ranges are calibrated on the train split, and the attacks use disjoint test queries. This fixes the overlap in the existing scripts.
       - Report R1 (edited values in range) and R2 (every claimed value in range).
    3. An LP mixture over the existing samplers on OPT-1.3B, at layers 6/12/18/23. It reuses G3.5's code (the helpers in `real_llm.py`, imported).
    4. G1e: the empirical G_q on the MNIST MLP, from the same script.

    The OPT-6.7B in-range attack is G3.5-L. A Qwen3-4B-Base repeat on the 2080 Ti runs only if the team says yes (Sec. 6).
  - **Why.** A-W2, A-Q3, and chg 5 (the hardened protocol with the best value-aware sampler). Three `\todo`s in attack.tex.
  - **Gain.** Fills the three `\todo`s with the best sampler's detection against in-range edits, as an upper bound.
  - **Effort / hardware.** 2 d. Laptop (MNIST, about 40 min); 2080 Ti (OPT-1.3B, about 1 h).
  - **Validation.** K-G3c (agreement within 1e-9); figures and table from `adaptive_game_plots.py`.
  - **Kill.**
    - The kill criteria K-G3a-d of the design apply as written. For example, if the LP mixture reaches d ≥ 27.7/N per path, the paper says that value-aware sampling helps on that model.
    - Without results by Mon 9 Nov, the three sentences are deleted.
    - The result does not depend on the platform, so it may run after the freeze.

### 3.7 Theory and paper

- **TH [WIP] must: close the theory gaps and finish the proofs the submission cites** (absorbs Plan 1 D7).
  - **Status (11 Oct).** TH-a, TH-c, TH-d integrated (0f843c8); TH-b filled by `params_table.py` (761b8a7); TH(g) sentence added; TH-e in the extended version as sections/ext_proofs.tex (45d592f). Open: the team's hand check (TH-f, [USER]) and the decisions listed in drafts/th_e_check.md.
  - **What.**
    - **(a) A block-mixing proposition.** Rows are grouped in blocks of d, and each block is RS-encoded and Merkle-committed on its own.
      - Then q = (N/d)(1 + t) queries per layer reach 2^-λ, with qd = (1 + t)N.
      - Together with Thm. dich(a), this makes the bound tight up to O(λ) at every locality d.
      - It replaces "intermediate locality is open" (B-W5, B-Q4).
    - **(b) The 9 `\todo`s of tab:params,** filled by a committed script from `setup_bytes.py`.
      - Today's values: 130 matrices in 6 trees, n = 65,536; β = 136.8 / 200.8 (interactive / Fiat-Shamir); t_j = 35-133 / 51-194; opened columns 124.7 / 182.5 MB; s = 7; t0_j = 1,103-2,063 at a = 100.
      - The table is regenerated after PS2's decision (21 Oct) and FS1's collision term.
      - The V1 rows get the 7B terms.
    - **(c) A remark that the setup proof's cost is forced for column-hash commitments.**
      - Certifying a radius e_j needs about β0 / -log2(1 - e_j/n_j) whole columns from any test, which is what ours opens.
      - Measured: the combined rows are the only part that FRI/STIR/WHIR/BaseFold could shrink. They are 45.2 MB of 2,833.5 MB (1.6%) at a = 100, and 0.4% at a = 1,000 (B-W3, B-Q2).
      - The scope is stated: column-hash commitments only.
    - **(d) Citations and positioning.**
      - Name one BCIKS20 version and its theorem number. The body says Thm. 1.7 with a "FOCS numbering" `\todo`; the D4 doc says 1.6.
      - State the Ligero radius used (d/4, or d/3 by Roth-Zémor).
      - State which version of Anchuri et al. the table uses, and that their two-server refereed variant lies outside Spot_d(q) (A-Q4, B-Q1, B-W6).
      - **B-W6:** position the integer-multiplicity logUp against deployed small-field logUp implementations.
    - **(e) Full proofs in the extended version.**
      - The proofs cover last-block pruning, the claim codec, the generation token rule and sampler, and V1 in Kpre.
      - They add the stateful-verifier caveat and the G4 acceptance-bound table.
      - They answer two of B's questions:
        - **B-Q5:** under Fiat-Shamir, V1's column check needs only that u_l is fixed before the columns are drawn. This is stated in the round-by-round argument.
        - **B-Q6:** the round-by-round bound uses the sum of the per-round errors. The text says whether the parameters use the tighter maximum.
      - A workflow drafts them in weeks 1-2, and the engineer reviews the drafts by Wed 28 Oct.
    - **(f) A team member checks each full proof by hand** (GenAI policy) [USER], on **Fri 30 Oct**.
    - **(g) One threat-model sentence on A-Q7:** who tokenises the prompt, and how provider-side system prompts are handled. Today `grep tokeni` and `grep "system prompt"` find nothing in the sections.
  - **Schedule.**
    - (b) and (d) are done by Sun 18.
    - (e) is drafted in weeks 1-2, so that (f) can run on Fri 30 and any code bug it exposes is fixed before Sun 1 Nov.
    - (a) and (c) start Mon 19 as a background workflow.
      - The workflow's verdict on (a) is a gate input on Fri 30.
      - The engineer's review lands by Tue 3 Nov.
      - The wording ("dichotomy", still in attack.tex, or "tight at the extremes") is therefore settled before the abstract freezes on Mon 9 Nov.
  - **Why.** B-W3, B-W5, B-W6, B-W7, B-Q5, B-Q6, A-Q7, chg 9. The submission cites the extended version 16 times. The CFP wants the proofs in the artifact at submission, and the artifact cannot be updated afterwards.
  - **Gain.** B 3 → 4 is plausible (this is a judgement). TH clears 11 `\todo`s.
  - **Effort / hardware.** 4.6 d of engineer time on the laptop:

    | Part | Days |
    |---|---|
    | (a) | 1 |
    | (b) | 0.5 |
    | (c) | 0.5 |
    | (d) | 0.5 |
    | (e) | 1.5 |
    | B-Q5, B-Q6, B-W6 | 0.5 |
    | (g) | 0.1 |
  - **Validation.**
    - An independent proof-check workflow, plus (f).
    - The table script is matched by H3.
    - 0 `\todo`s in the theory files.
  - **Kill.**
    - If (a) does not hold cleanly, keep "tight at the extremes" and drop the word "dichotomy" (chg 6(i)).
    - If writing (e) exposes a code bug, it is fixed before Sun 1 Nov, or the optimisation is switched off in the freeze.

- **EV [WIP] must: the deployment case, consistent platforms, non-interactive by default, statistics with spread.** It absorbs Plan 1 F7, the roofline of V-GPUFLOOR and the crossover of A11.
  - **Status (11 Oct).** Draft done (464a5fd): `ev_breakeven.py`, inputs with sources, formula tests, figure. Q* at 64 tokens 17.3 (FS, cnn16c), 22.3 (cnn18c); Kpre break-even link 0.59-0.70 Gb/s; V1 1.38 Gb/s today, 0.55 with V-V1N (projected). Open: tab:options literature cells and A7 column, the target-client paragraph, the final numbers from F6.
  - **What.**
    - **(a) A two-panel break-even figure from the pipeline.**
      1. **Client bytes against the number of queries Q:** downloading the int8 weights once (6.7 GB), against mode C.
         - The crossovers today are Q = 20.6 at 64 tokens interactive, and about 17.5 under Fiat-Shamir.
         - With PS2 (R64c, Fiat-Shamir) it becomes about 24.6. With C4b at B = 16 it would be about 72, but C4b is after 17 Nov, so that line is drawn as modelled.
         - At 2,048 tokens: 1.0 (v0) and 2.2 (V1).
      2. **Kpre client time plus transfer against re-execution, as a function of link speed.** The re-execution side is RX's strongest exact baseline, which is faster than fp32, so every break-even link below rises. Today's fp32 numbers are reference rows only.
         - Against fp32, the break-even link today is about 0.83 Gb/s for v0.
         - For V1, the draft's "about 0.45 Gb/s" assumed a V1 verifier at 1.6x v0 (about 27.5 s), measured before B5.
           - With today's projected V1 verifier of about 60 s, it is 22.4 Gb / (78.8 - 60) s ≈ 1.2 Gb/s, worse than v0.
           - With V-V1N (17-21 s, model) it would be about 0.36-0.39 Gb/s.
         - The figure uses the V1 re-time of Mon 12, which also replaces tab:options' "x1.6-1.9".
    - The roofline of the GPU verifier against fp16/int8 re-execution, and V1's bandwidth crossover, computed by a script from the L40S records.
    - The target client is named in the introduction: a CPU client of an open-weight model, plus a gateway or auditor for the GPU verifier.
    - **(b) Fill the "--" cells of tab:options from the literature.**
      - GPU TEE overhead: under 7%.
      - CommitLLM: 12-14% overhead, a key of about 48 MB for Llama 70B, and a statistical guarantee.
      - Add A7's provider-overhead column.
    - **(c)** Fiat-Shamir in every table (after FS1), or a label on each.
    - **(d) Statistics.**
      - Median [min-max] over ≥ 5 queries.
      - Wilson 95% intervals on every success count (40/40 → [91.2%, 100%]).
      - Bootstrap intervals on perplexity.
    - **(e)** One platform per table, checked by a test.
    - **(f) Draft by Thu 29 Oct (1.5 d).**
      - The break-even figure, the target-client paragraph and tab:options are built from 2080 Ti, Xeon and modelled numbers.
      - The draft feeds H5-mid and gate G1 criterion 6.
      - The final version (1 d, Thu-Fri 5-6 Nov) takes in the L40S/EPYC numbers.
  - **Why.** A-W1 and A-Q1 (one of A's three conditions), A-W6, B-W7, C-W8, chg 3, chg 11. A's deciding weakness is the deployment case, so the gate sees a draft of it.
  - **Gain.** A 2 → 3, together with F6. The honest result shows a narrow regime, and the figure states it.
  - **Effort / hardware.** 2.5 d (draft 1.5, final 1). Laptop.
  - **Validation.**
    - A test asserts that every plotted point equals a table cell.
    - The crossover formulas are unit-tested.
    - After F6, a grep check finds no value that disagrees between tables.

- **H4 [WIP] must: submission compliance.** Anonymous artifact and extended version, disclosure facts, HotCRP fields.
  - **Status (11 Oct).** `scripts/export_anonymous.py` (clean dry run: 4,107 files; again clean on 11 Oct at f45debd: 4,162 files), `scripts/artifact_README.md`, `docs/submission/HOTCRP_FIELDS.md` (de819fb, 9c4cfe2; write-up `docs/improvements/H4_submission_compliance.md`). Open: notices [USER], host and link. (D6a's certified digests are in the README since 12 Oct.)
  - **Snapshot dry run (12 Oct).** The README's quick tests run from an exported snapshot on the laptop: the first run failed on a missing model file (the export left out `code/artifacts/models`); fixed, plus an `.npz`-aware scan (random bytes in compressed weights had matched a short name; planted names in string/object/structured members are caught). Second run: 4,177 files, no identifying string, quick tests pass. H4 doc Sec. 4.1.
  - **What.**
    - **(a) A curated anonymous snapshot without git history** (Anonymous GitHub or GitFront, [USER] choice).
      - The history has 356 commits by named authors, 3 of them with mail.tau.ac.il addresses.
      - Scrub the 1,645 tracked result files that embed cluster home paths.
      - Leave out the plan, review and docs files that mention the team or the machines.
      - Add a README with reproduction commands per table, `main_ext.pdf` with 0 `\todo`s, and D6a's recipe and roots (the open-weight registry text of Plan 1 D6).
    - **(c) Disclosure facts.** Once the notices are sent, the conditional sentences in the introduction and Sec. 3.6 become facts, with the replies summarised.
    - **(d) Text for the mandatory HotCRP fields** "Ethics considerations" and "Generative AI usage".
      - The adversarial-review workflow of V1 is described accurately (chg 8).
      - Remove the anonymity leak "in our earlier evaluation" if it is still there.
  - **Why.** The CFP: anonymity, artifacts frozen at the deadline, proofs as artifacts, and disclosure by the rebuttal deadline. Non-compliance is grounds for desk rejection. Also A-W4, A-W7, chg 8, and Plan 1 H4 and G5.
  - **Effort / hardware.** 3 d. Laptop.
  - **Validation.**
    - The export script refuses to finish while any listed name, email address, host or path still matches.
    - A dry run of the README's quick tests on a clean machine, from the snapshot.
    - A dated log of the notices.

- **H5 [WIP] must: mock reviews, review checklist, page ledger and cut list.**
  - **Status (11 Oct).** H5a done (`paper_sp2027/REVIEW2.md`); the cut pass and defect fixes are running.
  - **Consistency audit (12 Oct).** A read-only workflow (six agents, one per section group) checked every number and "not yet" statement against the write-ups and the plan's latest statuses: 44 findings, each verified by hand against its source, all fixed (080b737; D6a doc 8e9b74c). Main build 18 pages, body ends p. 13, no undefined references; extended 51 pages, no overfull boxes. Worth repeating before H5-mid and H5b, since results keep landing.
  - **H5a, Wed 14 Oct:**
    - The three lenses of REVIEW.md, plus a compliance lens on the condensed draft: the page-13 boundary, appendix marking, anonymity, and the `\todo` count (71 + 34).
    - **Review checklist.** Each of REVIEW.md's 9 "Smaller issues", and each sub-item of changes 6 (i-vi) and 11, is confirmed closed or mapped to a fix with an owner item.
    - **Fonts are resolved; no fix is scheduled.**
      - The submission PDF embeds NimbusRomNo9L (a Times clone) for the body text. This was checked on 10 Oct by listing the PDF's fonts.
      - The `TU/ptm` warnings in the log come from IEEEtran's class load, before `\usepackage[T1]{fontenc}`.
      - H5a and H5b keep a font listing. The laptop has no `pdffonts`, so the listing uses PyMuPDF.
  - **Page ledger (H5a; applied by Sun 18 Oct).** Every new result has a destination (table below).
  - **Cut list, about 1.5 pages.**
    - The draft's 0.5 page, which makes room for the new figure:
      - move Thm. untrusted and its sketch to the appendix, keeping one paragraph with TH-c's remark;
      - drop the Proof-of-Learning row from tab:sok and recount the schemes;
      - compress the four CNN rows of tab:eval-compare to two.
    - About 1 more page, found at H5a now. One candidate is V1's body text, cut to about 1.25 pages as chg 11 asks. The compliance lens proposes the rest.
  - **H5-mid, Thu 29 Oct:** a mid mock review (background workflow) of the draft with EV's draft, weighted towards A's deployment lens. It feeds gate G1.
  - **Literature refresh, week 4 (0.5 d):** papers on verifiable inference from Oct-Nov 2026. New ones go into Related Work and tab:sok/tab:options as needed.
  - **H5b, Thu 12 Nov:** the same lenses on the draft with the final numbers, plus one external human reader who is not a PC member [USER].
  - **Effort.** 2.5 d: H5a with the checklist and ledger 0.75, H5-mid 0.25, literature refresh 0.5, H5b and fixes 1.
  - **Validation.**
    - 13 + 5 pages in Times (font listing).
    - 0 `\todo`s in both builds.
    - Every finding mapped to a fix or a rebuttal note.
    - Every ledger row placed.

**Page ledger.**

| New result | Item | Destination |
|---|---|---|
| Strongest re-execution baselines | RX | Body: rows in tab:eval-reexec, replacing the extrapolated rows (Smaller issues, bullet 3) |
| Break-even figure (two panels) | EV | Body: one column-wide figure in the deployment subsection |
| Provider overhead against unverified serving | A7 | Body: one column in tab:options plus one sentence. The `serve` table goes in the extended version. |
| Memory table for the 7B models at 2,048 tokens | F6 (F4) | Extended version. The body gets one sentence with the peaks (C-Q4). |
| Quality at 2,048 tokens | F2a, F2b | Body: replaces the 128-token rows of the quality table (same size) |
| Downstream accuracy | F2c | Body: one sentence (the mean drop with its interval). The table goes in the extended version. |
| Sampler TV distance | E2b | Body: one sentence in the generation section. The figure goes in the extended version. |
| Long-context rows | V-LONG-b | Body: one sentence. The cells and the fit go in the extended version. |
| Real-weight rows | F3a, F3b | Body: cells in the existing tables, labelled "real weights". Per-component bytes go in the extended version. |

### 3.8 Shapes and external baselines (could)

- **F8 [TODO] could:** add Llama-3-8B (GQA, 128k vocabulary) and Mistral-7B shapes to CONFIGS (C-W7).
  - The configs are written from the published numbers. The weights are random, so the shapes need no licence.
  - Add their byte and `cpuv` cells.
  - Effort: 1 d.
- **F5 [USER] could:** zkLLM on the same L40S. This moves A-W6 and C-W3 from a cross-hardware comparison to a matched one.
  - Effort: 1 d, plus the friend's approval.
  - It must pass its own smoke build before the freeze.
  - Otherwise the abstract states the hardware- and guarantee-matched comparison in words (chg 6(iii)).

### 3.9 Parked until after 17 Nov, and dropped

- **After 17 Nov (EuroS&P / USENIX track):**
  - PS1 (V1 windows for ReLU outputs, Sec. 3.5);
  - C4b (V1 batches, Sec. 3.4);
  - V-LONG-b's kernels past 2^15 and the measured 32k cells they enable;
  - V-STREAMCPU (a CPU verifier with bounded memory for full-model 32k prompts: 174 GB → about 5.6 GB at Llama-2-7B, 32k);
  - PS4 (context coder);
  - F2d (V1 with a few multipliers per op: per-row scales give x1.00-1.01 on OPT-125M);
  - E3 (verifier KV cache for multi-turn chat);
  - the V2/V3 implementation (PS5);
  - the PVC4 framed per-op codec;
  - A10 stage 2;
  - GKR rounds in lockstep;
  - C4c's implementation.
- **[DROP]:**
  - C5's per-op codec tweaks (measured: no room left);
  - G4 (scope rule K9);
  - V1 in mode K (Kpre dominates it);
  - the GPU-verifier targets as gate criteria (replaced by the roofline framing);
  - the course-report controls in `strong_gpu.sh must`, unless a table cites them.

---

## 4. Schedule (12 Oct to 17 Nov)

Every engineering block assumes one engineer. Job time runs in the background on the 2080 Ti and Xeon nodes and on the laptop. While GPU jobs run, the laptop carries the work that needs no GPU: D6a, E2b, G3-min on MNIST, and the builders for models up to 1.7B. Should items appear only in rank order, and only within their pool (Sec. 1).

| Week | Engineer: code | Engineer: writing | Jobs (2080 Ti / Xeon / laptop) | Team [USER] | Milestone |
|---|---|---|---|---|---|
| **1: Mon 12 - Sun 18 Oct** | Mon: **I5**; **FS1** part 1 (Kpre absorbs nothing, lands today); **PS2** byte tables. Tue-Wed: **FS1** parts 2-4 (BLAKE3/K12 digest against the tree mode; hash chosen and kill by Thu 15). Wed-Fri: **F2a** (lean path; `P_MAX` ablation on OPT-1.3B at 2,048). Thu-Fri: **RX** (exact int8-GEMM integer path, bf16, int8 engine). Thu: **D6a** test (laptop + Xeon). Sat-Sun: **F2b-Q** starts (laptop: Qwen3-0.6B beside TinyLlama-1.1B, then Qwen3-1.7B). | Mon: **TH-e** drafting workflow starts (background). Wed: **H5a** workflow (background) with the review checklist and the page ledger. Sun: H5a fixes, page ledger and about 1 more page of cuts; **TH-b** script; **TH-d** citations with B-W6; TH (g) A-Q7 sentence. | Mon: I5 checks (`/tmp`, `--mem`); **V-LONG-a** with today's code (interactive), then under Fiat-Shamir in Kpre once FS1 part 1 lands; **V1 re-time** (Llama-2-7B, 2-3 blocks, Kpre); PS2 2-block run. Thu-Sat: RX baselines on the Xeon (`--exclusive`); F2a on OPT-125M/1.3B, full test at 2,048 tokens, both `P_MAX`. | Mon 12: who sends the notices. Wed 14: Llama-2 licence; decisions on the attack scope (G4, Qwen repeat). Thu 15: notices, batch 1. | **Sun 18:** FS1 merged and every Fiat-Shamir suite green; full Llama-2-7B CPU verifier measured under Fiat-Shamir against RX's baselines; V-V1N decided by the re-time; OPT-1.3B quality at 2,048 tokens and the `P_MAX` ablation; D6a test result; first Qwen3/TinyLlama signal; page ledger fixed. |
| **2: Mon 19 - Sun 25 Oct** | **Mon 19: `P_MAX` and D6a's recipe fixed.** Mon-Wed: **F2b-Q** (Qwen3-4B builder; per-token exponent or fixed prefix if needed; the reported 2,048-token runs start once the `P_MAX` choice has landed). Wed 21: **PS2 decision**. Thu: **F3a** (`prune_last` in the real builders; OPT-1.3B full C/Kpre) with the `real` tier. Thu-Fri: **A7** harness and pilot with the `serve` tier; the `batch` (C4a) and `gen` tiers. **Fri 23: partial rehearsal.** Should slots, in rank order within the pre-freeze pool: **V-V1N** CPU parts 1-4 (from Mon 19 if the re-time confirmed the gap), **V-LONG-b** cells (start by Thu 22), **F2b-L** (kill Sun 25), **F3b**, **A6** (start by Sat 24). | Mon 19: **TH-a/TH-c** workflow starts (background). Sat-Sun: review of the TH-e drafts. Sun: **TH-b** re-run with the PS2 plan. | F2a: OPT-6.7B (HF_HOME on `/tmp`). F3a: OPT-6.7B Kpre (2080 Ti, or CPU prover on a Xeon node); OPT-6.7B real V1 bytes (Xeon). **G3.5-L** on a Xeon node (fp32, `--mem` ≥ 64 GB). Qwen3-4B quality runs. A7 pilot (HF fp16, our U1). | Mon 19: notices, batch 2; ask the friend for a smoke of at most 2 h between 19 and 27 Oct. **Fri 23:** L40S/EPYC window confirmed in writing (Mon 2 - Sun 8 Nov plus a backup), with exclusive use during the timing tiers, 24/7 operation, the EPYC's core count and RAM, `/tmp` for HF_HOME (40 GB), vLLM venv yes/no, zkLLM yes/no; the friend's own Llama-2 licence yes/no; anonymisation and artifact host. Fri 23 - Tue 27: the friend's smoke. | **Mon 19:** graph-changing decisions fixed. **Wed 21:** PS2 policy fixed. **Fri 23:** F2b-Q kill point; partial rehearsal passed. **Sun 25:** Qwen3-4B quality measured; OPT-1.3B real weights end to end; serving harness pilot ratios; F2b-L kill point. |
| **3: Mon 26 Oct - Sun 1 Nov** | **Mon 26: feature freeze.** Mon-Tue: **F6** package v2 completed + **H3** pipeline. Should slot: **V-V1N** exact by Wed 28. Wed-Thu: **rehearsal** on the 2080 Ti. Sat 31: rehearsal fixes only. **Sun 1 Nov 23:59: tag `sp2027-freeze`.** | Mon-Thu: **EV draft** (break-even figure, target-client paragraph, tab:options) from 2080 Ti, Xeon and modelled numbers. Wed 28: TH-e review complete. **Thu 29: H5-mid** (background) on the draft with EV. Fri 30: abstract draft 1, with numbers measured before the freeze only and the CPU ratio as "at least N times" (RX, V-LONG-a). | Rehearsal jobs (every tier's smallest cell). In the background, the work that does not depend on the platform: **F3b** (by Wed 28), **F2c**, Llama-2-7B quality (if F2b-L passed), **G3-min**, G3.5-L if not finished. | Mon 26: disclosure reminders. **Fri 30: gate G1 meeting; author list and conflicts fixed; TH-f** (a team member checks each full proof by hand). Sun 1: hand-over to the friend. | **Thu 29:** EV draft and H5-mid. **Fri 30:** gate G1 decided. **Sun 1:** frozen, rehearsed package handed over. |
| **4: Mon 2 - Sun 8 Nov: L40S run** | No code changes that affect reported numbers. Mon 2 - Tue 3: hotfix window for platform-only failures (`sp2027-freeze.1`, byte and verdict tests unchanged, new smoke). Daily: take in each finished tier through H3 and sanity-check it (about 0.25 d per day). | Mon-Tue: **TH-a**, **TH-c** review of the workflow drafts; the wording is fixed by Tue 3. Wed: **literature refresh** (0.5 d). Thu-Fri: **EV** final figure and tables. Sat-Sun: **H4** export and scrub script, README. Sun 8: abstract final draft. | Friend: smoke → `cpuv` → `v1` → `serve` → `strong_gpu must` (audited) → `gpuv` → `gen` → `batch` → `real` → `quality` (if any) → should tiers (`long`, vLLM, zkLLM). 2080 Ti/Xeon/laptop: F2c, E2b, G3-min continue. | Mon 2: disclosure reply cut-off; summary of replies. Line up the external reader. | **Sun 8:** ≥ 90% of the must cells in; the extended version's proofs complete. |
| **5: Mon 9 - Tue 17 Nov** | Mon 9 - Tue 10: re-runs of failed cells (hard stop Tue 10); tables regenerated. | Mon 9: abstract frozen (CPU ratio "at least N times", N the lower bound on both CPUs). Wed 11: text pass, 0 `\todo`s in both builds, platform-consistency test. Thu 12: **H5b**. Fri 13 - Sat 14: fixes, font listing (PyMuPDF), 13 + 5 pages, cut list if needed. Sun 15: artifact snapshot (grep gate), clean-machine test, final build. | none | **Tue 10:** register the abstract and HotCRP fields (title, abstract and authors as fixed). Thu 12: external reader. **Mon 16:** submit. | **Mon 16:** submitted. Tue 17 is buffer only. |

**L40S/EPYC run order and budget** (job-hours; the tiers run one after another so the timings stay clean):

| Order | Tier | Hours | Notes |
|---|---|---|---|
| 1 | `smoke` | 1 | Includes the FS1 and serving-harness tests. A platform-only failure on Mon 2 - Tue 3 triggers the hotfix rule. |
| 2 | `cpuv` (EPYC) | 14-19 | Llama-2-7B, Qwen3-4B and OPT-1.3B at 64 and 2,048 tokens, Fiat-Shamir. Every RX baseline runs on the same threads: exact integer with int8 GEMMs (VNNI), bf16 (AVX512_BF16), fp32, and the int8 engine labelled not exact. RX adds about 2-3 h. If the tier is done by Sun 8, the abstract's N is the smaller of the Xeon and EPYC ratios. |
| 3 | `v1` | 22-32 | lmax 27; ≥ 5 queries on the headline cells; plans chosen by PS2 |
| 4 | `serve` (A7) | 8-10 | U1, U2, prover C/Kpre under Fiat-Shamir, workloads (a) and (b) |
| 5 | `strong_gpu.sh must` (audited) | 20-30 | Table 3 "new" columns; Fiat-Shamir |
| 6 | `gpuv` | 5-7 | Plus fp16 and int8 re-execution on the same GPU, host-to-device bandwidth, memory |
| 7 | `gen` | 3-4 | Llama-2-7B and Qwen3-4B, P = 512, G ∈ {64, 256} |
| 8 | `batch` (C4a) | 2-3 | |
| 9 | `real` | 3-5 | Cells that did not fit the 2080 Ti; the setup-proof timing cell. The Llama-2 cells run only with the friend's own licence. |
| 10 | `quality` (reduced) | 0-2 | Llama-2-13B only, if F2b-L passed and the friend holds the licence |
| | **Must total** | **78-113** | 3.3-4.7 days of the 7-day window |
| 11+ | `long`, vLLM, zkLLM | 7-12 | Only if done or approved |

The `attack` tier (4-8 h) and OPT-6.7B's real V1 bytes moved to a Xeon node before the freeze (G3.5-L, F3a), which saves 6-12 L40S hours.

---

## 5. Risks and kill criteria

### 5.1 Gate G1 (Fri 30 Oct): go/no-go for S&P

**How the gate is weighed.**
- The S&P 2027 CFP's early-reject notification is Mon 18 Jan 2027, 8 days before USENIX Security cycle 2 (Tue 26 Jan).
- An early-rejected S&P paper can therefore still go to USENIX. A go mainly risks the one-year S&P block.
- The gate leans towards a degraded go.

**Go to S&P when all of these hold.** A single failure among criteria 1 and 3-6 is weighed at the meeting (see the no-go rule).
1. **FS1** is merged, and every Fiat-Shamir suite is green. Kpre v0 under Fiat-Shamir hashes nothing. Each party's hashing is inside its reported time.
2. **CPU verifier against the strongest exact baseline (RX, V-LONG-a).** The measured full-model Kpre CPU verifier under Fiat-Shamir (Llama-2-7B, 2,048 tokens, Xeon) is **at least 2x faster** than the strongest exact baseline on the same threads.
3. **OPT-6.7B runs with real weights end to end** in Kpre (accepted, tampers rejected), on the 2080 Ti or with the CPU prover on a Xeon node. OPT-1.3B runs in C and Kpre (F3a).
4. **The rehearsals of every must tier have passed** (partial on Fri 23, full on Wed-Thu 28-29), and the friend has **confirmed in writing** the window Mon 2 - Sun 8 Nov.
5. **Pages and root of trust.**
   - The submission build fits 13 + 5 pages in Times after H5a, and every new result has a place in the page ledger.
   - The D6a digest test has a result: the digests are equal on two machines, or the fallback root of trust is written.
6. **The EV draft (Thu 29 Oct) shows a regime in which the protocol wins.** It must show a non-empty regime of Q and link speed where mode C or Kpre beats downloading the weights and re-executing against RX's strongest exact baseline. H5-mid has read the draft.

**Degraded-go criterion.** A failure is reported as measured and is never a no-go.

7. **Quality (F2a).**
   - The criterion: smoothed OPT-1.3B and OPT-6.7B are measured at 2,048 tokens on the full WikiText-2 test set, with calibration on the train split and with confidence intervals, and are **within x1.10** of the fp reference. The cost of 8-bit probabilities is measured.
   - If it fails, the paper reports the measured ratios with their intervals and the attribution.

**A degraded go is still a go:**
- If criterion 7 fails, the quality numbers are reported as measured.
- If F2b-Q or F2b-L fails:
  - the cost tables lead with OPT-6.7B (and Qwen3, if measured);
  - the Llama rows are labelled "shape only, quality not measured".
- If V-V1N misses, V1's verifier is reported as measured, and V1 becomes a bytes-only option in EV and the abstract.
- If RX's int8-GEMM path is not exact, criterion 2 and the abstract use the smaller of the measured and modelled ratios.
- If only criterion 4 fails because the friend's window slips past Wed 4 Nov, there are two choices:
  - **(i) S&P with the 2080 Ti + Xeon as the single platform** for every cell that fits.
    - These cells are already rehearsed.
    - The cells that do not fit (Llama-2-7B V1 in mode C, Llama-2-13B at 2,048 tokens) are marked "does not fit in 11 GB".
  - **(ii) EuroS&P (2 Dec), with the L40S run shifted by at most about 1.5 weeks,** so that it finishes by about Wed 18 Nov.
    - A 3-week shift would put the run at 23-29 Nov, 3 days before 2 Dec, which is infeasible.

  The team decides on 30 Oct [USER]. The default is (i) if the window is lost entirely or would shift by more than about 1.5 weeks.

**Not a gate criterion: disclosure.** The CFP requires disclosure only by the rebuttal deadline (Tue 16 Feb 2027). The notices still go out on 15 and 19 Oct (Sec. 6).

**No-go:**
- If criterion 2 fails, or any two of criteria 1 and 3-6 fail, do not submit to S&P.
- Go to EuroS&P (2 Dec) if the gap is at most 3 weeks of work; otherwise go to USENIX (26 Jan 2027).
- The USENIX track adds PS1, C4b, PS5/V2, F2d, V-STREAMCPU, A10 stage 2 and the full L40S run in December.

### 5.2 Item kill criteria (summary)

| Item | Kill signal (date) | Action |
|---|---|---|
| FS1 | Chosen parallel hash < 3x faster than serial SHAKE-256 on the Xeon (Thu 15 Oct) | Keep part 1 (Kpre absorbs nothing) and the accounting fix; serial hash counted and reported. Not a gate criterion. |
| RX | int8-GEMM integer re-execution not bit-exact (Sun 18 Oct) | Modelled int8-GEMM row from the measured GEMM throughput; the gate and the abstract use the smaller ratio |
| F2a 11-bit | 8-bit cost ≤ 1% (decision Mon 19 Oct), or the exactness tests fail (Wed 21 Oct) | Keep 8 bits; report the cost |
| D6a | Digests differ and the fix is not in by Mon 19 Oct | The published int8 checkpoint and its hash become the root of trust; the threat model says so |
| PS2 | Bytes off by > 1.2%, or cnn18c setup > 1 h on the 2080 Ti node (Wed 21 Oct) | Keep cnn16c; lower-rate plans as modelled rows |
| F2b-Q | Qwen3-4B not within x1.10 (Fri 23 Oct) | OPT leads quality; Qwen rows labelled "shape only" |
| F2b-L | TinyLlama not within x1.15 (first result Sun 18; kill Sun 25 Oct) | Llama quality stated as a measured limitation |
| V-V1N | Not started if the Mon 12 re-time shows V1 within 1.2x of v0; not exact against `_final_ok` (Wed 28 Oct) | Off by default; V1 verifier reported as measured; V1 a bytes-only option in EV and the abstract |
| V-LONG-b | Cells not measured by Mon 9 Nov | Modelled rows, labelled |
| G3.5-L | No result by Mon 9 Nov | OPT-1.3B only; the limitation stated |
| A7-vLLM | No venv approval (Fri 30 Oct) | U1 + U2 plus the HF compile fallback |
| G3-min, E2b, F2c | No result by Mon 9 Nov | Delete the corresponding `\todo` sentences |
| Any should item | Not merged and rehearsed by Wed 28 Oct | Left out of the freeze (switch off). F2c, E2b and G3-min are exempt: they run after the freeze with the frozen builders. |
| Frozen code on the friend's platform | Smoke refused, or a platform-only failure on Mon 2 - Tue 3 Nov | Hotfix rule: platform-only fixes that leave the byte and verdict tests unchanged, re-tagged `sp2027-freeze.1`, then a new smoke |
| A tier in the run | Fails twice on the L40S | That cell comes from the 2080 Ti/Xeon with the platform labelled, or is dropped |
| Paper length | Body > 13 pages on Fri 13 Nov | Apply the H5 cut list |

### 5.3 Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | The friend's L40S window slips or is shorter, or co-tenant jobs disturb the timings | medium / high | Written confirmation by Fri 23 Oct with a backup window, exclusive use during the timing tiers and 24/7 operation; must tiers first; the package rehearsed on the 2080 Ti and smoked on the friend's machine; the abstract uses only numbers measured before the freeze; the 2080 Ti/Xeon single-platform fallback, or EuroS&P with a shift of at most about 1.5 weeks (5.1) |
| R2 | Integer Qwen3 or Llama quality fails under static per-tensor scales (massive activations at down_proj) | high / medium | The per-token exponent inside F2b-Q, and the fixed-prefix alternative; TinyLlama beside Qwen3-0.6B in week 1 (signal by Sun 18); early kills; the cost tables can lead with OPT-6.7B |
| R3 | FS1 invalidates Fiat-Shamir vectors and stored cells | certain / medium | Part 1 lands Mon 12 (Kpre) and part 2 by Thu 15 (mode C, V1), before any paper Fiat-Shamir number of that mode; all Fiat-Shamir suites re-run |
| R4 | 2080 Ti memory, node `/tmp` space and `--mem` unknown; 18 of 20 GB of quota used, against about 40 GB of checkpoints needed | high / medium | I5 on Mon 12: archive own old results, measure `/tmp`, pin model-heavy jobs to one node; F2a's lean path; HF_HOME on node-local `/tmp`, deleted at job end if it does not persist; outputs are small |
| R5 | Over-commitment: in engineer-days, the musts exceed the window before the freeze | high / medium | The unit is pessimistic (Plan 1's log); the job-hour budget is the binding plan; strict should ranking with last start dates; feature freeze on 26 Oct; switches off by default; TH and EV drafted by background workflows; F2c, E2b and G3-min use the post-freeze slack |
| R6 | A soundness regression from a protocol change (FS1 hash mode, V-V1N, `P_MAX`) | low / high | Exhaustive and tamper tests per item; an off-the-shelf digest preferred to a custom tree mode; the adversarial-review workflow re-run for V-V1N; the hash security paragraph; TH-e plus the team's hand check on Fri 30 Oct |
| R7 | Honest numbers are unfavourable: the CPU ratio against RX's strongest exact baseline may be about 3x on the Xeon and about 2x on the EPYC (model), not 4.6x; serving overhead of 9-40x at B = 1; mode C wins only below about 20-25 short queries; Kpre needs about 0.8 Gb/s against fp32 (more against RX); the GPU verifier is behind fp16 | high / medium | Report them as measured; gate at 2x; the abstract says "at least N times"; frame around the CPU client, proof on request, batching and the gateway/auditor; EV states the regime |
| R8 | The page limit is broken by the new figure and tables | medium / high (desk reject) | Page ledger and a cut list of about 1.5 pages, fixed at H5a; font listing at H5a and H5b (the body font is Times, NimbusRomNo9L); budget checked at H5a, H5-mid and H5b |
| R9 | Anonymity leak in the artifact or the PDF | medium / high | The export script refuses to finish while any listed name, email, host or path matches; clean-machine dry run; a team review of the snapshot |
| R10 | A disclosure reply arrives late or changes a verdict | low / medium | Reply cut-off 2 Nov; TensorCommitments' verdict already softened (chg 6(vi)); a later reply goes into the rebuttal, whose deadline (16 Feb 2027) is also the CFP's disclosure deadline |
| R11 | The abstract is frozen at registration (10 Nov) | certain / medium | Only numbers measured before the freeze; the CPU ratio only after RX and V-LONG-a, as "at least N times" with N a lower bound on both CPUs, so the EPYC run cannot contradict it |
| R12 | EPYC RAM too small for the 8k full-model cell (about 44 GB of claims, `--mem` about 200 GB) | medium / low | Ask the friend for the RAM and core count by Fri 23 Oct; otherwise 8k per block only |
| R13 | The laptop's 15.7 GB of RAM limits the builders to models up to 1.7B | certain / low | 4B/7B builds on the 2080 Ti nodes, whose host RAM is set by `--mem` |
| R14 | The branch `f2-quality` scripts that F2a reuses are marked unverified | medium / low | Port only `hybrid.py`'s softmax switch and the row-scale builder, with tests, before use |
| R15 | The frozen code first meets sm_89 (Triton, `torch._int_mm`), Zen 4 (`-march=native` kernels) and the friend's environment (CXX, HF_HOME, PVI_PY_INCLUDE) on Mon 2 Nov | medium / high | A smoke of at most 2 h between 19 and 27 Oct; otherwise the hotfix rule on Mon 2 - Tue 3 (`sp2027-freeze.1`) |
| R16 | The L40S Llama-2 cells lack a licence on the friend's machine | medium / low | The friend accepts the licence on their own account (Fri 23), or those cells are dropped or run with random weights; neither side uses the other's token |

---

## 6. [USER] actions, with dates

| Date | Action | Owner | Needed by |
|---|---|---|---|
| Mon 12 Oct | Decide who sends the disclosure notices (default in the drafts: Ofek from his TAU address, with Erel and Edo in CC), whether SPEX is notified, and whether Ambient appears anywhere (default: no) | team | G5, H4-c |
| Wed 14 Oct | Accept the Llama-2 licence on the **team's own** Hugging Face account (and Llama-3 only if F8 runs); never use another person's token | team | F2b-L, F3b |
| Wed 14 Oct | Confirm that G4 is dropped (scope rule K9), and say yes or no to a Qwen3-4B-Base in-range repeat on the 2080 Ti | team | G3-min |
| Thu 15 Oct | Send notices, batch 1: Anchuri et al., TensorCommitments | team | G5; CFP (by Tue 16 Feb 2027) |
| Mon 19 Oct | Send notices, batch 2: CommitLLM, NanoZK, SLP, SPEX | team | G5; CFP (by Tue 16 Feb 2027) |
| Mon 19 Oct | Ask the friend for a smoke of at most 2 h on the L40S/EPYC between 19 and 27 Oct (`smoke` plus `PVI_REHEARSAL=1` per tier; not the run itself) | team with the friend | F6 part 14, R15 |
| **Fri 23 Oct** | Book the friend's L40S + EPYC for **Mon 2 - Sun 8 Nov** (re-runs until Tue 10), plus a backup window. Ask for: exclusive use during the timing tiers (no co-tenant jobs on the GPU or the EPYC); 24/7 operation; the EPYC's core count and RAM (8k cells); about 40 GB for HF_HOME, readable by the compute nodes; and approval for a separate vLLM venv (about 5 GB, dummy weights). | team with the friend | F6, A7-vLLM, gate G1 criterion 4 |
| Fri 23 Oct | Ask whether the friend accepts the Llama-2 licence on their own Hugging Face account for the L40S Llama-2 cells (`quality` Llama-2-13B, `real` Llama-2-7B mode C). Otherwise those cells are dropped or run with random weights. | team with the friend | F6, F2b-L |
| Fri 23 Oct | Decide whether zkLLM runs on the friend's L40S (F5) | team with the friend | F5 |
| Fri 23 Oct | Anonymisation decisions (author block, the "Final Report" subtitle, the Contributions section) and the artifact host (Anonymous GitHub or GitFront) | team | H4 |
| Mon 26 Oct | Send the disclosure reminders | team | G5 |
| **Fri 30 Oct** | **Gate G1** go/no-go meeting (Sec. 5.1). If the L40S window has slipped, choose (i) the 2080 Ti/Xeon single platform or (ii) EuroS&P with a shift of at most about 1.5 weeks. | team and engineer | everything after it |
| Fri 30 Oct | Fix the author list and conflicts. The CFP confirms that the title, abstract and author list cannot change after 10 Nov. | team | H6 |
| Fri 30 Oct | A team member checks each full proof by hand (TH-f; GenAI policy) | team | TH |
| Sun 1 Nov | Receive the frozen tag and `L40S_RUN_SP2027.md`, run `smoke`, start the tiers in order | the friend | F6 |
| Mon 2 - Tue 3 Nov | Only if the smoke was refused or fails for a platform reason: receive `sp2027-freeze.1` and re-run `smoke` | the friend | F6 |
| Mon 2 Nov | Disclosure reply cut-off; send the engineer the replies for H4-c | team | H4 |
| Mon 9 Nov | Line up one external human reader (not a PC member) for Thu 12 - Fri 13 Nov | team | H5b |
| **Tue 10 Nov** | **Register the abstract:** title, abstract, authors and conflicts (as fixed on 30 Oct), and the mandatory "Ethics considerations" and "Generative AI usage" fields (texts from H4-d) | team | S&P |
| Sun 15 Nov | Approve the anonymous artifact snapshot before it goes public | team | H4 |
| **Mon 16 Nov** | **Submit** the PDF and the artifact link (deadline Tue 17 Nov) | team | S&P |

---

## 7. Changes from the draft after the critic's review

1. **RX added as a must** (Sec. 3.3, rank 3), measured in V-LONG-a and the `cpuv` tier (F6 part 5, run table).
   - Gate criterion 2 now requires at least 2x against the strongest exact baseline.
   - The abstract quotes the ratio only after RX and V-LONG-a, as "at least N times".
   - Also updated: the state on 10 Oct, the capacity, the job hours and R7.
2. **B-Q5, B-Q6 and B-W6 added to TH (d) and (e)**, +0.5 d (Sec. 3.7, ranked list, Sec. 1 goal B).
3. **A-Q7's threat-model sentence added as TH (g)**, 0.1 d.
4. **Review checklist added to H5a:** the 9 "Smaller issues" and the sub-items of changes 6 and 11 (Sec. 3.7).
5. **I5 added as an infrastructure must on Mon 12** (Sec. 3.1, rank 4, schedule week 1, R4).
6. **Llama-2 licence on the friend's machine:**
   - a [USER] row on Fri 23 (Sec. 6);
   - conditions on the `real` and `quality` tiers (F6, run table) and on F2b-L;
   - new risk R16.
7. **Fonts recorded as resolved; no fix is scheduled.**
   - NimbusRomNo9L is noted in the state and in H5.
   - `pdffonts` is replaced by a PyMuPDF font listing in H5, week 5 and R8.
8. **Page ledger added,** with a destination for each new result and about 1 more page of cuts at H5a (H5, gate criterion 5, R8).
9. **Author list and conflicts fixed on Fri 30 Oct** (Sec. 6, schedule week 3). A 0.5 d literature refresh is added in week 4 (H5, schedule).
10. **V1 re-timed with the B5 kernels on Mon 12** (V-LONG-a).
    - V-V1N's CPU parts become the first should item if the gap is confirmed; otherwise V1 is a bytes-only option.
    - EV's V1 break-even (about 1.2 Gb/s today) and tab:options' "x1.6-1.9" are corrected (EV, V-V1N, gate, kill table).
11. **F2b-Q's expectation qualified:** static per-tensor scales, and the x1.029 citation to be checked.
    - The per-token exponent moved in from F2b-L (now 1 d).
    - The PrefixQuant-style prefix added as an alternative.
    - TinyLlama runs beside Qwen3-0.6B in week 1.
    - Updated in F2b-Q, F2b-L, the schedule and R2.
12. **Quality criterion rewritten:** the full test set at 2,048 tokens with confidence intervals, within x1.10. It is now a degraded-go criterion (gate criterion 7, no-go rule).
13. **Capacity rewritten.**
    - It shows the pre-freeze over-commitment and why the engineer-day unit is pessimistic.
    - A pre-freeze job-hour budget is added (Sec. 1, R5).
14. **FS1 simplified.**
    - Part 1 lands on Mon 12 (0.25 d).
    - BLAKE3 or KangarooTwelve, absorbed as a digest, is evaluated against the tree mode, and the transcript stays SHAKE-256.
    - The 3x rule is FS1's own kill rule and no longer a gate criterion (FS1, gate criterion 1, kill table, R3, R6).
15. **A7's floor corrected** to about 1.8x at 2,048 tokens and about 2.6x at 64 tokens (A7).
16. **OPT-6.7B in-range attack moved to a Xeon node** as G3.5-L (must), together with OPT-6.7B's real V1 bytes (F3a).
    - The `attack` tier and those bytes leave the L40S run; the must total is now 78-113 h.
    - Updated in G3-min, Sec. 1, Sec. 2 and the run table.
17. **TH-e drafted by a workflow in weeks 1-2,** and TH-f moved to Fri 30 Oct (TH, schedule, Sec. 6).
18. **TH-a and TH-c start on Mon 19 as a background workflow.** Their wording is fixed by Tue 3 Nov, before the abstract freezes (TH, schedule).
19. **EV draft due Thu 29 Oct,** and H5-mid added. Both feed the gate (EV (f), H5, gate criterion 6, schedule week 3).
20. **Graph-changing decisions brought forward.** `P_MAX` and D6a's recipe are fixed on Mon 19 and PS2 on Wed 21, before any real-weight or byte cell (F2a, D6a, ranked list, kill table, schedule).
21. **Should items re-ranked exactly as given.**
    - PS1 and C4b moved to after 17 Nov.
    - V-LONG-b reduced to a 1 d measurement (2k-16k tokens); its kernel rewrite moves to after 17 Nov.
    - PS2 stays a must (Sec. 3.0, items, Sec. 3.9, gate, schedule).
22. **F6 built incrementally:** each tier lands with its item, and a partial rehearsal runs on Fri 23 Oct (F6 part 13, schedule, gate criterion 4).
23. **The friend's smoke and a hotfix rule added:**
    - a smoke of at most 2 h between 19 and 27 Oct;
    - the hotfix rule `sp2027-freeze.1` for Mon 2 - Tue 3 Nov.
    - Updated in F6 part 14, Sec. 1, the kill table, R15 and Sec. 6.
24. **Fri 23 booking extended:** exclusive use during the timing tiers, 24/7 operation, and the EPYC's core count and RAM (Sec. 6, R1).
25. **Fallback (ii) capped** at a shift of about 1.5 weeks; (i) is the default otherwise (gate).
26. **Gate reframed with the CFP.**
    - The early reject on 18 Jan 2027 comes before USENIX (26 Jan), so the gate leans towards a degraded go.
    - Disclosure leaves the gate (the CFP deadline is 16 Feb 2027), but the sending dates stay.
    - Updated in the header, the Sec. 1 dates, the gate, Sec. 2, Sec. 6 and R10.
27. **Abstract CPU ratio:** quoted only after RX and V-LONG-a, as "at least N times", with N a lower bound on both CPUs (Sec. 1 reason 3, `cpuv` row, R11, schedule).

**Consistency fix not raised by the critic.** If OPT-6.7B does not fit the 2080 Ti, its real-weight correctness run uses the CPU prover on a Xeon node. Gate criterion 3 can then be met before the L40S run (F3a).
