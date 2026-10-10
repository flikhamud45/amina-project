# HotCRP fields for IEEE S&P 2027, cycle 2 (Plan 2, H4 (d))

Both fields are mandatory at registration (10 Nov 2026). The texts below start from the drafts in
`paper_sp2027/REVIEW2.md` (Section 4), with the facts the repository records filled in. **[TEAM]** marks what only the
team can confirm or supply; nothing here has been submitted.

## Generative AI usage considerations

> We used large language models, run as coding and writing agents (Anthropic's Claude), in four ways.
> (1) Implementation support for the protocol code and the experiment scripts. Every optimised code path was validated
> against the path it replaces on the same queries (identical claims, proofs and verdicts), and the code base has about
> 1,100 automated tests, which include tamper tests for every check of the protocol.
> (2) Drafting, condensing and copy-editing the text of the paper and of its extended version; the authors revised all
> text and are responsible for it.
> (3) Drafting proofs for the extended version; **[TEAM: author role]** checked each proof by hand on **[TEAM: date]**.
> (4) Adversarial review: agent workflows tried to break the V1 verifier (four reviewers, each finding re-checked by an
> independent agent; five findings were confirmed and fixed, the most severe of medium severity, on crafted graphs, with
> no forgery against our models) and tried to find errors in the paper's statements and numbers.
> No measured number in the paper was produced by a model: every number comes from logged runs of our code, whose
> records are in the artifact. **[TEAM: confirm this sentence.]**

Sources: the test count is the Xeon run of 10 Oct 2026 (job 1006979: 1,095 passed; the one failure was fixed the same
day); the V1 review is `docs/improvements/C2_v1_proof_size.md`, Section 9; the proof drafts are
`paper_sp2027/drafts/th_*` with their check reports.

## Ethics considerations

> Our attack targets published verification protocols, not deployed services. All experiments used our own
> implementations, our own hardware, public datasets and open-weight models; no live system and no human subject was
> involved. On **[TEAM: dates]** we wrote to the authors of the path protocol, TensorCommitments, CommitLLM, NanoZK, SLP
> **[and SPEX]**, describing the findings, their assumptions and what we do not claim, and offering our draft and code.
> **[TEAM: replies, or "No reply by 2 Nov 2026."]** We do not claim that schemes meeting their stated probabilistic
> guarantees are broken. We release code for the edit against the path protocol on public models, but no method for
> steering the responses of instruction-tuned assistants. The defence has no harmful use that we are aware of.

The notices themselves are drafted in `sp2027_research/attack_theory/disclosure.md` (G5); sending them, from the team's
own addresses, is the team's step. The CFP requires disclosure no later than the rebuttal deadline (16 Feb 2027).

## Checklist before registration

- [ ] Notices sent; dates and replies filled in above and in `intro.tex` / `attack.tex` (H4 (c)).
- [ ] Proof hand-check done (TH-f); role and date filled in.
- [ ] Anonymous snapshot exported with `scripts/export_anonymous.py` (clean scan), link added to the `fullversion`
      entry of `extra.bib`, extended version with 0 `\todo`s.
- [ ] Title, abstract and author list final (frozen at registration).
