# D1 + D2. Fiat-Shamir binds the whole model; a SHAKE-256 transcript

Plan items D1 and D2 (rigor gaps R1 and R2 of the trust report, `S/sp2027_research/idea_trust-rigor.md`).
Commit `2dad363` (branch `sp2027`). Status: done. **It changes every Fiat-Shamir transcript, so the `C:fs`
benchmark cells must be re-run** (interactive cells are unchanged).

## 1. D1: the statement now includes the computation

**Gap.** Under Fiat-Shamir the challenges must be derived from a hash of the **whole statement**. Ours
absorbed the security parameters, each weight op's shape and commitment (tag, root, codeword length), a
plan's trees, col-layout matrices and lookup tables, and the input `x`, but **not** the computation the
verifier recomputes: the graph's structure and the integer constants of the operations without weights
(requantisation multipliers, norm gains, look-up tables such as the exp and GELU/SiLU tables, the RoPE
setting, residual scales). A transcript was therefore not bound to the exact computation it checks: the
same challenges would be derived for two models that differ only in those constants. This is the
"weak Fiat-Shamir" pattern (Bernhard et al.; Dao et al., IEEE S&P 2023), which matters when proofs are
transferred or the verifier key is obtained separately.

**Fix.** `IntGraph.digest()` returns SHA-256 of a canonical description of the public graph:

* every op in order: its kind, name, inputs, output and note;
* a weight op's shape, layout, convolution and input bound (not the weights: the commitment binds them);
* a cheap op's function by its module and qualified name, and **every constant it captured**: default
  arguments, closure cells and `functools.partial` arguments, recursively; integers, floats, strings,
  devices and dtypes by value, tensors by dtype, shape and bytes, dataclasses by their fields;
* the input and output names and `meta` (for example `lm_positions` of a generation graph, E1).

`_absorb_statement` absorbs it first (label `graph`). The digest is computed once per Verifier (before a GPU
verifier copies the constants to its device; the copies have the same values and names). It does not
depend on the Python version (no bytecode is hashed), so a prover and a verifier on different machines
derive the same digest from the same graph.

Tested: two builds of the same model give the same digest, a different calibration (other multipliers)
and a generation graph give other digests, `graph.public()` has the same digest, and the transcript-label
tests check that `graph` is absorbed first with exactly this digest.

## 2. D2: the transcript in SHAKE-256

**Gap.** Challenge keys were SHA-256 of the running SHA-256 transcript followed by a label. Merkle-Damgard
hashes are not indifferentiable from a random oracle when inputs are not prefix-free, so the Fiat-Shamir
argument needed an extra prefix-freeness argument.

**Fix.** The running transcript is a SHAKE-256 sponge (domain `pvi/fullcheck/v3`); a challenge key is 32
bytes read from a copy of it after absorbing `challenge/<label>`. The challenge expansion was already
SHAKE-256. Merkle trees stay SHA-256 (they need only collision resistance, with domain separation).

## 3. What the paper says

The Fiat-Shamir paragraph of Section 3 can now state that the transcript binds the full verifier key (graph
digest, parameters, commitments) and the input, and that challenges are SHAKE-256 outputs on prefix-free
labelled transcripts, before the state-restoration argument.

## 4. What remains (plan D3)

The claim codec is not canonical (other widths encode the same claims differently), so a prover can
re-encode to re-randomise Fiat-Shamir challenges at about one hash per attempt. The 64 grinding bits cover
it formally; for a cleaner proof the decoder should recompute the encoder's deterministic plan and reject
any other.
