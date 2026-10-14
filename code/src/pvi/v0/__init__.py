"""The base protocol (v0) of the paper's Section ``sec:protocol``.

The whole-network check: every weight product is verified, nothing is sampled.

Message flow for one query ``x`` (``mode="C"``, the verifier holds only the
32-byte commitment of each weight matrix -- the setting of Anchuri et al.):

1. prover -> verifier: the claimed pre-activations ``Z_l`` of every weight op.
2. verifier: recomputes every cheap op (requant, ReLU, pooling, residual adds,
   norms, softmax, attention, ...) from ``x`` and the claims, which gives it the
   input ``X_l`` of every weight op without trusting the prover for any of them
   (the paper's recomputation, (S1)).  It then draws ``r`` random rows ``chi_l`` per weight op.
3. prover -> verifier: ``u_l = chi_l^T [W_l | b_l]``.
4. verifier: checks ``chi_l^T Z_l == u_l^T [X_l ; 1]`` for every op and every
   column of ``Z_l`` (Freivalds' check, (S2)), then draws ``t`` column indices per op (under a
   commitment plan, per shared tree: see below).
5. prover -> verifier: the opened columns and their Merkle paths; the verifier
   checks each against its commitment and against ``Enc(u_l)`` (the column check, (S3)).

``mode="K"`` is the setting of SafetyNets, Slalom and Maverick, where the
verifier holds the weights: it computes ``u_l`` itself (fresh ``chi`` per query)
and steps 3 and 5 disappear.  ``mode="Kpre"`` additionally precomputes ``u_l``
for a secret ``chi`` once, so a query costs the verifier only step 2 and the two
inner products; the proof is just the claims (Slalom's and Maverick's
"preprocessing" mode).

Soundness (a false claim accepted), per weight op, is at most ``p**-r``
(Freivalds: a nonzero column of ``Z - A X`` survives a uniformly random row with
probability ``1/p``) plus, in mode C, ``prod_{i<t} (k-1-i)/(n-i) <= ((k-1)/n)**t`` (a wrong
``u`` survives ``t`` distinct columns of a Reed--Solomon code of distance ``n-k+1``), with a
union bound over ops (the paper's Theorem ``thm:sound``).  A commitment plan
(:mod:`pvi.commit.plans`, opt-in) keeps each op's budget and meets it with the op's exact ``t``,
trees shared by matrices of one codeword length and per-op codeword lengths -- and under its ``c``
policies commits some linear ops transposed (the col layout, alone or several reading one tensor in
one matrix).  A col-layout op sends no
``u`` and is not checked in step 4: step 2 also draws ``chi'`` over its claims' columns, the
verifier computes both sides of its code check itself (``w = chi' [X ; 1]^T`` and ``z = chi'
Z^T``), and step 5 opens columns of its transposed matrix, which must satisfy ``w . E'[:, c] ==
Enc(z)[c]``.  All trees' columns are drawn in step 4, so row and col matrices of one codeword
length share trees.  The ``c`` policies also commit every embedding table without a bias as a
lookup table (a Merkle tree over its rows): its claims are the looked-up rows, sent as int8 in step
1 with one multiproof over the distinct ids, and checked once derive has the ids, before any
challenge -- int8 and ids of the table (in derive), equal rows for equal ids
(``lookup_consistency``), the multiproof (``lookup_merkle``) -- with no ``u`` and no columns.  In
modes K and Kpre a verifier with ``lookups=True`` (opt-in) computes the embedding ops from its own
weights (the rows at the ids, plus the bias of an embedding that has one), and the prover sends no
claims for them.  Two preconditions make the *integer* claim, not just its residue, the thing
that is checked: claims are range-checked to ``|z| < 2**29`` (so two in-range integers with equal
residues are equal, since ``2 * 2**29 < p``), and every weight op is checked at commitment time to
have honest outputs inside that range (:func:`claim_bound`).  Completeness is exact: every
quantity is an integer and both parties compute cheap ops with the same integer rules.
"""

import importlib

_EXPORTS = {"Challenger": "transcript", "GraphCommitment": "graph_commitment", "Prover": "prover",
            "SecurityParams": "params", "Verifier": "verifier", "Z_BOUND": "params", "claim_bound": "params",
            "commit_graph": "graph_commitment", "params_for": "params", "run_query": "query",
            "soundness_bits": "params"}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """The public API, imported from its module on first use (so importing one submodule, e.g. the codec, does not
    import the others)."""
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"pvi.v0.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module 'pvi.v0' has no attribute {name!r}")
