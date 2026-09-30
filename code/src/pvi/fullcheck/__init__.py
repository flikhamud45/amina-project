"""Our defence: check every weight layer at once, on CNNs and transformers.

Each weight layer's claimed pre-activations are checked with Freivalds' algorithm;
with committed weights the folded rows are checked against a Reed--Solomon/Merkle
commitment (Ligero-style).  Modules:

* ``field``      -- BabyBear arithmetic, NTT and Reed--Solomon encoding (torch);
* ``commitment`` -- Merkle trees and the Ligero-style weight commitment;
* ``graph``      -- integer computation graphs with exact GPU/CPU semantics;
* ``quantize``   -- float CNN -> int8 graph (per-channel weights, BN folded);
* ``transformer``-- integer GPT-2 / OPT / Llama / Qwen style decoders;
* ``protocol``   -- prover, verifier, security parameters (modes C, K, Kpre);
* ``sampling``   -- Anchuri et al.'s path test on the same integer graphs, for a
                    like-for-like baseline;
* ``analytic``   -- decoder weight-op shapes and the expected multiproof size;
* ``reference``  -- the straightforward code of the verifier's fast routines (tests only);
* ``models``, ``datasets`` -- the CNNs of the benchmark and their data.
"""

from .field import P
from .graph import CheapOp, IntGraph, MatOp
from .protocol import (
    Challenger,
    Prover,
    SecurityParams,
    Verifier,
    commit_graph,
    params_for,
    run_query,
    soundness_bits,
)

__all__ = [
    "P",
    "CheapOp",
    "IntGraph",
    "MatOp",
    "Challenger",
    "Prover",
    "SecurityParams",
    "Verifier",
    "commit_graph",
    "params_for",
    "run_query",
    "soundness_bits",
]
