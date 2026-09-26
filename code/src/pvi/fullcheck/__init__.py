"""The whole-network defence, generalised to CNNs and transformers.

``protocol.batched`` (the Revision-1 prototype) checks MLPs with a 26-bit field
in numpy.  This package is the version the comparison with the literature is
run on:

* ``field``      -- BabyBear arithmetic, NTT and Reed--Solomon encoding (torch);
* ``commitment`` -- Merkle trees and the Ligero-style weight commitment;
* ``graph``      -- integer computation graphs with exact GPU/CPU semantics;
* ``quantize``   -- float CNN -> int8 graph (per-channel weights, BN folded);
* ``transformer``-- integer GPT-2 / OPT / Llama / Qwen style decoders;
* ``protocol``   -- prover, verifier, security parameters (modes C, K, Kpre);
* ``sampling``   -- Anchuri et al.'s path test on the same integer graphs, for a
                    like-for-like baseline;
* ``bench``      -- the benchmark runner that writes raw per-trial records.
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
