"""pvi -- Proofs of Verifiable Inference.

A faithful, self-contained reimplementation of the sampling-based proof-of-inference
protocol of Anchuri, Campanelli, Cesaretti, Gennaro, Jois, Kayman and Ozdemir,
*Towards Verifiable AI with Lightweight Cryptographic Proofs of Inference*
(SaTML 2026 / ePrint 2026/541), together with the attack and defence study built
on top of it.

Sub-packages:

``pvi.commitments``
    Vector commitments (Appendix B of the paper).  We provide the Merkle-tree
    instantiation used by the authors, in the row-wise optimised variant
    described in their Section 8.1.

``pvi.nn``
    The layered-DAG view of a neural network that the protocol needs: an explicit
    parent set ``G_j`` for every neuron, an execution trace ``trc`` holding one
    activation per neuron, and the local consistency relation
    ``a_j = phi(sum_{i in G_j} w_ij a_i)`` (Section 2).

``pvi.protocol``
    ``RandPathTest`` (Figure 3), the cryptographic protocol compiled from it
    (Figure 4), and the prover / verifier that realise Definition 4.

``pvi.attacks``, ``pvi.defences``, ``pvi.experiments``
    Our single-neuron / backdoor attacks, the alternative path samplers, and the
    exact acceptance probability of any trace under any sampler.

``pvi.fullcheck``
    Our defence (whole-layer Freivalds checks against a Reed--Solomon/Merkle weight
    commitment) on integer CNNs and transformers, plus the path test re-implemented
    on the same graphs as a baseline.  This part uses PyTorch and, optionally, a GPU.
"""

__all__ = ["attacks", "commitments", "defences", "experiments", "fullcheck", "nn", "protocol"]

__version__ = "1.0.0"
