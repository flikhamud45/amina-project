"""pvi: proofs of inference, the attack on spot-checking verifiers and a fully sound defence.

Sub-packages:

``pvi.core``
    The prime field and its degree-8 extension, and integer computation graphs: weight operations, cheap operations
    as registered kinds with explicit constants, exact evaluation and the graph digest; the contention record of a
    timed phase and pinned host memory.

``pvi.commit``
    Weight commitments (Merkle trees, Reed--Solomon column commitments, lookup tables), commitment plans, the setup
    proof of an untrusted commitment, the prover's setup cache and the byte model of a proof.

``pvi.v0``
    Protocol v0: security parameters, the Fiat--Shamir transcript, the commitment of a graph's weights, the prover,
    the verifier in modes C, K and Kpre (and its streaming form), the compact wire codec and the exact integer token
    sampler.

``pvi.v1``
    The V1 variant: requantised int8 values instead of claims, proved by windowed logUp and a fractional-sum GKR
    over the extension field.

``pvi.models``
    Integer models: random-weight decoders with the benchmark shapes, integer OPT and Llama/Qwen3 decoders from
    Hugging Face checkpoints, image models and their int8 quantiser.

``pvi.kernels``
    Exact native CPU kernels (C++/OpenMP) and Triton GPU kernels (attention, the claim encoder); each computes the
    same integers as the torch code it replaces.

``pvi.baseline``
    The spot-checking verifiers the attack targets: the path protocol of Anchuri et al. (float and integer
    implementations), value-aware samplers and the exact acceptance analysis.

``pvi.attacks``, ``pvi.mnist``
    The single-neuron and backdoor attacks, the reproduced forgeries of Anchuri et al., and the MNIST networks they
    are measured on.

``pvi.options``
    The implementation options: which of several equivalent code paths (native or torch kernels, fused GPU kernels,
    V1's optimised or baseline verifier) computes a step; every path gives the same integers, bytes and verdicts,
    and every script exposes the options as command-line flags.
"""

__all__ = ["attacks", "baseline", "commit", "core", "kernels", "mnist", "models", "options", "v0", "v1"]

__version__ = "2.0.0"
