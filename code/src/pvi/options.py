"""Implementation options: which of several equivalent code paths computes a step.

Every option chooses between implementations that give the same integers, bytes and verdicts (the tests compare
them); only time and memory change.  The defaults are the paths of the measured results.  The options are read
when a step runs, so they can be changed at any point::

    from pvi import options
    options.configure(native_gemm=False)              # for the rest of the process
    with options.using(fused_attention=False):        # for one block
        ...
    options.current().native_gemm

Scripts expose every option as a command-line flag (:func:`add_arguments`, :func:`apply_arguments`) and store
:func:`describe` in their records: the options in force and which accelerated paths this host can run, so that a
fallback (no C++ compiler, no Triton, a GPU whose int8 GEMMs fail their self-check) shows in the record.  With
``--require-native`` or ``--require-triton`` a script stops instead of falling back.

Three settings of the host are environment variables, because they name places on the machine rather than choose
a path: ``CXX`` (the C++ compiler of the native kernels; default ``g++``), ``PVI_NATIVE_DIR`` (where the compiled
library is kept; default ``~/.cache/pvi``) and ``PVI_PY_INCLUDE`` (a folder with the Python headers, for hosts where
Triton cannot find them).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

__all__ = ["Options", "current", "configure", "using", "add_arguments", "apply_arguments", "describe",
           "native_dir", "python_include", "V1_VERIFIERS", "GKR_MEMORY"]

V1_VERIFIERS = ("optimised", "baseline")
GKR_MEMORY = ("auto", "low", "host", "default")


def _option(default, help: str):
    return field(default=default, metadata={"help": help})


@dataclass(frozen=True)
class Options:
    """The implementation options (module docstring); every default is the measured path."""

    native: bool = _option(True, "the native C++/OpenMP kernels of a CPU verifier (Linux with a C++ compiler); "
                                 "off: torch everywhere")
    native_avx512: bool = _option(True, "the native attention's AVX-512 kernel where the CPU has AVX-512")
    native_cheap: bool = _option(True, "the native kernels of the cheap operations (requantisation, residual, table "
                                       "lookups, norms, RoPE)")
    native_decode: bool = _option(True, "the native decoder of the compact claim encoding")
    native_gemm: bool = _option(True, "the native int8 GEMM of the CPU verifier and of integer re-execution")
    int32_claims: bool = _option(True, "a CPU verifier keeps its claims as int32 (needs --native-cheap)")
    malloc_tune: bool = _option(True, "a CPU verifier keeps freed memory in the process (glibc) instead of mapping "
                                      "large blocks afresh")
    fused_attention: bool = _option(True, "the fused exact attention kernel (Triton) on a GPU, in the prover and the "
                                          "GPU verifier")
    fused_encoder: bool = _option(False, "the fused claim encoder (Triton) on a GPU: the same bytes, slower than the "
                                         "threaded encoder")
    gpu_decode: bool = _option(True, "a GPU verifier decodes the claims on its GPU (Triton); off: on the host")
    int8_fold: bool = _option(True, "mode Kpre's precomputation by int8 GEMMs (GPU) or the native field product "
                                    "(CPU); off: float64 limb products")
    int8_opening: bool = _option(True, "the prover's folds and column openings by int8 GEMMs on a GPU; off: float64 "
                                       "limb products")
    int8_forward: bool = _option(True, "the forward pass's weight products by int8 GEMMs on a GPU; off: exact float32 "
                                       "GEMMs in chunks")
    hash_threads: int = _option(0, "threads of the Fiat-Shamir tree absorption (0: torch's thread count)")
    merkle_processes: bool = _option(False, "a multi-threaded verifier checks large multiproofs in worker processes")
    v1_verifier: str = _option("optimised", "V1's verifier: optimised (window-table products, batched GKR check, "
                                            "batch-inverted logUp identity) or baseline (float64 limb products, one "
                                            "check per instance)")
    v1_native: bool = _option(True, "the native kernels of V1's window-table products on a CPU")
    gkr_triton: bool = _option(True, "V1's GKR prover in Triton on a GPU; off: the torch prover")
    gkr_profile: bool = _option(False, "synchronise and time each phase of the Triton GKR prover")
    gkr_memory: str = _option("auto", "V1's GPU prover memory during the GKR: low (free the sent values and the "
                                      "claims no fold reads before it, and each Triton tree's leaf numerators and "
                                      "layers once read), host (low, and the claims the folds read wait on the host "
                                      "during the GKR), default (keep them: the measured path), or auto (the first "
                                      "of default, low and host whose predicted peak is within "
                                      "--gkr-memory-budget)")
    gkr_memory_budget: int = _option(90, "with --gkr-memory auto: the percentage of the GPU's memory the default "
                                         "path's predicted GKR peak may take before the low-memory path is used")

    def __post_init__(self):
        if self.v1_verifier not in V1_VERIFIERS:
            raise ValueError(f"v1_verifier: one of {V1_VERIFIERS}, not {self.v1_verifier!r}")
        if self.gkr_memory not in GKR_MEMORY:
            raise ValueError(f"gkr_memory: one of {GKR_MEMORY}, not {self.gkr_memory!r}")
        if not 1 <= self.gkr_memory_budget <= 100:
            raise ValueError("gkr_memory_budget: a percentage in 1 .. 100")
        if self.hash_threads < 0:
            raise ValueError("hash_threads: 0 (torch's thread count) or a positive number")


_CURRENT = Options()


def current() -> Options:
    """The options in force."""
    return _CURRENT


def configure(options: Options | None = None, **changes) -> Options:
    """Set the options for the rest of the process: ``options`` (default: the current ones) with ``changes``."""
    global _CURRENT
    _CURRENT = replace(options if options is not None else _CURRENT, **changes)
    return _CURRENT


@contextlib.contextmanager
def using(**changes):
    """The current options with ``changes`` for the block, then the previous ones."""
    global _CURRENT
    saved = _CURRENT
    _CURRENT = replace(saved, **changes)
    try:
        yield _CURRENT
    finally:
        _CURRENT = saved


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per option (``--native/--no-native``, ``--hash-threads N``, ``--v1-verifier optimised|baseline``,
    ...), defaulting to the options in force, and ``--require-native`` / ``--require-triton``."""
    group = parser.add_argument_group("implementation options (the same results; only time and memory change)")
    for f in fields(Options):
        flag, default, text = "--" + f.name.replace("_", "-"), getattr(_CURRENT, f.name), f.metadata["help"]
        if isinstance(default, bool):
            group.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=text)
        elif f.name == "v1_verifier":
            group.add_argument(flag, choices=V1_VERIFIERS, default=default, help=text)
        elif f.name == "gkr_memory":
            group.add_argument(flag, choices=GKR_MEMORY, default=default, help=text)
        else:
            group.add_argument(flag, type=int, default=default, help=text)
    group.add_argument("--require-native", action="store_true", help="stop unless the native CPU kernels are available")
    group.add_argument("--require-triton", action="store_true", help="stop unless Triton runs on the GPU")


def apply_arguments(args: argparse.Namespace) -> Options:
    """Configure the options from :func:`add_arguments`'s flags, then check ``--require-native/--require-triton``."""
    opts = configure(Options(**{f.name: getattr(args, f.name) for f in fields(Options)}))
    if args.require_native:
        from pvi.kernels import native
        if not native.available():
            raise SystemExit("--require-native: the native CPU kernels are not available here (they need Linux, a C++ "
                             "compiler in CXX or on the PATH, and --native)")
    if args.require_triton:
        import torch

        from pvi.kernels import codec_triton
        if not (torch.cuda.is_available() and codec_triton.available("cuda")):
            raise SystemExit("--require-triton: Triton does not run on a GPU here")
    return opts


def describe(device=None) -> dict:
    """For a run record: the options in force and what this host runs (the native library, its AVX-512 kernel,
    Triton, and on a GPU ``device`` whether its int8 GEMMs passed their self-check)."""
    import torch

    from pvi.kernels import codec_triton, native
    out = {"options": dataclasses.asdict(_CURRENT), "native": native.available(), "avx512": native.has_avx512(),
           "triton": codec_triton.triton is not None}
    if device is not None and torch.device(device).type == "cuda":
        from pvi.core.field import int8_ok
        out["int8_gemm"] = int8_ok(device)
    return out


def native_dir() -> Path:
    """The folder of the compiled native library: ``$PVI_NATIVE_DIR``, else ``~/.cache/pvi``."""
    return Path(os.environ.get("PVI_NATIVE_DIR") or Path.home() / ".cache" / "pvi")


def python_include() -> str | None:
    """``$PVI_PY_INCLUDE``: a folder with the Python headers for Triton's launcher, or None."""
    return os.environ.get("PVI_PY_INCLUDE") or None
