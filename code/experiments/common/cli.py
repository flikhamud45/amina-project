"""Command-line flags shared by the experiment scripts.

The protocol flags default to the configuration of the paper's measured results (:data:`PAPER`): mode C's
commitment plan ``cnn18c``, Fiat--Shamir, the compact wire encoding, the lean prover, the last decoder block
pruned to the last position, and the verifier reading the embedding rows itself (lookups), at ``lambda = 128`` and
Reed--Solomon rate 4.  Each flag turns one of them off or changes it, so that every other configuration the paper
reports (interactive challenges, the per-op commitment, ...) is one flag away.  The runtime flags choose the
devices and threads and every implementation option (:mod:`pvi.options`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from pvi import options

__all__ = ["Protocol", "PAPER", "RATE", "CALIB_TOKENS", "add_protocol_arguments", "add_runtime_arguments",
           "apply_runtime", "protocol_of"]

RATE = 4
"""The Reed--Solomon rate (codeword length over row length) of the weight commitment."""
CALIB_TOKENS = 32
"""Calibration tokens of a random-weight decoder (``build_decoder(calib_tokens=...)``; fewer for shorter prompts)."""


@dataclass(frozen=True)
class Protocol:
    """The protocol configuration of a measured cell."""

    policy: str = "cnn18c"
    """Mode C's commitment plan (:func:`pvi.commit.plans.plan_commitment`)."""
    fiat_shamir: bool = True
    wire: bool = True
    """The proof in the compact encoding (``run_query(wire=True)``)."""
    lean: bool = True
    """The lean prover (``Prover(lean=True)``)."""
    prune_last: bool = True
    """Decoders: the last block computed at the last position only (``build_decoder(prune_last=True)``)."""
    lookups: bool = True
    """Modes K and Kpre: the verifier reads the embedding rows itself (``Verifier(lookups=True)``)."""
    lam: int = 128
    """The security level in bits."""
    rate: int = RATE


PAPER = Protocol()
"""The configuration of the paper's measured results."""


def add_protocol_arguments(parser: argparse.ArgumentParser, *, variants: bool = False,
                           defaults: Protocol = PAPER) -> None:
    """The protocol flags, defaulting to ``defaults``.  With ``variants`` the plan and the on/off settings take
    lists (``--policy cnn18c per-op``, ``--wire on off``), for scripts that time several variants side by side."""
    g = parser.add_argument_group("protocol (defaults: the paper's measured configuration)")
    if variants:
        g.add_argument("--policy", nargs="+", default=[defaults.policy],
                       help="mode C's commitment plans (per-op, auto, tight, cnn<e>, R<rate>, each with the suffix c "
                            "for the col layouts and lookup tables), timed side by side")
        for name, text in (("wire", "the compact wire encoding of the proof"), ("lean", "the lean prover"),
                           ("prune_last", "decoders: the last block at the last position only"),
                           ("lookups", "modes K and Kpre: the verifier reads the embedding rows itself")):
            g.add_argument("--" + name.replace("_", "-"), nargs="+", choices=["on", "off"],
                           default=["on" if getattr(defaults, name) else "off"], help=text + " (on, off, or both)")
    else:
        g.add_argument("--policy", default=defaults.policy,
                       help="mode C's commitment plan (per-op, auto, tight, cnn<e>, R<rate>, each with the suffix c "
                            "for the col layouts and lookup tables)")
        g.add_argument("--wire", action=argparse.BooleanOptionalAction, default=defaults.wire,
                       help="the compact wire encoding of the proof")
        g.add_argument("--lean", action=argparse.BooleanOptionalAction, default=defaults.lean,
                       help="the lean prover")
        g.add_argument("--prune-last", action=argparse.BooleanOptionalAction, default=defaults.prune_last,
                       help="decoders: the last block at the last position only")
        g.add_argument("--lookups", action=argparse.BooleanOptionalAction, default=defaults.lookups,
                       help="modes K and Kpre: the verifier reads the embedding rows itself")
    g.add_argument("--fs", action=argparse.BooleanOptionalAction, default=defaults.fiat_shamir,
                   help="Fiat-Shamir challenges (off: interactive)")
    g.add_argument("--lam", type=int, default=defaults.lam, help="security level in bits")
    g.add_argument("--rate", type=int, default=defaults.rate, help="Reed-Solomon rate of the weight commitment")
    g.add_argument("--setup-cache", default=None, metavar="DIR",
                   help="the prover's setup cache (pvi.commit.setup_cache): commitments are rebuilt from stored leaf "
                        "digests (the same roots and proofs); default none")


def protocol_of(args: argparse.Namespace) -> Protocol:
    """The :class:`Protocol` of single-valued protocol flags."""
    return Protocol(policy=args.policy, fiat_shamir=args.fs, wire=args.wire, lean=args.lean,
                    prune_last=args.prune_last, lookups=args.lookups, lam=args.lam, rate=args.rate)


def add_runtime_arguments(parser: argparse.ArgumentParser, *, threads: int = 8, verifier_device: bool = True) -> None:
    """``--device`` (the prover's; default the GPU if there is one), ``--verifier-device`` (default the CPU),
    ``--threads`` and every implementation option (:func:`pvi.options.add_arguments`)."""
    g = parser.add_argument_group("runtime")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="the prover's device")
    if verifier_device:
        g.add_argument("--verifier-device", default="cpu", help="the verifier's device (cpu or cuda)")
    g.add_argument("--threads", type=int, default=threads, help="torch's intra-op threads (and the native kernels')")
    options.add_arguments(parser)


def apply_runtime(args: argparse.Namespace) -> None:
    """Set the threads and the implementation options of :func:`add_runtime_arguments`."""
    torch.set_num_threads(args.threads)
    options.apply_arguments(args)
