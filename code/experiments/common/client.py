"""The CPU client in a process of its own: what it holds and its resident memory.

The timing scripts run the prover and the verifier in one process, so a record's peak resident memory is theirs
together. :class:`ClientProcess` starts a separate process (``multiprocessing``, spawn: a fresh interpreter) that
holds only what the client holds -- the public graph and the parameters; in mode C the commitment's roots (and a
plan's trees and lookup tables), in mode Kpre its secret key (the vectors ``chi`` and the folded rows ``u``) and, with
``lookups``, the embedding rows it reads itself, in mode K the weights -- and checks each proof it is given from the
prover's messages alone (:func:`pvi.v0.query.verify_wire`), with the verifier's implementation options and threads.
The key and each proof reach it through files in a temporary folder, each message read in one piece, so that the
process holds every message once, as a client that received it.

Its peak resident set (``VmHWM``) is the client's peak memory; its resident set after the imports (Python, PyTorch,
the library and its native kernels) and after loading the key are recorded with it.  On a host without ``/proc``
(not Linux) the three are ``None``.  The process inherits the parent's CPU affinity (``run.sh`` pins the CPU-timing
jobs) and runs while the parent waits, so that it never shares the CPUs with a timed query.

    with ClientProcess(verifier, folder=None) as client:      # after Verifier(...) and, in Kpre, precompute()
        got = client.verify(x, run_query(prover, verifier, x, wire=True, keep_proof=True)["proof"])
        memory = client.close()       # {"peak_rss", "baseline_rss", "key_rss"} in bytes
"""
from __future__ import annotations

import multiprocessing
import shutil
import tempfile
import time
from pathlib import Path

import torch

__all__ = ["ClientProcess", "client_key", "resident"]

KEY = "key.pt"
MESSAGES = ("claims", "rows", "u", "columns")
"""The byte messages of a :class:`pvi.v0.query.WireProof`, each written to a file of its own."""


def resident() -> dict:
    """This process's resident set (``rss``, ``VmRSS``) and its peak since the process started (``peak``,
    ``VmHWM``), in bytes; ``None`` where ``/proc`` has none."""
    out = {"rss": None, "peak": None}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    out["rss"] = 1024 * int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    out["peak"] = 1024 * int(line.split()[1])
    except (OSError, ValueError):
        pass
    return out


def _alone(t: torch.Tensor | None) -> torch.Tensor | None:
    """``t`` with a storage of its own (a copy of a view into a larger one), so that saving it saves ``t`` alone."""
    if t is None or t.untyped_storage().nbytes() == t.numel() * t.element_size():
        return t
    return t.clone()


def client_key(verifier) -> dict:
    """What the client of ``verifier`` holds between queries: the public graph and the parameters, the
    commitment's public parts (mode C), the precomputed ``(chi, u)`` of every op (mode Kpre), and the weights it
    reads: every op's in mode K, the embedding ops' with ``lookups`` (none otherwise)."""
    own = {op.name for op in verifier._own_rows()}
    weights = {k: tuple(map(_alone, w)) for k, w in verifier.weights.items() if verifier.mode == "K" or k in own}
    pre = {k: tuple(map(_alone, p)) for k, p in verifier._pre.items()} if verifier.mode == "Kpre" else {}
    return {"graph": verifier.graph, "params": verifier.params, "mode": verifier.mode, "publics": verifier.publics,
            "groups": verifier.groups, "tables": verifier.tables, "weights": weights, "pre": pre,
            "lean": verifier.lean, "lookups": verifier.lookups}


def _serve(conn, folder: str, opts, threads: int) -> None:
    """The client process: imports, loads the key, then verifies each proof it is sent until ``close``."""
    try:
        from pvi import options
        from pvi.kernels import native
        from pvi.v0.query import WireProof, verify_wire
        from pvi.v0.verifier import Verifier

        options.configure(opts)
        torch.set_num_threads(threads)
        native.available()                             # the native library loaded, as a verifier's first use loads it
        base = resident()
        key = torch.load(Path(folder) / KEY, weights_only=False)
        verifier = Verifier(key["graph"], key["params"], key["mode"], publics=key["publics"], weights=key["weights"],
                            lean=key["lean"], groups=key["groups"], tables=key["tables"], lookups=key["lookups"])
        verifier._pre = key["pre"]
        del key
        conn.send({"baseline_rss": base["rss"], "key_rss": resident()["rss"]})
        while True:
            msg = conn.recv()
            if msg["op"] == "close":
                conn.send({"peak_rss": resident()["peak"]})
                return
            proof = WireProof(**{name: (Path(folder) / f).read_bytes() for name, f in msg["files"].items()},
                              lookups=msg["lookups"], paths=msg["paths"])
            t0 = time.perf_counter()
            res = verify_wire(verifier, msg["x"], proof)
            seconds = time.perf_counter() - t0
            del proof
            verify = sum(v for k, v in res["timings"].items() if k.startswith("verify_") or k == "fs_hash")
            conn.send({"accepted": res["accepted"], "rejected_at": res["rejected_at"], "verify_s": verify,
                       "s": seconds})
    except BaseException as e:                         # reported to the parent, which records it as a failed check
        try:
            conn.send({"error": f"{type(e).__name__}: {e}"})
        except OSError:
            pass


class ClientProcess:
    """A client process for ``verifier`` (a CPU verifier, after :meth:`pvi.v0.verifier.Verifier.precompute` in
    mode Kpre): its key goes to a temporary folder under ``folder`` (default: the system's), the process starts and
    loads it, and :meth:`verify` sends it one proof at a time.  :meth:`close` ends it and returns its memory."""

    def __init__(self, verifier, folder: str | None = None, threads: int | None = None) -> None:
        from pvi import options
        if verifier.stream or torch.device(verifier.device).type != "cpu":
            raise ValueError("the client process runs the CPU verifier without stream")
        self.dir = Path(tempfile.mkdtemp(prefix="pvi-client-", dir=folder))
        self.proc = None
        try:
            torch.save(client_key(verifier), self.dir / KEY)
            ctx = multiprocessing.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_serve, args=(child, str(self.dir), options.current(),
                                                         threads or torch.get_num_threads()))
            self.proc.start()
            child.close()
            self.ready = self._answer()
            (self.dir / KEY).unlink()
        except BaseException:
            self._stop(0)
            raise

    def _answer(self) -> dict:
        try:
            got = self.conn.recv()
        except (EOFError, OSError) as e:
            self.proc.join(5)
            raise RuntimeError(f"the client process ended (exit code {self.proc.exitcode}): {e!r}") from None
        if "error" in got:
            raise RuntimeError(f"the client process failed: {got['error']}")
        return got

    def verify(self, x: torch.Tensor, proof) -> dict:
        """The client's verdict on ``proof`` (a :class:`pvi.v0.query.WireProof`) for the query ``x``:
        ``accepted``, ``rejected_at``, its verifier's time ``verify_s`` and the whole check ``s``."""
        files = {}
        try:
            for name in MESSAGES:
                blob = getattr(proof, name)
                if blob is not None:
                    (self.dir / f"{name}.bin").write_bytes(blob)
                    files[name] = f"{name}.bin"
            self.conn.send({"op": "verify", "x": x.cpu(), "files": files, "lookups": proof.lookups,
                            "paths": proof.paths})
            return self._answer()
        finally:
            for f in files.values():
                (self.dir / f).unlink(missing_ok=True)

    def close(self) -> dict:
        """End the process: ``{"peak_rss", "baseline_rss", "key_rss"}`` (bytes; ``None`` without ``/proc``)."""
        try:
            self.conn.send({"op": "close"})
            peak = self._answer()["peak_rss"]
        finally:
            self._stop(60)
        return {"peak_rss": peak, "baseline_rss": self.ready["baseline_rss"], "key_rss": self.ready["key_rss"]}

    def _stop(self, wait: float) -> None:
        """Wait up to ``wait`` seconds for the process to end, then end it; remove the folder."""
        if self.proc is not None:
            self.proc.join(wait)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(5)
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self) -> "ClientProcess":
        return self

    def __exit__(self, *exc) -> None:
        self._stop(0)
