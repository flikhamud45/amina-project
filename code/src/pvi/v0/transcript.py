"""The verifier's randomness: interactive draws, or a Fiat--Shamir transcript (SHAKE-256 over every
message, large messages absorbed as a one-level tree; the paper's Section ``sec:fs``)."""

from __future__ import annotations

import hashlib
import secrets
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from pvi import options
from pvi.core.field import P

_TREE_MIN = 1 << 24
"""Messages of at least this many bytes are absorbed as a one-level tree (:meth:`Challenger.absorb_large`)."""
_TREE_CHUNK = 1 << 22


def _hash_threads() -> int:
    """Threads for the tree absorption: the option ``hash_threads``, else torch's intra-op thread count."""
    return options.current().hash_threads or torch.get_num_threads()


class Challenger:
    """Verifier randomness.

    Interactive: every challenge is expanded with SHAKE-256 from a fresh 256-bit
    key drawn from the operating system's CSPRNG (``secrets``) at the moment the
    challenge is issued.  In particular the column indices are keyed only after
    the prover has sent ``u``, so nothing the prover saw earlier predicts them.
    Fiat--Shamir: the key is read from the running SHAKE-256 transcript of the statement
    (the graph's digest, parameters, commitments, input) and of every prover message so far:
    each challenge absorbs its label into a copy of the transcript and reads 32 bytes, so the
    challenges are random-oracle outputs of prefix-free inputs without a length-extension argument.
    ``seed`` makes interactive keys reproducible (tests only).  The transcript starts from the domain
    separator ``pvi/fullcheck/v3``, a fixed string kept from an earlier module name: it is part of every
    transcript, so changing it would change every Fiat--Shamir challenge.
    """

    def __init__(self, *, fiat_shamir: bool = False, seed: int | None = None) -> None:
        self.fiat_shamir = fiat_shamir
        self._state = hashlib.shake_256(b"pvi/fullcheck/v3")
        self._seed = seed
        self._count = 0

    def absorb(self, label: bytes, blob: bytes) -> None:
        """Length-prefixed, labelled absorption (an injective transcript encoding)."""
        self._state.update(len(label).to_bytes(2, "big") + label + len(blob).to_bytes(8, "big"))
        self._state.update(blob)

    def absorb_large(self, label: bytes, blob) -> None:
        """:meth:`absorb` for a large message (claims, packed rows, folds): from ``_TREE_MIN`` bytes, a one-level
        tree.  The message is cut into ``_TREE_CHUNK``-byte chunks; leaf ``j`` is
        ``SHAKE-256("pvi/leaf/v1" | len(label) | label | j | chunk_j)`` (32 bytes), computed on a thread pool
        (hashlib releases the GIL), and the transcript absorbs, under ``label + "/tree"``, the number of
        chunks, the chunk size, the length and the leaves in order.  The encoding is injective (fixed chunking,
        lengths and indices bound, a label of its own), so a collision of the transcript gives a collision
        of SHAKE-256; the Fiat--Shamir bound gains the birthday term of the leaves.  Both parties call it
        for the same messages, so their transcripts agree."""
        n = len(blob)
        if n < _TREE_MIN or not self.fiat_shamir:
            self.absorb(label, blob)
            return
        view = memoryview(blob).cast("B")
        chunks = [(j, view[a:a + _TREE_CHUNK]) for j, a in enumerate(range(0, n, _TREE_CHUNK))]
        head = b"pvi/leaf/v1" + len(label).to_bytes(2, "big") + label

        def leaf(item):
            j, c = item
            h = hashlib.shake_256(head + j.to_bytes(8, "big"))
            h.update(c)
            return h.digest(32)

        with ThreadPoolExecutor(_hash_threads()) as pool:
            leaves = list(pool.map(leaf, chunks))
        self.absorb(label + b"/tree", struct.pack(">QQQ", len(chunks), _TREE_CHUNK, n) + b"".join(leaves))

    def _key(self, label: str) -> bytes:
        if self.fiat_shamir:
            h = self._state.copy()
            h.update(b"challenge/" + label.encode())
            return h.digest(32)
        if self._seed is None:
            return secrets.token_bytes(32)
        self._count += 1
        return hashlib.sha256(f"test-seed/{self._seed}/{self._count}/{label}".encode()).digest()

    def transcript_digest(self) -> str:
        """The transcript so far as 32 bytes of hex (SHAKE-256 of everything absorbed, under the label
        ``digest``, read from a copy so the transcript is unchanged).  Under Fiat--Shamir it fingerprints the
        statement and every prover message absorbed so far: equal digests on two machines mean byte-identical
        messages, the records' check that the proof does not depend on the hardware."""
        h = self._state.copy()
        h.update(b"digest")
        return h.hexdigest(32)

    def fork(self) -> "Challenger":
        """A copy with the same transcript and test-seed counter: under Fiat--Shamir it draws exactly the
        challenges this one would, from the messages both absorb (the prover's copy in V1's GKR)."""
        out = Challenger(fiat_shamir=self.fiat_shamir, seed=self._seed)
        out._state, out._count = self._state.copy(), self._count
        return out

    @staticmethod
    def _stream(key: bytes, label: str, n_bytes: int) -> bytes:
        return hashlib.shake_256(key + label.encode()).digest(n_bytes)

    def folding(self, name: str, n_rows: int, reps: int) -> torch.Tensor:
        """``reps x n_rows`` uniform field elements (rejection sampling, no bias)."""
        n = reps * n_rows
        key = self._key("fold/" + name)
        parts, have, block = [], 0, 0
        while have < n:
            want = int((n - have) * 1.08) + 64
            words = np.frombuffer(self._stream(key, f"fold/{name}/{block}", 4 * want), dtype="<u4")
            words = words.astype(np.int64)
            words = words[words < 2 * P] % P
            parts.append(words)
            have += words.size
            block += 1
        return torch.from_numpy(np.concatenate(parts)[:n].reshape(reps, n_rows).copy())

    def ext(self, name: str, count: int = 1, *, nonbase: bool = False) -> torch.Tensor:
        """``count`` uniform elements of ``F_{p^8}`` (:mod:`pvi.core.extfield`) ``[count, 8]``, by the rejection
        sampling of :meth:`folding`; with ``nonbase``, an element in ``F_p + x F_p`` (its coefficients 2..7 all
        zero, probability ``p^-6``) is drawn again.  For V1's optimised verifier (:mod:`pvi.options`) one element
        (V1's GKR rounds) is drawn by :meth:`_ext_one`: the same bytes, words and choice, without numpy."""
        key = self._key("ext/" + name)
        if count == 1 and options.current().v1_verifier == "optimised":
            return self._ext_one(key, name, nonbase)
        return self._ext_draw(key, name, count, nonbase)

    def _ext_one(self, key: bytes, name: str, nonbase: bool) -> torch.Tensor:
        """:meth:`_ext_draw` for ``count = 1``: per block the same 192-byte stream (48 little-endian words), the
        accepted words (below ``2p``, reduced) in groups of 8, the first group taken unless ``nonbase`` rejects it."""
        block = 0
        while True:
            words = [w % P for w in struct.unpack("<48I", self._stream(key, f"ext/{name}/{block}", 192)) if w < 2 * P]
            for i in range(0, len(words) - 7, 8):
                v = words[i:i + 8]
                if nonbase and not any(v[2:]):
                    continue
                return torch.tensor([v], dtype=torch.int64)
            block += 1

    def _ext_draw(self, key: bytes, name: str, count: int, nonbase: bool) -> torch.Tensor:
        """:meth:`ext`'s rejection sampling from ``key``, for every ``count``."""
        out, block = [], 0
        while len(out) < count:
            words = np.frombuffer(self._stream(key, f"ext/{name}/{block}", 4 * 8 * (count - len(out) + 2) * 2),
                                  dtype="<u4").astype(np.int64)
            words = words[words < 2 * P] % P
            for i in range(0, words.size - 7, 8):
                v = words[i:i + 8]
                if nonbase and not v[2:].any():
                    continue
                out.append(v)
                if len(out) == count:
                    break
            block += 1
        return torch.from_numpy(np.stack(out).copy())

    def columns(self, name: str, n_points: int, t: int) -> torch.Tensor:
        """``t`` distinct uniform indices in ``[0, n_points)``, sorted."""
        t = min(t, n_points)
        key = self._key("cols/" + name)
        limit = (1 << 64) - ((1 << 64) % n_points)
        chosen: set[int] = set()
        block = 0
        while len(chosen) < t:
            vals = np.frombuffer(self._stream(key, f"cols/{name}/{block}", 8 * (4 * t + 64)), dtype="<u8")
            for v in vals.tolist():
                if v < limit:
                    chosen.add(v % n_points)
                    if len(chosen) == t:
                        break
            block += 1
        return torch.tensor(sorted(chosen), dtype=torch.int64)
