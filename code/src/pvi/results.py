"""Writing result files from a checkout shared by several accounts.

Three of us run these scripts against the same working copy, under a default
umask of 022.  A results file created by one account is read-only to the others,
so a plain ``open(path, "w")`` raises ``PermissionError`` for whoever did not
create it -- after the experiment has already run, which is the worst moment.

Writing to a temporary file in the same directory and renaming over the target
needs only directory write permission, which everyone has, and leaves the result
group-writable so the next person is not blocked either.

``git_sha`` names the commit a result came from; some compute nodes have no ``git`` binary, so it falls back to
reading the repository's ``HEAD`` and refs itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["git_sha", "write_json"]


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Atomically write ``payload`` as JSON, group-writable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_result_")
    try:
        with os.fdopen(handle, "w") as fh:
            json.dump(payload, fh, indent=indent)
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def _git_dir(path: Path) -> Path | None:
    for d in (path, *path.parents):
        dot = d / ".git"
        if dot.is_dir():
            return dot
        if dot.is_file():                            # a worktree or submodule: "gitdir: <path>"
            text = dot.read_text().strip()
            if text.startswith("gitdir:"):
                g = Path(text.split(":", 1)[1].strip())
                return g if g.is_absolute() else (d / g).resolve()
    return None


def _read_ref(git_dir: Path, ref: str) -> str | None:
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    for d in (git_dir, common):
        f = d / ref
        if f.is_file():
            return f.read_text().strip()
        packed = d / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    return None


def git_sha(path: Path | str = ".", *, short: int = 7) -> str:
    """The abbreviated commit of the repository holding ``path`` (``git rev-parse --short HEAD``); without a ``git``
    binary, ``HEAD`` and its ref are read from the repository directly. ``""`` when neither works."""
    path = Path(path).resolve()
    try:
        sha = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=path, timeout=30).stdout.strip()
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        git_dir = _git_dir(path)
        if git_dir is None:
            return ""
        head = (git_dir / "HEAD").read_text().strip()
        full = _read_ref(git_dir, head[4:].strip()) if head.startswith("ref:") else head
        return (full or "")[:short]
    except OSError:
        return ""
