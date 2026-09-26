"""Writing result files from a checkout shared by several accounts.

Three of us run these scripts against the same working copy, under a default
umask of 022.  A results file created by one account is read-only to the others,
so a plain ``open(path, "w")`` raises ``PermissionError`` for whoever did not
create it -- after the experiment has already run, which is the worst moment.

Writing to a temporary file in the same directory and renaming over the target
needs only directory write permission, which everyone has, and leaves the result
group-writable so the next person is not blocked either.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["write_json"]


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
