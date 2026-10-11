"""Plan 2, H4 (a): an anonymous snapshot of the artifact, without git history.

    python scripts/export_anonymous.py --out <empty directory outside the repository> [--ref HEAD] [--pdf main_ext.pdf]

1. **Select.** The files of ``--ref`` that :data:`INCLUDE` names and :data:`EXCLUDE` does not, read with ``git archive``,
   so untracked files never leave and the history is not copied.
2. **Rewrite.** In every text file, machine-specific strings become neutral placeholders (:data:`REWRITES`): the cluster
   project path, home directories on Linux and Windows, cluster host names. Nothing else is changed.
3. **Add.** ``scripts/artifact_README.md`` becomes the snapshot's ``README.md``; ``--pdf`` (the extended version, built
   without author names) is copied as ``extended_version.pdf``.
4. **Refuse.** Every exported file is scanned: text files as text, PDFs through their extracted text and metadata, other
   binary files as bytes. The deny list is built at run time (:func:`deny_patterns`): the names and e-mail addresses of
   every commit author, the owners and names of the configured remotes, and fixed institutional and cluster strings. While
   anything matches, the script lists every match and exits with status 1, so a snapshot with a leak is never reported
   as finished. No name is written into the repository by this script.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = ["code/src", "code/tests", "code/experiments", "code/artifacts/results", "code/artifacts/comparison",
           "code/artifacts/models",          # the small trained MNIST/digits models that the quick tests load
           "code/pyproject.toml", "code/requirements.txt", "code/requirements-llm.txt"]
"""Repository paths exported (directories recursively)."""

EXCLUDE = [re.compile(p) for p in (
    r"(^|/)__pycache__/", r"\.pyc$",
    r"\.sbatch$",                               # job scripts name the cluster's nodes and accounts
    r"^code/experiments/4_defence_benchmark/slurm/",
)]
"""Paths left out even when :data:`INCLUDE` covers them."""

REWRITES = [
    (re.compile(r"/home/[\w.-]+(?:/[\w.+-]+)*?/amina-project"), "$PROJECT"),
    (re.compile(r"/home/[\w.-]+(?:/[\w.+-]+)*/"), "$HOME/"),      # every directory under a home directory
    (re.compile(r"/home/[\w.-]+"), "$HOME"),
    (re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s\"']+"), "C:/Users/user"),
    (re.compile(r"/c/Users/[^/\s\"']+"), "/c/Users/user"),
    (re.compile(r"[\w.-]+\.tau\.ac\.il"), "cluster-login"),
    (re.compile(r"\b(?:[cs]-0\d\d|n-b?\d{3}|t-\d{3}|n-h200|rack-[\w-]+)\b"), "node"),
]
"""``(pattern, replacement)`` applied to every exported text file, in order."""

STATIC_DENY = ["tau.ac.il", "Tel Aviv", "Tel-Aviv", "sharifm", "erelbarzilay", "tml-0368", "ofeklavi", "amina-project",
               "amina_project"]
"""Institutional and cluster strings that must never appear (case-insensitive)."""

GENERIC_IDENTITIES = {"research", "user", "root", "admin", "claude", "github", "noreply", "bot", "agent", "workflow",
                      "test", "unknown"}
"""Commit identities that are common words, not names: not denied (they would match ordinary text)."""

BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".pt", ".pth", ".bin", ".npz", ".npy", ".gz", ".zip",
                   ".tar", ".safetensors", ".parquet", ".so", ".dll", ".pyd"}
"""Never treated as text, whatever their content."""


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def deny_patterns() -> list[tuple[str, re.Pattern]]:
    """``(what, pattern)``: every commit author's full name, each name token of at least 4 letters, and e-mail address;
    the owner and repository name of each remote URL; :data:`STATIC_DENY`."""
    out: list[tuple[str, re.Pattern]] = []
    seen: set[str] = set()

    def add(what: str, s: str, word: bool = True) -> None:
        key = s.lower()
        if len(s) < 3 or key in seen or key in GENERIC_IDENTITIES:
            return
        seen.add(key)
        body = re.escape(s)
        out.append((what, re.compile(rf"(?<![\w]){body}(?![\w])" if word else body, re.IGNORECASE)))

    for line in _git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").splitlines():
        for i, field in enumerate(line.split("\t")):
            field = field.strip()
            if not field:
                continue
            if i % 2:                                   # an e-mail address
                add("author e-mail", field, word=False)
                local = field.split("@")[0]
                if len(local) >= 5:
                    add("author e-mail user", local)
            else:                                       # a name
                add("author name", field)
                for tok in re.split(r"[\s._-]+", field):
                    if len(tok) >= 4 and tok.isalpha():
                        add("author name token", tok)
    for line in _git("remote", "-v").splitlines():
        url = line.split()[1] if len(line.split()) > 1 else ""
        m = re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
        if m and "/" not in m.group(1):
            add("remote owner", m.group(1))
            add("remote repository", m.group(2))
    for s in STATIC_DENY:
        add("institution/cluster", s, word=False)
    return out


def _excluded(name: str) -> bool:
    return any(p.search(name) for p in EXCLUDE)


def _is_text(path: Path, data: bytes) -> bool:
    """Text by content (valid UTF-8 without NUL bytes), not by name: stored records such as ``*.jsonl.part.<n>`` end in a
    number."""
    if path.suffix.lower() in BINARY_SUFFIXES or bytes(1) in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def export(ref: str, out: Path, pdf: Path | None) -> int:
    blob = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref, "--", *INCLUDE], check=True,
                          capture_output=True).stdout
    n_files = n_rewritten = 0
    # names in provenance notes ("run by ...") become a neutral phrase; the scan below still has the last word
    names = [pat for what, pat in deny_patterns() if what in ("author name", "author name token")]
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or _excluded(member.name):
                continue
            n_files += 1
            data = tar.extractfile(member).read()
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _is_text(dest, data):
                text = data.decode("utf-8")
                new = text
                for pat, rep in REWRITES:
                    new = pat.sub(rep, new)
                for pat in names:
                    new = pat.sub("a team member", new)
                n_rewritten += new != text
                dest.write_text(new, encoding="utf-8", newline="")
            else:
                dest.write_bytes(data)
    readme = ROOT / "scripts" / "artifact_README.md"
    shutil.copyfile(readme, out / "README.md")
    if pdf is not None:
        shutil.copyfile(pdf, out / "extended_version.pdf")
    print(f"exported {n_files} files from {ref} ({n_rewritten} rewritten) to {out}")
    return n_files


def _npz_text(data: bytes) -> str:
    """What a NumPy ``.npz`` archive can carry besides numbers: its zip comment, member names, each member's dtype
    and shape, and the contents of every member that is not a numeric array (strings and bytes decoded, structured
    arrays as values, object arrays and non-arrays as raw bytes). Numeric array data is not scanned as text: random
    bytes there match short names by chance (two of the trained models did), and numbers cannot identify anyone. An
    archive that does not open is scanned as bytes."""
    import zipfile
    import numpy as np
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return data.decode("latin-1")
    parts = [zf.comment.decode("latin-1")]
    for info in zf.infolist():
        parts.append(info.filename)
        raw = zf.read(info)
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception:                   # object arrays (pickles) and anything else: scan the bytes
            parts.append(raw.decode("latin-1"))
            continue
        parts.append(f"{arr.dtype} {arr.shape}")
        if arr.dtype.kind not in "biufc?":   # strings, bytes, structured or other: their values as text
            parts.append(repr(arr.tolist()))
    return "\n".join(parts)


def scan(out: Path) -> list[str]:
    pats = deny_patterns()
    hits: list[str] = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        for what, pat in pats:                         # file names first
            if pat.search(rel):
                hits.append(f"{rel}: file name matches {what}")
        data = path.read_bytes()
        if path.suffix.lower() == ".pdf":
            import pymupdf
            doc = pymupdf.open(path)
            texts = [str(doc.metadata)] + [page.get_text() for page in doc]
            text = "\n".join(texts)
        elif path.suffix.lower() == ".npz":
            text = _npz_text(data)
        elif _is_text(path, data):
            text = data.decode("utf-8")
        else:
            text = data.decode("latin-1")
        for what, pat in pats:
            m = pat.search(text)
            if m:
                a = max(0, m.start() - 30)
                hits.append(f"{rel}: {what} ... {text[a:m.end() + 30]!r}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--pdf", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if ROOT in out.parents or out == ROOT:
        sys.exit("refusing: the snapshot must be written outside the repository")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing: {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    export(args.ref, out, Path(args.pdf) if args.pdf else None)
    hits = scan(out)
    if hits:
        print(f"REFUSED: {len(hits)} identifying matches remain:")
        for h in hits:
            print("  " + h)
        sys.exit(1)
    print("OK: no identifying string found")


if __name__ == "__main__":
    main()
