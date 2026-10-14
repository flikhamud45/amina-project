"""An anonymous snapshot of the artifact for double-blind review: the files of one commit, without its history, with
machine-specific strings replaced, refused while any identifying string remains.

    python code/tools/export_anonymous.py --out <empty folder outside the repository> --private <file.json> \\
        [--ref HEAD] [--pdf <the extended version, built without author names>]

1. **Select.** The files of ``--ref`` under :data:`INCLUDE` and not matched by :data:`EXCLUDE`, read with
   ``git archive``: untracked files never leave and no history is copied.
2. **Rewrite.** In every text file, each ``rewrite`` rule of the private file applies (cluster paths and host names
   to placeholders), then home directories (Linux and Windows) become ``$HOME`` and ``C:/Users/user``, and the commit
   authors' names become "a team member". Nothing else changes.
3. **Add.** ``--pdf`` is copied as ``extended_version.pdf``.
4. **Refuse.** Every exported file is scanned: its path, text files as text, PDFs through their text and metadata,
   NumPy archives through their names and non-numeric members (:func:`npz_text`), other files as bytes. The patterns
   (:func:`deny_patterns`) are the commit authors' names, name parts and e-mail addresses, the owners and names of the
   configured remotes, and the private file's ``deny`` and ``deny_words`` strings. While anything matches, every match
   is listed and the exit status is 1.

The private file is JSON kept outside the repository, so that no name is written into the repository itself::

    {"deny": ["<institution>", "<cluster account>", ...],      # matched anywhere, ignoring case
     "deny_words": ["<short user name>", ...],                  # matched as whole words, ignoring case
     "rewrite": [["<regular expression>", "<placeholder>"], ...]}

Scanning PDFs needs PyMuPDF (``pip install pymupdf``).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INCLUDE = ["README.md", "code", "paper_sp2027"]
"""Repository paths exported (folders recursively)."""

EXCLUDE = [re.compile(p) for p in (r"(^|/)__pycache__/", r"\.py[cod]$", r"^paper_sp2027/main(_ext)?\.pdf$")]
"""Paths left out although :data:`INCLUDE` covers them (the built paper: the reviewers' copy is ``--pdf``)."""

HOME_REWRITES = [
    (re.compile(r"/home/[\w.-]+(?:/[\w.+-]+)*/"), "$HOME/"),       # every folder under a home directory
    (re.compile(r"/home/[\w.-]+"), "$HOME"),
    (re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s\"']+"), "C:/Users/user"),
    (re.compile(r"/c/Users/[^/\s\"']+"), "/c/Users/user"),
]
"""``(pattern, replacement)`` applied to every exported text file after the private file's rules."""

GENERIC_IDENTITIES = {"research", "user", "root", "admin", "claude", "github", "noreply", "bot", "agent", "workflow",
                      "test", "unknown"}
"""Commit identities that are ordinary words, not names: not denied (they would match ordinary text)."""

BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".pt", ".pth", ".bin", ".npz", ".npy", ".gz", ".zip",
                   ".tar", ".safetensors", ".parquet", ".so", ".dll", ".pyd"}
"""Never treated as text, whatever their content."""


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def deny_patterns(private: dict) -> list[tuple[str, re.Pattern]]:
    """``(what, pattern)``: every commit author's full name, each name part of at least 4 letters and each e-mail
    address; the owner and repository name of each remote; the private file's ``deny`` (anywhere) and
    ``deny_words`` (whole words) strings."""
    out: list[tuple[str, re.Pattern]] = []
    seen: set[str] = set()

    def add(what: str, s: str, word: bool = True) -> None:
        key = s.lower()
        if len(s) < 3 or key in seen or key in GENERIC_IDENTITIES:
            return
        seen.add(key)
        body = re.escape(s)
        out.append((what, re.compile(rf"(?<!\w){body}(?!\w)" if word else body, re.IGNORECASE)))

    for line in git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").splitlines():
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
                        add("author name part", tok)
    for line in git("remote", "-v").splitlines():
        url = line.split()[1] if len(line.split()) > 1 else ""
        m = re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
        if m and "/" not in m.group(1):
            add("remote owner", m.group(1))
            add("remote repository", m.group(2))
    for s in private.get("deny", []):
        add("private deny", s, word=False)
    for s in private.get("deny_words", []):
        add("private deny word", s)
    return out


def is_text(path: Path, data: bytes) -> bool:
    """Text by content (valid UTF-8 without NUL bytes), not by name: stored records such as ``*.jsonl.part.<n>`` end
    in a number."""
    if path.suffix.lower() in BINARY_SUFFIXES or bytes(1) in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def export(ref: str, out: Path, pdf: Path | None, private: dict) -> int:
    """Write the selected files of ``ref`` into ``out``, rewritten; returns the number of files."""
    blob = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref, "--", *INCLUDE], check=True,
                          capture_output=True).stdout
    rewrites = [(re.compile(p), r) for p, r in private.get("rewrite", [])] + HOME_REWRITES
    names = [pat for what, pat in deny_patterns(private) if what in ("author name", "author name part")]
    n_files = n_rewritten = 0
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or any(p.search(member.name) for p in EXCLUDE):
                continue
            n_files += 1
            data = tar.extractfile(member).read()
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if is_text(dest, data):
                text = data.decode("utf-8")
                new = text
                for pat, rep in rewrites:
                    new = pat.sub(rep, new)
                for pat in names:
                    new = pat.sub("a team member", new)
                n_rewritten += new != text
                dest.write_text(new, encoding="utf-8", newline="")
            else:
                dest.write_bytes(data)
    if pdf is not None:
        shutil.copyfile(pdf, out / "extended_version.pdf")
    print(f"exported {n_files} files of {ref} ({n_rewritten} rewritten) to {out}")
    return n_files


def npz_text(data: bytes) -> str:
    """What a NumPy ``.npz`` archive can carry besides numbers: its zip comment, member names, each member's dtype and
    shape, and every member that is not a numeric array (strings and bytes decoded, structured arrays as values,
    object arrays and anything else as raw bytes). Numeric array data is not scanned: random bytes there match short
    names by chance, and numbers cannot identify anyone. An archive that does not open is scanned as bytes."""
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
        if arr.dtype.kind not in "biufc?":
            parts.append(repr(arr.tolist()))
    return "\n".join(parts)


def scan(out: Path, private: dict) -> list[str]:
    """Every match of :func:`deny_patterns` in the snapshot, as ``"<file>: <what> ... <context>"``."""
    pats = deny_patterns(private)
    hits: list[str] = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        hits += [f"{rel}: file name matches {what}" for what, pat in pats if pat.search(rel)]
        data = path.read_bytes()
        if path.suffix.lower() == ".pdf":
            import pymupdf
            doc = pymupdf.open(path)
            text = "\n".join([str(doc.metadata)] + [page.get_text() for page in doc])
        elif path.suffix.lower() == ".npz":
            text = npz_text(data)
        elif is_text(path, data):
            text = data.decode("utf-8")
        else:
            text = data.decode("latin-1")
        for what, pat in pats:
            m = pat.search(text)
            if m:
                a = max(0, m.start() - 30)
                hits.append(f"{rel}: {what} ... {text[a:m.end() + 30]!r}")
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="an empty folder outside the repository")
    ap.add_argument("--private", required=True,
                    help="the JSON file of deny strings and rewrite rules, kept outside the repository")
    ap.add_argument("--ref", default="HEAD", help="the commit to export")
    ap.add_argument("--pdf", default=None, help="the extended version without author names, added as "
                                                "extended_version.pdf")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if ROOT in out.parents or out == ROOT:
        sys.exit("refusing: the snapshot must be written outside the repository")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing: {out} is not empty")
    private = json.loads(Path(args.private).read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    export(args.ref, out, Path(args.pdf) if args.pdf else None, private)
    hits = scan(out, private)
    if hits:
        print(f"REFUSED: {len(hits)} identifying matches remain:")
        print("\n".join("  " + h for h in hits))
        return 1
    print("OK: no identifying string found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
