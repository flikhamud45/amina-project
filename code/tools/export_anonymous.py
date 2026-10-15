"""An anonymous snapshot of the artifact for double-blind review: the files of one commit, without its history, with
machine-specific strings replaced, refused while any identifying string remains.

    python code/tools/export_anonymous.py --out <empty folder outside the repository> --private <file.json> \\
        [--ref HEAD] [--pdf <the extended version, built without author names>]

1. **Select.** The files of ``--ref`` under :data:`INCLUDE` and not matched by :data:`EXCLUDE`, read with
   ``git archive`` without line-ending conversion (each file has its blob's bytes): untracked files never leave and no
   history is copied.
2. **Rewrite.** In every text file, each ``rewrite`` rule of the private file applies (cluster paths and host names
   to placeholders), then :data:`HOME_REWRITES` (home directories become ``$HOME`` and ``C:/Users/user``, the first
   folder under ``/tmp``, ``/scratch`` and ``/data`` becomes ``user``), and the commit authors' names become "a team
   member". Archives (``.tar``, ``.tar.gz``, ``.tgz``, ``.gz``, ``.zip``) are opened and their members rewritten in
   the same way, member names included; a tar archive is written again with its members' owners cleared. Nothing
   else changes.
3. **Add.** ``--pdf`` is copied as ``extended_version.pdf``.
4. **Refuse.** Every exported file is scanned (:func:`inspect`): its path, text files as text, PDFs through their
   text and metadata, NumPy archives through their names and non-numeric members (:func:`npz_text`), images as
   bytes, and archives through every member's name, owner and content, recursively. Any other binary file is refused,
   since its content cannot be read. The patterns (:func:`deny_patterns`) are the commit authors' names, name parts,
   joined names, e-mail addresses and e-mail user names, the owners and names of the repository's hosted remotes and
   of the private file's ``remotes``, and the private file's ``deny``, ``deny_words`` and ``deny_case`` strings.
   While anything matches, every match is listed and the exit status is 1.

The private file is JSON kept outside the repository, so that no name is written into the repository itself::

    {"deny": ["<institution>", "<cluster account>", ...],      # matched anywhere, ignoring case
     "deny_words": ["<short user name>", ...],                  # whole words, ignoring case
     "deny_case": ["<acronym>", ...],                           # whole words, case-sensitive
     "remotes": ["git@<host>:<owner>/<repository>.git", ...],   # hosted copies of the history: owner, name denied
     "rewrite": [["<regular expression>", "<placeholder>"], ...]}

A whole word is one not preceded or followed by a letter or a digit (an underscore separates words). The rewrite
rules should give each host its own placeholder, so that the records of different hosts stay distinguishable, and
must name every host and checkout path of the machines whose records or logs are exported. The records keep the
commit hashes of the development history, so a hosted copy of that history must stay private until the review ends.

Scanning PDFs needs PyMuPDF (``pip install pymupdf``).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INCLUDE = ["README.md", "code", "paper_sp2027"]
"""Repository paths exported (folders recursively)."""

EXCLUDE = [re.compile(p) for p in (r"(^|/)__pycache__/", r"\.py[cod]$", r"^paper_sp2027/main(_ext)?\.pdf$")]
"""Paths left out although :data:`INCLUDE` covers them (the built paper: the reviewers' copy is ``--pdf``)."""

HOME_REWRITES = [
    (re.compile(r"/home/[\w.-]+(?:/[\w.+-]+)*/"), "$HOME/"),       # every folder under a home directory
    (re.compile(r"/home/[\w.-]+"), "$HOME"),
    (re.compile(r"(?<![\w.~$/-])/(tmp|scratch|data)/[^/\s\"']+/"), r"/\1/user/"),   # a user's folder there
    (re.compile(r"(?<![\w.~$/-])/(scratch|data)/[\w.+-]+"), r"/\1/user"),
    (re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s\"']+"), "C:/Users/user"),
    (re.compile(r"/c/Users/[^/\s\"']+"), "/c/Users/user"),
]
"""``(pattern, replacement)`` applied to every exported text file after the private file's rules."""

GENERIC_IDENTITIES = {"research", "user", "root", "admin", "claude", "github", "noreply", "bot", "agent", "workflow",
                      "test", "unknown"}
"""Commit identities that are ordinary words, not names: not denied (they would match ordinary text)."""

BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".pt", ".pth", ".bin", ".npz", ".npy", ".gz", ".tgz",
                   ".zip", ".tar", ".xz", ".bz2", ".7z", ".safetensors", ".parquet", ".so", ".dll", ".pyd"}
"""Never treated as text, whatever their content."""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif"}
"""Binary files scanned as bytes: their metadata (author, software, comments) is stored as plain text."""

WORD = r"(?<![^\W_]){}(?![^\W_])"
"""A whole word: not preceded or followed by a letter or a digit."""


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def remote_names(url: str) -> tuple[str, str] | None:
    """``(owner, repository)`` of a hosted remote's URL (``git@host:owner/repo.git``, ``https://host/owner/repo``);
    ``None`` for a local path, which names no hosted copy."""
    if url.startswith("file:") or ("://" not in url and not re.match(r"^[\w.-]+@[\w.-]+:", url)):
        return None
    m = re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url)
    return (m.group(1), m.group(2)) if m else None


def deny_patterns(private: dict) -> list[tuple[str, re.Pattern]]:
    """``(what, pattern)``: every commit author's full name, each name part of at least 4 letters, the name's parts
    joined (anywhere, as in a user name), each e-mail address and each e-mail user name of at least 5 characters
    (anywhere); the owner and repository name of each hosted remote; the private file's ``deny`` (anywhere),
    ``deny_words`` (whole words) and ``deny_case`` (whole words, case-sensitive) strings. A string denied both as a
    whole word and anywhere keeps both patterns, so the wider one always applies."""
    out: list[tuple[str, re.Pattern]] = []
    seen: set[tuple[str, bool, bool]] = set()

    def add(what: str, s: str, word: bool = True, case: bool = False) -> None:
        key = (s if case else s.lower(), word, case)
        if len(s) < 3 or key in seen or s.lower() in GENERIC_IDENTITIES:
            return
        seen.add(key)
        body = re.escape(s)
        out.append((what, re.compile(WORD.format(body) if word else body, 0 if case else re.IGNORECASE)))

    for line in git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").splitlines():
        for i, field in enumerate(line.split("\t")):
            field = field.strip()
            if not field:
                continue
            if i % 2:                                   # an e-mail address
                add("author e-mail", field, word=False)
                local = field.split("@")[0]
                if len(local) >= 5:
                    add("author e-mail user", local, word=False)
            else:                                       # a name
                add("author name", field)
                parts = [tok for tok in re.split(r"[\s._-]+", field) if tok]
                for tok in parts:
                    if len(tok) >= 4 and tok.isalpha():
                        add("author name part", tok)
                if len(parts) > 1 and len("".join(parts)) >= 6:
                    for sep in ("", ".", "_", "-"):
                        add("author name (joined)", sep.join(parts), word=False)
    urls = [line.split()[1] for line in git("remote", "-v").splitlines() if len(line.split()) > 1]
    for url in urls + list(private.get("remotes", [])):
        names = remote_names(url)
        if names:
            add("remote owner", names[0])
            add("remote repository", names[1])
    for s in private.get("deny", []):
        add("private deny", s, word=False)
    for s in private.get("deny_words", []):
        add("private deny word", s)
    for s in private.get("deny_case", []):
        add("private deny word (case)", s, case=True)
    return out


def is_text(path: Path | str, data: bytes) -> bool:
    """Text by content (valid UTF-8 without NUL bytes), not by name: stored records such as ``*.jsonl.part.<n>`` end
    in a number."""
    if Path(path).suffix.lower() in BINARY_SUFFIXES or bytes(1) in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def archive_kind(name: str) -> str | None:
    """``"tar"``, ``"tgz"`` (a gzip-compressed tar), ``"gz"`` (one gzip-compressed file) or ``"zip"`` by the file
    name; ``None`` for anything else (NumPy's ``.npz``, a zip of arrays, is read by :func:`npz_text`)."""
    low = name.lower()
    if low.endswith((".tar.gz", ".tgz")):
        return "tgz"
    for suffix in ("tar", "gz", "zip"):
        if low.endswith("." + suffix):
            return suffix
    return None


class Rewriter:
    """The rewrite of step 2: the private file's rules, :data:`HOME_REWRITES` and the authors' names."""

    def __init__(self, private: dict):
        self.rules = [(re.compile(p), r) for p, r in private.get("rewrite", [])] + HOME_REWRITES
        self.names = [pat for what, pat in deny_patterns(private) if what in ("author name", "author name part")]

    def text(self, text: str) -> str:
        for pat, rep in self.rules:
            text = pat.sub(rep, text)
        for pat in self.names:
            text = pat.sub("a team member", text)
        return text

    def file(self, name: str, data: bytes) -> bytes:
        """The rewritten bytes of a file (or archive member) called ``name``."""
        kind = archive_kind(name)
        if kind in ("tar", "tgz"):
            return self._tar(data, compress=kind == "tgz")
        if kind == "gz":
            inner = gzip.decompress(data)
            new = self.file(name[:-3], inner)
            return data if new == inner else gzip.compress(new, mtime=0)
        if kind == "zip":
            return self._zip(data)
        if is_text(name, data):
            return self.text(data.decode("utf-8")).encode("utf-8")
        return data

    def _tar(self, data: bytes, compress: bool) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as src, \
                tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as dst:
            for m in src.getmembers():
                info = tarfile.TarInfo(self.text(m.name))
                info.type, info.mode, info.mtime = m.type, m.mode, m.mtime
                info.linkname = self.text(m.linkname)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                if m.isfile():
                    content = self.file(m.name, src.extractfile(m).read())
                    info.size = len(content)
                    dst.addfile(info, io.BytesIO(content))
                else:
                    dst.addfile(info)
        return gzip.compress(buf.getvalue(), mtime=0) if compress else buf.getvalue()

    def _zip(self, data: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(buf, "w") as dst:
            dst.comment = self.text(src.comment.decode("utf-8", "replace")).encode("utf-8")
            for info in src.infolist():
                new = zipfile.ZipInfo(self.text(info.filename), info.date_time)
                new.compress_type, new.external_attr = info.compress_type, info.external_attr
                dst.writestr(new, self.file(info.filename, src.read(info)))
        return buf.getvalue()


def export(ref: str, out: Path, pdf: Path | None, private: dict) -> int:
    """Write the selected files of ``ref`` into ``out``, rewritten; returns the number of files."""
    blob = subprocess.run(["git", "-C", str(ROOT), "-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive",
                           "--format=tar", ref, "--", *INCLUDE], check=True, capture_output=True).stdout
    n_files, n_rewritten = write_tree(blob, out, Rewriter(private))
    if pdf is not None:
        shutil.copyfile(pdf, out / "extended_version.pdf")
    print(f"exported {n_files} files of {ref} ({n_rewritten} rewritten) to {out}")
    return n_files


def write_tree(blob: bytes, out: Path, rewriter: Rewriter) -> tuple[int, int]:
    """Write the regular files of the tar ``blob`` (``git archive``'s output) into ``out`` through ``rewriter``;
    returns the number of files written and of files changed."""
    n_files = n_rewritten = 0
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or any(p.search(member.name) for p in EXCLUDE):
                continue
            n_files += 1
            data = tar.extractfile(member).read()
            new = rewriter.file(member.name, data)
            n_rewritten += new != data
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(new)
    return n_files, n_rewritten


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


def _gzip_header(data: bytes) -> str:
    """The original file name and comment that a gzip header may carry."""
    flags, pos, parts = data[3], 10, []
    if flags & 4:                                       # FEXTRA
        pos += 2 + int.from_bytes(data[pos:pos + 2], "little")
    for bit in (8, 16):                                 # FNAME, FCOMMENT
        if flags & bit:
            end = data.index(b"\0", pos)
            parts.append(data[pos:end].decode("latin-1"))
            pos = end + 1
    return "\n".join(parts)


def inspect(name: str, data: bytes, label: str | None = None) -> list[tuple[str, str | None]]:
    """``(label, text)`` for the file ``name`` and, in an archive, for each member (labelled
    ``<archive>!/<member>``), recursively: what the scan reads. ``text`` is ``None`` for content that cannot be read
    (a binary file of another type, an archive that does not open)."""
    label = name if label is None else label
    kind = archive_kind(name)
    suffix = Path(name).suffix.lower()
    try:
        if kind in ("tar", "tgz"):
            items = []
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
                for m in tar.getmembers():
                    header = [m.name, m.linkname, m.uname, m.gname, *map(str, m.pax_headers.values())]
                    items.append((f"{label}!/{m.name} (header)", "\n".join(header)))
                    if m.isfile():
                        items += inspect(m.name, tar.extractfile(m).read(), f"{label}!/{m.name}")
            return items
        if kind == "gz":
            inner = name[:-3]
            return ([(label + " (header)", _gzip_header(data))]
                    + inspect(inner, gzip.decompress(data), f"{label}!/{Path(inner).name}"))
        if kind == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                items = [(label + " (comment)", zf.comment.decode("latin-1"))]
                for info in zf.infolist():
                    items.append((f"{label}!/{info.filename} (name)", info.filename))
                    items += inspect(info.filename, zf.read(info), f"{label}!/{info.filename}")
            return items
    except (tarfile.TarError, zipfile.BadZipFile, zlib.error, OSError, EOFError, ValueError, IndexError):
        return [(label, None)]
    if suffix == ".pdf":
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        return [(label, "\n".join([str(doc.metadata)] + [page.get_text() for page in doc]))]
    if suffix == ".npz":
        return [(label, npz_text(data))]
    if suffix in IMAGE_SUFFIXES:
        return [(label, data.decode("latin-1"))]
    if is_text(name, data):
        return [(label, data.decode("utf-8"))]
    return [(label, None)]


def scan(out: Path, private: dict) -> list[str]:
    """Every match of :func:`deny_patterns` in the snapshot, and every file that cannot be read, as
    ``"<file>: <what> ... <context>"``."""
    pats = deny_patterns(private)
    hits: list[str] = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        hits += [f"{rel}: file name matches {what}" for what, pat in pats if pat.search(rel)]
        for label, text in inspect(rel, path.read_bytes()):
            if text is None:
                hits.append(f"{label}: binary content the scan cannot read")
                continue
            for what, pat in pats:
                m = pat.search(text)
                if m:
                    a = max(0, m.start() - 30)
                    hits.append(f"{label}: {what} ... {text[a:m.end() + 30]!r}")
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
