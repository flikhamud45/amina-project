"""``tools/export_anonymous.py``: the anonymous snapshot refuses every identifying string it cannot remove, inside
words and inside archives, rewrites archive members and keeps each file's blob bytes. Runs on a throwaway
repository with a fictitious author."""
from __future__ import annotations

import gzip
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tools import export_anonymous as ea

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git to build the repository")

AUTHOR = ("Jane Roe", "jroe7@example.org")
PRIVATE = {"deny": ["jroe7", "Acme University"], "deny_words": ["jroe"], "rewrite": [[r"\bgpu-07\b", "node-A"]]}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def _tar_gz(members: dict[str, bytes], owner: str = "jroe") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.uname, info.gname, info.mtime = len(data), owner, owner, 1_700_000_000
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), mtime=0)


def _repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "core.autocrlf", "true")
    for name, data in {"README.md": b"# A\n", "paper_sp2027/main.tex": b"x\n", **files}.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "-c", f"user.name={AUTHOR[0]}", "-c", f"user.email={AUTHOR[1]}", "commit", "-q", "-m", "a")
    return repo


def _export(monkeypatch, tmp_path: Path, files: dict[str, bytes], private: dict = PRIVATE) -> tuple[Path, list[str]]:
    monkeypatch.setattr(ea, "ROOT", _repo(tmp_path, files))
    out = tmp_path / "out"
    out.mkdir()
    ea.export("HEAD", out, None, private)
    return out, ea.scan(out, private)


def test_a_deny_entry_equal_to_an_email_user_matches_inside_a_word(monkeypatch, tmp_path):
    out, hits = _export(monkeypatch, tmp_path, {"code/rec.json": b'{"cache": "jroe7_hf/e2b.npz"}\n',
                                                "code/b.txt": b"by janeroe and jane_roe\n"})
    assert [h.split(" ...")[0] for h in hits] == ["code/b.txt: author name (joined)",
                                                  "code/rec.json: author e-mail user"]
    pats = ea.deny_patterns({"deny_words": ["acme"], "deny": ["acme"]})      # the wider pattern is kept as well
    assert [what for what, pat in pats if pat.search("acmelab")] == ["private deny"]


def test_an_archive_member_with_a_denied_string_is_refused(monkeypatch, tmp_path):
    logs = _tar_gz({"run/log.txt": b"/home/grp/jroe/env/lib/x.py:9: warning\nrun at Acme University\n"})
    out, hits = _export(monkeypatch, tmp_path, {"code/artifacts/logs.tar.gz": logs})
    assert any(h.startswith("code/artifacts/logs.tar.gz!/run/log.txt: private deny") for h in hits)
    with tarfile.open(out / "code/artifacts/logs.tar.gz") as tar:
        (member,) = tar.getmembers()
        text = tar.extractfile(member).read().decode()
    assert (member.uname, member.gname, member.uid) == ("", "", 0)
    assert "/home/" not in text and "$HOME/" in text
    assert not any("deny word" in h for h in hits)          # the home path, and with it the user name, is gone


def test_archives_and_text_are_rewritten_and_a_clean_tree_passes(monkeypatch, tmp_path):
    files = {"code/a.json": b'{"host": "gpu-07", "cache": "/tmp/jroe_hf/x.npz", "out": "/tmp/attack.json"}\n',
             "code/b.md": b"See code/data/cifar10/ and $ROOT/data/x/.\nline two\n",
             "code/smi.csv.gz": gzip.compress(b"gpu-07,1\n", mtime=0),
             "code/logs.tar.gz": _tar_gz({"log.txt": b"/home/grp/jroe/env/x.py:9: warning\n"})}
    out, hits = _export(monkeypatch, tmp_path, files)
    assert hits == []
    assert (out / "code/a.json").read_bytes() == \
        b'{"host": "node-A", "cache": "/tmp/user/x.npz", "out": "/tmp/attack.json"}\n'
    assert (out / "code/b.md").read_bytes() == files["code/b.md"]           # LF kept, relative paths unchanged
    assert gzip.decompress((out / "code/smi.csv.gz").read_bytes()) == b"node-A,1\n"


def test_a_binary_file_the_scan_cannot_read_is_refused(monkeypatch, tmp_path):
    out, hits = _export(monkeypatch, tmp_path, {"code/blob.bin": bytes(range(256)) * 4})
    assert hits == ["code/blob.bin: binary content the scan cannot read"]


def test_hosted_remotes_name_their_owner_and_local_ones_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(ea, "ROOT", _repo(tmp_path, {}))
    assert ea.remote_names("C:/Users/someone/scratch/srv") is None
    assert ea.remote_names("file:///srv/repo.git") is None
    assert ea.remote_names("https://example.org/owner1/proj.git") == ("owner1", "proj")
    pats = ea.deny_patterns({"remotes": ["git@example.org:someowner/someproj.git"]})
    whats = {what for what, pat in pats if pat.search("see example.org/someowner/someproj")}
    assert whats == {"remote owner", "remote repository"}
