"""``experiments.common.records.git_sha``: the commit a result came from, with and without a ``git`` binary (some compute nodes
have none), in a plain repository and in a worktree with packed refs."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.common import records as results

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git to build the repositories")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "a.txt").write_text("a")
    _git(r, "add", "a.txt")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "a")
    (r / "sub").mkdir()
    return r


def _no_git(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError(2, "No such file or directory: 'git'")
    monkeypatch.setattr(results.subprocess, "run", missing)


def test_with_git_it_is_rev_parse(repo):
    assert results.git_sha(repo / "sub") == _git(repo, "rev-parse", "--short=7", "HEAD")


def test_without_git_it_reads_head_and_refs(repo, monkeypatch):
    full = _git(repo, "rev-parse", "HEAD")
    _no_git(monkeypatch)
    assert results.git_sha(repo / "sub") == full[:7]


def test_without_git_packed_refs_and_worktrees(repo, monkeypatch):
    full = _git(repo, "rev-parse", "HEAD")
    wt = repo.parent / "wt"
    _git(repo, "worktree", "add", "-q", "-b", "side", str(wt))
    _git(repo, "pack-refs", "--all")
    _no_git(monkeypatch)
    assert results.git_sha(repo) == full[:7]
    assert results.git_sha(wt) == full[:7]


def test_outside_a_repository_it_is_empty(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    assert results.git_sha(tmp_path) == ""
