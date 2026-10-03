"""Contention evidence on every timing (``pvi.fullcheck.contention``): measurement only.

The run-queue wait and on-CPU time of the process's threads (``/proc/self/task/*/schedstat``), the load
average at a phase's start and end and the context switches are recorded next to each timed phase of
``run_query``, the commitment, the build and the Kpre precomputation, and ``bench.py`` writes them as a
new ``contention`` field of the timing rows.  These tests check the arithmetic on a fake ``/proc``, that
nothing breaks without ``/proc`` (Windows, macOS), that the verdicts, labels and bytes of a query do not
depend on the logging, and that the readers of the raw records (aggregate.py, fingerprint_check.py,
count_outcomes.py) give exactly what they gave without the field.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

from pvi.fullcheck import contention
from pvi.fullcheck import protocol as proto

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _script(folder: str, name: str):
    path = EXPERIMENTS / folder / f"{name}.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"{name}_contention_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- a fake /proc
def _fake_proc(root: Path, threads: dict, load: str | None = "1.50 2.25 3.00 4/900 123", schedstat=True):
    """``root`` as a procfs: ``self/task/<tid>/schedstat`` = "<run ns> <wait ns> <slices>" per thread."""
    task = root / "self" / "task"
    if task.exists():
        for d in task.iterdir():
            (d / "schedstat").unlink()
            d.rmdir()
    task.mkdir(parents=True, exist_ok=True)
    (root / "self" / "schedstat").unlink(missing_ok=True)
    if schedstat:
        (root / "self" / "schedstat").write_text("1 2 3\n")
    for tid, (run, wait) in threads.items():
        (task / str(tid)).mkdir()
        (task / str(tid) / "schedstat").write_text(f"{run} {wait} 7\n")
    (root / "loadavg").unlink(missing_ok=True)
    if load is not None:
        (root / "loadavg").write_text(load + "\n")


def test_deltas_per_thread_new_threads_from_zero_ended_threads_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(contention, "PROC", str(tmp_path))
    _fake_proc(tmp_path, {10: (1000, 100), 11: (5000, 0), 12: (300, 30)})
    a = contention.snapshot()
    assert a.threads == {10: (1000, 100), 11: (5000, 0), 12: (300, 30)}
    assert a.load == [1.5, 2.25, 3.0, 4, 900]
    # thread 12 ended, 13 started during the phase, 11 is a new thread under a recycled id
    _fake_proc(tmp_path, {10: (4000, 1100), 11: (200, 50), 13: (700, 70)}, load="9.00 4.00 3.10 40/950 124")
    d = contention.delta(a, contention.snapshot())
    assert d["run_ns"] == 3000 + 200 + 700 and d["wait_ns"] == 1000 + 50 + 70
    assert d["threads"] == 3 and d["threads_exited"] == 1 and d["segments"] == 1
    assert d["load_start"] == [1.5, 2.25, 3.0, 4, 900] and d["load_end"] == [9.0, 4.0, 3.1, 40, 950]
    assert d["wait_frac"] == pytest.approx(1120 / (3900 + 1120))
    json.dumps(d)                                   # a JSON field as it stands


def test_a_phase_in_parts_sums_them(tmp_path, monkeypatch):
    monkeypatch.setattr(contention, "PROC", str(tmp_path))
    store: dict = {}
    _fake_proc(tmp_path, {1: (0, 0)}, load="1.00 1.00 1.00 1/10 5")
    c0 = contention.begin()
    _fake_proc(tmp_path, {1: (100, 10)}, load="2.00 1.00 1.00 2/10 5")
    contention.add(store, "prove_encode", c0)
    c0 = contention.begin()
    _fake_proc(tmp_path, {1: (400, 110), 2: (50, 0)}, load="3.00 1.00 1.00 3/11 5")
    contention.add(store, "prove_encode", c0)
    d = store["prove_encode"]
    assert d["segments"] == 2 and d["run_ns"] == 100 + 300 + 50 and d["wait_ns"] == 10 + 100
    assert d["load_start"][0] == 1.0 and d["load_end"][0] == 3.0 and d["threads"] == 2
    assert d["wait_frac"] == pytest.approx(110 / 560) and d["wall_s"] >= 0


def test_no_schedstat_no_loadavg_records_nulls_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(contention, "PROC", str(tmp_path / "missing"))
    assert not contention.available()
    store: dict = {}
    c0 = contention.begin()
    contention.add(store, "verify_total", c0)
    d = store["verify_total"]
    assert d["run_ns"] is None and d["wait_ns"] is None and d["wait_frac"] is None and d["threads"] is None
    assert d["load_start"] is None and d["load_end"] is None and d["wall_s"] >= 0
    # schedstat present but a task file unreadable or garbage: that thread is skipped, nothing raises
    monkeypatch.setattr(contention, "PROC", str(tmp_path))
    _fake_proc(tmp_path, {5: (10, 1)}, load="garbage")
    (tmp_path / "self" / "task" / "6").mkdir()
    (tmp_path / "self" / "task" / "6" / "schedstat").write_text("not numbers\n")
    s = contention.snapshot()
    assert s.threads == {5: (10, 1)} and s.load is None
    contention.add(store, "x", None)                # a failed begin(): nothing recorded
    assert "x" not in store
    # the kernel without per-task schedstat: run/wait are None even though task dirs exist
    _fake_proc(tmp_path, {5: (10, 1)}, schedstat=False)
    assert contention.snapshot().threads is None


@pytest.mark.skipif(not os.path.exists("/proc/self/schedstat"), reason="needs Linux schedstat")
def test_real_schedstat_sees_this_process_run():
    store: dict = {}
    c0 = contention.begin()
    t = time.perf_counter()
    while time.perf_counter() - t < 0.05:
        pass
    contention.add(store, "busy", c0)
    d = store["busy"]
    assert d["run_ns"] > 1e7 and d["wait_ns"] >= 0 and d["threads"] >= 1
    assert d["load_start"] is not None and d["nivcsw"] is not None


# ---------------------------------------------------------------- run_query
def _queries():
    """(name, prover factory, verifier, query) of every run_query path: mode C under a c policy (lookup
    tables), interactive and Fiat-Shamir, batched and streaming; K and Kpre, with the verifier's own rows."""
    from test_plans import _planned, _verifiers

    graph, x, coms = _planned("gpt", "R8c")
    out = []
    for fs in (False, True):
        for v in _verifiers(graph, coms, fs):
            out.append((f"C fs={fs} stream={v.stream}", lambda: proto.Prover(graph, commitments=coms), v, x))
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    for mode in ("K", "Kpre"):
        for lookups in (False, True):
            for stream in (False, True):
                v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), mode, weights=weights,
                                   lookups=lookups, stream=stream)
                if mode == "Kpre":
                    v.precompute(proto.Challenger(seed=1))
                out.append((f"{mode} lookups={lookups} stream={stream}", lambda: proto.Prover(graph), v, x))
    return out


def test_every_timed_phase_of_run_query_carries_evidence_and_nothing_else_changes(monkeypatch):
    for name, prover, v, x in _queries():
        for wire in (False, True):
            seed = None if v.params.fiat_shamir else 1
            res = proto.run_query(prover(), v, x, seed=seed, wire=wire)
            assert res["accepted"], name
            cs = res["contention"]
            timed = {k for k, s in res["timings"].items() if s > 0}
            assert timed <= set(cs) <= set(res["timings"]), (name, wire, timed ^ set(cs))
            assert all(d["segments"] >= 1 and d["wall_s"] >= 0 for d in cs.values())
            with monkeypatch.context() as m:        # the same query without any logging
                m.setattr(contention, "begin", lambda: None)
                off = proto.run_query(prover(), v, x, seed=seed, wire=wire)
            assert off["contention"] == {}
            assert (off["accepted"], off["rejected_at"], off["bytes"]) == (res["accepted"], res["rejected_at"],
                                                                         res["bytes"]), (name, wire)
            assert set(off["timings"]) == set(res["timings"])


def test_a_tampered_query_is_rejected_at_the_same_check(monkeypatch):
    for name, prover, v, x in _queries():
        p = prover()
        victim = p.graph.mat_ops[len(p.graph.mat_ops) // 2].name

        def tamper(o, z, victim=victim):
            if o.name == victim:
                z = z.clone()
                z.view(-1)[0] += 1
            return z

        seed = None if v.params.fiat_shamir else 1
        res = proto.run_query(p, v, x, seed=seed, wire=True, forward_kwargs={"tamper": tamper})
        with monkeypatch.context() as m:
            m.setattr(contention, "begin", lambda: None)
            off = proto.run_query(prover(), v, x, seed=seed, wire=True, forward_kwargs={"tamper": tamper})
        assert not res["accepted"] and res["rejected_at"] == off["rejected_at"], name
        assert res["bytes"] == off["bytes"]


# ---------------------------------------------------------------- bench.py records and their readers
def _bench_run(monkeypatch, root: Path):
    import pvi.fullcheck.transformer as tr
    from test_plans import _TINY

    bench = _script("4_defence_benchmark", "bench")
    monkeypatch.setitem(tr.CONFIGS, "tiny-gpt", _TINY["gpt"])
    monkeypatch.setattr(bench, "RAW", root)
    monkeypatch.setattr(bench, "WIRE", True)
    env = {"device": "cpu", "cpu": "test", "torch_threads": 1}
    monkeypatch.setattr(bench, "TAG", "_wire_polR8c")
    bench.suite_llm(bench.build_parser().parse_args(
        ["llm", "--model", "tiny-gpt", "--seq", "9", "--builds", "1,full", "--modes", "C:int,C:fs", "--lams", "40",
         "--queries", "2", "--policy", "R8c", "--wire", "--llm-tampers", "1", "--batches", "2"]), env)
    monkeypatch.setattr(bench, "TAG", "_wire_lookups")
    bench.suite_llm(bench.build_parser().parse_args(
        ["llm", "--model", "tiny-gpt", "--seq", "9", "--builds", "full", "--modes", "Kpre:int", "--lams", "40",
         "--queries", "2", "--lookups", "--wire"]), env)
    return bench


def _strip(root: Path, out: Path) -> None:
    """A copy of the raw tree without the new field."""
    for p in root.rglob("*"):
        q = out / p.relative_to(root)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.suffix == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            for r in rows:
                r.pop("contention", None)
            q.write_text("".join(json.dumps(r) + "\n" for r in rows))
        else:
            q.write_bytes(p.read_bytes())


def test_bench_writes_the_field_on_every_timing_row_and_the_readers_ignore_it(monkeypatch, tmp_path):
    root = tmp_path / "raw_test"
    _bench_run(monkeypatch, root)
    rows = [json.loads(line) for p in root.rglob("*.jsonl") for line in p.read_text().splitlines()]
    timed = [r for r in rows if r["unit"] == "s"]
    assert timed and all(isinstance(r.get("contention"), dict) for r in timed if r["value"] > 0)
    assert all("contention" not in r for r in rows if r["unit"] != "s")
    names = {r["metric"] for r in timed if "contention" in r}
    assert {"commit_total", "build_graph", "verifier_precompute", "prove_encode", "verify_decode"} <= names
    assert any(r.get("batch") == 2 and "contention" in r for r in timed)
    # aggregate.py: the same summary and full-model rows as without the field
    stripped = tmp_path / "raw_stripped"
    _strip(root, stripped)
    agg = _script("5_comparison", "aggregate")
    out = {}
    for label, raw in (("with", root), ("without", stripped)):
        monkeypatch.setattr(agg, "RAW", raw)
        loaded = agg.load_rows()
        summary = agg.summarise(loaded)
        out[label] = (summary, agg.llm_full(summary))
        agg._write(tmp_path / f"tables_{label}" / "measured_summary.csv", summary)
    assert out["with"] == out["without"] and out["with"][0]
    assert (tmp_path / "tables_with" / "measured_summary.csv").read_bytes() == \
        (tmp_path / "tables_without" / "measured_summary.csv").read_bytes()
    # fingerprint_check.py and count_outcomes.py: the same facts, and the check passes
    fp = _script("5_comparison", "fingerprint_check")
    a, b = fp.load(root), fp.load(stripped)
    assert {k: dict(v) for k, v in a.items()} == {k: dict(v) for k, v in b.items()} and a
    monkeypatch.setattr(sys, "argv", ["fingerprint_check.py", str(stripped), str(root)])
    with pytest.raises(SystemExit) as exc:
        fp.main()
    assert exc.value.code == 0
    co = _script("5_comparison", "count_outcomes")
    ca, cb = co.count(root), co.count(stripped)
    assert ca.keys() == cb.keys() and ca["records"]
    for k in ca:
        assert ca[k] == cb[k], k


def test_records_without_the_field_still_read(monkeypatch, tmp_path):
    """The earlier roots (no ``contention`` anywhere) and a root mixing old and new cells aggregate as before."""
    root = tmp_path / "raw_mixed"
    _bench_run(monkeypatch, root)
    stripped = tmp_path / "raw_old"
    _strip(root, stripped)
    one = next(stripped.rglob("defence_*lookups.jsonl"))       # one old-style cell among new ones
    target = root / one.relative_to(stripped)
    target.write_bytes(one.read_bytes())
    agg = _script("5_comparison", "aggregate")
    monkeypatch.setattr(agg, "RAW", root)
    s1 = agg.summarise(agg.load_rows())
    monkeypatch.setattr(agg, "RAW", stripped)
    s2 = agg.summarise(agg.load_rows())
    assert s1 == s2


def test_record_query_copies_without_sharing():
    """``_record_query`` writes each phase's own evidence (a copy in the JSON line, never another phase's)."""
    bench = _script("4_defence_benchmark", "bench")
    lines = []

    class R:
        def rec(self, metric, value, unit="", trial=None, **extra):
            lines.append(json.loads(json.dumps(dict(metric=metric, value=value, unit=unit, trial=trial, **extra),
                                               default=float)))

    res = {"timings": {"prove_forward": 0.5, "verify_fold": 0.0}, "bytes": {"claims": 3},
           "contention": {"prove_forward": {"wall_s": 0.5, "wait_ns": 7}}, "accepted": True}
    bench._record_query(R(), copy.deepcopy(res), 4, batch=2)
    by = {r["metric"]: r for r in lines}
    assert by["prove_forward"]["contention"] == {"wall_s": 0.5, "wait_ns": 7} and by["prove_forward"]["batch"] == 2
    assert "contention" not in by["verify_fold"] and "contention" not in by["bytes_claims"]
    assert "contention" not in by["accepted"]


def test_env_records_whether_schedstat_exists():
    pytest.importorskip("torchvision")
    bench = _script("4_defence_benchmark", "bench")
    assert bench._env_extra()["schedstat"] is contention.available()
