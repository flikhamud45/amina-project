"""Plan 2 F6 part 11 (H3): ``experiments/5_comparison/sp2027_numbers.py`` parses every tier's JSON lines into the
paper's tables and text numbers.

* Synthetic records with the fields each script writes (perf.py, v1_bytes.py, reexec.py, serve_ab.py, gen_bench.py,
  setup_timing.py, h2d_bandwidth.py, ppl_llama.py), in job logs named as sp2027.sh names them: each lands in its
  tier and kind, the tables fill, the abstract's N is the floor of the smaller Kpre ratio against the strongest
  exact baseline, a failed record is reported, and a dry-run listing gives the job coverage.
* The stored records (``--stored``): tab:eval-llm's base columns and tab:eval-compare's "ours" columns regenerate
  exactly the numbers the paper prints, and the "new" columns read from the same runs equal the base ones.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.filterwarnings("ignore:.enablePackrat. deprecated")   # matplotlib's own pyparsing call
CODE = Path(__file__).resolve().parents[1]
SCRIPT = CODE / "experiments" / "5_comparison" / "sp2027_numbers.py"


def _mod():
    spec = importlib.util.spec_from_file_location("sp2027_numbers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sp2027_numbers"] = mod
    spec.loader.exec_module(mod)
    return mod


sn = _mod()


def _perf(model, seq, mode, verify, proof, fs=True, vdev="cpu", layers=None, label="cpuv", accepted=5):
    return {"label": label, "model": model, "seq": seq, "layers": layers, "mode": mode, "fs": fs,
            "verifier_device": vdev, "prove_s": 1.0, "verify_s": verify, "prove_s_range": [0.9, 1.1],
            "verify_s_range": [verify * 0.95, verify * 1.05], "proof_bytes": proof, "timings": {"verify_derive": verify},
            "bytes": {"claims": proof}, "accepted": accepted, "queries": 5,
            "peak": {"host_peak_rss": 5e10, "gpu_peak_memory": 2e10}}


def _reexec(model, seq, variant, full, device="cpu", n_layers=32):
    return {"model": model, "seq": seq, "variant": variant, "device": device, "n_layers": n_layers,
            "measured_s": {"2": full / 16, str(n_layers): full}, "range_s": {str(n_layers): [full, full * 1.02]},
            "full_s": full, "peak": {"host_peak_rss": 3e10}}


def _v1(model, seq, mode, layers, v0, v1, timed=True):
    q = lambda p, v: {"prove_s": p, "verify_s": v, "fs_s": 0.1, "prove_s_range": [p, p], "verify_s_range": [v, v],
                      "accepted": 5}      # noqa: E731
    rec = {"model": model, "seq": seq, "mode": mode, "layers": layers, "v0_total": v0, "v1_total": v1,
           "v0_accepted": True, "accounting_ok": True, "real": None, "peak": {}}
    if timed:
        rec["timed"] = {"queries": 5, "v0": q(5.0, 19.0), "v1": q(30.0, 40.0),
                        "v1_paths": {"0": q(30.0, 70.0), "1": q(30.0, 40.0)}, "v1_path": "1"}
    return rec


def _write(folder: Path, name: str, records: list[dict], extra: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("host n-1 git abc\n" + extra + "".join(json.dumps(r) + "\n" for r in records),
                               encoding="utf-8")


def _logs(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    _write(logs, "sp-c-llama2-7b-64-101.out", [_perf("llama2-7b", 64, "Kpre", 1.4, 196e6),
                                               _perf("llama2-7b", 64, "C", 1.9, 383e6)])
    _write(logs, "sp-c-llama2-7b-2048-102.out", [_perf("llama2-7b", 2048, "Kpre", 19.6, 6.43e9),
                                                 _perf("llama2-7b", 2048, "C", 25.0, 6.6e9)])
    _write(logs, "sp-c-rx-llama2-7b-103.out", [_reexec("llama2-7b", s, v, f) for s, v, f in (
        (64, "int", 7.26), (64, "int_fp32gemm", 4.08), (64, "float", 4.53), (64, "bf16", 19.5), (64, "dynint8", 1.91),
        (2048, "int", 179.4), (2048, "int_fp32gemm", 141.6), (2048, "float", 93.2), (2048, "bf16", 306.5),
        (2048, "dynint8", 56.0))])
    _write(logs, "sp-c-v1-llama2-7b-2048-104.out", [_v1("llama2-7b", 2048, "Kpre", 32, 6.43e9, 2.9e9)])
    _write(logs, "sp-v1-llama7-2048-105.out", [_v1("llama2-7b", 2048, "C", 32, 6.6e9, 3.1e9)])
    _write(logs, "sp-g-rx-llama2-7b-106.out", [_reexec("llama2-7b", 2048, "int", 1.1, "cuda"),
                                               _reexec("llama2-7b", 2048, "float", 0.5, "cuda")])
    _write(logs, "sp-serve-llama7-107.out", [{
        "row": "prove", "model": "llama2-7b", "layers": None, "n_layers": 32, "seq": 2048, "batch": 1, "mode": "Kpre",
        "batch_proof": "single", "u1_s": {"median": 1.0, "ci95": [0.9, 1.1]},
        "u2_s": {"fp16": {"median": 0.5, "ci95": [0.4, 0.6]}}, "prove_s": {"median": 3.0, "ci95": [2.9, 3.1]},
        "prover_over_u1": {"ratio": 3.0, "ci95": [2.8, 3.2]}, "prover_over_u2": {"fp16": {"ratio": 6.0, "ci95": [5, 7]}},
        "proof_mb_per_token": 3.1, "egress_cap_tokens_per_s_per_100gbps": 4000.0, "ok": True},
        {"row": "on_request", "model": "llama2-7b", "seq": 2048, "batch": 1, "serve_s": {"median": 1.0}}])
    _write(logs, "sp-gen-llama2-7b-108.out", [{
        "kind": "gen", "model": "llama2-7b", "layers": None, "prompt": 512, "gen": 256, "decode": "sample",
        "mode": "Kpre", "prove_s": 4.0, "prove_s_range": [3.9, 4.2], "verify_s": 20.0, "verify_s_range": [19, 21],
        "proof_bytes": 8e9, "proof_mb_per_token": 31.25, "ok": True, "accepted": 4, "queries": 3,
        "tampers": {"wrong_token": {"rejected": True, "label": "token"},
                    "forged_logits": {"rejected": True, "label": "freivalds"}}, "peak": {}}])
    _write(logs, "sp-setup-llama2-7b-109.out", [{
        "kind": "setup", "model": "llama2-7b", "layers": None, "amortise": 100, "proof_bytes": 2.83e9, "commit_s": 500.0,
        "prove_s": 120.0, "verify_s": 60.0, "query_columns_bytes_honest": 124.7e6,
        "query_columns_bytes_untrusted": 168.5e6, "accepted": True, "peak": {}}])
    _write(logs, "sp-g-h2d-110.out", [{"kind": "h2d", "gpu": "NVIDIA L40S", "bytes": 1 << 30, "direction": "h2d",
                                       "memory": "pinned", "s": 0.05, "gb_s": 21.5}])
    _write(logs, "sp-q-llama13-111.out", [
        {"model": "meta-llama/Llama-2-13b-hf", "seq": 2048, "windows": 140, "cell": "reference", "ppl": 4.88},
        {"model": "meta-llama/Llama-2-13b-hf", "seq": 2048, "windows": 140, "cell": "smooth", "ppl": 5.1,
         "ratio": 5.1 / 4.88, "top1": 0.9}])
    _write(logs, "sp-l-llama7-4096-L2-112.out", [_perf("llama2-7b", 4096, "Kpre", 4.0, 1e9, layers=2, label="long")])
    _write(logs, "sp-l-llama7-4096-L3-113.out", [_perf("llama2-7b", 4096, "Kpre", 5.5, 1.4e9, layers=3, label="long")])
    _write(logs, "sp-l-llama7-8192-L2-114.out", [_perf("llama2-7b", 8192, "Kpre", 12.0, 2e9, layers=2, label="long")])
    _write(logs, "sp-l-llama7-8192-L3-115.out", [_perf("llama2-7b", 8192, "Kpre", 17.0, 2.8e9, layers=3, label="long")])
    return logs


def test_every_record_lands_in_its_tier_and_kind(tmp_path):
    problems = []
    recs = sn.read_logs(sn.log_files(_logs(tmp_path)), problems)
    assert not problems
    got = {(r.tier, r.kind) for r in recs}
    assert got == {("cpuv", "perf"), ("cpuv", "reexec"), ("cpuv", "v1"), ("v1", "v1"), ("gpuv", "reexec"),
                   ("serve", "serve"), ("gen", "gen"), ("setup", "setup"), ("gpuv", "h2d"), ("quality", "ppl"),
                   ("long", "perf")}
    assert not [m for r in recs for m in sn.checks(r)]


def test_the_tables_and_the_abstracts_n(tmp_path):
    recs = sn.read_logs(sn.log_files(_logs(tmp_path)), [])
    macros = {}
    opts = sn.tab_options(recs, macros)
    # the strongest exact baseline is int_fp32gemm at 64 tokens (4.08 s) and fp32 at 2,048 (93.2 s); dynint8 is
    # faster but not exact
    cells = {(c["row"], c["column"]): c["value"] for c in opts.cells}
    assert cells["Llama-2-7B (64)", "strongest exact re-execution (CPU)"] == 4.08
    assert cells["Llama-2-7B (2048)", "strongest exact re-execution (CPU)"] == 93.2
    assert cells["Llama-2-7B (2048)", "int8 re-execution (GPU)"] == 1.1
    assert macros["CpuRatioN"] == "2"                       # floor(min(4.08 / 1.4, 93.2 / 19.6)) = floor(2.91)
    assert cells["Llama-2-7B (2048)", "V1 / v0 CPU verifier"] == pytest.approx(40.1 / 19.1)
    v1 = sn.tab_eval_v1(recs, macros)
    assert len(v1.rows) == 2 and not v1.missing
    assert any(k.startswith("Crossover") for k in macros)
    reexec = sn.tab_eval_reexec(recs)
    assert any("0.50" in r or "0.5 " in r for r in reexec.rows if r.startswith("Llama-2-7B"))
    serve = sn.tab_serve(recs, macros)
    assert len(serve.rows) == 1 and macros["ServeLlamaTwoSevenBTwoZeroFourEightKpreOverUOne"] == "3.0"
    gen = sn.tab_gen(recs, macros)
    assert "wrong_token: token" in gen.rows[0] and "forged_logits: freivalds" in gen.rows[0]
    setup = sn.tab_setup(recs, macros)
    assert "2.8\\,GB" in setup.rows[0] and macros["SetupLlamaTwoSevenBAOneZeroZeroProver"] == "2.0\\,min"
    long = sn.tab_long(recs, macros)
    assert len(long.rows) == 2 and float(macros["LongFitMaxResidual"]) < 1e-9     # two points: an exact fit
    assert sn.tab_h2d(recs, macros).rows and macros["HtoDPinnedGBs"] == "21.5"
    quality = sn.tab_quality(recs)
    assert quality.rows and "1.045" in quality.rows[0]
    memory = sn.tab_memory(recs)
    assert memory.filled > 0
    spread = sn.tab_spread(recs)
    assert any("[" in r for r in spread.rows)


def test_a_failed_record_is_reported(tmp_path):
    logs = tmp_path / "logs"
    _write(logs, "sp-c-llama2-7b-64-201.out", [_perf("llama2-7b", 64, "Kpre", 1.4, 196e6, accepted=4)])
    bad = dict(_v1("opt-1.3b", 2048, "C", 24, 2e9, 1e9), accounting_ok=False)
    _write(logs, "sp-v1-opt13-2048-202.out", [bad], extra="{not json\n")
    problems = []
    recs = sn.read_logs(sn.log_files(logs), problems)
    assert any(p.startswith("PARSE") for p in problems)
    failed = [m for r in recs for m in sn.checks(r)]
    assert "accepted 4 of 5" in failed and "accounting_ok false" in failed


def test_job_coverage_from_a_dry_run_listing(tmp_path):
    logs = _logs(tmp_path)
    (logs / "l40s_sp2027-c-gpt2-64-300.out").write_text("...\ndone in 12s\n", encoding="utf-8")
    listing = tmp_path / "dry.txt"
    listing.write_text("sbatch -o x -J sp-c-llama2-7b-64 --wrap \"...\"\n"
                       "sbatch -o x -J sp-c-qwen3-4b-64 --wrap \"...\"\n"
                       "sbatch -o x -J l40s_sp2027-c-gpt2-64 --time=20 bench.sbatch llm\n"
                       "sbatch -o x -J l40s_sp2027-k-gpt2-64 --time=20 bench.sbatch llm\n# tier ...\n", encoding="utf-8")
    recs = sn.read_logs(sn.log_files(logs), [])
    names = sn.expected_jobs(listing, "l40s_sp2027")
    assert sn.job_coverage(names, logs, recs) == ["sp-c-qwen3-4b-64", "l40s_sp2027-k-gpt2-64"]


def test_the_stored_records_regenerate_the_papers_tables(capsys):
    pytest.importorskip("matplotlib")
    args = SimpleNamespace(stored=True, logs="", platform="", base="", policy="auto", challenges="int",
                           out="", paper=str(CODE.parent / "paper_sp2027" / "sections"), expect=None, check=True,
                           strict=False, rehearsal=False, compare_paper=True, verbose=False)
    status = sn.build(args)
    out = capsys.readouterr().out
    assert status == 0, out
    assert "tab:eval-llm against the paper: 63 base cells and proofs identical, 0 differ" in out
    assert "tab:eval-compare (ours) against the paper: 30 cells identical, 0 differ" in out
    assert "tab:eval-llm new columns from the same runs: 0 cells differ from the base ones" in out
