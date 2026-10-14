"""The benchmark tables (``experiments/comparison/aggregate.py``), the outcome counts (``count_outcomes.py``) and
the selection of benchmark cells (``experiments/paper/bench.py``).

The tables count the streaming verifier's time (``verify_total``), and only once: ``bench.py --verifier-impl stream``
records one ``verify_total`` per query instead of the verify phases it overlaps; ``aggregate.py`` and
``paper/bench.py`` must carry it into the tables and refuse a row that has both.  The counts keep the defining cells
only when asked, and on the stored runs of the base implementation they are the paper's.
"""

from __future__ import annotations

import csv
from collections import Counter

import pytest

from experiments.comparison import aggregate
from experiments.comparison import count_outcomes as co
from experiments.common.paths import BENCH
from experiments.paper import bench as B

STORED_RUNS = ("l40s_improved", "l40s_improved2", "l40s_improved3")
"""The stored runs of the base implementation, in order (a later run's cell wins)."""


def test_verify_total_is_carried_into_the_tables():
    assert "verify_total" in aggregate.ADDITIVE


def test_every_timing_of_run_query_is_carried_into_the_tables():
    """Every timing ``run_query`` records -- modes C, K and Kpre, a ``c`` policy's lookup tables and a K
    verifier's own rows, with and without wire, batched and streaming -- is additive in aggregate.py and part of
    paper/bench.py's prover or verifier time (once: never both), and the batched verify phases the streaming
    verifier's ``verify_total`` covers are those ``OVERLAPPED`` names."""
    import pvi.v0 as proto
    from test_plans import _planned, _verifiers

    graph, x, coms = _planned("gpt", "R8c")
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    verifiers = [*_verifiers(graph, coms, False), *_verifiers(graph, coms, True)]
    for mode in ("K", "Kpre"):
        for lookups in (False, True):
            for stream in (False, True):
                v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), mode, weights=weights,
                                   lookups=lookups, stream=stream)
                if mode == "Kpre":
                    v.precompute(proto.Challenger(seed=1))
                verifiers.append(v)
    keys = {False: set(), True: set()}                 # by stream
    for v in verifiers:
        for wire in (False, True):
            res = proto.run_query(proto.Prover(graph, commitments=coms), v, x, seed=None if v.params.fiat_shamir
                                  else 1, wire=wire)
            assert res["accepted"]
            keys[v.stream] |= set(res["timings"])
    emitted = keys[False] | keys[True]
    assert {"prove_lookups", "verify_lookups", "verify_total"} <= emitted
    assert emitted <= aggregate.ADDITIVE
    assert emitted <= set(B.PROVE) | set(B.VERIFY) and set(B.PROVE) & set(B.VERIFY) == {"fs_hash"}
    batched_only = {k for k in keys[False] - keys[True] if k.startswith("verify_")}
    assert batched_only <= set(B.OVERLAPPED) and not keys[True] & set(B.OVERLAPPED)
    assert set(B.OVERLAPPED) - batched_only == {"verify_upload"}    # (a GPU client's: none on the CPU)


LLM_HEAD = ["model", "seq", "mode", "challenges", "lam", "metric", "threads", "rate", "variant", "device", "prover_hw",
            "verifier_hw", "n_layers_full", "params_full", "value", "provenance", "batch"]


def _tables(tmp_path, rows: list[list], name: str = "t") -> B.Bench:
    d = tmp_path / f"tables_{name}"
    d.mkdir()
    with open(d / "llm_full_model.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(LLM_HEAD)
        w.writerows(rows)
    return B.Bench(d, "cnn18c")


def _row(metric, value, variant="_wire_gpuv_stream_prune_polcnn18c", provenance="measured", ch="fs"):
    return ["llama2-7b", 2048, "C", ch, 128, metric, 8, 4, variant, "cuda", "NVIDIA L40S",
            "NVIDIA L40S (GPU client) + AMD EPYC 9334 x8 threads", 32, 6738149376, value, provenance, ""]


def test_the_streaming_verifier_is_timed_once(tmp_path):
    base = [_row("prove_forward", 2.0), _row("prove_fold", 0.5), _row("fs_hash", 0.25), _row("bytes_total", 1000.0)]
    b = _tables(tmp_path, base + [_row("verify_total", 3.0)])
    c = b.llm_cost("llama2-7b", 2048, "c_gpu")
    assert (c["prove"], c["verify"], c["bytes"]) == (2.75, 3.25, 1000.0)
    assert c["cell"] == "defence_C_fs_lam128_T2048_L32_wire_gpuv_stream_prune_polcnn18c"
    both = _tables(tmp_path, base + [_row("verify_total", 3.0), _row("verify_products", 1.5)], "both")
    with pytest.raises(ValueError):
        both.llm_cost("llama2-7b", 2048, "c_gpu")


def test_a_cell_is_selected_by_its_settings_only(tmp_path):
    """A cell of other challenges, another plan, extrapolated rows or a batch never stands in for the cell."""
    rows = [_row("prove_forward", 1.0, "_wire_prune_polcnn18c"), _row("verify_products", 2.0, "_wire_prune_polcnn18c"),
            _row("prove_forward", 9.0, "_wire_prune_polauto"), _row("prove_forward", 8.0, "_wire_prune_polcnn18c",
                                                                     ch="int"),
            _row("prove_forward", 7.0, "_wire_prune_lookups", "extrapolated_from_1_and_2_blocks")]
    b = _tables(tmp_path, rows)
    assert b.llm_cost("llama2-7b", 2048, "c_cpu")["prove"] == 1.0
    assert b.llm_cost("llama2-7b", 2048, "c_cpu_int")["prove"] == 8.0
    assert b.llm_cost("llama2-7b", 2048, "kpre_cpu") is None          # extrapolated: not a full build
    assert B.tagset("_wire_gpuv_prune_polcnn18c") == B.tagset("_gpuv_wire_polcnn18c_prune")


def test_count_macros_refuse_an_accepted_attack():
    c = {"honest": Counter({("cnn", 1): 3}), "attacks": Counter({("cnn", 1): 2}), "stages": Counter({"freivalds": 2}),
         "by_type": Counter(), "policy": Counter(), "partial": Counter(), "records": 5, "rejected": []}
    m = co.macros(c, "Opt")
    assert m["OptNHonest"] == 3 and m["OptNAttacks"] == 2 and "OptNAttacksPlans" not in m
    c["attacks"][("llm", 0)] += 1
    with pytest.raises(SystemExit):
        co.macros(c, "")


def test_counts_of_several_runs_add_up():
    one = {"honest": Counter({("cnn", 1): 3}), "attacks": Counter({("cnn", 1): 2}), "stages": Counter({"freivalds": 2}),
           "by_type": Counter(), "policy": Counter(), "partial": Counter(), "records": 5, "rejected": []}
    c = co.merge([one, one])
    assert co.macros(c, "Final")["FinalNAttacks"] == 4 and c["records"] == 10


def test_counts_can_keep_the_definition_cells_only():
    d = B.definition_tagsets("auto")
    assert co.cell_tags("tamper_C_int_lam40_T64_L32_wire_prune_polauto") == ("C", {"wire", "prune", "polauto"})
    assert co.cell_tags("commit_T64_L12") == (None, None)
    keep = lambda suite, cell: co._kept(suite, cell, (), d)   # noqa: E731
    assert keep("cnn", "defence_C_int_lam40_rate4_wire_polauto") and keep("cnn", "tamper_C_int_lam40_wire_polauto")
    assert keep("cnn", "defence_Kpre_int_lam128_rate4_wire") and not keep("cnn", "defence_C_int_lam40_rate4_wire")
    assert keep("llm", "defence_C_int_lam128_T2048_L32_wire_gpuv_stream_prune_polauto")
    assert keep("llm", "defence_Kpre_int_lam40_T8_L36_thr1_wire_prune_lookups")
    assert keep("llm", "defence_C_int_lam128_T64_L32_wire_batch_prune_polauto")
    assert not keep("llm", "defence_C_int_lam128_T64_L32_wire_polauto")          # a stand-in, not the definition
    assert not keep("llm", "defence_C_int_lam128_T2048_L32_gpuv_stream")         # the uncompressed proof format
    assert not keep("llm", "defence_Kpre_int_lam40_T8_L36_thr1_prune_lookups")   # Maverick's uncompressed by-product


def test_the_stored_counts_are_the_papers():
    """The defining cells of the stored runs of the base implementation give the counts the paper prints today
    (``paper_sp2027/fallback/numbers.tex``: 3,792 honest queries, 2,512 attacks; 1,601, 853, 56 and 2 by check)."""
    roots = [BENCH / f"raw_{n}" for n in STORED_RUNS]
    if not all((r / "PLATFORM.json").exists() for r in roots):
        pytest.skip("the stored raw runs are not in this checkout")
    m = co.macros(co.count_runs(roots, "auto", (), B.definition_tagsets("auto")), "")
    assert (m["NHonest"], m["NAttacks"], m["NFreivalds"], m["NColumnsCode"], m["NColumnsMerkle"], m["NRange"]) == \
        (3792, 2512, 1601, 853, 56, 2)
