"""``experiments/paper/numbers.py``: the run's records into the paper's table bodies and text numbers.

* Synthetic records of every kind, with the fields each script writes, in job logs named as ``run.sh`` names them
  (plus the attack tier's stored documents and a synthetic aggregated benchmark table): each record lands in its
  tier and kind, and each builder fills its cells from the right records only -- a real-weight generation record
  never fills a random-weight cell, a baseline-verifier or GPU-verifier V1 record never fills the optimised CPU
  column, an alpha-0.55 quality cell never fills the certified row, and the abstract's N is the 2,048-token ratio.
* ``--strict`` fails on a missing cell, a failed check or an unparsable line gives exit status 1, and a dry-run
  listing gives the job coverage (an incomplete listing is refused).
* Exact re-execution is the fastest exact variant; llama.cpp is an inexact baseline of another model, which never
  changes the abstract's N, at its fastest configuration; mode C's plan comes from the records; the bounds round up.
* The 8- against 11-bit A/B of the improve tier gives its two macros, and a cost record whose decoder is not at the
  11-bit attention probabilities the text states is a text check that fails (a warning in a rehearsal).
* A document named by another model (a rehearsal's OPT-125M) counts for its tier and kind and fills no table cell or
  text number.
* The paper's wiring: the generated files and ``paper_sp2027/fallback/`` have the same names and macro sets, every
  macro the paper uses is defined, every table cell agrees with the macro that prints the same value (in the written
  files and in the fallback), and the stored attack documents regenerate the fallback's ``tab:inrange`` and its
  text numbers.

``PVI_SLOW=1`` also runs the check that the library's setup proof gives the fallback's setup-proof values (about ten
minutes on a laptop CPU).
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from experiments.paper import bench as B
from experiments.paper import format as F
from experiments.paper import macros as M
from experiments.paper import numbers as N
from experiments.paper import records as R
from experiments.paper import tables as T

CODE = Path(__file__).resolve().parents[1]
PAPER = CODE.parent / "paper_sp2027"
RESULTS = CODE / "artifacts" / "results"
ENV = {"gpu": "NVIDIA L40S", "cpu": "AMD EPYC 9334 32-Core Processor", "git": "abc1234", "argv": []}


def test_a_rehearsal_reports_only_the_texts_claims_as_warnings():
    failed = ["CHECK FAILED sp-setup-gpt2-1.out:3: setup proof rejected",
              f"{M.TEXT_CLAIM} ZeroshotLambadaOptS is +0.00 points, which the text calls a loss"]
    kept, warned = N.demote_text_claims(failed)
    assert kept == failed[:1]
    assert warned == ["WARNING (rehearsal) text: ZeroshotLambadaOptS is +0.00 points, which the text calls a loss"]


# ------------------------------------------------------------------------------------------------ the records
def _env(argv=()):
    return dict(ENV, argv=list(argv))


def perf(model, seq, mode, verify, proof, *, label="cpuv", layers=None, vdev="cpu", real=None, accepted=5,
         prove=1.0, timings=None, claims=None, tampers=None, stream=False, prob_max=2047, client=None, gpu_peak=None):
    """A ``costs/perf.py`` record (``client``: ``--client-memory``'s peak and baseline in GB)."""
    return {"label": label, "git": "abc1234", "model": model, "real": real, "seq": seq, "layers": layers, "mode": mode,
            "policy": "cnn18c" if mode == "C" else None, "wire": True, "lean": True, "lam": 128, "rate": 4,
            "prune_last": True, "lookups": mode != "C", "prob_max": prob_max, "device": "cuda", "verifier_device": vdev,
            "stream": stream,
            "threads": 8, "accepted": accepted, "queries": 5, "fs": True, "prove_s": prove, "verify_s": verify,
            "prove_s_range": [prove, prove], "verify_s_range": [verify, verify], "proof_bytes": proof,
            "timings": timings or {"verify_derive": verify}, "bytes": {"claims": claims or proof},
            **({"tampers": tampers} if tampers is not None else {}),
            "peak": {"host_peak_rss": 5e10, **({"gpu_peak_memory": gpu_peak} if gpu_peak else {})},
            **({"client_memory": {"peak_rss": client[0] * 1e9, "baseline_rss": client[1] * 1e9,
                                  "key_rss": client[1] * 1e9 + 2e8, "queries": accepted, "accepted": accepted,
                                  "agree": True, "verify_s": verify}} if client else {}),
            "env": _env()}


def reexec(model, seq, variant, full, *, device="cpu", n_layers=32, blocks=None):
    """A ``reexec/reexec.py`` record; ``blocks``: measured seconds by block count (default 2 and the full model)."""
    ms = blocks or {"2": full / 16, str(n_layers): full}
    return {"git": "abc1234", "model": model, "seq": seq, "variant": variant, "dtype": "fp32", "device": device,
            "threads": 8, "reps": 5, "prune_last": True, "prob_max": 2047 if variant in T.EXACT else None,
            "measured_s": ms,
            "range_s": {k: [v, v] for k, v in ms.items()}, "per_block_s": full / n_layers, "fixed_s": 0.0,
            "n_layers": n_layers, "full_s": full, "env": _env()}


def llamacpp(seq, wtype, t, *, fa=1, layers=32, threads=8):
    """A ``reexec/llamacpp.py`` record (written by the script's own :func:`record`) of a llama-bench line whose 5
    samples have the median ``t``."""
    from experiments.reexec import llamacpp as L
    samples = [round(t * x * 1e9) for x in (0.98, 0.99, 1.0, 1.01, 1.02)]
    line = {"build_commit": "d812350", "build_number": L.BUILD_NUMBER, "cpu_info": ENV["cpu"], "backends": "CPU",
            "model_type": f"llama 7B {wtype.upper()}", "model_size": 7.16e9 if wtype == "q8_0" else 1.348e10,
            "model_n_params": 6_738_415_616, "n_batch": seq, "n_ubatch": min(512, seq), "n_threads": threads,
            "cpu_mask": "0x0", "type_k": "f16", "type_v": "f16", "flash_attn": fa, "load_mode": "none",
            "n_prompt": seq, "n_gen": 0, "n_depth": 0, "samples_ns": samples}
    return L.record(line, model="llama2-7b", n_layers=32, layers=layers, wtype=wtype, threads=8, reps=5,
                    llamacpp={"tag": L.TAG, "commit": L.COMMIT}, extra={"git": "abc1234",
                                                                        "env": dict(_env(), cpu_affinity=8)})


def v1(model, seq, mode, layers, v0_total, v1_total, *, label="", real=None, argv=(), v0=(1.0, 5.0),
       opt=(4.0, 9.0), base=(4.0, 15.0), gkr=3.0):
    """A ``v1/v1_bytes.py --full --reps`` record (``timed.v1_paths`` by verifier)."""
    q = lambda p, v, t=None: {"accepted": 3, "timings": t or {}, "prove_s": p, "verify_s": v,    # noqa: E731
                              "prove_s_range": [p, p], "verify_s_range": [v, v], "fs_s": 0.1}
    return {"label": label, "git": "abc1234", "model": model, "real": real, "layers": layers, "seq": seq,
            "prune_last": True, "prob_max": 2047, "lmax": 27, "env": _env(argv), "mode": mode, "fs": True,
            "v1_verifier": "optimised",
            "v0_accepted": True, "v0_bytes": {"claims": 0.9 * v0_total},
            "v1_bytes": {"claims": 0.4 * v0_total, "cut_exc": 1.8e6, "cut_mult": 3.1e5, "cut_gkr": 5.249e5},
            "v0_total": v0_total, "v1_total": v1_total, "ratio": v0_total / v1_total, "n_leaves": 2_714_763_264,
            "n_instances": 156, "table": 553_917, "n_exceptions": 224_670,
            "timed": {"queries": 3, "v0": q(*v0), "v1": q(*opt, {"prove_gkr": gkr}),
                      "v1_paths": {"optimised": q(*opt, {"prove_gkr": gkr}), "baseline": q(*base, {"prove_gkr": gkr})},
                      "v1_path": "optimised"},
            "accounting": {}, "accounting_ok": True, "peak": {}}


def serve(model, seq, batch, mode, how, prove, u1, u2, *, mb=4.25):
    """A ``deployment/serve_ab.py`` prove line."""
    return {"label": "serve", "model": model, "layers": None, "n_layers": 24, "seq": seq, "batch": batch,
            "row": "prove", "mode": mode, "batch_proof": how, "u1_s": {"median": u1, "ci95": [u1, u1]},
            "u2_s": {"fp16": {"median": u2, "ci95": [u2, u2]}}, "prove_s": {"median": prove, "ci95": [prove, prove]},
            "prover_over_u1": {"ratio": prove / u1, "ci95": [0, 0]},
            "prover_over_u2": {"fp16": {"ratio": prove / u2, "ci95": [0, 0]}},
            "u1_over_u2": {"fp16": {"ratio": u1 / u2, "ci95": [0, 0]}}, "proof_mb_per_token": mb, "ok": True,
            "prob_max": 2047, "env": _env()}


def gen(model, mode, decode, g, prove, *, real=None, label="gen"):
    """A ``generation/gen_bench.py`` record."""
    return {"kind": "gen", "label": label, "model": model, "layers": None, "n_layers": 32, "prompt": 512, "gen": g,
            "decode": decode, "mode": mode, "real": real, "fs": True, "verifier_device": "cpu", "queries": 3,
            "accepted": 4, "prove_s": prove, "prove_s_range": [prove, prove], "verify_s": 2 * prove,
            "verify_s_range": [2 * prove, 2 * prove], "proof_bytes": 1e9, "proof_mb_per_token": 1e3 / g,
            "tampers": {"wrong_token": {"rejected": True, "label": "token", "expected": ["token"]}}, "ok": True,
            "prob_max": 2047, "env": _env()}


def ppl(model, cell, ppl_, ratio, *, split="test", calibration="train", seq=2048, draw=0, mass=0.975, top1=0.9,
        ci=(1.0, 1.1), alpha=None):
    """A ``quality/ppl_opt.py`` record (``cell`` ``fp32``: the reference)."""
    head = {"model": model, "tag": "", "seq": seq, "windows": 140, "tokens": 286_580, "split": split,
            "calibration": calibration, "calib_windows": 4, "env": _env()}
    if cell == "fp32":
        return dict(head, cell="fp32", ppl=ppl_, nll=[1.0])
    return dict(head, cell=cell, graph=cell.split("-")[0], alpha=alpha, pbits=int(cell[-2:].strip("p")), draw=draw,
                ppl=ppl_, fp32=ppl_ / ratio, ratio=ratio, ratio_ci95=list(ci), top1=top1, mass_min=mass, nll=[1.0])


def zeroshot(model, task, variant, acc, *, diff=None, recipe="certified"):
    """A ``quality/zeroshot.py`` stdout line (without the per-document lists): both variants carry the job's recipe
    (the script's head, written once the integer model is built)."""
    row = {"model": model, "tag": "", "env": _env(), "task": task, "variant": variant, "acc": acc, "recipe": recipe}
    if variant == "int":
        row.update(diff_acc=diff, diff_acc_ci95=[diff - 0.01, diff + 0.01], same_pick=0.9)
    return row


def llama_ppl(model, cell, ratio, *, split="test", ci=None, ref_dtype="float32", alpha=None, balance=None,
              ppl_=None):
    """A ``quality/ppl_llama.py`` record (``cell`` ``reference``: the reference)."""
    head = {"model": model, "smoke": False, "config": "llama2-7b", "split": split, "seq": 2048, "windows": 160,
            "ref_dtype": ref_dtype, "env": _env()}
    if cell == "reference":
        return dict(head, cell="reference", ppl=ppl_ or 6.0, nll=[1.0])
    return dict(head, cell=cell, graph=cell.split("-")[0], alpha=alpha, glu_smooth=balance, glu_exp=0, draw=0,
                ppl=(ppl_ or 6.0) * ratio, ref_ppl=ppl_ or 6.0, ratio=ratio, ratio_ci95=list(ci or (ratio, ratio)),
                top1=0.9)


def digest_rec(model, threads, *, device=None, digest_="d1", weights="w1", roots="r1"):
    """A ``quality/digest_check.py`` record (``device``: where the commitment is encoded; None: no commitment)."""
    row = {"model": model, "recipe": "certified", "threads": threads, "host": "n-1", "cpu": "x86_64",
           "torch": "2.5.1", "digest": digest_, "weights_sha256": weights, "env": _env()}
    if device:
        row.update(policy="cnn18c", rate=4, commit_device=device, roots_sha256=roots, n_roots=225)
    return row


ATTRIBUTION = {"weights": (72.5409, 66.6436), "w_emb": (75.5369, 64.8201), "w_qkv": (74.7597, 64.9653),
               "w_out": (74.7144, 65.1556), "w_fc1": (74.7254, 65.2792), "w_fc2": (74.8286, 65.0959),
               "w_head": (75.5774, 64.7015), "norm": (69.3907, 70.3329), "norm1": (70.9628, 68.6896),
               "norm2": (74.9013, 65.9101), "normf": (75.4596, 64.9366), "rq_qkv": (75.1141, 65.2619),
               "rq_fc1": (72.6652, 66.2708), "softmax": (75.8567, 64.9330), "att_out": (75.7001, 64.8082),
               "res": (75.8797, 64.7651), "relu": (75.5198, 64.7611)}
"""The attribution study's leave-one-out and add-one perplexities per group (OPT-125M, 16 windows of 128 tokens)."""


def attribution(*, digest="d125", int_ppl=75.5198, fp32=64.7611, ok=True, rows=ATTRIBUTION, sub=None, windows=16):
    """``quality/attribution.py``'s records: a row per group and mode, the feature sub-rows ``sub`` (``{key: add-one
    perplexity}``) and the check record."""
    head = {"kind": "attribution", "model": "facebook/opt-125m", "tag": "", "seq": 128, "windows": windows,
            "tokens": 127 * windows, "calibration": "test-end", "graph": "scalar", "graph_cell": "scalar-g4-p8",
            "env": _env()}
    out = [dict(head, part="row", group="all", mode="all_int", ppl=int_ppl),
           dict(head, part="row", group="all", mode="all_float", ppl=fp32 + 1e-6)]
    for g, (loo, add) in rows.items():
        out += [dict(head, part="row", group=g, mode="leave_one_out", ppl=loo),
                dict(head, part="row", group=g, mode="add_one", ppl=add)]
    for g, add in (sub or {"norm_round": 70.3329, "norm_clip": 64.8607}).items():
        out.append(dict(head, part="row", group=g, mode="add_one_feature", ppl=add))
    out.append(dict(head, part="check", fp32=fp32, int_ppl=int_ppl, all_int_exact=ok, all_int_max_abs_dlogit=0.0,
                    all_float_max_abs_dlogit=3.717e-5, digest=digest, weights_sha256="w", ok=ok))
    return out


def _write(folder: Path, name: str, records, extra: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("host n-1 git abc1234\n" + extra + "".join(json.dumps(r) + "\n" for r in records),
                               encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the run
def _bench(root: Path, platform: str = "test") -> None:
    """A synthetic ``tables_<platform>/`` and ``raw_<platform>/`` (one finished cell)."""
    t = root / f"tables_{platform}"
    t.mkdir(parents=True)
    head = ["model", "seq", "mode", "challenges", "lam", "metric", "threads", "rate", "variant", "device", "prover_hw",
            "verifier_hw", "n_layers_full", "params_full", "value", "provenance", "batch"]
    rows = []
    for m, seq, mode, ch, variant, scale in (
            ("llama2-7b", 64, "C", "fs", "_wire_prune_pmax2047_polcnn18c", 1.0),
            ("llama2-7b", 64, "C", "fs", "_wire_gpuv_stream_prune_pmax2047_polcnn18c", 0.4),
            ("llama2-7b", 64, "Kpre", "fs", "_wire_prune_lookups_pmax2047", 0.7),
            ("llama2-7b", 64, "C", "int", "_wire_prune_pmax2047_polcnn18c", 0.9),   # interactive: not tab:eval-llm's
            ("llama2-7b", 64, "Kpre", "fs", "_wire_prune_lookups", 5.0)):          # 8-bit: not the final run's
        hw = "AMD EPYC 9334 32-Core Processor x8 threads"
        for metric, v in (("prove_forward", 1.5 * scale), ("fs_hash", 0.1), ("verify_products", scale),
                          ("bytes_total", 3.25e8 * scale)):
            rows.append([m, seq, mode, ch, 128, metric, 8, 4, variant, "cuda", "NVIDIA L40S", hw, 32, 6738149376, v,
                         "measured", ""])
    with open(t / "llm_full_model.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    with open(t / "measured_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "model", "cell", "metric", "batch", "attack", "lam", "stage", "k", "n", "median", "mean",
                    "prover_hw", "verifier_hw"])
        cell = "defence_C_fs_lam128_rate4_wire_polcnn18c"
        for metric, v in (("prove_forward", 0.004), ("fs_hash", 0.001), ("verify_products", 0.002),
                          ("bytes_total", 63_300)):
            w.writerow(["cnn", "lenet5", cell, metric, "", "", "", "", "", 3, v, v, "NVIDIA L40S", "AMD EPYC 9334 "
                        "32-Core Processor x8 threads"])
        w.writerow(["cnn", "lenet5", "facts_wire", "n_params", "", "", "", "", "", 1, 61_706, 61_706, "", ""])
        for m in M.CNN_MODELS:              # float and int8 accuracy of every image model (CnnAccuracyChange)
            for metric, v in (("float_accuracy", 0.990), ("int8_accuracy", 0.9866 if m == "vgg16" else 0.989)):
                w.writerow(["cnn", m, "facts_wire", metric, "", "", "", "", "", 1, v, v, "", ""])
        # the batch tier: Llama-2-7B, mode C, 64 tokens; the single cell's claims and B = 4, 16 prompts per opening
        # (per prompt: claims + shared / B, exactly at B = 4 and 1.5% above at B = 16)
        single = "defence_C_fs_lam128_T64_L32_wire_prune_pmax2047_polcnn18c"
        w.writerow(["llm", "llama2-7b", single, "bytes_claims", "", "", "", "", "", 3, 2.0e8, 2.0e8, "", ""])
        for bsz, off in ((4, 1.0), (16, 1.015)):
            total = bsz * (2.0e8 + 1.25e8 / bsz) * off
            w.writerow(["llm", "llama2-7b", "defence_C_fs_lam128_T64_L32_wire_batch_prune_pmax2047_polcnn18c",
                        "bytes_total",
                        bsz, "", "", "", "", 3, total, total, "", ""])
    raw = root / f"raw_{platform}" / "cnn" / "lenet5"
    raw.mkdir(parents=True)
    cell = "defence_C_fs_lam128_rate4_wire_polcnn18c"
    recs = [{"metric": "env"}] + [{"cell": cell, "metric": "accepted", "value": 1} for _ in range(3)]
    (raw / f"{cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (raw / f"{cell}.done").write_text("", encoding="utf-8")
    tam = "tamper_C_fs_lam128_wire_polcnn18c"
    recs = [{"metric": "env"}, {"cell": tam, "metric": "rejected", "value": 1, "attack": "single_value",
                                "stage": "freivalds", "config": {}}]
    (raw / f"{tam}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (raw / f"{tam}.done").write_text("", encoding="utf-8")


def _logs(tmp: Path) -> Path:
    logs = tmp / "logs"
    m = "llama2-7b"
    _write(logs, "sp-smoke-1.out", [v1("gpt2", 64, "C", 2, 1e6, 8e5, argv=["--layers", "2"]),
                                    {"kind": "h2d", "label": "smoke", "gpu": "NVIDIA L40S", "bytes": 1 << 28,
                                     "direction": "h2d", "memory": "pinned", "s": 0.01, "gb_s": 25.0, "env": _env()}],
           extra="4 passed\nint8 GEMMs exact on this GPU: True\n")
    (logs / "sp-smoke-1.out").write_text((logs / "sp-smoke-1.out").read_text(encoding="utf-8") + "SMOKE OK\n",
                                         encoding="utf-8")
    # the client's own process (--client-memory): the peak and the resident set after its imports, in GB
    _write(logs, "sp-c-llama2-7b-64-2.out", [perf(m, 64, "Kpre", 1.4, 1.96e8, client=(1.15, 0.52), gpu_peak=7.4e9),
                                             perf(m, 64, "C", 1.9, 3.25e8, client=(1.42, 0.53))])
    _write(logs, "sp-c-llama2-7b-2048-3.out", [perf(m, 2048, "Kpre", 19.6, 6.43e9, client=(21.4, 0.54)),
                                               perf(m, 2048, "C", 25.0, 6.6e9, client=(23.06, 0.55))])
    # the gate tier (full-size checks before the run): its records fill no cell, however close their settings
    _write(logs, "sp-gate-v1-llama7-2k-2-61.out", [v1(m, 2048, "Kpre", 2, 2.5e8, 1.1e8, label="gate"),
                                                   v1(m, 2048, "C", 2, 2.7e8, 1.3e8, label="gate")])
    _write(logs, "sp-gate-l-llama7-16k-62.out", [perf(m, 16384, "Kpre", 99.0, 9e9, layers=2, label="gate")])
    _write(logs, "sp-c-rx-llama2-7b-4.out", [reexec(m, s, v, t) for s, v, t in (
        (64, "int", 7.26), (64, "int_fp32gemm", 4.08), (64, "float", 4.53), (64, "bf16", 19.5), (64, "dynint8", 1.91),
        (2048, "int", 179.4), (2048, "int_fp32gemm", 141.6), (2048, "float", 93.2), (2048, "bf16", 306.5),
        (2048, "dynint8", 56.0))])
    # llama.cpp (inexact, another model), each type with both attention kernels: slower here than dynamic int8
    _write(logs, "sp-c-llamacpp-llama2-7b-48.out", [llamacpp(s, w, t, fa=fa) for s, w, t, fa in (
        (64, "q8_0", 2.2, 1), (64, "q8_0", 2.4, 0), (2048, "q8_0", 61.3, 1), (2048, "q8_0", 65.0, 0),
        (64, "bf16", 3.1, 0), (2048, "bf16", 88.0, 1))])
    _write(logs, "sp-c-v1-llama2-7b-2048-5.out", [v1(m, 2048, "Kpre", 32, 6.43e9, 2.8e9, label="cpuv-v1",
                                                     v0=(1.0, 19.5), opt=(30.0, 45.3), base=(30.0, 70.0))])
    # the timed full V1 query of OPT-125M in C is 0.34% above the accounting (VOneAccountingWithin rounds it up)
    _write(logs, "sp-v1-opt125-2048-6.out", [dict(v1("opt-125m", 2048, "C", 12, 3.496e8, 1.723e8, v0=(0.4, 2.86),
                                                    opt=(4.75, 4.76), gkr=3.58),
                                                 v1_full={"bytes": {"claims": 1.723e8 * 1.0034}}),
                                             v1("opt-125m", 2048, "Kpre", 12, 3.439e8, 1.667e8)])
    _write(logs, "sp-v1-llama7-2048-7.out", [v1(m, 2048, "C", 32, 6.63e9, 2.97e9), v1(m, 2048, "Kpre", 32, 6.43e9,
                                                                                         2.8e9)])
    # Llama-2-13B on 2 of its 40 blocks: tab:eval-v1's row, never in the full models' range (VOneRatioRandom)
    _write(logs, "sp-v1-llama13-2k-2-47.out", [v1("llama2-13b", 2048, "C", 2, 1.0e8, 5.5e7)])
    # the GPU verifier's V1 cell of the gpuv tier: never a tab:eval-v1 or tab:options cell
    _write(logs, "sp-g-v1-llama2-7b-8.out", [v1(m, 2048, "Kpre", 32, 6.43e9, 1.0e9, label="gpuv-v1",
                                                argv=["--verifier-device", "cuda"], opt=(1.0, 1.0))])
    _write(logs, "sp-g-v0-opt-1.3b-Kpre-9.out", [perf("opt-1.3b", 2048, "Kpre", 1.1, 2e9, vdev="cuda", label="gpuv",
                                                      stream=True)])
    _write(logs, "sp-g-rx-llama2-7b-10.out", [reexec(m, 64, "int", 0.15, device="cuda"),
                                              reexec(m, 64, "float", 0.034, device="cuda"),
                                              reexec(m, 2048, "int", 1.65, device="cuda"),
                                              reexec(m, 2048, "float", 0.64, device="cuda")])
    _write(logs, "sp-serve-opt13-11.out", [serve("opt-1.3b", 2048, 1, "Kpre", "seq", 2.1, 0.51, 0.11),
                                           serve("opt-1.3b", 2048, 1, "C", "seq", 4.4, 0.51, 0.11),
                                           serve("opt-1.3b", 64, 64, "Kpre", "joint", 3.4, 0.71, 0.19),
                                           {"row": "on_request", "model": "opt-1.3b", "seq": 2048, "batch": 1,
                                            "serve_s": {"median": 0.5}, "prob_max": 2047, "env": _env()}])
    _write(logs, "sp-gen-llama2-7b-12.out", [gen(m, mode, d, g, 2.0 if mode == "C" else 1.0)
                                             for mode in ("C", "Kpre") for d in ("greedy", "sample") for g in (64, 256)])
    # Qwen3-4B's sampled responses cost 2.5% more than its greedy ones (GenSampledWithin)
    _write(logs, "sp-gen-qwen3-4b-35.out", [gen("qwen3-4b", mode, d, g, 1.5 * (1.025 if d == "sample" else 1.0))
                                            for mode in ("C", "Kpre") for d in ("greedy", "sample") for g in (64, 256)])
    # GPT-2 after a 64-token prompt (the per-token prover); a cell after another prompt never fills it
    _write(logs, "sp-gen-gpt2-36.out", [dict(gen("gpt2", mode, "greedy", 256, 0.29 if mode == "C" else 0.2),
                                             prompt=64, n_layers=12) for mode in ("C", "Kpre")]
           + [dict(gen("gpt2", "C", "greedy", 256, 5.0), prompt=16, n_layers=12)])
    # the real Llama-2-7B's generation (the real tier): never a cell of tab:gen
    _write(logs, "sp-r-llama7-gen-13.out", [gen(m, "C", "greedy", 64, 99.0, real="meta-llama/Llama-2-7b-hf",
                                                label="real")])
    _write(logs, "sp-r-opt13-v1-Kpre-64-14.out", [v1("opt-1.3b", 64, "Kpre", 24, 5.01e7, 3.91e7, label="real",
                                                     real="facebook/opt-1.3b", v0=(0.20, 0.51))])
    _write(logs, "sp-r-opt13rnd-v1-Kpre-64-15.out", [v1("opt-1.3b", 64, "Kpre", 24, 6.04e7, 3.97e7,
                                                        label="real-random", v0=(0.22, 0.54))])
    real_c = v1("opt-1.3b", 64, "C", 24, 5.2e7, 4.0e7, label="real", real="facebook/opt-1.3b")
    real_c.update(n_exceptions=2379, v1_bytes={"claims": 3.0e7, "cut_exc": 2.1e4})
    _write(logs, "sp-r-opt13-v1-C-64-37.out", [real_c])
    _write(logs, "sp-r-opt13rnd-v1-C-64-38.out", [dict(v1("opt-1.3b", 64, "C", 24, 6.2e7, 4.1e7, label="real-random"),
                                                       n_exceptions=1)])
    # the real Llama-2-7B (with the Llama-2 licence) against its random twins
    llama = "meta-llama/Llama-2-7b-hf"
    tam = {"single_value": {"attempts": 3, "rejected": 3}, "neuron": {"attempts": 6, "rejected": 6}}
    _write(logs, "sp-r-llama7-v0-2048-39.out", [perf(m, 2048, "Kpre", 21.0, 5.35e9, label="real", real=llama,
                                                     tampers=tam)])
    _write(logs, "sp-r-llama7rnd-v0-2048-40.out", [perf(m, 2048, "Kpre", 20.0, 6.45e9, label="real-random")])
    _write(logs, "sp-r-llama7-v0-64-41.out", [perf(m, 64, "Kpre", 1.1, 1.7e8, label="real", real=llama)])
    _write(logs, "sp-r-llama7rnd-v0-64-42.out", [perf(m, 64, "Kpre", 1.0, 2.0e8, label="real-random")])
    _write(logs, "sp-r-llama7-v0C-64-43.out", [perf(m, 64, "C", 1.9, 2.9e8, label="real", real=llama)])
    _write(logs, "sp-r-llama7rnd-v0C-64-44.out", [perf(m, 64, "C", 2.0, 3.25e8, label="real-random")])
    # its multiplicities and GKR transcripts (0.901 MB) are the largest of the run's V1 proofs (VOneExtrasMax)
    real_v1 = v1(m, 2048, "Kpre", 32, 5.35e9, 2.65e9, label="real", real=llama)
    real_v1["v1_bytes"].update(cut_mult=4.0e5, cut_gkr=5.01e5)
    _write(logs, "sp-r-llama7-v1-2048-45.out", [real_v1])
    _write(logs, "sp-setup-llama2-7b-16.out", [{
        "kind": "setup", "label": "setup", "model": m, "layers": None, "prob_max": 2047, "amortise": 100,
        "proof_bytes": 2.38e9,
        "commit_s": 500.0, "prove_s": 120.0, "verify_s": 60.0, "query_columns_bytes_honest": 6.56e7,
        "query_columns_bytes_untrusted": 1.135e8, "accepted": True, "env": _env()}])
    _write(logs, "sp-i-open-llama2-7b-17.out", [
        {"git": "a", "model": m, "layers": n, "seq": 64, "variant": v, "queries": 6, "accepted": 6, "identical": True,
         "prove_s": t, "open_fold_s": t / 2, "timings": {"prove_open": 0.6 * t}, "prob_max": 2047, "env": _env()}
        for n, v, t in ((1, "float64", 0.277), (1, "int8", 0.064), (4, "float64", 0.653), (4, "int8", 0.175))])
    _write(logs, "sp-i-cpuv-llama2-7b-18.out", [perf(m, 2048, "Kpre", 3.37, 1e9, layers=2, label="cpuv-before"),
                                                perf(m, 2048, "Kpre", 0.74, 1e9, layers=2, label="cpuv-after")])
    _write(logs, "sp-i-fold-llama2-7b-19.out", [
        {"git": "a", "model": m, "layers": 32, "device": "cpu", "path": p, "fs": True, "reps": 7, "precompute_s": t,
         "equal_to_float64": True, "prob_max": 2047, "env": _env()}
        for p, t in (("float64", 107.0), ("default", 12.7))])
    # the A/B of 8- against 11-bit attention probabilities on two Llama-2-7B blocks: the proof 0.02% larger in Kpre
    # and 0.045% smaller in C, the times at most 2.7% apart (Kpre's verifier)
    _write(logs, "sp-i-probbits-llama2-7b-48.out", [
        perf(m, 2048, mode, verify, proof, prove=prove, layers=2, label="prob-bits", prob_max=p)
        for mode, p, prove, verify, proof in (("Kpre", 255, 1.00, 0.74, 1.0e9), ("Kpre", 2047, 1.01, 0.76, 1.0002e9),
                                              ("C", 255, 1.20, 0.90, 1.1e9), ("C", 2047, 1.18, 0.91, 1.0995e9))])
    _write(logs, "sp-i-gkr-20.out", [{"n": 25, "leaves": 1 << 25, "T": 2048, "s": 0.52, "ns_per_leaf": 15.6,
                                      "env": _env()}])
    for q, job, gain, ref, scalar, cert in (("facebook/opt-125m", "opt125-21", 4, 27.66, 1.166, 1.076),
                                            ("facebook/opt-1.3b", "opt13-29", 3, 14.62, 1.096, 1.026),
                                            ("facebook/opt-6.7b", "opt67-30", 2, 10.86, 1.696, 1.008)):
        _write(logs, f"sp-q-{job}.out", [
            ppl(q, "fp32", ref, 1.0), ppl(q, f"scalar-g{gain}-p11", ref * scalar, scalar),
            ppl(q, f"scalar-g{gain}-p8", ref * scalar * 1.2, scalar * 1.2, mass=0.56 if gain == 4 else 0.7),
            ppl(q, "smooth-certified-a0.6-p11", ref * cert, cert, ci=(cert - 0.003, cert + 0.003)),
            ppl(q, "smooth-certified-a0.6-p8", ref * cert * 1.02, cert * 1.02, mass=0.6),
            # an alpha-0.55 cell (the legacy recipe's): never the certified row (OPT-125M's has the values of the
            # recipe study below, which measures the same cell)
            ppl(q, "smooth-legacy-a0.55-p11", *((55.0, 2.03) if gain == 4 else (ref * 1.956, 1.956)), alpha=0.55)])
    # the recipe studies: OPT-125M's percentile recipe at 2,048 tokens, the alpha sweeps on the test split (OPT-1.3B;
    # its alpha 0.6 is q-opt13's certified cell again, with the same values) and on the validation split (OPT-1.3B,
    # OPT-6.7B); a validation record never fills tab:quality's test-split row
    sweep = [ppl("facebook/opt-125m", "smooth-legacy-a0.55-p11", 55.0, 2.03, alpha=0.55)]
    for a, test, val in ((0.3, 1.040, None), (0.4, 1.030, None), (0.5, 1.027, 1.031), (0.6, 1.026, 1.029),
                         (0.7, 1.026, 1.028)):
        sweep.append(ppl("facebook/opt-1.3b", f"smooth-certified-a{a:g}-p11", 14.62 * test, test, alpha=a))
        if val:
            sweep.append(ppl("facebook/opt-1.3b", f"smooth-certified-a{a:g}-p11", 17.0, val, alpha=a,
                             split="validation"))
    for a, val in ((0.5, 1.010), (0.6, 1.012)):
        sweep.append(ppl("facebook/opt-6.7b", f"smooth-certified-a{a:g}-p11", 12.0, val, alpha=a, split="validation"))
    _write(logs, "sp-q-opt13-recipes-46.out", sweep)
    # the attribution study and the quality tier's cell of its graph (the same digest and perplexity)
    _write(logs, "sp-q-opt125-128-47.out", [dict(ppl("facebook/opt-125m", "scalar-g4-p8", 75.5198, 75.5198 / 64.7611,
                                                     calibration="test-end", seq=128), windows=16, digest="d125")])
    _write(logs, "sp-q-opt125-attr-48.out", attribution())
    _write(logs, "sp-q-zeroshot-opt-125m-22.out", [
        zeroshot("facebook/opt-125m", t, v, 0.379 if t == "lambada_openai" else 0.4, diff=-0.031 if t == "lambada_openai" else -0.005)
        for t, _ in T.ZEROSHOT for v in ("ref", "int")])
    _write(logs, "sp-q-openllama7-23.out", [
        {"model": "openlm-research/open_llama_7b_v2", "config": "llama2-7b", "seq": 2048, "windows": 160,
         "cell": "reference", "ppl": 6.0, "env": _env()},
        {"model": "openlm-research/open_llama_7b_v2", "config": "llama2-7b", "seq": 2048, "windows": 160,
         "cell": "main-a0.55-glus0.2", "graph": "main", "alpha": 0.55, "glu_smooth": 0.2, "draw": 0, "ppl": 6.44,
         "ratio": 1.073, "ratio_ci95": [1.07, 1.08], "top1": 0.9, "env": _env()}])
    # the legacy recipe's zero-shot job of OPT-125M, a later job: its fp16 reference never fills the certified columns
    _write(logs, "sp-q-zeroshot-legacy-opt-125m-48.out", [
        zeroshot("facebook/opt-125m", t, v, 0.5, diff=-0.02, recipe="legacy") for t, _ in T.ZEROSHOT
        for v in ("ref", "int")])
    # Llama-2-7B against its fp32 checkpoint, Llama-2-13B against its fp16 copy
    l7, l13, ol = "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "openlm-research/open_llama_7b_v2"
    _write(logs, "sp-q-llama7-49.out", [
        llama_ppl(l7, "reference", 1.0),
        llama_ppl(l7, "main-a0.55-glus0.2", 1.0805, ci=(1.0781, 1.0832), alpha=0.55, balance=0.2),
        llama_ppl(l7, "main-a0.65-glus0.2", 1.09, alpha=0.65, balance=0.2),
        llama_ppl(l7, "smooth-a0.55", 1.12, alpha=0.55)])
    _write(logs, "sp-q-llama13-50.out", [
        llama_ppl(l13, "reference", 1.0, ref_dtype="float16"),
        llama_ppl(l13, "main-a0.6-glus0.2", 1.0641, ci=(1.0622, 1.0660), alpha=0.6, balance=0.2,
                  ref_dtype="float16")])
    # OpenLLaMA-7B's grid on the validation split: alpha 0.45 with balance 0.3 is the best, the main recipe's ratio
    # 0.0035 above it; none of these fills the test split's \LlamaOpenRatio
    best = {(0.55, 0.2): 1.0735, (0.45, 0.3): 1.07}
    _write(logs, "sp-q-openllama7-validation-51.out", [llama_ppl(ol, "reference", 1.0, split="validation")] + [
        llama_ppl(ol, f"main-a{a:g}-glus{g:g}", best.get((a, g), 1.08), alpha=a, balance=g, split="validation")
        for a in (0.45, 0.55, 0.65) for g in (0.1, 0.2, 0.3)])
    # the reproducible build: OPT-125M with 1 and 8 threads, OPT-1.3B's commitment encoded on the GPU and the CPU
    _write(logs, "sp-q-digest-52.out", [
        digest_rec("facebook/opt-125m", 1), digest_rec("facebook/opt-125m", 8, device="cuda"),
        digest_rec("facebook/opt-1.3b", 8, device="cuda", digest_="d2", weights="w2", roots="r2"),
        digest_rec("facebook/opt-1.3b", 8, device="cpu", digest_="d2", weights="w2", roots="r2")])
    _write(logs, "sp-l-llama7-4096-L2-24.out", [perf(m, 4096, "Kpre", 4.0, 1e9, layers=2, label="long")])
    _write(logs, "sp-l-llama7-4096-L3-25.out", [perf(m, 4096, "Kpre", 5.4, 1.4e9, layers=3, label="long")])
    _write(logs, "sp-l-rx-llama7-4096-26.out", [reexec(m, 4096, v, t, blocks={"2": t / 16, "3": t * 3 / 32})
                                                for v, t in (("float", 188.5), ("int", 250.0))])
    _write(logs, "sp-b-c-lenet5-27.out", [{"pvi_path": "/x", "torch_threads": 8, "cpu": ENV["cpu"]}],
           extra="done in 12s\n")
    _write(logs, "sp-a-bound-28.out", [{"model": "llama2-7b", "params": 6738149376, "seq": 64, "lam": 40,
                                        "anchuri_width": 11008, "anchuri_paths": 305193,
                                        "anchuri_open_all_bytes": 13678673920}])
    att = logs / "attack"
    att.mkdir()
    for name in ("attack.json", "defence.json", "floor_sampler.json", "adaptive_game_mlp_mnist.json",
                 "real_llm_attack.json", "real_llm_attack_auto.json", "reproduction.json",
                 "inrange_llm_opt-1.3b_any.json", "inrange_llm_opt-1.3b_down.json", "inrange_llm_opt-6.7b_any.json",
                 "inrange_llm_opt-6.7b_down.json"):
        shutil.copy(RESULTS / name, att / name)
    shutil.copy(CODE / "artifacts" / "comparison" / "tables" / "analytic.csv", att / "analytic.csv")
    shutil.copy(RESULTS / "sampler_tv_opt-1.3b.json", logs / "sampler_tv_opt-1.3b.json")
    return logs


def _args(tmp: Path, logs: Path, *extra) -> list[str]:
    return ["--logs", str(logs), "--platform", "test", "--bench", str(tmp), "--out", str(tmp / "out"),
            "--figures", "", "--no-byte-model", *extra]


@pytest.fixture
def run_dir(tmp_path):
    _bench(tmp_path)
    return tmp_path, _logs(tmp_path)


def _ctx(tmp: Path, logs: Path):
    problems = []
    run = R.Run(R.read_logs(logs, problems), R.read_files(logs, problems))
    ctx = T.Ctx(run, B.Bench(tmp / "tables_test"), B.literature(), "cnn18c", "test platform", byte_model=False)
    return ctx, problems


# ------------------------------------------------------------------------------------------------ reading
def test_every_record_lands_in_its_tier_and_kind(run_dir):
    tmp, logs = run_dir
    problems = []
    recs = R.read_logs(logs, problems)
    assert not problems
    got = {(r.tier, r.kind) for r in recs}
    assert {("smoke", "v1"), ("smoke", "h2d"), ("cpuv", "perf"), ("cpuv", "reexec"), ("cpuv", "llamacpp"),
            ("cpuv", "v1"), ("v1", "v1"),
            ("gpuv", "v1"), ("gpuv", "perf"), ("gpuv", "reexec"), ("serve", "serve"), ("gen", "gen"), ("real", "gen"),
            ("real", "v1"), ("setup", "setup"), ("improve", "open_ab"), ("improve", "perf"), ("improve", "fold_ab"),
            ("improve", "gkr"), ("quality", "ppl"), ("quality", "zeroshot"), ("quality", "digest"),
            ("quality", "attribution"), ("long", "perf"), ("long", "reexec"),
            ("bench", "bench_env"), ("attack", "path_bound"), ("gate", "v1"), ("gate", "perf")} <= got
    assert not [m for r in recs for m in R.checks(r)] and not R.log_checks(logs)
    files = R.read_files(logs, problems)
    assert {r.kind for r in files} >= {"attack_mnist", "inrange", "tv", "path_bound"} and not problems
    assert R.verifier_device(v1("x", 1, "C", 1, 1, 1, argv=["--verifier-device", "cuda:0"])) == "cuda"


def test_the_formats():
    assert F.t(0.0761) == r"76.1\,ms" and F.t(0.712) == r"712\,ms" and F.t(27.4) == r"27.4\,s"
    assert F.t(174.0) == r"2.9\,min" and F.b(6.6e9) == r"6.6\,GB" and F.b(3.25e8) == r"325\,MB"
    assert F.sig(93.2) == "93.2" and F.sig(0.15) == "0.15" and F.sig(1722.4, 4) == "1{,}722"
    assert F.about(31_910) == "32{,}000" and F.about(305_193, 3) == "305{,}000"
    assert F.signed(-0.0004 * 100, 1) == "$-0.04$" and F.signed(0.06, 2) == "$+0.06$"
    assert F.sci(7.93e-4) == r"$7.9\cdot10^{-4}$" and F.rng(2.03, 2.29, lambda v: f"{v:.1f}") == "2.0--2.3"
    assert F.factor(88.3, 0.02)[0] == r"4{,}415$\times\downarrow$" and F.factor(1, 2)[0] == r"2.0$\times\uparrow$"
    assert F.macro_name("Crossover", "gpt2", 512, "Kpre") == "CrossoverGptTwoFiveOneTwoKpre"
    assert F.params(6.738e9) == "6.7B" and F.tokens(2048) == "2{,}048"
    # the unit is chosen after rounding: no "1000 ms", "100.0 s", "1000 MB" or "100.0 MB"
    assert (F.t(0.9996), F.t(99.97), F.t(0.0994), F.t(0.09996)) == (r"1.0\,s", r"1.7\,min", r"99.4\,ms", r"100\,ms")
    assert (F.b(999.7e6), F.b(99.97e6), F.b(960e6), F.b(999.7), F.b(500)) == (
        r"1.0\,GB", r"100\,MB", r"960\,MB", r"1.0\,kB", r"500\,B")
    # bounds: "at most" and "up to" round up, "at least" rounds down
    assert (M.ceil_sci(7.9346e-4), M.ceil_sci(9.95e-4), M.ceil_sci(8.0e-4)) == (
        r"$8.0\cdot10^{-4}$", r"$1.0\cdot10^{-3}$", r"$8.0\cdot10^{-4}$")
    assert (M.ceil_about(55_899), M.ceil_about(56_000), M.floor_sig(55.97), M.floor_sig(56.0)) == (
        "56{,}000", "56{,}000", "55.9", "56")
    assert M.ceil_pct(0.0949) == "10" and M.ceil_fixed(1.0264, 3) == "1.027"


# ------------------------------------------------------------------------------------------------ selection
def test_each_cell_comes_from_its_own_records(run_dir):
    tmp, logs = run_dir
    ctx, problems = _ctx(tmp, logs)
    tables = {t.name: t for t in T.build_all(ctx, problems)}
    assert not problems
    cells = lambda name: {(c["row"], c["column"]): c for c in tables[name].cells}    # noqa: E731
    # tab:gen: the random-weight greedy cell, never the real Llama-2-7B's (99 s)
    g = cells("tab_gen")
    assert g["Llama-2-7B G=64 C", "P"]["value"] == 2.0 and "sp-gen-" in g["Llama-2-7B G=64 C", "P"]["source"]
    assert all(c["value"] != 99.0 for c in tables["tab_gen_full"].cells)
    # tab:options: V1's optimised CPU verifier (45.3 + 0.1 s), never the baseline (70) or the GPU verifier's cell
    o = cells("tab_options")
    assert o["with V1", "time 2048"]["value"] == pytest.approx(45.4)
    assert o["with V1", "bytes 2048"]["value"] == 2.8e9 and "sp-c-v1-" in o["with V1", "time 2048"]["source"]
    assert o["Re-exec. CPU", "2048"]["value"] == 141.6 and o["Re-exec. CPU", "64"]["value"] == 4.08   # exact only
    assert o["Re-exec. GPU", "2048"]["value"] == 1.65
    # tab:eval-v1: the v1 tier's CPU-verifier record (1.0e9 is the GPU verifier's V1 bytes)
    v = cells("tab_eval_v1")
    assert v["Llama-2-7B T=2048 Kpre", "V1 MB"]["value"] == 2.8e9
    assert v["OPT-125M T=2048 C", "V1 verifier"]["value"] == pytest.approx(4.86)
    assert not any(c["value"] == 1.0e9 for c in tables["tab_eval_v1"].cells)
    # tab:quality: the certified cell (x1.076), never the alpha-0.55 one (x1.956)
    q = cells("tab_quality")
    assert q["OPT-125M", "certified ratio"]["value"] == 1.076 and q["OPT-125M", "scalar gain"]["value"] == 1.166
    assert not any(c["value"] == 1.956 for c in tables["tab_quality"].cells)
    assert tables["tab_quality"].rows[0] == r"OPT-125M & 27.66 & $\times$1.166 & 29.76 ($\times$1.076) & 0.90 \\"
    # tab:eval-llm: the Fiat-Shamir cells of the plan (the interactive cell, 0.9x, is tab:eval-compare's)
    e = cells("tab_eval_llm")
    assert e["Llama-2-7B (64)", "C prove"]["value"] == pytest.approx(1.6)
    assert e["Llama-2-7B (64)", "C GPU verify"]["value"] == pytest.approx(0.5)
    assert e["Llama-2-7B (64)", "Kpre proof"]["value"] == pytest.approx(0.7 * 3.25e8)
    # tab:eval-real: v0's timed queries of the V1 records, real against the random twin
    r = cells("tab_eval_real")
    assert r["Kpre T=64", "prove real"]["value"] == pytest.approx(0.30)
    assert r["Kpre T=64", "v0_total random"]["value"] == 6.04e7
    # tab:eval-improve: before / after of the same row
    i = cells("tab_eval_improve")
    assert i["open Llama-2-7B 1/32", "gain"]["value"] == pytest.approx(0.277 / 0.064)
    assert i["cpuv Llama-2-7B 2/32", "gain"]["value"] == pytest.approx(3.37 / 0.74)
    # tab:memory: the client's own process and the measuring process of the cpuv cells
    mem = cells("tab_memory")
    assert mem["Kpre T=2048", "client"]["value"] == pytest.approx(21.4e9)
    assert "sp-c-llama2-7b-2048-" in mem["Kpre T=2048", "client"]["source"]
    assert mem["C T=64", "client after imports"]["value"] == pytest.approx(0.53e9)
    assert mem["Kpre T=64", "prover GPU"]["value"] == 7.4e9 and mem["C T=2048", "process"]["value"] == 5e10
    assert tables["tab_memory"].rows[1] == r" & 2{,}048 & 21.4 & 0.5 & -- & 50.0 \\"
    # the gate tier's V1 records never fill tab:eval-v1 (its Llama-2-7B rows are the v1 tier's)
    assert not any(c["source"].startswith("sp-gate-") for t in tables.values() for c in t.cells)


def test_the_text_numbers(run_dir):
    tmp, logs = run_dir
    ctx, problems = _ctx(tmp, logs)
    m, _ = M.build(ctx, problems)
    assert not problems
    v = {k: x.value for k, x in m.items()}
    # the abstract's N: the 2,048-token ratio of exact re-execution (the integer forward pass, its faster GEMM path:
    # 141.6 s) to the Kpre CPU verifier (19.6 s), rounded down; not the minimum over the prompt lengths (64 tokens:
    # 4.08 / 1.4 = 2.9); fp32 (93.2 s), dynamic int8 (56 s) and llama.cpp (61.3 s) are inexact and never the baseline
    assert v["CpuRatioN"] == "7.2" and v["CpuRatio"] == "7.2" and v["CpuRatioShort"] == "2.9"
    assert v["CpuReexecName"] == "exact fp32 GEMMs" and v["CpuReexecShort"] == "4.08" and v["CpuReexecLong"] == "142"
    assert v["CpuInexactName"] == "dynamic int8" and v["CpuInexactLong"] == "56" and v["CpuRatioInexact"] == "2.9"
    # llama.cpp: each type at its faster attention kernel; Q8_0 is the faster type at 2,048 tokens (61.3 / 19.6 s)
    assert (v["CpuLlamaCppShort"], v["CpuLlamaCppLong"], v["CpuLlamaCppBfShort"], v["CpuLlamaCppBfLong"]) == \
        ("2.2", "61.3", "3.1", "88")
    assert v["CpuLlamaCppName"] == r"Q8\_0" and v["CpuRatioLlamaCpp"] == "3.1" and v["LlamaCppRelease"] == "v0.6.0"
    assert v["VOneVerifyLlama"] == "45.4" and v["VOneVerifyRatioLlama"] == "2.3"
    assert v["ServeOptMKpre"] == "2.1" and v["ServeOptMOverUOneRange"] == "4.1--8.6" and v["ServeBatchKpre"] == "3.4"
    assert v["QualityRatioOptS"] == "1.076" and v["QualityCIOptS"] == "1.073, 1.079" and v["QualityMissOptS"] == "0.016"
    assert v["QualityMassMinEight"] == "56" and v["LlamaOpenRatio"] == "1.073" and v["ZeroshotLambadaOptS"] == "3.1"
    assert v["GkrNsPerLeaf"] == "15.6" and v["KpreKeyLlama"] == "12.7" and v["ImproveCpuv"] == "4.6"
    assert v["ProbBitsBytes"] == "0.05" and v["ProbBitsTime"] == "3"           # bounds: rounded up
    assert v["FinalNHonest"] == "3" and v["FinalNFreivalds"] == "1" and v["TodoCounts"] == ""
    assert v["LenetProof"] == "63.3" and v["VOneRealOptMShortK"] == "1.28"
    assert v["OptMGpuVerify"] == "1.1" and v["GpuFloatLong"] == "0.64" and v["GpuIntShort"] == "0.15"
    # the attack's stored documents regenerate the paper's numbers
    assert v["AttackOptLPaths"] == "454{,}248" and v["PathsLlama"] == "305{,}193" and v["InrangeKMax"] == "60"
    assert v["InrangeDown"] == "9, 31, 38 and 39" and v["MixtureLayerOne"] == "3.1" and v["TvTopK"] == "0.0063"
    assert v["BreakevenQueriesShort"] is None                  # the byte model is off in this test
    # the client's own memory (tab:memory); the baseline is the largest over the cells, rounded up ("at most")
    assert (v["ClientPeakLongKpre"], v["ClientPeakLongC"], v["ClientPeakShortKpre"]) == ("21.4", "23.1", "1.1")
    assert v["ClientBaseline"] == "0.6"
    assert set(m) == set(M.names())


def test_the_settings_and_the_further_text_numbers(run_dir):
    """The platform and the plan that the captions state, the generation cells (GPT-2 per generated token after a
    64-token prompt, never another prompt's cell; sampled against greedy), the opening that B prompts share, the real
    Llama-2-7B and the certified OPT-1.3B's exceptions, the float64 opening's share, the recipe studies on the test
    and the validation split (a validation record never fills tab:quality), and the attack documents' numbers."""
    tmp, logs = run_dir
    ctx, problems = _ctx(tmp, logs)
    m, _ = M.build(ctx, problems)
    tables = {t.name: t for t in T.build_all(ctx, problems)}
    assert not problems
    v = {k: x.value for k, x in m.items()}
    assert v["PlatformGpu"] == "NVIDIA L40S" and v["PlatformCpu"] == "AMD EPYC 9334" and v["PlanLogPoints"] == "18"
    assert M.cpu_name("Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz") == "Intel Xeon Silver 4114"
    assert v["CpuDeriveShareLong"] == "100"
    # generation
    assert v["GenGptProveC"] == "0.29" and v["GenGptMsPerToken"] == "1.1"
    assert v["GenLlamaProveC"] == "2" and v["GenLlamaProofC"] == r"1.0\,GB" and v["GenQwenVerifyC"] == "3"
    assert v["GenLlamaMBPerTokenShort"] == "15.6" and v["GenLlamaMBPerTokenLong"] == "3.91"
    assert v["GenSampledWithin"] == "3"
    g = {(c["row"], c["column"]): c for c in tables["tab_gen"].cells}
    assert g["GPT-2 G=256 C", "P"]["value"] == 0.29 and "sp-gen-gpt2-" in g["GPT-2 G=256 C", "P"]["source"]
    assert tables["tab_gen"].rows[0].startswith("GPT-2 ($P{=}64$) & 256 & 0.29")
    assert not any(c["value"] == 5.0 for c in tables["tab_gen_full"].cells)
    # B prompts against one opening: claims + shared / B (exact at B = 4, 1.5% above at B = 16)
    assert (v["BatchLlamaSize"], v["BatchLlamaSingle"], v["BatchLlamaShared"]) == ("16", "325.0", "125.0")
    assert v["BatchLlamaPerQuery"] == "210.9" and v["BatchLlamaWithin"] == "2"
    # the bounds round up: VGG-16's int8 accuracy is 0.34 points below float, not "at most 0.3"
    assert v["CnnAccuracyChange"] == "0.4" and v["ImproveOpenShare"] == "60" and v["ImproveProveFloat"] == "0.65"
    # V1: the multiplicities and GKR transcripts of every V1 proof, the real tier's included (0.901 MB, never the
    # exceptions' 1.8 MB), and the full queries against the accounting (0.34%), both rounded up
    assert v["VOneVerifyLlamaBaseline"] == "70.1" and v["CrossoverLlamaLongKBuild"] == "the full model"
    assert v["VOneProverRatioAbout"] == "4" and v["VOneExtrasMax"] == "0.91" and v["VOneAccountingWithin"] == "0.4"
    assert v["VOneRandomExceptionsMax"] == "224{,}670" and v["GkrSecondsLarge"] == "0.52"
    assert (v["VOneRealOptMShortExceptions"], v["VOneRandomOptMShortExceptions"]) == ("2{,}379", "1")
    assert v["VOneRealOptMShortExceptionsMB"] == "0.02" and v["VOneRealOptMShortC"] == "1.30"
    assert v["VOneRealOptMLongC"] is None                      # the real tier's 2,048-token cells are not in this run
    # the real Llama-2-7B
    assert (v["RealLlamaAccepted"], v["RealLlamaQueries"], v["RealLlamaTampersRejected"], v["RealLlamaTampers"]) == \
        ("5", "5", "9", "9")
    assert (v["RealLlamaProof"], v["RealLlamaSmaller"], v["RealLlamaVerify"], v["RndLlamaVerify"]) == \
        ("5.35", "17", "21", "20")
    assert v["RealLlamaShortSmaller"] == "11--15" and v["RealLlamaShortVerifyWithin"] == "10"
    assert v["VOneRealLlamaRatio"] == "2.02"
    assert v["RealAcceptedClause"] == "the full real Llama-2-7B is accepted"
    assert M.accepted_clause(["OPT-6.7B", "Llama-2-7B"]) == "the full real OPT-6.7B and Llama-2-7B are accepted"
    # V1's range over the full models only: OPT-125M and Llama-2-7B (2.03 to 2.30), not the 2-block Llama-2-13B
    assert v["VOneRatioRandom"] == "2.0--2.3" and v["VOneRatioRandomExact"] == "2.03--2.30"
    assert v["VOneRatioLargeC"] == "1.82"
    # the Kpre client's storage, from the shapes: r = 7 vectors and folded rows, and the int8 embedding table
    assert (v["KpreKeyMB"], v["KpreEmbedMB"], v["KpreStoreGB"]) == ("71", "131", "0.2")
    # quality: the recipe studies
    assert v["QualityLegacyOptS"] == "2.0" and v["QualityAlphaGapOptM"] == "0.000" and v["QualityAlphaGapOptS"] is None
    assert v["QualityAlphaValGapOptM"] == "0.001" and v["QualityAlphaCostOptL"] == "0.002"
    q = {(c["row"], c["column"]): c for c in tables["tab_quality"].cells}
    assert q["OPT-1.3B", "certified ratio"]["value"] == 1.026
    # the attack's stored documents
    assert (v["ZeroingZeroShare"], v["MnistStealthNeurons"], v["MnistStealthPass"]) == ("65", "five", "99")
    assert v["StealthExceed"] == "5" and v["InrangeDownLast"] == "39"
    assert (v["InrangeChanged"], v["InrangeOptLChanged"]) == ("all 40", "all 40")
    assert M.all_of((40, 40)) == "all 40" and M.all_of((38, 40)) == "at least 38 of the 40"
    assert [M.count_word(x) for x in (5.43, 1, 10.2, 14.35)] == ["five", "one", "ten", "14"]
    assert (M.ceil_sig(0.8349), M.ceil_sig(1.8), M.ceil_sig(0.999), M.ceil_sig(12.3)) == ("0.84", "1.8", "1", "13")
    assert (M.ceil_fixed(0.34), M.ceil_fixed(0.3), M.ceil_fixed(0.0009999999999999, 3)) == ("0.4", "0.3", "0.001")
    # from the shapes alone, under the run's instance bound (the records' lmax 27)
    assert v["VOneDownClaims"] == r"2.6\cdot10^{8}" and v["VOneLargeLeaves"] == r"4.3\cdot10^{9}"
    assert (v["VOneLargeInstances"], v["VOneLargeBits"]) == ("39", "27")
    assert (v["GkrTranscriptBits"], v["GkrTranscriptKB"]) == ("27", "36.0")
    assert v["CompareVerifierSlowerRows"] is None              # most rows of tab:eval-compare are not in this run
    assert [M.rows_text(n) for n in (0, 1, 2, 12)] == ["no rows", "one row", "two rows", "12 rows"]


def test_the_llama_family_the_validation_grid_and_the_reproducible_build(run_dir):
    """Llama-2-7B against fp32 and Llama-2-13B against fp16, with their intervals; OpenLLaMA-7B's validation grid (its
    best cell, the main recipe's ratio above it rounded up, and never the test split's ratio); the reproducible
    build's thread counts and encoding devices; and tab:zeroshot's fp16 column from the certified job only."""
    tmp, logs = run_dir
    ctx, problems = _ctx(tmp, logs)
    m, _ = M.build(ctx, problems)
    tables = {t.name: t for t in T.build_all(ctx, problems)}
    assert not problems and not ctx.run.warnings
    v = {k: x.value for k, x in m.items()}
    assert (v["LlamaSevenRatio"], v["LlamaSevenCI"]) == ("1.081", "1.078, 1.083")
    assert (v["LlamaThirteenRatio"], v["LlamaThirteenCI"]) == ("1.065", "1.062, 1.066")
    assert v["LlamaOpenRatio"] == "1.073"                      # the test split's cell, never the validation grid's
    assert (v["LlamaOpenValAlpha"], v["LlamaOpenValBalance"], v["LlamaOpenValGap"]) == ("0.45", "0.3", "0.004")
    assert v["LlamaOpenValClause"] == r"$\alpha=0.45$ with balance 0.3 is the best, ours within 0.004 of its ratio"
    assert (v["DigestThreadsOptS"], v["DigestEncodingOptM"]) == ("1 and 8", "CPU and GPU")
    z = {(c["row"], c["column"]): c for c in tables["tab_zeroshot"].cells}
    assert z["facebook/opt-125m lambada_openai", "fp16"]["value"] == 0.379           # not the legacy job's 0.5
    assert "sp-q-zeroshot-opt-125m-22" in z["facebook/opt-125m lambada_openai", "fp16"]["source"]


def _quality_run(*records) -> tuple[dict, list]:
    """``({macro: printed value}, problems)`` of a run of the quality tier's ``records`` alone."""
    recs = [R.Rec("quality", "q-x", str(i), f"sp-q-x-{i}.out", 1, R.kind_of(d), d) for i, d in enumerate(records, 1)]
    problems: list = []
    m, _ = M.build(T.Ctx(R.Run(recs), B.Bench(None), {}, byte_model=False), problems)
    return {k: x.value for k, x in m.items()}, problems


def test_the_texts_claims_on_the_llama_family_and_a_build_that_differs():
    """The main recipe best on its validation grid (a tie goes to it); a validation cell outside the grid the text
    names, or a Llama-2-13B ratio against fp32, fails the text's claims (a rehearsal's warnings); two builds of one
    model that differ fail the build and name no setting, also in a rehearsal; one thread count names none."""
    ol, l13 = "openlm-research/open_llama_7b_v2", "meta-llama/Llama-2-13b-hf"
    grid = [llama_ppl(ol, f"main-a{a:g}-glus{g:g}", 1.07 if (a, g) in ((0.55, 0.2), (0.65, 0.1)) else 1.08, alpha=a,
                  balance=g, split="validation") for a in (0.45, 0.55, 0.65) for g in (0.1, 0.2, 0.3)]
    v, problems = _quality_run(*grid)
    assert not problems and v["LlamaOpenValClause"] == "ours are the best" and v["LlamaOpenValGap"] == "0.000"
    assert (v["LlamaOpenValAlpha"], v["LlamaOpenValBalance"]) == ("0.55", "0.2")
    v, problems = _quality_run(*grid[:8])                       # one cell missing: no clause, and no false claim
    assert not problems and v["LlamaOpenValClause"] is None and v["LlamaOpenValAlpha"] is None
    outside = llama_ppl(ol, "main-a0.75-glus0.2", 1.06, alpha=0.75, balance=0.2, split="validation")
    _, problems = _quality_run(*grid, outside)
    assert len(problems) == 1 and problems[0].startswith(M.TEXT_CLAIM) and "outside the grid" in problems[0]
    v, problems = _quality_run(llama_ppl(l13, "main-a0.6-glus0.2", 1.05, alpha=0.6, balance=0.2))
    assert v["LlamaThirteenRatio"] == "1.050" and problems == [
        f"{M.TEXT_CLAIM} LlamaThirteenRatio's reference is float32; the text states float16"]
    builds = [digest_rec("facebook/opt-1.3b", 8, device="cuda"),
              digest_rec("facebook/opt-1.3b", 8, device="cpu", roots="r9"), digest_rec("facebook/opt-125m", 8)]
    v, problems = _quality_run(*builds)
    assert v["DigestEncodingOptM"] is None and v["DigestThreadsOptS"] is None    # OPT-125M: one thread count only
    assert len(problems) == 1 and problems[0].startswith("CHECK FAILED digest: the builds of the certified "
                                                         "facebook/opt-1.3b differ in roots_sha256")
    kept, _ = N.demote_text_claims(problems)
    assert kept == problems                                     # a build that differs fails a rehearsal too
    v, problems = _quality_run(digest_rec("facebook/opt-125m", 1), digest_rec("facebook/opt-125m", 8, weights="w2"))
    assert v["DigestThreadsOptS"] is None and "differ in weights_sha256" in problems[0]
    assert M.commit_device({"roots_sha256": "r", "env": {"argv": ["--commit-device", "cuda:0"]}}) == "cuda"
    assert M.commit_device({"digest": "d"}) is None and M.val_clause((0.45, 0.1), 0.00012, (0.55, 0.2)) == \
        r"$\alpha=0.45$ with balance 0.1 is the best, ours within 0.001 of its ratio"


def test_two_records_of_one_quality_cell_that_differ_are_a_warning(run_dir, capsys):
    """A cell that two jobs measure (the certified OPT-125M of q-opt125 and of an alpha study) with the same values
    gives no warning; with different values a WARNING names both records, the later is used, and the exit status
    stays 0."""
    tmp, logs = run_dir
    same = ppl("facebook/opt-125m", "smooth-certified-a0.6-p11", 27.66 * 1.076, 1.076, ci=(1.073, 1.079))
    _write(logs, "sp-q-opt125-recipes-53.out", [same])
    ctx, problems = _ctx(tmp, logs)
    M.build(ctx, problems)
    assert not problems and not ctx.run.warnings
    _write(logs, "sp-q-opt125-recipes-53.out", [dict(same, ratio=1.0801, ppl=27.66 * 1.0801)])
    ctx, problems = _ctx(tmp, logs)
    m, _ = M.build(ctx, problems)
    assert m["QualityRatioOptS"].value == "1.081" and len(ctx.run.warnings) == 1
    w = ctx.run.warnings[0]
    assert w.startswith("WARNING quality: two records of facebook/opt-125m smooth-certified-a0.6-p11 (test split")
    assert "sp-q-opt125-21.out" in w and "sp-q-opt125-recipes-53.out" in w
    capsys.readouterr()
    assert N.main(_args(tmp, logs, "--check")) == 0
    assert "WARNING quality: two records of facebook/opt-125m smooth-certified-a0.6-p11" in capsys.readouterr().out


def test_the_llama_cells_are_the_main_recipes_and_the_grid_the_texts():
    """Each Llama-family macro reads its checkpoint's main recipe (``hf_llama.MAIN_SETTINGS``), and OpenLLaMA-7B's
    validation grid is the main recipe's alpha and balance with their neighbours 0.1 away, as the text names it."""
    from pvi.models import hf_llama
    for model, _, cell, _, _ in M.LLAMAS:
        r = hf_llama.main_recipe(model)
        assert cell == f"main-a{r['smooth']:g}-glus{r['glu_smooth']:g}", model
    model, alphas, balances = M.LLAMA_VALIDATION
    r = hf_llama.main_recipe(model)
    assert (alphas[1], balances[1]) == (r["smooth"], r["glu_smooth"])
    assert [round(b - a, 9) for x in (alphas, balances) for a, b in zip(x, x[1:])] == [0.1] * 4
    body = (PAPER / "sections" / "quality.tex").read_text(encoding="utf-8")
    ext = (PAPER / "sections" / "ext_generation.tex").read_text(encoding="utf-8")
    assert r"$\alpha$ and balance $\pm0.1$ from ours, \LlamaOpenValClause{}" in body
    assert (rf"$\alpha\in\{{{','.join(f'{a:g}' for a in alphas)}\}}$ and balance "
            rf"$\in\{{{','.join(f'{g:g}' for g in balances)}\}}$") in ext
    assert rf"($\alpha={alphas[1]:g}$, balance {balances[1]:g})" in ext


def test_the_attribution_its_numbers_and_what_the_text_says(run_dir):
    """``tab:quality-attr`` and its macros from ``attribution.py``'s records (the stored study's values: the
    fallback's), the cross-check of its integer graph against the quality tier's cell of the same graph, a failed
    check of the float copy, and the checks of what the text states: the LayerNorm outputs dominate and remove more
    than half of the gap, the weight matrices and fc1's requantisation follow, the loss comes from rounding, not
    clipping, and the softmax, residual and ReLU rows each cost at most the printed bound."""
    tmp, logs = run_dir
    ctx, problems = _ctx(tmp, logs)
    m, _ = M.build(ctx, problems)
    tb = T.tab_quality_attr(ctx)
    assert not problems and not tb.missing and R.cross_checks(ctx.run) == []
    assert tuple(T.ATTRIBUTION_GROUPS) == tuple(ATTRIBUTION)
    v = {k: x.value for k, x in m.items()}
    assert (v["AttrIntPpl"], v["AttrFpPpl"], v["AttrNormLoo"]) == ("75.5", "64.8", "69.4")
    assert v["AttrSmallCost"] == "0.2" and v["AttrFloatTol"] == r"$4\cdot10^{-5}$"    # the softmax costs 0.17
    fallback = (PAPER / "fallback" / "tab_quality_attr.tex").read_text(encoding="utf-8")
    assert [r.split() for r in tb.rows] == [x.split() for x in fallback.splitlines() if not x.startswith("%")]
    assert "digest d125" in tb.note

    def build(recs):
        _write(logs, "sp-q-opt125-attr-48.out", recs)
        ctx, problems = _ctx(tmp, logs)
        failed = []
        M.build(ctx, failed)
        assert not problems
        return ctx.run, failed

    run, _ = build(attribution(digest="other"))
    assert [x for x in R.cross_checks(run) if "digest is not that of the cell scalar-g4-p8" in x]
    run, _ = build(attribution(int_ppl=75.6))
    assert [x for x in R.cross_checks(run) if "perplexity 75.6 is not the cell's 75.5198" in x]
    run, _ = build(attribution(windows=2, int_ppl=75.6))       # a rehearsal's: the same graph, other windows
    assert R.cross_checks(run) == [] and run.attribution("facebook/opt-125m", part="check") is None
    run, _ = build(attribution(ok=False))
    check = run.attribution("facebook/opt-125m", part="check")
    assert R.checks(check) == ["the attribution's float copy failed a check (ok false)"]
    # what the text states, against other values
    rows = dict(ATTRIBUTION, norm=(73.0, 70.3329), relu=(72.0, 64.7611))
    _, failed = build(attribution(rows=rows, sub={"norm_round": 64.85, "norm_clip": 64.8607}))
    claims = [x for x in failed if x.startswith(M.TEXT_CLAIM)]
    assert len(claims) == 4 and all("attribution" in x for x in claims)
    for text in ("is not the lowest", "the text says more than half", "the text names the weight matrices",
                 "rounding, not clipping"):
        assert any(text in x for x in claims), text


def test_the_models_from_the_library():
    """The values that follow from the code alone: the split commitment's columns per query and their bytes against
    the honest ones (Llama-2-7B under cnn18c), and V1's cut from the shapes, which reproduces the leaves of a v1
    record of the full Llama-2-7B at 2,048 tokens."""
    assert M.split_columns(136.82, 1 << 18) == 137
    assert M.split_factor("llama2-7b", "cnn18c") == pytest.approx(4.45, abs=0.01)
    shape = T.v1_shape_model("llama2-7b", 2048, 27, "cnn18c")
    assert shape["n_leaves"] == v1("llama2-7b", 2048, "C", 32, 1, 1)["n_leaves"] == 2_714_763_264
    assert shape["max_bits"] <= 27 and shape["lookup_log2"] < -215 and shape["gkr_log2"] < -230
    assert T.v1_shape_model("llama2-13b", 2048, 26, "cnn18c")["n_instances"] == 79
    # the Kpre client of Llama-2-7B: r = 7 secret vectors and folded rows of its 225 linear operations (4-byte field
    # elements) and its int8 embedding table
    assert M.kpre_storage() == {"key": 7 * 4 * 2_534_656, "embed": 32_000 * 4_096, "reps": 7}
    # the prefill computes the LM head and the pruned last block's query, output projection and MLP at every position
    assert M.prefill_extra_claims("llama2-7b") == (32_000 + 2 * 4_096 + 2 * 11_008 + 4_096, 1_325_568)


def _reexec_run(*rows):
    return R.Run([R.Rec("cpuv", "c-rx-llama2-7b", str(i), f"sp-c-rx-llama2-7b-{i}.out", 1, "reexec",
                        dict(reexec("llama2-7b", 2048, v, t), **extra)) for i, (v, t, extra) in enumerate(rows, 1)])


def test_exact_reexecution_is_the_fastest_exact_variant():
    """Exact re-execution is the fastest of the integer forward pass's exact variants (tables.EXACT); the oneDNN
    variant counts only when its record says its logits equal the integer reference's bit for bit."""
    assert T.EXACT == ("int", "int_fp32gemm", "int_mm") and "float" not in T.EXACT
    base = (("int", 179.4, {}), ("int_fp32gemm", 141.6, {}), ("float", 93.2, {}))
    assert T.fastest(_reexec_run(*base), "cpuv", "llama2-7b", 2048)[:2] == (141.6, "int_fp32gemm")
    for extra, want in (({}, "int_fp32gemm"), ({"exact": False}, "int_fp32gemm"), ({"exact": True}, "int_mm")):
        run = _reexec_run(*base, ("int_mm", 60.0, extra))
        assert T.fastest(run, "cpuv", "llama2-7b", 2048)[1] == want
    # a faster int_mm record that was not exact never hides an exact one
    run = _reexec_run(*base, ("int_mm", 50.0, {"exact": False}), ("int_mm", 60.0, {"exact": True}))
    assert T.fastest(run, "cpuv", "llama2-7b", 2048)[:2] == (60.0, "int_mm")
    assert T.REEXEC_NAME["int_mm"] == "oneDNN int8 GEMMs"


def test_llamacpp_is_inexact_and_never_the_headline_baseline():
    """llama.cpp (``reexec/llamacpp.py``) is an inexact baseline that computes another model: a llama.cpp record
    faster than every re-execution never changes the abstract's N or the exact baseline's name, but it becomes the
    fastest inexact engine; of its configurations the fastest is chosen, a one-block build fills no full-model cell,
    and a record not marked inexact and a different model, or not run on its threads, fails the checks."""
    m = "llama2-7b"
    recs = [R.Rec("cpuv", "c-rx-llama2-7b", "1", "sp-c-rx-llama2-7b-1.out", i, "reexec", reexec(m, 2048, v, t))
            for i, (v, t) in enumerate((("int", 179.4), ("int_fp32gemm", 141.6), ("float", 93.2),
                                        ("dynint8", 56.0)), 1)]
    recs.append(R.Rec("cpuv", "c-llama2-7b-2048", "2", "sp-c-llama2-7b-2048-2.out", 1, "perf",
                      perf(m, 2048, "Kpre", 19.6, 6.43e9)))
    fast = [llamacpp(2048, "q8_0", 12.0, fa=0), llamacpp(2048, "q8_0", 10.0, fa=1), llamacpp(2048, "bf16", 15.0),
            llamacpp(64, "q8_0", 0.1, layers=1)]
    recs += [R.Rec("cpuv", "c-llamacpp-llama2-7b", "3", "sp-c-llamacpp-llama2-7b-3.out", i, "llamacpp", d)
             for i, d in enumerate(fast, 1)]
    assert all(R.kind_of(d) == "llamacpp" for d in fast)
    run = R.Run(recs)
    ctx = T.Ctx(run, B.Bench(None), {}, "cnn18c", "test platform", byte_model=False)
    c = M.Collector()
    M.cpu_client(c, ctx)
    v = {k: x.value for k, x in c.m.items()}
    # the abstract's N and the exact baseline: exact re-execution only (141.6 / 19.6 s)
    assert v["CpuRatioN"] == v["CpuRatio"] == "7.2" and v["CpuReexecName"] == "exact fp32 GEMMs"
    assert T.fastest(run, "cpuv", m, 2048)[:2] == (141.6, "int_fp32gemm")
    # the fastest inexact engine: llama.cpp's faster attention kernel (10 s, not 12 s)
    assert v["CpuInexactName"] == r"llama.cpp Q8\_0" and v["CpuInexactLong"] == "10" and v["CpuRatioInexact"] == "0.5"
    assert (v["CpuLlamaCppLong"], v["CpuLlamaCppBfLong"], v["CpuLlamaCppName"], v["CpuRatioLlamaCpp"]) == \
        ("10", "15", r"Q8\_0", "0.5")
    assert v["CpuLlamaCppShort"] is None and v["LlamaCppRelease"] == "v0.6.0"      # one block: no full-model cell
    q8 = run.reexec("cpuv", m, 2048, "cpu", T.LLAMACPP)["llamacpp_q8_0"]
    assert q8[0] == pytest.approx(10.0) and q8[1].d["flash_attn"] == "on"
    # the tables: tab:options' llama.cpp row (Q8_0 with its bytes) and tab:eval-reexec's two llama.cpp rows
    opts = T.tab_options(ctx)
    o = {(x["row"], x["column"]): x for x in opts.cells}
    assert o["llama.cpp", "2048"]["value"] == pytest.approx(10.0) and o["llama.cpp", "bytes"]["value"] == 7.16e9
    assert "sp-c-llamacpp-" in o["llama.cpp", "2048"]["source"] and o["llama.cpp", "64"]["value"] is None
    assert o["Re-exec. CPU", "2048"]["value"] == 141.6
    row = next(r for r in opts.rows if r.startswith("llama.cpp"))
    assert row == r"llama.cpp~\cite{llamacpp}$^g$ & weights & \multicolumn{2}{c}{7.2\,GB Q8\_0 once} & -- & " \
                  r"10\,s & inexact \\"
    e = {(x["row"], x["column"]): x["value"] for x in T.tab_eval_reexec(ctx).cells}
    assert e["cpu 2048", r"llama.cpp Q8\_0"] == pytest.approx(10.0) and e["cpu 2048", "llama.cpp bf16"] == 15.0
    assert e["cpu 2048", "integer re-execution"] == 141.6
    # the checks: marked inexact and a different model, on its threads (no more than the CPUs it was allowed)
    good = R.Rec("cpuv", "c-llamacpp-llama2-7b", "3", "sp-c-llamacpp-llama2-7b-3.out", 2, "llamacpp", fast[1])
    assert R.checks(good) == []
    unmarked = {k: x for k, x in fast[1].items() if k != "exact"}
    for d in (unmarked, dict(fast[1], exact=True), dict(fast[1], same_model=True), dict(fast[1], threads=16),
              dict(fast[1], env=dict(_env(), cpu_affinity=4))):
        assert R.checks(R.Rec("cpuv", "c-llamacpp-llama2-7b", "3", "x.out", 1, "llamacpp", d)), d


def test_the_plan_comes_from_the_records(tmp_path):
    """Mode C's plan: --policy, else the one plan the records name (a mode-C record's policy, the benchmark tables'
    pol<plan> tags), else the paper's; a plan the records do not name is a WARNING.  The paper's plan has one
    torch-free copy, which the run package and the experiments' defaults equal."""
    from experiments.common import cli
    from experiments.comparison import count_outcomes as co
    rec = R.Rec("cpuv", "c-llama2-7b-64", "1", "x", 1, "perf", perf("llama2-7b", 64, "C", 1.9, 3.25e8))
    assert N.record_plans(R.Run([rec]), B.Bench(None)) == {"cnn18c"}
    assert N.record_plans(R.Run([]), B.Bench(CODE / "artifacts" / "comparison" / "tables_l40s_improved2")) == {"auto"}
    assert N.choose_plan(None, {"auto"}) == ("auto", []) and N.choose_plan(None, set()) == (B.DEFAULT_POLICY, [])
    chosen, warn = N.choose_plan("cnn18c", {"auto"})
    assert chosen == "cnn18c" and "WARNING plan" in warn[0]
    assert N.choose_plan(None, {"auto", "cnn18c"})[1]
    run_sh = (CODE / "experiments" / "package" / "run.sh").read_text(encoding="utf-8")
    assert re.search(r"^POL=(\S+)", run_sh, re.M).group(1) == B.DEFAULT_POLICY == cli.PAPER.policy
    assert co.PAPER_POLICY == T.Ctx.policy == B.DEFAULT_POLICY
    assert co.PAPER_PROB_MAX == B.DEFAULT_PROB_MAX == cli.PAPER.prob_max == B.Bench(None).prob_max


def _setup_size(setup, columns, combined, at_lam, untrusted=79.4, factor=1.21):
    """One model's ``costs/setup_bytes.setup_model`` output (MB), with 1% more columns drawn with replacement."""
    return {"setup_MB": setup, "columns_MB": columns, "combined_rows_MB": combined, "paths_MB_bound": 12.0,
            "columns_MB_at_lam": at_lam, "query_columns_MB_honest": 65.6, "query_columns_MB_untrusted": untrusted,
            "t0": {"t1": 100, "t2": 50}, "t0_with_replacement": {"t1": 101, "t2": 50.5}, "query_factor": factor}


def test_the_setup_proof_from_a_synthetic_byte_model(monkeypatch):
    """The setup-proof family (``macros.setup_proof``) from a synthetic byte model and plan: Llama-2-7B's sizes and
    shares at a = 100 and 1,000 amortised queries, the maxima and ranges over the four models, the largest radius in
    words (GPT-2's half, not Llama-2-7B's quarter), Llama-2-7B's largest t0 share at a = 1,000, and the growth of the
    run's 2,048-token proof (rounded up)."""
    from types import SimpleNamespace as NS
    plan = NS(coded=[NS(name="q", n_points=1 << 20, row_length=1 << 18), NS(name="o", n_points=1 << 18,
                                                                                row_length=1 << 16)],
              groups=[("t1", ["q"]), ("t2", ["o"])])

    def setup_of(model, policy, a):
        assert policy == "cnn18c"
        if model != "llama2-7b":                              # radii of half of the distance
            return _setup_size(100.0, 90.0, 2.0, 40.0), NS(t0=[("t1", 10), ("t2", 10)],
                                                           e=[("q", 3 << 16), ("o", 3 << 14)]), plan
        size = _setup_size(2400.0, 2300.0, 52.8, 1200.0) if a == 100 else _setup_size(3000.0, 2950.0, 30.0, 1350.0,
                                                                                         70.1, 1.02)
        return size, NS(t0=[("t1", 19_923), ("t2", 300)], e=[("q", 3 << 15), ("o", 3 << 13)]), plan

    monkeypatch.setattr(T, "setup_of", setup_of)
    monkeypatch.setattr(T, "params_values", lambda pol, v1: {"int": {"t_min": 23}, "fs": {"t_min": 25}, "trees": 7})
    monkeypatch.setattr(M, "split_factor", lambda m, pol: 4.46 if m == "llama2-7b" else 4.0)
    # the per-query columns are interactive, so their growth is taken over the interactive 2,048-token proof
    bench = NS(name="stub", llm_cost=lambda m, s, k: {"bytes": 6.6e9, "cell": "c"} if (m, s, k) == (
        "llama2-7b", 2048, "c_cpu_int") else None)
    c = M.Collector()
    M.setup_proof(c, T.Ctx(R.Run([]), bench, {}, "cnn18c", byte_model=True))
    v = {k: x.value for k, x in c.m.items()}
    assert (v["ParamsColumnsMinInt"], v["ParamsColumnsMinFs"], v["ParamsTrees"]) == ("23", "25", "7")
    assert v["SplitColumnFactorRange"] == "4.0--4.5"
    assert (v["SetupGBLlamaA"], v["SetupMBLlamaA"], v["SetupGBLlamaB"]) == ("2.4", "2{,}400.0", "3.0")
    assert (v["SetupUntrustedMBLlamaA"], v["SetupUntrustedMBLlamaB"], v["SetupHonestMBLlama"]) == \
        ("79.4", "70.1", "65.6")
    assert (v["SetupCombinedShareLlamaA"], v["SetupCombinedShareLlamaB"], v["SetupCombinedMBLlama"]) == \
        ("2.2", "1.0", "52.8")
    assert (v["SetupColumnsShareLlamaA"], v["SetupNonColumnsShareLlamaA"], v["SetupNonColumnsShareMaxA"]) == \
        ("95.8", "4.2", "10.0")
    assert (v["SetupPathsMBLlamaA"], v["SetupPathsShareLlamaA"], v["SetupAtLamShareLlamaA"]) == ("12.0", "0.50", "50.0")
    assert v["SetupAtLamShareRange"] == "40.0--50.0" and v["SetupReplacementExcessA"] == "1.0"
    assert (v["SetupQueryGrowthLlamaA"], v["SetupQueryGrowthLlamaB"]) == ("21", "2")
    assert (v["SetupQueryShareLlamaA"], v["SetupQueryShareLlamaB"]) == ("1", "1")    # 13.8 and 4.5 MB of 6.6 GB
    assert v["SetupTZeroMaxShareB"] == "1.9" and v["SetupRadiusMax"] == "half"
    assert None not in v.values()
    off = M.Collector()
    M.setup_proof(off, T.Ctx(R.Run([]), bench, {}, "cnn18c", byte_model=False))
    assert set(off.m) == set(c.m) and all(x.value is None for x in off.m.values())


@pytest.mark.skipif(os.environ.get("PVI_SLOW") != "1", reason="about ten minutes on a laptop CPU (PVI_SLOW=1)")
def test_the_setup_proof_from_the_library_gives_the_fallback():
    """The setup-proof family from the library under cnn18c (``costs/setup_bytes.py`` and ``pvi.commit.setup_proof``
    for the four models of ``tab:setup``) equals the values that ``paper_sp2027/fallback/numbers.tex`` prints; only
    the growth of the run's 2,048-token proof needs a run."""
    c = M.Collector()
    M.setup_proof(c, T.Ctx(R.Run([]), B.Bench(None), {}, "cnn18c", byte_model=True))
    got = {k: x.value for k, x in c.m.items() if x.value is not None}
    values = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (PAPER / "fallback" / "numbers.tex")
                             .read_text(encoding="utf-8")))
    assert got == {k: values[k] for k in got}
    assert set(c.m) - set(got) == {"SetupQueryShareLlamaA", "SetupQueryShareLlamaB"}
    assert (got["SetupGBLlamaA"], got["SetupUntrustedMBLlamaB"], got["SetupCombinedShareLlamaA"]) == \
        ("2.4", "79.4", "2.2")
    assert (got["SetupTZeroMaxShareB"], got["SetupRadiusMax"], got["ParamsColumnsMinInt"]) == ("1.9", "a quarter", "23")


def test_the_stored_attack_documents_regenerate_the_fallback(run_dir):
    """The attack tier's stored documents (the records behind today's paper) give the fallback's tab:inrange body and
    the fallback values of the text numbers they define."""
    tmp, logs = run_dir
    ctx, _ = _ctx(tmp, logs)
    tb = T.tab_inrange(ctx)
    fallback = (PAPER / "fallback" / "tab_inrange.tex").read_text(encoding="utf-8").splitlines()[1:]
    assert [r.strip() for r in tb.rows] == [r.strip() for r in fallback]
    m, _ = M.build(ctx, [])
    values = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (PAPER / "fallback" / "numbers.tex")
                             .read_text(encoding="utf-8")))
    same = ("AttackOptLSucceeded", "AttackOptLPrompts", "AttackOptLPaths", "AttackOptLBackdoorBack",
            "AttackOptLBackdoorHacked", "PathsLlama", "PathsLlamaAbout", "WidthLlama", "OpenAllLlama",
            "MnistSinglePass",
            "MnistBackdoorPass", "MnistBackdoorRuns", "MnistBackdoorPredicted", "MnistForgeryNodes", "SamplerUniform",
            "SamplerContribution", "SamplerStatic", "SamplerSaliency", "ZeroingNeurons", "ZeroingRuns", "FloorBest",
            "FloorVsUniform", "FloorOthers", "MixtureLayerOne", "MixtureLayerTwo", "LocatorOne", "LocatorTwo",
            "StealthNeurons", "StealthPass", "ReproRuns", "ReproCaught", "ReproZeroTolerance", "ReproQueries",
            "ReachTokens", "InrangeCalibTokens",
            "InrangeChanged", "InrangeKMedian", "InrangeKMax", "InrangePMedianMax", "InrangePMax", "InrangeDown",
            "InrangeDownKMedian", "InrangeOptLPMedianMax", "InrangeOptLKRange", "InrangePathsOptM", "InrangePathsOptL",
            "TvPositions", "ZeroingZeroShare", "MnistStealthPass", "MnistStealthNeurons", "StealthExceed",
            "InrangeDownLast", "InrangeOptLChanged")
    assert {k: m[k].value for k in same} == {k: values[k] for k in same}
    # the stored sampler document is the legacy recipe's: the certified model's distances wait for the tv tier
    assert m["TvTopK"].value == "0.0063" and all(values[k] == r"\todo{tv tier}"
                                                  for k in ("TvTopK", "TvFull", "TvModel", "TvTwenty"))


# ------------------------------------------------------------------------------------------------ the command
def test_the_command_writes_every_file_and_strict_fails_on_a_missing_cell(run_dir, capsys):
    tmp, logs = run_dir
    assert N.main(_args(tmp, logs)) == 0
    out = tmp / "out"
    names = {p.name for p in out.glob("*.tex")}
    assert names == {p.name for p in (PAPER / "fallback").glob("*.tex")}
    macro = re.compile(r"\\newcommand\{\\(\w+)\}")
    assert set(macro.findall((out / "numbers.tex").read_text(encoding="utf-8"))) == \
        set(macro.findall((PAPER / "fallback" / "numbers.tex").read_text(encoding="utf-8")))
    assert "not in the run" in (out / "numbers.tex").read_text(encoding="utf-8")     # a missing number is visible
    rows = list(csv.DictReader(open(out / "cells.csv", encoding="utf-8")))
    assert any(r["table"] == "tab:options" and r["source"].startswith("sp-c-") for r in rows)
    assert "tab:eval-llm" in (out / "coverage.txt").read_text(encoding="utf-8")
    tables, macros = N.folder_files(out)                     # the written cells agree with the written macros
    assert len(tables) == len(T.TABLE_FILES) and macros["CpuReexecLong"] == "142" and N.same_value(tables, macros) == []
    capsys.readouterr()
    assert N.main(_args(tmp, logs, "--strict", "--check")) == 1           # most cells are missing in this run
    assert "STRICT:" in capsys.readouterr().out


def test_a_failed_check_or_an_unparsable_line_fails(run_dir, capsys):
    tmp, logs = run_dir
    _write(logs, "sp-c-qwen3-4b-64-90.out", [perf("qwen3-4b", 64, "Kpre", 0.7, 1.6e8, accepted=4)])
    assert N.main(_args(tmp, logs, "--check")) == 1
    assert "CHECK FAILED sp-c-qwen3-4b-64-90.out" in capsys.readouterr().out
    (logs / "sp-c-qwen3-4b-64-90.out").unlink()
    differs = perf("qwen3-4b", 64, "Kpre", 0.7, 1.6e8, client=(1.0, 0.5))
    differs["client_memory"].update(agree=False, accepted=4)
    _write(logs, "sp-c-qwen3-4b-64-92.out", [differs])
    assert N.main(_args(tmp, logs, "--check")) == 1
    assert "the client process's verdicts differ" in capsys.readouterr().out
    (logs / "sp-c-qwen3-4b-64-92.out").unlink()
    _write(logs, "sp-v1-gpt2-64-91.out", [], extra="{not json\n")
    assert N.main(_args(tmp, logs, "--check")) == 1
    assert "PARSE sp-v1-gpt2-64-91.out:2" in capsys.readouterr().out
    (logs / "sp-v1-gpt2-64-91.out").unlink()
    text = (logs / "sp-smoke-1.out").read_text(encoding="utf-8").replace("SMOKE OK", "")
    (logs / "sp-smoke-1.out").write_text(text, encoding="utf-8")
    assert N.main(_args(tmp, logs, "--check")) == 1
    assert "did not end in SMOKE OK" in capsys.readouterr().out


def _node(host: str, gpu: str | None = "NVIDIA L40S", cpu: str = ENV["cpu"], driver: str = "535.183.01") -> str:
    """The node line ``experiments/package/node.py`` writes at the start of a job's log."""
    return "node: " + json.dumps({"host": host, "gpu": gpu, "gpu_count": 1 if gpu else 0, "driver": driver,
                                  "cpu": cpu, "cuda_visible_devices": "0"}) + "\n"


def test_jobs_of_another_gpu_or_cpu_model_fail_and_nodes_and_drivers_are_listed(run_dir, capsys):
    """Every job's GPU and CPU (its log's node line and its records' environments) must be one model each: a job on
    another GPU model, one that saw no GPU, or one on another CPU model is a ``CHECK FAILED platform`` line that
    names it; several nodes and drivers of the same models are allowed and listed in the report's machines line."""
    tmp, logs = run_dir
    _write(logs, "sp-g-h2d-95.out", [], extra=_node("n-805", driver="595.99.02"))
    _write(logs, "sp-tv-checks-96.out", [], extra=_node("n-801"))
    assert N.main(_args(tmp, logs, "--check")) == 0
    out = capsys.readouterr().out
    assert "CHECK FAILED platform" not in out
    machines = next(line for line in out.splitlines() if line.startswith("machines: "))
    assert "n-805 (1 jobs)" in machines and "n-801 (1 jobs)" in machines and "(allowed)" in machines
    assert "drivers: 535.183.01 (1 jobs), 595.99.02 (1 jobs) (allowed)" in machines
    _write(logs, "sp-a-mnist-97.out", [], extra=_node("t-806", cpu="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz"))
    _write(logs, "sp-q-opt125-98.out", [], extra=_node("n-802", gpu=None))
    assert N.main(_args(tmp, logs, "--check")) == 1
    out = capsys.readouterr().out
    gpu = next(line for line in out.splitlines() if "GPU models" in line)
    cpu = next(line for line in out.splitlines() if "CPU models" in line)
    assert gpu.startswith("CHECK FAILED platform: the jobs name 2 GPU models: 'NVIDIA L40S' (") and \
        "'none' (1 jobs: sp-q-opt125-98.out)" in gpu
    assert "'Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz' (1 jobs: sp-a-mnist-97.out)" in cpu
    assert R.nodes(logs)["sp-a-mnist-97.out"]["host"] == "t-806"


def test_the_cost_records_are_at_the_texts_probability_scale(run_dir, capsys):
    """The text states 11-bit attention probabilities for the certified models and every cost measurement: a cost
    record of another scale, or of none, is a text check that fails (a warning in a rehearsal); the 8-bit side of the
    improve tier's A/B, an image model's ``perf.py`` record, a floating-point re-execution and the quality tier's
    records are not cost records of an integer decoder at the paper's scale."""
    tmp, logs = run_dir
    unscaled = perf("qwen3-4b", 64, "C", 1.0, 1e8)
    del unscaled["prob_max"]
    _write(logs, "sp-c-qwen3-4b-64-92.out", [perf("qwen3-4b", 64, "Kpre", 0.7, 1.6e8, prob_max=255), unscaled,
                                            perf("lenet5", None, "C", 0.1, 1e5, prob_max=None),
                                            reexec("qwen3-4b", 64, "float", 3.0, n_layers=36)])
    ctx, problems = _ctx(tmp, logs)
    M.build(ctx, problems)
    assert [p.split(" builds")[0] for p in problems] == [f"{M.TEXT_CLAIM} sp-c-qwen3-4b-64-92.out:2",
                                                         f"{M.TEXT_CLAIM} sp-c-qwen3-4b-64-92.out:3"], problems
    assert "prob_max 255" in problems[0] and "prob_max None" in problems[1]
    kept, warned = N.demote_text_claims(problems)
    assert kept == [] and len(warned) == 2
    assert N.main(_args(tmp, logs, "--check")) == 1
    assert "builds its decoder at prob_max 255" in capsys.readouterr().out


def test_job_coverage_from_a_dry_run_listing(run_dir):
    tmp, logs = run_dir
    listing = tmp / "dry.txt"
    listing.write_text(
        'sbatch -o logs/x/%x-%j.out -J sp-c-llama2-7b-64 --dependency=singleton --wrap "..."\n'
        'sbatch -o logs/x/%x-%j.out -J sp-c-qwen3-4b-64 --dependency=singleton --wrap "..."\n'
        'sbatch -o logs/x/%x-%j.out -J sp-b-c-lenet5 --dependency=singleton --wrap "..."\n'
        'sbatch -o logs/x/%x-%j.out -J sp-b-k-lenet5 --dependency=singleton --wrap "..."\n'
        'sbatch -o logs/x/%x-%j.out -J sp-a-opt67 --wrap "mkdir -p /r/logs/x/attack && python -m a --out '
        '/r/logs/x/attack/real_llm_attack.json && python -m a --out /r/logs/x/attack/real_llm_attack_auto.json"\n'
        'sbatch -o logs/x/%x-%j.out -J sp-a-game-opt13 --wrap "python -m g --out '
        '/r/logs/x/attack/adaptive_game_opt-1.3b.json"\n# tier attack: 2 jobs, job limits 9 h, estimate 4-6 h\n'
        '# cpuv bench attack: 6 jobs, job limits 12 h\n', encoding="utf-8")
    jobs = N.expected_jobs(listing)
    assert jobs["sp-a-opt67"] == ["real_llm_attack.json", "real_llm_attack_auto.json"]
    recs = R.read_logs(logs, [])
    assert N.job_coverage(jobs, logs, recs) == ["sp-c-qwen3-4b-64", "sp-b-k-lenet5", "sp-a-game-opt13"]
    # a dry run that stopped early (no summary line, or a summary of other jobs) is refused, and --expect fails
    lines = listing.read_text(encoding="utf-8").splitlines()
    for bad in (lines[:-2], lines[:3] + lines[-1:]):
        listing.write_text("\n".join(bad) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            N.expected_jobs(listing)
    assert N.main(_args(tmp, logs, "--check", "--expect", str(listing))) == 1


def test_the_rehearsal_needs_every_tier(run_dir, capsys):
    tmp, logs = run_dir
    assert N.main(_args(tmp, logs, "--rehearsal", "--check")) == 1
    out = capsys.readouterr().out
    assert "REHEARSAL: tier tv has no record" not in out          # the tv document is there
    assert "REHEARSAL: tier batch has no record of ['bench_env']" in out


def test_a_document_named_by_another_model_has_its_kind_and_fills_no_cell(tmp_path):
    """The OPT scripts' documents of a rehearsal, named by their model (OPT-125M), count for their tier and kind by
    their names' patterns (``records.DOCUMENTS``) but fill no table cell and no text number, which select the full
    run's names; under those names the same documents fill them."""
    assert [R.doc_kind(n) for n in ("attack.json", "adaptive_game_mlp_mnist.json", "adaptive_game_opt-125m.json",
                                    "inrange_llm_opt-125m_down.json", "real_llm_attack_opt-125m_auto.json",
                                    "sampler_tv_opt-350m.json", "inrange_llm_opt-125m.json", "sampler_tv_x.npz",
                                    "notes.json")] \
        == ["attack_mnist", "adaptive_game", "adaptive_game_llm", "inrange", "attack_llm", "tv", None, None, None]
    game = {"per_layer": {"6": {"d_ff": 8192, "caps": {c: {"mixture": {"detection_per_path": 0.4},
                                                           "samplers": {"uniform": {"detection_per_path": 0.003}}}
                                                       for c in ("90", "95", "99", "100")}}}}
    names = {"real_llm_attack.json": "real_llm_attack_opt-125m.json",
             "real_llm_attack_auto.json": "real_llm_attack_opt-125m_auto.json",
             "inrange_llm_opt-1.3b_any.json": "inrange_llm_opt-125m_any.json",
             "inrange_llm_opt-1.3b_down.json": "inrange_llm_opt-125m_down.json",
             "inrange_llm_opt-6.7b_any.json": "inrange_llm_opt-350m_any.json",
             "inrange_llm_opt-6.7b_down.json": "inrange_llm_opt-350m_down.json",
             "adaptive_game_opt-1.3b.json": "adaptive_game_opt-125m.json",
             "sampler_tv_opt-1.3b.json": "sampler_tv_opt-125m.json"}

    def build(folder: str | None):
        """``(documents, filled macros, filled cells per table)`` of a run with the documents under their full-run
        names (``final``), their rehearsal names (``rehearsal``) or none (None)."""
        files, problems = [], []
        if folder:
            logs = tmp_path / folder
            (logs / "attack").mkdir(parents=True)
            for full, other in names.items():
                name = other if folder == "rehearsal" else full
                path = (logs if name.startswith("sampler_tv") else logs / "attack") / name
                if full.startswith("adaptive_game"):
                    path.write_text(json.dumps(game), encoding="utf-8")
                else:
                    shutil.copy(RESULTS / full, path)
            files = R.read_files(logs, problems)
        ctx = T.Ctx(R.Run([], files), B.Bench(None), B.literature(), "cnn18c", "test platform", byte_model=False)
        macros, _ = M.build(ctx, problems)
        cells = {tb.name: tb.filled for tb in T.build_all(ctx, problems)}
        assert not problems
        return files, {k: m.value for k, m in macros.items() if m.value is not None}, cells

    _, macros0, cells0 = build(None)
    files, macros, cells = build("rehearsal")
    assert {(Path(r.path).name, r.tier, r.kind) for r in files} == {
        (other, "tv" if other.startswith("sampler_tv") else "attack", R.FILES[full])
        for full, other in names.items()}
    assert macros == macros0 and cells == cells0                 # nothing filled
    _, macros, cells = build("final")
    assert macros["InrangeKMax"] == "60" and macros["AttackOptLPaths"] == "454{,}248" and macros["TvTopK"] == "0.0063"
    assert macros["GameLayerSix"] == "40" and cells["tab_inrange"] > cells0["tab_inrange"]


# ------------------------------------------------------------------------------------------------ the paper
def test_the_paper_inputs_every_table_and_defines_every_macro():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    listed = re.search(r"\\@for\\pvi@name:=(.*?)\\do", main, re.S).group(1)
    assert [n.strip() for n in listed.replace("%", "").split(",")] == ["numbers", *T.TABLE_FILES]
    tex = "".join(p.read_text(encoding="utf-8") for p in (PAPER / "sections").glob("*.tex")) + main
    used = set(re.findall(r"\\pvitable\{(\w+)\}", tex))
    assert used == {"numbers", *T.TABLE_FILES}
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "fallback" / "numbers.tex").read_text(
        encoding="utf-8")))
    assert defined == set(M.names())
    macros_used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", tex))
    assert macros_used - defined == set() and len(macros_used & defined) > 250


def test_the_fallback_tables_agree_with_their_macros():
    """Every cell of ``paper_sp2027/fallback/`` that prints the same value of the same record as a macro
    (``numbers.SAME_VALUE``) agrees with the fallback's macro up to their rounding, and a changed cell or macro is
    caught; one quantity from two jobs (``numbers.TWO_JOBS``) only warns."""
    tables, macros = N.folder_files(PAPER / "fallback")
    assert N.same_value(tables, macros) == []
    compared = [s for s in N.SAME_VALUE if N.printed(N._cell(tables, s), s[5], s[6]) is not None
                and N.printed(macros.get(s[7]), 0, s[8]) is not None]
    assert len(compared) >= 60
    assert all(N._cell(tables, s) is not None for s in N.SAME_VALUE)        # every pair names a cell of the fallback
    broken = dict(tables, tab_options=tables["tab_options"].replace(r"& 142\,s &", r"& 93\,s &"))
    assert [x for x in N.same_value(broken, macros) if "CpuReexecLong" in x]
    assert len(N.same_value(tables, dict(macros, CpuVerifyLong="18.5"))) == 2    # tab:options and tab:eval-reexec
    assert N.two_jobs(tables, macros) == []
    assert N.two_jobs(tables, dict(macros, VOneVZeroVerifyLlama="25.0"))
    assert N.printed(r"2.9--3.5$\times$") == (2.9, 3.5, pytest.approx(0.1))
    assert N.printed(r"\textbf{17.6\,s (46$\times\downarrow$)}", 1) == (46.0, 46.0, 1.0)
    assert N.printed(r"431\,ms")[0] == pytest.approx(0.431) and N.printed(r"\todo{bench tier}") is None


def test_the_definition_of_the_cells():
    """The final run's language-model cells carry the tag of its 11-bit attention probabilities (``pmax2047``), the
    stored runs' 8-bit cells none; the image models have no attention."""
    d = B.definition_tagsets("cnn18c")
    assert frozenset({"wire", "prune", "pmax2047", "polcnn18c"}) in d[("llm", "C")]
    assert frozenset({"wire", "batch", "prune", "lookups", "pmax2047"}) in d[("llm", "Kpre")]
    assert frozenset({"thr1", "wire", "prune", "lookups", "pmax2047"}) in d[("llm", "Kpre")]
    assert frozenset({"thr1", "prune", "lookups", "pmax2047"}) not in d[("llm", "Kpre")]  # the uncompressed by-product
    assert d[("cnn", "C")] == {frozenset({"wire", "polcnn18c"})} and d[("cnn", "Kpre")] == {frozenset({"wire"})}
    stored = B.definition_tagsets("cnn18c", B.STORED_PROB_MAX)
    assert frozenset({"wire", "prune", "polcnn18c"}) in stored[("llm", "C")] and stored[("cnn", "C")] == d[("cnn", "C")]
    bench = B.Bench(None, "cnn18c")
    assert bench.cnn_cell("vgg16", "C", "fs") == "defence_C_fs_lam128_rate4_wire_polcnn18c"
    assert bench.cnn_cell("vgg16", "Kpre", "int") == "defence_Kpre_int_lam128_rate4_wire"


def test_the_figure_checks_catch_small_overlapping_and_clipped_text():
    plt = pytest.importorskip("matplotlib.pyplot")
    from experiments.paper import figures
    fig = plt.figure(figsize=(2.0, 1.0))
    fig.text(0.1, 0.5, "small", fontsize=6)
    fig.text(0.1, 0.5, "on top")
    fig.text(0.98, 0.5, "clipped at the edge")
    bad = " ".join(figures.layout_problems(fig))
    assert "6.0 pt" in bad and "overlaps" in bad and "leaves the canvas" in bad
    plt.close(fig)


def test_the_schematic_figures(tmp_path):
    """The schematics pass their layout checks with the pinned matplotlib (``requirements.txt``), whose text
    metrics they are laid out for."""
    matplotlib = pytest.importorskip("matplotlib")
    pin = re.search(r"^matplotlib==(\S+)", (CODE / "requirements.txt").read_text(encoding="utf-8"), re.M)
    if pin is None or matplotlib.__version__ != pin.group(1):
        pytest.skip(f"matplotlib {matplotlib.__version__}; the figures are laid out for {pin and pin.group(1)}")
    from experiments.paper import figures
    paths = figures.schematics(tmp_path)
    assert [p.name for p in paths] == ["overview_attack.pdf", "overview_defence.pdf", "protocol.pdf"]
    assert all(p.stat().st_size > 1000 for p in paths)
