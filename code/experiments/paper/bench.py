"""The benchmark cells behind the paper's tables, from one platform's aggregated tables.

``costs/bench.py`` writes the run's benchmark records to ``artifacts/comparison/raw_<platform>/`` and
``comparison/aggregate.py --platform <name>`` turns them into ``tables_<platform>/``: ``measured_summary.csv`` (per
cell and metric the median over the queries) and ``llm_full_model.csv`` (the full-model costs of the language models;
the final run builds every model in full, so every row used here is ``measured``).  This module defines, in one
place, which cell holds each number of the cost tables (:data:`CELLS`), and reads it.

A cell is named by its settings (``costs/bench.py``'s docstring): the mode, the challenges (``fs`` for Fiat--Shamir,
``int`` for interactive), lambda, the prompt length and blocks of a decoder, then the tags of the settings that
differ from the reference -- ``thr<n>`` (verifier threads other than 8), ``wire`` (the compact encoding), ``gpuv``
and ``stream`` (the streaming GPU verifier), ``batch``, ``prune`` (the pruned last block), ``lookups`` (the verifier's
own embedding rows, modes K and Kpre) and ``pol<plan>`` (mode C under a commitment plan).  The final run's cells
(``package/run.sh``'s ``bench`` and ``batch`` tiers) are all under Fiat--Shamir, except the interactive cells of the
comparison with zkLLM (``C:int`` at 2,048 tokens) and with Maverick (``Kpre:int``, lambda 40, one thread).

    ===================  =====  ==========  ======  =======  ====================================
    cell                 mode   challenges  lambda  threads  tags
    ===================  =====  ==========  ======  =======  ====================================
    ``c_cpu``            C      fs          128     8        wire, prune, pol<plan>
    ``c_gpu``            C      fs          128     8        wire, gpuv, stream, prune, pol<plan>
    ``kpre_cpu``         Kpre   fs          128     8        wire, prune, lookups
    ``kpre_gpu``         Kpre   fs          128     8        wire, gpuv, stream, prune, lookups
    ``c_cpu_int``        C      int         128     8        wire, prune, pol<plan>
    ``c_gpu_int``        C      int         128     8        wire, gpuv, stream, prune, pol<plan>
    ``maverick``         Kpre   int         40      1        thr1, prune, lookups (no compact encoding)
    ``maverick_wire``    Kpre   int         40      1        thr1, wire, prune, lookups
    image models, C      C      fs / int    l       8        wire, pol<plan>
    image models, Kpre   K/Kpre int         l       8        wire
    ===================  =====  ==========  ======  =======  ====================================

The batch tier's cells add the tag ``batch``.  :func:`definition_tagsets` gives these tag sets to
``comparison/count_outcomes.py --definition``, so that the outcome counts cover exactly the defining cells.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from experiments.common.paths import BENCH, RESULTS

PROVE = ("prove_forward", "prove_lookups", "prove_fold", "prove_open", "prove_encode", "fs_hash")
"""The timing parts of one proof (``fs_hash``: the Fiat--Shamir hashing, paid by both parties)."""
VERIFY = ("verify_decode", "verify_derive", "verify_lookups", "verify_fold", "verify_products", "verify_columns",
          "verify_upload", "verify_total", "fs_hash")
"""The timing parts of one verification (``verify_total``: the streaming verifier's overlapped phases, timed as one;
``verify_upload``: a GPU client's copy of the proof)."""
OVERLAPPED = ("verify_derive", "verify_lookups", "verify_products", "verify_columns", "verify_upload")
"""The verify phases that the streaming verifier's ``verify_total`` covers (never counted with it)."""
DEFAULT_POLICY = "cnn18c"
"""Mode C's commitment plan of the final run: the torch-free copy of ``experiments.common.cli.PAPER.policy``, which
``paper/tables.py``, ``paper/numbers.py`` and ``comparison/count_outcomes.py`` read (``tests/test_paper_numbers.py``
checks that it equals ``cli.PAPER.policy`` and run.sh's ``POL`` default)."""


@dataclass(frozen=True)
class Spec:
    """The settings of one kind of cell: mode, challenges, lambda, verifier threads, tags (``pol`` stands for the
    run's plan)."""

    mode: str
    challenges: str
    lam: int
    threads: int
    tags: tuple


CELLS = {
    "c_cpu": Spec("C", "fs", 128, 8, ("wire", "prune", "pol")),
    "c_gpu": Spec("C", "fs", 128, 8, ("wire", "gpuv", "stream", "prune", "pol")),
    "kpre_cpu": Spec("Kpre", "fs", 128, 8, ("wire", "prune", "lookups")),
    "kpre_gpu": Spec("Kpre", "fs", 128, 8, ("wire", "gpuv", "stream", "prune", "lookups")),
    "c_cpu_int": Spec("C", "int", 128, 8, ("wire", "prune", "pol")),
    "c_gpu_int": Spec("C", "int", 128, 8, ("wire", "gpuv", "stream", "prune", "pol")),
    "maverick": Spec("Kpre", "int", 40, 1, ("thr1", "prune", "lookups")),
    "maverick_wire": Spec("Kpre", "int", 40, 1, ("thr1", "wire", "prune", "lookups")),
}
"""The language-model cells of the final run (module docstring)."""
CNN_C = ("wire", "pol")
CNN_K = ("wire",)


def tags_of(spec_tags, policy: str) -> frozenset:
    """A spec's tags with ``pol`` replaced by ``pol<policy>``."""
    return frozenset(f"pol{policy}" if t == "pol" else t for t in spec_tags)


def tagset(variant: str) -> frozenset:
    """``'_wire_gpuv_prune_polcnn18c'`` -> ``{'wire', 'gpuv', 'prune', 'polcnn18c'}``."""
    return frozenset(t for t in (variant or "").split("_") if t)


def definition_tagsets(policy: str = DEFAULT_POLICY) -> dict:
    """``{(suite, mode): {tag sets}}`` of the defining defence and tamper cells under the plan ``policy``: what
    ``comparison/count_outcomes.py --definition`` counts.  Only cells of the compact encoding (``wire``): the
    uncompressed cell of the comparison with Maverick is a by-product.  A language model's CPU-verifier cell may also
    be a batch cell (the tag ``batch`` added)."""
    out: dict = defaultdict(set)
    for spec in CELLS.values():
        tags = tags_of(spec.tags, policy)
        if "wire" not in tags:
            continue
        out[("llm", spec.mode)].add(tags)
        if "gpuv" not in tags and "thr1" not in tags:
            out[("llm", spec.mode)].add(tags | {"batch"})
    out[("cnn", "C")].add(tags_of(CNN_C, policy))
    for mode in ("K", "Kpre"):
        out[("cnn", mode)].add(frozenset(CNN_K))
    return dict(out)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _one_verifier(names, where: str) -> None:
    """The verify phases of one row come from one verifier: never ``verify_total`` with the phases it overlaps."""
    if "verify_total" in names and any(k in names for k in OVERLAPPED):
        raise ValueError(f"{where}: verify_total (the streaming verifier) together with {OVERLAPPED}")


class Bench:
    """The aggregated tables of one platform (``tables_<platform>/``); an absent folder reads as empty."""

    def __init__(self, tables: Path | None, policy: str = DEFAULT_POLICY) -> None:
        self.dir = tables
        self.policy = policy
        self.summary = self._read("measured_summary.csv")
        self.llm = self._read("llm_full_model.csv")
        self.idx = {}
        for r in self.summary:
            key = (r["suite"], r["model"], r["cell"], r["metric"], r.get("batch", ""), r.get("attack", ""),
                   r.get("lam", ""), r.get("stage", ""), r.get("k", ""))
            self.idx[key] = r

    def _read(self, name: str) -> list[dict]:
        path = self.dir / name if self.dir else None
        if path is None or not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    @property
    def name(self) -> str:
        return self.dir.name if self.dir else "(no tables)"

    def hardware(self) -> tuple[set, set]:
        """The provers and verifier CPUs of the tables (one each on one platform)."""
        rows = self.summary + self.llm
        return ({r.get("prover_hw") for r in rows if r.get("prover_hw")},
                {re.sub(r"^.* \(GPU client\) \+ ", "", r.get("verifier_hw", "")) for r in rows if r.get("verifier_hw")})

    # ---------------------------------------------------------------- image models
    def cnn_cell(self, model: str, mode: str, challenges: str, lam: int = 128, batch: bool = False) -> str:
        """The name of an image model's defence cell of the final run."""
        tags = tags_of(CNN_C if mode == "C" else CNN_K, self.policy) | ({"batch"} if batch else set())
        order = ("thr1", "wire", "batch", "prune", "lookups")
        known = [t for t in order if t in tags] + sorted(t for t in tags if t.startswith("pol"))
        return f"defence_{mode}_{challenges}_lam{lam}_rate4" + "".join("_" + t for t in known)

    def cnn_get(self, model: str, cell: str, metric: str, **group) -> float | None:
        key = ("cnn", model, cell, metric, group.get("batch", ""), group.get("attack", ""), str(group.get("lam", "")),
               group.get("stage", ""), group.get("k", ""))
        r = self.idx.get(key)
        return _f(r["median"]) if r else None

    def cnn_cost(self, model: str, cell: str) -> dict | None:
        """``{prove, verify, bytes, cell}`` of an image model's cell, or None."""
        vals = {}
        for part in PROVE + VERIFY:
            v = self.cnn_get(model, cell, part)
            if v is not None:
                vals[part] = v
        if "prove_forward" not in vals:
            return None
        _one_verifier(vals, f"cnn/{model}/{cell}")
        return {"prove": sum(vals.get(k, 0.0) for k in PROVE), "verify": sum(vals.get(k, 0.0) for k in VERIFY),
                "bytes": self.cnn_get(model, cell, "bytes_total"), "cell": cell}

    def cnn_metric(self, model: str, cell: str, metric: str, **group) -> float | None:
        """One metric of a cell that is not a defence cell (``facts``, ``attack_float``, ``sampling``)."""
        return self.cnn_get(model, cell, metric, **group)

    def cnn_cells(self, model: str, base: str) -> list[str]:
        """An image model's stored cells of one base name (``facts``, ``attack_float``, ``sampling``: their names
        carry the run's tags), the reference cell first (``<base>_wire``, else ``<base>``), cells of other batch
        sizes last."""
        cells = {k[2] for k in self.idx if k[0] == "cnn" and k[1] == model and (k[2] == base
                                                                               or k[2].startswith(base + "_"))}
        return sorted(cells, key=lambda c: (c not in (f"{base}_wire", base), "batch" in tagset(c[len(base):]), c))

    # ---------------------------------------------------------------- language models
    def llm_cost(self, model: str, seq: int, cell: str, batch: str = "") -> dict | None:
        """``{prove, verify, bytes, params, n_layers, cell}`` of one language model's full-model cell (``cell`` a key
        of :data:`CELLS`; ``batch``: a batch size of the batch tier), from measured rows only, or None."""
        spec = CELLS[cell]
        want = tags_of(spec.tags, self.policy) | ({"batch"} if batch else set())
        d, hit = {}, None
        for r in self.llm:
            if (r["model"], r["mode"], r["challenges"]) != (model, spec.mode, spec.challenges):
                continue
            if int(float(r["seq"])) != seq or _f(r["lam"]) != spec.lam or str(r.get("threads", "")) != str(spec.threads):
                continue
            if tagset(r.get("variant", "")) != want or str(r.get("batch") or "") != str(batch):
                continue
            if r.get("provenance") != "measured":
                continue
            d[r["metric"]] = _f(r["value"])
            hit = r
        if hit is None or "prove_forward" not in d:
            return None
        _one_verifier(d, f"llm/{model}/T{seq}/{cell}")
        verify = sum(d[k] for k in VERIFY if k in d) if any(k in d for k in ("verify_products", "verify_total")) \
            else None
        n_layers = hit.get("n_layers_full")
        variant = hit.get("variant", "")
        return {"prove": sum(d[k] for k in PROVE if k in d), "verify": verify, "bytes": d.get("bytes_total"),
                "params": _f(hit.get("params_full")), "n_layers": n_layers,
                "cell": f"defence_{spec.mode}_{spec.challenges}_lam{spec.lam}_T{seq}_L{n_layers}{variant}"}

    def llm_metric(self, model: str, seq: int, cell: str, metric: str) -> float | None:
        """One metric of a full-model cell that ``llm_full_model.csv`` does not sum (e.g. ``bytes_claims``,
        ``bytes_columns``): the cell's median in ``measured_summary.csv``."""
        c = self.llm_cost(model, seq, cell)
        if c is None:
            return None
        r = self.idx.get(("llm", model, c["cell"], metric, "", "", "", "", ""))
        return _f(r["median"]) if r else None

    def llm_batches(self, model: str, seq: int, cell: str) -> dict:
        """``{B: {bytes, prove, verify}}`` per query of a batch cell (``measured_summary.csv``: the batch's median
        over its trials, divided by ``B``)."""
        spec = CELLS[cell]
        want = tags_of(spec.tags, self.policy) | {"batch"}
        out: dict = defaultdict(dict)
        pat = re.compile(rf"^defence_{spec.mode}_{spec.challenges}_lam{spec.lam}_T{seq}_L\d+(.*)$")
        for (suite, m, c, metric, batch, *_), r in self.idx.items():
            mm = pat.match(c) if suite == "llm" and m == model else None
            if not mm or tagset(mm.group(1)) != want or not batch:
                continue
            out[int(batch)][metric] = _f(r["median"])
        res = {}
        for bsz, ms in out.items():
            if "bytes_total" not in ms:
                continue
            res[bsz] = {"bytes": ms["bytes_total"] / bsz, "prove": sum(ms.get(k, 0.0) for k in PROVE) / bsz,
                        "verify": sum(ms.get(k, 0.0) for k in VERIFY) / bsz}
        return res


ZKLLM_RUNS = "zkLLM (stored runs)"
"""The system name under which :func:`literature` lists zkLLM's stored runs (:func:`zkllm_runs`)."""
ZKLLM_MODELS = {"llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B"}


def zkllm_runs(folder: Path | None = None) -> dict:
    """zkLLM's public demo run on the hardware of the run package (``artifacts/results/zkllm_l40s/``, its README):
    ``{(ZKLLM_RUNS, model, seq): row}`` from ``summary.csv`` and each run's ``env.txt``.  ``prover_s`` is the run's
    ``whole_model_s``: the median times of zkLLM's proof binaries on two decoder layers (three repetitions each; the
    one-time ``ppgen`` and ``commit-param`` and the Python scripts around the binaries excluded), with the skip
    connection that its demo does not run counted as the one it runs, times the model's layers.  ``prover_hw`` and
    ``cpu`` are the GPU and CPU models that the run's ``env.txt`` records (its folder ``<run>/`` or
    ``<model>-T<seq>/``; None if it names several)."""
    folder = folder or RESULTS / "zkllm_l40s"
    path = folder / "summary.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            env = next((p / "env.txt" for p in (folder / r["run"], folder / f"{r['model']}-T{r['seq']}")
                        if (p / "env.txt").exists()), None)
            lines = env.read_text(encoding="utf-8").splitlines() if env else []
            gpus = {x.split(",")[0].strip() for x in lines if x.startswith("NVIDIA")}
            cpus = {x.split(":", 1)[1].strip() for x in lines if x.startswith("Model name:")}
            model = ZKLLM_MODELS.get(r["model"], r["model"])
            out[(ZKLLM_RUNS, model, r["seq"])] = {
                "system": ZKLLM_RUNS, "model": model, "seq": r["seq"], "prover_s": r["whole_model_s"],
                "per_layer_s": r["per_layer_s"], "layers": r["layers"], "run": r["run"],
                "prover_hw": next(iter(gpus)) if len(gpus) == 1 else None,
                "cpu": next(iter(cpus)) if len(cpus) == 1 else None,
                "source": f"artifacts/results/zkllm_l40s/summary.csv ({r['run']})"}
    return out


def literature(path: Path | None = None, runs: Path | None = None) -> dict:
    """The published systems' values (``artifacts/comparison/tables/reported_curated.csv``, ``literature.py``) and
    zkLLM's stored runs on the run package's GPU model (``runs``, default ``artifacts/results/zkllm_l40s/``:
    :func:`zkllm_runs`): ``{(system, model, seq): row}``."""
    path = path or BENCH / "tables" / "reported_curated.csv"
    out = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            out = {(r["system"], r["model"], r["seq"]): r for r in csv.DictReader(fh)}
    return {**out, **zkllm_runs(runs)}
