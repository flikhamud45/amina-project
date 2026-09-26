"""Shared readers for the stored benchmark tables (``artifacts/comparison/tables``).

Used by ``report_tables.py``, ``count_outcomes.py`` and ``paper_assets.py``.  Headline
settings (``DEFAULT``) pick one configuration per point; nothing is chosen by sort order.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "artifacts" / "comparison" / "tables"

DEFAULT = {"rate": "4", "threads": "8", "variant": "", "lam": 128}
CNN_ORDER = ["mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224"]
LLM_LABEL = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
             "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}
# the timing parts that make up one proof and one verification (fs_hash is paid by both)
PROVE = ("prove_forward", "prove_fold", "prove_open", "fs_hash")
VERIFY = ("verify_derive", "verify_fold", "verify_products", "verify_columns", "fs_hash")


def _read(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Measured:
    """Index over ``measured_summary.csv`` (median / IQR / min / max per model, cell, metric)."""

    def __init__(self) -> None:
        self.idx = {}
        for r in _read("measured_summary.csv"):
            key = (r["suite"], r["model"], r["cell"], r["metric"], r.get("batch", ""), r.get("attack", ""),
                   r.get("lam", ""), r.get("stage", ""), r.get("k", ""))
            self.idx[key] = r

    def get(self, model, cell, metric, stat="median", suite="cnn", batch="", attack="", lam="", stage=""):
        r = self.idx.get((suite, model, cell, metric, batch, attack, str(lam) if lam != "" else "", stage, ""))
        return _f(r[stat]) if r else None

    def total(self, model, cell, parts, suite="cnn"):
        vals = [self.get(model, cell, k, suite=suite) for k in parts]
        return None if all(v is None for v in vals) else sum(v or 0 for v in vals)

    def attack_rate(self, model, cell, attack):
        """(fraction rejected, number of attempts) for one attack type."""
        vals = [(_f(r["mean"]), _f(r["n"])) for k, r in self.idx.items()
                if k[1] == model and k[2] == cell and k[3] == "rejected" and k[5] == attack]
        n = sum(v[1] for v in vals)
        return (sum(v[0] * v[1] for v in vals) / n, int(n)) if n else (None, 0)


def llm_rows(mode="C", lam=DEFAULT["lam"], chal="int", threads=DEFAULT["threads"], variant=DEFAULT["variant"]) -> dict:
    """``{(model, seq): {metric: (value, provenance, params)}}`` from ``llm_full_model.csv``."""
    out: dict = defaultdict(dict)
    for r in _read("llm_full_model.csv"):
        if r["metric"] == "commit_total":
            continue
        if (r["mode"], r["challenges"]) != (mode, chal) or _f(r["lam"]) != lam:
            continue
        if (r.get("rate", ""), r.get("threads", ""), r.get("variant", "")) != (DEFAULT["rate"], threads, variant):
            continue
        key = (r["model"], int(float(r["seq"])))
        if r["metric"] in out[key]:
            raise SystemExit(f"two headline rows for {key} {r['metric']}")
        out[key][r["metric"]] = (float(r["value"]), r["provenance"], _f(r["params_full"]))
    return out


def _llm_bytes(d: dict, part: str = "total"):
    """Proof bytes, with the Merkle term as a multiproof (native or expected; see aggregate.py)."""
    key = f"bytes_{part}_multiproof"
    return d[key][0] if key in d else d.get(f"bytes_{part}", (None,))[0]


def _llm_cost(d: dict):
    """(prove seconds, verify seconds, proof bytes, parameters) of one LLM row."""
    prove = sum(d[k][0] for k in PROVE if k in d)
    verify = sum(d[k][0] for k in VERIFY if k in d)
    return prove, verify, _llm_bytes(d), d["prove_forward"][2]
