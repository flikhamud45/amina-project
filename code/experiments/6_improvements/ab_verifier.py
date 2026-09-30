"""Back-to-back A/B timing of the verifier: a baseline checkout (the paper code) vs this one.

    PYTHONPATH=src python experiments/6_improvements/ab_verifier.py --base <checkout>/code/src \\
        --out ab.csv --threads 1,4 lenet5:C:128 vgg16:C:128 gpt2:C:128:64 gpt2:Kpre:128:64

A case is ``model:mode:lam[:seq[:layers[:vocab]]]`` (``seq`` tokens for decoders; ``layers``
blocks, default all; ``vocab`` overrides the vocabulary, e.g. to measure the per-block slope
cheaply: the block shapes do not change).  The baseline ``pvi.fullcheck`` is imported under
another module name, so both run in this process on the same machine state.

For each case both packages build the same random-init model (same seeds; costs depend on
shapes, not on weight values).  The baseline prover makes one transcript (claims, ``chi``,
``u``, column indices, openings), and this checkout's prover must send the same messages.
Each tampered transcript must be rejected by both verifiers at the same check.  Then both
run derive / check_products / check_columns on the honest transcript ``--reps`` times,
alternating which goes first; every repetition must give identical derived tensors and
verdicts.  Transcripts are cached in ``--cache`` (the prover-side checks run when one is
made).  See README.md for the results.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import importlib
import importlib.util
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

import pvi.fullcheck as N

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("derive", "products", "columns")


def load_base(src: Path, name: str = "pvi_base_fullcheck"):
    """The baseline checkout's ``pvi.fullcheck``, imported as ``name`` (its imports are relative)."""
    pkg = src / "pvi" / "fullcheck"
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sub(pkg, name: str):
    return importlib.import_module(f"{pkg.__name__}.{name}")


@dataclasses.dataclass(frozen=True)
class Case:
    model: str
    mode: str
    lam: int
    seq: int = 64
    layers: int | None = None
    vocab: int | None = None

    @classmethod
    def parse(cls, spec: str) -> "Case":
        p = spec.split(":")
        opt = [None if v in ("", "-") else int(v) for v in p[3:]] + [None] * 3
        return cls(p[0], p[1], int(p[2]), opt[0] or 64, opt[1], opt[2])

    @property
    def key(self) -> str:
        return f"{self.model}_{self.mode}_lam{self.lam}_T{self.seq}_L{self.layers}_V{self.vocab}"


# -- models -------------------------------------------------------------------------------------

def _chunked_weight(self, rows, cols, std=40.0):
    """``_Builder.weight`` without a float32 copy of the whole matrix (Qwen's 389M-entry
    tables); installed in both packages, so their graphs stay identical."""
    out = torch.empty(rows, cols, dtype=torch.int8)
    step = max(1, (1 << 24) // cols)
    for r0 in range(0, rows, step):
        w = torch.randn(min(step, rows - r0), cols, generator=self.g) * std
        out[r0:r0 + w.shape[0]] = w.round_().clamp_(-127, 127).to(torch.int8)
    return out


def build(pkg, case: Case):
    """``(graph with weights, query, number of weight ops of the full model)``."""
    if case.model in ("lenet5", "vgg11", "vgg16", "resnet18_cifar"):
        torch.manual_seed(0)
        model = sub(pkg, "models").build_float_model(case.model, 10).eval()
        shape = (1, 28, 28) if case.model == "lenet5" else (3, 32, 32)
        g = torch.Generator().manual_seed(5)
        q = sub(pkg, "quantize")
        graph = q.quantize_model(model, torch.randn(64, *shape, generator=g))
        return graph, q.quantize_input(graph, torch.randn(1, *shape, generator=g)), len(graph.mat_ops)
    tr = sub(pkg, "transformer")
    cfg = tr.CONFIGS[case.model]
    n_ops = len(sub(pkg, "analytic").decoder_shapes(cfg))
    if case.vocab:
        cfg = dataclasses.replace(cfg, vocab=case.vocab)
    tr._Builder.weight = _chunked_weight
    graph = tr.build_decoder(cfg, n_layers=case.layers, calib_tokens=min(case.seq, 32), seed=0)
    tokens = torch.randint(0, cfg.vocab, (1, case.seq), generator=torch.Generator().manual_seed(1))
    return graph, tokens, n_ops


def fold_rows(pkg, op, chi, rows=256):
    """``chi [W | b] mod P`` (the verifier's Kpre precompute) in row blocks, without an int64
    copy of a whole weight matrix."""
    fld = sub(pkg, "field")
    w, u = op.weight, None
    step = max(1, min(rows, (1 << 23) // w.shape[1]))
    for r0 in range(0, w.shape[0], step):
        part = fld.small_matmul_mod(w[r0:r0 + step].T.to(torch.int64).contiguous(),
                                    chi[:, r0:r0 + step].T.contiguous()).T
        u = part if u is None else (u + part) % fld.P
    if op.bias is not None:
        u = torch.cat([u, fld.field_matmul_mod(chi, fld.to_field(op.bias)[:, None])], 1)
    return u


# -- transcript -----------------------------------------------------------------------------------

def _equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b) if torch.is_tensor(a) else a == b


def make_transcript(base, case: Case, cache: Path | None) -> dict:
    path = cache / f"{case.key}.pt" if cache else None
    if path is not None and path.exists():
        return torch.load(path, weights_only=False)
    graph, x, n_ops = build(base, case)
    params = base.params_for(case.lam, n_ops)
    mats = graph.mat_ops
    coms = base.commit_graph(graph, params.rate) if case.mode == "C" else {}
    prover = base.Prover(graph, commitments=coms)
    ch = base.Challenger(seed=7)
    tr = {"x": x, "claims": prover.claims(x), "n_ops": n_ops,
          "chis": {op.name: ch.folding(op.name, op.n_rows, params.reps) for op in mats}}
    if case.mode == "C":
        tr["us"] = prover.fold(tr["chis"])
        # (as this package's dataclass: a cached transcript must not depend on the baseline's name)
        tr["publics"] = {k: sub(N, "commitment").CommitmentPublic(*dataclasses.astuple(c.public))
                         for k, c in coms.items()}
        tr["cols"] = {op.name: ch.columns(op.name, tr["publics"][op.name].n_points, params.columns) for op in mats}
        tr["openings"] = prover.open(tr["cols"])
    else:
        tr["us"] = {op.name: fold_rows(base, op, tr["chis"][op.name]) for op in mats}
    del graph, prover
    gc.collect()
    # this checkout's prover sends the same messages (commitment building is unchanged code:
    # its Merkle trees are shared, so fold/open run the new code on the same committed data)
    graph, x_new, _ = build(N, case)
    assert torch.equal(x, x_new)
    new_coms = {k: sub(N, "commitment").WeightCommitment(tag=c.tag, weight=c.weight, bias=c.bias, rate=c.rate,
                                                         tree=c.tree, n_points=c.n_points) for k, c in coms.items()}
    prover = N.Prover(graph, commitments=new_coms)
    assert _equal(prover.claims(x), tr["claims"]), "the claims differ"
    if case.mode == "C":
        assert _equal(prover.fold(tr["chis"]), tr["us"]), "u differs"
        assert _equal(prover.open(tr["cols"]), tr["openings"]), "the openings differ"
    else:
        us = {op.name: fold_rows(N, op, tr["chis"][op.name]) for op in graph.mat_ops}
        assert _equal(us, tr["us"]), "u differs"
    del graph, prover, coms, new_coms
    gc.collect()
    if path is not None:
        cache.mkdir(parents=True, exist_ok=True)
        torch.save(tr, path)
    return tr


def verifiers(base, case: Case, tr: dict):
    out = []
    for pkg in (base, N):
        graph = build(pkg, case)[0].public()
        gc.collect()
        params = pkg.params_for(case.lam, tr["n_ops"])
        v = pkg.Verifier(graph, params, case.mode, publics=tr.get("publics", {}))
        if case.mode == "Kpre":
            v._pre.update({k: (tr["chis"][k], tr["us"][k]) for k in tr["chis"]})
        out.append(v)
    return out


# -- timing and verdicts ------------------------------------------------------------------------

def stages(v, tr: dict, mode: str):
    t, res = {}, {}
    t0 = time.perf_counter()
    res["inputs"] = v.derive(tr["x"], tr["claims"])
    t["derive"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res["products"] = v.check_products(tr["claims"], res["inputs"], tr["chis"], tr["us"])
    t["products"] = time.perf_counter() - t0
    if mode == "C":
        t0 = time.perf_counter()
        res["columns"] = v.check_columns(tr["chis"], tr["us"], tr["cols"], tr["openings"])
        t["columns"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return t, res


def verdict(v, tr: dict, mode: str) -> str:
    """The rejection label of ``run_query`` (the same order of checks)."""
    inputs = v.derive(tr["x"], tr["claims"])
    if inputs is None:
        return "range_or_shape"
    if not v.check_products(tr["claims"], inputs, tr["chis"], tr["us"]):
        return "freivalds"
    if mode == "C":
        return v.check_columns(tr["chis"], tr["us"], tr["cols"], tr["openings"]) or "accepted"
    return "accepted"


def tampered(tr: dict, mode: str):
    P, Z = N.P, sub(N, "protocol").Z_BOUND
    names = list(tr["chis"])
    first, mid, last = names[0], names[len(names) // 2], names[-1]

    def with_(key, name, val):
        return dict(tr, **{key: dict(tr[key], **{name: val})})

    def claim(name, index, value):
        z = tr["claims"][name].clone()
        z.view(-1)[index] = value(int(z.view(-1)[index]))
        return with_("claims", name, z)

    out = [(f"claim+1@{n}", claim(n, tr["claims"][n].numel() // 2, lambda v: v + 1)) for n in (first, mid, last)]
    out += [("claim=Z_BOUND", claim(mid, 0, lambda v: Z)), ("claim=INT64_MIN", claim(mid, 0, lambda v: -(1 << 63))),
            ("claim=1-Z_BOUND", claim(mid, -1, lambda v: 1 - Z)),
            ("claim int32", with_("claims", mid, tr["claims"][mid].to(torch.int32)))]
    u = tr["us"][mid].clone()
    u[0, -1] = (u[0, -1] + 1) % P
    out.append(("u+1", with_("us", mid, u)))
    u = tr["us"][mid].clone()
    u[-1, -1] = P
    out.append(("u=P", with_("us", mid, u)))
    if mode == "C":
        o, proof = tr["openings"][mid]
        o1 = o.clone()
        o1[0, 0] = (o1[0, 0] + 1) % P
        o2 = o.clone()
        o2[0, 0] = -1
        out += [("opened+1", with_("openings", mid, (o1, proof))), ("opened=-1", with_("openings", mid, (o2, proof))),
                ("opened narrow", with_("openings", mid, (o[:, :-1].clone(), proof)))]
        if proof:
            out += [("path[0]=0", with_("openings", mid, (o, [bytes(32)] + list(proof)[1:]))),
                    ("path short", with_("openings", mid, (o, list(proof)[:-1]))),
                    ("path long", with_("openings", mid, (o, list(proof) + [bytes(32)])))]
    return out


def run_case(base, case: Case, reps: int, threads: list[int], cache: Path | None, writer) -> None:
    t0 = time.perf_counter()
    tr = make_transcript(base, case, cache)
    vb, vn = verifiers(base, case, tr)
    labels = [(label, verdict(vb, t, case.mode), verdict(vn, t, case.mode)) for label, t in tampered(tr, case.mode)]
    bad = [row for row in labels if row[1] != row[2] or row[1] == "accepted"]
    assert not bad, f"tampered transcripts: {bad}"
    print(f"{case.key}: {sum(z.numel() for z in tr['claims'].values())} claims, setup {time.perf_counter() - t0:.0f} s, "
          f"tampered -> {', '.join(f'{a}: {b}' for a, b, _ in labels)}")
    for n_threads in threads:
        torch.set_num_threads(n_threads)
        for _ in range(2):   # warm-up: lru caches, the allocator, the Kpre stacks
            stages(vb, tr, case.mode)
            stages(vn, tr, case.mode)
        times = {"base": [], "new": []}
        for i in range(reps):
            results = {}
            for tag, v in ((("base", vb), ("new", vn)) if i % 2 == 0 else (("new", vn), ("base", vb))):
                gc.collect()
                t, results[tag] = stages(v, tr, case.mode)
                times[tag].append(t)
            rb, rn = results["base"], results["new"]
            assert _equal(rb["inputs"], rn["inputs"]), "derive differs"
            assert rb["products"] is rn["products"] is True and rb.get("columns") == rn.get("columns") is None
        for stage in STAGES + ("total",):
            if stage not in times["base"][0]:
                continue
            b = [t[stage] for t in times["base"]]
            n = [t[stage] for t in times["new"]]
            mb, mn = statistics.median(b), statistics.median(n)
            writer.writerow([case.model, case.mode, case.lam, case.seq, case.layers, case.vocab, n_threads, stage,
                             reps, f"{mb:.6f}", f"{mn:.6f}", f"{min(b):.6f}", f"{min(n):.6f}", f"{mb / mn:.3f}",
                             len(labels)])
            print(f"  threads={n_threads} {stage:9s} base {mb * 1e3:10.3f} ms   new {mn * 1e3:10.3f} ms   "
                  f"x{mb / mn:5.2f}   (min {min(b) * 1e3:.3f} / {min(n) * 1e3:.3f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--base", type=Path, required=True, help="the baseline checkout's code/src")
    ap.add_argument("--out", type=Path, required=True, help="CSV to append to")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--threads", type=lambda v: [int(t) for t in v.split(",")], default=[1],
                    help="comma-separated thread counts, e.g. 1,4")
    ap.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "fullcheck" / "cache" / "ab_verifier")
    a = ap.parse_args()
    base = load_base(a.base.resolve())
    assert Path(base.__file__).resolve() != Path(N.__file__).resolve(), "--base is this checkout"
    print(f"base {Path(base.__file__).parent}\nnew  {Path(N.__file__).parent}\n"
          f"torch {torch.__version__}  {platform.processor()}")
    new_file = not a.out.exists()
    with a.out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["model", "mode", "lam", "seq", "layers", "vocab", "threads", "stage", "reps",
                        "base_median_s", "new_median_s", "base_min_s", "new_min_s", "ratio_median", "tampered"])
        for spec in a.cases:
            run_case(base, Case.parse(spec), a.reps, a.threads, a.cache, w)
            fh.flush()
            gc.collect()


if __name__ == "__main__":
    main()
