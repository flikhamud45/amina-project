"""The benchmark of the report (Sec. 4.3-4.4): our defence and Anchuri et al.'s path
test, on the models the literature benchmarks.  Every measurement is appended, one JSON
object per line, to ``artifacts/comparison/raw/<suite>/<model>/<cell>.jsonl``;
nothing is aggregated here, so figures can be re-made later without re-running.

    python experiments/4_defence_benchmark/bench.py cnn --model vgg16
    python experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 64

A cell that has a ``.done`` marker is skipped (resume after pre-emption); pass
``--force`` to redo it.
"""

from __future__ import annotations

import argparse
import copy
import json
import zlib
import math
import os
import platform
import socket
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pvi.fullcheck.field import P
from pvi.fullcheck.graph import CheapOp, MatOp
from pvi.fullcheck.protocol import (
    Challenger,
    Prover,
    Verifier,
    Z_BOUND,
    commit_graph,
    params_for,
    run_query,
    soundness_bits,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "comparison" / "raw"
MODELS = ROOT / "artifacts" / "fullcheck" / "models"
LAMBDAS = (40, 80, 128)
TAG = ""  # appended to every cell name (``--tag``), so variant runs never collide


# --------------------------------------------------------------------------- env
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.fileMode=false", *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def env_info() -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except Exception:
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "host": socket.gethostname(), "cpu": cpu, "torch_threads": torch.get_num_threads(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "gpu": gpu,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }


class Recorder:
    """Append-only per-cell record file with an atomic ``.done`` marker."""

    def __init__(self, suite: str, model: str, cell: str, config: dict, env: dict) -> None:
        self.dir = RAW / suite / model
        self.dir.mkdir(parents=True, exist_ok=True)
        cell = cell + TAG
        self.path = self.dir / f"{cell}.jsonl"
        self.done_path = self.dir / f"{cell}.done"
        on_gpu = str(env.get("device", "")).startswith("cuda")
        prover_hw = env.get("gpu") if on_gpu else env.get("cpu")
        self.base = {"run_id": uuid.uuid4().hex[:12], "suite": suite, "model": model, "cell": cell,
                     "config": dict(config, variant=TAG, device=env.get("device"), merkle="multiproof"),
                     "prover_hw": prover_hw, "verifier_hw": f"{env.get('cpu')} x{env.get('torch_threads')} threads",
                     "hw": prover_hw, "host": env.get("host"), "git_sha": env.get("git_sha")}
        self.fh = open(self.path.with_suffix(".jsonl.part"), "w")
        self.rec("env", 0, "", **{"env": env})

    def rec(self, metric: str, value, unit: str = "", trial: int | None = None, **extra) -> None:
        row = dict(self.base, metric=metric, value=value, unit=unit, trial=trial, ts=time.time(), **extra)
        self.fh.write(json.dumps(row, default=float) + "\n")
        self.fh.flush()

    def done(self) -> None:
        self.fh.close()
        os.replace(self.path.with_suffix(".jsonl.part"), self.path)
        self.done_path.write_text(json.dumps({"ts": time.time()}))
        for p in (self.path, self.done_path):
            try:
                os.chmod(p, 0o664)
            except OSError:
                pass


ONLY: set[str] = set()


def is_done(suite, model, cell) -> bool:
    return (RAW / suite / model / f"{cell}{TAG}.done").exists()


def _todo(args, suite, model, cell) -> bool:
    """Run this cell?  Selected by ``--only`` (if given), and forced or not yet done."""
    if ONLY and not any(cell.startswith(o) for o in ONLY):
        return False
    return args.force or not is_done(suite, model, cell)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _timeit(fn, device, n=30, warmup=5) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out.append(time.perf_counter() - t0)
    return out


def _claims_zlib(claims: dict) -> int:
    """Size of the claimed pre-activations after zlib (level 6), as int32 -- the
    comparison point for Anchuri et al.'s Brotli-compressed proof sizes."""
    raw = b"".join(claims[k].to(torch.int32).numpy().tobytes() for k in sorted(claims))
    return len(zlib.compress(raw, 6))


def _record_query(r: Recorder, res: dict, trial: int, **extra) -> None:
    if "claims" in res:
        r.rec("bytes_claims_zlib", _claims_zlib(res.pop("claims")), "B", trial, **extra)
    for k, v in res["timings"].items():
        r.rec(k, v, "s", trial, **extra)
    for k, v in res["bytes"].items():
        r.rec("bytes_" + k, v, "B", trial, **extra)
    r.rec("bytes_total", sum(res["bytes"].values()), "B", trial, **extra)
    r.rec("accepted", int(res["accepted"]), "", trial, **extra)


def _shapes(graph, commitments) -> list[tuple[int, int]]:
    return [(c.row_length, c.n_points) for c in commitments.values()]


# ------------------------------------------------------------------ CNN models
def _mlp_model() -> tuple[nn.Module, dict]:
    from pvi.data import load_classification
    from pvi.training import load_network
    from pvi.zoo import mlp_architecture_for

    net = load_network(mlp_architecture_for(10), ROOT / "artifacts" / "models" / "mlp_mnist_full.npz")
    layers = [l for l in net.architecture.layers if l.name in net.parameters]
    mods: list[nn.Module] = []
    for i, layer in enumerate(layers):
        w, b = net.parameters[layer.name]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = torch.from_numpy(np.asarray(w, dtype=np.float32))
        lin.bias.data = torch.from_numpy(np.asarray(b, dtype=np.float32))
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ReLU())
    ds = load_classification("mnist")
    data = {"train_x": torch.from_numpy(ds.train_x[:512]).reshape(-1, 784),
            "test_x": torch.from_numpy(ds.test_x).reshape(-1, 784), "test_y": torch.from_numpy(ds.test_y),
            "shape": (784,), "dataset": "mnist (pvi, [0,1] inputs)"}
    return nn.Sequential(*mods).eval(), data


def _cnn_model(name: str) -> tuple[nn.Module, dict]:
    from pvi.fullcheck.datasets import load_dataset, normalise
    from pvi.fullcheck.models import MODEL_SPECS, build_float_model

    kwargs, dataset, shape = MODEL_SPECS[name]
    tx, ty, vx, vy, family, n_classes = load_dataset(dataset)
    model = build_float_model(kwargs["kind"], n_classes)
    model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
    crop = (lambda x: x[:, :, 16:240, 16:240]) if family == "imagenet" else (lambda x: x)
    g = torch.Generator().manual_seed(0)
    cal = tx[torch.randperm(len(tx), generator=g)[: (64 if family == "imagenet" else 512)]]
    data = {"train_x": normalise(crop(cal), family), "test_x_uint8": vx, "test_y": vy,
            "norm": lambda x: normalise(crop(x), family), "shape": shape, "dataset": dataset}
    return model.eval(), data


def _test_batches(data, batch):
    if "test_x" in data:
        x = data["test_x"]
        for s in range(0, len(x), batch):
            yield x[s:s + batch], data["test_y"][s:s + batch]
    else:
        x = data["test_x_uint8"]
        for s in range(0, len(x), batch):
            yield data["norm"](x[s:s + batch]), data["test_y"][s:s + batch]


def _penultimate_mat(graph) -> MatOp:
    """The weight op whose output (through cheap ops only) feeds the final layer."""
    prod = {op.output: op for op in graph.ops}
    name = graph.mat_ops[-1].inputs[0]
    while not isinstance(prod[name], MatOp):
        name = prod[name].inputs[0]
    return prod[name]


def suite_cnn(args, env) -> None:
    from pvi.fullcheck.quantize import dequantize_logits, quantize_input, quantize_model
    from pvi.fullcheck.sampling import (TraceCommitment, _base as _base_name, neuron_tensors, paths_for,
                                        sample_path, shared_path_bytes, visit_probabilities)

    device = torch.device(args.device)
    name = args.model
    model, data = _mlp_model() if name == "mlp_mnist" else _cnn_model(name)
    graph = quantize_model(model, data["train_x"], model_name=name)
    mats = graph.mat_ops
    n_params = graph.n_params()
    model_f = model.float().to(device)
    xs, ys = next(_test_batches(data, args.queries + 64))
    q_inputs = quantize_input(graph, xs)          # int8 queries, one per trial

    # ---- model facts and fidelity ------------------------------------------------
    cell = "facts"
    if _todo(args, "cnn", name, cell):
        r = Recorder("cnn", name, cell, {}, env)
        r.rec("n_params", n_params, "")
        r.rec("n_weight_ops", len(mats), "")
        r.rec("n_cheap_ops", sum(1 for op in graph.ops if not isinstance(op, MatOp)), "")
        r.rec("model_bytes_int8", sum(op.weight.numel() + 4 * (op.bias.numel() if op.bias is not None else 0)
                                      for op in mats), "B")
        r.rec("model_bytes_fp32", 4 * n_params, "B")
        _, claims1 = graph.forward(q_inputs[:1].to(device))
        r.rec("claim_values_per_query", sum(z.numel() for z in claims1.values()), "")
        correct_f = correct_i = agree = total = 0
        with torch.no_grad():
            for xb, yb in _test_batches(data, 128 if "test_x" in data else 32):
                lf = model_f(xb.to(device)).argmax(1).cpu()
                _, cl = graph.forward(quantize_input(graph, xb).to(device))
                li = dequantize_logits(graph, cl[mats[-1].name]).argmax(1).cpu()
                correct_f += int((lf == yb).sum())
                correct_i += int((li == yb).sum())
                agree += int((lf == li).sum())
                total += len(yb)
        r.rec("float_accuracy", correct_f / total, "", n=total)
        r.rec("int8_accuracy", correct_i / total, "", n=total)
        r.rec("float_int8_agreement", agree / total, "", n=total)
        x1 = xs[:1].to(device)
        with torch.no_grad():
            for t_ in _timeit(lambda: model_f(x1), device):
                r.rec("float_inference", t_, "s")
            # the "download the model and re-run it" reference: the verifier's CPU
            model_cpu = copy.deepcopy(model_f).cpu()
            x1c = xs[:1].float()
            for t_ in _timeit(lambda: model_cpu(x1c), "cpu", n=10, warmup=2):
                r.rec("cpu_float_inference", t_, "s")
            qi = q_inputs[:1].to(device)
            for t_ in _timeit(lambda: graph.forward(qi), device, n=10, warmup=2):
                r.rec("int8_inference", t_, "s")
        r.done()

    # ---- the single-neuron attack on the float model --------------------------------
    cell = "attack_float"
    if _todo(args, "cnn", name, cell):
        r = Recorder("cnn", name, cell, {"n": len(xs)}, env)
        seq = isinstance(model_f, nn.Sequential)
        head = model_f[-1] if seq else model_f.head
        feat_fn = (lambda b_: model_f[:-1](b_)) if seq else model_f.features
        with torch.no_grad():
            feats = feat_fn(xs.to(device)).double()
            fcal = torch.cat([feat_fn(data["train_x"][s0:s0 + 64].float().to(device)).double()
                              for s0 in range(0, len(data["train_x"]), 64)])
            unit_max = torch.maximum(fcal.max(0).values, feats.max(0).values)   # per unit, clean data
            live = unit_max > 0
            layer_max = float(unit_max.max())
            w, b = head.weight.double(), head.bias.double()
            logits = feats @ w.T + b
            pred = logits.argmax(1)

            def best_flip(i, allowed):
                best = None
                l_ = logits[i]
                for c in range(w.shape[0]):
                    if c == int(pred[i]):
                        continue
                    others = torch.cat([w[:c], w[c + 1:]])       # rows k != c
                    lo = torch.cat([l_[:c], l_[c + 1:]])
                    ok_j = (w[c][None, :] > others).all(0) & allowed  # c wins for a large enough delta
                    if not bool(ok_j.any()):
                        continue
                    gap = (lo - l_[c])[:, None] / (w[c][None, :] - others)
                    need = gap.clamp_min(0).amax(0) * 1.0001 + 1e-9
                    need = torch.where(ok_j, need, torch.full_like(need, math.inf))
                    j = int(need.argmin())
                    if best is None or float(need[j]) < best[0]:
                        best = (float(need[j]), c, j)
                return best

            everyone = torch.ones_like(live)
            for i in range(len(xs)):
                for tag, allowed in (("", everyone), ("_live", live)):
                    best = best_flip(i, allowed)
                    if best is None:
                        r.rec("flip_success" + tag, 0, "", i)
                        continue
                    delta, c, j = best
                    new = logits[i] + delta * w[:, j]
                    r.rec("flip_success" + tag, int(int(new.argmax()) == c), "", i)
                    r.rec("flip_delta" + tag, delta, "", i)
                    r.rec("flip_unit_dead" + tag, int(not bool(live[j])), "", i)
                    r.rec("flip_delta_over_layer_max" + tag, delta / layer_max, "", i)
                    if bool(live[j]):
                        r.rec("flip_delta_over_unit_max" + tag, delta / float(unit_max[j]), "", i)
        r.rec("penultimate_width", int(w.shape[1]), "")
        r.rec("penultimate_live_units", int(live.sum()), "")
        r.done()

    # ---- the sampling baseline (Anchuri et al.) on the integer graph -----------------------
    cell = "sampling"
    if _todo(args, "cnn", name, cell):
        n_paths = args.paths
        r = Recorder("cnn", name, cell, {"n_paths": n_paths}, env)
        env1, _ = graph.forward(q_inputs[:1])
        tc = TraceCommitment(graph, env1)
        r.rec("trace_commit", tc.commit_seconds, "s")
        r.rec("weight_commit", tc.weight_commit_seconds, "s")
        r.rec("trace_bytes", tc.trace_bytes, "B")
        r.rec("model_bytes", tc.model_bytes, "B")
        r.rec("open_all_bytes", tc.open_all_bytes, "B")
        rng = np.random.default_rng(0)
        per_path = []
        for i in range(n_paths):
            res = sample_path(tc, env1, rng)
            per_path.append(res["bytes"])
            r.rec("path_ok", int(res["ok"]), "", i)
            r.rec("path_bytes", res["bytes"], "B", i)
            r.rec("path_open", res["open_s"], "s", i)
            r.rec("path_verify", res["verify_s"], "s", i)
            r.rec("path_steps", res["steps"], "", i)
        mass = visit_probabilities(graph, env1)
        prod = {op.output: op for op in graph.ops}
        src = _base_name(prod, mats[-1].inputs[0])
        p_attack = float(mass[src].min())
        p_min = min(float(mass[n].min()) for n in neuron_tensors(graph))
        r.rec("p_detect_penultimate", p_attack, "")
        r.rec("p_detect_min_node", p_min, "")
        lams = (10, 20, 40, 64, 80, 128)
        ks = {lam: paths_for(lam, p_attack) for lam in lams}
        grid = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
        shared = shared_path_bytes(tc, env1, sorted(set(ks.values()) | set(grid)), max_paths=args.max_shared_paths)
        for k_ in grid:   # the cost-vs-security curve of the sampling protocol
            if shared.get(k_) is not None:
                r.rec("paths_bytes_shared_k", shared[k_], "B", k=k_,
                      bits=-k_ * math.log2(1.0 - p_attack))
        median_path = statistics.median(per_path)
        for lam in lams:
            r.rec("paths_needed_penultimate", ks[lam], "", lam=lam)
            r.rec("paths_needed_min_node", paths_for(lam, p_min), "", lam=lam)
            r.rec("paths_bytes_naive", ks[lam] * median_path, "B", lam=lam)
            if shared.get(ks[lam]) is not None:
                r.rec("paths_bytes_shared", shared[ks[lam]], "B", lam=lam)
        r.done()

    # ---- our defence: commitment, honest queries, all modes and security levels ----------
    if ONLY and not any(o.startswith(("commit", "defence", "tamper")) for o in ONLY):
        return
    # The commitment is one-time and takes seconds for these models, so it is
    # always rebuilt (the Merkle roots are deterministic); it is recorded once.
    rate = args.rate
    _sync(device)
    t0 = time.perf_counter()
    coms = commit_graph(graph, rate, device=device)
    _sync(device)
    commit_s = time.perf_counter() - t0
    cell = f"commit_rate{rate}"
    if _todo(args, "cnn", name, cell):
        r = Recorder("cnn", name, cell, {"rate": rate}, env)
        r.rec("commit_total", commit_s, "s")
        for k, c in coms.items():
            r.rec("commit_op", c.commit_seconds, "s", op=k, n_rows=c.weight.shape[0],
                  row_length=c.row_length, n_points=c.n_points)
        r.rec("commitment_bytes", 32 * len(coms), "B")
        r.done()

    prover = Prover(graph, device=device, commitments=coms)
    weights = {op.name: (op.weight, op.bias) for op in mats}
    for lam in LAMBDAS:
        for mode, chal in (("C", "int"), ("C", "fs"), ("K", "int"), ("Kpre", "int")):
            cell = f"defence_{mode}_{chal}_lam{lam}_rate{rate}"
            if not _todo(args, "cnn", name, cell):
                continue
            params = params_for(lam, len(mats), rate=rate, fiat_shamir=(chal == "fs"))
            cfg = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": params.rate,
                   "columns": params.columns, "threads": torch.get_num_threads()}
            r = Recorder("cnn", name, cell, cfg, env)
            shapes = [(op.row_length, rate * (1 << max(0, (op.row_length - 1).bit_length()))) for op in mats]
            r.rec("soundness_bits", soundness_bits(params, shapes, "C" if mode == "C" else "K"), "bits")
            if mode == "C":
                v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
            else:
                v = Verifier(graph.public(), params, mode, weights=weights)
                if mode == "Kpre":
                    r.rec("verifier_precompute", v.precompute(Challenger()), "s")
            run_query(prover, v, q_inputs[:1])  # untimed warm-up (lazy CUDA/BLAS initialisation)
            for i in range(args.queries):
                res = run_query(prover, v, q_inputs[i:i + 1], keep_claims=(i < 3))
                _record_query(r, res, i)
            # batch amortisation: B queries in one interaction (mode C, interactive only)
            if mode == "C" and chal == "int":
                for bsz in (8, 32):
                    if bsz > len(q_inputs):
                        continue
                    res = run_query(prover, v, q_inputs[:bsz])
                    _record_query(r, res, 0, batch=bsz)
            r.done()

    # ---- soundness experiments: every attack must be rejected ----------------------------
    cell = "tamper_C_int_lam40"
    if _todo(args, "cnn", name, cell):
        params = params_for(40, len(mats), rate=rate)
        v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
        r = Recorder("cnn", name, cell, {"lam": 40, "mode": "C", "reps": params.reps, "columns": params.columns}, env)
        g = torch.Generator().manual_seed(123)
        pen = _penultimate_mat(graph)
        x0 = q_inputs[:1]

        def attempt(kind, i, **kw):
            res = run_query(prover, v, x0, **kw)
            r.rec("rejected", int(not res["accepted"]), "", i, attack=kind, stage=res["rejected_at"])

        for i in range(args.tampers):
            op = mats[int(torch.randint(len(mats), (1,), generator=g))]
            pos = int(torch.randint(10 ** 9, (1,), generator=g))
            delta = int(torch.randint(1, 1000, (1,), generator=g)) * (1 if i % 2 else -1)

            def t_single(o, z, op=op, pos=pos, delta=delta):
                if o.name == op.name:
                    z = z.clone()
                    z.view(-1)[pos % z.numel()] += delta
                return z

            attempt("single_value", i, forward_kwargs={"tamper": t_single})
            j = int(torch.randint(pen.n_rows, (1,), generator=g))

            def t_pen(o, z, j=j):
                if o.name == pen.name:
                    z = z.clone()
                    z[j] = Z_BOUND // 4
                return z

            attempt("penultimate_neuron", i, forward_kwargs={"tamper": t_pen})

            def t_logit(o, z):
                if o.name == mats[-1].name:
                    z = z.clone()
                    k = int(z[:, 0].argmin())
                    z[k, 0] = int(z[:, 0].max()) + 1000
                return z

            attempt("output_logit", i, forward_kwargs={"tamper": t_logit})
        # model substitution: 1% of every weight matrix nudged by +-1
        for i in range(max(3, args.tampers // 10)):
            other = {}
            for op in mats:
                w = op.weight.clone().to(torch.int16)
                mask = torch.rand(w.shape, generator=g) < 0.01
                w = (w + mask.to(torch.int16) * (2 * torch.randint(0, 2, w.shape, generator=g, dtype=torch.int16) - 1))
                other[op.name] = MatOp(op.name, op.inputs, op.output, weight=w.clamp(-127, 127).to(torch.int8),
                                       bias=op.bias, layout=op.layout, conv=op.conv)
            attempt("substituted_model_1pct", i, forward_kwargs={"weights_override": other})
        if name == "resnet18_224" and (MODELS / "resnet18_224_squirrel.pt").exists():
            m2, d2 = _cnn_model("resnet18_224_squirrel")
            g2 = quantize_model(m2, d2["train_x"])
            other = {a.name: b_ for a, b_ in zip(mats, g2.mat_ops)}
            for i in range(max(3, args.tampers // 10)):
                attempt("substituted_model_M_tilde", i, forward_kwargs={"weights_override": other})
        # Attacks aimed at the two parts of mode C that Freivalds alone cannot cover.
        # (1) forged fold: a wrong logit plus a u' solved so that Freivalds passes;
        #     only the Reed-Solomon column check can reject it.
        # (2) forged column: an honest u plus an opened column shifted by a vector in
        #     the kernel of chi, so the code check passes; only the Merkle path can reject it.
        real_fold, real_open = prover.fold, prover.open
        final = mats[-1]
        state: dict = {}

        def t_forge(o, z):
            if o.name == final.name:
                z = z.clone()
                k = int(z[:, 0].argmin())
                new = int(z[:, 0].max()) + 1000
                state["k"], state["delta"] = k, new - int(z[k, 0])
                z[k, 0] = new
            return z

        def forged_fold(chis):
            us = real_fold(chis)
            u = us[final.name].clone()
            kb = final.n_in                      # the bias coordinate multiplies the constant 1
            for i_ in range(u.shape[0]):
                u[i_, kb] = (int(u[i_, kb]) + int(chis[final.name][i_, state["k"]]) * state["delta"]) % P
            us[final.name] = u
            return us

        def kernel_vector(chi):
            """A nonzero delta with chi @ delta = 0 mod P (Gaussian elimination)."""
            r_, n_ = chi.shape
            if n_ <= r_:
                return None
            m = [[int(v) % P for v in chi[i_, :r_].tolist()] + [(-int(chi[i_, r_])) % P] for i_ in range(r_)]
            for col in range(r_):
                piv = next((i_ for i_ in range(col, r_) if m[i_][col]), None)
                if piv is None:
                    return None
                m[col], m[piv] = m[piv], m[col]
                inv = pow(m[col][col], P - 2, P)
                m[col] = [(v_ * inv) % P for v_ in m[col]]
                for i_ in range(r_):
                    if i_ != col and m[i_][col]:
                        f_ = m[i_][col]
                        m[i_] = [(vi - f_ * vc) % P for vi, vc in zip(m[i_], m[col])]
            delta = torch.zeros(n_, dtype=torch.int64)
            delta[:r_] = torch.tensor([m[i_][r_] for i_ in range(r_)], dtype=torch.int64)
            delta[r_] = 1
            return delta

        for i in range(max(3, args.tampers // 10)):
            prover.fold = forged_fold
            attempt("forged_fold", i, forward_kwargs={"tamper": t_forge})
            prover.fold = real_fold
            victim = mats[i % len(mats)].name

            def capture_fold(chis):
                state["chis"] = chis
                return real_fold(chis)

            def forged_open(cols, victim=victim):
                out = real_open(cols)
                c, pth = out[victim]
                delta = kernel_vector(state["chis"][victim])
                if delta is not None:
                    c = c.clone()
                    c[:, 0] = (c[:, 0] + delta) % P
                out[victim] = (c, pth)
                return out

            prover.fold, prover.open = capture_fold, forged_open
            attempt("forged_column", i)
            prover.fold, prover.open = real_fold, real_open
        r.done()


# ------------------------------------------------------------------ LLM shapes
def suite_llm(args, env) -> None:
    from pvi.fullcheck.analytic import decoder_shapes, soundness_error_bits
    from pvi.fullcheck.transformer import CONFIGS, build_decoder, decoder_param_count

    device = torch.device(args.device)
    cfg = CONFIGS[args.model]
    full = cfg.n_layers <= 12 and not args.extrapolate
    builds = [cfg.n_layers] if full else [1, 2]
    modes = [("C", "int"), ("C", "fs"), ("Kpre", "int"), ("K", "int")] if not args.modes else \
        [tuple(m.split(":")) for m in args.modes.split(",")]
    for seq in args.seq:
        for n_layers in builds:
            base = f"T{seq}_L{n_layers}" + ("" if args.rate == 4 else f"_rate{args.rate}")
            todo = [f"defence_{m}_{c}_lam{l}_{base}" for m, c in modes for l in args.lams]
            if not args.force and all(is_done("llm", cfg.name, t) for t in todo):
                continue
            t0 = time.perf_counter()
            graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, 32), seed=0)
            build_s = time.perf_counter() - t0
            mats = graph.mat_ops
            tokens = torch.randint(0, cfg.vocab, (args.queries, seq), generator=torch.Generator().manual_seed(1))
            coms = None
            if any(m == "C" for m, _ in modes):
                torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
                t0 = time.perf_counter()
                coms = commit_graph(graph, args.rate, device=device)
                commit_s = time.perf_counter() - t0
                cell = f"commit_{base}"
                if _todo(args, "llm", cfg.name, cell):
                    r = Recorder("llm", cfg.name, cell, {"seq": seq, "n_layers": n_layers, "rate": args.rate,
                                                         "n_layers_full": cfg.n_layers,
                                                         "params_full": decoder_param_count(cfg)}, env)
                    r.rec("commit_total", commit_s, "s")
                    r.rec("build_graph", build_s, "s")
                    for k, c in coms.items():
                        r.rec("commit_op", c.commit_seconds, "s", op=k, n_rows=c.weight.shape[0],
                              row_length=c.row_length, n_points=c.n_points)
                    r.done()
            prover = Prover(graph, device=device, commitments=coms)
            weights = {op.name: (op.weight, op.bias) for op in mats}
            for lam in args.lams:
                for mode, chal in modes:
                    cell = f"defence_{mode}_{chal}_lam{lam}_{base}"
                    if not _todo(args, "llm", cfg.name, cell):
                        continue
                    # size (r, t) for the FULL model's op count, so extrapolated rows keep their lambda
                    full_shapes = decoder_shapes(cfg, seq)
                    params = params_for(lam, len(full_shapes), rate=args.rate, fiat_shamir=(chal == "fs"))
                    conf = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": args.rate,
                            "n_checks_full": len(full_shapes),
                            "columns": params.columns, "seq": seq, "n_layers": n_layers,
                            "n_layers_full": cfg.n_layers, "params_full": decoder_param_count(cfg),
                            "params_built": graph.n_params(), "threads": torch.get_num_threads()}
                    r = Recorder("llm", cfg.name, cell, conf, env)
                    shapes = [(op.row_length, args.rate * (1 << max(0, (op.row_length - 1).bit_length())))
                              for op in mats]
                    r.rec("soundness_bits_built", soundness_bits(params, shapes, "C" if mode == "C" else "K"), "bits")
                    r.rec("soundness_bits_full", soundness_error_bits(full_shapes, params, "C" if mode == "C" else "K"),
                          "bits")
                    if mode == "C":
                        v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
                    else:
                        v = Verifier(graph.public(), params, mode, weights=weights)
                        if mode == "Kpre":
                            r.rec("verifier_precompute", v.precompute(Challenger()), "s")
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    run_query(prover, v, tokens[:1])  # untimed warm-up
                    for i in range(args.queries):
                        res = run_query(prover, v, tokens[i:i + 1], keep_claims=(i == 0 and seq <= 512))
                        _record_query(r, res, i)
                    if device.type == "cuda":
                        r.rec("gpu_peak_memory", torch.cuda.max_memory_allocated(), "B")
                    # one tampered query per cell: must be rejected
                    victim = mats[len(mats) // 2].name

                    def tamper(o, z, victim=victim):
                        if o.name == victim:
                            z = z.clone()
                            z.view(-1)[0] += 1
                        return z

                    res = run_query(prover, v, tokens[:1], forward_kwargs={"tamper": tamper})
                    r.rec("tamper_rejected", int(not res["accepted"]), "", stage=res["rejected_at"])
                    r.done()
            del prover, coms, graph
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["cnn", "llm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--tampers", type=int, default=100)
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--seq", type=int, nargs="+", default=[64])
    ap.add_argument("--lams", type=int, nargs="+", default=list(LAMBDAS))
    ap.add_argument("--modes", default="")
    ap.add_argument("--extrapolate", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rate", type=int, default=4, help="Reed-Solomon rate (codeword / message length)")
    ap.add_argument("--max-shared-paths", type=int, default=100_000,
                    help="cap on paths simulated for the shared-opening sampling cost")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for cell names, e.g. _thr1")
    ap.add_argument("--only", default="", help="comma-separated cell prefixes to (re)run, e.g. sampling")
    args = ap.parse_args()
    global TAG, ONLY
    TAG = args.tag
    ONLY = {c for c in args.only.split(",") if c}
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = env_info()
    env["device"] = args.device
    print(json.dumps(env), flush=True)
    t0 = time.time()
    (suite_cnn if args.suite == "cnn" else suite_llm)(args, env)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
