"""Summarise runs of zkLLM's public demo (``run_zkllm.sh prove``): the per-stage times (the script's wall time, model
load included; the proof binary alone, from the run's ``bin_times.log``, which the shims of
``install_timing_shims.sh`` write), the time per layer and for the whole model, and the run's environment.

Usage (standard library only):
    summarise.py <run dir> [<layers per model>]     # print the run's figures
    summarise.py --csv [<folder>]                   # write <folder>/summary.csv, one row per run folder in it,
                                                    # with its record's ok (default: the stored runs,
                                                    # artifacts/results/zkllm_l40s/)
    summarise.py --record <run dir>                 # print the run's record (one JSON line, kind "zkllm");
                                                    # exit status 1 unless it is ok

A run folder is named ``<model>-T<seq>`` (``llama2-7b-T2048``; a trailing ``-<number>`` is accepted) and holds
``bin_times.log``, ``env.txt``, ``verdicts.txt`` and ``time-*.txt``, as ``run_zkllm.sh prove`` writes them (one
folder per prompt length; ``errors.txt``: the first error line of each script that printed one, at a length where
zkLLM's demo failed)."""
import csv
import gzip
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORED = HERE.parents[2] / "artifacts" / "results" / "zkllm_l40s"
"""The stored runs (``code/artifacts/results/zkllm_l40s/``)."""
LAYERS = {"llama2-7b": 32, "llama2-13b": 40}
"""The decoder layers of each model of zkLLM's demo."""
RUN_DIR = re.compile(r"(?P<model>llama2-\d+b)-T(?P<seq>\d+)(?:-\d+)?$")
VERDICTS = {"qkv": "QKV linear proof successfully verified!", "attn": "Self attention proof successfully verified!",
            "ffn": "SwiGLU proof complete."}
"""The verifier messages that each proved layer prints once (``verdicts.txt`` counts them)."""


def binary_times(run_dir: Path) -> dict:
    """{stage: [(seconds, exit code), ...]} of the proof binaries, from the run's bin_times.log."""
    binary = {}
    for line in (Path(run_dir) / "bin_times.log").read_text().splitlines():
        m = re.match(r"BIN (\S+) ([\d.]+) s\s+maxrss (\d+) KB\s+exit (\d+)\s+args (\S+)", line)
        if not m:
            continue
        name, t, rss, rc, a0 = m.groups()
        key = {"self-attn": f"attn-{a0}", "rmsnorm": f"rmsnorm-{a0}"}.get(name, name)
        binary.setdefault(key, []).append((float(t), int(rc)))
    return binary


def per_layer_s(run_dir: Path) -> float:
    """Proving time of one layer: the median times of the proof binaries that ran (setup, i.e. ppgen and
    commit-param, excluded), with the measured skip connection counted twice.  A layer has two skip
    connections, but only the second one runs: zkLLM's attention stage never writes its output, so the
    first skip has no input (counted twice only when there is one skip run per attention run, as in
    the stored runs)."""
    binary = binary_times(run_dir)
    ran = sum(st.median(t for t, _ in v) for k, v in binary.items() if k not in ("ppgen", "commit-param"))
    runs = binary.get("skip-connection", [])
    skip = st.median(t for t, _ in runs) if runs and len(runs) == len(binary.get("attn-attn", [])) else 0.0
    return ran + skip


def run_name(run_dir: Path) -> tuple[str, int]:
    """``(model, seq)`` of a run folder's name."""
    m = RUN_DIR.match(Path(run_dir).name)
    if not m:
        raise ValueError(f"{run_dir}: not a run folder <model>-T<seq>")
    return m["model"], int(m["seq"])


def environment(run_dir: Path) -> dict:
    """The run's ``env.txt``: the host, the GPU (``nvidia-smi``'s name, driver and memory; None if it names several
    models), the CPU model, zkLLM's commit and the ``key: value`` lines that follow (build and library versions)."""
    lines = (Path(run_dir) / "env.txt").read_text(encoding="utf-8").splitlines()
    gpus = [[x.strip() for x in s.split(",")] for s in lines if s.startswith("NVIDIA")]
    cpus = {s.split(":", 1)[1].strip() for s in lines if s.startswith("Model name:")}
    out = {"host": lines[0].strip() if lines else None,
           "gpu": gpus[0][0] if gpus and len({g[0] for g in gpus}) == 1 else None,
           "driver": gpus[0][1] if gpus and len(gpus[0]) > 1 else None,
           "gpu_memory": gpus[0][2] if gpus and len(gpus[0]) > 2 else None,
           "cpu": next(iter(cpus)) if len(cpus) == 1 else None,
           "zkllm_commit": next((s.strip() for s in lines if re.fullmatch(r"[0-9a-f]{40}", s.strip())), None)}
    for s in lines:
        m = re.match(r"([a-z_]+): (.*)$", s)
        if m and m[1] not in out:
            out[m[1]] = m[2].strip()
    return out


def verdicts(run_dir: Path) -> dict:
    """``{qkv, attn, ffn}``: how often each verifier message of :data:`VERDICTS` appears in ``verdicts.txt``."""
    path = Path(run_dir) / "verdicts.txt"
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        m = re.match(r"\s*(\d+) (.*)$", line)
        if m:
            counts[m[2].strip()] = int(m[1])
    return {k: counts.get(text, 0) for k, text in VERDICTS.items()}


def errors(run_dir: Path) -> list[str]:
    """The run's ``errors.txt`` (``run_zkllm.sh prove``: the first error line of each script that printed one, counted
    with ``uniq -c``), one entry per line; empty when there is none."""
    path = Path(run_dir) / "errors.txt"
    return [" ".join(x.split()) for x in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if x.strip()] if path.exists() else []


def record(run_dir: Path) -> dict:
    """The run's record: the model, prompt length, repetitions and layers proved, the time per layer and for the whole
    model, the verifier messages, the proof binaries that failed, zkLLM's error lines (:func:`errors`) and the
    environment (``env``: GPU, driver, CPU, zkLLM's commit).  ``ok``: every proof binary exited 0 and each of the
    ``reps`` x ``layers`` proved layers (``run.json``) printed each message of :data:`VERDICTS` once.  ``required``
    (``run.json``; true unless ``run_zkllm.sh prove --allow-failed-lengths``): whether a record that is not ``ok`` is
    a failed check (``paper/records.py``) or a length at which zkLLM's demo does not run, which the paper reports as
    such."""
    run_dir = Path(run_dir)
    model, seq = run_name(run_dir)
    binary = binary_times(run_dir)
    proved = len(binary.get("ffn", []))
    per = per_layer_s(run_dir)
    failed = sum(rc != 0 for k, v in binary.items() for _, rc in v)
    meta = {}
    if (run_dir / "run.json").exists():
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    expected = meta["reps"] * len(meta["layers"]) if meta else proved
    said = verdicts(run_dir)
    ok = bool(expected) and proved == expected and failed == 0 and all(n == expected for n in said.values())
    return {"kind": "zkllm", "model": model, "seq": seq, "layers": LAYERS[model], "reps": meta.get("reps"),
            "layer_ids": meta.get("layers"), "layers_expected": expected, "layers_proved": proved,
            "per_layer_s": round(per, 4), "whole_model_s": round(per * LAYERS[model], 2), "verdicts": said,
            "binary_failures": failed, "errors": errors(run_dir), "ok": ok,
            "required": meta.get("required", True) is not False, "run": run_dir.name, "env": environment(run_dir)}


def write_csv(folder: Path) -> list[dict]:
    """``<folder>/summary.csv``: one row per run folder, model, seq, layers, per_layer_s, whole_model_s and ok (the
    run's :func:`record`; the times of a run that is not ``ok`` cover the binaries that ran, not a proof)."""
    rows = []
    for run in sorted(p for p in folder.iterdir() if p.is_dir() and (p / "bin_times.log").exists()):
        model, seq = run_name(run)
        per = per_layer_s(run)
        rows.append({"model": model, "seq": seq, "layers": LAYERS[model], "run": run.name,
                     "per_layer_s": f"{per:.4f}", "whole_model_s": f"{per * LAYERS[model]:.2f}",
                     "ok": str(record(run)["ok"]).lower()})
    if not rows:
        raise SystemExit(f"{folder}: no run folder with a bin_times.log")
    with open(folder / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def report(R: Path, L: int) -> None:
    wall = {}
    for f in R.glob("time-r*-L*-*.txt"):
        rep, layer, stage = re.match(r"time-r(\d+)-L(\d+)-(.+)\.txt", f.name).groups()
        m = re.search(r"Elapsed.*: (?:(\d+):)?(\d+):([\d.]+)", f.read_text())
        h, mi, s = m.groups()
        wall.setdefault(stage, []).append((int(h or 0) * 3600 + int(mi) * 60 + float(s)))
    print(f"{R.name}: layers per model {L}")
    log = R / "bin_times.log"
    if log.exists():
        binary = binary_times(R)
        print("binary stage           n   median s   (min-max)     failed")
        per_layer = 0.0
        for k in sorted(binary):
            ts = [t for t, rc in binary[k]]
            bad = sum(rc != 0 for _, rc in binary[k])
            print(f"  {k:20s} {len(ts):3d}   {st.median(ts):8.2f}   ({min(ts):.2f}-{max(ts):.2f})   {bad}")
            if k not in ("ppgen", "commit-param"):
                per_layer += st.median(ts)
        print(f"per-layer proving, binaries that ran (median): {per_layer:.2f} s")
        full = per_layer_s(R)
        print(f"per-layer proving, second skip counted twice (the README's figure): {full:.2f} s"
              f"  -> x{L} layers = {full*L:.0f} s")
    else:
        print(f"binary times: not stored (no {log.name} in this run dir)")
    print("script wall (incl. Python + model load), median per stage:",
          {k: round(st.median(v), 1) for k, v in sorted(wall.items())})
    per_layer_wall = sum(st.median(v) for v in wall.values())
    print(f"per-layer script wall (median, the failing first-skip script included): {per_layer_wall:.2f} s"
          f"  -> x{L} layers = {per_layer_wall*L:.1f} s")
    # nvidia-smi.csv.gz has one row per GPU that the job's nvidia-smi queried per 0.5 s sample, in index order and
    # without an index column; with several (the stored runs recorded every GPU of a shared node) this job's GPU is
    # taken to be the one whose memory use changes most often (a heuristic: the node ran other jobs)
    ngpu = sum(s.startswith("NVIDIA") for s in (R / "env.txt").read_text().splitlines())
    rows = [r for r in csv.reader(gzip.open(R / "nvidia-smi.csv.gz", "rt")) if len(r) >= 3][1:]
    mem = [[int(r[1].split()[0]) for r in rows[i::ngpu]] for i in range(ngpu)]
    job = max(range(ngpu), key=lambda i: len(set(mem[i])))
    print("peak GPU memory per GPU queried (MiB):", [max(m) for m in mem])
    print(f"this job's GPU (heuristic): index {job}, peak {max(mem[job])} MiB = {max(mem[job]) / 1024:.1f} GiB")


def main(argv: list[str]) -> int:
    if argv[:1] == ["--csv"]:
        for r in write_csv(Path(argv[1]) if len(argv) > 1 else STORED):
            print(r)
    elif argv[:1] == ["--record"] and len(argv) == 2:
        rec = record(Path(argv[1]))
        print(json.dumps(rec), flush=True)
        return 0 if rec["ok"] else 1
    elif len(argv) in (1, 2) and not argv[0].startswith("-"):
        run = Path(argv[0])
        report(run, int(argv[1]) if len(argv) == 2 else LAYERS[run_name(run)[0]])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
