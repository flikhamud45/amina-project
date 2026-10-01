"""Summarise one zkLLM run of zkllm.sbatch (report Sec. 5.5): per-stage times (script wall incl. model
load; binary alone, from the run's bin_times.log, written by the shims of install_timing_shims.sh).
Usage: summarise.py <run dir> <layers per model>      # print the run's figures
       summarise.py --csv                             # write summary.csv for every stored run (paper_assets.py)"""
import csv, gzip, re, sys, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = {"llama2-7b": 32, "llama2-13b": 40}   # decoder layers of the stored runs' models


def binary_times(run_dir: Path) -> dict:
    """{stage: [(seconds, exit code), ...]} of the proof binaries, from the run's bin_times.log."""
    binary = {}
    for line in (Path(run_dir) / "bin_times.log").read_text().splitlines():
        m = re.match(r"BIN (\S+) ([\d.]+) s\s+maxrss (\d+) KB\s+exit (\d+)\s+args (\S+)", line)
        if not m: continue
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


def write_csv() -> None:
    """summary.csv: one row per stored run, model, seq, layers, per_layer_s, whole_model_s."""
    rows = []
    for run in sorted(p for p in HERE.iterdir() if p.is_dir() and (p / "bin_times.log").exists()):
        model, seq, _job = re.match(r"(.+)-T(\d+)-(\d+)$", run.name).groups()
        per = per_layer_s(run)
        rows.append({"model": model, "seq": seq, "layers": LAYERS[model], "run": run.name,
                     "per_layer_s": f"{per:.4f}", "whole_model_s": f"{per * LAYERS[model]:.2f}"})
    with open(HERE / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


def report(R: Path, L: int) -> None:
    wall = {}
    for f in R.glob("time-r*-L*-*.txt"):
        rep, layer, stage = re.match(r"time-r(\d+)-L(\d+)-(.+)\.txt", f.name).groups()
        m = re.search(r"Elapsed.*: (?:(\d+):)?(\d+):([\d.]+)", f.read_text())
        h, mi, s = m.groups(); wall.setdefault(stage, []).append((int(h or 0)*3600 + int(mi)*60 + float(s)))
    print(f"{R.name}: layers per model {L}")
    log = R / "bin_times.log"
    if log.exists():
        binary = binary_times(R)
        print("binary stage           n   median s   (min-max)     failed")
        per_layer = 0.0
        for k in sorted(binary):
            ts = [t for t, rc in binary[k]]; bad = sum(rc != 0 for _, rc in binary[k])
            print(f"  {k:20s} {len(ts):3d}   {st.median(ts):8.2f}   ({min(ts):.2f}-{max(ts):.2f})   {bad}")
            if k not in ("ppgen", "commit-param"): per_layer += st.median(ts)
        print(f"per-layer proving, binaries that ran (median): {per_layer:.2f} s")
        full = per_layer_s(R)
        print(f"per-layer proving, second skip counted twice (the report's figure): {full:.2f} s"
              f"  -> x{L} layers = {full*L:.0f} s")
    else:
        print(f"binary times: not stored (no {log.name} in this run dir)")
    print("script wall (incl. Python + model load), median per stage:",
          {k: round(st.median(v), 1) for k, v in sorted(wall.items())})
    per_layer_wall = sum(st.median(v) for v in wall.values())
    print(f"per-layer script wall (median, the failing first-skip script included): {per_layer_wall:.2f} s"
          f"  -> x{L} layers = {per_layer_wall*L:.1f} s")
    # nvidia-smi.csv.gz has one row per GPU of the (shared) node per 0.5 s sample, in index order and
    # without an index column; this job's GPU is taken to be the one whose memory use changes most often
    # (a heuristic: the node ran other jobs)
    ngpu = sum(l.startswith("NVIDIA") for l in (R / "env.txt").read_text().splitlines())
    rows = [r for r in csv.reader(gzip.open(R / "nvidia-smi.csv.gz", "rt")) if len(r) >= 3][1:]
    mem = [[int(r[1].split()[0]) for r in rows[i::ngpu]] for i in range(ngpu)]
    job = max(range(ngpu), key=lambda i: len(set(mem[i])))
    print("peak GPU memory per GPU of the node (MiB):", [max(m) for m in mem])
    print(f"this job's GPU (heuristic): index {job}, peak {max(mem[job])} MiB = {max(mem[job]) / 1024:.1f} GiB")


if __name__ == "__main__":
    if sys.argv[1:] == ["--csv"]:
        write_csv()
    else:
        report(Path(sys.argv[1]), int(sys.argv[2]))
