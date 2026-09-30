"""Summarise one zkLLM run of zkllm.sbatch (report Sec. 4.4): per-stage times (script wall incl. model
load; binary alone, from the run's bin_times.log, written by the shims of install_timing_shims.sh).
Usage: summarise.py <run dir> <layers per model>."""
import csv, gzip, re, sys, statistics as st
from pathlib import Path
R = Path(sys.argv[1]); L = int(sys.argv[2])
wall = {}
for f in R.glob("time-r*-L*-*.txt"):
    rep, layer, stage = re.match(r"time-r(\d+)-L(\d+)-(.+)\.txt", f.name).groups()
    m = re.search(r"Elapsed.*: (?:(\d+):)?(\d+):([\d.]+)", f.read_text())
    h, mi, s = m.groups(); wall.setdefault(stage, []).append((int(h or 0)*3600 + int(mi)*60 + float(s)))
print(f"{R.name}: layers per model {L}")
log = R / "bin_times.log"
if log.exists():
    binary = {}
    for line in log.read_text().splitlines():
        m = re.match(r"BIN (\S+) ([\d.]+) s\s+maxrss (\d+) KB\s+exit (\d+)\s+args (\S+)", line)
        if not m: continue
        name, t, rss, rc, a0 = m.groups()
        key = {"self-attn": f"attn-{a0}", "rmsnorm": f"rmsnorm-{a0}"}.get(name, name)
        binary.setdefault(key, []).append((float(t), int(rc)))
    print("binary stage           n   median s   (min-max)     failed")
    per_layer = 0.0
    for k in sorted(binary):
        ts = [t for t, rc in binary[k]]; bad = sum(rc != 0 for _, rc in binary[k])
        print(f"  {k:20s} {len(ts):3d}   {st.median(ts):8.2f}   ({min(ts):.2f}-{max(ts):.2f})   {bad}")
        if k not in ("ppgen", "commit-param"): per_layer += st.median(ts)
    print(f"per-layer proving, binaries that ran (median): {per_layer:.2f} s")
    # A layer has two skip connections, but only the second one runs: zkLLM's attention stage never
    # writes its output, so the first skip has no input. Count the measured skip a second time
    # (only when there is one skip run per attention run, as in the stored runs).
    runs = binary.get("skip-connection", [])
    skip = st.median(t for t, _ in runs) if runs and len(runs) == len(binary.get("attn-attn", [])) else 0.0
    full = per_layer + skip
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
# without an index column; this job's GPU is the one whose memory use changes most often.
ngpu = sum(l.startswith("NVIDIA") for l in (R / "env.txt").read_text().splitlines())
rows = [r for r in csv.reader(gzip.open(R / "nvidia-smi.csv.gz", "rt")) if len(r) >= 3][1:]
mem = [[int(r[1].split()[0]) for r in rows[i::ngpu]] for i in range(ngpu)]
job = max(range(ngpu), key=lambda i: len(set(mem[i])))
print("peak GPU memory per GPU of the node (MiB):", [max(m) for m in mem])
print(f"this job's GPU: index {job}, peak {max(mem[job])} MiB = {max(mem[job]) / 1024:.1f} GiB")
