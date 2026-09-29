"""Summarise one zkLLM run of zkllm.sbatch (report Sec. 4.4): per-stage times (script wall incl. model
load; binary alone, from bin_times.log). Usage: summarise.py <run dir> <layers per model>.
The stored runs have no bin_times.log, so for them only the script wall times are printed."""
import re, sys, statistics as st
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
    print(f"per-layer proving, binaries only (median): {per_layer:.2f} s  -> x{L} layers = {per_layer*L:.1f} s")
else:
    print(f"binary times: not stored (no {log.name} in this run dir)")
print("script wall (incl. Python + model load), median per stage:",
      {k: round(st.median(v), 1) for k, v in sorted(wall.items())})
per_layer_wall = sum(st.median(v) for v in wall.values())
print(f"per-layer script wall (median): {per_layer_wall:.2f} s  -> x{L} layers = {per_layer_wall*L:.1f} s")
