"""Summarise a zkLLM D.1 run: per-stage times (script wall incl. model load; binary alone)."""
import re, sys, statistics as st
from pathlib import Path
R = Path(sys.argv[1]); L = int(sys.argv[2])
wall = {}
for f in R.glob("time-r*-L*-*.txt"):
    rep, layer, stage = re.match(r"time-r(\d+)-L(\d+)-(.+)\.txt", f.name).groups()
    m = re.search(r"Elapsed.*: (?:(\d+):)?(\d+):([\d.]+)", f.read_text())
    h, mi, s = m.groups(); wall.setdefault(stage, []).append((int(h or 0)*3600 + int(mi)*60 + float(s)))
binary = {}
for line in (R / "bin_times.log").read_text().splitlines():
    m = re.match(r"BIN (\S+) ([\d.]+) s\s+maxrss (\d+) KB\s+exit (\d+)\s+args (\S+)", line)
    if not m: continue
    name, t, rss, rc, a0 = m.groups()
    key = {"self-attn": f"attn-{a0}", "rmsnorm": f"rmsnorm-{a0}"}.get(name, name)
    binary.setdefault(key, []).append((float(t), int(rc)))
print(f"{R.name}: layers per model {L}")
print("binary stage           n   median s   (min-max)     failed")
per_layer = 0.0
for k in sorted(binary):
    ts = [t for t, rc in binary[k]]; bad = sum(rc != 0 for _, rc in binary[k])
    print(f"  {k:20s} {len(ts):3d}   {st.median(ts):8.2f}   ({min(ts):.2f}-{max(ts):.2f})   {bad}")
    if k not in ("ppgen", "commit-param"): per_layer += st.median(ts)
print(f"per-layer proving, binaries only (median): {per_layer:.2f} s  -> x{L} layers = {per_layer*L:.1f} s")
print("script wall (incl. Python + model load), median per stage:",
      {k: round(st.median(v), 1) for k, v in sorted(wall.items())})
