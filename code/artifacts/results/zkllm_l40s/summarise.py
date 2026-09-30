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
print(f"per-layer proving, binaries that ran (median): {per_layer:.2f} s")
# A layer has two skip connections, but only the second one runs: zkLLM's attention stage never
# writes its output, so the first skip has no input. Count the measured skip a second time.
skip = st.median([t for t, _ in binary.get("skip-connection", [])]) if "skip-connection" in binary else 0.0
full = per_layer + skip
print(f"per-layer proving incl. the first skip (the published figure): {full:.2f} s"
      f"  -> x{L} layers = {full*L:.0f} s")
print("script wall (incl. Python + model load), median per stage:",
      {k: round(st.median(v), 1) for k, v in sorted(wall.items())})
