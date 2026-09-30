#!/bin/bash
# The timing wrapper used for D.1 (ours, not zkLLM's). Run inside a built zkLLM checkout
# (commit 993311e). Each zkLLM binary X is renamed X.real and replaced by a script that runs it
# under /usr/bin/time and appends one line to $ZK_TIMELOG (default ./bin_times.log):
#     BIN <name> <wall s> s  maxrss <KB> KB  exit <code>  args <arguments>
# zkLLM's Python scripts call ./X, so they are timed unchanged. `touch` keeps make from rebuilding.
set -euo pipefail
cd "${1:-.}"
for b in ppgen commit-param self-attn ffn rmsnorm skip-connection; do
  [ -f "$b.real" ] || mv "$b" "$b.real"
  cat > "$b" <<SHIM
#!/bin/bash
# timing shim (ours, not zkLLM's): logs the real binary's wall time and peak RSS
d=\$(dirname "\$0")
/usr/bin/time -f "BIN $b %e s  maxrss %M KB  exit %x  args \$*" -a -o "\${ZK_TIMELOG:-\$d/bin_times.log}" "\$d/$b.real" "\$@"
SHIM
  chmod +x "$b"; touch "$b"
done
