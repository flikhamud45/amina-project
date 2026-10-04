"""A/B timing of one group of attention heads: a baseline checkout (the paper code) against this
one's int32 path and its int64 fallback, in the same process, on ``--device``.

    cd code && export PYTHONPATH=$PWD/src
    python experiments/6_improvements/attention_ab.py --base ../../base/code/src --heads 4 --seq 2048 \\
        --head-dim 128 --threads 1 --reps 7 --out $SCRATCH/attention_ab.json

The heads are random int8 values with every third key equal to its query (peaked softmax rows as
well as flat ones), at the real models' temperature (``--m-s``, about 2.6M for Llama-2-7B).
Every repetition must give identical integers in all three; medians, minima and ratios go to
``--out`` (JSON).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from pvi.fullcheck import transformer as tr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_verifier import load_base, sub  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="the baseline checkout's code/src")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--m-s", type=int, default=2_600_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    base_tr = sub(load_base(a.base.resolve()), "transformer")
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randint(-127, 128, (1, a.heads, a.seq, a.head_dim), generator=g).to(a.device) for _ in range(3))
    k[:, :, ::3] = q[:, :, ::3]
    notmask = tr._causal_notmask(a.seq, 1, str(q.device))
    assert tr._int32_scores(a.m_s, a.head_dim, a.seq), "these sizes take the int64 fallback"
    runs = {"base": lambda: base_tr._attention_heads(q, k, v, a.m_s, None),
            "int64": lambda: tr._attention_int64(q, k, v, a.m_s, notmask),
            "int32": lambda: tr._attention_heads(q, k, v, a.m_s)}
    want = runs["base"]()
    times = {name: [] for name in runs}
    for i in range(a.reps + 1):                 # the first round warms up
        for name in list(runs) if i % 2 == 0 else list(runs)[::-1]:
            gc.collect()
            if q.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            got = runs[name]()
            if q.is_cuda:
                torch.cuda.synchronize()
            if i:
                times[name].append(time.perf_counter() - t0)
            assert torch.equal(got, want), name
            del got
    rec = {"heads": a.heads, "seq": a.seq, "head_dim": a.head_dim, "m_s": a.m_s, "device": a.device,
           "threads": a.threads, "reps": a.reps, "torch": torch.__version__,
           **{f"{n}_median_s": statistics.median(t) for n, t in times.items()},
           **{f"{n}_min_s": min(t) for n, t in times.items()}}
    for n in ("int64", "int32"):
        rec[f"ratio_{n}"] = rec["base_median_s"] / rec[f"{n}_median_s"]
    print(json.dumps(rec))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
