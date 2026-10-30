"""The benchmark of protocol v0 and of the path test: the stored cells behind the cost tables and the outcome counts.

Measures protocol v0 (modes C, K and Kpre; interactive and Fiat--Shamir challenges) and the path test of Anchuri et
al. (re-implemented on the same integer graphs, :mod:`pvi.baseline.path_int`) on the trained image models and on
random-weight decoders with the shapes of the paper's language models.  Its records are the benchmark cells of
``tab:eval-llm`` (prover, CPU and GPU verifier, proof of modes C and Kpre), the ``ours`` rows of
``tab:eval-compare``, the soundness and completeness counts of Sec. ``sec:eval-setup`` (every honest query
accepted, every attack rejected, with the check that rejected it; ``comparison/count_outcomes.py``), the
single-neuron attack and the path test on the image models (Sec. ``sec:break``) and, with ``--gen``, the
generated-response cells of ``tab:gen``.  Nothing is aggregated here: ``comparison/aggregate.py`` makes the tables.

    python -m experiments.costs.bench cnn --model lenet5 --platform <name>
    python -m experiments.costs.bench llm --model llama2-7b --seq 64 --builds full --modes C:fs --lams 128 \\
        --queries 5 --llm-tampers 30 --platform <name>
    python -m experiments.costs.bench llm --model llama2-7b --seq 64 --builds full --modes Kpre:fs --lams 128 \\
        --queries 5 --platform <name>
    python -m experiments.costs.bench llm --model llama2-7b --real meta-llama/Llama-2-7b-hf --seq 64 2048 \\
        --builds full --modes C:fs,Kpre:fs --lams 128 --queries 5 --platform <name>
    python -m experiments.costs.bench llm --model tiny-gpt --seq 16 --queries 2 --lams 40 --platform laptop
        # a laptop check (minutes on a CPU)

The prover runs on ``--device`` (default the GPU if there is one) and the verifier on ``--verifier-device`` (default
the CPU; ``cuda``: the prover's GPU).  The protocol settings default to the paper's configuration
(:data:`experiments.common.cli.PAPER`): mode C under the commitment plan ``cnn18c``, the compact wire encoding, the
lean prover and verifier, the last decoder block pruned to the last position, in modes K and Kpre the verifier
reading the embedding rows itself, and decoders with 11-bit attention probabilities (``--prob-max 2047``).  Each also
has a ``--no-...`` form (the per-op commitment: ``--policy per-op``; the 8-bit decoders: ``--prob-max 255``).

**Records.**  Every measurement is appended, one JSON object per line, to
``artifacts/comparison/raw_<platform>/<suite>/<model>/<cell>.jsonl``: a first row with ``metric`` ``env`` (the run's
environment, :func:`env_info`, with the implementation options of :mod:`pvi.options` under ``implementation``),
then rows ``{run_id, suite, model, cell, config, prover_hw, verifier_hw, host, git_sha, metric, value, unit, trial,
ts}`` with, where they apply, ``batch``, ``attack``, ``stage``, ``lam``, ``k`` and ``contention`` (the scheduler
evidence of :mod:`pvi.core.contention` for each timed phase).  Two rows carry hex digests instead of numbers (the
aggregates skip them; ``comparison/fingerprint_check.py`` compares them between platforms): ``commit_roots_sha256``
in the commitment cells (:func:`experiments.common.records.roots_sha256`) and, per Fiat--Shamir query,
``proof_transcript`` (the transcript's digest, :func:`pvi.v0.query.run_query`).  A platform names one prover GPU and one CPU model:
the root's ``PLATFORM.json`` records them, and a job on other hardware, or in a frozen root, stops.  A finished cell
has a ``.done`` marker and is skipped, so an interrupted job resumes where it stopped.  An honest query that is
rejected stops the job and keeps the cell's records as ``<cell>.rejected-<run_id>.jsonl``.

**Cell names** are a function of the settings that change the cell: a base name (``facts``, ``attack_float``,
``sampling``, ``commit_rate<r>``, ``defence_<mode>_<int|fs>_lam<l>_rate<r>``, ``tamper_C_int_lam<l>``; decoders
``commit_T<seq>_L<blocks>``, ``defence_<mode>_<int|fs>_lam<l>_T<seq>_L<blocks>``, ``tamper_C_int_lam<l>_T<seq>_L<b>``,
with ``_rate<r>`` after the base of the commitment and mode-C cells at a rate other than 4), then the suffixes, in
this order, of the settings that differ from the reference:  ``_thr<n>`` (``--threads`` other than 8), ``_nolean``
(decoders, ``--no-lean``), ``_tf32``, ``_wire``, ``_gpuv`` (a verifier on the GPU), ``_stream`` (the streaming
verifier), ``_batch`` (batch sizes other than the defaults), ``_prune`` (decoders, the pruned last block),
``_lookups`` (the K and Kpre cells of decoders with ``--lookups``), ``_pmax<n>`` (decoders whose attention
probabilities' scale is not the stored cells' 255, :data:`STORED_PROB_MAX`: ``_pmax2047``, 11 bits), ``_gen<G>``,
``_pol<name>`` (the commitment and mode-C cells under a plan other than ``per-op``) and ``_real`` (decoders,
``--real``: a checkpoint's weights).  A cell measured under the same
settings therefore has the name of the stored cell measured under them, and the cells a setting does not change (e.g.
the K and Kpre cells under a commitment plan) are shared between runs.

**Real weights.**  The decoder cells take random weights of the model's shapes (``--seed``) and random prompts
(``--prompt-seed``) unless ``--real`` names a Hugging Face checkpoint (``--model`` names its shapes): its integer graph
is then built as ``perf.py --real`` builds it (:mod:`experiments.common.real_cells`: an OPT checkpoint with
``hf_opt.CERTIFIED_RECIPE``, a Llama-family checkpoint, Qwen3 included, with ``hf_llama.MAIN_RECIPE`` and its fixed
public prefix), loaded and calibrated once per job, and the prompts are the WikiText-2 test split's windows from its
start (``perf.py --real``'s queries: the ``i``-th prompt of ``T`` tokens is the ``i``-th window of ``T`` tokens).  A
build of ``n`` blocks is the checkpoint's first ``n``, the full graph's first ``n`` (OPT and Llama-family alike).  The
cells are named ``_real`` and their ``config`` holds ``real`` (the checkpoint) and ``recipe``
(``certified`` or ``main``); the commitment cell also records ``load_checkpoint``, the seconds of loading the
checkpoint in fp32 and its calibration, which ``build_graph`` (the integer graph's own calibration and build) does not
include.  Refused: the image models, GPT-2 (no integer builder of a real checkpoint), a prompt (with ``--gen``, the
prompt and the generated tokens, behind a Llama-family recipe's fixed prefix) longer than the checkpoint's trained
context, a checkpoint of other shapes than ``--model``'s, and an attention probabilities' scale other than the
paper's.  Without ``--real`` the cells and their
records are the random-weight ones.

Hardware and time: the paper's cells ran on an NVIDIA L40S prover with 8 threads of an AMD EPYC 9334 as the CPU
verifier.  A CNN or GPT-2 job takes minutes, a 7B or 13B decoder at 2,048 tokens one to two hours and up to 128 GB
of host memory; a build whose int8 weights exceed 85% of the GPU's memory is skipped (and the job then fails).  With
``--real`` the checkpoint is also held in fp32 until the job's last build (about 4 bytes per parameter: 27 GB for
Llama-2-7B and OPT-6.7B, 52 GB for Llama-2-13B, 16 GB for Qwen3-4B), and its calibration's forward passes on the CPU
add tens of minutes to a 7B or 13B job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pvi
from experiments.common import checkpoints, cli, real_cells, records
from experiments.common.models import decoder_config, test_batches, trained_cnn, trained_mlp
from experiments.common.paths import BENCH, CNN_MODELS, CODE
from pvi.commit.cost import setup_size
from pvi.commit.plans import plan_commitment
from pvi.core import contention
from pvi.core.field import P
from pvi.core.graph import MatOp
from pvi.v0.graph_commitment import commit_graph
from pvi.v0.params import Z_BOUND, params_for, soundness_bits
from pvi.v0.prover import Prover
from pvi.v0.query import run_query
from pvi.v0.transcript import Challenger
from pvi.v0.verifier import Verifier

RAW = BENCH / "raw"
"""The raw root of this run (set by :func:`main` from ``--platform``, see :func:`raw_root`)."""
LAMBDAS = (40, 80, 128)
THREADS = 8
"""The verifier threads of the reference cells (others are named ``_thr<n>``)."""
CNN_BATCHES = (8, 32)
"""The batch sizes of the image models' reference ``defence_C_int`` cells (others are named ``_batch``)."""
PATH_GRID = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)
PLAN_CELLS = ("commit_", "defence_C_", "tamper_C_")
"""The cells a commitment plan or rate changes (the others are the same under every plan)."""
FROZEN_ALIAS = "rtx2080ti"
"""The platform name of the root ``raw/`` (the first stored run, frozen)."""
STORED_PROB_MAX = 255
"""The attention probabilities' scale of the stored decoder cells (8 bits); a decoder cell of another scale is named
``_pmax<n>``."""


def _require_this_pvi() -> Path:
    """Refuse to run on another checkout's ``pvi`` (e.g. a shared environment with pvi installed from a different
    clone): the records would say this code ran when it did not."""
    here = (CODE / "src" / "pvi").resolve()
    got = Path(pvi.__file__).resolve().parent
    if got != here:
        raise SystemExit(f"pvi is imported from {got}, not from this checkout ({here}): set PYTHONPATH={here.parent}")
    return got


def raw_root(platform: str) -> Path:
    """``raw_<platform>/``: every prover/verifier pair writes its own root, so records of different hardware never
    share a cell (or a median); :data:`FROZEN_ALIAS` names ``raw/``."""
    return BENCH / ("raw" if platform == FROZEN_ALIAS else f"raw_{platform}")


def claim_platform(env: dict, platform: str) -> None:
    """Refuse to write into a root that holds another machine's records.  The first run in an empty root creates
    ``PLATFORM.json``; every run (the first included) must match its GPU and CPU.  The lock appears atomically and
    complete (a hard link to a finished temporary file), so jobs that start together cannot each claim the root for
    their own machine, and no job ever reads a half-written lock."""
    lock = RAW / "PLATFORM.json"
    want = {"gpu": env.get("gpu") if str(env.get("device", "")).startswith("cuda") else None, "cpu": env.get("cpu")}
    if not lock.exists():
        if any(RAW.glob("*/*/*.jsonl")):
            raise SystemExit(f"{RAW} has records but no PLATFORM.json; create it before adding records")
        RAW.mkdir(parents=True, exist_ok=True)
        tmp = RAW / f"PLATFORM.json.{uuid.uuid4().hex}"
        tmp.write_text(json.dumps(dict(want, platform=platform, host=env.get("host"), torch=env.get("torch"),
                                       cuda=env.get("cuda"), gpu_total_memory=env.get("gpu_total_memory"),
                                       driver=env.get("driver"), created=time.time()), indent=1) + chr(10))
        try:
            os.link(tmp, lock)          # fails if another job created the lock first
        except FileExistsError:
            pass                        # it did: compare with its lock below
        finally:
            tmp.unlink()
    have = json.loads(lock.read_text())
    if have.get("frozen"):
        raise SystemExit(f"{RAW} is frozen (stored records that must not change): pass another --platform")
    if {k: have.get(k) for k in want} != want:
        raise SystemExit(f"{RAW} belongs to {have.get('gpu')} + {have.get('cpu')}; this job runs on "
                         f"{want['gpu']} + {want['cpu']}: pass another --platform")


# --------------------------------------------------------------------------- the environment
def _env_extra() -> dict:
    """What a benchmark record keeps besides :func:`records.environment`: library versions, the TF32 and cuDNN
    settings, the scheduler's and the BLAS libraries' environment variables, the CPU topology and the GPU's
    details."""
    import torchvision

    out = {"raw_dir": str(RAW), "numpy": np.__version__, "torchvision": torchvision.__version__,
           "tf32_cudnn": torch.backends.cudnn.allow_tf32, "fp32_matmul_precision": torch.get_float32_matmul_precision(),
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
           "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "vars": {k: v for k, v in os.environ.items()          # recorded only; no code path reads them here
                    if k.startswith(("NVIDIA_TF32", "TORCH_ALLOW_TF32", "CUBLAS", "OMP_", "MKL_", "CUDA_VISIBLE",
                                     "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS"))}}
    try:  # Linux: physical cores in the affinity set ('x8 threads' can be 4 cores x 2 hyperthreads)
        topo = "/sys/devices/system/cpu/cpu{}/topology/{}"
        out["cpu_affinity_cores"] = len({tuple(Path(topo.format(c, k)).read_text().strip()
                                               for k in ("physical_package_id", "core_id"))
                                         for c in os.sched_getaffinity(0)})
    except (AttributeError, OSError):
        pass
    queries = ((["lscpu"], "lscpu"),
               (["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit,"
                 "mig.mode.current,compute_mode,pci.bus_id", "--format=csv,noheader"], "nvidia_smi"))
    for cmd, key in queries:
        try:
            txt = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if key == "lscpu":
            keep = ("Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "NUMA node(s)",
                    "CPU max MHz")
            txt = {a.strip(): b.strip() for a, _, b in (x.partition(":") for x in txt.splitlines())
                   if a.strip() in keep}
        out[key] = txt
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        out["gpu_props"] = {"total_memory": pr.total_memory, "capability": f"{pr.major}.{pr.minor}",
                            "sms": pr.multi_processor_count}
    return out


def env_info(device, platform: str = "") -> dict:
    """The ``env`` row of every cell: :func:`records.environment` (commit, host, CPU, GPU, versions, implementation
    options) with the prover's ``device``, the ``platform``, the full commit ``git_sha``, ``torch_threads`` and
    ``pvi_path`` (the names the stored roots use) and :func:`_env_extra`."""
    env = records.environment(device)
    env.update(device=str(device), platform=platform, torch_threads=torch.get_num_threads(),
               git_sha=records.git_sha(CODE, short=40), pvi_path=env["pvi"], **_env_extra())
    return env


# --------------------------------------------------------------------------- cell names and records
def _batch_variant(args, suite: str) -> bool:
    if suite == "llm":
        return bool(args.batches)
    return tuple(args.cnn_batches) != CNN_BATCHES or args.batch_trials != 1


def variant(args, suite: str, base: str) -> str:
    """The suffix of the cell ``base`` (module docstring): the settings of this run that change it."""
    llm, plan = suite == "llm", base.startswith(PLAN_CELLS)
    tags = [f"thr{args.threads}" if args.threads != THREADS else "",
            "nolean" if llm and not args.lean else "", "tf32" if args.tf32 else "", "wire" if args.wire else "",
            "gpuv" if args.verifier_device != "cpu" else "", "stream" if args.verifier_impl == "stream" else "",
            "batch" if _batch_variant(args, suite) else "", "prune" if llm and args.prune_last else "",
            "lookups" if llm and args.lookups and base.startswith(("defence_K_", "defence_Kpre_")) else "",
            f"pmax{args.prob_max}" if llm and args.prob_max != STORED_PROB_MAX else "",
            f"gen{args.gen}" if llm and args.gen > 1 else "",
            f"pol{args.policy}" if plan and args.policy != "per-op" else "",
            "real" if llm and getattr(args, "real", "") else ""]
    return "".join("_" + t for t in tags if t)


def is_done(args, suite: str, model: str, base: str) -> bool:
    return (RAW / suite / model / f"{base}{variant(args, suite, base)}.done").exists()


def _verifier(args, *a, **kw) -> Verifier:
    """A verifier on ``--verifier-device`` (``--verifier-impl stream``: the streaming verifier, which records one
    ``verify_total`` per query instead of the verify phases it overlaps)."""
    return Verifier(*a, device=args.verifier_device, stream=args.verifier_impl == "stream", **kw)


def _query(args, prover: Prover, v: Verifier, x: torch.Tensor, **kw) -> dict:
    """One interaction (``run_query``); with ``--wire`` in the compact encoding of the proof (``prove_encode`` and
    ``verify_decode`` recorded, and the encoded sizes as ``bytes_*``)."""
    return run_query(prover, v, x, wire=args.wire, **kw)


def _verifier_hw(env: dict, vdev: str) -> str:
    cpu = f"{env.get('cpu')} x{env.get('torch_threads')} threads"
    if vdev == "cpu":
        return cpu
    return f"{torch.cuda.get_device_name(torch.device(vdev))} (GPU client) + {cpu}"


def _lock(fh, part: Path) -> None:
    """An exclusive lock on a cell's ``.part`` file: a second job on the same cell stops at once instead of
    interleaving its rows with the first job's.  Where the filesystem has no ``flock`` (or on Windows), the job
    scheduler's ``--dependency=singleton`` is the only guard."""
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(f"{part} is being written by another job")
    except (ImportError, OSError):
        pass


class Recorder:
    """Append-only per-cell record file with an atomic ``.done`` marker.  A ``.part`` file left by an interrupted
    attempt is kept (renamed ``.part.<time>``), never truncated."""

    def __init__(self, args, suite: str, model: str, base: str, config: dict, env: dict) -> None:
        self.dir = RAW / suite / model
        self.dir.mkdir(parents=True, exist_ok=True)
        tag = variant(args, suite, base)
        cell = base + tag
        self.path = self.dir / f"{cell}.jsonl"
        self.done_path = self.dir / f"{cell}.done"
        on_gpu = str(env.get("device", "")).startswith("cuda")
        prover_hw = env.get("gpu") if on_gpu else env.get("cpu")
        self.base = {"run_id": uuid.uuid4().hex[:12], "suite": suite, "model": model, "cell": cell,
                     "config": dict(config, variant=tag, device=env.get("device"), merkle="multiproof",
                                    verifier_device=args.verifier_device,
                                    **({"platform": args.platform} if args.platform else {}),
                                    **({"verifier_impl": args.verifier_impl}
                                       if args.verifier_impl != "default" else {}),
                                    **({"wire": True} if args.wire else {})),
                     "prover_hw": prover_hw, "verifier_hw": _verifier_hw(env, args.verifier_device),
                     "host": env.get("host"), "git_sha": env.get("git_sha")}
        part = self.path.with_suffix(".jsonl.part")
        self.fh = open(part, "a")
        _lock(self.fh, part)
        if os.fstat(self.fh.fileno()).st_size:   # an earlier attempt's records: keep them aside
            self.fh.close()                       # (Windows cannot rename an open file)
            os.replace(part, part.with_name(f"{part.name}.{time.time():.6f}"))
            try:
                self.fh = open(part, "x")
            except FileExistsError:
                raise SystemExit(f"{part} is being written by another job")
            _lock(self.fh, part)
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


def _commit(args, graph, device, model_ops=None, cs: dict | None = None):
    """``(commitment, seconds)``: the weight commitment under ``--policy`` at ``--rate``, timed to its end on the
    GPU; its contention evidence goes to ``cs["commit_total"]`` (:func:`_cs`)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    c0 = contention.begin()
    t0 = time.perf_counter()
    coms = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=model_ops,
                        cache=args.setup_cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if cs is not None:
        contention.add(cs, "commit_total", c0)
    return coms, dt


def _cs(store: dict | None, phase: str) -> dict:
    """The ``contention`` field of a timing row: the evidence of whether its machine was quiet while ``phase`` ran
    (:mod:`pvi.core.contention`: run-queue wait and on-CPU time of this process's threads, the load average at
    start and end, context switches); nothing if none was taken.  The readers of the records key on metric, value
    and the group keys, and ignore it."""
    c = (store or {}).get(phase)
    return {"contention": c} if c is not None else {}


def _plan_config(coms, params=None) -> dict:
    """What a cell records of a commitment plan, and with ``params`` its trees' column counts (nothing for the
    per-op commitment)."""
    if coms is None or coms.plan is None:
        return {}
    return {"policy": coms.plan.policy, **({"policy_requested": coms.plan.requested} if coms.plan.requested else {}),
            "trees": len(coms.groups),
            **({"group_columns": dict(params.group_columns)} if params is not None else {})}


def _record_setup(r: Recorder, coms, ops) -> None:
    """A commitment plan's setup size (``cost.setup_size``: encoded entries, leaves, trees)."""
    for k, v in setup_size(ops, plan=coms.plan).items():
        r.rec("setup_" + k, v, "")


def _plan_soundness(params, plan, mode: str) -> float:
    return soundness_bits(params, plan.shapes(), mode, columns=plan.matrix_columns(params.group_columns))


def _opening_of(coms, name: str) -> tuple[str, int]:
    """Where op ``name``'s opened columns are: its own opening, or from row ``offset`` of its group's (those of its
    committed matrix: under a ``c`` policy maybe a col-layout one, of ``k`` rows)."""
    if coms.plan is None:
        return name, 0
    matrix = coms.plan.matrix_of(name).name
    g, members = next((g, ms) for g, ms in coms.plan.groups if matrix in ms)
    return g, sum(coms[m].public.n_rows for m in members[:members.index(matrix)])


def _record_query(r: Recorder, res: dict, trial: int, **extra) -> None:
    for k, v in res["timings"].items():
        r.rec(k, v, "s", trial, **extra, **_cs(res.get("contention"), k))
    for k, v in res["bytes"].items():
        r.rec("bytes_" + k, v, "B", trial, **extra)
    r.rec("bytes_total", sum(res["bytes"].values()), "B", trial, **extra)
    r.rec("accepted", int(res["accepted"]), "", trial, **extra)
    if res.get("transcript"):       # Fiat--Shamir: the proof's fingerprint (pvi.v0.query.run_query)
        r.rec("proof_transcript", res["transcript"], "sha", trial, **extra)
    if res.get("claims_staged"):    # claims the prover staged in host memory (--host-claims)
        r.rec("claims_staged", res["claims_staged"], "B", trial, **extra)
    if not res["accepted"]:  # an honest query must never fail: GPU/CPU disagreement, stop the job
        # keep the evidence where a re-run cannot overwrite it and count_outcomes.py finds it
        # (no .done: the aggregates never read it)
        r.fh.close()
        kept = r.path.with_name(f"{r.path.stem}.rejected-{r.base['run_id']}.jsonl")
        os.replace(r.path.with_suffix(".jsonl.part"), kept)
        raise SystemExit(f"HONEST QUERY REJECTED at {res['rejected_at']} in {r.path.name} trial {trial} "
                         f"(records kept in {kept.name})")


def _record_peaks(r: Recorder, device, **extra) -> None:
    """The peaks since :func:`records.reset_peaks` (the prover's GPU, which with ``--verifier-device cuda`` also
    holds the client, and the host's resident set), and the GPU allocator's device allocations."""
    p = records.peaks(device)
    if device.type == "cuda":
        r.rec("gpu_peak_memory", p["gpu_peak_memory"], "B", **extra)
        r.rec("gpu_peak_reserved", p["gpu_peak_reserved"], "B", **extra)
        r.rec("gpu_alloc_retries", p["gpu_alloc_retries"], "", **extra)
        r.rec("gpu_device_allocs", torch.cuda.memory_stats(device).get("num_device_alloc", -1), "", **extra)
    if p["host_peak_rss"] is not None:
        r.rec("host_peak_rss", p["host_peak_rss"], "B", **extra)


def _modes(args, default) -> list[tuple[str, str]]:
    """``--modes`` (``C:int,Kpre:fs,...``) or ``default``."""
    if not args.modes:
        return list(default)
    out = [tuple(m.split(":")) for m in args.modes.split(",")]
    if any(len(m) != 2 or m[0] not in ("C", "K", "Kpre") or m[1] not in ("int", "fs") for m in out):
        raise SystemExit(f"--modes: comma-separated <C|K|Kpre>:<int|fs>, not {args.modes!r}")
    return out


# ------------------------------------------------------------------ image models
def _penultimate_mat(graph) -> MatOp:
    """The weight op whose output (through cheap ops only) feeds the final layer."""
    prod = {op.output: op for op in graph.ops}
    name = graph.mat_ops[-1].inputs[0]
    while not isinstance(prod[name], MatOp):
        name = prod[name].inputs[0]
    return prod[name]


def suite_cnn(args, env) -> None:
    from pvi.baseline.path_int import (TraceCommitment, base_tensor as _base_name, neuron_tensors, paths_for,
                                       shared_path_bytes, visit_probabilities)
    from pvi.models.quantize import dequantize_logits, quantize_input, quantize_model

    device = torch.device(env["device"])
    name = args.model
    modes = _modes(args, (("C", "int"), ("C", "fs"), ("K", "int"), ("Kpre", "int")))
    rate = args.rate
    tamper_cell = f"tamper_C_int_lam{args.tamper_lam}" + (f"_rate{rate}" if rate != cli.RATE else "")
    substitute = CNN_MODELS / "resnet18_224_squirrel.pt"
    if (name == "resnet18_224" and args.tampers and any(m == "C" for m, _ in modes) and not substitute.exists()
            and not is_done(args, "cnn", name, tamper_cell)):
        raise SystemExit(f"{substitute} is missing: the attacks on resnet18_224 include Anchuri et al.'s substitute "
                         "model (python -m experiments.costs.train_cnn --model resnet18_224_squirrel)")
    model, data = trained_mlp() if name == "mlp_mnist" else trained_cnn(name)
    from experiments.common import checkpoints
    env["sources"] = {**(env.get("sources") or {}), **checkpoints.SOURCES}   # the weights' SHA-256 in every env row
    graph = quantize_model(model, data["train_x"])
    mats = graph.mat_ops
    n_params = graph.n_params()
    model_f = model.float().to(device)
    xs, _ = next(test_batches(data, max([args.queries + 64, *args.cnn_batches])))
    q_inputs = quantize_input(graph, xs)          # int8 queries, one per trial

    # ---- model facts and fidelity ------------------------------------------------
    cell = "facts"
    if not is_done(args, "cnn", name, cell):
        r = Recorder(args, "cnn", name, cell, {}, env)
        r.rec("n_params", n_params, "")
        r.rec("model_bytes_int8", sum(op.weight.numel() + 4 * (op.bias.numel() if op.bias is not None else 0)
                                      for op in mats), "B")
        correct_f = correct_i = total = 0
        with torch.no_grad():
            for xb, yb in test_batches(data, 128 if "test_x" in data else 32):
                lf = model_f(xb.to(device)).argmax(1).cpu()
                _, cl = graph.forward(quantize_input(graph, xb).to(device))
                li = dequantize_logits(graph, cl[mats[-1].name]).argmax(1).cpu()
                correct_f += int((lf == yb).sum())
                correct_i += int((li == yb).sum())
                total += len(yb)
        r.rec("float_accuracy", correct_f / total, "", n=total)
        r.rec("int8_accuracy", correct_i / total, "", n=total)
        r.done()

    # ---- the single-neuron attack on the float model --------------------------------
    cell = "attack_float"
    if not is_done(args, "cnn", name, cell):
        r = Recorder(args, "cnn", name, cell, {"n": len(xs)}, env)
        seq = isinstance(model_f, nn.Sequential)
        head = model_f[-1] if seq else model_f.head
        feat_fn = (lambda b_: model_f[:-1](b_)) if seq else model_f.features
        with torch.no_grad():
            feats = feat_fn(xs.to(device)).double()
            w, b = head.weight.double(), head.bias.double()
            logits = feats @ w.T + b
            pred = logits.argmax(1)

            def best_flip(i):
                best = None
                l_ = logits[i]
                for c in range(w.shape[0]):
                    if c == int(pred[i]):
                        continue
                    others = torch.cat([w[:c], w[c + 1:]])       # rows k != c
                    lo = torch.cat([l_[:c], l_[c + 1:]])
                    ok_j = (w[c][None, :] > others).all(0)  # c wins for a large enough delta
                    if not bool(ok_j.any()):
                        continue
                    gap = (lo - l_[c])[:, None] / (w[c][None, :] - others)
                    need = gap.clamp_min(0).amax(0) * 1.0001 + 1e-9
                    need = torch.where(ok_j, need, torch.full_like(need, math.inf))
                    j = int(need.argmin())
                    if best is None or float(need[j]) < best[0]:
                        best = (float(need[j]), c, j)
                return best

            for i in range(len(xs)):
                best = best_flip(i)
                if best is None:
                    r.rec("flip_success", 0, "", i)
                    continue
                delta, c, j = best
                new = logits[i] + delta * w[:, j]
                r.rec("flip_success", int(int(new.argmax()) == c), "", i)
        r.done()

    # ---- the path test (Anchuri et al.) on the integer graph -----------------------
    cell = "sampling"
    if not is_done(args, "cnn", name, cell):
        r = Recorder(args, "cnn", name, cell, {}, env)
        env1, _ = graph.forward(q_inputs[:1])
        tc = TraceCommitment(graph, env1)
        r.rec("open_all_bytes", tc.open_all_bytes, "B")
        mass = visit_probabilities(graph, env1)
        prod = {op.output: op for op in graph.ops}
        src = _base_name(prod, mats[-1].inputs[0])
        p_attack = float(mass[src].min())
        p_min = min(float(mass[n].min()) for n in neuron_tensors(graph))
        r.rec("p_detect_penultimate", p_attack, "")
        r.rec("p_detect_min_node", p_min, "")
        k_lam = paths_for(args.path_lam, p_attack)    # paths for 2^-lam against the single-neuron attack
        grid = list(args.path_grid)
        shared = shared_path_bytes(tc, env1, sorted({k_lam, *grid}))
        for k_ in grid:   # the cost-vs-security curve of the path test
            r.rec("paths_bytes_shared_k", shared[k_], "B", k=k_, bits=-k_ * math.log2(1.0 - p_attack))
        r.rec("paths_needed_penultimate", k_lam, "", lam=args.path_lam)
        r.rec("paths_bytes_shared", shared[k_lam], "B", lam=args.path_lam)
        r.done()

    # ---- protocol v0: commitment, honest queries, every mode and security level ----------
    # The commitment is one-time and takes seconds for these models, so it is always rebuilt (the Merkle roots are
    # deterministic).
    records.reset_peaks(device)
    commit_cs: dict = {}
    coms, commit_s = _commit(args, graph, device, cs=commit_cs)
    cell = f"commit_rate{rate}"
    if coms.plan is not None and any(m == "C" for m, _ in modes) and not is_done(args, "cnn", name, cell):
        r = Recorder(args, "cnn", name, cell, {"rate": rate, **_plan_config(coms)}, env)
        r.rec("commit_total", commit_s, "s", **_cs(commit_cs, "commit_total"))
        r.rec("commit_roots_sha256", records.roots_sha256(coms)[0], "sha256")
        _record_setup(r, coms, mats)
        _record_peaks(r, device)
        r.done()

    prover = Prover(graph, device=device, commitments=coms)
    weights = {op.name: (op.weight, op.bias) for op in mats}
    for lam in args.lams:
        for mode, chal in modes:
            cell = f"defence_{mode}_{chal}_lam{lam}_rate{rate}"
            if is_done(args, "cnn", name, cell):
                continue
            plan = coms.plan if mode == "C" else None        # (modes K and Kpre open no columns)
            params = params_for(lam, len(mats), rate=rate, fiat_shamir=(chal == "fs"), plan=plan)
            cfg = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": rate,
                   "columns": params.columns, "threads": torch.get_num_threads(),
                   **(_plan_config(coms, params) if plan is not None else {})}
            r = Recorder(args, "cnn", name, cell, cfg, env)
            if plan is None:
                shapes = [(op.row_length, rate * (1 << max(0, (op.row_length - 1).bit_length()))) for op in mats]
                r.rec("soundness_bits", soundness_bits(params, shapes, "C" if mode == "C" else "K"), "bits")
            else:
                r.rec("soundness_bits", _plan_soundness(params, coms.plan, "C"), "bits")
            if mode == "C":
                v = _verifier(args, graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                              tables=coms.table_publics)
            else:
                v = _verifier(args, graph.public(), params, mode, weights=weights)
                if mode == "Kpre":
                    v.precompute(Challenger())
            records.reset_peaks(device)
            _query(args, prover, v, q_inputs[:1])  # untimed warm-up (lazy CUDA/BLAS initialisation)
            for i in range(args.queries):
                _record_query(r, _query(args, prover, v, q_inputs[i:i + 1]), i)
            _record_peaks(r, device)
            # batch amortisation: B queries in one interaction (mode C, interactive only)
            if mode == "C" and chal == "int":
                for bsz in args.cnn_batches:
                    if bsz > len(q_inputs):
                        continue
                    records.reset_peaks(device)
                    for tr in range(args.batch_trials):
                        _record_query(r, _query(args, prover, v, q_inputs[:bsz]), tr, batch=bsz)
                    _record_peaks(r, device, batch=bsz)
            r.done()

    # ---- soundness experiments: every attack must be rejected (--tampers 0: none) ------------
    lam, cell = args.tamper_lam, tamper_cell
    if args.tampers and any(m == "C" for m, _ in modes) and not is_done(args, "cnn", name, cell):
        params = params_for(lam, len(mats), rate=rate, plan=coms.plan)
        v = _verifier(args, graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                      tables=coms.table_publics)
        r = Recorder(args, "cnn", name, cell, {"lam": lam, "mode": "C", "reps": params.reps, "columns": params.columns,
                                               **_plan_config(coms, params)}, env)
        g = torch.Generator().manual_seed(args.tamper_seed)
        pen = _penultimate_mat(graph)
        x0 = q_inputs[:1]

        def attempt(kind, i, **kw):
            res = _query(args, prover, v, x0, **kw)
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
        if name == "resnet18_224":                  # (checked above: the substitute model exists)
            m2, d2 = trained_cnn("resnet18_224_squirrel")
            g2 = quantize_model(m2, d2["train_x"])
            other = {a.name: b_ for a, b_ in zip(mats, g2.mat_ops)}
            for i in range(max(3, args.tampers // 10)):
                attempt("substituted_model_M_tilde", i, forward_kwargs={"weights_override": other})
        # Attacks aimed at the two parts of mode C that Freivalds alone cannot cover.
        # (1) forged fold: a wrong logit plus a u' solved so that Freivalds passes;
        #     only the Reed-Solomon column check can reject it.
        # (2) forged column: an honest u plus an opened column shifted by a vector in
        #     the kernel of chi, so the code check passes; only the Merkle path can reject it.
        #     (A col-layout matrix's check folds its columns with w = chi' [X ; 1]^T: a shift in
        #     the kernel of [X ; 1]^T, which the prover knows, passes it for every chi'.)
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

        def kernel_vector(left):
            """A nonzero delta with left @ delta = 0 mod P (Gaussian elimination; None: full column rank)."""
            m = [[int(v) % P for v in row] for row in left.tolist()]
            pivots = []
            for col in range(left.shape[1]):
                piv = next((i_ for i_ in range(len(pivots), len(m)) if m[i_][col]), None)
                if piv is None:
                    continue
                r_ = len(pivots)
                m[r_], m[piv] = m[piv], m[r_]
                inv = pow(m[r_][col], P - 2, P)
                m[r_] = [(v_ * inv) % P for v_ in m[r_]]
                for i_ in range(len(m)):
                    if i_ != r_ and m[i_][col]:
                        f_ = m[i_][col]
                        m[i_] = [(vi - f_ * vc) % P for vi, vc in zip(m[i_], m[r_])]
                pivots.append(col)
            free = next((c_ for c_ in range(left.shape[1]) if c_ not in pivots), None)
            if free is None:
                return None
            delta = torch.zeros(left.shape[1], dtype=torch.int64)
            delta[free] = 1
            for i_, c_ in enumerate(pivots):
                delta[c_] = (-m[i_][free]) % P
            return delta

        def folded_with(victim):
            """What the code check folds the victim's opened columns with: its chi, or for a col-layout victim
            [X ; 1]^T of the honest query."""
            matrix = coms.plan.matrix_of(victim) if coms.plan is not None else None
            if matrix is None or matrix.layout == "row":
                return state["chis"][victim]
            first = next(op for op in mats if op.name == matrix.members[0])
            xt = first.unfold(prover.graph.forward(x0.to(prover.device))[0][first.inputs[0]]).T.cpu()
            return torch.cat([xt, torch.ones(xt.shape[0], 1, dtype=torch.int64)], 1) if first.has_bias else xt

        folded_final = coms.plan is None or coms.plan.matrix_of(final.name).layout == "row"
        for i in range(max(3, args.tampers // 10)):
            if folded_final:                 # (a col-layout final op has no u to forge: output_logit covers it)
                prover.fold = forged_fold
                attempt("forged_fold", i, forward_kwargs={"tamper": t_forge})
                prover.fold = real_fold
            victim = mats[i % len(mats)].name

            def capture_fold(chis):
                state["chis"] = chis
                return real_fold(chis)

            def forged_open(cols, victim=victim):
                out = real_open(cols)
                key, off = _opening_of(coms, victim)      # (under a plan: the victim's rows of its group's)
                c, pth = out[key]
                delta = kernel_vector(folded_with(victim))
                if delta is not None:
                    c = c.clone()
                    c[off:off + len(delta), 0] = (c[off:off + len(delta), 0] + delta) % P
                out[key] = (c, pth)
                return out

            prover.fold, prover.open = capture_fold, forged_open
            attempt("forged_column", i)
            prover.fold, prover.open = real_fold, real_open
        r.done()


# ------------------------------------------------------------------ decoders
def suite_llm(args, env) -> None:
    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder, decoder_param_count, greedy_tokens, with_lm_positions

    device = torch.device(env["device"])
    cfg = decoder_config(args.model)
    prune = args.prune_last
    rate = args.rate
    rtag = f"_rate{rate}" if rate != cli.RATE else ""      # (in the base name of the plan cells)
    if args.builds == "auto":      # 12 blocks or fewer: all of them; else 1 and 2
        builds = [cfg.n_layers] if cfg.n_layers <= 12 else [1, 2]
    else:                          # e.g. "full" or "1,2,full" (L1, L2 and the full model in one job)
        builds = [cfg.n_layers if b == "full" else int(b) for b in args.builds.split(",")]
    modes = _modes(args, (("C", "int"), ("C", "fs"), ("Kpre", "int"), ("K", "int")))
    n_prompts = max([args.queries, *args.batches])
    ck, fam, rw = None, None, {}
    if args.real:        # a checkpoint's weights (experiments.common.real_cells), loaded at the first build to run
        fam = real_cells.check(cfg, seqs=[s + max(args.gen - 1, 0) for s in args.seq], layers=builds,
                               prob_max=args.prob_max)
        rw = {"real": args.real, "recipe": "main" if fam == "llama" else "certified"}
    skipped = []

    def todo_of(seq: int, n_layers: int) -> list[str]:
        base = f"T{seq}_L{n_layers}"
        todo = [f"defence_{m}_{c}_lam{lam}_{base}" + (rtag if m == "C" else "") for m, c in modes
                for lam in args.lams]
        if args.llm_tampers and n_layers == cfg.n_layers and any(m == "C" for m, _ in modes):
            todo.append(f"tamper_C_int_lam{args.tamper_lam}_{base}{rtag}")
        return todo

    # the builds still to run (a real checkpoint's fp32 model is dropped after the last of them)
    left = sum(not all(is_done(args, "llm", cfg.name, t) for t in todo_of(s, n)) for s in args.seq for n in builds)
    for seq in args.seq:
        for n_layers in builds:
            base = f"T{seq}_L{n_layers}"
            if all(is_done(args, "llm", cfg.name, t) for t in todo_of(seq, n_layers)):
                continue
            left -= 1
            if device.type == "cuda":  # a build whose int8 weights alone fill the GPU would only run out of memory
                need = sum(sh.n_rows * sh.row_length for sh in decoder_shapes(cfg, n_layers=n_layers,
                                                                              prune_last=prune))
                have = torch.cuda.get_device_properties(device).total_memory
                if need > 0.85 * have:     # ... after its build and commit: skip it, run the others
                    print(f"SKIP {cfg.name} T{seq} L{n_layers}: {need / 2**30:.1f} GiB of int8 weights "
                          f"> 85% of this GPU's {have / 2**30:.0f} GiB", flush=True)
                    skipped.append(f"T{seq}_L{n_layers}")
                    if ck is not None and not left:
                        ck.free()
                    continue
            records.reset_peaks(device)
            load_s = None
            if args.real and ck is None:   # once per job: the checkpoint in fp32 and its calibration
                t0 = time.perf_counter()
                long = max(args.seq) + max(args.gen - 1, 0)
                ck = real_cells.load(args.real, cfg, seq=long, n_queries=max(1, -(-n_prompts * max(args.seq) // long)))
                load_s = time.perf_counter() - t0
                env["sources"] = {**(env.get("sources") or {}), **checkpoints.SOURCES}   # its revision in env rows
            setup_cs: dict = {}
            c0 = contention.begin()
            t0 = time.perf_counter()
            if ck is not None:
                graph, _ = ck.graph(prune, n_layers)
                if not left:               # the job's last build: the fp32 model is no longer needed
                    ck.free()
            else:
                graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, args.calib_tokens),
                                      seed=args.seed, prune_last=prune, prob_max=args.prob_max)
            build_s = time.perf_counter() - t0
            contention.add(setup_cs, "build_graph", c0)
            mats = graph.mat_ops
            # the first ``queries`` rows are the single-prompt queries
            if ck is not None:
                tokens = real_cells.prompts(ck, seq, n_prompts)
            else:
                tokens = torch.randint(0, cfg.vocab, (n_prompts, seq),
                                       generator=torch.Generator().manual_seed(args.prompt_seed))
            generate_s = None
            if args.gen > 1:   # each prompt and the gen - 1 tokens this build generates after it, one prefill
                t0 = time.perf_counter()
                tokens = torch.cat([greedy_tokens(graph, tokens[i:i + 1].to(device), args.gen - 1).cpu()
                                    for i in range(tokens.shape[0])])
                generate_s = (time.perf_counter() - t0) / tokens.shape[0]
                graph = with_lm_positions(graph, args.gen)
            coms = None
            if any(m == "C" for m, _ in modes):
                # a real OPT's full graph is planned from its own ops (its LM head has a bias column); any other
                # build for the whole model's shapes as its graph has them
                plan_ops = (None if fam == "opt" and n_layers == cfg.n_layers
                            else real_cells.model_ops(cfg, prune, bool(args.real)))
                coms, commit_s = _commit(args, graph, device, plan_ops, cs=setup_cs)
                cell = f"commit_{base}{rtag}"
                if not is_done(args, "llm", cfg.name, cell):
                    r = Recorder(args, "llm", cfg.name, cell, {"seq": seq, "n_layers": n_layers, "rate": rate,
                                                               "n_layers_full": cfg.n_layers, "lean": args.lean,
                                                               "prob_max": args.prob_max,
                                                               "params_full": decoder_param_count(cfg),
                                                               **({"prune_last": True} if prune else {}),
                                                               **_plan_config(coms), **rw}, env)
                    if load_s is not None:
                        r.rec("load_checkpoint", load_s, "s")
                    r.rec("build_graph", build_s, "s", **_cs(setup_cs, "build_graph"))
                    if coms.setup_cache == "hit":   # rebuilt from the setup cache: a load time, never commit_total
                        r.rec("commit_cached_load", commit_s, "s", setup_cache="hit")
                    else:
                        r.rec("commit_total", commit_s, "s", setup_cache=coms.setup_cache,
                              **_cs(setup_cs, "commit_total"))
                    r.rec("commit_roots_sha256", records.roots_sha256(coms)[0], "sha256")
                    if coms.plan is not None:
                        _record_setup(r, coms, mats)
                    _record_peaks(r, device)
                    r.done()
            prover = Prover(graph, device=device, commitments=coms, lean=args.lean)
            weights = {op.name: (op.weight, op.bias) for op in mats}
            for lam in args.lams:
                for mode, chal in modes:
                    cell = f"defence_{mode}_{chal}_lam{lam}_{base}" + (rtag if mode == "C" else "")
                    if is_done(args, "llm", cfg.name, cell):
                        continue
                    # size (r, t) for the FULL model's op count, so extrapolated rows keep their lambda
                    full_shapes = decoder_shapes(cfg, prune_last=prune)
                    plan = coms.plan if coms is not None and mode == "C" else None   # (K, Kpre: no columns)
                    params = params_for(lam, len(full_shapes), rate=rate, fiat_shamir=(chal == "fs"), plan=plan)
                    conf = {"mode": mode, "challenges": chal, "lam": lam, "reps": params.reps, "rate": rate,
                            "columns": params.columns, "seq": seq, "n_layers": n_layers,
                            "n_layers_full": cfg.n_layers, "params_full": decoder_param_count(cfg),
                            "params_built": graph.n_params(), "lean": args.lean, "prob_max": args.prob_max,
                            "threads": torch.get_num_threads(), **({"prune_last": True} if prune else {}),
                            **({"lookups": True} if args.lookups and mode != "C" else {}),
                            **({"gen": args.gen, "positions": tokens.shape[1]} if args.gen > 1 else {}),
                            **(_plan_config(coms, params) if plan is not None else {}), **rw}
                    r = Recorder(args, "llm", cfg.name, cell, conf, env)
                    if generate_s is not None:   # untimed by the protocol: the honest generation, per query
                        r.rec("generate", generate_s, "s")
                    if plan is not None:      # the whole model's bound, under the whole model's plan
                        # (a real OPT's plan is made from its own ops: its LM head has a folded bias column)
                        whole = plan_commitment(mats if fam == "opt" else full_shapes, args.policy, rate=rate)
                        r.rec("soundness_bits", _plan_soundness(params_for(
                            lam, len(full_shapes), fiat_shamir=(chal == "fs"), plan=whole), whole, "C"), "bits")
                    if mode == "C":
                        v = _verifier(args, graph.public(), params, "C", publics=coms.publics,
                                      groups=coms.group_publics, tables=coms.table_publics, lean=args.lean)
                    else:
                        v = _verifier(args, graph.public(), params, mode, weights=weights, lean=args.lean,
                                      lookups=args.lookups)
                        if mode == "Kpre":
                            pre_cs: dict = {}
                            c0 = contention.begin()
                            t0 = time.perf_counter()
                            v.precompute(Challenger())   # ends with a sync of the verifier's device
                            dt = time.perf_counter() - t0
                            contention.add(pre_cs, "verifier_precompute", c0)
                            r.rec("verifier_precompute", dt, "s", **_cs(pre_cs, "verifier_precompute"))
                    records.reset_peaks(device)
                    _query(args, prover, v, tokens[:1])  # untimed warm-up
                    for i in range(args.queries):
                        _record_query(r, _query(args, prover, v, tokens[i:i + 1]), i)
                    _record_peaks(r, device)
                    # batch amortisation: B prompts against one set of u and openings (as for the image models);
                    # interactive or Fiat-Shamir
                    if mode in ("C", "Kpre"):
                        for bsz in args.batches:
                            records.reset_peaks(device)
                            for tr in range(args.batch_trials):
                                _record_query(r, _query(args, prover, v, tokens[:bsz]), tr, batch=bsz)
                            _record_peaks(r, device, batch=bsz)
                    # one tampered query per cell: must be rejected
                    victim = mats[len(mats) // 2].name

                    def tamper(o, z, victim=victim):
                        if o.name == victim:
                            z = z.clone()
                            z.view(-1)[0] += 1
                        return z

                    res = _query(args, prover, v, tokens[:1], forward_kwargs={"tamper": tamper})
                    r.rec("tamper_rejected", int(not res["accepted"]), "", stage=res["rejected_at"])
                    r.done()
            # ---- attacks on this build: random single values anywhere, the top logit, the last hidden state, and
            # one random block run with 1%-perturbed weights; mode C, lambda = --tamper-lam
            lam = args.tamper_lam
            cell = f"tamper_C_int_lam{lam}_{base}{rtag}"
            if (args.llm_tampers and coms is not None and n_layers == cfg.n_layers
                    and not is_done(args, "llm", cfg.name, cell)):
                params = params_for(lam, len(decoder_shapes(cfg)), rate=rate, plan=coms.plan)
                v = _verifier(args, graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                              tables=coms.table_publics, lean=args.lean)
                r = Recorder(args, "llm", cfg.name, cell, {"lam": lam, "mode": "C", "reps": params.reps,
                                                           "columns": params.columns, "seq": seq,
                                                           "n_layers": n_layers, "n_layers_full": cfg.n_layers,
                                                           "lean": args.lean, "prob_max": args.prob_max,
                                                           **({"prune_last": True} if prune else {}),
                                                           **_plan_config(coms, params), **rw}, env)
                g = torch.Generator().manual_seed(args.tamper_seed)
                x0 = tokens[:1]

                def attempt(kind, i, **kw):
                    res = _query(args, prover, v, x0, forward_kwargs=kw)
                    r.rec("rejected", int(not res["accepted"]), "", i, attack=kind, stage=res["rejected_at"])

                def t_logit(o, z):      # column 0: the LM head's only position (with --gen G, the first of G)
                    if o.name == mats[-1].name:
                        z = z.clone()
                        k = int(z[:, 0].argmin())
                        z[k, 0] = int(z[:, 0].max()) + 1000
                    return z

                for i in range(args.llm_tampers):
                    op = mats[int(torch.randint(len(mats), (1,), generator=g))]
                    pos = int(torch.randint(10 ** 9, (1,), generator=g))
                    delta = int(torch.randint(1, 1000, (1,), generator=g)) * (1 if i % 2 else -1)

                    def t_single(o, z, op=op, pos=pos, delta=delta):
                        if o.name == op.name:
                            z = z.clone()
                            z.view(-1)[pos % z.numel()] += delta
                        return z

                    attempt("single_value", i, tamper=t_single)
                    attempt("output_logit", i, tamper=t_logit)
                    # the single-neuron tamper: one coordinate of the op feeding the LM head (the last block's
                    # down/fc2 projection), at the last position, set to Z_BOUND // 4
                    j = int(torch.randint(mats[-2].n_rows, (1,), generator=g))

                    def t_last(o, z, j=j):
                        if o.name == mats[-2].name:
                            z = z.clone()
                            z[j, -1] = Z_BOUND // 4
                        return z

                    attempt("last_hidden", i, tamper=t_last)
                proj = 1 if getattr(cfg, "embed_dim", 0) else 0    # OPT-350M: project_in / project_out
                first = (2 if cfg.pos == "learned" else 1) + proj  # token (and position) embedding
                per_block = (len(mats) - first - 1 - proj) // n_layers   # the LM head is last
                assert per_block * n_layers == len(mats) - first - 1 - proj, "unexpected decoder op layout"
                for i in range(max(3, args.llm_tampers // 10)):
                    b0 = first + per_block * int(torch.randint(n_layers, (1,), generator=g))
                    other = {}
                    for op in mats[b0:b0 + per_block]:
                        w = op.weight.to(torch.int16)
                        mask = (torch.rand(w.shape, generator=g) < 0.01).to(torch.int16)
                        sign = 2 * torch.randint(0, 2, w.shape, generator=g, dtype=torch.int16) - 1
                        other[op.name] = MatOp(op.name, op.inputs, op.output, bias=op.bias, layout=op.layout,
                                               weight=(w + mask * sign).clamp(-127, 127).to(torch.int8))
                    attempt("substituted_block_1pct", i, weights_override=other)
                    other = None
                r.done()
            # drop every reference to this build (``mats`` and ``weights`` keep its MatOps, and with them the GPU
            # weight copies, alive otherwise)
            prover = coms = graph = mats = weights = v = r = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if skipped:   # the job must not look complete
        raise SystemExit(f"{cfg.name}: builds {', '.join(skipped)} do not fit this GPU (the others ran)")


# ------------------------------------------------------------------ the command line
def build_parser() -> argparse.ArgumentParser:
    paper = cli.PAPER
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("suite", choices=["cnn", "llm"])
    ap.add_argument("--model", required=True,
                    help="image models: mlp_mnist or a pvi.models.cnn.MODEL_SPECS name; decoders: a "
                         "pvi.models.decoder.CONFIGS name or a small shape of experiments.common.models.TINY")
    ap.add_argument("--platform", required=True,
                    help="a name for this GPU + CPU: the records go to artifacts/comparison/raw_<platform>/ "
                         f"(the stored roots are frozen; {FROZEN_ALIAS!r} names raw/)")
    ap.add_argument("--queries", type=int, default=30, help="honest queries per defence cell, after a warm-up")
    ap.add_argument("--modes", default="",
                    help="e.g. C:fs,Kpre:fs (default: image models C:int,C:fs,K:int,Kpre:int; decoders "
                         "C:int,C:fs,Kpre:int,K:int)")
    ap.add_argument("--lams", type=int, nargs="+", default=list(LAMBDAS), help="security levels of the defence cells")
    g = ap.add_argument_group("protocol (defaults: the paper's measured configuration)")
    g.add_argument("--policy", default=paper.policy,
                   help="mode C's commitment plan (pvi.commit.plans): per-op (every weight op on a tree of its own), "
                        "tight, cnn<e> or R<rate>, each also with the suffix c (col layouts and lookup tables), or "
                        "auto (the planning rule; the plan it picks is recorded as the cells' policy); the commitment "
                        "and mode-C cells of a plan other than per-op are named _pol<name>")
    g.add_argument("--wire", action=argparse.BooleanOptionalAction, default=paper.wire,
                   help="the proof in the compact encoding of pvi.v0.claimcodec (run_query(wire=True); _wire)")
    g.add_argument("--lean", action=argparse.BooleanOptionalAction, default=paper.lean,
                   help="decoders: the lean prover and verifier (dead activations freed, claims streamed to the "
                        "host): the same integers and proof, less memory (off: _nolean)")
    g.add_argument("--prune-last", action=argparse.BooleanOptionalAction, default=paper.prune_last,
                   help="decoders: the last block at the last position only (build_decoder(prune_last=True); _prune)")
    g.add_argument("--lookups", action=argparse.BooleanOptionalAction, default=paper.lookups,
                   help="decoders, modes K and Kpre: the verifier reads the embedding rows itself and the prover "
                        "sends no claims for them (Verifier(lookups=True); _lookups)")
    cli.add_prob_max_argument(g, default=paper.prob_max,
                              note=f"; a scale other than the stored cells' {STORED_PROB_MAX}: _pmax<n>")
    g.add_argument("--rate", type=int, default=cli.RATE, help="Reed-Solomon rate of the weight commitment")
    g.add_argument("--setup-cache", default=None, metavar="DIR",
                   help="the prover's setup cache (pvi.commit.setup_cache): commitments are rebuilt from stored leaf "
                        "digests (the same roots and proofs; recorded as commit_cached_load); default none")
    g = ap.add_argument_group("cells")
    g.add_argument("--tampers", type=int, default=100,
                   help="image models: attacks per type in the tamper cell (0: no tamper cell)")
    g.add_argument("--llm-tampers", type=int, default=0,
                   help="decoders, full builds: attacks per type in a tamper_C_int_lam<l>_<build> cell (0: none)")
    g.add_argument("--tamper-lam", type=int, default=40, help="the security level of the tamper cells")
    g.add_argument("--tamper-seed", type=int, default=123, help="the attacks' random choices")
    g.add_argument("--seq", type=int, nargs="+", default=[64], help="decoders: prompt lengths")
    g.add_argument("--builds", default="auto",
                   help="decoder blocks: 'auto' (all if <= 12, else 1,2), 'full', or e.g. '1,2,full'")
    g.add_argument("--gen", type=int, default=0,
                   help="decoders: a generated response of G tokens per query, proved as one prefill: the prompt of "
                        "--seq tokens and the G - 1 tokens the build generates greedily after it, with the logits at "
                        "the last G positions (decoder.with_lm_positions; the verifier checks the tokens); _gen<G>")
    g.add_argument("--cnn-batches", type=int, nargs="+", default=list(CNN_BATCHES),
                   help="image models: batch sizes timed in the defence_C_int cells (others than 8 32: _batch)")
    g.add_argument("--batches", type=int, nargs="*", default=[],
                   help="decoders: also time B prompts in one interaction (the C and Kpre cells; any: _batch)")
    g.add_argument("--batch-trials", type=int, default=1, help="repetitions per batch size")
    g.add_argument("--path-lam", type=int, default=40,
                   help="image models: the path test's paths for 2^-lam against the single-neuron attack")
    g.add_argument("--path-grid", type=int, nargs="+", default=list(PATH_GRID),
                   help="image models: the path counts of the path test's cost-against-security curve")
    g.add_argument("--seed", type=int, default=0, help="decoders: the random weights' seed")
    g.add_argument("--prompt-seed", type=int, default=1, help="decoders: the random prompts' seed")
    g.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                   help="decoders: calibration tokens of the random-weight build (at most --seq)")
    real_cells.add_arguments(g, real_cells.HELP + "; _real")
    g = ap.add_argument_group("verifier and numerics")
    g.add_argument("--verifier-impl", default="default", choices=["default", "stream"],
                   help="'stream': the streaming verifier, the same verdicts (_stream)")
    g.add_argument("--tf32", action="store_true",
                   help="TF32 tensor cores for the float32 GEMMs (exact: operands <= 255; _tf32)")
    cli.add_runtime_arguments(ap, threads=THREADS)
    return ap


def check(args) -> None:
    """Refuse settings that do not apply or would mislabel the records."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.platform):
        raise SystemExit("--platform: a name of letters, digits, '.', '_' and '-'")
    if args.suite == "cnn" and args.gen:
        raise SystemExit("--gen is for the decoders (the llm suite)")
    if args.suite == "cnn" and args.real:
        raise SystemExit("--real is for the decoders (the llm suite)")
    if args.verifier_device not in ("cpu", "cuda") and not args.verifier_device.startswith("cuda:"):
        raise SystemExit("--verifier-device: cpu or cuda")
    if args.tf32 and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        raise SystemExit("--tf32 with NVIDIA_TF32_OVERRIDE=0: cuBLAS would ignore it and the _tf32 cells would hold "
                         "non-TF32 timings")
    try:
        plan_commitment([], args.policy)
    except ValueError as exc:
        raise SystemExit(f"--policy: {exc}")


def main() -> None:
    global RAW
    _require_this_pvi()
    args = build_parser().parse_args()
    check(args)
    cli.apply_runtime(args)
    RAW = raw_root(args.platform)
    # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE (NGC containers) only sets torch's default, which these assignments
    # override.  NVIDIA_TF32_OVERRIDE=0 is different: cuBLAS then ignores allow_tf32 altogether (see check()).
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = False
    env = env_info(args.device, args.platform)
    print(json.dumps(env), flush=True)
    print(f"pvi from {env['pvi_path']}; verifier threads {env['torch_threads']} on {env.get('cpu_affinity', '?')} "
          f"CPUs / {env.get('cpu_affinity_cores', '?')} physical cores", flush=True)
    claim_platform(env, args.platform)
    t0 = time.time()
    (suite_cnn if args.suite == "cnn" else suite_llm)(args, env)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
