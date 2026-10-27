"""Reading a run's records: the job logs and the JSON files of the run package, each record's kind and tier, and
the correctness checks.

The run package (``experiments/package/run.sh``) writes every job's output to ``logs/<name>/sp-<cell>-<jobid>.out``;
each line that starts with ``{`` is one record (one JSON object).  The cell's prefix names the tier (:data:`TIERS`),
and the record's fields name the script that wrote it (:func:`kind_of`).  The attack tier writes JSON documents to
``logs/<name>/attack/`` and the tv tier to ``logs/<name>/sampler_tv_*.json``; :func:`read_files` reads them and
:func:`doc_kind` names their kinds from their file names (:data:`FILES`, :data:`DOCUMENTS`).  Nothing here runs a
model.

The checks (:func:`checks`): an honest query rejected or a tamper accepted (``perf``, ``v1``, ``gen``), a client
process whose verdicts differ from the in-process ones (``perf``'s ``client_memory``), a byte accounting that
disagrees with the model (``v1``), a serving check that failed (``serve``), a setup proof rejected (``setup``), an
implementation A/B whose two paths disagree (the ``improve`` tier), a zkLLM run whose proof binaries failed or
whose layers did not all print zkLLM's verifier messages (``zkllm``), a llama.cpp record not marked inexact and a
different model or not run on its threads (``llamacpp``), a cpux cell that ran on fewer CPUs than its threads, an
attribution study whose float copy failed a check (``attribution``), a cell whose proofs or commitments differ between
the GPU and the CPU (``digests``), and the smoke's own verdicts (no ``SMOKE OK``, int8 GEMMs that are not exact, a test
of its pytest run that failed: the ``tests`` record of its summary line) or a ``FAILED`` line
of a script in a job's log; between two jobs (:func:`cross_checks`), the attribution's integer graph against the
quality tier's cell of the same graph.  Each job's GPU, CPU, driver and node come from the node line at the start of
its log (:func:`nodes`) and its records' environments (:func:`machines`).  Each is reported as a ``CHECK FAILED`` line; a line that starts with ``{`` and
does not parse, or parses as no known kind, is a ``PARSE`` line.  Either makes ``paper/numbers.py`` exit with status
1.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

TIERS = (("smoke", "smoke"), ("xcheck", "smoke"), ("c-", "cpuv"), ("x-", "cpux"), ("v1-", "v1"), ("serve-", "serve"),
         ("b-", "bench"), ("g-", "gpuv"), ("gen-", "gen"), ("bt-", "batch"), ("r-", "real"), ("setup-", "setup"),
         ("i-", "improve"), ("q-", "quality"), ("tv-", "tv"), ("a-", "attack"), ("l-", "long"), ("zk-", "zkllm"),
         ("gate-", "gate"))
"""The job-name prefixes of ``run.sh`` (``sp-<prefix>...``) and their tiers (``gate``: the full-size checks run once
before the run, whose records fill no table cell or text number; ``zkllm``: zkLLM's public demo on the run's GPU,
whose figures ``paper/bench.py`` reads from ``logs/<name>/zkllm/summary.csv``; the smoke tier's job ``xcheck``: the
proofs of fixed cells on the GPU and on the CPU, ``comparison/proof_digests.py``)."""
KINDS = {"smoke": {"v1", "h2d", "tests", "digests"}, "cpuv": {"perf", "reexec", "llamacpp", "v1"},
         "cpux": {"perf", "reexec"}, "v1": {"v1"},
         "serve": {"serve"}, "bench": {"bench_env"}, "gpuv": {"perf", "v1", "reexec", "h2d"}, "gen": {"gen"},
         "batch": {"bench_env"}, "real": {"perf", "v1", "gen"}, "setup": {"setup"},
         "improve": {"open_ab", "forward_ab", "encode_ab", "perf", "gpu_profile", "fold_ab", "v1_phases", "gkr"},
         "quality": {"ppl", "zeroshot", "digest", "llama_diag", "llama_attrib", "attribution"}, "tv": {"tv"},
         "attack": {"reproduction", "attack_mnist", "samplers", "floor_sampler", "adaptive_game", "path_bound",
                    "attack_llm", "inrange", "adaptive_game_llm"},
         "long": {"perf", "reexec"}, "zkllm": {"zkllm"}, "gate": {"v1", "perf", "reexec"}}
"""The record kinds each tier writes (the real tier's ``gen`` with ``--llama-licence``; the smoke's ``tests``: the
summary line of its pytest run, :data:`PYTEST_SUMMARY`; ``digests``: its ``xcheck`` job's proofs on two devices)."""
PYTEST_SUMMARY = re.compile(r"^=*\s*(?P<counts>\d+ [a-z]+(?:, \d+ [a-z]+)*)(?: in [0-9.]+s(?: \([^)]*\))?)?\s*=*$")
"""pytest's summary line (``798 passed, 4 skipped in 1203.51s (0:20:03)``), which the smoke's log ends its pytest
output with: :func:`read_logs` turns it into a record of kind ``tests`` (``passed``, ``failed``, ``skipped``,
``errors``, ...)."""
PYTEST_WORDS = {"passed": "passed", "failed": "failed", "skipped": "skipped", "error": "errors", "errors": "errors",
                "xfailed": "xfailed", "xpassed": "xpassed", "deselected": "deselected", "warning": "warnings",
                "warnings": "warnings"}
REHEARSAL_KINDS = {tier: kinds for tier, kinds in KINDS.items() if tier not in ("gate", "zkllm")}
"""The kinds of a rehearsal (``run.sh --rehearsal``: each tier's smallest cells, which run every script of every
tier): each tier but ``gate`` (full size only, never rehearsed) and ``zkllm`` (zkLLM's demo, which runs only with
``--llama-licence``; its rehearsal job's record is checked as any other) must have at least one record of each of its
:data:`KINDS`."""
FILES = {"reproduction.json": "reproduction", "attack.json": "attack_mnist", "defence.json": "samplers",
         "floor_sampler.json": "floor_sampler", "adaptive_game_mlp_mnist.json": "adaptive_game",
         "analytic.csv": "path_bound", "real_llm_attack.json": "attack_llm", "real_llm_attack_auto.json": "attack_llm",
         "inrange_llm_opt-1.3b_any.json": "inrange", "inrange_llm_opt-1.3b_down.json": "inrange",
         "inrange_llm_opt-6.7b_any.json": "inrange", "inrange_llm_opt-6.7b_down.json": "inrange",
         "adaptive_game_opt-1.3b.json": "adaptive_game_llm", "sampler_tv_opt-1.3b.json": "tv",
         "sampler_tv_selfcheck.json": "tv", "sampler_tv_opt-125m.json": "tv"}
"""The documents the full run's attack tier (``logs/<name>/attack/``) and tv tier (``logs/<name>/``) write, by file
name, and their kinds: the names the tables and macros select (:meth:`Run.file`)."""
DOCUMENTS = ((re.compile(r"real_llm_attack(_.+)?\.json"), "attack_llm"),
             (re.compile(r"inrange_llm_.+_(any|down)\.json"), "inrange"),
             (re.compile(r"adaptive_game_opt-.+\.json"), "adaptive_game_llm"),
             (re.compile(r"sampler_tv_.+\.json"), "tv"))
"""The name patterns of the same scripts' documents on other models (the rehearsal's OPT-125M:
``inrange_llm_opt-125m_any.json``, ``adaptive_game_opt-125m.json``, ...) and their kinds.  Such a document counts
for its tier and kind, and fills no table cell or text number, which select the full run's names."""
LOG = re.compile(r"^sp-(?P<cell>.+)-(?P<job>[0-9]+|DRY[0-9]+)\.out$")
ATTRIBUTION_CELL = "scalar-g4-p8"
"""The integer graph of the attribution study (``attribution.py``'s default): ``ppl_opt.py``'s cell of OPT-125M with
norm gain 4 and 8-bit probabilities, calibrated on one window at the end of the test split."""


@dataclass
class Rec:
    """One record: its tier and job (from the log's name), where it was read and what wrote it."""

    tier: str | None
    cell: str
    job: str
    path: str
    line: int
    kind: str
    d: dict = field(repr=False)

    @property
    def source(self) -> str:
        """``<file>:<line>`` (or the file name of a document), for ``cells.csv``."""
        name = Path(self.path).name
        return f"{name}:{self.line}" if self.line else name

    def get(self, key, default=None):
        return self.d.get(key, default)


def tier_of(cell: str) -> str | None:
    """The tier of a job's cell name (``c-llama2-7b-64`` -> ``cpuv``), or None."""
    for prefix, tier in TIERS:
        if cell == prefix or (prefix.endswith("-") and cell.startswith(prefix)):
            return tier
    return None


def kind_of(d: dict) -> str | None:
    """The script that wrote one record, from its fields; None for a record of no known kind."""
    k = d.get("kind")
    if k in ("gen", "setup", "h2d", "gpu_profile", "llamacpp", "attribution", "zkllm", "digests"):
        return k
    if d.get("row") in ("prove", "on_request"):
        return "serve"
    if "variant" in d and "measured_s" in d and "full_s" in d:
        return "reexec"
    if "v1_total" in d or d.get("codec_only"):
        return "v1"
    if "cut" in d and "path" in d and "verify_s" in d:
        return "v1_phases"
    if "ns_per_leaf" in d:
        return "gkr"
    if "open_fold_s" in d:
        return "open_ab"
    if "vs_float32" in d:
        return "forward_ab"
    if "encode_s" in d and "same_bytes" in d:
        return "encode_ab"
    if "precompute_s" in d and "equal_to_float64" in d:
        return "fold_ab"
    if {"prove_s", "verify_s", "timings", "mode", "proof_bytes"} <= set(d):
        return "perf"
    if "cell" in d and "ppl" in d:
        return "ppl"
    if "task" in d and "variant" in d:
        return "zeroshot"
    if "first_flag" in d or "worst" in d:
        return "llama_attrib"
    if "blocks" in d and "s_residual" in d or "prefix_choice" in d and "blocks" not in d and "cell" not in d:
        return "llama_diag"
    if "digest" in d and "weights_sha256" in d:
        return "digest"
    if set(d) == {"recipe", "opts", "digest", "build_s"}:
        return "info"                               # zeroshot.py's build line
    if "pvi_path" in d and "torch_threads" in d:
        return "bench_env"                          # bench.py's environment line (its records go to raw_<name>/)
    if "anchuri_paths" in d:
        return "path_bound"
    if "moves" in d and "p_uniform_median" in d:
        return "inrange"
    if "calibration_s" in d:
        return "info"                               # the attack scripts' calibration line
    if d.get("script") == "sampler_tv.py":
        return "tv"
    return None


def pytest_summary(line: str) -> dict | None:
    """``{"kind": "tests", "passed": n, ...}`` of pytest's summary line (:data:`PYTEST_SUMMARY`), else None."""
    m = PYTEST_SUMMARY.match(line.strip())
    if m is None:
        return None
    counts = {}
    for part in m["counts"].split(", "):
        n, word = part.split(" ", 1)
        if word not in PYTEST_WORDS:
            return None
        counts[PYTEST_WORDS[word]] = int(n)
    if not {"passed", "failed", "errors"} & set(counts):
        return None
    return {"kind": "tests", "passed": 0, "failed": 0, "errors": 0, "skipped": 0, **counts, "line": line.strip()}


def read_logs(logs: Path, problems: list) -> list[Rec]:
    """Every record of the job logs ``sp-*.out`` in ``logs``; a line that starts with ``{`` and does not parse, or
    is of no known kind, is a ``PARSE`` problem.  In the smoke's logs, pytest's summary line is a record of kind
    ``tests`` (:func:`pytest_summary`)."""
    out = []
    for path in sorted(logs.glob("sp-*.out")):
        m = LOG.match(path.name)
        cell, job = (m["cell"], m["job"]) if m else (path.stem[3:], "")
        tier = tier_of(cell)
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if not line.startswith("{"):
                    summary = pytest_summary(line) if tier == "smoke" else None
                    if summary is not None:
                        out.append(Rec(tier, cell, job, str(path), i, "tests", summary))
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"PARSE {path.name}:{i}: {e}")
                    continue
                kind = kind_of(d) if isinstance(d, dict) else None
                if kind is None:
                    keys = sorted(d)[:8] if isinstance(d, dict) else type(d).__name__
                    problems.append(f"PARSE {path.name}:{i}: a record of no known kind ({keys})")
                    continue
                out.append(Rec(tier, cell, job, str(path), i, kind, d))
    return out


def doc_kind(name: str) -> str | None:
    """The kind of an attack or tv document by its file name (:data:`FILES`, else :data:`DOCUMENTS`); None for any
    other file."""
    return FILES.get(name) or next((kind for pattern, kind in DOCUMENTS if pattern.fullmatch(name)), None)


def read_files(logs: Path, problems: list) -> list[Rec]:
    """The attack and tv tiers' documents under ``logs`` (``attack/`` and the folder itself): every file with a kind
    (:func:`doc_kind`)."""
    out = []
    for folder in (logs / "attack", logs):
        for path in sorted(folder.iterdir()) if folder.is_dir() else ():
            kind = doc_kind(path.name) if path.is_file() else None
            if kind is None:
                continue
            name, tier = path.name, "tv" if kind == "tv" else "attack"
            try:
                if path.suffix == ".csv":
                    import csv
                    with open(path, newline="", encoding="utf-8") as fh:
                        d = {"rows": list(csv.DictReader(fh))}
                else:
                    d = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                problems.append(f"PARSE {path}: {e}")
                continue
            out.append(Rec(tier, name, "", str(path), 0, kind, d))
    return out


def job_order(r: Rec) -> int:
    """A record's job id as a number (later jobs, e.g. a tier run again, sort later)."""
    return int(r.job) if r.job.isdigit() else -1


def checks(r: Rec) -> list[str]:
    """The correctness checks of one record that failed."""
    d, bad = r.d, []

    def tampers(t):
        for k, v in (t or {}).items():
            if isinstance(v, dict) and "attempts" in v and v.get("rejected") != v.get("attempts"):
                bad.append(f"tamper {k}: {v.get('rejected')} of {v.get('attempts')} rejected")

    if r.kind == "perf":
        if d.get("accepted") != d.get("queries"):
            bad.append(f"honest queries accepted: {d.get('accepted')} of {d.get('queries')}")
        tampers(d.get("tampers"))
        client = d.get("client_memory")
        if client is not None and client.get("agree") is not True:
            bad.append("the client process's verdicts differ from the in-process ones, or it failed "
                       f"({client.get('error') or 'client_memory.agree false'})")
    elif r.kind == "v1" and not d.get("codec_only"):
        if d.get("accounting_ok") is False:
            bad.append("accounting_ok false (a byte accounting, acceptance or tamper check)")
        if d.get("v0_accepted") is False:
            bad.append("v0 query rejected")
        tampers(d.get("tampers"))
    elif r.kind == "serve" and d.get("row") == "prove" and "skipped" not in d and d.get("ok") is False:
        bad.append("serving check failed (ok false)")
    elif r.kind == "gen" and d.get("ok") is False:
        bad.append("an honest response rejected or a tampered one accepted")
    elif r.kind == "setup" and d.get("accepted") is False:
        bad.append("setup proof rejected")
    elif r.kind == "open_ab" and (d.get("identical") is False or d.get("accepted") != d.get("queries")):
        bad.append("the two opening paths differ or a query was rejected")
    elif r.kind == "forward_ab" and d.get("identical") is False:
        bad.append("the two forward paths give different claims")
    elif r.kind == "encode_ab" and d.get("same_bytes") is False:
        bad.append("the encoders give different bytes")
    elif r.kind == "fold_ab" and d.get("equal_to_float64") is False:
        bad.append("the Kpre precomputation differs from the float64 path")
    elif r.kind in ("gpu_profile",) and d.get("accepted") is False:
        bad.append("a profiled query was rejected")
    elif r.kind == "v1_phases" and d.get("accepted") != d.get("queries"):
        bad.append(f"V1 queries accepted: {d.get('accepted')} of {d.get('queries')}")
    elif r.kind == "gkr" and d.get("same_transcript") is False:
        bad.append("the Triton and PyTorch GKR transcripts differ")
    elif r.kind == "llamacpp":
        # llama.cpp computes another model, never exact; it ran on the record's threads, no more than the CPUs the
        # process was allowed (where the record has them)
        threads, ran = d.get("threads"), (d.get("llamacpp") or {}).get("n_threads")
        allowed = (d.get("env") or {}).get("cpu_affinity")
        if d.get("exact") is not False or d.get("same_model") is not False:
            bad.append("a llama.cpp record not marked inexact (exact false) and a different model (same_model false)")
        if ran != threads or (allowed is not None and threads is not None and allowed < threads):
            bad.append(f"llama.cpp ran on {ran} threads, not its {threads} (on {allowed} allowed CPUs)")
    elif r.kind == "tests" and (d.get("failed") or d.get("errors")):
        bad.append(f"the smoke's tests: {d.get('failed')} failed, {d.get('errors')} errors ({d.get('line')})")
    elif r.kind == "attribution" and d.get("part") == "check" and d.get("ok") is False:
        bad.append("the attribution's float copy failed a check (ok false)")
    elif r.kind == "digests" and "skipped" not in d and d.get("agree") is not True:
        bad.append(f"{d.get('model')} {d.get('mode')} {d.get('protocol')}: the proofs or commitments on "
                   f"{', '.join(d.get('devices') or {})} differ, or a query was rejected (agree false)")
    elif r.kind == "zkllm" and d.get("ok") is not True:
        bad.append(f"zkLLM's demo: {d.get('binary_failures')} proof binaries failed, verifier messages "
                   f"{d.get('verdicts')} for {d.get('layers_expected')} proved layers (ok false)")
    # a CPU timing of the paper ran on the native kernels (the improve tier's before-cells run without them on purpose;
    # a V1 record names its verifier's device on its command line)
    device = verifier_device(d) if r.kind == "v1" else str(d.get("verifier_device") or d.get("device") or "cpu")
    on_cpu = device.startswith("cpu")
    if (r.tier in ("cpuv", "cpux", "long", "real", "v1", "gate") and r.kind in ("perf", "v1", "reexec", "gen")
            and on_cpu
            and not d.get("codec_only")
            and ((d.get("env") or {}).get("implementation") or {}).get("native") is False):
        bad.append("the native CPU kernels were not used")
    # the cpux tier's cells time a CPU of their threads' cores: never on fewer CPUs than threads (where the record has
    # them)
    allowed = (d.get("env") or {}).get("cpu_affinity")
    if (r.tier == "cpux" and r.kind in ("perf", "reexec") and on_cpu and allowed is not None
            and d.get("threads") is not None and allowed < d["threads"]):
        bad.append(f"{d['threads']} threads on {allowed} allowed CPUs")
    return bad


def cross_checks(run: "Run") -> list[str]:
    """The checks between two jobs' records, as ``CHECK FAILED`` lines: the attribution's integer graph (each check
    record of ``attribution.py`` that names a ``ppl_opt.py`` cell) is the quality tier's graph of that cell with the
    same calibration and window length (the same digest), and over the same windows has its perplexity (to a relative
    1e-6); a pair of which one record is missing is not checked."""
    bad = []
    for r in run.kind("attribution", "quality"):
        d = r.d
        if d.get("part") != "check" or not d.get("graph_cell") or d.get("tag"):
            continue
        cell = run.ppl(d.get("model"), d["graph_cell"], calibration=d.get("calibration", "test-end"), seq=d.get("seq"))
        if cell is None:
            continue
        where = f"{Path(r.path).name}:{r.line} against {Path(cell.path).name}:{cell.line}"
        if d.get("digest") != cell.d.get("digest"):
            bad.append(f"CHECK FAILED attribution {where}: the integer graph's digest is not that of the cell "
                       f"{d['graph_cell']}")
        elif d.get("windows") == cell.d.get("windows") and not (
                abs(d.get("int_ppl", math.nan) - cell.d.get("ppl", math.nan)) <= 1e-6 * abs(cell.d.get("ppl", 0.0))):
            bad.append(f"CHECK FAILED attribution {where}: the integer graph's perplexity {d.get('int_ppl')!r} is not "
                       f"the cell's {cell.d.get('ppl')!r}")
    return bad


def log_checks(logs: Path) -> list[str]:
    """The verdicts written as text in the job logs: the smoke's ``SMOKE OK`` and int8 self-check, and the
    ``FAILED`` lines of the scripts (``FAILED: ...``, ``A CORRECTNESS CHECK FAILED``)."""
    out = []
    for path in sorted(logs.glob("sp-*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        m = LOG.match(path.name)
        cell = m["cell"] if m else path.stem
        if cell == "smoke":
            if "SMOKE OK" not in text:
                out.append(f"CHECK FAILED {path.name}: the smoke did not end in SMOKE OK")
            if "int8 GEMMs exact on this GPU: False" in text:
                out.append(f"CHECK FAILED {path.name}: int8 GEMMs are not exact on this GPU")
        for i, line in enumerate(text.splitlines(), 1):
            if line.startswith("FAILED:") or "A CORRECTNESS CHECK FAILED" in line:
                out.append(f"CHECK FAILED {path.name}:{i}: {line.strip()[:160]}")
    return out


NODE = re.compile(r"^node: (\{.*\})\s*$")
"""The line ``experiments/package/node.py`` writes at the start of every job's log: the node, its GPU, driver and
CPU."""


WRONG_NODE = "wrong node:"
"""The start of the line ``experiments/package/node.py`` writes, after the node line, when the node is of another GPU or
CPU model than the run's: the job stopped there (exit status 86) and did no work."""


def _node_line(path: Path) -> tuple[dict | None, bool]:
    """A job log's node line's fields (None without one) and whether the job stopped at the node check (a
    :data:`WRONG_NODE` line among the lines that follow the node line)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = NODE.match(line)
            if not m:
                continue
            try:
                fields = json.loads(m.group(1))
            except json.JSONDecodeError:
                return None, False
            for _, after in zip(range(5), fh):
                if after.startswith(WRONG_NODE):
                    return fields, True
            return fields, False
    return None, False


def nodes(logs: Path) -> dict[str, dict]:
    """``{job log name: its node line's fields}`` (``host``, ``gpu``, ``driver``, ``cpu``, ...) of every job log
    ``sp-*.out`` in ``logs`` that has one (the first; logs of jobs submitted before the line existed have none), but
    the logs of jobs that stopped at the node check (:func:`stopped_at_node_check`), which ran on no node of the
    run."""
    out = {}
    for path in sorted(logs.glob("sp-*.out")):
        fields, stopped = _node_line(path)
        if fields is not None and not stopped:
            out[path.name] = fields
    return out


def stopped_at_node_check(logs: Path) -> list[str]:
    """The job logs ``sp-*.out`` in ``logs`` of jobs that stopped at the node check (``wrong node: ...``, exit status
    86): they did no work, and :func:`nodes` leaves them out."""
    return [path.name for path in sorted(logs.glob("sp-*.out")) if _node_line(path)[1]]


def machines(run: "Run", logs: Path | None) -> dict[str, dict[str, set]]:
    """``{job log name: {"gpu", "cpu", "driver", "host": set of values}}``: what each job's node line
    (:func:`nodes`) and the environments of its records (``env``) name; a job without a GPU in its node line names
    GPU ``none``."""
    out: dict = {}

    def add(job, key, value):
        if value:
            out.setdefault(job, {k: set() for k in ("gpu", "cpu", "driver", "host")})[key].add(str(value).strip())

    for job, n in (nodes(logs) if logs is not None and logs.is_dir() else {}).items():
        add(job, "gpu", n.get("gpu") or "none")
        for key in ("cpu", "driver", "host"):
            add(job, key, n.get(key))
    for r in run.recs:
        env = r.d.get("env") or {}
        job = Path(r.path).name
        for key in ("gpu", "cpu", "driver", "host"):
            add(job, key, env.get(key))
    return out


def verifier_device(d: dict) -> str:
    """A record's verifier device: its ``verifier_device`` field, else the ``--verifier-device`` of the command line
    stored in ``env.argv`` (``v1_bytes.py`` records), else ``cpu``."""
    if d.get("verifier_device"):
        return str(d["verifier_device"]).split(":")[0]
    argv = (d.get("env") or {}).get("argv") or []
    for i, a in enumerate(argv[:-1]):
        if a == "--verifier-device":
            return str(argv[i + 1]).split(":")[0]
    return "cpu"


# ------------------------------------------------------------------------------------------------ selection
class Run:
    """The records of one run and the selection of a table cell's record.

    Every selector keys on the settings that distinguish the package's jobs, so that a record never fills a cell of
    another configuration: the tier, the model, the prompt length, the mode, the blocks (``layers``; None: the full
    model), the verifier device, the label and real or random weights (``real``: None for random weights, else the
    checkpoint); a ``perf.py`` record also by its attention probabilities' scale where an A/B of the improve tier
    times two scales under one label (``prob_max``), by its threads where the cpux tier times one label on several
    (``threads``) and by mode C's plan where it times several plans side by side (``policy``); a re-execution record
    of the cpux tier by its threads and implementation options (:func:`experiments.paper.tables.cpu_point`).  Of
    several matching records (a tier run again) the latest job's
    is used; of several re-execution records of one variant (e.g. two allocator settings), the fastest.  Several
    quality records of one cell whose values differ are a ``WARNING`` line of :attr:`warnings` (:meth:`ppl`).
    """

    SAME = 1e-6
    """The relative difference above which two records of one quality cell are said to differ."""

    def __init__(self, recs: list[Rec], files: list[Rec] | None = None) -> None:
        self.recs = sorted(recs, key=job_order)
        self.files = {Path(r.path).name: r for r in files or []}
        self.by_kind: dict = {}
        for r in self.recs:
            self.by_kind.setdefault(r.kind, []).append(r)
        self.warnings: list[str] = []
        self._warned: set = set()

    def kind(self, kind: str, tier: str | None = None) -> list[Rec]:
        return [r for r in self.by_kind.get(kind, []) if tier is None or r.tier == tier]

    @staticmethod
    def _last(rs):
        return rs[-1] if rs else None

    def file(self, name: str) -> Rec | None:
        return self.files.get(name)

    def environment(self) -> dict:
        """The run's GPU and CPU (the most frequent ones of the records' environments)."""
        from collections import Counter
        gpus, cpus = Counter(), Counter()
        for r in self.recs:
            env = r.d.get("env") or {}
            if env.get("gpu"):
                gpus[env["gpu"]] += 1
            if env.get("cpu"):
                cpus[env["cpu"]] += 1
        return {"gpu": gpus.most_common(1)[0][0] if gpus else None,
                "cpu": cpus.most_common(1)[0][0] if cpus else None}

    # ---------------------------------------------------------------- protocol v0 (perf.py)
    def perf(self, tier, model, seq, mode, *, vdev="cpu", layers=None, label=None, real=None, fs=True,
             stream=None, prob_max=None, threads=None, policy=None) -> Rec | None:
        def ok(r):
            d = r.d
            return (d.get("model") == model and d.get("seq") == seq and d.get("mode") == mode
                    and str(d.get("verifier_device", "cpu")).split(":")[0] == vdev and d.get("layers") == layers
                    and (label is None or d.get("label") == label) and (d.get("real") or None) == real
                    and bool(d.get("fs")) == fs and (stream is None or bool(d.get("stream")) == stream)
                    and (prob_max is None or d.get("prob_max") == prob_max)
                    and (threads is None or d.get("threads") == threads)
                    and (policy is None or d.get("policy") == policy))
        return self._last([r for r in self.kind("perf", tier) if ok(r)])

    # ---------------------------------------------------------------- V1 (v1_bytes.py)
    def v1(self, tier, model, seq, mode, *, vdev="cpu", label=None, real=None, layers=None) -> Rec | None:
        def ok(r):
            d = r.d
            return (not d.get("codec_only") and d.get("model") == model and d.get("seq") == seq
                    and d.get("mode") == mode and verifier_device(d) == vdev
                    and (label is None or d.get("label") == label) and (d.get("real") or None) == real
                    and (layers is None or d.get("layers") == layers))
        return self._last([r for r in self.kind("v1", tier) if ok(r)])

    # ---------------------------------------------------------------- re-execution (reexec.py)
    def reexec(self, tier, model, seq, device, variants, keep=None) -> dict:
        """``{variant: (seconds, record)}``: each variant's full model measured at full depth (no extrapolation),
        from ``reexec.py``'s records and ``llamacpp.py``'s (variants ``llamacpp_<type>``); of several records (e.g. two
        allocator settings, or llama.cpp's configurations), the faster; ``keep(variant, record)``: the records that may
        stand for their variant (default all)."""
        out = {}
        for r in self.kind("reexec", tier) + self.kind("llamacpp", tier):
            d = r.d
            if d.get("model") != model or d.get("seq") != seq or not str(d.get("device", "")).startswith(device):
                continue
            if d.get("variant") not in variants or (keep is not None and not keep(d["variant"], r)):
                continue
            v = (d.get("measured_s") or {}).get(str(d.get("n_layers")))
            if v is None:
                continue
            if d["variant"] not in out or v < out[d["variant"]][0]:
                out[d["variant"]] = (v, r)
        return out

    def reexec_blocks(self, tier, model, seq, device, variant, keep=None) -> tuple[dict, Rec | None]:
        """``(measured_s by block count, record)`` of one variant (the long tier's 2- and 3-block builds; ``keep`` as
        in :meth:`reexec`)."""
        r = self._last([r for r in self.kind("reexec", tier) if r.d.get("model") == model and r.d.get("seq") == seq
                        and str(r.d.get("device", "")).startswith(device) and r.d.get("variant") == variant
                        and (keep is None or keep(variant, r))])
        return ((r.d.get("measured_s") or {}) if r else {}), r

    # ---------------------------------------------------------------- the other scripts
    def serve(self, model, seq, batch, mode, how, layers=None) -> Rec | None:
        return self._last([r for r in self.kind("serve", "serve") if r.d.get("row") == "prove"
                           and (r.d.get("model"), r.d.get("seq"), r.d.get("batch"), r.d.get("mode"),
                                r.d.get("batch_proof"), r.d.get("layers")) == (model, seq, batch, mode, how, layers)])

    def gen(self, model, mode, decode, g, *, prompt=None, real=None, label="gen", tier="gen") -> Rec | None:
        """A ``gen_bench.py`` record of the full ``model`` (``prompt``: the prompt's tokens, None for any)."""
        return self._last([r for r in self.kind("gen", tier)
                           if (r.d.get("model"), r.d.get("mode"), r.d.get("decode"), r.d.get("gen"))
                           == (model, mode, decode, g) and (r.d.get("real") or None) == real
                           and (prompt is None or r.d.get("prompt") == prompt)
                           and r.d.get("layers") is None and (label is None or r.d.get("label") == label)])

    def setup(self, model, amortise) -> Rec | None:
        return self._last([r for r in self.kind("setup", "setup") if r.d.get("model") == model
                           and r.d.get("amortise") == amortise and r.d.get("layers") is None])

    def ppl(self, model, cell, *, split="test", calibration="train", seq=2048, draw=0) -> Rec | None:
        """A quality record of ``ppl_opt.py`` (``cell`` names the graph, its recipe, alpha and probability width:
        ``smooth-certified-a0.6-p11``; ``fp32`` the reference) or ``ppl_llama.py`` (``main-a0.55-glus0.2``;
        ``reference``), on ``split`` with ``calibration`` (``ppl_llama.py``'s records name no calibration: train;
        records without a split are the test split's).  Two jobs that measure one cell (a cell of two studies)
        should give the same values: records of one cell whose perplexity or ratio differ by more than :data:`SAME`
        (relative) are a ``WARNING`` of :attr:`warnings`."""
        def ok(r):
            d = r.d
            return (d.get("model") == model and d.get("cell") == cell and d.get("seq") == seq
                    and d.get("split", "test") == split and d.get("calibration", "train") == calibration
                    and d.get("draw") in (None, draw) and not d.get("tag") and not d.get("smoke"))
        rs = [r for r in self.kind("ppl", "quality") if ok(r)]
        self._same_values(rs, ("ppl", "ratio"), f"{model} {cell} ({split} split, {calibration} calibration, "
                                                f"{seq} tokens, draw {draw})")
        return self._last(rs)

    def _same_values(self, rs: list[Rec], keys: tuple, what: str) -> None:
        """A ``WARNING`` (once per selection) when the records ``rs`` of one selection disagree on a value of
        ``keys`` by more than :data:`SAME` (relative): the latest is used."""
        if len(rs) < 2 or what in self._warned:
            return
        last = rs[-1]
        for r in rs[:-1]:
            for k in keys:
                a, b = r.d.get(k), last.d.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                        and abs(a - b) > self.SAME * max(abs(a), abs(b)):
                    self._warned.add(what)
                    self.warnings.append(f"WARNING quality: two records of {what} differ: {k} {a:.6g} in "
                                         f"{r.source} and {b:.6g} in {last.source}, which is used")
                    return

    def zeroshot(self, model, task, variant, recipe="certified") -> Rec | None:
        """A ``zeroshot.py`` record of ``model`` on ``task``: the fp16 reference (``ref``) or the integer model
        (``int``) of the job whose integer model has ``recipe``, so that a table's fp16 column and its difference
        come from one job."""
        return self._last([r for r in self.kind("zeroshot", "quality")
                           if (r.d.get("model"), r.d.get("task"), r.d.get("variant")) == (model, task, variant)
                           and r.d.get("recipe") == recipe and not r.d.get("tag")])

    def attribution(self, model, group="all", mode=None, *, part="row", graph_cell=ATTRIBUTION_CELL, seq=128,
                    windows=16) -> Rec | None:
        """A record of ``attribution.py``: a row (``group``, ``mode``: ``leave_one_out``, ``add_one``, ...) or, with
        ``part="check"``, the check record, of the graph ``graph_cell`` on ``windows`` windows of ``seq`` tokens."""
        def ok(r):
            d = r.d
            return (d.get("model") == model and d.get("part") == part and d.get("graph_cell") == graph_cell
                    and d.get("seq") == seq and d.get("windows") == windows and not d.get("tag")
                    and (part != "row" or (d.get("group"), d.get("mode")) == (group, mode)))
        return self._last([r for r in self.kind("attribution", "quality") if ok(r)])

    def first(self, kind, tier=None, **match) -> Rec | None:
        """The latest record of ``kind`` (in ``tier``) whose fields equal ``match``."""
        return self._last([r for r in self.kind(kind, tier) if all(r.d.get(k) == v for k, v in match.items())])


def v1_timed(rec: Rec | None, proto: str, verifier: str | None = None):
    """``(prove, verify)`` medians of ``v1_bytes.py``'s timed queries of ``proto`` (``v0``, or ``v1`` with the V1
    verifier ``verifier``, ``optimised`` or ``baseline``; None: the record's main one), each party's Fiat--Shamir
    hashing added to both (as ``perf.py`` counts it); ``(None, None)`` if not timed."""
    tm = (rec.d.get("timed") or {}) if rec else {}
    q = (tm.get("v1_paths") or {}).get(verifier) if proto == "v1" and verifier is not None else tm.get(proto)
    if not q:
        return None, None
    fs = q.get("fs_s") or 0.0
    return q["prove_s"] + fs, q["verify_s"] + fs
