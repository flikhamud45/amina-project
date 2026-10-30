"""The implementation options (:mod:`pvi.options`): flags, scoping, validation, records, and that every option is
read by the library."""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import pytest

from pvi import options

SRC = Path(__file__).resolve().parents[1] / "src" / "pvi"


def test_flags_round_trip_and_default_to_the_options_in_force(set_options):
    ap = argparse.ArgumentParser()
    options.add_arguments(ap)
    args = ap.parse_args([])
    assert options.apply_arguments(args) == options.Options()
    args = ap.parse_args(["--no-native-gemm", "--fused-encoder", "--hash-threads", "3", "--v1-verifier", "baseline"])
    got = options.apply_arguments(args)
    assert (got.native_gemm, got.fused_encoder, got.hash_threads, got.v1_verifier) == (False, True, 3, "baseline")
    assert options.current() == got
    set_options(int8_fold=False)
    ap = argparse.ArgumentParser()
    options.add_arguments(ap)
    assert ap.parse_args([]).int8_fold is False                     # the defaults are the options in force


def test_using_restores_the_previous_options(set_options):
    before = options.current()
    with options.using(native=False, gpu_decode=False) as inside:
        assert options.current() is inside and not inside.native and not inside.gpu_decode
    assert options.current() == before
    with pytest.raises(RuntimeError), options.using(native=False):
        raise RuntimeError
    assert options.current() == before


def test_invalid_values_are_refused():
    with pytest.raises(ValueError):
        options.Options(v1_verifier="fast")
    with pytest.raises(ValueError):
        options.Options(hash_threads=-1)
    with pytest.raises(ValueError):
        options.Options(gkr_memory="small")
    with pytest.raises(ValueError):
        options.Options(gkr_memory_budget=0)
    with pytest.raises(TypeError):
        options.configure(no_such_option=True)


def test_describe_records_the_options_and_the_host():
    d = options.describe("cpu")
    assert d["options"] == dataclasses.asdict(options.current())
    assert {"native", "avx512", "triton"} <= d.keys() and "int8_gemm" not in d


def test_every_option_is_read_by_the_library():
    text = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py") if p.name != "options.py")
    for f in dataclasses.fields(options.Options):
        assert re.search(r"options\.current\(\)\." + f.name + r"\b|\bo\." + f.name + r"\b", text), f.name


def test_the_library_reads_no_switch_from_the_environment():
    """Only the host settings named in the docstring of :mod:`pvi.options` (compiler, native build folder, Python
    headers) and the dataset folders are environment variables."""
    allowed = {"PVI_NATIVE_DIR", "PVI_PY_INCLUDE", "PVI_CACHE", "PVI_CIFAR_ROOT", "PVI_IMAGENET_ROOT"}
    for p in SRC.rglob("*.py"):
        for name in re.findall(r"environ(?:\.get)?\(?\[?\"(\w+)\"", p.read_text(encoding="utf-8")):
            assert name in allowed | {"CXX"}, (p.name, name)
