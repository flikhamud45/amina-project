"""V1's query end to end (``pvi.fullcheck.cut_protocol``; ``V1_SPEC.md`` tests T4 and T5).

T4: honest queries are accepted in modes C (one tree per op, and commitment plans with col matrices and lookup
tables) and Kpre, interactive and under Fiat--Shamir, on GPT-, Llama- (pruned), OPT- and Qwen-shaped decoders and a
generation graph.  T5: every way a prover can deviate at one message is rejected with its label: the
trace-tampering attack (``cut_final``), fake or malformed exceptions, bad multiplicities, a tampered GKR, a wrong
fold, a wrong clear claim.
"""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import cut as cutmod
from pvi.fullcheck import cut_protocol as cp
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.field import P
from pvi.fullcheck.transformer import build_decoder, greedy_tokens, with_lm_positions
from test_plans import _TINY

_CACHE: dict = {}


def _setup(kind, mode, policy=None, fs=False, pruned=False, gen=False):
    key = (kind, mode, policy, fs, pruned, gen)
    if key in _CACHE:
        return _CACHE[key]
    g = build_decoder(_TINY[kind], calib_tokens=8, seed=1, prune_last=pruned)
    if gen:
        g = with_lm_positions(g, 3)
    if mode == "C":
        coms = proto.commit_graph(g, 4, policy=policy or "paper")
        params = proto.params_for(40, len(g.mat_ops), fiat_shamir=fs, plan=coms.plan, cut=True)
        v = proto.Verifier(g.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                           tables=coms.table_publics)
        out = g, proto.Prover(g, commitments=coms), v
    else:
        params = proto.params_for(40, len(g.mat_ops), fiat_shamir=fs, cut=True)
        v = proto.Verifier(g.public(), params, mode, weights={op.name: (op.weight, op.bias) for op in g.mat_ops})
        v.precompute(proto.Challenger())
        out = g, proto.Prover(g), v
    _CACHE[key] = out
    return out


def _x(n=9, seed=2):
    return torch.randint(0, 97, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("kind,mode,policy,fs,pruned", [
    ("gpt", "C", None, False, False), ("gpt", "C", None, True, False), ("llama", "C", "cnn16c", True, True),
    ("opt", "C", "tightc", False, False), ("qwen", "C", "auto", True, False), ("gpt", "Kpre", None, True, False),
    ("llama", "Kpre", None, False, True), ("opt", "Kpre", None, True, True)])
def test_t4_honest_queries_are_accepted(kind, mode, policy, fs, pruned):
    g, prover, v = _setup(kind, mode, policy, fs, pruned)
    out = cp.run_cut_query(prover, v, _x(), seed=None if fs else 5)
    assert out["accepted"], out["rejected_at"]
    assert out["cut"]["n_ops"] >= 2 * _TINY[kind].n_layers and out["bytes"]["cut_gkr"] > 0
    assert v._cut_plan is None                         # the per-query state is cleared
    # a v0 query on the same verifier is unaffected
    assert proto.run_query(prover, v, _x(), wire=True)["accepted"]


def test_t4_a_generation_graph_and_another_length():
    g, prover, v = _setup("llama", "C", "cnn16c", True, True, gen=True)
    base = build_decoder(_TINY["llama"], calib_tokens=8, seed=1, prune_last=True)
    x = greedy_tokens(base, _x(10, seed=4), 2)                  # a prompt and its 2 greedy tokens
    out = cp.run_cut_query(prover, v, x)
    assert out["accepted"], out["rejected_at"]
    x[0, -1] = (x[0, -1] + 1) % 97                               # another token: the token rule still holds
    assert cp.run_cut_query(prover, v, x)["rejected_at"] in (None, "token")
    g, prover, v = _setup("gpt", "Kpre", None, True)
    assert cp.run_cut_query(prover, v, _x(33, seed=6))["accepted"]


def test_mode_k_and_streaming_are_not_supported():
    g, prover, v = _setup("gpt", "Kpre", None, True)
    v2 = proto.Verifier(g.public(), v.params, "K", weights={op.name: (op.weight, op.bias) for op in g.mat_ops})
    with pytest.raises(NotImplementedError):
        cp.run_cut_query(prover, v2, _x())


# ------------------------------------------------------------------ T5: cheats
def _first_cut(v, x, kind="requant"):
    plan = cp.cut_plan(v.graph, v.claim_columns(x), x.shape[1], v.publics if v.groups else None, v.params.cut_lmax)
    return plan, next(o for o in plan.ops if o.kind == kind)


@pytest.mark.parametrize("mode,label", [("C", "cut_final"), ("Kpre", "kpre_y")])
def test_the_trace_tampering_attack_is_caught(mode, label):
    # mode C: the GKR proves the prover's own (in-window) leaves, the final check compares them with A [X; 1]
    # through u; Kpre: the prover's y = Z' e already disagrees with the secret rows
    g, prover, v = _setup("gpt", mode, None, True)
    x = _x()
    plan, o = _first_cut(v, x)

    def tamper(op, z):
        if op.name != o.name:
            return z
        z = z.clone()
        z[3, 4] += 3 * o.widths[1]                     # another requantised value, consistent downstream
        return z

    out = cp.run_cut_query(prover, v, x, forward_kwargs={"tamper": tamper})
    assert out["rejected_at"] == label


def _cheat_exception(o, offs: dict, honest_leaf: bool):
    def edit(values, witness, exc, mult):
        s = values[o.name]
        d, w = witness[o.name]
        flat = s.reshape(-1)
        k = int(torch.nonzero(flat.abs() < 100)[0])
        a = int(flat[k])
        lower, width = cutmod.windows(torch.tensor([a]), o.mult, o.sh)
        z_true = int(lower) + int(d.reshape(-1)[k])
        z = int(lower) if z_true != int(lower) else int(lower) + 1
        idx, ez = exc[o.name]
        exc[o.name] = (torch.cat([idx, torch.tensor([k])]).sort().values,
                       torch.cat([ez, torch.tensor([z])])[torch.cat([idx, torch.tensor([k])]).argsort()])
        old = offs[int(w.reshape(-1)[k])] + int(d.reshape(-1)[k])
        if honest_leaf:                                # the leaf of the listed z: delta = z_true - z, off the table
            d.reshape(-1)[k], w.reshape(-1)[k] = z_true - z, 1
        else:                                          # an in-table leaf the public windows do not give
            d.reshape(-1)[k], w.reshape(-1)[k] = 0, 1
            mult[old] -= 1
            mult[offs[1]] += 1
    return edit


@pytest.mark.parametrize("honest_leaf,label", [(True, "cut_logup"), (False, "cut_final")])
def test_a_fake_exception_is_caught(honest_leaf, label):
    g, prover, v = _setup("gpt", "C", None, True)
    x = _x()
    plan, o = _first_cut(v, x)
    out = cp.run_cut_query(prover, v, x, cheat={"witness": _cheat_exception(o, plan.table()[1], honest_leaf)})
    assert out["rejected_at"] == label


def test_malformed_exceptions_are_rejected():
    g, prover, v = _setup("gpt", "C", None, True)
    x = _x()
    plan, o = _first_cut(v, x)
    res = next(op for op in plan.ops if op.kind == "residual")

    def at(name, k, z_of):
        def edit(values, witness, exc, mult):
            idx, ez = exc[name]
            exc[name] = (torch.cat([idx, torch.tensor([k])]), torch.cat([ez, torch.tensor([z_of(values[name])])]))
        return edit

    lo = lambda s: int(cutmod.lo(s.reshape(-1)[:1], o.mult, o.sh))                      # noqa: E731
    cheats = [at(res.name, 0, lambda s: 0),                                              # a residual op
              at(o.name, 0, lambda s: lo(s) + 40 * o.widths[1]),                         # requant(z) != a
              at(o.name, o.n_rows * plan.T, lambda s: 0),                                # outside the op
              at(o.name, 0, lambda s: 1 << 30)]                                          # |z| >= 2^29
    for cheat in cheats:
        assert cp.run_cut_query(prover, v, x, cheat={"witness": cheat})["rejected_at"] == "cut_exception"

    def twice(values, witness, exc, mult):                                               # not increasing
        a = lo(values[o.name])
        exc[o.name] = (torch.tensor([0, 0]), torch.tensor([a, a]))
    assert cp.run_cut_query(prover, v, x, cheat={"witness": twice})["rejected_at"] == "cut_exception"


def test_multiplicity_cheats():
    g, prover, v = _setup("gpt", "Kpre", None, True)
    x = _x()

    def off_by_one(values, witness, exc, mult):
        mult[0] += 1
    assert cp.run_cut_query(prover, v, x, cheat={"witness": off_by_one})["rejected_at"] == "cut_multiplicities"

    def plus_p(values, witness, exc, mult):           # m_t >= p: the same fractions, but not an integer count
        mult[int(torch.nonzero(mult)[0])] += P
    assert cp.run_cut_query(prover, v, x, cheat={"witness": plus_p})["rejected_at"] == "cut_multiplicities"

    def moved(values, witness, exc, mult):
        i = int(torch.nonzero(mult)[0])
        mult[i] -= 1
        mult[i + 1] += 1
    assert cp.run_cut_query(prover, v, x, cheat={"witness": moved})["rejected_at"] == "cut_logup"


def test_gkr_and_fold_cheats():
    g, prover, v = _setup("gpt", "Kpre", None, True)
    x = _x()

    def flip(blobs):
        b = bytearray(blobs[0])
        b[40] ^= 1
        return [bytes(b)] + blobs[1:]
    assert cp.run_cut_query(prover, v, x, cheat={"gkr": flip})["rejected_at"] in ("cut_gkr", "cut_final")
    assert cp.run_cut_query(prover, v, x, cheat={"gkr": lambda bl: [bl[0][:-3]] + bl[1:]})["rejected_at"] == "cut_gkr"

    def fold(parts):
        parts = [t.clone() for t in parts]
        parts[0][0, 0] = (parts[0][0, 0] + 1) % P
        return parts
    assert cp.run_cut_query(prover, v, x, cheat={"fold": fold})["rejected_at"] == "kpre_y"
    g, prover, v = _setup("gpt", "C", None, True)
    plan, _ = _first_cut(v, x)
    n_clear = len([op for op in v._row_ops() if op.name not in plan.names()])

    def fold_cut(parts):                               # a cut row op's u (after the clear ops' u)
        parts = [t.clone() for t in parts]
        parts[n_clear][3, 2] = (parts[n_clear][3, 2] + 1) % P
        return parts
    assert cp.run_cut_query(prover, v, x, cheat={"fold": fold_cut})["rejected_at"] == "cut_final"


def test_a_wrong_clear_claim_fails_freivalds():
    g, prover, v = _setup("gpt", "C", None, True)
    x = _x()
    head = next(op for op in g.mat_ops if op.output == g.output_name)

    def edit(values, witness, exc, mult):
        values[head.name] = values[head.name].clone()
        values[head.name][0, 0] += 1
    assert cp.run_cut_query(prover, v, x, cheat={"witness": edit})["rejected_at"] == "freivalds"


@pytest.mark.parametrize("kind,mode,policy,fs,pruned", [
    ("gpt", "C", None, True, False), ("llama", "C", "cnn16c", True, True), ("opt", "Kpre", None, True, True)])
def test_t7_the_byte_accounting_is_that_of_a_query(kind, mode, policy, fs, pruned):
    g, prover, v = _setup(kind, mode, policy, fs, pruned)
    real = cp.run_cut_query(prover, v, _x())
    est = cp.cut_proof_bytes(prover, v, _x(), seed=3)
    for k in ("claims", "cut_exc", "cut_mult", "cut_gkr", "u", "columns"):
        assert est[k] == real["bytes"][k], k
    assert est["n_leaves"] == real["cut"]["n_leaves"] and v._cut_plan is None
    if mode == "C":                                    # other columns: about the same multiproofs
        assert 0.5 < est["paths"] / real["bytes"]["paths"] < 2
