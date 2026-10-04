"""Tests for the whole-network defence on integer CNN graphs (``pvi.fullcheck``)."""

from __future__ import annotations

import math

import pytest
import torch

from pvi.fullcheck import field as fld
from pvi.fullcheck.commitment import MerkleTree, WeightCommitment, column_leaf
from pvi.fullcheck.graph import CheapOp, MatOp, exact_matmul
from pvi.fullcheck.models import LeNet5, ResNet18
from pvi.fullcheck.protocol import (
    Challenger,
    Prover,
    Verifier,
    commit_graph,
    params_for,
    run_query,
    soundness_bits,
)
from pvi.fullcheck.quantize import dequantize_logits, quantize_input, quantize_model

P = fld.P


# -- field -------------------------------------------------------------------------
def test_ntt_matches_naive_dft():
    n = 16
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, P, (3, n), generator=g, dtype=torch.int64)
    w = fld.root_of_unity(n)
    naive = torch.tensor([[sum(int(a[r, j]) * pow(w, i * j, P) for j in range(n)) % P for i in range(n)]
                          for r in range(3)], dtype=torch.int64)
    assert torch.equal(fld.ntt(a), naive)


def test_rs_code_distance_on_random_rows():
    k, n = 20, 64
    g = torch.Generator().manual_seed(1)
    a = torch.randint(0, P, (1, k), generator=g, dtype=torch.int64)
    b = a.clone()
    b[0, 3] = (b[0, 3] + 1) % P
    diff = (fld.rs_encode(a, n) != fld.rs_encode(b, n)).sum().item()
    assert diff >= n - k + 1


def test_modular_products_are_exact():
    g = torch.Generator().manual_seed(2)
    small = torch.randint(-127, 128, (7, 20000), generator=g, dtype=torch.int64)
    field = torch.randint(0, P, (20000, 5), generator=g, dtype=torch.int64)
    ref = torch.tensor([[sum(int(small[i, k]) * int(field[k, j]) for k in range(20000)) % P
                         for j in range(5)] for i in range(2)], dtype=torch.int64)
    assert torch.equal(fld.small_matmul_mod(small, field)[:2], ref)
    left = torch.randint(0, P, (2, 3000), generator=g, dtype=torch.int64)
    right = torch.randint(0, P, (3000, 3), generator=g, dtype=torch.int64)
    ref2 = torch.tensor([[sum(int(left[i, k]) * int(right[k, j]) for k in range(3000)) % P
                          for j in range(3)] for i in range(2)], dtype=torch.int64)
    assert torch.equal(fld.field_matmul_mod(left, right), ref2)


def test_exact_matmul_matches_int64_reference():
    g = torch.Generator().manual_seed(3)
    w = torch.randint(-127, 128, (33, 5000), generator=g, dtype=torch.int64)
    x = torch.randint(-127, 128, (5000, 17), generator=g, dtype=torch.int64)
    assert torch.equal(exact_matmul(w.to(torch.int8), x.to(torch.float32)), w @ x)


# -- commitment --------------------------------------------------------------------
def test_weight_commitment_open_matches_full_encoding_and_fold():
    g = torch.Generator().manual_seed(4)
    w = torch.randint(-127, 128, (9, 30), generator=g, dtype=torch.int64).to(torch.int8)
    b = torch.randint(-(1 << 20), 1 << 20, (9,), generator=g, dtype=torch.int64)
    com = WeightCommitment.build(b"t", w, b, rate=4)
    a = torch.cat([w.to(torch.int64), b[:, None]], 1)
    enc = fld.rs_encode(a, com.n_points)
    cols = torch.tensor([0, 5, 77, com.n_points - 1])
    opened, proof = com.open(cols)
    assert torch.equal(opened, enc[:, cols])
    from pvi.fullcheck.commitment import verify_multiproof

    leaves = {c: column_leaf(b"t", c, opened[:, j].numpy()) for j, c in enumerate(cols.tolist())}
    assert verify_multiproof(com.tree.root, com.tree.depth, leaves, proof)
    chi = torch.randint(0, P, (2, 9), generator=g, dtype=torch.int64)
    ref = torch.tensor([[sum(int(chi[r, i]) * int(a[i, j]) for i in range(9)) % P for j in range(31)]
                        for r in range(2)])
    assert torch.equal(com.fold(chi), ref)


# -- whole protocol on quantised CNNs --------------------------------------------------
def _tiny_graph(kind: str):
    torch.manual_seed(0)
    if kind == "lenet":
        model, shape = LeNet5(10), (1, 28, 28)
    else:
        model, shape = ResNet18(10), (3, 32, 32)
    model.eval()
    for m in model.modules():  # give BN non-trivial statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    x_cal = torch.randn(8, *shape)
    return model, quantize_model(model, x_cal), shape


@pytest.fixture(scope="module", params=["lenet", "resnet"])
def setup(request):
    model, graph, shape = _tiny_graph(request.param)
    params = params_for(20, len(graph.mat_ops))
    commitments = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=commitments)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in commitments.items()})
    x = quantize_input(graph, torch.randn(1, *shape))
    return model, graph, prover, verifier, x


def test_public_graph_holds_no_weights(setup):
    _, _, _, verifier, _ = setup
    assert all(op.weight is None and op.bias is None for op in verifier.graph.mat_ops)


def test_honest_prover_is_accepted(setup):
    _, _, prover, verifier, x = setup
    for seed in range(3):
        assert run_query(prover, verifier, x, seed=seed)["accepted"]


def test_integer_model_tracks_float_model(setup):
    model, graph, _, _, _ = setup
    # accuracy agreement is measured on real data in the benchmark; here the
    # integer logits must at least track the float ones on random inputs.
    shape = (1, 28, 28) if isinstance(model, LeNet5) else (3, 32, 32)
    xf = torch.randn(16, *shape)
    _, claims = graph.forward(quantize_input(graph, xf))
    logits = dequantize_logits(graph, claims[graph.mat_ops[-1].name]).float()
    ref = model.float()(xf)
    corr = torch.corrcoef(torch.stack([logits.flatten(), ref.flatten()]))[0, 1]
    assert corr > 0.9


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_single_value_tamper_is_rejected(setup, where):
    _, graph, prover, verifier, x = setup
    names = [op.name for op in graph.mat_ops]
    target = {"first": names[0], "middle": names[len(names) // 2], "last": names[-1]}[where]

    def tamper(op, z):
        if op.name == target:
            z = z.clone()
            z.view(-1)[z.numel() // 3] += 1
        return z

    assert not run_query(prover, verifier, x, seed=7, forward_kwargs={"tamper": tamper})["accepted"]


def _forge_logit(graph, state):
    final = graph.mat_ops[-1]

    def tamper(op, z):
        if op.name == final.name:
            z = z.clone()
            k = int(z[:, 0].argmin())
            new = int(z[:, 0].max()) + 1000
            state["k"], state["delta"] = k, new - int(z[k, 0])
            z[k, 0] = new
        return z

    return tamper


def test_forged_fold_passes_freivalds_and_is_caught_by_the_columns(setup):
    """A wrong logit with a u' solved to satisfy Freivalds: only the RS columns can reject it."""
    _, graph, prover, verifier, x = setup
    final = graph.mat_ops[-1]
    real = prover.fold
    state: dict = {}

    def forged_fold(chis):
        us = real(chis)
        u = us[final.name].clone()
        for i in range(u.shape[0]):   # bias coordinate: multiplies the constant 1
            u[i, final.n_in] = (int(u[i, final.n_in]) + int(chis[final.name][i, state["k"]]) * state["delta"]) % P
        us[final.name] = u
        return us

    prover.fold = forged_fold
    try:
        out = run_query(prover, verifier, x, seed=3, forward_kwargs={"tamper": _forge_logit(graph, state)})
    finally:
        prover.fold = real
    assert not out["accepted"] and out["rejected_at"] == "columns_code"


def test_forged_column_passes_the_code_check_and_is_caught_by_merkle(setup):
    """An opened column shifted by a vector in ker(chi): only the Merkle path can reject it."""
    _, graph, prover, verifier, x = setup
    victim = graph.mat_ops[0].name
    real_fold, real_open = prover.fold, prover.open
    state: dict = {}

    def capture(chis):
        state["chi"] = chis[victim]
        return real_fold(chis)

    def forged_open(cols):
        out = real_open(cols)
        c, pth = out[victim]
        chi = state["chi"]
        r = chi.shape[0]
        # solve chi[:, :r] d = -chi[:, r] mod P for d, then delta = (d, 1, 0, ...)
        a = [[int(v) % P for v in chi[i, :r].tolist()] + [(-int(chi[i, r])) % P] for i in range(r)]
        for col in range(r):
            piv = next(i for i in range(col, r) if a[i][col])
            a[col], a[piv] = a[piv], a[col]
            inv = pow(a[col][col], P - 2, P)
            a[col] = [(v * inv) % P for v in a[col]]
            for i in range(r):
                if i != col and a[i][col]:
                    f = a[i][col]
                    a[i] = [(vi - f * vc) % P for vi, vc in zip(a[i], a[col])]
        delta = torch.zeros(chi.shape[1], dtype=torch.int64)
        delta[:r] = torch.tensor([a[i][r] for i in range(r)])
        delta[r] = 1
        assert torch.equal(fld.field_matmul_mod(chi, delta[:, None]), torch.zeros(r, 1, dtype=torch.int64))
        c = c.clone()
        c[:, 0] = (c[:, 0] + delta) % P
        out[victim] = (c, pth)
        return out

    prover.fold, prover.open = capture, forged_open
    try:
        out = run_query(prover, verifier, x, seed=5)
    finally:
        prover.fold, prover.open = real_fold, real_open
    assert not out["accepted"] and out["rejected_at"] == "columns_merkle"


def test_claims_are_pinned_down_as_integers(setup):
    """2 * Z_BOUND < p, the range check survives INT64_MIN, and an aliased claim is rejected."""
    from pvi.fullcheck.protocol import Z_BOUND

    assert 2 * Z_BOUND < P
    _, graph, prover, verifier, x = setup
    first = graph.mat_ops[0].name

    def put(value):
        def tamper(op, z):
            if op.name == first:
                z = z.clone()
                z.view(-1)[0] = value(int(z.view(-1)[0]))
            return z
        return tamper

    for value in (lambda v: -(1 << 63), lambda v: v + P, lambda v: v - P):
        out = run_query(prover, verifier, x, seed=1, forward_kwargs={"tamper": put(value)})
        assert not out["accepted"] and out["rejected_at"] == "range_or_shape"


def test_commitment_refuses_a_model_whose_honest_claims_could_reach_the_bound():
    from pvi.fullcheck.protocol import Z_BOUND

    w = torch.ones(4, 8, dtype=torch.int8)
    b = torch.tensor([0, 0, Z_BOUND, 0])
    graph_op = MatOp("fc", ("x",), "fc.z", weight=w, bias=b)
    from pvi.fullcheck.graph import IntGraph

    with pytest.raises(ValueError):
        commit_graph(IntGraph([graph_op], "x", "fc.z"), 4)


def test_challenger_columns_are_distinct_and_cover_the_range():
    ch = Challenger()
    seen = set()
    for i in range(200):
        idx = ch.columns(f"op{i}", 64, 10)
        assert len(set(idx.tolist())) == 10 and idx.min() >= 0 and idx.max() < 64
        seen |= set(idx.tolist())
    assert seen == set(range(64))
    chi = ch.folding("op", 1000, 3)
    assert chi.shape == (3, 1000) and int(chi.min()) >= 0 and int(chi.max()) < P


def test_fiat_shamir_challenges_depend_on_the_statement(setup):
    """With FS, the same claims under a different input give different challenges."""
    from pvi.fullcheck.protocol import _absorb_statement, _tensor_blob

    _, graph, prover, verifier, x = setup
    fs = params_for(20, len(graph.mat_ops), fiat_shamir=True)
    v = Verifier(verifier.graph, fs, "C", publics=verifier.publics)
    claims = prover.claims(x)
    chis = []
    for xx in (x, x.clone().flip(-1)):
        ch = Challenger(fiat_shamir=True)
        _absorb_statement(ch, v, xx)
        for k in sorted(claims):
            ch.absorb(b"claim/" + k.encode(), _tensor_blob(claims[k]))
        chis.append(ch.folding(graph.mat_ops[0].name, 4, 1))
    assert not torch.equal(chis[0], chis[1])


def test_model_substitution_is_rejected(setup):
    _, graph, prover, verifier, x = setup
    other = {}
    for op in graph.mat_ops:
        clone = MatOp(op.name, op.inputs, op.output, weight=op.weight.clone(), bias=op.bias.clone(),
                      layout=op.layout, conv=op.conv)
        clone.weight.view(-1)[0] = -clone.weight.view(-1)[0] if clone.weight.view(-1)[0] != 0 else 1
        other[op.name] = clone
    out = run_query(prover, verifier, x, seed=1, forward_kwargs={"weights_override": other})
    assert not out["accepted"]


def test_mode_k_and_kpre_accept_honest_and_reject_tamper(setup):
    _, graph, prover, _, x = setup
    params = params_for(20, len(graph.mat_ops))
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    for mode in ("K", "Kpre"):
        v = Verifier(graph.public(), params, mode, weights=weights)
        if mode == "Kpre":
            v.precompute(Challenger(seed=11))
        assert run_query(prover, v, x, seed=2)["accepted"]

        def tamper(op, z):
            z = z.clone()
            z.view(-1)[0] += 5
            return z

        assert not run_query(prover, v, x, seed=2, forward_kwargs={"tamper": tamper})["accepted"]


def test_fiat_shamir_variant_accepts_honest(setup):
    _, graph, prover, verifier, x = setup
    fs = params_for(20, len(graph.mat_ops), fiat_shamir=True)
    v = Verifier(verifier.graph, fs, "C", publics=verifier.publics)
    # columns grew with the grinding margin; commitments are rate-bound, not t-bound
    assert run_query(prover, v, x)["accepted"]


def test_params_meet_their_target():
    for lam in (40, 80, 128):
        p = params_for(lam, 20)
        bits = soundness_bits(p, [(4609, 4 * 8192)] * 20)
        assert bits >= lam
        assert p.reps == math.ceil((lam + math.log2(40)) / fld.LOG2_P)


# -- integer transformers -------------------------------------------------------------
from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder, decoder_param_count  # noqa: E402

_TINY = {
    "gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64),
    "opt": DecoderConfig("tiny-opt", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=64),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                           pos="rope", bias=False, tied=False),
    "qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                          pos="rope", rope_theta=1e6, bias=False, qk_norm=True),
}


@pytest.mark.parametrize("family", list(_TINY))
def test_decoder_protocol_honest_and_tampered(family):
    cfg = _TINY[family]
    graph = build_decoder(cfg, calib_tokens=12, seed=3)
    tokens = torch.randint(0, cfg.vocab, (1, 12), generator=torch.Generator().manual_seed(5))
    env, claims = graph.forward(tokens)
    # activations are neither saturated nor dead
    logits = claims[graph.mat_ops[-1].name]
    assert logits.double().std() > 10
    params = params_for(20, len(graph.mat_ops))
    coms = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=coms)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    assert run_query(prover, verifier, tokens, seed=0)["accepted"]
    for target in (graph.mat_ops[0].name, graph.mat_ops[len(graph.mat_ops) // 2].name):
        def tamper(op, z, target=target):
            if op.name == target:
                z = z.clone()
                z.view(-1)[7] -= 3
            return z
        assert not run_query(prover, verifier, tokens, seed=1, forward_kwargs={"tamper": tamper})["accepted"]


def test_real_configs_have_published_parameter_counts():
    # weight-product parameters, within 6% of the published (rounded) sizes
    for name, published in [("gpt2", 124e6), ("llama2-7b", 6.74e9), ("llama2-13b", 13.0e9),
                            ("opt-1.3b", 1.3e9), ("qwen3-4b", 4.0e9)]:
        n = decoder_param_count(CONFIGS[name])
        assert abs(n - published) / published < 0.06, (name, n)


# -- the sampling baseline (Anchuri et al.'s RandPathTest) on the same graphs --------------
from pvi.fullcheck.sampling import (TraceCommitment, neuron_tensors, paths_for,  # noqa: E402
                                    sample_path, shared_path_bytes, visit_probabilities)


@pytest.mark.parametrize("kind", ["lenet", "resnet"])
def test_sampling_baseline_accepts_honest_paths(kind):
    import numpy as np

    _, graph, shape = _tiny_graph(kind)
    env, _ = graph.forward(quantize_input(graph, torch.randn(1, *shape)))
    tc = TraceCommitment(graph, env)
    rng = np.random.default_rng(0)
    paths = [sample_path(tc, env, rng) for _ in range(20)]
    assert all(p["ok"] for p in paths)
    assert all(p["data_bytes"] > 0 for p in paths)


def test_visit_mass_is_a_distribution_on_a_chain():
    _, graph, shape = _tiny_graph("lenet")
    env, _ = graph.forward(quantize_input(graph, torch.randn(1, *shape)))
    mass = visit_probabilities(graph, env)
    for name in neuron_tensors(graph):   # every path crosses every neuron layer exactly once
        assert abs(float(mass[name].sum()) - 1.0) < 1e-9


def test_single_node_tamper_detected_at_the_visit_rate():
    import numpy as np

    _, graph, shape = _tiny_graph("lenet")
    env, _ = graph.forward(quantize_input(graph, torch.randn(1, *shape)))
    final = graph.mat_ops[-1]
    src = final.inputs[0]
    j = 5
    env = dict(env)
    env[src] = env[src].clone()
    env[src][0, j] += 9                       # the attack: one node...
    env[final.output] = final.fold(final.compute(env[src]), env[src])  # ...honestly propagated
    p = float(visit_probabilities(graph, env)[src][0, j])
    assert abs(p - 1 / final.n_in) < 1e-12
    tc = TraceCommitment(graph, env)
    rng = np.random.default_rng(0)
    n = 4000
    caught = sum(not sample_path(tc, env, rng)["ok"] for _ in range(n))
    assert abs(caught / n - p) < 4 * math.sqrt(p * (1 - p) / n)


def _tampered_env(graph, env, tensor, flat, delta):
    """Tamper one neuron and re-propagate every later op honestly (the attack)."""
    env = dict(env)
    env[tensor] = env[tensor].clone()
    env[tensor].view(-1)[flat] += delta
    seen = False
    for op in graph.ops:
        if op.output == tensor:
            seen = True
            continue
        if not seen:
            continue
        if isinstance(op, MatOp):
            xin = env[op.inputs[0]]
            env[op.output] = op.fold(op.compute(xin), xin)
        else:
            env[op.output] = op.fn(*[env[n] for n in op.inputs])
    return env


def test_single_neuron_tamper_inside_a_residual_block_matches_the_visit_rate():
    import numpy as np

    _, graph, shape = _tiny_graph("resnet")
    env, _ = graph.forward(quantize_input(graph, torch.randn(1, *shape)))
    adds = [op.output for op in graph.ops if isinstance(op, CheapOp) and op.note == "residual add+relu"]
    target = adds[len(adds) // 2]
    flat = env[target].numel() // 3
    bad = _tampered_env(graph, env, target, flat, 7)
    p = float(visit_probabilities(graph, bad)[target].view(-1)[flat])
    tc = TraceCommitment(graph, bad)
    rng = np.random.default_rng(2)
    n = 20000
    caught = sum(not sample_path(tc, bad, rng)["ok"] for _ in range(n))
    assert p > 0
    assert abs(caught / n - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-4


def test_a_spliced_dense_input_is_caught_through_its_producer():
    """Flatten is a view: replacing the whole flattened vector means changing the
    pooled tensor it views, whose every node is on every path."""
    import numpy as np

    _, graph, shape = _tiny_graph("lenet")
    env, _ = graph.forward(quantize_input(graph, torch.randn(1, *shape)))
    flat_op = next(op for op in graph.ops if isinstance(op, CheapOp) and op.note == "flatten")
    pooled = flat_op.inputs[0]
    bad = dict(env)
    bad[pooled] = torch.randint(0, 127, env[pooled].shape)
    bad = _tampered_env(graph, bad, pooled, 0, 0)
    tc = TraceCommitment(graph, bad)
    rng = np.random.default_rng(0)
    assert all(not sample_path(tc, bad, rng)["ok"] for _ in range(50))
    assert flat_op.output not in tc.tensors


def test_shared_openings_never_exceed_opening_everything():
    _, graph, shape = _tiny_graph("lenet")
    env, _ = graph.forward(quantize_input(graph, torch.randn(1, *shape)))
    tc = TraceCommitment(graph, env)
    out = shared_path_bytes(tc, env, [1, 10, 100, 10000])
    assert out[1] <= out[10] <= out[100] <= out[10000] <= tc.open_all_bytes
    assert out[10000] == tc.open_all_bytes


def test_paths_for_matches_closed_form():
    assert paths_for(40, 1 / 512) == math.ceil(40 / -math.log2(1 - 1 / 512))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kind", ["lenet", "resnet"])
def test_gpu_prover_matches_cpu_verifier_bit_for_bit(kind):
    _, graph, shape = _tiny_graph(kind)
    x = quantize_input(graph, torch.randn(2, *shape))
    _, cpu_claims = graph.forward(x)
    _, gpu_claims = graph.forward(x.cuda())
    assert all(torch.equal(cpu_claims[k], gpu_claims[k].cpu()) for k in cpu_claims)
    params = params_for(20, len(graph.mat_ops))
    coms = commit_graph(graph, params.rate, device="cuda")
    prover = Prover(graph, device="cuda", commitments=coms)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()})
    assert run_query(prover, verifier, x, seed=4)["accepted"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_decoder_matches_cpu():
    cfg = _TINY["qwen"]
    graph = build_decoder(cfg, calib_tokens=12, seed=3)
    tokens = torch.randint(0, cfg.vocab, (1, 12), generator=torch.Generator().manual_seed(5))
    _, a = graph.forward(tokens)
    _, b = graph.forward(tokens.cuda())
    assert all(torch.equal(a[k], b[k].cpu()) for k in a)


# -- decoder shapes (used by aggregate.py and the benchmark) ---------------------------------
from pvi.fullcheck.analytic import decoder_shapes  # noqa: E402


@pytest.mark.parametrize("family", ["gpt", "opt", "llama", "qwen"])
def test_decoder_shapes_match_built_graph(family):
    cfg = _TINY[family]
    graph = build_decoder(cfg, calib_tokens=12, seed=3)
    built = [(op.n_rows, op.row_length, op.layout, op.has_bias) for op in graph.mat_ops]
    formula = [(s.n_rows, s.row_length, s.layout, s.has_bias) for s in decoder_shapes(cfg)]
    assert built == formula


def test_direct_codeword_evaluation_equals_full_encoding():
    from pvi.fullcheck.commitment import vandermonde_columns

    g = torch.Generator().manual_seed(9)
    u = torch.randint(0, P, (3, 700), generator=g, dtype=torch.int64)
    n = 4 * 1024
    idx = torch.tensor([0, 1, 17, 1023, 4095])
    assert torch.equal(fld.field_matmul_mod(u, vandermonde_columns(n, 700, idx)), fld.rs_encode(u, n)[:, idx])


def test_multiproof_matches_individual_paths_and_rejects_forgery():
    from pvi.fullcheck.commitment import multiproof, multiproof_size, verify_multiproof

    import random
    rnd = random.Random(0)
    leaves = [bytes([i % 256, i // 256]) * 16 for i in range(64)]
    tree = MerkleTree(leaves)
    for _ in range(50):
        idx = sorted(rnd.sample(range(64), rnd.randint(1, 20)))
        proof = multiproof(tree, idx)
        assert len(proof) == multiproof_size(idx, tree.depth) <= len(idx) * tree.depth
        assert verify_multiproof(tree.root, tree.depth, {i: leaves[i] for i in idx}, proof)
        bad = {i: leaves[i] for i in idx}
        bad[idx[0]] = b"x" * 32
        assert not verify_multiproof(tree.root, tree.depth, bad, proof)
        if proof:
            assert not verify_multiproof(tree.root, tree.depth, {i: leaves[i] for i in idx}, proof[:-1])
    assert multiproof_size(range(64), tree.depth) == 0


def test_expected_multiproof_size_matches_monte_carlo():
    import random

    from pvi.fullcheck.analytic import expected_multiproof_nodes
    from pvi.fullcheck.commitment import multiproof_size

    rnd = random.Random(3)
    for n, t in ((64, 5), (512, 67), (4096, 23), (128, 100)):
        sims = [multiproof_size(rnd.sample(range(n), t), n.bit_length() - 1) for _ in range(3000)]
        mean = sum(sims) / len(sims)
        assert abs(mean - expected_multiproof_nodes(n, t)) < 0.02 * mean + 0.1


@pytest.mark.parametrize("m_s", [1 << 20, 1000])          # int32 scores, and the int64 fallback
def test_grouped_attention_equals_all_heads_at_once(m_s, monkeypatch):
    import pvi.fullcheck.transformer as tr

    g = torch.Generator().manual_seed(4)
    b, t, h, dh = 1, 40, 6, 8
    q = torch.randint(-127, 128, (b, t, h * dh), generator=g)
    k = torch.randint(-127, 128, (b, t, h * dh), generator=g)
    v = torch.randint(-127, 128, (b, t, h * dh), generator=g)
    m_o = 1 << 22
    heads = []
    attention_heads = tr._attention_heads
    monkeypatch.setattr(tr, "_attention_heads",
                        lambda q, *args: heads.append(q.shape[1]) or attention_heads(q, *args))
    whole = tr._attention(q, k, v, m_s, m_o, h, h, dh)
    width = 4 if tr._int32_scores(m_s, dh, t) else 8             # bytes per score on the path taken
    monkeypatch.setattr(tr, "ATTN_BYTES", width * b * t * t * 4)  # force groups of 4 heads (6 = 4 + 2)
    assert torch.equal(tr._attention(q, k, v, m_s, m_o, h, h, dh), whole)
    assert heads == [6, 4, 2]
