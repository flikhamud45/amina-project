"""Last-block pruning: ``build_decoder(..., prune_last=True)`` builds the last block at the last position
(k and v at every position, which its one query row attends to), with the logits of the whole block.

Its pieces: attention of the queries of the last positions (``transformer._attention``,
``_attention_heads``: the last rows of the causal mask), RoPE at an offset, and the shapes of the
pruned graph (``analytic.decoder_shapes`` / ``decoder_claim_columns(..., prune_last=True)``).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pvi.fullcheck import analytic
from pvi.fullcheck import reference as ref
from pvi.fullcheck import transformer as tr
from pvi.fullcheck.transformer import CONFIGS, build_decoder

from test_plans import _TINY   # the plan tests' tiny decoders

_REAL = {name: dataclasses.replace(CONFIGS[name], vocab=512) for name in ("gpt2", "opt-350m", "llama2-7b", "qwen3-4b")}


@pytest.mark.parametrize("kind", [*_TINY, *_REAL])
def test_a_pruned_decoder_gives_the_same_logits(kind):
    """The last block at the last position gives the logits of the whole block (the same integers), on
    the tiny decoders and on GPT-2, OPT-350M (embed_dim), Llama-2-7B and Qwen3-4B (GQA, q/k norm) blocks,
    with the same weights; its q, o and MLP claims have one column, k and v all of them."""
    cfg = _TINY.get(kind) or _REAL[kind]
    layers = 2 if kind in _TINY else 1
    full = build_decoder(cfg, n_layers=layers, calib_tokens=8, seed=1)
    pruned = build_decoder(cfg, n_layers=layers, calib_tokens=8, seed=1, prune_last=True)
    assert [(op.n_rows, op.row_length) for op in full.mat_ops] == [(op.n_rows, op.row_length) for op in pruned.mat_ops]
    assert all(torch.equal(a.weight, b.weight) for a, b in zip(full.mat_ops, pruned.mat_ops))
    for t in (1, 2, 9):
        x = torch.randint(0, cfg.vocab, (1, t), generator=torch.Generator().manual_seed(t))
        (env_a, claims_a), (env_b, claims_b) = full.forward(x), pruned.forward(x)
        assert torch.equal(env_a[full.output_name], env_b[pruned.output_name])
        cols = analytic.decoder_claim_columns(cfg, t, layers, prune_last=True)
        assert [z.shape for z in claims_b.values()] == [(s.n_rows, cols[s.name])
                                                        for s in analytic.decoder_shapes(cfg, layers, True)]
        assert [s.row_length for s in analytic.decoder_shapes(cfg, layers, True)] == [
            op.row_length for op in pruned.mat_ops]
        assert (sum(z.numel() for z in claims_b.values()) < sum(z.numel() for z in claims_a.values())) == (t > 1)
        assert pruned.claim_columns(x) == [z.shape[1] for z in claims_b.values()]


def test_the_pruned_shapes_read_the_last_position_in_the_last_block_only():
    cfg = CONFIGS["llama2-7b"]
    for layers in (1, 2, None):
        shapes = analytic.decoder_shapes(cfg, layers, prune_last=True)
        last = (layers or cfg.n_layers) - 1
        reads = {s.name: s.input for s in shapes}
        assert reads[f"q{last}"] == f"attn_last{last}" and reads[f"k{last}"] == reads[f"v{last}"] == f"attn_in{last}"
        assert all(reads[f"q{i}"] == reads[f"k{i}"] for i in range(last))
        cols = analytic.decoder_claim_columns(cfg, 64, layers, prune_last=True)
        assert {n for n, c in cols.items() if c == 1} == {f"{p}{last}" for p in ("q", "o", "gate", "up", "down")} | {"head"}
        unpruned = analytic.decoder_shapes(cfg, layers)
        assert [(s.name, s.n_rows, s.row_length) for s in shapes] == [(s.name, s.n_rows, s.row_length) for s in unpruned]


@pytest.mark.parametrize("m_s", [1 << 20, 1000])          # the int32 path and its int64 fallback
@pytest.mark.parametrize("hq,hkv,group_heads", [(8, 2, None), (4, 4, None), (8, 8, 3), (8, 2, 3)])
def test_the_queries_of_the_last_positions_attend_as_in_the_whole_attention(hq, hkv, group_heads, m_s, monkeypatch):
    g = torch.Generator().manual_seed(hq + hkv)
    t, dh = 12, 16
    q = torch.randint(-127, 128, (2, t, hq * dh), generator=g)
    k = torch.randint(-127, 128, (2, t, hkv * dh), generator=g)
    v = torch.randint(-127, 128, (2, t, hkv * dh), generator=g)
    whole = tr._attention(q, k, v, m_s, 1 << 21, hq, hkv, dh)
    if group_heads:                                  # a few heads per group (the stacked GQA path otherwise)
        width = 4 if tr._int32_scores(m_s, dh, t) else 8
        monkeypatch.setattr(tr, "ATTN_BYTES", width * 2 * t * group_heads)
    for tq in (1, 3, t):
        got = tr._attention(q[:, -tq:], k, v, m_s, 1 << 21, hq, hkv, dh)
        assert torch.equal(got, whole[:, -tq:])
        assert torch.equal(got, ref.attention(q[:, -tq:], k, v, m_s, 1 << 21, hq, hkv, dh))

    def heads(a, rep=1):                             # [B, T, H*dh] -> [B, H*rep, T, dh]
        return a.reshape(2, t, -1, dh).transpose(1, 2).repeat_interleave(rep, 1)

    last = (heads(q)[:, :, -1:], heads(k, hq // hkv), heads(v, hq // hkv))
    assert torch.equal(tr._attention_heads(*last, m_s), ref.attention_heads(*last, m_s, None))
    assert not tr._causal_notmask(t, 2, "cpu", 1).any()          # the last query row attends to every key
    assert torch.equal(tr._causal_notmask(t, 2, "cpu", 3), tr._causal_notmask(t, 1, "cpu")[-3:].repeat(2, 1))


def test_rope_at_an_offset_rotates_as_at_those_positions():
    x = torch.randint(-127, 128, (1, 9, 4, 16), generator=torch.Generator().manual_seed(3))
    for theta in (1e4, 1e6):
        whole = tr._rope(x, theta)
        for tq in (1, 4):
            assert torch.equal(tr._rope(x[:, -tq:], theta, 9 - tq), whole[:, -tq:])
            assert torch.equal(ref.rope(x[:, -tq:], theta, 9 - tq), whole[:, -tq:])

