"""Integer OPT decoders built from *real* pretrained weights: the certified OPT models (Sec. ``sec:quality``).

:func:`pvi.models.decoder.build_decoder` measures cost with random int8 weights, which is sound because the prover's
and verifier's work depend only on the shapes. It says nothing about the quality of the int8 model the defence
proves. This module answers that for OPT: it builds the same integer graph -- the same op kinds, in the same order, with
the same matrix shapes -- from a Hugging Face checkpoint, so its perplexity is the perplexity of the model the
defence actually proves.

What changes relative to ``build_decoder`` is only the *constants*:

* **Weights** are the real ones, quantised per tensor to int8 (``s_w = max|W| / 127``).
* **LayerNorm gain and bias are folded** into the following matrix
  (``W' = W diag(g)``, ``b' = b + W beta``). The proven norm op stays exactly what it is
  -- parameter-free, integer mean/variance, int8 output G x (normalised value) with
  G = ``norm_gain`` a public constant (``build_decoder`` uses 32), or a public gain vector
  (SmoothQuant through the norm's gains, ``smooth``) -- so the verifier recomputes the same ops.
* **Requantisation multipliers come from true scales**, not from matching a target
  standard deviation. ``build_decoder`` rescales every branch to a fixed std (fine
  for cost, where values are meaningless); here every integer tensor carries its
  real scale, and each multiplier is the ratio that preserves real values.

One structural difference, stated rather than hidden: folding the *final* LayerNorm
leaves a bias ``W beta`` on the LM head, and the benchmark graph's LM head has no
bias slot. The real-weight graph therefore has one extra column per vocabulary row
(``row_length`` d+1 instead of d): the LM head's rows grow by ``1/d``, 0.02% at OPT-6.7B's d = 4096. Every other
op matches ``build_decoder`` one for one; ``matches_benchmark_graph`` checks it.

Activations stay per-tensor int8, as proved. The norm's gain G is a public constant:
the benchmark's G = 32 clips OPT's outlier features; the best scalar gains are 4, 3 and 2 for OPT-125M,
OPT-1.3B and OPT-6.7B (``sec:ext-quality``), at the same cost.

``prune_last`` (opt-in, as ``build_decoder(..., prune_last=True)``) builds the last block at the
last position only, but for its keys and values; the pruned block is calibrated on every position,
so the graph has the weights, gains and multipliers of the graph without pruning and gives the same
next-token logits (``tests/test_pruning.py``).

Two non-default options of :func:`build_opt_from_hf` build the variants of the extended version's quality section
(``sec:ext-quality``), each a change of constants or of the offline weight quantisation only: per-row power-of-two
weight scales (``row_pow2``; protocol v0 only) and cross-layer equalisation between fc1 and fc2 (``cle_mlp``).
:func:`probability_mass` measures the share of probability mass the integer attention's rounded rows keep.
"""

from __future__ import annotations

import math

import torch

from pvi.core.graph import (INT8_MAX, CheapOp, IntGraph, MatOp, OpFn, identity_op, op_fn, requant_fn, requant_op,
                            residual_op)
from pvi.models.decoder import (EXP_TABLE, CERTIFIED_PROB_MAX, CONFIGS, RES_MAX, SHIFT, DecoderConfig,
                                integer_attention, _isqrt, attention_op, build_decoder, last_position_op, lut_op,
                                norm_op, positions_op, scale_op)

__all__ = ["CERTIFIED_RECIPE", "LEGACY_RECIPE", "build_opt_from_hf", "is_pruned", "matches_benchmark_graph",
           "real_logits", "every_position", "attention_mass", "probability_mass"]

_RELU = torch.tensor([max(0, x) for x in range(-128, 128)], dtype=torch.int64)


def _mult(s_in, s_out):
    """The multiplier ``round(2^SHIFT s_in / s_out)`` of a requantisation from scale ``s_in`` to ``s_out``: an integer,
    or an int64 vector when ``s_in`` is a per-row scale vector."""
    if torch.is_tensor(s_in):
        return torch.round((1 << SHIFT) * s_in / s_out).to(torch.int64)
    return round((1 << SHIFT) * s_in / s_out)


class _RealBuilder:
    """Like ``pvi.models.decoder._Builder``, but every tensor carries its real scale."""

    def __init__(self, tokens: torch.Tensor, pct: float, norm_gain: float, smooth: float | None = None,
                 smooth_pct: float = 99.9, row_pow2: int = 0) -> None:
        self.norm_gain = norm_gain
        self.smooth, self.smooth_pct = smooth, smooth_pct
        self.row_pow2 = row_pow2
        self.ops: list = []
        self.env = {"x": tokens}
        self.scale: dict[str, float] = {}
        self.pct = pct
        self._n = 0

    def name(self, p):
        self._n += 1
        return f"{p}{self._n}"

    def _run(self, op, scale: float):
        self.ops.append(op)
        if isinstance(op, MatOp):
            xin = self.env[op.inputs[0]]
            self.env[op.output] = op.fold(op.compute(xin), xin)
        else:
            self.env[op.output] = op.fn(*[self.env[n] for n in op.inputs])
        self.scale[op.output] = scale
        return op.output

    def mat(self, inp, w: torch.Tensor, b: torch.Tensor | None, s_in, layout="linear",
            s_w: float | None = None, rows: bool = False):
        """Real ``w [rows, cols]`` -> int8; the product's real scale is ``s_w * s_in``.  A per-channel input scale
        ``s_in`` (a smoothed norm's, a vector) is folded into ``w``'s columns first, then one scale per matrix.
        With ``rows`` and ``row_pow2``: one power-of-two scale per row (:func:`build_opt_from_hf`), so the product's
        real scale is a vector over the rows."""
        n = self.name("mm")
        if rows and self.row_pow2:
            w_eff = w.double() * (s_in.double()[None, :] if torch.is_tensor(s_in) else s_in)   # on the integer input
            row = w_eff.abs().amax(1).clamp_min(1e-30) / INT8_MAX
            top = float(row.max())
            s_z = top * torch.pow(2.0, -torch.floor(torch.log2(top / row)).clamp(0, self.row_pow2 - 1))
            w_int = (w_eff / s_z[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            b_int = None if b is None else (b.double() / s_z).round().to(torch.int64)
            return self._run(MatOp(n, (inp,), n + ".z", weight=w_int, bias=b_int, layout=layout), s_z)
        if torch.is_tensor(s_in):
            w_eff = w.double() * s_in.double()[None, :]                 # acts on the integer input
            s_z = float(w_eff.abs().max()) / INT8_MAX
            w_int = (w_eff / s_z).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            b_int = None if b is None else (b.double() / s_z).round().to(torch.int64)
            return self._run(MatOp(n, (inp,), n + ".z", weight=w_int, bias=b_int, layout=layout), s_z)
        s_w = float(w.abs().max()) / INT8_MAX if s_w is None else s_w
        w_int = (w / s_w).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        s_z = s_w * s_in
        b_int = None if b is None else (b.double() / s_z).round().to(torch.int64)
        return self._run(MatOp(n, (inp,), n + ".z", weight=w_int, bias=b_int, layout=layout), s_z)

    def cheap(self, prefix, inputs, fn, scale, note="", params=None):
        n = self.name(prefix)
        return self._run(CheapOp(n, tuple(inputs), n + ".y", fn=fn, note=note, params=params or {}), scale)

    def requant(self, prefix, inputs, scale, note, **params):
        """A requantising consumer of a weight op (:func:`pvi.core.graph.requant_fn`, so V1 can cut it).  A vector
        multiplier (per-row weight scales) takes the same op kinds with a vector constant; V1 does not cut such
        an op."""
        mult = params["mult"]
        if torch.is_tensor(mult):
            if params["kind"] == "requant":
                fn = op_fn(requant_op, mult=mult, shift=SHIFT, lo=-INT8_MAX, hi=INT8_MAX)
            else:
                fn = op_fn(residual_op, mult=mult, shift=SHIFT, res_max=RES_MAX)
            return self.cheap(prefix, inputs, fn, scale, note, {"vec_requant": {"kind": params["kind"]}})
        return self.cheap(prefix, inputs, requant_fn(params), scale, note, {"requant": params})

    def last(self, name):
        """The last position of ``name``, calibrated on every position (``pvi.models.decoder._Builder.last(...,
        calibrate_all=True)``): the op takes the last position, while the calibration keeps every position and
        ``name``'s real scale, so the ops after it get the percentiles, gains and multipliers of the graph
        without it."""
        n = self.name("last")
        self.ops.append(CheapOp(n, (name,), n + ".y", fn=op_fn(last_position_op), note="last position"))
        self.env[n + ".y"] = self.env[name]
        self.scale[n + ".y"] = self.scale[name]
        return n + ".y"

    def release(self, keep) -> None:
        """Drop every calibration tensor but the tokens and ``keep`` (the residual stream between blocks): no
        later op reads them, so no constant changes, and a 2,048-token calibration window of OPT-6.7B holds one block's
        tensors (25 int64 tensors of ``T x d``: 1.7 GB) instead of all 32 blocks' (54 GB)."""
        self.env = {k: self.env[k] for k in ("x", *keep)}

    def to_int8(self, z):
        real = self.env[z].double().abs() * self.scale[z]
        top = float(torch.quantile(real.flatten()[:1 << 24], self.pct / 100.0)) if self.pct < 100 \
            else float(real.max())
        s_out = max(top, 1e-12) / INT8_MAX
        m = _mult(self.scale[z], s_out)
        return self.requant("rq", [z], s_out, "requant", kind="requant", mult=m, shift=SHIFT, lo=-INT8_MAX,
                            hi=INT8_MAX)

    def residual(self, r, z):
        """r + z, with z brought to the residual's real scale (not to a target std)."""
        m = _mult(self.scale[z], self.scale[r])
        return self.requant("res", [r, z], self.scale[r], "residual add", kind="residual", mult=m, shift=SHIFT,
                            res_max=RES_MAX)

    def norm(self, r, d, consumers=None):
        # _norm_int with gain G << 16 emits G * (normalised value), clamped to int8: real
        # scale 1/G, range +-127/G standard deviations. G is a public constant of the op,
        # so choosing it does not change any shape or cost.
        if self.smooth is None or consumers is None:
            gain = torch.full((d,), round(self.norm_gain * (1 << 16)), dtype=torch.int64)
            return self.cheap("norm", [r], op_fn(norm_op, gain=gain, center=True), 1.0 / self.norm_gain, "norm")
        # SmoothQuant through the gain vector: channel j gets G_j = c / s_j, s_j = a_j^alpha / w_j^(1-alpha)
        # (a_j: a high percentile of |normalised_j| on calibration data; w_j: the largest |W_ij| of the matrices this
        # norm feeds), c such that the largest a_j G_j is 127; the next matrices take 1/G_j into column j (mat()).
        x = self.env[r]
        mean = torch.div(x.sum(-1, keepdim=True) + d // 2, d, rounding_mode="floor")
        xc = x - mean
        sigma = _isqrt(torch.div((xc * xc).sum(-1, keepdim=True), d, rounding_mode="floor"))
        n = (xc.double() / sigma.double()).reshape(-1, d).abs()
        a = (torch.quantile(n[:1 << 14], self.smooth_pct / 100.0, dim=0) if self.smooth_pct < 100
             else n.amax(0)).clamp_min(1e-3)
        wj = torch.stack([w.double().abs().amax(0) for w in consumers]).amax(0).clamp_min(1e-8)
        s_ = a.pow(self.smooth) / wj.pow(1 - self.smooth)
        g_real = ((INT8_MAX / float((a / s_).max())) / s_).clamp(1.0 / 64, float(1 << 14))
        gain = torch.round(g_real * (1 << 16)).to(torch.int64)
        return self.cheap("norm", [r], op_fn(norm_op, gain=gain, center=True), (1 << 16) / gain.double(), "norm")


def _fold(w: torch.Tensor, b: torch.Tensor | None, gamma: torch.Tensor, beta: torch.Tensor):
    """``W (x*g + beta) + b  ==  (W diag g) x + (W beta + b)``."""
    w32 = w.double()
    wf = w32 * gamma.double()[None, :]
    bf = w32 @ beta.double() + (b.double() if b is not None else 0.0)
    return wf, bf


def _fc2_input_max(model, calib_ids: torch.Tensor) -> list[torch.Tensor]:
    """Per block, each channel's largest ``|value|`` at fc2's input (the ReLU's output) in the checkpoint on
    ``calib_ids``."""
    stats, hooks = [], []
    for layer in model.model.decoder.layers:
        hooks.append(layer.fc2.register_forward_hook(
            lambda mod, inp, out: stats.append(inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0).double())))
    try:
        with torch.no_grad():
            model(calib_ids.to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return stats


def _cle_scale(act: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, alpha: float, ratio: float) -> torch.Tensor:
    """Cross-layer equalisation's per-channel divisor ``s_j = sqrt(act_j^alpha r1_j^(1-alpha) / r2_j)`` of a tensor
    produced by the rows of one matrix (row ranges ``r1``) and read by the columns of the next (column ranges
    ``r2``): afterwards the producer's row range (``alpha = 0``, data-free equalisation) or the activation's range
    (``alpha = 1``) meets the reader's column range ``r2_j s_j``; clamped to ``[median / ratio, median ratio]``."""
    a = act.clamp_min(1e-8 * float(act.max()))
    s = (a.pow(alpha) * r1.double().clamp_min(1e-12).pow(1 - alpha) / r2.double().clamp_min(1e-12)).sqrt()
    med = float(s.median())
    return s.clamp(med / ratio, med * ratio)


CERTIFIED_RECIPE = dict(norm_gain=4, smooth=0.6, smooth_pct=100.0, pct=100.0, pct_att=100.0)
"""The certified OPT models' recipe (Sec. ``sec:quality``): SmoothQuant with alpha 0.6 through the norm gains, each
channel's maximum as its statistic and no clipping ("smooth-max"), norm gain 4, and ``prob_max`` =
:data:`CERTIFIED_PROB_MAX` (the default).  Calibration: the first four 2,048-token windows of the WikiText-2 (raw)
*train* split (``experiments.common.checkpoints.certified_calibration``).  WikiText-2 test perplexity at 2,048
tokens, measured: x1.076 (OPT-125M), x1.026 (OPT-1.3B), x1.008 (OPT-6.7B) of fp32 (``tab:quality``)."""
LEGACY_RECIPE = dict(norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
"""The percentile recipe of the first smoothing study (the extended version's quality section): SmoothQuant with
alpha 0.55 through the norm gains with the 99.9th-percentile channel statistic, 99.99th-percentile clipping of the
weights and of the attention inputs, norm gain 4."""


def build_opt_from_hf(model, calib_ids: torch.Tensor, *, pct: float = 100.0, norm_gain: float = 32,
                      residual_headroom: float = 1.25, device=None, smooth: float | None = None,
                      smooth_pct: float = 99.9, pct_att: float = 100.0,
                      prune_last: bool = False, prob_max: int = CERTIFIED_PROB_MAX, row_pow2: int = 0,
                      cle_mlp: float | None = None, cle_ratio: float = 32.0,
                      n_layers: int | None = None) -> tuple[IntGraph, dict]:
    """An integer OPT decoder with the checkpoint's real weights.

    ``calib_ids`` ``[B, T]`` (``B`` calibration windows) sets every activation scale. The knobs are all
    public constants of cheap ops or the offline weight quantisation -- none changes a shape, so none changes the
    cost:

    * ``pct``: percentile of |value| mapped to 127 at each requantisation of q, k, v and fc1
      (100 = never clip on calibration data); ``pct_att`` the same at the attention output;
    * ``norm_gain``: the norm's output is ``norm_gain * normalised``, so its range is
      +-127/norm_gain standard deviations (``build_decoder`` uses 32);
    * ``smooth`` (alpha; None: one scalar gain): SmoothQuant through the norm's gain vector, with
      ``smooth_pct`` the percentile of each channel's normalised magnitude (100: its maximum);
    * ``prob_max``: the attention probabilities' scale (:data:`pvi.models.decoder.PROB_MAX` = 255: 8 bits; the default,
      :data:`CERTIFIED_PROB_MAX` = 2047: 11 bits), whose output multiplier is calibrated at that scale;
    * the embed-to-residual multiplier is chosen so the embedding keeps full int8
      resolution (``build_decoder`` uses 8).

    ``device`` is where the integer calibration pass runs (the fp model stays where it is).

    ``info`` holds the builder's settings, the residual stream's real scale ``s_residual``, the logits' real scale
    ``logit_scale`` and ``scales``: the real scale of every op's output by tensor name (a float, or a vector for a
    smoothed norm's channels or per-row weight scales), which the attribution study's float copy of the graph reads
    (``experiments/quality/attribution.py``).  ``info`` is not part of the graph or its digest.

    ``prune_last`` (opt-in): the last block's q, attention output, out-projection, residuals and MLP
    read the last position only, its k and v every position (the ops and order of
    ``build_decoder(..., prune_last=True)``: a "last position" op before q and one before the first
    residual add).  The calibration still sees every position wherever the graph without pruning does
    (:meth:`_RealBuilder.last`), so every weight, norm gain and multiplier is that graph's, and the
    next-token logits are its integers (``tests/test_pruning.py``).

    ``n_layers`` (None: every block): build the checkpoint's first ``n_layers`` blocks only, then the final norm and
    the LM head (``build_llama_from_hf``'s ``n_layers``).  The residual stream's scale is still set on the whole
    checkpoint's hidden states, so the blocks built are the full graph's first ``n_layers`` blocks, op for op.

    Two variants of the extended version (``sec:ext-quality``), off by default:

    * ``row_pow2`` (0: one scale per weight matrix, the certified graphs): with ``n > 0``, the linear weight ops but
      the LM head (q, k, v, the out-projection, fc1, fc2) get one power-of-two weight scale per row, ``s 2^-e`` with
      ``0 <= e < n`` and ``s`` the matrix's largest row scale, and the requantisation or residual add that reads
      such an op takes a vector multiplier with at most ``n`` distinct values (params ``{"vec_requant": ...}``).
      Protocol v0 recomputes it like any other cheap op; V1's cut needs one multiplier per op
      (:func:`pvi.core.graph.requant_fn`), so these graphs are v0 only.
    * ``cle_mlp`` (alpha; None: off): cross-layer equalisation between fc1 and fc2.  Channel ``j`` of fc1's output
      is divided by ``s_j`` (:func:`_cle_scale`: ``a_j`` the channel's largest ``|value|`` at fc2's input in the
      checkpoint on ``calib_ids``, ``r1_j`` and ``r2_j`` the largest ``|.|`` in row ``j`` of the folded fc1 and in
      column ``j`` of fc2, clamped to a factor ``cle_ratio`` around the median) and column ``j`` of fc2 is
      multiplied by it; ``relu(x / s) = relu(x) / s`` for ``s > 0``, so the float function is unchanged.  With
      ``cle_mlp > 0`` the graph depends on the checkpoint's float forward pass, whose last bits differ between
      floating-point libraries and CPUs (a few units of a requantisation multiplier); ``cle_mlp = 0`` (data-free,
      the extended version's cells) and every other statistic here are values of the integer graph or of the
      weights, the same on every host.
    """
    cfg = model.config
    assert cfg.do_layer_norm_before, "post-LN OPT-350M is not handled"
    assert cfg.word_embed_proj_dim == cfg.hidden_size, "project_in/out OPT-350M is not handled"
    assert row_pow2 >= 0
    assert 1 <= prob_max <= 2047, prob_max            # the bound of the exact int32 attention
    dec = model.model.decoder
    d, n_heads = cfg.hidden_size, cfg.num_attention_heads
    dh = d // n_heads

    # The residual stream's real scale: large enough that the real residual never
    # clips at RES_MAX on calibration data. The embedding's int8 scale is then tied to
    # it by the graph's fixed "embed to residual" x8 op: s_embed = 8 * s_residual.
    with torch.no_grad():
        hs = model(calib_ids.to(model.device), output_hidden_states=True).hidden_states
    res_max = max(float(h.abs().max()) for h in hs) * residual_headroom
    del hs
    s_res = max(res_max / RES_MAX, float(dec.embed_tokens.weight.abs().max()) / (INT8_MAX * 8))
    fc2_in = _fc2_input_max(model, calib_ids) if cle_mlp is not None else None

    B = _RealBuilder(calib_ids.to(device or calib_ids.device), pct, norm_gain, smooth, smooth_pct, row_pow2)
    # Integer multiplier c with s_embed = c * s_residual, as close as possible to giving the
    # embedding its full int8 resolution (build_decoder hard-codes c = 8).
    embed_mult = max(1, round(float(dec.embed_tokens.weight.abs().max()) / INT8_MAX / s_res))
    tok = B.mat("x", dec.embed_tokens.weight.T.float(), None, 1.0, layout="embed",
                s_w=embed_mult * s_res)
    pos_ids = B.cheap("pos", ["x"], op_fn(positions_op), 1.0, "position ids")
    # OPT's learned positions are offset by 2 rows.
    max_pos = cfg.max_position_embeddings
    pos_table = dec.embed_positions.weight[2:2 + max_pos].T.float()
    pos = B.mat(pos_ids, pos_table, None, 1.0, layout="embed")
    r = B.residual(B.cheap("scale", [tok], op_fn(scale_op, c=embed_mult), s_res, "embed to residual"), pos)

    layers = list(dec.layers)[:n_layers]
    if not layers:
        raise ValueError(f"n_layers {n_layers}: the checkpoint has {len(dec.layers)} blocks")
    for i, layer in enumerate(layers):
        prune = prune_last and i == len(layers) - 1
        at = layer.self_attn
        ln1 = layer.self_attn_layer_norm
        folded = [_fold(proj.weight.float(), proj.bias, ln1.weight, ln1.bias)
                  for proj in (at.q_proj, at.k_proj, at.v_proj)]
        h = B.norm(r, d, [wf for wf, _ in folded])
        # (pruned) q reads the last position; k and v every position, which its one query row attends to
        q, k, v = (B.to_int8(B.mat(B.last(h) if prune and j == 0 else h, wf.float(), bf, B.scale[h], rows=True))
                   for j, (wf, bf) in enumerate(folded))
        # HF scales q by dh^-0.5 inside the module; here that lives in m_s, as in build_decoder.
        sq, sk, sv = B.scale[q], B.scale[k], B.scale[v]
        m_s = round((1 << SHIFT) * 16.0 * sq * sk / math.sqrt(dh))
        raw = integer_attention(B.env[q], B.env[k], B.env[v], m_s, None, n_heads, n_heads, dh, prob_max)
        s_raw = sv / float(prob_max)               # probabilities 0..prob_max x int8 values (8 bits: 255)
        rabs = raw.double().abs().flatten()
        top = (float(torch.quantile(rabs[:1 << 24], pct_att / 100.0)) if pct_att < 100 else float(rabs.max())) * s_raw
        s_att = max(top, 1e-12) / INT8_MAX
        m_o = round((1 << SHIFT) * s_raw / s_att)
        att = B.cheap("attn", [q, k, v], op_fn(attention_op, m_s=m_s, m_o=m_o, heads=n_heads, kv_heads=n_heads, dh=dh,
                                               pmax=prob_max), s_att, "attention")
        # (pruned) the residual stream's last position takes the attention branch (its last position only)
        r = B.residual(B.last(r) if prune else r, B.mat(att, at.out_proj.weight.float(), at.out_proj.bias, s_att,
                                                        rows=True))

        ln2 = layer.final_layer_norm
        wf, bf = _fold(layer.fc1.weight.float(), layer.fc1.bias, ln2.weight, ln2.bias)
        w2 = layer.fc2.weight.float()
        if cle_mlp is not None:      # offline: fc1's rows / s_j, fc2's columns * s_j
            s_ = _cle_scale(fc2_in[i], wf.abs().amax(1), w2.double().abs().amax(0), cle_mlp, cle_ratio)
            wf, bf = wf / s_[:, None], bf / s_
            w2 = (w2.double() * s_[None, :]).float()
        h = B.norm(r, d, [wf])
        f1 = B.to_int8(B.mat(h, wf.float(), bf, B.scale[h], rows=True))
        act = B.cheap("act", [f1], op_fn(lut_op, table=_RELU), B.scale[f1], "relu")
        r = B.residual(r, B.mat(act, w2, layer.fc2.bias, B.scale[act], rows=True))
        B.release([r])

    lnf = dec.final_layer_norm
    wf, bf = _fold(model.lm_head.weight.float(), None, lnf.weight, lnf.bias)
    h = B.norm(r, d, [wf])
    # (in a pruned graph the one position of its last block; calibrated, as the head, on the last position)
    h = B.cheap("last", [h], op_fn(last_position_op), B.scale[h], "last position")
    logits = B.mat(h, wf.float(), bf, B.scale[h])
    info = {"s_residual": s_res, "residual_real_max": res_max, "logit_scale": B.scale[logits],
            "pct": pct, "norm_gain": norm_gain, "embed_mult": embed_mult, "smooth": smooth,
            "smooth_pct": smooth_pct, "pct_att": pct_att, "prune_last": prune_last, "row_pow2": row_pow2,
            "cle_mlp": cle_mlp, "cle_ratio": cle_ratio if cle_mlp is not None else None,
            "scales": dict(B.scale)}
    return IntGraph(B.ops, "x", logits), info


def is_pruned(graph: IntGraph) -> bool:
    """Whether ``graph`` (of ``build_opt_from_hf`` or ``build_decoder``) has its last block pruned: its last block
    adds two "last position" ops to the one before the LM head."""
    return sum(isinstance(o, CheapOp) and o.note == "last position" for o in graph.ops) > 1


def every_position(graph: IntGraph) -> IntGraph:
    """``graph`` with every "last position" slice replaced by the identity, so one pass gives its output at every
    position (:func:`real_logits`); the other ops are ``graph``'s own objects."""
    ops = [CheapOp(o.name, o.inputs, o.output, fn=op_fn(identity_op), note=o.note, params=o.params)
           if isinstance(o, CheapOp) and o.note == "last position" else o for o in graph.ops]
    return IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta))


def real_logits(graph: IntGraph, info: dict, ids: torch.Tensor, *, all_positions: bool = False):
    """Run the integer graph and return real-valued logits.

    With ``all_positions`` the "last position" slice is skipped, giving logits at every
    position in one pass. That is exactly what one query per prefix would produce:
    every op is per-position except causal attention, and no constant depends on the
    input, so position t's integers depend only on tokens 0..t.

    A graph with its last block pruned (``prune_last``) gives every position too: the two
    slices of its last block are skipped as well, so in that block position t's query row
    attends to the keys 0..t and its residual add reads position t -- the integers the pruned
    graph gives on the prefix 0..t, as one query per prefix would.  The result is that of the
    graph without pruning, whose constants the pruned graph shares (``tests/test_pruning.py``).
    """
    if all_positions:
        graph = every_position(graph)
    env, _ = graph.forward(ids)
    return env[graph.output_name].double() * info["logit_scale"]


def attention_mass(q: torch.Tensor, k: torch.Tensor, *, m_s: int, heads: int, kv_heads: int, dh: int, pmax: int,
                   heads_per_chunk: int = 8) -> float:
    """The share of probability mass the integer attention (``attention_op``'s probabilities ``p_j = round(pmax e_j /
    tot)``) keeps: the mean over batch rows, heads and query rows of ``sum_j p_j / pmax``.  Rounding loses the mass
    of a diffuse row: at most ``2 pmax`` keys of a row get ``p_j > 0``; a short row, whose probabilities round half
    up, may keep slightly more than all of it (``sum_j p_j <= pmax + T/2``).  ``q [B, Tq, heads dh]`` holds the queries
    of the last ``Tq`` positions, ``k [B, T, kv_heads dh]``.  The scores are exact float32 products of int8 values
    (``|s| <= 127^2 dh < 2^24`` for ``dh <= 1040``), the rest the integers of
    :func:`pvi.models.decoder.integer_attention`."""
    b, tq, _ = q.shape
    t = k.shape[1]
    assert dh <= 1040
    qh = q.reshape(b, tq, heads, dh).transpose(1, 2)
    kh = k.reshape(b, t, kv_heads, dh).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    allowed = torch.ones(tq, t, dtype=torch.bool, device=q.device).tril(diagonal=t - tq)
    table = EXP_TABLE.to(q.device)
    mass = []
    for h0 in range(0, heads, heads_per_chunk):
        h1 = min(heads, h0 + heads_per_chunk)
        s = torch.matmul(qh[:, h0:h1].to(torch.float32), kh[:, h0:h1].to(torch.float32).transpose(-1, -2))
        s = s.to(torch.int64).masked_fill_(~allowed, -(1 << 40))
        gap = s.amax(-1, keepdim=True) - s
        del s
        idx = ((gap.clamp_(max=1 << 31) * m_s + (1 << (SHIFT - 1))) >> SHIFT).clamp_(0, table.numel() - 1)
        del gap
        e = table[idx].masked_fill_(~allowed, 0)
        del idx
        tot = e.sum(-1, keepdim=True)
        p = (tot + 2 * pmax * e).div_(2 * tot, rounding_mode="floor")
        del e, tot
        mass.append(p.sum(-1).double().mean() / pmax * (h1 - h0))
    return float(sum(mass)) / heads


def probability_mass(graph: IntGraph, ids: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """``graph``'s output on ``ids`` (the integers of ``graph.forward(ids, free=True)``) and, for each attention op in
    order (one per block), :func:`attention_mass` on its inputs: a diagnostic of the probabilities' scale
    (``sec:quality``: the share of mass a rounded row keeps at 8 and at 11 bits)."""
    env = {graph.input_name: ids}
    last = graph.last_use()
    masses = []
    for i, op in enumerate(graph.ops):
        if isinstance(op, MatOp):
            xin = env[op.inputs[0]]
            env[op.output] = op.fold(op.compute(xin), xin)
        else:
            args = [env[n] for n in op.inputs]
            if isinstance(op.fn, OpFn) and op.fn.kind == attention_op.op_kind:
                c = op.fn.consts
                masses.append(attention_mass(args[0], args[1], m_s=c["m_s"], heads=c["heads"],
                                             kv_heads=c["kv_heads"], dh=c["dh"], pmax=c["pmax"]))
            env[op.output] = op.fn(*args)
        for n in op.inputs:
            if last.get(n, -1) <= i:
                env.pop(n, None)
    return env[graph.output_name], masses


def matches_benchmark_graph(graph: IntGraph, config_name: str | DecoderConfig, *,
                            prune_last: bool | None = None) -> list[str]:
    """Differences from ``build_decoder``'s graph (op kinds, order, matrix shapes, cheap ops' notes) for the
    config ``config_name`` (a name of ``CONFIGS``, or a ``DecoderConfig``), built with ``prune_last`` (default:
    as ``graph`` is, :func:`is_pruned`).

    Returns the list of mismatches; the only one expected is the LM head's bias.
    """
    cfg = CONFIGS[config_name] if isinstance(config_name, str) else config_name
    ref = build_decoder(cfg, prune_last=is_pruned(graph) if prune_last is None else prune_last)
    problems = []
    if len(ref.ops) != len(graph.ops):
        return [f"op count {len(graph.ops)} vs benchmark {len(ref.ops)}"]
    for a, b in zip(graph.ops, ref.ops):
        if type(a) is not type(b):
            problems.append(f"{a.name}: {type(a).__name__} vs {type(b).__name__}")
        elif isinstance(a, MatOp) and (a.n_rows, a.n_in) != (b.n_rows, b.n_in):
            problems.append(f"{a.name}: shape {(a.n_rows, a.n_in)} vs {(b.n_rows, b.n_in)}")
        elif isinstance(a, MatOp) and a.has_bias != b.has_bias:
            problems.append(f"{a.name}: bias {a.has_bias} vs benchmark {b.has_bias}")
        elif isinstance(a, CheapOp) and a.note != b.note:
            problems.append(f"{a.name}: '{a.note}' vs '{b.note}'")
    return problems
