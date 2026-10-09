"""Integer OPT decoders built from *real* pretrained weights (the report's int8 perplexities, Section 4.1).

``transformer.build_decoder`` measures cost with random int8 weights, which is
sound because the prover's and verifier's work depend only on the shapes. It says
nothing about whether the int8 model we prove is any *good*. This module answers
that for OPT: it builds the same integer graph -- the same op kinds, in the same
order, with the same matrix shapes -- from a Hugging Face checkpoint, so its
perplexity is the perplexity of the model the defence actually proves.

What changes relative to ``build_decoder`` is only the *constants*:

* **Weights** are the real ones, quantised per tensor to int8 (``s_w = max|W| / 127``).
* **LayerNorm gain and bias are folded** into the following matrix
  (``W' = W diag(g)``, ``b' = b + W beta``). The proven norm op stays exactly what it is
  -- parameter-free, integer mean/variance, int8 output G x (normalised value) with
  G = ``norm_gain`` a public constant (``build_decoder`` uses 32) -- so the verifier
  recomputes the same ops.
* **Requantisation multipliers come from true scales**, not from matching a target
  standard deviation. ``build_decoder`` rescales every branch to a fixed std (fine
  for cost, where values are meaningless); here every integer tensor carries its
  real scale, and each multiplier is the ratio that preserves real values.

One structural difference, stated rather than hidden: folding the *final* LayerNorm
leaves a bias ``W beta`` on the LM head, and the benchmark graph's LM head has no
bias slot. The real-weight graph therefore has one extra column per vocabulary row
(``row_length`` d+1 instead of d) -- about 0.02% of OPT-6.7B's proof. Every other
op matches ``build_decoder`` one for one; ``matches_benchmark_graph`` checks it.

Activations stay per-tensor int8, as proved. The norm's gain G is a public constant:
the benchmark's G = 32 clips OPT's outlier features (perplexity in the hundreds to tens
of thousands); G = 2-4 does not, at the same cost.
"""

from __future__ import annotations

import math

import torch

from .graph import INT8_MAX, CheapOp, IntGraph, MatOp, requant_fn
from .transformer import (
    CONFIGS, RES_MAX, SHIFT, _attention, _isqrt, _lut, _norm_int, build_decoder,
)

__all__ = ["build_opt_from_hf", "matches_benchmark_graph", "real_logits"]

_RELU = torch.tensor([max(0, x) for x in range(-128, 128)], dtype=torch.int64)


class _RealBuilder:
    """Like ``transformer._Builder``, but every tensor carries its real scale."""

    def __init__(self, tokens: torch.Tensor, pct: float, norm_gain: float, smooth: float | None = None,
                 smooth_pct: float = 99.9) -> None:
        self.norm_gain = norm_gain
        self.smooth, self.smooth_pct = smooth, smooth_pct
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
            s_w: float | None = None):
        """Real ``w [rows, cols]`` -> int8; the product's real scale is ``s_w * s_in``.  A per-channel input scale
        ``s_in`` (a smoothed norm's, a vector) is folded into ``w``'s columns first, then one scale per matrix."""
        n = self.name("mm")
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
        """A requantising consumer of a weight op (:func:`graph.requant_fn`, so V1 can cut it)."""
        return self.cheap(prefix, inputs, requant_fn(params), scale, note, {"requant": params})

    def to_int8(self, z):
        real = self.env[z].double().abs() * self.scale[z]
        top = float(torch.quantile(real.flatten()[:1 << 24], self.pct / 100.0)) if self.pct < 100 \
            else float(real.max())
        s_out = max(top, 1e-12) / INT8_MAX
        m = round((1 << SHIFT) * self.scale[z] / s_out)
        return self.requant("rq", [z], s_out, "requant", kind="requant", mult=m, shift=SHIFT, lo=-INT8_MAX,
                            hi=INT8_MAX)

    def residual(self, r, z):
        """r + z, with z brought to the residual's real scale (not to a target std)."""
        m = round((1 << SHIFT) * self.scale[z] / self.scale[r])
        return self.requant("res", [r, z], self.scale[r], "residual add", kind="residual", mult=m, shift=SHIFT,
                            res_max=RES_MAX)

    def norm(self, r, d, consumers=None):
        # _norm_int with gain G << 16 emits G * (normalised value), clamped to int8: real
        # scale 1/G, range +-127/G standard deviations. G is a public constant of the op,
        # so choosing it does not change any shape or cost.
        if self.smooth is None or consumers is None:
            gain = torch.full((d,), round(self.norm_gain * (1 << 16)), dtype=torch.int64)
            return self.cheap("norm", [r], lambda a, g=gain: _norm_int(a, g, True),
                              1.0 / self.norm_gain, "norm")
        # SmoothQuant through the gain vector (plan F2): channel j gets G_j = c / s_j, s_j = a_j^alpha / w_j^(1-alpha)
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
        return self.cheap("norm", [r], lambda a_, g=gain: _norm_int(a_, g, True),
                          (1 << 16) / gain.double(), "norm")


def _fold(w: torch.Tensor, b: torch.Tensor | None, gamma: torch.Tensor, beta: torch.Tensor):
    """``W (x*g + beta) + b  ==  (W diag g) x + (W beta + b)``."""
    w32 = w.double()
    wf = w32 * gamma.double()[None, :]
    bf = w32 @ beta.double() + (b.double() if b is not None else 0.0)
    return wf, bf


def build_opt_from_hf(model, calib_ids: torch.Tensor, *, pct: float = 100.0, norm_gain: float = 32,
                      residual_headroom: float = 1.25, device=None, smooth: float | None = None,
                      smooth_pct: float = 99.9, pct_att: float = 100.0) -> tuple[IntGraph, dict]:
    """An integer OPT decoder with the checkpoint's real weights.

    ``calib_ids`` ``[1, T]`` sets every activation scale. The three knobs are all
    public constants of cheap ops -- none changes a shape, so none changes the cost:

    * ``pct``: percentile of |value| mapped to 127 at each requantisation
      (100 = never clip on calibration data);
    * ``norm_gain``: the norm's output is ``norm_gain * normalised``, so its range is
      +-127/norm_gain standard deviations (``build_decoder`` uses 32);
    * the embed-to-residual multiplier is chosen so the embedding keeps full int8
      resolution (``build_decoder`` uses 8).

    ``device`` is where the integer calibration pass runs (the fp model stays where it is).
    """
    cfg = model.config
    assert cfg.do_layer_norm_before, "post-LN OPT-350M is not handled"
    assert cfg.word_embed_proj_dim == cfg.hidden_size, "project_in/out OPT-350M is not handled"
    dec = model.model.decoder
    d, n_heads = cfg.hidden_size, cfg.num_attention_heads
    dh = d // n_heads

    # The residual stream's real scale: large enough that the real residual never
    # clips at RES_MAX on calibration data. The embedding's int8 scale is then tied to
    # it by the graph's fixed "embed to residual" x8 op: s_embed = 8 * s_residual.
    with torch.no_grad():
        hs = model(calib_ids.to(model.device), output_hidden_states=True).hidden_states
    res_max = max(float(h.abs().max()) for h in hs) * residual_headroom
    s_res = max(res_max / RES_MAX, float(dec.embed_tokens.weight.abs().max()) / (INT8_MAX * 8))

    B = _RealBuilder(calib_ids.to(device or calib_ids.device), pct, norm_gain, smooth, smooth_pct)
    # Integer multiplier c with s_embed = c * s_residual, as close as possible to giving the
    # embedding its full int8 resolution (build_decoder hard-codes c = 8).
    embed_mult = max(1, round(float(dec.embed_tokens.weight.abs().max()) / INT8_MAX / s_res))
    tok = B.mat("x", dec.embed_tokens.weight.T.float(), None, 1.0, layout="embed",
                s_w=embed_mult * s_res)
    pos_ids = B.cheap("pos", ["x"], lambda x: torch.arange(x.shape[1], device=x.device)
                      .expand(x.shape[0], -1).contiguous(), 1.0, "position ids")
    # OPT's learned positions are offset by 2 rows.
    max_pos = cfg.max_position_embeddings
    pos_table = dec.embed_positions.weight[2:2 + max_pos].T.float()
    pos = B.mat(pos_ids, pos_table, None, 1.0, layout="embed")
    r = B.residual(B.cheap("scale", [tok], lambda a, c=embed_mult: a * c, s_res, "embed to residual"), pos)

    for layer in dec.layers:
        at = layer.self_attn
        ln1 = layer.self_attn_layer_norm
        folded = [_fold(proj.weight.float(), proj.bias, ln1.weight, ln1.bias)
                  for proj in (at.q_proj, at.k_proj, at.v_proj)]
        h = B.norm(r, d, [wf for wf, _ in folded])
        q, k, v = (B.to_int8(B.mat(h, wf.float(), bf, B.scale[h])) for wf, bf in folded)
        # HF scales q by dh^-0.5 inside the module; here that lives in m_s, as in build_decoder.
        sq, sk, sv = B.scale[q], B.scale[k], B.scale[v]
        m_s = round((1 << SHIFT) * 16.0 * sq * sk / math.sqrt(dh))
        raw = _attention(B.env[q], B.env[k], B.env[v], m_s, None, n_heads, n_heads, dh)
        s_raw = sv / 255.0                         # 8-bit probabilities x int8 values
        rabs = raw.double().abs().flatten()
        top = (float(torch.quantile(rabs[:1 << 24], pct_att / 100.0)) if pct_att < 100 else float(rabs.max())) * s_raw
        s_att = max(top, 1e-12) / INT8_MAX
        m_o = round((1 << SHIFT) * s_raw / s_att)
        att = B.cheap("attn", [q, k, v],
                      lambda a, b_, c, ms=m_s, mo=m_o: _attention(a, b_, c, ms, mo, n_heads, n_heads, dh),
                      s_att, "attention")
        r = B.residual(r, B.mat(att, at.out_proj.weight.float(), at.out_proj.bias, s_att))

        ln2 = layer.final_layer_norm
        wf, bf = _fold(layer.fc1.weight.float(), layer.fc1.bias, ln2.weight, ln2.bias)
        h = B.norm(r, d, [wf])
        f1 = B.to_int8(B.mat(h, wf.float(), bf, B.scale[h]))
        act = B.cheap("act", [f1], lambda a, t=_RELU: _lut(a, t), B.scale[f1], "relu")
        r = B.residual(r, B.mat(act, layer.fc2.weight.float(), layer.fc2.bias, B.scale[act]))

    lnf = dec.final_layer_norm
    wf, bf = _fold(model.lm_head.weight.float(), None, lnf.weight, lnf.bias)
    h = B.norm(r, d, [wf])
    h = B.cheap("last", [h], lambda a: a[:, -1:, :].contiguous(), B.scale[h], "last position")
    logits = B.mat(h, wf.float(), bf, B.scale[h])
    info = {"s_residual": s_res, "residual_real_max": res_max, "logit_scale": B.scale[logits],
            "pct": pct, "norm_gain": norm_gain, "embed_mult": embed_mult, "smooth": smooth,
            "smooth_pct": smooth_pct, "pct_att": pct_att}
    return IntGraph(B.ops, "x", logits), info


def real_logits(graph: IntGraph, info: dict, ids: torch.Tensor, *, all_positions: bool = False):
    """Run the integer graph and return real-valued logits.

    With ``all_positions`` the "last position" slice is skipped, giving logits at every
    position in one pass. That is exactly what one query per prefix would produce:
    every op is per-position except causal attention, and no constant depends on the
    input, so position t's integers depend only on tokens 0..t.
    """
    if all_positions:
        ops = [CheapOp(o.name, o.inputs, o.output, fn=(lambda a: a), note=o.note)
               if isinstance(o, CheapOp) and o.note == "last position" else o for o in graph.ops]
        graph = IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta))
    env, _ = graph.forward(ids)
    return env[graph.output_name].double() * info["logit_scale"]


def matches_benchmark_graph(graph: IntGraph, config_name: str) -> list[str]:
    """Differences from ``build_decoder``'s graph (op kinds, order, matrix shapes).

    Returns the list of mismatches; the only one expected is the LM head's bias.
    """
    ref = build_decoder(CONFIGS[config_name])
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
