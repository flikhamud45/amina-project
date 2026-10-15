"""The tiny decoder shapes of the test suites, each defined once and named after its shape.

Where a shape differs from the rehearsals' shape of the same family (``experiments.common.models.TINY``), its
name says how: GPT and OPT have 64 positions here (``-64``, against 256), and ``tiny-opt-proj`` adds OPT-350M's
token embedding in 32 dimensions projected to the model's 64.  ``tiny-gpt2`` has a wider MLP (256) and the default
2,048 positions.  ``tiny-llama`` and ``tiny-qwen`` are the rehearsals' shapes.
"""

from pvi.models.decoder import DecoderConfig

GPT = DecoderConfig("tiny-gpt-64", 64, 2, 4, 4, 16, 128, 97, max_pos=64)
GPT2 = DecoderConfig("tiny-gpt2", 64, 2, 4, 4, 16, 256, 97)
OPT = DecoderConfig("tiny-opt-64", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=64)
OPT_PROJ = DecoderConfig("tiny-opt-proj", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=64, embed_dim=32)
LLAMA = DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope", bias=False,
                      tied=False)
QWEN = DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                     rope_theta=1e6, bias=False, qk_norm=True)                # GQA, q/k norm
