"""Shared helpers for the quality experiments: model and data loading, perplexity scoring.

The windows are those of ``experiments/2_attack/real_weights_ppl.py``: WikiText-2 (raw) test,
non-overlapping windows of ``seq`` tokens from the start of the split; calibration windows are
taken from the END of the split (disjoint from every evaluation window as long as
``(windows + calib_windows) * seq`` is below the split's length, about 287k tokens).
"""

from __future__ import annotations

import glob
import math
import os

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

DEFAULT_PARQUET = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-00000-of-00001.parquet")


def find_parquet(path: str | None = None) -> str:
    if path:
        return path
    hits = glob.glob(DEFAULT_PARQUET)
    if not hits:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id="Salesforce/wikitext", filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
                               repo_type="dataset")
    return hits[0]


def load_opt(name: str):
    """OPT in fp32, eval mode.  transformers >= 4.5x refuses ``torch.load`` under torch < 2.6, and the
    official OPT repos ship ``pytorch_model.bin`` only, so the state dict is loaded here with
    ``weights_only=True`` (no pickle code runs) and handed to a model built from the config."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM
    path = snapshot_download(name, allow_patterns=["*.json", "*.bin", "*.txt"])
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):                     # no second copy of the weights while loading
        model = OPTForCausalLM(cfg)
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    sd = {(k if k.startswith(("model.", "lm_head.")) else "model." + k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del sd
    missing = [m for m in missing if m != "lm_head.weight"]          # tied to the token embedding
    assert not missing and not unexpected, (missing, unexpected)
    model.tie_weights()
    for p_ in model.parameters():                   # fp16 checkpoints -> fp32, one tensor at a time
        p_.data = p_.data.float()
    model = model.eval()
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def load_windows(tok, parquet: str | None = None, *, seq: int = 128, windows: int = 16, calib_windows: int = 1):
    """``(eval windows, calibration windows)``: lists of ``[1, seq]`` id tensors."""
    text = "".join(pq.read_table(find_parquet(parquet)).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert (windows + calib_windows) * seq < ids.numel()
    wins = [ids[i * seq:(i + 1) * seq].unsqueeze(0) for i in range(windows)]
    # calibration window j = the j-th window counted back from the end (j = 0 is real_weights_ppl.py's)
    n = ids.numel()
    calib = [ids[n - (j + 1) * seq:n - j * seq].unsqueeze(0) for j in range(calib_windows)]
    return wins, calib


class Scorer:
    """Perplexity and top-1 agreement (with a reference's argmaxes) over fixed windows."""

    def __init__(self, wins):
        self.wins = wins
        self.ref_argmax = None

    def score(self, logits_fn):
        tot, n, am = 0.0, 0, []
        for w in self.wins:
            lg = logits_fn(w)[0, :-1].double()
            tgt = w[0, 1:]
            tot += float(F.cross_entropy(lg, tgt, reduction="sum"))
            n += tgt.numel()
            am.append(lg.argmax(-1))
        ppl = math.exp(tot / n)
        agree = None
        if self.ref_argmax is not None:
            agree = sum(int((a == b).sum()) for a, b in zip(am, self.ref_argmax)) / sum(a.numel() for a in am)
        return ppl, agree, am
