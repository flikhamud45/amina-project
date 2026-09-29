"""Layer-0 input for the zkLLM demo (ours): the embeddings of the first SEQ tokens of
WikiText-2 test, in zkLLM's fixed-point format (int32, scale 2^16)."""
import sys
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fileio_utils import save_int

size, seq, parquet, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
card = f"meta-llama/Llama-2-{size}b-hf"
tok = AutoTokenizer.from_pretrained(card, local_files_only=True, cache_dir="./model-storage")
model = AutoModelForCausalLM.from_pretrained(card, local_files_only=True, cache_dir="./model-storage")
text = "".join(pq.read_table(parquet).column("text").to_pylist())
ids = tok(text, return_tensors="pt").input_ids[:, :seq]
assert ids.shape[1] == seq
with torch.no_grad():
    x = model.model.embed_tokens(ids)[0].float()
save_int(x, 1 << 16, out)
print("input", tuple(x.shape), "->", out)
