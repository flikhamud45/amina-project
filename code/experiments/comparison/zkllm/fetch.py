"""Download what zkLLM's demo reads (run with zkLLM's environment; ``run_zkllm.sh build`` calls it): the Llama-2
checkpoints (``meta-llama/Llama-2-<size>b-hf``: configuration, tokenizer and safetensors weights; gated: the
account of ``HF_TOKEN`` needs access) into the Hugging Face hub cache that zkLLM's ``./model-storage`` is, and
WikiText-2's raw test split (``Salesforce/wikitext``, ``wikitext-2-raw-v1``, one parquet file), copied to
``--wikitext-out``.  Files already in the cache are not downloaded again.

    fetch.py --cache <hub cache> --models 7 13 --wikitext-out <file.parquet>"""
import argparse
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")
PATTERNS = ["*.json", "*.safetensors", "tokenizer.model"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="the Hugging Face hub cache (zkLLM's ./model-storage)")
    ap.add_argument("--models", nargs="*", default=["7", "13"], choices=["7", "13"], help="Llama-2 sizes")
    ap.add_argument("--wikitext-out", required=True, help="where the WikiText-2 test split's parquet file goes")
    args = ap.parse_args()
    for size in args.models:
        path = snapshot_download(f"meta-llama/Llama-2-{size}b-hf", cache_dir=args.cache, allow_patterns=PATTERNS)
        print(f"Llama-2-{size}B: {path}", flush=True)
    path = hf_hub_download(WIKITEXT[0], WIKITEXT[1], repo_type="dataset", cache_dir=args.cache)
    shutil.copyfile(path, args.wikitext_out)
    print(f"WikiText-2 test split: {args.wikitext_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
