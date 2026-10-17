"""Runs inside `docker build`: stops with a clear message if the trained models are missing,
then downloads the Gemma base model into the image so nothing is downloaded at run time."""
import os, sys
from pathlib import Path

M = Path(os.environ.get("HG_MODELS_DIR", "/models"))
need = {
    "Gemma adapter": M / "gemma_cpu_lora" / "adapter_config.json",
    "sahajBERT model": M / "sahajbert" / "config.json",
}
bad = [f"  - {n}: expected {p.relative_to(M.parent)}" for n, p in need.items() if not p.exists()]
if bad:
    sys.exit("\n\nBUILD STOPPED: your trained models are missing from the models/ folder next to the Dockerfile.\n"
             + "\n".join(bad)
             + "\nCopy your gemma_cpu_lora folder to models/gemma_cpu_lora and the contents of sahajbert/model "
               "to models/sahajbert, then run the build again. See models/README.txt.\n")

base_file = M / "gemma_cpu_lora" / "base_model.txt"
base = base_file.read_text().strip() if base_file.exists() else os.environ.get("HG_GEMMA_BASE", "unsloth/gemma-3-270m-it")
print("downloading Gemma base model into the image:", base, flush=True)
from huggingface_hub import snapshot_download
snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])
print("done")
