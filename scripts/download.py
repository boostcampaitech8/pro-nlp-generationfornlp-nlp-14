import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

target = Path("/data/ephemeral/home/.cache/huggingface")

if target.exists():
    shutil.rmtree(target)

REPO_ID = "Qwen/Qwen3-32B-GGUF"
FILENAME = "Qwen3-32B-Q6_K.gguf"

path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, resume_download=True)
print(path)
