import os

# HF via FlClash proxy (falls back gracefully if proxy down -> script retries)
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
from huggingface_hub import snapshot_download

p = snapshot_download(
    repo_id="ltnghia/DNS-Challenge",
    repo_type="dataset",
    allow_patterns=["datasets/noise/*"],
    local_dir=r"D:\data\dns-challenge-mirror",
    max_workers=6,
)
print("SNAPSHOT DONE:", p)
