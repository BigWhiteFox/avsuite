import os

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
from huggingface_hub import snapshot_download

# 1) ltnghia mirror clean speech (~10GB, 10007 wavs)
p1 = snapshot_download(
    repo_id="ltnghia/DNS-Challenge", repo_type="dataset",
    allow_patterns=["datasets/clean/*"],
    local_dir=r"D:\data\dns-challenge-mirror", max_workers=6,
)
print("CLEAN DONE:", p1)

# 2) DNS-2020 dev test set (small, for eval)
p2 = snapshot_download(
    repo_id="nkdem/DNS-Challenge-2020-DevTest-16k", repo_type="dataset",
    local_dir=r"D:\data\dns2020-devtest", max_workers=4,
)
print("DEVTEST DONE:", p2)
