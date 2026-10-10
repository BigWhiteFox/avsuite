"""数据集就绪度检查（WP-00 骨架，report-only，不改任何文件）。

批次2/3/4 启动前跑一遍确认：
  1. VoiceBank-DEMAND 16k    D:/data/voicebank-demand-16k              6 个 parquet
  2. DNS-Challenge 子集      D:/data/dns-challenge-mirror/datasets     clean/noise 计数
  3. AEC-Challenge 测试集    D:/data/aec-challenge/test_set_icassp2022 三场景 vs .filelist
  4. RIR（WP-07D 合成用）    D:/data/repos/AEC-Challenge/.../impulse_responses
"""
import sys
from pathlib import Path

VB = Path("D:/data/voicebank-demand-16k")
DNS = Path("D:/data/dns-challenge-mirror/datasets")
AEC_REPO = Path("D:/data/repos/AEC-Challenge")
AEC_TESTSET = AEC_REPO / "datasets/test_set_icassp2022"   # 仓库克隆内 = 权威源
RIRS = AEC_REPO / "datasets/RIRs"
VB_EXPECTED = [f"train-{i:05d}-of-00005.parquet" for i in range(5)] + ["test-00000-of-00001.parquet"]


def mb(p: Path) -> float:
    return p.stat().st_size / 1024 ** 2


def check_voicebank():
    print("\n[1] VoiceBank-DEMAND 16k（WP-03，~2GB）")
    if not VB.exists():
        print("  MISS 目录不存在")
        return
    for name in VB_EXPECTED:
        p = VB / name
        flag = "OK " if p.exists() and p.stat().st_size > 1e6 else "MISS"
        size = f"{mb(p):8.1f}MB" if p.exists() else "         -"
        print(f"  {flag} {name:36s} {size}")
    extra = [p.name for p in VB.glob("*.parquet") if p.name not in VB_EXPECTED]
    if extra:
        print(f"  WARN 额外文件: {extra}")


def check_dns():
    print("\n[2] DNS-Challenge 子集（WP-04）")
    if not DNS.exists():
        print("  MISS 目录不存在")
        return
    for sub in sorted(p.name for p in DNS.iterdir() if p.is_dir()):
        n = sum(1 for p in (DNS / sub).rglob("*") if p.is_file())
        print(f"  ..  {sub:24s} {n:6d} 个文件（参考满额：clean 10000 / noise 660）")


def check_aec():
    print("\n[3] AEC-Challenge（WP-06/07，仓库克隆内 = 权威源）")
    if not AEC_TESTSET.exists():
        print(f"  MISS 测试集不存在: {AEC_TESTSET}")
        return
    for s in ("doubletalk", "farend-singletalk", "nearend-singletalk"):
        n = sum(1 for p in (AEC_TESTSET / s).rglob("*.wav"))
        mark = "OK " if n in (600, 400) else ".."
        print(f"  {mark} {s:22s} {n} wavs（满额参考：FEST/DT 600、NST 400）")
    print(f"  {'OK ' if RIRS.exists() else 'MISS'} RIR（WP-07D 回声合成用）: {RIRS}")
    print(f"  {'OK ' if (AEC_REPO / 'AECMOS').exists() else 'MISS'} AECMOS 官方脚本: {AEC_REPO / 'AECMOS'}")
    extra = Path("D:/data/aec-challenge/test_set_icassp2022")
    if extra.exists():
        print(f"  NOTE 单独下载的副本 {extra} 与克隆内容重复，可删除省空间")


if __name__ == "__main__":
    check_voicebank()
    check_dns()
    check_aec()
    print("\n（report-only，退出码恒 0）")
    sys.exit(0)
