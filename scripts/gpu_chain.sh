#!/usr/bin/env bash
# GPU 接力链（2026-10-09 下午挂起）：
#   1. 等 DCCRN 训练结束（train.log 出现 [done]）
#   2. 自动跑 DCCRN devtest + synth 评估
#   3. 自动启动 NN-AEC 夜训（60 epochs，early-stop 8）
# 所有产物带 checkpoint，可 Ctrl-C / kill 后 --resume。
set -u
export PYTHONNOUSERSITE=1
PY=/c/Users/14805/.conda/envs/avsuite/python.exe
cd /d/Project_for_Python/Zcoder/avsuite

echo "[chain $(date '+%T')] 等待 DCCRN 训练结束..."
while ! grep -q "\[done\]" exp/04_dccrn/outputs/train.log 2>/dev/null; do sleep 60; done
echo "[chain $(date '+%T')] DCCRN 训练结束，开始评估"

$PY -s exp/04_dccrn/eval.py --ckpt exp/04_dccrn/outputs/dccrn/best.ckpt --devtest --num 150 \
  >> exp/04_dccrn/outputs/eval_devtest.log 2>&1
echo "[chain $(date '+%T')] devtest 评估完成"
$PY -s exp/04_dccrn/eval.py --ckpt exp/04_dccrn/outputs/dccrn/best.ckpt --synth 100 \
  >> exp/04_dccrn/outputs/eval_synth.log 2>&1
echo "[chain $(date '+%T')] synth 评估完成，启动 NN-AEC 训练"

$PY -s -u exp/07_nn_aec/train.py --device cuda --epochs 60 --batch-size 16 \
  --num-workers 4 --early-stop 8 > exp/07_nn_aec/outputs/train.log 2>&1
echo "[chain $(date '+%T')] NN-AEC 训练结束"
