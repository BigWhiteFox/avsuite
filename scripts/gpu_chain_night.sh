#!/usr/bin/env bash
# GPU 夜跑接力链（2026-10-09 晚挂起，目标：明早全项目验收）
#   1. DCCRN 续训 400ep（val 仍在降）→ devtest+synth 评估
#   2. DCCRN --no-dm 消融 100ep → 评估
#   3. DCCRN --real-conv 消融 100ep → 评估
#   4. NN-AEC 续训（w-sisnr 0.5）300ep → 对比表 + 时延扫描 + AECMOS
#   5. NN-AEC γ=0 消融 300ep → 评估
#   6. 生成验收报告 docs/ACCEPTANCE.md
# 每步独立容错（失败不中断链），全部 checkpoint 化，可随时 kill 后手动 resume。
set -u
export PYTHONNOUSERSITE=1
PY=/c/Users/14805/.conda/envs/avsuite/python.exe
cd /d/Project_for_Python/Zcoder/avsuite
LOG=outputs_night.log
echo "[night $(date '+%F %T')] 夜跑接力链启动" | tee -a $LOG

step() { echo "[night $(date '+%T')] $1" | tee -a $LOG; }
run() { "$@" >> $LOG 2>&1 || { echo "[night $(date '+%T')] FAILED: $*" | tee -a $LOG; }; }

# ---------- 1. DCCRN 续训 ----------
step "DCCRN 续训（resume, 400ep, early-stop 20）"
run $PY -s -u exp/04_dccrn/train.py --device cuda --epochs 400 --epoch-len 1500 \
  --val-len 64 --batch-size 8 --num-workers 4 --early-stop 20 \
  --out exp/04_dccrn/outputs --resume exp/04_dccrn/outputs/dccrn/last.ckpt
run $PY -s exp/04_dccrn/eval.py --ckpt exp/04_dccrn/outputs/dccrn/best.ckpt --devtest --num 150
run $PY -s exp/04_dccrn/eval.py --ckpt exp/04_dccrn/outputs/dccrn/best.ckpt --synth 100

# ---------- 2. DCCRN --no-dm 消融 ----------
step "DCCRN 消融 --no-dm"
run $PY -s -u exp/04_dccrn/train.py --device cuda --epochs 100 --epoch-len 1500 \
  --val-len 64 --batch-size 8 --num-workers 4 --early-stop 12 \
  --out exp/04_dccrn/outputs_abl --no-dm
run $PY -s exp/04_dccrn/eval.py --ckpt exp/04_dccrn/outputs_abl/dccrn_nodm/best.ckpt --devtest --num 150

# ---------- 3. DCCRN --real-conv 消融 ----------
step "DCCRN 消融 --real-conv"
run $PY -s -u exp/04_dccrn/train.py --device cuda --epochs 100 --epoch-len 1500 \
  --val-len 64 --batch-size 8 --num-workers 4 --early-stop 12 \
  --out exp/04_dccrn/outputs_abl --real-conv
run $PY -s exp/04_dccrn/eval.py --ckpt exp/04_dccrn/outputs_abl/dccrn_realconv/best.ckpt --devtest --num 150

# ---------- 4. NN-AEC 续训（近端保真） ----------
step "NN-AEC 续训（w-sisnr 0.5, lr 3e-4, 300ep）"
run $PY -s -u exp/07_nn_aec/train.py --device cuda --epochs 300 --batch-size 16 \
  --lr 3e-4 --w-sisnr 0.5 --num-workers 4 --early-stop 30 \
  --out exp/07_nn_aec/outputs_v3 --resume exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/last.ckpt
run $PY -s exp/07_nn_aec/eval.py --ckpt exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/best.ckpt --compare-trad
run $PY -s exp/07_nn_aec/eval.py --ckpt exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/best.ckpt --jitter-sweep
run $PY -s exp/07_nn_aec/aecmos_eval.py --nn-ckpt exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/best.ckpt --out exp/07_nn_aec/outputs/aecmos_nn_night.json

# ---------- 5. NN-AEC γ=0 消融 ----------
step "NN-AEC γ=0 消融（300ep fresh）"
run $PY -s -u exp/07_nn_aec/train.py --device cuda --epochs 300 --batch-size 16 \
  --lr 3e-4 --w-sisnr 0.5 --gamma 0 --num-workers 4 --early-stop 30 \
  --out exp/07_nn_aec/outputs_g0
run $PY -s exp/07_nn_aec/eval.py --ckpt exp/07_nn_aec/outputs_g0/nnaec_g0_j10/best.ckpt --compare-trad --out exp/07_nn_aec/outputs/eval/g0

# ---------- 6. 验收报告 ----------
step "生成验收报告"
run $PY -s tools/acceptance_report.py
step "夜跑接力链全部完成"
