# AECMOS 采样率修复 + 自合成集全量重评分（R1 / t19-P0-1）

- 日期：2026-10-11 ｜ 执行：WS-V5 / data-eng（t20）｜ 依据：`reports/refactor_wave5/triage_v3.md` §2 P0-1（blocker）+ 队长追加的同族 mic/enh 传参缺陷
- 结论一句话：**修复已落地并通过单点验证；自合成集全部 AECMOS 数字需按新口径替换**——DT 层 echo/deg 系统性**上升**、FEST 层 echo 系统性**下降**（旧偏差不是单向均匀量），且**"NN 近端保真（NST deg）结构性落后"这条结论在新口径下翻转**（7/7 集），其余三条主结论方向与显著性保持。
- 纪律：本轮**不改** `docs/NUMBERS.md`、`CHEATSHEET.md` 与任何既有报告/json；数字回填归队长。所有新产物落新目录。

---

## 1. 缺陷与修复

### 1.1 主缺陷（P0-1，blocker）：16k 波形送 48k AECMOS 模型

`AECMOS_local/aecmos.py:45-50` 对 `Run_1668423760_Stage_0.onnx` 设 `sampling_rate=48000`、`dft_size=1536`；官方入口 `read_and_process_audio_files()` 用 `librosa.load(path, sr=self.sampling_rate)` 统一重采样。但 `exp/07_nn_aec/aecmos_eval.py` 用 `sf.read`（原生 16k）直接调 `est.run()`，**绕过重采样** ⇒ mel 频谱按 48k 解释 16k 波形（`_mel_transform(sample, sr=48000)`，`n_fft=1537`），全部自合成集分数失真。

**修复**（`exp/07_nn_aec/aecmos_eval.py`）：

```python
def resample_to_model_sr(x, sr_in, sr_out):
    """与 t19 auditor 探针 _t19_p0_aecmos.py:53 同法（gcd 约简 + resample_poly）。"""
    if int(sr_in) == int(sr_out):
        return x
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)
```

- 在 `score_all()` 中对 **lpb/mic/enh 三路**统一重采样到 `est.sampling_rate`，**送评前三路等长断言保留**（并显式给出不等长时的报错信息）。
- 输入采样率自动探测（`sf.info(farend wav).samplerate`），可用 `--in-sr` 覆盖。
- 输出 JSON 增加口径元数据：`aecmos_model` / `aecmos_sr_model` / `input_sr` / `resample`，便于回溯。
- 与官方口径统一为「**AECMOS@48k 模型协议**」。

**同类传参缺陷（队长追加，已一并修）**：`exp/10.5_capstone/cascade.py` 的 `aecmos_scores()` 旧实现 `est.run(talk_type, lpb[:n], sig[:n], sig[:n])` 把**同一路信号同时当 mic 与 enh**（官方签名 `run(talk_type, lpb, mic, enh)`，`mic` 应为**未处理原始 mic**、`enh` 为本级输出）⇒ 分阶段 deg 分数恒等于"零处理"参照。已改为 `mic_raw = signals["mic"]`，三路 `n = min(len(lpb), len(mic_raw), len(sig))` 截齐后分别传入；同时把 AECMOSEstimator 改为**进程内单例**（不再每请求重建）。

**核对**：`aecmos_eval.py` 三处 `est.run`/pairs 组装（`--trad`/`--nn-ckpt`/`--enh-dir`）传参均为 `(lpb, mic, enh)`，**无同款 mic/enh 同源问题**（`pairs.append((name, sc, lpb, mic, out/enh))`）。`exp/10.5_capstone/demo_app.py` 的 `v[0]/v[1]` 解包经实测**可正常工作**（`aecmos_scores` 返回 tuple），triage v3b-054 的"float 不可下标"在现行工作树**不可复现**；仍按同族做**容错加固**：新增 `_fmt_score()` 支持标量形态，并把评分失败与音频处理失败**拆成两阶段**（评分失败只降级文本，三级音频照常返回，不再被外层 except 吞掉）。

### 1.2 单点验证（复跑 auditor 口径对照）

同一内容（eval_v3 前 2 条），三路口径：

| item | 场景 | (a) 16k 直送（修复前） | (b) 升采样（官方/auditor） | (c) 16k 模型 | **修复后 score_all** | ==(b)? |
|---|---|---|---|---|---|---|
| fileid_0_dt | dt | echo 2.459 / deg 2.763 | **echo 3.812 / deg 3.670** | echo 4.078 / deg 3.329 | echo 3.812 / deg 3.670 | ✅ 逐位相同 |
| fileid_1_fest | fest | echo 2.417 / deg 5.000 | **echo 3.070 / deg 4.998** | echo 3.279 / deg 5.000 | echo 3.070 / deg 4.998 | ✅ 逐位相同 |

- 与 t19 报告数值完全一致（dt 2.459/3.812/4.078），说明修复复现了 auditor 判定的口径。
- `gcd(16000,48000)=16000 → up=3, down=1`，即 `resample_poly(x,3,1)`——与 auditor 的 `up` 逐位同源。
- **结论：工作树行为 = 升采样口径（PASS）**。

### 1.3 官方 48k 源集链**未受影响**（已核）

| 脚本 | 送评前加载 | 判定 |
|---|---|---|
| `exp/07_nn_aec/official_aecmos_eval.py:291-303` | `librosa.load(clip[...], sr=SR_AECMOS)`，`SR_AECMOS=48000`（:68） | ✅ 已正确重采样，不受影响 |
| `exp/07_nn_aec/ood_real_echoes_eval.py:148-160` | `librosa.load(clip[...], sr=48000)` | ✅ 同上 |

⇒ `docs/NUMBERS.md` 的**官方集 800 段**表（3.256/3.143 等）与 `reports/ood_real_echoes.md`、`reports/official_aecmos.md` **无需改动**；受影响的只有走 `aecmos_eval.py` 的**自合成 16k 集**与 capstone。

---

## 2. 重评分战役（修复后口径，8 集 × 三方）

- 管线与 `data_inventory.md` §5.4 完全同配方，**只换 AECMOS 口径**：trad = `TradAEC(mdf,8,0.7)` 全链；NN = `outputs_v4/nnaec_g0.5_j10/best.ckpt`；级联 = `precompute_cascade` 同款线性级 + `09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt`。
- 产物：`exp/07_nn_aec/outputs/aecmos48_v5fix/<set>/{trad,nn,cascade}.json`（**全部新文件，未覆盖任何既有 json**，含口径元数据与逐条 rows）。
- 级联增强 wav：eval_v3/v5/v6/v7 复用既有 enh；其余 4 集新生成于 `D:/data/aec-synth/cascade_stage/v5fix/<set>/enh_cascade`（线性级同目录 `*_lin`），`lin-check mismatch=0`。
- **耗时 18.9 min**（3 个 arm × 8 集 = 24 次评分，全部 exit=0，**零拒评/零失败**）。CI 用 `tools/compute_ci.py` 的 importlib 复用方式（B=2000/seed=777），**未运行其 main**。

---

## 3. 新旧对照（逐集逐层 echo_mos：旧 16k 直送 → 新 @48k）

| 集 | 层 | 传统 旧→新 | NN v4 旧→新 | 级联 旧→新 |
|---|---|---|---|---|
| eval_v5 | fest | 4.002 → **3.518** (−0.484) | 3.956 → **3.881** (−0.075) | 4.589 → **4.196** (−0.393) |
| eval_v5 | nst | 5.000 → 4.998 | 4.999 → 4.998 | 5.000 → 4.998 |
| **eval_v5** | **dt** | 2.525 → **3.442** (**+0.918**) | 3.721 → **4.248** (**+0.527**) | 3.741 → **4.317** (**+0.576**) |
| eval_v7 | fest | 3.879 → **3.543** (−0.336) | 3.749 → **3.595** (−0.154) | 4.401 → **4.157** (−0.244) |
| eval_v7 | nst | 5.000 → 4.998 | 4.999 → 4.998 | 5.000 → 4.998 |
| **eval_v7** | **dt** | 2.449 → **3.424** (**+0.974**) | 3.615 → **4.223** (**+0.608**) | 3.703 → **4.291** (**+0.587**) |
| eval_v3 | fest | 3.836 → **3.207** (−0.628) | 3.762 → **3.194** (−0.568) | 4.502 → **3.938** (−0.564) |
| eval_v3 | dt | 2.467 → **3.494** (+1.027) | 3.754 → **4.175** (+0.421) | 3.919 → **4.261** (+0.342) |
| eval_ood_delay | dt | — → 2.147 | 1.919 → **2.671** (+0.753) | — → 2.853 |
| eval_v6_realrir | dt | 2.000 → **2.769** (+0.769) | 2.565 → **3.504** (+0.939) | 2.627 → **3.500** (+0.873) |
| eval_v3_drift | dt | 1.649 → **2.120** (+0.471) | 3.802 → **4.227** (+0.425) | — → 3.321 |
| eval_v4_speaker_heldout | dt | 2.066 → **3.433** (+1.367) | 3.213 → **4.161** (+0.948) | — → 4.222 |
| eval_speaker_only_holdout | dt | 2.517 → **3.482** (+0.965) | 3.777 → **4.209** (+0.432) | — → 4.268 |

**层相关偏差（新 − 旧）汇总**：

| 集 | DT echo 偏差（trad / NN / 级联） | FEST echo 偏差（trad / NN / 级联） |
|---|---|---|
| eval_v5 | +0.918 / +0.527 / +0.576 | −0.484 / −0.075 / −0.393 |
| eval_v7 | +0.974 / +0.608 / +0.587 | −0.336 / −0.154 / −0.244 |
| eval_v3 | +1.027 / +0.421 / +0.342 | −0.628 / −0.568 / −0.564 |
| eval_v6_realrir | +0.769 / +0.939 / +0.873 | −0.285 / −0.356 / −0.257 |
| eval_v3_drift | +0.471 / +0.425 | −0.241 / −0.608 |
| eval_v4 / speaker_only | +1.367 / +0.965（trad） | −0.653 / −0.474（trad） |

> **重要修正**：t19 的表述「全部 echo_mos 被系统性压低 −0.65~−1.35」只在 **DT 层**成立（旧值偏低，新值升高 +0.34~+1.37）；**FEST 层方向相反**（旧值偏高，新值下降 −0.08~−0.65）。NST echo 两侧都饱和在 ≈5.0。旧偏差是**层/指标相关**的，不是一个可整体平移的常数。

### 3.1 deg_mos 旧→新（口径变化同样巨大，且以"上升"为主）

| 集 | NST deg（trad/NN/级联） | DT deg（trad/NN/级联） |
|---|---|---|
| eval_v5 | 2.645→**3.776** / 2.091→**3.827** / 2.062→**3.750** | 2.430→**3.590** / 1.811→**3.220** / 1.851→**3.375** |
| eval_v7 | 2.579→**3.829** / 2.012→**3.876** / 2.010→**3.846** | 2.376→**3.590** / 1.864→**3.269** / 1.877→**3.451** |
| eval_v3 | 2.665→**3.842** / 2.073→**3.941** / 2.081→**3.851** | 2.401→**3.598** / 1.743→**3.237** / 1.785→**3.250** |
| eval_v6_realrir | 2.588→3.792 / 2.051→3.905 / 2.068→3.770 | 2.518→3.951 / 2.044→3.571 / 1.842→3.453 |
| 两个 speaker 集 | +1.14~+1.99（trad/NN） | +1.24~+1.45 |

FEST deg 两侧都饱和 ≈5.0（无判别力）。**deg_mos 的绝对尺度整体上移约 +1.0~+2.0**，旧 deg 绝对值（如 1.811）不可再与新值并列。

---

## 4. 三方相对差：新口径下方向与显著性是否保持

**判定总表**（✓=bootstrap/解析 CI 不含 0；旧值取既有 json）：

| 相对差 | 集 | 旧口径 | **新口径** | 判定 |
|---|---|---|---|---|
| NN − trad, DT echo | eval_v5 | +1.196±0.111 ✓ | **+0.806±0.087 ✓** | ✅ 保持（幅度 −33%） |
| NN − trad, DT echo | eval_v7 | +1.166±0.078 ✓ | **+0.799±0.057 ✓** | ✅ 保持（幅度 −31%） |
| NN − trad, DT echo | eval_v3 | +1.287±0.397 ✓ | **+0.681±0.212 ✓** | ✅ 保持 |
| NN − trad, DT deg | eval_v5 | −0.619±0.071 ✓ | **−0.369±0.089 ✓** | ✅ 保持（幅度 −40%） |
| NN − trad, DT deg | eval_v7 | −0.512±0.046 ✓ | **−0.322±0.055 ✓** | ✅ 保持 |
| **NN − trad, NST deg** | eval_v5 | −0.554±0.077 ✓ | **+0.051±0.086 ✗** | ❌ **翻转** |
| **NN − trad, NST deg** | eval_v7 | −0.567±0.050 ✓ | **+0.047±0.057 ✗** | ❌ **翻转** |
| 级联 − NN, FEST echo | eval_v5 | +0.633±0.189 ✓ | **+0.315±0.203 ✓** | ✅ 保持（幅度 −50%） |
| 级联 − NN, FEST echo | eval_v7 | +0.653±0.145 ✓ | **+0.562±0.128 ✓** | ✅ 保持 |
| 级联 − NN, DT echo | eval_v5 | +0.020±0.090 ✗ | **+0.069±0.043 ✓** | ⚠️ 由 n.s. 变边缘显著（幅度仍极小） |
| 级联 − NN, DT echo | eval_v7 | +0.088±0.064 ✓ | **+0.068±0.031 ✓** | ⚠️ 同前，量级 ≤0.07 |
| 级联 − trad, DT echo | eval_v5 / v7 | +1.217 / +1.254 ✓ | **+0.875 / +0.867 ✓** | ✅ 保持 |

### 4.1 有意义的"变号"清单（排除 |Δ|≈0 的饱和噪声后）

| 集 | 相对差 | 旧 | 新 |
|---|---|---|---|
| eval_v5 | NN−trad, FEST echo | −0.046±0.281 ✗ | **+0.363±0.194 ✓** |
| eval_v7 | NN−trad, FEST echo | −0.130±0.189 ✗ | +0.052±0.146 ✗ |
| **eval_v5 / v7 / v3 / v6 / drift / speaker_only** | **NN−trad, NST deg** | −0.44~−0.68 ✓ | **+0.05~+0.11 ✗** |
| eval_v4_speaker_heldout | NN−trad, NST deg | −0.273±0.131 ✓ | **+0.144±0.100 ✓**（转正且显著） |

> NST echo / FEST deg 上出现的若干"变号"是 **4.998↔5.000 的饱和噪声**（|Δ|<0.002），不构成结论变化，已在报告与 JSON 中标注。
> `eval_v5` FEST `NN−trad` echo 由 n.s. 变显著正（+0.363），但 `eval_v7` 同项仍 n.s.（+0.052）⇒ **跨集不一致，不作为新结论主张**。

### 4.2 结论级影响（供队长裁决）

1. **必须替换（口径错误，全部自合成集）**：所有走 `aecmos_eval.py` 得到的 echo/deg 数字——`docs/NUMBERS.md` 的 eval_v3/eval_v5/eval_v7/drift/ood_delay/v6/两个 speaker held-out 小节的绝对值与 CI、`reports/DATA_CI.md` 的对应行、`reports/eval_v5_dt300.md`、`reports/eval_v7_fest_nst.md`、`reports/drift_sweep.md`、`reports/drift_retrain.md`、`reports/realrir_domain.md`（v6 部分）、`reports/deg_mechanism.md`（v5 存档比对）、capstone README 的 AECMOS 示例值。
2. **无需改动（官方 48k 链，已核）**：`official_aecmos.md`、`ood_real_echoes.md`、NUMBERS 官方 800 段表、级联官方/v6 报告里走 48k 加载的部分。
3. **结论变化**：
   - **翻转**：「NN 近端保真（NST deg）结构性落后」**不再成立**（7/7 集：旧 −0.44~−0.68 显著 → 新 +0.05~+0.14，多数 n.s.，held-out 集甚至转正且显著）。这条是 NUMBERS 中"deg 结构性瓶颈"叙事的主证据之一，需重写。
   - **保持**：NN DT echo 优势（+0.80 左右，仍显著）、NN DT deg 落后（−0.32~−0.37，仍显著）、级联 FEST echo 最优（+0.32~+0.56，仍显著）、级联≈NN on DT echo（差值 ≤0.07）。
   - **建议的新叙事**：deg 的问题应从"NN 全面落后"收窄为"**DT 段 NN deg 落后（−0.32~−0.37）**"；NST/近端保真维度在新口径下**没有落后证据**。
4. **旧数字处置**：建议 NUMBERS 保留旧值**仅作历史脚注**（一行说明"旧值为 16k→48k 模型错配口径，已作废，见 `reports/aecmos_rescore_v5fix.md`"），主表全部换成新值；CI 一并替换。**不要**新旧混排。

---

## 5. 产物与纪律

| 类型 | 路径 |
|---|---|
| 修复 | `exp/07_nn_aec/aecmos_eval.py`（重采样 + 口径元数据 + `--in-sr`）、`exp/10.5_capstone/cascade.py`（mic/enh 传参 + 重采样 + 估计器单例）、`exp/10.5_capstone/demo_app.py`（两阶段容错 + `_fmt_score`） |
| 新评分（24 份） | `exp/07_nn_aec/outputs/aecmos48_v5fix/<set>/{trad,nn,cascade}.json` |
| 分析与驱动 | `exp/07_nn_aec/outputs/aecmos48_v5fix/{_campaign.py,_analyze.py,_analysis_v5fix.json}` |
| 级联增强（新生成 4 集） | `D:/data/aec-synth/cascade_stage/v5fix/<set>/{enh_cascade,*_lin}` |
| 本文 | `reports/aecmos_rescore_v5fix.md`（新文件） |

- **未改动**：`docs/NUMBERS.md`、`docs/CHEATSHEET.md`、任何既有报告/json（新评分全部写新目录）。
- **未运行** `tools/compute_ci.py` 的 main（importlib 复用其常量与 bootstrap 算法）。
- 验证：契约 `pytest tests/test_nn_aec.py tests/test_official_protocol.py -q` → exit 0（17 passed）。
- 已知遗留（只记录，未擅自扩修）：官方 `AECMOS_local/aecmos.py:113` 在 **每次 `run()` 内重建 `ort.InferenceSession`**（本仓 1740 条 × 3 法共数千次会话创建），属上游代码，改它超出本仓范围；本轮通过 CLI 逐进程调用未受影响。
