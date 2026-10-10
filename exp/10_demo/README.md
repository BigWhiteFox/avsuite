# exp/10_demo — Gradio 在线 Demo（WP-10 / 模块10）

GTCRN（23.67K 参数，DNS3 官方权重）ONNX 推理的浏览器 demo：
上传/麦克风录音 → 增强 → A/B 对比播放 + **DNSMOS 无参考分数**（有参考才给 PESQ，
无参考场景的纪律，WORKLIST 明示）。

## 本地运行

```bash
export PYTHONNOUSERSITE=1
cd exp/10_demo
python app.py          # http://127.0.0.1:7860，同局域网手机可访问（0.0.0.0 绑定）
```

首次运行会自动下载 DNSMOS 模型（~1.5MB，GitHub raw 经代理，缓存 `D:/data/models/dnsmos/`）；
下载失败时增强功能不受影响，仅 DNSMOS 分数显示提示。

## 实现要点

- **模型加载一次**：模块级单例 onnxruntime Session（启动后每次请求零重载）
- 任意采样率输入自动 resample_poly 到 16k；立体声转单声道
- 处理异常吞掉并返回友好提示（不崩页面）
- 引擎可替换：`app.py` 的 `enhance()` 换成 CRN/DCCRN 权重即可（预留 `--model` 扩展位）

## HF Spaces 部署（WORKLIST 要求文档化）

1. 推仓库到 GitHub（或只推 `app.py + requirements + 模型文件`）
2. HF 新建 Space：SDK=Gradio，app 文件指向本 `app.py`
3. 路径适配：Space 环境无 `D:/` 盘——把 `DEFAULT_ONNX` 改为仓库相对路径
   `../05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx`（需把 onnx 文件纳入 git/或 LFS）
4. requirements：根 requirements.txt + gradio；DNSMOS 模型走 GitHub raw（Space 出口
   可直连，无需代理）

**降级路径（WORKLIST 预案）**：HF 被墙/排队 → 本地局域网部署 + 录屏进简历。
本 demo 已满足本地部署形态（0.0.0.0 绑定 + 手机可访问）。

## 已验证

- `process()`：合成带噪音频 → 增强 + DNSMOS（ovr 2.93 @4s SNR~5dB 样本）✅
- UI 服务器：HTTP 200，端口 7860 ✅
