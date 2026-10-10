"""传统 AEC 组件库（WP-06 / 模块6，纯 numpy+scipy）。

模块清单（均逐帧/逐块流式友好——状态只在对象里，未来可直接搬 C）：
  gcc.py      GCC-PHAT 时延估计
  nlms.py     时域 NLMS 自适应滤波
  mdf.py      MDF/PBFDAF 分块频域自适应滤波（车载 DSP 主流方案）
  dtd.py      双讲检测（Geigel 能量比 + 相关系数 双判据）
  res.py      频域残余回声抑制（最小统计量驱动的 Wiener 增益）
  pipeline.py delay 对齐 -> 自适应滤波(DTD 软冻结) -> 可选 RES
"""
