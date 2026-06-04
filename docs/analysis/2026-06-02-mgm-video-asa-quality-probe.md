# MGM-Video MMDiT × BLADE ASA 精度可行性探针报告（A 阶段）

- 日期：2026-06-02（实际跑测日期：TBD-by-runner）
- 分支：`asa`
- 设计 spec：`docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md`
- 探针脚本：`scripts/probe_mgm_asa.py`
- 新增 P4.5 sweep：通过环境变量 `VLLM_MGM_ASA_WARMUP_STEPS=N` 驱动 step-warmup 混合稀疏度扫描，详见下方 "Step-warmup sweep (P4.5)" 章节

## 实验配置

| 项 | 值 |
|---|---|
| 输入分辨率 | 1×16×16×90×160（B,C,T,H,W） |
| video token 数 (T) | 57600（W=80, H=45, T=16） |
| text token 数 (L) | 256 |
| 蒸馏步数 | 8 |
| cache scheme | `cache_scheme_tdm_8step_dit_per_12_5_v1_speedup.txt`（生产路径，启用） |
| seed | 1234（5 prompt 共用） |
| dtype | bf16 |
| 实验 NPU | TBD-by-runner |

## Prompt 集合

（与 `scripts/probe_mgm_asa.py` 中的 PROMPTS 同步）

1. p01_person — A close-up portrait of an elderly fisherman mending his nets at sunrise, warm golden light, realistic, cinematic.
2. p02_motion — A leopard sprinting through tall grass at golden hour, motion blur on the grass, wildlife documentary style.
3. p03_scene — Snow falling slowly over a quiet Tokyo back-alley at night, neon reflections on wet cobblestones, atmospheric.
4. p04_object — A ceramic teacup tipping over onto a wooden table in slow motion, water splashing, macro lens.
5. p05_crowd — A bustling street market in Marrakech at midday, vibrant colors, people moving, handheld camera feel.

## 数值结果（vs baseline）

| run | prompt | PSNR (dB) | SSIM proxy |
|---|---|---|---|
| dense_probe | p01_person | TBD | TBD |
| dense_probe | p02_motion | TBD | TBD |
| dense_probe | p03_scene  | TBD | TBD |
| dense_probe | p04_object | TBD | TBD |
| dense_probe | p05_crowd  | TBD | TBD |
| asa_0p20    | p01_person | TBD | TBD |
| asa_0p20    | p02_motion | TBD | TBD |
| asa_0p20    | p03_scene  | TBD | TBD |
| asa_0p20    | p04_object | TBD | TBD |
| asa_0p20    | p05_crowd  | TBD | TBD |

P3 验收门槛：dense_probe 所有 prompt PSNR > 35 dB。

## 视频路径

| run | prompt | 路径 |
|---|---|---|
| baseline    | p01_person | runs/asa_probe_xxx/baseline_p01_person.mp4 |
| ...         | ...        | ... |

（runner 跑完后用 `find runs/asa_probe_xxx -name '*.mp4'` 自动填充）

## 人工目检结论

每 prompt 一段（baseline / dense_probe / asa_0p20 三栏对比）：

- p01_person：TBD-by-runner
- p02_motion：TBD-by-runner
- p03_scene：TBD-by-runner
- p04_object：TBD-by-runner
- p05_crowd：TBD-by-runner

## 结论与下一步建议

（runner 综合数值与目检填）

下一步选项：

- **推进 P5 (ASA_G)**：A₀ 通过且仍想进一步提升质量 → 实施 spec §3.3 双路径融合
- **扩展 C₀ 阶梯**：A₀ 通过 → 跑 `max_retain ∈ {0.15, 0.10}` 找最低可用稀疏度
- **C₀' 高位扫描**：A₀ 失败 → 跑 `max_retain ∈ {0.30, 0.40}` 找最高可用上限
- **暂停**：所有档位都崩 → ASA 在该模型不适用，结论入库并停止

## Step-warmup sweep (P4.5)

### 背景与动机

BLADE Fig 8 / FastDiT / ToDo 等工作均观察到：扩散模型早期若干步主要负责"全局结构"建立，激进稀疏化在这些步骤上极易破坏构图。"Step-warmup hybrid sparsity" 通过在前 `N` 步保持稠密注意力（warmup_steps=N），仅在 `step ≥ N` 后启用 ASA 稀疏化，以前置保护全局结构、后续提速换精度。

本 sweep 的目的：在 8 步 TDM 模型上量化 `warmup_steps ∈ {0, 1, 2, 3}` 在 PSNR / SSIM proxy / wall-time 上的 Pareto 取舍，确定推荐默认值。

### 固定配置

- `max_retain = TBD-by-runner`（建议默认 `0.20`，即上方 `asa_0p20` 行使用的最优档位；以 `数值结果` 章节得出的最优档位为准）
- 其余 ASA 参数与 `asa_0p20` 行保持一致（block_size、Gilbert rearranger、cache scheme 等）
- 仅 `warmup_steps` 一个变量在 `{0, 1, 2, 3}` 间扫描

### 环境变量配方

```bash
export VLLM_MGM_ASA_ENABLE=1
export VLLM_MGM_ASA_MAX_RETAIN=0.20      # 替换为最优档位
export VLLM_MGM_ASA_WARMUP_STEPS=N       # N ∈ {0, 1, 2, 3}
# 其余 VLLM_MGM_ASA_* 变量沿用 asa_0p20 配置
```

`warmup_steps=0` 等价于纯 ASA（无 warmup），用作回归基准；`warmup_steps=2` 是 BLADE 经验默认。

### 主表（5 prompts 平均）

| warmup_steps | avg PSNR (dB) | avg SSIM proxy | wall-time/video (s) | wall-time vs warmup=0 (%) |
|---|---|---|---|---|
| 0 | TBD | TBD | TBD | 0.0 (基准) |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |

### 详细表（5 × 4 = 20 cells，可选）

<details>
<summary>展开按 prompt × warmup_steps 的逐 cell 数据</summary>

| prompt | warmup=0 PSNR | warmup=1 PSNR | warmup=2 PSNR | warmup=3 PSNR |
|---|---|---|---|---|
| p01_person | TBD | TBD | TBD | TBD |
| p02_motion | TBD | TBD | TBD | TBD |
| p03_scene  | TBD | TBD | TBD | TBD |
| p04_object | TBD | TBD | TBD | TBD |
| p05_crowd  | TBD | TBD | TBD | TBD |

| prompt | warmup=0 SSIM | warmup=1 SSIM | warmup=2 SSIM | warmup=3 SSIM |
|---|---|---|---|---|
| p01_person | TBD | TBD | TBD | TBD |
| p02_motion | TBD | TBD | TBD | TBD |
| p03_scene  | TBD | TBD | TBD | TBD |
| p04_object | TBD | TBD | TBD | TBD |
| p05_crowd  | TBD | TBD | TBD | TBD |

| prompt | warmup=0 wall (s) | warmup=1 wall (s) | warmup=2 wall (s) | warmup=3 wall (s) |
|---|---|---|---|---|
| p01_person | TBD | TBD | TBD | TBD |
| p02_motion | TBD | TBD | TBD | TBD |
| p03_scene  | TBD | TBD | TBD | TBD |
| p04_object | TBD | TBD | TBD | TBD |
| p05_crowd  | TBD | TBD | TBD | TBD |

</details>

### Pareto 取舍 / 推荐默认

请 runner 在主表填完后，沿 (wall-time vs warmup=0 %) × (avg PSNR) 两轴画一张 Pareto 散点图（4 个点），并在下方标注推荐 `warmup_steps` 默认值：

- 推荐 `warmup_steps = TBD-by-runner`
- 选择依据：满足下方验收门槛中"PSNR 与 wall-time"双约束的最小 `warmup_steps`（如同时多个满足，取 wall-time 最低者）

### 验收门槛（plan §P4.5）

直接引用计划 §P4.5 验收要点，runner 跑完后逐项 check：

1. `warmup_steps=0` bit-exact：单元测试已验证（`tests/diffusion/models/mgm_video/test_asa_warmup.py` 或同等用例）；本 sweep 仍跑一次，作为 PSNR sanity 对照纯 ASA run。
2. `warmup_steps=2` PSNR ≥ `dense_probe - 3 dB`（在 8 步 TDM 模型上，5 prompts 平均）。
3. wall-time 增幅 `warmup_steps=2 vs warmup_steps=0` ≤ 25%。

若三条全部满足 → P4.5 验收通过，按推荐默认值落库；若 (2) 不通过 → 上调 `warmup_steps` 至 3 或改用 ASA_G；若 (3) 不通过 → 下调 `max_retain` 或下调 `warmup_steps` 至 1。

## 已知风险触发情况

（参照 spec §6.3 R1–R8，runner 标注实际是否触发）

- R1 稠密 mask OOM：未触发 / 触发（处理：xx）
- R2 GilbertRearranger 初始化慢：未触发 / 触发（处理：xx）
- R3 Gilbert 逆运算非 bit-exact：未触发 / 触发（处理：xx）
- R4 block_size 不整除 T：未触发 / 触发（处理：xx）
- R5 dense_probe PSNR 偏低：未触发 / 触发（处理：xx）
- R6 asa@0.20 PSNR 崩溃：未触发 / 触发（处理：xx）
- R7 NPU cumsum bf16 精度：未触发 / 触发（处理：xx）
- R8 text_length 不匹配运行时：未触发 / 触发（处理：xx）
