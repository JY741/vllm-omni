# MGM-Video ASA 质量探针设计（BLADE Adaptive Block-Sparse Attention，纯 torch 推理期套用）

**日期**: 2026-05-30
**作者**: jiangyu741 / Claude
**状态**: 设计待评审
**关联**: BLADE 论文解读 `/home/j00935189/code/omnia/omnia/paper_reading/2026.05/2026-05-30-BLADE.md`

---

## 0. TL;DR

在 MGM-Video MMDiT denoise 阶段的 **video↔video 自注意力**上，按 BLADE 论文推理期的
**二值块稀疏掩码（ASA）**逻辑，用**纯 torch** 生成 `atten_mask` 喂给现有
`npu_fusion_attention`，**只为肉眼验证"块稀疏会不会把这个已蒸馏 8-step 模型的视频质量搞崩"**。

**本设计明确不追求墙钟加速**：dense FA + mask 不跳块，被屏蔽位置照算，因此总时间只增不减。
这是一次"质量探针"，用来决定后续是否值得投入**真正能加速的 prefill-MHA block-sparse 融合算子**。

---

## 1. 背景与动机

### 1.1 现状

- `vllm_omni/diffusion/models/mgm_video` 是把 pangu_t2v 推理代码适配到 vllm-omni 的产物。
- denoise 是**计算 bound**，瓶颈在 MMDiT 的 full attention（已做 CP 切头 + 通算并行优化，仍很重）。
- 默认 11B preset：`hidden_size=3072, depth=42, num_heads=24, head_dim=128, model_max_length=400`，
  720×1280×121f → latent `16×45×80`，video token = **57600**，text ≤ 400，单步注意力序列 **S≈58K**、**双向全注意力**。
- 模型默认启用 **TDM 蒸馏 8-step** + `cache_scheme_tdm_8step_dit_per_12_5_v1_speedup.txt`
  （42层×8步状态表，部分层/步 skip 整个 attention）。

### 1.2 为何先做"质量探针"而非直接上融合算子

可行性前置评估（已完成）得出两条硬结论：

1. **用户最初指定的算子 `npu_sparse_flash_attention_enhance` 与 MMDiT 注意力几何不兼容。**
   它是 **MLA-decode 形态**（DeepSeek 风格），tiling 硬约束：
   - `qk_head_dim` 必须 == **512**（MMDiT 是 128）
   - `kv_head_num` 必须 == **1**（MMDiT 是 24 头 MHA）
   - `attention_mode` 必须 == **2**（MLA-absorb，q/k 与 rope 沿 D 拼接、k/v 共享底层）
   - 示例/kernel 面向 **S1=1 decode** + causal，而 MMDiT 是 58K 双向 prefill
   - 重要度由外部 `lightning_indexer` 选择器算，算子本身不算

   同目录所有"能真正跳块省墙钟"的算子（`sparse_flash_attention_enhance`、
   `ai_infra_attention_pioneer`）**全是 MLA 形态**；唯一的 MHA 算子
   `flash_attention_score_enhance` 是**训练用 dense FA + mask，不跳块**。
   **即：omni-ops 现有算子里没有任何一个能对 MMDiT（D=128/24头/双向/58K prefill）做带墙钟收益的 block-sparse。**

2. **模型已是 TDM 蒸馏少步学生**，把训练无关稀疏直接套上去，正是 BLADE 论文
   **明确警告的"朴素组合"失败模式**（§动机：少步步进大、对每步精度敏感，稀疏近似误差被放大、质量明显下降）。
   BLADE 的质量保证来自 **sparsity-aware 联合训练（ASA_G 全局通道 + TDM）**，而非推理期套用。

**结论**：在投入融合算子前，先用一版纯 torch ASA 把"质量风险"单独验证清楚——
解耦"质量"与"加速"两个问题。用户已确认走此路径。

---

## 2. 目标与非目标

### 2.1 目标

- 在 video↔video 注意力上套用 BLADE 推理期二值块稀疏掩码，跑出可肉眼对比的视频。
- 提供 τ / block_size / 作用域（layer/step 范围）的可配置 sweep 能力。
- 输出实际块稀疏率统计，给"是否值得做融合算子"一个量化锚点。
- 单一注入点、开关保护、可零风险回退到 dense。

### 2.2 非目标（YAGNI，明确不做）

- **不追求墙钟加速**（dense+mask 不跳块）。
- **不做 Gilbert 空间填充曲线重排**（见 §3.5 诚实标注）。
- **不做 k=16 采样近似**（我们用精确块重要度，见 §3.2）。
- **不做融合算子 / kernel 改造。**
- **不做 ASA_G 全局通道**（那是训练用的可微通路）。
- **不做性能 benchmark。**

---

## 3. 架构与算法设计

### 3.1 注入点

**唯一注入点：`JoinAttentionInference`（`mmdit/mmdit_blocks_inference.py`）的 `infer()` → `fa()` 注意力调用。**

- `infer()` 在 CP all-to-all **之后**，每个 rank 持有**完整序列 S、本 rank 的部分头**
  （24 头，CP=2 → 每 rank 12 头）。块稀疏掩码在本地 full-S 上独立生成，
  **完全不碰 all-to-all 通信逻辑**。
- `infer()` 里 `q = cat([x_q_chunk, y_q_chunk], dim=2)`，S 轴上 video 段 `[0:T]`、
  text 段 `[T:T+L]` 边界清晰 → 能精确只对 video↔video 加掩码、text 永远 dense。

**精确实现位置**：`fa()` 的签名不接收 `T/L/step`，因此 **ASA 掩码在 `infer()` 中
`out = self.fa(...)` 调用前构建**（`T`、`L` 在该作用域内是局部变量），与既有 `mask`
合并后作为 `mask` 实参传入 `fa()`。`fa()` 自身逻辑不改（它已消费 `mask`）。

> 这是对"注入点=fa()"口头表述的精确落地：掩码生成放在 infer() 里 fa 调用前，
> fa() 透明消费。语义等价，位置更准确。

### 3.2 精确块重要度算法（块粒度，规避 58K×58K 显存爆炸）

**核心洞察**：BLADE 的 k=16 采样是为省算力引入的*近似误差*；质量探针要的是"ASA 质量上界"，
应当用精确块重要度。但 full `P=softmax(QK^T)`（58K×58K×2B≈6.7GB/head）materialize 不出来。
解法：重要度本就只需块粒度，**全程块粒度归约，从不 materialize full P**。

设 video 段 `Q_v, K_v ∈ [S_v, D]`，`S_v=57600`，`block_size=B`（默认 128，对齐 kernel 友好的 64/128）。
默认配置下 `S_v=57600` 能被 64/128 整除，`nB = S_v / B = 450`；通用情形下 `nB = ceil(S_v/B)`，
尾部以 `F.pad` 把 `Q_v/K_v` 补到 `nB*B`（补零 token，最后再裁掉 pad 列/行）。下文按整除写，pad 仅为通用兜底。
每个 head 独立：

```
1. 块内聚合 K（query 无关，每层每步只算一次）：
   K_blk = K_v.reshape(nB, B, D).mean(dim=1)               # [nB, D]，mean-pool 代表块
   （pad 兜底后整除；mean 比单 token 采样更稳）

2. 近似分数矩阵（token 行 × 块列，约 50MB/head fp32，可接受）：
   S_imp = (Q_v @ K_blk.transpose(-1,-2)) * scale          # [S_v, nB]
   scale = head_dim ** -0.5

3. query 块归约（对 query 块内取 max，呼应论文 max-pool 共享掩码）：
   S_blk = S_imp.reshape(nB, B, nB).amax(dim=1)            # [nB, nB]

4. 行 softmax + τ 累积 top-k：
   P = softmax(S_blk, dim=-1)                              # [nB, nB]
   按行降序累积，选到累积概率 >= τ（默认 0.95）的最小块集 → 二值块掩码 M[nB, nB]
   （True = 保留 / 参与计算）

5. 对角块兜底自选：M[i, i] = True，保证局部性、防退化。

6. 展开回 token 掩码：
   M_tok = M.repeat_interleave(B, dim=0).repeat_interleave(B, dim=1)[:S_v, :S_v]   # [S_v, S_v]
   pad text 段：行/列扩到 [S, S]，text 行与 text 列全 dense（保留）
   取反成 npu_fusion_attention 约定（atten_mask: True = 屏蔽）：attn_mask = ~M_full
   广播到 [B, N_local, S, S]
```

**显存账**：
- step 2 的 `[S_v, nB]` ≈ 50MB/head（逐头算可进一步省）。
- 最终 `[S, S]` bool 掩码 ≈ 58K²×1B ≈ **3.3GB**。这是 dense-mask 路线的**固有代价**——
  `npu_fusion_attention` 需要显式 `[*, S, S]` mask。**这正是"不跳块、无加速"的物理体现**：
  花显存换"看质量"，不换速度。
- 若单卡显存吃紧：掩码可按 `[B,1,S,S]` 生成（所有头共享同一掩码——则 step 1-5 改为对头取平均/任一头；
  作为可选退化项，默认逐头）。实现里优先 `[B,1,S,S]` 共享掩码以省显存，并在文档标注"共享掩码是显存友好近似"。

> **显存策略决定**：默认生成 **`[B, 1, S, S]` 全头共享掩码**（重要度对头维取 mean 后再选块），
> 把 3.3GB 控制为单份而非 ×N_local。逐头掩码作为 `VLLM_MGM_ASA_PER_HEAD=1` 的可选项。

**复杂度**：`Q@K_blk` ≈ 57600×450×128 ≈ 3.3 GFLOP/head，相比 full attention 57600²×128
的 ~0.05%，可忽略。但 dense+mask 的 FA 本身一点没省——符合"先不要加速"。

### 3.3 与 TDM-cache 的协同（关键坑）

模型默认启用 cache（42层×8步状态表），其中 **state 3/4 的 (layer, step) 会 `skip_compute` 整个 attention**
（`mmdit_inference.py:232-237`）。ASA 若在这些位置生成掩码纯属浪费且误导结论。

**设计**：ASA 作用域**自动与 cache 表求交**，仅在 `current_status ∈ {0,1,2}`（真正 compute attention）
的 (layer, step) 上生效。
- layer 号：`self._debug_block_idx`（block.forward 已设到 attention 上）。
- step 号：需把 `cur_time_index` 从 block 透传到 attention。
  **wiring**：在 `MMDiTBlockInference.forward` 里增加 `self.attention.cur_time_index = self.cur_time_index`
  （cache 启用时 block.cur_time_index 已被 `blocks_forward` 设置）。
  ASA 在 `infer()` 里读 `getattr(self, 'cur_time_index', None)` 判作用域；为 None（cache 关闭等）时按 step=全部处理。

ASA 探针与既有加速方案正交，不互相干扰。

### 3.4 作用范围与 skiparse 互斥

- **只对 video↔video 子块加掩码**：`q[:,:,0:T]` × `k[:,:,0:T]`。text 段（`T:T+L`）作为 query 行、
  作为 key 列**永远 dense**（text 仅 400 token，稀疏它无收益且伤 cross-modal 对齐）。
- **downscale ≠ 1 的 skiparse 层互斥**：当前 11B 默认 preset 未开 skiparse（`downscale` 全 1）。
  仍加保护：`if self.downscale != 1: 跳过 ASA 并日志告警`（两种静态稀疏不叠加，避免语义混乱）。

### 3.5 诚实标注（写入文档与运行日志）

1. **省略 Gilbert 重排**（raster 顺序分块）。论文消融：Gilbert 带来明显质量提升（块内语义连贯）。
   故本版是**"质量下界探针"**：raster 块稀疏若质量 OK，Gilbert 只会更好；若 raster 崩了，
   需补 Gilbert 再判，不能直接下"ASA 不适配"结论。日志每次打印 `[ASA][raster, no-gilbert]` 标识。
2. **mean-pool 精确代表 vs 论文 k=16 采样近似**：我们用前者，是 ASA 重要度的**精确上界**。
3. **全头共享掩码**（默认）是显存友好近似；逐头掩码经 `VLLM_MGM_ASA_PER_HEAD=1` 开启。

### 3.6 CP 交互

ASA 在 all-to-all 之后的本地 full-S 上算，每 rank 独立生成自己头的掩码，
**无需跨 rank 通信**，不碰现有 all-to-all。

---

## 4. 配置接口（环境变量，零代码改动 sweep）

沿用现有 `VLLM_MGM_*` 风格：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_MGM_ASA_ENABLE` | `0` | 总开关，关=完全走原 dense 路径 |
| `VLLM_MGM_ASA_TAU` | `0.95` | 累积阈值（质量/稀疏权衡） |
| `VLLM_MGM_ASA_BLOCK` | `128` | 块大小（64/128） |
| `VLLM_MGM_ASA_LAYERS` | `all` | 生效层范围，如 `8-41`（跳过浅层） |
| `VLLM_MGM_ASA_STEPS` | `all` | 生效 step 范围，如 `2-7`（跳过前几步） |
| `VLLM_MGM_ASA_PER_HEAD` | `0` | 1=逐头掩码（更精确、更费显存）；0=全头共享 |
| `VLLM_MGM_ASA_LOG` | `1` | 1=每次打印 `layer/step/sparsity` 统计 |

范围解析：`all` 或 `lo-hi`（闭区间，按 layer/step 0-based 索引）。

---

## 5. 验证方法

用现成离线脚本 `examples/offline_inference/text_to_video/mgm_video_t2v.py`，零新增测试框架。

1. **基线对照**：同 prompt + 同 seed(42)，`ASA_ENABLE=0` 跑 dense 基线视频，`=1` 跑 ASA 视频，
   逐帧肉眼比 + 文件并排。**核心判据**。
2. **稀疏率自检**：`build_asa_block_mask` 统计实际块稀疏率，日志 `[ASA] layer=L step=S sparsity=XX%`。
   τ=0.95 下若稀疏率<20% → 注意力本就稠密、ASA 无意义（直接"不适配"）；50-80% 且不崩才有融合算子价值。
3. **数值健全性**（可选）：首帧 dense vs ASA 的 latent L2 / PSNR，给"崩"量化锚点。
4. **sweep 矩阵**：`τ ∈ {0.90,0.95,0.98} × layers ∈ {all,8-41} × steps ∈ {all,2-7}`，各跑一条，
   形成"质量 vs 稀疏率"曲线。

### 5.1 判定逻辑（指导看完视频后的决策）

- **质量基本无损 + 稀疏率>40%** → 值得投入做 prefill-MHA block-sparse 融合算子（回到真正能加速的路）。
- **视频崩**（模糊/闪烁/运动断裂）→ 印证 BLADE 警告 → 此模型需 sparsity-aware 重训才能用 ASA，
  推理期套用不可行。
- **中间态**（浅层崩深层不崩 / 前几步敏感）→ 作用域配置帮定位"哪些层/步能稀疏"，为算子方案缩小范围。

---

## 6. 交付物

1. `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py`
   —— ASA 块掩码生成（纯 torch，~150 行，独立可单测）。核心函数：
   `build_asa_block_mask(q, k, T, L, block_size, tau, per_head, scale) -> attn_mask | None`
2. `mmdit_blocks_inference.py` 的 `infer()` 注入（fa 调用前构建掩码）+ `_should_apply_asa(layer, step)`
   作用域判定（~20 行，开关保护）。
3. `mmdit_inference.py`：`MMDiTBlockInference.forward` 增加
   `self.attention.cur_time_index = getattr(self, 'cur_time_index', None)`（1 行 wiring）。
4. 本设计文档（已提交 git）。
5. 最小 pytest `tests/.../test_mmdit_asa.py`：固定小 shape（S=512, block=64）验证
   - 对角块自选；text 行/列全 dense；稀疏率随 τ 单调；
   - 与朴素 full-P 参考实现（直接 softmax(QK^T) → 块 max-pool → τ）数值对齐。
6. README 片段：环境变量用法 + sweep 命令示例。

---

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| 3.3GB 掩码 OOM | 默认全头共享掩码（单份）；必要时缩小 block 数无益（掩码 size 与 S² 绑定），
  根本退路是按 query-block 分块构建+分块调 FA（属未来融合算子范畴，本探针不做） |
| ASA 在 cache-skip 层生成掩码 | §3.3 自动与 cache 表求交 |
| 与 skiparse 静态稀疏叠加语义混乱 | §3.4 downscale≠1 时跳过 ASA + 告警 |
| 结论被 raster（无 Gilbert）拖累 | §3.5 明确标注为"质量下界探针" |
| 开关回退不干净 | `ASA_ENABLE=0` 时 `infer()` 不触碰任何 ASA 代码路径，等价原始 dense |

---

## 8. 实现顺序（供 writing-plans 展开）

1. `mmdit_asa.py`：`build_asa_block_mask` + 小 shape 参考实现 + pytest（TDD）。
2. step-index wiring（`mmdit_inference.py` 1 行）+ 作用域解析工具。
3. `infer()` 注入：构建掩码、合并既有 mask、传入 fa。
4. 日志/统计 + README。
5. 离线脚本跑 dense vs ASA 对照，sweep。
