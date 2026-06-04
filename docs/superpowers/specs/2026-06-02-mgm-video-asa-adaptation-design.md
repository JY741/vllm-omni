# MGM-Video MMDiT × BLADE ASA 适配方案设计（A 阶段：精度可行性探针）

- 状态：草案，待用户复核
- 日期：2026-06-02
- 适用分支：`asa`（基于 `feature/mgm-video-asa-quality-probe`）
- 论文与源码：
  - 论文解析 `/home/j00935189/code/omnia/omnia/paper_reading/2026.05/2026-05-29-BLADE.md`
  - ASA 源码梳理 `/home/j00935189/code/t2v/BLADE/ASA_Implementation.md`

## 1. 目标与范围

### 1.1 单一目标
回答一个问题：**BLADE ASA 算法的块稀疏掩码在 mgm_video 8-step TDM 蒸馏模型上是否能保住生成视频质量。**

### 1.2 包含
- Gilbert 3D 重排（仅 video token 段；text token 段保持原顺序）
- 采样近似估计块重要性（用 SDPA / `npu_fusion_attention` 替代 BLADE 的 Triton 池化 kernel）
- 能量阈值剪枝生成块掩码
- **关键差异**：把块掩码展开为 token 级稠密 mask，喂给现有 `torch_npu.npu_fusion_attention(atten_mask=...)`，**不引入新算子**
- C 阶段渐进路线：先标准 ASA（A 子阶段），再 ASA_G（B 子阶段，需要从 fa 取 LSE）
- 探针报告：PSNR/SSIM 数值对照 + 人工目检

### 1.3 不包含（YAGNI）
- 任何 NPU 块稀疏注意力算子开发（无加速目标）
- 替换或修改 skiparse 机制（已确认未启用）
- 修改 8-step TDM 蒸馏权重（训练无关 ASA）
- VBench 等大规模自动化评测
- 修改 `pipeline_mgm_video.py` 与 cache_algo 调度

### 1.4 已澄清的关键约束
- **skiparse 未启用**：所有 42 层 mmdit block 均为 full attention，ASA 介入面 100%
- **cache_algo 启用**（生产路径，scheme 文件 `cache_scheme_tdm_8step_dit_per_12_5_v1_speedup.txt`）：
  - 8 step × 42 layer = 336 个 attention 调用位
  - state 0 占 292 个、state 1/2/3 各 1 个、state 4 占 41 个
  - **step 5 整步全部 skip**（layer 0 state=1 记录 ori、layer 41 state=2 记录 residual、其余 state=4）
  - 其余 7 步全部 state=0（regular compute）
  - **ASA 实际介入**：state ∈ {0,1,2} 共 294 / 336 ≈ 87.5% attention 调用
  - 探针 baseline 与 ASA 实验**均启用相同 cache scheme**，单变量对照

## 2. 高层架构

### 2.1 模块划分
```
vllm_omni/diffusion/models/mgm_video/mmdit/
├── asa.py                          (NEW, ~400 行)
│   ├── AsaConfig                   dataclass + from_env() 工厂
│   ├── GilbertRearranger           lazy init + register_buffer 索引
│   ├── sample_pool_attn            块重要性估计
│   ├── build_asa_block_mask        能量阈值剪枝
│   ├── expand_block_to_token_mask  块 mask → 稠密 token mask
│   ├── asa_attention               A 子阶段:variant ∈ {dense_probe, asa}
│   └── asa_g_attention             B 子阶段:variant=asa_g（双路径融合）
└── mmdit_blocks_inference.py       改动 ~30 行
    └── JoinAttentionInference      接收 asa_cfg；fa() 内分支调用 asa.py

tests/mgm_video/test_asa.py         (NEW)
docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md  (报告产出，P4 完成时写)
scripts/probe_mgm_asa.py            (NEW, 自动跑探针并落盘)
```

### 2.2 配置入口
```python
@dataclass(frozen=True)
class AsaConfig:
    enable: bool = False
    variant: Literal["dense_probe", "asa", "asa_g"] = "asa"
    max_retain_ratio: float = 0.20
    min_retain_ratio: float = 0.05
    energy_threshold: float = 0.95
    block_size: int = 128            # 采样池化的块大小
    num_keep: int = 32               # 每块采样 token 数
    sample_gap: int = 30             # ASA_G 全局池化窗口
    use_gilbert: bool = True
    text_length: int = 256           # text token 数, 用于 video/text 段切分
    video_shape: tuple[int, int, int] | None = None   # (W, H, T) lazy init
    collect_stats: bool = False      # R6 风险预案:开启时收集每层 attn-out L2 norm
```
- 模型构造参数（`mmdit_xl_2_inference(asa_cfg=None)`）为唯一权威入口，仿 `cache_algo_cfg` 模式
- `AsaConfig.from_env()` 仅供探针脚本便利使用，不改变模型签名
- `asa_cfg=None` ⇒ 模型行为 bit-by-bit 与未引入 ASA 前一致

### 2.3 与现有代码的耦合点
- `mmdit_xl_2_inference()`：新增 kwarg `asa_cfg`
- `MMDiTInference.__init__`：透传到每个 block
- `MMDiTBlockInference.__init__`：透传到 `JoinAttentionInference`
- `JoinAttentionInference.__init__`：保存 `self.asa_cfg`
- `JoinAttentionInference.fa`：当 `asa_cfg.enable=True` 走 `asa_attention`，否则走原 `npu_fusion_attention`
- pipeline 层：**完全不动**，由独立脚本 `scripts/probe_mgm_asa.py` 构造 `AsaConfig` 注入

## 3. 端到端数据流（A 子阶段）

### 3.1 张量布局约定
进入 `JoinAttentionInference.fa` 时形状为 `[B, N_heads, S, D]`（BNSD）。S 由两段组成：
```
S = T (video tokens) + L (text tokens)
T = f * hh * ww    例: 16 * 45 * 80 = 57,600
L = ~256
```
ASA 仅对 video 段做空间重排；text 段保留在尾部不动（与 cogvideox 模式一致，wanx 全序列重排不适用 join attention）。

### 3.2 数据流
```
q,k,v: [B, N, T+L, D]
   │
   ├── 1. 拆分: q_v=q[:,:,:T,:]  q_t=q[:,:,T:,:]   (k,v 同理)
   │
   ├── 2. Gilbert 重排 (仅 video 段):
   │      q_v_g = q_v.index_select(-2, perm_idx)
   │      k_v_g, v_v_g 同理
   │
   ├── 3. 拼回: q_g = cat([q_v_g, q_t], dim=-2)   形状不变 [B,N,T+L,D]
   │           k_g, v_g 同理
   │
   ├── 4. 块重要性估计 (无梯度,head mean 共享):
   │      ├ pad q_g/k_g 到 cfg.block_size 倍数 (默认 128)
   │      ├ 每块随机采样 cfg.num_keep 个 token: q_smp, k_smp
   │      ├ scores = (q_smp @ k_smp.T) / sqrt(D)
   │      ├ softmax 沿 -1
   │      ├ block-pool: scores.view(...).amax(over within-block dims)
   │      │             中间形状 [B, N, nq, nk]
   │      └ head 维度 mean: P = scores_blockpool.mean(dim=1, keepdim=True)
   │                       形状 [B, 1, nq, nk]   nq=nk=⌈(T+L)/cfg.block_size⌉
   │      (head mean 共享是 §4.2 内存约束的硬决策,所有 head 共享同一 mask)
   │
   ├── 5. 阈值剪枝 build_asa_block_mask(P):
   │      ├ 行内降序排 (fp32 cast)
   │      ├ cumsum 找累积能量 ≥ 0.95 的位置
   │      └ clamp 到 [min_retain=0.05, max_retain=0.20] × nk
   │      输出 M_block: [B, 1, nq, nk] bool
   │
   ├── 6. 块 mask → token 稠密 mask (仅约束 video×video):
   │      ├ M_block_video = M_block 中对应 video 段块的子矩阵 [B,1,nq_v,nk_v]
   │      │                 nq_v = nk_v = ⌈T / cfg.block_size⌉
   │      ├ M_token_vv = M_block_video.repeat_interleave(cfg.block_size, -2)
   │      │                            .repeat_interleave(cfg.block_size, -1)
   │      ├ 截到 [B, 1, T, T] 去掉 pad
   │      └ 拼成完整 token mask:
   │            M_token = ones([B, 1, T+L, T+L])
   │            M_token[:, :, :T, :T] = M_token_vv
   │        即仅 video×video 块用稀疏 mask;video×text、text×video、
   │        text×text 三块全部保留为 True (text 段不剪)
   │      输出 M_token: [B, 1, T+L, T+L] bool
   │
   ├── 7. fa with mask:
   │      out_g = npu_fusion_attention(q_g, k_g, v_g,
   │                                   atten_mask=~M_token, ...)[0]
   │      (npu_fusion_attention 约定 True=mask 掉,故 logical_not;
   │       N=1 维 broadcast 给 24 head)
   │      out_g: [B, N, T+L, D]
   │
   └── 8. Gilbert 逆重排 (仅 video 段):
          out_v_g = out_g[:,:,:T,:]
          out_v   = out_v_g.index_select(-2, inv_perm_idx)
          out     = cat([out_v, out_g[:,:,T:,:]], dim=-2)
                    [B, N, T+L, D]
```

### 3.3 ASA_G（B 子阶段）增量
```
路径 1（局部稀疏）: 走 3.2 整套，但 fa 返回 (out1, lse1)
   lse1 = softmax_max + log(softmax_sum)   每 query 一个标量

路径 2（全局池化）:
   k_pool = simple_pooling(k_g, sample_gap=30)   # 均值池化
   v_pool = simple_pooling(v_g, sample_gap=30)
   out2, (sm_max2, sm_sum2) = npu_fusion_attention(q_g, k_pool, v_pool, atten_mask=None)
   lse2 = sm_max2 + log(sm_sum2)

LSE 加权融合（与 BLADE 源码一致）:
   log_w1 = lse1
   log_w2 = lse2 + log(sample_gap)
   m = max(log_w1, log_w2)
   alpha = exp(log_w1 - m) / (exp(log_w1 - m) + exp(log_w2 - m))
   out_fused = out1 * alpha + out2 * (1 - alpha)

最后过 Gilbert 逆重排
```

## 4. 关键工程决策

### 4.1 采样池化用 SDPA 替代 Triton 内核
BLADE 的 `attn_pooling_kernel.py` 在一个 Triton kernel 内同时输出 attention out 和块池化图，NPU 无对应实现。改成两步：
1. 采样得 `q_smp, k_smp`，shape `[B, N, nq*32, D]`；用 small SDPA / `npu_fusion_attention` 跑 small attention 得到 scores
2. 行 softmax + reshape 到 `[B, N, nq, 32, nk, 32]` + `amax(dim=(3,5))` 得到 P

成本可控（小尺寸 attention），且无随机性外的不确定性。
**A 阶段不计入加速预算**，写明：块重要性估计本身的开销不优化。

### 4.2 块 mask 展开成稠密 token mask（A 阶段核心妥协）
NPU 当前没有块稀疏 attention 算子，必须把块 mask 展开为 token mask 喂给 `npu_fusion_attention`：

- T+L ≈ 57856；token mask bool 矩阵约 **57856 × 57856 / 8 ≈ 400 MiB / batch / head**
- 多头 24 时 batch=1 单层 ≈ 9.6 GiB —— **必须做 head 共享**

**实际方案**（已写入 §3.2 step 4 数据流）：块重要性估计在 head 维度做 `mean`，得到 `[B, 1, nq, nk]`；展开后稠密 mask `[B, 1, T+L, T+L]` ≈ 400 MiB，在 fa 阶段沿 N 维 broadcast 给 24 head。

如果 broadcast 在 NPU 上仍展开为实体内存，回退顺序见 §6 R1。

### 4.3 Gilbert 索引一次性预计算 + register_buffer
- `GilbertRearranger.__init__(W, H, T, text_length)`：调用 `gilbert3d` 生成两个 LongTensor（`original2gilbert`, `gilbert2original`），长度 W×H×T
- `register_buffer` 注册，自动跟随模型 `.to(npu)`
- 首次 forward 触发的 H2D 拷贝在 warmup 完成；之后每次 forward 仅 `index_select`
- `gilbert3d` 算法直接移植 BLADE 源码（递归生成 (x,y,z) 序列），不重写

### 4.4 video_shape lazy init
`AsaConfig.video_shape` 默认 `None`：
- `JoinAttentionInference.infer` 入参带 `f, hh, ww`（mmdit_blocks_inference.py:134），首次 forward 时构造 `GilbertRearranger(W=ww, H=hh, T=f, text_length=cfg.text_length)`，缓存为 `self._asa_rearranger`
- 后续 forward 检查 `(f, hh, ww)` 与缓存一致；不一致触发 `AssertionError`（mgm_video 推理定长，正常情况下不应触发）
- 探针脚本无需提前算 video_shape

### 4.5 mask 生成路径强制 fp32
`build_asa_block_mask` 内部（sort、cumsum、阈值比较）强制 cast 到 fp32 后再生成 bool mask 输出。原因：
- bf16 长行（nk≈450）的 cumsum 累积误差大
- 部分 NPU 算子在 bf16 sort 上有版本敏感行为
- mask 输出是 bool，类型转换零损失

### 4.6 ASA_G 的 LSE 来源
`torch_npu.npu_fusion_attention` 在多输出模式返回 `(attention_out, softmax_max, softmax_sum, ...)`。当前 `JoinAttentionInference.fa` 只取 `[0]`，B 子阶段改成：
```python
out, sm_max, sm_sum = torch_npu.npu_fusion_attention(...)[:3]
lse = sm_max + torch.log(sm_sum.clamp(min=1e-30))
```
启动 B 子阶段前先用版本检测脚本确认接口；接口缺失则 `RuntimeError`，**不做静默 fallback**。

## 5. 超参数（A 子阶段，max_retain 阶梯实验）

| 超参 | 值 | 来源 / 理由 |
|---|---|---|
| `block_size` | 128 | 与 BLADE 源码（cogvideox/wanx）默认对齐；非模型相关 |
| `num_keep` | 32 | 同上 |
| `min_retain_ratio` | 0.05 | 同上 |
| `energy_threshold` | 0.95 | 同上 |
| `sample_gap`（仅 ASA_G） | 30 | 与 wanx 配置一致；mgm_video 序列长度 57k 比 wanx 33k 还长，不取 cogvideox 的 15 |
| `use_gilbert` | True | 关闭仅作为 R3 风险时的备选 |
| `video_shape` | (W=80, H=45, T=16) | 由 `[1,16,16,90,160]` 推算；lazy init 自动获取 |
| `text_length` | 256 | mgm_video 默认；与 cogvideox 226 接近 |

**`max_retain_ratio` 阶梯实验设计（D → A → C 三步走）**：

| 步骤 | `max_retain_ratio` | `variant` | 目的 |
|---|---|---|---|
| **D₀ (消歧基线)** | 1.0 | `dense_probe` | 跑通 Gilbert + 采样池化 + 全 1 mask 路径，验证不破坏数值（latent MSE < 1e-4） |
| **A₀ (主探针)** | 0.20 | `asa` | 与 wanx 论文配置对齐，约 0.8 稀疏比；这是探针主结论 |
| **C₀ (扩展，仅 A₀ 通过时跑)** | 0.15 / 0.10 | `asa` | 找最低可用稀疏度，画"质量–稀疏度"曲线 |
| **C₀' (扩展，仅 A₀ 失败时跑)** | 0.30 / 0.40 | `asa` | 找最高保住质量的稀疏度上限 |

不预先扫所有档位（避免×4 工作量）；按 D₀→A₀ 的结果决定 C₀ 方向。

## 6. 错误处理与已知风险

### 6.1 错误处理矩阵
| 场景 | 处理 |
|---|---|
| `asa_cfg=None` 或 `enable=False` | `JoinAttentionInference.fa` 完全走原路径；`asa.py` 延迟 import，零 import 副作用 |
| 运行时 `(f, hh, ww)` 与首次缓存不一致 | `AssertionError`（mgm_video 推理定长，触发说明上层 pipeline 异常） |
| `T % block_size != 0` | `pad_to_multiple` 在采样池化前 pad；token mask 展开后裁回原 T；pad 部分 mask 中标 0 |
| `dense_probe` 模式 | 采样固定种子 42（消除随机源），mask 跳过生成强制全 1，仅验证路径无损 |
| B 子阶段 fa 接口取不到 LSE | `RuntimeError`，无静默 fallback |
| 稠密 mask 内存超限 | 见 R1 回退顺序 |
| Gilbert buffer 在 CP 切分下行为 | 注入点在 `all_to_all` 之后（mmdit_blocks_inference.py:242-249），看到完整 video 段，无 CP 干扰 |

### 6.2 性能软约束（A 阶段）
| 指标 | 约束 | 不达标处理 |
|---|---|---|
| 单步注意力额外开销 | < 2× baseline | 先优化算法层（采样数减半、mask 缓存复用） |
| 显存峰值 | < baseline + 4 GiB | 触发 R1 回退路径 |
| 探针单 prompt 端到端 | < 5 min | baseline 已知量级 1 min；4× 预算合理 |

### 6.3 已知风险
| ID | 风险 | 触发条件 | 缓解 |
|---|---|---|---|
| **R1** | 稠密 mask OOM | 即使 per-head 共享后仍超显存 | 回退顺序: (a) `mean over heads` 已默认；(b) 改 `[B,1,1,T+L]` 列共享（仅按列剪枝，损失行精度但内存×T+L）；(c) 逐 head 循环 fa（牺牲并行）；(d) 试 `npu_fusion_attention` 的 `sparse_mode` 接口（如版本可用） |
| **R2** | bf16 长序列 sort/cumsum 卡死或精度不足 | block 重要性矩阵在 bf16 上 cumsum | mask 生成路径强制 fp32（已在 §4.5 兜底） |
| **R3** | NPU `index_select` 在 T=57600 上慢 | 首次 forward H2D + 长序列 gather | 索引张量 `register_buffer` warmup 一次；实测单层 > 50ms 则关 Gilbert（`use_gilbert=False`）做对照 |
| **R4** | 8-step 蒸馏模型对稀疏更敏感 | 关键 step 上稀疏破坏生成 | 探针报告**逐 step**而非逐 layer 对照；找出最敏感 step；B 子阶段 ASA_G 全局路径优先补这几步 |
| **R5** | dense_probe 模式 MSE 仍 > 1e-4 | Gilbert 重排不严格等价 / pad 路径有 bug | 单测 `test_gilbert_rearrange_inverse_is_identity` 是必过门槛；触发先修再继续 |
| **R6** | 视频质量退化但定位不到原因 | Q5 选 B 不收逐层 MSE | `AsaConfig.collect_stats=True` 预留收集**每层 attn-out L2 范数**（不是逐元素 MSE，开销可控）；只在退化无法解释时打开 |
| **R7** | cache_algo 与 ASA 耦合 | 已澄清生产 scheme 是 7 步全算 + step 5 整步 skip | A 阶段 baseline 与 ASA 实验**同时启用**生产 cache scheme，单变量对照；ASA 实际介入 294/336 ≈ 87.5% attention 调用 |
| **R8** | per-head mean 共享 mask 损失差异化 | 不同 head 的关注模式被强行平均 | A 子阶段接受此妥协（首要看是否保住质量）；如果 A₀ 通过且决定推 B 阶段，那时再评估是否需要 per-head 独立 mask + kernel 实现 |

## 7. 实施阶段

每个阶段独立可合并、可回滚。`asa_cfg=None` 始终是生产默认。

| 阶段 | 内容 | 输出 | 通过判据 |
|---|---|---|---|
| **P0 · 骨架** | 新建 `mmdit/asa.py`；`AsaConfig` + `from_env`；构造参数沿 `mmdit_xl_2_inference → MMDiTInference → MMDiTBlockInference → JoinAttentionInference` 透传；`enable=False` 时 lazy import 不触发 | PR-1 (~150 行) | `asa_cfg=None` 跑生产 prompt，输出与未引入 ASA 前 bit-by-bit 一致 |
| **P1 · Gilbert** | `GilbertRearranger`（lazy init + register_buffer）；`gilbert3d` 索引生成（移植 BLADE 源码）；rearrange / reversed_rearrange | PR-2 (~150 行) + 单测 | 单测 `test_gilbert_rearrange_inverse_is_identity` 通过；NPU 单层重排实测 < 50ms |
| **P2 · 块重要性 + mask 生成** | `sample_pool_attn`（采样 + small SDPA + block max-pool）；`build_asa_block_mask`（行排序 + cumsum + clamp，强制 fp32）；`expand_block_to_token_mask`（含 text 段强制保留） | PR-3 (~150 行) + 单测 | 单测全过；P 矩阵在 dense_probe 模式下 mask 全 1 |
| **P3 · 接入 + dense_probe** | `asa_attention(variant='dense_probe' \| 'asa')`；`JoinAttentionInference.fa` 当 `enable=True` 走 ASA 分支；走 `npu_fusion_attention` + 稠密 atten_mask | PR-4 (~50 行) + 集成测试 | dense_probe 模式（max_retain=1.0）跑 1 prompt：最终 latent MSE vs baseline < 1e-4；目检视频与 baseline 视觉等同 |
| **P4 · 启用 ASA + 探针报告** | `scripts/probe_mgm_asa.py` 跑 5 prompt × {dense_probe@1.0, asa@0.20}；输出 PSNR/SSIM 表 + 视频；写 `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md` | 报告 commit | 报告交付；结论明确（推进 B / 暂停 / 调超参） |
| **P5（可选） · ASA_G** | 取 LSE，加 `simple_pooling` + 全局 fa + log-sum-exp 融合；新增 `asa_g_attention` | PR-5 + 报告 update | 由 P4 报告结论决定是否启动 |

## 8. 测试策略

### 8.1 单元测试 `tests/mgm_video/test_asa.py`
（CPU/CUDA 也可跑，不强制 NPU）

```python
def test_gilbert_rearrange_inverse_is_identity():
    rearr = GilbertRearranger(W=80, H=45, T=16, text_length=256)
    x = torch.randn(1, 24, 80*45*16 + 256, 128)
    assert torch.equal(rearr.reversed_rearrange(rearr.rearrange(x)), x)

def test_build_asa_block_mask_bounds():
    P = torch.softmax(torch.randn(1, 1, 450, 450), dim=-1)
    M = build_asa_block_mask(P, max_retain=0.20, min_retain=0.05, threshold=0.95)
    row_sums = M.sum(-1)
    assert (row_sums >= int(450 * 0.05)).all()
    assert (row_sums <= int(450 * 0.20)).all()

def test_build_asa_block_mask_energy_threshold():
    # 行 [0.5,0.4,0.05,0.05]，cum=[0.5,0.9,0.95,1.0]，threshold=0.95 应保留前 3 块
    P = torch.tensor([[[[0.5, 0.4, 0.05, 0.05]]]])
    M = build_asa_block_mask(P, max_retain=1.0, min_retain=0.0, threshold=0.95)
    assert M.sum().item() == 3

def test_expand_block_to_token_mask_layout():
    # 小尺寸手算对照
    M_block = torch.tensor([[[[True, False], [False, True]]]])
    M_tok = expand_block_to_token_mask(M_block, block_size=2, T=4, L=2)
    expected = torch.tensor([
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [0,0,1,1,1,1],
        [0,0,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
    ]).bool()
    assert torch.equal(M_tok[0,0], expected)

def test_asa_attention_dense_probe_matches_full_attention():
    torch.manual_seed(42)
    q = torch.randn(1, 24, 1024, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    cfg = AsaConfig(enable=True, variant='dense_probe', max_retain_ratio=1.0,
                    video_shape=(8, 8, 16))
    out_asa = asa_attention(q, k, v, cfg, T=8*8*16, L=0)
    out_ref = reference_full_attention(q, k, v)
    torch.testing.assert_close(out_asa, out_ref, atol=1e-3, rtol=1e-3)

def test_asa_config_from_env(monkeypatch):
    monkeypatch.setenv('VLLM_MGM_ASA_ENABLE', '1')
    monkeypatch.setenv('VLLM_MGM_ASA_MAX_RETAIN', '0.15')
    cfg = AsaConfig.from_env()
    assert cfg.enable and cfg.max_retain_ratio == 0.15
```

### 8.2 集成测试 `scripts/probe_mgm_asa.py`（半自动）
```
固定 5 prompt + 固定 seed
跑 baseline (asa_cfg=None) → 保存 latent + 视频
跑 dense_probe (max_retain=1.0)
   验证: latent MSE vs baseline < 1e-4   [P3 验收门槛]
跑 asa@0.20
   计算: PSNR/SSIM vs baseline
   逐 step 输出对比帧
打印对照表 + 输出视频路径
```

prompt 集合（5 条覆盖人物 / 动作 / 场景）由探针脚本作者选定并写入 spec 附录或脚本顶部 docstring；报告中固定记录 prompt + seed + 视频路径。

## 9. 验收

A 阶段探针完成判据（全部满足）：
1. P0–P4 五个 PR 全部合入 `asa` 分支
2. `asa_cfg=None` 生产路径与 ASA 引入前完全一致（diff 走查 + 单测）
3. `tests/mgm_video/test_asa.py` 全过
4. P3 dense_probe 模式 latent MSE < 1e-4
5. P4 报告 `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md` 交付
6. 报告中给出"是否推进 B 子阶段（ASA_G）/ 是否启动 NPU kernel 工作"的明确结论

## 10. 决策摘要

| 问题 | 选择 |
|---|---|
| Q1 首要目标 | A：精度可行性探针（不追求加速） |
| Q2 ASA 变体 | C：先标准 ASA → 再 ASA_G 渐进 |
| Q3 Gilbert 粒度 | A：仅 video 段重排，text 尾接（cogvideox 模式） |
| Q4 max_retain | D→A→C：1.0 消歧 → 0.20 主探针 → 阶梯扩展 |
| Q5 质量判定 | B：数值对照（PSNR/SSIM/MSE）+ 人工目检 |
| Q6 代码组织 | B：新增 `mmdit/asa.py` 独立模块 |
| Q7 配置入口 | D：`AsaConfig` dataclass + `from_env()` 工厂 |
| Q8 旧探针处理 | 推倒重来，新设计不依赖旧探针代码（用户决策） |
| Q9 交付物 | B：代码 + 单测 + PSNR/SSIM 报告 + 5 prompt 视频 |

## 11. 参考
- BLADE 论文：<https://arxiv.org/abs/2508.10774v2>
- BLADE 源码 cogvideox：`/home/j00935189/code/t2v/BLADE/cogvideox/train/special_attentions_local/TrainRelated/cogvideo_blocksparseattn.py`
- BLADE 源码 wanx：`/home/j00935189/code/t2v/BLADE/wanx/train/special_attentions_local/TrainRelated/wanx_blocksparseattn.py`
- ASA 源码梳理：`/home/j00935189/code/t2v/BLADE/ASA_Implementation.md`
- 论文解析：`/home/j00935189/code/omnia/omnia/paper_reading/2026.05/2026-05-29-BLADE.md`
- mgm_video MMDiT inference 注入点：`vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:64-98`（`fa` 方法）
- cache scheme：`vllm_omni/diffusion/models/mgm_video/cache_scheme/cache_scheme_tdm_8step_dit_per_12_5_v1_speedup.txt`

