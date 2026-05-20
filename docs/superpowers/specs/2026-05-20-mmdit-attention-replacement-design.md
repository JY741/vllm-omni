# mgm_video MMDiT Attention 替换方案

## 背景

当前 `mgm_video` 模型的 MMDiT 模块使用自研 attention 实现（`JoinAttention`、`SelfAttention`、`CrossAttention` 等），其 attention kernel 直接调用 `torch.nn.functional.scaled_dot_product_attention`（CUDA 路径）或 `torch_npu.npu_fusion_attention`（NPU 路径）。

vllm-omni 提供了一套统一的 diffusion attention 框架（`vllm_omni.diffusion.attention`），具备以下优势：
- 自动 backend 分发（CUDA→FlashAttn, NPU→mindiesd, fallback→SDPA）
- 统一的序列并行（Ulysses + Ring）支持
- KV-cache 量化（FP8）支持
- 跨平台一致性

本方案将 mgm_video MMDiT 的 attention kernel 替换为 vllm-omni 的统一实现。

## 目标

1. 将 `JoinAttention`、`JoinAttentionInference`、`SelfAttention`、`CrossAttention` 中的 attention kernel 替换为 vllm-omni `Attention`
2. 保持 mgm_video 特有的外层逻辑（QKV 投影、QKNorm、3D RoPE、Skiparse、CP 通信、TDM Cache）不变
3. 精度对齐：替换后数值差异在 1e-4 以内
4. 不替换 VAE 中的 attention（原因见"排除范围"）

## 排除范围

以下模块**不参与**本次替换：

| 模块 | 原因 |
|------|------|
| VAE attention (`AttnBlock3D`, `TemporalAttention`, `SpatialTempAttention`) | 1. VAE attention 多为单头（head_size=channels，不固定），FlashAttn 支持列表有限；2. `AttnBlock3D` 使用 `torch.bmm` 而非标准 multi-head 格式，reshape 后可能引入额外开销；3. VAE 不是当前性能瓶颈 |
| `WindowAttention` | 仅在训练路径使用，当前推理不走此分支 |

## 架构设计

### 替换原则

采用"**内核替换，外壳保留**"策略：

```
┌──────────────────────────────────────────────┐
│           JoinAttention (保留外壳)            │
│  ┌─────────┐ ┌─────────┐ ┌──────────────┐   │
│  │qkv_x/y  │ │QKNorm   │ │3D RoPE       │   │
│  │Linear   │ │         │ │              │   │
│  └────┬────┘ └────┬────┘ └──────┬───────┘   │
│       │           │             │            │
│       └───────────┴─────────────┘            │
│                   │                          │
│       ┌───────────▼──────────────┐          │
│       │  CP通信 / Skiparse       │          │
│       │  (mgm_video 自己实现)     │          │
│       └───────────┬──────────────┘          │
│                   │                          │
│       ┌───────────▼──────────────┐          │
│       │  [替换] vllm Attention   │          │
│       │  (backend 自动分发)       │          │
│       └───────────┬──────────────┘          │
│                   │                          │
│       ┌───────────▼──────────────┐          │
│       │  CP通信 / 输出投影        │          │
│       │  (mgm_video 自己实现)     │          │
│       └──────────────────────────┘          │
└──────────────────────────────────────────────┘
```

### 各 Attention 类型替换策略

#### 1. JoinAttention（训练路径）

**当前实现：**
- `fa()` 方法中：`if flash: F.sdpa(...)` → `elif npu_fusion: torch_npu.npu_fusion_attention(...)` → `else: manual_softmax`

**替换后：**
- 新增 `vllm_attn` 实例
- `fa()` 中调用 `self.vllm_attn.forward(q, k, v, attn_metadata)`

**接口对齐：**
- vllm-omni `Attention` 接收 `(B, S, H, D)` 格式（BSND），但 NPU backend 内部会 transpose 为 BNSD
- mgm_video 当前使用 `(B, H, S, D)` 格式（BNSD），需要确认是否需要 transpose
- `AttentionMetadata` 需要传递 `attn_mask`

#### 2. JoinAttentionInference（推理路径）

**当前实现：**
- `fa()` 中：NPU 用 `torch_npu.npu_fusion_attention` 或 `mindiesd.attention_forward`（laser attention）

**替换后：**
- 和 JoinAttention 相同，替换 `fa()` 方法
- laser attention 场景需评估：vllm-omni 目前不直接支持 laser attention，如需要可保留 laser 分支作为 fallback

#### 3. SelfAttention

**当前实现：**
- `forward()` 中：`q, k, v = self.qkv(x).split(...)` → `F.sdpa(q, k, v)`

**替换后：**
- 保留 QKV 投影和 RoPE
- attention 计算替换为 `vllm_attn.forward(q, k, v, attn_metadata)`

#### 4. CrossAttention

**当前实现：**
- `forward()` 中：Q 来自 x，KV 来自 y，`F.sdpa(q, k, v)`

**替换后：**
- 保留 Q/K/V 投影
- attention 计算替换为 `vllm_attn.forward(q, k, v, attn_metadata)`
- `role="cross"`

### vllm-omni Attention 配置

```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

self.vllm_attn = Attention(
    num_heads=self.n_head,
    head_size=self.n_embd // self.n_head,
    causal=False,
    softmax_scale=(self.n_embd // self.n_head) ** -0.5,
    num_kv_heads=self.n_head,  # MHA
    role="joint",  # JoinAttention
    # role="self"  # SelfAttention
    # role="cross" # CrossAttention
    qkv_layout="BNSD",  # mgm_video 使用 BNSD
    skip_sequence_parallel=True,  # CP 由 mgm_video 自己处理
)
```

### AttentionMetadata 构建

```python
# 从 mgm_video 的 mask 构建 AttentionMetadata
# mgm_video 的 mask 格式：bool 类型，True 表示需要 mask 的位置
# vllm-omni SDPA 的 _maybe_reshape_attn_mask：2D mask → 4D 或 broadcast
attn_metadata = AttentionMetadata(
    attn_mask=mask,  # 2D bool mask (B, S)
)
```

## 关键差异与处理

### 1. 张量格式差异

| 属性 | mgm_video | vllm-omni |
|------|-----------|-----------|
| 默认格式 | BNSD (B, H, S, D) | BSND (B, S, H, D) |
| NPU backend | 内部 transpose | `forward_npu` 中 transpose |

**处理：**
- `JoinAttention` 的 `fa()` 输出是 BNSD，直接传入 vllm `Attention`
- vllm `Attention` 的 `_run_local_attention` 会调用 backend 的 `forward()`
- `SDPAImpl._forward_impl` 中有 `query.permute(0, 2, 1, 3)` 将 BSND → BNSD
- 但 `FlashAttentionImpl` 和 `mindiesd` 期望的格式不同

**需要验证：** 传入 BNSD 时，各 backend 是否能正确处理。根据代码分析：
- `SDPAImpl._forward_impl`：先 permute `(0, 2, 1, 3)` 把 BSND→BNSD，然后调用 `F.scaled_dot_product_attention`
- `FlashAttentionImpl.forward_npu`：调用 `mindiesd.attention_forward`，参数是 `layout="BNSD"`
- `FlashAttentionImpl.forward_fa_quant_npu`：`transpose(1, 2)` 把 BSND→BNSD

**结论：** vllm-omni `Attention` 接收 BSND，但 mgm_video 用 BNSD。有两种处理方式：
1. **方式 A**：在传入前 transpose BNSD→BSND，vllm 内部再处理
2. **方式 B**：设置 `qkv_layout="BNSD"`，让 NPU backend 直接处理

推荐**方式 B**，减少不必要的 transpose。

### 2. mask 格式差异

| 属性 | mgm_video | vllm-omni SDPA |
|------|-----------|----------------|
| mask 含义 | `True` = 需要 mask（padding） | `True` = 参与 attention |
| mask 维度 | 2D (B, S) 或 4D (B, 1, S, S) | 2D (B, S) |

**处理：**
- mgm_video 的 mask 在 `before_fa()` 中已经被处理为 `logical_not()` 后的格式
- 需要确认 mask 的语义是否一致
- `SDPAImpl._maybe_reshape_attn_mask` 会将 2D mask reshape 为 broadcast 或 full_qk 格式
- `forward_npu` 使用 `mask_mode="full_qk"`，生成 (B, 1, S, S) mask

### 3. Skiparse 与 CP 通信

Skiparse 和 CP 通信在 `before_fa()` 和 `after_fa()` 中处理，不受 attention kernel 替换影响。

### 4. TDM Cache

TDM Cache 在 `MMDiTBlockInference._forward()` 中控制是否跳过整个 block，不受 attention kernel 替换影响。

### 5. FA Offload

`JoinAttention.fa()` 中的 `offload_fa` 参数用于异步 H2D/D2H。vllm-omni `Attention` 不支持此特性。

**处理：** 如需要保留 FA Offload，可在调用 vllm `Attention` 前/后手动管理。

## 精度影响分析

| 因素 | 风险等级 | 说明 |
|------|---------|------|
| 数学公式一致性 | 极低 | 都是 softmax(QK^T/√d)V |
| kernel 实现差异 | 低 | `npu_fusion_attention` vs `mindiesd.attention_forward` 累加顺序不同 |
| RoPE 前置 | 无 | RoPE 在 attention 前应用，不受影响 |
| QKNorm 前置 | 无 | Norm 在 attention 前应用，不受影响 |
| Skiparse | 无 | 在 attention 前后处理 |
| CP 通信 | 无 | 通信逻辑不变 |

**预期数值差异：** 1e-5 ~ 1e-4（不同 fused attention kernel 的正常差异范围）

## 性能影响分析

| 场景 | 当前 | 替换后 | 预期影响 |
|------|------|--------|---------|
| NPU 训练 | `torch_npu.npu_fusion_attention` | `mindiesd.attention_forward` | ±5%，需实测 |
| NPU 推理 | `torch_npu.npu_fusion_attention` / laser | `mindiesd.attention_forward` | ±5%，laser 场景需评估 |
| CUDA 训练/推理 | `F.sdpa` | FlashAttn（若支持） | 可能提升 10-30% |

**性能风险缓解：**
1. 保留原始 `fa()` 作为 fallback（通过环境变量或 config 切换）
2. 首次替换后做端到端性能 benchmark

## 实现步骤

### Step 1: JoinAttention 替换

修改文件：`vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks.py`

1. 在 `JoinAttention.__init__` 中创建 `vllm_attn` 实例
2. 修改 `JoinAttention.fa()` 方法，调用 `self.vllm_attn.forward()`
3. 构建 `AttentionMetadata`，传递 mask
4. 处理 BNSD / BSND 格式问题

### Step 2: JoinAttentionInference 替换

修改文件：`vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py`

1. 在 `JoinAttentionInference.__init__` 中创建 `vllm_attn` 实例
2. 修改 `JoinAttentionInference.fa()` 方法
3. 评估 laser attention 场景，决定保留或替换

### Step 3: SelfAttention 替换

修改文件：`vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks.py`

1. 在 `SelfAttention.__init__` 中创建 `vllm_attn` 实例
2. 修改 `SelfAttention.forward()`，替换 `F.sdpa` 调用

### Step 4: CrossAttention 替换

修改文件：`vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks.py`

1. 在 `CrossAttention.__init__` 中创建 `vllm_attn` 实例
2. 修改 `CrossAttention.forward()`，替换 `F.sdpa` 调用

### Step 5: 精度验证

1. 使用相同输入，对比替换前后的输出
2. 验证数值差异在 1e-4 以内
3. 跑端到端推理，确认生成结果一致性

### Step 6: 性能 benchmark

1. 测量替换前后的 latency
2. 评估不同 backend（SDPA / FlashAttn / mindiesd）的性能

## 回滚方案

如需回滚，保留原始 `fa()` 实现作为 fallback：

```python
def fa(self, q, k, v, mask, C, ...):
    if os.environ.get("MGM_USE_ORIGINAL_ATTN", "0") == "1":
        return self._original_fa(q, k, v, mask, C, ...)
    # 新实现
    attn_metadata = AttentionMetadata(attn_mask=mask)
    return self.vllm_attn.forward(q, k, v, attn_metadata)
```

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 精度偏差超出预期 | 低 | 高 | 保留 fallback，逐步验证 |
| NPU 性能下降 | 中 | 中 | benchmark 对比，保留原始实现开关 |
| mask 格式不兼容 | 低 | 高 | 仔细验证 mask 语义和 reshape 逻辑 |
| 张量格式 transpose 错误 | 中 | 高 | 单测验证 shape |
| vllm-omni Attention 接口变更 | 低 | 中 | 依赖版本锁定 |

## 附录：关键代码路径

| 文件 | 用途 |
|------|------|
| `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks.py` | JoinAttention, SelfAttention, CrossAttention |
| `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py` | JoinAttentionInference |
| `vllm_omni/diffusion/attention/layer.py` | vllm-omni Attention 入口 |
| `vllm_omni/diffusion/attention/backends/abstract.py` | AttentionBackend, AttentionImpl, AttentionMetadata |
| `vllm_omni/diffusion/attention/backends/sdpa.py` | SDPA backend |
| `vllm_omni/diffusion/attention/backends/flash_attn.py` | Flash Attention backend |
| `vllm_omni/diffusion/attention/selector.py` | Backend 选择逻辑 |
