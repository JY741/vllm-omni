# MGM-Video ASA 切换至 `npu_block_sparse_attention` 设计方案

- 状态：草案，待用户复核
- 日期：2026-06-24
- 适用分支：`asa`
- 依赖文档：
  - `docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md`
  - `docs/superpowers/specs/2026-06-08-mgm-video-sta-hybrid-design.md`

## 1. 目标与范围

### 1.1 单一目标
将当前 ASA 路径中“块掩码 → token 级稠密掩码 → `npu_fusion_attention`”的实现，替换为直接调用 `torch_npu.npu_block_sparse_attention`，从而：
- 省掉 `expand_block_to_token_mask` 的内存与计算开销；
- 利用算子原生的块稀疏计算，获得稀疏率加速；
- 降低端到端推理时延。

### 1.2 包含
- 推理路径（`mmdit_blocks_inference.py`）新增块稀疏注意力回调；
- `asa.py` 中 `asa_attention` 支持通过回调注入块稀疏注意力；
- 环境变量 `VLLM_MGM_USE_BLOCK_SPARSE_ATTN` 控制新/旧路径切换；
- 块掩码 `[B, 1, nq, nk] bool` → `[B, N, nq, nk] int8` 的转换封装；
- 针对 `torch_npu.npu_block_sparse_attention` 的单元测试与 NPU smoke test；
- 与稠密路径的精度/性能对照实验。

### 1.3 不包含（YAGNI）
- 训练路径接入（`mmdit_blocks.py` 当前未启用 ASA）；
- 修改 ASA 的 Gilbert 重排、块重要性估计、能量阈值剪枝逻辑；
- 修改 STA 掩码生成逻辑；
- 替换 `npu_fusion_attention` 在非 ASA 路径中的使用；
- 自定义 NPU kernel 开发。

## 2. 高层架构

### 2.1 模块划分

```
vllm_omni/diffusion/models/mgm_video/mmdit/
├── asa.py                          (少量改动)
│   └── asa_attention               接收 dense-FA 或 block-sparse 回调
├── mmdit_blocks_inference.py       (新增 env-var 分支与回调构造)
│   └── JoinAttentionInference
│       ├── 读取 VLLM_MGM_USE_BLOCK_SPARSE_ATTN
│       └── 构造 fa_full_dense 或 fa_block_sparse 回调
└── block_sparse_attn.py            (新增独立文件)
    └── npu_block_sparse_attention_wrapper
        bool 块掩码 → int8 per-head 掩码，调用 torch_npu 算子

tests/mgm_video/test_asa.py         (新增 mock-callback 与 parity 测试)
docs/superpowers/specs/2026-06-24-mgm-video-block-sparse-attn-design.md
```

### 2.2 设计原则

- **解耦**：`asa.py` 不直接引用 `torch_npu`，与当前 `fa_full_dense` 注入方式保持一致；
- **可回退**：环境变量为 `0` 时，行为与当前稠密路径 bit-by-bit 一致；
- **可测试**：块稀疏注意力通过回调注入，单元测试可用 mock 替代真实 NPU 算子；
- **最小侵入**：不改 ASA 上游（Gilbert、重要性估计、STA OR）逻辑。

## 3. 端到端数据流

### 3.1 张量布局约定

进入 `JoinAttentionInference.fa` 时形状为 `[B, N_heads, S, D]`（BNSD）。其中：
- `S = T + L`，`T = f * hh * ww` 为 video token 数，`L` 为 text token 数；
- 当前模型 `hidden_size = 1152`，`head_dim = 64`，`N_heads = 18`（MHA，非 GQA）；
- 推理 dtype 为 `torch.bfloat16`。

### 3.2 新块稀疏路径数据流

```
q,k,v: [B, N, T+L, D]  (bf16)
   │
   ├── 1. Gilbert 重排（仅 video 段）── 不变
   │      q_g, k_g, v_g
   │
   ├── 2. sample_pool_attn ── 不变
   │      P: [B, 1, nq, nk]
   │
   ├── 3. build_asa_block_mask ── 不变
   │      M_block: [B, 1, nq, nk] bool
   │
   ├── 4. 与 STA 块掩码 OR ── 不变
   │      M_block: [B, 1, nq, nk] bool
   │
   ├── 5. 块掩码转 per-head int8
   │      M_int8 = M_block.expand(-1, N, -1, -1).to(torch.int8)
   │      M_int8: [B, N, nq, nk]
   │
   ├── 6. 块稀疏注意力回调
   │      out_g = fa_block_sparse(q_g, k_g, v_g, M_int8)
   │      out_g: [B, N, T+L, D]
   │
   └── 7. Gilbert 逆重排（仅 video 段）── 不变
          out: [B, N, T+L, D]
```

### 3.3 旧稠密路径数据流（保留，env-var 关闭时）

与当前实现完全一致：
```
M_token = expand_block_to_token_mask(M_block, block_size, t_len, l_len)  # [B, 1, S, S]
out_g = fa_full_dense(q_g, k_g, v_g, M_token)
```

## 4. 关键工程决策

### 4.1 回调注入而非直接调用

`asa_attention` 当前接收 `fa_full_dense(q, k, v, atten_mask)` 回调以保持与 `torch_npu` 解耦。新设计增加一个可选回调：

```python
def asa_attention(
    q, k, v,
    cfg: AsaConfig,
    rearranger: GilbertRearranger,
    t_len: int,
    l_len: int,
    fa_full_dense,                       # 稠密路径回调（旧）
    fa_block_sparse=None,                # 新增：块稀疏路径回调
    generator=None,
    sta_cache=None,
):
```

回调签名约定：
- `fa_full_dense(q, k, v, atten_mask) -> out`，其中 `atten_mask` 为 `[B, 1, S, S]` bool，`True` 表示保留；
- `fa_block_sparse(q, k, v, block_mask) -> out`，其中 `block_mask` 为 `[B, 1, nq, nk]` bool，函数内部负责扩成 per-head int8。

当 `fa_block_sparse is not None` 且 `cfg.variant == "asa"` 时，走块稀疏路径；否则走稠密路径。这样：
- `asa.py` 不直接 import `torch_npu`；
- 单元测试可传入 mock 回调验证调用参数；
- `mmdit_blocks_inference.py` 负责构造真实 NPU 回调。

### 4.2 块稀疏算子调用

```python
out, _ = torch_npu.npu_block_sparse_attention(
    query=q, key=k, value=v,
    block_sparse_mask=block_mask_int8,
    block_shape=[block_size, block_size],
    q_input_layout="BNSD",
    kv_input_layout="BNSD",
    num_key_value_heads=num_heads,       # MHA: N_kv = N
    scale_value=head_dim ** -0.5,
    inner_precise=0,                     # bf16 强制 fp32 softmax
)
```

约束与匹配：
- `blockShapeY` 必须是 128 的倍数；`cfg.block_size = 128` 满足；
- 输入 dtype 为 bf16，因此 `inner_precise=0`；
- 当前为 MHA，`num_key_value_heads = N_heads`；
- `scale_value` 与当前 `npu_fusion_attention` 的 `scale=(C // n_head) ** -0.5` 一致；
- 序列长度不需要整除 `block_shape`，算子内部会向上取整。

### 4.3 块掩码转 int8

`npu_block_sparse_attention` 的 `block_sparse_mask` 形状要求为 `[B, N, ceil(QS/blockShapeX), ceil(KVS/blockShapeY)]`。当前 `M_block` 是 head-mean 共享的 `[B, 1, nq, nk]` bool，因此：

```python
block_mask_int8 = m_block.expand(-1, num_heads, -1, -1).to(torch.int8)
```

其中 `nq = nk = ceil((T_pad + L_pad) / block_size)`，与算子内部的分块一致。

### 4.4 text token 处理

当前 `expand_block_to_token_mask` 强制 `video×text`、`text×video`、`text×text` 全部为 `True`。在块稀疏算子中，必须显式保证同样语义：

- text tokens 位于序列尾部，从 block index `t_len // block_size` 开始的所有 block 行/列必须保留为 1；
- 即使最后一个 video block 与 text block 有重叠，也整体保留为 1。

实现：在 `cfg.variant == "asa"` 分支中，当 `fa_block_sparse is not None` 时，对 `m_block` 做如下强制：

```python
n_blocks_total = m_block.shape[-1]
n_text_start_block = t_len // cfg.block_size
if n_text_start_block < n_blocks_total:
    m_block[..., n_text_start_block:, :] = True
    m_block[..., :, n_text_start_block:] = True
```

然后才把 `m_block` 传给 `fa_block_sparse`。稠密路径保持原有 `expand_block_to_token_mask` 行为不变。

### 4.5 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_MGM_USE_BLOCK_SPARSE_ATTN` | `0` | `1` 启用 `npu_block_sparse_attention`，`0` 保持稠密路径 |
| `VLLM_MGM_BLOCK_SPARSE_SHAPE` | `128,128` | 可选，块稀疏算子的 `block_shape`，格式 `"X,Y"` |

读取位置：`mmdit_blocks_inference.py` 中 `JoinAttentionInference` 的初始化或 `infer` 方法内，避免在 `asa.py` 中引入环境依赖。

## 5. 改动点详述

### 5.1 `asa.py`

- `asa_attention` 新增可选参数 `fa_block_sparse=None`；
- 在 `cfg.variant == "asa"` 分支中：
  - 若 `fa_block_sparse is not None`，调用块稀疏回调；
  - 否则保持现有 `expand_block_to_token_mask` + `fa_full_dense` 路径；
- `dense_probe` 路径完全不变。

### 5.2 `mmdit_blocks_inference.py`

- 在 `JoinAttentionInference.infer` 中读取 `VLLM_MGM_USE_BLOCK_SPARSE_ATTN`（每次 forward 读取，便于运行时切换而无需重建实例）；
- 构造两个回调：
  - `_fa_full_dense`：包装 `self.fa`（现有的 `npu_fusion_attention` 路径）；
  - `_fa_block_sparse`：调用 `npu_block_sparse_attention_wrapper`；
- 根据环境变量决定传入 `asa_attention` 的回调；
- 若环境变量为 `1` 但当前 torch_npu 版本不支持该算子，抛出清晰错误。

### 5.3 `block_sparse_attn.py`（新增）

```python
import torch
import torch_npu


def npu_block_sparse_attention_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,  # [B, 1, nq, nk] bool
    block_size: int = 128,
) -> torch.Tensor:
    """Wrap torch_npu.npu_block_sparse_attention for ASA."""
    num_heads = q.shape[1]
    head_dim = q.shape[-1]
    block_mask_int8 = block_mask.expand(-1, num_heads, -1, -1).to(torch.int8)

    out, _ = torch_npu.npu_block_sparse_attention(
        q, k, v,
        block_sparse_mask=block_mask_int8,
        block_shape=[block_size, block_size],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=num_heads,
        scale_value=head_dim ** -0.5,
        inner_precise=0,
    )
    return out
```

## 6. 测试计划

### 6.1 单元测试（CPU 可运行）

- `test_asa_attention_block_sparse_callback_invoked`：
  - 使用 mock `fa_block_sparse` 回调；
  - 验证回调被调用，且传入的块掩码形状为 `[B, N, nq, nk]`、dtype 为 `int8`；
  - 验证 q/k/v 已经过 Gilbert 重排。
- `test_block_sparse_mask_conversion`：
  - 验证 `[B, 1, nq, nk] bool` 正确扩展为 `[B, N, nq, nk] int8`。

### 6.2 NPU smoke test

- 在真实 NPU 上用极小输入（`B=1, N=2, S=256, D=64`）调用 `npu_block_sparse_attention_wrapper`；
- 验证输出形状为 `[B, N, S, D]`；
- 验证无显存异常。

### 6.3 精度对照

- 使用相同输入分别运行稠密路径与块稀疏路径；
- 在 block mask 全 1 的退化情况下，两者输出应接近（允许算子级 tolerance，如 `rtol=1e-2, atol=1e-3` for bf16）；
- 在真实 ASA 稀疏掩码下，记录 attention out 的 L2 相对误差。

## 7. 性能与精度验证

### 7.1 性能指标

- **端到端 step latency**：相同输入下，开启/关闭 `VLLM_MGM_USE_BLOCK_SPARSE_ATTN` 的单步耗时；
- **attention 子模块耗时**：通过 profiler 对比 `asa_attention` 内部耗时；
- **显存峰值**：`torch.npu.max_memory_allocated()`；
- **稀疏率**：统计每层 `M_block.sum() / M_block.numel()`，与理论 `max_retain_ratio` 对比。

### 7.2 精度指标

- 与稠密路径生成视频对比 PSNR/SSIM；
- 人工目检关键 case；
- 若出现明显退化，回退至稠密路径并分析原因。

## 8. 风险与回退

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| `npu_block_sparse_attention` 不支持当前 head_dim=64 的反向（训练场景） | 低（仅推理） | 本设计仅用于推理；若未来训练接入需重新评估 |
| 算子在某些序列长度/稀疏模式下性能不如稠密路径 | 中 | 环境变量可立即回退；perf 验证后再默认开启 |
| bf16 + inner_precise=0 的精度与当前 npu_fusion_attention 有差异 | 中 | 全 1 mask 退化对照 + 视频质量指标验证 |
| `block_shape` 调优空间未探索 | 低 | 默认 `128,128`；后续可暴露 `VLLM_MGM_BLOCK_SPARSE_SHAPE` 调参 |
| text token 所在 block 会拉低稀疏率 | 中 | 这是当前语义所需，需在 perf 报告中单独说明 |

## 9. 验收标准

- [ ] `VLLM_MGM_USE_BLOCK_SPARSE_ATTN=1` 时推理成功完成，输出视频与稠密路径在视觉上一致；
- [ ] 单元测试全部通过（含 mock callback 测试）；
- [ ] NPU smoke test 通过；
- [ ] 端到端 latency 优于或等于稠密路径（在 ASA 实际稀疏率下）；
- [ ] 设计文档合并后，通过 `writing-plans` 生成实施计划。
