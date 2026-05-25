# MGM-Video Inference RoPE 替换性能分析

## 背景

将 mgm_video 推理路径的 RoPE 实现从原始 `apply_3drotary_pos`（`mmdit_blocks.py`）替换为 vllm-omni 的 `RotaryEmbedding`（`diffusion/layers/rope.py`）后，精度对齐（fp32 零差异，bf16 ~1.56e-2 NPU kernel 差异），但去噪阶段整体性能劣化约 1.5s。

本文档逐算子对比两条路径的实现差异，定位性能劣化根因。

## 环境说明

- 远端环境无 mindiesd，`RotaryEmbedding.forward_npu` fallback 到 `forward_native` → `apply_rotary_emb_torch`
- NPU 设备（Ascend），对非连续内存访问（stride indexing）性能敏感
- 模型配置：42 个 MMDiT block，每 block 对 q/k 做 3 段（h/w/t）× 2（q+k）= 6 次 rope 调用，总计 252 次/forward

## 旧路径：`apply_rotary_pos_emb` + `rotate_every_two`

来源：`mmdit_blocks.py`，被 `apply_3drotary_pos` 调用。

### 代码

```python
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)  # view 操作，零拷贝
    x1, x2 = x.chunk(2, dim=-1)                      # view 操作，零拷贝
    x = torch.cat((-x2, x1), dim=-1)                  # 1 次内存分配
    return x.flatten(-2)                               # view 操作，零拷贝

def apply_rotary_pos_emb(tensor, sin, cos):
    sin = sin.unsqueeze(0).unsqueeze(1)  # view 操作，零拷贝
    cos = cos.unsqueeze(0).unsqueeze(1)  # view 操作，零拷贝
    return (tensor * cos) + (rotate_every_two(tensor) * sin)
```

### 输入格式

- `tensor`：`[B, N, S, D]`（BNSD）
- `sin, cos`：`[S, dim]`（全维度，interleaved 排列）

### 算子分解（每次调用）

| 步骤 | 操作 | 类型 | 内存开销 |
|------|------|------|----------|
| 1 | `sin.unsqueeze(0).unsqueeze(1)` | view | 零拷贝，仅修改 stride 元数据 |
| 2 | `cos.unsqueeze(0).unsqueeze(1)` | view | 零拷贝 |
| 3 | `rearrange(x, '... (d j) -> ... d j', j=2)` | view | 零拷贝，reshape |
| 4 | `x.chunk(2, dim=-1)` | view | 零拷贝，返回两个 view |
| 5 | `torch.cat((-x2, x1), dim=-1)` | **kernel** | 1 次内存分配 + 拷贝 |
| 6 | `x.flatten(-2)` | view | 零拷贝 |
| 7 | `tensor * cos` | **kernel** | 广播乘法（cos 通过 stride 广播，无实际展开） |
| 8 | `rotate_every_two(tensor) * sin` | **kernel** | 广播乘法 |
| 9 | `... + ...` | **kernel** | 逐元素加 |

**实际 NPU kernel launch：4 次**（cat、2 × mul、1 × add）

### 关键特点

- sin/cos 通过 `unsqueeze` 产生的 stride 广播参与乘法，**不触发实际数据拷贝**
- `rotate_every_two` 全程使用连续内存操作（`rearrange` → reshape view，`chunk` → 连续切片 view）
- 中间张量数量少，只有 `cat` 步骤产生 1 个新 tensor

## 新路径：`apply_rotary_emb_torch` + `rotate_half(interleaved=True)`

来源：`diffusion/layers/rope.py`，被 `RotaryEmbedding.forward_native` 调用（NPU 无 mindiesd 时的 fallback）。

### 代码

```python
def rotate_half(x, interleaved=False):
    # interleaved=True 分支
    x1, x2 = x[..., ::2], x[..., 1::2]  # 步长索引，非连续内存
    return rearrange(
        torch.stack((-x2, x1), dim=-1),
        "... d two -> ... (d two)", two=2
    )

def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    # interleaved=True 分支
    ro_dim = cos.shape[-1] * 2
    cos = repeat(cos, "... d -> ... 1 (d 2)")  # 实际展开，内存分配
    sin = repeat(sin, "... d -> ... 1 (d 2)")  # 实际展开，内存分配
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )
```

### 输入格式

- `x`：`[B, S, N, D]`（BSND）
- `cos, sin`：`[S, dim/2]`（半维度，非 interleaved）

### 算子分解（每次调用）

| 步骤 | 操作 | 类型 | 内存开销 |
|------|------|------|----------|
| 1 | `repeat(cos, "... d -> ... 1 (d 2)")` | **kernel** | 将 `[S, dim/2]` 展开为 `[S, 1, dim]`，需分配新 tensor 并交错填充 |
| 2 | `repeat(sin, "... d -> ... 1 (d 2)")` | **kernel** | 同上 |
| 3 | `x[..., ::2]` | view | 步长索引，产生**非连续**内存视图 |
| 4 | `x[..., 1::2]` | view | 步长索引，产生**非连续**内存视图 |
| 5 | `-x2` | **kernel** | 对非连续 tensor 取负，需隐式 contiguous |
| 6 | `torch.stack((-x2, x1), dim=-1)` | **kernel** | 对非连续 tensor stack，需内存分配 + 拷贝 |
| 7 | `rearrange("... d two -> ... (d two)")` | view | reshape |
| 8 | `x[..., :ro_dim] * cos` | **kernel** | 乘法 |
| 9 | `rotate_half(...) * sin` | **kernel** | 乘法 |
| 10 | `... + ...` | **kernel** | 逐元素加 |
| 11 | `torch.cat([..., x[..., ro_dim:]], dim=-1)` | **kernel** | 拼接（ro_dim == D 时 `x[..., ro_dim:]` 为空，但 cat 仍会 dispatch） |

**实际 NPU kernel launch：7-8 次**（2 × repeat、neg、stack、2 × mul、add、cat）

### 关键特点

- sin/cos 通过 `repeat` 实际展开数据（`[S, dim/2]` → `[S, 1, dim]`），每次调用产生 2 个新 tensor
- `rotate_half(interleaved=True)` 使用 `x[..., ::2]` 步长索引，产生**非连续内存视图**
  - 后续 `torch.stack` 操作需要先将非连续 tensor 拷贝为连续内存，再执行 stack
  - **NPU 对步长非连续访问性能极差**，远劣于连续内存的 chunk/split
- 最终 `torch.cat` 即使在 `x[..., ro_dim:]` 为空时仍会触发 kernel dispatch

## 逐项对比

| 对比维度 | 旧路径 `apply_rotary_pos_emb` | 新路径 `apply_rotary_emb_torch` | 影响 |
|----------|-------------------------------|--------------------------------|------|
| **sin/cos 广播方式** | `unsqueeze` → stride 广播（零拷贝） | `repeat("d -> 1 (d 2)")` → 实际分配 + 交错拷贝 | 每次调用多 **2 次内存分配 + 数据拷贝** |
| **旋转操作内存模式** | `chunk(2, -1)` → 连续内存切片 | `[::2]` / `[1::2]` → 步长索引，非连续内存 | NPU stride 访问性能差，后续 stack 需隐式 contiguous |
| **旋转操作算子数** | `cat` 1 次 | `neg` + `stack` + `rearrange` | 多 **1-2 次 kernel dispatch** |
| **kernel launch 总数** | ~4 次 | ~7-8 次 | **每次调用多约 4 次 kernel dispatch** |
| **中间 tensor 分配** | 1 个（cat 结果） | 4-5 个（2 × repeat + neg + stack + cat） | 更多 NPU 内存分配开销 |
| **输入 sin/cos 维度** | `[S, dim]` 全维度 | `[S, dim/2]` 半维度 | 新路径需要 repeat 展开才能参与运算 |

## 累积开销估算

| 参数 | 值 |
|------|-----|
| MMDiT block 数量 | 42 |
| 每 block rope 调用次数 | 6（3 段 × q/k） |
| 每次 forward 总调用次数 | 252 |
| 每次调用额外 kernel launch | ~4 次 |
| 每次 forward 额外 kernel launch | **~1008 次** |
| 去噪步数（典型值） | 8 步 |
| 整个推理额外 kernel launch | **~8064 次** |

在 NPU 上，单次 kernel launch 开销约 0.1-0.2ms（含调度 + 同步），1008 次额外 launch 对应约 **0.1s-0.2s/step**，8 步去噪累积约 **0.8s-1.6s**，与观测到的 1.5s 劣化吻合。

## 根因总结

性能劣化并非来自数学计算量增加，而是来自 **算子实现模式差异**：

1. **sin/cos 展开方式**：旧路径 `unsqueeze` 零拷贝广播 vs 新路径 `repeat` 实际数据拷贝
2. **旋转操作内存访问模式**：旧路径 `chunk` 连续内存切片 vs 新路径 `[::2]` 步长非连续索引
3. **算子数量**：新路径每次调用多 ~4 个 kernel launch，252 次调用累积成为瓶颈

`apply_rotary_emb_torch` 是一个通用实现，同时支持 interleaved 和 non-interleaved 两种模式。其 interleaved 分支通过 `repeat` 展开和步长索引实现通用性，但牺牲了 NPU 上的性能。而旧路径 `apply_rotary_pos_emb` 专为 interleaved（GPT-J）模式设计，全程使用连续内存操作，在 NPU 上更高效。

## 优化方向

| 方案 | 描述 | 预期收益 |
|------|------|----------|
| **安装 mindiesd** | `RotaryEmbedding.forward_npu` 走 mindiesd fused kernel，单次调用替代全部中间算子 | 完全消除劣化，预期反超旧路径 |
| **NPU fallback 快速路径** | 在 `RotaryEmbedding.forward_npu` 无 mindiesd 时，复用旧路径的 `unsqueeze` 广播 + `rotate_every_two` 逻辑，而非走通用 `apply_rotary_emb_torch` | 消除 ~1.5s 劣化 |
| **预展开 sin/cos** | 在 `_convert_spatial_freq_for_inference` 中将 `[S, dim/2]` 预展开为 `[S, dim]`，避免 per-block repeat | 消除 ~0.3-0.5s（252 次 repeat × 2） |
