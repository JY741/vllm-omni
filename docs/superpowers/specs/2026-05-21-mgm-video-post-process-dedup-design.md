# MGM Video Post-Process 去重设计

## 背景

`vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py` 将裸仓推理代码适配到 vllm-omni 框架后，`MGMVideoPipeline.forward()` 中在 NPU 上完成了 VAE decode 结果的后处理（clamp → normalize → uint8 → CPU），目的是：
- 性能：NPU 并行处理比 CPU 串行遍历快约 20 倍
- 稳定性：及时释放 float32 大 tensor 占用的 NPU 内存，避免 VAE decode 耗时跳变（5s → 9~10s）
- 传输量：IPC 数据量从 float32 的 1.27 GB 降至 uint8 的 334 MB

但 `get_mgm_video_post_process_func()`（引擎层调用）仍会执行同样的后处理，导致：
1. 后处理被执行了两次
2. 对已经是 uint8 的数据重复执行 clamp/sub/div/mul 会产生数值错误
3. 之前尝试用 `_post_processed` 标志让引擎跳过，但 `DiffusionOutput` 无此字段，引擎也不检查它

目标：**在不修改 vllm-omni 核心代码的前提下，消除重复后处理。**

## 方案对比

| 方案 | 描述 | 侵入性 | 是否保留 NPU 后处理 | 推荐度 |
|------|------|--------|---------------------|--------|
| A（推荐） | `post_process_func` 通过 `dtype` 隐式检测状态 | 仅改 MGM pipeline | 是 | ★★★ |
| B | 新增 `DiffusionOutput._post_processed` 字段，引擎检查 | 改 `data.py` + `diffusion_engine.py` | 是 | ★★☆ |
| C | 移除 `forward()` NPU 后处理，完全交给引擎 | 仅改 MGM pipeline | 否 | ★☆☆ |

选择方案 A 的理由：
- **零侵入 vllm-omni 核心**：不新增字段、不改引擎、不改 registry
- **保留性能收益**：端到端快 ~10%，VAE decode 耗时稳定
- **兜底安全**：若未来有路径绕过 NPU 后处理，`post_process_func` 仍可正确处理 float32 输入

## 设计方案

### 核心思路

利用 tensor 的 `dtype` 作为**隐式状态标志**：
- `float32` → 原始 VAE 输出，需要完整后处理
- `uint8` → `forward()` 已在 NPU 上完成后处理，只需 `.numpy()`

### 修改点

#### 1. `get_mgm_video_post_process_func`（`pipeline_mgm_video.py:111-137`）

```python
def get_mgm_video_post_process_func(od_config: OmniDiffusionConfig):
    """Post-process function for MGM-Video.

    forward() 中已在 NPU 上完成的后处理：
      clamp → normalize → uint8 → permute → CPU

    本函数通过 dtype 检测状态：
      - uint8：NPU 后处理已完成，直接转 numpy
      - float32：兜底路径，执行完整 CPU 后处理
    """

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video

        # forward() 已在 NPU 上完成后处理（dtype=uint8, 已 permute+CPU）
        if video.dtype == torch.uint8:
            return video.numpy()

        # 兜底：原始 VAE 输出，执行完整后处理
        low, high = -1.0, 1.0
        video = torch.clamp(video, min=low, max=high)
        video = video.sub(low).div(max(high - low, 1e-5))
        video = video.mul(255).add(0.5).clamp(0, 255)
        video = video.permute(0, 2, 3, 4, 1)
        video = video.to("cpu", torch.uint8).numpy()
        return video

    return post_process_func
```

#### 2. `forward()` 返回语句（`pipeline_mgm_video.py:731`）

```python
# 修改前（_post_processed 字段不存在，运行时报 TypeError）：
# return DiffusionOutput(output=output, _post_processed=(output_type != "latent"))

# 修改后：
return DiffusionOutput(output=output)
```

#### 3. `forward()` NPU 后处理注释（`pipeline_mgm_video.py:718-724`）

更新注释，说明 post_process_func 会通过 dtype 检测自动跳过重复处理，而非声称"引擎会检测"（引擎实际不检测）。

### 状态流转图

```
output_type == "latent"
  └── 直接返回 latents（float32）
      └── post_process_func: output_type=="latent" → 直接返回

output_type != "latent"（正常推理）
  ├── forward():
  │     ├── VAE decode → float32 [B,C,T,H,W]
  │     ├── clamp / normalize / uint8
  │     ├── permute → [B,T,H,W,C]
  │     └── .cpu() → uint8 CPU tensor
  │         └── DiffusionOutput(output=uint8_tensor)
  │             └── engine.post_process_func(video=uint8_tensor)
  │                 └── dtype==uint8 → video.numpy() → 返回
  │
  └── 某兜底路径（未来可能）
        └── DiffusionOutput(output=float32_tensor)
            └── engine.post_process_func(video=float32_tensor)
                └── dtype==float32 → 完整后处理 → 返回
```

## 兼容性

| 场景 | 行为 |
|------|------|
| `output_type="latent"` | 直接返回，不受影响 |
| 正常推理（forward 已处理） | `post_process_func` 检测到 uint8，仅 `.numpy()` |
| 兜底/旁路路径（float32 输入） | `post_process_func` 完整后处理，安全兜底 |

## 不修改的文件

- `vllm_omni/diffusion/data.py` — 不新增 `_post_processed` 字段
- `vllm_omni/diffusion/diffusion_engine.py` — 不改引擎调用逻辑
- `vllm_omni/diffusion/registry.py` — 不改 registry 映射

## 验证计划

1. 跑通 MGM Video 端到端推理，确认输出结果正确（数值不变）
2. 对比前后性能：端到端时间、VAE decode 耗时稳定性
3. 确认 `post_process_func` 只执行 `.numpy()`，不再重复 clamp/normalize/permute
