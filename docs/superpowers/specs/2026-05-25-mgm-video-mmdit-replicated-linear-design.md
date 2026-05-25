# MGM-Video MMDiT: nn.Linear → ReplicatedLinear Replacement

**Date:** 2026-05-25
**Status:** Draft
**Scope:** `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks.py`

## Problem

mgm_video 的 mmdit 模块中 attention 和 MLP 核心层仍使用 `nn.Linear`，无法利用 vllm-omni 并行框架的 `ReplicatedLinear` 特性（统一量化接口、weight_loader 标准化、后续 TP 扩展基础）。

之前简单穿刺替换后，在 `diffusers_loader.py` 的 `_process_weights_after_loading` 中 `module.to(target_device)` 处报空指针错误（`target_device` 为 cpu）。

## Root Cause Analysis

### 加载流程

mgm_video 已注册在 `_DIFFUSION_MODELS` registry 中，走 `diffusers_loader.py` 标准路径：

```
diffusers_loader.load_model()
  → target_device = torch.device(load_device)   # "cpu" when layerwise offload enabled
  → with target_device:
      → initialize_model(od_config)
          → MGMVideoPipeline(od_config=od_config)
              → create_transformer_from_config()  # ReplicatedLinear params created on target_device
  → self.load_weights(model)                      # AutoWeightsLoader via weight_loader
  → _process_weights_after_loading(model, target_device)
```

### 为什么正确替换不会报错

1. **参数创建**：模型在 `with target_device:` 上下文中初始化 → `ReplicatedLinear.__init__` 调用 `torch.empty(...)` → 参数创建在 `target_device`（CPU 或 NPU）上
2. **权重加载**：`AutoWeightsLoader` 通过 `weight_loader` 加载权重到参数所在设备
3. **后处理检查**：`_process_weights_after_loading` 检测 `module_device == target_device` → `needs_device_move = False` → **不调用 `.to()`**
4. **no-op 路径**：`UnquantizedLinearMethod.process_weights_after_loading()` 在非 CPU 平台（NPU）上是 no-op

### 穿刺报错原因

穿刺时替换不完整（构造参数缺失、weight_loader 未正确衔接等），导致参数处于异常设备状态（如 meta device），触发 `.to("cpu")` 失败。

## Design

### Approach

方案 A：直接替换 + 利用现有 weight_loader 流程。**仅修改 `mmdit_blocks.py`**，不侵入 vllm-omni 框架层代码（`diffusers_loader.py` 等）。

### Replacement Scope

替换 mmdit 中 attention/MLP 核心层的 6 个 `nn.Linear`，保留 embedding 和 adaLN 等小层不变。

| Class | Attribute | Before | After |
|-------|-----------|--------|-------|
| `JoinAttention` | `qkv_x` | `nn.Linear(n_embd, 3*n_embd)` | `ReplicatedLinear(n_embd, 3*n_embd, bias=True, return_bias=False)` |
| `JoinAttention` | `qkv_y` | `nn.Linear(n_embd, 3*n_embd)` | 同上 |
| `JoinAttention` | `proj_x` | `nn.Linear(n_embd, n_embd)` | `ReplicatedLinear(n_embd, n_embd, bias=True, return_bias=False)` |
| `JoinAttention` | `proj_y` | `nn.Linear(n_embd, n_embd)` | 同上 |
| `MLP` | `dense_h_to_4h` | `nn.Linear(n_embd, 4*n_embd)` | `ReplicatedLinear(n_embd, 4*n_embd, bias=True, return_bias=False)` |
| `MLP` | `dense_4h_to_h` | `nn.Linear(4*n_embd, n_embd)` | `ReplicatedLinear(4*n_embd, n_embd, bias=True, return_bias=False)` |

### Layers NOT Replaced

保留为 `nn.Linear`（不在计算密集路径上，且多在 `nn.Sequential` 中，替换收益低）：

- `MMDiTBlock.adaLN_modulation_x` / `adaLN_modulation_y`（Sequential 内，4 个 nn.Linear）
- `FinalLayer.linear` / `FinalLayer.adaLN_modulation`（2 个 nn.Linear）
- `TimestepEmbedder.mlp`（Sequential 内，2 个 nn.Linear）
- `SizeEmbedder.mlp`（Sequential 内，2 个 nn.Linear）
- `CaptionEmbedder.y_proj`（timm Mlp，外部模块）

### Why No Forward Changes

`ReplicatedLinear.forward()` 在 `return_bias=False` 时直接返回单个 tensor（`linear.py:393-394`），行为与 `nn.Linear` 完全一致。flux_transformer.py 已验证此模式（如 `self.proj_mlp(x)` 直接使用，无 tuple 解包）。

### Why No load_weights Changes

1. `AutoWeightsLoader`（`pipeline_mgm_video.py:758`）自动识别参数上的 `weight_loader` 属性
2. `ReplicatedLinear.weight_loader` 直接 `param.data.copy_(loaded_weight)`（`linear.py:382`），shape 校验与原始 nn.Linear 权重一致
3. 现有的 fc1→dense_h_to_4h / fc2→dense_4h_to_h 重映射仅是字符串替换，不受层类型影响

### File Changes

| File | Change |
|------|--------|
| `mmdit_blocks.py` | 新增 `from vllm.model_executor.layers.linear import ReplicatedLinear`；MLP 替换 2 个 nn.Linear；JoinAttention 替换 4 个 nn.Linear |

**不改动**：`diffusers_loader.py`、`pipeline_mgm_video.py`、`mmdit.py`、所有 forward 方法

## Verification

1. **权重加载**：验证 `AutoWeightsLoader` 成功加载所有 6 个 ReplicatedLinear 的权重，无 missing/unexpected keys
2. **精度对齐**：替换前后用相同输入比对 mmdit 输出，应完全一致（`return_bias=False` 时与 nn.Linear 计算等价）
3. **端到端推理**：运行完整的 mgm_video 推理流程，生成视频质量无退化
4. **layerwise offload**：验证 `self.transformer.to(self.device)` 不报错（`pipeline_mgm_video.py:776`）

## Risks

1. **TP 初始化**：`LinearBase.__init__` 调用 `get_tensor_model_parallel_rank()`，需确保分布式环境已初始化。vllm-omni 在模型创建前初始化 TP group，风险低。
2. **权重 dtype**：`ReplicatedLinear` 使用 `torch.get_default_dtype()` 作为默认 dtype。模型在 `set_default_torch_dtype(od_config.dtype)` 上下文中创建，与原始行为一致。
3. **nn.Linear bias=True 默认值**：原始代码 `nn.Linear(n_embd, 3*n_embd)` 默认 `bias=True`，ReplicatedLinear 同样默认 `bias=True`，显式指定确保无遗漏。
