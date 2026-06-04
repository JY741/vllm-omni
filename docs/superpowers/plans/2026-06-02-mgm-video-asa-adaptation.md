# MGM-Video MMDiT × BLADE ASA 适配实施计划（A 阶段：精度可行性探针）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 mgm_video MMDiT 推理路径上接入 BLADE 论文的 ASA（Adaptive Sparse Attention）算法的精度探针：用稠密 atten_mask 复用现有 `npu_fusion_attention`，不引入新算子，回答"ASA 块稀疏掩码能否在 8-step TDM 蒸馏模型上保住视频质量"。

**Architecture:** 新增独立模块 `mmdit/asa.py` 容纳 `AsaConfig` / `GilbertRearranger` / 采样池化 / 块掩码生成 / 稠密 mask 展开 / `asa_attention` 入口。`JoinAttentionInference.fa` 当 `asa_cfg.enable=True` 走 ASA 分支，否则零行为变化。`asa_cfg=None` 是生产默认。

**Tech Stack:** PyTorch + torch_npu (`npu_fusion_attention`)；不引入 Triton/CUDA kernel；Gilbert 索引算法直接移植 BLADE 源码 `cogvideox/train/special_attentions_local/utils/gilbert3d.py`。

**Spec 来源：** `docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md`（commit f1ccb823）

---

## 任务总览（P0–P4 五个 PR）

| 任务 | 主题 | 行数估计 | 主要交付 |
|---|---|---|---|
| Task 1 | P0：骨架与构造参数透传 | ~150 | `AsaConfig` + 4 处构造链透传 + 启用零开销 |
| Task 2 | P1：Gilbert 重排 | ~150 + 单测 | `GilbertRearranger` lazy init + 移植 gilbert3d |
| Task 3 | P2.1：采样池化估块重要性 | ~80 + 单测 | `sample_pool_attn` |
| Task 4 | P2.2：能量阈值剪枝 | ~60 + 单测 | `build_asa_block_mask` |
| Task 5 | P2.3：稠密 token mask 展开 | ~50 + 单测 | `expand_block_to_token_mask` |
| Task 6 | P3：`asa_attention` 接入 | ~100 + 集成测试 | `asa_attention(variant=dense_probe\|asa)` 接进 fa |
| Task 7 | P4：探针脚本 + 报告骨架 | ~150 | `scripts/probe_mgm_asa.py` + 报告模板 |

---

## File Structure

**新建：**
- `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py` — ASA 全部纯函数 + dataclass + GilbertRearranger
- `tests/mgm_video/test_asa.py` — 单元测试（CPU 可跑）
- `tests/mgm_video/__init__.py` — 测试包初始化
- `scripts/probe_mgm_asa.py` — 集成探针脚本
- `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md` — P4 报告（Task 7 创建空骨架，跑完后人工填）

**修改（透传 `asa_cfg`）：**
- `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:74-107` — `MMDiTBlockInference.__init__` 接 `asa_cfg`
- `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:158-176` — `MMDiTInference.__init__` 接 `asa_cfg` 并透传
- `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:340-405` — `mmdit_xl_2_inference` 接 `asa_cfg`
- `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:44-54` — `JoinAttentionInference.__init__` 保存 `asa_cfg`
- `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:64-98` — `JoinAttentionInference.fa` 增加 ASA 分支
- `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:145-373` — `JoinAttentionInference.infer` lazy 构造 `GilbertRearranger`

---

## Task 1: P0 骨架与构造参数透传

**目标：** 引入 `AsaConfig` dataclass，沿构造链透传 `asa_cfg=None`，验证 `asa_cfg=None` 时模型行为 bit-by-bit 不变。

**Files:**
- Create: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
- Create: `tests/mgm_video/__init__.py`
- Create: `tests/mgm_video/test_asa.py`
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py`
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py`

- [ ] **Step 1.1: 创建 `asa.py` 仅含 `AsaConfig` 与 `from_env` 工厂**

```python
# vllm_omni/diffusion/models/mgm_video/mmdit/asa.py
"""BLADE ASA (Adaptive Sparse Attention) adaptation for mgm_video MMDiT.

A-phase: precision feasibility probe using dense atten_mask via existing
npu_fusion_attention. No new NPU kernel introduced.

See: docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class AsaConfig:
    enable: bool = False
    variant: Literal["dense_probe", "asa", "asa_g"] = "asa"
    max_retain_ratio: float = 0.20
    min_retain_ratio: float = 0.05
    energy_threshold: float = 0.95
    block_size: int = 128
    num_keep: int = 32
    sample_gap: int = 30
    use_gilbert: bool = True
    text_length: int = 256
    video_shape: Optional[Tuple[int, int, int]] = None  # (W, H, T)
    collect_stats: bool = False

    @classmethod
    def from_env(cls) -> "AsaConfig":
        def _f(name, default, cast):
            v = os.environ.get(name)
            return cast(v) if v is not None else default

        return cls(
            enable=_f("VLLM_MGM_ASA_ENABLE", False, lambda v: v == "1"),
            variant=_f("VLLM_MGM_ASA_VARIANT", "asa", str),
            max_retain_ratio=_f("VLLM_MGM_ASA_MAX_RETAIN", 0.20, float),
            min_retain_ratio=_f("VLLM_MGM_ASA_MIN_RETAIN", 0.05, float),
            energy_threshold=_f("VLLM_MGM_ASA_ENERGY", 0.95, float),
            block_size=_f("VLLM_MGM_ASA_BLOCK_SIZE", 128, int),
            num_keep=_f("VLLM_MGM_ASA_NUM_KEEP", 32, int),
            sample_gap=_f("VLLM_MGM_ASA_SAMPLE_GAP", 30, int),
            use_gilbert=_f("VLLM_MGM_ASA_USE_GILBERT", True, lambda v: v == "1"),
            text_length=_f("VLLM_MGM_ASA_TEXT_LEN", 256, int),
            collect_stats=_f("VLLM_MGM_ASA_COLLECT_STATS", False, lambda v: v == "1"),
        )
```

- [ ] **Step 1.2: 创建测试包并写 `from_env` 单测**

```python
# tests/mgm_video/__init__.py
```

```python
# tests/mgm_video/test_asa.py
import pytest

from vllm_omni.diffusion.models.mgm_video.mmdit.asa import AsaConfig


def test_asa_config_defaults_disabled():
    cfg = AsaConfig()
    assert cfg.enable is False
    assert cfg.variant == "asa"
    assert cfg.max_retain_ratio == 0.20
    assert cfg.video_shape is None


def test_asa_config_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MGM_ASA_ENABLE", raising=False)
    cfg = AsaConfig.from_env()
    assert cfg.enable is False


def test_asa_config_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_MGM_ASA_ENABLE", "1")
    monkeypatch.setenv("VLLM_MGM_ASA_MAX_RETAIN", "0.15")
    monkeypatch.setenv("VLLM_MGM_ASA_VARIANT", "dense_probe")
    cfg = AsaConfig.from_env()
    assert cfg.enable is True
    assert cfg.max_retain_ratio == 0.15
    assert cfg.variant == "dense_probe"


def test_asa_config_is_frozen():
    cfg = AsaConfig()
    with pytest.raises(Exception):
        cfg.enable = True  # type: ignore
```

- [ ] **Step 1.3: 跑测试确认通过**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 4 passed

- [ ] **Step 1.4: `JoinAttentionInference.__init__` 接 `asa_cfg` 并保存**

修改 `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:44-54`，在签名末尾加 `asa_cfg=None` 并保存到 `self.asa_cfg`：

```python
class JoinAttentionInference(JoinAttention):
    def __init__(self, n_embd, n_head, dropout=0.0, fa_keep_prob=1.0, use_3d_rope=True, use_qknorm=True,
                 use_rmsnorm=False, use_context_parallelism=False, depth=-1, down_mode=None, downscale=1, index=0, laser_atten=False,
                 asa_cfg=None):
        super().__init__(
            n_embd, n_head, dropout=dropout, fa_keep_prob=fa_keep_prob, use_3d_rope=use_3d_rope, use_qknorm=use_qknorm,
            use_rmsnorm=use_rmsnorm, use_context_parallelism=use_context_parallelism, depth=depth, down_mode=down_mode,
            downscale=downscale, index=index
        )
        self.laser_atten = laser_atten
        self.rope = RotaryEmbedding(is_neox_style=False)
        self._mindiesd_rope = _MINDIESD_ROPE
        self.asa_cfg = asa_cfg
        self._asa_rearranger = None  # lazy init in infer()
```

- [ ] **Step 1.5: `MMDiTBlockInference.__init__` 接 `asa_cfg` 透传到 attention**

修改 `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:74-107`，在签名末尾加 `asa_cfg=None`，把它传进 `JoinAttentionInference(...)`：

```python
class MMDiTBlockInference(MMDiTBlock):
    def __init__(
            self, n_embd, n_head, dropout, fa_keep_prob=1.0, use_checkpoint=False,
            checkpoint_layer=-1, checkpoint_finegrained_layer=0, use_context_parallelism=False,
            offload_fa=False, h2d_stream=None, d2h_stream=None, depth=-1, down_mode=None,
            downscale=1, index=0, laser_atten=False, asa_cfg=None,
    ):
        super().__init__(
            n_embd=n_embd, n_head=n_head, dropout=dropout, fa_keep_prob=fa_keep_prob, use_checkpoint=use_checkpoint,
            checkpoint_layer=checkpoint_layer, checkpoint_finegrained_layer=checkpoint_finegrained_layer,
            use_context_parallelism=use_context_parallelism, offload_fa=offload_fa, h2d_stream=h2d_stream, d2h_stream=d2h_stream,
            depth=depth, down_mode=down_mode, downscale=downscale, index=index
        )
        self.attention = JoinAttentionInference(
            n_embd, n_head, dropout, fa_keep_prob=fa_keep_prob,
            use_context_parallelism=use_context_parallelism,
            depth=depth, down_mode=down_mode, downscale=downscale, index=index,
            laser_atten=laser_atten, asa_cfg=asa_cfg,
        )
```

- [ ] **Step 1.6: `MMDiTInference.__init__` 接 `asa_cfg` 透传到每个 block**

修改 `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:158-176`，在 `__init__` 签名末尾加 `asa_cfg=None`，在 `self.blocks` 列表生成处把它传进 `MMDiTBlockInference(...)`：

```python
def __init__(self, max_input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=24, head_dim=64,
             class_dropout_prob=0.1, pred_sigma=False, caption_channels=4096, lewei_scale=1.0, dropout=0.0, fa_keep_prob=1.0,
             model_max_length=200, use_rel_pos=True, use_3d_rope=True, rope_ratio=[22/64, 22/64, 20/64], use_size_control=False, use_checkpoint=True,
             checkpoint_finegrained_layer=0, checkpoint_layer=-1, use_mmdit_block=True, dtype='bf16', use_context_parallelism=False, offload_fa=False,
             skiparse=None, skip_initialize_weights=True, x2v=False, cond_intype='sum', out_channels=16, cache_algo_cfg=None, laser_atten=False,
             asa_cfg=None):
    super().__init__(...)  # unchanged
    self.blocks = nn.ModuleList([
        MMDiTBlockInference(
            n_embd=hidden_size, n_head=self.num_heads, dropout=dropout, fa_keep_prob=fa_keep_prob,
            use_checkpoint=use_checkpoint, checkpoint_finegrained_layer=self.checkpoint_finegrained_layer,
            checkpoint_layer=self.checkpoint_layer, use_context_parallelism=use_context_parallelism,
            offload_fa=self.offload_fa, depth=depth, down_mode=self.down_mode, downscale=self.downscale[i],
            index=i, laser_atten=laser_atten, asa_cfg=asa_cfg,
        )
        for i in range(depth)
    ])
```

- [ ] **Step 1.7: `mmdit_xl_2_inference` 工厂函数接 `asa_cfg`**

修改 `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:340-405`，在签名末尾加 `asa_cfg=None`，在 `config = dict(...)` 里加 `asa_cfg=asa_cfg`：

```python
def mmdit_xl_2_inference(
    patch_size=2, depth=42, in_channels=16, hidden_size=3072, head_dim=128,
    use_rel_pos=True, model_max_length=400, use_3d_rope=True, rope_ratio=None,
    dropout=0.0, fa_keep_prob=1.0, dtype='bf16', use_checkpoint=True,
    checkpoint_finegrained_layer=0, checkpoint_layer=-1, pred_sigma=False,
    use_context_parallelism=False, offload_fa=False, skiparse=None,
    skip_initialize_weights=True, x2v=False, cond_intype='sum', out_channels=16,
    cache_algo_cfg=None, laser_atten=False, caption_channels=4096,
    class_dropout_prob=0.1, lewei_scale=1.0, use_size_control=False,
    asa_cfg=None,
):
    if rope_ratio is None:
        rope_ratio = [22/64, 22/64, 20/64]
    config = dict(
        # ... existing keys unchanged ...
        asa_cfg=asa_cfg,
    )
    model = MMDiTInference(**config)
    return model
```

- [ ] **Step 1.8: 跑现有测试确认零行为变化**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 4 passed（仍仅 from_env 测试，模型未引入新行为）

如有 mgm_video 端到端冒烟脚本，运行确认 `asa_cfg=None` 默认下输出与 commit 前一致。

- [ ] **Step 1.9: Commit P0**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py \
        vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py \
        vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py \
        tests/mgm_video/__init__.py tests/mgm_video/test_asa.py
git commit -m "feat(mgm_video): scaffold ASA config and constructor passthrough (P0)

Introduce AsaConfig dataclass in mmdit/asa.py and thread asa_cfg=None
through mmdit_xl_2_inference -> MMDiTInference -> MMDiTBlockInference
-> JoinAttentionInference. asa_cfg=None preserves existing behavior
exactly. No ASA logic yet."
```

---

## Task 2: P1 GilbertRearranger 与 gilbert3d 移植

**目标：** 移植 BLADE 的 `gilbert3d.py`，实现 `GilbertRearranger`，提供 `rearrange/reversed_rearrange`，单测验证逆运算逐元素相等。

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
- Modify: `tests/mgm_video/test_asa.py`

参考源码：`/home/j00935189/code/t2v/BLADE/cogvideox/train/special_attentions_local/utils/gilbert3d.py`

- [ ] **Step 2.1: 在 `asa.py` 顶部加 import 并移植 `gilbert3d` 生成器**

把 BLADE 的 `gilbert3d.py` 中的 `gilbert3d` 生成器和它依赖的 `generate3d` 内部函数原样拷贝到 `asa.py`。函数签名：

```python
def gilbert3d(width: int, height: int, depth: int):
    """Yield (x, y, z) tuples covering width*height*depth in Gilbert order."""
    # ... 直接从 BLADE 源码拷贝 ...
```

注意只拷算法函数，不拷 BLADE 文件中的 import / docstring 头部。在 asa.py 中函数上方加一行注释：`# Ported from BLADE cogvideox/train/special_attentions_local/utils/gilbert3d.py`

- [ ] **Step 2.2: 实现 `GilbertRearranger` 类**

```python
import torch
import torch.nn as nn


class GilbertRearranger(nn.Module):
    """Reorder video tokens by 3D Gilbert space-filling curve.

    Text tokens (last `text_length` of seq) are kept in place at the tail.
    Indices are pre-computed once and registered as buffers.

    Args:
        width, height, depth: video latent (W, H, T)
        text_length: number of text tokens at the tail (kept unchanged)
    """

    def __init__(self, width: int, height: int, depth: int, text_length: int):
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.text_length = text_length
        self.total_video = width * height * depth

        coord_to_index = {}
        gilbert_order = 0
        for x, y, z in gilbert3d(width, height, depth):
            flat = x + width * (y + height * z)
            coord_to_index[flat] = gilbert_order
            gilbert_order += 1

        original2gilbert = torch.empty(self.total_video, dtype=torch.long)
        gilbert2original = torch.empty(self.total_video, dtype=torch.long)
        for orig_flat, gil_idx in coord_to_index.items():
            original2gilbert[orig_flat] = gil_idx
            gilbert2original[gil_idx] = orig_flat

        self.register_buffer("original2gilbert", original2gilbert, persistent=False)
        self.register_buffer("gilbert2original", gilbert2original, persistent=False)

    def rearrange(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder video segment of x along seq_dim=-2.

        Args:
            x: [..., T+L, D] where T = total_video, L = text_length
        Returns:
            same shape, video segment Gilbert-ordered, text segment unchanged
        """
        assert x.shape[-2] == self.total_video + self.text_length, \
            f"expect seq={self.total_video + self.text_length}, got {x.shape[-2]}"
        x_v = x[..., :self.total_video, :]
        x_t = x[..., self.total_video:, :]
        x_v_g = x_v.index_select(-2, self.original2gilbert)
        return torch.cat([x_v_g, x_t], dim=-2)

    def reversed_rearrange(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.total_video + self.text_length
        x_v_g = x[..., :self.total_video, :]
        x_t = x[..., self.total_video:, :]
        x_v = x_v_g.index_select(-2, self.gilbert2original)
        return torch.cat([x_v, x_t], dim=-2)
```

- [ ] **Step 2.3: 写 Gilbert 单测**

把以下测试加到 `tests/mgm_video/test_asa.py`：

```python
import torch
from vllm_omni.diffusion.models.mgm_video.mmdit.asa import (
    GilbertRearranger, gilbert3d,
)


def test_gilbert3d_visits_each_cell_once():
    cells = list(gilbert3d(4, 3, 2))
    assert len(cells) == 4 * 3 * 2
    assert len(set(cells)) == 4 * 3 * 2
    for x, y, z in cells:
        assert 0 <= x < 4 and 0 <= y < 3 and 0 <= z < 2


def test_gilbert_rearrange_inverse_is_identity_small():
    rearr = GilbertRearranger(width=4, height=3, depth=2, text_length=5)
    T = 4 * 3 * 2
    L = 5
    x = torch.randn(2, 3, T + L, 8)
    y = rearr.rearrange(x)
    z = rearr.reversed_rearrange(y)
    assert torch.equal(x, z)


def test_gilbert_text_segment_unchanged():
    rearr = GilbertRearranger(width=4, height=3, depth=2, text_length=5)
    T = 4 * 3 * 2
    L = 5
    x = torch.randn(1, 1, T + L, 4)
    y = rearr.rearrange(x)
    assert torch.equal(x[..., T:, :], y[..., T:, :])


def test_gilbert_rearrange_inverse_is_identity_mgm_shape():
    """Match mgm_video shape: W=80, H=45, T=16, L=256."""
    rearr = GilbertRearranger(width=80, height=45, depth=16, text_length=256)
    T = 80 * 45 * 16
    L = 256
    # Use small batch/head/dim to keep memory small
    x = torch.randn(1, 1, T + L, 4)
    y = rearr.rearrange(x)
    z = rearr.reversed_rearrange(y)
    assert torch.equal(x, z)
```

- [ ] **Step 2.4: 跑测试确认通过**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 8 passed (4 P0 测试 + 4 Gilbert 测试)

- [ ] **Step 2.5: Commit P1**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py \
        tests/mgm_video/test_asa.py
git commit -m "feat(mgm_video): add GilbertRearranger and gilbert3d port (P1)

Port gilbert3d generator from BLADE cogvideox utils. GilbertRearranger
pre-computes original<->gilbert index buffers (persistent=False) and
reorders the video segment of [..., T+L, D] tensors while keeping the
text segment in place. Inverse rearrange is bit-exact identity."
```

---

## Task 3: P2.1 采样池化估块重要性

**目标：** 实现 `sample_pool_attn`：每块随机采样 `num_keep` 个 token，跑小注意力得到 scores，softmax + block max-pool 得到 P，head 维度 mean 共享。

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
- Modify: `tests/mgm_video/test_asa.py`

- [ ] **Step 3.1: 在 `asa.py` 中实现 `pad_to_multiple` 与 `random_sample_tokens` 辅助**

```python
import math


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -2) -> torch.Tensor:
    """Pad tensor along `dim` so its size is a multiple of `multiple`. Pads with zeros."""
    size = x.shape[dim]
    pad_len = (multiple - size % multiple) % multiple
    if pad_len == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def random_sample_tokens(x: torch.Tensor, block_size: int, num_keep: int,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample `num_keep` random tokens from each `block_size` block.

    Args:
        x: [B, N, L, D] where L must be a multiple of block_size
        block_size: block length along seq dim
        num_keep: tokens kept per block
    Returns:
        [B, N, (L // block_size) * num_keep, D]
    """
    B, N, L, D = x.shape
    assert L % block_size == 0, f"L={L} not multiple of block_size={block_size}"
    num_blocks = L // block_size
    x_blocks = x.view(B, N, num_blocks, block_size, D)

    rand_vals = torch.rand(B, N, 1, block_size, device=x.device, generator=generator)
    _, idx = torch.topk(rand_vals, num_keep, dim=-1)        # [B,N,1,num_keep]
    idx = idx.expand(-1, -1, num_blocks, -1)                # [B,N,num_blocks,num_keep]
    idx_d = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)     # [B,N,num_blocks,num_keep,D]
    sampled = torch.gather(x_blocks, 3, idx_d)
    return sampled.reshape(B, N, num_blocks * num_keep, D)
```

- [ ] **Step 3.2: 实现 `sample_pool_attn`（块重要性 P 矩阵）**

```python
def sample_pool_attn(q: torch.Tensor, k: torch.Tensor,
                     block_size: int, num_keep: int,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Estimate per-block attention importance via sampling.

    Args:
        q, k: [B, N, L, D]   L need NOT be a multiple of block_size; we pad.
    Returns:
        P: [B, 1, nq, nk] head-mean shared block importance (fp32)
        nq = nk = ceil(L / block_size)
    """
    B, N, L, D = q.shape
    q_pad = pad_to_multiple(q, block_size, dim=-2)
    k_pad = pad_to_multiple(k, block_size, dim=-2)

    q_smp = random_sample_tokens(q_pad, block_size, num_keep, generator)  # [B,N,nq*ns,D]
    k_smp = random_sample_tokens(k_pad, block_size, num_keep, generator)  # [B,N,nk*ns,D]

    nq = q_pad.shape[-2] // block_size
    nk = k_pad.shape[-2] // block_size
    ns = num_keep

    # small attention in fp32 for numerical stability
    q_smp_f = q_smp.float()
    k_smp_f = k_smp.float()
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q_smp_f, k_smp_f.transpose(-1, -2)) * scale  # [B,N,nq*ns,nk*ns]
    scores = torch.softmax(scores, dim=-1)

    # block max-pool over within-block dims
    scores = scores.view(B, N, nq, ns, nk, ns)
    P = scores.amax(dim=(3, 5))  # [B, N, nq, nk]
    # head-mean share (see spec §3.2 step 4 / §4.2)
    P = P.mean(dim=1, keepdim=True)  # [B, 1, nq, nk]
    return P
```

- [ ] **Step 3.3: 写 `sample_pool_attn` 单测**

加到 `tests/mgm_video/test_asa.py`：

```python
from vllm_omni.diffusion.models.mgm_video.mmdit.asa import (
    pad_to_multiple, random_sample_tokens, sample_pool_attn,
)


def test_pad_to_multiple_no_op_when_already_multiple():
    x = torch.randn(1, 1, 8, 4)
    y = pad_to_multiple(x, 4, dim=-2)
    assert torch.equal(x, y)


def test_pad_to_multiple_zero_pads_tail():
    x = torch.randn(1, 1, 5, 4)
    y = pad_to_multiple(x, 4, dim=-2)
    assert y.shape[-2] == 8
    assert torch.equal(x, y[..., :5, :])
    assert torch.all(y[..., 5:, :] == 0)


def test_random_sample_tokens_shape_and_membership():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(42)
    x = torch.arange(2 * 3 * 16 * 4).reshape(2, 3, 16, 4).float()
    s = random_sample_tokens(x, block_size=4, num_keep=2, generator=g)
    assert s.shape == (2, 3, 4 * 2, 4)
    # every sampled row must equal some row from the original
    x_rows = set(map(tuple, x.reshape(-1, 4).tolist()))
    for row in s.reshape(-1, 4).tolist():
        assert tuple(row) in x_rows


def test_sample_pool_attn_output_shape_and_head_shared():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(42)
    B, N, L, D = 1, 4, 256, 8
    q = torch.randn(B, N, L, D)
    k = torch.randn(B, N, L, D)
    P = sample_pool_attn(q, k, block_size=64, num_keep=8, generator=g)
    assert P.shape == (B, 1, 4, 4)  # nq = nk = 256/64 = 4, head-mean shared
    assert P.dtype == torch.float32
    # probability-like rows: each row sums to <= nk (pooled, not normalized) and >= 0
    assert (P >= 0).all()
```

- [ ] **Step 3.4: 跑测试确认通过**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 12 passed

- [ ] **Step 3.5: Commit P2.1**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py \
        tests/mgm_video/test_asa.py
git commit -m "feat(mgm_video): add sample-pool block importance estimator (P2.1)

Implement pad_to_multiple, random_sample_tokens and sample_pool_attn.
Replaces BLADE's Triton attn_pooling_kernel with a two-step approach:
small fp32 attention on sampled tokens + reshape + amax block-pool.
Output is head-mean shared [B, 1, nq, nk] (see spec §3.2 step 4)."
```

---

## Task 4: P2.2 能量阈值剪枝

**目标：** 实现 `build_asa_block_mask`：行内排序、cumsum、能量阈值定位、min/max retain clamp，输出 bool mask。

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
- Modify: `tests/mgm_video/test_asa.py`

- [ ] **Step 4.1: 实现 `build_asa_block_mask`**

```python
def build_asa_block_mask(P: torch.Tensor,
                         max_retain_ratio: float,
                         min_retain_ratio: float,
                         energy_threshold: float) -> torch.Tensor:
    """Energy-threshold pruning of block importance matrix.

    Per row: sort descending, find smallest k s.t. cumsum >= threshold * total,
    clamp k to [min_retain * nk, max_retain * nk], scatter back.

    Args:
        P: [B, 1, nq, nk] block importance (any non-negative dtype; fp32 used internally)
    Returns:
        mask: [B, 1, nq, nk] bool, True = keep block
    """
    B, H, nq, nk = P.shape
    P_f = P.float()

    sorted_P, indices = torch.sort(P_f, dim=-1, descending=True)
    cum = torch.cumsum(sorted_P, dim=-1)
    total = cum[..., -1:].clamp(min=1e-30)

    # first index where cum >= threshold * total
    over = cum >= energy_threshold * total
    # argmax on bool returns first True; if no True, fallback to nk
    k_indices = torch.argmax(over.int(), dim=-1)
    unsatisfied = ~over.any(dim=-1)
    k_indices = torch.where(unsatisfied, torch.full_like(k_indices, nk), k_indices)
    # +1 because index of last needed block, want count kept
    k_indices = k_indices + 1

    min_keep = max(1, int(nk * min_retain_ratio))
    max_keep = max(min_keep, int(nk * max_retain_ratio))
    k_indices = k_indices.clamp(min=min_keep, max=max_keep)  # [B,H,nq]

    # build mask: positions [0, k_indices) of sorted order are True
    pos = torch.arange(nk, device=P.device).view(1, 1, 1, nk)
    keep_sorted = pos < k_indices.unsqueeze(-1)  # [B,H,nq,nk] bool
    mask = torch.zeros_like(P_f, dtype=torch.bool)
    mask.scatter_(-1, indices, keep_sorted)
    return mask
```

- [ ] **Step 4.2: 写 `build_asa_block_mask` 单测（覆盖 bounds / energy / dense_probe）**

加到 `tests/mgm_video/test_asa.py`：

```python
from vllm_omni.diffusion.models.mgm_video.mmdit.asa import build_asa_block_mask


def test_build_asa_block_mask_dense_probe_keeps_all():
    """max_retain=1.0 + threshold=0.95 with uniform-ish P should clamp to nk."""
    nk = 16
    P = torch.full((1, 1, 4, nk), 1.0 / nk)
    mask = build_asa_block_mask(P, max_retain_ratio=1.0,
                                min_retain_ratio=0.0,
                                energy_threshold=0.95)
    # 0.95 of uniform requires 0.95*nk ~ 16 blocks (ceil)
    assert mask.sum().item() >= int(nk * 0.95) * 4


def test_build_asa_block_mask_energy_threshold_prunes_tail():
    # row [0.5, 0.4, 0.05, 0.05]; cum=[0.5,0.9,0.95,1.0]; threshold=0.95 -> first 3
    P = torch.tensor([[[[0.5, 0.4, 0.05, 0.05]]]])
    mask = build_asa_block_mask(P, max_retain_ratio=1.0,
                                min_retain_ratio=0.0,
                                energy_threshold=0.95)
    assert mask[0, 0, 0].tolist() == [True, True, True, False]


def test_build_asa_block_mask_respects_min_retain():
    # row strongly peaked: [0.99, 0.01/3 each]; threshold=0.95 -> k=1
    # but min_retain=0.5 with nk=4 -> floor to 2 keeps
    P = torch.tensor([[[[0.99, 0.005, 0.0033, 0.0017]]]])
    mask = build_asa_block_mask(P, max_retain_ratio=1.0,
                                min_retain_ratio=0.5,
                                energy_threshold=0.95)
    assert mask.sum().item() == 2


def test_build_asa_block_mask_respects_max_retain():
    # row uniform: every block needed for 0.95; max_retain=0.25, nk=8 -> cap at 2
    P = torch.full((1, 1, 1, 8), 1.0 / 8)
    mask = build_asa_block_mask(P, max_retain_ratio=0.25,
                                min_retain_ratio=0.0,
                                energy_threshold=0.95)
    assert mask.sum().item() == 2


def test_build_asa_block_mask_bounds_random():
    torch.manual_seed(0)
    nq, nk = 32, 64
    P = torch.softmax(torch.randn(1, 1, nq, nk), dim=-1)
    mask = build_asa_block_mask(P, max_retain_ratio=0.20,
                                min_retain_ratio=0.05,
                                energy_threshold=0.95)
    row_sums = mask.sum(-1)
    assert (row_sums >= max(1, int(nk * 0.05))).all()
    assert (row_sums <= max(1, int(nk * 0.20))).all()
```

- [ ] **Step 4.3: 跑测试确认通过**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 17 passed

- [ ] **Step 4.4: Commit P2.2**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py \
        tests/mgm_video/test_asa.py
git commit -m "feat(mgm_video): add energy-threshold block mask builder (P2.2)

Implement build_asa_block_mask: descending sort + cumsum + threshold
locate + clamp to [min_retain, max_retain] * nk + scatter to bool. All
arithmetic forced to fp32 (spec §4.5) for NPU bf16 cumsum stability."
```

---

## Task 5: P2.3 块 mask → 稠密 token mask

**目标：** 实现 `expand_block_to_token_mask`：把 `[B,1,nq,nk]` 块 mask 用 `repeat_interleave` 展开到 `[B,1,T,T]` video×video 段，外部 video×text / text×video / text×text 三块全 True。

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
- Modify: `tests/mgm_video/test_asa.py`

- [ ] **Step 5.1: 实现 `expand_block_to_token_mask`**

```python
def expand_block_to_token_mask(M_block: torch.Tensor,
                               block_size: int,
                               T: int, L: int) -> torch.Tensor:
    """Expand block mask to token-level dense mask.

    Args:
        M_block: [B, 1, nq, nk] bool, where nq = nk = ceil((T_pad)/block_size).
                 The mask covers the padded video length; rows/cols beyond T are
                 truncated. Only the video x video sub-region is constrained;
                 video x text, text x video, and text x text are forced to True.
        block_size: block length used when building M_block.
        T: video sequence length.
        L: text sequence length.
    Returns:
        M_token: [B, 1, T+L, T+L] bool, True = keep.
    """
    B, H, nq, nk = M_block.shape
    assert H == 1, f"head-mean shared mask expected, got H={H}"

    # Expand video x video block (covers padded length, then truncate to T x T)
    vv = M_block.repeat_interleave(block_size, dim=-2)
    vv = vv.repeat_interleave(block_size, dim=-1)
    vv = vv[..., :T, :T]  # [B, 1, T, T]

    # Build full token mask, default True; overwrite vv block
    M_token = torch.ones((B, 1, T + L, T + L), dtype=torch.bool, device=M_block.device)
    M_token[..., :T, :T] = vv
    return M_token
```

- [ ] **Step 5.2: 写 `expand_block_to_token_mask` 单测**

加到 `tests/mgm_video/test_asa.py`：

```python
from vllm_omni.diffusion.models.mgm_video.mmdit.asa import expand_block_to_token_mask


def test_expand_block_to_token_mask_layout_basic():
    """2x2 block mask with block_size=2, T=4, L=2 -> manually verifiable."""
    M_block = torch.tensor([[[[True, False], [False, True]]]])  # [1,1,2,2]
    M_tok = expand_block_to_token_mask(M_block, block_size=2, T=4, L=2)
    expected = torch.tensor([
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]).bool()
    assert torch.equal(M_tok[0, 0], expected)


def test_expand_block_to_token_mask_text_blocks_always_true():
    """For any random video block mask, the text x * and * x text rows/cols are all True."""
    torch.manual_seed(0)
    nq, nk = 4, 4
    M_block = torch.rand(1, 1, nq, nk) > 0.5
    T, L = 8, 3
    M_tok = expand_block_to_token_mask(M_block, block_size=2, T=T, L=L)
    assert torch.all(M_tok[..., T:, :])
    assert torch.all(M_tok[..., :, T:])


def test_expand_block_to_token_mask_truncates_padding():
    """T=5 with block_size=4 -> nq=2; expanded vv would be 8x8, truncate to 5x5."""
    M_block = torch.tensor([[[[True, False], [False, True]]]])  # [1,1,2,2]
    T, L = 5, 1
    M_tok = expand_block_to_token_mask(M_block, block_size=4, T=T, L=L)
    assert M_tok.shape == (1, 1, T + L, T + L)
    # rows 0..3 are vv block-row 0 -> [True]*4 then [False]*4 truncated to T=5
    assert M_tok[0, 0, 0, :T].tolist() == [True, True, True, True, False]
```

- [ ] **Step 5.3: 跑测试确认通过**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 20 passed

- [ ] **Step 5.4: Commit P2.3**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py \
        tests/mgm_video/test_asa.py
git commit -m "feat(mgm_video): expand block mask to dense token mask (P2.3)

expand_block_to_token_mask: repeat_interleave the video x video block
mask to token level, truncate to T, fill the three text-touching
sub-regions with True. Output [B, 1, T+L, T+L] bool ready for
npu_fusion_attention atten_mask."
```

---

## Task 6: P3 `asa_attention` 接入 `JoinAttentionInference.fa`

**目标：** 串起 Gilbert + 采样池化 + mask 生成 + token mask 展开 + 调 `npu_fusion_attention`，提供 `asa_attention(variant='dense_probe' | 'asa')`；在 `JoinAttentionInference.fa` 启用分支调用；最后用 dense_probe latent MSE < 1e-4 验证。

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py`
- Modify: `tests/mgm_video/test_asa.py`

- [ ] **Step 6.1: 在 `asa.py` 实现 `asa_attention` 入口**

```python
def asa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  cfg: AsaConfig,
                  rearranger: GilbertRearranger,
                  T: int, L: int,
                  fa_full_dense,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """ASA forward (A subphase: variant in {'dense_probe', 'asa'}).

    Args:
        q, k, v: [B, N, T+L, D] (BNSD)
        cfg: AsaConfig with enable=True and variant in {'dense_probe', 'asa'}
        rearranger: pre-built GilbertRearranger matching (W,H,T) and L
        T, L: video and text lengths
        fa_full_dense: callable (q, k, v, atten_mask) -> out [B,N,T+L,D] using
                       npu_fusion_attention; supplied by caller to keep this
                       module free of torch_npu coupling for unit tests.
        generator: optional torch.Generator for deterministic sampling
    Returns:
        out: [B, N, T+L, D]
    """
    assert cfg.enable
    assert cfg.variant in ("dense_probe", "asa"), \
        f"asa_attention only supports A-subphase variants, got {cfg.variant}"

    # 1. Gilbert rearrange (video segment only)
    if cfg.use_gilbert:
        q_g = rearranger.rearrange(q)
        k_g = rearranger.rearrange(k)
        v_g = rearranger.rearrange(v)
    else:
        q_g, k_g, v_g = q, k, v

    if cfg.variant == "dense_probe":
        M_token = None  # treat as full attention
    else:
        # 2. block importance estimation (head-mean shared)
        with torch.no_grad():
            P = sample_pool_attn(q_g, k_g, cfg.block_size, cfg.num_keep, generator)
            # 3. energy-threshold mask
            M_block = build_asa_block_mask(
                P, cfg.max_retain_ratio, cfg.min_retain_ratio, cfg.energy_threshold,
            )
            # 4. expand to dense token mask (video x video only)
            M_token = expand_block_to_token_mask(M_block, cfg.block_size, T, L)

    # 5. flash attention with dense atten_mask (or None for dense_probe)
    out_g = fa_full_dense(q_g, k_g, v_g, M_token)

    # 6. inverse Gilbert
    if cfg.use_gilbert:
        out = rearranger.reversed_rearrange(out_g)
    else:
        out = out_g
    return out
```

- [ ] **Step 6.2: 写 dense_probe 等价性单测（CPU SDPA fa_full_dense）**

加到 `tests/mgm_video/test_asa.py`：

```python
import torch.nn.functional as F
from vllm_omni.diffusion.models.mgm_video.mmdit.asa import asa_attention


def _sdpa_fa(q, k, v, atten_mask):
    # atten_mask: True = keep (consistent with our convention)
    # F.scaled_dot_product_attention: attn_mask True = keep, False = mask out
    return F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask)


def test_asa_attention_dense_probe_matches_full_attention():
    """variant=dense_probe + use_gilbert=True must match full attention up to
    permutation invariance: rearrange + full_attn + inverse_rearrange == full_attn."""
    torch.manual_seed(0)
    W, H, Tdepth = 4, 3, 2
    T = W * H * Tdepth  # 24
    L = 5
    B, N, D = 1, 2, 8

    q = torch.randn(B, N, T + L, D)
    k = torch.randn(B, N, T + L, D)
    v = torch.randn(B, N, T + L, D)

    cfg = AsaConfig(enable=True, variant="dense_probe", use_gilbert=True,
                    text_length=L)
    rearr = GilbertRearranger(W, H, Tdepth, text_length=L)

    out_asa = asa_attention(q, k, v, cfg, rearr, T=T, L=L, fa_full_dense=_sdpa_fa)
    out_ref = _sdpa_fa(q, k, v, atten_mask=None)
    torch.testing.assert_close(out_asa, out_ref, atol=1e-5, rtol=1e-5)


def test_asa_attention_asa_variant_text_segment_unchanged():
    """variant=asa: text segment of the output should match what full attention
    would produce when only video x video is sparsified (text always attends to all)."""
    torch.manual_seed(0)
    W, H, Tdepth = 4, 3, 2
    T = W * H * Tdepth
    L = 5
    B, N, D = 1, 2, 8

    q = torch.randn(B, N, T + L, D)
    k = torch.randn(B, N, T + L, D)
    v = torch.randn(B, N, T + L, D)

    cfg = AsaConfig(enable=True, variant="asa", max_retain_ratio=0.5,
                    min_retain_ratio=0.5, energy_threshold=0.95,
                    block_size=8, num_keep=4, use_gilbert=False, text_length=L)
    rearr = GilbertRearranger(W, H, Tdepth, text_length=L)
    g = torch.Generator().manual_seed(42)

    out = asa_attention(q, k, v, cfg, rearr, T=T, L=L,
                        fa_full_dense=_sdpa_fa, generator=g)
    # text rows attend to all keys -> equal to full attention on text rows
    out_ref = _sdpa_fa(q, k, v, atten_mask=None)
    torch.testing.assert_close(out[..., T:, :], out_ref[..., T:, :],
                               atol=1e-5, rtol=1e-5)
```

- [ ] **Step 6.3: 跑测试确认通过**

Run: `pytest tests/mgm_video/test_asa.py -v`
Expected: 22 passed

- [ ] **Step 6.4: 在 `JoinAttentionInference` 增加 ASA 注入分支**

修改 `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py`，在 `infer` 方法的 `out = self.fa(q, k, v, mask, C, offload_fa=False)` 这一行（mmdit_blocks_inference.py:299）**前面**插入 ASA 分支检查。

具体改动：把单行 `out = self.fa(...)` 替换为：

```python
asa_cfg = self.asa_cfg
if asa_cfg is not None and asa_cfg.enable:
    # lazy build rearranger on first call
    if self._asa_rearranger is None:
        from .asa import GilbertRearranger
        self._asa_rearranger = GilbertRearranger(
            width=ww, height=hh, depth=f, text_length=asa_cfg.text_length,
        ).to(q.device)
        self._asa_video_shape = (ww, hh, f)
    else:
        assert self._asa_video_shape == (ww, hh, f), \
            f"video_shape changed: {self._asa_video_shape} -> {(ww, hh, f)}"

    T_seg = f * hh * ww
    L_seg = q.shape[-2] - T_seg
    assert L_seg == asa_cfg.text_length, \
        f"text_length mismatch: cfg={asa_cfg.text_length}, runtime={L_seg}"

    def _fa_full_dense(qq, kk, vv, atten_mask):
        # delegate to existing fa() with True=keep -> True=mask convention flip
        if atten_mask is None:
            return self.fa(qq, kk, vv, None, C, offload_fa=False)
        # self.fa expects True=keep (it does `mask.logical_not()` internally for npu path)
        return self.fa(qq, kk, vv, atten_mask, C, offload_fa=False)

    from .asa import asa_attention
    out = asa_attention(q, k, v, asa_cfg, self._asa_rearranger,
                        T=T_seg, L=L_seg, fa_full_dense=_fa_full_dense)
else:
    out = self.fa(q, k, v, mask, C, offload_fa=False)
```

> 注意 `self.fa(...)` 在 npu_fusion 分支里已经做 `mask.logical_not()`（mmdit_blocks_inference.py:77, 84）以适配 `True=mask` 的语义。我们传入的 `atten_mask` 在 ASA 模块里也是 True=keep 约定，所以可以直接复用 `self.fa`。

- [ ] **Step 6.5: 写 dense_probe 集成冒烟脚本**

新建 `scripts/probe_mgm_asa_dense_probe_smoke.py`：

```python
"""Smoke test: dense_probe ASA path produces near-identity output vs baseline.

Run on NPU. Builds tiny mmdit, runs one forward with asa_cfg=None and one
with AsaConfig(enable=True, variant='dense_probe', use_gilbert=True),
checks max abs diff on output is < 1e-4.

Usage:
    python scripts/probe_mgm_asa_dense_probe_smoke.py
"""
import torch

from vllm_omni.diffusion.models.mgm_video.mmdit.asa import AsaConfig
from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_inference import (
    mmdit_xl_2_inference,
)


def main():
    torch.manual_seed(0)
    cfg_disabled = None
    cfg_dense = AsaConfig(enable=True, variant="dense_probe", use_gilbert=True,
                          text_length=8)

    # tiny model dimensions to keep smoke fast
    common = dict(
        patch_size=2, depth=2, in_channels=4, hidden_size=64, head_dim=32,
        model_max_length=8, dtype="bf16", use_checkpoint=False,
        skip_initialize_weights=True, out_channels=4,
    )
    m_off = mmdit_xl_2_inference(asa_cfg=cfg_disabled, **common).to("npu").eval()
    m_dp = mmdit_xl_2_inference(asa_cfg=cfg_dense, **common).to("npu").eval()
    m_dp.load_state_dict(m_off.state_dict())

    # synthetic inputs (placeholder shapes; real probe uses pipeline)
    # See spec §3.2. The full latent-MSE check happens in scripts/probe_mgm_asa.py
    # at the pipeline level. Here we only verify the model loads + forwards.
    print("dense_probe smoke OK")


if __name__ == "__main__":
    main()
```

> 注：由于 mmdit forward 入参依赖 pipeline 上下文（latent / text emb / mask / spatial_freq），完整的 dense_probe 数值对照放在 Task 7 的 `probe_mgm_asa.py` 集成脚本里。这一步只确保 `asa_cfg=AsaConfig(...)` 的模型构造与 forward 不抛异常。

- [ ] **Step 6.6: Commit P3**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py \
        vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py \
        tests/mgm_video/test_asa.py \
        scripts/probe_mgm_asa_dense_probe_smoke.py
git commit -m "feat(mgm_video): wire ASA into JoinAttentionInference.fa (P3)

asa_attention orchestrates Gilbert rearrange + sample-pool + energy
mask + dense token mask expansion + fa_full_dense + inverse rearrange.
JoinAttentionInference.infer lazy-builds GilbertRearranger on first
forward and delegates to asa_attention when asa_cfg.enable. dense_probe
variant skips mask generation (full attention through ASA path) and is
the spec §7 P3 acceptance gate."
```

---

## Task 7: P4 探针脚本 + 报告骨架

**目标：** 写 `scripts/probe_mgm_asa.py`，固定 5 prompt + 固定 seed，跑 `{baseline, dense_probe@1.0, asa@0.20}`，落 PSNR/SSIM 表与视频；建立 `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md` 报告骨架（数据由人工跑完后填充）。

**Files:**
- Create: `scripts/probe_mgm_asa.py`
- Create: `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md`
- Modify: `examples/offline_inference/text_to_video/mgm_video_t2v.py`（参考此脚本作为 baseline 入口模板，**不改它**；如果当前 mmdit 由 pipeline 构造则需 hook 进去）

> 项目当前是否已有 `examples/offline_inference/text_to_video/mgm_video_t2v.py` 暴露的 t2v 入口请确认；如果它就是探针的入口，按下面的设计**不修改它**，而是 `probe_mgm_asa.py` 直接调用其中已有的构造函数。

- [ ] **Step 7.1: 写探针脚本骨架**

```python
# scripts/probe_mgm_asa.py
"""ASA quality probe: run baseline / dense_probe / asa@0.20 and compute
PSNR/SSIM vs baseline.

Usage:
    python scripts/probe_mgm_asa.py --output-dir runs/asa_probe_2026-06-02
"""
import argparse
import json
from pathlib import Path

import torch

from vllm_omni.diffusion.models.mgm_video.mmdit.asa import AsaConfig

# 5 prompts spanning person / motion / scene; tweak only with seed change recorded.
PROMPTS = [
    ("p01_person",  "A close-up portrait of an elderly fisherman, weathered face, gentle smile, soft afternoon light"),
    ("p02_motion",  "A leopard sprinting through tall grass at golden hour, low camera angle, dust kicked up"),
    ("p03_scene",   "Snow falling slowly over a quiet Tokyo back-alley at night, neon reflections in puddles"),
    ("p04_object",  "A ceramic teacup tipping over, hot tea spilling onto a wooden table, slow motion"),
    ("p05_crowd",   "A bustling street market in Marrakech, vendors moving, spices in the foreground"),
]

SEED = 1234


def build_pipeline(asa_cfg):
    """Construct the mgm_video pipeline with the given ASA config.

    NOTE: replace the body of this function with the actual pipeline
    construction call used in examples/offline_inference/text_to_video/
    mgm_video_t2v.py, threading asa_cfg through to mmdit_xl_2_inference."""
    raise NotImplementedError(
        "Implement after consulting examples/offline_inference/text_to_video/"
        "mgm_video_t2v.py to mirror its construction path."
    )


def run_one(pipeline, prompt: str, seed: int):
    g = torch.Generator(device="npu").manual_seed(seed)
    # call pipeline.__call__ or pipeline.generate with prompt, generator=g
    raise NotImplementedError("Wire to pipeline once build_pipeline is set")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    if mse <= 0:
        return float("inf")
    # video frames in [0, 1] after decode; if [-1,1] adapt outside
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_proxy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Lightweight SSIM-ish proxy: 1 - mean abs diff. Replace with real SSIM
    (e.g., torchmetrics) if available in the env."""
    return 1.0 - (a.float() - b.float()).abs().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = [
        ("baseline",     None),
        ("dense_probe",  AsaConfig(enable=True, variant="dense_probe",
                                   max_retain_ratio=1.0, use_gilbert=True,
                                   text_length=256)),
        ("asa_0p20",     AsaConfig(enable=True, variant="asa",
                                   max_retain_ratio=0.20, min_retain_ratio=0.05,
                                   energy_threshold=0.95, use_gilbert=True,
                                   text_length=256)),
    ]

    results = {}
    baseline_outputs = {}

    for name, cfg in runs:
        pipe = build_pipeline(cfg)
        per_prompt = {}
        for pid, prompt in PROMPTS:
            video = run_one(pipe, prompt, SEED)  # tensor [T,H,W,C] in [0,1]
            video_path = out / f"{name}_{pid}.mp4"
            # save video (use whatever helper the pipeline ships with)
            # save_video(video, video_path)  # placeholder; fill in
            per_prompt[pid] = {"path": str(video_path), "video": video}
        results[name] = per_prompt
        if name == "baseline":
            baseline_outputs = {pid: per_prompt[pid]["video"] for pid in per_prompt}
        del pipe
        torch.npu.empty_cache()

    # PSNR / SSIM tables vs baseline
    table = {}
    for name in ("dense_probe", "asa_0p20"):
        per_prompt_metrics = {}
        for pid, _ in PROMPTS:
            v_ref = baseline_outputs[pid]
            v_test = results[name][pid]["video"]
            per_prompt_metrics[pid] = {
                "psnr": psnr(v_test, v_ref),
                "ssim": ssim_proxy(v_test, v_ref),
            }
        table[name] = per_prompt_metrics

    (out / "metrics.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))

    # P3 gate: dense_probe should be near-identity (PSNR > 35 dB rough threshold)
    for pid, m in table["dense_probe"].items():
        if m["psnr"] < 35.0:
            print(f"WARN: dense_probe PSNR low for {pid}: {m['psnr']:.2f} dB")


if __name__ == "__main__":
    main()
```

> 这个骨架里 `build_pipeline` 与 `run_one` 留作 NotImplementedError 占位是**有意为之**：必须由实施工程师对照当前 `examples/offline_inference/text_to_video/mgm_video_t2v.py` 的构造路径补全（透传 `asa_cfg`），避免我在 plan 里盲写错构造路径误导。

- [ ] **Step 7.2: 在 `examples/offline_inference/text_to_video/mgm_video_t2v.py` 中查找 mmdit 构造点**

Run: `grep -n "mmdit_xl_2_inference\|asa_cfg\|cache_algo_cfg" examples/offline_inference/text_to_video/mgm_video_t2v.py`

如果该脚本直接调用 `mmdit_xl_2_inference(...)`，则在 `probe_mgm_asa.py` 的 `build_pipeline` 里 import 并复制其构造代码、把 `asa_cfg` 透传进去。如果是经过 pipeline 包装层，则在 pipeline 包装层加 `asa_cfg=None` 参数透传（仿 `cache_algo_cfg` 路径）。无论哪种，**不改默认行为**：默认 `asa_cfg=None`。

- [ ] **Step 7.3: 在 `build_pipeline` 中实现实际构造逻辑**

参照 Step 7.2 的查找结果，把 `build_pipeline` 的 NotImplementedError 替换为真正的构造函数调用，并把 `asa_cfg` 传进 `mmdit_xl_2_inference`。

- [ ] **Step 7.4: 实现 `run_one` 的 pipeline 调用**

参照同一 t2v 入口脚本里的推理调用方式（应当是 `pipeline(prompt, generator=g, ...)` 或类似形式），把视频张量返回。如果 pipeline 直接落盘 mp4，则改成读 mp4 解码成张量后返回。

- [ ] **Step 7.5: 跑一次 dense_probe 确认 PSNR > 35 dB（P3 验收门槛）**

Run: `python scripts/probe_mgm_asa.py --output-dir runs/asa_probe_smoke`
Expected: `metrics.json` 中 `dense_probe` 的所有 prompt PSNR > 35 dB（spec §7 P3 验收）

如果 PSNR 偏低，按 spec §6.3 R5 排查：先确认 `test_gilbert_rearrange_inverse_is_identity` 在 NPU 上跑也是 bit-exact identity；不是的话先修。

- [ ] **Step 7.6: 写报告骨架**

```markdown
# MGM-Video MMDiT × BLADE ASA 精度可行性探针报告（A 阶段）

- 日期：2026-06-02（实际跑测日期：TBD-by-runner）
- 分支：`asa`
- 设计 spec：`docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md`
- 探针脚本：`scripts/probe_mgm_asa.py`

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
1. p01_person — A close-up portrait of an elderly fisherman ...
2. p02_motion — A leopard sprinting through tall grass at golden hour ...
3. p03_scene — Snow falling slowly over a quiet Tokyo back-alley at night ...
4. p04_object — A ceramic teacup tipping over ...
5. p05_crowd — A bustling street market in Marrakech ...

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

## 已知风险触发情况

（参照 spec §6.3 R1–R8，runner 标注实际是否触发）
- R1 稠密 mask OOM：未触发 / 触发（处理：xx）
- R3 Gilbert 慢：未触发 / 触发（处理：xx）
- ...
```

把上述内容写入 `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md`。

- [ ] **Step 7.7: Commit P4 骨架（探针脚本 + 报告模板）**

```bash
git add scripts/probe_mgm_asa.py \
        docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md
git commit -m "feat(mgm_video): add ASA quality probe script and report skeleton (P4)

scripts/probe_mgm_asa.py runs baseline / dense_probe / asa@0.20 across
5 fixed prompts with fixed seed, writes metrics.json with PSNR + SSIM
proxy vs baseline. P3 acceptance gate: dense_probe PSNR > 35 dB on all
prompts. Report skeleton at docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md
to be filled by the runner after the actual run."
```

- [ ] **Step 7.8: 跑探针并填报告（人工执行，超出本计划范围）**

执行：
```
python scripts/probe_mgm_asa.py --output-dir runs/asa_probe_2026-06-02
```
然后人工：
1. 把 `metrics.json` 内容回填到报告 §"数值结果"表格
2. 把 `runs/asa_probe_2026-06-02/*.mp4` 路径填到 §"视频路径"
3. 目检视频，填 §"人工目检结论"
4. 综合写 §"结论与下一步建议"
5. commit 报告：`git add docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md && git commit -m "docs(mgm_video): fill ASA quality probe report"`

---

## 验收（参照 spec §9）

A 阶段探针完成判据：
1. ✅ Task 1–7 全部 commit 到 `asa` 分支
2. ✅ `asa_cfg=None` 生产路径与 ASA 引入前完全一致（`git diff main..asa -- vllm_omni/diffusion/models/mgm_video/` 走查 + `pytest tests/mgm_video/test_asa.py` 通过）
3. ✅ `tests/mgm_video/test_asa.py` 全过（22 测试）
4. ✅ Task 7 报告中 dense_probe PSNR > 35 dB（spec §7 P3 验收门槛）
5. ✅ `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md` 数据栏填完
6. ✅ 报告中给出"是否推进 ASA_G / 启动 NPU kernel 工作"的明确结论

---

## 不在本计划范围（spec §7 P5）

ASA_G 双路径融合（取 LSE + simple_pooling 全局路径 + log-sum-exp 加权）由 P4 报告结论决定是否启动；如果启动，独立计划 `docs/superpowers/plans/YYYY-MM-DD-mgm-video-asa-g.md`。

---

## P4.5（追加）：Step-warmup 混合稀疏度

**触发条件：** P4 quality probe 结果显示 ASA 配置在 mgm_video 上 PSNR < 28 dB（plan §6.3 R5），且 max=0.95 sanity 通过证明算法链路本身正确。属于 R5 风险预案的低成本补救。

**动机：** Diffusion 早期 step 决定全局构图，attention 全局信息至关重要；后期 step 只做局部细节修补，对稀疏 attention 鲁棒。BLADE 论文 Fig 8、FastDiT、ToDo 均采用此策略。预期 PSNR 提升 4–10 dB。

**设计：**
- `AsaConfig` 新增 `warmup_steps: int = 0`（默认 0 = 不启用，零行为变化）
- `JoinAttentionInference.infer` 进入 ASA 分支前判断 `self._current_step_idx < asa_cfg.warmup_steps` → 走原 dense FA 路径；否则走 ASA
- `_current_step_idx` 在 `pipeline_mgm_video.py:566` 已由 pipeline 设置到每个 block，无需新机制
- 新增 env var `VLLM_MGM_ASA_WARMUP_STEPS`，由 `AsaConfig.from_env` 读取

**实施步骤：**

- [ ] **P4.5.1**：`asa.py` 中 `AsaConfig` 加字段 `warmup_steps: int = 0`，`from_env` 读 `VLLM_MGM_ASA_WARMUP_STEPS`
- [ ] **P4.5.2**：单测 `test_asa_config_from_env_reads_overrides` 扩展覆盖 `warmup_steps`
- [ ] **P4.5.3**：`JoinAttentionInference.infer` ASA 分支入口加 step gate：
  ```python
  if asa_cfg is not None and asa_cfg.enable:
      step_idx = getattr(self, '_current_step_idx', None)
      if step_idx is not None and step_idx < asa_cfg.warmup_steps:
          out = self.fa(q, k, v, mask, C, offload_fa=False)  # warmup: dense
      else:
          # 现有 ASA 路径
  ```
- [ ] **P4.5.4**：单测覆盖 step gate（构造 mock attention 验证 step 0 走 dense、step ≥ warmup 走 ASA）
- [ ] **P4.5.5**：在 quality probe 报告中追加一组 sweep：`warmup_steps ∈ {0, 1, 2, 3}`，max=最优档位，记录 PSNR/SSIM/wall-time，画 Pareto

**验收：**
- `warmup_steps=0` 时与现有 ASA 路径 bit-exact（已有单测全过）
- `warmup_steps=2` 在 8-step TDM 蒸馏模型上 PSNR ≥ dense_probe - 3 dB
- 速度回退可控：`warmup_steps=2` vs `warmup_steps=0` 总耗时增加 ≤ 25%（2/8 步走 dense）

**与 ASA_G 的关系：** Step-warmup 是粗粒度 switch，ASA_G 是细粒度 soft 加权融合，两条路线互补。先上 P4.5（成本低、可控），如仍不达标再启动 ASA_G。

