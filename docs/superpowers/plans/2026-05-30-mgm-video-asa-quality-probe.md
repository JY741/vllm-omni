# MGM-Video ASA 质量探针 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MGM-Video MMDiT denoise 的 video↔video 注意力上，用纯 torch 生成 BLADE 风格二值块稀疏掩码喂给现有 attention，肉眼验证块稀疏是否会破坏这个已蒸馏 8-step 模型的视频质量（不追求加速）。

**Architecture:** 新增独立模块 `mmdit_asa.py`（块重要度→二值块掩码，纯 torch，CPU 可单测）；在 `JoinAttentionInference.infer()` 的 `fa()` 调用前注入掩码构建（开关保护、与既有 mask 合并）；在 `MMDiTBlockInference.forward` 透传 step 索引。全部经 `VLLM_MGM_ASA_*` 环境变量控制，关闭时零代码路径触碰。

**Tech Stack:** Python 3.10, PyTorch / torch_npu, pytest（CPU 小 shape 单测）, 现有 `npu_fusion_attention` / SDPABackend。

**关联 spec:** `docs/superpowers/specs/2026-05-30-mgm-video-asa-quality-probe-design.md`

---

## 关键实现事实（必读，影响每个任务）

1. **注入点是 `infer()` 而非 `fa()`。** `mmdit_blocks_inference.py` 的 `infer()` 里，CP all-to-all 之后有
   `for idx in range(un):` 循环，**每次迭代处理一个 head、完整序列**，循环内
   `q = torch.cat([x_q_chunk, y_q_chunk], dim=2)` 形状 `[bs, 1, T+L, hs]`（bs=1 video，N=1）。
   `out = self.fa(q, k, v, mask, C, offload_fa=False)` 在该文件**第 299 行**。掩码在此行前构建。
   因为 N=1，掩码天然是"每 head 各自一份"，spec 里的 3.3GB 全头共享问题在 CP 结构下自动化解
   （峰值是单 head 的 `[1,1,S,S]`，逐 idx 串行、用完即释放）。
2. **掩码约定 True = 屏蔽（masked）。** 三条 fa 路径（`use_vllm_attn` 默认 / `flash` / `npu_fusion`）
   入口拿到的 `mask` 都是 True=屏蔽：vllm/flash 路径会 `mask.logical_not()` 转成 SDPA 的 True=keep，
   npu_fusion 直接当 atten_mask（True=不算）。所以 ASA 产出的 `atten_mask` 必须 **True=屏蔽**。
3. **与 TDM-cache 求交是自动的。** cache state 3/4 的层/步走 `block.forward(skip_compute=True)`，
   在调用 `infer()` 之前就 `return x, y`（`mmdit_inference.py:112-113`），根本不进 `infer()`。
   因此 ASA 永远不会在 cache-skip 的层/步生效，**无需在 ASA 内读 cache 表**。
4. **step 索引来源。** `block.cur_time_index` 在 cache 路径（`mmdit_inference.py:220`）和非 cache 路径
   （`mmdit.py:612`）都会被设置。需在 `MMDiTBlockInference.forward` 把它透传到 attention：
   `self.attention.cur_time_index = getattr(self, 'cur_time_index', None)`。
5. **layer 索引来源。** `self._debug_block_idx`（`MMDiTBlockInference.forward` 已设到 attention 上，
   `mmdit_inference.py:129`）。
6. **skiparse 互斥。** 当前 11B preset `downscale` 全 1。`should_apply_asa` 在 `downscale != 1` 时返回 False。
7. **scale 值。** fa 内用 `(C // self.n_head) ** -0.5` == `head_dim ** -0.5`。ASA 重要度用同一 scale。

---

## File Structure

- **Create** `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py`
  —— `AsaConfig`（env 解析）、`_parse_range`、`should_apply_asa`、`_block_importance_keep`、`build_asa_block_mask`。纯 torch，无 NPU 依赖，CPU 可跑。
- **Modify** `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py`
  —— `MMDiTBlockInference.forward` 透传 `cur_time_index` 到 attention（1 行）。
- **Modify** `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py`
  —— `infer()` 在 fa 调用前注入 ASA 掩码构建与合并（~12 行，开关保护）。
- **Create** `tests/mgm_video/test_mmdit_asa.py`
  —— CPU pytest：结构性质 + 向量化/朴素循环参考实现对齐 + full-P 近似 sanity。
- **Create** `examples/offline_inference/text_to_video/README_asa.md`
  —— 环境变量与 sweep 命令用法。

---

## Task 1: ASA 配置与范围解析

**Files:**
- Create: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py`
- Test: `tests/mgm_video/test_mmdit_asa.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/mgm_video/test_mmdit_asa.py`：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for BLADE ASA quality-probe block-sparse mask.

Run: pytest tests/mgm_video/test_mmdit_asa.py -v
"""

import torch

from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import (
    AsaConfig,
    _parse_range,
    should_apply_asa,
)


def test_parse_range_all():
    assert _parse_range("all") is None


def test_parse_range_span():
    assert _parse_range("8-11") == {8, 9, 10, 11}


def test_parse_range_single():
    assert _parse_range("5") == {5}


def test_should_apply_respects_enable():
    cfg = AsaConfig(enable=False, tau=0.95, block=128, layers="all",
                    steps="all", per_head=True, log=False)
    assert should_apply_asa(cfg, layer_idx=0, step_idx=0, downscale=1) is False


def test_should_apply_skiparse_mutex():
    cfg = AsaConfig(enable=True, tau=0.95, block=128, layers="all",
                    steps="all", per_head=True, log=False)
    assert should_apply_asa(cfg, layer_idx=0, step_idx=0, downscale=2) is False


def test_should_apply_layer_step_scope():
    cfg = AsaConfig(enable=True, tau=0.95, block=128, layers="8-41",
                    steps="2-7", per_head=True, log=False)
    assert should_apply_asa(cfg, layer_idx=0, step_idx=5, downscale=1) is False
    assert should_apply_asa(cfg, layer_idx=10, step_idx=0, downscale=1) is False
    assert should_apply_asa(cfg, layer_idx=10, step_idx=5, downscale=1) is True


def test_should_apply_step_none_passes():
    cfg = AsaConfig(enable=True, tau=0.95, block=128, layers="all",
                    steps="2-7", per_head=True, log=False)
    # step unknown (cache disabled) -> step scope not enforced
    assert should_apply_asa(cfg, layer_idx=0, step_idx=None, downscale=1) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v`
Expected: FAIL with `ModuleNotFoundError: ...mmdit_asa` / `ImportError`.

- [ ] **Step 3: Write minimal implementation**

创建 `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py`：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BLADE Adaptive Block-Sparse Attention (ASA) — inference-time quality probe.

Pure-torch reference implementation that builds a binary block-sparse
attention mask over the video<->video sub-block of MGM-Video MMDiT attention.

This is a QUALITY PROBE: it does NOT skip blocks and yields NO wall-clock
speedup (dense FA + mask recomputes masked positions). Its sole purpose is to
let us eyeball whether block sparsity degrades the already-distilled 8-step
model's video quality before investing in a fused block-sparse kernel.

Faithful-but-simplified vs the paper:
- raster block partition (NO Gilbert reorder) -> this is a quality LOWER bound.
- exact mean-pooled block importance (NO k=16 sampling approximation).
"""

import os


def _parse_range(spec):
    """Parse 'all' or 'lo-hi' (inclusive, 0-based) or 'n' into a set of ints.

    Returns None for 'all' (means "every index").
    """
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return set(range(int(lo), int(hi) + 1))
    return {int(spec)}


class AsaConfig:
    """ASA probe configuration, sourced from VLLM_MGM_ASA_* env vars."""

    def __init__(self, enable, tau, block, layers, steps, per_head, log):
        self.enable = enable
        self.tau = tau
        self.block = block
        self.layers = layers  # raw range string, parsed lazily
        self.steps = steps    # raw range string, parsed lazily
        self.per_head = per_head
        self.log = log

    @classmethod
    def from_env(cls):
        return cls(
            enable=os.environ.get("VLLM_MGM_ASA_ENABLE", "0") == "1",
            tau=float(os.environ.get("VLLM_MGM_ASA_TAU", "0.95")),
            block=int(os.environ.get("VLLM_MGM_ASA_BLOCK", "128")),
            layers=os.environ.get("VLLM_MGM_ASA_LAYERS", "all"),
            steps=os.environ.get("VLLM_MGM_ASA_STEPS", "all"),
            per_head=os.environ.get("VLLM_MGM_ASA_PER_HEAD", "1") == "1",
            log=os.environ.get("VLLM_MGM_ASA_LOG", "1") == "1",
        )


def should_apply_asa(cfg, layer_idx, step_idx, downscale):
    """Decide whether ASA masking applies at this (layer, step).

    Note: TDM-cache skip is handled UPSTREAM (skip_compute short-circuits
    before infer() is called), so no cache-table lookup is needed here.
    """
    if not cfg.enable:
        return False
    if downscale != 1:
        # skiparse static sparsity is mutually exclusive with ASA
        return False
    layers = _parse_range(cfg.layers)
    if layers is not None and layer_idx not in layers:
        return False
    if step_idx is not None:
        steps = _parse_range(cfg.steps)
        if steps is not None and step_idx not in steps:
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py tests/mgm_video/test_mmdit_asa.py
git commit -m "feat(mgm_video): ASA probe config + scope resolution"
```

---

## Task 2: 块重要度核心 `_block_importance_keep`

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py`
- Test: `tests/mgm_video/test_mmdit_asa.py`

- [ ] **Step 1: Write the failing test**

在 `tests/mgm_video/test_mmdit_asa.py` 末尾追加：

```python
from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import (
    _block_importance_keep,
)


def _reference_block_keep(q_v, k_v, block, tau, scale):
    """Naive loop reference of the SAME mean-pool block-importance algorithm."""
    Sv, D = q_v.shape
    nB = (Sv + block - 1) // block
    pad = nB * block - Sv
    if pad:
        q_v = torch.cat([q_v, torch.zeros(pad, D, dtype=q_v.dtype)], 0)
        k_v = torch.cat([k_v, torch.zeros(pad, D, dtype=k_v.dtype)], 0)
    k_blk = torch.stack([k_v[i * block:(i + 1) * block].mean(0) for i in range(nB)])
    keep = torch.zeros(nB, nB, dtype=torch.bool)
    for i in range(nB):
        qi = q_v[i * block:(i + 1) * block]
        s_blk = ((qi @ k_blk.t()) * scale).amax(0)
        p = torch.softmax(s_blk.float(), -1)
        order = torch.argsort(p, descending=True)
        acc = 0.0
        for j in order.tolist():
            keep[i, j] = True
            acc += p[j].item()
            if acc >= tau:
                break
        keep[i, i] = True
    return keep


def test_block_importance_diagonal_self_select():
    torch.manual_seed(0)
    q = torch.randn(256, 16)
    k = torch.randn(256, 16)
    keep, nB, pad = _block_importance_keep(q, k, block=64, tau=0.5, scale=16 ** -0.5)
    assert nB == 4 and pad == 0
    assert bool(keep.diagonal().all())


def test_block_importance_tau_monotonic():
    torch.manual_seed(1)
    q = torch.randn(256, 16)
    k = torch.randn(256, 16)
    keep_low, _, _ = _block_importance_keep(q, k, block=64, tau=0.50, scale=16 ** -0.5)
    keep_high, _, _ = _block_importance_keep(q, k, block=64, tau=0.99, scale=16 ** -0.5)
    # higher tau keeps >= blocks
    assert int(keep_high.sum()) >= int(keep_low.sum())


def test_block_importance_matches_reference():
    torch.manual_seed(2)
    q = torch.randn(320, 16)  # non-divisible -> exercises padding (nB=5, pad=0 for 320/64)
    k = torch.randn(320, 16)
    keep, _, _ = _block_importance_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    ref = _reference_block_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    assert torch.equal(keep, ref)


def test_block_importance_padding():
    torch.manual_seed(3)
    q = torch.randn(200, 16)  # 200/64 -> nB=4, pad=56
    k = torch.randn(200, 16)
    keep, nB, pad = _block_importance_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    assert nB == 4 and pad == 56
    ref = _reference_block_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    assert torch.equal(keep, ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v -k block_importance`
Expected: FAIL with `ImportError: cannot import name '_block_importance_keep'`.

- [ ] **Step 3: Write minimal implementation**

在 `mmdit_asa.py` 顶部加 `import torch` 和 `import torch.nn.functional as F`，并追加函数：

```python
import torch
import torch.nn.functional as F


def _block_importance_keep(q_v, k_v, block, tau, scale):
    """Exact mean-pooled block importance -> binary block keep-mask.

    Args:
        q_v, k_v: [Sv, D] single-head video query/key.
        block: block size b.
        tau: cumulative threshold (nucleus selection).
        scale: softmax scale (head_dim ** -0.5).

    Returns:
        keep: [nB, nB] bool, True = KEEP (block participates).
        nB: number of blocks.
        pad: zero-padding added to reach nB*block.
    """
    Sv, D = q_v.shape
    nB = (Sv + block - 1) // block
    pad = nB * block - Sv
    if pad:
        q_v = F.pad(q_v, (0, 0, 0, pad))
        k_v = F.pad(k_v, (0, 0, 0, pad))

    # 1. mean-pool each KV block (query-independent)
    k_blk = k_v.reshape(nB, block, D).mean(dim=1)            # [nB, D]
    # 2. token-row x block-col approximate scores
    s_imp = (q_v @ k_blk.transpose(0, 1)) * scale            # [nB*block, nB]
    # 3. max-pool over query block rows (shared mask per query block)
    s_blk = s_imp.reshape(nB, block, nB).amax(dim=1)         # [nB, nB]
    # 4. row softmax + nucleus (cumulative >= tau) selection
    p = torch.softmax(s_blk.float(), dim=-1)                 # [nB, nB]
    sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    # keep a block if cumulative-before-it < tau (so the block crossing tau is kept)
    keep_sorted = (cum - sorted_p) < tau
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    # 5. diagonal self-select (locality safety net)
    diag = torch.arange(nB, device=q_v.device)
    keep[diag, diag] = True
    return keep, nB, pad
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v -k block_importance`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py tests/mgm_video/test_mmdit_asa.py
git commit -m "feat(mgm_video): exact mean-pooled block importance for ASA"
```

---

## Task 3: token 掩码组装 `build_asa_block_mask`

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py`
- Test: `tests/mgm_video/test_mmdit_asa.py`

- [ ] **Step 1: Write the failing test**

在 `tests/mgm_video/test_mmdit_asa.py` 末尾追加：

```python
from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import (
    build_asa_block_mask,
)


def test_build_mask_shape_and_dtype():
    torch.manual_seed(4)
    T, L, D = 256, 32, 16
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m = build_asa_block_mask(q, k, T, L, block=64, tau=0.9, scale=D ** -0.5)
    assert m.shape == (1, 1, S, S)
    assert m.dtype == torch.bool


def test_build_mask_text_is_dense():
    torch.manual_seed(5)
    T, L, D = 256, 32, 16
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m = build_asa_block_mask(q, k, T, L, block=64, tau=0.9, scale=D ** -0.5)
    # True = masked. Text rows and text cols must be fully dense (never masked).
    assert not bool(m[0, 0, T:, :].any())   # text query rows
    assert not bool(m[0, 0, :, T:].any())   # text key cols


def test_build_mask_diagonal_video_unmasked():
    torch.manual_seed(6)
    T, L, D, block = 256, 32, 16, 64
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m = build_asa_block_mask(q, k, T, L, block=block, tau=0.5, scale=D ** -0.5)
    # diagonal video blocks self-select -> their on-diagonal token region unmasked
    for i in range(T // block):
        s = i * block
        assert not bool(m[0, 0, s:s + block, s:s + block].any())


def test_build_mask_higher_tau_less_masked():
    torch.manual_seed(7)
    T, L, D = 256, 32, 16
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m_low = build_asa_block_mask(q, k, T, L, block=64, tau=0.50, scale=D ** -0.5)
    m_high = build_asa_block_mask(q, k, T, L, block=64, tau=0.99, scale=D ** -0.5)
    assert int(m_high.sum()) <= int(m_low.sum())


def test_build_mask_fullp_sanity():
    """Soft check: ASA-kept blocks should cover most of the true full-P mass."""
    torch.manual_seed(8)
    T, L, D, block = 256, 0, 16, 64
    q = torch.randn(1, 1, T, D)
    k = torch.randn(1, 1, T, D)
    scale = D ** -0.5
    m = build_asa_block_mask(q, k, T, L, block=block, tau=0.9, scale=scale)
    keep = ~m[0, 0, :T, :T]
    # true post-softmax attention probabilities
    p_full = torch.softmax((q[0, 0] @ k[0, 0].t()) * scale, dim=-1)
    covered = (p_full * keep.float()).sum(-1)  # per query row
    # kept blocks should cover a large majority of probability mass on average
    assert covered.mean().item() > 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v -k build_mask`
Expected: FAIL with `ImportError: cannot import name 'build_asa_block_mask'`.

- [ ] **Step 3: Write minimal implementation**

在 `mmdit_asa.py` 追加：

```python
def build_asa_block_mask(q, k, T, L, block, tau, scale, log=False,
                         layer=-1, step=None):
    """Build a video<->video block-sparse attention mask.

    Args:
        q, k: [bs, 1, S, D] single-head (CP processes one head per fa call),
              S = T + L, video tokens [0:T], text tokens [T:T+L].
        T, L: video / text sequence lengths.
        block, tau, scale: ASA hyper-parameters.
        log: print per-call sparsity stat.
        layer, step: for logging only.

    Returns:
        atten_mask: [bs, 1, S, S] bool, True = MASKED (npu_fusion convention).
                    text rows and text cols are always dense (never masked).
    """
    bs, n, S, D = q.shape
    assert n == 1, "ASA probe expects single-head fa calls (CP head-major loop)"
    device = q.device

    q_v = q[0, 0, :T]                                        # [T, D]
    k_v = k[0, 0, :T]
    keep_blk, nB, pad = _block_importance_keep(q_v, k_v, block, tau, scale)

    # atten_mask: start all-keep (False), then mask the dropped video token blocks.
    atten = torch.zeros(S, S, dtype=torch.bool, device=device)
    mask_blk = ~keep_blk                                     # [nB, nB] True = masked
    mask_tok = mask_blk.repeat_interleave(block, 0).repeat_interleave(block, 1)
    atten[:T, :T] = mask_tok[:T, :T]
    # text rows/cols stay False (dense) by construction.

    if log:
        sparsity = mask_blk.float().mean().item()
        print(f"[ASA][raster, no-gilbert] layer={layer} step={step} "
              f"nB={nB} tau={tau} block={block} block_sparsity={sparsity:.1%}")

    return atten.unsqueeze(0).unsqueeze(0).expand(bs, 1, S, S)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v`
Expected: PASS (all tests, ~16 passed).

- [ ] **Step 5: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_asa.py tests/mgm_video/test_mmdit_asa.py
git commit -m "feat(mgm_video): assemble video<->video ASA token mask"
```

---

## Task 4: 透传 step 索引到 attention

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py:129`

- [ ] **Step 1: 定位现有代码**

`MMDiTBlockInference.forward` 现有（`mmdit_inference.py` 约 128-131 行）：

```python
        # Propagate block_idx to JoinAttentionInference for internal debug
        self.attention._debug_block_idx = block_idx

        x1, y1 = self.attention.infer(
```

- [ ] **Step 2: 修改 — 增加 step 透传**

把上面那段改为：

```python
        # Propagate block_idx to JoinAttentionInference for internal debug
        self.attention._debug_block_idx = block_idx
        # Propagate step index (set by blocks_forward / cache path) for ASA scope.
        self.attention.cur_time_index = getattr(self, 'cur_time_index', None)

        x1, y1 = self.attention.infer(
```

- [ ] **Step 3: 校验导入与语法**

Run: `python -c "import ast; ast.parse(open('vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_inference.py
git commit -m "feat(mgm_video): wire step index to attention for ASA scope"
```

---

## Task 5: 在 `infer()` 注入 ASA 掩码

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:299`

- [ ] **Step 1: 定位现有代码**

`infer()` 现有（`mmdit_blocks_inference.py` 约 295-299 行）：

```python
            else:
                # mask is already processed in MMDiT.forward of mimogpt/models/dit/mmdit.py
                pass

            out = self.fa(q, k, v, mask, C, offload_fa=False)
```

- [ ] **Step 2: 修改 — fa 调用前注入 ASA**

替换为：

```python
            else:
                # mask is already processed in MMDiT.forward of mimogpt/models/dit/mmdit.py
                pass

            # --- BLADE ASA quality probe: optional video<->video block-sparse mask ---
            # Builds a True=masked attention mask and ORs it into the existing mask.
            # No speedup (dense FA still computes masked positions); quality probe only.
            from .mmdit_asa import AsaConfig, build_asa_block_mask, should_apply_asa
            _asa_cfg = AsaConfig.from_env()
            _asa_layer = getattr(self, '_debug_block_idx', -1)
            _asa_step = getattr(self, 'cur_time_index', None)
            if (should_apply_asa(_asa_cfg, _asa_layer, _asa_step, self.downscale)
                    and not isinstance(mask, list)):
                _asa_mask = build_asa_block_mask(
                    q, k, T, L, _asa_cfg.block, _asa_cfg.tau,
                    (C // self.n_head) ** -0.5,
                    log=_asa_cfg.log, layer=_asa_layer, step=_asa_step,
                )
                mask = _asa_mask if mask is None else (mask | _asa_mask)

            out = self.fa(q, k, v, mask, C, offload_fa=False)
```

- [ ] **Step 3: 校验语法**

Run: `python -c "import ast; ast.parse(open('vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: 校验开关关闭时零行为变更（CPU 冒烟）**

Run:
```bash
VLLM_MGM_ASA_ENABLE=0 python -c "
from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import AsaConfig, should_apply_asa
cfg = AsaConfig.from_env()
assert cfg.enable is False
assert should_apply_asa(cfg, 0, 0, 1) is False
print('disabled-path OK')
"
```
Expected: `disabled-path OK`

- [ ] **Step 5: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py
git commit -m "feat(mgm_video): inject ASA block-sparse mask in infer() (probe)"
```

---

## Task 6: 用法文档

**Files:**
- Create: `examples/offline_inference/text_to_video/README_asa.md`

- [ ] **Step 1: 创建用法文档**

```markdown
# MGM-Video ASA 质量探针用法

把 BLADE 风格的二值块稀疏注意力（ASA）套到 MMDiT denoise 的 video↔video 注意力上，
**仅用于肉眼验证视频质量是否受损**——不跳块、无墙钟加速。

详见设计文档：`docs/superpowers/specs/2026-05-30-mgm-video-asa-quality-probe-design.md`。

## 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_MGM_ASA_ENABLE` | `0` | 总开关。`1` 开启 |
| `VLLM_MGM_ASA_TAU` | `0.95` | 累积阈值（越大保留越多、越接近 dense） |
| `VLLM_MGM_ASA_BLOCK` | `128` | 块大小（建议 64 / 128） |
| `VLLM_MGM_ASA_LAYERS` | `all` | 生效层范围，如 `8-41` |
| `VLLM_MGM_ASA_STEPS` | `all` | 生效 step 范围，如 `2-7` |
| `VLLM_MGM_ASA_PER_HEAD` | `1` | 1=逐头掩码；0=保留位（CP 下每 fa 调用本就单头） |
| `VLLM_MGM_ASA_LOG` | `1` | 1=打印每次 layer/step/block_sparsity |

范围格式：`all` 或 `lo-hi`（闭区间，0-based）或单个索引 `n`。

## 跑法

```bash
# 1) dense 基线（同 seed）
VLLM_MGM_ASA_ENABLE=0 \
  python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model /path/to/mgm_video_11b_vllm \
  --prompt "A cat playing piano in a cozy room" \
  --seed 42 --output asa_dense.mp4

# 2) ASA 开启
VLLM_MGM_ASA_ENABLE=1 VLLM_MGM_ASA_TAU=0.95 VLLM_MGM_ASA_BLOCK=128 \
  python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model /path/to/mgm_video_11b_vllm \
  --prompt "A cat playing piano in a cozy room" \
  --seed 42 --output asa_tau095.mp4

# 3) sweep 示例：只对深层、后段 step 稀疏
VLLM_MGM_ASA_ENABLE=1 VLLM_MGM_ASA_LAYERS=8-41 VLLM_MGM_ASA_STEPS=2-7 \
  python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model /path/to/mgm_video_11b_vllm --seed 42 --output asa_deep.mp4
```

逐帧/并排比较 `asa_dense.mp4` 与 ASA 输出，结合日志里的 `block_sparsity` 判断
"质量损失 vs 稀疏率"。判定逻辑见设计文档 §5.1。

## 已知限制（诚实标注）

- **无加速**：dense FA + mask 不跳块，总时间只增不减。
- **无 Gilbert 重排**（raster 分块）：这是"质量下界探针"。raster 质量 OK → Gilbert 只会更好；
  raster 崩 → 需补 Gilbert 再判，不能直接下"ASA 不适配"。
- 3.3GB 量级的 `[S,S]` 掩码是 dense-mask 路线的固有显存代价（逐 head 串行、用完即释放）。
```

- [ ] **Step 2: Commit**

```bash
git add examples/offline_inference/text_to_video/README_asa.md
git commit -m "docs(mgm_video): ASA quality-probe usage guide"
```

---

## Task 7: 端到端人工验证（手动，非自动化）

**Files:** 无（运行验证）

> 此任务在有 NPU + 权重的环境手动执行，产出对比视频与 sweep 结论。无自动断言。

- [ ] **Step 1: 全套单测回归**

Run: `pytest tests/mgm_video/test_mmdit_asa.py -v`
Expected: all PASS。

- [ ] **Step 2: dense 基线**

Run:
```bash
VLLM_MGM_ASA_ENABLE=0 python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model <MODEL> --prompt "<PROMPT>" --seed 42 --output asa_dense.mp4
```
Expected: 正常出 `asa_dense.mp4`，日志无 `[ASA]` 行。

- [ ] **Step 3: ASA 开启（默认 τ=0.95）**

Run:
```bash
VLLM_MGM_ASA_ENABLE=1 python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model <MODEL> --prompt "<PROMPT>" --seed 42 --output asa_tau095.mp4
```
Expected: 日志出现 `[ASA][raster, no-gilbert] layer=.. step=.. block_sparsity=..%`；出 `asa_tau095.mp4`。

- [ ] **Step 4: 记录稀疏率与肉眼质量**

- 记录各层/步 `block_sparsity` 区间。
- 逐帧/并排比较 dense vs ASA。
- 按设计文档 §5.1 三分支判定（无损+高稀疏→值得做融合算子；崩→需重训；中间态→缩小作用域）。

- [ ] **Step 5: sweep（可选）**

按需扫 `VLLM_MGM_ASA_TAU ∈ {0.90,0.95,0.98}`、`VLLM_MGM_ASA_LAYERS`、`VLLM_MGM_ASA_STEPS`，
各出一条视频，形成"质量 vs 稀疏率"曲线，写入结论。

---

## Self-Review 记录

- **Spec 覆盖**：§3.1 注入点→Task5；§3.2 块重要度→Task2/3；§3.3 cache 求交→Task1(`should_apply`)+关键事实#3(自动)；
  §3.4 作用范围/skiparse 互斥→Task1/Task3(text dense)；§3.5 诚实标注→Task3 日志+Task6 文档；
  §3.6 CP→关键事实#1；§4 配置→Task1+Task6；§5 验证→Task7；§6 交付物→全任务覆盖。
- **占位符扫描**：无 TBD/TODO；每个 code step 含完整代码；测试均含真实断言。
- **类型/命名一致性**：`AsaConfig`/`_parse_range`/`should_apply_asa`/`_block_importance_keep`/
  `build_asa_block_mask` 跨任务签名一致；`keep`(True=保留) 与 `atten_mask`(True=屏蔽) 语义在
  Task2/3 明确区分（`build` 内 `~keep_blk`）。
- **与 spec 的一处澄清**：spec §6 "与朴素 full-P 参考实现数值对齐"。full-P pooling 与 mean-pool-K
  是两种不同重要度，不会逐值相等。本计划把**精确对齐目标**定为"向量化 vs 同算法朴素循环"
  （Task2 `test_block_importance_matches_reference`，exact equal），另对 full-P 用**软 sanity**
  （Task3 `test_build_mask_fullp_sanity`，覆盖率>0.7）。这是对 spec 措辞的合理修正，已在此记录。
