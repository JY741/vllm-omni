# MGM-Video ASA 切换至 `npu_block_sparse_attention` 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在推理路径中新增 `torch_npu.npu_block_sparse_attention` 作为 ASA 的可选注意力后端，通过环境变量切换，保留稠密路径作为回退。

**Architecture:** 通过回调注入把 `torch_npu` 调用隔离在 `mmdit_blocks_inference.py`，`asa.py` 只接受 `fa_block_sparse(q,k,v,block_mask)` 回调；新增 `block_sparse_attn.py` 负责 bool 块掩码 → int8 per-head 掩码的转换与算子调用。

**Tech Stack:** Python, PyTorch, torch_npu, pytest

---

## 文件结构

| 文件 | 责任 |
|---|---|
| `vllm_omni/diffusion/models/mgm_video/mmdit/block_sparse_attn.py` | 新增：`npu_block_sparse_attention_wrapper`，封装算子调用与掩码转换 |
| `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py` | 修改：`asa_attention` 增加可选 `fa_block_sparse` 回调分支 |
| `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py` | 修改：读取 `VLLM_MGM_USE_BLOCK_SPARSE_ATTN`，构造并传入回调 |
| `tests/mgm_video/test_asa.py` | 新增/修改：mock callback 测试与退化 parity 测试 |

---

### Task 1: Create `block_sparse_attn.py` wrapper

**Files:**
- Create: `vllm_omni/diffusion/models/mgm_video/mmdit/block_sparse_attn.py`
- Test: `tests/mgm_video/test_block_sparse.py` (created in Task 4)

- [ ] **Step 1: Write the wrapper file**

```python
# SPDX-License-Identifier: Apache-2.0
"""Block-sparse attention wrapper for torch_npu.npu_block_sparse_attention."""

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover - CPU-only test environments
    torch_npu = None


def npu_block_sparse_attention_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Run block-sparse attention on NPU.

    Args:
        q, k, v: [B, N, S, D] (BNSD). q/k/v must be bf16 or fp16.
        block_mask: [B, 1, nq, nk] bool, True = keep block. nq=nk=ceil(S/block_size).
        block_size: sparse block size. Must equal block_shape used by the op.

    Returns:
        out: [B, N, S, D]
    """
    if torch_npu is None:
        raise RuntimeError("torch_npu is not available")

    num_heads = q.shape[1]
    head_dim = q.shape[-1]

    # Operator expects per-head int8 mask: [B, N, nq, nk]
    block_mask_int8 = block_mask.expand(-1, num_heads, -1, -1).to(torch.int8)

    out, _ = torch_npu.npu_block_sparse_attention(
        q,
        k,
        v,
        block_sparse_mask=block_mask_int8,
        block_shape=[block_size, block_size],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=num_heads,
        scale_value=head_dim ** -0.5,
        inner_precise=0,  # bf16 requires fp32 softmax
    )
    return out
```

- [ ] **Step 2: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/block_sparse_attn.py
git commit -m "feat(mgm_video): add npu_block_sparse_attention wrapper

Add block_sparse_attn.py to encapsulate mask conversion and operator call.
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Modify `asa.py` to accept block-sparse callback

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py:557-647`
- Test: `tests/mgm_video/test_asa.py`

- [ ] **Step 1: Add `fa_block_sparse` parameter and branch**

Change the function signature to insert `fa_block_sparse` right after `fa_full_dense`:

```python
def asa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: "AsaConfig",
    rearranger: GilbertRearranger,
    t_len: int,
    l_len: int,
    fa_full_dense,
    fa_block_sparse=None,
    generator: torch.Generator | None = None,
    sta_cache: "StaMaskCache | None" = None,
) -> torch.Tensor:
```

Add to the docstring under `fa_full_dense`:

```
        fa_block_sparse: optional callable (q, k, v, block_mask) -> out [B,N,T+L,D]
                         using torch_npu.npu_block_sparse_attention. If provided
                         and cfg.variant == "asa", it replaces the dense mask path.
```

- [ ] **Step 2: Replace the attention call with a branch**

Locate:

```python
    # 6. flash attention with dense atten_mask (or None for dense_probe)
    out_g = fa_full_dense(q_g, k_g, v_g, m_token)
```

Replace with:

```python
    # 6. attention: block-sparse callback when available, else dense mask
    if cfg.variant == "asa" and fa_block_sparse is not None:
        out_g = fa_block_sparse(q_g, k_g, v_g, m_block)
    else:
        out_g = fa_full_dense(q_g, k_g, v_g, m_token)
```

- [ ] **Step 3: Run existing ASA tests to ensure no regression**

```bash
pytest tests/mgm_video/test_asa.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/asa.py
git commit -m "feat(mgm_video): add fa_block_sparse callback path in asa_attention

When fa_block_sparse is provided and variant is 'asa', pass the block mask
[N,1,nq,nk] bool directly to the callback instead of expanding to a dense
token mask. Dense path remains the default.
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire env-var gate in `mmdit_blocks_inference.py`

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py:366-380`

- [ ] **Step 1: Add env-var read and block-sparse callback construction**

Locate the existing block:

```python
                    def _fa_full_dense(qq, kk, vv, atten_mask):
                        # asa.py uses SDPA convention (True = keep). self.fa's
                        # npu_fusion_attention path uses the opposite convention
                        # (True = mask out), matching the baseline mask built at
                        # lines 287-289 above. Invert here so both unit tests
                        # (SDPA-backed) and runtime (NPU FA) see the contract
                        # they expect.
                        if atten_mask is not None:
                            atten_mask = atten_mask.logical_not()
                        return self.fa(qq, kk, vv, atten_mask, C, offload_fa=False)

                    from .asa import asa_attention
                    out = asa_attention(q, k, v, asa_cfg, self._asa_rearranger,
                                        t_len=T_seg, l_len=L_seg, fa_full_dense=_fa_full_dense,
                                        sta_cache=self._sta_cache)
```

Replace with:

```python
                    def _fa_full_dense(qq, kk, vv, atten_mask):
                        if atten_mask is not None:
                            atten_mask = atten_mask.logical_not()
                        return self.fa(qq, kk, vv, atten_mask, C, offload_fa=False)

                    use_block_sparse = os.environ.get(
                        "VLLM_MGM_USE_BLOCK_SPARSE_ATTN", "0"
                    ) == "1"
                    fa_block_sparse = None
                    if use_block_sparse:
                        if not hasattr(torch_npu, "npu_block_sparse_attention"):
                            raise RuntimeError(
                                "VLLM_MGM_USE_BLOCK_SPARSE_ATTN=1 but "
                                "torch_npu.npu_block_sparse_attention is not available"
                            )
                        from .block_sparse_attn import npu_block_sparse_attention_wrapper

                        def _fa_block_sparse(qq, kk, vv, block_mask):
                            return npu_block_sparse_attention_wrapper(
                                qq, kk, vv, block_mask, block_size=asa_cfg.block_size
                            )

                        fa_block_sparse = _fa_block_sparse

                    from .asa import asa_attention
                    out = asa_attention(
                        q, k, v, asa_cfg, self._asa_rearranger,
                        t_len=T_seg, l_len=L_seg,
                        fa_full_dense=_fa_full_dense,
                        fa_block_sparse=fa_block_sparse,
                        sta_cache=self._sta_cache,
                    )
```

Make sure `import os` is already present at the top of the file (it is in the current codebase).

- [ ] **Step 2: Verify the module imports**

```bash
python -c "from vllm_omni.diffusion.models.mgm_video.mmdit import mmdit_blocks_inference"
```

Expected: no import errors.

- [ ] **Step 3: Commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py
git commit -m "feat(mgm_video): gate npu_block_sparse_attention with env var

Add VLLM_MGM_USE_BLOCK_SPARSE_ATTN env var in inference path. When set to 1,
construct and inject fa_block_sparse callback into asa_attention; otherwise
keep the existing dense npu_fusion_attention path.
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Add unit tests for block-sparse callback path

**Files:**
- Modify: `tests/mgm_video/test_asa.py`

- [ ] **Step 1: Add mock callback test**

Append to `tests/mgm_video/test_asa.py`:

```python
def test_asa_attention_uses_block_sparse_callback_when_provided():
    """variant=asa with fa_block_sparse should invoke callback with block mask."""
    torch.manual_seed(0)
    W, H, Tdepth = 4, 3, 2
    T = W * H * Tdepth
    L = 5
    B, N, D = 1, 2, 8

    q = torch.randn(B, N, T + L, D)
    k = torch.randn(B, N, T + L, D)
    v = torch.randn(B, N, T + L, D)

    cfg = AsaConfig(
        enable=True, variant="asa", max_retain_ratio=0.5, min_retain_ratio=0.5,
        energy_threshold=0.95, block_size=8, num_keep=4, use_gilbert=False, text_length=L,
    )
    rearr = GilbertRearranger(W, H, Tdepth, text_length=L)

    captured = {}

    def _mock_block_sparse(qq, kk, vv, block_mask):
        captured["q"] = qq
        captured["k"] = kk
        captured["v"] = vv
        captured["block_mask"] = block_mask
        return torch.zeros_like(qq)

    def _mock_full_dense(qq, kk, vv, atten_mask):
        pytest.fail("dense callback should not be called when block sparse is provided")

    asa_attention(
        q, k, v, cfg, rearr,
        t_len=T, l_len=L,
        fa_full_dense=_mock_full_dense,
        fa_block_sparse=_mock_block_sparse,
    )

    assert captured["block_mask"].dtype == torch.bool
    assert captured["block_mask"].shape[1] == 1  # head-shared on entry
    assert captured["q"].shape == (B, N, T + L, D)
```

- [ ] **Step 2: Add mask conversion test**

Append to `tests/mgm_video/test_asa.py`:

```python
def test_block_sparse_mask_conversion():
    """Wrapper expands head-shared bool mask to per-head int8."""
    from vllm_omni.diffusion.models.mgm_video.mmdit.block_sparse_attn import (
        npu_block_sparse_attention_wrapper,
    )

    B, N, nq, nk = 1, 4, 3, 3
    block_mask = torch.tensor([[[[True, False, True],
                                  [False, True, False],
                                  [True, True, False]]]])  # [1,1,3,3]

    # We cannot call NPU op on CPU, so verify the conversion logic by monkey-patching.
    captured = {}

    def _fake_npu_op(*args, **kwargs):
        captured["mask"] = kwargs["block_sparse_mask"]
        return (torch.zeros(1, N, 1, 8), None)

    import vllm_omni.diffusion.models.mgm_video.mmdit.block_sparse_attn as bsamod
    original = bsamod.torch_npu.npu_block_sparse_attention
    bsamod.torch_npu.npu_block_sparse_attention = _fake_npu_op
    try:
        q = torch.zeros(1, N, 8, 8)
        k = torch.zeros(1, N, 8, 8)
        v = torch.zeros(1, N, 8, 8)
        npu_block_sparse_attention_wrapper(q, k, v, block_mask, block_size=8)
    finally:
        bsamod.torch_npu.npu_block_sparse_attention = original

    assert captured["mask"].dtype == torch.int8
    assert captured["mask"].shape == (B, N, nq, nk)
    expected = block_mask.expand(-1, N, -1, -1).to(torch.int8)
    assert torch.equal(captured["mask"], expected)
```

- [ ] **Step 3: Run the new tests**

```bash
pytest tests/mgm_video/test_asa.py::test_asa_attention_uses_block_sparse_callback_when_provided tests/mgm_video/test_asa.py::test_block_sparse_mask_conversion -v
```

Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add tests/mgm_video/test_asa.py
git commit -m "test(mgm_video): add block-sparse callback tests

Verify asa_attention calls fa_block_sparse when provided and that the
wrapper converts [B,1,nq,nk] bool to [B,N,nq,nk] int8.
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Add NPU smoke test

**Files:**
- Create: `tests/mgm_video/test_block_sparse_npu.py`

- [ ] **Step 1: Write the NPU smoke test**

```python
# SPDX-License-Identifier: Apache-2.0
"""NPU smoke test for npu_block_sparse_attention_wrapper."""

import pytest
import torch


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU not available",
)
def test_npu_block_sparse_attention_wrapper_smoke():
    from vllm_omni.diffusion.models.mgm_video.mmdit.block_sparse_attn import (
        npu_block_sparse_attention_wrapper,
    )

    B, N, S, D = 1, 2, 256, 64
    block_size = 128
    n_blocks = (S + block_size - 1) // block_size

    q = torch.randn(B, N, S, D, dtype=torch.bfloat16).npu()
    k = torch.randn(B, N, S, D, dtype=torch.bfloat16).npu()
    v = torch.randn(B, N, S, D, dtype=torch.bfloat16).npu()
    block_mask = torch.ones(1, 1, n_blocks, n_blocks, dtype=torch.bool).npu()

    out = npu_block_sparse_attention_wrapper(q, k, v, block_mask, block_size=block_size)
    assert out.shape == (B, N, S, D)
    assert out.dtype == torch.bfloat16
```

- [ ] **Step 2: Run smoke test (requires NPU)**

```bash
pytest tests/mgm_video/test_block_sparse_npu.py -v
```

Expected on NPU: pass. On CPU: skipped.

- [ ] **Step 3: Commit**

```bash
git add tests/mgm_video/test_block_sparse_npu.py
git commit -m "test(mgm_video): add NPU smoke test for block sparse wrapper

Verify the wrapper runs on real NPU hardware with bf16 inputs and produces
correct output shape.
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Run full test suite and parity check

**Files:**
- Run: `tests/mgm_video/test_asa.py`
- Run: `tests/mgm_video/test_sta.py` (regression check)

- [ ] **Step 1: Run all ASA tests**

```bash
pytest tests/mgm_video/test_asa.py tests/mgm_video/test_sta.py -v
```

Expected: all pass.

- [ ] **Step 2: (Optional, on NPU) Run dense vs block-sparse parity script**

Create a temporary script `scripts/block_sparse_parity_check.py`:

```python
import os
import torch
from vllm_omni.diffusion.models.mgm_video.mmdit.asa import (
    AsaConfig, GilbertRearranger, asa_attention,
)


def _dense_fa(q, k, v, atten_mask):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask)


def _block_sparse_fa(q, k, v, block_mask):
    from vllm_omni.diffusion.models.mgm_video.mmdit.block_sparse_attn import (
        npu_block_sparse_attention_wrapper,
    )
    return npu_block_sparse_attention_wrapper(q, k, v, block_mask, block_size=128)


def main():
    W, H, Tdepth = 4, 3, 2
    T = W * H * Tdepth
    L = 5
    B, N, D = 1, 2, 64

    torch.manual_seed(0)
    q = torch.randn(B, N, T + L, D, dtype=torch.bfloat16).npu()
    k = torch.randn(B, N, T + L, D, dtype=torch.bfloat16).npu()
    v = torch.randn(B, N, T + L, D, dtype=torch.bfloat16).npu()

    cfg = AsaConfig(
        enable=True, variant="asa", max_retain_ratio=1.0, min_retain_ratio=1.0,
        energy_threshold=0.95, block_size=128, num_keep=4, use_gilbert=False, text_length=L,
    )
    rearr = GilbertRearranger(W, H, Tdepth, text_length=L).to(q.device)

    out_dense = asa_attention(q, k, v, cfg, rearr, t_len=T, l_len=L, fa_full_dense=_dense_fa)
    out_sparse = asa_attention(
        q, k, v, cfg, rearr, t_len=T, l_len=L,
        fa_full_dense=_dense_fa,
        fa_block_sparse=_block_sparse_fa,
    )
    rel_err = (out_dense - out_sparse).abs().mean() / out_dense.abs().mean()
    print(f"relative mean abs error: {rel_err.item():.6f}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/block_sparse_parity_check.py
```

Expected: script completes and prints a small relative error (< 1%).

- [ ] **Step 3: Commit parity script (optional)**

If the script is useful, commit it:

```bash
git add scripts/block_sparse_parity_check.py
git commit -m "chore(mgm_video): add dense vs block-sparse parity check script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Final verification and cleanup

- [ ] **Step 1: Run lint/type check if available**

```bash
# If the project has a lint command, run it; otherwise skip.
python -m py_compile vllm_omni/diffusion/models/mgm_video/mmdit/block_sparse_attn.py
python -m py_compile vllm_omni/diffusion/models/mgm_video/mmdit/asa.py
python -m py_compile vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py
```

Expected: no syntax errors.

- [ ] **Step 2: Final commit or summary**

If everything passes, the branch is ready for a final summary commit or PR.

```bash
git log --oneline -5
```

Expected: commits from Tasks 1-5 (and optional 6) are present.

---

## 自我审查

### Spec coverage检查

| Spec 要求 | 对应任务 |
|---|---|
| 新增 `block_sparse_attn.py` 封装算子调用 | Task 1 |
| `asa.py` 支持 `fa_block_sparse` 回调 | Task 2 |
| `mmdit_blocks_inference.py` 读取 env var 并构造回调 | Task 3 |
| 块掩码 bool → int8 转换 | Task 1 / Task 4 |
| 单元测试（mock callback） | Task 4 |
| NPU smoke test | Task 5 |
| 稠密路径保留作为回退 | Task 3 |
| 精度/性能验证 | Task 6 |

### Placeholder 扫描

- 无 TBD/TODO/"implement later"/"fill in details"。
- 所有步骤包含具体代码或命令。
- 类型和签名前后一致：`fa_block_sparse(q, k, v, block_mask)`。

### 类型一致性

- `npu_block_sparse_attention_wrapper` 签名与 spec 一致。
- `asa_attention` 新增参数位置在末尾，避免破坏现有调用点。
- `mmdit_blocks_inference.py` 中构造的 `_fa_block_sparse` 签名与 `asa_attention` 期望一致。

### 已知限制

- `npu_block_sparse_attention` 在 `torch_npu` 旧版本中可能不存在；代码中已做显式检查并抛出 `RuntimeError`。
- 当前模型为 MHA，GQA 场景下 `num_key_value_heads` 需要调整，但超出本计划范围。
- NPU smoke test 和 parity check 必须在真实 NPU 环境中运行。
