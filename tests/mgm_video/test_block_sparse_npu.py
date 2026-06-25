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
