# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""Block-sparse attention wrapper for torch_npu.npu_block_sparse_attention."""

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover - CPU-only test environments
    torch_npu = None

__all__ = ["npu_block_sparse_attention_wrapper"]


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

    if block_mask.dim() != 4 or block_mask.shape[1] != 1:
        raise ValueError(
            f"block_mask must be [B, 1, nq, nk], got {block_mask.shape}"
        )

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
        inner_precise=0,  # 0 = fp32 softmax intermediate; required for bf16 inputs
    )
    return out
