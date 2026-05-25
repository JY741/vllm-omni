"""Precision verification: old apply_3drotary_pos vs new RotaryEmbedding-based 3D RoPE.

Run: VLLM_PLUGINS=omni-npu python tests/mgm_video/test_rope_precision.py
"""

import torch

from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import (
    apply_3drotary_pos,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


def _create_sinusoidal_positions_cpu(num_pos: int, dim: int) -> torch.Tensor:
    """CPU version of create_sinusoidal_positions for testing."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    sinusoid_inp = torch.stack((sinusoid_inp, sinusoid_inp), dim=-1).flatten(-2)
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def _convert_spatial_freq(spatial_freq):
    """Same logic as MMDiTInference._convert_spatial_freq_for_inference."""
    result = []
    for sincos in spatial_freq:
        dim = sincos.shape[-1] // 2
        sin_interleaved = sincos[:, :dim]
        cos_interleaved = sincos[:, dim:]
        result.append((cos_interleaved[:, ::2], sin_interleaved[:, ::2]))
    return result


def _apply_3d_rope_new(rope, q, k, spatial_freq):
    """Same logic as JoinAttentionInference._apply_3d_rope.

    Args:
        q, k: [B, S, N, D] (BSND layout)
        spatial_freq: [(cos_h, sin_h), (cos_w, sin_w), (cos_t, sin_t)]
    Returns:
        q, k: [B, S, N, D] with RoPE applied
    """
    (cos_h, sin_h), (cos_w, sin_w), (cos_t, sin_t) = spatial_freq
    dim_h = cos_h.shape[-1] * 2
    dim_w = cos_w.shape[-1] * 2
    dim_t = cos_t.shape[-1] * 2

    q_h, q_w, q_t = q.split([dim_h, dim_w, dim_t], dim=-1)
    k_h, k_w, k_t = k.split([dim_h, dim_w, dim_t], dim=-1)

    q_h = rope(q_h, cos_h, sin_h)
    k_h = rope(k_h, cos_h, sin_h)
    q_w = rope(q_w, cos_w, sin_w)
    k_w = rope(k_w, cos_w, sin_w)
    q_t = rope(q_t, cos_t, sin_t)
    k_t = rope(k_t, cos_t, sin_t)

    q = torch.cat([q_h, q_w, q_t], dim=-1)
    k = torch.cat([k_h, k_w, k_t], dim=-1)
    return q, k


def test_rope_precision():
    torch.manual_seed(42)
    device = torch.device("npu" if torch.npu.is_available() else "cpu") if hasattr(torch, "npu") else torch.device("cpu")

    B, S, N, D = 1, 64, 24, 128
    rope_ratio = [22 / 64, 22 / 64, 20 / 64]
    base_size_h, base_size_w, base_size_t = 8, 8, 1
    dim_h = int(rope_ratio[0] * D)
    dim_w = int(rope_ratio[1] * D)
    dim_t = int(rope_ratio[2] * D)

    # bf16 atol is relaxed: on NPU the new path uses mindiesd fused kernel while
    # the old path uses PyTorch elementwise ops, producing ~1e-2 numerical difference.
    # fp32 confirms mathematical equivalence (zero diff).
    for dtype, atol in [(torch.float32, 1e-6), (torch.bfloat16, 2e-2)]:
        # Old path (apply_3drotary_pos) expects BNSD
        q_bnsd = torch.randn(B, N, S, D, dtype=dtype, device=device)
        k_bnsd = torch.randn(B, N, S, D, dtype=dtype, device=device)

        embed_h = _create_sinusoidal_positions_cpu(base_size_h, dim_h).to(device)
        embed_w = _create_sinusoidal_positions_cpu(base_size_w, dim_w).to(device)
        embed_t = _create_sinusoidal_positions_cpu(base_size_t, dim_t).to(device)

        spatial_ids_h = [i for _ in range(base_size_t) for i in range(base_size_h) for _ in range(base_size_w)]
        spatial_ids_w = [j for _ in range(base_size_t) for _ in range(base_size_h) for j in range(base_size_w)]
        temporal_ids = [tt for tt in range(base_size_t) for _ in range(base_size_h) for _ in range(base_size_w)]

        sincos_h = embed_h[spatial_ids_h].to(dtype)
        sincos_w = embed_w[spatial_ids_w].to(dtype)
        sincos_t = embed_t[temporal_ids].to(dtype)
        spatial_freq_old = [sincos_h, sincos_w, sincos_t]

        q_old, k_old = apply_3drotary_pos(q_bnsd.clone(), k_bnsd.clone(), freqs_cis=spatial_freq_old)

        # New path (_apply_3d_rope_new) expects BSND — transpose for input, transpose back for comparison
        q_bsnd = q_bnsd.clone().transpose(1, 2)
        k_bsnd = k_bnsd.clone().transpose(1, 2)
        spatial_freq_new = _convert_spatial_freq(spatial_freq_old)
        rope = RotaryEmbedding(is_neox_style=False)
        q_new_bsnd, k_new_bsnd = _apply_3d_rope_new(rope, q_bsnd, k_bsnd, spatial_freq_new)
        q_new = q_new_bsnd.transpose(1, 2)
        k_new = k_new_bsnd.transpose(1, 2)

        q_match = torch.allclose(q_old, q_new, atol=atol, rtol=1e-5)
        k_match = torch.allclose(k_old, k_new, atol=atol, rtol=1e-5)
        q_maxdiff = (q_old - q_new).abs().max().item()
        k_maxdiff = (k_old - k_new).abs().max().item()

        status = "PASS" if q_match and k_match else "FAIL"
        print(f"[{status}] dtype={dtype}, q_maxdiff={q_maxdiff:.2e}, k_maxdiff={k_maxdiff:.2e}, atol={atol}")
        assert q_match and k_match, f"Precision mismatch for {dtype}"

    print("All precision tests passed.")


if __name__ == "__main__":
    test_rope_precision()
