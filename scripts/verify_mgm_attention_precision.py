#!/usr/bin/env python3
"""
Single-op precision verification script for mgm_video attention backends.

Compares JoinAttention.fa and JoinAttentionInference.infer outputs between:
- vllm-omni backend (default, VLLM_MGM_USE_NATIVE_FA unset or != "1")
- native backend (VLLM_MGM_USE_NATIVE_FA=1)

Usage:
    python scripts/verify_mgm_attention_precision.py
    python scripts/verify_mgm_attention_precision.py --device npu --atol 1e-4 --rtol 1e-3

Exit codes:
    0 - All tests PASS
    1 - At least one test FAIL
    2 - Runtime error
"""

import argparse
import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def init_distributed():
    """Initialize single-process distributed environment (JoinAttention needs CP group)."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group("gloo", rank=0, world_size=1)


def compute_metrics(out1: torch.Tensor, out2: torch.Tensor) -> Dict[str, float]:
    """Compute numerical difference metrics between two tensors."""
    diff = (out1 - out2).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


def print_result(name: str, metrics: Dict[str, float], passed: bool, atol: float, rtol: float):
    """Print test result with metrics."""
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {name}")
    print(f"    max_abs_diff : {metrics['max_abs_diff']:.6e}")
    print(f"    mean_abs_diff: {metrics['mean_abs_diff']:.6e}")
    print(f"    threshold    : atol={atol}, rtol={rtol}")


def build_spatial_freq(seq_len: int, head_dim: int, device: torch.device):
    """Build simulated 3D RoPE frequency tensors matching the model's approach.

    The model uses rope_ratio * attention_head_dim as the dim for
    create_sinusoidal_positions. For the default model config:
    head_dim=128, rope_ratio=[22/64, 22/64, 20/64].
    """
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import (
        create_sinusoidal_positions,
    )

    hw = int(seq_len ** 0.5)
    if hw * hw != seq_len:
        hw = int(seq_len ** 0.5) + 1

    # Match the model's rope_ratio logic from mmdit.py
    # For head_dim=128: rope_ratio=[22/64, 22/64, 20/64]
    # dim_h = int(22/64 * 128) = 44, dim_w = 44, dim_t = 40
    rope_ratio = [22 / 64, 22 / 64, 20 / 64]
    dim_h = int(rope_ratio[0] * head_dim)
    dim_w = int(rope_ratio[1] * head_dim)
    dim_t = int(rope_ratio[2] * head_dim)

    embed_positions_h = create_sinusoidal_positions(hw, dim_h)
    embed_positions_w = create_sinusoidal_positions(hw, dim_w)
    embed_positions_t = create_sinusoidal_positions(hw, dim_t)

    position_ids_h = torch.arange(seq_len, device=device) % hw
    position_ids_w = torch.arange(seq_len, device=device) // hw

    sincos_h = embed_positions_h[position_ids_h].to(device=device)
    sincos_w = embed_positions_w[position_ids_w].to(device=device)
    sincos_t = embed_positions_t[position_ids_h].to(device=device)

    return [sincos_h, sincos_w, sincos_t]


def create_join_attention(n_embd: int, n_head: int, use_vllm: bool, device: torch.device):
    """Create a JoinAttention instance with the specified backend.

    The VLLM_MGM_USE_NATIVE_FA env var is checked at __init__ time, so we must
    set/unset it before creating the instance.
    """
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import JoinAttention

    if use_vllm:
        os.environ.pop("VLLM_MGM_USE_NATIVE_FA", None)
    else:
        os.environ["VLLM_MGM_USE_NATIVE_FA"] = "1"

    attn = JoinAttention(
        n_embd=n_embd,
        n_head=n_head,
        dropout=0.0,
        fa_keep_prob=1.0,
        use_3d_rope=True,
        use_qknorm=True,
        use_context_parallelism=False,
        downscale=1,
    )
    attn = attn.to(device)
    attn.eval()

    # On CPU, force native backend to use torch SDPA instead of NPU-specific
    # npu_fusion_attention, so both backends use the same underlying kernel.
    if device.type == "cpu":
        attn.npu_fusion = False
        attn.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    return attn


def create_join_attention_inference(n_embd: int, n_head: int, use_vllm: bool, device: torch.device):
    """Create a JoinAttentionInference instance with the specified backend."""
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks_inference import JoinAttentionInference

    if use_vllm:
        os.environ.pop("VLLM_MGM_USE_NATIVE_FA", None)
    else:
        os.environ["VLLM_MGM_USE_NATIVE_FA"] = "1"

    attn = JoinAttentionInference(
        n_embd=n_embd,
        n_head=n_head,
        dropout=0.0,
        fa_keep_prob=1.0,
        use_3d_rope=True,
        use_qknorm=True,
        use_context_parallelism=False,
        downscale=1,
    )
    attn = attn.to(device)
    attn.eval()

    # On CPU, force native backend to use torch SDPA instead of NPU-specific
    # npu_fusion_attention, so both backends use the same underlying kernel.
    if device.type == "cpu":
        attn.npu_fusion = False
        attn.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    return attn


def test_join_attention_precision(
    device: torch.device,
    B: int = 1,
    T: int = 128,
    L: int = 64,
    C: int = 768,
    num_heads: int = 12,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Tuple[bool, Dict[str, float]]:
    """Test JoinAttention.fa with identical inputs on both backends."""
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import JoinAttention

    init_distributed()

    head_dim = C // num_heads

    # Generate identical random inputs
    x = torch.randn(B, T, C, device=device)
    y = torch.randn(B, L, C, device=device)
    spatial_freq = build_spatial_freq(T, head_dim, device)

    # Run with vllm-omni backend
    attn_vllm = create_join_attention(C, num_heads, use_vllm=True, device=device)
    with torch.no_grad():
        q, k, v, prepared_mask, B_out, T_out, C_out, L_out, _, _ = attn_vllm.before_fa(
            x, None, y, spatial_freq=spatial_freq,
            x_padding_size=0, y_padding_size=0,
            mask=None, f=None, hh=None, ww=None,
        )
        out_vllm = attn_vllm.fa(q, k, v, prepared_mask, C_out, offload_fa=False)
        x_vllm, y_vllm = attn_vllm.after_fa(
            out=out_vllm, B=B_out, T=T_out, C=C_out, L=L_out,
            q=q, x1_cts=None, frame=None, spatial_tn=None,
            x_padding_size=0, y_padding_size=0,
        )

    # Run with native backend - copy weights from vllm instance for fair comparison
    attn_native = create_join_attention(C, num_heads, use_vllm=False, device=device)
    attn_native.load_state_dict(attn_vllm.state_dict())
    with torch.no_grad():
        q, k, v, prepared_mask, B_out, T_out, C_out, L_out, _, _ = attn_native.before_fa(
            x, None, y, spatial_freq=spatial_freq,
            x_padding_size=0, y_padding_size=0,
            mask=None, f=None, hh=None, ww=None,
        )
        out_native = attn_native.fa(q, k, v, prepared_mask, C_out, offload_fa=False)
        x_native, y_native = attn_native.after_fa(
            out=out_native, B=B_out, T=T_out, C=C_out, L=L_out,
            q=q, x1_cts=None, frame=None, spatial_tn=None,
            x_padding_size=0, y_padding_size=0,
        )

    # Compare outputs
    out_vllm_cat = torch.cat([x_vllm, y_vllm], dim=1)
    out_native_cat = torch.cat([x_native, y_native], dim=1)

    metrics = compute_metrics(out_vllm_cat, out_native_cat)
    passed = torch.allclose(out_vllm_cat, out_native_cat, atol=atol, rtol=rtol)

    return passed, metrics


def test_join_attention_inference_precision(
    device: torch.device,
    B: int = 1,
    T: int = 128,
    L: int = 64,
    C: int = 768,
    num_heads: int = 12,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Tuple[bool, Dict[str, float]]:
    """Test JoinAttentionInference.infer with identical inputs on both backends.

    Note: JoinAttentionInference.infer requires context parallelism setup.
    We test with use_context_parallelism=False, so CP is not active.
    """
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks_inference import JoinAttentionInference

    init_distributed()

    head_dim = C // num_heads

    # Generate identical random inputs
    x = torch.randn(B, T, C, device=device)
    y = torch.randn(B, L, C, device=device)
    spatial_freq = build_spatial_freq(T, head_dim, device)

    # Run with vllm-omni backend
    attn_vllm = create_join_attention_inference(C, num_heads, use_vllm=True, device=device)
    with torch.no_grad():
        x_vllm, y_vllm = attn_vllm.infer(
            x, y, None, spatial_freq=spatial_freq,
            x_padding_size=0, y_padding_size=0,
            mask=None, f=None, hh=None, ww=None,
        )

    # Run with native backend - copy weights from vllm instance for fair comparison
    attn_native = create_join_attention_inference(C, num_heads, use_vllm=False, device=device)
    attn_native.load_state_dict(attn_vllm.state_dict())
    with torch.no_grad():
        x_native, y_native = attn_native.infer(
            x, y, None, spatial_freq=spatial_freq,
            x_padding_size=0, y_padding_size=0,
            mask=None, f=None, hh=None, ww=None,
        )

    # Compare outputs
    out_vllm_cat = torch.cat([x_vllm, y_vllm], dim=1)
    out_native_cat = torch.cat([x_native, y_native], dim=1)

    metrics = compute_metrics(out_vllm_cat, out_native_cat)
    passed = torch.allclose(out_vllm_cat, out_native_cat, atol=atol, rtol=rtol)

    return passed, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Verify mgm_video attention precision: vllm-omni backend vs native backend"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to test on (cpu/cuda/npu)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (image tokens)")
    parser.add_argument("--ctx-len", type=int, default=64, help="Context length (text tokens)")
    parser.add_argument("--hidden-size", type=int, default=3072, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=24, help="Number of attention heads")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for allclose")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for allclose")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Force SDPA backend for reproducible CPU testing
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

    set_seed(args.seed)

    # Determine device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device_str = "cpu"
    if device_str == "npu":
        try:
            import torch_npu
            if not torch.npu.is_available():
                print("WARNING: NPU not available, falling back to CPU")
                device_str = "cpu"
        except ImportError:
            print("WARNING: torch_npu not available, falling back to CPU")
            device_str = "cpu"

    device = torch.device(device_str)
    print(f"Testing on device: {device}")
    print(f"Config: batch={args.batch_size}, seq={args.seq_len}, ctx={args.ctx_len}, "
          f"hidden={args.hidden_size}, heads={args.num_heads}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print("=" * 60)

    all_passed = True

    # Test 1: JoinAttention.fa
    print("\n[1/2] Testing JoinAttention.fa ...")
    try:
        passed, metrics = test_join_attention_precision(
            device=device,
            B=args.batch_size,
            T=args.seq_len,
            L=args.ctx_len,
            C=args.hidden_size,
            num_heads=args.num_heads,
            atol=args.atol,
            rtol=args.rtol,
        )
        print_result("JoinAttention.fa", metrics, passed, args.atol, args.rtol)
        all_passed = all_passed and passed
    except Exception as e:
        print(f"\n  [ERROR] JoinAttention.fa: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: JoinAttentionInference.infer
    print("\n[2/2] Testing JoinAttentionInference.infer ...")
    try:
        passed, metrics = test_join_attention_inference_precision(
            device=device,
            B=args.batch_size,
            T=args.seq_len,
            L=args.ctx_len,
            C=args.hidden_size,
            num_heads=args.num_heads,
            atol=args.atol,
            rtol=args.rtol,
        )
        print_result("JoinAttentionInference.infer", metrics, passed, args.atol, args.rtol)
        all_passed = all_passed and passed
    except Exception as e:
        print(f"\n  [ERROR] JoinAttentionInference.infer: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED")
        return 0
    else:
        print("Some tests FAILED")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
