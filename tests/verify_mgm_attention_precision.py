#!/usr/bin/env python3
"""
精度验证脚本：对比 mgm_video attention 替换前后的输出差异。

用法：
    # 对比模式（自动运行原始 vs 新实现）
    python scripts/verify_mgm_attention_precision.py --attention-type join --compare
    python scripts/verify_mgm_attention_precision.py --attention-type self --compare
    python scripts/verify_mgm_attention_precision.py --attention-type cross --compare

    # 单独运行新实现
    python scripts/verify_mgm_attention_precision.py --attention-type join

    # 指定参数
    python scripts/verify_mgm_attention_precision.py \
        --attention-type join \
        --batch-size 1 \
        --seq-len 256 \
        --ctx-len 120 \
        --hidden-size 3072 \
        --num-heads 24 \
        --use-rope \
        --use-qknorm \
        --compare

退出码：
    0 - 通过（差异在阈值内）
    1 - 失败（差异超出阈值）
    2 - 错误（运行异常）
"""

import argparse
import os
import sys
import warnings
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 忽略一些常见的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(out1: torch.Tensor, out2: torch.Tensor) -> Dict[str, float]:
    """计算两个 tensor 之间的数值差异指标。"""
    diff = (out1 - out2).abs()
    rel_diff = diff / (out1.abs() + 1e-8)

    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            out1.flatten(), out2.flatten(), dim=0
        ).item(),
    }


def init_distributed():
    """初始化单进程分布式环境（JoinAttention 需要 CP group）。"""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group("gloo", rank=0, world_size=1)


def build_spatial_freq(batch_size: int, seq_len: int, head_dim: int, device: str):
    """构建模拟的 3D RoPE 频率张量。"""
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import (
        create_sinusoidal_positions,
    )

    # 简化为 2D RoPE（h, w）
    hw = int(seq_len ** 0.5)
    if hw * hw != seq_len:
        # 如果 seq_len 不是完全平方数，向上取整
        hw = int(seq_len ** 0.5) + 1

    embed_positions_h = create_sinusoidal_positions(hw, head_dim // 2)
    embed_positions_w = create_sinusoidal_positions(hw, head_dim // 2)

    # 扩展为 seq_len
    position_ids_h = torch.arange(seq_len, device=device) % hw
    position_ids_w = torch.arange(seq_len, device=device) // hw

    sincos_h = embed_positions_h[position_ids_h].to(device=device)
    sincos_w = embed_positions_w[position_ids_w].to(device=device)

    return [sincos_h, sincos_w, sincos_h]  # 复用 h 作为 t


def test_join_attention(args) -> torch.Tensor:
    """测试 JoinAttention 的 fa() 输出。"""
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import JoinAttention

    init_distributed()

    attn = JoinAttention(
        n_embd=args.hidden_size,
        n_head=args.num_heads,
        dropout=0.0,
        fa_keep_prob=1.0,
        use_3d_rope=args.use_rope,
        use_qknorm=args.use_qknorm,
        use_context_parallelism=False,
        downscale=1,
    )
    attn.eval()

    B = args.batch_size
    T = args.seq_len
    L = args.ctx_len
    C = args.hidden_size

    x = torch.randn(B, T, C)
    y = torch.randn(B, L, C)
    spatial_freq = None
    if args.use_rope:
        spatial_freq = build_spatial_freq(B, T, C // args.num_heads, x.device)

    # 调用 before_fa 准备 q, k, v
    q, k, v, mask, B_out, T_out, C_out, L_out, frame, spatial_tn = attn.before_fa(
        x1=x,
        x1_cts=None,
        y=y,
        spatial_freq=spatial_freq,
        x_padding_size=0,
        y_padding_size=0,
        mask=None,
        f=None,
        hh=None,
        ww=None,
    )

    # 调用 fa() 计算 attention
    out = attn.fa(q, k, v, mask, C_out, offload_fa=False)

    # 调用 after_fa 获取最终输出
    x_out, y_out = attn.after_fa(
        out=out,
        B=B_out,
        T=T_out,
        C=C_out,
        L=L_out,
        q=q,
        x1_cts=None,
        frame=frame,
        spatial_tn=spatial_tn,
        x_padding_size=0,
        y_padding_size=0,
    )

    # 拼接 x 和 y 作为整体输出用于对比
    return torch.cat([x_out, y_out], dim=1)


def test_self_attention(args) -> torch.Tensor:
    """测试 SelfAttention 的 forward() 输出。"""
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import SelfAttention

    attn = SelfAttention(
        n_embd=args.hidden_size,
        n_head=args.num_heads,
        dropout=0.0,
    )
    attn.eval()

    B = args.batch_size
    T = args.seq_len
    C = args.hidden_size

    x = torch.randn(B, T, C)
    spatial_freq = None
    if args.use_rope:
        spatial_freq = build_spatial_freq(B, T, C // args.num_heads, x.device)

    mask = None
    if args.use_mask:
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -10:] = False

    return attn.forward(x, mask=mask, spatial_freq=spatial_freq)


def test_cross_attention(args) -> torch.Tensor:
    """测试 CrossAttention 的 forward() 输出。"""
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks import CrossAttention

    attn = CrossAttention(
        n_embd=args.hidden_size,
        n_head=args.num_heads,
        dropout=0.0,
    )
    attn.eval()

    B = args.batch_size
    T = args.seq_len
    L = args.ctx_len
    C = args.hidden_size

    x = torch.randn(B, T, C)  # query source
    y = torch.randn(B, L, C)  # key/value source

    mask = None
    if args.use_mask:
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[:, -5:] = False

    return attn.forward(x, y, mask=mask)


def run_test(args) -> torch.Tensor:
    """根据类型分发到对应的测试函数。"""
    if args.attention_type == "join":
        return test_join_attention(args)
    elif args.attention_type == "self":
        return test_self_attention(args)
    elif args.attention_type == "cross":
        return test_cross_attention(args)
    else:
        raise ValueError(f"Unknown attention type: {args.attention_type}")


def print_metrics(metrics: Dict[str, float], threshold: float = 1e-3):
    """打印精度对比结果。"""
    print(f"\n{'='*60}")
    print(f"Precision Comparison Results")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.6e}")
    print(f"{'='*60}")

    max_diff = metrics["max_abs_diff"]
    cos_sim = metrics["cosine_sim"]
    passed = max_diff < threshold and cos_sim > 0.999

    status = "PASS" if passed else "FAIL"
    print(f"\nThreshold: max_abs_diff < {threshold}, cosine_sim > 0.999")
    print(f"Result: {status}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify mgm_video attention precision after replacement"
    )
    parser.add_argument(
        "--attention-type",
        choices=["join", "self", "cross"],
        required=True,
        help="Which attention type to test",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (image tokens)")
    parser.add_argument("--ctx-len", type=int, default=64, help="Context length (text tokens)")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--use-rope", action="store_true", help="Enable 3D RoPE")
    parser.add_argument("--use-qknorm", action="store_true", help="Enable QK norm")
    parser.add_argument("--use-mask", action="store_true", help="Use attention mask")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare original vs new implementation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Max absolute diff threshold for PASS",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/npu)")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device != "cpu":
        # 尝试设置设备
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)
        elif args.device == "npu":
            try:
                import torch_npu

                torch.npu.set_device(0)
            except ImportError:
                print("WARNING: torch_npu not available, falling back to cpu")
                args.device = "cpu"

    try:
        if args.compare:
            print(f"Testing {args.attention_type} attention...")
            print(f"  Config: batch={args.batch_size}, seq={args.seq_len}, "
                  f"ctx={args.ctx_len}, hidden={args.hidden_size}, heads={args.num_heads}")
            print(f"  RoPE={args.use_rope}, QKNorm={args.use_qknorm}, Mask={args.use_mask}")

            # 运行原始实现
            print("\n[1/2] Running original implementation...")
            os.environ["MGM_USE_ORIGINAL_ATTN"] = "1"
            out_orig = run_test(args)
            print(f"  Output shape: {out_orig.shape}, dtype: {out_orig.dtype}")

            # 运行新实现
            print("\n[2/2] Running new implementation...")
            os.environ["MGM_USE_ORIGINAL_ATTN"] = "0"
            out_new = run_test(args)
            print(f"  Output shape: {out_new.shape}, dtype: {out_new.dtype}")

            # 对比
            metrics = compute_metrics(out_orig, out_new)
            passed = print_metrics(metrics, threshold=args.threshold)

            return 0 if passed else 1
        else:
            print(f"Running {args.attention_type} attention (new implementation)...")
            os.environ["MGM_USE_ORIGINAL_ATTN"] = "0"
            out = run_test(args)
            print(f"Output shape: {out.shape}")
            print(f"Output mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
            return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 2

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
