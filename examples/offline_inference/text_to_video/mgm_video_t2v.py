#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline inference example for MGM-Video (MimoGPT) T2V generation.

Usage:
    # First, setup the weight directory:
    python -m vllm_omni.diffusion.models.mgm_video.setup_weights \
        --dit-ckpt /path/to/iter_799.pth \
        --vae-ckpt /path/to/causal_vae_v3.1_sd3.pth \
        --t5-path /path/to/t5_ckpts

    # Then run inference:
    python mgm_video_t2v.py \
        --model /path/to/mgm_video_11b_vllm \
        --prompt "A cat playing piano in a cozy room" \
        --height 720 --width 1280 --num-frames 121
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

_MODEL_PRESETS = {
    "mgm_11b": {
        "height": 720,
        "width": 1280,
        "num_frames": 121,
        "num_inference_steps": 9,
        "guidance_scale": 1.0,
        "fps": 24,
        "output": "mgm_video_output.mp4",
    },
    "mgm_11b_480p": {
        "height": 480,
        "width": 864,
        "num_frames": 121,
        "num_inference_steps": 9,
        "guidance_scale": 1.0,
        "fps": 24,
        "output": "mgm_video_480p_output.mp4",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="MGM-Video T2V generation with vllm-omni")
    parser.add_argument("--model", type=str, required=True, help="Path to MGM-Video model directory")
    parser.add_argument("--prompt", type=str, default="A serene lakeside sunrise with mist over the water.",
                        help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale (default: 1.0 = no CFG)")
    parser.add_argument("--height", type=int, default=None, help="Video height")
    parser.add_argument("--width", type=int, default=None, help="Video width")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--flow-shift", type=float, default=None,
                        help="Time shifting factor (default: 12.0)")
    parser.add_argument("--output", type=str, default=None, help="Output path (mp4)")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise offloading")
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses SP degree")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring SP degree")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE patch parallel size")
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help="Enable Hybrid Sharded Data Parallel to shard model weights across NPUs.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=-1,
        help=(
            "Number of NPUs to shard model weights across within each replica group. "
            "-1 (default) auto-calculates as world_size / replicate_size."
        ),
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help=(
            "Number of replica groups for HSDP. Each replica holds a full sharded copy. "
            "Default 1 means pure sharding (no replication)."
        ),
    )
    parser.add_argument(
        "--use-cp",
        action="store_true",
        help="Use Context Parallelism (Ulysses) + full_denoise offload instead of HSDP. "
             "Recommended for MGM-Video on 2 NPUs.",
    )
    parser.add_argument(
        "--skip-dummy-run",
        action="store_true",
        help="Skip dummy run during engine init (auto-enabled with --enforce-eager).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducible inference "
             "(torch.use_deterministic_algorithms + HCCL_DETERMINISTIC).",
    )
    return parser.parse_args()


def main():
    # Reduce NPU memory allocator fragmentation (used by MGM-Video-Ascend)
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    # Match original MGM-Video-Ascend NPU precision settings.
    # The original repo sets allow_internal_format=False in distributed_vae.py,
    # which prevents NPU internal format (5HD/3ND) conversions that introduce
    # precision differences. In torch_npu >= 2.x this maps to ACL_OP_SELECT_IMPL_MODE.
    # Must be set before torch_npu initializes.
    os.environ.setdefault("ACL_OP_SELECT_IMPL_MODE", "high_precision")

    args = parse_args()

    # Apply defaults
    preset = _MODEL_PRESETS["mgm_11b"]
    for key, default_val in preset.items():
        if getattr(args, key.replace("-", "_"), None) is None:
            setattr(args, key.replace("-", "_"), default_val)

    # When --use-cp is set, use Context Parallelism + DiT inference offload
    # (no HSDP — this matches the original MGM-Video-Ascend repo's approach)
    # CP is implemented as Ulysses sequence parallelism: set ulysses_degree=2
    # so that DiffusionParallelConfig.world_size=2, launching 2 worker processes.
    # The MGM-Video pipeline then creates its own CP group via
    # mmdit_parallel_states.initialize_distributed() and handles
    # all-to-all communication inside mmdit_blocks.py.
    if args.use_cp:
        args.use_hsdp = False
        args.ulysses_degree = 2
        args.enable_layerwise_offload = True
        args.skip_dummy_run = True

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    omni_kwargs = dict(
        model=args.model,
        model_class_name="MGMVideoPipeline",
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        skip_dummy_run=args.skip_dummy_run if args.skip_dummy_run else None,
        deterministic=args.deterministic,
    )
    if args.flow_shift is not None:
        omni_kwargs["flow_shift"] = args.flow_shift

    print(f"\n{'=' * 60}")
    print("MGM-Video T2V Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Flow shift: {args.flow_shift or 12.0}")
    print(f"{'=' * 60}\n")
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))
    if profiler_enabled:
        from vllm.config import ProfilerConfig
        omni_kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=os.environ["VLLM_TORCH_PROFILER_DIR"],
            torch_profiler_with_memory=True,
        )

    omni = Omni(**omni_kwargs)


    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    sampling_kwargs = dict(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )

    for _ in range(3):
        start = time.perf_counter()
        if profiler_enabled:
            omni.start_profile()
        frames = omni.generate(prompt_dict, OmniDiffusionSamplingParams(**sampling_kwargs))
        if profiler_enabled:
            profile_results = omni.stop_profile()

        elapsed = time.perf_counter() - start
        print(f"Generation time: {elapsed:.2f}s")

    # Extract video frames
    if isinstance(frames, list):
        frames = frames[0] if frames else None
    if isinstance(frames, OmniRequestOutput):
        if frames.is_pipeline_output and frames.request_output is not None:
            frames = frames.request_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if isinstance(frames.images[0], dict):
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images

    # Convert frames to [T, H, W, C] uint8 tensor for torchvision write_video.
    # The post_process_func returns uint8 numpy of shape [B, T, H, W, C] with
    # values in [0, 255], matching the original MGM-Video-Ascend save_sample.
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5 and video_tensor.dtype == torch.uint8:
            # Already post-processed: [B, T, H, W, C] uint8
            video_tensor = video_tensor[0]  # [T, H, W, C]
        elif video_tensor.dim() == 5:
            # [B, C, T, H, W] float -> [T, H, W, C] uint8
            video_tensor = video_tensor[0].clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
        elif video_tensor.dim() == 4 and video_tensor.dtype == torch.uint8:
            # [T, H, W, C] uint8 — already ready
            pass
        elif video_tensor.dim() == 4:
            # [C, T, H, W] float -> [T, H, W, C] uint8
            video_tensor = video_tensor.clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
    elif isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]  # [B, T, H, W, C] -> [T, H, W, C]
        if frames.ndim == 4 and frames.shape[-1] in (1, 3, 4):
            # Already [T, H, W, C] — may be uint8 from post_process_func
            # or float from older code paths
            if frames.dtype == np.uint8:
                video_tensor = torch.from_numpy(frames.copy())
            else:
                # Float [0,1] path: apply round compensation and convert to uint8
                video_tensor = torch.from_numpy(frames.copy())
                video_tensor = video_tensor.mul(255).add(0.5).clamp(0, 255).to(torch.uint8)
        elif frames.ndim == 4:
            # [C, T, H, W] -> [T, H, W, C]
            video_tensor = torch.from_numpy(frames.transpose(1, 2, 3, 0).copy())
        else:
            video_tensor = torch.from_numpy(frames)
    elif isinstance(frames, list):
        # frames may be [torch.Tensor] or [np.ndarray of shape [B, T, H, W, C]] from images list
        if len(frames) > 0 and isinstance(frames[0], torch.Tensor):
            # Pipeline may return pre-processed uint8 [B, T, H, W, C] or
            # raw float [B, C, T, H, W]. Handle both.
            t = frames[0].detach().cpu()
            if t.dim() == 5 and t.dtype == torch.uint8:
                # Already post-processed: [B, T, H, W, C] uint8
                video_tensor = t[0]  # [T, H, W, C]
            elif t.dim() == 5:
                # [B, C, T, H, W] float -> [T, H, W, C] uint8
                t = t[0].clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
                video_tensor = t
            elif t.dim() == 4 and t.dtype == torch.uint8:
                # [T, H, W, C] uint8 — already ready
                video_tensor = t
            elif t.dim() == 4:
                # [C, T, H, W] float -> [T, H, W, C] uint8
                t = t.clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
                video_tensor = t
            else:
                raise ValueError(f"Unsupported tensor shape/dtype in list: dim={t.dim()}, dtype={t.dtype}")
        elif len(frames) > 0 and isinstance(frames[0], np.ndarray):
            arr = frames[0]
            if arr.ndim == 5:
                arr = arr[0]  # [B, T, H, W, C] -> [T, H, W, C]
            if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
                if arr.dtype == np.uint8:
                    video_tensor = torch.from_numpy(arr.copy())
                else:
                    video_tensor = torch.from_numpy(arr.copy())
                    video_tensor = video_tensor.mul(255).add(0.5).clamp(0, 255).to(torch.uint8)
            elif arr.ndim == 4:
                video_tensor = torch.from_numpy(arr.transpose(1, 2, 3, 0).copy())
            else:
                video_tensor = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported frames list content type: {type(frames[0]) if frames else 'empty'}")
    else:
        raise ValueError(f"Unsupported frames type: {type(frames)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use torchvision write_video with CRF=10 to match the original
    # MGM-Video-Ascend repo's encoding quality (H.264, CRF=10, preset=medium).
    from torchvision.io import write_video
    write_video(
        str(output_path),
        video_tensor,
        fps=args.fps,
        video_codec="h264",
        options={"crf": "10", "preset": "medium"},
    )
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()