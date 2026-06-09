#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MGM-Video × BLADE ASA quality probe script (A-phase).

Runs three configurations — baseline, dense_probe, asa@0.20 — across a fixed
set of 5 prompts with a fixed seed. Writes per-prompt MP4s and a metrics.json
with PSNR and SSIM proxy vs baseline.

Usage::

    python scripts/probe_mgm_asa.py \
        --model /path/to/mgm_video_11b_vllm \
        --output-dir runs/asa_probe_smoke

P3 acceptance gate: dense_probe PSNR > 35 dB on all prompts.

See: docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md §7
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

# Ensure NPU allocator is pre-configured before any torch_npu import
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ACL_OP_SELECT_IMPL_MODE", "high_precision")

SEED = 1234

PROMPTS = [
    (
        "p01_person",
        "A close-up portrait of an elderly fisherman mending his nets at sunrise,"
        " warm golden light, realistic, cinematic.",
    ),
    (
        "p02_motion",
        "A leopard sprinting through tall grass at golden hour,"
        " motion blur on the grass, wildlife documentary style.",
    ),
    (
        "p03_scene",
        "Snow falling slowly over a quiet Tokyo back-alley at night,"
        " neon reflections on wet cobblestones, atmospheric.",
    ),
    (
        "p04_object",
        "A ceramic teacup tipping over onto a wooden table in slow motion,"
        " water splashing, macro lens.",
    ),
    (
        "p05_crowd",
        "A bustling street market in Marrakech at midday,"
        " vibrant colors, people moving, handheld camera feel.",
    ),
]

# Default inference parameters (TDM 9-step, 720p)
_DEFAULT_PRESET = {
    "height": 720,
    "width": 1280,
    "num_frames": 121,
    "num_inference_steps": 9,
    "guidance_scale": 1.0,
    "fps": 24,
}


def parse_args():
    p = argparse.ArgumentParser(description="MGM-Video ASA quality probe (A-phase)")
    p.add_argument("--model", required=True, help="Path to MGM-Video model directory")
    p.add_argument("--output-dir", required=True, help="Directory to write MP4s and metrics.json")
    p.add_argument("--height", type=int, default=_DEFAULT_PRESET["height"])
    p.add_argument("--width", type=int, default=_DEFAULT_PRESET["width"])
    p.add_argument("--num-frames", type=int, default=_DEFAULT_PRESET["num_frames"])
    p.add_argument("--num-inference-steps", type=int, default=_DEFAULT_PRESET["num_inference_steps"])
    p.add_argument("--guidance-scale", type=float, default=_DEFAULT_PRESET["guidance_scale"])
    p.add_argument("--fps", type=int, default=_DEFAULT_PRESET["fps"])
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--use-cp",
        action="store_true",
        help="Use Context Parallelism (Ulysses degree=2) + DiT layerwise offload, "
             "matching mgm_video_t2v.py's --use-cp 2-NPU mode.",
    )
    p.add_argument("--enable-sta", action="store_true",
                   help="Enable STA hybrid OR with ASA (sets VLLM_MGM_STA_ENABLE=1)")
    p.add_argument("--sta-window", default="7,13,13",
                   help="STA window 'wT,wH,wW' (default 7,13,13). Used only when --enable-sta is set.")
    return p.parse_args()


def build_pipeline(model: str, asa_cfg, args):
    """Build an Omni pipeline with the given AsaConfig injected via env vars."""
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni

    # Inject ASA config via environment variables so create_transformer_from_config
    # picks them up via AsaConfig.from_env() during pipeline construction.
    if asa_cfg is not None and asa_cfg.enable:
        os.environ["VLLM_MGM_ASA_ENABLE"] = "1"
        os.environ["VLLM_MGM_ASA_VARIANT"] = asa_cfg.variant
        os.environ["VLLM_MGM_ASA_MAX_RETAIN"] = str(asa_cfg.max_retain_ratio)
        os.environ["VLLM_MGM_ASA_MIN_RETAIN"] = str(asa_cfg.min_retain_ratio)
        os.environ["VLLM_MGM_ASA_ENERGY"] = str(asa_cfg.energy_threshold)
        os.environ["VLLM_MGM_ASA_USE_GILBERT"] = "1" if asa_cfg.use_gilbert else "0"
        os.environ["VLLM_MGM_ASA_TEXT_LEN"] = str(asa_cfg.text_length)
    else:
        os.environ.pop("VLLM_MGM_ASA_ENABLE", None)

    # STA hybrid (Task 7 of 2026-06-08 plan)
    if getattr(args, "enable_sta", False):
        os.environ["VLLM_MGM_STA_ENABLE"] = "1"
        os.environ["VLLM_MGM_STA_WINDOW"] = args.sta_window
    else:
        os.environ.pop("VLLM_MGM_STA_ENABLE", None)
        os.environ.pop("VLLM_MGM_STA_WINDOW", None)

    if args.use_cp:
        parallel_config = DiffusionParallelConfig(
            ulysses_degree=2,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        omni = Omni(
            model=model,
            model_class_name="MGMVideoPipeline",
            parallel_config=parallel_config,
            enforce_eager=args.enforce_eager,
            enable_layerwise_offload=True,
            skip_dummy_run=True,
        )
    else:
        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=args.tensor_parallel_size,
        )
        omni = Omni(
            model=model,
            model_class_name="MGMVideoPipeline",
            parallel_config=parallel_config,
            enforce_eager=args.enforce_eager,
        )
    return omni


def run_one(omni, prompt: str, args) -> torch.Tensor:
    """Run inference and return [T, H, W, C] uint8 tensor."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )

    result = omni.generate({"prompt": prompt}, sampling_params)
    frames = result

    # Unwrap output into [T, H, W, C] uint8 — mirrors mgm_video_t2v.py logic
    if isinstance(frames, list):
        frames = frames[0] if frames else None
    if isinstance(frames, OmniRequestOutput):
        if frames.is_pipeline_output and frames.request_output is not None:
            frames = frames.request_output
        if isinstance(frames, OmniRequestOutput) and frames.images:
            img = frames.images[0]
            frames = img.get("frames") or img.get("video") if isinstance(img, dict) else img

    if isinstance(frames, torch.Tensor):
        t = frames.detach().cpu()
        if t.dim() == 5 and t.dtype == torch.uint8:
            return t[0]
        if t.dim() == 5:
            return t[0].clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
        if t.dim() == 4 and t.dtype == torch.uint8:
            return t
        return t.clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8)

    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.dtype == np.uint8:
            return torch.from_numpy(frames.copy())
        return torch.from_numpy(frames.copy()).mul(255).add(0.5).clamp(0, 255).to(torch.uint8)

    raise ValueError(f"Unsupported frames type from pipeline: {type(frames)}")


def save_video(video_tensor: torch.Tensor, path: Path, fps: int) -> None:
    from torchvision.io import write_video
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(path), video_tensor, fps=fps, video_codec="h264", options={"crf": "10", "preset": "medium"})


def psnr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """PSNR in dB between uint8 [T,H,W,C] tensors."""
    mse = (pred.float() - ref.float()).pow(2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim_proxy(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Lightweight SSIM proxy: mean structural correlation over 8x8 patches."""
    p = pred.float() / 255.0
    r = ref.float() / 255.0
    # flatten spatial to [N, C] where N = T*H*W
    p_flat = p.reshape(-1, p.shape[-1])
    r_flat = r.reshape(-1, r.shape[-1])
    mu_p = p_flat.mean(0)
    mu_r = r_flat.mean(0)
    sigma_p = ((p_flat - mu_p) ** 2).mean(0).clamp(min=0).sqrt()
    sigma_r = ((r_flat - mu_r) ** 2).mean(0).clamp(min=0).sqrt()
    cov = ((p_flat - mu_p) * (r_flat - mu_r)).mean(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_channels = (2 * mu_p * mu_r + c1) * (2 * cov + c2) / (
        (mu_p ** 2 + mu_r ** 2 + c1) * (sigma_p ** 2 + sigma_r ** 2 + c2)
    )
    return ssim_channels.mean().item()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from vllm_omni.diffusion.models.mgm_video.mmdit.asa import AsaConfig

    runs = [
        ("baseline", None),
        ("dense_probe", AsaConfig(enable=True, variant="dense_probe", use_gilbert=True, text_length=256)),
        ("asa_0p20", AsaConfig(
            enable=True, variant="asa",
            max_retain_ratio=0.20, min_retain_ratio=0.05,
            energy_threshold=0.95, use_gilbert=True,
            text_length=256,
        )),
    ]

    videos: dict[str, dict[str, torch.Tensor]] = {}
    baseline_videos: dict[str, torch.Tensor] = {}

    for run_name, asa_cfg in runs:
        print(f"\n{'=' * 60}")
        print(f"Run: {run_name}")
        print(f"{'=' * 60}")

        pipeline = build_pipeline(args.model, asa_cfg, args)
        per_prompt: dict[str, torch.Tensor] = {}

        for pid, prompt in PROMPTS:
            print(f"  Prompt: {pid}")
            video = run_one(pipeline, prompt, args)
            video_path = out / f"{run_name}_{pid}.mp4"
            save_video(video, video_path, args.fps)
            print(f"  Saved: {video_path}")
            per_prompt[pid] = video

        videos[run_name] = per_prompt
        if run_name == "baseline":
            baseline_videos = {pid: per_prompt[pid] for pid, _ in PROMPTS}

        del pipeline
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except Exception:
            torch.cuda.empty_cache()

    # Compute metrics vs baseline
    table: dict[str, dict[str, dict[str, float]]] = {}
    for run_name in ("dense_probe", "asa_0p20"):
        table[run_name] = {}
        for pid, _ in PROMPTS:
            v_ref = baseline_videos[pid]
            v_test = videos[run_name][pid]
            table[run_name][pid] = {
                "psnr": psnr(v_test, v_ref),
                "ssim_proxy": ssim_proxy(v_test, v_ref),
            }

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(table, indent=2))
    print(f"\nMetrics written to {metrics_path}")
    print(json.dumps(table, indent=2))

    # P3 acceptance gate
    gate_ok = True
    for pid, _ in PROMPTS:
        p_val = table["dense_probe"][pid]["psnr"]
        if p_val < 35.0:
            print(f"WARN: dense_probe PSNR below 35 dB for {pid}: {p_val:.2f} dB")
            gate_ok = False
    if gate_ok:
        print("\nP3 gate PASSED: dense_probe PSNR >= 35 dB on all prompts.")
    else:
        print("\nP3 gate FAILED: see WARN lines above.")


if __name__ == "__main__":
    main()
