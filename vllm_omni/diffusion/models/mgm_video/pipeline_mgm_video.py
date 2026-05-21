# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MGM-Video (MimoGPT) T2V Pipeline for vllm-omni.

This pipeline adapts the MGM-Video-Ascend T2V generation pipeline to work
within the vllm-omni DiffusionEngine framework. It follows the same pattern
as Wan22Pipeline but uses the MMDiT architecture and RectifiedFlow scheduler.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.mgm_video.original_distributed_vae import OriginalDistributedVAE
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.mgm_video.mmdit_transformer import MMDiTInference, mmdit_xl_2_inference
from vllm_omni.diffusion.models.mgm_video.t5_text_encoder import T5TextEncoder
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)

# Default frame scale/bias for the 11B 720p model (16 channels)
DEFAULT_FRAME_SCALE = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]
DEFAULT_FRAME_BIAS = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/config.json")
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict,
    cp_size: int = 1,
    cache_algo_cfg: dict | None = None,
) -> MMDiTInference:
    """Create MMDiTInference from config dict.
    """
    kwargs = {}

    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "hidden_size" in config:
        kwargs["hidden_size"] = config["hidden_size"]
    if "depth" in config:
        kwargs["depth"] = config["depth"]
    if "head_dim" in config:
        kwargs["head_dim"] = config["head_dim"]
    if "patch_size" in config:
        ps = config["patch_size"]
        if isinstance(ps, (list, tuple)):
            kwargs["patch_size"] = ps[1] if len(ps) > 1 else ps[0]
        else:
            kwargs["patch_size"] = ps
    if "caption_channels" in config:
        kwargs["caption_channels"] = config["caption_channels"]
    if "model_max_length" in config:
        kwargs["model_max_length"] = config["model_max_length"]
    if "pred_sigma" in config:
        kwargs["pred_sigma"] = config["pred_sigma"]
    if "rope_ratio" in config:
        kwargs["rope_ratio"] = config["rope_ratio"]
    if "use_3d_rope" in config:
        kwargs["use_3d_rope"] = config["use_3d_rope"]

    kwargs["use_context_parallelism"] = cp_size > 1
    kwargs["skip_initialize_weights"] = True
    kwargs["cache_algo_cfg"] = cache_algo_cfg

    return mmdit_xl_2_inference(**kwargs)


def get_mgm_video_post_process_func(od_config: OmniDiffusionConfig):
    """Post-process function for MGM-Video.

    forward() 中已在 NPU 上完成的后处理：
      clamp → normalize → uint8 → permute → CPU

    本函数通过 dtype 检测状态：
      - uint8：NPU 后处理已完成，直接转 numpy
      - float32：兜底路径，执行完整 CPU 后处理
    """

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video

        # forward() 已在 NPU 上完成后处理（dtype=uint8, 已 permute+CPU）
        if video.dtype == torch.uint8:
            return video.numpy()

        # 兜底：原始 VAE 输出，执行完整后处理
        low, high = -1.0, 1.0
        video = torch.clamp(video, min=low, max=high)
        video = video.sub(low).div(max(high - low, 1e-5))
        video = video.mul(255).add(0.5).clamp(0, 255)
        video = video.permute(0, 2, 3, 4, 1)
        video = video.to("cpu", torch.uint8).numpy()
        return video

    return post_process_func


def get_mgm_video_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process function for MGM-Video (no-op for T2V)."""

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return request

    return pre_process_func


class MGMVideoPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    """MGM-Video (MimoGPT) T2V Pipeline for vllm-omni.

    This pipeline generates video from text prompts using the MMDiT architecture
    with RectifiedFlow scheduling.

    Default configuration (11B model):
    - Resolution: 720x1280, 121 frames
    - Steps: 9, t_shift=12, cfg_scale=1.0
    - VAE: AutoencoderKL3D (8x8x8 compression)
    - Text encoder: T5-v1.1-XXL (d_model=4096, max_length=400)
    """

    # Reference to offload backend (set by DiffusionModelRunner after enable())
    offload_backend = None

    # Dummy run at a reduced resolution to avoid OOM during engine init.
    # The dummy run only validates that the transformer forward pass works;
    # it does not need to match the actual inference resolution.
    # VAE decode at high resolutions (720x1280) OOMs on 61 GiB NPUs because
    # the VAE runs in float32 and is not HSDP-sharded. Use 256x256 which
    # is safe for both transformer + VAE on 2 NPUs with HSDP.
    dummy_run_height = 720
    dummy_run_width = 1280

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        import torch
        import os
        torch.use_deterministic_algorithms(True)
        os.environ["HCCL_DETERMINISTIC"] = "True"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Read model_index.json for configuration
        self.frame_scale = DEFAULT_FRAME_SCALE
        self.frame_bias = DEFAULT_FRAME_BIAS
        self.use_framescale = True
        self.decode_pad = 3

        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.frame_scale = model_index.get("frame_scale", DEFAULT_FRAME_SCALE)
                    self.frame_bias = model_index.get("frame_bias", DEFAULT_FRAME_BIAS)
                    self.use_framescale = model_index.get("use_framescale", True)
                    self.decode_pad = model_index.get("decode_pad", 3)

        # Weights sources for the transformer
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Initialize T5 text encoder (lazy device movement in HSDP/offload mode)
        _keep_on_cpu = (
            od_config.parallel_config.use_hsdp
            or od_config.parallel_config.ulysses_degree > 1
            or getattr(od_config, "enable_layerwise_offload", False)
        ) if od_config else False
        self.t5_encoder = T5TextEncoder(
            model_path=model,
            device=self.device,
            dtype=torch.bfloat16,
            local_files_only=local_files_only,
            keep_on_cpu=_keep_on_cpu,
        )

        # Load VAE (using original repo's distributed VAE logic)
        self.vae = OriginalDistributedVAE.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32
        )
        if not _keep_on_cpu:
            self.vae = self.vae.to(self.device)

        # Create transformer from config (weights loaded later via load_weights)
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        if not transformer_config:
            # Default 11B config
            transformer_config = {
                "in_channels": 16,
                "out_channels": 16,
                "hidden_size": 3072,
                "depth": 42,
                "num_heads": 24,
                "head_dim": 128,
                "patch_size": [1, 2, 2],
                "caption_channels": 4096,
                "model_max_length": 400,
                "pred_sigma": False,
                "rope_ratio": [22 / 64, 22 / 64, 20 / 64],
                "use_3d_rope": True,
            }

        # Detect CP (Context Parallelism) from world_size when layerwise offload is enabled
        # CP is used when multiple NPUs are available but we don't want HSDP/SP
        # (e.g., for MGM-Video with layerwise offload on 2 NPUs)
        cp_size = 1
        if getattr(od_config, "enable_layerwise_offload", False) and not od_config.parallel_config.use_hsdp:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        cp_size = world_size
                        # Initialize CP group using original repo's parallel_states
                        from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_parallel_states import (
                            initialize_distributed,
                        )
                        initialize_distributed(context_parallel_size=cp_size)
                        logger.info("CP enabled: cp_size=%d (initialized via mmdit_parallel_states)", cp_size)
            except Exception:
                logger.warning("CP requested but DIT group not initialized, falling back to cp_size=1")
                cp_size = 1

        # TDM cache: skip compute on certain blocks/timesteps to speed up inference.
        # Matches the original MGM-Video-Ascend inference_algo.cache config.
        cache_algo_cfg = None
        cache_scheme_path = os.path.join(
            os.path.dirname(__file__), "cache_scheme",
            "cache_scheme_tdm_8step_dit_per_12_5_v1_speedup.txt",
        )
        if os.path.isfile(cache_scheme_path):
            cache_algo_cfg = {"enable": True, "scheme": cache_scheme_path}
            logger.info("TDM cache enabled: scheme=%s", cache_scheme_path)
        else:
            logger.warning("TDM cache scheme not found at %s, running without cache", cache_scheme_path)

        self.transformer = create_transformer_from_config(
            transformer_config, cp_size=cp_size, cache_algo_cfg=cache_algo_cfg,
        )
        self.transformer_config = {
            "in_channels": self.transformer.in_channels,
            "out_channels": self.transformer.out_channels,
            "patch_size": self.transformer.patch_size,
        }

        # Initialize RectifiedFlow scheduler
        t_shift = od_config.flow_shift if od_config.flow_shift is not None else 12.0
        self.scheduler = RectifiedFlowScheduler(
            num_train_timesteps=1000,
            time_shifting_factor=t_shift,
            prediction_type="velocity",
        )

        # VAE scale factors
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

        # Store offload config for use in load_weights (enable_dit_inference_offload
        # must be called after weights are loaded)
        self._enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
        self._num_inference_steps = 9  # default, will be set per-request

        # Signal to vllm-omni's LayerWiseOffloadBackend that this pipeline
        # manages its own offloading via enable_dit_inference_offload().
        # Setting skip_layerwise_offload=True prevents the backend from
        # installing conflicting hooks.
        self.skip_layerwise_offload = self._enable_layerwise_offload

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ):
        """Prepare random noise latents.

        For MGM-Video with 8x8x8 VAE:
            num_latent_frames = (num_frames - 1) // 8 + 1
            latent_height = height // 8
            latent_width = width // 8
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # latents = torch.randn(shape, device=device, dtype=dtype)
        return latents

    def predict_noise(self, current_model: nn.Module | None = None, **kwargs) -> torch.Tensor:
        """Forward pass through transformer to predict noise/velocity.

        Calls MMDiTInference.forward() with the original repo's signature:
            forward(x, timestep, y, y_mask=y_mask, cur_time_index=step_idx)
        """
        if current_model is None:
            current_model = self.transformer
        # Extract positional args from kwargs for the original repo's signature
        x = kwargs.pop("hidden_states")
        timestep = kwargs.pop("timestep")
        y = kwargs.pop("encoder_hidden_states")
        y_mask = kwargs.pop("y_mask", None)
        cur_time_index = kwargs.pop("cur_time_index", -1)
        # Remove any remaining kwargs that the original forward() doesn't accept
        kwargs.pop("attention_kwargs", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("current_model", None)

        return current_model(x=x, timestep=timestep, y=y, y_mask=y_mask,
                             cur_time_index=cur_time_index, **kwargs)

    def _offload_transformer(self) -> bool:
        """Release transformer NPU memory after DiT inference.

        The original MGM-Video repo uses enable_dit_inference_offload()
        with scheduler=1, which offloads all weights to CPU at the last
        denoising step. We just need to empty_cache() here since the
        offload hooks handle the weight movement.

        Returns:
            True if offload was performed, False otherwise.
        """
        # Release freed memory back to the allocator
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass

        return True

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 9,
        guidance_scale: float = 1.0,
        frame_num: int = 121,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """Generate video from text prompt.

        Args:
            req: The diffusion request
            prompt: Text prompt (overridden by req.prompts)
            negative_prompt: Negative prompt
            height: Video height (default 720)
            width: Video width (default 1280)
            num_inference_steps: Number of ODE evaluation points (default 9, producing 8 Euler steps)
            guidance_scale: CFG scale (default 1.0 = no CFG)
            frame_num: Number of frames (default 121)
            output_type: Output type ("np", "latent", etc.)
            generator: Random generator
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            attention_kwargs: Additional attention kwargs

        Returns:
            DiffusionOutput containing the generated video
        """
        # Parse request parameters
        if len(req.prompts) > 1:
            raise ValueError("MGM-Video only supports a single prompt per request.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = (
                None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for MGM-Video generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        # Respect per-request guidance_scale
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        # Ensure dimensions are compatible with VAE and patch size
        patch_size = self.transformer_config["patch_size"]  # (1, 2, 2)
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 8*2=16
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        # Ensure num_frames is compatible with VAE temporal compression
        if (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            num_frames = (num_frames - 1) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype if hasattr(self.transformer, 'dtype') else torch.bfloat16

        # Seed / generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)
        
        # Encode text
        # NOTE: Removed the pre-encoding DiT offload loop. In the
        # full_denoise schedule (scheduler=1), DiT blocks are already
        # offloaded to CPU at the end of the previous denoising loop,
        # so re-offloading here was redundant and wasted ~10 GiB of
        # PCIe bandwidth per request.
        encoder_time_start = time.time()

        y_mask = None
        if prompt_embeds is None:
            with torch.profiler.record_function("t5_encoder"):
                prompt_embeds, negative_prompt_embeds, y_mask = self.t5_encoder.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=guidance_scale > 1.0,
                    num_videos_per_prompt=1,
                    max_sequence_length=req.sampling_params.max_sequence_length or 400,
                    device=device,
                    dtype=dtype,
                )
        else:
            # prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            prompt_embeds = prompt_embeds.to(device=device)
            if negative_prompt_embeds is not None:
                # negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(device=device)
        encoder_time_end = time.time()
        print(f"encoder cost time={encoder_time_end - encoder_time_start}")

        # Set timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Initialize offload backend with num_sampling_steps for full_denoise schedule
        if hasattr(self, 'offload_backend') and self.offload_backend is not None:
            self.offload_backend.set_num_sampling_steps(len(timesteps))

        # Prepare latents
        num_channels_latents = self.transformer_config["in_channels"]
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.sampling_params.latents,
        )
        # # 打印 prepare_latents 输出的 latents 张量信息
        # print(f"vLLM latents shape      : {latents.shape}")
        # print(f"vLLM latents mean       : {latents.float().mean().item():.8f}")
        # print(f"vLLM latents min/max    : {latents.float().min().item():.8f} / {latents.float().max().item():.8f}")

        if attention_kwargs is None:
            attention_kwargs = {}

        # Denoising loop
        mmdit_time_start = time.time()
        with torch.profiler.record_function("denosing"):
            with self.progress_bar(total=len(timesteps)) as pbar:
                for step_idx, t in enumerate(timesteps):
                    self._current_timestep = t

                    # Propagate step index to transformer for offload schedule
                    self.transformer._current_step_idx = step_idx
                    # Also set on each block so hooks can see it before forward()
                    for blk in self.transformer.blocks:
                        blk._current_step_idx = step_idx

                    # latent_model_input = latents.to(dtype)
                    latent_model_input = latents
                    timestep = t.expand(latents.shape[0])
                    
                    # ====================== vLLM 排查打印（和原仓完全对齐） ======================
                    # print(f"\n==================== vLLM STEP {t.item():.4f} ====================")
                    # print(f"vLLM x shape      : {latent_model_input.shape}")
                    # print(f"vLLM x mean       : {latent_model_input.float().mean().item():.8f}")
                    # print(f"vLLM x min/max    : {latent_model_input.float().min().item():.8f} / {latent_model_input.float().max().item():.8f}")
                    # print(f"vLLM y shape      : {prompt_embeds.shape}")
                    # print(f"vLLM y mean       : {prompt_embeds.float().mean().item():.8f}")
                    # if y_mask is not None:
                    #     print(f"vLLM y_mask       : {y_mask.shape}, mean: {y_mask.float().mean().item():.4f}")
                    # ============================================================================

                    do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

                    # Prepare kwargs for positive prediction
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": prompt_embeds,
                        "y_mask": y_mask,
                        "cur_time_index": step_idx,
                    }

                    if do_true_cfg:
                        negative_kwargs = {
                            "hidden_states": latent_model_input,
                            "timestep": timestep,
                            "encoder_hidden_states": negative_prompt_embeds,
                            "y_mask": y_mask,
                            "cur_time_index": step_idx,
                        }
                    else:
                        negative_kwargs = None

                    # Predict noise with automatic CFG parallel handling
                    noise_pred = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_true_cfg,
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                    )
                    
                    
                    # ====================== vLLM noise_pred 打印 ======================
                    # print(f"vLLM noise_pred shape : {noise_pred.shape}")
                    # print(f"vLLM noise_pred mean  : {noise_pred.float().mean().item():.8f}")
                    # print(f"vLLM noise_pred min/max: {noise_pred.float().min().item():.8f} / {noise_pred.float().max().item():.8f}")
                    # print("="*70 + "\n")
                    # ==================================================================

                    # Scheduler step
                    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

                    pbar.update()
        mmdit_time_end = time.time()
        print(f"mmdit cost time={mmdit_time_end - mmdit_time_start}")

        # Clear cache before VAE decode
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass
        self._current_timestep = None

        # Apply per-frame scale/bias normalization before VAE decode
        # frame_scale/frame_bias are per-temporal-position vectors (length = num_latent_frames).
        # Broadcast along [B, C, T, H, W] as [1, 1, T, 1, 1] — matching the original repo.
        if self.use_framescale:
            num_latent_frames = latents.shape[2]
            # Use float32 for frame_scale/frame_bias (matching original repo's
            # torch.tensor(cfg.vae.frame_scale, dtype=torch.float32)) to avoid
            # precision loss from bfloat16 division on scale values ~0.58-0.70.
            frame_scale = torch.tensor(self.frame_scale, device=latents.device, dtype=torch.float32)[:num_latent_frames]
            frame_bias = torch.tensor(self.frame_bias, device=latents.device, dtype=torch.float32)[:num_latent_frames]
            # Compute in float32 for precision, then cast back
            latents_fp32 = latents.float()
            latents_fp32 = latents_fp32 / frame_scale[None, None, :, None, None] + frame_bias[None, None, :, None, None]
            latents = latents_fp32.to(latents.dtype)


        # ====================== 对比打印 ======================
        # print("========== vLLM FINAL LATENT ==========")
        # print("shape:", latents.shape)
        # print("mean:", latents.mean().item())
        # print("min:", latents.min().item())
        # print("max:", latents.max().item())

        # ====================== Cross-repo VAE decode debug ======================
        # If /tmp/mgm_original_latent.pt exists, load it and replace latents
        # before VAE decode. This isolates VAE decode differences from DiT
        # inference differences. The saved tensor is the pre-frame_scale latent
        # from the original MGM-Video-Ascend repo.
        # _latent_override_path = "/tmp/mgm_original_latent.pt"
        # if os.path.exists(_latent_override_path):
        #     override_latent = torch.load(_latent_override_path, map_location=latents.device, weights_only=True)
        #     # The original saves with shape [1, 16, T, H, W] in bfloat16
        #     override_latent = override_latent.to(latents.dtype)
        #     print(f"🔥 LOADED original latent from {_latent_override_path}: "
        #           f"shape={list(override_latent.shape)}, dtype={override_latent.dtype}, "
        #           f"mean={override_latent.float().mean().item():.6f}, "
        #           f"min={override_latent.min().item():.6f}, max={override_latent.max().item():.6f}")
        #     print(f"🔥 REPLACING vllm-omni latent (mean={latents.float().mean().item():.6f}) "
        #           f"with original latent (mean={override_latent.float().mean().item():.6f})")
        #     latents = override_latent

        # VAE decode
        if output_type == "latent":
            output = latents
        else:
            # Release transformer NPU memory before VAE decode.
            # The original repo's enable_dit_inference_offload() with
            # scheduler=1 offloads weights to CPU at the last denoising step.
            # Just empty_cache() to release any remaining device memory.
            self._offload_transformer()

            # Lazy move VAE to device (kept on CPU during init to save memory,
            # but stays on device after first move — matching the original repo
            # where VAE is never offloaded back to CPU).
            if self.vae.device.type == "cpu":
                self.vae = self.vae.to(self.device)
            latents = latents.to(self.vae.dtype)
            vae_time_start = time.time()
            with torch.profiler.record_function("vae_decode"):
                output = self.vae.decode(latents, return_dict=False, num_frames=num_frames)[0]
            vae_time_end = time.time()
            print(f"vae decode cost time={vae_time_end-vae_time_start}s")

            # Post-process on GPU before returning: convert float32 [-1,1]
            # to uint8 [0,255] and permute [B,C,T,H,W] -> [B,T,H,W,C].
            # Doing this on GPU is ~20x faster than CPU (NPU parallel vs
            # serial memory traversal), and reduces the data transferred
            # through IPC from 1.27 GB (float32) to 334 MB (uint8) for
            # 121-frame 720p video.
            #
            # get_mgm_video_post_process_func() detects the pre-processed
            # uint8 tensor (via dtype check) and skips redundant ops.
            output = torch.clamp(output, min=-1.0, max=1.0)
            output = output.sub(-1.0).div(2.0)  # [-1,1] -> [0,1]
            output = output.mul(255).add(0.5).clamp(0, 255)
            output = output.permute(0, 2, 3, 4, 1)  # [B,C,T,H,W] -> [B,T,H,W,C]
            output = output.to(torch.uint8).cpu()

        return DiffusionOutput(output=output)

    def load_weights(self, weights):
        """Load weights using the vLLM AutoWeightsLoader.

        We only load transformer weights (prefixed with "transformer.").
        The text_encoder and vae are already loaded in __init__.

        Key remapping: old checkpoints used fc1/fc2 for MLP layers
        (from setup_weights.py conversion), but the original repo uses
        dense_h_to_4h/dense_4h_to_h. We remap checkpoint keys to match
        the model's parameter names.
        """
        # Remap weight keys: fc1 -> dense_h_to_4h, fc2 -> dense_4h_to_h
        remapped_weights = []
        for name, param in weights:
            new_name = name
            new_name = new_name.replace("mlp_x.fc1.", "mlp_x.dense_h_to_4h.")
            new_name = new_name.replace("mlp_x.fc2.", "mlp_x.dense_4h_to_h.")
            new_name = new_name.replace("mlp_y.fc1.", "mlp_y.dense_h_to_4h.")
            new_name = new_name.replace("mlp_y.fc2.", "mlp_y.dense_4h_to_h.")
            remapped_weights.append((new_name, param))

        loader = AutoWeightsLoader(self, skip_prefixes=["text_encoder.", "vae."])
        result = loader.load_weights(remapped_weights)

        # After weights are loaded, enable DiT inference offload (original repo
        # approach: scheduler=1 loads all weights on first timestep, offloads on
        # last). We skip vllm-omni's LayerWiseOffloadBackend for this pipeline
        # since enable_dit_inference_offload manages CPU<->device transfers
        # directly via forward hooks, matching the original MGM-Video-Ascend repo.
        if getattr(self, "_enable_layerwise_offload", False):
            num_steps = getattr(self, "_num_inference_steps", 9)
            if hasattr(self.transformer, "enable_dit_inference_offload"):
                # The original repo loads the full model to device first, then
                # enable_dit_inference_offload copies blocks 1..N weights to
                # pinned CPU memory and resizes their device storage to 0.
                # Block 0 stays on device. Since load_device="cpu" when
                # layerwise offload is enabled, we must move the entire
                # transformer to device before calling enable_dit_inference_offload.
                logger.info("Moving transformer to %s before enable_dit_inference_offload", self.device)
                self.transformer.to(self.device)

                logger.info("Enabling DiT inference offload (scheduler=1, num_steps=%d)", num_steps)
                self.transformer.enable_dit_inference_offload(
                    offload_scheduler=1, num_sampling_steps=num_steps
                )

        # VAE decode warm-up: run a tiny decode to pre-allocate NPU memory
        # for VAE intermediate tensors. Without this, the first real request
        # suffers ~30s cold-start latency because the NPU memory pool has
        # never allocated space for VAE feature maps and must request new
        # segments from the OS (very slow on NPU). The original repo avoids
        # this because VAE is moved to device at init time and the memory
        # pool is already warm.
        # self._warmup_vae_decode()

        return result

    def _warmup_vae_decode(self) -> None:
        """Warm-up VAE decode to pre-allocate NPU memory for intermediate tensors.

        The first VAE decode call is ~30s slower than subsequent calls because
        the NPU memory pool must allocate new segments from the OS for VAE
        feature maps. By running a decode at init time, we ensure the memory
        pool is warm before the first real request arrives.
        """
        try:
            # Move VAE to device if still on CPU (lazy init)
            if self.vae.device.type == "cpu":
                self.vae = self.vae.to(self.device)

            # Use a latent with the same spatial resolution as real requests
            # to ensure the NPU memory pool allocates segments large enough
            # for actual VAE feature maps. Shape: [1, 16, 4, H/8, W/8].
            # 4 latent frames = 32 output frames (patch_size=8), enough to
            # exercise all VAE decoder layers including Conv3D ops.
            warmup_latent = torch.randn(
                1, 16, 4, 90, 160,
                device=self.device,
                dtype=self.vae.dtype,
            )
            logger.info("Running VAE decode warm-up to pre-allocate NPU memory...")
            with torch.no_grad():
                _ = self.vae.decode(warmup_latent, return_dict=False, num_frames=32)
            # Release warm-up tensors. Do NOT call empty_cache() here —
            # we want the NPU memory pool to retain the expanded segments
            # so the first real request can reuse them without OS-level
            # allocation. empty_cache() would shrink segments back and
            # defeat the purpose of the warm-up.
            del warmup_latent, _
            logger.info("VAE decode warm-up complete.")
        except Exception as e:
            logger.warning("VAE decode warm-up failed (non-fatal): %s", e)


class RectifiedFlowScheduler:
    """Rectified Flow scheduler with time-shifting support.

    Implements the Linear interpolation path:
        x_t = t * x_1 + (1 - t) * x_0
    where x_0 is noise and x_1 is data.

    The model predicts velocity: v = x_1 - x_0

    Time shifting concentrates sampling steps at low-noise region:
        t_shifted = t / (t + s - s * t)  where s = time_shifting_factor

    Integration direction: forward from noise (t~0) to data (t~1),
    matching the original MGM-Video-Ascend ode solver.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        time_shifting_factor: float = 12.0,
        prediction_type: str = "velocity",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.time_shifting_factor = time_shifting_factor
        self.prediction_type = prediction_type

        self.timesteps = None
        self._step_index = None

        # Config attributes expected by some downstream code
        self.config = type("Config", (), {
            "num_train_timesteps": num_train_timesteps,
            "prediction_type": prediction_type,
        })()

    def set_timesteps(self, num_inference_steps: int, device: torch.device | None = None):
        """Generate the timestep sequence with time shifting.

        Matches the original MGM-Video-Ascend repo's odeint-based solver:
            t = linspace(0, 1, num_steps)  -- num_steps evaluation points
            t_shifted = t / (t + s - s*t)

        The original odeint with Euler method uses num_steps evaluation points
        and performs num_steps-1 Euler integration steps between them. The model
        is called at t[0]..t[num_steps-2] (not at t[num_steps-1]=1.0).

        We generate num_steps points internally but only expose the first
        num_steps-1 as the iteration schedule. The last point (t=1.0) is
        used by step() to compute dt for the final Euler step.
        """
        self.num_inference_steps = num_inference_steps

        # Generate num_steps evaluation points (matching odeint's t span),
        # but only iterate over the first num_steps-1 (model evaluation points).
        # The last point (t=1.0) is the ODE endpoint, not a model call point.
        t = torch.linspace(0, 1, num_inference_steps, device=device)

        # Apply time shifting: t_shifted = t / (t + s - s*t)
        if self.time_shifting_factor > 0:
            s = self.time_shifting_factor
            t = t / (t + s - s * t)

        # Store all points for dt computation in step(), but only expose
        # the first num_steps-1 as the iteration schedule.
        self._all_timesteps = t.to(device=device)
        self.timesteps = self._all_timesteps[:num_inference_steps - 1]

        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        **kwargs,
    ) -> SchedulerOutput:
        """Perform a single Euler step.

        For velocity prediction with Linear path:
            dx/dt = v(x,t) = model_output
            x_{t+dt} = x_t + dt * v(x_t, t)

        Since we integrate forward (t increasing), dt > 0.
        """
        step_index = self._get_step_index(timestep)

        # Get current and next sigma values from the full timestep grid
        # (including the endpoint t=1.0 that is not in the iteration schedule).
        all_ts = getattr(self, '_all_timesteps', self.timesteps)
        sigma = all_ts[step_index]
        if step_index + 1 < len(all_ts):
            sigma_next = all_ts[step_index + 1]
        else:
            # Should not happen with correct set_timesteps, but safety fallback
            sigma_next = sigma

        # Euler step: x_{t+dt} = x_t + (sigma_next - sigma) * model_output
        # For velocity prediction, model_output is v = dx/dt
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output

        return SchedulerOutput(prev_sample=prev_sample)

    def _get_step_index(self, timestep: torch.Tensor) -> int:
        """Find the index of the given timestep in self.timesteps."""
        if self._step_index is not None:
            return self._step_index

        # Find closest timestep
        if isinstance(timestep, torch.Tensor):
            t = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t = float(timestep)

        diffs = (self.timesteps - t).abs()
        step_index = diffs.argmin().item()
        return step_index

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples according to the Linear interpolation path.

        x_t = t * x_1 + (1 - t) * x_0
        where x_1 = original_samples and x_0 = noise.
        """
        # Apply time shifting to timesteps
        s = self.time_shifting_factor
        if s > 0:
            t = timesteps / (timesteps + s - s * timesteps)
        else:
            t = timesteps

        t = t.flatten()
        while t.dim() < original_samples.dim():
            t = t.unsqueeze(-1)

        return t * original_samples + (1 - t) * noise

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """No scaling needed for rectified flow."""
        return sample