# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Distributed VAE wrapper that instantiates the MGM-Video DistributedVAE
from the local vae/ package (no dependency on the original MGM-Video-Ascend repo).
"""

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

logger = init_logger(__name__)


class OriginalDistributedVAE(nn.Module):
    """Thin wrapper around the MGM-Video DistributedVAE class.

    The underlying DistributedVAE handles everything:
    - model construction from YAML config
    - weight loading
    - convert_to_distributed (DistributedConv3D, SyncGroupNorm, etc.)
    - bf16 conversion
    - decode with pad_split_frames + model.decode(is_distributed=True) + gather_video
    """

    scale_factor_temporal = 8
    scale_factor_spatial = 8

    def __init__(self, dist_vae):
        nn.Module.__init__(self)
        self._dist_vae = dist_vae
        self._dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_path: str, subfolder: str = "vae",
                        torch_dtype=torch.float32, **kwargs):
        """Load VAE by instantiating the DistributedVAE from the local vae/ package."""
        from easydict import EasyDict
        from vllm_omni.diffusion.models.mgm_video.vae.vae_distributed import DistributedVAE

        # Read model_index.json for decode_pad
        decode_pad = 3
        model_index_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(model_index_path):
            import json
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
            decode_pad = model_index.get("decode_pad", 3)

        # Find VAE checkpoint
        vae_dir = os.path.join(model_path, subfolder)
        ckpt_file = None
        for fname in ["causal_vae_v3.1_sd3.pth", "vae.pth", "model.pth"]:
            candidate = os.path.join(vae_dir, fname)
            if os.path.exists(candidate):
                ckpt_file = candidate
                break
        if ckpt_file is None:
            import glob
            safetensor_files = glob.glob(os.path.join(vae_dir, "*.safetensors"))
            if safetensor_files:
                ckpt_file = safetensor_files[0]
        if ckpt_file is None:
            raise FileNotFoundError(f"No VAE checkpoint found in {vae_dir}")

        logger.info("Loading VAE via DistributedVAE from %s", ckpt_file)

        # Construct args exactly as motionvae_16ch_dist() does
        height = kwargs.get("height", 480)
        width = kwargs.get("width", 720)
        max_frame = kwargs.get("max_frame", 121)

        # Use the local decode.yml config
        vae_pkg_dir = os.path.dirname(
            os.path.abspath(
                __import__("vllm_omni.diffusion.models.mgm_video.vae", fromlist=["__init__"]).__file__
            )
        )
        config_path = os.path.join(vae_pkg_dir, "decode.yml")

        args = EasyDict({})
        args.model_name = "motion_vae"
        args.config_path = config_path
        args.ckpt_path = ckpt_file
        args.height = height
        args.width = width
        args.max_frame = max_frame
        args.infer_type = "op"
        args.task = "decode"
        args.patch_size = (8, 8, 8)
        args.decode_pad = decode_pad
        args.is_casual = kwargs.get("is_casual", False)

        # vae_conv_split_config and vae_gn_split_infer from original inference config
        args.vae_conv_split_config = {
            "enable_conv_split": 1,
            "split_nums_in": 2,
            "split_nums_out": 4,
            "split_type": "split_in_out_channel",
            "split_conv_names": [
                "decoder.align.conv",
                "decoder.align.dcnpack.conv_offset_mask",
                "decoder.align.fusion.0",
                "decoder.align.fusion.1",
                "decoder.up.0.block.0.conv1",
                "decoder.up.0.block.0.nin_shortcut",
                "decoder.up.0.upsample.conv",
                "decoder.up.1.upsample.conv",
                "decoder.up.1.upsample.conv",
                "decoder.up.2.upsample.conv",
                "decoder.up.2.upsample.conv",
                "decoder.up.3.attn.0.q",
                "decoder.up.3.attn.0.k",
                "decoder.up.3.attn.0.v",
                "decoder.up.3.attn.0.proj_out",
                "decoder.up.3.upsample.conv",
                "decoder.up.3.upsample.conv",
            ],
        }
        args.vae_gn_split_infer = True

        # Instantiate DistributedVAE — this does everything:
        # model construction, weight loading, convert_to_distributed, bf16, eval
        dist_vae = DistributedVAE(args)

        instance = cls(dist_vae)
        return instance

    def decode(self, z: torch.Tensor, return_dict: bool = True,
               num_frames: int = 121):
        """Decode latent z to video, delegating to DistributedVAE."""
        with torch.no_grad():
            result = self._dist_vae.decode(z)
        if return_dict:
            return DecoderOutput(sample=result)
        return (result,)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return next(self._dist_vae.model.parameters()).device

    @property
    def config(self):
        return SimpleNamespace(
            scale_factor_temporal=8,
            scale_factor_spatial=8,
            z_channels=16,
            embed_dim=16,
            double_z=True,
            in_channels=3,
            out_ch=3,
        )