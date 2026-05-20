# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MGM-Video VAE (MotionVAE) — self-contained, no dependency on the original
MGM-Video-Ascend repo.

This package provides:
- ``DistributedVAE``: the distributed VAE decoder that handles model
  construction, weight loading, convert_to_distributed, and H-split
  distributed decode — matching the original repo's behavior.
- ``AutoencoderKL3D``: the underlying 3D VAE model class.
"""

import os

# ---------------------------------------------------------------------------
# NPU environment setup (must happen before any torch_npu-dependent import)
# ---------------------------------------------------------------------------
os.environ["COMBINED_ENABLE"] = "1"
os.environ["INF_NAN_MODE_ENABLE"] = "1"
os.environ.setdefault("ACL_OP_SELECT_IMPL_MODE", "high_precision")

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

try:
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if hasattr(torch_npu.npu.config, "allow_internal_format"):
        torch_npu.npu.config.allow_internal_format = False
except (ImportError, AttributeError):
    pass

# Set device_type=npu so the VAE attention uses forward_npu
# (cross-rank all_gather of KV for distributed attention).
os.environ.setdefault("device_type", "npu")

from vllm_omni.diffusion.models.mgm_video.vae.vae_distributed import DistributedVAE
from vllm_omni.diffusion.models.mgm_video.vae.vae_model import AutoencoderKL3D

__all__ = ["DistributedVAE", "AutoencoderKL3D"]