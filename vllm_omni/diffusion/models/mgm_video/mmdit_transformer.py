# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin entrypoint for MMDiT transformer.

Re-exports MMDiTInference so that pipeline_mgm_video.py only imports
from this module, not directly from the mmdit/ subdirectory.
"""

from .mmdit.mmdit_inference import MMDiTInference, mmdit_xl_2_inference

__all__ = ["MMDiTInference", "mmdit_xl_2_inference"]
