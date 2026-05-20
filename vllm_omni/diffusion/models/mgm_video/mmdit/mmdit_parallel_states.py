# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Context parallel group management for MMDiT inference.

Ported from MGM-Video-Ascend mimogpt/models/dit/parallel_states.py.
Creates a dedicated CP group via dist.new_group() — does NOT reuse
vllm-omni's get_dit_group().
"""

import torch.distributed as dist

_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("data", None)


def set_context_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_context_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def get_data_parallel_ranks():
    return _GLOBAL_PARALLEL_GROUPS.get("data_ranks", None)


def set_data_parallel_ranks(ranks):
    _GLOBAL_PARALLEL_GROUPS["data_ranks"] = ranks


def initialize_distributed(context_parallel_size, tensor_parallel_size=1):
    """Initialize context parallel groups.

    Must be called after dist.is_initialized(). Creates CP groups via
    dist.new_group() matching the original MGM-Video-Ascend logic.
    """
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert tensor_parallel_size == 1
    model_parallel_size = context_parallel_size * tensor_parallel_size

    # Build the sequence parallel groups
    if context_parallel_size > 1:
        num_sequence_parallel_groups: int = world_size // context_parallel_size
        for i in range(num_sequence_parallel_groups):
            ranks = range(i * context_parallel_size,
                          (i + 1) * context_parallel_size)
            group = dist.new_group(ranks)
            if rank in ranks:
                set_context_parallel_group(group)