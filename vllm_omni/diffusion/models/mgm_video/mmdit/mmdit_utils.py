# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared utilities for the MMDiT module.

Consolidates tuple helpers, cache I/O, parallel-group management,
and distributed communication primitives that were previously spread
across mmdit_functional, mmdit_cache_utils, mmdit_parallel_states,
and mmdit_communications.
"""

import numpy as np
import torch
import torch.distributed as dist


# ── Section 1: Tuple helpers (from mmdit_functional.py) ──────────


def _ntuple(n):
    def parse(x):
        if isinstance(x, (list, tuple)):
            return x
        return (x,) * n

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


# ── Section 2: Cache scheme I/O (from mmdit_cache_utils.py) ─────


def read_2d_array_from_file_int(file_path):
    """Read a 2D array of integers from a text file.

    Each line should contain exactly 8 space-separated integers.
    Used by MMDiTInference to load the cache scheme that determines
    which blocks to compute vs skip at each denoising timestep.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    numbers = [int(num) for num in line.split()]
                    if len(numbers) == 8:
                        data.append(numbers)
                    else:
                        print(f"Warning: Line '{line}' does not contain exactly 8 numbers.")
                except ValueError:
                    print(f"Warning: Could not convert line '{line}' to numbers.")
    return np.array(data)


# ── Section 3: Parallel group management (from mmdit_parallel_states.py) ─


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

    if context_parallel_size > 1:
        num_sequence_parallel_groups: int = world_size // context_parallel_size
        for i in range(num_sequence_parallel_groups):
            ranks = range(i * context_parallel_size,
                          (i + 1) * context_parallel_size)
            group = dist.new_group(ranks)
            if rank in ranks:
                set_context_parallel_group(group)


# ── Section 4: Communication primitives (from mmdit_communications.py) ──


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _all_to_all(input_, dist.get_world_size(process_group), process_group, scatter_dim, gather_dim)


def _split(input_, pg: dist.ProcessGroup, dim=-1):
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        return input_

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    assert input_.device.type in ("cuda", "npu")
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def split_forward_gather_backward(input_, process_group, dim):
    return _split(input_, process_group, dim)


def gather_forward_split_backward(input_, process_group, dim):
    return _gather(input_, process_group, dim)
