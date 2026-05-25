# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Consolidated utility functions for the MGM-Video VAE.

This module gathers small helper functions and classes that were originally
spread across multiple files in the MGM-Video-Ascend repository
(mimogpt).  They have been made self-contained here so that the VAE
code has **no** dependency on the original mimogpt package.

Sources (original locations in MGM-Video-Ascend):
  - mimogpt/models/modules/ldm/utils.py
  - mimogpt/models/modules/ldm/distributions/distributions.py
  - mimogpt/utils/log_utils.py            (Registry only)
  - mimogpt/utils/txt_utils.py            (merge_args, read_from_yaml)
  - mimogpt/models/dit/communications.py
  - mimogpt/models/modules/ldm/diffusion/model.py
      (time2batch, batch2time, get_timestep_embedding,
       nonlinearity, Normalize)
"""

import importlib
import math
from inspect import isfunction
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from easydict import EasyDict
from einops import rearrange


# ---------------------------------------------------------------------------
# From mimogpt/models/modules/ldm/utils.py
# ---------------------------------------------------------------------------

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ---------------------------------------------------------------------------
# From mimogpt/models/modules/ldm/distributions/distributions.py
# ---------------------------------------------------------------------------

class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


# ---------------------------------------------------------------------------
# From mimogpt/utils/txt_utils.py  (merge_args, read_from_yaml only)
# ---------------------------------------------------------------------------

def merge_args(args, cfg):
    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    def _register(self, obj, name=None):
        if name is None:
            name = obj.__name__

        assert (
                name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        if obj is not None:
            self._register(obj=obj, name=name)
            return obj

        def _register_cls(cls):
            self._register(obj=cls, name=name)
            return cls
        return _register_cls

def merge_args(args, cfg):
    for key, val in args.__dict__.items():
        setattr(cfg, key, val)
    return cfg


def read_from_yaml(txt_path):
    with open(txt_path, 'r') as fd:
        cont = fd.read()
        try:
            y = yaml.safe_load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.safe_load(cont)
        return EasyDict(y)


# ---------------------------------------------------------------------------
# From mimogpt/models/dit/communications.py
# ---------------------------------------------------------------------------

# ====================
# All-To-All
# ====================

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


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


# ====================
# Gather-Split
# ====================

def _split(input_, pg: dist.ProcessGroup, dim=-1):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    assert input_.device.type == "cuda" or input_.device.type == "npu"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim):
        ctx.mode = process_group
        ctx.dim = dim
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):

        return _split(grad_output, ctx.mode, ctx.dim), None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim):
        ctx.mode = process_group
        ctx.dim = dim
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


def split_forward_gather_backward(input_, process_group, dim):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim)


def gather_forward_split_backward(input_, process_group, dim):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim)


# ---------------------------------------------------------------------------
# From mimogpt/models/modules/ldm/diffusion/model.py
#   (time2batch, batch2time, get_timestep_embedding,
#    nonlinearity, Normalize only)
# ---------------------------------------------------------------------------

def time2batch(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    batch_size = x.shape[0]
    return rearrange(x, "b c t h w -> (b t) c h w"), batch_size


def batch2time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return torch.nn.functional.silu(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)