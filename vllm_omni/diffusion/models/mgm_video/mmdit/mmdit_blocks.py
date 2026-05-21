# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# -*- coding: utf-8 -*-
import os
import math
import torch

import torch.nn as nn
from typing import Tuple
from torch.nn import functional as F
from torch.profiler import record_function
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, Attention as Attention_

# Stub: add_decomposed_rel_pos is only used by WindowAttention (not in inference path)
def add_decomposed_rel_pos(*args, **kwargs):
    raise NotImplementedError("add_decomposed_rel_pos not used in inference path")

from .mmdit_parallel_states import get_context_parallel_group
from .mmdit_communications import all_to_all, split_forward_gather_backward, gather_forward_split_backward

import importlib

from .mmdit_async_offload import async_save_on_cpu

try:
    '''ascend'''
    import torch_npu
except ImportError:
    print("training on gpu")


def is_npu_available():
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


import torch.distributed as dist

if is_npu_available():
    DEVICE_TYPE = 'npu'
else:
    DEVICE_TYPE = 'cuda'


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        assert isinstance(normalized_shape, int)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        weight = torch.ones(self.normalized_shape, dtype=torch.float32, device=x.device)
        bias = torch.zeros(self.normalized_shape, dtype=torch.float32, device=x.device)
        if DEVICE_TYPE == 'npu':
            with torch.cuda.amp.autocast(enabled=False):
                ret_x = torch.nn.functional.layer_norm(x, [self.normalized_shape], weight=weight, bias=bias, eps=self.eps)
            return ret_x
        else:
            with torch.cuda.amp.autocast(enabled=False):
                ret_x = torch.nn.functional.layer_norm(x.to(torch.float32), [self.normalized_shape], weight=weight, bias=bias, eps=self.eps)
            return ret_x.to(torch.bfloat16) 

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def trans_BNSD2BSH(tensor):
    tensor = torch.transpose(tensor, 1, 2)
    tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
    return tensor


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).to(device=DEVICE_TYPE) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float).to(device=DEVICE_TYPE), inv_freq).float()
    # sinusoid_inp = torch.cat((sinusoid_inp, sinusoid_inp), dim=-1)
    sinusoid_inp = torch.stack((sinusoid_inp, sinusoid_inp), dim=-1).flatten(-2)
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


# Copied from transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = sin.unsqueeze(0).unsqueeze(1)
    cos = cos.unsqueeze(0).unsqueeze(1)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


def apply_2drotary_pos(q, k, freqs_cis):
    sincos_h, sincos_w = freqs_cis
    sin_h, cos_h = torch.split(sincos_h, sincos_h.shape[-1] // 2, dim=-1)
    sin_w, cos_w = torch.split(sincos_w, sincos_w.shape[-1] // 2, dim=-1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)

    q1 = apply_rotary_pos_emb(q1, sin_h, cos_h)
    k1 = apply_rotary_pos_emb(k1, sin_h, cos_h)
    q2 = apply_rotary_pos_emb(q2, sin_w, cos_w)
    k2 = apply_rotary_pos_emb(k2, sin_w, cos_w)
    q = torch.concat([q1, q2], dim=-1)
    k = torch.concat([k1, k2], dim=-1)

    return q, k


def apply_3drotary_pos(q, k, freqs_cis, freqs_cis_ctx=None):
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = torch.split(sincos_h, sincos_h.shape[-1] // 2, dim=-1)
    sin_w, cos_w = torch.split(sincos_w, sincos_w.shape[-1] // 2, dim=-1)
    sin_t, cos_t = torch.split(sincos_t, sincos_t.shape[-1] // 2, dim=-1)
    sin_h, cos_h = sin_h.contiguous(), cos_h.contiguous()
    sin_w, cos_w = sin_w.contiguous(), cos_w.contiguous()
    sin_t, cos_t = sin_t.contiguous(), cos_t.contiguous()

    # q1_c, q2_c, q3_c = q.chunk(3, dim=-1)
    # k1_c, k2_c, k3_c = k.chunk(3, dim=-1)
    q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
    k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
    q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
    k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()

    if freqs_cis_ctx == None:
        q1 = apply_rotary_pos_emb(q1, sin_h, cos_h)
        k1 = apply_rotary_pos_emb(k1, sin_h, cos_h)
        q2 = apply_rotary_pos_emb(q2, sin_w, cos_w)
        k2 = apply_rotary_pos_emb(k2, sin_w, cos_w)
        q3 = apply_rotary_pos_emb(q3, sin_t, cos_t)
        k3 = apply_rotary_pos_emb(k3, sin_t, cos_t)
    else:
        sincos_h_ctx, sincos_w_ctx, sincos_t_ctx = freqs_cis_ctx
        sin_h_ctx, cos_h_ctx = torch.split(sincos_h_ctx, sincos_h_ctx.shape[-1] // 2, dim=-1)
        sin_w_ctx, cos_w_ctx = torch.split(sincos_w_ctx, sincos_w_ctx.shape[-1] // 2, dim=-1)
        sin_t_ctx, cos_t_ctx = torch.split(sincos_t_ctx, sincos_t_ctx.shape[-1] // 2, dim=-1)

        q1 = apply_rotary_pos_emb(q1, sin_h, cos_h)
        q2 = apply_rotary_pos_emb(q2, sin_w, cos_w)
        q3 = apply_rotary_pos_emb(q3, sin_t, cos_t)
        k1 = apply_rotary_pos_emb(k1, sin_h_ctx, cos_h_ctx)
        k2 = apply_rotary_pos_emb(k2, sin_w_ctx, cos_w_ctx)
        k3 = apply_rotary_pos_emb(k3, sin_t_ctx, cos_t_ctx)

    q = torch.concat([q1, q2, q3], dim=-1)
    # k = torch.concat([k1, k2, q3], dim=-1)
    k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


# def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
# inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).cuda().float() / dim))
# t = torch.arange(num_pos, dtype=torch.float, device=inv_freq.device)
# freqs = torch.outer(t, inv_freq)
# emb = torch.cat((freqs, freqs), dim=-1)

# return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

# def rotate_half(x):
# x1, x2 = x.chunk(2, dim=-1)
# return torch.cat((-x2, x1), dim=-1)

# def apply_rotary_pos_emb(q, k, freqs):
# cos,sin = freqs.unsqueeze(0).unsqueeze(1).chunk(2, dim=-1)
# cos = cos.contiguous()
# sin = sin.contiguous()
# outputq = q * cos + rotate_half(q) * sin
# outputk = k * cos + rotate_half(k) * sin
# return outputq.to(q.dtype), outputk.to(k.dtype)

# def apply_3drotary_pos(q, k, freqs_cis):
# freqs_h, freqs_w, freqs_t = freqs_cis
# q1, q2, q3 = q.chunk(3, dim=-1)
# k1, k2, k3 = k.chunk(3, dim=-1)
# q1, k1 = apply_rotary_pos_emb(q1.contiguous(), k1.contiguous(), freqs_h)
# q2, k2 = apply_rotary_pos_emb(q2.contiguous(), k2.contiguous(), freqs_w)
# q3, k3 = apply_rotary_pos_emb(q3.contiguous(), k3.contiguous(), freqs_t)
# q = torch.concat([q1, q2, q3], dim=-1)
# k = torch.concat([k1, k2, k3], dim=-1)
# return q,k

class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_linear = nn.Linear(n_embd, n_embd)
        self.kv_linear = nn.Linear(n_embd, 2 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, y, mask):
        B, T, C = x.size()
        _, L, _ = y.size()
        q = self.q_linear(x)
        k, v = self.kv_linear(y).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                                   dropout_p=self.dropout if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        x = out.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        x = self.proj_drop(self.proj(x))
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0., use_rmsnorm=False):
        super(MultiHeadCrossAttention, self).__init__()
        assert hidden_size % num_heads == 0, "d_model must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout)
        self.use_rmsnorm = use_rmsnorm

        if self.use_rmsnorm:
            self.norm1 = RMSNorm(self.head_dim)
            self.norm2 = RMSNorm(self.head_dim)
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x, y, mask):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        q = self.q_linear(x).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        kv = self.kv_linear(y).view(B, -1, 2, self.num_heads, C // self.num_heads).transpose(1, 3)
        k, v = kv.unbind(2)
        if self.use_rmsnorm:
            q = self.norm1(q)
            k = self.norm2(k)

        if self.flash:  # and self.training:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                                   dropout_p=self.dropout if self.training else 0)
            out = out.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            out = out.transpose(1, 2).contiguous().view(B, -1, C)  # re-assemble all head outputs side by side

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class WindowAttention(Attention_):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        dropout=0.,
        use_3d_rope=False,
        use_rmsnorm=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        head_dim = dim // num_heads
        self.use_rmsnorm = use_rmsnorm
        self.use_3d_rope = use_3d_rope
        if use_rmsnorm:
            self.norm1 = RMSNorm(head_dim)
            self.norm2 = RMSNorm(head_dim)

    def forward(self, x, mask=None, spatial_freq=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = qkv.unbind(2)

        if self.use_rmsnorm:
            q = self.norm1(q)
            k = self.norm2(k)

        if spatial_freq is not None:
            if self.use_3d_rope:
                q, k = apply_3drotary_pos(q, k, freqs_cis=spatial_freq)
            else:
                q, k = apply_2drotary_pos(q, k, freqs_cis=spatial_freq)

        if self.flash:  # and self.training:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                                   dropout_p=self.dropout if self.training else 0)
            x = out.transpose(1, 2).contiguous().view(B, -1, C)  # r
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            x = out.transpose(1, 2).contiguous().view(B, -1, C)  #

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################
class Attention(Attention_):
    def forward(self, x, mask=None, spatial_freq=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        if spatial_freq is not None:
            q, k = apply_2drotary_pos(q, k, freqs_cis=spatial_freq)

        use_fp32_attention = getattr(self, 'fp32_attention', False)
        if use_fp32_attention:
            q, k = q.float(), k.float()
        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        x = (att @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MaskFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=DEVICE_TYPE)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).to(device=DEVICE_TYPE) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU, token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size,
                          act_layer=act_layer, drop=0)
        # self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    # def token_drop(self, caption, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
    #     return caption

    def forward(self, caption, train, force_drop_ids=None):
        # if train:
        #     assert caption.shape[2:] == self.y_embedding.shape
        # use_dropout = self.uncond_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class CaptionEmbedderDoubleBr(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU, token_num=120):
        super().__init__()
        self.proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size,
                        act_layer=act_layer, drop=0)
        self.embedding = nn.Parameter(torch.randn(1, in_channels) / 10 ** 0.5)
        self.y_embedding = nn.Parameter(torch.randn(token_num, in_channels) / 10 ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, global_caption, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(global_caption.shape[0]).to(device=DEVICE_TYPE) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        global_caption = torch.where(drop_ids[:, None], self.embedding, global_caption)
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return global_caption, caption

    def forward(self, caption, train, force_drop_ids=None):
        assert caption.shape[2:] == self.y_embedding.shape
        global_caption = caption.mean(dim=2).squeeze()
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            global_caption, caption = self.token_drop(global_caption, caption, force_drop_ids)
        y_embed = self.proj(global_caption)
        return y_embed, caption


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(n_embd, 4 * n_embd)
        self.dense_4h_to_h = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.gelu(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x


###################### wangnanyang dit ####################
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, mask, spatial_freq=None):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if spatial_freq is not None:
            if True:  # self.use_3d_rope:
                q, k = apply_3drotary_pos(q, k, freqs_cis=spatial_freq)
            else:
                q, k = apply_2drotary_pos(q, k, freqs_cis=spatial_freq)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                                   dropout_p=self.dropout if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        x = out.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        x = self.proj_drop(self.proj(x))
        return x


# class JoinAttention(nn.Module):
#     def __init__(self, n_embd, n_head, dropout=0.0, use_3d_rope=True, use_rmsnorm=False):
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.use_3d_rope = use_3d_rope
#         self.use_rmsnorm = use_rmsnorm
#         # key, query, value projections for all heads, but in a batch
#         self.qkv_x = nn.Linear(n_embd, 3 * n_embd)
#         self.qkv_y = nn.Linear(n_embd, 3 * n_embd)
#         # output projection
#         self.proj_x = nn.Linear(n_embd, n_embd)
#         self.proj_y = nn.Linear(n_embd, n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_drop_x = nn.Dropout(dropout)
#         self.proj_drop_y = nn.Dropout(dropout)
#         self.n_head = n_head
#         self.n_embd = n_embd
#         self.dropout = dropout
#         head_dim = n_embd // n_head
#         if self.use_rmsnorm:
#             self.norm1 = RMSNorm(head_dim)
#             self.norm2 = RMSNorm(head_dim)
#             self.norm3 = RMSNorm(head_dim)
#             self.norm4 = RMSNorm(head_dim)
#         # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
#         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
#         self.npu_fusion = DEVICE_TYPE == "ascend" and hasattr(torch_npu, "npu_fusion_attention")
#         if self.npu_fusion:
#             self.flash = False  # disable torch flash attn in npu
#         if not self.flash and not self.npu_fusion:
#             print(
#                 "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
#             )

#     def trans_BNSD2BSH(self, tensor):
#         tensor = torch.transpose(tensor, 1, 2)
#         tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
#         return tensor

#     def forward(self, x, y, mask, spatial_freq=None):
#         B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
#         q_x, k_x, v_x = self.qkv_x(x).split(self.n_embd, dim=2)
#         q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

#         # text
#         _, L, _ = y.size()  # batch size, sequence length, embedding dimensionality (n_embd)
#         q_y, k_y, v_y = self.qkv_y(y).split(self.n_embd, dim=2)
#         q_y = q_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         k_y = k_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v_y = v_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

#         if self.use_rmsnorm:
#             q_x = self.norm1(q_x)
#             k_x = self.norm2(k_x)
#             q_y = self.norm3(q_y)
#             k_y = self.norm4(k_y)

#         if spatial_freq is not None:
#             if self.use_3d_rope:
#                 q_x, k_x = apply_3drotary_pos(q_x, k_x, freqs_cis=spatial_freq)
#             else:
#                 q_x, k_x = apply_2drotary_pos(q_x, k_x, freqs_cis=spatial_freq)

#         q = torch.cat([q_x, q_y], dim=2)
#         k = torch.cat([k_x, k_y], dim=2)
#         v = torch.cat([v_x, v_y], dim=2)

#         if self.flash: # and self.training
#             out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
#                                                                    dropout_p=self.dropout if self.training else 0)
#             out = out.transpose(1, 2).contiguous().view(B, T + L, C)  # re-assemble all head outputs side by side
#         # elif self.npu_fusion:
#         #     assert q.dtype in [torch.float16, torch.bfloat16] and k.dtype in [torch.float16, torch.bfloat16]
#         #     mask_npu = mask.repeat(1,1,T+L,1).bool()

#         #     qa = self.trans_BNSD2BSH(q)
#         #     ka = self.trans_BNSD2BSH(k)
#         #     va = self.trans_BNSD2BSH(v
#         #     out = torch_npu.npu_fusion_attention(
#         #         qa, ka, va, self.n_head,
#         #         atten_mask=mask_npu.logical_not(),
#         #         scale=(C // self.n_head) ** -0.5,
#         #         keep_prob=1.0,
#         #         input_layout="BSH",
#         #     )[0]
#         #     out = out.reshape(B, T + L, C)
#         else:
#             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#             att = att.masked_fill(mask == 0, float("-inf"))
#             att = F.softmax(att, dim=-1)
#             att = self.attn_drop(att)
#             out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#             out = out.transpose(1, 2).contiguous().view(B, T + L, C)  # re-assemble all head outputs side by side

#         x, y = out.split([T, L], dim=1)
#         # output projection
#         x = self.proj_drop_x(self.proj_x(x))
#         y = self.proj_drop_y(self.proj_y(y))
#         return x, y


class JoinAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, fa_keep_prob=1.0, use_3d_rope=True, use_qknorm=True,
                 use_rmsnorm=False, use_context_parallelism=False, depth=-1, down_mode=None, downscale=1, index=0):
        super().__init__()
        assert n_embd % n_head == 0
        self.use_context_parallelism = use_context_parallelism
        self.cp_group = get_context_parallel_group()
        self.cp_size = dist.get_world_size(self.cp_group)
        self.use_3d_rope = use_3d_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_qknorm = use_qknorm
        # key, query, value projections for all heads, but in a batch
        self.qkv_x = nn.Linear(n_embd, 3 * n_embd)
        self.qkv_y = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.proj_x = nn.Linear(n_embd, n_embd)
        self.proj_y = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop_x = nn.Dropout(dropout)
        self.proj_drop_y = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.fa_keep_prob = fa_keep_prob
        self.depth = depth
        head_dim = n_embd // n_head
        if self.use_qknorm:
            if self.use_rmsnorm:
                self.norm1 = RMSNorm(head_dim)
                self.norm2 = RMSNorm(head_dim)
                self.norm3 = RMSNorm(head_dim)
                self.norm4 = RMSNorm(head_dim)
            else:
                # self.norm1 = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
                # self.norm2 = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
                # self.norm3 = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
                # self.norm4 = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
                self.norm1 = CustomLayerNorm(head_dim, eps=1e-6)
                self.norm2 = CustomLayerNorm(head_dim, eps=1e-6)
                self.norm3 = CustomLayerNorm(head_dim, eps=1e-6)
                self.norm4 = CustomLayerNorm(head_dim, eps=1e-6)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.npu_fusion = DEVICE_TYPE == "npu" and hasattr(torch_npu, "npu_fusion_attention")
        if self.npu_fusion:
            self.flash = False  # disable torch flash attn in npu
        if not self.flash and not self.npu_fusion:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
        self.index = index
        self.down_mode = down_mode
        assert self.down_mode in ["roll", None, ""]
        self.downscale = downscale
        if self.downscale == 1:
            self.down_mode = None

        # vllm-omni attention backend (optional replacement for fa kernel)
        self.use_vllm_attn = os.environ.get("VLLM_MGM_USE_NATIVE_FA", "0") != "1"
        if self.use_vllm_attn:
            from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
            head_dim = n_embd // n_head
            # Use SDPA backend: on NPU torch SDPA delegates to npu_fusion_attention
            # (verified equivalent), while FlashAttentionBackend uses mindiesd
            # which produces numerically different results.
            self.vllm_attn = SDPABackend.get_impl_cls()(
                num_heads=n_head,
                head_size=head_dim,
                causal=False,
                softmax_scale=head_dim ** -0.5,
            )

        if self.downscale != 1:
            self.sparse_n = self.downscale
            self.sparse_n_2 = int(self.sparse_n ** 0.5)
            assert self.sparse_n == self.sparse_n_2 * self.sparse_n_2

    # def trans_BNSD2BSH(self, tensor):
    #     tensor = torch.transpose(tensor, 1, 2)
    #     tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
    #     return tensor

    def _sparse_1d(self, x, frame, height, width):
        nn = x.shape[1]
        pad_len1 = 0
        pad_len2 = 0
        if self.index % 3 != 1:
            if self.index % 3 == 0:
                x = rearrange(x, "b n (f h w) d -> b n (f w h) d", f=frame, h=height, w=width)
                if self.down_mode is not None:
                   x =  torch.roll(x, 1, dims=-2)
            x = rearrange(x, "b n x d -> x b (n d)")
            l = x.shape[0]
            assert l == frame * height * width
            if l % self.sparse_n != 0:
                pad_len1 = self.sparse_n - l % self.sparse_n
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len1))
            x = rearrange(x, '(g k) b d -> g (k b) d', k=self.sparse_n)
        else:
            x = rearrange(x, "b n x d -> x b (n d)")
            x = rearrange(x, '(f h w) b d -> f h w b d', f=frame, h=height, w=width)
            if width % self.sparse_n_2 != 0:
                pad_len1 = self.sparse_n_2 - width % self.sparse_n_2
            if pad_len1 != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_len1))
            if height % self.sparse_n_2 != 0:
                pad_len2 = self.sparse_n_2 - height % self.sparse_n_2
            if pad_len2 != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_len2))
            x = rearrange(x, 'f (y m) (x k) b d -> (f y x) (m k b) d', f=frame, m=self.sparse_n_2, k=self.sparse_n_2)
        x = rearrange(x, "x b (n d) -> b n x d", n=nn)
        return x, pad_len1, pad_len2

    def _reverse_sparse_1d(self, x, frame, height, width, pad_len1=0, pad_len2=0):
        nn = x.shape[1]
        x = rearrange(x, "b n x d -> x b (n d)")
        if self.index % 3 != 1:
            assert x.shape[0] == (frame * height * width + pad_len1) // self.sparse_n
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.sparse_n)
            x = x[:frame * height * width, :, :]
            x = rearrange(x, "x b (n d) -> b n x d", n=nn)
            if self.index % 3 == 0:
                if self.down_mode is not None:
                   x =  torch.roll(x, -1, dims=-2)
                x = rearrange(x, "b n (f w h) d -> b n (f h w) d", f=frame, h=height, w=width)
        else:
            assert x.shape[0] == (frame * ((height + pad_len2) * (width + pad_len1))) // self.sparse_n
            x = rearrange(x, '(f y x) (m k b) d -> f (y m) (x k) b d', k=self.sparse_n_2, f=frame, m=self.sparse_n_2,
                          y=(height + pad_len2) // self.sparse_n_2, x=(width + pad_len1) // self.sparse_n_2)
            x = x[:, :height, :width]
            x = rearrange(x, 'f h w b d -> (f h w) b d')
            x = rearrange(x, "x b (n d) -> b n x d", n=nn)
        return x


    def proj_and_drop(self, x, y):
        x = self.proj_drop_x(self.proj_x(x))
        y = self.proj_drop_y(self.proj_y(y))
        return x, y

    def qk_norm(self, q_x, k_x, q_y, k_y):
        q_x = self.norm1(q_x)
        k_x = self.norm2(k_x)
        q_y = self.norm3(q_y)
        k_y = self.norm4(k_y)

        return q_x, k_x, q_y, k_y

    def before_fa(self, x, x1_cts, y, spatial_freq, x_padding_size, y_padding_size, mask, f=None, hh=None, ww=None):
        assert x1_cts is None, "Ulysses sequence parallel is not supported for window attention."

        # Use parent MMDiTBlock's _debug_is_block0 flag (set by before_attention)
        # instead of self.index which is hidden by FSDP
        _is_block0 = getattr(self, '_debug_is_block0', False)

        if x1_cts != None:
            _, frame, spatial_tn, _ = x.shape
            _, _, spatial_tn_cts, _ = x1_cts.shape
            x = rearrange(x, "b f s d -> b (f s) d")
            x1_cts = rearrange(x1_cts, "b f s d -> b (f s) d")

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv_x_out = self.qkv_x(x)
        q_x, k_x, v_x = qkv_x_out.split(self.n_embd, dim=2)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        if x1_cts != None:
            _, k_x, v_x = self.qkv_x(x1_cts).split(self.n_embd, dim=2)
            k_x = k_x.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v_x = v_x.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # text
        _, L, _ = y.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv_y_out = self.qkv_y(y)
        q_y, k_y, v_y = qkv_y_out.split(self.n_embd, dim=2)
        q_y = q_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k_y = k_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v_y = v_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        if self.use_qknorm:
            q_x, k_x, q_y, k_y = self.qk_norm(q_x, k_x, q_y, k_y)


        if spatial_freq is not None:
            if self.use_3d_rope:
                q_x, k_x = apply_3drotary_pos(q_x, k_x, freqs_cis=spatial_freq)
            else:
                q_x, k_x = apply_2drotary_pos(q_x, k_x, freqs_cis=spatial_freq)

        if x1_cts != None:
            q_x = rearrange(q_x, "b n (f s) d -> (b f) n s d", f=frame, s=spatial_tn)
            k_x = rearrange(k_x, "b n (f s) d -> (b f) n s d", f=frame, s=spatial_tn_cts)
            v_x = rearrange(v_x, "b n (f s) d -> (b f) n s d", f=frame, s=spatial_tn_cts)

            q_y = q_y.repeat(frame, 1, 1, 1)
            k_y = k_y.repeat(frame, 1, 1, 1)
            v_y = v_y.repeat(frame, 1, 1, 1)

        if self.use_context_parallelism:
            if q_x.shape[1] % self.cp_size != 0:
                self.x_padding_head = self.cp_size - q_x.shape[1] % self.cp_size
            else:
                self.x_padding_head = 0

            if q_y.shape[1] % self.cp_size != 0:
                self.y_padding_head = self.cp_size - q_y.shape[1] % self.cp_size
            else:
                self.y_padding_head = 0
            assert self.x_padding_head == self.y_padding_head

            # sequence parallel split for x
            qkv_x = torch.stack((q_x, k_x, v_x), dim=0)
            if self.x_padding_head > 0:
                qkv_x = torch.nn.functional.pad(qkv_x, (0, 0, 0, 0, 0, self.x_padding_head, 0, 0, 0, 0))

            qkv_x = all_to_all(qkv_x, self.cp_group, scatter_dim=2, gather_dim=3)

            if x_padding_size > 0:
                qkv_x = qkv_x[:, :, :, :-x_padding_size, :]

            q_x, k_x, v_x = qkv_x.unbind(0)
            T = qkv_x.shape[3]

            # sequence parallel split for y
            qkv_y = torch.stack((q_y, k_y, v_y), dim=0)
            if self.y_padding_head > 0:
                qkv_y = torch.nn.functional.pad(qkv_y, (0, 0, 0, 0, 0, self.y_padding_head, 0, 0, 0, 0))

            qkv_y = all_to_all(qkv_y, self.cp_group, scatter_dim=2, gather_dim=3)
            if y_padding_size > 0:
                qkv_y = qkv_y[:, :, :, :-y_padding_size, :]
            q_y, k_y, v_y = qkv_y.unbind(0)
            L = qkv_y.shape[3]

        if self.downscale != 1:
            # before_fa, 先CP(聚合S, 切head), 再skiparse(对S重组, q/k/v/mask)
            self.T_ori = q_x.shape[-2]
            q_x, self.pad_len1, self.pad_len2 = self._sparse_1d(q_x, f, hh, ww)
            k_x, self.pad_len1, self.pad_len2 = self._sparse_1d(k_x, f, hh, ww)
            v_x, self.pad_len1, self.pad_len2 = self._sparse_1d(v_x, f, hh, ww)
            q_y = torch.cat([q_y] * self.sparse_n, dim=0)
            k_y = torch.cat([k_y] * self.sparse_n, dim=0)
            v_y = torch.cat([v_y] * self.sparse_n, dim=0)
            T = q_x.shape[-2]

        q = torch.cat([q_x, q_y], dim=2)
        k = torch.cat([k_x, k_y], dim=2)
        v = torch.cat([v_x, v_y], dim=2)

        if isinstance(mask, list):
            # 针对skiparse算法对mask做同等变换
            # TODO, 整合skiparse和非skiparse的mask操作
            x_mask, y_mask = mask
            if x_mask is not None and y_mask is not None:
                assert x_mask.shape[1] == f * hh * ww
                if self.downscale != 1:
                    x_mask = x_mask.unsqueeze(1).unsqueeze(-1)
                    x_mask, x_mask_pad1, x_mask_pad2 = self._sparse_1d(x_mask, f, hh, ww)
                    assert x_mask_pad1 == self.pad_len1
                    assert x_mask_pad2 == self.pad_len2
                    x_mask = x_mask.squeeze(1).squeeze(-1)
                    y_mask = torch.cat([y_mask] * self.sparse_n, dim=0)
                    mask = torch.cat((x_mask, y_mask), dim=1).bool()
                    mask = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T+L)
                    mask = mask.repeat(1, 1, x_mask.shape[-1] + y_mask.shape[-1], 1).bool().logical_not()
                else:
                    mask = torch.cat((x_mask, y_mask), dim=1).bool()  # (B, T+L)
                    mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T+L)
                    mask = mask.repeat(1, 1, x_mask.shape[-1] + y_mask.shape[-1], 1).bool().logical_not()
            elif x_mask is None and y_mask is None:
                mask = None
            else:
                raise NotImplementedError
        else:
            # mask在mimogpt/models/dit/mmdit.py的MMDiT类的forward中已完成操作
            pass

        if x1_cts != None:
            B, _, T, _ = q_x.shape

        if x1_cts != None:
            return q, k, v, mask, B, T, C, L, frame, spatial_tn
        else:
            return q, k, v, mask, B, T, C, L, None, None

    def after_fa(self, out, B, T, C, L, q, x1_cts, frame, spatial_tn, x_padding_size, y_padding_size, f=None, hh=None, ww=None):
        _is_block0 = getattr(self, '_debug_is_block0', False)
        if self.downscale != 1:
            # after_fa, 先skiparse(对S反重组, out), 再CP(恢复head1, 切S)，
            out_x, out_y = out.split([T, L], dim=-2)
            out_x = self._reverse_sparse_1d(out_x, f, hh, ww, self.pad_len1, self.pad_len2)
            out_y = rearrange(out_y, "(q p) n x d -> p q n x d", q=self.sparse_n)
            out_y = torch.mean(out_y, dim=1)
            assert out_y.shape[0] == out_x.shape[0]
            T = self.T_ori
            out = torch.cat([out_x, out_y], dim=-2)

        if self.use_context_parallelism:
            out = out.transpose(1, 2)
            x, y = out.split([T, L], dim=1)
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, x_padding_size))
            y = torch.nn.functional.pad(y, (0, 0, 0, 0, 0, y_padding_size))

            T += x_padding_size
            L += y_padding_size

            x = all_to_all(x, self.cp_group, scatter_dim=1, gather_dim=2)
            y = all_to_all(y, self.cp_group, scatter_dim=1, gather_dim=2)

            if self.x_padding_head > 0:
                x = x[:, :, :-self.x_padding_head, :]
                y = y[:, :, :-self.y_padding_head, :]

            x = x.reshape(B, T // self.cp_group.size(), C)
            y = y.reshape(B, L // self.cp_group.size(), C)
        else:
            out = out.transpose(1, 2)
            out = out.reshape(B, T + L, C).type_as(q)
            x, y = out.split([T, L], dim=1)

        x, y = self.proj_and_drop(x, y)

        if x1_cts != None:
            x = rearrange(x, "(b f) s d -> b f s d", f=frame, s=spatial_tn)
            y = rearrange(y, "(b f) l d -> b f l d", f=frame, l=L)
            y = torch.mean(y, dim=1)
        return x, y

    def fa(self, q, k, v, mask, C, offload_fa, h2d_stream=None, d2h_stream=None, num_layer=-1):
        _is_block0 = getattr(self, '_debug_is_block0', False)
        if self.use_vllm_attn:
            from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
            # mask semantic conversion: mmdit True=masked -> vllm-omni True=keep
            if mask is not None:
                mask = mask.logical_not()
            metadata = AttentionMetadata(attn_mask=mask)
            # vllm-omni SDPA backend expects BSND [B, S, N, D] input format,
            # while mmdit produces BNSD [B, N, S, D]. Permute before calling.
            q_bsnd = q.permute(0, 2, 1, 3)
            k_bsnd = k.permute(0, 2, 1, 3)
            v_bsnd = v.permute(0, 2, 1, 3)
            out = self.vllm_attn.forward(q_bsnd, k_bsnd, v_bsnd, metadata)
            return out.permute(0, 2, 1, 3)

        if self.flash:
            mask = mask.logical_not() if mask != None else None
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=1 - self.fa_keep_prob)
        elif self.npu_fusion:
            # q,k,v shape-[B,N,S,D]
            n_head = q.shape[1]

            if offload_fa:
                with async_save_on_cpu(h2d_stream=h2d_stream, d2h_stream=d2h_stream, num_layer=num_layer, depth=self.depth):
                    out = torch_npu.npu_fusion_attention(
                        q, k, v, n_head,
                        atten_mask=mask,
                        scale=(C // self.n_head) ** -0.5,
                        keep_prob=self.fa_keep_prob,
                        input_layout="BNSD",
                    )[0]
            else:
                out = torch_npu.npu_fusion_attention(
                    q, k, v, n_head,
                    atten_mask=mask,
                    scale=(C // self.n_head) ** -0.5,
                    keep_prob=self.fa_keep_prob,
                    input_layout="BNSD",
                )[0]
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = att @ v

        return out

    def forward(self, x, x1_cts, y, mask, spatial_freq=None, spatial_freq_ctx=None, use_finegrained=False):
        # 已挪到mimogpt/models/dit/mmdit.py的MMDiTBlock类的_forward实现
        raise NotImplementedError
