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
from vllm.model_executor.layers.linear import ReplicatedLinear

from .mmdit_parallel_states import get_context_parallel_group
from .mmdit_communications import all_to_all, split_forward_gather_backward, gather_forward_split_backward

import importlib

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
    sinusoid_inp = torch.stack((sinusoid_inp, sinusoid_inp), dim=-1).flatten(-2)
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return x.flatten(-2)


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
    k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


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
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU, token_num=120):
        super().__init__()
        from timm.models.vision_transformer import Mlp
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size,
                          act_layer=act_layer, drop=0)
        self.uncond_prob = uncond_prob

    def forward(self, caption, train, force_drop_ids=None):
        caption = self.y_proj(caption)
        return caption


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.dense_h_to_4h = ReplicatedLinear(n_embd, 4 * n_embd, bias=True, return_bias=False)
        self.dense_4h_to_h = ReplicatedLinear(4 * n_embd, n_embd, bias=True, return_bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.gelu(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x


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
        self.qkv_x = ReplicatedLinear(n_embd, 3 * n_embd, bias=True, return_bias=False)
        self.qkv_y = ReplicatedLinear(n_embd, 3 * n_embd, bias=True, return_bias=False)
        self.proj_x = ReplicatedLinear(n_embd, n_embd, bias=True, return_bias=False)
        self.proj_y = ReplicatedLinear(n_embd, n_embd, bias=True, return_bias=False)
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
            self.norm1 = CustomLayerNorm(head_dim, eps=1e-6)
            self.norm2 = CustomLayerNorm(head_dim, eps=1e-6)
            self.norm3 = CustomLayerNorm(head_dim, eps=1e-6)
            self.norm4 = CustomLayerNorm(head_dim, eps=1e-6)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.npu_fusion = DEVICE_TYPE == "npu" and hasattr(torch_npu, "npu_fusion_attention")
        if self.npu_fusion:
            self.flash = False
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

        self.use_vllm_attn = os.environ.get("VLLM_MGM_USE_NATIVE_FA", "0") != "1"
        if self.use_vllm_attn:
            from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
            head_dim = n_embd // n_head
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

        _is_block0 = getattr(self, '_debug_is_block0', False)

        if x1_cts != None:
            _, frame, spatial_tn, _ = x.shape
            _, _, spatial_tn_cts, _ = x1_cts.shape
            x = rearrange(x, "b f s d -> b (f s) d")
            x1_cts = rearrange(x1_cts, "b f s d -> b (f s) d")

        B, T, C = x.size()
        qkv_x_out = self.qkv_x(x)
        q_x, k_x, v_x = qkv_x_out.split(self.n_embd, dim=2)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        if x1_cts != None:
            _, k_x, v_x = self.qkv_x(x1_cts).split(self.n_embd, dim=2)
            k_x = k_x.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            v_x = v_x.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)

        _, L, _ = y.size()
        qkv_y_out = self.qkv_y(y)
        q_y, k_y, v_y = qkv_y_out.split(self.n_embd, dim=2)
        q_y = q_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        k_y = k_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v_y = v_y.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)


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

            qkv_x = torch.stack((q_x, k_x, v_x), dim=0)
            if self.x_padding_head > 0:
                qkv_x = torch.nn.functional.pad(qkv_x, (0, 0, 0, 0, 0, self.x_padding_head, 0, 0, 0, 0))

            qkv_x = all_to_all(qkv_x, self.cp_group, scatter_dim=2, gather_dim=3)

            if x_padding_size > 0:
                qkv_x = qkv_x[:, :, :, :-x_padding_size, :]

            q_x, k_x, v_x = qkv_x.unbind(0)
            T = qkv_x.shape[3]

            qkv_y = torch.stack((q_y, k_y, v_y), dim=0)
            if self.y_padding_head > 0:
                qkv_y = torch.nn.functional.pad(qkv_y, (0, 0, 0, 0, 0, self.y_padding_head, 0, 0, 0, 0))

            qkv_y = all_to_all(qkv_y, self.cp_group, scatter_dim=2, gather_dim=3)
            if y_padding_size > 0:
                qkv_y = qkv_y[:, :, :, :-y_padding_size, :]
            q_y, k_y, v_y = qkv_y.unbind(0)
            L = qkv_y.shape[3]

        if self.downscale != 1:
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
                    mask = mask.unsqueeze(1).unsqueeze(2)
                    mask = mask.repeat(1, 1, x_mask.shape[-1] + y_mask.shape[-1], 1).bool().logical_not()
                else:
                    mask = torch.cat((x_mask, y_mask), dim=1).bool()
                    mask = mask.unsqueeze(1).unsqueeze(2)
                    mask = mask.repeat(1, 1, x_mask.shape[-1] + y_mask.shape[-1], 1).bool().logical_not()
            elif x_mask is None and y_mask is None:
                mask = None
            else:
                raise NotImplementedError
        else:
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
        if self.use_vllm_attn:
            from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
            if mask is not None:
                mask = mask.logical_not()
            metadata = AttentionMetadata(attn_mask=mask)
            q_bsnd = q.permute(0, 2, 1, 3)
            k_bsnd = k.permute(0, 2, 1, 3)
            v_bsnd = v.permute(0, 2, 1, 3)
            out = self.vllm_attn.forward(q_bsnd, k_bsnd, v_bsnd, metadata)
            return out.permute(0, 2, 1, 3)

        if self.flash:
            mask = mask.logical_not() if mask != None else None
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=1 - self.fa_keep_prob)
        elif self.npu_fusion:
            n_head = q.shape[1]
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
