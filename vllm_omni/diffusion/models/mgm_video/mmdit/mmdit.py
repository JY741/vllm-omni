# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from timm.models.vision_transformer import Mlp
from einops import rearrange, repeat
from .mmdit_functional import to_2tuple
from .mmdit_blocks import t2i_modulate, modulate, CaptionEmbedder, \
    TimestepEmbedder, FinalLayer, MLP, create_sinusoidal_positions, JoinAttention, SizeEmbedder
from .mmdit_parallel_states import get_context_parallel_group
from .mmdit_communications import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)

import torch.distributed as dist

# Global counter for block-level debug printing (bypasses FSDP attribute hiding)
_block_debug_counter = [0]
_block_debug_step = [0]

class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        assert isinstance(normalized_shape, int)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x, scale, shift):
        with torch.cuda.amp.autocast(enabled=False):
            weight = (1 + scale.float()).squeeze()
            bias = shift.float().squeeze()
            ret = torch.nn.functional.layer_norm(x.float(), [self.normalized_shape], weight=weight, bias=bias, eps=self.eps)
        return ret

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MMDiTBlock(nn.Module):
    def __init__(
            self,
            n_embd,
            n_head,
            dropout,
            fa_keep_prob=1.0,
            use_checkpoint=False,
            checkpoint_layer=-1,
            checkpoint_finegrained_layer=0,
            use_context_parallelism=False,offload_fa=False,
            h2d_stream=None,
            d2h_stream=None,
            depth=-1,
            down_mode=None, downscale=1, index=0
    ):
        super().__init__()
        self.checkpoint_layer = checkpoint_layer
        self.checkpoint_finegrained_layer = checkpoint_finegrained_layer
        self.use_checkpoint = use_checkpoint
        self.norm_x_1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.norm_x_2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.norm_y_1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.norm_y_2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.fusednorm_x_1 = FusedLayerNorm(n_embd, eps=1e-6)
        self.fusednorm_x_2 = FusedLayerNorm(n_embd, eps=1e-6)
        self.fusednorm_y_1 = FusedLayerNorm(n_embd, eps=1e-6)
        self.fusednorm_y_2 = FusedLayerNorm(n_embd, eps=1e-6)
        self.attention = JoinAttention(
			n_embd,
			n_head,
			dropout,
			fa_keep_prob=fa_keep_prob,
			use_context_parallelism=use_context_parallelism,
            depth=depth, down_mode=down_mode, downscale=downscale, index=index
		)
        self.mlp_x = MLP(n_embd, dropout)
        self.mlp_y = MLP(n_embd, dropout)

        self.h2d_stream = h2d_stream
        self.d2h_stream = d2h_stream
        self.depth = depth
        self.offload_fa = offload_fa
        self.index = index

        self.adaLN_modulation_x = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4),
            nn.SiLU(),
            nn.Linear(n_embd // 4, n_embd * 6),
        )
        self.adaLN_modulation_y = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4),
            nn.SiLU(),
            nn.Linear(n_embd // 4, n_embd * 6),
        )

    def collect_spatial_contexts(self, s_win_size, x):
        if s_win_size <= 1:
            return None
        B, F, S, C = x.shape

        all_ctx = x
        p = s_win_size // 2
        pad_ctx = torch.cat((
            torch.zeros(B, p, S, C, dtype=x.dtype, device=x.device),
            all_ctx,
            torch.zeros(B, p, S, C, dtype=x.dtype, device=x.device),
        ), dim=1)  # B p+T+p S C
        contexts = []
        for i in range(-p, p + 1):
            ctx = pad_ctx[:, p + i:p + i + F, :, :]
            contexts.append(ctx)
        contexts = torch.cat(contexts, dim=2)  # B AT s_win_size*S C

        return contexts

    def collect_spatial_freq_contexts(self, s_win_size, frame, spatial_freq):
        if s_win_size <= 1:
            return None

        sincos_h, sincos_w, sincos_t = spatial_freq
        token_num = sincos_h.shape[0]
        sincos_h = rearrange(sincos_h, "(f s) d -> f s d", f=frame)
        sincos_w = rearrange(sincos_w, "(f s) d -> f s d", f=frame)
        sincos_t = rearrange(sincos_t, "(f s) d -> f s d", f=frame)
        sincos_h = sincos_h.unsqueeze(0)
        sincos_w = sincos_w.unsqueeze(0)
        sincos_t = sincos_t.unsqueeze(0)

        B, F, S, C1 = sincos_h.shape
        B, F, S, C2 = sincos_w.shape
        B, F, S, C3 = sincos_t.shape

        sincos_h_ctx = sincos_h
        sincos_w_ctx = sincos_w
        sincos_t_ctx = sincos_t

        p = s_win_size // 2
        pad_sincos_h_ctx = torch.cat((
            torch.zeros(B, p, S, C1, dtype=sincos_h_ctx.dtype, device=sincos_h_ctx.device),
            sincos_h_ctx,
            torch.zeros(B, p, S, C1, dtype=sincos_h_ctx.dtype, device=sincos_h_ctx.device),
        ), dim=1)

        pad_sincos_w_ctx = torch.cat((
            torch.zeros(B, p, S, C2, dtype=sincos_w_ctx.dtype, device=sincos_w_ctx.device),
            sincos_w_ctx,
            torch.zeros(B, p, S, C2, dtype=sincos_w_ctx.dtype, device=sincos_w_ctx.device),
        ), dim=1)

        pad_sincos_t_ctx = torch.cat((
            torch.zeros(B, p, S, C3, dtype=sincos_t_ctx.dtype, device=sincos_t_ctx.device),
            sincos_t_ctx,
            torch.zeros(B, p, S, C3, dtype=sincos_t_ctx.dtype, device=sincos_t_ctx.device),
        ), dim=1)


        contexts_sincos_h = []
        contexts_sincos_w = []
        contexts_sincos_t = []
        for i in range(-p, p + 1):
            ctx_sincos_h = pad_sincos_h_ctx[:, p + i:p + i + F, :, :]
            contexts_sincos_h.append(ctx_sincos_h)
            ctx_sincos_w = pad_sincos_w_ctx[:, p + i:p + i + F, :, :]
            contexts_sincos_w.append(ctx_sincos_w)
            ctx_sincos_t = pad_sincos_t_ctx[:, p + i:p + i + F, :, :]
            contexts_sincos_t.append(ctx_sincos_t)

        contexts_sincos_h = torch.cat(contexts_sincos_h, dim=2)
        contexts_sincos_w = torch.cat(contexts_sincos_w, dim=2)
        contexts_sincos_t = torch.cat(contexts_sincos_t, dim=2)

        contexts_sincos_h = rearrange(contexts_sincos_h, "b f s d -> (b f s) d", f=frame)
        contexts_sincos_w = rearrange(contexts_sincos_w, "b f s d -> (b f s) d", f=frame)
        contexts_sincos_t = rearrange(contexts_sincos_t, "b f s d -> (b f s) d", f=frame)

        spatial_freq_ctx = (contexts_sincos_h, contexts_sincos_w, contexts_sincos_t)
        return spatial_freq_ctx

    def t2i_modulate_xy(self, x, y, shift_msa_x, scale_msa_x, shift_msa_y, scale_msa_y):
        x1 = t2i_modulate(self.norm_x_1(x), shift_msa_x, scale_msa_x)
        y1 = t2i_modulate(self.norm_y_1(y), shift_msa_y, scale_msa_y)
        return x1, y1

    def t2i_modulate_xy_v2(self, x, y, shift_msa_x, scale_msa_x, shift_msa_y, scale_msa_y):
        x1 = self.fusednorm_x_1(x, scale_msa_x, shift_msa_x).type_as(x)
        y1 = self.fusednorm_y_1(y, scale_msa_y, shift_msa_y).type_as(y)
        return x1, y1

    def mlp_xy(self, x, y, x1, y1, gate_mlp_x, gate_mlp_y, shift_mlp_x, scale_mlp_x, shift_mlp_y, scale_mlp_y):
        x = x + gate_mlp_x * self.mlp_x(t2i_modulate(self.norm_x_2(x1), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(t2i_modulate(self.norm_y_2(y1), shift_mlp_y, scale_mlp_y))
        return x, y

    def mlp_xy_v2(self, x, y, x1, y1, gate_mlp_x, gate_mlp_y, shift_mlp_x, scale_mlp_x, shift_mlp_y, scale_mlp_y):
        t2i_x = self.fusednorm_x_2(x1, scale_mlp_x, shift_mlp_x).type_as(x1)
        t2i_y = self.fusednorm_y_2(y1, scale_mlp_y, shift_mlp_y).type_as(y1)
        x = x + gate_mlp_x * self.mlp_x(t2i_x)
        y = y + gate_mlp_y * self.mlp_y(t2i_y)
        return x, y

    def add_mul(self, x, scale, mul_x):
        return x + scale*mul_x

    def xy_add_mul(self, x, gate_msa_x, x1, y, gate_msa_y, y1):
        x1 = self.add_mul(x, gate_msa_x, x1)
        y1 = self.add_mul(y, gate_msa_y, y1)
        return x1, y1

    def before_attention(self, x, y, t, x1_cts, mask, spatial_freq, x_padding_size, y_padding_size, f=None, hh=None, ww=None):
        import torch
        import sys
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1

        B = x.shape[0]
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = \
            self.adaLN_modulation_x(t).reshape(B, 6, -1).chunk(6, dim=1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = \
            self.adaLN_modulation_y(t).reshape(B, 6, -1).chunk(6, dim=1)


        if x.shape[0] == 1:  ## video data
            x1, y1 = self.t2i_modulate_xy_v2(x, y, shift_msa_x, scale_msa_x, shift_msa_y, scale_msa_y)
        else:  ## image data
            x1, y1 = self.t2i_modulate_xy(x, y, shift_msa_x, scale_msa_x, shift_msa_y, scale_msa_y)


        self.attention._debug_is_block0 = _is_block0

        q, k, v, mask, B, T, C, L, frame, spatial_tn = self.attention.before_fa(x1, x1_cts, y1, spatial_freq, x_padding_size, y_padding_size, mask=mask, f=f, hh=hh, ww=ww)

        fa_info = {}
        fa_info['q'] = q
        fa_info['k'] = k
        fa_info['v'] = v
        fa_info['mask'] = mask

        shape_info = {}
        shape_info['B'] = B
        shape_info['T'] = T
        shape_info['C'] = C
        shape_info['L'] = L
        shape_info['frame'] = frame
        shape_info['spatial_tn'] = spatial_tn

        modulate_info = {}
        modulate_info['gate_msa_x'] = gate_msa_x
        modulate_info['gate_msa_y'] = gate_msa_y
        modulate_info['gate_mlp_x'] = gate_mlp_x
        modulate_info['gate_mlp_y'] = gate_mlp_y
        modulate_info['shift_mlp_x'] = shift_mlp_x
        modulate_info['scale_mlp_x'] = scale_mlp_x
        modulate_info['shift_mlp_y'] = shift_mlp_y
        modulate_info['scale_mlp_y'] = scale_mlp_y
        return fa_info, shape_info, modulate_info

    def after_attention(self, x, y, out, fa_info, shape_info, modulate_info, x1_cts, x_padding_size, y_padding_size, f=None, hh=None, ww=None):
        _is_block0 = getattr(self, '_debug_is_block0', False)
        x1, y1 = self.attention.after_fa(
            out, shape_info['B'], shape_info['T'], shape_info['C'],
            shape_info['L'], fa_info['q'], x1_cts, shape_info['frame'], shape_info['spatial_tn'],
            x_padding_size, y_padding_size, f=f, hh=hh, ww=ww
        )

        x1, y1 = self.xy_add_mul(x, modulate_info['gate_msa_x'], x1, y, modulate_info['gate_msa_y'], y1)

        if x.shape[0] == 1:   ## video data
            x, y = self.mlp_xy_v2(x, y, x1, y1, modulate_info['gate_mlp_x'], modulate_info['gate_mlp_y'], modulate_info['shift_mlp_x'], modulate_info['scale_mlp_x'], modulate_info['shift_mlp_y'], modulate_info['scale_mlp_y'])
        else: ## image data
            x, y = self.mlp_xy(x, y, x1, y1, modulate_info['gate_mlp_x'], modulate_info['gate_mlp_y'], modulate_info['shift_mlp_x'], modulate_info['scale_mlp_x'], modulate_info['shift_mlp_y'], modulate_info['scale_mlp_y'])

        return x, y

    def _forward(self, x, y, t, x_padding_size, y_padding_size, mask=None, spatial_freq=None, use_finegrained=False, num_layer=-1, f=None, hh=None, ww=None):
        x1_cts = None
        if use_finegrained:
            with record_function("before_attention"):
                fa_info, shape_info, modulate_info = torch.utils.checkpoint.checkpoint(
                    self.before_attention,
                    x, y, t, x1_cts, mask, spatial_freq, x_padding_size, y_padding_size, f, hh, ww,
                    use_reentrant=False
                )
        else:
            fa_info, shape_info, modulate_info = self.before_attention(
                    x, y, t, x1_cts, mask, spatial_freq, x_padding_size, y_padding_size, f, hh, ww)

        offload_fa = self.offload_fa and use_finegrained and self.training
        out = self.attention.fa(fa_info['q'], fa_info['k'], fa_info['v'], fa_info['mask'], shape_info['C'], offload_fa, h2d_stream=self.h2d_stream, d2h_stream=self.d2h_stream, num_layer=num_layer)

        if use_finegrained:
            with record_function("after_attention"):
                x, y = torch.utils.checkpoint.checkpoint(
                    self.after_attention,
                    x, y, out, fa_info, shape_info, modulate_info, x1_cts, x_padding_size, y_padding_size, f, hh, ww,
                    use_reentrant=False
                )
        else:
            x, y = self.after_attention(
                    x, y, out, fa_info, shape_info, modulate_info, x1_cts, x_padding_size, y_padding_size, f, hh, ww)

        return x, y

    def forward(self, x, y, t, x_padding_size, y_padding_size, mask=None, spatial_freq=None, num_layer=-1, f=None, hh=None, ww=None):
        if self.use_checkpoint and num_layer < self.checkpoint_layer:
            with record_function("MMDiTBlock_{}".format(num_layer)):
                if num_layer < self.checkpoint_layer - self.checkpoint_finegrained_layer:
                    return torch.utils.checkpoint.checkpoint(self._forward, x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, False, num_layer, f, hh, ww, use_reentrant=False)
                else:
                    return self._forward(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=num_layer, use_finegrained=True, f=f, hh=hh, ww=ww)
        else:
            return self._forward(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=num_layer, f=f, hh=hh, ww=ww)


#################################################################################
#                                 Core MMDiT Model                              #
#################################################################################
class MMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, max_input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=24, head_dim=64,
                 class_dropout_prob=0.1, pred_sigma=False, caption_channels=4096, lewei_scale=1.0, dropout=0.0, fa_keep_prob=1.0,
                 model_max_length=200, use_rel_pos=True, use_3d_rope=True, rope_ratio=[22/64, 22/64, 20/64], use_size_control=False, use_checkpoint=True,
                 checkpoint_finegrained_layer=0, checkpoint_layer=-1, use_mmdit_block=True, dtype='bf16', use_context_parallelism=False, offload_fa=False,
                 skiparse=None, skip_initialize_weights=True, x2v=False, cond_intype='sum', out_channels=16):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.x2v = x2v
        self.cond_intype = cond_intype
        self.num_heads = hidden_size // head_dim
        self.attention_head_dim = head_dim
        self.lewei_scale = lewei_scale,
        self.text_max_length = model_max_length
        self.checkpoint_layer = checkpoint_layer if checkpoint_layer != -1 else depth
        self.depth = depth
        self.checkpoint_finegrained_layer = min(checkpoint_finegrained_layer, self.checkpoint_layer)
        print(f'--->>> MMDiT: max_input_size:{max_input_size} dtype:{dtype} use_context_parallelism:{use_context_parallelism}')
        self.x_embedder = PatchEmbed(None, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.use_size_control = use_size_control
        if self.use_size_control:
            self.size_embedder = SizeEmbedder(hidden_size // 2)  # c_size embed

        self.input_size = None
        self.patch_size = (1, patch_size, patch_size)
        self.use_3d_rope = use_3d_rope
        self.rope_ratio = rope_ratio
        self.use_rel_pos = use_rel_pos

        self.offload_fa = offload_fa
        if self.offload_fa:
            h2d_stream = torch.cuda.Stream()
            d2h_stream = torch.cuda.Stream()
        else:
            h2d_stream = None
            d2h_stream = None

        self.use_context_parallelism = use_context_parallelism
        print("use_context_parallelism...", use_context_parallelism)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size,
                                          uncond_prob=class_dropout_prob, act_layer=approx_gelu,
                                          token_num=self.text_max_length)
        if skiparse:
            mode = skiparse.split(",")
            assert len(mode) == 3
            sparse_n = int(mode[0].strip()[-1])
            self.down_mode = None if mode[-1].strip() != "roll" else "roll"
            down_layers = mode[1].split("-")
            down_layers[0] = int(down_layers[0].strip())
            down_layers[1] = int(down_layers[1].strip())
            downscale = [1 for _ in range(depth)]
            for i in range(down_layers[0], down_layers[1]+1):
                downscale[i] = sparse_n
            print(["downscale", downscale])
            self.skiparse = True
        else:
            self.down_mode = ""
            downscale = [1 for i in range(depth)]
            self.skiparse = False

        self.downscale = downscale

        self.use_mmdit_block = use_mmdit_block
        self.blocks = nn.ModuleList([
            MMDiTBlock(n_embd=hidden_size, n_head=self.num_heads, dropout=dropout, fa_keep_prob=fa_keep_prob, use_checkpoint=use_checkpoint,
            checkpoint_finegrained_layer=self.checkpoint_finegrained_layer, checkpoint_layer=self.checkpoint_layer, use_context_parallelism=use_context_parallelism,
            offload_fa=self.offload_fa, h2d_stream=h2d_stream, d2h_stream=d2h_stream, depth=depth,
            down_mode=self.down_mode, downscale=downscale[i], index=i)
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.dtype = dtype
        if self.dtype == 'bf16' or self.dtype == torch.bfloat16:
            self.dtype = torch.bfloat16
        elif self.dtype == 'fp16' or self.dtype == torch.float16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if use_context_parallelism:
            self.cp_rank = dist.get_rank(get_context_parallel_group())
        else:
            self.cp_rank = None
        print(f'Warning: lewei scale: {self.lewei_scale}')

    def forward(self, x, timestep, y, x_mask=None, y_mask=None, data_info=None, mask=None, hh=None, ww=None, cond=None, **kwargs):
        """
        Forward pass of MMDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        _block_debug_counter[0] = 0
        _block_debug_step[0] += 1

        raw_input_size = x.shape[2:]

        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # padding for patch
        _, _, T, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = torch.nn.functional.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        if self.x2v and not self.training and cond is not None:
            _, _, T1, H1, W1 = cond.size()
            if W1 % self.patch_size[2] != 0:
                cond = torch.nn.functional.pad(cond, (0, self.patch_size[2] - W1 % self.patch_size[2]))
            if H1 % self.patch_size[1] != 0:
                cond = torch.nn.functional.pad(cond, (0, 0, 0, self.patch_size[1] - H1 % self.patch_size[1]))
            if T1 % self.patch_size[0] != 0:
                cond = torch.nn.functional.pad(cond, (0, 0, 0, 0, 0, self.patch_size[0] - T1 % self.patch_size[0]))
            x = torch.cat([x,cond.to(self.dtype)],dim=1)

        bs, _, fn, lh, lw = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        if self.x2v and self.cond_intype == 'sum':
            x, masked_x, mask_x = x[:, :self.in_channels], x[:, self.in_channels: 2 * self.in_channels], x[:, 2 * self.in_channels:]
            masked_x = self.maskedx_embedder(masked_x.to(self.dtype))
            mask_x = self.mask_embedder(mask_x.to(self.dtype))
            x = self.x_embedder(x)
            x = x + masked_x + mask_x
        else:
            x = self.x_embedder(x)

        _, ls, _ = x.shape
        x = rearrange(x, "(b f) s d -> b (f s) d", b=bs)

        if not self.use_rel_pos:
            x = x + self.pos_embed.to(self.dtype)  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(timestep.float()).to(x.dtype)  # (N, D)

        if bs != 1:
            num_nonzero = torch.count_nonzero(y_mask, dim=1).max()
            y = y[:, :, :num_nonzero, :] # (N, 1, L, D)
            y_mask = y_mask[:, :num_nonzero]
            y = self.y_embedder(y, self.training)  # (N, 1, L, D)
            y = y.squeeze(1) # (N, L, D)
            # process mask
            if x_mask is None:
                x_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
            if y_mask is None:
                y_mask = torch.ones((y.shape[0], y.shape[1]), dtype=torch.bool, device=y.device)
            if not self.skiparse:
                mask = torch.cat((x_mask, y_mask), dim=1).bool() # (B, T+L)
                mask = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T+L)
                mask = mask.repeat(1, 1, x.shape[1] + y.shape[1], 1).bool().logical_not()
            else:
                mask = [x_mask, y_mask]
        else:
            num_nonzero = torch.count_nonzero(y_mask, dim=1)[0]
            y = y[:, :, :num_nonzero, :] # (N, 1, L, D)
            y = self.y_embedder(y, self.training)  # (N, 1, L, D)
            y = y.squeeze(1) # (N, L, D)

            if not self.skiparse:
                mask = None
            else:
                x_mask, y_mask = None, None
                mask = [x_mask, y_mask]

        if self.use_size_control:
            c_size = data_info['img_hw'].to(self.dtype)
            c_size = self.size_embedder(c_size, y.shape[0])  # (N, D)
            y = torch.cat((c_size.unsqueeze(1), y), dim=1)
            y_mask = torch.cat((
                torch.ones((y_mask.shape[0], 1), dtype=torch.bool, device=y.device),
                y_mask
            ), dim=1)

        self.input_size = (fn, lh, lw)
        base_size_t = self.input_size[0] // self.patch_size[0]
        base_size_h = self.input_size[1] // self.patch_size[1]
        base_size_w = self.input_size[2] // self.patch_size[2]
        if self.use_3d_rope:
            spatial_position_ids = [[i for k in range(base_size_t) for i in range(base_size_h) for j in range(base_size_w)],
                                    [j for k in range(base_size_t) for i in range(base_size_h) for j in range(base_size_w)]]
            temporal_position_ids = [k for k in range(base_size_t) for i in range(base_size_h) for j in range(base_size_w)]
            embed_positions_h = create_sinusoidal_positions(base_size_h, int(self.rope_ratio[0]*self.attention_head_dim))
            embed_positions_w = create_sinusoidal_positions(base_size_w, int(self.rope_ratio[1]*self.attention_head_dim))
            embed_positions_t = create_sinusoidal_positions(base_size_t, int(self.rope_ratio[2]*self.attention_head_dim))
            sincos_h = embed_positions_h[spatial_position_ids[0]]
            sincos_w = embed_positions_w[spatial_position_ids[1]]
            sincos_t = embed_positions_t[temporal_position_ids]
            spatial_freq = [sincos_h.to(self.dtype), sincos_w.to(self.dtype), sincos_t.to(self.dtype)]
        else:
            spatial_freq = None

        if self.use_context_parallelism:
            cp_size = get_context_parallel_group().size()
            x_dim_size = x.shape[1]
            y_dim_size = y.shape[1]

            if x_dim_size % cp_size != 0:
                x_padding_size = cp_size - x_dim_size % cp_size
            else:
                x_padding_size = 0

            if y_dim_size % cp_size != 0:
                y_padding_size = cp_size - y_dim_size % cp_size
            else:
                y_padding_size = 0

            if x_padding_size > 0:
                x = torch.nn.functional.pad(x, (0,0,0,x_padding_size,0,0))
                for s_f_idx in range(len(spatial_freq)):
                    spatial_freq[s_f_idx] = torch.nn.functional.pad(spatial_freq[s_f_idx], (0,0,0,x_padding_size))
                spatial_freq = tuple(spatial_freq)

            if y_padding_size > 0:
                y = torch.nn.functional.pad(y, (0,0,0,y_padding_size,0,0))

            x = split_forward_gather_backward(x, get_context_parallel_group(), dim=1)
            y = split_forward_gather_backward(y, get_context_parallel_group(), dim=1)

            spatial_freq = tuple(torch.chunk(s_f, cp_size, dim=0)[self.cp_rank].contiguous() for s_f in spatial_freq)
        else:
            x_padding_size = 0
            y_padding_size = 0

        x, y = self.blocks_forward(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, fn, base_size_h, base_size_w, **kwargs)


        if self.use_context_parallelism:
            x = gather_forward_split_backward(x, get_context_parallel_group(), dim=1)
            y = gather_forward_split_backward(y, get_context_parallel_group(), dim=1)

            if x_padding_size > 0:
                x = x[:, :-x_padding_size, :]
            if y_padding_size > 0:
                y = y[:, :-y_padding_size, :]

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify3D(x)

        # remove padding
        T, H, W = raw_input_size
        x = x[:, :, :T, :H, :W]

        x = x.to(torch.float32)
        return x

    def blocks_forward(self, x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, fn, base_size_h, base_size_w, **kwargs):
        if self.use_mmdit_block:
            _rank = torch.distributed.get_rank()
            for i, block in enumerate(self.blocks):
                block.cur_time_index = kwargs['cur_time_index']
                x, y = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w)
        else:
            raise NotImplementedError
        return x, y

    def unpatchify3D(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=self.patch_size[0],
            H_p=self.patch_size[1],
            W_p=self.patch_size[2],
            C_out=self.out_channels,
        )
        return x
