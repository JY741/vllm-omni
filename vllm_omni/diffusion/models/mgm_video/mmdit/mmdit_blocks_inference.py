# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-optimized JoinAttention for MGM-Video MMDiT.

Ported from MGM-Video-Ascend mimogpt/models/dit/mmdit_blocks_inference.py.
Provides JoinAttentionInference which overrides the forward path with
NPU-accelerated flash attention (npu_fusion_attention / laser attention)
and explicit all-to-all communication for context parallelism.
"""

import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple
from torch.nn import functional as F
from einops import rearrange, repeat
from .mmdit_communications import all_to_all
from .mmdit_blocks import JoinAttention
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

try:
    '''ascend'''
    import torch_npu
except Exception as e:
    print("training on gpu !!")

try:
    from mindiesd import attention_forward
except Exception as e:
    print("no mindiesd !!!")


class JoinAttentionInference(JoinAttention):
    def __init__(self, n_embd, n_head, dropout=0.0, fa_keep_prob=1.0, use_3d_rope=True, use_qknorm=True,
                 use_rmsnorm=False, use_context_parallelism=False, depth=-1, down_mode=None, downscale=1, index=0, laser_atten=False):
        super().__init__(
            n_embd, n_head, dropout=dropout, fa_keep_prob=fa_keep_prob, use_3d_rope=use_3d_rope, use_qknorm=use_qknorm,
            use_rmsnorm=use_rmsnorm, use_context_parallelism=use_context_parallelism, depth=depth, down_mode=down_mode,
            downscale=downscale, index=index
        )
        self.laser_atten = laser_atten
        self.rope = RotaryEmbedding(is_neox_style=False)

    def la(self, query, key, value):
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)
        attention_out = attention_forward(query, key, value, opt_mode="manual",
                                        op_type="ascend_laser_attention", layout="BNSD")
        return attention_out.transpose(1,2)

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
            # q,k,v shape-[B,N,S,D]
            n_head = q.shape[1]

            if self.laser_atten:
                if mask is not None:
                    raise NotImplementedError

                out = self.la(q, k, v)
            else:
                out = torch_npu.npu_fusion_attention(
                        q, k, v, n_head,
                        atten_mask=mask,
                        scale=(C // self.n_head) ** -0.5,
                        keep_prob=self.fa_keep_prob,
                        input_layout="BNSD",
                    )[0]
        else:
            raise NotImplementedError
        return out

    def _apply_3d_rope(self, q, k, spatial_freq):
        """Apply 3D RoPE using vllm-omni RotaryEmbedding.

        Args:
            q, k: [B, S, N, D] (BSND layout)
            spatial_freq: [(cos_h, sin_h), (cos_w, sin_w), (cos_t, sin_t)]
                each cos/sin: [S, dim/2]
        Returns:
            q, k: [B, S, N, D] with RoPE applied
        """
        (cos_h, sin_h), (cos_w, sin_w), (cos_t, sin_t) = spatial_freq
        dim_h = cos_h.shape[-1] * 2
        dim_w = cos_w.shape[-1] * 2
        dim_t = cos_t.shape[-1] * 2

        q_h, q_w, q_t = q.split([dim_h, dim_w, dim_t], dim=-1)
        k_h, k_w, k_t = k.split([dim_h, dim_w, dim_t], dim=-1)

        q_h = self.rope(q_h, cos_h, sin_h)
        k_h = self.rope(k_h, cos_h, sin_h)
        q_w = self.rope(q_w, cos_w, sin_w)
        k_w = self.rope(k_w, cos_w, sin_w)
        q_t = self.rope(q_t, cos_t, sin_t)
        k_t = self.rope(k_t, cos_t, sin_t)

        q = torch.cat([q_h, q_w, q_t], dim=-1)
        k = torch.cat([k_h, k_w, k_t], dim=-1)
        return q, k

    def infer(self, x, y, x1_cts, spatial_freq, x_padding_size, y_padding_size, mask, f, hh, ww):
        assert x1_cts is None, "Ulysses sequence parallel is not supported for window attention."

        _is_block0 = getattr(self, '_debug_block_idx', -1) == 0

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv_x_out = self.qkv_x(x)
        q_x, k_x, v_x = qkv_x_out.split(self.n_embd, dim=2)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)


        # text
        _, L, _ = y.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv_y_out = self.qkv_y(y)
        q_y, k_y, v_y = qkv_y_out.split(self.n_embd, dim=2)
        q_y = q_y.view(B, L, self.n_head, C // self.n_head)  # (B, L, nh, hs)
        k_y = k_y.view(B, L, self.n_head, C // self.n_head)  # (B, L, nh, hs)
        v_y = v_y.view(B, L, self.n_head, C // self.n_head)  # (B, T, nh, hs)

        if self.use_qknorm:
            q_x, k_x, q_y, k_y = self.qk_norm(q_x, k_x, q_y, k_y)


        if spatial_freq is not None:
            q_x, k_x = self._apply_3d_rope(q_x, k_x, spatial_freq)


        if q_x.shape[2] % self.cp_size != 0:
            self.x_padding_head = self.cp_size - q_x.shape[2] % self.cp_size
        else:
            self.x_padding_head = 0

        if q_y.shape[2] % self.cp_size != 0:
            self.y_padding_head = self.cp_size - q_y.shape[2] % self.cp_size
        else:
            self.y_padding_head = 0
        assert self.x_padding_head == self.y_padding_head

        bs, x_shard_seqlen, hc, hs = q_x.shape
        bs, y_shard_seqlen, hc, hs = q_y.shape
        un = hc // self.cp_size

        # (3*bs, seq, hc, hs)
        qkv_x = torch.cat([q_x, k_x, v_x], dim=0)
        qkv_y = torch.cat((q_y, k_y, v_y), dim=0)

        if self.x_padding_head > 0:
            qkv_x = torch.nn.functional.pad(qkv_x, (0, 0, 0, self.x_padding_head, 0, 0, 0, 0))
        if self.y_padding_head > 0:
            qkv_y = torch.nn.functional.pad(qkv_y, (0, 0, 0, self.y_padding_head, 0, 0, 0, 0))

        # (3*bs, seqlen/P, hc, hs) -> (hc, seqlen/P, 3*bs, hs) -> (un, ud, seqlen/P, 3*bs, hs), where hc = un*ud
        qkv_x = qkv_x.transpose(0, 2).contiguous().reshape(un, self.cp_size, x_shard_seqlen, 3 * bs, hs)
        qkv_y = qkv_y.transpose(0, 2).contiguous().reshape(un, self.cp_size, y_shard_seqlen, 3 * bs, hs)

        x_first_all2all_output_list = [torch.zeros(self.cp_size, 1, x_shard_seqlen, 3 * bs, hs, dtype=q_x.dtype, device=q_x.device,) for _ in range(un)]
        x_second_all2all_output_list = [torch.zeros(self.cp_size, 1, x_shard_seqlen, bs, hs, dtype=q_x.dtype, device=q_x.device,) for _ in range(un)]
        x_first_all2all_handle_list = []
        x_second_all2all_handle_list = []

        y_first_all2all_output_list = [torch.zeros(self.cp_size, 1, y_shard_seqlen, 3 * bs, hs, dtype=q_y.dtype, device=q_y.device,) for _ in range(un)]
        y_second_all2all_output_list = [torch.zeros(self.cp_size, 1, y_shard_seqlen, bs, hs, dtype=q_y.dtype, device=q_y.device,) for _ in range(un)]
        y_first_all2all_handle_list = []
        y_second_all2all_handle_list = []

        # first all to all for text
        for idx in range(un):
            ret = dist.all_to_all_single(
                y_first_all2all_output_list[idx],
                qkv_y[idx:(idx+1), ...].contiguous(),
                group=self.cp_group,
                async_op=True,
            )
            y_first_all2all_handle_list.append(ret)

        # first all to all for video
        for idx in range(un):
            ret = dist.all_to_all_single(
                x_first_all2all_output_list[idx],
                qkv_x[idx:(idx+1), ...].contiguous(),
                group=self.cp_group,
                async_op=True,
            )
            x_first_all2all_handle_list.append(ret)

        for idx in range(un):
            x_ret = x_first_all2all_handle_list[idx]
            y_ret = y_first_all2all_handle_list[idx]
            if x_ret is not None and y_ret is not None:
                x_ret.wait()
                y_ret.wait()
            else:
                raise Exception

            # (b,n,s,d)
            x_qkv_chunk = x_first_all2all_output_list[idx].permute(1,0,2,3,4).reshape(1, -1, 3 * bs, hs).permute(2,0,1,3).contiguous()
            y_qkv_chunk = y_first_all2all_output_list[idx].permute(1,0,2,3,4).reshape(1, -1, 3 * bs, hs).permute(2,0,1,3).contiguous()

            if x_padding_size > 0:
                x_qkv_chunk = x_qkv_chunk[:, :, :-x_padding_size, :]
            if y_padding_size > 0:
                y_qkv_chunk = y_qkv_chunk[:, :, :-y_padding_size, :]

            T = x_qkv_chunk.shape[2]
            L = y_qkv_chunk.shape[2]

            x_q_chunk, x_k_chunk, x_v_chunk = x_qkv_chunk.chunk(3, dim=0)
            y_q_chunk, y_k_chunk, y_v_chunk = y_qkv_chunk.chunk(3, dim=0)

            if self.downscale != 1:
                # before_fa, first CP (gather S, split head), then skiparse (reorganize S, q/k/v/mask)
                T_ori = x_qkv_chunk.shape[2]
                x_q_chunk, self.pad_len1, self.pad_len2 = self._sparse_1d(x_q_chunk, f, hh, ww)
                x_k_chunk, self.pad_len1, self.pad_len2 = self._sparse_1d(x_k_chunk, f, hh, ww)
                x_v_chunk, self.pad_len1, self.pad_len2 = self._sparse_1d(x_v_chunk, f, hh, ww)
                y_q_chunk = torch.cat([y_q_chunk] * self.sparse_n, dim=0)
                y_k_chunk = torch.cat([y_k_chunk] * self.sparse_n, dim=0)
                y_v_chunk = torch.cat([y_v_chunk] * self.sparse_n, dim=0)
                T = x_q_chunk.shape[2]

            q = torch.cat([x_q_chunk, y_q_chunk], dim=2)
            k = torch.cat([x_k_chunk, y_k_chunk], dim=2)
            v = torch.cat([x_v_chunk, y_v_chunk], dim=2)

            if isinstance(mask, list):
                # skiparse algorithm applies equivalent transform to mask
                # TODO, integrate skiparse and non-skiparse mask operations
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
                # mask is already processed in MMDiT.forward of mimogpt/models/dit/mmdit.py
                pass

            out = self.fa(q, k, v, mask, C, offload_fa=False)

            if self.downscale != 1:
                # after_fa, first skiparse (reverse reorganize S, out), then CP (restore head1, split S),
                out_x, out_y = out.split([T, L], dim=-2)
                out_x = self._reverse_sparse_1d(out_x, f, hh, ww, self.pad_len1, self.pad_len2)
                out_y = rearrange(out_y, "(q p) n x d -> p q n x d", q=self.sparse_n)
                out_y = torch.mean(out_y, dim=1)
                assert out_y.shape[0] == out_x.shape[0]
                T = T_ori
                out = torch.cat([out_x, out_y], dim=-2)

            out = out.transpose(1, 2)
            x, y = out.split([T, L], dim=1)
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, x_padding_size))
            y = torch.nn.functional.pad(y, (0, 0, 0, 0, 0, y_padding_size))

            T += x_padding_size
            L += y_padding_size

            x = rearrange(x, "b s n d->s n b d").contiguous()
            y = rearrange(y, "b s n d->s n b d").contiguous()

            _, head, batch, dim = x.shape
            x = x.reshape(self.cp_size, x_shard_seqlen, head, batch, dim).permute(0,2,1,3,4).contiguous()
            y = y.reshape(self.cp_size, y_shard_seqlen, head, batch, dim).permute(0,2,1,3,4).contiguous()

            # with torch.cuda.stream(all2all_overlap_stream):
            y_ret = dist.all_to_all_single(
                y_second_all2all_output_list[idx],
                y,
                group=self.cp_group,
                async_op=True,
            )
            y_second_all2all_handle_list.append(y_ret)

            x_ret = dist.all_to_all_single(
                x_second_all2all_output_list[idx],
                x,
                group=self.cp_group,
                async_op=True,
            )
            x_second_all2all_handle_list.append(x_ret)

        for idx in range(un):
            x_ret = x_second_all2all_handle_list[idx]
            y_ret = y_second_all2all_handle_list[idx]
            if x_ret is not None and y_ret is not None:
                x_ret.wait()
                y_ret.wait()
            else:
                raise Exception
            x_second_all2all_output_list[idx] = (
                x_second_all2all_output_list[idx].contiguous()
                .reshape(self.cp_size, x_shard_seqlen, bs, hs)
                .permute(2,1,0,3).contiguous()
            )
            y_second_all2all_output_list[idx] = (
                y_second_all2all_output_list[idx].contiguous()
                .reshape(self.cp_size, y_shard_seqlen, bs, hs)
                .permute(2,1,0,3).contiguous()
            )
        # bsnd
        x = torch.cat(x_second_all2all_output_list, dim=2)
        y = torch.cat(y_second_all2all_output_list, dim=2)

        if self.x_padding_head > 0:
            x = x[:, :, :-self.x_padding_head, :]
            y = y[:, :, :-self.y_padding_head, :]

        x = x.reshape(B, T // self.cp_size, C)
        y = y.reshape(B, L // self.cp_size, C)

        x, y = self.proj_and_drop(x, y)

        return x, y