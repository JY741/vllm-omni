# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Consolidated MGM-Video VAE model (dcn + align + spynet + autoencoder3d).

This module merges the following original files into a single module:
  - dcn.py       (DCNv2, DCN, DCN_sep_off, DCN_sep_off_local)
  - align.py     (AlignDCN, XFlowResEncoder)
  - spynet.py    (SPyNet, SPyNetBasicModule)
  - autoencoder3d.py (Upsample3D, Downsample3D, ResnetBlock3D, Encoder3D,
                      AttnBlock3D, make_attn_3d, Decoder_flow, AutoencoderKL3D, etc.)

3D-specific classes have been renamed with a "3D" suffix to avoid conflicts
with any future 2D equivalents.
"""

import math
import os
from functools import partial

import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import contextmanager
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.distributed.distributed_c10d import _get_default_group
from einops import rearrange
from math import sqrt, log
from torch.profiler import record_function

from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

from vllm_omni.diffusion.models.mgm_video.vae.vae_utils import (
    nonlinearity, time2batch, batch2time, Normalize,
    DiagonalGaussianDistribution, instantiate_from_config, all_to_all,
)

try:
    import moxing as mox
except:
    print("no moxing")


# ---------------------------------------------------------------------------
# DCN (Deformable Convolution) modules — originally from dcn.py
# ---------------------------------------------------------------------------

# from torch.autograd.function import once_differentiable
# from torch.cuda.amp import custom_fwd
# from ..models.basic_module.raft_core.utils import bilinear_sampler 


# class _DCNv2(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, input, offset, mask, weight, bias, stride, padding,
#                 dilation, deformable_groups):
#         ctx.stride = _pair(stride)
#         ctx.padding = _pair(padding)
#         ctx.dilation = _pair(dilation)
#         ctx.kernel_size = _pair(weight.shape[2:4])
#         ctx.deformable_groups = deformable_groups
#         output = _backend.dcn_v2_forward(
#             input, weight, bias, offset, mask, ctx.kernel_size[0],
#             ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0],
#             ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
#             ctx.deformable_groups)
#         ctx.save_for_backward(input, offset, mask, weight, bias)
#         return output
 
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         input, offset, mask, weight, bias = ctx.saved_tensors
#         grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
#             _backend.dcn_v2_backward(input, weight,
#                                      bias,
#                                      offset, mask,
#                                      grad_output,
#                                      ctx.kernel_size[0], ctx.kernel_size[1],
#                                      ctx.stride[0], ctx.stride[1],
#                                      ctx.padding[0], ctx.padding[1],
#                                      ctx.dilation[0], ctx.dilation[1],
#                                      ctx.deformable_groups)
 
#         return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
#             None, None, None, None,


# dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
 
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
 
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
 
    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        # return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
        #                    self.stride, self.padding, self.dilation,
        #                    self.deformable_groups)
 
 
class DCN(DCNv2):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, deformable_groups)
 
        channels_ = self.deformable_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()
 
    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
 
    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
        #                    self.stride, self.padding, self.dilation,
        #                    self.deformable_groups)
 
 
class DCN_sep_off(DCNv2):
    '''Use other features to generate offsets and masks'''
 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 deformable_groups=1, mask=True):
        super(DCN_sep_off, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, deformable_groups)
        self.mask = mask
        if mask:
            channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        else:
            channels_ = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()
 
    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
 
    def forward(self, input, fea):
        '''input: input features for deformable conv
        fea: other features used for generating offsets and mask'''
        out = self.conv_offset_mask(fea)
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
        else:
            o1, o2 = torch.chunk(out, 2, dim=1)
            mask = torch.zeros_like(o1).to(o1.device)
 
        offset = torch.cat((o1, o2), dim=1)
 
        offset_mean = torch.mean(torch.abs(offset))
        offset_mean = torch.clamp(offset_mean, 0, 100)
 
        mask = torch.sigmoid(mask)
        # return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding,
        #                    self.dilation, self.deformable_groups), offset

class DCN_sep_off_local(DCN_sep_off):
 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 deformable_groups=1, mask=True):
        super(DCN_sep_off_local, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, deformable_groups)

    def prepare_grid_xy(self, device, w, h):
        gridX, gridY = np.meshgrid(np.arange(w), np.arange(h))
        gridX = torch.tensor(gridX, requires_grad=False, device=device)
        gridY = torch.tensor(gridY, requires_grad=False, device=device)
        return gridX, gridY

    def get_grid(self, flow, grid_x, grid_y, w, h):
        u = flow[:,0]
        v = flow[:,1]
        x = grid_x.unsqueeze(0).expand_as(u).float() + u
        y = grid_y.unsqueeze(0).expand_as(v).float() + v
        x = 2 * (x / (w - 1) - 0.5)
        y = 2 * (y / (h - 1) - 0.5)
        grid = torch.stack((x, y), dim=3)
        return grid
 
    def forward(self, input, fea):
        '''input: input features for deformable conv
        fea: other features used for generating offsets and mask'''
        # print('dcn dtype', input.dtype, fea.dtype)
        out = self.conv_offset_mask(fea)
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
        else:
            o1, o2 = torch.chunk(out, 2, dim=1)
            mask = torch.zeros_like(o1).to(o1.device)
 
        offset = torch.cat((o1, o2), dim=1)

        o1 = None
        o2 = None
        out = None

        offset_mean = torch.mean(torch.abs(offset))
        offset_mean = torch.clamp(offset_mean, 0, 100)
 
        mask = torch.sigmoid(mask)
 
        channel_per_group = input.size(1) // self.deformable_groups
        aligned_fea = []
 
        output = 0.0
        chunk_size = self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]

        grid_x, grid_y = self.prepare_grid_xy(offset.device, input.size(3), input.size(2))

        for i in range(self.deformable_groups):
            for j in range(self.kernel_size[0] * self.kernel_size[1]):
                cur_idx = i * self.kernel_size[0] * self.kernel_size[1] + j
                cur_mask = mask[:, cur_idx: cur_idx + 1]
                cur_offset = self.get_grid(offset[:, cur_idx * 2: (cur_idx + 1) * 2], grid_x, grid_y, input.size(3), input.size(2))
                aligned_fea.append(
                    F.grid_sample(
                        (input[:, channel_per_group * i: channel_per_group * (i + 1)] * cur_mask).to(cur_offset.dtype),
                        cur_offset, mode='bilinear', align_corners=False).to(input.dtype)
                )

            aligned_fea = torch.cat(aligned_fea, dim=1)
            output += F.conv2d(
                aligned_fea,
                self.weight.view(
                    self.out_channels, 
                    self.in_channels * self.kernel_size[0] * self.kernel_size[1], 1, 1
                )[:, i*chunk_size:(i+1)*chunk_size, :, :],
                bias=self.bias, 
                stride=self.stride, 
                dilation=self.dilation
            ).to(torch.float32)
            aligned_fea = []
        
        mask = None
        offset = None
        input = None
        return output.to(torch.bfloat16), offset


# ---------------------------------------------------------------------------
# Alignment modules — originally from align.py
# ---------------------------------------------------------------------------

# from dcn import DCN_sep_off_local


class XFlowResEncoder(nn.Module):
    r"""The full version encoder
        to flow with SOTA optical flow.
    """

    def __init__(self, c_in=128, cmp=1, act_type='relu'):
        super(XFlowResEncoder, self).__init__()
        self._conv = partial(nn.Conv2d,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False)
        self._conv2 = partial(nn.Conv2d,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)

        self.offset_init = nn.Sequential(self._conv2(2, c_in),
                                         nn.ReLU(),
                                         self._conv(c_in, c_in), )
        # self.layer_forward = nn.Sequential(self._conv(192, 64),
        #                                    nn.ReLU(),
        #                                    self._conv(64, 64))

        self.dcn_align = AlignDCN(N=c_in, groups=8)

    def forward(self, x_ref, flow):
        # print('flow encoder input', x_ref.shape, flow.shape)
        offset_map = self.offset_init(flow)
        x_pred, _ = self.dcn_align(x_ref, offset_map)
        # offset_map_res = self.layer_forward(torch.cat([x_pred, x_ref, x_cur], dim=1))
        # offset_map = offset_map + offset_map_res
        return offset_map, x_pred
    def forward_dec(self, x_ref, flow):
        x_pred, _ = self.dcn_align(x_ref, flow)
        return  x_pred




class AlignDCN(nn.Module):
    r"""Align feature with dcn.
    """
    def __init__(self, N=3, ch=64, groups=3, ks=3, align_type='dcn'):
        super(AlignDCN, self).__init__()
        self._conv = partial(nn.Conv2d,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False)
        if align_type == 'dcn':
            self.dcnpack = DCN_sep_off_local(N, N, ks,
                                       stride=1,
                                       padding=ks//2, 
                                       deformable_groups=groups)
        elif align_type == 'dcn_nmask':
            self.dcnpack = DCN_off(N, N, ks,
                                   stride=1,
                                   padding=ks//2,
                                   deformable_groups=groups,
                                   mask=False)
        else:
            raise NotImplementedError
        self.conv = nn.Conv2d(N, N, 3, padding=1)
        self.fusion = nn.Sequential(self._conv(N*2, N),
                                    self._conv(N, N))
    
    def init_dcn(self):
        self.dcnpack.init_offset()

    def forward(self, img, flow):
        # print('before dcnpack', img.shape, flow.shape)
        img_warp, offset = self.dcnpack(img, flow)
        # print('after dcnpack', img_warp.shape, flow.shape)
        img_warp = self.conv(img_warp)
        img_warp = self.fusion(torch.cat([img, img_warp], dim=1)) + img_warp
        return img_warp, offset


# ---------------------------------------------------------------------------
# SPyNet (Optical Flow) — originally from spynet.py
# ---------------------------------------------------------------------------

class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1].to(torch.float),
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False).to(torch.bfloat16))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1].to(torch.float),
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False).to(torch.bfloat16))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow.to(torch.float),
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True).to(torch.bfloat16) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level].to(torch.float),
                        flow_up.permute(0, 2, 3, 1).to(torch.float),
                        # padding_mode='border').to(torch.bfloat16), flow_up
                        padding_mode='border'), flow_up
                ], 1).to(torch.bfloat16))

        return flow

    def forward(self, ref, supp, fix_norm = True):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        def fix_normalize(x):
            return  (x * 128 + 127.5) / 255
        if fix_norm:
                ref = fix_normalize(ref)
                supp = fix_normalize(supp)   
        # print('spynet_input', ref.min(), ref.max())
        # print('spynet_input', ref.shape, supp.shape)
        ## upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref.float(), size=(h_up, w_up), mode='bilinear', align_corners=False).to(torch.bfloat16)
        supp = F.interpolate(
            input=supp.float(),
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False).to(torch.bfloat16)

        ## compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False).to(torch.bfloat16)
        #flow = self.compute_flow(ref, supp)
        
        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


# ---------------------------------------------------------------------------
# 3D Autoencoder — originally from autoencoder3d.py
# 3D-specific classes renamed with "3D" suffix.
# ---------------------------------------------------------------------------

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class DiagonalGaussianDistribution3D(DiagonalGaussianDistribution):
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4])


def interpolate(x, scale_factor):
    return torch.cat([torch.nn.functional.interpolate(xx, scale_factor=scale_factor, mode="nearest") for xx in
                      torch.split(x, 16, dim=1)], dim=1)


# class Upsample3D(nn.Module):
#     def __init__(self, in_channels, with_conv, only_sp=False, only_temp=False):
#         super().__init__()
#         self.with_conv = with_conv
#         self.only_sp = only_sp
#         self.only_temp = only_temp
#         if self.with_conv:
#             self.conv = torch.nn.Conv3d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=0)

#     def forward(self, x, first_frame=True):
#         the_type = x.dtype
#         x = torch.nn.functional.interpolate(x.float(), scale_factor=(1.0,2.0,2.0) if self.only_sp else ((2.0,1.0,1.0) if self.only_temp else 2.0), mode="nearest").to(the_type)
#         #x = torch.nn.functional.interpolate(x[:,:,0,:,:], scale_factor=2.0).unsqueeze(2) if self.only_sp else torch.nn.functional.interpolate(x.float(), scale_factor=2.0)
#         #x = interpolate(x.float(), scale_factor=(1.0,2.0,2.0) if self.only_sp else ((2.0,1.0,1.0) if self.only_temp else 2.0)).to(the_type)
#         if self.with_conv:
#             x = torch.nn.functional.pad(x, (1,1,1,1,2,0), mode="constant", value=0)
#             x = self.conv(x)
#         return x[:,:,1:,:,:] if (not self.only_sp) and first_frame else x

class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, only_sp=False, only_temp=False):
        super().__init__()
        self.with_conv = with_conv
        self.only_sp = only_sp
        self.only_temp = only_temp
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=0)

    def forward(self, x, first_frame=True, is_casual=False):
        # print('new')
        the_type = x.dtype
        # x = torch.nn.functional.interpolate(x.float(), scale_factor=(1.0,2.0,2.0) if self.only_sp else ((2.0,1.0,1.0) if self.only_temp else 2.0), mode="nearest").to(the_type)
        # x = torch.nn.functional.interpolate(x[:,:,0,:,:], scale_factor=2.0).unsqueeze(2) if self.only_sp else torch.nn.functional.interpolate(x.float(), scale_factor=2.0)
        # x = interpolate(x.float(), scale_factor=(1.0,2.0,2.0) if self.only_sp else ((2.0,1.0,1.0) if self.only_temp else 2.0)).to(the_type)
        b, c, d, h, w = x.shape
        if not is_casual:
            x = x.view(b, -1, h, w)
        else:
            x = x.reshape(b, -1, h, w)
        if not self.only_temp:
            # interpolate bf16输入仅适用推理
            if not self.training:
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest").to(torch.bfloat16)
            else:
                x = torch.nn.functional.interpolate(x.float(), scale_factor=2.0, mode="nearest").to(torch.bfloat16)
        h, w = x.shape[-2:]
        if not is_casual:
            x = x.view(b, c, d, -1)
        else:
            x = x.reshape(b, c, d, -1)
        if not self.only_sp:
            # interpolate bf16输入仅适用推理
            if not self.training:
                x = torch.nn.functional.interpolate(x, scale_factor=(2.0, 1.0), mode="nearest").to(torch.bfloat16)
            else:
                x = torch.nn.functional.interpolate(x.float(), scale_factor=(2.0, 1.0), mode="nearest").to(torch.bfloat16)
        if not is_casual:
            x = x.view(b, c, -1, h, w)
        else:
            x = x.reshape(b, c, -1, h, w)

        if self.with_conv:
            x = torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
            x = self.conv(x)
        return x[:, :, 1:, :, :] if (not self.only_sp) and first_frame else x


class Downsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, only_sp=False, only_temp=False):
        super().__init__()
        self.with_conv = with_conv
        self.only_sp = only_sp
        self.only_temp = only_temp
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=(1, 2, 2) if only_sp else ((2, 1, 1) if only_temp else 2),
                                        padding=0)

    def forward(self, x, first_frame=True):
        first_frame = (x.shape[2] % 2 == 1)
        if self.with_conv:
            pad = ((0, 1, 0, 1, 2, 0) if first_frame else (0, 1, 0, 1, 1, 0)) if not self.only_temp else (
            1, 1, 1, 1, 2, 0)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock_Causal(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=0)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=0)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=0)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h, batch_size = time2batch(h)
        h = self.norm1(h)
        h = batch2time(h, batch_size)
        h = nonlinearity(h)
        h = torch.nn.functional.pad(h, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h, batch_size = time2batch(h)
        h = self.norm2(h)
        h = batch2time(h, batch_size)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = torch.nn.functional.pad(h, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=0)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=0)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=0)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = torch.nn.functional.pad(h, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = torch.nn.functional.pad(h, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.device_type = os.environ.get("device_type", "npu")
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


# def nonlinearity(x):
#     # swish
#     #return x*torch.sigmoid(x)
#     return torch.nn.functional.silu(x)


# def Normalize(in_channels, num_groups=32):
#     return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2d(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, is_casual=False):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample2d(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock2d(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class TemporalAttention(nn.Module):  # 2D + 1D attention
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)

        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)
        self.to_out = nn.Linear(in_channels, in_channels)

        self.sp_att = AttnBlock3D(in_channels)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.sp_att(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        h_ = x
        h_ = self.norm(h_)
        h_ = rearrange(h_, "(b t) c h w -> (b h w) t c", t=t)
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)
        attn_out = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=None, dropout_p=0.0)[0]
        attn_out = rearrange(attn_out, "(b h w) t c -> b c t h w", h=h, w=w)
        return attn_out + x


class SpatialTempAttention(AttnBlock3D):  # 3D attention
    def ori_forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> b c (t h) w")
        x = super().forward(x)
        x = rearrange(x, "b c (t h) w -> b c t h w", t=t)
        return x

    def forward_sdpa(self, x):
        b, c, t, h, w = x.shape
        h_ = x
        h_ = self.norm(h_)
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        q = rearrange(q, "(b t) c h w -> b (t h w) c", b=b)
        k = rearrange(k, "(b t) c h w -> b (t h w) c", b=b)
        v = rearrange(v, "(b t) c h w -> b (t h w) c", b=b)

        attn_out = F.scaled_dot_product_attention(q.unsqueeze(0).contiguous(), k.unsqueeze(0).contiguous(),
                                                  v.unsqueeze(0).contiguous(), attn_mask=None, dropout_p=0.0)[0]
        attn_out = rearrange(attn_out, "b (t h w) c -> (b t) c h w", h=h, w=w)
        attn_out = self.proj_out(attn_out)
        attn_out = rearrange(attn_out, "(b t) c h w -> b c t h w", t=t)
        return attn_out + x

    def forward_npu(self, x):
        b, c, t, h, w = x.shape
        h_ = x
        h_ = self.norm(h_)
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        q = rearrange(q, "(b t) c h w -> b (t h w) c", b=b)
        k = rearrange(k, "(b t) c h w -> b (t h w) c", b=b)
        v = rearrange(v, "(b t) c h w -> b (t h w) c", b=b)

        if dist.get_world_size() > 1:
            kv = torch.stack([k, v], dim=0)
            all_shapes = [torch.zeros([4], device=x.device, dtype=torch.int32) for _ in range(dist.get_world_size())]
            shape_tensor = torch.tensor(kv.shape, device=x.device, dtype=torch.int32)
            dist.all_gather(all_shapes, shape_tensor)
            max_axis_val = max([i[2] for i in all_shapes])
            b, _, before_seq, c = kv.shape

            if max_axis_val - before_seq > 0:
                kv = torch.nn.functional.pad(kv, (0, 0, 0, max_axis_val - before_seq), mode="constant", value=0)

            tensor_list = [torch.zeros(kv.shape, device=x.device, dtype=kv.dtype) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list, kv)

            for idx, shape in enumerate(all_shapes):
                tensor_list[idx] = tensor_list[idx][:, :, :shape[2], :]

            k = torch.cat([kv_tmp[0] for kv_tmp in tensor_list], dim=1)
            v = torch.cat([kv_tmp[1] for kv_tmp in tensor_list], dim=1)

        attn_out = torch_npu.npu_fusion_attention(
                q, k, v, 1,
                atten_mask=None,
                scale=(c // 1) ** -0.5,
                keep_prob=1.0,
                input_layout="BSH",  # BNSD
            )[0]

        attn_out = rearrange(attn_out, "b (t h w) c -> (b t) c h w", h=h, w=w)
        attn_out = self.proj_out(attn_out)
        attn_out = rearrange(attn_out, "(b t) c h w -> b c t h w", t=t)
        return attn_out + x

    def forward(self, x):
        if self.device_type == "npu":
            return self.forward_npu(x)
        elif self.device_type == "gpu":
            return self.forward_sdpa(x)
        else:
            raise NotImplementedError


class SpatialAttention(AttnBlock3D):  # 2D attention
    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


def make_attn_3d(in_channels, attn_type="vanilla"):
    assert attn_type in ["3d", "2d+1d", "2d", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "3d":
        return SpatialTempAttention(in_channels)
    elif attn_type == "2d+1d":
        return TemporalAttention(in_channels)
    elif attn_type == "2d":
        print('2D Attention')
        return SpatialAttention(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        pass


class Encoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", num_resize=3,
                 spynet_pretrained_local=None,
                 spynet_pretrained_cloud=None,
                 # N次时间上下采样, 0,1,2,3,4,5
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        # self.conv_in = torch.nn.Conv3d(in_channels,
        #                                self.ch,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=0)
        # # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch // 2,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            print('block_in', block_in)
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if i_level == 1 and i_block == 0:
                    block_in = 192
                if i_level == 0:
                    block_in, block_out = 64, 64
                    block.append(ResnetBlock2d(in_channels=block_in,
                                               out_channels=block_out,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout))
                else:
                    block.append(ResnetBlock3D(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_3d(block_in, attn_type=attn_type))
            if i_level > 2:
                attn.append(make_attn_3d(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level == 0:
                down.downsample = Downsample2d(block_in, resamp_with_conv)
            elif i_level != self.num_resolutions - 1:
                down.downsample = Downsample3D(block_in, resamp_with_conv, only_sp=(
                            i_level < (self.num_resolutions - 1 - num_resize + 1)))  # 只有第一层是only_sp
                curr_res = curr_res // 2
            elif num_resize > 2:
                down.downsample = Downsample3D(block_in, resamp_with_conv, only_temp=True)
            self.down.append(down)
            # print('####down_layers###', self.down)
        ##########flow
        self.flow_enc = XFlowResEncoder(c_in=64)

        if not os.path.exists(spynet_pretrained_local):
            try:
                mox.file.copy(spynet_pretrained_cloud, spynet_pretrained_local)
            except:
                raise Exception("failed load ckpt from {}...".format(spynet_pretrained_local))

        self.spynet_pretrained = spynet_pretrained_local
        self.flow_gen = SPyNet(pretrained=self.spynet_pretrained)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn_3d(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        if num_resize > 4:
            self.mid.downsample = Downsample3D(block_in, resamp_with_conv, only_temp=True)
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=0)
        # print('Training', self.training)

    def middle_end_forward(self, hs, temb):
        # middle
        h = self.mid.block_1(hs[-1], temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = torch.nn.functional.pad(h, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        return self.conv_out(h)

    #     def flow_estimate_parallel(self, video_seq):
    #         _,_,t,_,_ = video_seq.shape # (b, c, 17, h, w)
    #         assert t >1
    #         video_seq = video_seq.unfold(dimension=2, size=2, step=1)  # (b, c, t-1, 2, h, w)
    #         print('flow_estimate', video_seq.shape, video_seq.min(), video_seq.max())
    #         video_seq = rearrange(video_seq, 'b c t2 t1 h w -> (b t2) t1 c h w')
    #         print('flow_estimate', video_seq.shape)
    #         video_seq = video_seq.to(torch.float)
    #         _, flow = self.flow_gen(video_seq[:,0], video_seq[:,1]) # flow(i, i-1)
    #         print('flow', flow.shape)

    #         # flow = rearrange(flow, '(b t2) c h w -> b c t2 h w')

    #         # if pre
    #         # video_pre = self.align.....
    #         return flow

    def flow_estimate_parallel(self, video_seq, is_distributed=False):
        _, _, t, _, _ = video_seq.shape  # (b, c, 17, h, w)
        # assert t >1
        if is_distributed:
            ########################### split T frame offset ###########################
            if dist.get_rank() == 0:
                video_seq = torch.cat((video_seq[:, :, :1], video_seq), dim=2)
            ########################### split T frame offset ###########################
        else:
            video_seq = torch.cat((video_seq[:, :, :1], video_seq), dim=2)

        n, t, c, h, w = video_seq.size()
        lrs_1 = video_seq[:, :, :-1, :, :]
        lrs_2 = video_seq[:, :, 1:, :, :]

        lrs_1 = rearrange(lrs_1, 'b c t2 h w -> (b t2) c h w')
        lrs_2 = rearrange(lrs_2, 'b c t2 h w -> (b t2) c h w')

        flows_backward = self.flow_gen(lrs_2, lrs_1)  # flow(i, i-1)

        #### verification with align2d
        # print('###############verificaiton', )
        #         self.align2d = Align2D()
        #         pred1 = self.align2d(lrs_2, flows_backward)
        #         pred2 = self.align2d(lrs_1, flows_backward)

        #         print('flow_0 flow1', flows_backward[:,:,0].mean(), flows_backward[:,:,1].mean(), flows_backward[:,:,2].mean())
        #         print('du_forard',(pred1-lrs_1).abs().mean(), (pred2-lrs_2).abs().mean())
        #         print('dif_1_2',  (lrs_2-lrs_1).abs().mean())
        #         assert 0
        # flow = rearrange(flow, '(b t2) c h w -> b c t2 h w')

        # if pre
        # video_pre = self.align.....
        return flows_backward

    def forward_res_0(self, x, temb, is_distributed=False):
        # xp = torch.nn.functional.pad(x, (1,1,1,1,2,0), mode="constant", value=0)
        # downsampling
        b, c, t, H, W = x.shape
        flow_x = self.flow_estimate_parallel(x, is_distributed=is_distributed)  # no padding input
        x = rearrange(x, "b c t h w -> (b t) c h w")

        h = self.conv_in(x)
        for i_block in range(self.num_res_blocks):
            h = self.down[0].block[i_block](h, temb)
        h = self.down[0].downsample(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", t=t)

        ######flow align  128->64->2D
        if is_distributed:
            ########################### split T frame offset ###########################
            if dist.get_rank() == 0:
                pframe = torch.cat((h[:, :, :1], h[:, :, :-1]), dim=2)  # I帧前copy一帧
            else:
                pframe = h[:, :, :-1]
            ########################### split T frame offset ###########################
        else:
            pframe = torch.cat((h[:, :, :1], h[:, :, :-1]), dim=2)  # I帧前copy一帧
        # print('l1_enc_before_warp', (h-pframe).abs().mean())
        pframe = rearrange(pframe, "b c t h w -> (b t) c h w")
        # print('flow_x', flow_x.shape,flow_x.min(), flow_x.max())
        flow_f, pframe = self.flow_enc(pframe, flow_x)  # h[:,1:] 非首帧
        # print('pred_l', pframe.shape, flow_f.shape) # ([16, 128, 128, 128])

        if is_distributed:
            if dist.get_rank() == 0:
                pframe = rearrange(pframe, "(b t) c h w ->b c t h w", t=t)
                flow_f = rearrange(flow_f, "(b t) c h w ->b c t h w", t=t)
            else:
                pframe = rearrange(pframe, "(b t) c h w ->b c t h w", t=t - 1)
                flow_f = rearrange(flow_f, "(b t) c h w ->b c t h w", t=t - 1)

            ########################### split T frame offset ###########################
            if dist.get_rank() == 0:
                h = torch.cat((h, pframe, flow_f), dim=1).contiguous()
            else:
                h = torch.cat((h[:, :, 1:], pframe, flow_f), dim=1).contiguous()
            ########################### split T frame offset ###########################
        else:
            pframe = rearrange(pframe, "(b t) c h w ->b c t h w", t=t)
            flow_f = rearrange(flow_f, "(b t) c h w ->b c t h w", t=t)
            h = torch.cat((h, pframe, flow_f), dim=1).contiguous()
        return h, flow_f

    def forward_res_1(self, h, temb):
        for i_block in range(self.num_res_blocks):
            h = self.down[1].block[i_block](h, temb)
        h = self.down[1].downsample(h)
        return h

    def forward_res_2(self, h, temb):
        for i_block in range(self.num_res_blocks):
            h = self.down[2].block[i_block](h, temb)
        h = self.down[2].downsample(h)
        return h

    def forward_res_3(self, h, temb):
        for i_block in range(self.num_res_blocks):
            h = self.down[3].block[i_block](h, temb)
        h = self.down[3].downsample(h)
        h = self.down[3].attn[0](h)
        return h

    def gather_t_split_h(self, x):
        all_shapes = [torch.zeros([5], device=torch.cuda.current_device(), dtype=torch.int32) for _ in range(dist.get_world_size())]
        shape_tensor = torch.tensor(x.shape, device=torch.cuda.current_device(), dtype=torch.int32)
        dist.all_gather(all_shapes, shape_tensor)
        max_axis_val = max([i[2] for i in all_shapes])
        min_axis_val = min([i[2] for i in all_shapes])
        b, c, t_before, h, w = x.shape
        max_shape = []

        if max_axis_val - t_before == 0:
            assert dist.get_rank() == dist.get_world_size()-1,"rank={}--->{}".format(dist.get_rank(), max_axis_val - t_before)  # causal vae t轴切分最后一张卡一定除不尽
            x, rest = torch.split(x, min_axis_val, dim=2)
            rest_shape_tensor = torch.tensor(rest.shape, device=torch.cuda.current_device(), dtype=torch.int32)
        else:
            rest_shape_tensor = torch.zeros([5], device=torch.cuda.current_device(), dtype=torch.int32)
            rest = None

        dist.broadcast(rest_shape_tensor, src=int(dist.get_world_size() - 1))

        if rest is None:
            rest = torch.zeros(rest_shape_tensor.tolist(), device=torch.cuda.current_device(), dtype=torch.bfloat16)

        x = all_to_all(x, _get_default_group(), scatter_dim=3, gather_dim=2)

        dist.broadcast(rest.contiguous(), src=int(dist.get_world_size() - 1))

        chunk_size = rest_shape_tensor[3] // dist.get_world_size()
        x = torch.cat([x, rest[:, :, :, dist.get_rank() * chunk_size:(dist.get_rank() + 1) * chunk_size, :]], dim=2)

        return x

    def forward(self, x, is_distributed=False):
        temb = None

        if self.training:
            with record_function(f"Encoder_0"):
                h, flow_f = torch.utils.checkpoint.checkpoint(self.forward_res_0, x, temb, use_reentrant=False)
            # print("tmp_shape_0", h.shape)
            with record_function(f"Encoder_1"):
                h = torch.utils.checkpoint.checkpoint(self.forward_res_1, h, temb, use_reentrant=False)
            # print("tmp_shape_1", h.shape)
            with record_function(f"Encoder_2"):
                h = torch.utils.checkpoint.checkpoint(self.forward_res_2, h, temb, use_reentrant=False)
            # print("tmp_shape_2", h.shape)
            with record_function(f"Encoder_3"):
                h = torch.utils.checkpoint.checkpoint(self.forward_res_3, h, temb, use_reentrant=False)
            # print("tmp_shape_3", h.shape)
            with record_function("middle_end_forward"):
                h = torch.utils.checkpoint.checkpoint(self.middle_end_forward, [h], temb, use_reentrant=False)
            # print("tmp_shape_4", h.shape)

        else:
            h, flow_f = self.forward_res_0(x, temb)  # [1, 3, 41, 480, 720], None
            if is_distributed:
                if dist.get_world_size() > 1:
                    h = self.gather_t_split_h(h)
            h = self.forward_res_1(h, temb)
            h = self.forward_res_2(h, temb)
            h = self.forward_res_3(h, temb)
            h = self.middle_end_forward([h], temb)
        return h, flow_f


class Decoder_flow(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, num_resize=3,
                 # N次时间上下采样, 0,1,2,3,4,5
                 attn_type="vanilla", is_casual=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.is_casual = is_casual

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=0)

        self.align = AlignDCN(N=ch * ch_mult[0] // 2, groups=8)
        # self.refine = ResnetBlock3D(in_channels=ch*ch_mult[0]//2,
        #                                out_channels=ch*ch_mult[0]//2,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)
        self.mergeip = torch.nn.Conv2d(ch * ch_mult[0],
                                       ch * ch_mult[0] // 2,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)  # padding 最好写在外面

        # middle
        self.mid = nn.Module()
        if is_casual:
            self.mid.block_1 = ResnetBlock_Causal(in_channels=block_in,
                                              out_channels=block_in,
                                              temb_channels=self.temb_ch,
                                              dropout=dropout)
            self.mid.attn_1 = make_attn_3d(block_in, attn_type='2d')
            self.mid.block_2 = ResnetBlock_Causal(in_channels=block_in,
	                                              out_channels=block_in,
	                                              temb_channels=self.temb_ch,
	                                              dropout=dropout)
        else:
            self.mid.block_1 = ResnetBlock3D(in_channels=block_in,
	                                       out_channels=block_in,
	                                       temb_channels=self.temb_ch,
	                                       dropout=dropout)
            self.mid.attn_1 = make_attn_3d(block_in, attn_type=attn_type)
            self.mid.block_2 = ResnetBlock3D(in_channels=block_in,
	                                       out_channels=block_in,
	                                       temb_channels=self.temb_ch,
										   dropout=dropout)
        if num_resize > 3:
            self.mid.upsample = Upsample3D(block_in, resamp_with_conv, only_temp=True)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_level == 0 and i_block == 1:
                    block_out = block_out // 2
                    block_in = block_in // 2
                    print('block_out_i1', block_out)
                if i_level == 0 and i_block != 0:
                    block.append(ResnetBlock2d(in_channels=block_in,
                                               out_channels=block_out,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout))
                else:
                    if is_casual:
                        block.append(ResnetBlock_Causal(in_channels=block_in,
					                                    out_channels=block_out,
														temb_channels=self.temb_ch,
														dropout=dropout))
                    else:
                        block.append(ResnetBlock3D(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                # if i_level==0 and i_block ==0:
                #     block_out =block_out//2
                #     print('block_out_i2', block_out)
                block_in = block_out
                if curr_res in attn_resolutions:
                    if is_casual:
                        attn.append(make_attn_3d(block_in, attn_type='2d'))
                    else:
                        attn.append(make_attn_3d(block_in, attn_type=attn_type))
            if i_level > 2:
                if is_casual:
                    attn.append(make_attn_3d(block_in, attn_type='2d'))
                else:
                    attn.append(make_attn_3d(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level < 4:
                # up.upsample = Upsample3D(block_in, resamp_with_conv, only_sp=(i_level>num_resize))
                if i_level == 0:
                    up.upsample = Upsample2d(block_in, resamp_with_conv)
                else:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, only_sp=(i_level < 1), only_temp=(i_level > 2))
                curr_res = curr_res * 2
            elif num_resize > 3:
                up.upsample = Upsample3D(block_in, resamp_with_conv, only_temp=True)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv3d(block_in,
        #                                 out_ch,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=0)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def post_process(self, h):
        _rank = dist.get_rank() if dist.is_initialized() else 0
        #print(f"[R{_rank}][VAE-DBG] post_process input: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")
        h = self.norm_out(h)
        #print(f"[R{_rank}][VAE-DBG] post_process after norm_out: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")
        h = nonlinearity(h)
        # h = torch.nn.functional.pad(h, (1,1,1,1,2,0), mode="constant", value=0)
        h = self.conv_out(h)
        #print(f"[R{_rank}][VAE-DBG] post_process after conv_out: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")
        if self.tanh_out:
            h = torch.tanh(h)
        return h

    def pre_process(self, z, temb):
        _rank = dist.get_rank() if dist.is_initialized() else 0
        # z to block_in
        #print(f"[R{_rank}][VAE-DBG] pre_process input: shape={list(z.shape)}, dtype={z.dtype}, mean={z.float().mean().item():.6f}")
        z = torch.nn.functional.pad(z, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        #print(f"[R{_rank}][VAE-DBG] pre_process after F.pad: shape={list(z.shape)}, dtype={z.dtype}, mean={z.float().mean().item():.6f}")
        h = self.conv_in(z)
        #print(f"[R{_rank}][VAE-DBG] pre_process after conv_in: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

        # middle
        h = self.mid.block_1(h, temb)
        #print(f"[R{_rank}][VAE-DBG] pre_process after mid.block_1: shape={list(h.shape)}, mean={h.float().mean().item():.6f}")
        h = self.mid.attn_1(h)
        #print(f"[R{_rank}][VAE-DBG] pre_process after mid.attn_1: shape={list(h.shape)}, mean={h.float().mean().item():.6f}")
        h = self.mid.block_2(h, temb)
        #print(f"[R{_rank}][VAE-DBG] pre_process after mid.block_2: shape={list(h.shape)}, mean={h.float().mean().item():.6f}")

        if hasattr(self.mid, 'upsample'):
            h = self.mid.upsample(h, is_casual=self.is_casual)
            #print(f"[R{_rank}][VAE-DBG] pre_process after mid.upsample: shape={list(h.shape)}, mean={h.float().mean().item():.6f}")

        return h

    def alignlatent(self, h, flow, is_distributed=False):
        _rank = dist.get_rank() if dist.is_initialized() else 0
        b, c, t, H, W = h.shape
        #print(f"[R{_rank}][VAE-DBG] alignlatent input: h shape={list(h.shape)}, flow shape={list(flow.shape)}, is_distributed={is_distributed}")
        if is_distributed:
            ########################### split T frame offset ###########################
            if dist.get_rank() == 0:
                pframe = torch.cat((h[:, :, :1], h[:, :, :-1]), dim=2)
            else:
                pframe = h[:, :, :-1]
                flow = flow[:, :, 1:]
            ########################### split T frame offset ###########################
        else:
            pframe = torch.cat((h[:, :, :1], h[:, :, :-1]), dim=2)

        #print(f"[R{_rank}][VAE-DBG] alignlatent pframe: shape={list(pframe.shape)}, mean={pframe.float().mean().item():.6f}, flow: shape={list(flow.shape)}, mean={flow.float().mean().item():.6f}")

        if dist.get_rank() == 0:
            pframe = rearrange(pframe, "b c t h w -> (b t) c h w", t=t)
            flow = rearrange(flow, "b c t h w -> (b t) c h w", t=t)
            h = rearrange(h, "b c t h w -> (b t) c h w", t=t)
        else:
            pframe = rearrange(pframe, "b c t h w -> (b t) c h w", t=t - 1)
            flow = rearrange(flow, "b c t h w -> (b t) c h w", t=t - 1)
            h = rearrange(h, "b c t h w -> (b t) c h w", t=t)

        # cloned_h = h.clone()
        # cloned_flow = flow.clone()
        h1, _ = self.align(pframe, flow)
        #print(f"[R{_rank}][VAE-DBG] alignlatent after align: h1 shape={list(h1.shape)}, mean={h1.float().mean().item():.6f}, h shape={list(h.shape)}, mean={h.float().mean().item():.6f}")

        # print('dec_pframe_shape', h1.shape)
        if dist.get_rank() == 0:
            h1 = self.mergeip(torch.cat((h1, h), dim=1))
        else:
            # b=1
            h1 = self.mergeip(torch.cat((h1, h[1:]), dim=1))
        #print(f"[R{_rank}][VAE-DBG] alignlatent after mergeip: shape={list(h1.shape)}, mean={h1.float().mean().item():.6f}, min={h1.min().item():.6f}, max={h1.max().item():.6f}")
        return h1

    def get_p2p_groups(self):
        groups = {"send": [], "recv": []}
        world_size = dist.get_world_size()
        # 例如8卡并行，第一组通信0->1 2->3 4->5 6->7，第二组通信1->2 3->4 5->6
        send_group1 = {}
        recv_group1 = {}
        for idx in range(0, world_size - 1, 2):
            send_group1[idx] = idx + 1
            recv_group1[idx + 1] = idx
        groups["send"].append(send_group1)
        groups["recv"].append(recv_group1)
        send_group2 = {}
        recv_group2 = {}
        for idx in range(1, world_size - 1, 2):
            send_group2[idx] = idx + 1
            recv_group2[idx + 1] = idx
        groups["send"].append(send_group2)
        groups["recv"].append(recv_group2)
        return groups

    def frame_p2p(self, x):
        rank = dist.get_rank()
        groups = self.get_p2p_groups()

        # x.shape=[b,c,t,h,w]
        for group_send, group_recv in zip(groups["send"], groups["recv"]):
            if rank in group_send:
                data_send = x[:, :, -1:, :, :].contiguous()
                dist.send(data_send, group_send[rank], group=None)

            if rank in group_recv:
                recv_buffer_head = torch.zeros_like(x[:, :, 0:1, :, :])
                dist.recv(recv_buffer_head, group_recv[rank], group=None)

        if dist.get_rank() != 0:
            x = torch.cat([recv_buffer_head, x], dim=2)

        return x

    # motion_vae decode设计是3D部分切H轴，2D部分切T轴，这里使用all_to_all实现切H转切T的实现
    def gather_h_split_t(self, x):
        all_shapes = [torch.zeros([5], device=torch.cuda.current_device(), dtype=torch.int32) for _ in range(dist.get_world_size())]
        shape_tensor = torch.tensor(x.shape, device=torch.cuda.current_device(), dtype=torch.int32)
        # decoder多卡并行，被切分的轴存在不能整除的场景，从而会导致最后一张卡的shape与其他卡不同，
        # 在all_to_all之前会all_gather获取各个卡的shape信息，最大的shape补齐
        dist.all_gather(all_shapes, shape_tensor)
        max_axis_val = max([i[3] for i in all_shapes])
        min_axis_val = min([i[3] for i in all_shapes])
        b, c, t, h_before, w = x.shape
        world_size = dist.get_world_size()

        # 因为切H转切T，需要考虑T是否被整除，不能整除进行padding
        if t % world_size != 0:
            padding_size = world_size - t % world_size
        else:
            padding_size = 0
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padding_size, 0, 0, 0, 0), mode="constant", value=0)

        if max_axis_val != min_axis_val:  # H轴不能被整除场景
            if max_axis_val - h_before == 0:
                # H轴切分例如H=135被8卡切分，chunk=135//8，所以只有最后一张卡shape是不同的，而且最后一张卡shape一定是最多的
                assert dist.get_rank() == world_size-1,"rank={}--->{}".format(dist.get_rank(), max_axis_val - h_before)
                # all_to_all先处理与其他7张卡相同的shape
                x, rest = torch.split(x, min_axis_val, dim=3)
                rest_shape_tensor = torch.tensor(rest.shape, device=torch.cuda.current_device(), dtype=torch.int32)
            else:
                rest_shape_tensor = torch.zeros([5], device=torch.cuda.current_device(), dtype=torch.int32)
                rest = None

        x = all_to_all(x, _get_default_group(), scatter_dim=2, gather_dim=3)

        if padding_size > 0 and dist.get_rank() == world_size - 1:  # 因为在T轴padding只会影响最后一张卡，所以只在最后一张卡在all_to_all之后去除padding部分
            x = x[:, :, :-padding_size, :, :]

        if max_axis_val != min_axis_val:  # H轴不能被整除场景
            # 最后一张卡的shape剩下的部分tensor在T轴切分下应该拼接到all_to_all后每张卡tensor的最后，broadcast后按照rank取出对应的tensor进行拼接
            dist.broadcast(rest_shape_tensor, src=int(world_size - 1))
            if rest is None:
                rest = torch.zeros(rest_shape_tensor.tolist(), device=torch.cuda.current_device(), dtype=torch.bfloat16)

            dist.broadcast(rest.contiguous(), src=int(world_size - 1))
            chunk_size = rest_shape_tensor[2] // world_size

            if dist.get_rank() == world_size - 1:
                rest = rest[:, :, dist.get_rank() * chunk_size:(dist.get_rank() + 1) * chunk_size - padding_size, :, :]
            else:
                rest = rest[:, :, dist.get_rank() * chunk_size:(dist.get_rank() + 1) * chunk_size, :, :]

            x = torch.cat([x, rest], dim=3)  # x.shape=[b,c,t,h,w]
        return x

    def forward(self, h, first_frame=True, is_distributed=False):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = h.shape
        _rank = dist.get_rank() if dist.is_initialized() else 0

        # timestep embedding
        temb = None
        if self.training:
            with record_function("pre_process"):
                h = torch.utils.checkpoint.checkpoint(self.pre_process, h, temb, use_reentrant=False)
        else:
            h = self.pre_process(h, temb)
        #print(f"[R{_rank}][VAE-DBG] after pre_process: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

        # upsampling
        for i_level in reversed(range(1, self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if self.training:
                    with record_function(f"block_{i_level}_{i_block}"):
                        h = torch.utils.checkpoint.checkpoint(self.up[i_level].block[i_block], h, temb,
                                                              use_reentrant=False)
                else:
                    h = self.up[i_level].block[i_block](h, temb)
                #print(f"[R{_rank}][VAE-DBG] after up[{i_level}].block[{i_block}]: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

            if len(self.up[i_level].attn) > 0:
                h = self.up[i_level].attn[0](h)
                #print(f"[R{_rank}][VAE-DBG] after up[{i_level}].attn: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")
            if hasattr(self.up[i_level], 'upsample'):
                if self.training:
                    with record_function(f"block_{i_level}_{i_block}_upsample"):
                        h = torch.utils.checkpoint.checkpoint(self.up[i_level].upsample, h, first_frame)
                else:
                    h = self.up[i_level].upsample(h, first_frame=first_frame, is_casual=self.is_casual)
                #print(f"[R{_rank}][VAE-DBG] after up[{i_level}].upsample: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

        # end
        ############i_level==0
        i_level = 0
        _, _, t, _, _ = h.shape
        for i_block in range(self.num_res_blocks + 1):
            if self.training:
                with record_function(f"block_{i_level}_{i_block}"):
                    h = torch.utils.checkpoint.checkpoint(self.up[i_level].block[i_block], h, temb, use_reentrant=False)
            else:
                h = self.up[i_level].block[i_block](h, temb)
            #print(f"[R{_rank}][VAE-DBG] after up[0].block[{i_block}]: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")
            ########################flow align
            if i_block == 0:  # 第一层 ch128 shape128
                if dist.get_world_size() > 1:
                    #print(f"[R{_rank}][VAE-DBG] before gather_h_split_t: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}")
                    h = self.gather_h_split_t(h)
                    #print(f"[R{_rank}][VAE-DBG] after gather_h_split_t: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}")
                    # 在不做多卡并行情况下motion_vae2D部分需要后一帧参考前一帧的操作，但在切分后，每张卡的第一帧参考不到前一帧，通过send/recv方式把
                    # 前一张卡的最后一帧cat到当前卡的第一帧位置
                    h = self.frame_p2p(h)
                    #print(f"[R{_rank}][VAE-DBG] after frame_p2p: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}")
                h, flow = torch.chunk(h, chunks=2, dim=1)
                #print(f"[R{_rank}][VAE-DBG] after chunk: h shape={list(h.shape)}, flow shape={list(flow.shape)}, h_mean={h.float().mean().item():.6f}, flow_mean={flow.float().mean().item():.6f}")
                h = self.alignlatent(h, flow, is_distributed=is_distributed)
                #print(f"[R{_rank}][VAE-DBG] after alignlatent: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")
                flow = None

        if len(self.up[i_level].attn) > 0:
            h = self.up[i_level].attn[0](h)
            #print(f"[R{_rank}][VAE-DBG] after up[0].attn: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}")
        if hasattr(self.up[i_level], 'upsample'):
            if self.training:
                with record_function(f"block_{i_level}_{i_block}_upsample"):
                    h = torch.utils.checkpoint.checkpoint(self.up[i_level].upsample, h)
            else:
                # h = self.up[i_level].upsample(h, first_frame=first_frame)
                h = self.up[i_level].upsample(h, is_casual=self.is_casual)
            #print(f"[R{_rank}][VAE-DBG] after up[0].upsample: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

        if self.give_pre_end:
            return h

        if self.training:
            with record_function("post_process"):
                h = torch.utils.checkpoint.checkpoint(self.post_process, h, use_reentrant=False)
        else:
            h = self.post_process(h)
        #print(f"[R{_rank}][VAE-DBG] after post_process: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

        h = rearrange(h, "(b t) c h w -> b c t h w", t=h.shape[0])
        #print(f"[R{_rank}][VAE-DBG] after rearrange: shape={list(h.shape)}, dtype={h.dtype}, mean={h.float().mean().item():.6f}, min={h.min().item():.6f}, max={h.max().item():.6f}")

        return h, flow


class AutoencoderKL3D(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_checkpoint=False,
                 inflation=True,
                 mode="repeat",
                 ):
        super().__init__()
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        self.image_key = image_key
        self.encoder = Encoder3D(**ddconfig, use_checkpoint=self.use_checkpoint)
        self.decoder = Decoder_flow(**ddconfig, use_checkpoint=self.use_checkpoint)
        print(ignore_keys)
        print('num_resize=', ddconfig["num_resize"])
        # self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        # tile
        self.use_tiling = False
        self.tile_sample_min_size = 256
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(ddconfig["ch_mult"]) - 1)))
        self.tile_overlap_factor = 0.25
        self.input_shape = None
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            if inflation:
                self.init_from_inflation(ckpt_path, mode=mode)
            else:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        print('spynet_weight', self.encoder.flow_gen.basic_module[0].basic_module[1].conv.weight.abs().mean())

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                # print('ik', ik)
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"VAE-KL: Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        # tile
        if self.use_tiling and (
                x.shape[-1] > self.tile_sample_min_size
                or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.tiled_encode2d(x)
        # total
        h, flow = self.encoder(x)  # [1,3,41,480,720]
        moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution3D(moments)
        return moments  # posterior, flow

    def decode(self, z, first_frame, is_distributed=False):
        # tile
        if self.use_tiling and (
                z.shape[-1] > self.tile_latent_min_size
                or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.tiled_decode2d(z, first_frame=first_frame)
        # total
        # print(f"🔥 AutoencoderKL3D.decode INPUT: shape={list(z.shape)}, dtype={z.dtype}, mean={z.float().mean().item():.6f}, min={z.min().item():.6f}, max={z.max().item():.6f}, use_tiling={self.use_tiling}")
        z = self.post_quant_conv(z)
        # print(f"🔥 After post_quant_conv: shape={list(z.shape)}, dtype={z.dtype}, mean={z.float().mean().item():.6f}, min={z.min().item():.6f}, max={z.max().item():.6f}")
        dec, flow = self.decoder(z, first_frame=first_frame, is_distributed=is_distributed)
        # print(f"🔥 After Decoder_flow: shape={list(dec.shape)}, dtype={dec.dtype}, mean={dec.float().mean().item():.6f}, min={dec.min().item():.6f}, max={dec.max().item():.6f}")
        return dec

    def pad_to_multiple_of(self, x, multiple=256):
        height, width = x.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
        x = F.pad(x, padding, mode='constant', value=0)
        return x

    def forward(self, input, num_frames, sample_posterior=True, name=None):
        # print('#####infer####')
        self.input_shape = input.shape
        if self.use_tiling:
            input = self.pad_to_multiple_of(input)
        posterior, flow_enc = self.encode(input)
        # print('posterior size', posterior.shape)

        first_frame = (num_frames % 2 == 1)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = z.to(input.dtype)
        dec = self.decode(z, first_frame=first_frame)
        dec = dec[:, :, :, :self.input_shape[-2], :self.input_shape[-1]]
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def blend_v(
            self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                    1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
            self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                    1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode2d(self, x):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        flows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            flow = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                       :,
                       :,
                       :,
                       i: i + self.tile_sample_min_size,
                       j: j + self.tile_sample_min_size,
                       ]
                tile, tile_flow = self.encoder(tile)
                # print('enc_per_tile', tile.shape, tile_flow.shape)
                tile = self.quant_conv(tile)
                row.append(tile)
                flow.append(tile_flow)

            rows.append(row)
            flows.append(flow)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        result_flows = []
        for i, flow in enumerate(flows):
            result_flow = []
            for j, tile_flow in enumerate(flow):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile_flow = self.blend_v(flows[i - 1][j], tile_flow, blend_extent)
                if j > 0:
                    tile_flow = self.blend_h(flow[j - 1], tile_flow, blend_extent)
                result_flow.append(tile_flow[:, :, :, :row_limit, :row_limit])
            result_flows.append(torch.cat(result_flow, dim=4))

        moments = torch.cat(result_rows, dim=3)
        moments_flow = torch.cat(result_flows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)
        # print('encoder size after tile', moments.shape, moments_flow.shape)

        return posterior, moments_flow

    def tiled_decode2d(self, z, first_frame):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        flows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            flow = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                       :,
                       :,
                       :,
                       i: i + self.tile_latent_min_size,
                       j: j + self.tile_latent_min_size,
                       ]
                tile = self.post_quant_conv(tile)
                decoded, dec_flow = self.decoder(tile, first_frame, is_distributed=True)
                row.append(decoded)
                flow.append(dec_flow)
            rows.append(row)
            flows.append(flow)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))
        result_flows = []
        for i, flow in enumerate(flows):
            result_flow = []
            for j, tile_flow in enumerate(flow):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile_flow = self.blend_v(flows[i - 1][j], tile_flow, blend_extent)
                if j > 0:
                    tile_flow = self.blend_h(flow[j - 1], tile_flow, blend_extent)
                result_flow.append(tile_flow[:, :, :, :row_limit, :row_limit])
            result_flows.append(torch.cat(result_flow, dim=4))

        dec = torch.cat(result_rows, dim=3)
        dec_flows = torch.cat(result_flows, dim=3)
        return dec, dec_flows

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def init_from_inflation(self, path, mode='repeat'):
        def translate_key(key):
            item = key.split('.')
            if 'mid' in key:
                return item[0] + '.mid.block_2.conv2.' + item[-1]
            else:
                return '.'.join(item[:3]) + '.block.1.conv2.' + item[-1]

        sd = torch.load(path, map_location="cpu")

        non_inflation_keys = [k for k, v in self.state_dict().items() if len(v.size()) != 5 and k in sd]
        inflation_keys = [k for k, v in self.state_dict().items() if len(v.size()) == 5 and k in sd]
        non_inflation_sd = {k: (sd[k] if k in sd else sd[translate_key(k)]) for k in non_inflation_keys}
        inflation_sd = {k: (sd[k] if k in sd else sd[translate_key(k)]) for k in inflation_keys}

        # for k, v in non_inflation_sd.items():
        #     print(k, v.size(), len(v.size()))
        # for k, v in inflation_sd.items():
        #     print('inflation', k, v.size(), len(v.size()))

        missing_keys, unexpected_keys = self.load_state_dict(non_inflation_sd, strict=False)
        print(f'missing_keys: {missing_keys}, unexpected: {unexpected_keys}')
        conv_layer_name = []
        inflation_layer = []
        no_inflation_conv = []
        qkv = []
        others = []
        for name, p in self.named_parameters():
            if name in inflation_keys:
                # print(name, p.mean())
                t_size = p.data.size()[2]
                if mode == 'repeat':
                    for i in range(t_size):
                        p.data[:, :, i, :, :] = inflation_sd[name].data / t_size
                elif mode == 'last_slice':
                    for i in range(t_size):
                        p.data[:, :, i, :, :] = inflation_sd[name].data * 0.0
                    p.data[:, :, -1, :, :] = inflation_sd[name].data
        print('Succeed to INFLATION!!!!!!!!!!!!')

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class IdentityFirstStage(nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


if __name__ == '__main__':
    # import sys
    # sys.path.append('/home/weikanggong/MGM/mimo/')

    ddconfig = {'double_z': False,
                'z_channels': 8,
                'resolution': 256,
                'in_channels': 3,
                'out_ch': 3,
                'ch_mult': [1, 1, 2, 2, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [16],
                'dropout': 0.0,
                'ch': 128}
    lossconfig = {'target': 'torch.nn.Identity'}
    model = VQModelInterface(embed_dim=8, n_embed=16384, ddconfig=ddconfig, lossconfig=lossconfig)

    out = model.encode(torch.randn(10, 3, 256, 256))
    print(out.shape)
    print(out)
    # for n,p in model.named_parameters():
    #     print(n)
