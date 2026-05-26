# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MMDiT inference model with DiT offloading support.

Ported from MGM-Video-Ascend mimogpt/models/dit/mmdit_inference.py.

Key changes from the original:
- Removed BACKBONE_REGISTRY decorator and cfg-based construction
- Replaced torch_npu top-level import with try/except for portability
- Replaced torch.cuda Stream/Event with NPU-compatible helpers
- mmdit_xl_2_inference accepts explicit kwargs instead of cfg object
"""

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
from torch.profiler import record_function

try:
    import torch_npu
except ImportError:
    torch_npu = None

from .mmdit import MMDiTBlock, MMDiT
from .mmdit_blocks_inference import JoinAttentionInference
from .mmdit_utils import read_2d_array_from_file_int

# ---------------------------------------------------------------------------
# NPU / CUDA stream & event helpers
# ---------------------------------------------------------------------------
try:
    import torch_npu as _torch_npu_check
    _USE_NPU = True
except ImportError:
    _USE_NPU = False


def _get_stream():
    if _USE_NPU:
        return torch.npu.Stream()
    return torch.cuda.Stream()


def _get_event():
    if _USE_NPU:
        return torch.npu.Event()
    return torch.cuda.Event()


def _get_current_stream():
    if _USE_NPU:
        return torch.npu.current_stream()
    return torch.cuda.current_stream()


def _get_default_stream():
    if _USE_NPU:
        return torch.npu.default_stream(torch.npu.current_device())
    return torch.cuda.default_stream(torch.cuda.current_device())


def _stream_ctx(stream):
    """Return the appropriate stream context manager (torch.npu.stream or torch.cuda.stream)."""
    if _USE_NPU:
        return torch.npu.stream(stream)
    return torch.cuda.stream(stream)


class MMDiTBlockInference(MMDiTBlock):
    def __init__(
            self,
            n_embd,
            n_head,
            dropout,
            fa_keep_prob=1.0,
            use_checkpoint=False,
            checkpoint_layer=-1,
            checkpoint_finegrained_layer=0,
            use_context_parallelism=False,
            offload_fa=False,
            h2d_stream=None,
            d2h_stream=None,
            depth=-1,
            down_mode=None,
            downscale=1,
            index=0,
            laser_atten=False,
    ):
        super().__init__(
            n_embd=n_embd, n_head=n_head, dropout=dropout, fa_keep_prob=fa_keep_prob, use_checkpoint=use_checkpoint,
            checkpoint_layer=checkpoint_layer, checkpoint_finegrained_layer=checkpoint_finegrained_layer,
            use_context_parallelism=use_context_parallelism, offload_fa=offload_fa, h2d_stream=h2d_stream, d2h_stream=d2h_stream,
            depth=depth, down_mode=down_mode, downscale=downscale, index=index
        )
        self.attention = JoinAttentionInference(
			n_embd,
			n_head,
			dropout,
			fa_keep_prob=fa_keep_prob,
			use_context_parallelism=use_context_parallelism,
            depth=depth, down_mode=down_mode, downscale=downscale, index=index,
            laser_atten=laser_atten,
		)

    _debug_block_idx = 0  # class-level counter for debug printing

    def forward(self, x, y, t, x_padding_size, y_padding_size, mask=None, spatial_freq=None, num_layer=-1, f=None, hh=None, ww=None, skip_compute=False):
        if skip_compute: # skip compute, but load others
            return x, y
        x1_cts = None
        B = x.shape[0]
        block_idx = MMDiTBlockInference._debug_block_idx

        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = \
            self.adaLN_modulation_x(t).reshape(B, 6, -1).chunk(6, dim=1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = \
            self.adaLN_modulation_y(t).reshape(B, 6, -1).chunk(6, dim=1)

        if x.shape[0] == 1:  ## video data
            x1, y1 = self.t2i_modulate_xy_v2(x, y, shift_msa_x, scale_msa_x, shift_msa_y, scale_msa_y)
        else:  ## image data
            x1, y1 = self.t2i_modulate_xy(x, y, shift_msa_x, scale_msa_x, shift_msa_y, scale_msa_y)

        # Propagate block_idx to JoinAttentionInference for internal debug
        self.attention._debug_block_idx = block_idx

        x1, y1 = self.attention.infer(
            x1, y1, x1_cts, spatial_freq,
            x_padding_size, y_padding_size,
            mask=mask, f=f, hh=hh, ww=ww
        )

        x1, y1 = self.xy_add_mul(x, gate_msa_x, x1, y, gate_msa_y, y1)

        if x.shape[0] == 1:   ## video data
            x, y = self.mlp_xy_v2(x, y, x1, y1, gate_mlp_x, gate_mlp_y, shift_mlp_x, scale_mlp_x, shift_mlp_y, scale_mlp_y)
        else: ## image data
            x, y = self.mlp_xy(x, y, x1, y1, gate_mlp_x, gate_mlp_y, shift_mlp_x, scale_mlp_x, shift_mlp_y, scale_mlp_y)


        MMDiTBlockInference._debug_block_idx += 1

        return x, y


#################################################################################
#                                 Core MMDiT Model                              #
#################################################################################
class MMDiTInference(MMDiT):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, max_input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=24, head_dim=64,
                 class_dropout_prob=0.1, pred_sigma=False, caption_channels=4096, lewei_scale=1.0, dropout=0.0, fa_keep_prob=1.0,
                 model_max_length=200, use_rel_pos=True, use_3d_rope=True, rope_ratio=[22/64, 22/64, 20/64], use_size_control=False, use_checkpoint=True,
                 checkpoint_finegrained_layer=0, checkpoint_layer=-1, use_mmdit_block=True, dtype='bf16', use_context_parallelism=False, offload_fa=False,
                 skiparse=None, skip_initialize_weights=True, x2v=False, cond_intype='sum', out_channels=16, cache_algo_cfg=None, laser_atten=False):
        super().__init__(
            max_input_size=max_input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, head_dim=head_dim,
            class_dropout_prob=class_dropout_prob, pred_sigma=pred_sigma, caption_channels=caption_channels, lewei_scale=lewei_scale, dropout=dropout,
            fa_keep_prob=fa_keep_prob, model_max_length=model_max_length, use_rel_pos=use_rel_pos, use_3d_rope=use_3d_rope, rope_ratio=rope_ratio,
            use_size_control=use_size_control, use_checkpoint=use_checkpoint, checkpoint_finegrained_layer=checkpoint_finegrained_layer,
            checkpoint_layer=checkpoint_layer, use_mmdit_block=use_mmdit_block, dtype=dtype, use_context_parallelism=use_context_parallelism,
            offload_fa=offload_fa, skiparse=skiparse, skip_initialize_weights=skip_initialize_weights, x2v=x2v, cond_intype=cond_intype, out_channels=out_channels
        )
        self.blocks = nn.ModuleList([
            MMDiTBlockInference(n_embd=hidden_size, n_head=self.num_heads, dropout=dropout, fa_keep_prob=fa_keep_prob, use_checkpoint=use_checkpoint,
            checkpoint_finegrained_layer=self.checkpoint_finegrained_layer, checkpoint_layer=self.checkpoint_layer, use_context_parallelism=use_context_parallelism,
            offload_fa=self.offload_fa, depth=depth, down_mode=self.down_mode, downscale=self.downscale[i], index=i, laser_atten=laser_atten)
            for i in range(depth)
        ])

        # load infer_algo related cfgs
        if cache_algo_cfg is not None:
            self.cache_algo_cfg = cache_algo_cfg
            self.cache_algo_enable = cache_algo_cfg['enable']
            if self.cache_algo_enable == True:
                cache_algo_cfg_path = cache_algo_cfg['scheme']
                self.blk_step_status_2d = read_2d_array_from_file_int(cache_algo_cfg_path)
                print(f"load cache scheme {cache_algo_cfg_path}")
                assert self.blk_step_status_2d.shape[1] == 8 # only support 8 steps tdm model
        else:
            self.cache_algo_cfg = None
            self.cache_algo_enable = False


    @staticmethod
    def _convert_spatial_freq_for_inference(spatial_freq):
        """Convert legacy spatial_freq format to (cos, sin) tuples for RotaryEmbedding.

        Legacy format per axis: [S, 2*dim] where [:, :dim] is sin (interleaved), [:, dim:] is cos (interleaved).
        Target format per axis: (cos[S, dim/2], sin[S, dim/2]) — non-interleaved half-dim.
        """
        result = []
        for sincos in spatial_freq:
            dim = sincos.shape[-1] // 2
            sin_interleaved = sincos[:, :dim]
            cos_interleaved = sincos[:, dim:]
            result.append((cos_interleaved[:, ::2].contiguous(), sin_interleaved[:, ::2].contiguous()))
        return result

    def blocks_forward(self, x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, fn, base_size_h, base_size_w, **kwargs):
        if spatial_freq is not None:
            spatial_freq = self._convert_spatial_freq_for_inference(spatial_freq)
        # Reset block debug counter for each forward pass (each denoising step)
        MMDiTBlockInference._debug_block_idx = 0
        cur_time_index = kwargs['cur_time_index']
        if self.cache_algo_cfg is not None and self.cache_algo_enable == True:
            for i, block in enumerate(self.blocks):
                block.cur_time_index = cur_time_index
                current_status = self.blk_step_status_2d[i, cur_time_index]
                if current_status == 0:  # state 0, regular compute
                    x, y = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w)
                elif current_status == 1:  # state 1, recored ori feature + regular compute
                    ori_x = x.clone()
                    ori_y = y.clone()
                    x, y = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w)
                elif current_status == 2:  # state 2, recored residual + regular compute
                    x, y = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w)
                    self.previous_residual = x - ori_x
                    self.previous_residual_encoder = y - ori_y
                elif current_status == 3:  # state 3, use residual + skip compute
                    x_tmp, y_tmp = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w, skip_compute=True)
                    x += self.previous_residual
                    y += self.previous_residual_encoder
                elif current_status == 4: # state 4, skip compute
                    x_tmp, y_tmp = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w, skip_compute=True)
                else:
                    x, y = block(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, num_layer=i, f=fn, hh=base_size_h, ww=base_size_w)
        else:
            x, y = super().blocks_forward(x, y, t, x_padding_size, y_padding_size, mask, spatial_freq, fn, base_size_h, base_size_w, **kwargs)
        return x, y


    def parameter_to_device_hook(self, module, input):
        def async_copy(forward_event):
            self.eval_h2d_stream.wait_event(forward_event)
            if p.is_slice_tensor:
                p.data.copy_(p.p_cpu, non_blocking=True)
            else:
                p.data.untyped_storage().copy_(p.p_cpu.untyped_storage(), non_blocking=True)

        to_device_index = module.index + self.block_on_npu_nums
        if to_device_index < module.depth:
            if self.offload_scheduler == 0:
                with _stream_ctx(self.eval_h2d_stream):
                    for p in self.blocks[to_device_index].parameters():
                        p.data.untyped_storage().resize_(p.storage_size)
                        forward_event = _get_event()
                        forward_event.record()
                        async_copy(forward_event)

            else:
                if module.cur_time_index in self.to_device_timestep_list:
                    for p in self.blocks[to_device_index].parameters():
                        p.data.untyped_storage().resize_(p.storage_size)
                        forward_event = _get_event()
                        forward_event.record()
                        with _stream_ctx(self.eval_h2d_stream):
                            async_copy(forward_event)

    def parameter_to_resize_hook(self, module, input, output):
        def resize_parameter(to_resize_index):
            if to_resize_index > self.block_on_npu_nums-1:  # 第0,1层参数常驻在decice上
                for p in self.blocks[to_resize_index].parameters():
                    p.data.untyped_storage().resize_(0)

        to_resize_index = module.index
        if self.offload_scheduler == 1:
            if module.cur_time_index in self.to_host_timestep_list:
                resize_parameter(to_resize_index)
        else:
            resize_parameter(to_resize_index)

        _get_current_stream().wait_stream(self.eval_h2d_stream)
        _get_default_stream().wait_stream(self.eval_h2d_stream)


    def enable_dit_inference_offload(self, offload_scheduler=0, num_sampling_steps=9):
        self.eval_h2d_stream = _get_stream()
        self.eval_d2h_stream = _get_stream()
        self.offload_scheduler = offload_scheduler

        if offload_scheduler == 0:  # 每个timestep都进行offload
            self.block_on_npu_nums = 2  # 兼容fsdp zero3 常驻2个block在npu上
            self.to_device_timestep_list = list(range(self.block_on_npu_nums, num_sampling_steps-1))
            self.to_host_timestep_list = list(range(self.block_on_npu_nums, num_sampling_steps-1))
        elif offload_scheduler == 1:
            self.block_on_npu_nums = 1
            self.to_device_timestep_list = [0]
            self.to_host_timestep_list = [num_sampling_steps-2]
        else:
            raise Exception("unknow offload sheduler")

        print("to_device_timestep_list: {}...".format(self.to_device_timestep_list))
        print("to_host_timestep_list: {}...".format(self.to_host_timestep_list))

        # parameter to host warmup
        for blk_idx in range(self.block_on_npu_nums, self.depth, 1):
            for p in self.blocks[blk_idx].parameters():
                with _stream_ctx(self.eval_d2h_stream):
                    if not hasattr(p, "p_cpu"):
                        p_cpu = torch.empty(p.data.shape, dtype=p.dtype, pin_memory=True, device='cpu')
                        setattr(p, "p_cpu", p_cpu)
                    is_slice_tensor = p.data.untyped_storage().size() != p.data.numel()
                    storage_size = p.data.untyped_storage().size()
                    if is_slice_tensor:
                        p.p_cpu.copy_(p.data, non_blocking=True)
                    else:
                        p.p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)
                    setattr(p, "storage_size", storage_size)
                    setattr(p, "is_slice_tensor", is_slice_tensor)

        _get_current_stream().wait_stream(self.eval_d2h_stream)
        _get_default_stream().wait_stream(self.eval_d2h_stream)

        for blk_idx in range(self.block_on_npu_nums, self.depth, 1):
            for p in self.blocks[blk_idx].parameters():
                p.data.untyped_storage().resize_(0)

        for blk_idx, blk in enumerate(self.blocks):
            blk.register_forward_pre_hook(self.parameter_to_device_hook)
            if blk_idx > self.block_on_npu_nums-1:
                blk.register_forward_hook(self.parameter_to_resize_hook)


#################################################################################
#                                   Dit Configs                                 #
#################################################################################
def mmdit_xl_2_inference(
    patch_size=2,
    depth=42,
    in_channels=16,
    hidden_size=3072,
    head_dim=128,
    use_rel_pos=True,
    model_max_length=400,
    use_3d_rope=True,
    rope_ratio=None,
    dropout=0.0,
    fa_keep_prob=1.0,
    dtype='bf16',
    use_checkpoint=True,
    checkpoint_finegrained_layer=0,
    checkpoint_layer=-1,
    pred_sigma=False,
    use_context_parallelism=False,
    offload_fa=False,
    skiparse=None,
    skip_initialize_weights=True,
    x2v=False,
    cond_intype='sum',
    out_channels=16,
    cache_algo_cfg=None,
    laser_atten=False,
    caption_channels=4096,
    class_dropout_prob=0.1,
    lewei_scale=1.0,
    use_size_control=False,
):
    if rope_ratio is None:
        rope_ratio = [22/64, 22/64, 20/64]
    config = dict(
        patch_size=patch_size,
        depth=depth,
        in_channels=in_channels,
        hidden_size=hidden_size,
        head_dim=head_dim,
        use_rel_pos=use_rel_pos,
        model_max_length=model_max_length,
        use_3d_rope=use_3d_rope,
        rope_ratio=rope_ratio,
        dropout=dropout,
        fa_keep_prob=fa_keep_prob,
        use_mmdit_block=True,
        dtype=dtype,
        use_checkpoint=use_checkpoint,
        checkpoint_finegrained_layer=checkpoint_finegrained_layer,
        checkpoint_layer=checkpoint_layer,
        pred_sigma=pred_sigma,
        use_context_parallelism=use_context_parallelism,
        offload_fa=offload_fa,
        skiparse=skiparse,
        skip_initialize_weights=skip_initialize_weights,
        x2v=x2v,
        cond_intype=cond_intype,
        out_channels=out_channels,
        cache_algo_cfg=cache_algo_cfg,
        laser_atten=laser_atten,
        caption_channels=caption_channels,
        class_dropout_prob=class_dropout_prob,
        lewei_scale=lewei_scale,
        use_size_control=use_size_control,
    )
    model = MMDiTInference(**config)
    return model