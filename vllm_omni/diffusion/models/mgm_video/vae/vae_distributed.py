import os
os.environ["COMBINED_ENABLE"] = "1"

os.environ["INF_NAN_MODE_ENABLE"] = "1"
import torch
try:
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = False
    from torch_npu.contrib import transfer_to_npu
except:
    print("no torch_npu")

import copy
import time
import json
import torch.distributed as dist

try:
    import moxing as mox
except:
    print("no moxing")

from vllm_omni.diffusion.models.mgm_video.vae.vae_utils import instantiate_from_config, read_from_yaml, merge_args
from vllm_omni.diffusion.models.mgm_video.vae.vae_distributed_modules import SyncGroupNormWithGather, SyncGroupNormWithSyncBN, DistributedConv3D, DistributedConv2D


def print_by_rank(content, output_path=None):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0:
        if output_path is not None:
            with open(output_path, 'a') as f:
                f.write(str(content))
                f.write("\n")
        print(content)


def set_distributed_config(model, args, keys, conv3d_pad_config):
    distributed_config = {}
    vae_conv_split_config = args.vae_conv_split_config
    if vae_conv_split_config is not None:
        enable_conv_split = vae_conv_split_config.enable_conv_split
        split_nums_in = vae_conv_split_config.get("split_nums_in", 1)
        split_nums_out = vae_conv_split_config.get("split_nums_out", 1)
        split_conv_names = vae_conv_split_config.get("split_conv_names", [])
    else:
        enable_conv_split = False

    for name, module in model.named_modules():
        for subname, submodule in module.named_children():
            if name != "":
                op_name = name + "." + subname
            else:
                op_name = subname
            if isinstance(submodule, (torch.nn.Conv2d,)):
                conv_config = {
                    "conv_type": None,
                }
                for key in keys:
                    if key in name:
                        if not enable_conv_split:
                            conv_config["conv_type"] = "original"
                        else:
                            if op_name in split_conv_names:
                                conv_config["conv_type"] = "split_in_out_channel"
                                assert submodule.in_channels % split_nums_in == 0
                                assert submodule.out_channels % split_nums_out == 0
                                conv_config["split_nums_in"] = split_nums_in
                                conv_config["split_nums_out"] = split_nums_out
                                print("run {} in split_in_out_channel".format(op_name))
                distributed_config[op_name] = conv_config
            elif isinstance(submodule, (torch.nn.Conv3d,)):
                split_stragety = copy.deepcopy(args.split_stragety)
                if args.task == "encode_decode":
                    if "decoder" in op_name:
                        split_stragety["method"] = split_stragety["method"][1]
                    else:
                        split_stragety["method"] = split_stragety["method"][0]

                conv_config = {
                    "conv_type": None,
                    "padding_info": None,
                    "is_downsample": 0,
                    "is_upsample": 0,
                    "split_stragety": split_stragety,
                    "in_channel": None,
                    "out_channel": None
                }
                for key in conv3d_pad_config.keys():
                    if key in op_name:
                        channel = max([submodule.in_channels, submodule.out_channels])
                        input_feature_nums = keys[key] * channel / dist.get_world_size()
                        conv_config["conv_type"] = "original"
                        conv_config["in_channel"] = submodule.in_channels
                        conv_config["out_channel"] = submodule.out_channels
                        conv_config["padding_info"] = conv3d_pad_config[key]
                        if "downsample" in op_name:
                            if submodule.stride == tuple([2, 1, 1]):
                                conv_config["is_downsample"] = 0
                            else:
                                conv_config["is_downsample"] = 1
                        elif "upsample" in op_name:
                            conv_config["is_upsample"] = 1
                        
                        if enable_conv_split:
                            if op_name in split_conv_names:
                                conv_config["conv_type"] = "split_in_out_channel"
                                assert submodule.in_channels % split_nums_in == 0
                                assert submodule.out_channels % split_nums_out == 0
                                conv_config["split_nums_in"] = split_nums_in
                                conv_config["split_nums_out"] = split_nums_out
                                print("run {} in split_in_out_channel".format(op_name))
                            

                distributed_config[op_name] = conv_config
            elif isinstance(submodule, (torch.nn.GroupNorm,)):
                # find = False
                for key in keys:
                    if key in op_name:
                        norm_config = {
                            "groups": submodule.num_groups,
                            "batch_size": 1,
                            "split_infer": args.vae_gn_split_infer
                        }
                        distributed_config[op_name] = norm_config
                        find = True
    return distributed_config


def set_p2p_config(args):
    h, w, t, c = args.height, args.width, args.max_frame, 128

    gpu_nums = dist.get_world_size()
    communication_type = args.infer_config.communication_type

    def check(h, w):
        is_pass = False
        if h % (gpu_nums * 8) == 0:
            return 0
        elif w % (gpu_nums * 8) == 0:
            return 1
        else:
            raise Exception("split axis must be divided by gpu_nums")

    if args.task in ["encode", "encode_decode"]:
        permute_hw = check(h, w)
        args.infer_config.permute_hw = permute_hw

    return {
        "method": communication_type,
        "split_type": "split_h",
    }


def convert_to_distributed(model, keys, distributed_config=None, args=None):
    for name, module in model.named_modules():
        for subname, submodule in module.named_children():
            if subname == "ops":
                continue
            if name != "":
                op_name = name + "." + subname
            else:
                op_name = subname
            if isinstance(submodule, (torch.nn.Conv3d,)):
                for key in keys:
                    if key in op_name:
                        try:
                            op_config = distributed_config[op_name]
                        except:
                            continue
                        stride = submodule.stride
                        if isinstance(stride, (tuple, list,)):
                            stride_nums = len(set(list(stride)))
                            if stride_nums > 1:
                                pass
                            else:
                                stride = stride[0]
                        distributed_conv = DistributedConv3D(
                            submodule.in_channels,
                            submodule.out_channels,
                            submodule.kernel_size,
                            conv_config=op_config,
                            stride=stride,
                            padding=submodule.padding
                        )
                        distributed_conv.conv.weight = submodule.weight
                        distributed_conv.conv.bias = submodule.bias
                        setattr(module, subname, distributed_conv)
            elif isinstance(submodule, (torch.nn.Conv2d,)):
                for key in keys:
                    if key in op_name:
                        try:
                            op_config = distributed_config[op_name]
                        except:
                            continue
                        stride = submodule.stride
                        if isinstance(stride, (tuple, list,)):
                            stride_nums = len(set(list(stride)))
                            if stride_nums > 1:
                                raise Exception("not support multi stride")
                        stride = stride[0]
                        distributed_conv = DistributedConv2D(
                            submodule.in_channels,
                            submodule.out_channels,
                            submodule.kernel_size,
                            conv_config=op_config,
                            stride=stride,
                            padding=submodule.padding
                        )
                        distributed_conv.conv.weight = submodule.weight
                        distributed_conv.conv.bias = submodule.bias
                        setattr(module, subname, distributed_conv)
            elif isinstance(submodule, (torch.nn.GroupNorm,)):
                for key in keys:
                    if key in op_name:
                        op_config = distributed_config[op_name]
                        if args.infer_config.infer_type == "graph":
                            # print("----JYY convert_to_distributed----SyncGroupNormWithGather")
                            sync_groupnorm = SyncGroupNormWithGather(
                                submodule.num_channels,
                                **op_config
                            )
                        else:
                            # print("----JYY convert_to_distributed----SyncGroupNormWithSyncBN")
                            sync_groupnorm = SyncGroupNormWithSyncBN(
                                submodule.num_channels,
                                **op_config
                            )

                        sync_groupnorm.weight = submodule.weight
                        sync_groupnorm.bias = submodule.bias

                        setattr(module, subname, sync_groupnorm)

    return model

def pad_split_frames(x, args,
                     split_type=None):
    if split_type is None:
        split_type = args.data_config.split_type
    is_pad = args.data_config.is_pad
    permute_hw = args.infer_config.permute_hw
    is_split = dist.get_world_size() > 1

    if is_pad:
        x_frame = x.shape[2]
        if args.max_frame < x_frame:
            x = x[:, :, :args.max_frame, :, :].contiguous()
        else:
            chunk_nums = args.max_frame // x_frame
            chunk_delta = args.max_frame - chunk_nums * x_frame
            x = [x] * chunk_nums + [x[:, :, :chunk_delta, :, :]]
            x = torch.cat(x, dim=2)

    if permute_hw:
        x = x.permute(0, 1, 2, 4, 3)

    if is_split:
        rank = dist.get_rank()

        if split_type == "split_h":
            chunk_size = x.shape[3] // dist.get_world_size()
            if rank == 0:
                x = x[:, :, :, :(rank+1)*chunk_size, :].contiguous()
            elif rank == dist.get_world_size() - 1:
                x = x[:, :, :, rank*chunk_size:, :].contiguous()
            else:
                x = x[:, :, :, rank*chunk_size:(rank+1)*chunk_size, :].contiguous()
        elif split_type == "split_w":
            chunk_size = x.shape[4] // dist.get_world_size()
            if rank == 0:
                x = x[:, :, :, :, :(rank+1)*chunk_size].contiguous()
            elif rank == dist.get_world_size() - 1:
                x = x[:, :, :, :, rank*chunk_size:].contiguous()
            else:
                x = x[:, :, :, :, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
        elif split_type == "split_t":
            chunk_size = x.shape[2] // dist.get_world_size()
            if rank == 0:
                x = x[:, :, :(rank + 1) * chunk_size, :, :].contiguous()
            elif rank == dist.get_world_size() - 1:
                if not (args.model_name == "motion_vae"):
                    x = x[:, :, rank * chunk_size:, :, :].contiguous()
                else:
                    x = x[:, :, rank * chunk_size - 1:, :, :].contiguous()
            else:
                if not (args.model_name == "motion_vae"):
                    x = x[:, :, rank * chunk_size:(rank + 1) * chunk_size, :, :].contiguous()
                else:
                    x = x[:, :, rank * chunk_size - 1:(rank + 1) * chunk_size, :, :].contiguous()
        else:
            raise NotImplementedError

    return x


def gather_video(dec, args, split_type=None, x=None):
    def slice_tensor(tensor_list, all_shapes, axis):
        for idx, shape in enumerate(all_shapes):
            if split_type == "split_h":
                tensor_list[idx] = tensor_list[idx][:, :, :, :shape[axis], :]
            elif split_type == "split_w":
                tensor_list[idx] = tensor_list[idx][:, :, :, :, :shape[axis]]
            elif split_type == "split_t":
                tensor_list[idx] = tensor_list[idx][:, :, :shape[axis], :, :]
        return tensor_list

    if split_type is None:
        split_type = args.data_config.split_type

    all_shapes = [torch.zeros([5], device=torch.cuda.current_device(), dtype=torch.int32) for _ in range(dist.get_world_size())]
    shape_tensor = torch.tensor(dec.shape, device=torch.cuda.current_device(), dtype=torch.int32)
    dist.all_gather(all_shapes, shape_tensor)
    gather_axis = {
        "split_h": [3, 3],
        "split_w": [4, 1],
        "split_t": [2, 5],
    }

    axis = gather_axis[split_type][0]
    max_axis_val = max([i[axis] for i in all_shapes])
    padding_info = [0, 0, 0, 0, 0, 0]
    padding_info[gather_axis[split_type][1]] = max_axis_val - dec.shape[axis]
    padding_info = tuple(padding_info)
    b, c, t, h, w = dec.shape
    if max_axis_val - dec.shape[axis] > 0:
        dec = torch.nn.functional.pad(dec, padding_info, mode="constant", value=0)

    tensor_list = [torch.zeros(dec.shape, device=torch.cuda.current_device(), dtype=dec.dtype) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, dec)
    tensor_list = slice_tensor(tensor_list, all_shapes, axis)
    dec = torch.cat(tensor_list, dim=axis)

    if x is not None:
        if max_axis_val - x.shape[axis] > 0:
            x = torch.nn.functional.pad(x, padding_info, mode="constant", value=0)
        tensor_list = [torch.zeros(x.shape, device=torch.cuda.current_device(), dtype=x.dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, x)
        tensor_list = slice_tensor(tensor_list, all_shapes, axis)
        x = torch.cat(tensor_list, dim=axis)
        return dec, x
    else:
        return dec, None


class DistributedVAE(torch.nn.Module):
    def __init__(self, args, dataloader=[]):
        super().__init__()

        self.patch_size = args.patch_size
        self.decode_pad = args.decode_pad
        model_config = {
            "vae-1.2": {
                "keys": {
                    # encoder
                    "encoder.conv_in": args.height * args.width * args.max_frame,
                    "encoder.down.0": args.height * args.width * args.max_frame,
                    "encoder.down.0.downsample": args.height // 2 * args.width // 2 * args.max_frame,
                    "encoder.down.1": args.height // 2 * args.width // 2 * args.max_frame,
                    "encoder.down.1.downsample": args.height // 4 * args.width // 4 * args.max_frame // 2,
                    "encoder.down.2": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "encoder.down.2.downsample": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.down.3": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.mid.block": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.conv_out": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.mid.attn_1.norm": 0,
                    "encoder.norm_out": 0,
                    # decoder
                    "decoder.mid.attn_1.norm": 0,
                    "decoder.conv_in": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "decoder.up.3": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "decoder.up.3.upsample": args.height // 4 * args.width // 4 * args.max_frame // 4,
                    "decoder.up.2": args.height // 4 * args.width // 4 * args.max_frame // 4,
                    "decoder.up.2.upsample": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "decoder.up.1": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "decoder.up.1.upsample": args.height * args.width * args.max_frame,
                    "decoder.up.0": args.height * args.width * args.max_frame,
                    "decoder.mid.block": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "decoder.conv_out": args.height * args.width * args.max_frame,
                    "decoder.norm_out": 0,
                    # extra
                    "quant_conv": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "post_quant_conv": args.height // 8 * args.width // 8 * args.max_frame // 4
                },
                "conv3d_pad_config": {
                    "encoder.conv_in": (1, 1, 1, 1, 2, 0),  # wht
                    "encoder.down.0": (1, 1, 1, 1, 2, 0),
                    "encoder.down.0.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.1": (1, 1, 1, 1, 2, 0),
                    "encoder.down.1.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.2": (1, 1, 1, 1, 2, 0),
                    "encoder.down.2.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.3": (1, 1, 1, 1, 2, 0),
                    "encoder.mid.block": (1, 1, 1, 1, 2, 0),
                    "encoder.conv_out": (1, 1, 1, 1, 2, 0),
                    "decoder.conv_in": (1, 1, 1, 1, 2, 0),
                    "decoder.mid.block": (1, 1, 1, 1, 2, 0),
                    "decoder.up.0": (1, 1, 1, 1, 2, 0),
                    "decoder.up.1.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.1": (1, 1, 1, 1, 2, 0),
                    "decoder.up.2.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.2": (1, 1, 1, 1, 2, 0),
                    "decoder.up.3.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.3": (1, 1, 1, 1, 2, 0),
                    "decoder.conv_out": (1, 1, 1, 1, 2, 0),
                },
                "downsample": {
                    "frame": 4,
                    "height": 8,
                    "width": 8,
                }
            },
            "vae-2.4": {
                "keys": {
                    # encoder
                    "encoder.conv_in": args.height * args.width * args.max_frame,
                    "encoder.down.0": args.height * args.width * args.max_frame,
                    "encoder.down.0.downsample": args.height // 2 * args.width // 2 * args.max_frame,
                    "encoder.down.1": args.height // 2 * args.width // 2 * args.max_frame,
                    "encoder.down.1.downsample": args.height // 4 * args.width // 4 * args.max_frame // 2,
                    "encoder.down.2": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "encoder.down.2.downsample": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.down.3": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.mid.block": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.conv_out": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.mid.attn_1.norm": 0,
                    "encoder.norm_out": 0,
                    # decoder
                    "decoder.mid.attn_1.norm": 0,
                    "decoder.conv_in": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "decoder.up.3": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "decoder.up.3.upsample": args.height // 4 * args.width // 4 * args.max_frame // 4,
                    "decoder.up.2": args.height // 4 * args.width // 4 * args.max_frame // 4,
                    "decoder.up.2.upsample": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "decoder.up.1": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "decoder.up.1.upsample": args.height * args.width * args.max_frame,
                    "decoder.up.0": args.height * args.width * args.max_frame,
                    "decoder.mid.block": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "decoder.conv_out": args.height * args.width * args.max_frame,
                    "decoder.norm_out": 0,
                    # extra
                    "quant_conv": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "post_quant_conv": args.height // 8 * args.width // 8 * args.max_frame // 4
                },
                "conv3d_pad_config": {
                    "encoder.conv_in": (1, 1, 1, 1, 2, 0),  # wht
                    "encoder.down.0": (1, 1, 1, 1, 2, 0),
                    "encoder.down.0.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.1": (1, 1, 1, 1, 2, 0),
                    "encoder.down.1.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.2": (1, 1, 1, 1, 2, 0),
                    "encoder.down.2.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.3": (1, 1, 1, 1, 2, 0),
                    "encoder.mid.block": (1, 1, 1, 1, 2, 0),
                    "encoder.conv_out": (1, 1, 1, 1, 2, 0),
                    "decoder.conv_in": (1, 1, 1, 1, 2, 0),
                    "decoder.mid.block": (1, 1, 1, 1, 2, 0),
                    "decoder.up.0": (1, 1, 1, 1, 2, 0),
                    "decoder.up.1.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.1": (1, 1, 1, 1, 2, 0),
                    "decoder.up.2.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.2": (1, 1, 1, 1, 2, 0),
                    "decoder.up.3.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.3": (1, 1, 1, 1, 2, 0),
                    "decoder.conv_out": (1, 1, 1, 1, 2, 0),
                },
                "downsample": {
                    "frame": 4,
                    "height": 8,
                    "width": 8,
                }
            },
            "motion_vae": {
                "keys": {
                    # encoder
                    "encoder.down.1": args.height // 2 * args.width // 2 * args.max_frame,
                    "encoder.down.1.downsample": args.height // 4 * args.width // 4 * args.max_frame // 2,
                    "encoder.down.2": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "encoder.down.2.downsample": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.down.3": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "encoder.down.3.downsample": args.height // 8 * args.width // 8 * args.max_frame // 8,
                    "encoder.mid.block": args.height // 8 * args.width // 8 * args.max_frame // 8,
                    "encoder.conv_out": args.height // 8 * args.width // 8 * args.max_frame // 8,
                    "encoder.mid.attn_1.norm": 0,
                    "encoder.norm_out": 0,
                    # decoder
                    "decoder.conv_in": args.height // 8 * args.width // 8 * args.max_frame // 8,
                    "decoder.mid.attn_1.norm": 0,
                    "decoder.up.3": args.height // 8 * args.width // 8 * args.max_frame // 8,
                    "decoder.up.3.upsample": args.height // 4 * args.width // 4 * args.max_frame // 4,
                    "decoder.up.2": args.height // 4 * args.width // 4 * args.max_frame // 4,
                    "decoder.up.2.upsample": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "decoder.up.1": args.height // 2 * args.width // 2 * args.max_frame // 2,
                    "decoder.up.1.upsample": args.height * args.width * args.max_frame,
                    "decoder.up.0.block.0": args.height * args.width * args.max_frame // 2,
                    "decoder.mid.block": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    # extra
                    "quant_conv": args.height // 8 * args.width // 8 * args.max_frame // 4,
                    "post_quant_conv": args.height // 8 * args.width // 8 * args.max_frame // 4
                },
                "conv3d_pad_config": {
                    "encoder.down.1": (1, 1, 1, 1, 2, 0),
                    "encoder.down.1.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.2": (1, 1, 1, 1, 2, 0),
                    "encoder.down.2.downsample": (0, 1, 0, 1, 2, 0),
                    "encoder.down.3": (1, 1, 1, 1, 2, 0),
                    "encoder.down.3.downsample": (1, 1, 1, 1, 2, 0),
                    "encoder.mid.block": (1, 1, 1, 1, 2, 0),
                    "encoder.conv_out": (1, 1, 1, 1, 2, 0),
                    "decoder.conv_in": (1, 1, 1, 1, 2, 0),
                    "decoder.mid.block": (1, 1, 1, 1, 2, 0),
                    "decoder.up.0.block.0": (1, 1, 1, 1, 2, 0),
                    "decoder.up.1.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.1": (1, 1, 1, 1, 2, 0),
                    "decoder.up.2.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.2": (1, 1, 1, 1, 2, 0),
                    "decoder.up.3.upsample": (1, 1, 1, 1, 2, 0),
                    "decoder.up.3": (1, 1, 1, 1, 2, 0),
                },
                "downsample": {
                    "frame": 8,
                    "height": 8,
                    "width": 8,
                }
            }
        }

        keys = model_config[args.model_name]["keys"]
        conv3d_pad_config = model_config[args.model_name]["conv3d_pad_config"]

        config = read_from_yaml(args.config_path)
        args = merge_args(args, config)
        args.infer_config.infer_type = args.infer_type
        config.model_config.params.ddconfig.is_casual = args.is_casual
        self.model = instantiate_from_config(config.model_config)
        self.out_channels = self.model.embed_dim

        if args.infer_config.deterministic:
            print_by_rank("use_deterministic...")
            torch.use_deterministic_algorithms(True)
            os.environ["HCCL_DETERMINISTIC"] = "True"

        if dist.get_world_size() > 1:
            args.split_stragety = set_p2p_config(args)

        if args.ckpt_path is not None:
            args.model_config.ckpt_path_local = args.ckpt_path
        if not os.path.exists(args.model_config.ckpt_path_local):
            src = args.model_config.ckpt_path_cloud
            dst = args.model_config.ckpt_path_local
            try:
                mox.file.copy(src, dst)
            except:
                print("failed load ckpt from {}...".format(args.model_config.ckpt_path_local))

        print("pretrained_state......", args.model_config.ckpt_path_local)
        pretrained_state = torch.load(args.model_config.ckpt_path_local)
        missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_state, strict=False)
        if dist.get_rank() == 0:
            if len(missing_keys) > 0:
                for miss_key in missing_keys:
                    print(miss_key)
                print("===================")
                for unexpected_key in unexpected_keys:
                    print(unexpected_key)
                print("failed load ckpt from {}...".format(args.model_config.ckpt_path_local))
            else:
                print("successfully load ckpt from {}...".format(args.model_config.ckpt_path_local))

        if dist.get_world_size() > 1:
            distributed_config = set_distributed_config(self.model, args, keys, conv3d_pad_config)
            self.model = convert_to_distributed(self.model, keys, distributed_config, args)

        self.model = self.model.cuda().to(torch.bfloat16)
        self.model = self.model.eval()

        if args.infer_config.infer_type == "graph":
            import torchair
            from torchair import patch_for_hcom
            patch_for_hcom()
            config = torchair.CompilerConfig()
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
            print("run with graph mode...")
            self.model.decoder = torch.compile(
                self.model.decoder,
                mode="default",
                backend=npu_backend,
                dynamic=False,
                fullgraph=False
            )

        self.args = args

    def decode(self, x):
        with torch.no_grad():
            if self.decode_pad > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.decode_pad), mode="constant", value=0)
            x = pad_split_frames(x, self.args, "split_h")
            dec = self.model.decode(x, first_frame=self.args.max_frame % 2 == 1, is_distributed=True)
            if isinstance(dec, tuple):
                dec = dec[0]
            dec = gather_video(dec.contiguous(), self.args)[0]
            result = dec[:,:,:dec.shape[2]-self.decode_pad*self.patch_size[0]]
            return result

