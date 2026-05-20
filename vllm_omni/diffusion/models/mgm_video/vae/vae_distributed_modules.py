# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Consolidated distributed VAE modules (syncbatchnorm + distributed_modules).

This module merges:
  - syncbatchnorm.py       (SyncBatchNorm)
  - distributed_modules.py (RingAttention, SyncGroupNormWithGather,
                            SyncGroupNormWithSyncBN, DistributedConv3D,
                            DistributedConv2D, SplitReslutionConv3D, etc.)
"""

import os
import argparse
import json
import collections
import copy
from functools import partial, lru_cache, wraps
from collections import namedtuple
from typing import Optional

import torch
import torch_npu
import torch.distributed as dist
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import Function
from tqdm import tqdm

try:
    import moxing as mox
except:
    print("no moxing")


# ---------------------------------------------------------------------------
# SyncBatchNorm — originally from syncbatchnorm.py
# ---------------------------------------------------------------------------


class SyncBatchNorm(Function):
 
    @staticmethod
    def forward(self, input_tensor, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size, ori_shape, split_infer=True, split_nums=2):
        input_tensor = input_tensor.contiguous()
        input_shape = input_tensor.shape
        input_tensor_ = input_tensor.reshape(input_shape[0], input_shape[1], 1, -1)
        # calculate sum/sum_square for input.
        if split_infer:
            sum_val_list = []
            sum_square_val_list = []
            assert input_tensor_.shape[1] % split_nums == 0
            split_chunk_size = input_tensor_.shape[1] // split_nums
            for chunk_idx in range(split_nums):
                sum_val, sum_square_val = torch_npu.batch_norm_reduce(input_tensor_[:, chunk_idx*split_chunk_size:(chunk_idx+1)*split_chunk_size, ...], eps)
                sum_val_list.append(sum_val)
                sum_square_val_list.append(sum_square_val)
            sum_val = torch.cat(sum_val_list)
            sum_square_val = torch.cat(sum_square_val_list)
        else:
            sum_val, sum_square_val = torch_npu.batch_norm_reduce(input_tensor_, eps)
 
        count = torch.full((1,),
                           input_tensor.numel() // input_tensor.size(1),
                           dtype=sum_val.dtype,
                           device=sum_val.device)
 
        num_channels = input_tensor.shape[1]
        # C, C, 1 -> (2C + 1)
        combined = torch.cat([sum_val, sum_square_val, count], dim=0)
        # world_size * (2C + 1)
        combined_list = [torch.empty_like(combined) for k in range(world_size)]
        dist.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        sum_all, square_sum_all, count_all = torch.split(combined, num_channels, dim=1)
 
        # calculate global mean & invstd
        mean, invstd = torch_npu.batch_norm_gather_stats_update(input_tensor,
                                                                sum_all,
                                                                square_sum_all,
                                                                running_mean,
                                                                running_var,
                                                                momentum,
                                                                eps,
                                                                count_all.view(-1))
 
        # self.save_for_backward(input_tensor, weight, mean, invstd, count_all)
        self.process_group = process_group
        # apply element-wise normalization
        if len(ori_shape) == 4:
            ratio = input_tensor.shape[2]
            dim = mean.shape[0]  # 4
            chn = weight.shape[0]
            b, _, _, h, w = input_tensor.shape
            input_tensor = input_tensor.view([b, -1, 1, h, w])
            mean = mean.view(-1, 1).expand([dim, ratio]).reshape(-1)
            invstd = invstd.view(-1, 1).expand([dim, ratio]).reshape(-1)
            weight = weight.view(1, -1).expand([ori_shape[0], chn]).reshape(-1)
            bias = bias.view(1, -1).expand([ori_shape[0], chn]).reshape(-1)
            return torch.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps).view(ori_shape)
        else:
            input_tensor = input_tensor.view(ori_shape)
            ratio = input_tensor.shape[1] // 32
            mean = mean.view(-1, 1).expand([32, ratio]).reshape(-1)
            invstd = invstd.view(-1, 1).expand([32, ratio]).reshape(-1)
            return torch.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps)
 
    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_contiguous(memory_format=torch.channels_last):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
 
        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output,
                                                                                      saved_input,
                                                                                      mean,
                                                                                      invstd,
                                                                                      weight,
                                                                                      self.needs_input_grad[0],
                                                                                      self.needs_input_grad[1],
                                                                                      self.needs_input_grad[2])
 
        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(
                combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)
 
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(grad_output,
                                                         saved_input,
                                                         mean,
                                                         invstd,
                                                         weight,
                                                         sum_dy,
                                                         sum_dy_xmu,
                                                         count_tensor)
 
        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None
 
        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None
 
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


# Alias used by distributed_modules code below
sync_bn_ops = SyncBatchNorm


# ---------------------------------------------------------------------------
# Distributed modules — originally from distributed_modules.py
# (SyncBatchNorm is referenced as sync_bn_ops within this section)
# ---------------------------------------------------------------------------


cache = partial(lru_cache, maxsize = None)
RingInfo = namedtuple('RingInfo', ['ring_rank', 'iter_info'])

class RingAttention:
    def __init__(self):
        self.name = "ring_attention"
        self.rank = dist.get_rank()

    def exists(self, v):
        return v is not None

    @cache()
    def get_rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def cast_tuple(self, t, length=1):
        return t if isinstance(t, tuple) else ((t,) * length)

    def default(self, v, d):
        return v if self.exists(v) else d

    def circular_index_left(self, pos, ring_size, num=1):
        return ((pos - num) + ring_size) % ring_size

    def circular_index_right(self, pos, ring_size, num=1):
        return (pos + num) % ring_size

    def circular_rank_left(self, rank=None, ring_size=None, num=1):
        rank = self.default(rank, self.get_rank())
        ring_size = self.default(ring_size, dist.get_world_size())
        ring_set_num = rank // ring_size
        offset = ring_set_num * ring_size
        return self.circular_index_left(rank, ring_size, num) + offset

    def circular_rank_right(self, rank=None, ring_size=None, num=1):
        rank = self.default(rank, self.get_rank())
        ring_size = self.default(ring_size, dist.get_world_size())
        ring_set_num = rank // ring_size
        offset = ring_set_num * ring_size
        return self.circular_index_right(rank, ring_size, num) + offset

    def send_and_receive_(self, x, receive_buffer, send_to_rank, receive_from_rank):
        send_op = dist.P2POp(dist.isend, x, send_to_rank)
        recv_op = dist.P2POp(dist.irecv, receive_buffer, receive_from_rank)

        reqs = dist.batch_isend_irecv([send_op, recv_op])

        for req in reqs:
            req.wait()

    def ring_pass(
        self, 
        x: Tensor,
        receive_buffer: Optional[Tensor] = None,
        ring_size: Optional[int] = None
    ):
        ring_size = self.default(ring_size, dist.get_world_size())
        x = x.contiguous()

        if not self.exists(receive_buffer):
            receive_buffer = torch.zeros_like(x)
        else:
            receive_buffer = receive_buffer.contiguous()

        self.send_and_receive_(x, receive_buffer, self.circular_rank_right(ring_size=ring_size), self.circular_rank_left(ring_size=ring_size))
        return receive_buffer, x

    def all_ring_pass(self, *tensors, max_iters = None, receive_buffers = None, ring_size = None):
        ring_size = self.default(ring_size, dist.get_world_size())  # world_size
        max_iters = self.default(max_iters, ring_size)  # world_size

        receive_buffers = self.cast_tuple(receive_buffers, len(tensors))

        # make sure iteration is between 1 and world size

        total_iters = max(1, min(ring_size, max_iters))  # world_size

        curr_ring_pos = self.get_rank()

        for ind in range(total_iters):
            is_first = ind == 0
            is_last = ind == (total_iters - 1)

            yield RingInfo(curr_ring_pos, (is_first,  is_last)), (tensors, receive_buffers)

            curr_ring_pos = self.circular_index_left(curr_ring_pos, ring_size)

            if is_last:
                continue

            new_tensors = []
            new_receive_buffers = []

            for tensor, receive_buffer in zip(tensors, receive_buffers):
                if self.exists(tensor):
                    new_tensor, new_receive_buffer = self.ring_pass(tensor, receive_buffer, ring_size)
                else:
                    new_tensor, new_receive_buffer = None, None

                new_tensors.append(new_tensor)
                new_receive_buffers.append(new_receive_buffer)

            tensors = new_tensors
            receive_buffers = new_receive_buffers

    def run_ring_attention(self, q, k, v):
        b, s, c = q.shape
        ring_size = dist.get_world_size()
        receive_kv = None
        receive_mask = None
        mask = None
        max_ring_passes = None

        kv = torch.stack((k, v))
        k_ori, v_ori = k.clone(), v.clone()

        prev_attn_out = None
        prev_m = None
        prev_lse = None

        for (ring_rank, (is_first, is_last)), ((kv, mask), (receive_kv, receive_mask)) in self.all_ring_pass(
            kv, mask, receive_buffers=(receive_kv, receive_mask), max_iters=max_ring_passes, ring_size=ring_size
        ):
            k, v = kv

            cur_attn_out, _, cur_lse, _, _, _, _ = torch_npu.npu_fusion_attention(
                q, k, v, 1,
                atten_mask=None,
                scale=(c // 1) ** -0.5,
                keep_prob=1.0,
                input_layout="BSH",
            )

            # cur_lse = cur_lse[:, 0, :, 0].unsqueeze(-1).to(torch.float32)
            # cur_attn_out = cur_attn_out.to(torch.float32)
            cur_lse = cur_lse[:, 0, :, 0].unsqueeze(-1)

            if prev_attn_out is not None:
                lse_merge = prev_lse + torch.log(1+torch.exp(cur_lse-prev_lse))
                prev_attn_out = prev_attn_out*torch.exp(prev_lse-lse_merge) + cur_attn_out*torch.exp(cur_lse-lse_merge)
                prev_lse = lse_merge
            else:
                prev_attn_out = cur_attn_out
                prev_lse = cur_lse
        
        return prev_attn_out.to(torch.bfloat16)

class SyncGroupNormWithSyncBN(nn.Module):
    def __init__(self, in_channels, batch_size=1, groups=32, eps=1e-6, split_infer=False):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.running_batch_size = batch_size
        self.weight = nn.Parameter(torch.zeros(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.running_mean = (None)
        self.running_var = (None)
        self.eps = eps
        self.process_group = torch.distributed.group.WORLD
        self.world_size = torch.distributed.get_world_size(self.process_group)
        self.exponential_average_factor = 0.0
        self.split_infer = split_infer
    
    def forward(self, x):
        # x:[b,c,t,h,w]
        if len(x.shape) == 5:
            b, c, t, h, w = x.shape
            assert self.running_batch_size == b
            x = x.reshape(1, b*self.groups, c//self.groups*t, h, w)
            x = sync_bn_ops.apply(
                x, self.weight, self.bias,
                self.running_mean, self.running_var,
                self.eps, self.exponential_average_factor,
                self.process_group, self.world_size,
                [b, c, t, h, w],
                self.split_infer
            )
        else:
            b, c, h, w = x.shape
            x = x.reshape(1, b*self.groups, c//self.groups, h, w)
            x = sync_bn_ops.apply(
                x, self.weight, self.bias,
                self.running_mean, self.running_var,
                self.eps, self.exponential_average_factor,
                self.process_group, self.world_size,
                [b, c, h, w],
                self.split_infer
            )
        return x

def dump_tensor(x, name, split_type="split_h", is_distributed=True, is_gather=True):
    dump_path = "dumps"
    os.makedirs(dump_path, exist_ok=True)
    if dist.get_world_size() == 1:
        name = "one_card_"+name
        torch.save(x, os.path.join(dump_path, "{}.pth".format(name)))
    elif not is_distributed:
        if is_gather:
            name = "distributed_"+name
        else:
            name = "one_card_"+name
        print("save..........................................", name)
        torch.save(x, os.path.join(dump_path, "{}.pth".format(name)))
    elif len(x.shape) == 3:  # gather sequence
        tensor_list = [torch.zeros(x.shape, device=torch.cuda.current_device(), dtype=x.dtype) for _ in range(dist.get_world_size())]
        x = x.contiguous()
        dist.all_gather(tensor_list, x)
        x = torch.cat(tensor_list, dim=1).cpu()
        name = "distributed_"+name
        torch.save(x, os.path.join(dump_path, "{}.pth".format(name)))
    else:
        def slice_tensor(tensor_list, all_shapes, axis):
            for idx, shape in enumerate(all_shapes):
                if split_type == "split_h":
                    tensor_list[idx] = tensor_list[idx][:, :, :, :shape[axis], :]
                elif split_type == "split_w":
                    tensor_list[idx] = tensor_list[idx][:, :, :, :, :shape[axis]]
                elif split_type == "split_t":
                    tensor_list[idx] = tensor_list[idx][:, :, :shape[axis], :, :]
            return tensor_list

        all_shapes = [torch.zeros([5], device=torch.cuda.current_device(), dtype=torch.int32) for _ in range(dist.get_world_size())]
        shape_tensor = torch.tensor(x.shape, device=torch.cuda.current_device(), dtype=torch.int32)
        dist.all_gather(all_shapes, shape_tensor)
        gather_axis = {
            "split_h": [3, 3],
            "split_w": [4, 1],
            "split_t": [2, 5],
        }

        axis = gather_axis[split_type][0]
        max_axis_val = max([i[axis] for i in all_shapes])
        padding_info = [0,0,0,0,0,0]

        padding_info[gather_axis[split_type][1]] = max_axis_val-x.shape[axis]
        padding_info = tuple(padding_info)
        b, c, t, h, w = x.shape
        if max_axis_val - x.shape[axis] > 0:
            x = torch.nn.functional.pad(x, padding_info, mode="constant", value=0)

        tensor_list = [torch.zeros(x.shape, device=torch.cuda.current_device(), dtype=x.dtype) for _ in range(dist.get_world_size())]
        x = x.contiguous()
        dist.all_gather(tensor_list, x)
        tensor_list = slice_tensor(tensor_list, all_shapes, axis)
        x = torch.cat(tensor_list, dim=axis).cpu()
        name = "distributed_"+name
        torch.save(x, os.path.join(dump_path, "{}.pth".format(name)))


class SyncGroupNormWithGather(nn.Module):
    def __init__(self, in_channels, batch_size=1, groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))
    
    def forward(self, x, eps=1e-06):
        b, c, t, h, w = x.shape
        assert self.batch_size == b
        x = x.contiguous()
        x = x.view(b*self.groups, -1)

        x_square_sum = torch.mean(x*x, dim=-1, keepdim=True)
        x_sum = torch.mean(x, dim=-1, keepdim=True)

        # rms_weight = nn.Parameter(torch.ones(x.shape[-1], dtype=x.dtype, device=torch.cuda.current_device()))
        # x_square_sum = ((1.0/torch_npu.npu_rms_norm(x, rms_weight, epsilon=0)[1])**2).to(torch.bfloat16)
        # x_sum = torch.mean(x, dim=-1, keepdim=True)

        dist.all_reduce(x_square_sum)
        dist.all_reduce(x_sum)

        global_mean = x_sum / dist.get_world_size()
        global_var = x_square_sum / dist.get_world_size() - torch.square(global_mean)
        # inv_global_var = 1.0 / (torch.sqrt(global_var + eps))

        # # apply element-wise normalization  1/(sqrt(var+eps))
        # x = x.view([b, c, t, h, w])
        # ratio = x.shape[1] // 32
        # global_mean = global_mean.view(-1, 1).expand([32, ratio]).reshape(-1)
        # inv_global_var = inv_global_var.view(-1, 1).expand([32, ratio]).reshape(-1)
        # return torch.batch_norm_elemt(x, self.weight, self.bias, global_mean, inv_global_var, eps)

        x = (x-global_mean) / torch.sqrt(global_var+eps)
        x = x.unsqueeze(0)

        x = x.view(b, c, t, h, w)

        weight = self.weight.unsqueeze(0)
        bias = self.bias.unsqueeze(0)
        while len(weight.size()) != len(x.size()):
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        return torch.addcmul(bias, x, weight)

    def forward_1(self, x, eps=1e-06):
        b, c, t, h, w = x.shape
        assert self.batch_size == b
        x = x.contiguous()
        x = x.view(b*self.groups, -1)

        x_square_sum = torch.mean(x*x, dim=-1, keepdim=True)
        x_sum = torch.mean(x, dim=-1, keepdim=True)

        dist.all_reduce(x_square_sum)
        dist.all_reduce(x_sum)

        rank_size = dist.get_world_size()

        global_mean = x_sum / rank_size
        global_var = x_square_sum / rank_size - torch.square(global_mean)
        
        inv_global_var = 1.0 / (torch.sqrt(global_var + eps))

        # # apply element-wise normalization  1/(sqrt(var+eps))
        ratio = c // self.groups
        global_mean = global_mean.view(-1, 1).expand([self.groups, ratio]).reshape(-1)
        inv_global_var = inv_global_var.view(-1, 1).expand([self.groups, ratio]).reshape(-1)
        return torch.batch_norm_elemt(x.view(b, c, t, h, w), self.weight, self.bias, global_mean, inv_global_var, 0)

class SearchP2P:
    def __init__(self):
        self.name = "search_p2p"

    def build_dependency_intra(self, node_name):
        self.all_inter_combinations = []
        self.all_inter_combinations_keys = set()
        self.generate_candidates(node_name)
    
    def find_combination(self, visited, combinations, combination_keys):
        if len(list(visited)) == len(self.inter_node_communicate):
            key = "".join([str(i) for i in sorted(combination_keys)])
            if key not in self.all_inter_combinations_keys:
                self.all_inter_combinations.append(copy.deepcopy(combinations))
                self.all_inter_combinations_keys.add(key)
            return
        for idx1, item1 in enumerate(self.inter_node_communicate):
            if tuple(item1) in visited:
                continue
            for idx2, item2 in enumerate(self.inter_node_communicate):
                if tuple(item2) in visited:
                    continue
                if item1[0] not in item2 and item1[1] not in item2:
                    visited.add(tuple(item1))
                    visited.add(tuple(item2))
                    combination_keys.append(int("".join([str(i) for i in sorted([idx1, idx2])])))
                    combinations.append([item1, item2])
                    self.find_combination(visited, combinations, combination_keys)
                    visited.remove(tuple(item1))
                    visited.remove(tuple(item2))
                    combinations.pop(-1)
                    combination_keys.pop(-1)
    
    def search_one_group(self, name, visited, visited_ranks, combinations, combination_keys, node_communicate, prefix, is_intra):
        current_recrusive_visited = set()
        for idx2 in node_communicate:
            if is_intra:
                item2 = self.intra_node_communicate[idx2]
            else:
                item2 = self.inter_node_communicate[idx2]
            if tuple(item2) in visited:
                continue
            if item2[0] not in visited_ranks and item2[1] not in visited_ranks:
                visited_ranks.add(item2[0])
                visited_ranks.add(item2[1])
                visited.add(tuple(item2))
                combinations.append(item2)
                combination_keys.append(idx2)
                self.search_one_group(name, visited, visited_ranks, combinations, combination_keys, node_communicate, prefix, is_intra)
                visited_ranks.remove(item2[0])
                visited_ranks.remove(item2[1])
                visited.remove(tuple(item2))
                combinations.pop(-1)
                combination_keys.pop(-1)
            else:
                current_recrusive_visited.add(tuple(item2))
                visited.add(tuple(item2))

        if len(visited) == len(node_communicate):
            if len(combination_keys) > 0:
                key = ",".join([str(i) for i in sorted(combination_keys)])  # intra node idx
                if not is_intra:
                    if key not in self.all_combinations_keys_inter[name]:
                        if len(self.communicate_groups_with_inter[name]) == 0:
                            self.all_combinations_max_length_inter[name] = len(combinations)
                            self.communicate_groups_with_inter[name].append(copy.deepcopy(combinations))
                            self.all_combinations_keys_inter[name].append(key)
                        else:
                            if len(combinations) > self.all_combinations_max_length_inter[name]:
                                self.communicate_groups_with_inter[name] = []
                                self.communicate_groups_with_inter[name].append(copy.deepcopy(combinations))
                                self.all_combinations_keys_inter[name] = []
                                self.all_combinations_keys_inter[name].append(key)
                                self.all_combinations_max_length_inter[name] = len(combinations)
                            elif len(combinations) == self.all_combinations_max_length_inter[name]:
                                self.communicate_groups_with_inter[name].append(copy.deepcopy(combinations))
                                self.all_combinations_keys_inter[name].append(key)
                else:
                    if key not in self.all_combinations_keys_intra[name]:
                        if len(self.communicate_groups_with_intra[name]) == 0:
                            self.all_combinations_max_length_intra[name] = len(combinations)
                            self.communicate_groups_with_intra[name].append(copy.deepcopy(combinations))
                            self.all_combinations_keys_intra[name].append(key)
                        else:
                            if len(combinations) > self.all_combinations_max_length_intra[name]:
                                self.communicate_groups_with_intra[name] = []
                                self.communicate_groups_with_intra[name].append(copy.deepcopy(combinations))
                                self.all_combinations_keys_intra[name] = []
                                self.all_combinations_keys_intra[name].append(key)
                                self.all_combinations_max_length_intra[name] = len(combinations)
                            elif len(combinations) == self.all_combinations_max_length_intra[name]:
                                self.communicate_groups_with_intra[name].append(copy.deepcopy(combinations))
                                self.all_combinations_keys_intra[name].append(key)

        for item in list(current_recrusive_visited):
            visited.remove(tuple(item))

    def update_tree(self, name, idx, cur_idx, node_communicate, visited_ranks, combinations, prefix="", is_intra=False):
        if len(node_communicate) == 0:
            return
        if idx == cur_idx:
            self.search_one_group(name, set(), visited_ranks, combinations, [], node_communicate, prefix, is_intra)
            return
        
        if is_intra:
            prev_combinations_keys = self.all_combinations_keys_intra[name]
        else:
            prev_combinations_keys = self.all_combinations_keys_inter[name]

        for prev_cand_idx, cand_key in enumerate(prev_combinations_keys):
            if cand_key == "":
                index = []
            else:
                index = [int(i) for i in cand_key.split(",")]
            new_node_communicate = []
            for del_idx in node_communicate:
                if del_idx not in index:
                    new_node_communicate.append(del_idx)  # update intra using the rest
                # elif cur_idx in index and len(combinations) == 1:
                #     combinations = []
                #     visited_ranks = set()

            if len(new_node_communicate) == 0:
                return

            next_name = name + "-{}-{}L{}".format(prev_cand_idx, prefix, idx+1)
            self.update_tree(next_name, idx+1, cur_idx, new_node_communicate, visited_ranks, combinations, prefix, is_intra=is_intra)

    def build_candidate_tree(self, cur_idx, visited_ranks, combinations, is_intra, prefix=""):
        """
        cur_idx: self.all_inter_combinations[0] index
        visited_ranks: current visited ranks
        combinations: self.all_inter_combinations[0][index]
        """
        if is_intra:
            node_communicate = list(range(len(self.intra_node_communicate)))
        else:
            node_communicate = list(range(len(self.inter_node_communicate)))
        if cur_idx == 0:
            name = "{}L{}".format(prefix, cur_idx)
            self.search_one_group(name, set(), visited_ranks, combinations, [], node_communicate, prefix, is_intra)
        else:
            self.update_tree("{}L0".format(prefix), 0, cur_idx, node_communicate, visited_ranks, combinations, prefix, is_intra)
    
    def list_to_str(self, item):
        return [str(i) for i in item]
    
    def filter_groups(self, groups, communicate_groups, all_combinations_keys, is_intra=False):
        for key in communicate_groups:
            key_info = key.split("-")
            visited_ranks = []
            combinations = []
            for idx in range(len(key_info)-1):
                if idx % 2 == 0:
                    prev_keys = "-".join(key_info[:idx+1])
                    prev_idxs = int(key_info[idx+1])
                    try:
                        visited_keys = [int(i) for i in all_combinations_keys[prev_keys][prev_idxs].split(",")]
                    except:
                        breakpoint()
                    visited_ranks.extend(visited_keys)
                    combinations.append(communicate_groups[prev_keys][prev_idxs])

            last_keys = all_combinations_keys[key]
            visited_ranks = list(set(visited_ranks))

            if is_intra:
                node_communicate = self.intra_node_communicate
            else:
                node_communicate = self.inter_node_communicate

            for idx, last_key in enumerate(last_keys):
                last_ranks = [int(i) for i in all_combinations_keys[key][idx].split(",")]
                if len(visited_ranks)+len(last_ranks) == len(node_communicate):
                    print(key)
                    try:
                        assert len(communicate_groups[key]) == 1 and len(last_keys) == 1
                    except:
                        breakpoint()
                    groups[key] = combinations + communicate_groups[key]
        return groups
        
    def generate_candidates(self, node_name):
        self.communicate_groups_with_intra = collections.defaultdict(list)
        self.all_combinations_keys_intra = collections.defaultdict(list)
        self.all_combinations_max_length_intra = collections.defaultdict(list)

        self.communicate_groups_with_inter = collections.defaultdict(list)
        self.all_combinations_keys_inter = collections.defaultdict(list)
        self.all_combinations_max_length_inter = collections.defaultdict(list)

        print("building intra")
        if len(self.intra_node_communicate) > 0:
            for idx1, item1 in enumerate(tqdm(self.intra_node_communicate)):
                self.build_candidate_tree(idx1, set(), [], is_intra=True)

            # gather groups
            groups_intra = collections.defaultdict(list)
            self.groups_intra = self.filter_groups(
                groups_intra, 
                self.communicate_groups_with_intra,
                self.all_combinations_keys_intra,
                is_intra=True
            )

            json.dump(self.groups_intra, open("groups_intra_{}.json".format(node_name), "w"))

        print("building inter")
        if len(self.inter_node_communicate) > 0:
            for idx1, item1 in enumerate(tqdm(self.inter_node_communicate)):
                self.build_candidate_tree(idx1, set(), [], is_intra=False)
            
            groups_inter = collections.defaultdict(list)
            self.groups_inter = self.filter_groups(
                groups_inter, 
                self.communicate_groups_with_inter,
                self.all_combinations_keys_inter,
                is_intra=False
            )

            json.dump(self.groups_inter, open("groups_inter_{}.json".format(node_name), "w"))

    def communicate_filter(self, json_file, intra_comm_volumn, relative_comm_volumn):
        comm_topo_groups = json.load(open(json_file))
        topo_groups_comm_attr = collections.defaultdict(list)
        topo_groups_comm_attr_name = collections.defaultdict(list)
        for group_name, topo_groups in comm_topo_groups.items():
            for group_idx, groups in enumerate(topo_groups):
                group_comm_attr = set()
                for group in groups:
                    for comm_key in intra_comm_volumn:
                        comm_list = intra_comm_volumn[comm_key]
                        if group in comm_list:
                            group_comm_attr.add(comm_key)
                topo_groups_comm_attr[group_name].append(len(group_comm_attr))
                topo_groups_comm_attr_name[group_name].append(list(group_comm_attr))
        
        valid_topo_group_names = []
        valid_topo_groups = []
        valid_topo_groups_comm_name = []
        for group_name in topo_groups_comm_attr:
            if sum(topo_groups_comm_attr[group_name]) / len(topo_groups_comm_attr[group_name]) == 1:
                valid_topo_group_names.append(group_name)
                valid_topo_groups.append(comm_topo_groups[group_name])
                valid_topo_groups_comm_name.append(topo_groups_comm_attr_name[group_name])
        return valid_topo_group_names, valid_topo_groups, valid_topo_groups_comm_name

    def merge_intra_comm_groups(self, intra_node_groups, intra_node_groups_comm_names):
        length_codes = set()
        for groups in intra_node_groups:
            length_codes.add("".join([str(len(group)) for group in groups]))
        assert len(length_codes) == 1

        for items in zip(*intra_node_groups_comm_names):
            tmp_item = set()
            for item in items:
                tmp_item.add(tuple(item))
            if len(tmp_item) > 1:
                raise Exception

        intra_comm_groups = []
        for items in zip(*intra_node_groups):
            tmp_item = []
            for item in items:
                tmp_item += item
            intra_comm_groups.append(tmp_item)
        return intra_comm_groups, intra_node_groups_comm_names[0]

class DistributedConv3D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        conv_config, 
        stride=1, 
        padding=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        self.kernel_size = kernel_size
        assert kernel_size in [1, 3],"{} not support kernel_size except 1,3".format(kernel_size)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.conv_config = conv_config
        self.padding_info = conv_config["padding_info"]
        self.is_upsample = conv_config["is_upsample"]
        self.is_downsample = conv_config["is_downsample"]
        self.split_stragety = conv_config["split_stragety"]

        self.func_map = {
            "height_left":self.height_left,
            "height_right":self.height_right,
            "width_bottom":self.width_bottom,
            "width_upper":self.width_upper,
            "corner_right_bottom":self.corner_right_bottom,
            "corner_left_upper":self.corner_left_upper,
            "corner_left_bottom":self.corner_left_bottom,
            "corner_right_upper":self.corner_right_upper,
        }

        self.search_p2p = SearchP2P()

        if conv_config["conv_type"] == "split_in_out_channel":
            self.split_nums_in = conv_config["split_nums_in"]
            self.split_nums_out = conv_config["split_nums_out"]
            self.conv = SplitInOutchannelConv3d(
                self.split_nums_in,
                self.split_nums_out,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            self.conv = torch.nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        
        self.neighbour_size = self.kernel_size - 2
    
    def configure_split_h_p2p(self, t, w):
        if self.world_size == 2:
            self.p2p_send_matrix_group1 = {
                1:0
            }
            self.p2p_recv_matrix_group1 = {
                0:1
            }

            self.p2p_send_matrix_group2 = {
                0:1
            }
            self.p2p_recv_matrix_group2 = {
                1:0
            }
        elif self.world_size == 4:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3
            }
            self.p2p_send_matrix_group2 = {
                0:1, 2:3
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2
            }
            self.p2p_send_matrix_group3 = {
                2:1
            }
            self.p2p_recv_matrix_group3 = {
                1:2
            }
            self.p2p_send_matrix_group4 = {
                1:2
            }
            self.p2p_recv_matrix_group4 = {
                2:1
            }
        elif self.world_size == 8:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2, 5:4, 7:6
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3, 4:5, 6:7
            }
            
            self.p2p_send_matrix_group2 = {
                0:1, 2:3, 4:5, 6:7
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2, 5:4, 7:6
            }

            self.p2p_send_matrix_group3 = {
                2:1, 4:3, 6:5
            }
            self.p2p_recv_matrix_group3 = {
                1:2, 3:4, 5:6
            }

            self.p2p_send_matrix_group4 = {
                1:2, 3:4, 5:6
            }
            self.p2p_recv_matrix_group4 = {
                2:1, 4:3, 6:5
            }
        elif self.world_size == 16:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2, 5:4, 7:6, 9:8, 11:10, 13:12, 15:14
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3, 4:5, 6:7, 8:9, 10:11, 12:13, 14:15
            }
            
            self.p2p_send_matrix_group2 = {
                0:1, 2:3, 4:5, 6:7, 8:9, 10:11, 12:13, 14:15
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2, 5:4, 7:6, 9:8, 11:10, 13:12, 15:14
            }

            self.p2p_send_matrix_group3 = {
                2:1, 4:3, 6:5, 8:7, 10:9, 11:10, 13:12
            }
            self.p2p_recv_matrix_group3 = {
                1:2, 3:4, 5:6, 7:8, 9:10, 10:11, 12:13
            }

            self.p2p_send_matrix_group4 = {
                1:2, 3:4, 5:6, 7:8, 9:10, 10:11, 12:13
            }
            self.p2p_recv_matrix_group4 = {
                2:1, 4:3, 6:5, 8:7, 10:9, 11:10, 13:12
            }
        self.recv_buffer_head = torch.zeros([1, self.in_channels, t, self.kernel_size-2, w], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_tail = torch.zeros([1, self.in_channels, t, self.kernel_size-2, w], device=torch.cuda.current_device(), dtype=torch.bfloat16)

    def configure_split_w_p2p(self, t, h):
        if self.world_size == 2:
            self.p2p_send_matrix_group1 = {
                1:0
            }
            self.p2p_recv_matrix_group1 = {
                0:1
            }

            self.p2p_send_matrix_group2 = {
                0:1
            }
            self.p2p_recv_matrix_group2 = {
                1:0
            }
        elif self.world_size == 4:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3
            }
            self.p2p_send_matrix_group2 = {
                0:1, 2:3
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2
            }
            self.p2p_send_matrix_group3 = {
                2:1
            }
            self.p2p_recv_matrix_group3 = {
                1:2
            }
            self.p2p_send_matrix_group4 = {
                1:2
            }
            self.p2p_recv_matrix_group4 = {
                2:1
            }
        elif self.world_size == 8:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2, 5:4, 7:6
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3, 4:5, 6:7
            }
            
            self.p2p_send_matrix_group2 = {
                0:1, 2:3, 4:5, 6:7
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2, 5:4, 7:6
            }

            self.p2p_send_matrix_group3 = {
                2:1, 4:3, 6:5
            }
            self.p2p_recv_matrix_group3 = {
                1:2, 3:4, 5:6
            }

            self.p2p_send_matrix_group4 = {
                1:2, 3:4, 5:6
            }
            self.p2p_recv_matrix_group4 = {
                2:1, 4:3, 6:5
            }
        elif self.world_size == 16:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2, 5:4, 7:6, 9:8, 11:10, 13:12, 15:14
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3, 4:5, 6:7, 8:9, 10:11, 12:13, 14:15
            }
            
            self.p2p_send_matrix_group2 = {
                0:1, 2:3, 4:5, 6:7, 8:9, 10:11, 12:13, 14:15
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2, 5:4, 7:6, 9:8, 11:10, 13:12, 15:14
            }

            self.p2p_send_matrix_group3 = {
                2:1, 4:3, 6:5, 8:7, 10:9, 11:10, 13:12
            }
            self.p2p_recv_matrix_group3 = {
                1:2, 3:4, 5:6, 7:8, 9:10, 10:11, 12:13
            }

            self.p2p_send_matrix_group4 = {
                1:2, 3:4, 5:6, 7:8, 9:10, 10:11, 12:13
            }
            self.p2p_recv_matrix_group4 = {
                2:1, 4:3, 6:5, 8:7, 10:9, 11:10, 13:12
            }
        self.recv_buffer_head = torch.zeros([1, self.in_channels, t, h, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_tail = torch.zeros([1, self.in_channels, t, h, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)

    def configure_split_t_p2p(self, h, w):
        if self.world_size == 2:
            self.p2p_send_matrix_group1 = {
                1:0
            }
            self.p2p_recv_matrix_group1 = {
                0:1
            }

            self.p2p_send_matrix_group2 = {
                0:1
            }
            self.p2p_recv_matrix_group2 = {
                1:0
            }
        elif self.world_size == 4:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3
            }
            self.p2p_send_matrix_group2 = {
                0:1, 2:3
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2
            }
            self.p2p_send_matrix_group3 = {
                2:1
            }
            self.p2p_recv_matrix_group3 = {
                1:2
            }
            self.p2p_send_matrix_group4 = {
                1:2
            }
            self.p2p_recv_matrix_group4 = {
                2:1
            }
        elif self.world_size == 8:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2, 5:4, 7:6
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3, 4:5, 6:7
            }
            
            self.p2p_send_matrix_group2 = {
                0:1, 2:3, 4:5, 6:7
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2, 5:4, 7:6
            }

            self.p2p_send_matrix_group3 = {
                2:1, 4:3, 6:5
            }
            self.p2p_recv_matrix_group3 = {
                1:2, 3:4, 5:6
            }

            self.p2p_send_matrix_group4 = {
                1:2, 3:4, 5:6
            }
            self.p2p_recv_matrix_group4 = {
                2:1, 4:3, 6:5
            }
        elif self.world_size == 16:
            self.p2p_send_matrix_group1 = {
                1:0, 3:2, 5:4, 7:6, 9:8, 11:10, 13:12, 15:14
            }
            self.p2p_recv_matrix_group1 = {
                0:1, 2:3, 4:5, 6:7, 8:9, 10:11, 12:13, 14:15
            }
            
            self.p2p_send_matrix_group2 = {
                0:1, 2:3, 4:5, 6:7, 8:9, 10:11, 12:13, 14:15
            }
            self.p2p_recv_matrix_group2 = {
                1:0, 3:2, 5:4, 7:6, 9:8, 11:10, 13:12, 15:14
            }

            self.p2p_send_matrix_group3 = {
                2:1, 4:3, 6:5, 8:7, 10:9, 11:10, 13:12
            }
            self.p2p_recv_matrix_group3 = {
                1:2, 3:4, 5:6, 7:8, 9:10, 10:11, 12:13
            }

            self.p2p_send_matrix_group4 = {
                1:2, 3:4, 5:6, 7:8, 9:10, 10:11, 12:13
            }
            self.p2p_recv_matrix_group4 = {
                2:1, 4:3, 6:5, 8:7, 10:9, 11:10, 13:12
            }
        self.recv_buffer_head = torch.zeros([1, self.in_channels, self.kernel_size-2, h, w], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_tail = torch.zeros([1, self.in_channels, self.kernel_size-2, h, w], device=torch.cuda.current_device(), dtype=torch.bfloat16)

    def configure_search_p2p(self, t, h, w):
        self.groups = []
        self.groups_info = []
        communication_info = self.split_stragety["communication_info"]
        for group in self.split_stragety["communication_groups"]:
            group_send_info = {}
            group_recv_info = {}
            # send
            g_0 = group[0]
            group_info = communication_info["{},{}".format(g_0[0], g_0[1])]
            self.groups_info.append(group_info)
            for g in group:
                group_send_info[g[0]] = g[1]
            self.groups.append(group_send_info)
            # recv
            group_reverse = [g[::-1] for g in group]
            g_0_reverse = group_reverse[0]
            group_info_reverse = communication_info["{},{}".format(g_0_reverse[0], g_0_reverse[1])]
            for g in group_reverse:
                group_recv_info[g[0]] = g[1]
            self.groups.append(group_recv_info)
            self.groups_info.append(group_info_reverse)

        self.recv_buffer_height_left = torch.zeros([1, self.in_channels, t, h, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_height_right = torch.zeros([1, self.in_channels, t, h, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_width_bottom = torch.zeros([1, self.in_channels, t, self.kernel_size-2, w], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_width_upper = torch.zeros([1, self.in_channels, t, self.kernel_size-2, w], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_corner_right_bottom = torch.zeros([1, self.in_channels, t, self.kernel_size-2, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_corner_left_upper = torch.zeros([1, self.in_channels, t, self.kernel_size-2, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_corner_left_bottom = torch.zeros([1, self.in_channels, t, self.kernel_size-2, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.recv_buffer_corner_right_upper = torch.zeros([1, self.in_channels, t, self.kernel_size-2, self.kernel_size-2], device=torch.cuda.current_device(), dtype=torch.bfloat16)

    def h_split_p2p(self, x):
        h_head, h_tail = self.padding_info[2], self.padding_info[3]
        if self.world_size == 2:
            if self.rank in self.p2p_send_matrix_group1:
                data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group1[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group1:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group1[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group2:
                data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group2[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group2:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group2[self.rank], group=None)
        else:
            if self.rank in self.p2p_send_matrix_group1:
                data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group1[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group1:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group1[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group2:
                data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group2[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group2:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group2[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group3:
                data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group3[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group3:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group3[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group4:
                data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group4[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group4:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group4[self.rank], group=None)

    def w_split_p2p(self, x):
        w_head, w_tail = self.padding_info[0], self.padding_info[1]
        if self.world_size == 2:
            if self.rank in self.p2p_send_matrix_group1:
                data_send = x[:, :, :, :, w_head:self.kernel_size-2+w_head].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group1[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group1:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group1[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group2:
                data_send = x[:, :, :, :, -self.kernel_size+2-w_tail:-w_tail].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group2[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group2:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group2[self.rank], group=None)
        else:
            if self.rank in self.p2p_send_matrix_group1:
                data_send = x[:, :, :, :, w_head:self.kernel_size-2+w_head].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group1[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group1:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group1[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group2:
                data_send = x[:, :, :, :, -self.kernel_size+2-w_tail:-w_tail].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group2[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group2:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group2[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group3:
                data_send = x[:, :, :, :, w_head:self.kernel_size-2+w_head].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group3[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group3:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group3[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group4:
                data_send = x[:, :, :, :, -self.kernel_size+2-w_tail:-w_tail].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group4[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group4:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group4[self.rank], group=None)

    def t_split_p2p(self, x):
        h_head, h_tail = self.padding_info[2], self.padding_info[3]
        if self.world_size == 2:
            if self.rank in self.p2p_send_matrix_group1:
                data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group1[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group1:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group1[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group2:
                data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group2[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group2:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group2[self.rank], group=None)
        else:
            if self.rank in self.p2p_send_matrix_group1:
                data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group1[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group1:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group1[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group2:
                data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group2[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group2:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group2[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group3:
                data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group3[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group3:
                dist.recv(self.recv_buffer_tail, self.p2p_recv_matrix_group3[self.rank], group=None)
            
            if self.rank in self.p2p_send_matrix_group4:
                data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
                dist.send(data_send, self.p2p_send_matrix_group4[self.rank], group=None)
            if self.rank in self.p2p_recv_matrix_group4:
                dist.recv(self.recv_buffer_head, self.p2p_recv_matrix_group4[self.rank], group=None)

    def forward_w_split_p2p(self, x):
        b, c, t, h, w = x.shape
        if self.kernel_size == 1:
            x = self.conv(x)
        else:
            self.configure_split_w_p2p(t, h)
            self.w_split_p2p(x)
            if self.is_downsample:
                if self.rank != self.world_size-1:
                    x[:, :, :, :, -self.recv_buffer_tail.shape[4]:] = self.recv_buffer_tail
            else:
                if self.rank == 0:
                    x[:, :, :, :, -self.recv_buffer_tail.shape[4]:] = self.recv_buffer_tail
                elif self.rank == self.world_size-1:
                    x[:, :, :, :, :self.recv_buffer_head.shape[4]] = self.recv_buffer_head
                else:
                    x[:, :, :, :, :self.recv_buffer_head.shape[4]] = self.recv_buffer_head
                    x[:, :, :, :, -self.recv_buffer_tail.shape[4]:] = self.recv_buffer_tail

            x = self.conv(x)
            del self.recv_buffer_head
            del self.recv_buffer_tail

        return x

    def forward_h_split_p2p(self, x):
        b, c, t, h, w = x.shape
        if self.kernel_size == 1:
            x = self.conv(x)
        else:
            self.configure_split_h_p2p(t, w)
            self.h_split_p2p(x)
            if self.is_downsample:
                if self.rank != self.world_size-1:
                    x[:, :, :, -self.recv_buffer_tail.shape[3]:, :] = self.recv_buffer_tail
            else:
                if self.rank == 0:
                    x[:, :, :, -self.recv_buffer_tail.shape[3]:, :] = self.recv_buffer_tail
                elif self.rank == self.world_size-1:
                    x[:, :, :, :self.recv_buffer_head.shape[3], :] = self.recv_buffer_head
                else:
                    x[:, :, :, :self.recv_buffer_head.shape[3], :] = self.recv_buffer_head
                    x[:, :, :, -self.recv_buffer_tail.shape[3]:, :] = self.recv_buffer_tail

            x = self.conv(x)
            del self.recv_buffer_head
            del self.recv_buffer_tail

        return x

    def forward_t_split_p2p(self, x):
        b, c, t, h, w = x.shape
        if self.kernel_size == 1:
            x = self.conv(x)
        else:
            self.configure_split_t_p2p(h, w)
            self.t_split_p2p(x)
            if self.is_downsample:
                if self.rank != self.world_size-1:
                    x[:, :, :, -self.recv_buffer_tail.shape[3]:, :] = self.recv_buffer_tail
            else:
                if self.rank == 0:
                    x[:, :, :, -self.recv_buffer_tail.shape[3]:, :] = self.recv_buffer_tail
                elif self.rank == self.world_size-1:
                    x[:, :, :, :self.recv_buffer_head.shape[3], :] = self.recv_buffer_head
                else:
                    x[:, :, :, :self.recv_buffer_head.shape[3], :] = self.recv_buffer_head
                    x[:, :, :, -self.recv_buffer_tail.shape[3]:, :] = self.recv_buffer_tail

            x = self.conv(x)
            del self.recv_buffer_head
            del self.recv_buffer_tail

        return x

    def height_left(self, x, is_send, rank_info):
        w_head, w_tail = self.padding_info[0], self.padding_info[1]
        if is_send:
            data_send = x[:, :, :, :, w_head:self.kernel_size-2+w_head].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_height_right, rank_info[self.rank], group=None)
            # paste data
            x[:, :, :, :, -self.recv_buffer_height_right.shape[4]:] = self.recv_buffer_height_right

    def height_right(self, x, is_send, rank_info):
        w_head, w_tail = self.padding_info[0], self.padding_info[1]
        if is_send:
            data_send = x[:, :, :, :, -self.kernel_size+2-w_tail:-w_tail].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_height_left, rank_info[self.rank], group=None)
            # paste data
            if not self.is_downsample:
                x[:, :, :, :, :self.recv_buffer_height_left.shape[4]] = self.recv_buffer_height_left
    
    def width_bottom(self, x, is_send, rank_info):
        h_head, h_tail = self.padding_info[2], self.padding_info[3]
        if is_send:
            data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_width_upper, rank_info[self.rank], group=None)
            # paste data
            if not self.is_downsample:
                x[:, :, :, :self.recv_buffer_width_upper.shape[3], :] = self.recv_buffer_width_upper
    
    def width_upper(self, x, is_send, rank_info):
        h_head, h_tail = self.padding_info[2], self.padding_info[3]
        if is_send:
            data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_width_bottom, rank_info[self.rank], group=None)
            # paste data
            x[:, :, :, -self.recv_buffer_width_bottom.shape[3]:, :] = self.recv_buffer_width_bottom

    def corner_right_bottom(self, x, is_send, rank_info):
        h_head, h_tail, w_head, w_tail = self.padding_info[2], self.padding_info[3], self.padding_info[0], self.padding_info[1]
        if is_send:
            data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, -self.kernel_size+2-w_tail:-w_tail].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_corner_left_upper, rank_info[self.rank], group=None)
            # paste data
            x[:, :, :, :self.recv_buffer_corner_left_upper.shape[3], :self.recv_buffer_corner_left_upper.shape[4]] = self.recv_buffer_corner_left_upper
    
    def corner_left_upper(self, x, is_send, rank_info):
        h_head, h_tail, w_head, w_tail = self.padding_info[2], self.padding_info[3], self.padding_info[0], self.padding_info[1]
        if is_send:
            data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, w_head:self.kernel_size-2+w_head].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_corner_right_bottom, rank_info[self.rank], group=None)
            # paste data
            x[:, :, :, -self.recv_buffer_corner_right_bottom.shape[3]:, -self.recv_buffer_corner_right_bottom.shape[4]:] = self.recv_buffer_corner_right_bottom

    def corner_left_bottom(self, x, is_send, rank_info):
        h_head, h_tail, w_head, w_tail = self.padding_info[2], self.padding_info[3], self.padding_info[0], self.padding_info[1]
        if is_send:
            data_send = x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, w_head:self.kernel_size-2+w_head].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_corner_right_upper, rank_info[self.rank], group=None)
            # paste data
            x[:, :, :, :self.recv_buffer_corner_right_upper.shape[3], -self.recv_buffer_corner_right_upper.shape[4]:] = self.recv_buffer_corner_right_upper

    def corner_right_upper(self, x, is_send, rank_info):
        h_head, h_tail, w_head, w_tail = self.padding_info[2], self.padding_info[3], self.padding_info[0], self.padding_info[1]
        if is_send:
            data_send = x[:, :, :, h_head:self.kernel_size-2+h_head, -self.kernel_size+2-w_tail:-w_tail].contiguous()
            dist.send(data_send, rank_info[self.rank], group=None)
        else:
            dist.recv(self.recv_buffer_corner_left_bottom, rank_info[self.rank], group=None)
            # paste data
            x[:, :, :, -self.recv_buffer_corner_left_bottom.shape[3]:, :self.recv_buffer_corner_left_bottom.shape[4]] = self.recv_buffer_corner_left_bottom

    def forward_search_p2p(self, x):
        b, c, t, h, w = x.shape
        if self.kernel_size == 1:
            return self.conv(x)
        else:
            self.configure_search_p2p(t, h, w)

            for group_idx, (group_info, group) in enumerate(zip(self.groups_info, self.groups)):
                is_send = group_idx % 2 == 0
                if self.rank in group:
                    self.func_map[group_info](x, is_send=is_send, rank_info={self.rank:group[self.rank]})

            return self.conv(x)

    def dummy_tensor_like(self, dtype, device):
        # return torch.empty([0, 1], dtype=dtype, device=device)
        return torch.zeros([1], dtype=dtype, device=device)

    def before_alltoall_split_h(self, x):
        h_head, h_tail = self.padding_info[2], self.padding_info[3]
        if self.rank == 0:
            send_tensors = [self.dummy_tensor_like(x.dtype, x.device), x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()]
            input_tensors = send_tensors + [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.world_size-2)]
        elif self.rank == self.world_size - 1:
            send_tensors = [x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous(), self.dummy_tensor_like(x.dtype, x.device)]
            input_tensors = [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.world_size-2)] + send_tensors
        else:
            send_tensors = [x[:, :, :, h_head:self.kernel_size-2+h_head, :].contiguous(), self.dummy_tensor_like(x.dtype, x.device), x[:, :, :, -self.kernel_size+2-h_tail:-h_tail, :].contiguous()]
            input_tensors = [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.rank-1)] + send_tensors + [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.world_size-2 - self.rank)]
        return input_tensors
    
    def before_alltoall_split_w(self, x):
        w_head, w_tail = self.padding_info[0], self.padding_info[1]
        if self.rank == 0:
            send_tensors = [self.dummy_tensor_like(x.dtype, x.device), x[:, :, :, :, -self.kernel_size+2-w_tail:-w_tail].contiguous()]
            input_tensors = send_tensors + [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.world_size-2)]
        elif self.rank == self.world_size - 1:
            send_tensors = [x[:, :, :, :, w_head:self.kernel_size-2+w_head].contiguous(), self.dummy_tensor_like(x.dtype, x.device)]
            input_tensors = [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.world_size-2)] + send_tensors
        else:
            send_tensors = [x[:, :, :, :, w_head:self.kernel_size-2+w_head].contiguous(), self.dummy_tensor_like(x.dtype, x.device), x[:, :, :, :, -self.kernel_size+2-w_tail:-w_tail].contiguous()]
            input_tensors = [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.rank-1)] + send_tensors + [self.dummy_tensor_like(x.dtype, x.device) for _ in range(self.world_size-2 - self.rank)]
        return input_tensors

    def after_alltoall_split_h(self, x, output_tensors):
        # hard code: self.neighbour_size = 1
        if self.is_downsample:
            if self.rank != self.world_size-1:
                if self.rank == 0:
                    # x[:, :, :, -self.neighbour_size:, :] = output_tensors[1]
                    indices = torch.tensor([-1],dtype=torch.int32).npu()
                    torch_npu.scatter_update_(x, indices, output_tensors[1], axis=-2)
                else:
                    # x[:, :, :, :self.neighbour_size, :] = output_tensors[self.rank-1]
                    indices = torch.tensor([0],dtype=torch.int32).npu()
                    torch_npu.scatter_update_(x, indices, output_tensors[self.rank-1], axis=-2)
                    # x[:, :, :, -self.neighbour_size:, :] = output_tensors[self.rank+1]
                    indices = torch.tensor([-1],dtype=torch.int32).npu()
                    torch_npu.scatter_update_(x, indices, output_tensors[self.rank+1], axis=-2)
                x = x.contiguous()
                return x
            else:
                return x
        else:
            if self.rank == 0:
                # x[:, :, :, -self.neighbour_size:, :] = output_tensors[self.rank+1]
                indices = torch.tensor([-1],dtype=torch.int32).npu()
                torch_npu.scatter_update_(x, indices, output_tensors[self.rank+1], axis=-2)
            elif self.rank == self.world_size-1:
                # x[:, :, :, :self.neighbour_size, :] = output_tensors[self.rank-1]
                indices = torch.tensor([0],dtype=torch.int32).npu()
                torch_npu.scatter_update_(x, indices, output_tensors[self.rank-1], axis=-2)
            else:
                # x[:, :, :, :self.neighbour_size, :] = output_tensors[self.rank-1]
                indices = torch.tensor([0],dtype=torch.int32).npu()
                torch_npu.scatter_update_(x, indices, output_tensors[self.rank-1], axis=-2)
                # x[:, :, :, -self.neighbour_size:, :] = output_tensors[self.rank+1]
                indices = torch.tensor([-1],dtype=torch.int32).npu()
                torch_npu.scatter_update_(x, indices, output_tensors[self.rank+1], axis=-2)
            x = x.contiguous()
            return x
    
    def after_alltoall_split_w(self, x, output_tensors):
        if self.is_downsample:
            if self.rank != self.world_size-1:
                if self.rank == 0:
                    x[:, :, :, :, -self.neighbour_size:] = output_tensors[1]
                else:
                    x[:, :, :, :, :self.neighbour_size] = output_tensors[self.rank-1]
                    x[:, :, :, :, -self.neighbour_size:] = output_tensors[self.rank+1]
                x = x.contiguous()
                return x
            else:
                return x
        else:
            if self.rank == 0:
                x[:, :, :, :, -self.neighbour_size:] = output_tensors[self.rank+1]
            elif self.rank == self.world_size-1:
                x[:, :, :, :, :self.neighbour_size] = output_tensors[self.rank-1]
            else:
                x[:, :, :, :, :self.neighbour_size] = output_tensors[self.rank-1]
                x[:, :, :, :, -self.neighbour_size:] = output_tensors[self.rank+1]
            x = x.contiguous()
            return x

    def forward_alltoall(self, x, split_type):  # only support split h
        if self.kernel_size == 1:
            return self.conv(x)
        else:
            if split_type == "split_h":
                input_tensors = self.before_alltoall_split_h(x)
            elif split_type == "split_w":
                input_tensors = self.before_alltoall_split_w(x)
            else:
                raise NotImplementedError
            
            output_tensors = [torch.empty(each.shape, dtype=x.dtype, device=x.device) for each in input_tensors]
            dist.all_to_all(output_tensors, input_tensors)

            if split_type == "split_h":
                x = self.after_alltoall_split_h(x, output_tensors)
            elif split_type == "split_w":
                x = self.after_alltoall_split_w(x, output_tensors)
            else:
                raise NotImplementedError
            
            return self.conv(x)

    def forward(self, x):
        if self.split_stragety["method"] == "send_recv":
            if self.split_stragety["split_type"] == "split_h":
                return self.forward_h_split_p2p(x)
            elif self.split_stragety["split_type"] == "split_w":
                return self.forward_w_split_p2p(x)
            elif self.split_stragety["split_type"] == "split_t":
                return self.forward_t_split_p2p(x)
            else:
                raise NotImplementedError
        elif self.split_stragety["method"] == "alltoall":
            return self.forward_alltoall(x, self.split_stragety["split_type"])
        elif self.split_stragety["method"] == "search":
            return self.forward_search_p2p(x)
        else:
            print("split_stragety_method:", self.split_stragety["method"])
            raise NotImplementedError

class DistributedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_config, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        self.kernel_size = kernel_size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.conv_config = conv_config
        self.only_sp = conv_config.get("only_sp", None)
        self.only_temp = conv_config.get("only_temp", None)

        if conv_config["conv_type"] == "split_in_out_channel":
            self.split_nums_in = conv_config["split_nums_in"]
            self.split_nums_out = conv_config["split_nums_out"]
            self.conv = SplitInOutchannelConv2d(
                self.split_nums_in,
                self.split_nums_out,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            self.conv = torch.nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class SplitInOutchannelConv2d(torch.nn.Conv2d):
    def __init__(self, split_num_in, split_num_out, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if isinstance(kernel_size, tuple):
            assert len(set(list(kernel_size))) == 1
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            if len(set(list(stride))) == 1:
                stride = stride[0]
        self.split_num_in = split_num_in
        self.split_num_out = split_num_out
    
    def forward(self, x):
        assert len(x.shape) == 4
        in_channel = x.shape[1]
        out_channel = self.weight.shape[0]
        assert in_channel % self.split_num_in == 0
        assert out_channel % self.split_num_out == 0
        chunk_size_in = in_channel // self.split_num_in
        chunk_size_out = out_channel // self.split_num_out
        output = None

        output_list = []

        for chunk_idx_out in range(self.split_num_out):
            output = None
            for chunk_idx_in in range(self.split_num_in):
                if output is None:
                    output = torch.nn.functional.conv2d(
                        x[:, chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, :, :], 
                        self.weight[
                            chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out, 
                            chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, 
                            :, :
                        ], 
                        None if self.bias is None else self.bias[chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out], 
                        stride=self.stride,
                        padding=self.padding
                    )
                else:
                    output += torch.nn.functional.conv2d(
                        x[:, chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, :, :], 
                        self.weight[
                            chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out, 
                            chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, 
                            :, :
                        ], 
                        None if self.bias is None else self.bias[chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out], 
                        stride=self.stride,
                        padding=self.padding
                    )
            output_list.append(output)
        return torch.cat(output_list, dim=1)

class SplitInOutchannelConv3d(torch.nn.Conv3d):
    def __init__(self, split_num_in, split_num_out, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if isinstance(kernel_size, tuple):
            assert len(set(list(kernel_size))) == 1
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            if len(set(list(stride))) == 1:
                stride = stride[0]
        self.split_num_in = split_num_in
        self.split_num_out = split_num_out
    
    def forward(self, x):
        assert len(x.shape) == 5
        in_channel = x.shape[1]
        out_channel = self.weight.shape[0]
        assert in_channel % self.split_num_in == 0
        assert out_channel % self.split_num_out == 0
        chunk_size_in = in_channel // self.split_num_in
        chunk_size_out = out_channel // self.split_num_out
        output = None

        output_list = []

        for chunk_idx_out in range(self.split_num_out):
            output = None
            for chunk_idx_in in range(self.split_num_in):
                if output is None:
                    output = torch.nn.functional.conv3d(
                        x[:, chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, :, :, :], 
                        self.weight[
                            chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out, 
                            chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, 
                            :, :, :
                        ], 
                        self.bias[chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out], 
                        stride=self.stride,
                        padding=self.padding
                    )
                else:
                    output += torch.nn.functional.conv3d(
                        x[:, chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, :, :, :], 
                        self.weight[
                            chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out, 
                            chunk_idx_in*chunk_size_in:(chunk_idx_in+1)*chunk_size_in, 
                            :, :, :
                        ], 
                        self.bias[chunk_idx_out*chunk_size_out:(chunk_idx_out+1)*chunk_size_out], 
                        stride=self.stride,
                        padding=self.padding
                    )
            output_list.append(output)
        return torch.cat(output_list, dim=1)
