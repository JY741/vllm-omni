# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Async CPU offloading for activation tensors during training.

Ported from MGM-Video-Ascend mimogpt/models/dit/async_offload.py.
Adapted torch.cuda.Stream/Event to be NPU-compatible.
"""

import torch
from torch.autograd.graph import saved_tensors_hooks

# NPU compatibility: use torch.npu streams/events when available,
# fall back to torch.cuda otherwise.
try:
    import torch_npu  # noqa: F401
    _USE_NPU = True
except ImportError:
    _USE_NPU = False


def _get_stream():
    if _USE_NPU:
        return torch.npu.Stream()
    return torch.cuda.Stream()


def _get_current_stream():
    if _USE_NPU:
        return torch.npu.current_stream()
    return torch.cuda.current_stream()


def _get_default_stream():
    if _USE_NPU:
        return torch.npu.default_stream(torch.npu.current_device())
    return torch.cuda.default_stream(torch.cuda.current_device())


def _get_event():
    if _USE_NPU:
        return torch.npu.Event()
    return torch.cuda.Event()


class GetCnt:
    def __init__(self, ):
        self._block_idx = -1
        self._part_idx = []

    def get_cnt(self, i):
        finish_fa = False
        if i != self._block_idx:
            self._part_idx.clear()
            self._block_idx = i

        part_idx = len(self._part_idx)
        if part_idx > 2:
            finish_fa = True
        self._part_idx.append(part_idx)
        return "{}_{}".format(i, part_idx), finish_fa


Ggetgnt = GetCnt()


class SwapTensor:
    def __init__(self, tensor, key):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.untyped_storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

        self.is_slice_tensor = tensor.untyped_storage().size() != tensor.numel()
        self.stat = "device"
        self.key = key

        self.h2d_event = _get_event()

    # device to host
    def launch_d2h(self, stream):
        if self.stat != "device":
            return
        forward_event = _get_event()
        forward_event.record()
        with torch.no_grad():
            with _get_stream().__class__(stream):  # use the stream
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.untyped_storage().copy_(self.tensor.untyped_storage(), non_blocking=True)
                self.stat = "host"

    # synchronize d2h and resize 0
    def wait_d2h_finished(self, stream):
        if self.stat != "host":
            return

        _get_current_stream().wait_stream(stream)
        _get_default_stream().wait_stream(stream)
        self.tensor.untyped_storage().resize_(0)
        self.stat = "host"

    # resize storage_size and host to device
    def launch_h2d(self, h2d_stream):
        if self.stat != "host":
            return
        backward_event = _get_event()
        backward_event.record()

        self.tensor.untyped_storage().resize_(self.storage_size)

        with torch.no_grad():
            with _get_stream().__class__(h2d_stream):  # use the h2d_stream
                h2d_stream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.untyped_storage().copy_(self.tensor_cpu.untyped_storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"

    # synchronize h2d
    def wait_h2d_finished(self):
        if self.stat != "device":
            return
        if self.h2d_event:
            _get_current_stream().wait_event(self.h2d_event)
            _get_default_stream().wait_event(self.h2d_event)
        self.stat = "device"


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OffloadItem:

    def __init__(self, act=None, ref_cnt=0, event=None):
        self.act = act
        self.ref_cnt = ref_cnt


class OffloadManager(metaclass=SingletonMeta):

    def __init__(self, ):
        self.items = {}

    def assert_exist(self, key):
        assert key in self.items

    def exist(self, key):
        return key in self.items

    def assert_not_exist(self, key):
        assert key not in self.items

    def put(self, key, act):
        if key in self.items:
            self.items[key].act = act
            self.items[key].ref_cnt += 1
        else:
            self.items[key] = OffloadItem(act, 1)

    def del_cuda_tensor(self, prefile_key, d2h_stream):
        for key in self.items.keys():
            if key.startswith(prefile_key):
                self.items[key].act.wait_d2h_finished(d2h_stream)

    def get(self, key, is_prefetch=False):
        self.assert_exist(key)
        item = self.items[key]
        act = item.act

        if not is_prefetch:
            item.ref_cnt -= 1
            if item.ref_cnt == 0:
                self.clear(key)
        return act

    def empty(self):
        return len(self.items) == 0

    def clear(self, key=None):
        if key in self.items:
            self.items.pop(key)


GOffloadManager = OffloadManager()


class async_save_on_cpu(saved_tensors_hooks):
    def __init__(self, h2d_stream, d2h_stream, num_layer, depth, prefetch=True) -> None:

        def pack_to_cpu(tensor):
            key, finish_fa = Ggetgnt.get_cnt(num_layer)

            if finish_fa and num_layer < depth - 1:
                GOffloadManager.del_cuda_tensor("{}_".format(num_layer), d2h_stream)
                return tensor

            swap_tensor = SwapTensor(tensor, key)
            if num_layer < depth - 1:
                working_stream = _get_current_stream()
                d2h_stream.wait_stream(working_stream)
                swap_tensor.launch_d2h(d2h_stream)
            GOffloadManager.put(key, swap_tensor)
            return swap_tensor

        def unpack_from_cpu(swap_tensor) -> torch.Tensor:
            if isinstance(swap_tensor, torch.Tensor):
                return swap_tensor

            key = swap_tensor.key
            swap_tensor = GOffloadManager.get(key)
            h2d_stream.wait_stream(d2h_stream)
            swap_tensor.launch_h2d(h2d_stream)

            if prefetch:
                prefetch_key = "{}_{}".format(num_layer - 1, swap_tensor.key.split("_")[-1])
                if GOffloadManager.exist(prefetch_key):
                    prefetch_swap_tensor = GOffloadManager.get(prefetch_key, True)
                    h2d_stream.wait_stream(h2d_stream)
                    prefetch_swap_tensor.launch_h2d(h2d_stream)
                    prefetch_swap_tensor.tensor.record_stream(h2d_stream)

            return swap_tensor.tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)