"""Opt-in, bounded PyTorch aten::add.Tensor compatibility for Maxwell."""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Mapping
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cuda_legacy_compat import unrouted

_library: torch.library.Library | None = None
_bridge: ctypes.CDLL | None = None
_fallback_mode: TorchDispatchMode | None = None
_dispatch_state = threading.local()
_CPU_FALLBACK_MAX_BYTES = 64 * 1024 * 1024


def _load_bridge() -> tuple[ctypes.CDLL, bytes]:
    bridge_path = os.environ.get("CUDA_COMPAT_BRIDGE")
    ptx_path = os.environ.get("CUDA_COMPAT_VECTOR_ADD_PTX")
    if not bridge_path or not ptx_path:
        raise RuntimeError("CUDA_COMPAT_BRIDGE and CUDA_COMPAT_VECTOR_ADD_PTX must be set")
    bridge = ctypes.CDLL(bridge_path)
    bridge.cuda_compat_launch_vector_add.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_vector_add.restype = ctypes.c_int
    bridge.cuda_compat_launch_vector_add_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_vector_add_f32.restype = ctypes.c_int
    bridge.cuda_compat_launch_vector_add_f16.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_vector_add_f16.restype = ctypes.c_int
    bridge.cuda_compat_launch_vector_add_scaled_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_vector_add_scaled_f32.restype = ctypes.c_int
    bridge.cuda_compat_launch_arange_i64.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_arange_i64.restype = ctypes.c_int
    bridge.cuda_compat_copy_device_to_host.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_copy_device_to_host.restype = ctypes.c_int
    bridge.cuda_compat_launch_embedding_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_embedding_f32.restype = ctypes.c_int
    bridge.cuda_compat_launch_layer_norm_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_layer_norm_f32.restype = ctypes.c_int
    bridge.cuda_compat_launch_matrix_multiply_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_matrix_multiply_f32.restype = ctypes.c_int
    bridge.cuda_compat_launch_relu_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    bridge.cuda_compat_launch_relu_f32.restype = ctypes.c_int
    bridge.cuda_compat_cached_context_count.argtypes = []
    bridge.cuda_compat_cached_context_count.restype = ctypes.c_size_t
    return bridge, os.fsencode(ptx_path)


def _validate_pair(left: torch.Tensor, right: torch.Tensor, operation: str) -> None:
    if not left.is_cuda or not right.is_cuda or left.device != right.device:
        raise NotImplementedError(f"Maxwell {operation} requires tensors on the same CUDA device")
    if left.requires_grad or right.requires_grad:
        raise NotImplementedError(f"autograd is not implemented for Maxwell {operation}")
    if left.layout != torch.strided or right.layout != torch.strided:
        raise NotImplementedError(f"Maxwell {operation} supports only dense strided tensors")


def _record_stream(tensors: tuple[torch.Tensor, ...], stream: Any) -> None:
    for tensor in tensors:
        tensor.record_stream(stream)


def _add(self: torch.Tensor, other: torch.Tensor, *, alpha: int | float = 1) -> torch.Tensor:
    global _bridge
    if self.requires_grad or other.requires_grad:
        return _cpu_fallback(torch.ops.aten.add.Tensor, "add")(self, other, alpha=alpha)
    _validate_pair(self, other, "addition")

    if (
        self.dtype == torch.float16
        and other.dtype == torch.float16
        and self.shape == other.shape
        and self.is_contiguous()
        and other.is_contiguous()
        and alpha == 1
    ):
        element_count = self.numel()
        if element_count > 1048576:
            raise NotImplementedError("float16 Maxwell addition is limited to 1,048,576 elements")
        output = torch.empty_like(self, memory_format=torch.contiguous_format)
        if element_count == 0:
            _dispatch_state.last_path = "llvm-sm52-f16-empty"
            return output
        if _bridge is None:
            _bridge, ptx_path = _load_bridge()
        else:
            ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
        current_stream = torch.cuda.current_stream(self.device)
        _record_stream((self, other), current_stream)
        result = _bridge.cuda_compat_launch_vector_add_f16(
            ptx_path,
            self.data_ptr(),
            other.data_ptr(),
            output.data_ptr(),
            element_count,
            current_stream.cuda_stream,
        )
        if result != 0:
            raise RuntimeError(f"CUDA Driver API Maxwell float16 add failed with CUresult={result}")
        _dispatch_state.last_path = "llvm-sm52-f16-add"
        return output

    if (
        self.dtype == torch.float32
        and other.dtype == torch.float32
        and self.shape == other.shape
        and self.is_contiguous()
        and other.is_contiguous()
        and alpha == 1
    ):
        element_count = self.numel()
        if element_count > 1048576:
            raise NotImplementedError("float32 Maxwell addition is limited to 1,048,576 elements")
        output = torch.empty_like(self, memory_format=torch.contiguous_format)
        if element_count == 0:
            _dispatch_state.last_path = "llvm-sm52-f32-empty"
            return output
        if _bridge is None:
            _bridge, ptx_path = _load_bridge()
        else:
            ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
        current_stream = torch.cuda.current_stream(self.device)
        _record_stream((self, other), current_stream)
        stream = current_stream.cuda_stream
        result = _bridge.cuda_compat_launch_vector_add_f32(
            ptx_path,
            self.data_ptr(),
            other.data_ptr(),
            output.data_ptr(),
            element_count,
            stream,
        )
        if result != 0:
            raise RuntimeError(f"CUDA Driver API Maxwell float32 add failed with CUresult={result}")
        _dispatch_state.last_path = "llvm-sm52-f32-add"
        return output

    if (
        self.dtype == torch.int32
        and other.dtype == torch.int32
        and self.shape == other.shape
        and self.is_contiguous()
        and other.is_contiguous()
        and alpha == 1
    ):
        element_count = self.numel()
        if element_count == 0:
            _dispatch_state.last_path = "llvm-sm52-int32-empty"
            return torch.empty_like(self, memory_format=torch.contiguous_format)
        if element_count > 2147483647:
            raise NotImplementedError("int32 Maxwell kernel is limited to 2,147,483,647 elements")
        if _bridge is None:
            _bridge, ptx_path = _load_bridge()
        else:
            ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
        output = torch.empty_like(self, memory_format=torch.contiguous_format)
        current_stream = torch.cuda.current_stream(self.device)
        _record_stream((self, other), current_stream)
        stream = current_stream.cuda_stream
        result = _bridge.cuda_compat_launch_vector_add(
            ptx_path,
            self.data_ptr(),
            other.data_ptr(),
            output.data_ptr(),
            element_count,
            stream,
        )
        if result != 0:
            raise RuntimeError(f"CUDA Driver API Maxwell kernel failed with CUresult={result}")
        _dispatch_state.last_path = "llvm-sm52-int32"
        return output

    if (
        self.dtype == torch.int64
        and other.dtype == torch.int64
        and self.shape == other.shape
        and self.is_contiguous()
        and other.is_contiguous()
        and alpha == 1
    ):
        _dispatch_state.last_path = "pytorch-cpu-int64"
        result = torch.add(self.cpu(), other.cpu(), alpha=alpha)
        return result.to(self.device)

    return _cpu_fallback(torch.ops.aten.add.Tensor, "add")(
        self, other, alpha=alpha
    )


def _add_inplace(
    self: torch.Tensor, other: torch.Tensor, *, alpha: int | float = 1
) -> torch.Tensor:
    if not isinstance(other, torch.Tensor):
        raise NotImplementedError("Maxwell in-place addition does not support scalar operands")
    if torch.is_grad_enabled() and (self.requires_grad or other.requires_grad):
        raise NotImplementedError(
            "Maxwell in-place addition does not implement autograd; optimizer updates require no_grad"
        )
    if (
        not self.is_cuda
        or not other.is_cuda
        or self.device != other.device
        or self.dtype != torch.float32
        or other.dtype != torch.float32
        or self.shape != other.shape
        or not self.is_contiguous()
        or not other.is_contiguous()
    ):
        raise NotImplementedError(
            "Maxwell in-place addition supports same-shape contiguous float32 CUDA tensors"
        )
    if self.numel() > 1048576:
        raise NotImplementedError("Maxwell in-place addition is limited to 1,048,576 elements")
    left_start = self.data_ptr()
    right_start = other.data_ptr()
    byte_count = self.numel() * self.element_size()
    if (
        left_start != right_start
        and left_start < right_start + byte_count
        and right_start < left_start + byte_count
    ):
        raise NotImplementedError("Maxwell in-place addition rejects partially overlapping tensors")
    if self.numel() == 0:
        _dispatch_state.last_path = "llvm-sm52-f32-add-inplace-empty"
        return self
    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    current_stream = torch.cuda.current_stream(self.device)
    _record_stream((self, other), current_stream)
    result = _bridge.cuda_compat_launch_vector_add_scaled_f32(
        ptx_path,
        self.data_ptr(),
        other.data_ptr(),
        self.data_ptr(),
        self.numel(),
        float(alpha),
        current_stream.cuda_stream,
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell in-place float32 add failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-f32-add-inplace"
    return self


def _mm(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if self.requires_grad or other.requires_grad:
        return _cpu_fallback(torch.ops.aten.mm.default, "mm")(self, other)
    _validate_pair(self, other, "matrix multiplication")
    if self.dtype != torch.float32 or other.dtype != torch.float32:
        raise NotImplementedError("Maxwell matrix multiplication supports float32 tensors only")
    if self.dim() != 2 or other.dim() != 2:
        raise NotImplementedError("Maxwell matrix multiplication requires two-dimensional tensors")
    if self.shape[1] != other.shape[0]:
        raise RuntimeError("matrix multiplication shapes do not align")
    left_strides = self.stride()
    right_strides = other.stride()
    if any(stride < 0 for stride in (*left_strides, *right_strides)):
        raise NotImplementedError("Maxwell matrix multiplication does not support negative strides")

    rows, inner = self.shape
    columns = other.shape[1]
    if (
        rows > 256
        or columns > 256
        or inner > 256
        or rows * columns * inner > 16777216
    ):
        raise NotImplementedError("Maxwell float32 matrix multiplication is limited to 256 dimensions")

    output = torch.empty((rows, columns), dtype=torch.float32, device=self.device)
    if rows == 0 or columns == 0:
        _dispatch_state.last_path = "llvm-sm52-mm-f32-empty"
        return output

    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    current_stream = torch.cuda.current_stream(self.device)
    _record_stream((self, other), current_stream)
    stream = current_stream.cuda_stream
    result = _bridge.cuda_compat_launch_matrix_multiply_f32(
        ptx_path,
        self.data_ptr(),
        other.data_ptr(),
        output.data_ptr(),
        rows,
        columns,
        inner,
        left_strides[0],
        left_strides[1],
        right_strides[0],
        right_strides[1],
        stream,
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell matrix multiplication failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-mm-f32"
    return output


def _arange(
    end: int,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    if dtype not in (None, torch.int64):
        raise NotImplementedError("Maxwell arange supports int64 values only")
    if layout not in (None, torch.strided) or pin_memory is True:
        raise NotImplementedError("Maxwell arange supports ordinary strided device tensors only")
    if device is None or torch.device(device).type != "cuda":
        raise NotImplementedError("Maxwell arange requires an explicit CUDA device")
    count = int(end)
    if count < 0:
        count = 0
    if count > 1048576:
        raise NotImplementedError("Maxwell arange is limited to 1,048,576 values")
    output = torch.empty((count,), dtype=torch.int64, device=device)
    if count == 0:
        return output

    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    stream = torch.cuda.current_stream(device).cuda_stream
    result = _bridge.cuda_compat_launch_arange_i64(
        ptx_path, output.data_ptr(), count, stream
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell arange failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-arange-i64"
    return output


def _embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    if weight.requires_grad:
        return _cpu_fallback(torch.ops.aten.embedding.default, "embedding")(
            weight, indices, padding_idx, scale_grad_by_freq, sparse
        )
    if (
        not weight.is_cuda
        or not indices.is_cuda
        or weight.device != indices.device
        or weight.dtype != torch.float32
        or indices.dtype != torch.int64
        or indices.dim() not in (1, 2)
        or weight.dim() != 2
        or not weight.is_contiguous()
    ):
        raise NotImplementedError(
            "Maxwell embedding requires contiguous float32 weights and int64 CUDA indices"
        )
    if scale_grad_by_freq or sparse:
        raise NotImplementedError("Maxwell embedding does not implement gradient modes")
    rows, embedding_size = weight.shape
    if padding_idx < -rows or padding_idx >= rows:
        raise IndexError("padding_idx is out of range")
    token_count = indices.numel()
    if token_count * embedding_size > 1048576:
        raise NotImplementedError("Maxwell embedding output is limited to 1,048,576 values")
    output = torch.empty(
        (*indices.shape, embedding_size), dtype=weight.dtype, device=weight.device
    )
    if token_count == 0:
        return output

    index_strides = tuple(indices.stride())
    if any(stride < 0 for stride in index_strides):
        raise NotImplementedError("Maxwell embedding does not support negative index strides")
    maximum_offset = sum(
        (size - 1) * stride
        for size, stride in zip(indices.shape, index_strides, strict=True)
    )
    available_storage = indices.untyped_storage().nbytes() // indices.element_size()
    if maximum_offset >= available_storage - indices.storage_offset():
        raise RuntimeError("embedding index view exceeds its underlying storage")

    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    span_count = maximum_offset + 1
    host_indices = ctypes.create_string_buffer(span_count * indices.element_size())
    current_stream = torch.cuda.current_stream(weight.device)
    current_stream.synchronize()
    copy_result = _bridge.cuda_compat_copy_device_to_host(
        indices.data_ptr(), ctypes.addressof(host_indices), len(host_indices)
    )
    if copy_result != 0:
        raise RuntimeError(f"CUDA Driver API embedding index copy failed with CUresult={copy_result}")
    index_storage = torch.frombuffer(host_indices, dtype=torch.int64, count=span_count)
    logical_indices = index_storage.as_strided(indices.shape, index_strides)
    if int(logical_indices.min()) < 0 or int(logical_indices.max()) >= rows:
        raise IndexError("embedding index is out of range")

    stream = current_stream.cuda_stream
    _record_stream((weight, indices), current_stream)
    result = _bridge.cuda_compat_launch_embedding_f32(
        ptx_path,
        weight.data_ptr(),
        indices.data_ptr(),
        output.data_ptr(),
        token_count,
        embedding_size,
        indices.shape[1] if indices.dim() == 2 else 1,
        index_strides[0],
        index_strides[1] if indices.dim() == 2 else 0,
        indices.dim(),
        stream,
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell embedding failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-embedding-f32"
    return output


def _layer_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    cudnn_enable: bool = True,
) -> torch.Tensor:
    del cudnn_enable
    if input.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad):
        return _cpu_fallback(torch.ops.aten.layer_norm.default, "layer_norm")(
            input, normalized_shape, weight, bias, eps, False
        )
    if (
        not input.is_cuda
        or input.dtype != torch.float32
        or input.layout != torch.strided
        or not input.is_contiguous()
        or input.dim() == 0
        or tuple(normalized_shape) != (input.shape[-1],)
    ):
        raise NotImplementedError(
            "Maxwell layer normalization supports contiguous float32 input normalized on its last dimension"
        )
    if (
        weight is None
        or bias is None
        or weight.device != input.device
        or bias.device != input.device
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.shape != (input.shape[-1],)
        or bias.shape != (input.shape[-1],)
        or not weight.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise NotImplementedError("Maxwell layer normalization requires contiguous float32 weight and bias")
    normalized_size = input.shape[-1]
    row_count = input.numel() // normalized_size
    if row_count * normalized_size > 1048576:
        raise NotImplementedError("Maxwell layer normalization is limited to 1,048,576 values")
    output = torch.empty_like(input)
    if row_count == 0:
        return output

    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    current_stream = torch.cuda.current_stream(input.device)
    _record_stream((input, weight, bias), current_stream)
    stream = current_stream.cuda_stream
    result = _bridge.cuda_compat_launch_layer_norm_f32(
        ptx_path,
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        row_count,
        normalized_size,
        eps,
        stream,
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell layer normalization failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-layernorm-f32"
    return output


def _relu(self: torch.Tensor) -> torch.Tensor:
    if not self.is_cuda:
        raise NotImplementedError("Maxwell ReLU requires a CUDA tensor")
    if self.requires_grad:
        return _cpu_fallback(torch.ops.aten.relu.default, "relu")(self)
    if self.layout != torch.strided or not self.is_contiguous():
        raise NotImplementedError("Maxwell ReLU requires a contiguous dense tensor")
    if self.dtype != torch.float32:
        raise NotImplementedError("Maxwell ReLU supports float32 tensors only")
    if self.numel() > 1048576:
        raise NotImplementedError("Maxwell float32 ReLU is limited to 1,048,576 elements")
    output = torch.empty_like(self)
    if self.numel() == 0:
        _dispatch_state.last_path = "llvm-sm52-relu-f32-empty"
        return output

    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    current_stream = torch.cuda.current_stream(self.device)
    _record_stream((self,), current_stream)
    stream = current_stream.cuda_stream
    result = _bridge.cuda_compat_launch_relu_f32(
        ptx_path, self.data_ptr(), output.data_ptr(), self.numel(), stream
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell ReLU failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-relu-f32"
    return output


def _cuda_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and tensor.requires_grad:
        _dispatch_state.last_path = "differentiable-cuda-to-cpu-copy"
        return tensor.to("cpu")
    if tensor.numel() == 0 or tensor.is_contiguous():
        return tensor.to("cpu")
    strides = tuple(tensor.stride())
    if any(stride < 0 for stride in strides):
        raise NotImplementedError("Maxwell CPU fallback does not support negative strides")
    maximum_offset = sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, strides, strict=True)
    )
    span_count = maximum_offset + 1
    span_bytes = span_count * tensor.element_size()
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    if maximum_offset >= storage_elements - tensor.storage_offset():
        raise RuntimeError("CUDA tensor view exceeds its underlying storage")
    if span_bytes > _CPU_FALLBACK_MAX_BYTES:
        raise NotImplementedError("Maxwell CPU fallback is limited to 64 MiB per tensor view")

    global _bridge
    if _bridge is None:
        _bridge, _ = _load_bridge()
    host_storage = ctypes.create_string_buffer(span_bytes)
    torch.cuda.current_stream(tensor.device).synchronize()
    result = _bridge.cuda_compat_copy_device_to_host(
        tensor.data_ptr(), ctypes.addressof(host_storage), span_bytes
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API tensor copy failed with CUresult={result}")
    cpu_storage = torch.frombuffer(host_storage, dtype=tensor.dtype, count=span_count)
    return cpu_storage.as_strided(tensor.shape, strides).clone()


def _contiguous(
    input: torch.Tensor,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    if not input.is_cuda or torch.is_grad_enabled():
        raise NotImplementedError("Maxwell contiguous fallback requires CUDA inference tensors")
    if memory_format != torch.contiguous_format:
        raise NotImplementedError("Maxwell contiguous fallback supports contiguous_format only")
    if input.is_contiguous(memory_format=memory_format):
        _dispatch_state.last_path = "pytorch-contiguous-alias"
        return input
    result = torch.ops.aten.contiguous.default(
        _cuda_tensor_to_cpu(input), memory_format=memory_format
    ).to(input.device)
    _dispatch_state.last_path = "cpu-fallback-contiguous"
    return result


def _tensor_inputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_inputs(item)]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _tensor_inputs(item)]
    return []


def _cpu_fallback(operation: Callable[..., Any], name: str) -> Callable[..., Any]:
    def run(*args: Any, **kwargs: Any) -> Any:
        tensors = _tensor_inputs((args, kwargs))
        cuda_tensors = [tensor for tensor in tensors if tensor.is_cuda]
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1 or not cuda_tensors:
            raise NotImplementedError(
                f"Maxwell CPU fallback for {name} requires tensors from one CUDA device"
            )
        transfer_bytes = sum(_cuda_tensor_transfer_bytes(tensor) for tensor in cuda_tensors)
        if transfer_bytes > _CPU_FALLBACK_MAX_BYTES:
            raise NotImplementedError(
                f"Maxwell CPU fallback for {name} is limited to 64 MiB of CUDA inputs"
            )

        def move_to_cpu(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return _cuda_tensor_to_cpu(value) if value.is_cuda else value
            if isinstance(value, tuple):
                return tuple(move_to_cpu(item) for item in value)
            if isinstance(value, list):
                return [move_to_cpu(item) for item in value]
            if isinstance(value, Mapping):
                return {key: move_to_cpu(item) for key, item in value.items()}
            return value

        cpu_args = move_to_cpu(args)
        cpu_kwargs = move_to_cpu(kwargs)
        device = next(iter(devices))
        if name == "aten::linear_backward":
            input_tensor, grad_output, weight, output_mask = cpu_args
            input_matrix = input_tensor.reshape(-1, input_tensor.shape[-1])
            grad_matrix = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = grad_matrix.matmul(weight).reshape_as(input_tensor) if output_mask[0] else None
            grad_weight = grad_matrix.transpose(0, 1).matmul(input_matrix) if output_mask[1] else None
            grad_bias = grad_matrix.sum(dim=0) if output_mask[2] else None
            cpu_result = (grad_input, grad_weight, grad_bias)
        else:
            cpu_result = operation(*cpu_args, **cpu_kwargs)

        def result_bytes(value: Any) -> int:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, (tuple, list)):
                return sum(result_bytes(item) for item in value)
            if isinstance(value, Mapping):
                return sum(result_bytes(item) for item in value.values())
            return 0

        if result_bytes(cpu_result) > _CPU_FALLBACK_MAX_BYTES:
            raise NotImplementedError(
                f"Maxwell CPU fallback for {name} is limited to 64 MiB of outputs"
            )

        def move_to_device(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device) if value.device.type == "cpu" else value
            if isinstance(value, tuple):
                return tuple(move_to_device(item) for item in value)
            if isinstance(value, list):
                return [move_to_device(item) for item in value]
            if isinstance(value, Mapping):
                return {key: move_to_device(item) for key, item in value.items()}
            return value

        result = move_to_device(cpu_result)
        _dispatch_state.last_path = f"cpu-fallback-{name}"
        return result

    return run


def _cuda_tensor_transfer_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout != torch.strided:
        raise NotImplementedError("Maxwell CPU fallback supports only dense strided tensors")
    if tensor.numel() == 0:
        return 0
    strides = tuple(tensor.stride())
    if any(stride < 0 for stride in strides):
        raise NotImplementedError("Maxwell CPU fallback does not support negative strides")
    span_count = (
        tensor.numel()
        if tensor.is_contiguous()
        else sum((size - 1) * stride for size, stride in zip(tensor.shape, strides, strict=True)) + 1
    )
    return span_count * tensor.element_size()


def _cuda_tensors(value: Any) -> list[torch.Tensor]:
    return [tensor for tensor in _tensor_inputs(value) if tensor.is_cuda]


def _is_architecture_failure(error: Exception) -> bool:
    message = str(error).lower()
    if "could not run 'aten::" in message and "arguments from the 'cuda' backend" in message:
        return True
    return any(
        marker in message
        for marker in (
            "no kernel image is available",
            "no binary for gpu",
            "cublas_status_arch_mismatch",
            "cudnn_status_arch_mismatch",
            "unable to find an engine to execute this computation",
        )
    )


class _MaxwellInferenceFallback(TorchDispatchMode):
    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = getattr(operation, "_schema", None)
        name = schema.name if schema is not None else str(operation)
        if not _cuda_tensors((args, kwargs)):
            # No CUDA tensor to copy to the CPU (a factory such as torch.ones(device="cuda")): no route.
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                if _is_architecture_failure(exc):
                    unrouted.record(name, exc)
                raise
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            if not _is_architecture_failure(exc):
                raise
            if schema is not None and (
                schema.is_mutable
                or any(result.alias_info is not None for result in schema.returns)
            ):
                error = NotImplementedError(
                    f"CUDA operator {operation} has no Maxwell image and its aliasing semantics cannot use CPU fallback"
                )
                unrouted.record(name, error)
                raise error from exc
            return _cpu_fallback(operation, name)(*args, **kwargs)


def install() -> None:
    """Install the opt-in operator adapter only on a homogeneous sm_52 process."""
    global _fallback_mode, _library
    if _library is not None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capabilities = [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]
    if not capabilities or any(capability != (5, 2) for capability in capabilities):
        raise RuntimeError(f"adapter requires only sm_52 devices; observed {capabilities}")
    _library = torch.library.Library("aten", "IMPL", "CUDA")
    _library.impl("add.Tensor", _add)
    _library.impl("add_.Tensor", _add_inplace)
    _library.impl("arange", _arange)
    _library.impl("embedding", _embedding)
    _library.impl("layer_norm", _layer_norm)
    _library.impl("contiguous", _contiguous)
    _library.impl("linear", _cpu_fallback(torch.ops.aten.linear.default, "linear"))
    _library.impl("mm", _mm)
    _library.impl("bmm", _cpu_fallback(torch.ops.aten.bmm.default, "bmm"))
    _library.impl("_softmax", _cpu_fallback(torch.ops.aten._softmax.default, "softmax"))
    _library.impl(
        "scaled_dot_product_attention",
        _cpu_fallback(
            torch.ops.aten.scaled_dot_product_attention.default,
            "scaled_dot_product_attention",
        ),
    )
    _library.impl("gelu", _cpu_fallback(torch.ops.aten.gelu.default, "gelu"))
    _library.impl("mul.Tensor", _cpu_fallback(torch.ops.aten.mul.Tensor, "mul"))
    _library.impl("div.Tensor", _cpu_fallback(torch.ops.aten.div.Tensor, "div"))
    _library.impl("sum.dim_IntList", _cpu_fallback(torch.ops.aten.sum.dim_IntList, "sum"))
    _library.impl("clamp_min", _cpu_fallback(torch.ops.aten.clamp_min.default, "clamp_min"))
    _library.impl("cat", _cpu_fallback(torch.ops.aten.cat.default, "cat"))
    _library.impl("relu", _relu)
    torch.backends.mha.set_fastpath_enabled(False)
    _fallback_mode = _MaxwellInferenceFallback()
    _fallback_mode.__enter__()


def last_path() -> str:
    return getattr(_dispatch_state, "last_path", "not-run")


def cached_context_count() -> int:
    """Return the number of CUDA contexts with the adapter PTX module loaded."""
    if _bridge is None:
        return 0
    return int(_bridge.cuda_compat_cached_context_count())
