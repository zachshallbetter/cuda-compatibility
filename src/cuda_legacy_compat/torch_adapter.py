"""Opt-in, bounded PyTorch aten::add.Tensor compatibility for Maxwell."""

from __future__ import annotations

import ctypes
import functools
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
# The CPU fallback's bounds (#178): no single copy between a GPU and host memory moves more than
# _CPU_FALLBACK_MAX_BYTES (a larger contiguous tensor moves in pieces of that size, each a plain memory copy), and
# no operation stages more than _CPU_FALLBACK_TOTAL_BYTES, so a huge fallback still stops with one plain line.
_CPU_FALLBACK_MAX_BYTES = 64 * 1024 * 1024
_MIB = 1024 * 1024


def _total_limit() -> int:
    raw = os.environ.get("CUDA_COMPAT_FALLBACK_TOTAL_MIB", "")
    return int(raw) * _MIB if raw.isdigit() and int(raw) > 0 else 1024 * _MIB


_CPU_FALLBACK_TOTAL_BYTES = _total_limit()
_autograd_library: torch.library.Library | None = None


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
    bridge.cuda_compat_launch_matrix_multiply_f32.argtypes = (
        [ctypes.c_char_p] + [ctypes.c_uint64] * 18 + [ctypes.c_float, ctypes.c_float, ctypes.c_size_t])
    bridge.cuda_compat_launch_matrix_multiply_f32.restype = ctypes.c_int
    bridge.cuda_compat_launch_named.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int] + [ctypes.c_uint] * 7 + [
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    bridge.cuda_compat_launch_named.restype = ctypes.c_int
    bridge.cuda_compat_ensure_context.argtypes = [ctypes.c_int]
    bridge.cuda_compat_ensure_context.restype = ctypes.c_int
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


def _falls_back_to_cpu(operation_name: str, oversize: Callable[..., Any] | None = None):
    """A native kernel that refuses a case it has no Maxwell kernel for (another dtype, a strided input, a size
    past its limit) hands it to the CPU fallback instead, like every unregistered operator. `oversize`, when given,
    takes the cases whose CUDA inputs exceed the fallback limit (embedding: only the looked-up rows move, #175)."""
    def wrap(kernel: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(kernel)
        def run(*args: Any, **kwargs: Any) -> Any:
            try:
                return kernel(*args, **kwargs)
            except NotImplementedError:
                operation = _resolve(operation_name)
                if not _cuda_tensors((args, kwargs)):
                    result = _factory_on_cpu(operation, operation._schema, f"aten::{operation_name}", args, kwargs)
                    if result is None:
                        raise
                    return result
                if oversize is not None and _over_fallback_limit(args, kwargs, _CPU_FALLBACK_MAX_BYTES):
                    return oversize(*args, **kwargs)
                if _writes_only_self(operation._schema):
                    # An in-place operator (fill_, zero_) must update its own tensor: the out-of-place fallback
                    # would return a new one and leave the target as it was (uninitialised for ones/zeros).
                    return _inplace_on_cpu(operation, f"aten::{operation_name.split('.')[0]}", args, kwargs)
                return _cpu_fallback(operation, operation_name.split(".")[0])(*args, **kwargs)
        return run
    return wrap


def _resolve(operation_name: str):
    name, _, overload = operation_name.partition(".")
    return getattr(getattr(torch.ops.aten, name), overload or "default")


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


@_falls_back_to_cpu("add.Tensor")
def _add(self: torch.Tensor, other: torch.Tensor, *, alpha: int | float = 1) -> torch.Tensor:
    global _bridge
    # float32 with broadcasting, a scalar, any alpha or size, in training too (#184): the elementwise kernel.
    if self.dtype == torch.float32 and self.is_cuda and (
            _scalar(other) is not None or (isinstance(other, torch.Tensor) and other.dtype == torch.float32
                                           and other.is_cuda and other.device == self.device)):
        output = _binary(self, other, 0, float(alpha))
        _dispatch_state.last_path = "llvm-sm52-binary-f32"
        return output
    if not isinstance(other, torch.Tensor) or (other.dim() == 0 and not other.is_cuda):
        # x + 1, 1 + x, x + 0.5, x + torch.tensor(1.0): a scalar operand has no Maxwell kernel here.
        scalar = other.item() if isinstance(other, torch.Tensor) else other
        return _cpu_fallback(torch.ops.aten.add.Tensor, "add")(self, scalar, alpha=alpha)
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
        _ensure_context()
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
        _ensure_context()
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
        _ensure_context()
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
    """self += alpha * other on the elementwise kernel, any size, other broadcast or a number (#184 B: SGD's step and
    gradient accumulation stay on the GPU). A case it can't take is computed on the CPU and written back."""
    try:
        if self.numel():
            _binary(self, other, 0, float(alpha), out=self)
    except NotImplementedError:
        return _inplace_on_cpu(torch.ops.aten.add_.Tensor, "add_", (self, other), {"alpha": alpha})
    _dispatch_state.last_path = "llvm-sm52-f32-add-inplace"
    return self


def _foreach_add_list(self: list[torch.Tensor], other: list[torch.Tensor], *, alpha: int | float = 1) -> None:
    """torch._foreach_add_(params, grads, alpha=-lr), SGD's step (#184 B): each pair through the in-place add."""
    if len(self) != len(other):
        raise RuntimeError("_foreach_add_ needs lists of the same length")
    for tensor, update in zip(self, other):
        _add_inplace(tensor, update, alpha=alpha)


class _Shape(ctypes.Structure):
    """compat_kernels.cu's CompatShape: up to six sizes and two operands' strides, passed by value."""
    _fields_ = [("size", ctypes.c_int64 * 6), ("a", ctypes.c_int64 * 6), ("b", ctypes.c_int64 * 6)]


def _shape(sizes, a_strides, b_strides=None) -> _Shape:
    if len(sizes) > 6:
        raise NotImplementedError("Maxwell elementwise kernels take at most six dimensions")
    shape = _Shape()
    for d, size in enumerate(sizes):
        shape.size[d], shape.a[d] = size, a_strides[d]
        shape.b[d] = b_strides[d] if b_strides is not None else 0
    return shape


def _launch(name: str, grid: int, *arguments, block: int = 256, device: torch.device | None = None) -> None:
    """Run one kernel of the adapter's module by name (cuda_compat_launch_named), on the current stream."""
    global _bridge
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    pointers = (ctypes.c_void_p * len(arguments))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in arguments])
    index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
    stream = torch.cuda.current_stream(index).cuda_stream
    result = _bridge.cuda_compat_launch_named(ptx_path, name.encode(), index, max(1, grid), 1, 1, block, 1, 1, 0,
                                              stream, pointers)
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell {name} failed with CUresult={result}")


def _ensure_context() -> None:
    """Make the current device's primary context current for the driver-API launch that follows (#188): autograd's
    backward thread may have none, since a cached allocation makes no CUDA call."""
    result = _bridge.cuda_compat_ensure_context(torch.cuda.current_device())
    if result != 0:
        raise RuntimeError(f"CUDA Driver API could not make a context current (CUresult={result})")


def _ptr(tensor: torch.Tensor | None) -> ctypes.c_uint64:
    return ctypes.c_uint64(tensor.data_ptr() if tensor is not None else 0)


def _blocks(count: int) -> int:
    return min(4096, (count + 255) // 256)


def _f32(*tensors: torch.Tensor | None, contiguous: bool = False) -> None:
    """Refuse (to the CPU fallback) anything but float32 CUDA tensors of one device, without negative strides."""
    present = [t for t in tensors if t is not None]
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not t.is_cuda for t in present):
        raise NotImplementedError("Maxwell elementwise kernels take float32 CUDA tensors only")
    if len({t.device for t in present}) > 1 or any(s < 0 for t in present for s in t.stride()):
        raise NotImplementedError("Maxwell elementwise kernels need one device and nonnegative strides")
    if contiguous and any(not t.is_contiguous() for t in present):
        raise NotImplementedError("Maxwell elementwise kernel needs contiguous tensors")


def _scalar(value: Any) -> float | None:
    """A Python number, or a 0-d CPU tensor (x * torch.tensor(2.0)); None for anything else."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.dim() == 0 and not value.is_cuda and value.dtype.is_floating_point else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _binary(left: torch.Tensor, right: Any, op: int, alpha: float = 1.0,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """left (op) right with broadcasting (#184): op 0 add, 1 sub, 2 mul, 3 div against a float32 CUDA tensor, or
    the scalar forms 4-7 (add, mul, div, reverse-subtract) when right is a number. `out`, when given, is a contiguous
    tensor of the result's shape that receives it (in place when it is `left`); an operand that overlaps it any other
    way is refused."""
    scalar = _scalar(right)
    if scalar is not None:
        _f32(left)
        shape, other = left.shape, None
        # add -> 4 (x + alpha*s), sub -> 4 (x - alpha*s), mul -> 5, div -> 6.
        op, value = {0: (4, alpha * scalar), 1: (4, -alpha * scalar), 2: (5, scalar), 3: (6, scalar)}[op]
    else:
        _f32(left, right)
        shape, other, value = torch.broadcast_shapes(left.shape, right.shape), right, 0.0
    if out is None:
        output = torch.empty(shape, dtype=torch.float32, device=left.device)
    else:
        if out.dtype != torch.float32 or out.shape != shape or not out.is_contiguous() or out.device != left.device:
            raise NotImplementedError("Maxwell elementwise output must be a contiguous float32 tensor of the result's shape")
        if any(operand is not None and _overlaps_otherwise(operand, out) for operand in (left, other)):
            raise NotImplementedError("Maxwell elementwise kernels refuse an operand that partly overlaps the output")
        output = out
    count = output.numel()
    if count:
        lv = left.expand(shape)
        rv = other.expand(shape) if other is not None else lv
        _launch("compat_binary_f32", _blocks(count), _ptr(lv), _ptr(rv), _ptr(output), ctypes.c_int64(count),
                ctypes.c_int(len(shape)), _shape(shape, lv.stride(), rv.stride()), ctypes.c_int(op),
                ctypes.c_float(alpha), ctypes.c_float(value), device=lv.device)
    return output


def _overlaps_otherwise(operand: torch.Tensor, output: torch.Tensor) -> bool:
    """Whether operand shares memory with output other than as the very same view (x += y reads x[i] before writing
    x[i]; any other sharing could read a value already written)."""
    if operand.untyped_storage().data_ptr() != output.untyped_storage().data_ptr():
        return False
    return not (operand.data_ptr() == output.data_ptr() and operand.shape == output.shape
                and operand.stride() == output.stride())


@_falls_back_to_cpu("sub.Tensor")
def _sub(self: torch.Tensor, other: Any, *, alpha: int | float = 1) -> torch.Tensor:
    output = _binary(self, other, 1, float(alpha))
    _dispatch_state.last_path = "llvm-sm52-binary-f32"
    return output


@_falls_back_to_cpu("rsub.Scalar")
def _rsub(self: torch.Tensor, other: Any, alpha: int | float = 1) -> torch.Tensor:
    """other - alpha * self (1.0 - x): the kernel's reverse-subtract form, for alpha 1."""
    scalar = _scalar(other)
    if scalar is None or alpha != 1:
        raise NotImplementedError("Maxwell reverse subtraction takes a number and alpha=1")
    _f32(self)
    output = torch.empty(self.shape, dtype=torch.float32, device=self.device)
    count = output.numel()
    if count:
        sv = self.expand(self.shape)
        _launch("compat_binary_f32", _blocks(count), _ptr(sv), _ptr(sv), _ptr(output), ctypes.c_int64(count),
                ctypes.c_int(self.dim()), _shape(self.shape, sv.stride(), sv.stride()), ctypes.c_int(7),
                ctypes.c_float(1.0), ctypes.c_float(scalar), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-binary-f32"
    return output


@_falls_back_to_cpu("mul.Tensor")
def _mul(self: torch.Tensor, other: Any) -> torch.Tensor:
    output = _binary(self, other, 2)
    _dispatch_state.last_path = "llvm-sm52-binary-f32"
    return output


@_falls_back_to_cpu("div.Tensor")
def _div(self: torch.Tensor, other: Any) -> torch.Tensor:
    output = _binary(self, other, 3)
    _dispatch_state.last_path = "llvm-sm52-binary-f32"
    return output


@_falls_back_to_cpu("gelu")
def _gelu(self: torch.Tensor, *, approximate: str = "none") -> torch.Tensor:
    _f32(self, contiguous=True)
    output = torch.empty_like(self)
    if self.numel():
        _launch("compat_unary_f32", _blocks(self.numel()), _ptr(self), _ptr(output), ctypes.c_int64(self.numel()),
                ctypes.c_int(0 if approximate == "none" else 1), ctypes.c_float(0.0), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-gelu-f32"
    return output


@_falls_back_to_cpu("gelu_backward")
def _gelu_backward(grad_output: torch.Tensor, self: torch.Tensor, *, approximate: str = "none") -> torch.Tensor:
    _f32(grad_output, self, contiguous=True)
    if grad_output.shape != self.shape:
        raise NotImplementedError("Maxwell GELU backward needs the gradient in the input's shape")
    output = torch.empty_like(self)
    if self.numel():
        _launch("compat_gelu_backward_f32", _blocks(self.numel()), _ptr(grad_output), _ptr(self), _ptr(output),
                ctypes.c_int64(self.numel()), ctypes.c_int(0 if approximate == "none" else 1), device=grad_output.device)
    _dispatch_state.last_path = "llvm-sm52-gelu-backward-f32"
    return output


def _rows(tensor: torch.Tensor, dim: int) -> tuple[int, int]:
    if dim % max(1, tensor.dim()) != tensor.dim() - 1:
        raise NotImplementedError("Maxwell softmax runs over the last dimension only")
    columns = tensor.shape[-1] if tensor.dim() else 1
    return (tensor.numel() // columns if columns else 0), columns


def _softmax_rows(self: torch.Tensor, dim: int, mode: int) -> torch.Tensor:
    """mode 0 softmax, 1 log-softmax, 2 safe softmax (fully masked rows give zeros)."""
    _f32(self, contiguous=True)
    rows, columns = _rows(self, dim)
    output = torch.empty_like(self)
    if rows and columns:
        _launch("compat_softmax_rows_f32", rows, _ptr(self), _ptr(output), ctypes.c_int64(rows),
                ctypes.c_int64(columns), ctypes.c_int(mode), device=self.device)
    return output


def _softmax_rows_backward(grad_output: torch.Tensor, output: torch.Tensor, dim: int, log: bool) -> torch.Tensor:
    _f32(grad_output, output, contiguous=True)
    rows, columns = _rows(output, dim)
    grad_input = torch.empty_like(output)
    if rows and columns:
        _launch("compat_softmax_backward_rows_f32", rows, _ptr(grad_output), _ptr(output), _ptr(grad_input),
                ctypes.c_int64(rows), ctypes.c_int64(columns), ctypes.c_int(int(log)), device=grad_output.device)
    return grad_input


@_falls_back_to_cpu("_softmax")
def _softmax(self: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    output = _softmax_rows(self, dim, 0)
    _dispatch_state.last_path = "llvm-sm52-softmax-f32"
    return output


@_falls_back_to_cpu("_log_softmax")
def _log_softmax(self: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    output = _softmax_rows(self, dim, 1)
    _dispatch_state.last_path = "llvm-sm52-log-softmax-f32"
    return output


@_falls_back_to_cpu("_safe_softmax")
def _safe_softmax(self: torch.Tensor, dim: int, dtype=None) -> torch.Tensor:
    """Attention with a mask (#184 B): softmax, except a row masked everywhere gives zeros."""
    if dtype is not None and dtype != torch.float32:
        raise NotImplementedError("Maxwell safe softmax keeps float32")
    output = _softmax_rows(self, dim, 2)
    _dispatch_state.last_path = "llvm-sm52-safe-softmax-f32"
    return output


@_falls_back_to_cpu("_softmax_backward_data")
def _softmax_backward(grad_output: torch.Tensor, output: torch.Tensor, dim: int, input_dtype) -> torch.Tensor:
    grad_input = _softmax_rows_backward(grad_output, output, dim, False)
    _dispatch_state.last_path = "llvm-sm52-softmax-backward-f32"
    return grad_input


@_falls_back_to_cpu("_log_softmax_backward_data")
def _log_softmax_backward(grad_output: torch.Tensor, output: torch.Tensor, dim: int, input_dtype) -> torch.Tensor:
    grad_input = _softmax_rows_backward(grad_output, output, dim, True)
    _dispatch_state.last_path = "llvm-sm52-log-softmax-backward-f32"
    return grad_input


def _norm_rows(input: torch.Tensor, normalized_shape) -> tuple[int, int, tuple[int, ...]]:
    columns = 1
    for size in normalized_shape:
        columns *= int(size)
    keep = input.dim() - len(normalized_shape)
    if keep < 0 or tuple(input.shape[keep:]) != tuple(int(s) for s in normalized_shape):
        raise NotImplementedError("Maxwell layer normalization: the normalized shape must end the input's shape")
    rows = input.numel() // columns if columns else 0
    return rows, columns, tuple(input.shape[:keep]) + (1,) * len(normalized_shape)


@_falls_back_to_cpu("native_layer_norm")
def _native_layer_norm(input: torch.Tensor, normalized_shape, weight: torch.Tensor | None, bias: torch.Tensor | None,
                       eps: float):
    _f32(input, weight, bias, contiguous=True)
    rows, columns, stat_shape = _norm_rows(input, normalized_shape)
    output = torch.empty_like(input)
    mean = torch.empty(stat_shape, dtype=torch.float32, device=input.device)
    rstd = torch.empty(stat_shape, dtype=torch.float32, device=input.device)
    if rows and columns:
        _launch("compat_layer_norm_forward_f32", rows, _ptr(input), _ptr(weight), _ptr(bias), _ptr(output), _ptr(mean),
                _ptr(rstd), ctypes.c_int64(rows), ctypes.c_int64(columns), ctypes.c_float(eps), device=input.device)
    _dispatch_state.last_path = "llvm-sm52-native-layer-norm-f32"
    return output, mean, rstd


@_falls_back_to_cpu("native_layer_norm_backward")
def _native_layer_norm_backward(grad_out: torch.Tensor, input: torch.Tensor, normalized_shape, mean: torch.Tensor,
                                rstd: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None,
                                output_mask):
    _f32(grad_out, input, mean, rstd, weight, contiguous=True)
    rows, columns, _ = _norm_rows(input, normalized_shape)
    grad_input = torch.empty_like(input) if output_mask[0] else None
    grad_weight = torch.empty(tuple(int(s) for s in normalized_shape), dtype=torch.float32, device=input.device) \
        if output_mask[1] and weight is not None else None
    grad_bias = torch.empty(tuple(int(s) for s in normalized_shape), dtype=torch.float32, device=input.device) \
        if output_mask[2] and bias is not None else None
    if rows and columns:
        if grad_input is not None:
            _launch("compat_layer_norm_backward_input_f32", rows, _ptr(grad_out), _ptr(input), _ptr(mean), _ptr(rstd),
                    _ptr(weight), _ptr(grad_input), ctypes.c_int64(rows), ctypes.c_int64(columns), device=grad_out.device)
        if grad_weight is not None or grad_bias is not None:
            _launch("compat_layer_norm_backward_params_f32", columns, _ptr(grad_out), _ptr(input), _ptr(mean),
                    _ptr(rstd), _ptr(grad_weight), _ptr(grad_bias), ctypes.c_int64(rows), ctypes.c_int64(columns), device=grad_out.device)
    _dispatch_state.last_path = "llvm-sm52-native-layer-norm-backward-f32"
    return grad_input, grad_weight, grad_bias


@_falls_back_to_cpu("sum.dim_IntList")
def _sum(self: torch.Tensor, dim, keepdim: bool = False, *, dtype=None) -> torch.Tensor:
    if dtype is not None and dtype != torch.float32:
        raise NotImplementedError("Maxwell sum keeps float32")
    _f32(self)
    ndim = self.dim()
    dims = sorted({d % ndim for d in dim} if dim else set(range(ndim))) if ndim else []
    kept = [d for d in range(ndim) if d not in dims]
    if len(kept) > 6 or len(dims) > 6:
        raise NotImplementedError("Maxwell sum takes at most six dimensions")
    out_shape = [1 if d in dims else self.shape[d] for d in range(ndim)] if keepdim else [self.shape[d] for d in kept]
    output = torch.empty(out_shape, dtype=torch.float32, device=self.device)
    outputs = output.numel()
    reduced_count = 1
    for d in dims:
        reduced_count *= self.shape[d]
    if ndim == 0:
        output.copy_(self)   # same dtype and device: a plain memory copy
    elif outputs:
        if reduced_count == 0:
            _launch("compat_unary_f32", _blocks(outputs), _ptr(self), _ptr(output), ctypes.c_int64(outputs),
                    ctypes.c_int(2), ctypes.c_float(0.0), device=self.device)
        else:
            _launch("compat_reduce_sum_f32", outputs, _ptr(self), _ptr(output), ctypes.c_int64(outputs),
                    ctypes.c_int(len(kept)), _shape([self.shape[d] for d in kept], [self.stride(d) for d in kept]),
                    ctypes.c_int(len(dims)), _shape([self.shape[d] for d in dims], [self.stride(d) for d in dims]),
                    ctypes.c_int64(reduced_count), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-sum-f32"
    return output


@_falls_back_to_cpu("fill_.Scalar")
def _fill(self: torch.Tensor, value) -> torch.Tensor:
    _f32(self, contiguous=True)
    if self.numel():
        _launch("compat_unary_f32", _blocks(self.numel()), _ptr(self), _ptr(self), ctypes.c_int64(self.numel()),
                ctypes.c_int(2), ctypes.c_float(float(value)), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-fill-f32"
    return self


@_falls_back_to_cpu("zero_")
def _zero(self: torch.Tensor) -> torch.Tensor:
    _f32(self, contiguous=True)
    if self.numel():
        _launch("compat_unary_f32", _blocks(self.numel()), _ptr(self), _ptr(self), ctypes.c_int64(self.numel()),
                ctypes.c_int(2), ctypes.c_float(0.0), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-fill-f32"
    return self


def _gemm(left: torch.Tensor, right: torch.Tensor, output: torch.Tensor, bias: torch.Tensor | None = None,
          alpha: float = 1.0, beta: float = 0.0, name: str = "matrix multiplication") -> torch.Tensor:
    """output = alpha * left @ right + beta * bias on the Maxwell tiled kernel (#184), for 2D operands or a batch of
    them (3D). Every operand goes by its strides, so transposed and broadcast views need no copy; a case the kernel
    doesn't take raises NotImplementedError, which the wrapper hands to the CPU fallback."""
    global _bridge
    tensors = [left, right] + ([bias] if bias is not None and beta != 0 else [])
    if any(t.dtype != torch.float32 for t in tensors):
        raise NotImplementedError(f"Maxwell {name} supports float32 tensors only")
    if any(not t.is_cuda or t.device != output.device for t in tensors):
        raise NotImplementedError(f"Maxwell {name} requires tensors on one CUDA device")
    if any(stride < 0 for t in tensors for stride in t.stride()):
        raise NotImplementedError(f"Maxwell {name} does not support negative strides")
    batched = left.dim() == 3
    batch = left.shape[0] if batched else 1
    rows, inner = left.shape[-2:]
    columns = right.shape[-1]
    if (rows + 63) // 64 > 65535 or batch > 65535 or max(rows, columns, inner) > 2147483647:
        raise NotImplementedError(f"Maxwell {name} is too large for one launch")
    ls, rs = left.stride(), right.stride()
    bias_pointer, bias_strides = 0, (0, 0, 0)
    if bias is not None and beta != 0:
        view = bias.expand(output.shape)
        bs = view.stride()
        bias_pointer, bias_strides = view.data_ptr(), ((bs[-2], bs[-1], bs[0] if batched else 0))
    if output.numel() == 0:
        _dispatch_state.last_path = "llvm-sm52-gemm-f32-empty"
        return output
    if _bridge is None:
        _bridge, ptx_path = _load_bridge()
    else:
        ptx_path = os.fsencode(os.environ["CUDA_COMPAT_VECTOR_ADD_PTX"])
    current_stream = torch.cuda.current_stream(output.device)
    _record_stream(tuple(tensors), current_stream)
    _ensure_context()
    result = _bridge.cuda_compat_launch_matrix_multiply_f32(
        ptx_path, left.data_ptr(), right.data_ptr(), bias_pointer, output.data_ptr(), batch, rows, columns, inner,
        ls[-2], ls[-1], rs[-2], rs[-1], bias_strides[0], bias_strides[1], ls[0] if batched else 0,
        rs[0] if batched else 0, bias_strides[2], rows * columns if batched else 0, float(alpha), float(beta),
        current_stream.cuda_stream)
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell {name} failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-gemm-f32"
    return output


@_falls_back_to_cpu("mm")
def _mm(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if self.dim() != 2 or other.dim() != 2:
        raise NotImplementedError("Maxwell matrix multiplication requires two-dimensional tensors")
    if self.shape[1] != other.shape[0]:
        raise RuntimeError("matrix multiplication shapes do not align")
    output = _gemm(self, other, torch.empty((self.shape[0], other.shape[1]), dtype=torch.float32, device=self.device))
    if output.numel():
        _dispatch_state.last_path = "llvm-sm52-mm-f32"
    return output


@_falls_back_to_cpu("addmm")
def _addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta: float = 1,
           alpha: float = 1) -> torch.Tensor:
    if mat1.dim() != 2 or mat2.dim() != 2 or mat1.shape[1] != mat2.shape[0]:
        raise NotImplementedError("Maxwell addmm requires aligned two-dimensional matrices")
    output = torch.empty((mat1.shape[0], mat2.shape[1]), dtype=torch.float32, device=mat1.device)
    if beta != 0 and input.dim() > 2:
        raise NotImplementedError("Maxwell addmm takes a bias that broadcasts to the output")
    output = _gemm(mat1, mat2, output, input, float(alpha), float(beta), "addmm")
    if output.numel():
        _dispatch_state.last_path = "llvm-sm52-addmm-f32"
    return output


@_falls_back_to_cpu("bmm")
def _bmm(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if self.dim() != 3 or other.dim() != 3 or self.shape[0] != other.shape[0] or self.shape[2] != other.shape[1]:
        raise NotImplementedError("Maxwell bmm requires aligned three-dimensional batches")
    output = torch.empty((self.shape[0], self.shape[1], other.shape[2]), dtype=torch.float32, device=self.device)
    output = _gemm(self, other, output, name="batched matrix multiplication")
    if output.numel():
        _dispatch_state.last_path = "llvm-sm52-bmm-f32"
    return output


@_falls_back_to_cpu("arange")
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
    _ensure_context()
    result = _bridge.cuda_compat_launch_arange_i64(
        ptx_path, output.data_ptr(), count, stream
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell arange failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-arange-i64"
    return output


def _embedding_rows_on_cpu(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Look up only the rows the indices name (#175): a 2D contiguous weight's rows, sorted and copied in
    consecutive runs, are plain memory copies, so a vocabulary larger than the fallback limit still works when the
    rows it touches fit under it. The backward is PyTorch's own (embedding_dense_backward)."""
    if weight.dim() != 2 or not weight.is_contiguous():
        return _cpu_fallback(torch.ops.aten.embedding.default, "embedding")(
            weight, indices, padding_idx, scale_grad_by_freq, sparse)
    cpu_indices = _cuda_tensor_to_cpu(indices) if _is_cuda(indices) else indices
    rows, inverse = torch.unique(cpu_indices, return_inverse=True)
    if rows.numel() * weight.shape[1] * weight.element_size() > _CPU_FALLBACK_TOTAL_BYTES:
        raise NotImplementedError(f"Maxwell CPU fallback for embedding is limited to "
                                  f"{_CPU_FALLBACK_TOTAL_BYTES // _MIB} MiB of looked-up rows")
    picked = torch.empty((rows.numel(), weight.shape[1]), dtype=weight.dtype)
    numbers = rows.tolist()
    start = 0
    with torch.no_grad():
        while start < len(numbers):
            end = start + 1
            while end < len(numbers) and numbers[end] == numbers[end - 1] + 1:
                end += 1
            first = numbers[start]
            picked[start:end] = _cuda_tensor_to_cpu(weight[first:first + end - start])
            start = end
    output = torch.nn.functional.embedding(inverse, picked)
    _dispatch_state.last_path = "cpu-fallback-embedding-rows"
    return _move(output, weight.device)


@_falls_back_to_cpu("embedding", oversize=_embedding_rows_on_cpu)
def _embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
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
        _bridge, _ = _load_bridge()
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

    _record_stream((weight, indices), current_stream)
    count = token_count * embedding_size
    _launch("compat_embedding_f32", (count + 255) // 256, _ptr(weight), _ptr(indices), _ptr(output),
            ctypes.c_uint64(token_count), ctypes.c_uint64(embedding_size),
            ctypes.c_uint64(indices.shape[1] if indices.dim() == 2 else 1), ctypes.c_uint64(index_strides[0]),
            ctypes.c_uint64(index_strides[1] if indices.dim() == 2 else 0), ctypes.c_uint64(indices.dim()),
            device=weight.device)
    _dispatch_state.last_path = "llvm-sm52-embedding-f32"
    return output


@_falls_back_to_cpu("layer_norm")
def _layer_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    cudnn_enable: bool = True,
) -> torch.Tensor:
    """Layer normalization without autograd (training decomposes to native_layer_norm, _composite_autograd): the
    native_layer_norm kernel. Parameters that require grad are fine here, since nothing differentiates this call."""
    del cudnn_enable
    output = _native_layer_norm.__wrapped__(input, normalized_shape, weight, bias, eps)[0]
    _dispatch_state.last_path = "llvm-sm52-layernorm-f32"
    return output


@_falls_back_to_cpu("embedding_dense_backward")
def _embedding_dense_backward(grad_output: torch.Tensor, indices: torch.Tensor, num_weights: int, padding_idx: int,
                              scale_grad_by_freq: bool) -> torch.Tensor:
    """The embedding's weight gradient on the GPU (#184 B): each looked-up row's gradient added into a zeroed table."""
    if scale_grad_by_freq:
        raise NotImplementedError("Maxwell embedding backward does not scale by frequency")
    _f32(grad_output, contiguous=True)
    if indices.dtype != torch.int64 or not indices.is_cuda or indices.device != grad_output.device \
            or not indices.is_contiguous():
        raise NotImplementedError("Maxwell embedding backward takes contiguous int64 CUDA indices")
    columns = grad_output.shape[-1] if grad_output.dim() else 1
    tokens = indices.numel()
    if grad_output.numel() != tokens * columns:
        raise NotImplementedError("Maxwell embedding backward needs one gradient row per index")
    grad_weight = torch.empty((int(num_weights), columns), dtype=torch.float32, device=grad_output.device)
    if grad_weight.numel():
        _launch("compat_unary_f32", _blocks(grad_weight.numel()), _ptr(grad_weight), _ptr(grad_weight),
                ctypes.c_int64(grad_weight.numel()), ctypes.c_int(2), ctypes.c_float(0.0), device=grad_output.device)
    if tokens * columns:
        _launch("compat_embedding_backward_f32", _blocks(tokens * columns), _ptr(grad_output), _ptr(indices),
                _ptr(grad_weight), ctypes.c_int64(tokens), ctypes.c_int64(columns), ctypes.c_int64(int(num_weights)),
                ctypes.c_int64(int(padding_idx)), device=grad_output.device)
    _dispatch_state.last_path = "llvm-sm52-embedding-backward-f32"
    return grad_weight


_REDUCTIONS = {0: "none", 1: "mean", 2: "sum"}


def _nll_shapes(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> tuple[int, int]:
    """Rows and classes of a (classes,) or (rows, classes) nll_loss, after checking the targets on the host (a
    target out of range raises as on the CPU, never reads past the row)."""
    _f32(input, weight, contiguous=True)
    if input.dim() not in (1, 2) or target.dtype != torch.int64 or not target.is_cuda \
            or target.device != input.device or not target.is_contiguous():
        raise NotImplementedError("Maxwell nll_loss takes a 1D or 2D float32 input and contiguous int64 CUDA targets")
    rows, columns = (input.shape[0] if input.dim() == 2 else 1), input.shape[-1]
    if target.numel() != rows or (weight is not None and weight.numel() != columns):
        raise NotImplementedError("Maxwell nll_loss needs one target per row and one weight per class")
    return rows, columns


def _check_targets(target: torch.Tensor, columns: int, ignore_index: int) -> None:
    host = _cuda_tensor_to_cpu(target)
    kept = host[host != ignore_index]
    if kept.numel() and (int(kept.min()) < 0 or int(kept.max()) >= columns):
        raise IndexError(f"Target {int(kept.max()) if int(kept.max()) >= columns else int(kept.min())} is out of bounds.")


@_falls_back_to_cpu("nll_loss_forward")
def _nll_loss_forward(self: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None, reduction: int,
                      ignore_index: int):
    """cross_entropy's second half (#184 B): the loss of a batch of log-probabilities without moving them."""
    rows, columns = _nll_shapes(self, target, weight)
    if reduction not in _REDUCTIONS:
        raise NotImplementedError(f"Maxwell nll_loss has no reduction {reduction}")
    _check_targets(target, columns, ignore_index)
    output = torch.empty((rows,) if reduction == 0 and self.dim() == 2 else (), dtype=torch.float32,
                         device=self.device)
    total_weight = torch.empty((), dtype=torch.float32, device=self.device)
    _launch("compat_nll_loss_forward_f32", 1, _ptr(self), _ptr(target), _ptr(weight), _ptr(output),
            _ptr(total_weight), ctypes.c_int64(rows), ctypes.c_int64(columns), ctypes.c_int64(int(ignore_index)),
            ctypes.c_int(reduction), ctypes.c_int(int(self.dim() == 2)), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-nll-loss-f32"
    return output, total_weight


@_falls_back_to_cpu("nll_loss_backward")
def _nll_loss_backward(grad_output: torch.Tensor, self: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor | None, reduction: int, ignore_index: int,
                       total_weight: torch.Tensor) -> torch.Tensor:
    rows, columns = _nll_shapes(self, target, weight)
    _f32(grad_output, total_weight, contiguous=True)
    if reduction not in _REDUCTIONS or grad_output.numel() != (rows if reduction == 0 else 1):
        raise NotImplementedError("Maxwell nll_loss backward needs one gradient per row, or one for a reduction")
    grad_input = torch.empty_like(self)
    if grad_input.numel():
        _launch("compat_unary_f32", _blocks(grad_input.numel()), _ptr(grad_input), _ptr(grad_input),
                ctypes.c_int64(grad_input.numel()), ctypes.c_int(2), ctypes.c_float(0.0), device=self.device)
    if rows:
        _launch("compat_nll_loss_backward_f32", _blocks(rows), _ptr(grad_output), _ptr(target), _ptr(weight),
                _ptr(total_weight), _ptr(grad_input), ctypes.c_int64(rows), ctypes.c_int64(columns),
                ctypes.c_int64(int(ignore_index)), ctypes.c_int(reduction), device=self.device)
    _dispatch_state.last_path = "llvm-sm52-nll-loss-backward-f32"
    return grad_input


@_falls_back_to_cpu("relu")
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
    _ensure_context()
    result = _bridge.cuda_compat_launch_relu_f32(
        ptx_path, self.data_ptr(), output.data_ptr(), self.numel(), stream
    )
    if result != 0:
        raise RuntimeError(f"CUDA Driver API Maxwell ReLU failed with CUresult={result}")
    _dispatch_state.last_path = "llvm-sm52-relu-f32"
    return output


def _copy_in_pieces(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Copy a contiguous tensor into a contiguous one of the same size, at most _CPU_FALLBACK_MAX_BYTES at a time:
    between a GPU and host memory each piece is a plain memory copy, so no Maxwell kernel is needed (#178)."""
    flat_destination, flat_source = destination.view(-1), source.reshape(-1)
    step = max(1, _CPU_FALLBACK_MAX_BYTES // max(1, source.element_size()))
    with torch.no_grad():
        for start in range(0, flat_source.numel(), step):
            flat_destination[start:start + step].copy_(flat_source[start:start + step])
    return destination


class _MoveInPieces(torch.autograd.Function):
    """A copy to another device in pieces that autograd can go back through (the backward moves in pieces too)."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, device) -> torch.Tensor:
        ctx.source_device = tensor.device
        return _copy_in_pieces(torch.empty(tensor.shape, dtype=tensor.dtype, device=device), tensor)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _move(grad if grad.device.type != "cpu" else grad.contiguous(), ctx.source_device), None


def _move(tensor: torch.Tensor, device) -> torch.Tensor:
    """`tensor.to(device)`, in pieces of at most _CPU_FALLBACK_MAX_BYTES when a contiguous tensor is bigger."""
    if tensor.numel() * tensor.element_size() <= _CPU_FALLBACK_MAX_BYTES or not tensor.is_contiguous():
        return tensor.to(device)
    if torch.is_grad_enabled() and tensor.requires_grad:
        return _MoveInPieces.apply(tensor, device)
    return _copy_in_pieces(torch.empty(tensor.shape, dtype=tensor.dtype, device=device), tensor)


def _cuda_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and tensor.requires_grad:
        _dispatch_state.last_path = "differentiable-cuda-to-cpu-copy"
        return _move(tensor, "cpu")
    if tensor.numel() == 0 or tensor.is_contiguous():
        return _move(tensor, "cpu")
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
        raise NotImplementedError("Maxwell CPU fallback is limited to 64 MiB per strided tensor view")

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


def _strided_copy(source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """output[...] = source[...] element by element through both tensors' strides, for any 1-, 2-, 4- or 8-byte
    dtype (#184 B): a strided view's clone stays on the GPU instead of a round trip through the CPU."""
    if not source.is_cuda or not output.is_cuda or source.device != output.device or source.dtype != output.dtype \
            or source.shape != output.shape:
        raise NotImplementedError("Maxwell strided copy needs one CUDA device, one dtype and one shape")
    if source.layout != torch.strided or source.element_size() not in (1, 2, 4, 8) or source.dim() > 6:
        raise NotImplementedError("Maxwell strided copy takes dense tensors of up to six dimensions, 1 to 8 bytes")
    count = output.numel()
    if count:
        _launch("compat_strided_copy", _blocks(count), _ptr(source), _ptr(output), ctypes.c_int64(count),
                ctypes.c_int(source.dim()), _shape(source.shape, source.stride(), output.stride()),
                ctypes.c_int(source.element_size()), device=source.device)
    return output


@_falls_back_to_cpu("clone")
def _clone(self: torch.Tensor, *, memory_format=None) -> torch.Tensor:
    output = torch.empty_like(self, memory_format=memory_format or torch.preserve_format)
    _strided_copy(self, output)
    _dispatch_state.last_path = "llvm-sm52-strided-copy"
    return output


def _contiguous(
    input: torch.Tensor,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> torch.Tensor:
    if not input.is_cuda:
        raise NotImplementedError("Maxwell contiguous fallback requires a CUDA tensor")
    if memory_format != torch.contiguous_format:
        raise NotImplementedError("Maxwell contiguous fallback supports contiguous_format only")
    if input.is_contiguous(memory_format=memory_format):
        _dispatch_state.last_path = "pytorch-contiguous-alias"
        return input
    try:
        result = _strided_copy(input, torch.empty(input.shape, dtype=input.dtype, device=input.device))
        _dispatch_state.last_path = "llvm-sm52-strided-copy"
    except NotImplementedError:
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


# Operators that write some inputs without declaring it: (positions written, position of the "training" flag).
_UNDECLARED_WRITES = {
    "aten::native_batch_norm": ((3, 4), 5),
    "aten::_native_batch_norm_legit": ((3, 4), 5),
}

# Overrides of CompositeImplicitAutograd operators. Autograd differentiates those through PyTorch's own
# decomposition, so a CUDA kernel in its place has no backward: _composite_autograd sends training through the
# decomposition, and _inference_only refuses if anything still reaches the kernel needing a gradient.
_INFERENCE_ONLY = {"contiguous", "layer_norm", "scaled_dot_product_attention"}


def _composite_autograd(operation) -> Callable[..., Any]:
    """An AutogradCUDA kernel: decompose when a gradient is needed, otherwise use the CUDA override."""
    def run(*args: Any, **kwargs: Any) -> Any:
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in _tensor_inputs((args, kwargs))):
            return operation.decompose(*args, **kwargs)
        with torch._C._AutoDispatchBelowAutograd():
            return operation(*args, **kwargs)
    return run


def _inference_only(kernel: Callable[..., Any], name: str) -> Callable[..., Any]:
    def run(*args: Any, **kwargs: Any) -> Any:
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in _tensor_inputs((args, kwargs))):
            error = NotImplementedError(
                f"Maxwell {name} runs only without autograd; training through it is not supported yet")
            unrouted.record(f"aten::{name}", error)
            raise error
        return kernel(*args, **kwargs)
    return run


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
        if transfer_bytes > _CPU_FALLBACK_TOTAL_BYTES:
            raise NotImplementedError(
                f"Maxwell CPU fallback for {name} is limited to {_CPU_FALLBACK_TOTAL_BYTES // _MIB} MiB of CUDA "
                "inputs per operation (CUDA_COMPAT_FALLBACK_TOTAL_MIB)"
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
        # Batch norm in training updates its running statistics in place without declaring it in its schema,
        # and the CPU kernel writes them directly: copy the CPU statistics back into the CUDA buffers.
        if name in _UNDECLARED_WRITES and cpu_args[_UNDECLARED_WRITES[name][1]]:
            for index in _UNDECLARED_WRITES[name][0]:
                original, updated = args[index], cpu_args[index]
                if isinstance(original, torch.Tensor) and original.is_cuda:
                    if not original.is_contiguous():
                        raise NotImplementedError(f"Maxwell CPU fallback for {name} cannot copy back a strided buffer")
                    original.copy_(updated)

        def result_bytes(value: Any) -> int:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, (tuple, list)):
                return sum(result_bytes(item) for item in value)
            if isinstance(value, Mapping):
                return sum(result_bytes(item) for item in value.values())
            return 0

        if result_bytes(cpu_result) > _CPU_FALLBACK_TOTAL_BYTES:
            raise NotImplementedError(
                f"Maxwell CPU fallback for {name} is limited to {_CPU_FALLBACK_TOTAL_BYTES // _MIB} MiB of outputs "
                "per operation (CUDA_COMPAT_FALLBACK_TOTAL_MIB)"
            )

        def move_to_device(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return _move(value, device) if value.device.type == "cpu" else value
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


def _is_cuda(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _written_arguments(schema) -> list[int] | None:
    """Positions of the tensor arguments an in-place operator writes, starting with its first, when it returns
    only those (or nothing: torch.optim's foreach steps and the AMP scaler ops); None for anything else."""
    if schema is None or not schema.arguments:
        return None
    written = [index for index, argument in enumerate(schema.arguments)
               if argument.alias_info is not None and argument.alias_info.is_write]
    if not written:
        return None
    if any(not str(schema.arguments[index].type).startswith(("Tensor", "List[Tensor]")) for index in written):
        return None
    sets = set().union(*(schema.arguments[index].alias_info.before_set for index in written))
    if not all(result.alias_info is not None and result.alias_info.before_set <= sets for result in schema.returns):
        return None
    return written


def _writes_only_self(schema) -> bool:
    return _written_arguments(schema) is not None


def _inplace_on_cpu(operation, name: str, args: tuple, kwargs: dict, schema=None) -> Any:
    """Run an in-place operator on CPU copies of the CUDA tensors it writes and copy the results back.

    A written tensor that is a strided view (x[:, 1:3]) is staged through its contiguous base: the base goes to
    the CPU, the operator updates the same view of it there, and the whole base is copied back.
    """
    schema = schema if schema is not None else getattr(operation, "_schema", None)
    positions = (_written_arguments(schema) if schema is not None else None) or [0]
    names = [argument.name for argument in schema.arguments] if schema is not None else []

    def value_at(index: int) -> Any:
        return args[index] if index < len(args) else kwargs.get(names[index]) if index < len(names) else None

    targets: list[torch.Tensor] = []
    for index in positions:
        value = value_at(index)
        targets.extend(value if isinstance(value, (list, tuple)) else [value])
    if not targets or any(not isinstance(target, torch.Tensor) or not _is_cuda(target) for target in targets):
        raise NotImplementedError(f"Maxwell in-place fallback for {name} requires CUDA target tensors")

    staged: dict[int, torch.Tensor] = {}
    write_back: list[tuple[torch.Tensor, torch.Tensor]] = []
    storages: set[int] = set()
    staged_bytes = 0
    for target in targets:
        if id(target) in staged:
            continue
        home = target if target.is_contiguous() else target._base
        if target.layout != torch.strided or home is None or not home.is_contiguous() or home.dtype != target.dtype:
            raise NotImplementedError(
                f"Maxwell in-place fallback for {name} requires a contiguous target or a view of a contiguous tensor")
        if home.data_ptr() in storages:
            raise NotImplementedError(f"Maxwell in-place fallback for {name} writes two views of one tensor")
        storages.add(home.data_ptr())
        cpu_home = _cuda_tensor_to_cpu(home)
        staged[id(target)] = cpu_home if home is target else cpu_home.as_strided(
            target.size(), target.stride(), target.storage_offset() - home.storage_offset())
        write_back.append((home, cpu_home))
        staged_bytes += _cuda_tensor_transfer_bytes(home)
    tensors = _tensor_inputs((args, kwargs))
    cuda_tensors = [tensor for tensor in tensors if _is_cuda(tensor)]
    if len({tensor.device for tensor in cuda_tensors}) != 1:
        raise NotImplementedError(f"Maxwell in-place fallback for {name} requires tensors from one CUDA device")
    read_bytes = sum(_cuda_tensor_transfer_bytes(tensor) for tensor in cuda_tensors if id(tensor) not in staged)
    if staged_bytes + read_bytes > _CPU_FALLBACK_TOTAL_BYTES:
        raise NotImplementedError(f"Maxwell in-place fallback for {name} is limited to "
                                  f"{_CPU_FALLBACK_TOTAL_BYTES // _MIB} MiB of CUDA inputs per operation "
                                  "(CUDA_COMPAT_FALLBACK_TOTAL_MIB)")

    def move_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if id(value) in staged:
                return staged[id(value)]
            return _cuda_tensor_to_cpu(value) if _is_cuda(value) else value
        if isinstance(value, (tuple, list)):
            return type(value)(move_to_cpu(item) for item in value)
        if isinstance(value, Mapping):
            return {key: move_to_cpu(item) for key, item in value.items()}
        return value

    cpu_args, cpu_kwargs = move_to_cpu(args), move_to_cpu(kwargs)
    operation(*cpu_args, **cpu_kwargs)
    if any(staged[id(target)].shape != target.shape or staged[id(target)].dtype != target.dtype for target in targets):
        # Operators such as resize_ or transpose_ change metadata, not values; those cannot be copied back.
        raise NotImplementedError(f"Maxwell in-place fallback for {name} only supports value updates")
    # A same-dtype copy from host memory into a contiguous CUDA tensor is a plain memory copy, no kernel.
    for home, cpu_home in write_back:
        _copy_in_pieces(home, cpu_home)
    _dispatch_state.last_path = f"cpu-fallback-inplace-{name}"
    if schema is None:
        return args[0]
    # Each result aliases one written argument: return that argument, as the CUDA kernel would have.
    by_set = {alias: index for index in positions for alias in schema.arguments[index].alias_info.before_set}
    results = tuple(value_at(by_set[next(iter(result.alias_info.before_set))]) for result in schema.returns)
    return None if not results else results[0] if len(results) == 1 else results


def _lstm_cell(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    """aten::_thnn_fused_lstm_cell on the CPU: gates in i, f, g, o order; workspace is the activated gates."""
    gates = input_gates + hidden_gates
    if input_bias is not None:
        gates = gates + input_bias + hidden_bias
    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    cy = f * cx + i * g
    return o * cy.tanh(), cy, torch.cat((i, f, g, o), 1)


def _lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    """aten::_thnn_fused_lstm_cell_backward_impl on the CPU: (grad_gates, grad_cx, grad_bias)."""
    i, f, g, o = workspace.chunk(4, 1)
    grad_hy = torch.zeros_like(cy) if grad_hy is None else grad_hy
    grad_cy = torch.zeros_like(cy) if grad_cy is None else grad_cy
    tanh_cy = cy.tanh()
    grad_c = grad_hy * o * (1 - tanh_cy * tanh_cy) + grad_cy
    grad_gates = torch.cat((grad_c * g * i * (1 - i), grad_c * cx * f * (1 - f),
                            grad_c * i * (1 - g * g), grad_hy * tanh_cy * o * (1 - o)), 1)
    return grad_gates, grad_c * f, grad_gates.sum(0) if has_bias else None


def _gru_cell(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    """aten::_thnn_fused_gru_cell on the CPU: gates in r, z, n order; workspace is r, z, n, hx, hn."""
    if input_bias is not None:
        input_gates, hidden_gates = input_gates + input_bias, hidden_gates + hidden_bias
    ir, iz, in_ = input_gates.chunk(3, 1)
    hr, hz, hn = hidden_gates.chunk(3, 1)
    r, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
    n = (in_ + r * hn).tanh()
    return n + z * (hx - n), torch.cat((r, z, n, hx, hn), 1)


def _gru_cell_backward(grad_hy, workspace, has_bias):
    """aten::_thnn_fused_gru_cell_backward on the CPU: (grad_input_gates, grad_hidden_gates, grad_hx, biases)."""
    r, z, n, hx, hn = workspace.chunk(5, 1)
    grad_z = grad_hy * (hx - n) * z * (1 - z)
    grad_n = grad_hy * (1 - z) * (1 - n * n)
    grad_r = grad_n * hn * r * (1 - r)
    grad_input = torch.cat((grad_r, grad_z, grad_n), 1)
    grad_hidden = torch.cat((grad_r, grad_z, grad_n * r), 1)
    biases = (grad_input.sum(0), grad_hidden.sum(0)) if has_bias else (None, None)
    return (grad_input, grad_hidden, grad_hy * z) + biases


# Operators with a CUDA kernel and no CPU one: the CPU fallback runs these instead.
_CPU_ONLY_HERE = {
    "aten::_thnn_fused_lstm_cell": _lstm_cell,
    "aten::_thnn_fused_lstm_cell_backward_impl": _lstm_cell_backward,
    "aten::_thnn_fused_gru_cell": _gru_cell,
    "aten::_thnn_fused_gru_cell_backward": _gru_cell_backward,
}


def _factory_on_cpu(operation, schema, name: str, args: tuple, kwargs: dict) -> Any:
    """Build a CUDA factory's result on the CPU and copy it to its device, or None when it is not a factory."""
    device = kwargs.get("device")
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    if schema is not None and (schema.is_mutable or any(result.alias_info is not None for result in schema.returns)):
        return None
    cpu_result = operation(*args, **{**kwargs, "device": torch.device("cpu")})
    outputs = _tensor_inputs(cpu_result)
    if not outputs:
        return None
    if sum(tensor.numel() * tensor.element_size() for tensor in outputs) > _CPU_FALLBACK_TOTAL_BYTES:
        raise NotImplementedError(f"Maxwell CPU fallback for {name} is limited to "
                                  f"{_CPU_FALLBACK_TOTAL_BYTES // _MIB} MiB of outputs per operation "
                                  "(CUDA_COMPAT_FALLBACK_TOTAL_MIB)")
    result = _move(cpu_result, device) if isinstance(cpu_result, torch.Tensor) else type(cpu_result)(
        _move(item, device) if isinstance(item, torch.Tensor) else item for item in cpu_result)
    _dispatch_state.last_path = f"cpu-fallback-{name}"
    return result


# Foreach operators that reduce each tensor: they may be split between tensors, never inside one.
_FOREACH_REDUCTIONS = {"aten::_foreach_norm", "aten::_foreach_max"}


def _foreach_in_pieces(operation, name: str, args: tuple, kwargs: dict, schema, inplace: bool) -> Any:
    """Run a foreach operator whose CUDA tensors exceed the fallback limit as several calls under it (#174).

    Each list element is independent, so the lists are split between elements; an element too big on its own is
    split into flat ranges when the operator is elementwise (every in-place foreach operator is) and every tensor
    at that position is contiguous with the same number of elements. Results come back in the original order.
    """
    names = [argument.name for argument in schema.arguments]
    kinds = [str(argument.type) for argument in schema.arguments]

    def value_at(index: int) -> Any:
        return args[index] if index < len(args) else kwargs.get(names[index])

    lists = [index for index, kind in enumerate(kinds) if kind == "List[Tensor]" and value_at(index) is not None]
    scalar_lists = [index for index, kind in enumerate(kinds) if kind.startswith("List[") and index not in lists
                    and isinstance(value_at(index), (list, tuple))]
    per_element = [index for index, kind in enumerate(kinds) if kind == "Tensor" and names[index] == "scalars"]
    count = len(value_at(lists[0]))
    if any(len(value_at(index)) != count for index in lists):
        raise NotImplementedError(f"Maxwell CPU fallback for {name} needs tensor lists of one length")
    listed = {id(tensor) for index in lists for tensor in value_at(index)}
    shared = sum(_cuda_tensor_transfer_bytes(tensor) for tensor in _cuda_tensors((args, kwargs)) if id(tensor) not in listed)
    room = _CPU_FALLBACK_TOTAL_BYTES - shared

    def element_bytes(position: int) -> int:
        return sum(_cuda_tensor_transfer_bytes(value_at(index)[position]) for index in lists
                   if _is_cuda(value_at(index)[position]))

    pieces: list[list[tuple[int, int | None, int | None]]] = []   # (element, flat start, flat end)
    current, used = [], 0
    for position in range(count):
        size = element_bytes(position)
        if size <= room:
            if used + size > room:
                pieces.append(current)
                current, used = [], 0
            current.append((position, None, None))
            used += size
            continue
        tensors = [value_at(index)[position] for index in lists]
        if (name in _FOREACH_REDUCTIONS or any(not tensor.is_contiguous() for tensor in tensors)
                or len({tensor.numel() for tensor in tensors}) != 1):
            raise NotImplementedError(f"Maxwell CPU fallback for {name} is limited to "
                                      f"{_CPU_FALLBACK_TOTAL_BYTES // _MIB} MiB of CUDA inputs per operation "
                                      "(CUDA_COMPAT_FALLBACK_TOTAL_MIB)")
        if current:
            pieces.append(current)
            current, used = [], 0
        numel = tensors[0].numel()
        step = max(1, room // max(1, sum(tensor.element_size() for tensor in tensors)))
        pieces.extend([(position, start, min(numel, start + step))] for start in range(0, numel, step))
    if current:
        pieces.append(current)

    results: list[list[Any]] = [[] for _ in range(count)]
    for piece in pieces:
        call = list(args) + [None] * max(0, len(names) - len(args))
        call_kwargs = dict(kwargs)
        for index, kind in enumerate(kinds):
            if index not in lists and index not in scalar_lists and index not in per_element:
                continue
            value = value_at(index)
            if value is None:
                continue
            if index in lists:
                part = [value[p] if start is None else value[p].reshape(-1)[start:end] for p, start, end in piece]
            elif index in scalar_lists:
                part = [value[p] for p, _, _ in piece]
            else:
                # A piece covers consecutive elements, so its scalars are a slice (a view, no kernel).
                part = value[piece[0][0]:piece[-1][0] + 1] if value.dim() else value
            if index < len(args):
                call[index] = part
            else:
                call_kwargs[names[index]] = part
        call = tuple(call[:len(args)])
        if inplace:
            _inplace_on_cpu(operation, name, call, call_kwargs, schema)
            continue
        out = _cpu_fallback(operation, name)(*call, **call_kwargs)
        for (p, start, end), tensor in zip(piece, out, strict=True):
            results[p].append((start, end, tensor))
    _dispatch_state.last_path = f"cpu-fallback-pieces-{name}"
    if inplace:
        written = _written_arguments(schema) or [0]
        return None if not schema.returns else value_at(written[0])
    assembled = []
    for position, parts in enumerate(results):
        if len(parts) == 1 and parts[0][0] is None:
            assembled.append(parts[0][2])
            continue
        model = value_at(lists[0])[position]
        whole = torch.empty(model.numel(), dtype=parts[0][2].dtype, device=parts[0][2].device)
        for start, end, tensor in parts:
            whole[start:end].copy_(tensor.reshape(-1))   # same dtype, same device: a plain memory copy
        assembled.append(whole.view(model.shape))
    return assembled


def _over_fallback_limit(args: tuple, kwargs: dict, limit: int | None = None) -> bool:
    try:
        return sum(_cuda_tensor_transfer_bytes(tensor) for tensor in _cuda_tensors((args, kwargs))) > (
            _CPU_FALLBACK_TOTAL_BYTES if limit is None else limit)
    except NotImplementedError:
        return False


class _MaxwellInferenceFallback(TorchDispatchMode):
    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = getattr(operation, "_schema", None)
        name = schema.name if schema is not None else str(operation)
        if not _cuda_tensors((args, kwargs)):
            # No CUDA tensor input: a factory such as torch.ones(device="cuda") is built on the CPU and copied over.
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                if not _is_architecture_failure(exc):
                    raise
                try:
                    result = _factory_on_cpu(operation, schema, name, args, kwargs)
                except NotImplementedError as error:
                    unrouted.record(name, error)
                    raise error from exc
                if result is None:
                    unrouted.record(name, exc)
                    raise
                return result
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            if not _is_architecture_failure(exc):
                raise
            if name.startswith("aten::_foreach_") and schema is not None and _over_fallback_limit(args, kwargs):
                try:
                    return _foreach_in_pieces(operation, name, args, kwargs, schema, _writes_only_self(schema))
                except NotImplementedError as error:
                    unrouted.record(name, error)
                    raise error from exc
            if _writes_only_self(schema):
                try:
                    return _inplace_on_cpu(operation, name, args, kwargs)
                except NotImplementedError as error:
                    unrouted.record(name, error)
                    raise error from exc
            if schema is not None and (
                schema.is_mutable
                or any(result.alias_info is not None for result in schema.returns)
            ):
                error = NotImplementedError(
                    f"CUDA operator {operation} has no Maxwell image and its aliasing semantics cannot use CPU fallback"
                )
                unrouted.record(name, error)
                raise error from exc
            return _cpu_fallback(_CPU_ONLY_HERE.get(name, operation), name)(*args, **kwargs)


def install() -> None:
    """Install the opt-in operator adapter only on a homogeneous sm_52 process."""
    global _fallback_mode, _library, _autograd_library
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
    _library.impl("_foreach_add_.List", _foreach_add_list)
    _library.impl("clone", _clone)
    _library.impl("arange", _arange)
    _library.impl("embedding", _embedding)
    _library.impl("layer_norm", _inference_only(_layer_norm, "layer_norm"))
    _library.impl("contiguous", _inference_only(_contiguous, "contiguous"))
    _library.impl("mm", _mm)
    _library.impl("addmm", _addmm)
    _library.impl("bmm", _bmm)
    _library.impl("_softmax", _softmax)
    _library.impl("_log_softmax", _log_softmax)
    _library.impl("_softmax_backward_data", _softmax_backward)
    _library.impl("_log_softmax_backward_data", _log_softmax_backward)
    _library.impl("_safe_softmax", _safe_softmax)
    _library.impl("embedding_dense_backward", _embedding_dense_backward)
    _library.impl("nll_loss_forward", _nll_loss_forward)
    _library.impl("nll_loss_backward", _nll_loss_backward)
    _library.impl("native_layer_norm", _native_layer_norm)
    _library.impl("native_layer_norm_backward", _native_layer_norm_backward)
    _library.impl("gelu_backward", _gelu_backward)
    _library.impl("sub.Tensor", _sub)
    _library.impl("rsub.Scalar", _rsub)
    _library.impl("fill_.Scalar", _fill)
    _library.impl("zero_", _zero)
    _library.impl(
        "scaled_dot_product_attention",
        _inference_only(_cpu_fallback(
            torch.ops.aten.scaled_dot_product_attention.default,
            "scaled_dot_product_attention",
        ), "scaled_dot_product_attention"),
    )
    _library.impl("gelu", _gelu)
    _library.impl("mul.Tensor", _mul)
    _library.impl("div.Tensor", _div)
    _library.impl("sum.dim_IntList", _sum)
    _library.impl("clamp_min", _cpu_fallback(torch.ops.aten.clamp_min.default, "clamp_min"))
    _library.impl("cat", _cpu_fallback(torch.ops.aten.cat.default, "cat"))
    _library.impl("relu", _relu)
    torch.backends.mha.set_fastpath_enabled(False)
    # cuDNN in the CUDA 13 wheels has no Maxwell support and fails with CUDNN_STATUS_EXECUTION_FAILED_CUDART;
    # without it, convolution and batch norm use PyTorch's own kernels, which fall back to the CPU when missing.
    torch.backends.cudnn.enabled = False
    # Training through the composite overrides goes through their decompositions (with real backward formulas).
    _autograd_library = torch.library.Library("aten", "IMPL", "AutogradCUDA")
    for name in sorted(_INFERENCE_ONLY):
        _autograd_library.impl(name, _composite_autograd(getattr(torch.ops.aten, name).default))
    # Attention decomposes by device: for CUDA tensors PyTorch would pick fused kernels that exist on neither
    # Maxwell nor the CPU fallback, so only the math backend (bmm, softmax) is left on.
    for backend in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        getattr(torch.backends.cuda, backend)(False)
    torch.backends.cuda.enable_math_sdp(True)
    _fallback_mode = _MaxwellInferenceFallback()
    _fallback_mode.__enter__()


def last_path() -> str:
    return getattr(_dispatch_state, "last_path", "not-run")


def cached_context_count() -> int:
    """Return the number of CUDA contexts with the adapter PTX module loaded."""
    if _bridge is None:
        return 0
    return int(_bridge.cuda_compat_cached_context_count())
