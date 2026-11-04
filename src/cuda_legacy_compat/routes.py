"""What the adapter routes, without importing PyTorch: the source for `cuda-legacy ops` and the support matrix.

tests/test_routes.py checks that every `_library.impl(...)` in torch_adapter.py appears here and nothing else does.
"""

from __future__ import annotations

import textwrap
from typing import Any

CPU_FALLBACK_LIMIT = "at most 64 MiB of CUDA inputs and 64 MiB of output per operation, one CUDA device per operation"

# (aten registrations, operation, supported case)
NATIVE = (
    (("add.Tensor",), "Tensor addition",
     "Same-device, same-shape contiguous float32 (up to 1,048,576 values) or int32 (up to 2,147,483,647 elements), "
     "alpha=1. float16 up to 1,048,576 values, alpha=1. Other cases use the CPU fallback."),
    (("add_.Tensor",), "Float32 in-place addition",
     "Same-shape contiguous float32 CUDA tensors up to 1,048,576 values; scaled updates as in the verified SGD case."),
    (("arange",), "Range creation", "CUDA int64 arange, up to 1,048,576 values. Other cases use the CPU fallback."),
    (("embedding",), "Embedding",
     "Contiguous float32 weight, int64 CUDA indices in 1D or 2D, up to 1,048,576 output values. "
     "Other cases, and training, use the CPU fallback."),
    (("layer_norm",), "Layer normalization",
     "Contiguous float32 input normalized over the last dimension, float32 weight and bias, up to 1,048,576 "
     "values. Other cases use the CPU fallback; training uses PyTorch's decomposition."),
    (("mm",), "Matrix multiplication",
     "2D float32 with nonnegative strides; each dimension at most 256 and at most 16,777,216 multiply-accumulates. "
     "Other dtypes and larger inputs use the CPU fallback."),
    (("relu",), "ReLU", "Contiguous float32 tensors up to 1,048,576 values. Other cases, and training, use the CPU fallback."),
)

# (aten registration, operation)
CPU_FALLBACK = (
    ("linear", "linear"), ("bmm", "batched matrix multiplication"), ("_softmax", "softmax"),
    ("scaled_dot_product_attention", "scaled dot-product attention"), ("gelu", "GELU"),
    ("mul.Tensor", "multiplication"), ("div.Tensor", "division"), ("sum.dim_IntList", "sum over dimensions"),
    ("clamp_min", "clamp_min"), ("cat", "concatenation"), ("contiguous", "contiguous copy"),
)

NOTES = (
    "Everything else runs as the stock wheel does, and on sm_52 that usually means 'no kernel image is available'.",
    "Creating a tensor directly on the GPU (torch.zeros, ones, full, rand, randn, eye, ... with device='cuda') "
    "builds it on the CPU and copies it over, within the 64 MiB output limit. Random values come from the CPU "
    "generator: torch.manual_seed keeps them reproducible, but they differ from a GPU that has kernels.",
    "Training is verified against the CPU for small Linear/GELU MLPs with SGD, momentum SGD, Adam and AdamW "
    "(their default foreach updates included), for a Conv2d/BatchNorm2d/MaxPool CNN with momentum SGD, and for "
    "LSTM, GRU and RNN layers (the fused CUDA cells run on the CPU). "
    "cuDNN is switched off under the adapter: it has no Maxwell support in the CUDA 13 wheels.",
    "Training through layer_norm, scaled_dot_product_attention and contiguous uses PyTorch's own "
    "decomposition, so Transformers, attention and LayerNorm models train; attention uses the math backend. "
    "Inference keeps the adapter's kernels for them.",
    "In-place operators (x += 1, mul_, zero_, fill_, x[i] = v, relu_, nn.ReLU(inplace=True), and the "
    "GradScaler steps of torch.amp) run on a CPU copy and are copied back, within the 64 MiB limit; a strided "
    "view such as x[:, 1:3] is updated through its contiguous base. Other mutating or alias-returning "
    "operators fail closed.",
)


def registered() -> set[str]:
    return {name for names, _, _ in NATIVE for name in names} | {name for name, _ in CPU_FALLBACK}


def as_json() -> dict[str, Any]:
    return {
        "native": [{"aten": list(names), "operation": operation, "supported": case} for names, operation, case in NATIVE],
        "cpu_fallback": {"limit": CPU_FALLBACK_LIMIT,
                         "operations": [{"aten": name, "operation": operation} for name, operation in CPU_FALLBACK]},
        "notes": list(NOTES),
        "details": "https://cudacompat.com/docs/compatibility.html",
    }


def render() -> str:
    width = max(len(operation) for operation in [*(n[1] for n in NATIVE), *(f[1] for f in CPU_FALLBACK)]) + 2
    lines = ["Runs natively on sm_52:"]
    for names, operation, case in NATIVE:
        lines.append(f"  {operation:<{width}}{', '.join(names)}")
        lines.extend(textwrap.wrap(case, width=100, initial_indent=" " * (width + 2),
                                   subsequent_indent=" " * (width + 2)))
    lines.append("")
    lines.append(f"Runs on the CPU and copies back ({len(CPU_FALLBACK)} operations):")
    lines.extend(textwrap.wrap(f"Limit: {CPU_FALLBACK_LIMIT}.", width=100, initial_indent="  ", subsequent_indent="  "))
    for name, operation in CPU_FALLBACK:
        lines.append(f"  {operation:<{width}}{name}")
    lines.append("")
    for note in NOTES:
        lines.extend(textwrap.wrap(note, width=100, initial_indent="Note: ", subsequent_indent="      "))
    lines.append("Details and evidence: https://cudacompat.com/docs/compatibility.html")
    return "\n".join(lines) + "\n"
