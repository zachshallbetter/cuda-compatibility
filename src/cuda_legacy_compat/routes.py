"""What the adapter routes, without importing PyTorch: the source for `cuda-legacy ops` and the support matrix.

tests/test_routes.py checks that every `_library.impl(...)` in torch_adapter.py appears here and nothing else does.
"""

from __future__ import annotations

import textwrap
from typing import Any

CPU_FALLBACK_LIMIT = ("each copy between the GPU and host memory at most 64 MiB (larger tensors move in pieces), at "
                      "most 1 GiB staged per operation (CUDA_COMPAT_FALLBACK_TOTAL_MIB), one CUDA device per "
                      "operation; an embedding over 64 MiB moves only the rows it looks up")

# (aten registrations, operation, supported case)
NATIVE = (
    (("add.Tensor",), "Tensor addition",
     "float32 of any size with broadcasting (up to six dimensions), any alpha, or a number; int32 and float16 "
     "same-shape contiguous cases as before. Other dtypes use the CPU fallback."),
    (("sub.Tensor", "rsub.Scalar", "mul.Tensor", "div.Tensor"), "Subtraction, multiplication and division",
     "float32 of any size with broadcasting (up to six dimensions), or a number. Other dtypes use the CPU fallback."),
    (("gelu", "gelu_backward"), "GELU", "Contiguous float32, exact or tanh, and its gradient."),
    (("_softmax", "_safe_softmax", "_log_softmax", "_softmax_backward_data", "_log_softmax_backward_data"),
     "Softmax and log-softmax",
     "Contiguous float32 over the last dimension, masked (-inf) entries included, and their gradients; safe softmax "
     "(attention with a mask) gives zeros for a fully masked row."),
    (("native_layer_norm", "native_layer_norm_backward"), "Layer normalization in training",
     "Contiguous float32 over the trailing dimensions, with or without weight and bias, and its gradients."),
    (("sum.dim_IntList", "mean.dim"), "Sum and mean over dimensions",
     "float32, any dimensions (up to six), keepdim or not."),
    (("fill_.Scalar", "zero_"), "Fill", "Contiguous float32 tensors. Other dtypes are filled in place on the CPU."),
    (("add_.Tensor", "_foreach_add_.List"), "Float32 in-place addition",
     "A contiguous float32 tensor of any size updated by a broadcast float32 tensor or a number, with any alpha; "
     "SGD's foreach step and gradient accumulation. Other cases are computed on the CPU and written back."),
    (("arange",), "Range creation", "CUDA int64 arange, up to 1,048,576 values. Other cases use the CPU fallback."),
    (("embedding", "embedding_dense_backward"), "Embedding",
     "Contiguous float32 weight of any size and int64 CUDA indices in 1D or 2D, in training too: the weight "
     "gradient (padding_idx included, scale_grad_by_freq on the CPU) stays on the GPU."),
    (("layer_norm",), "Layer normalization",
     "Inference: contiguous float32 normalized over the trailing dimensions, any size, with or without weight and "
     "bias. Training uses PyTorch's decomposition onto native_layer_norm."),
    (("nll_loss_forward", "nll_loss_backward"), "Negative log likelihood",
     "1D or 2D float32 log-probabilities with int64 CUDA targets: mean, sum or none, ignore_index and class weights, "
     "and its gradient; cross_entropy runs through it. A target out of range raises IndexError."),
    (("_foreach_mul_.Scalar", "_foreach_add_.Scalar", "_foreach_div_.ScalarList", "_foreach_lerp_.Scalar",
      "_foreach_addcmul_.Scalar", "_foreach_addcdiv_.ScalarList", "_foreach_sqrt", "_foreach_add.List"),
     "Optimizer steps",
     "Adam, AdamW and momentum or weight-decay SGD's default foreach updates over contiguous float32 parameters on "
     "one device. Other tensors (float16, another device) form their own group, which runs on the CPU; amsgrad, RMSprop "
     "and Adagrad's other steps use the CPU route."),
    (("max_pool2d_with_indices", "max_pool2d_with_indices_backward"), "Max pooling",
     "2-D float32, a contiguous (C, H, W) or (N, C, H, W) input, any kernel, stride, padding, dilation and ceil mode: "
     "values and indices as the CPU computes them (NaN included), and the gradient."),
    (("clone", "contiguous", "select_backward"), "Copies of strided views",
     "Any dtype of 1, 2, 4 or 8 bytes, up to six dimensions (transposes, slices, broadcasts). Training through "
     "contiguous uses PyTorch's decomposition onto clone; the gradient of x[:, i] and unbind (float32) too."),
    (("mm",), "Matrix multiplication",
     "2D float32 of any size with nonnegative strides (transposed views included), on a tiled kernel; training "
     "too. Other dtypes use the CPU fallback."),
    (("addmm",), "Matrix multiplication with bias",
     "As mm, with alpha and a bias of any broadcastable shape; linear layers run through it."),
    (("bmm",), "Batched matrix multiplication",
     "3D float32 batches (up to 65,535) of any size with nonnegative strides; attention runs through it."),
    (("relu",), "ReLU", "Contiguous float32 tensors up to 1,048,576 values. Other cases, and training, use the CPU fallback."),
)

# (aten registration, operation)
CPU_FALLBACK = (
    ("scaled_dot_product_attention", "scaled dot-product attention"),
    ("clamp_min", "clamp_min"), ("cat", "concatenation"),
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
    "Other in-place operators (mul_, fill_ and zero_ of other dtypes, x[i] = v, relu_, nn.ReLU(inplace=True), "
    "and the GradScaler steps of torch.amp) run on a CPU copy and are copied back, within the 64 MiB limit; a strided "
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
        lines.extend(textwrap.wrap(f"{operation:<{width}}{', '.join(names)}", width=98, initial_indent="  ",
                                   subsequent_indent=" " * (width + 2), break_on_hyphens=False))
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
