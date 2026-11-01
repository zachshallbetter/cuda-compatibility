"""Compare a small convolutional PyTorch workload on CPU and CUDA."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode


class _OperationTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.last_operation = "unknown"

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        schema = getattr(function, "_schema", None)
        self.last_operation = schema.name if schema is not None else str(function)
        return function(*args, **(kwargs or {}))


def main() -> int:
    torch.manual_seed(2026)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 5),
    ).eval()
    inputs = torch.randn((2, 3, 32, 32), dtype=torch.float32)
    with torch.inference_mode():
        reference = model(inputs)
    if not torch.cuda.is_available():
        print("FAIL device=cuda unavailable", flush=True)
        return 2

    properties = torch.cuda.get_device_properties(0)
    print(
        f"device={properties.name} capability={properties.major}.{properties.minor} "
        f"torch={torch.__version__} built_cuda={torch.version.cuda}",
        flush=True,
    )
    model = model.cuda().eval()
    inputs = inputs.cuda()
    trace = _OperationTrace()
    try:
        with torch.inference_mode(), trace:
            actual = model(inputs)
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"FAIL stage=cnn_forward op={trace.last_operation} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return 3

    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-5)
    print(
        f"PASS workload=conv-relu-pool-conv-adaptive-pool-linear "
        f"shape=2x3x32x32 output={tuple(actual.shape)} rtol=1e-4 atol=1e-5",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
