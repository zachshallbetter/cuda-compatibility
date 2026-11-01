"""Record stock CUDA 13 wheel behavior for selected built-in operators."""

from __future__ import annotations

import os

import torch


def main() -> int:
    operation = os.environ.get("CUDA_COMPAT_BASELINE_OPERATOR", "")
    if operation not in {"mm", "relu"}:
        raise SystemExit("CUDA_COMPAT_BASELINE_OPERATOR must be mm or relu")

    print(f"torch={torch.__version__} built_cuda={torch.version.cuda} operator={operation}", flush=True)
    if not torch.cuda.is_available():
        print("FAIL stage=device_discovery cuda_available=false", flush=True)
        return 2
    properties = torch.cuda.get_device_properties(0)
    print(
        f"device=0 name={properties.name} compute={properties.major}.{properties.minor}",
        flush=True,
    )

    try:
        if operation == "mm":
            left = torch.arange(16, dtype=torch.float32).reshape(4, 4).to("cuda")
            right = torch.eye(4, dtype=torch.float32).to("cuda")
            result = torch.mm(left, right).cpu()
            expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        else:
            values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).to("cuda")
            result = torch.relu(values).cpu()
            expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
    except Exception as error:
        print(f"OBSERVED_FAILURE operator={operation} error={type(error).__name__}: {error}", flush=True)
        return 0

    print(f"PASS stock_operator={operation} result_elements={result.numel()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
