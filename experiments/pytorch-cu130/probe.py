"""Probe the official PyTorch CUDA 13 wheel on visible NVIDIA devices."""

from __future__ import annotations

import sys
import traceback

import torch


def main() -> int:
    print(f"torch={torch.__version__} built_cuda={torch.version.cuda}", flush=True)
    try:
        stage = "device discovery"
        device_count = torch.cuda.device_count()
        print(f"cuda_available={torch.cuda.is_available()} device_count={device_count}", flush=True)
        if device_count == 0:
            return 2
        for index in range(device_count):
            stage = "device property query"
            properties = torch.cuda.get_device_properties(index)
            print(
                f"device={index} name={properties.name} "
                f"compute={properties.major}.{properties.minor} "
                f"memory_bytes={properties.total_memory}",
                flush=True,
            )
        for index in range(device_count):
            stage = "host-to-device copy"
            host_values = torch.arange(1024, dtype=torch.int64)
            values = host_values.to(f"cuda:{index}")
            torch.cuda.synchronize(index)
            print(f"PASS device={index} host_to_device elements=1024", flush=True)
            stage = "CUDA tensor addition"
            result = (values + 1).cpu()
            expected = torch.arange(1024, dtype=torch.int64) + 1
            torch.testing.assert_close(result, expected)
            print(f"PASS device={index} elements=1024 result={result[-1].item()}", flush=True)
    except Exception as error:
        print(f"FAIL stage={stage}: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
