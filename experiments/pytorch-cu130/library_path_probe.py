"""Record CUDA libraries mapped by a bounded PyTorch CUDA 13 workload."""

from __future__ import annotations

import re

import torch
import torch.nn.functional as functional

from cuda_legacy_compat import torch_adapter


def main() -> int:
    torch_adapter.install()
    device = torch.device("cuda:0")
    left = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(device)
    right = torch.eye(4, dtype=torch.float32).to(device)
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8).to(device)
    weight = torch.ones((4, 3, 3, 3), dtype=torch.float32)
    with torch.inference_mode():
        product = torch.mm(left, right)
        matrix_path = torch_adapter.last_path()
        convolution = functional.conv2d(image, weight.to(device), padding=1)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(product.cpu(), torch.mm(left.cpu(), right.cpu()))
    torch.testing.assert_close(
        convolution.cpu(),
        functional.conv2d(image.cpu(), weight, padding=1),
        rtol=1e-5,
        atol=1e-6,
    )
    print(f"PASS mm_path={matrix_path} conv_path={torch_adapter.last_path()}")

    library_pattern = re.compile(
        r"/(?:libcuda|libcudart|libcublas|libcudnn|libnccl|libnvrtc|libtorch_cuda)[^/]*\.so(?:\.[^/]*)?$"
    )
    mapped: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            fields = line.rstrip().split(maxsplit=5)
            if len(fields) == 6 and library_pattern.search(fields[5]):
                mapped.add(fields[5])
    if not mapped:
        raise RuntimeError("no CUDA wheel/driver libraries appeared in /proc/self/maps")
    for path in sorted(mapped):
        print(f"MAPPED_LIBRARY {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
