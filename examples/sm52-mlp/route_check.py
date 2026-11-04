"""Public example: a small float32 MLP on an sm_52 GPU, with each operation's execution route.

Run it twice on the same machine:

    python route_check.py                     # stock PyTorch CUDA 13 wheel: expect the matrix multiply to fail
    cuda-legacy run --llvm-root ... -- route_check.py   # through the adapter: expect every check to pass

It prints one JSON document and exits 0 only when the observed outcome matches the expected one for its mode.
The workload is synthetic and deterministic. It measures correctness and routes, not performance.
"""
from __future__ import annotations

import json
import platform
import sys

import torch

UNIT_ROUNDOFF = 2.0 ** -24


def _adapter():
    return sys.modules.get("cuda_legacy_compat.torch_adapter")


def _route() -> str:
    adapter = _adapter()
    return adapter.last_path() if adapter else "stock-pytorch"


def _environment(device: torch.device) -> dict:
    major, minor = torch.cuda.get_device_capability(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": f"{major}.{minor}",
        "adapter_loaded": _adapter() is not None,
    }


def _case(name: str, gpu: torch.Tensor, cpu: torch.Tensor, route: str, bound: torch.Tensor | None = None) -> dict:
    """Compare one GPU result with the CPU result of the same operation on the same inputs.

    With no bound the results must be identical. A bound is an elementwise absolute error limit.
    """
    error = (gpu.cpu() - cpu).abs()
    limit = torch.zeros_like(error) if bound is None else bound
    return {
        "case": name,
        "route": route,
        "shape": list(error.shape),
        "criterion": "identical" if bound is None else "float32 dot-product bound 2*n*u*(|A|@|B|)",
        "matches_cpu": bool((error <= limit).all()),
        "max_abs_error": float(error.max()) if error.numel() else 0.0,
    }


def _mm_case(name: str, a: torch.Tensor, b: torch.Tensor) -> tuple[dict, torch.Tensor]:
    out = torch.mm(a, b)
    route = _route()
    ca, cb = a.cpu(), b.cpu()
    # Standard forward error bound for n-term float32 dot products (gamma_n ~ n*u), doubled because
    # the CPU reference is itself rounded. Fixed before measurement; not fitted to observed errors.
    bound = 2 * ca.shape[1] * UNIT_ROUNDOFF * (ca.abs() @ cb.abs())
    return _case(name, out, ca @ cb, route, bound), out


def adapted(device: torch.device) -> list[dict]:
    generator = torch.Generator().manual_seed(52)
    x = torch.randn(8, 16, generator=generator).to(device)
    w1 = torch.randn(16, 32, generator=generator).to(device)
    b1 = torch.randn(8, 32, generator=generator).to(device)
    w2 = torch.randn(32, 8, generator=generator).to(device)

    cases = []
    case, hidden = _mm_case("mm 8x16 @ 16x32", x, w1)
    cases.append(case)
    biased = torch.add(hidden, b1)
    cases.append(_case("add bias 8x32", biased, hidden.cpu() + b1.cpu(), _route()))
    activated = torch.relu(biased)
    cases.append(_case("relu 8x32", activated, torch.relu(biased.cpu()), _route()))
    case, output = _mm_case("mm 8x32 @ 32x8", activated, w2)
    cases.append(case)
    probabilities = torch.softmax(output, dim=1)
    cases.append(_case("softmax 8x8 (CPU fallback)", probabilities, torch.softmax(output.cpu(), dim=1), _route()))

    # Past the native kernel's 256-dimension limit, matrix multiplication uses the CPU fallback.
    case, _ = _mm_case("mm 1x257 @ 257x1 (CPU fallback)", torch.ones(1, 257).to(device), torch.ones(257, 1).to(device))
    cases.append(case)
    # Constructors that fill memory on the GPU have no Maxwell image; the adapter builds them on the CPU and copies.
    ones = torch.ones(2, 2, device=device)
    cases.append(_case("torch.ones on the GPU (CPU fallback)", ones, torch.ones(2, 2), _route()))
    return cases


def stock(device: torch.device) -> list[dict]:
    # Inputs are built on the CPU and copied, because the stock wheel cannot even fill memory on sm_52.
    a = torch.ones(8, 16).to(device)
    try:
        torch.mm(a, torch.ones(16, 32).to(device))
        torch.cuda.synchronize(device)
        return [{"case": "stock mm 8x16 @ 16x32", "fails": False}]
    except RuntimeError as error:
        return [{"case": "stock mm 8x16 @ 16x32", "fails": True, "error": str(error).splitlines()[0]}]


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA is unavailable"}))
        return 2
    device = torch.device("cuda", 0)
    mode = "adapter" if _adapter() else "stock"
    with torch.inference_mode():
        cases = adapted(device) if mode == "adapter" else stock(device)
    if mode == "adapter":
        passed = all(c.get("matches_cpu", c.get("fails_closed")) for c in cases)
    else:
        passed = cases[0]["fails"]
    print(json.dumps({"mode": mode, "environment": _environment(device),
                      "cases": cases, "expected_outcome_observed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
