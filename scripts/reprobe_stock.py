"""Re-run the two contradicted stock-wheel probes (support matrix D1 and D2) without the adapter.

Usage: PYTHON scripts/reprobe_stock.py [--device N]
Run it with the Python of an environment that holds the PyTorch CUDA wheel to test, on a machine whose visible
GPU is sm_52, and not under `cuda-legacy run`: the point is what the stock wheel does on its own. It prints one
JSON object. A probe that now succeeds means its contradicted claim should be reopened (see
.agents/records/negative-results.jsonl and docs/SITE_PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CLAIMS = {
    "mm": "CLC-009-stock-mm-cublas-arch-mismatch",
    "relu": "CLC-009-stock-relu-no-kernel-image",
}


def verdict(error: str | None) -> str:
    return "reopen: the stock wheel ran it" if error is None else "still contradicted"


def adapter_active() -> bool:
    # `cuda-legacy run` sets CUDA_COMPAT_ENABLE=1 for the child and its startup hook imports the package.
    return os.environ.get("CUDA_COMPAT_ENABLE") == "1" or "cuda_legacy_compat" in sys.modules


def probe(device: int) -> dict:
    import torch

    torch.cuda.set_device(device)
    report = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": torch.cuda.get_device_name(device),
              "capability": "sm_%d%d" % torch.cuda.get_device_capability(device), "probes": {}}
    cases = {
        "mm": lambda: torch.mm(torch.ones(4, 4, device="cpu").cuda(), torch.ones(4, 4, device="cpu").cuda()),
        "relu": lambda: torch.relu(torch.tensor([-1.0, 2.0]).cuda()),
    }
    for name, run in cases.items():
        error = None
        try:
            run()
            torch.cuda.synchronize(device)
        except Exception as exc:  # the failure text is the observation
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        report["probes"][name] = {"claim": CLAIMS[name], "error": error, "verdict": verdict(error)}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    if adapter_active():
        print("reprobe_stock: the cuda-legacy adapter is active; run this with the environment's python directly",
              file=sys.stderr)
        return 2
    print(json.dumps(probe(args.device), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
