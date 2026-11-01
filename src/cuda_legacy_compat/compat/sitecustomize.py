"""Install the opt-in PyTorch dispatcher adapter during interpreter startup."""

import os
import sys

if os.environ.get("CUDA_COMPAT_ENABLE") == "1":
    try:
        from cuda_legacy_compat.torch_adapter import install

        install()
    except BaseException as exc:
        print(f"cuda-legacy-compat: adapter startup failed: {exc}", file=sys.stderr)
        raise SystemExit(78) from exc
