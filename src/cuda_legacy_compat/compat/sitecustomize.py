"""Install the opt-in PyTorch dispatcher adapter during interpreter startup."""

import os
import sys
import warnings

if os.environ.get("CUDA_COMPAT_ENABLE") == "1":
    # The stock wheel warns that sm_52 is unsupported and suggests reinstalling PyTorch. Under the adapter that
    # advice is wrong for the supported operations, so it is replaced with one accurate line (unless verbose).
    if os.environ.get("CUDA_COMPAT_VERBOSE") != "1":
        for message in (
            r"Found GPU\d+ .* which is of compute capability",
            r"(?s).*with CUDA capability sm_52 is not compatible",
        ):
            warnings.filterwarnings("ignore", message=message, category=UserWarning)
        print("cuda-legacy: adapter active for sm_52. Supported operations run on the GPU or the CPU fallback; "
              "others fail as before. Set CUDA_COMPAT_VERBOSE=1 to see PyTorch's own warnings.", file=sys.stderr)
    try:
        from cuda_legacy_compat.torch_adapter import install

        install()
    except BaseException as exc:
        print(f"cuda-legacy-compat: adapter startup failed: {exc}", file=sys.stderr)
        raise SystemExit(78) from exc
