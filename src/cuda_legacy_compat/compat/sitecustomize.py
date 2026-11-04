"""Install the opt-in PyTorch dispatcher adapter during interpreter startup."""

import os
import sys
import warnings

STARTUP_FAILED = 78


def _why(exc: BaseException) -> str:
    """The startup failure in the user's terms."""
    reason = str(exc) or type(exc).__name__
    if reason == "CUDA is unavailable":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
            return "no GPU is visible to this process (CUDA_VISIBLE_DEVICES is set but empty)"
        return "PyTorch cannot reach a CUDA GPU from this process"
    if reason.startswith("adapter requires only sm_52 devices"):
        return f"{reason}; select only sm_52 cards with CUDA_VISIBLE_DEVICES"
    return reason


if os.environ.get("CUDA_COMPAT_ENABLE") == "1":
    verbose = os.environ.get("CUDA_COMPAT_VERBOSE") == "1"
    # The stock wheel warns that sm_52 is unsupported and suggests reinstalling PyTorch. Under the adapter that
    # advice is wrong for the supported operations, so it is replaced with one accurate line (unless verbose).
    if not verbose:
        for message in (
            r"Found GPU\d+ .* which is of compute capability",
            r"(?s).*with CUDA capability sm_52 is not compatible",
        ):
            warnings.filterwarnings("ignore", message=message, category=UserWarning)
    try:
        from cuda_legacy_compat.torch_adapter import install
        from cuda_legacy_compat.unrouted import install_excepthook

        install()
        install_excepthook()
    except BaseException as exc:
        # SystemExit raised while site imports is a fatal interpreter error with two tracebacks; leave quietly.
        print(f"cuda-legacy: the adapter could not start: {_why(exc)}. Run `cuda-legacy doctor` to see what is missing.",
              file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.stderr.flush()
        os._exit(STARTUP_FAILED)
    if not verbose:
        print("cuda-legacy: adapter active for sm_52. Supported operations run on the GPU or the CPU fallback; "
              "others fail as before. Set CUDA_COMPAT_VERBOSE=1 to see PyTorch's own warnings.", file=sys.stderr)
