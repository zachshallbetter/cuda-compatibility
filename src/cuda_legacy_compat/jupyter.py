"""A Jupyter kernel that runs through the adapter (#225).

The kernel spec's command is this environment's `cuda-legacy run -m ipykernel_launcher`, so a notebook gets the same
per-process adapter as `run`, and Jupyter's interrupt reaches the kernel as Ctrl-C does (#217). The spec carries the
setup marker, so only a spec this command wrote is ever replaced or removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from cuda_legacy_compat.toolchain import MANAGED_MARKER

KERNEL_NAME = "cuda-legacy"
DISPLAY_NAME = "Python (cuda-legacy, sm_52)"


class JupyterError(RuntimeError):
    """The kernel could not be registered; the message says what to do next."""


def kernels_dir() -> Path:
    """Jupyter's per-user kernels directory, as jupyter_core finds it on Linux."""
    if os.environ.get("JUPYTER_DATA_DIR"):
        return Path(os.environ["JUPYTER_DATA_DIR"]) / "kernels"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "jupyter" / "kernels"


def spec_dir() -> Path:
    return kernels_dir() / KERNEL_NAME


def launcher() -> str:
    """This environment's cuda-legacy command: the kernel starts the same one that registered it."""
    candidate = Path(sysconfig.get_path("scripts")) / "cuda-legacy"
    return str(candidate) if candidate.is_file() else shutil.which("cuda-legacy") or str(candidate)


def has_ipykernel(python: str) -> bool:
    try:
        return subprocess.run([python, "-c", "import ipykernel"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install(python: str) -> Path:
    """Write the kernel spec, replacing only one this command wrote."""
    if not has_ipykernel(python):
        raise JupyterError(f"ipykernel is not installed in the PyTorch environment ({python}). "
                           f"Install it, then run this again: {python} -m pip install ipykernel")
    target = spec_dir()
    if target.exists() and not (target / MANAGED_MARKER).is_file():
        raise JupyterError(f"{target} exists and cuda-legacy did not create it; move it aside to register this kernel")
    target.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [launcher(), "run", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": DISPLAY_NAME,
        "language": "python",
        "interrupt_mode": "signal",
        "metadata": {"debugger": False},
    }
    (target / "kernel.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    (target / MANAGED_MARKER).write_text("Created by cuda-legacy jupyter; `cuda-legacy jupyter --remove` deletes it.\n")
    return target


def remove() -> Path | None:
    """Delete the kernel spec if this command wrote it; None when there was none of ours."""
    target = spec_dir()
    if (target / MANAGED_MARKER).is_file():
        shutil.rmtree(target)
        return target
    return None
