"""Build and launch the bounded PyTorch CUDA compatibility adapter."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


class LaunchError(RuntimeError):
    """A required compiler, toolkit component, or build step is unavailable."""


_RESOURCE_DIR = Path(__file__).with_name("compat")
_LLVM_VERSION = "23.1.4"


def _run_build(command: Sequence[str], *, timeout: int = 120) -> None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError(f"build command failed to run: {command[0]}: {exc}") from exc
    if result.returncode:
        details = (result.stderr or result.stdout).strip()[-4000:]
        rendered = " ".join(command)
        raise LaunchError(
            f"build command failed ({result.returncode}): {rendered}\n{details}"
        )


def _compiler_include_paths() -> tuple[Path, Path]:
    compiler = shutil.which("g++")
    if compiler is None:
        raise LaunchError("g++ is required to build the CUDA driver bridge")
    try:
        version_result = subprocess.run(
            [compiler, "-dumpversion"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchError(f"could not query g++ version: {exc}") from exc
    version = version_result.stdout.strip()
    include = Path("/usr/include/c++") / version
    arch_include = Path("/usr/include/x86_64-linux-gnu/c++") / version
    if not (include / "climits").is_file() or not arch_include.is_dir():
        raise LaunchError(f"C++ standard library headers not found for g++ {version}")
    return include, arch_include


def build_adapter(llvm_root: Path, cuda_root: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise LaunchError("the PyTorch Maxwell adapter currently supports Linux x86_64 only")
    clangxx = llvm_root / "bin" / "clang++"
    if not clangxx.is_file() or not os.access(clangxx, os.X_OK):
        raise LaunchError(f"LLVM clang++ is unavailable: {clangxx}")
    try:
        version_result = subprocess.run(
            [str(clangxx), "--version"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchError(f"could not query LLVM version: {exc}") from exc
    if _LLVM_VERSION not in version_result.stdout:
        found_version = version_result.stdout.splitlines()[0]
        raise LaunchError(f"LLVM {_LLVM_VERSION} is required; found {found_version}")
    for header in (cuda_root / "include" / "cuda.h", cuda_root / "include" / "cuda_runtime.h"):
        if not header.is_file():
            raise LaunchError(f"CUDA toolkit header is unavailable: {header}")
    bridge_source = (_RESOURCE_DIR / "driver_add_bridge.cpp").read_text()
    if re.search(r"cu(Stream|Ctx)Synchronize", bridge_source):
        raise LaunchError("the driver bridge must preserve asynchronous stream execution")
    cxx_include, arch_include = _compiler_include_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    ptx = output_dir / "compat.ptx"
    bridge = output_dir / "libcuda_compat_bridge.so"
    facade = output_dir / "libcuda_runtime_version_facade.so"
    _run_build([
        str(clangxx), "-x", "cuda", f"--cuda-path={cuda_root}", "--cuda-gpu-arch=sm_52",
        "--cuda-device-only", "-S", "-O2", "-std=c++17", "-isystem", str(cxx_include),
        "-isystem", str(arch_include), "-o", str(ptx), str(_RESOURCE_DIR / "compat_kernels.cu"),
    ])
    ptx_text = ptx.read_text(errors="replace")
    if not re.search(r"^\.target\s+sm_52\b", ptx_text, re.MULTILINE):
        raise LaunchError("compiler output is not PTX targeting sm_52")
    _run_build([
        "g++", "-std=c++17", "-O2", "-pthread", "-fPIC", "-shared",
        f"-I{cuda_root / 'include'}", str(_RESOURCE_DIR / "driver_add_bridge.cpp"),
        "-lcuda", "-o", str(bridge),
    ])
    _run_build([
        "g++", "-std=c++17", "-O2", "-fPIC", "-shared",
        f"-I{cuda_root / 'include'}", str(_RESOURCE_DIR / "cuda_runtime_version_facade.cpp"),
        "-ldl", "-o", str(facade),
    ])
    return ptx, bridge, facade


def launch(
    script: Path,
    script_args: Sequence[str],
    *,
    llvm_root: Path,
    cuda_root: Path,
    environ: Mapping[str, str] | None = None,
) -> int:
    script = script.resolve()
    if not script.is_file():
        raise LaunchError(f"Python script not found: {script}")
    env = dict(os.environ if environ is None else environ)
    with tempfile.TemporaryDirectory(prefix="cuda-legacy-compat-") as temporary:
        ptx, bridge, facade = build_adapter(llvm_root, cuda_root, Path(temporary))
        alternate_ptx = Path(temporary) / "compat-alternate.ptx"
        shutil.copy2(ptx, alternate_ptx)
        env.update({
            "CUDA_COMPAT_BRIDGE": str(bridge),
            "CUDA_COMPAT_VECTOR_ADD_PTX": str(ptx),
            "CUDA_COMPAT_ALTERNATE_PTX": str(alternate_ptx),
            "CUDA_COMPAT_RUNTIME_VERSION": "13000",
            "CUDA_COMPAT_ENABLE": "1",
        })
        env["LD_PRELOAD"] = ":".join(filter(None, [str(facade), env.get("LD_PRELOAD", "")]))
        env["PYTHONPATH"] = ":".join(filter(None, [str(_RESOURCE_DIR), env.get("PYTHONPATH", "")]))
        try:
            return subprocess.run(
                [sys.executable, str(script), *script_args], env=env, check=False
            ).returncode
        except OSError as exc:
            raise LaunchError(f"could not start Python: {exc}") from exc
