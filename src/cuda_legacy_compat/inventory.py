"""Bounded, read-only discovery of NVIDIA devices and local CUDA compilers."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

MAX_COMMAND_OUTPUT = 16_384
MAX_GPU_COUNT = 64
COMMAND_TIMEOUT_SECONDS = 5


def _tool_path(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if name != "nvcc":
        return None
    roots = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        roots.append(Path(cuda_home))
    roots.extend((Path("/usr/local/cuda"), Path("/opt/cuda")))
    for root in roots:
        candidate = root / "bin" / "nvcc"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run(arguments: list[str]) -> tuple[str | None, str | None, int | None]:
    try:
        result = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout", None
    except OSError:
        return None, "execution_failed", None
    if len(result.stdout.encode("utf-8", errors="replace")) > MAX_COMMAND_OUTPUT:
        return None, "output_limit_exceeded", result.returncode
    if result.returncode != 0:
        return None, "command_failed", result.returncode
    return result.stdout, None, result.returncode


def _parse_compute_capability(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_gpu_inventory(output: str) -> list[dict[str, Any]]:
    rows = list(csv.reader(StringIO(output), skipinitialspace=True))
    if len(rows) > MAX_GPU_COUNT:
        raise ValueError("GPU inventory exceeds the device limit")
    devices = []
    for row in rows:
        if len(row) != 5:
            raise ValueError("GPU inventory row has an unexpected number of fields")
        index, name, capability, driver, memory = (field.strip() for field in row)
        parsed_capability = _parse_compute_capability(capability)
        memory_match = re.fullmatch(r"\s*(\d+)(?:\s*MiB)?\s*", memory)
        if parsed_capability is None or not index.isdecimal() or memory_match is None:
            raise ValueError("GPU inventory contains an invalid field")
        devices.append(
            {
                "index": int(index),
                "name": name,
                "compute_capability": f"{parsed_capability[0]}.{parsed_capability[1]}",
                "compute_capability_parts": list(parsed_capability),
                "driver_version": driver,
                "memory_total_mib": int(memory_match.group(1)),
            }
        )
    return devices


def _inspect_gpus() -> dict[str, Any]:
    executable = _tool_path("nvidia-smi")
    if executable is None:
        return {"status": "not_found", "devices": []}
    output, error, code = _run(
        [
            executable,
            "--query-gpu=index,name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if error:
        return {"status": error, "exit_code": code, "devices": []}
    try:
        devices = parse_gpu_inventory(output or "")
    except ValueError as exc:
        return {"status": "invalid_output", "reason": str(exc), "devices": []}
    if not devices:
        return {"status": "no_devices", "devices": []}
    return {"status": "available", "devices": devices}


def _inspect_nvcc() -> dict[str, Any]:
    executable = _tool_path("nvcc")
    if executable is None:
        return {"status": "not_found", "version": None, "supports_sm_52": None}
    version_output, error, code = _run([executable, "--version"])
    if error:
        return {"status": error, "exit_code": code, "version": None, "supports_sm_52": None}
    match = re.search(r"\brelease\s+(\d+)\.(\d+)", version_output or "")
    if match is None:
        return {"status": "unrecognized_version", "version": None, "supports_sm_52": None}
    arch_output, arch_error, _ = _run([executable, "--list-gpu-arch"])
    supports_sm_52 = None
    if arch_error is None:
        supports_sm_52 = bool(re.search(r"\bcompute_52\b", arch_output or ""))
    return {
        "status": "available",
        "version": f"{match.group(1)}.{match.group(2)}",
        "supports_sm_52": supports_sm_52,
    }


def inspect_system() -> dict[str, Any]:
    """Return a deterministic inventory; omit timestamps and host-specific identifiers."""
    return {
        "schema_version": 1,
        "gpu_inventory": _inspect_gpus(),
        "nvcc": _inspect_nvcc(),
    }
