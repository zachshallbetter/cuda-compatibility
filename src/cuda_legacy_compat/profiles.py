"""Strict profile loading and explicit hardware/toolkit compatibility audits."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MAX_PROFILE_BYTES = 65_536


class ProfileError(ValueError):
    """A profile is unreadable or violates the supported profile shape."""


def load_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    try:
        data = profile_path.read_bytes()
    except OSError as exc:
        raise ProfileError(f"cannot read profile: {exc.strerror or 'I/O error'}") from exc
    if len(data) > MAX_PROFILE_BYTES:
        raise ProfileError("profile exceeds the 65536-byte limit")
    try:
        profile = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError("profile is not valid UTF-8 JSON") from exc
    if not isinstance(profile, dict):
        raise ProfileError("profile root must be an object")
    validate_profile(profile)
    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    required = {"schema_version", "id", "name", "source", "build", "gpu"}
    if not required.issubset(profile):
        raise ProfileError("profile is missing required fields")
    unexpected = set(profile) - required - {"notes"}
    if unexpected:
        raise ProfileError("profile contains unsupported fields")
    if profile["schema_version"] != 1:
        raise ProfileError("unsupported profile schema_version")
    for key in ("id", "name", "source"):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ProfileError(f"profile {key} must be a non-empty string")
    build = profile["build"]
    gpu = profile["gpu"]
    if not isinstance(build, dict) or not isinstance(gpu, dict):
        raise ProfileError("profile build and gpu fields must be objects")
    if set(build) - {"cuda_version", "host_toolkit_minimum", "artifact"}:
        raise ProfileError("profile build contains unsupported fields")
    if set(gpu) - {"minimum_compute_capability", "architectures"}:
        raise ProfileError("profile gpu contains unsupported fields")
    cuda_version = build.get("cuda_version")
    if not isinstance(cuda_version, str) or re.fullmatch(r"\d+(?:\.\d+){1,2}", cuda_version) is None:
        raise ProfileError("profile build.cuda_version must be a dotted numeric version")
    host_toolkit_minimum = build.get("host_toolkit_minimum")
    if host_toolkit_minimum is not None and (
        not isinstance(host_toolkit_minimum, str)
        or re.fullmatch(r"\d+(?:\.\d+){1,2}", host_toolkit_minimum) is None
    ):
        raise ProfileError("profile build.host_toolkit_minimum must be null or a dotted numeric version")
    minimum = gpu.get("minimum_compute_capability")
    if (
        not isinstance(minimum, list)
        or len(minimum) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) or part < 0 for part in minimum)
    ):
        raise ProfileError("profile gpu.minimum_compute_capability must be [major, minor]")


def _version_tuple(version: object) -> tuple[int, ...] | None:
    if not isinstance(version, str):
        return None
    parts = version.split(".")
    if not parts or any(not part.isdecimal() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def audit_profile(profile: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    validate_profile(profile)
    issues: list[dict[str, str]] = []
    gpu_state = inventory.get("gpu_inventory", {})
    gpu_status = gpu_state.get("status", "unknown")
    min_capability = tuple(profile["gpu"]["minimum_compute_capability"])
    devices = gpu_state.get("devices", [])
    device_results = []
    if gpu_status != "available":
        issues.append(
            {
                "code": "gpu_inventory_unavailable",
                "detail": f"GPU inventory status is {gpu_status}",
            }
        )
    else:
        for device in devices:
            capability = tuple(device["compute_capability_parts"])
            compatible = capability >= min_capability
            device_results.append(
                {
                    "index": device["index"],
                    "compute_capability": device["compute_capability"],
                    "status": "compatible" if compatible else "incompatible",
                }
            )
            if not compatible:
                issues.append(
                    {
                        "code": "gpu_compute_capability_below_minimum",
                        "detail": (
                            f"GPU {device['index']} has {device['compute_capability']}; "
                            f"profile requires {min_capability[0]}.{min_capability[1]} or newer"
                        ),
                    }
                )

    toolkit_result: dict[str, Any]
    host_toolkit_minimum = profile["build"]["host_toolkit_minimum"]
    if host_toolkit_minimum is None:
        toolkit_result = {
            "status": "not_required_for_runtime",
            "package_cuda_version": profile["build"]["cuda_version"],
            "observed_nvcc_version": inventory.get("nvcc", {}).get("version"),
        }
    else:
        nvcc = inventory.get("nvcc", {})
        requested = _version_tuple(host_toolkit_minimum)
        observed = _version_tuple(nvcc.get("version", ""))
        if nvcc.get("status") != "available" or observed is None:
            toolkit_result = {"status": "unknown", "minimum_host_toolkit": host_toolkit_minimum}
            issues.append({"code": "required_host_toolkit_unavailable", "detail": "nvcc version is unavailable"})
        elif requested is None:
            raise ProfileError("profile build.host_toolkit_minimum must be numeric dotted version")
        elif (observed + (0, 0, 0))[:3] < (requested + (0, 0, 0))[:3]:
            toolkit_result = {
                "status": "incompatible",
                "minimum_host_toolkit": host_toolkit_minimum,
                "observed_nvcc_version": nvcc["version"],
            }
            issues.append(
                {
                    "code": "host_toolkit_below_requirement",
                    "detail": f"nvcc {nvcc['version']} is older than CUDA {host_toolkit_minimum}",
                }
            )
        else:
            toolkit_result = {
                "status": "compatible",
                "minimum_host_toolkit": host_toolkit_minimum,
                "observed_nvcc_version": nvcc["version"],
            }

    incompatible_codes = {
        "gpu_compute_capability_below_minimum",
        "host_toolkit_below_requirement",
    }
    if any(issue["code"] in incompatible_codes for issue in issues):
        status = "incompatible"
    elif issues:
        status = "unknown"
    else:
        status = "compatible"
    return {
        "schema_version": 1,
        "profile_id": profile["id"],
        "profile_name": profile["name"],
        "status": status,
        "toolkit": toolkit_result,
        "devices": device_results,
        "issues": issues,
    }
