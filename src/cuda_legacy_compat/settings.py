"""Saved setup: where the PyTorch environment, LLVM and CUDA headers are, and what setup created."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_CUDA_ROOT = "/usr/local/cuda-12.4"


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "cuda-legacy"


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "cuda-legacy"


def config_path() -> Path:
    return config_dir() / "config.json"


def problem() -> str | None:
    """Why the saved setup exists but can't be used, or None when it is fine or simply absent."""
    path = config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"the saved setup at {path} can't be read ({exc.__class__.__name__}: {exc})"
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return f"the saved setup at {path} is from an unknown version"
    return None


def load() -> dict[str, Any]:
    """The saved setup, or {} when there is none or it can't be read (see problem())."""
    if problem():
        return {}
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(settings: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({**settings, "schema_version": SCHEMA_VERSION}, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def resolve(flag: str | None, env_name: str, key: str, default: str | None = None) -> tuple[str | None, str]:
    """A value and where it came from: the flag, then the environment, then the saved setup, then the default."""
    if flag:
        return flag, "flag"
    if os.environ.get(env_name):
        return os.environ[env_name], env_name
    saved = load().get(key)
    if saved:
        return str(saved), "setup"
    return default, "default"
