"""Inspect CUDA hardware, review compatibility profiles, and launch tested adapters."""

from __future__ import annotations

import argparse
import ctypes.util
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Sequence

from cuda_legacy_compat import __version__
from cuda_legacy_compat.inventory import inspect_system
from cuda_legacy_compat.launch import LaunchError, launch
from cuda_legacy_compat.profiles import ProfileError, audit_profile, load_profile


def _profile_directories() -> tuple[Path, ...]:
    candidates = (
        Path(__file__).resolve().parents[2] / "profiles",
        Path(sysconfig.get_path("data")) / "share" / "cuda-legacy-compat" / "profiles",
        Path(__file__).resolve().parents[1] / "share" / "cuda-legacy-compat" / "profiles",
    )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir() and candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _resolve_profile_path(value: str) -> Path:
    requested = Path(value)
    if requested.is_file():
        return requested
    filename = requested.name if requested.suffix == ".json" else f"{requested.name}.json"
    return next(
        (
            directory / filename
            for directory in _profile_directories()
            if (directory / filename).is_file()
        ),
        requested,
    )


def _profile_summaries() -> list[dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for directory in _profile_directories():
        for path in sorted(directory.glob("*.json")):
            profile = load_profile(path)
            profile_id = profile["id"]
            profiles.setdefault(
                profile_id,
                {
                    "id": profile_id,
                    "name": profile["name"],
                    "source": profile["source"],
                    "cuda_version": profile["build"]["cuda_version"],
                    "minimum_compute_capability": ".".join(
                        str(part)
                        for part in profile["gpu"]["minimum_compute_capability"]
                    ),
                },
            )
    if not profiles:
        raise ProfileError("no compatibility profiles were found in this installation")
    return [profiles[profile_id] for profile_id in sorted(profiles)]


def _doctor_check(
    check_id: str,
    status: str,
    detail: str,
    next_step: str | None = None,
) -> dict[str, str]:
    check = {"id": check_id, "status": status, "detail": detail}
    if next_step:
        check["next_step"] = next_step
    return check


def _selected_gpu_devices(
    devices: list[dict[str, Any]], visibility: str | None
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if visibility is None:
        return devices, None
    tokens = [token.strip() for token in visibility.split(",")]
    if not visibility.strip() or "-1" in tokens:
        return [], None
    if not all(token.isdecimal() for token in tokens):
        return None, (
            "CUDA_VISIBLE_DEVICES contains UUIDs or selectors that inventory cannot resolve"
        )
    by_index = {device["index"]: device for device in devices}
    selected = [by_index[int(token)] for token in tokens if int(token) in by_index]
    if len(selected) != len(tokens):
        return None, "CUDA_VISIBLE_DEVICES names a GPU index absent from nvidia-smi inventory"
    return selected, None


_INTERPRETER_PROBE = (
    "import importlib.metadata as m, json, sys\n"
    "try:\n    torch = m.version('torch')\n"
    "except m.PackageNotFoundError:\n    torch = None\n"
    "print(json.dumps({'version': list(sys.version_info[:3]), 'torch': torch}))"
)


def _interpreter_facts(python: str | None) -> tuple[tuple[int, int, int] | None, str | None, str | None]:
    """Return (version, torch version, error) for this interpreter or the one named by --python."""
    if python is None:
        try:
            torch_version: str | None = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        major, minor, micro = sys.version_info[:3]
        return (major, minor, micro), torch_version, None
    try:
        result = subprocess.run(
            [python, "-c", _INTERPRETER_PROBE], capture_output=True, text=True, timeout=30, check=False
        )
        facts = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return None, None, f"could not query {python}: {exc}"
    if not isinstance(facts, dict) or not isinstance(facts.get("version"), list):
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        return None, None, f"could not query {python}: {detail}"
    major, minor, micro = (int(part) for part in facts["version"][:3])
    return (major, minor, micro), facts.get("torch"), None


def _doctor_report(llvm_root: str | None, cuda_root: str, python: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        checks.append(_doctor_check("platform", "ready", f"{system} {machine}"))
    else:
        checks.append(
            _doctor_check(
                "platform",
                "unsupported",
                f"observed {system} {machine}; the adapter requires Linux x86_64",
                "Use a Linux x86_64 host for the current PyTorch adapter.",
            )
        )

    interpreter_version, torch_version, interpreter_error = _interpreter_facts(python)
    python_version = ".".join(str(part) for part in interpreter_version or ())
    if interpreter_version is None:
        checks.append(
            _doctor_check(
                "python",
                "not_ready",
                interpreter_error or "the target interpreter could not be queried",
                "Pass --python with the interpreter of the environment that contains PyTorch.",
            )
        )
    elif interpreter_version < (3, 10):
        checks.append(
            _doctor_check(
                "python",
                "unsupported",
                f"observed Python {python_version}; Python 3.10+ is required",
                "Use Python 3.10 or newer in the environment that contains PyTorch.",
            )
        )
    elif interpreter_version[:2] == (3, 10):
        checks.append(_doctor_check("python", "ready", f"Python {python_version} (tested series)"))
    else:
        checks.append(
            _doctor_check(
                "python",
                "unknown",
                f"Python {python_version}; the adapter was tested on Python 3.10.12",
                "Use Python 3.10.12 for the verified configuration, "
                "or validate this interpreter separately.",
            )
        )

    if interpreter_version is not None and torch_version is None:
        checks.append(
            _doctor_check(
                "pytorch",
                "not_ready",
                "PyTorch is not installed in this Python environment",
                "Install the CUDA-enabled PyTorch wheel into this same environment.",
            )
        )
    elif interpreter_version is not None and torch_version is not None:
        if torch_version == "2.14.0+cu130":
            status = "ready"
            next_step = None
            detail = f"torch {torch_version} (the tested CUDA 13 wheel)"
        elif "+cu" in torch_version:
            status = "unknown"
            next_step = "The verified adapter evidence currently targets torch 2.14.0+cu130."
            detail = f"found CUDA wheel torch {torch_version}; this build has not been qualified"
        else:
            status = "not_ready"
            next_step = "Install the CUDA-enabled PyTorch wheel into this same environment."
            detail = f"found torch {torch_version}; a CUDA wheel is required"
        checks.append(_doctor_check("pytorch", status, detail, next_step))

    inventory = inspect_system()
    gpu_inventory = inventory["gpu_inventory"]
    if gpu_inventory["status"] != "available":
        status = "not_ready"
        detail = f"GPU inventory is {gpu_inventory['status']}"
        next_step = "Install or enable the NVIDIA driver and confirm `nvidia-smi` can see the GPU."
    else:
        devices, selection_error = _selected_gpu_devices(
            gpu_inventory["devices"], os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        if selection_error:
            status = "unknown"
            detail = selection_error
            next_step = (
                "Use numeric nvidia-smi indices in CUDA_VISIBLE_DEVICES, or unset it for all GPUs."
            )
        elif not devices:
            status = "not_ready"
            detail = "no GPUs are visible to this process"
            next_step = "Set CUDA_VISIBLE_DEVICES to one or more supported GPU indices."
        else:
            capabilities = [device["compute_capability"] for device in devices]
            if all(capability == "5.2" for capability in capabilities):
                status = "ready"
                selected = os.environ.get("CUDA_VISIBLE_DEVICES")
                selection = f" selected by CUDA_VISIBLE_DEVICES={selected}" if selected else ""
                detail = f"{len(devices)} GPU(s){selection}, all compute capability 5.2"
                next_step = None
            else:
                status = "unsupported"
                detail = (
                    f"visible compute capabilities: {', '.join(capabilities)}; "
                    "this adapter requires only 5.2 devices"
                )
                next_step = (
                    "Select only supported sm_52 devices with CUDA_VISIBLE_DEVICES "
                    "before running the adapter."
                )
    checks.append(_doctor_check("gpu", status, detail, next_step))

    if llvm_root:
        clangxx = Path(llvm_root) / "bin" / "clang++"
        if not clangxx.is_file() or not os.access(clangxx, os.X_OK):
            checks.append(
                _doctor_check(
                    "llvm",
                    "not_ready",
                    f"LLVM clang++ is unavailable at {clangxx}",
                    "Install the verified LLVM 23.1.4 build and pass its root with --llvm-root.",
                )
            )
        else:
            try:
                result = subprocess.run(
                    [str(clangxx), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                version_text = ""
                version_error = str(exc)
                version_ok = False
            else:
                version_text = result.stdout
                version_error = result.stderr.strip() or f"exit code {result.returncode}"
                version_ok = result.returncode == 0
            if version_ok and "23.1.4" in version_text:
                checks.append(_doctor_check("llvm", "ready", f"LLVM 23.1.4 at {clangxx}"))
            else:
                checks.append(
                    _doctor_check(
                        "llvm",
                        "unsupported",
                        f"expected LLVM 23.1.4 at {clangxx}; {version_error}",
                        "Install the verified LLVM 23.1.4 build.",
                    )
                )
    else:
        checks.append(
            _doctor_check(
                "llvm",
                "not_ready",
                "LLVM root was not provided",
                "Pass --llvm-root or set CUDA_COMPAT_LLVM_ROOT to LLVM 23.1.4.",
            )
        )

    cuda_path = Path(cuda_root)
    missing_headers = [
        str(path)
        for path in (cuda_path / "include" / "cuda.h", cuda_path / "include" / "cuda_runtime.h")
        if not path.is_file()
    ]
    if missing_headers:
        checks.append(
            _doctor_check(
                "cuda_headers",
                "not_ready",
                f"missing: {', '.join(missing_headers)}",
                "Install CUDA 12.4 development headers and pass their root with --cuda-root.",
            )
        )
    else:
        checks.append(
            _doctor_check("cuda_headers", "ready", f"CUDA headers found under {cuda_path}")
        )

    cxx = shutil.which("g++")
    if cxx:
        try:
            version_result = subprocess.run(
                [cxx, "-dumpversion"], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            version = ""
            version_error = str(exc)
            version_ok = False
        else:
            version = version_result.stdout.strip()
            version_ok = version_result.returncode == 0
        cxx_include = Path("/usr/include/c++") / version
        arch_include = Path("/usr/include/x86_64-linux-gnu/c++") / version
        missing_cxx_paths = []
        if not (cxx_include / "climits").is_file():
            missing_cxx_paths.append(str(cxx_include / "climits"))
        if not arch_include.is_dir():
            missing_cxx_paths.append(str(arch_include))
        if version_ok and version and not missing_cxx_paths:
            checks.append(
                _doctor_check(
                    "cxx_compiler", "ready", f"{cxx} {version} with standard library headers"
                )
            )
        else:
            checks.append(
                _doctor_check(
                    "cxx_compiler",
                    "not_ready",
                    (
                        f"g++ is present at {cxx}, but standard headers are missing: "
                        f"{', '.join(missing_cxx_paths)}"
                        if version_ok and version
                        else f"could not query g++ version at {cxx}: {version_error}"
                    ),
                    "Install g++ and its C++ standard library development headers.",
                )
            )
    else:
        checks.append(
            _doctor_check(
                "cxx_compiler", "not_ready", "g++ was not found on PATH",
                "Install g++ and its C++ standard library development headers.",
            )
        )

    driver_library = ctypes.util.find_library("cuda")
    if driver_library:
        checks.append(_doctor_check("cuda_driver_library", "ready", driver_library))
    else:
        checks.append(
            _doctor_check(
                "cuda_driver_library",
                "not_ready",
                "the CUDA Driver API library was not found",
                "Install a compatible NVIDIA driver and expose its driver library to the linker.",
            )
        )

    return {
        "schema_version": 1,
        "status": (
            "ready" if all(check["status"] == "ready" for check in checks) else "needs_attention"
        ),
        "checks": checks,
        "note": (
            "Ready checks confirm launcher prerequisites only; "
            "they do not certify a model or operator workload."
        ),
    }


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _print_profiles(profiles: list[dict[str, Any]]) -> None:
    print("Available compatibility profiles:")
    for profile in profiles:
        print(
            f"  {profile['id']}  CUDA {profile['cuda_version']}  "
            f"minimum GPU {profile['minimum_compute_capability']}  {profile['name']}"
        )


def _print_doctor(report: dict[str, Any]) -> None:
    print(f"CUDA Compatibility doctor: {report['status']}")
    labels = {
        "ready": "PASS",
        "not_ready": "NEEDS SETUP",
        "unsupported": "UNSUPPORTED",
        "unknown": "UNVERIFIED",
    }
    for check in report["checks"]:
        print(f"{labels[check['status']]:<12} {check['id']}: {check['detail']}")
        if "next_step" in check:
            print(f"             Next: {check['next_step']}")
    print(report["note"])


def _program_name() -> str:
    invoked = Path(sys.argv[0]).name
    if invoked in {"cuda-legacy-compat", "cuda-legacy"}:
        return invoked
    return "cuda-legacy-compat"


def _build_parser(
    prog: str = "cuda-legacy-compat",
) -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Inspect your NVIDIA setup, review compatibility profiles, "
            "and run the experimental PyTorch adapter."
        ),
        epilog=(
            "Examples:\n"
            f"  {prog} inspect\n"
            f"  {prog} profiles list\n"
            f"  {prog} doctor --llvm-root /opt/LLVM-23.1.4\n"
            f"  {prog} audit pytorch-2.14-cu130\n"
            f"  {prog} run --llvm-root /opt/LLVM-23.1.4 -- app.py\n\n"
            "The run command affects only its child Python process. "
            "Its current adapter is experimental, "
            "Linux x86_64-only, and restricted to homogeneous sm_52 devices."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    command_parsers: dict[str, argparse.ArgumentParser] = {}

    command_parsers["inspect"] = commands.add_parser(
        "inspect", help="show local NVIDIA GPU and CUDA compiler inventory as JSON",
        description=(
            "Report what NVIDIA GPU and CUDA compiler tools "
            "are visible to this Python environment."
        ),
    )
    command_parsers["inspect"].add_argument(
        "--json", action="store_true", help="accepted for consistency; inspect always prints JSON"
    )

    profiles_parser = commands.add_parser(
        "profiles", help="list or inspect the compatibility profiles included with the package"
    )
    profiles_commands = profiles_parser.add_subparsers(dest="profiles_command", required=True)
    profile_list = profiles_commands.add_parser("list", help="list available target profiles")
    profile_list.add_argument("--json", action="store_true", help="print machine-readable JSON")
    profile_show = profiles_commands.add_parser("show", help="show one profile and its source")
    profile_show.add_argument("profile", help="profile ID, filename, or JSON path")
    profile_show.add_argument("--json", action="store_true", help="print machine-readable JSON")

    doctor_parser = commands.add_parser(
        "doctor", help="check prerequisites for the experimental PyTorch launcher"
    )
    doctor_parser.add_argument(
        "--llvm-root", default=os.environ.get("CUDA_COMPAT_LLVM_ROOT"),
        help="LLVM 23.1.4 installation (or CUDA_COMPAT_LLVM_ROOT)",
    )
    doctor_parser.add_argument(
        "--cuda-root", default=os.environ.get("CUDA_COMPAT_CUDA_ROOT", "/usr/local/cuda-12.4"),
        help="CUDA development headers root (default: /usr/local/cuda-12.4)",
    )
    doctor_parser.add_argument(
        "--python", default=os.environ.get("CUDA_COMPAT_PYTHON"),
        help="interpreter of the environment that contains PyTorch "
        "(or CUDA_COMPAT_PYTHON; default: the interpreter running cuda-legacy)",
    )
    doctor_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    audit_parser = commands.add_parser(
        "audit", help="compare machine inventory with a target profile"
    )
    audit_parser.add_argument("profile", help="profile ID, filename, or JSON path")

    run_parser = commands.add_parser(
        "run", help="run a Python script through the opt-in PyTorch Maxwell adapter"
    )
    run_parser.add_argument(
        "--llvm-root", default=os.environ.get("CUDA_COMPAT_LLVM_ROOT"),
        help="LLVM 23.1.4 installation (or CUDA_COMPAT_LLVM_ROOT)",
    )
    run_parser.add_argument(
        "--cuda-root", default=os.environ.get("CUDA_COMPAT_CUDA_ROOT", "/usr/local/cuda-12.4"),
        help="CUDA development headers root (default: /usr/local/cuda-12.4)",
    )
    run_parser.add_argument(
        "--python", default=os.environ.get("CUDA_COMPAT_PYTHON"),
        help="interpreter of the environment that contains PyTorch "
        "(or CUDA_COMPAT_PYTHON; default: the interpreter running cuda-legacy)",
    )
    run_parser.add_argument(
        "script", help="Python script to execute with the selected interpreter"
    )
    run_parser.add_argument(
        "script_args", nargs=argparse.REMAINDER,
        help="arguments forwarded to the script; separate them with --",
    )

    help_parser = commands.add_parser("help", help="show general or command-specific help")
    help_parser.add_argument(
        "topic",
        nargs="?",
        choices=tuple(command_parsers) + ("profiles", "doctor", "audit", "run"),
    )
    command_parsers.update(
        {
            "profiles": profiles_parser,
            "doctor": doctor_parser,
            "audit": audit_parser,
            "run": run_parser,
            "help": help_parser,
        }
    )
    return parser, command_parsers


def main(argv: Sequence[str] | None = None) -> int:
    parser, command_parsers = _build_parser(_program_name())
    args = parser.parse_args(argv)

    if args.command == "help":
        (command_parsers[args.topic] if args.topic else parser).print_help()
        return 0
    if args.command == "inspect":
        _print_json(inspect_system())
        return 0
    if args.command == "profiles":
        try:
            if args.profiles_command == "list":
                profiles = _profile_summaries()
                if args.json:
                    _print_json(profiles)
                else:
                    _print_profiles(profiles)
                return 0
            profile = load_profile(_resolve_profile_path(args.profile))
        except ProfileError as exc:
            print(f"profile error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            _print_json(profile)
        else:
            print(f"{profile['name']} ({profile['id']})")
            print(f"CUDA package version: {profile['build']['cuda_version']}")
            minimum = ".".join(map(str, profile["gpu"]["minimum_compute_capability"]))
            print(f"Minimum GPU capability: {minimum}")
            print(f"Source: {profile['source']}")
            if profile.get("notes"):
                print(f"Notes: {profile['notes']}")
        return 0
    if args.command == "doctor":
        report = _doctor_report(args.llvm_root, args.cuda_root, args.python)
        if args.json:
            _print_json(report)
        else:
            _print_doctor(report)
        return 0 if report["status"] == "ready" else 1
    if args.command == "run":
        if not args.llvm_root:
            print("launch error: pass --llvm-root or set CUDA_COMPAT_LLVM_ROOT", file=sys.stderr)
            return 2
        script_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
        try:
            return launch(
                Path(args.script),
                script_args,
                llvm_root=Path(args.llvm_root),
                cuda_root=Path(args.cuda_root),
                python=args.python,
            )
        except LaunchError as exc:
            print(f"launch error: {exc}", file=sys.stderr)
            return 2
    try:
        profile = load_profile(_resolve_profile_path(args.profile))
        result = audit_profile(profile, inspect_system())
    except ProfileError as exc:
        print(f"profile error: {exc}", file=sys.stderr)
        return 2
    _print_json(result)
    if result["status"] == "compatible":
        return 0
    if result["status"] == "incompatible":
        return 1
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
