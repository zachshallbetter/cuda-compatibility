"""Guided setup: find or fetch the pieces the launcher needs, save them once, and undo it cleanly.

Setup never installs drivers, never changes the system CUDA install and never edits the PyTorch environment.
The only things it creates are the LLVM files under the data directory and the saved setup, and both are
listed in the setup's manifest so `setup --remove` deletes exactly those.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any, Callable, Iterable
from urllib.request import Request, urlopen

from cuda_legacy_compat import __version__, settings

LLVM_VERSION = "23.1.4"
MANAGED_MARKER = ".cuda-legacy-managed"


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str
    size: int
    decompressor: str | None  # an external program the archive needs, or None for Python's own


# The official llvm-project release; digests are the ones GitHub publishes for each asset.
LLVM_ARCHIVES = (
    Archive("https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.4/LLVM-23.1.4-Linux-X64.tar.zst",
            "efdd1dbd089b99bfcfcde7b8eb1236163f2e88df5a981cca4f783ec9cb5e0378", 1_190_060_068, "zstd"),
    Archive("https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.4/LLVM-23.1.4-Linux-X64.tar.xz",
            "25d814fda50d5fb0084b42be09605d8b1595d49d5e2e03a3345d7e112ff6f9f3", 2_017_061_664, None),
)

# The launcher needs only the compiler driver and its resource headers (about 370 MB of a 12 GB unpack).
_LLVM_KEEP = ("bin/clang-23", "bin/clang", "bin/clang++", "lib/clang/")

_CUDA_CANDIDATES = ("/usr/local/cuda-12.4", "/usr/local/cuda", "/usr/local/cuda-12", "/usr")


class SetupError(RuntimeError):
    """Setup could not finish; the message says what to do next."""


# ---------- discovery ----------

_INTERPRETER_PROBE = (
    "import importlib.metadata as m, json, sys\n"
    "try:\n    torch = m.version('torch')\n"
    "except m.PackageNotFoundError:\n    torch = None\n"
    "print(json.dumps({'version': list(sys.version_info[:3]), 'torch': torch, 'executable': sys.executable}))"
)


def interpreter_facts(python: str | None) -> tuple[tuple[int, int, int] | None, str | None, str | None]:
    """(version, torch version, error) for this interpreter, or the one named by `python`."""
    if python is None:
        try:
            torch_version: str | None = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        major, minor, micro = sys.version_info[:3]
        return (major, minor, micro), torch_version, None
    try:
        result = subprocess.run([python, "-c", _INTERPRETER_PROBE], capture_output=True, text=True,
                                timeout=30, check=False)
        facts = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return None, None, f"could not query {python}: {exc}"
    if not isinstance(facts, dict) or not isinstance(facts.get("version"), list):
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        return None, None, f"could not query {python}: {detail}"
    major, minor, micro = (int(part) for part in facts["version"][:3])
    return (major, minor, micro), facts.get("torch"), None


@dataclass
class Found:
    value: str | None
    detail: str
    considered: list[str] = field(default_factory=list)


def find_python(explicit: str | None = None) -> Found:
    """The interpreter whose environment holds a CUDA build of PyTorch."""
    candidates: list[tuple[str, str]] = []
    if explicit:
        candidates.append((explicit, "--python"))
    for env_name, suffix in (("CUDA_COMPAT_PYTHON", ""), ("VIRTUAL_ENV", "/bin/python"), ("CONDA_PREFIX", "/bin/python")):
        if os.environ.get(env_name):
            candidates.append((os.environ[env_name] + suffix, env_name))
    candidates.append((sys.executable, "this interpreter"))
    if shutil.which("python3"):
        candidates.append((shutil.which("python3") or "python3", "python3 on PATH"))
    considered = []
    seen = set()
    for path, origin in candidates:
        resolved = shutil.which(path) or path
        if resolved in seen:
            continue
        seen.add(resolved)
        version, torch_version, error = interpreter_facts(None if resolved == sys.executable else resolved)
        if error or version is None:
            considered.append(f"{resolved} ({origin}): {error or 'not runnable'}")
        elif not torch_version:
            considered.append(f"{resolved} ({origin}): no PyTorch")
        elif "+cu" not in torch_version:
            considered.append(f"{resolved} ({origin}): torch {torch_version} is not a CUDA build")
        else:
            return Found(resolved, f"{resolved} (torch {torch_version}, from {origin})", considered)
        if explicit and origin == "--python":
            break
    return Found(None, "no Python environment with a CUDA build of PyTorch was found", considered)


def _has_headers(root: Path) -> bool:
    return (root / "include" / "cuda.h").is_file() and (root / "include" / "cuda_runtime.h").is_file()


def find_cuda_root(explicit: str | None = None) -> Found:
    candidates = [explicit] if explicit else [os.environ.get("CUDA_COMPAT_CUDA_ROOT"), *_CUDA_CANDIDATES]
    if not explicit:
        candidates += sorted((str(p) for p in Path("/usr/local").glob("cuda-12.*")), reverse=True)
    considered = []
    for candidate in filter(None, dict.fromkeys(candidates)):
        if _has_headers(Path(candidate)):
            return Found(candidate, f"CUDA headers under {candidate}", considered)
        considered.append(f"{candidate}: no cuda.h and cuda_runtime.h")
    return Found(None, "CUDA development headers (cuda.h, cuda_runtime.h) were not found", considered)


def llvm_version_ok(root: Path) -> bool:
    clangxx = root / "bin" / "clang++"
    if not clangxx.is_file() or not os.access(clangxx, os.X_OK):
        return False
    try:
        result = subprocess.run([str(clangxx), "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and LLVM_VERSION in result.stdout


def managed_llvm_dir() -> Path:
    return settings.data_dir() / f"llvm-{LLVM_VERSION}"


def find_llvm(explicit: str | None = None) -> Found:
    candidates = [explicit] if explicit else [
        os.environ.get("CUDA_COMPAT_LLVM_ROOT"), settings.load().get("llvm_root"), str(managed_llvm_dir()),
        f"/opt/LLVM-{LLVM_VERSION}-Linux-X64", f"/opt/LLVM-{LLVM_VERSION}", str(Path.home() / f"LLVM-{LLVM_VERSION}-Linux-X64"),
    ]
    considered = []
    for candidate in filter(None, dict.fromkeys(candidates)):
        if llvm_version_ok(Path(candidate)):
            return Found(candidate, f"LLVM {LLVM_VERSION} at {candidate}", considered)
        considered.append(f"{candidate}: no working clang++ {LLVM_VERSION}")
    return Found(None, f"LLVM {LLVM_VERSION} was not found", considered)


# ---------- download and unpack ----------

def choose_archive(archives: Iterable[Archive] | None = None) -> Archive:
    archives = list(LLVM_ARCHIVES if archives is None else archives)
    for archive in archives:
        if archive.decompressor is None or shutil.which(archive.decompressor):
            return archive
    raise SetupError("no LLVM archive format can be unpacked here")


def _human(size: int) -> str:
    return f"{size / 1_000_000_000:.1f} GB" if size >= 1_000_000_000 else f"{size / 1_000_000:.0f} MB"


def download(archive: Archive, destination: Path, progress: Callable[[int, int], None] | None = None,
             opener: Callable[..., IO[bytes]] = urlopen) -> Path:
    """Fetch the archive to `destination`, checking its SHA-256; a partial or wrong file never survives."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    done = 0
    try:
        with opener(Request(archive.url, headers={"User-Agent": f"cuda-legacy/{__version__}"}), timeout=60) as response, \
                partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, archive.size)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise SetupError(f"download failed: {exc}. Check the network and run `cuda-legacy setup` again.") from exc
    if digest.hexdigest() != archive.sha256:
        partial.unlink(missing_ok=True)
        raise SetupError(f"the downloaded archive's SHA-256 is {digest.hexdigest()}, expected {archive.sha256}; "
                         "it was deleted. Run `cuda-legacy setup` again; if this repeats, report it.")
    partial.replace(destination)
    return destination


def _wanted(name: str) -> str | None:
    """The path inside the LLVM root for a member we keep, or None to skip it."""
    parts = PurePosixPath(name).parts
    if len(parts) < 2 or any(part in ("..", "") for part in parts) or PurePosixPath(name).is_absolute():
        return None
    inner = "/".join(parts[1:])
    return inner if any(inner == keep or (keep.endswith("/") and inner.startswith(keep)) for keep in _LLVM_KEEP) else None


def unpack(archive_path: Path, decompressor: str | None, target: Path) -> Path:
    """Extract the compiler subset into `target` atomically (a partial unpack never becomes `target`)."""
    staging = Path(tempfile.mkdtemp(prefix=".llvm-unpack-", dir=target.parent))
    process = None
    try:
        if decompressor:
            process = subprocess.Popen([decompressor, "-dc", str(archive_path)], stdout=subprocess.PIPE)
            stream = tarfile.open(fileobj=process.stdout, mode="r|")
        else:
            stream = tarfile.open(archive_path, mode="r|*")
        with stream:
            for member in stream:
                inner = _wanted(member.name)
                if inner is None:
                    continue
                destination = staging / inner
                if member.issym():
                    link = PurePosixPath(member.linkname)
                    if link.is_absolute() or ".." in link.parts:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.unlink(missing_ok=True)
                    destination.symlink_to(member.linkname)
                elif member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = stream.extractfile(member)
                    with destination.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    destination.chmod(0o755 if member.mode & 0o111 else 0o644)
        if process and process.wait():
            raise SetupError(f"{decompressor} could not read {archive_path.name}")
        if not llvm_version_ok(staging):
            raise SetupError(f"the unpacked compiler does not report LLVM {LLVM_VERSION}")
        (staging / MANAGED_MARKER).write_text("Created by cuda-legacy setup; `cuda-legacy setup --remove` deletes it.\n")
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        return target
    except (tarfile.TarError, OSError) as exc:
        raise SetupError(f"could not unpack {archive_path.name}: {exc}") from exc
    finally:
        if process:
            if process.poll() is None:
                process.kill()
            if process.stdout:
                process.stdout.close()
            process.wait()
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# ---------- the command ----------

@dataclass
class Plan:
    python: Found
    cuda: Found
    llvm: Found
    archive: Archive | None

    def as_json(self) -> dict[str, Any]:
        return {
            "python": vars(self.python), "cuda_root": vars(self.cuda), "llvm_root": vars(self.llvm),
            "llvm_download": None if self.archive is None else {"url": self.archive.url, "sha256": self.archive.sha256,
                                                                "bytes": self.archive.size,
                                                                "destination": str(managed_llvm_dir())},
        }


def plan(python: str | None, cuda_root: str | None, llvm_root: str | None) -> Plan:
    llvm = find_llvm(llvm_root)
    archive = None
    if llvm.value is None and not llvm_root:
        try:
            archive = choose_archive()
        except SetupError:
            archive = None
    return Plan(find_python(python), find_cuda_root(cuda_root), llvm, archive)


def _say(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        stream = open("/dev/tty", "r+") if not sys.stdin.isatty() else None
    except OSError:
        stream = None
    if stream is None and not sys.stdin.isatty():
        raise SetupError("setup needs a yes to download; run it in a terminal or pass --yes")
    out = stream or sys.stdout
    out.write(f"{question} [Y/n] ")
    out.flush()
    answer = (stream.readline() if stream else input()).strip().lower()
    if stream:
        stream.close()
    return answer in ("", "y", "yes")


def _progress(quiet: bool) -> Callable[[int, int], None]:
    state = {"last": -1, "start": time.monotonic()}

    def show(done: int, total: int) -> None:
        if quiet or not total:
            return
        percent = done * 100 // total
        if percent != state["last"] and (percent % 5 == 0 or done == total):
            state["last"] = percent
            rate = done / max(time.monotonic() - state["start"], 0.001) / 1_000_000
            print(f"  {percent:3d}%  {_human(done)} of {_human(total)}  ({rate:.0f} MB/s)", file=sys.stderr, flush=True)
    return show


def run_setup(*, python: str | None, cuda_root: str | None, llvm_root: str | None, assume_yes: bool,
              dry_run: bool, quiet: bool = False) -> dict[str, Any]:
    """Find or fetch everything, save it, and return what was decided."""
    steps = plan(python, cuda_root, llvm_root)
    result: dict[str, Any] = {"plan": steps.as_json(), "saved": None, "created": []}
    _say("CUDA Compatibility setup", quiet)
    for label, found in (("PyTorch environment", steps.python), ("CUDA headers", steps.cuda), ("LLVM", steps.llvm)):
        _say(f"  {label + ':':<21}{found.detail}", quiet)
    if steps.llvm.value is None and steps.archive:
        _say(f"  {'':<21}will download {_human(steps.archive.size)} from llvm-project (SHA-256 checked) and keep "
             f"about 370 MB in {managed_llvm_dir()}", quiet)
    if dry_run:
        _say("Dry run: nothing was downloaded or saved.", quiet)
        return result
    if steps.llvm.value is None:
        if llvm_root:
            raise SetupError(f"{llvm_root} is not a working LLVM {LLVM_VERSION}; fix the path or omit --llvm-root "
                             "to let setup download it")
        if steps.archive is None:
            raise SetupError("no LLVM archive format can be unpacked here; install `zstd` or `xz` and run setup again")
        if not _confirm(f"Download LLVM {LLVM_VERSION} ({_human(steps.archive.size)})?", assume_yes):
            raise SetupError("setup stopped before downloading; pass --llvm-root to use an LLVM you already have")
        target = managed_llvm_dir()
        target.parent.mkdir(parents=True, exist_ok=True)
        archive_path = target.parent / Path(steps.archive.url).name
        if not archive_path.is_file():
            _say(f"Downloading {Path(steps.archive.url).name} ...", quiet)
            download(steps.archive, archive_path, _progress(quiet))
        _say("Checking and unpacking the compiler ...", quiet)
        try:
            unpack(archive_path, steps.archive.decompressor, target)
        finally:
            archive_path.unlink(missing_ok=True)
        steps.llvm = Found(str(target), f"LLVM {LLVM_VERSION} at {target} (installed by setup)")
        result["created"].append(str(target))
    saved = {
        "python": steps.python.value, "cuda_root": steps.cuda.value, "llvm_root": steps.llvm.value,
        "llvm_managed": steps.llvm.value == str(managed_llvm_dir()),
        "package_version": __version__, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = settings.save({key: value for key, value in saved.items() if value is not None or key == "llvm_managed"})
    result["saved"] = str(path)
    _say(f"Saved to {path}.", quiet)
    return result


def remove(*, quiet: bool = False) -> list[str]:
    """Delete what setup created: the managed LLVM (only if it carries our marker) and the saved setup."""
    removed = []
    target = managed_llvm_dir()
    if (target / MANAGED_MARKER).is_file():
        shutil.rmtree(target)
        removed.append(str(target))
    for leftover in target.parent.glob("LLVM-*.partial"):
        leftover.unlink(missing_ok=True)
        removed.append(str(leftover))
    path = settings.config_path()
    if path.is_file():
        path.unlink()
        removed.append(str(path))
    for directory in (settings.data_dir(), settings.config_dir()):
        try:
            directory.rmdir()
        except OSError:
            pass
    for item in removed:
        _say(f"Removed {item}", quiet)
    if not removed:
        _say("Nothing to remove: setup has not saved anything here.", quiet)
    return removed
