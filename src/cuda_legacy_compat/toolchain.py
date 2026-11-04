"""Guided setup: find or fetch the pieces the launcher needs, save them once, and undo it cleanly.

Setup never installs drivers, never changes the system CUDA install and never edits the PyTorch environment.
The only things it creates are the LLVM files and, when the system has none, the CUDA 12.4 headers under the data
directory, and the saved setup; each carries a marker so `setup --remove` deletes exactly those.
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
import zipfile
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

# Without a system CUDA 12.4, setup assembles the part the build reads (#204): headers and libdevice from NVIDIA's own
# wheels on PyPI, checked by SHA-256. clang's CUDA wrapper also includes cuRAND's mtgp32 header, which needs CCCL's
# <nv/target>; only the .post1 CCCL wheel carries nv/. Built from these, the PTX is byte-identical to the toolkit's.
CUDA_VERSION = "12.4"
_PYPI = "https://files.pythonhosted.org/packages"
CUDA_WHEELS = (
    Archive(f"{_PYPI}/ea/27/1795d86fe88ef397885f2e580ac37628ed058a92ed2c39dc8eac3adf0619/"
            "nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl",
            "64403288fa2136ee8e467cdc9c9427e0434110899d07c779f25b5c068934faa5", 883_737, None),
    Archive(f"{_PYPI}/54/5c/c2f2592f3a5a041d546501cdcfab0f19ccbd31714de238f381a6e302a783/"
            "nvidia_cuda_nvcc_cu12-12.4.131-py3-none-manylinux2014_x86_64.whl",
            "e4c9a35435c3b0e36c6397c5696334c9ea6650e524736cbd1c5d345ea099bb04", 22_265_199, None),
    Archive(f"{_PYPI}/8a/6d/44ad094874c6f1b9c654f8ed939590bdc408349f137f9b98a3a23ccec411/"
            "nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl",
            "a88f583d4e0bb643c49743469964103aa59f7f708d862c3ddb0fc07f851e3b8b", 56_305_206, None),
    Archive(f"{_PYPI}/7f/62/6da9278c0d06e9ed48520d0ac23e300cc209f8cb4fdc854ef2e133cc2aa0/"
            "nvidia_cuda_cccl_cu12-12.4.127.post1-py3-none-manylinux2014_x86_64.whl",
            "2027fa5fc75fbee703bf24e8275a40a605cf94a3e8fef8a7187897d49b34facb", 3_111_329, None),
)
_CUDA_KEEP = ("include", "nvvm")   # under each wheel's nvidia/<package>/


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
    if explicit and considered:
        return Found(None, considered[0], considered)
    return Found(None, "no Python environment with a CUDA build of PyTorch was found", considered)


def _has_headers(root: Path) -> bool:
    return (root / "include" / "cuda.h").is_file() and (root / "include" / "cuda_runtime.h").is_file()


def _has_libdevice(root: Path) -> bool:
    return (root / "nvvm" / "libdevice" / "libdevice.10.bc").is_file()


def managed_cuda_dir() -> Path:
    return settings.data_dir() / f"cuda-{CUDA_VERSION}"


def find_cuda_root(explicit: str | None = None) -> Found:
    candidates = [explicit] if explicit else [os.environ.get("CUDA_COMPAT_CUDA_ROOT"), *_CUDA_CANDIDATES,
                                              str(managed_cuda_dir())]
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
    if explicit:
        return Found(None, f"no working clang++ {LLVM_VERSION} under {explicit} (expected {explicit}/bin/clang++)",
                     considered)
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
            # llvm-project's .tar.zst uses a 1 GiB window; zstd refuses windows over 128 MiB without --long.
            window = ["--long=31"] if decompressor == "zstd" else []
            process = subprocess.Popen([decompressor, "-dc", *window, str(archive_path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        reason = str(exc)
        if process and process.stderr and process.poll() not in (None, 0):
            reason = process.stderr.read().decode(errors="replace").strip().splitlines()[-1:] or [reason]
            reason = f"{decompressor}: {reason[0]}"
        raise SetupError(f"could not unpack {archive_path.name}: {reason}") from exc
    finally:
        if process:
            if process.poll() is None:
                process.kill()
            if process.stdout:
                process.stdout.close()
            process.wait()
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _wanted_cuda(name: str) -> str | None:
    """The path inside the CUDA root for a wheel member we keep (nvidia/<package>/include|nvvm/...), or None."""
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or any(part in ("..", "") for part in parts) or len(parts) < 4 or parts[0] != "nvidia":
        return None
    return "/".join(parts[2:]) if parts[2] in _CUDA_KEEP else None


def unpack_cuda(wheels: Iterable[Path], target: Path) -> Path:
    """Assemble the CUDA root from the wheels into `target` atomically: headers, libdevice and an empty bin/
    (clang looks for it). A partial assembly never becomes `target`."""
    staging = Path(tempfile.mkdtemp(prefix=".cuda-unpack-", dir=target.parent))
    try:
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                for member in archive.infolist():
                    inner = _wanted_cuda(member.filename)
                    if inner is None or member.is_dir():
                        continue
                    destination = staging / inner
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, destination.open("wb") as out:
                        shutil.copyfileobj(source, out)
        (staging / "bin").mkdir(exist_ok=True)
        if not _has_headers(staging) or not _has_libdevice(staging):
            raise SetupError("NVIDIA's CUDA wheels did not hold cuda.h, cuda_runtime.h and libdevice; report this")
        (staging / MANAGED_MARKER).write_text("Created by cuda-legacy setup; `cuda-legacy setup --remove` deletes it.\n")
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        return target
    except (zipfile.BadZipFile, OSError) as exc:
        raise SetupError(f"could not unpack the CUDA headers: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# ---------- the command ----------

@dataclass
class Plan:
    python: Found
    cuda: Found
    llvm: Found
    archive: Archive | None
    cuda_wheels: tuple[Archive, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "python": vars(self.python), "cuda_root": vars(self.cuda), "llvm_root": vars(self.llvm),
            "llvm_download": None if self.archive is None else {"url": self.archive.url, "sha256": self.archive.sha256,
                                                                "bytes": self.archive.size,
                                                                "destination": str(managed_llvm_dir())},
            "cuda_download": None if not self.cuda_wheels else {
                "urls": [wheel.url for wheel in self.cuda_wheels], "bytes": sum(w.size for w in self.cuda_wheels),
                "destination": str(managed_cuda_dir())},
        }


def plan(python: str | None, cuda_root: str | None, llvm_root: str | None) -> Plan:
    llvm = find_llvm(llvm_root)
    archive = None
    if llvm.value is None and not llvm_root:
        try:
            archive = choose_archive()
        except SetupError:
            archive = None
    cuda = find_cuda_root(cuda_root)
    return Plan(find_python(python), cuda, llvm, archive, CUDA_WHEELS if cuda.value is None and not cuda_root else ())


def _say(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def _confirm(question: str, assume_yes: bool) -> bool:
    """Ask on the terminal even when stdin is a pipe (`curl ... | sh`); without any terminal, require --yes."""
    if assume_yes:
        return True
    if sys.stdin.isatty():
        reader, writer = sys.stdin, sys.stdout
        owned = False
    else:
        try:
            # Separate handles: a terminal can't seek, so a single "r+" text handle fails to open.
            reader, writer = open("/dev/tty", encoding="utf-8"), open("/dev/tty", "w", encoding="utf-8")
        except OSError:
            raise SetupError("setup needs a yes to download; run it in a terminal or pass --yes") from None
        owned = True
    try:
        writer.write(f"{question} [Y/n] ")
        writer.flush()
        answer = reader.readline().strip().lower()
    finally:
        if owned:
            reader.close()
            writer.close()
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
    # Parent directories that don't exist yet are ours to remove later (only if they are empty again).
    created_dirs = sorted(set(settings.load().get("created_dirs", [])) | {
        str(directory) for directory in _parents_setup_may_create() if not directory.exists()
    })
    _say("CUDA Compatibility setup", quiet)
    fetches = {
        "CUDA headers": steps.cuda.value is None and steps.cuda_wheels and
        f"will download {_human(sum(w.size for w in steps.cuda_wheels))} of NVIDIA's CUDA {CUDA_VERSION} wheels from "
        f"PyPI (SHA-256 checked) and keep about 47 MB of headers in {managed_cuda_dir()}",
        "LLVM": steps.llvm.value is None and steps.archive and
        f"will download {_human(steps.archive.size)} from llvm-project (SHA-256 checked) and keep about 370 MB in "
        f"{managed_llvm_dir()}",
    }
    for label, found in (("PyTorch environment", steps.python), ("CUDA headers", steps.cuda), ("LLVM", steps.llvm)):
        _say(f"  {label + ':':<21}{found.detail}", quiet)
        if fetches.get(label):
            _say(f"  {'':<21}{fetches[label]}", quiet)
    if dry_run:
        # Report what a real run would end with: stop (exit 2), saved but not ready (1), or ready (0).
        if steps.llvm.value is None and (llvm_root or steps.archive is None):
            result["would"] = "stop"
        elif steps.python.value is None or (steps.cuda.value is None and not steps.cuda_wheels):
            result["would"] = "needs_attention"
        else:
            result["would"] = "ready"
        verdict = {"stop": "setup would stop: fix the LLVM path above or omit --llvm-root",
                   "needs_attention": "setup would save this, but doctor would still need the items marked above",
                   "ready": "setup would finish with everything it needs"}[result["would"]]
        _say(f"Dry run: nothing was downloaded or saved; {verdict}.", quiet)
        return result
    if steps.cuda.value is None and steps.cuda_wheels:
        total = sum(wheel.size for wheel in steps.cuda_wheels)
        if not _confirm(f"Download the CUDA {CUDA_VERSION} headers from NVIDIA's wheels ({_human(total)})?", assume_yes):
            raise SetupError("setup stopped before downloading; install the CUDA 12.4 development headers, or pass "
                             "--cuda-root to use ones you already have")
        target = managed_cuda_dir()
        target.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        try:
            for wheel in steps.cuda_wheels:
                path = target.parent / Path(wheel.url).name
                if not path.is_file():
                    _say(f"Downloading {path.name} ...", quiet)
                    download(wheel, path, _progress(quiet))
                paths.append(path)
            _say("Assembling the CUDA headers ...", quiet)
            unpack_cuda(paths, target)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)
        steps.cuda = Found(str(target), f"CUDA {CUDA_VERSION} headers at {target} (installed by setup)")
        result["created"].append(str(target))
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
        "cuda_managed": steps.cuda.value == str(managed_cuda_dir()),
        "package_version": __version__, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "created_dirs": created_dirs,
    }
    path = settings.save({key: value for key, value in saved.items()
                          if value is not None or key in ("llvm_managed", "cuda_managed")})
    result["saved"] = str(path)
    _say(f"Saved to {path}.", quiet)
    return result


def _parents_setup_may_create() -> list[Path]:
    """The data and config directories and their parents, deepest first."""
    data, config = settings.data_dir(), settings.config_dir()
    return [data, data.parent, data.parent.parent, config, config.parent]


def remove(*, quiet: bool = False) -> list[str]:
    """Delete what setup created: the managed LLVM and CUDA headers (only if they carry our marker) and the saved
    setup."""
    removed = []
    created_dirs = [Path(item) for item in settings.load().get("created_dirs", [])]
    target = managed_llvm_dir()
    for managed in (target, managed_cuda_dir()):
        if (managed / MANAGED_MARKER).is_file():
            shutil.rmtree(managed)
            removed.append(str(managed))
    for pattern in ("LLVM-*.partial", "nvidia_*.whl", "nvidia_*.whl.partial"):
        for leftover in target.parent.glob(pattern):
            leftover.unlink(missing_ok=True)
            removed.append(str(leftover))
    path = settings.config_path()
    if path.is_file():
        path.unlink()
        removed.append(str(path))
    # Our own directories, then any parents setup created; rmdir only succeeds when they are empty.
    for directory in (settings.data_dir(), settings.config_dir(),
                      *sorted(created_dirs, key=lambda item: len(item.parts), reverse=True)):
        try:
            directory.rmdir()
        except OSError:
            pass
    for item in removed:
        _say(f"Removed {item}", quiet)
    if not removed:
        _say("Nothing to remove: setup has not saved anything here.", quiet)
    return removed
