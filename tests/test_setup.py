from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from cuda_legacy_compat import cli, settings, toolchain

FAKE_CLANG = "#!/bin/sh\necho 'clang version 23.1.4 (fixture)'\n"


def fake_llvm_archive(path: Path, version: str = "23.1.4") -> Path:
    """A tar.xz shaped like the llvm-project release: a top directory with bin/ and lib/clang/."""
    with tarfile.open(path, "w:xz") as archive:
        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            info = tarfile.TarInfo(f"LLVM-23.1.4-Linux-X64/{name}")
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))

        def link(name: str, target: str) -> None:
            info = tarfile.TarInfo(f"LLVM-23.1.4-Linux-X64/{name}")
            info.type, info.linkname = tarfile.SYMTYPE, target
            archive.addfile(info)

        add("bin/clang-23", FAKE_CLANG.replace("23.1.4", version).encode(), 0o755)
        link("bin/clang", "clang-23")
        link("bin/clang++", "clang")
        add("lib/clang/23/include/__clang_cuda_runtime_wrapper.h", b"/* fixture */\n")
        add("bin/clang-tidy", b"not kept", 0o755)
        add("lib/libLLVM.so", b"not kept")
        add("../escape.txt", b"never written")
        link("bin/evil", "/etc/passwd")
    return path


def archive_for(path: Path) -> toolchain.Archive:
    return toolchain.Archive(path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, None)


class IsolatedHome(unittest.TestCase):
    """Every test gets its own config and data directories; nothing touches the real home."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cuda-legacy-setup-test-"))
        self.env = mock.patch.dict(os.environ, {
            "XDG_CONFIG_HOME": str(self.tmp / "config"), "XDG_DATA_HOME": str(self.tmp / "data"),
        })
        self.env.start()
        for name in ("CUDA_COMPAT_LLVM_ROOT", "CUDA_COMPAT_CUDA_ROOT", "CUDA_COMPAT_PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX"):
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        self.env.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class DownloadAndUnpackTests(IsolatedHome):
    def test_download_checks_the_digest_and_keeps_only_a_verified_file(self) -> None:
        source = fake_llvm_archive(self.tmp / "src.tar.xz")
        good = archive_for(source)
        target = self.tmp / "dl" / "LLVM.tar.xz"
        toolchain.download(good, target)
        self.assertEqual(target.read_bytes(), source.read_bytes())

        bad = toolchain.Archive(good.url, "0" * 64, good.size, None)
        other = self.tmp / "dl" / "other.tar.xz"
        with self.assertRaisesRegex(toolchain.SetupError, "SHA-256"):
            toolchain.download(bad, other)
        self.assertFalse(other.exists())
        self.assertFalse(other.with_name(other.name + ".partial").exists())

    def test_interrupted_download_leaves_nothing(self) -> None:
        def broken(*args, **kwargs):
            raise OSError("connection reset")
        target = self.tmp / "dl" / "LLVM.tar.xz"
        with self.assertRaisesRegex(toolchain.SetupError, "download failed"):
            toolchain.download(toolchain.Archive("https://example.invalid/x", "0" * 64, 1, None), target, opener=broken)
        self.assertEqual(list((self.tmp / "dl").iterdir()), [])

    def test_unpack_keeps_only_the_compiler_and_refuses_escapes(self) -> None:
        source = fake_llvm_archive(self.tmp / "src.tar.xz")
        target = self.tmp / "data" / "llvm"
        target.parent.mkdir(parents=True)
        toolchain.unpack(source, None, target)
        kept = sorted(str(p.relative_to(target)) for p in target.rglob("*"))
        self.assertIn("bin/clang++", kept)
        self.assertIn("lib/clang/23/include/__clang_cuda_runtime_wrapper.h", kept)
        self.assertNotIn("bin/clang-tidy", kept)
        self.assertNotIn("lib/libLLVM.so", kept)
        self.assertNotIn("bin/evil", kept)
        self.assertFalse((self.tmp / "data" / "escape.txt").exists())
        self.assertTrue((target / toolchain.MANAGED_MARKER).is_file())
        self.assertTrue(toolchain.llvm_version_ok(target))
        self.assertTrue(os.stat(target / "bin" / "clang-23").st_mode & stat.S_IXUSR)

    def test_unpack_rejects_the_wrong_version_and_leaves_no_target(self) -> None:
        source = fake_llvm_archive(self.tmp / "src.tar.xz", version="22.1.0")
        target = self.tmp / "data" / "llvm"
        target.parent.mkdir(parents=True)
        with self.assertRaisesRegex(toolchain.SetupError, "does not report LLVM 23.1.4"):
            toolchain.unpack(source, None, target)
        self.assertFalse(target.exists())
        self.assertEqual([p.name for p in target.parent.iterdir()], [])


class DiscoveryTests(IsolatedHome):
    def test_cuda_root_needs_both_headers(self) -> None:
        root = self.tmp / "cuda"
        (root / "include").mkdir(parents=True)
        (root / "include" / "cuda.h").write_text("")
        self.assertIsNone(toolchain.find_cuda_root(str(root)).value)
        (root / "include" / "cuda_runtime.h").write_text("")
        self.assertEqual(toolchain.find_cuda_root(str(root)).value, str(root))

    def test_python_must_hold_a_cuda_build_of_torch(self) -> None:
        facts = {"/env/cpu/python": ((3, 10, 12), "2.14.0+cpu", None), "/env/cuda/bin/python": ((3, 10, 12), "2.14.0+cu130", None)}
        with mock.patch.object(toolchain, "interpreter_facts", side_effect=lambda p: facts.get(p, ((3, 10, 12), None, None))), \
                mock.patch.object(toolchain.shutil, "which", side_effect=lambda p: p if p.startswith("/env") else None):
            os.environ["CUDA_COMPAT_PYTHON"] = "/env/cpu/python"
            os.environ["VIRTUAL_ENV"] = "/env/cuda"
            found = toolchain.find_python()
        self.assertEqual(found.value, "/env/cuda/bin/python")
        self.assertTrue(any("not a CUDA build" in item for item in found.considered))

    def test_an_explicit_python_without_torch_is_not_replaced(self) -> None:
        with mock.patch.object(toolchain, "interpreter_facts", return_value=((3, 10, 12), None, None)), \
                mock.patch.object(toolchain.shutil, "which", side_effect=lambda p: p):
            found = toolchain.find_python("/env/plain/python")
        self.assertIsNone(found.value)


class SetupCommandTests(IsolatedHome):
    def _cuda_root(self) -> Path:
        root = self.tmp / "cuda"
        (root / "include").mkdir(parents=True)
        for header in ("cuda.h", "cuda_runtime.h"):
            (root / "include" / header).write_text("")
        return root

    def _run(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), \
                mock.patch.object(cli, "_doctor_report", return_value={"status": "ready", "checks": [], "note": "n"}):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_setup_downloads_saves_and_remove_undoes_it(self) -> None:
        source = fake_llvm_archive(self.tmp / "src.tar.xz")
        cuda = self._cuda_root()
        with mock.patch.object(toolchain, "LLVM_ARCHIVES", (archive_for(source),)), \
                mock.patch.object(toolchain, "find_python", return_value=toolchain.Found("/env/bin/python", "fixture")):
            code, out, err = self._run("setup", "--yes", "--cuda-root", str(cuda))
        self.assertEqual(code, 0, err)
        saved = settings.load()
        self.assertEqual(saved["llvm_root"], str(toolchain.managed_llvm_dir()))
        self.assertTrue(saved["llvm_managed"])
        self.assertEqual(saved["cuda_root"], str(cuda))
        self.assertEqual(saved["python"], "/env/bin/python")
        self.assertIn("cuda-legacy run app.py", out)
        self.assertEqual(list(toolchain.managed_llvm_dir().parent.glob("*.tar.*")), [])

        code, out, _ = self._run("setup", "--remove")
        self.assertEqual(code, 0)
        self.assertFalse(toolchain.managed_llvm_dir().exists())
        self.assertFalse(settings.config_path().exists())
        self.assertFalse(settings.data_dir().exists())

    def test_remove_never_deletes_an_llvm_it_did_not_create(self) -> None:
        own = toolchain.managed_llvm_dir()
        own.mkdir(parents=True)
        (own / "user-file").write_text("mine")
        settings.save({"llvm_root": str(own)})
        self._run("setup", "--remove")
        self.assertTrue((own / "user-file").is_file())
        self.assertFalse(settings.config_path().exists())

    def test_dry_run_saves_and_downloads_nothing(self) -> None:
        source = fake_llvm_archive(self.tmp / "src.tar.xz")
        with mock.patch.object(toolchain, "LLVM_ARCHIVES", (archive_for(source),)), \
                mock.patch.object(toolchain, "find_python", return_value=toolchain.Found(None, "none")):
            code, out, _ = self._run("setup", "--dry-run", "--json", "--cuda-root", str(self._cuda_root()))
        self.assertEqual(code, 0)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["llvm_download"]["sha256"], archive_for(source).sha256)
        self.assertFalse(settings.config_path().exists())
        self.assertFalse(toolchain.managed_llvm_dir().exists())

    def test_without_a_terminal_setup_asks_for_yes_instead_of_hanging(self) -> None:
        source = fake_llvm_archive(self.tmp / "src.tar.xz")
        with mock.patch.object(toolchain, "LLVM_ARCHIVES", (archive_for(source),)), \
                mock.patch.object(toolchain, "find_python", return_value=toolchain.Found(None, "none")), \
                mock.patch.object(toolchain.sys.stdin, "isatty", return_value=False), \
                mock.patch("builtins.open", side_effect=OSError("no tty")):
            code, _, err = self._run("setup", "--cuda-root", str(self._cuda_root()))
        self.assertEqual(code, 2)
        self.assertIn("--yes", err)


class SavedSetupTests(IsolatedHome):
    def test_flag_then_environment_then_saved_then_default(self) -> None:
        settings.save({"llvm_root": "/saved/llvm"})
        self.assertEqual(settings.resolve(None, "CUDA_COMPAT_LLVM_ROOT", "llvm_root"), ("/saved/llvm", "setup"))
        os.environ["CUDA_COMPAT_LLVM_ROOT"] = "/env/llvm"
        self.assertEqual(settings.resolve(None, "CUDA_COMPAT_LLVM_ROOT", "llvm_root")[0], "/env/llvm")
        self.assertEqual(settings.resolve("/flag/llvm", "CUDA_COMPAT_LLVM_ROOT", "llvm_root")[0], "/flag/llvm")
        self.assertEqual(settings.resolve(None, "CUDA_COMPAT_CUDA_ROOT", "cuda_root", "/default")[0], "/default")

    def test_a_corrupt_saved_setup_is_ignored(self) -> None:
        settings.config_path().parent.mkdir(parents=True)
        settings.config_path().write_text("{not json")
        self.assertEqual(settings.load(), {})

    def test_run_without_setup_says_to_run_setup(self) -> None:
        script = self.tmp / "app.py"
        script.write_text("print('hi')\n")
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(["run", str(script)])
        self.assertEqual(code, 2)
        self.assertIn("cuda-legacy setup", err.getvalue())

    def test_run_uses_the_saved_setup(self) -> None:
        settings.save({"llvm_root": "/saved/llvm", "cuda_root": "/saved/cuda", "python": "/saved/python"})
        script = self.tmp / "app.py"
        script.write_text("")
        with mock.patch.object(cli, "launch", return_value=0) as launched:
            self.assertEqual(cli.main(["run", str(script), "--", "--epochs", "1"]), 0)
        kwargs = launched.call_args.kwargs
        self.assertEqual((str(kwargs["llvm_root"]), str(kwargs["cuda_root"]), kwargs["python"]),
                         ("/saved/llvm", "/saved/cuda", "/saved/python"))
        self.assertEqual(launched.call_args.args[1], ["--epochs", "1"])


class FirstStepsTests(IsolatedHome):
    def test_bare_command_points_to_setup_then_run(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main([]), 0)
        self.assertIn("Not set up yet", out.getvalue())
        self.assertIn("setup", out.getvalue())
        settings.save({"llvm_root": "/x"})
        out = io.StringIO()
        with redirect_stdout(out):
            cli.main([])
        self.assertIn("run app.py", out.getvalue())

if __name__ == "__main__":
    unittest.main()
