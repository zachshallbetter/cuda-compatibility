from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cuda_legacy_compat import launch as launch_module
from cuda_legacy_compat.launch import LaunchError, build_adapter


class LauncherPreflightTests(unittest.TestCase):
    def test_missing_clang_fails_before_any_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(launch_module.platform, "system", return_value="Linux"), patch.object(
                launch_module.platform, "machine", return_value="x86_64"
            ), self.assertRaisesRegex(LaunchError, "LLVM clang\\+\\+ is unavailable"):
                build_adapter(root / "missing-llvm", root / "cuda", root / "out")

    def test_launch_rejects_missing_script_before_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(launch_module, "build_adapter") as build:
                with self.assertRaisesRegex(LaunchError, "Python script not found"):
                    launch_module.launch(
                        root / "missing.py", [], llvm_root=root / "llvm", cuda_root=root / "cuda"
                    )
            build.assert_not_called()

    def test_launch_rejects_missing_interpreter_before_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "app.py"
            script.write_text("print('hi')\n")
            with patch.object(launch_module, "build_adapter") as build:
                with self.assertRaisesRegex(LaunchError, "Python interpreter not found"):
                    launch_module.launch(
                        script, [], llvm_root=root / "llvm", cuda_root=root / "cuda",
                        python=str(root / "missing-python"),
                    )
            build.assert_not_called()

    def test_launch_runs_script_with_selected_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "app.py"
            script.write_text("print('hi')\n")
            artifacts = (root / "a.ptx", root / "bridge.so", root / "facade.so")
            artifacts[0].write_text("ptx")
            with patch.object(launch_module, "build_adapter", return_value=artifacts), patch.object(
                launch_module.subprocess, "Popen"
            ) as run:
                run.return_value.wait.return_value = 0
                launch_module.launch(
                    script, ["--x"], llvm_root=root / "llvm", cuda_root=root / "cuda",
                    python=sys.executable,
                )
            self.assertEqual(run.call_args.args[0][:3], [sys.executable, str(script.resolve()), "--x"])

    def test_launch_passes_module_code_and_interactive_to_the_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "app.py"
            script.write_text("print('hi')\n")
            artifacts = (root / "a.ptx", root / "bridge.so", root / "facade.so")
            artifacts[0].write_text("ptx")
            cases = [
                (dict(script=None, module="pytest"), ["--x"], ["-m", "pytest", "--x"]),
                (dict(script=None, code="print(1)"), ["a"], ["-c", "print(1)", "a"]),
                (dict(script=None, interactive=True), [], ["-i"]),
                (dict(script=script, interactive=True), ["--x"], ["-i", str(script.resolve()), "--x"]),
            ]
            for kwargs, args, expected in cases:
                with self.subTest(kwargs=kwargs), patch.object(launch_module, "build_adapter", return_value=artifacts), \
                        patch.object(launch_module.subprocess, "Popen") as run:
                    run.return_value.wait.return_value = 0
                    script_arg = kwargs.pop("script")
                    launch_module.launch(script_arg, args, llvm_root=root / "llvm", cuda_root=root / "cuda",
                                         python=sys.executable, **kwargs)
                    self.assertEqual(run.call_args.args[0], [sys.executable, *expected])
            with patch.object(launch_module, "build_adapter") as build:
                with self.assertRaisesRegex(LaunchError, "name a script"):
                    launch_module.launch(None, [], llvm_root=root / "llvm", cuda_root=root / "cuda", python=sys.executable)
                build.assert_not_called()

    @unittest.skipIf(sys.platform == "win32", "POSIX signals")
    def test_ctrl_c_reaches_the_program_and_sigterm_is_passed_on(self) -> None:
        # A real launcher process and program (#217); the build is stubbed, and the interpreter wrapper drops the
        # adapter's switch, so no GPU is needed. The program catches SIGINT to "save a checkpoint" for half a second.
        import signal
        import subprocess
        import textwrap
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "program.py").write_text(textwrap.dedent("""
                import signal, sys, time
                def save(signum, frame):
                    time.sleep(0.5)
                    open(sys.argv[1], "w").write(str(signum))
                    sys.exit(42)
                signal.signal(signal.SIGINT, save)
                signal.signal(signal.SIGTERM, save)
                print("ready", flush=True)
                time.sleep(30)
            """))
            (root / "python").write_text(f"#!/bin/sh\nunset CUDA_COMPAT_ENABLE\nexec {sys.executable} \"$@\"\n")
            (root / "python").chmod(0o755)
            (root / "launcher.py").write_text(textwrap.dedent(f"""
                import sys
                from pathlib import Path
                from unittest import mock
                from cuda_legacy_compat import launch
                root = Path({str(root)!r}); artifacts = (root / "a.ptx", root / "b.so", root / "f.so")
                artifacts[0].write_text("ptx")
                with mock.patch.object(launch, "build_adapter", return_value=artifacts):
                    sys.exit(launch.launch(root / "program.py", [sys.argv[1]], llvm_root=root, cuda_root=root,
                                           python=str(root / "python")))
            """))
            src = str(Path(__file__).parents[1] / "src")
            for name, signum, to_group in (("ctrl-c", signal.SIGINT, True), ("sigterm", signal.SIGTERM, False)):
                with self.subTest(name):
                    flag = root / f"{name}.flag"
                    launcher = subprocess.Popen([sys.executable, str(root / "launcher.py"), str(flag)],
                                                env={**os.environ, "PYTHONPATH": src}, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, text=True, start_new_session=True)
                    try:
                        self.assertEqual(launcher.stdout.readline().strip(), "ready")
                        if to_group:
                            os.killpg(launcher.pid, signum)   # what a terminal and Jupyter do
                        else:
                            launcher.send_signal(signum)      # kill <launcher pid>
                        _, err = launcher.communicate(timeout=20)
                    finally:
                        if launcher.poll() is None:
                            os.killpg(launcher.pid, signal.SIGKILL)
                            launcher.communicate()
                    self.assertEqual(flag.read_text() if flag.exists() else None, str(int(signum)))   # handler finished
                    self.assertEqual(launcher.returncode, 42, err)
                    self.assertNotIn("KeyboardInterrupt", err)

    def test_package_import_does_not_eagerly_import_torch(self) -> None:
        init_source = (Path(__file__).parents[1] / "src/cuda_legacy_compat/__init__.py").read_text()
        self.assertNotIn("torch_adapter", init_source)


if __name__ == "__main__":
    unittest.main()
