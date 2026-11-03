from __future__ import annotations

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
                launch_module.subprocess, "run"
            ) as run:
                run.return_value.returncode = 0
                launch_module.launch(
                    script, ["--x"], llvm_root=root / "llvm", cuda_root=root / "cuda",
                    python=sys.executable,
                )
            self.assertEqual(run.call_args.args[0][:3], [sys.executable, str(script.resolve()), "--x"])

    def test_package_import_does_not_eagerly_import_torch(self) -> None:
        init_source = (Path(__file__).parents[1] / "src/cuda_legacy_compat/__init__.py").read_text()
        self.assertNotIn("torch_adapter", init_source)


if __name__ == "__main__":
    unittest.main()
