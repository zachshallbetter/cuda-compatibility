from __future__ import annotations

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

    def test_package_import_does_not_eagerly_import_torch(self) -> None:
        init_source = (Path(__file__).parents[1] / "src/cuda_legacy_compat/__init__.py").read_text()
        self.assertNotIn("torch_adapter", init_source)


if __name__ == "__main__":
    unittest.main()
