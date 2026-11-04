from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

COMPAT = Path(__file__).resolve().parents[1] / "src" / "cuda_legacy_compat" / "compat"


class StartupHookTests(unittest.TestCase):
    """The startup hook as the launcher runs it: sitecustomize first on the path, adapter enabled."""

    def start(self, failure: str | None, **env: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as stubs:
            package = Path(stubs) / "cuda_legacy_compat"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            install = f"raise RuntimeError({failure!r})" if failure else "pass"
            (package / "torch_adapter.py").write_text(f"def install():\n    {install}\n", encoding="utf-8")
            (package / "unrouted.py").write_text("def install_excepthook():\n    pass\n", encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-c", "print('script ran')"], capture_output=True, text=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{COMPAT}:{stubs}", "CUDA_COMPAT_ENABLE": "1", **env},
            )

    def test_a_failed_start_is_one_line_and_exit_78(self) -> None:
        result = self.start("CUDA is unavailable", CUDA_VISIBLE_DEVICES="")
        self.assertEqual(result.returncode, 78)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.strip().splitlines(), [
            "cuda-legacy: the adapter could not start: no GPU is visible to this process "
            "(CUDA_VISIBLE_DEVICES is set but empty). Run `cuda-legacy doctor` to see what is missing.",
        ])

    def test_names_the_cause_in_the_users_terms(self) -> None:
        unreachable = self.start("CUDA is unavailable")
        self.assertIn("PyTorch cannot reach a CUDA GPU from this process", unreachable.stderr)
        mixed = self.start("adapter requires only sm_52 devices; observed [(5, 2), (6, 1)]")
        self.assertIn("select only sm_52 cards with CUDA_VISIBLE_DEVICES", mixed.stderr)
        self.assertNotIn("adapter active", mixed.stderr)

    def test_verbose_keeps_the_traceback(self) -> None:
        result = self.start("CUDA is unavailable", CUDA_COMPAT_VERBOSE="1")
        self.assertEqual(result.returncode, 78)
        self.assertIn("Traceback", result.stderr)
        self.assertNotIn("Fatal Python error", result.stderr)

    def test_a_good_start_prints_the_banner_and_runs_the_script(self) -> None:
        result = self.start(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "script ran")
        self.assertIn("adapter active for sm_52", result.stderr)

    def test_not_enabled_does_nothing(self) -> None:
        result = subprocess.run([sys.executable, "-c", "print('script ran')"], capture_output=True, text=True,
                                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(COMPAT)})
        self.assertEqual((result.returncode, result.stdout.strip(), result.stderr), (0, "script ran", ""))


if __name__ == "__main__":
    unittest.main()
