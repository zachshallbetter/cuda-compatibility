from __future__ import annotations

import io
import subprocess
import sys
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from cuda_legacy_compat import __version__, unrouted

SRC = Path(__file__).resolve().parents[1] / "src"


class UnroutedHintTests(unittest.TestCase):
    def setUp(self) -> None:
        unrouted._recorded = None

    def test_names_the_operation_and_the_next_step(self) -> None:
        error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        unrouted.record("aten::ones", error)
        hint = unrouted.hint_for(error)
        self.assertIn("aten::ones has no sm_52 route", hint)
        self.assertIn(f"cuda-legacy {__version__}", hint)
        self.assertIn("`cuda-legacy ops` lists what runs", hint)
        self.assertIn(unrouted.COVERAGE_URL, hint)
        self.assertIn(unrouted.REPORT_URL, hint)

    def test_follows_errors_raised_from_the_recorded_one(self) -> None:
        recorded = RuntimeError("no kernel image is available")
        unrouted.record("aten::ones", recorded)
        try:
            try:
                raise recorded
            except RuntimeError as exc:
                raise ValueError("model failed to build") from exc
        except ValueError as wrapped:
            self.assertIsNotNone(unrouted.hint_for(wrapped))

    def test_unrelated_errors_get_no_hint(self) -> None:
        unrouted.record("aten::ones", RuntimeError("no kernel image is available"))
        self.assertIsNone(unrouted.hint_for(KeyError("config")))
        unrouted._recorded = None
        self.assertIsNone(unrouted.hint_for(RuntimeError("no kernel image is available")))

    def test_hook_prints_the_traceback_then_one_hint(self) -> None:
        error = RuntimeError("no kernel image is available")
        unrouted.record("aten::ones", error)
        previous = mock.Mock()
        with mock.patch.object(sys, "excepthook", previous):
            unrouted.install_excepthook()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                sys.excepthook(RuntimeError, error, None)
        previous.assert_called_once_with(RuntimeError, error, None)
        self.assertEqual(stderr.getvalue().count("has no sm_52 route"), 1)

    def test_an_unhandled_error_still_exits_1_with_its_traceback(self) -> None:
        script = textwrap.dedent("""
            from cuda_legacy_compat import unrouted
            unrouted.install_excepthook()
            error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
            unrouted.record("aten::ones", error)
            raise error
        """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("Traceback", result.stderr)
        self.assertIn("RuntimeError: CUDA error: no kernel image", result.stderr)
        self.assertTrue(result.stderr.rstrip().splitlines()[-1].startswith("cuda-legacy: aten::ones has no sm_52 route"))


if __name__ == "__main__":
    unittest.main()
