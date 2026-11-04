from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# A launcher that runs the program through launch._run_forwarding_signals, as `cuda-legacy run` does.
LAUNCHER = textwrap.dedent("""
    import sys
    from cuda_legacy_compat.launch import _run_forwarding_signals
    sys.exit(_run_forwarding_signals([sys.executable, "-c", sys.argv[1]], dict(__import__("os").environ)))
""")
PROGRAM = textwrap.dedent("""
    import signal, sys, time
    def stop(number, frame):
        print("program saved its checkpoint", flush=True)
        sys.exit(7)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("ready", flush=True)
    time.sleep(30)
""")


@unittest.skipUnless(hasattr(os, "killpg"), "needs POSIX process groups")
class RunSignalTests(unittest.TestCase):
    def start(self) -> subprocess.Popen:
        launcher = subprocess.Popen([sys.executable, "-c", LAUNCHER, PROGRAM], stdout=subprocess.PIPE, text=True,
                                    env={**os.environ, "PYTHONPATH": str(SRC)}, start_new_session=True)
        self.assertEqual(launcher.stdout.readline().strip(), "ready")
        return launcher

    def finish(self, launcher: subprocess.Popen) -> tuple[int, str]:
        out, _ = launcher.communicate(timeout=20)
        return launcher.returncode, out

    def test_ctrl_c_reaches_the_program_and_its_exit_code_comes_back(self) -> None:
        launcher = self.start()
        os.killpg(launcher.pid, signal.SIGINT)  # what a terminal's Ctrl-C does
        code, out = self.finish(launcher)
        self.assertEqual(code, 7)
        self.assertIn("program saved its checkpoint", out)

    def test_sigterm_to_the_launcher_alone_stops_the_program(self) -> None:
        launcher = self.start()
        launcher.send_signal(signal.SIGTERM)
        code, out = self.finish(launcher)
        self.assertEqual(code, 7)
        self.assertIn("program saved its checkpoint", out)

    def test_a_signal_death_is_reported_as_128_plus_n(self) -> None:
        launcher = subprocess.Popen([sys.executable, "-c", LAUNCHER, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
                                    env={**os.environ, "PYTHONPATH": str(SRC)})
        self.assertEqual(launcher.wait(timeout=20), 128 + signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
