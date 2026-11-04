from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from cuda_legacy_compat import cli, jupyter, settings


class JupyterKernelTests(unittest.TestCase):
    """`cuda-legacy jupyter` writes a kernel spec that starts through `run -m ipykernel_launcher` (#225)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cuda-legacy-jupyter-test-"))
        self.env = mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(self.tmp / "data"),
                                                "XDG_CONFIG_HOME": str(self.tmp / "config")})
        self.env.start()
        for name in ("JUPYTER_DATA_DIR", "CUDA_COMPAT_PYTHON"):
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        self.env.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_registers_a_kernel_that_starts_through_run_and_removes_only_its_own(self) -> None:
        with mock.patch.object(jupyter, "has_ipykernel", return_value=True):
            code, out, err = self.run_cli("jupyter", "--python", sys.executable)
        self.assertEqual(code, 0, err)
        target = self.tmp / "data" / "jupyter" / "kernels" / "cuda-legacy"
        spec = json.loads((target / "kernel.json").read_text())
        self.assertEqual(spec["argv"][1:], ["run", "-m", "ipykernel_launcher", "-f", "{connection_file}"])
        self.assertTrue(spec["argv"][0].endswith("cuda-legacy"))
        self.assertEqual(spec["display_name"], "Python (cuda-legacy, sm_52)")
        self.assertEqual(spec["interrupt_mode"], "signal")
        self.assertIn("Pick it from Jupyter's kernel menu", out)
        code, out, _ = self.run_cli("jupyter", "--remove")
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        self.assertIn("Removed the Jupyter kernel", out)

    def test_names_the_ipykernel_install_when_it_is_missing(self) -> None:
        with mock.patch.object(jupyter, "has_ipykernel", return_value=False):
            code, _, err = self.run_cli("jupyter", "--python", "/env/bin/python")
        self.assertEqual(code, 2)
        self.assertIn("/env/bin/python -m pip install ipykernel", err)
        self.assertFalse(jupyter.spec_dir().exists())

    def test_never_replaces_or_removes_a_kernel_it_did_not_write(self) -> None:
        own = jupyter.spec_dir()
        own.mkdir(parents=True)
        (own / "kernel.json").write_text("{}")
        with mock.patch.object(jupyter, "has_ipykernel", return_value=True):
            code, _, err = self.run_cli("jupyter", "--python", sys.executable)
        self.assertEqual(code, 2)
        self.assertIn("did not create it", err)
        self.assertEqual(self.run_cli("jupyter", "--remove")[0], 0)
        self.assertEqual((own / "kernel.json").read_text(), "{}")

    def test_jupyter_data_dir_wins_and_setup_remove_deletes_the_kernel(self) -> None:
        os.environ["JUPYTER_DATA_DIR"] = str(self.tmp / "jd")
        with mock.patch.object(jupyter, "has_ipykernel", return_value=True):
            target = jupyter.install(sys.executable)
        self.assertEqual(target, self.tmp / "jd" / "kernels" / "cuda-legacy")
        settings.save({"python": sys.executable})
        code, out, _ = self.run_cli("setup", "--remove")
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        self.assertIn(str(target), out)


if __name__ == "__main__":
    unittest.main()
