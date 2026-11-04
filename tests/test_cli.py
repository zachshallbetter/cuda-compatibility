from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cuda_legacy_compat import cli
from cuda_legacy_compat.cli import _resolve_profile_path


def sample_profile() -> dict:
    return {
        "schema_version": 1,
        "id": "fixture",
        "name": "Fixture profile",
        "source": "fixture://profile",
        "build": {"cuda_version": "13.0", "host_toolkit_minimum": None},
        "gpu": {"minimum_compute_capability": [7, 5]},
    }


def sample_inventory(capability: str) -> dict:
    major, minor = (int(part) for part in capability.split("."))
    return {
        "gpu_inventory": {
            "status": "available",
            "devices": [
                {
                    "index": 0,
                    "compute_capability": capability,
                    "compute_capability_parts": [major, minor],
                }
            ],
        },
        "nvcc": {"status": "not_found", "version": None},
    }


class ProfilePathTests(unittest.TestCase):
    def test_resolves_installed_profile_by_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir) / "share" / "cuda-legacy-compat" / "profiles"
            profile_dir.mkdir(parents=True)
            profile_file = profile_dir / "pytorch-2.14-cu130.json"
            profile_file.write_text("{}", encoding="utf-8")
            module_file = Path(temp_dir) / "site-packages" / "cuda_legacy_compat" / "cli.py"
            with patch("cuda_legacy_compat.cli.__file__", str(module_file)), patch(
                "cuda_legacy_compat.cli.sysconfig.get_path", return_value=temp_dir
            ):
                resolved = _resolve_profile_path("pytorch-2.14-cu130")
        self.assertEqual(resolved.resolve(), profile_file.resolve())
    def test_resolves_profile_from_pip_target_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target"
            module_dir = target / "cuda_legacy_compat"
            module_dir.mkdir(parents=True)
            profile_dir = target / "share" / "cuda-legacy-compat" / "profiles"
            profile_dir.mkdir(parents=True)
            profile_file = profile_dir / "pytorch-2.14-cu126.json"
            profile_file.write_text("{}", encoding="utf-8")
            module_file = module_dir / "cli.py"
            module_file.write_text("", encoding="utf-8")
            with patch("cuda_legacy_compat.cli.__file__", str(module_file)), patch(
                "cuda_legacy_compat.cli.sysconfig.get_path",
                return_value=str(Path(temp_dir) / "global"),
            ):
                resolved = _resolve_profile_path("pytorch-2.14-cu126")
        self.assertEqual(resolved.resolve(), profile_file.resolve())

    def test_prefers_an_explicit_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_file = Path(temp_dir) / "profile.json"
            profile_file.write_text("{}", encoding="utf-8")
            self.assertEqual(_resolve_profile_path(str(profile_file)), profile_file)


class DoctorSelectionTests(unittest.TestCase):
    def test_numeric_cuda_visible_devices_selects_only_requested_gpu(self) -> None:
        devices = [
            {"index": 0, "compute_capability": "8.6"},
            {"index": 1, "compute_capability": "5.2"},
        ]
        selected, error = cli._selected_gpu_devices(devices, "1")
        self.assertIsNone(error)
        self.assertEqual(selected, [devices[1]])

    def test_unresolvable_gpu_uuid_is_not_reported_as_compatible(self) -> None:
        selected, error = cli._selected_gpu_devices([], "GPU-abc123")
        self.assertIsNone(selected)
        self.assertIn("cannot resolve", error)


class CliOutputTests(unittest.TestCase):
    def test_version_flag_prints_package_version(self) -> None:
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            cli.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"cuda-legacy {cli.__version__}")

    def test_short_command_uses_its_name_in_version_and_help(self) -> None:
        output = StringIO()
        with patch("cuda_legacy_compat.cli.sys.argv", ["cuda-legacy"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"cuda-legacy {cli.__version__}")

        output = StringIO()
        with patch("cuda_legacy_compat.cli.sys.argv", ["cuda-legacy"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("cuda-legacy inspect", output.getvalue())

    def test_incompatible_profile_returns_nonzero_and_machine_readable_json(self) -> None:
        output = StringIO()
        with patch.object(cli, "load_profile", return_value=sample_profile()), patch.object(
            cli, "inspect_system", return_value=sample_inventory("5.2")
        ), redirect_stdout(output):
            exit_code = cli.main(["audit", "fixture.json"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "incompatible")


    def test_run_forwards_script_arguments_and_toolchain_paths(self) -> None:
        with patch.object(cli, "launch", return_value=7) as run:
            exit_code = cli.main([
                "run", "--llvm-root", "/opt/llvm", "--cuda-root", "/opt/cuda",
                "--", "app.py", "--mode", "fast",
            ])
        self.assertEqual(exit_code, 7)
        args, kwargs = run.call_args
        self.assertEqual(args[0], Path("app.py"))
        self.assertEqual(args[1], ["--mode", "fast"])
        self.assertEqual(kwargs["llvm_root"], Path("/opt/llvm"))
        self.assertEqual(kwargs["cuda_root"], Path("/opt/cuda"))

    def test_run_takes_module_code_and_interactive_like_python(self) -> None:
        cases = [
            (["run", "-m", "pytest", "-x", "tests/"], dict(script=None, args=["-x", "tests/"], module="pytest")),
            (["run", "-c", "import torch", "a"], dict(script=None, args=["a"], code="import torch")),
            (["run", "-i"], dict(script=None, args=[], interactive=True)),
            (["run", "-i", "-m", "mypkg.train"], dict(script=None, args=[], module="mypkg.train", interactive=True)),
            # After a script, -m belongs to the script, as with python.
            (["run", "app.py", "-m", "x"], dict(script=Path("app.py"), args=["-m", "x"], module=None)),
        ]
        for argv, expected in cases:
            with self.subTest(argv=argv), patch.object(cli, "launch", return_value=0) as run, \
                    patch.object(cli, "_locations", return_value=("/opt/llvm", "/opt/cuda", None)):
                self.assertEqual(cli.main(argv), 0)
                args, kwargs = run.call_args
                self.assertEqual(args[0], expected["script"])
                self.assertEqual(args[1], expected["args"])
                for key in ("module", "code", "interactive"):
                    if key in expected:
                        self.assertEqual(kwargs[key], expected[key])

    def test_run_needs_a_program_and_a_value_for_m_and_c(self) -> None:
        for argv in (["run"], ["run", "-m"], ["run", "-c"]):
            with self.subTest(argv=argv), patch.object(cli, "launch") as run, redirect_stderr(StringIO()) as err:
                self.assertEqual(cli.main(argv), 2)
                run.assert_not_called()
            self.assertIn("run", err.getvalue())

    def test_compatible_profile_returns_zero(self) -> None:
        output = StringIO()
        with patch.object(cli, "load_profile", return_value=sample_profile()), patch.object(
            cli, "inspect_system", return_value=sample_inventory("8.6")
        ), redirect_stdout(output):
            exit_code = cli.main(["audit", "fixture.json"])
        self.assertEqual(exit_code, 0)

    def test_profiles_list_json_includes_source_backed_profiles(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["profiles", "list", "--json"])
        self.assertEqual(exit_code, 0)
        profiles = json.loads(output.getvalue())
        self.assertIn("pytorch-2.14-cu130", {profile["id"] for profile in profiles})
        self.assertTrue(all(profile["source"].startswith("https://") for profile in profiles))

    def test_profiles_show_prints_target_boundary(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["profiles", "show", "pytorch-2.14-cu130"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Minimum GPU capability: 7.5", output.getvalue())
        self.assertIn("github.com/pytorch/pytorch", output.getvalue())

    def test_profiles_show_missing_profile_returns_actionable_error(self) -> None:
        output = StringIO()
        with patch("sys.stderr", output):
            exit_code = cli.main(["profiles", "show", "missing-profile"])
        self.assertEqual(exit_code, 2)
        self.assertIn("profile error:", output.getvalue())

    def test_doctor_unknown_environment_returns_needs_attention(self) -> None:
        report = {
            "schema_version": 1,
            "status": "needs_attention",
            "checks": [
                {
                    "id": "gpu",
                    "status": "not_ready",
                    "detail": "GPU inventory is not_found",
                    "next_step": "Install or enable the NVIDIA driver.",
                }
            ],
            "note": "Ready checks are not a workload certification.",
        }
        output = StringIO()
        with patch.object(cli, "_doctor_report", return_value=report), redirect_stdout(output):
            exit_code = cli.main(["doctor", "--json"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "needs_attention")

    def test_doctor_does_not_mark_missing_gpu_ready(self) -> None:
        inventory = {
            "gpu_inventory": {"status": "not_found", "devices": []},
            "nvcc": {"status": "not_found", "version": None},
        }
        with patch.object(cli, "inspect_system", return_value=inventory), patch.object(
            cli.importlib.metadata,
            "version",
            side_effect=cli.importlib.metadata.PackageNotFoundError,
        ):
            report = cli._doctor_report(None, "/missing-cuda")
        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(report["status"], "needs_attention")
        self.assertEqual(checks["gpu"]["status"], "not_ready")
        self.assertIn("nvidia-smi", checks["gpu"]["next_step"])

    def test_doctor_queries_the_interpreter_named_by_python(self) -> None:
        inventory = {
            "gpu_inventory": {"status": "not_found", "devices": []},
            "nvcc": {"status": "not_found", "version": None},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            fake = Path(temp_dir) / "python"
            fake.write_text(
                "#!/bin/sh\n"
                "echo '{\"version\": [3, 10, 12], \"torch\": \"2.14.0+cu130\"}'\n"
            )
            fake.chmod(0o755)
            with patch.object(cli, "inspect_system", return_value=inventory):
                report = cli._doctor_report(None, "/missing-cuda", str(fake))
        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["python"]["status"], "ready")
        self.assertEqual(checks["pytorch"]["status"], "ready")

    def test_doctor_reports_unreachable_interpreter_without_guessing_pytorch(self) -> None:
        inventory = {
            "gpu_inventory": {"status": "not_found", "devices": []},
            "nvcc": {"status": "not_found", "version": None},
        }
        with patch.object(cli, "inspect_system", return_value=inventory):
            report = cli._doctor_report(None, "/missing-cuda", "/nonexistent/python")
        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["python"]["status"], "not_ready")
        self.assertIn("--python", checks["python"]["next_step"])
        self.assertNotIn("pytorch", checks)

    def test_help_command_shows_run_usage(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["help", "run"])
        self.assertEqual(exit_code, 0)
        self.assertIn("the Python script to run", output.getvalue())
        self.assertIn("a -- before them is optional", " ".join(output.getvalue().split()))


if __name__ == "__main__":
    unittest.main()
