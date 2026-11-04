from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "site" / "install.sh"
STUBS = {
    # The installer reads only these from the host; the stubs pretend to be a Linux GPU box.
    "uname": '#!/bin/sh\ncase $1 in -s) echo "${FAKE_OS:-Linux}" ;; -m) echo "${FAKE_ARCH:-x86_64}" ;; *) echo Linux ;; esac\n',
    "id": '#!/bin/sh\necho "${FAKE_UID:-1000}"\n',
    "nvidia-smi": "#!/bin/sh\nexit 0\n",
}


def _script_value(name: str) -> str:
    match = re.search(rf"^{name}=(.*)$", SCRIPT.read_text(encoding="utf-8"), re.M)
    return match.group(1).strip('"') if match else ""


class InstallScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="cuda-legacy-install-test-"))
        cls.bin = cls.tmp / "stubs"
        cls.bin.mkdir()
        for name, body in STUBS.items():
            stub = cls.bin / name
            stub.write_text(body, encoding="utf-8")
            stub.chmod(0o755)
        # A virtual environment whose "torch" is a CUDA build, as far as the installer can tell.
        # Only the end-to-end test runs pip, so the shared environment is made without it.
        cls.env = cls.make_torch_env("torch-env", with_pip=False)
        cls.python = cls.env / "bin" / "python"
        cls.plain = cls.tmp / "plain-env"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(cls.plain)], check=True)

    @classmethod
    def make_torch_env(cls, name: str, with_pip: bool) -> Path:
        env = cls.tmp / name
        command = [sys.executable, "-m", "venv", str(env)] + ([] if with_pip else ["--without-pip"])
        subprocess.run(command, check=True)
        site = subprocess.run([str(env / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                              check=True, capture_output=True, text=True).stdout.strip()
        (Path(site) / "torch").mkdir()
        (Path(site) / "torch" / "__init__.py").write_text('__version__ = "2.14.0+cu130"\n', encoding="utf-8")
        info = Path(site) / "torch-2.14.0+cu130.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.14.0+cu130\n", encoding="utf-8")
        return env

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(dir=self.tmp, prefix="home-"))

    def run_script(self, *args: str, stdin: str | None = None, **env: str) -> subprocess.CompletedProcess[str]:
        base = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home), "LANG": "C"}
        command = ["sh", "-s", "--", *args] if stdin is not None else ["sh", str(SCRIPT), *args]
        # A new session has no controlling terminal, as when a CI job pipes the script into sh.
        return subprocess.run(command, input=stdin, capture_output=True, text=True, env={**base, **env},
                              start_new_session=True, timeout=600)

    def home_is_empty(self) -> bool:
        return not any(self.home.iterdir())

    def test_shellcheck_is_clean(self) -> None:
        if not shutil.which("shellcheck"):
            self.skipTest("shellcheck is not installed")
        result = subprocess.run(["shellcheck", "-s", "sh", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_published_checksum_matches_the_script(self) -> None:
        published = (SCRIPT.parent / "install.sh.sha256").read_text(encoding="utf-8").split()
        self.assertEqual(published, [hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), "install.sh"])

    def test_pinned_release_is_the_package_version(self) -> None:
        version = re.search(r'^version = "(.+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
        pinned, sha = _script_value("VERSION"), _script_value("ARCHIVE_SHA256")
        if sha:
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertEqual(pinned, version)
        else:
            # Until the release is cut, the script names the version that release will carry.
            as_tuple = lambda v: tuple(int(part) for part in v.split("."))
            self.assertGreaterEqual(as_tuple(pinned), as_tuple(version))

    def test_a_partial_download_runs_nothing(self) -> None:
        lines = SCRIPT.read_text(encoding="utf-8").splitlines(keepends=True)
        cuts = range(1, len(lines))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = pool.map(lambda cut: self.run_script(stdin="".join(lines[:cut])), cuts)
            for cut, result in zip(cuts, results):
                self.assertNotIn("==>", result.stdout, f"cut after line {cut}")
        self.assertTrue(self.home_is_empty())

    def test_version_and_help(self) -> None:
        self.assertEqual(self.run_script("--version").stdout.strip(), f"cuda-legacy installer {_script_value('VERSION')}")
        piped_help = self.run_script("--help", stdin=SCRIPT.read_text(encoding="utf-8"))
        self.assertEqual(piped_help.returncode, 0)
        self.assertIn("--uninstall", piped_help.stdout)

    def test_unknown_option_stops(self) -> None:
        result = self.run_script("--frobnicate")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown option: --frobnicate", result.stderr)

    def test_refuses_root(self) -> None:
        result = self.run_script("--dry-run", FAKE_UID="0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not root or sudo", result.stderr)

    def test_refuses_other_platforms(self) -> None:
        result = self.run_script("--dry-run", FAKE_OS="Darwin", FAKE_ARCH="arm64")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Linux x86_64", result.stderr)
        self.assertIn("Darwin arm64", result.stderr)

    def test_explains_a_missing_pytorch(self) -> None:
        result = self.run_script("--dry-run", "--python", str(self.plain / "bin" / "python"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("has no PyTorch CUDA wheel", result.stderr)
        self.assertIn("download.pytorch.org/whl/cu130", result.stderr)

    def test_finds_the_active_environment(self) -> None:
        result = self.run_script("--dry-run", "--yes", VIRTUAL_ENV=str(self.env), CUDA_LEGACY_ARCHIVE_SHA256="0" * 64)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.python), result.stdout)
        self.assertIn("setup --python", result.stdout)
        self.assertIn("Dry run: nothing was changed", result.stdout)
        self.assertTrue(self.home_is_empty())

    def test_says_whether_it_installs_upgrades_or_reinstalls(self) -> None:
        pinned = _script_value("VERSION")
        cases = [(None, f"Installing cuda-legacy {pinned} into"), ("0.1.0", f"Upgrading cuda-legacy 0.1.0 to {pinned} in"),
                 (pinned, f"Reinstalling cuda-legacy {pinned} in"), ("9.0.0", f"Replacing cuda-legacy 9.0.0 with {pinned} in")]
        for installed, expected in cases:
            env = self.make_torch_env(f"torch-env-{installed}", with_pip=False)
            if installed:
                site = next((env / "lib").glob("python*/site-packages"))
                info = site / f"cuda_legacy_compat-{installed}.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: cuda-legacy-compat\nVersion: {installed}\n", encoding="utf-8")
            result = self.run_script("--dry-run", "--yes", "--python", str(env / "bin" / "python"), CUDA_LEGACY_ARCHIVE_SHA256="0" * 64)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)

    def test_asks_for_yes_without_a_terminal(self) -> None:
        result = self.run_script("--python", str(self.python), CUDA_LEGACY_ARCHIVE_SHA256="0" * 64)
        self.assertEqual(result.returncode, 2)
        self.assertIn("sh -s -- --yes", result.stderr)

    def test_unpinned_release_stops_before_downloading(self) -> None:
        if _script_value("ARCHIVE_SHA256"):
            self.skipTest("a release is pinned")
        result = self.run_script("--yes", "--python", str(self.python))
        self.assertEqual(result.returncode, 2)
        self.assertIn("no release pinned", result.stderr)

    def test_checksum_mismatch_installs_nothing(self) -> None:
        archive = self.tmp / "tampered.tar.gz"
        archive.write_bytes(b"not the release")
        result = self.run_script("--yes", "--python", str(self.python),
                                 CUDA_LEGACY_ARCHIVE=str(archive), CUDA_LEGACY_ARCHIVE_SHA256="0" * 64)
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not match the published checksum", result.stderr)
        self.assertIn("Nothing was installed", result.stderr)
        check = subprocess.run([str(self.python), "-c", "import cuda_legacy_compat"], capture_output=True,
                               env={"PATH": "/usr/bin:/bin"}, cwd=self.tmp)
        self.assertNotEqual(check.returncode, 0)

    def test_uninstall_with_nothing_installed(self) -> None:
        result = self.run_script("--uninstall", VIRTUAL_ENV=str(self.env))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Nothing to remove", result.stdout)

    @unittest.skipUnless(os.environ.get("CUDA_LEGACY_INSTALL_E2E") == "1",
                         "set CUDA_LEGACY_INSTALL_E2E=1 to install this checkout for real (needs network for pip)")
    def test_install_then_uninstall_this_checkout(self) -> None:
        # The archive GitHub serves for a tag is `git archive` of that commit with this prefix.
        archive = self.tmp / "release.tar.gz"
        subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar.gz",
                        f"--prefix=cuda-compatibility-{_script_value('VERSION')}/", "-o", str(archive), "HEAD"], check=True)
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        env = {"CUDA_LEGACY_ARCHIVE": str(archive), "CUDA_LEGACY_ARCHIVE_SHA256": sha}
        torch_env = self.make_torch_env("torch-env-pip", with_pip=True)
        python = torch_env / "bin" / "python"
        installed = self.run_script("--yes", "--python", str(python), "--", "--dry-run", **env)
        self.assertIn("SHA-256 verified", installed.stdout)
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
        self.assertTrue((torch_env / "bin" / "cuda-legacy").is_file())
        removed = self.run_script("--uninstall", "--python", str(python))
        self.assertEqual(removed.returncode, 0, removed.stdout + removed.stderr)
        self.assertFalse((torch_env / "bin" / "cuda-legacy").exists())
        self.assertTrue(self.home_is_empty(), list(self.home.rglob("*")))


if __name__ == "__main__":
    unittest.main()
