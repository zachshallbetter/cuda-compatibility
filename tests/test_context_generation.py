import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen-context.py"
SPEC = importlib.util.spec_from_file_location("gen_context", SCRIPT)
GEN_CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN_CONTEXT)


class ContextSourceSelectionTests(unittest.TestCase):
    def test_includes_untracked_sources_and_omits_ignored_build_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("*.egg-info/\n", encoding="utf-8")
            (root / "tracked.md").write_text("tracked source\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.md"], check=True)
            (root / "new.md").write_text("untracked source\n", encoding="utf-8")
            generated = root / "package.egg-info"
            generated.mkdir()
            (generated / "PKG-INFO").write_text("generated metadata\n", encoding="utf-8")

            policy = {
                "excludeDirectories": [],
                "excludePaths": [],
                "includeExtensions": [".md"],
                "maxFileBytes": 1024,
            }
            with patch.object(GEN_CONTEXT, "ROOT", root):
                sources, _ = GEN_CONTEXT.source_files(policy)

        self.assertEqual([path for path, _ in sources], ["new.md", "tracked.md"])


if __name__ == "__main__":
    unittest.main()
