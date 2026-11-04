from __future__ import annotations

import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from cuda_legacy_compat import cli, routes

ADAPTER = Path(__file__).resolve().parents[1] / "src" / "cuda_legacy_compat" / "torch_adapter.py"
SHEET = Path(__file__).resolve().parents[1] / "site" / "docs" / "compatibility.html"


class RoutesTests(unittest.TestCase):
    def test_table_matches_the_adapter_registrations_exactly(self) -> None:
        source = ADAPTER.read_text(encoding="utf-8")
        registered = set(re.findall(r'_library\.impl\(\s*"([^"]+)"', source))
        self.assertTrue(registered)
        self.assertEqual(routes.registered(), registered)

    def test_counts_match_the_published_support_matrix(self) -> None:
        self.assertEqual(len(routes.CPU_FALLBACK), 9)

    def test_the_compatibility_sheet_names_every_route(self) -> None:
        sheet = SHEET.read_text(encoding="utf-8")
        labels = set(re.findall(r'<td class="op"[^>]*>([^<]+)</td>', sheet))
        codes = set(re.findall(r"<code>([^<]+)</code>", sheet))
        self.assertEqual({operation for _, operation, _ in routes.NATIVE} - labels, set())
        self.assertEqual({name.split(".")[0].lstrip("_") for name, _ in routes.CPU_FALLBACK} - codes, set())
        self.assertIn(f"These {len(routes.CPU_FALLBACK)} registered fallback routes", sheet)

    def test_the_compatibility_sheet_states_each_general_rule(self) -> None:
        # One phrase per routes.NOTES rule; a new rule needs a line here and on the sheet.
        sheet = SHEET.read_text(encoding="utf-8")
        for phrase in ("cuDNN is switched off", "<code>foreach</code>", 'device="cuda"', "<code>x += 1</code>",
                       "PyTorch's own decomposition"):
            self.assertTrue(phrase in sheet, phrase)
        self.assertEqual(len(routes.NOTES), 5, "routes.NOTES changed: update the sheet and this list")

    def test_ops_prints_both_groups_and_json(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["ops"]), 0)
        text = out.getvalue()
        self.assertIn("Runs natively on sm_52", text)
        self.assertIn("Runs on the CPU and copies back (9 operations)", text)
        self.assertTrue(all(len(line) <= 100 for line in text.splitlines()))
        out = io.StringIO()
        with redirect_stdout(out):
            cli.main(["ops", "--json"])
        data = json.loads(out.getvalue())
        self.assertEqual({op["aten"] for op in data["cpu_fallback"]["operations"]}, {name for name, _ in routes.CPU_FALLBACK})


class RunUsageTests(unittest.TestCase):
    def test_missing_script_says_what_to_type(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(cli.main(["run"]), 2)
        self.assertIn("run app.py", err.getvalue())
        self.assertNotIn("ARGS", err.getvalue().splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
