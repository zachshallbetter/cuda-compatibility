from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from scripts import reprobe_stock

RECORDS = Path(__file__).resolve().parents[1] / ".agents" / "records" / "negative-results.jsonl"


class ReprobeTests(unittest.TestCase):
    def test_each_probe_names_a_recorded_contradicted_claim(self) -> None:
        ids = {json.loads(line)["id"] for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()}
        self.assertLessEqual(set(reprobe_stock.CLAIMS.values()), ids)

    def test_a_probe_that_runs_reopens_its_claim(self) -> None:
        self.assertEqual(reprobe_stock.verdict(None), "reopen: the stock wheel ran it")
        self.assertEqual(reprobe_stock.verdict("RuntimeError: no kernel image"), "still contradicted")

    def test_refuses_to_run_under_the_adapter(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_COMPAT_ENABLE": "1"}):
            with mock.patch("sys.stderr"):
                self.assertEqual(reprobe_stock.main([]), 2)


if __name__ == "__main__":
    unittest.main()
