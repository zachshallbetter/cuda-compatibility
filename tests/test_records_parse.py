from __future__ import annotations

import json
import unittest
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[1] / ".agents" / "records"


class RecordsParseTests(unittest.TestCase):
    """Every committed ledger line is one JSON object; a merge conflict left in a ledger fails here."""

    def test_every_jsonl_line_parses(self) -> None:
        ledgers = sorted(RECORDS.glob("*.jsonl"))
        self.assertTrue(ledgers)
        for ledger in ledgers:
            for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                self.assertFalse(line.startswith(("<<<<<<<", "=======", ">>>>>>>")), f"{ledger.name}:{number}: conflict marker")
                try:
                    self.assertIsInstance(json.loads(line), dict, f"{ledger.name}:{number}")
                except json.JSONDecodeError as error:
                    self.fail(f"{ledger.name}:{number}: {error}")


if __name__ == "__main__":
    unittest.main()
