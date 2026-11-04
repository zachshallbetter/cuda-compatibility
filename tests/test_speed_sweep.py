from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU sweep")
class SpeedSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        from scripts import speed_sweep

        self.sweep = speed_sweep

    def test_every_model_trains_and_reports_its_measurements(self) -> None:
        import torch

        for model in self.sweep.MODELS.values():
            row = self.sweep.measure(model, torch.device("cpu"), steps=1)
            self.assertEqual(row["model"], model.name)
            self.assertGreater(row["step_s"], 0)
            self.assertGreater(row[f"train_{model.unit}_per_s"], 0)
            self.assertEqual(row["host_copy_mib_per_step"], 0)

    def test_the_json_carries_the_environment_and_rejects_unknown_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep.json"
            self.assertEqual(self.sweep.main(["--models", "mlp", "--steps", "1", "--json", str(out)]), 0)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([row["model"] for row in data["results"]], ["mlp"])
        self.assertIn("torch", data["environment"])
        with self.assertRaises(SystemExit):
            self.sweep.main(["--models", "nope"])


if __name__ == "__main__":
    unittest.main()
