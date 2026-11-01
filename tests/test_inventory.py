from __future__ import annotations

import unittest

from cuda_legacy_compat.inventory import parse_gpu_inventory


class InventoryParserTests(unittest.TestCase):
    def test_parses_multiple_gpu_rows(self) -> None:
        devices = parse_gpu_inventory(
            "0, NVIDIA GeForce GTX TITAN X, 5.2, 580.178.04, 12288 MiB\n"
            "1, NVIDIA GeForce GTX TITAN X, 5.2, 580.178.04, 12288 MiB\n"
        )
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["compute_capability_parts"], [5, 2])
        self.assertEqual(devices[1]["memory_total_mib"], 12288)

    def test_rejects_unexpected_row_shape(self) -> None:
        with self.assertRaises(ValueError):
            parse_gpu_inventory("0, NVIDIA GPU\n")

    def test_rejects_invalid_compute_capability(self) -> None:
        with self.assertRaises(ValueError):
            parse_gpu_inventory("0, NVIDIA GPU, 5.x, 580.1, 12288 MiB\n")


if __name__ == "__main__":
    unittest.main()
