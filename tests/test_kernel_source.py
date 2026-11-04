from __future__ import annotations

import re
import unittest
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[1] / "src" / "cuda_legacy_compat" / "compat" / "compat_kernels.cu"


class KernelSourceTests(unittest.TestCase):
    """compat_kernels.cu is compiled only on an sm_52 host; catch a broken merge of it here first."""

    def test_braces_balance_and_no_conflict_markers_remain(self) -> None:
        source = KERNELS.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^(<<<<<<<|=======|>>>>>>>)", source, re.M))
        code = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", source, flags=re.S)
        depth = 0
        for character in code:
            depth += {"{": 1, "}": -1}.get(character, 0)
            self.assertGreaterEqual(depth, 0)
        self.assertEqual(depth, 0)

    def test_every_kernel_begins_at_file_scope(self) -> None:
        source = KERNELS.read_text(encoding="utf-8")
        for match in re.finditer(r'extern "C" __global__', source):
            before = re.sub(r"//[^\n]*", "", source[:match.start()])
            self.assertEqual(before.count("{"), before.count("}"), source[match.start():match.start() + 80])


if __name__ == "__main__":
    unittest.main()
