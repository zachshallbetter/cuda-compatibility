from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class AddScalarTests(unittest.TestCase):
    """x + 1, 1 + x and x + torch.tensor(1.0) reach the native add with a scalar operand."""

    def test_a_scalar_operand_takes_the_cpu_fallback(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter

        seen = []

        def fallback(operation, name):
            return lambda *args, **kwargs: seen.append((name, args[1], kwargs)) or "fallback"

        x = torch.ones(3)
        with mock.patch.object(torch_adapter, "_cpu_fallback", fallback):
            for other, scalar in ((1, 1), (0.5, 0.5), (torch.tensor(2.0), 2.0)):
                self.assertEqual(torch_adapter._add(x, other, alpha=3), "fallback")
                self.assertEqual(seen[-1], ("add", scalar, {"alpha": 3}))


if __name__ == "__main__":
    unittest.main()
