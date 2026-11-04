from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class ReluInPlaceTests(unittest.TestCase):
    """relu_ uses the native kernel in place; a case it can't take is refused so the CPU route still runs it."""

    def test_cases_the_kernel_cannot_take_are_refused(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter as A

        for tensor in (torch.zeros(4, 4).t(), torch.zeros(4, dtype=torch.float64)):
            with mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)), self.assertRaises(NotImplementedError):
                A._relu.__wrapped__(tensor, tensor)


if __name__ == "__main__":
    unittest.main()
