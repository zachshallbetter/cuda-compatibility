from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class FusedDefaultTests(unittest.TestCase):
    """Adam/AdamW left to choose take the fused step on float32 CUDA parameters; an explicit choice is kept (#257)."""

    def test_selection(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter as A

        cuda = [torch.zeros(3), torch.zeros(2, 2)]
        with mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            self.assertTrue(A._fused_by_default(cuda, {}))
            self.assertTrue(A._fused_by_default([{"params": cuda}], {"lr": 1e-3}))
            self.assertTrue(A._fused_by_default([("w", cuda[0])], {}))
            for explicit in ({"fused": False}, {"foreach": True}, {"foreach": False}, {"capturable": True},
                             {"differentiable": True}, {"lr": torch.tensor(1e-3)}):
                self.assertFalse(A._fused_by_default(cuda, explicit), explicit)
            self.assertFalse(A._fused_by_default([torch.zeros(3, dtype=torch.float16)], {}))
            self.assertFalse(A._fused_by_default([], {}))
        self.assertFalse(A._fused_by_default([torch.zeros(3)], {}))  # CPU parameters

    def test_the_hook_fills_fused_only_when_left_open(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter as A

        seen = []
        original = torch.optim.Adam.__init__
        try:
            torch.optim.Adam.__init__ = lambda self, params, *args, **kwargs: seen.append(kwargs.get("fused"))
            A._default_adam_to_fused()
            with mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
                torch.optim.Adam.__init__(None, [torch.zeros(2)], 1e-3)
                torch.optim.Adam.__init__(None, [torch.zeros(2)], 1e-3, foreach=True)
                torch.optim.Adam.__init__(None, iter([torch.zeros(2)]))
            self.assertEqual(seen, [True, None, True])
        finally:
            torch.optim.Adam.__init__ = original


if __name__ == "__main__":
    unittest.main()
