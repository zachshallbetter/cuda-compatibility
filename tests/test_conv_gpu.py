from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class ConvDecompositionTests(unittest.TestCase):
    """_conv_on_gpu's arithmetic (selection-matrix padding, im2col, transposed conv), checked on the CPU."""

    def test_forward_and_backward_match_pytorch(self) -> None:
        import torch
        import torch.nn.functional as F
        from cuda_legacy_compat import torch_adapter as A

        aten = torch.ops.aten
        torch.manual_seed(0)
        cases = [((2, 3, 7, 7), 4, (3, 3), (1, 1), (1, 1), (1, 1)), ((2, 3, 8, 9), 5, (3, 3), (2, 2), (1, 1), (1, 1)),
                 ((1, 2, 9, 9), 3, (3, 3), (2, 2), (2, 2), (2, 2)), ((2, 4, 6, 6), 2, (1, 1), (1, 1), (0, 0), (1, 1)),
                 ((1, 3, 10, 7), 4, (3, 5), (1, 2), (1, 2), (1, 1))]
        with mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
            for shape, co, k, s, p, d in cases:
                x, w, b = torch.randn(shape), torch.randn(co, shape[1], *k), torch.randn(co)
                want = F.conv2d(x, w, b, s, p, d)
                handled, got = A._conv_on_gpu("aten.convolution.default", (x, w, b, list(s), list(p), list(d), False, [0, 0], 1))
                self.assertTrue(handled)
                self.assertLess(float((got - want).abs().max()), 1e-4)
                g = torch.randn_like(want)
                args = (g, x, w, [co], list(s), list(p), list(d), False, [0, 0], 1, [True, True, True])
                _, got_grads = A._conv_on_gpu("aten.convolution_backward.default", args)
                for have, ref in zip(got_grads, aten.convolution_backward.default(*args)):
                    self.assertLess(float((have - ref).abs().max()), 1e-4, (shape, k, s, p, d))
            x = torch.randn(1, 4, 5, 5)
            self.assertEqual(A._conv_on_gpu("aten.convolution.default", (x, torch.randn(4, 2, 3, 3), None, [1, 1], [0, 0],
                                                                           [1, 1], False, [0, 0], 2)), (False, None))


if __name__ == "__main__":
    unittest.main()
