from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class BatchNormDecompositionTests(unittest.TestCase):
    """_batch_norm_on_gpu's arithmetic, checked on the CPU against PyTorch's own batch norm."""

    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter as A

        self.torch, self.A = torch, A
        sqrt = lambda op, x, out=None, **_: out.copy_(x.sqrt())  # no GPU here: the sqrt kernel as torch.sqrt
        patcher = mock.patch.object(A, "_pointwise", sqrt)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_forward_and_backward_match_pytorch(self) -> None:
        torch, aten, A = self.torch, self.torch.ops.aten, self.A
        torch.manual_seed(0)
        x, w, b = torch.randn(4, 3, 5, 5), torch.randn(3), torch.randn(3)
        for training in (True, False):
            rm, rv = torch.randn(3), torch.rand(3) + 0.5
            want = aten.native_batch_norm.default(x, w, b, rm.clone(), rv.clone(), training, 0.1, 1e-5)
            rm2, rv2 = rm.clone(), rv.clone()
            handled, got = A._batch_norm_on_gpu("aten.native_batch_norm.default", (x, w, b, rm2, rv2, training, 0.1, 1e-5), {})
            self.assertTrue(handled)
            self.assertLess(float((got[0] - want[0]).abs().max()), 1e-5)
            if training:
                ref_rm, ref_rv = rm.clone(), rv.clone()
                aten.native_batch_norm.default(x, w, b, ref_rm, ref_rv, True, 0.1, 1e-5)
                self.assertLess(float((rm2 - ref_rm).abs().max()), 1e-6)
                self.assertLess(float((rv2 - ref_rv).abs().max()), 1e-6)
            grad = torch.randn_like(x)
            want_b = aten.native_batch_norm_backward.default(grad, x, w, rm, rv, want[1], want[2], training, 1e-5,
                                                             [True, True, True])
            handled, got_b = A._batch_norm_on_gpu("aten.native_batch_norm_backward.default",
                                                  (grad, x, w, rm, rv, got[1], got[2], training, 1e-5, [True, True, True]), {})
            self.assertTrue(handled)
            for have, ref in zip(got_b, want_b):
                self.assertLess(float((have - ref).abs().max()), 1e-4)

    def test_other_dtypes_keep_the_cpu_route(self) -> None:
        x = self.torch.randn(2, 3, 4).double()
        self.assertEqual(self.A._batch_norm_on_gpu("aten.native_batch_norm.default", (x, None, None, None, None, True, 0.1, 1e-5), {}),
                         (False, None))


if __name__ == "__main__":
    unittest.main()
