from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class SingleAsForeachTests(unittest.TestCase):
    """foreach=False optimizer steps reach the foreach kernels as one-tensor calls."""

    def test_each_step_maps_to_its_foreach_kernel(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter as A

        aten, calls = torch.ops.aten, []
        x, a, b = torch.ones(2), torch.ones(2), torch.ones(2)

        def kernel(name):
            return lambda *args: calls.append((name, args)) or [torch.zeros(2)]

        with mock.patch.object(A, "_foreach_kernel", kernel):
            for operation, args, kwargs, want in (
                (aten.mul_.Tensor, (x, 0.9), {}, "_foreach_mul_.Scalar"),
                (aten.div_.Tensor, (x, 2.0), {}, "_foreach_div_.ScalarList"),
                (aten.lerp_.Scalar, (x, a, 0.1), {}, "_foreach_lerp_.Scalar"),
                (aten.addcmul_.default, (x, a, b), {"value": 0.5}, "_foreach_addcmul_.Scalar"),
                (aten.addcdiv_.default, (x, a, b), {"value": -0.1}, "_foreach_addcdiv_.ScalarList"),
            ):
                handled, result = A._single_as_foreach(operation, args, kwargs)
                self.assertTrue(handled)
                self.assertIs(result, x)
                self.assertEqual(calls[-1][0], want)
            handled, result = A._single_as_foreach(aten.sqrt.default, (x,), {})
            self.assertTrue(handled)
            self.assertEqual(calls[-1][0], "_foreach_sqrt")
        for operation, args in ((aten.mul_.Tensor, (x, a)), (aten.relu_.default, (x,))):
            self.assertEqual(A._single_as_foreach(operation, args, {}), (False, None))


if __name__ == "__main__":
    unittest.main()
