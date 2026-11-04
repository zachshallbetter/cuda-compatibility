from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None
NO_KERNEL = "CUDA error: no kernel image is available for execution on the device"


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class InplaceFallbackTests(unittest.TestCase):
    """An in-place operator with no Maxwell kernel runs on a CPU copy and is copied back into its target."""

    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter, unrouted

        self.torch, self.adapter, self.unrouted = torch, torch_adapter, unrouted
        unrouted._recorded = None
        # No GPU here: every tensor counts as a CUDA tensor and the device copy is an ordinary clone.
        for target, value in (("_is_cuda", lambda tensor: True), ("_cuda_tensor_to_cpu", lambda tensor: tensor.clone())):
            patcher = mock.patch.object(torch_adapter, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def dispatch(self, operation, *args, **kwargs):
        """Fail like the stock wheel on the first call, then behave like the CPU."""
        calls = []

        def failing_then_cpu(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError(NO_KERNEL)
            return operation(*a, **k)
        failing_then_cpu._schema = operation._schema
        mode = self.adapter._MaxwellInferenceFallback()
        with mock.patch.object(self.adapter, "_cuda_tensors", lambda value: [1]):
            return mode.__torch_dispatch__(failing_then_cpu, (), args, kwargs)

    def test_in_place_updates_reach_the_target(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertIs(self.dispatch(aten.mul_.Scalar, x, 2), x)
        self.assertEqual(x.tolist(), [[0, 2, 4], [6, 8, 10]])
        self.assertEqual(self.adapter.last_path(), "cpu-fallback-inplace-aten::mul_")
        self.dispatch(aten.fill_.Scalar, x[0], 7.0)
        self.assertEqual(x.tolist(), [[7, 7, 7], [6, 8, 10]])
        y = torch.tensor([-1.0, 2.0])
        self.dispatch(aten.relu_.default, y)
        self.assertEqual(y.tolist(), [0.0, 2.0])
        self.assertIsNone(self.unrouted._recorded)

    def test_a_strided_target_or_a_metadata_change_fails_closed(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        x = torch.zeros(3, 4)
        with self.assertRaisesRegex(NotImplementedError, "requires a contiguous target"):
            self.dispatch(aten.zero_.default, x[:, 1:3])
        with self.assertRaisesRegex(NotImplementedError, "only supports value updates"):
            self.dispatch(aten.resize_.default, torch.zeros(4), [2, 2])
        self.assertIsNotNone(self.unrouted._recorded)

    def test_more_than_64_mib_is_refused(self) -> None:
        big = self.torch.zeros(17, 1024, 1024)
        with self.assertRaisesRegex(NotImplementedError, "limited to 64 MiB"):
            self.dispatch(self.torch.ops.aten.zero_.default, big)

    def test_only_operators_that_write_just_self_qualify(self) -> None:
        aten, qualifies = self.torch.ops.aten, self.adapter._writes_only_self
        for operation in (aten.add_.Tensor, aten.mul_.Scalar, aten.zero_.default, aten.fill_.Tensor, aten.relu_.default):
            self.assertTrue(qualifies(operation._schema), operation)
        for operation in (aten.add.out, aten.view.default, aten.add.Tensor):
            self.assertFalse(qualifies(operation._schema), operation)


if __name__ == "__main__":
    unittest.main()
