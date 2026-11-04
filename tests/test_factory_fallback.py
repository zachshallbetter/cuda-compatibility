from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None
NO_KERNEL = "CUDA error: no kernel image is available for execution on the device"


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class FactoryFallbackTests(unittest.TestCase):
    """A CUDA factory with no Maxwell kernel is built on the CPU, then copied to its device."""

    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter, unrouted

        self.torch, self.adapter, self.unrouted = torch, torch_adapter, unrouted
        unrouted._recorded = None
        self.cuda = torch.device("cuda", 0)
        self.moved: list[object] = []
        # No GPU here: record the copy instead of making it.
        patcher = mock.patch.object(torch.Tensor, "to", lambda tensor, device: self.moved.append(device) or tensor)
        patcher.start()
        self.addCleanup(patcher.stop)

    def factory(self, make):
        """An operator that fails like the stock wheel on CUDA and works on the CPU."""
        def operation(*args, device=None, **kwargs):
            if device is not None and device.type == "cuda":
                raise RuntimeError(NO_KERNEL)
            return make(*args, device=device, **kwargs)
        return operation

    def dispatch(self, operation, *args, **kwargs):
        mode = self.adapter._MaxwellInferenceFallback()
        return mode.__torch_dispatch__(operation, (), args, kwargs)

    def test_a_cuda_factory_is_built_on_the_cpu_and_copied(self) -> None:
        result = self.dispatch(self.factory(self.torch.zeros), (2, 3), device=self.cuda, dtype=self.torch.float16)
        self.assertEqual((tuple(result.shape), result.dtype), ((2, 3), self.torch.float16))
        self.assertTrue(bool((result == 0).all()))
        self.assertEqual(self.moved, [self.cuda])
        self.assertTrue(self.adapter.last_path().startswith("cpu-fallback-"))
        self.assertIsNone(self.unrouted._recorded)

    def test_a_seeded_random_factory_repeats(self) -> None:
        draw = lambda: self.dispatch(self.factory(self.torch.randn), (4,), device=self.cuda)
        self.torch.manual_seed(7)
        first = draw()
        self.torch.manual_seed(7)
        self.assertTrue(self.torch.equal(first, draw()))

    def test_more_than_64_mib_is_refused_with_the_hint(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "limited to 64 MiB of outputs"):
            self.dispatch(self.factory(self.torch.zeros), (17, 1024, 1024), device=self.cuda)
        self.assertEqual(self.moved, [])
        self.assertIsNotNone(self.unrouted.hint_for(self.unrouted._recorded[1]))

    def test_other_failures_and_non_cuda_calls_pass_through(self) -> None:
        def broken(*args, device=None, **kwargs):
            raise RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.dispatch(broken, (2,), device=self.cuda)
        def no_device(*args, **kwargs):
            raise RuntimeError(NO_KERNEL)
        with self.assertRaisesRegex(RuntimeError, "no kernel image"):
            self.dispatch(no_device, (2,))
        self.assertIsNotNone(self.unrouted._recorded)
        self.assertEqual(self.moved, [])


if __name__ == "__main__":
    unittest.main()
