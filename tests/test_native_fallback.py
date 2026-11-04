from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class NativeFallbackTests(unittest.TestCase):
    """A native kernel that refuses a case hands it to the CPU fallback instead of failing."""

    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter

        self.torch, self.adapter = torch, torch_adapter

    def test_a_refused_case_takes_the_cpu_fallback(self) -> None:
        seen = []

        @self.adapter._falls_back_to_cpu("relu")
        def kernel(tensor):
            raise NotImplementedError("Maxwell ReLU supports float32 tensors only")

        def fallback(operation, name):
            return lambda *args, **kwargs: seen.append((operation, name)) or "fallback"

        with mock.patch.object(self.adapter, "_cuda_tensors", lambda value: [1]), \
                mock.patch.object(self.adapter, "_cpu_fallback", fallback):
            self.assertEqual(kernel(self.torch.ones(2)), "fallback")
        self.assertEqual(seen, [(self.torch.ops.aten.relu.default, "relu")])

    def test_a_refused_factory_takes_the_factory_fallback(self) -> None:
        @self.adapter._falls_back_to_cpu("arange")
        def kernel(*args, **kwargs):
            raise NotImplementedError("Maxwell arange supports int64 values only")

        with mock.patch.object(self.adapter, "_factory_on_cpu", lambda *a: "factory"):
            self.assertEqual(kernel(4, device=self.torch.device("cuda")), "factory")

    def test_other_errors_and_successes_pass_through(self) -> None:
        @self.adapter._falls_back_to_cpu("mm")
        def failing(a, b):
            raise RuntimeError("out of memory")

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            failing(1, 2)
        self.assertEqual(self.adapter._falls_back_to_cpu("mm")(lambda a, b: "native")(1, 2), "native")

    def test_a_refused_in_place_case_updates_its_own_tensor(self) -> None:
        # torch.ones(..., dtype=torch.bool, device="cuda") fills a bool tensor: the float32 kernel refuses it, and the
        # fallback must write into that tensor, never return a new one (a causal mask was left uninitialised).
        torch = self.torch
        target = torch.zeros(3, 4, dtype=torch.bool)
        with mock.patch.object(self.adapter, "_cuda_tensors", lambda value: self.adapter._tensor_inputs(value)), \
                mock.patch.object(self.adapter, "_is_cuda", lambda tensor: True), \
                mock.patch.object(self.adapter, "_cuda_tensor_to_cpu", lambda tensor: tensor.clone()):
            result = self.adapter._fill(target, True)
            self.assertIs(result, target)
            self.assertTrue(bool(target.all()))
            ints = torch.ones(5, dtype=torch.int64)
            self.assertIs(self.adapter._zero(ints), ints)
            self.assertEqual(ints.tolist(), [0] * 5)

    def test_every_native_kernel_is_wrapped(self) -> None:
        for kernel in ("_add", "_mm", "_relu", "_arange", "_embedding", "_layer_norm"):
            self.assertTrue(hasattr(getattr(self.adapter, kernel), "__wrapped__"), kernel)


if __name__ == "__main__":
    unittest.main()
