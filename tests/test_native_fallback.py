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

    def test_a_foreach_step_the_kernel_cannot_take_goes_the_cpu_route(self) -> None:
        seen = []
        with mock.patch.object(self.adapter, "_foreach_on_cpu", lambda op, args, kwargs: seen.append(op) or "cpu"):
            step = self.adapter._foreach_kernel("_foreach_mul_.Scalar")
            self.assertEqual(step([self.torch.ones(2)], 2.0), "cpu")   # a CPU tensor: not ready for the kernel
            self.assertEqual(step([], 2.0), "cpu")
        self.assertEqual(seen, [self.torch.ops.aten._foreach_mul_.Scalar] * 2)

    def test_every_foreach_step_is_registered(self) -> None:
        import inspect
        source = inspect.getsource(self.adapter.install)
        for overload in self.adapter._FOREACH:
            self.assertIn(f'_library.impl("{overload}", _foreach_kernel("{overload}"))', source)

    def test_max_pool_output_size_matches_pytorch(self) -> None:
        import itertools
        torch, pool_size = self.torch, self.adapter._pool_size
        for size, k, s, p, d, ceil in itertools.product(range(1, 10), (1, 2, 3), (1, 2, 3), (0, 1), (1, 2), (False, True)):
            if p > k // 2:
                continue
            try:
                want = torch.nn.functional.max_pool2d(torch.zeros(1, 1, size, size), k, s, p, d, ceil_mode=ceil).shape[-1]
            except RuntimeError:
                want = 0
            self.assertEqual(max(pool_size(size, k, s, p, d, ceil), 0), want, (size, k, s, p, d, ceil))

    def test_max_pool_leaves_arguments_pytorch_rejects_to_the_cpu_path(self) -> None:
        pairs = self.adapter._pool_pairs
        self.assertEqual(pairs(3, [], 1, 1), ([3, 3], [3, 3], [1, 1], [1, 1]))
        self.assertEqual(pairs([2, 3], [1, 2], [1, 0], [1, 1]), ([2, 3], [1, 2], [1, 0], [1, 1]))
        for bad in ((2, [], 2, 1), (0, [], 0, 1), (3, [0], 0, 1), ([1, 2, 3], [], 0, 1)):
            with self.assertRaises(NotImplementedError, msg=bad):
                pairs(*bad)

    def test_every_native_kernel_is_wrapped(self) -> None:
        for kernel in ("_add", "_mm", "_relu", "_arange", "_embedding", "_layer_norm", "_max_pool2d_with_indices",
                       "_max_pool2d_with_indices_backward"):
            self.assertTrue(hasattr(getattr(self.adapter, kernel), "__wrapped__"), kernel)


if __name__ == "__main__":
    unittest.main()
