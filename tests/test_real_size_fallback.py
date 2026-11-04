from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None
NO_KERNEL = "CUDA error: no kernel image is available for execution on the device"


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class RealSizeFallbackTests(unittest.TestCase):
    """Real-size models (#174, #175, #176): work over the fallback limit runs in pieces under it."""

    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter, unrouted

        self.torch, self.adapter, self.unrouted = torch, torch_adapter, unrouted
        unrouted._recorded = None
        # No GPU here: every tensor counts as a CUDA tensor, the device copy is a clone, and both limits are 64 bytes.
        for target, value in (("_is_cuda", lambda tensor: True), ("_cuda_tensor_to_cpu", lambda tensor: tensor.clone()),
                              ("_CPU_FALLBACK_MAX_BYTES", 64), ("_CPU_FALLBACK_TOTAL_BYTES", 64)):
            patcher = mock.patch.object(torch_adapter, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def cpu_results(self):
        """The general fallback checks real CUDA tensors; here it runs the operator (already on the CPU) under the
        same limit, so a call that wasn't split still fails."""
        adapter = self.adapter

        def fallback(operation, name):
            def run(*args, **kwargs):
                if sum(t.numel() * t.element_size() for t in adapter._tensor_inputs((args, kwargs))) > adapter._CPU_FALLBACK_MAX_BYTES:
                    raise NotImplementedError(f"Maxwell CPU fallback for {name} is limited to 64 MiB of CUDA inputs")
                return operation(*args, **kwargs)
            return run
        return mock.patch.object(self.adapter, "_cpu_fallback", fallback)

    def dispatch(self, operation, *args, **kwargs):
        """Fail like the stock wheel on the first call, then behave like the CPU; count the calls."""
        self.calls = []

        def failing_then_cpu(*a, **k):
            self.calls.append(1)
            if len(self.calls) == 1:
                raise RuntimeError(NO_KERNEL)
            return operation(*a, **k)
        failing_then_cpu._schema = operation._schema
        mode = self.adapter._MaxwellInferenceFallback()
        with mock.patch.object(self.adapter, "_cuda_tensors", lambda value: self.adapter._tensor_inputs(value)):
            return mode.__torch_dispatch__(failing_then_cpu, (), args, kwargs)

    def test_a_foreach_step_is_split_between_tensors(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        params = [torch.arange(4, dtype=torch.float32) + i for i in range(3)]   # 16 bytes each, with its grad 32
        grads = [torch.ones(4) * (i + 1) for i in range(3)]
        expected = [p + 0.5 * g for p, g in zip(params, grads)]
        self.dispatch(aten._foreach_add_.List, params, grads, alpha=0.5)
        self.assertEqual([p.tolist() for p in params], [e.tolist() for e in expected])
        self.assertGreater(len(self.calls) - 1, 1)   # more than one call after the refused one
        self.assertEqual(self.adapter.last_path(), "cpu-fallback-pieces-aten::_foreach_add_")

    def test_one_tensor_over_the_limit_is_split_into_ranges(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        param, other = torch.arange(40, dtype=torch.float32).reshape(5, 8), torch.full((5, 8), 2.0)
        expected = param * 2.0
        self.dispatch(aten._foreach_mul_.List, [param], [other])
        self.assertEqual(param.tolist(), expected.tolist())

    def test_per_element_scalars_follow_their_tensors(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        params = [torch.zeros(4) for _ in range(3)]
        self.dispatch(aten._foreach_add_.ScalarList, params, [1.0, 2.0, 3.0])
        self.assertEqual([p.tolist() for p in params], [[1.0] * 4, [2.0] * 4, [3.0] * 4])
        params = [torch.zeros(4) for _ in range(3)]
        t1, t2 = [torch.ones(4) for _ in range(3)], [torch.full((4,), 2.0) for _ in range(3)]
        self.dispatch(aten._foreach_addcdiv_.Tensor, params, t1, t2, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual([p.tolist() for p in params], [[1.0] * 4, [2.0] * 4, [3.0] * 4])

    def test_an_out_of_place_foreach_returns_whole_tensors_in_order(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        values = [torch.arange(40, dtype=torch.float32) ** 2, torch.tensor([4.0, 9.0])]
        with self.cpu_results():
            result = self.dispatch(aten._foreach_sqrt.default, values)
        self.assertEqual([r.tolist() for r in result], [torch.arange(40, dtype=torch.float32).tolist(), [2.0, 3.0]])

    def test_a_reduction_is_never_split_inside_a_tensor(self) -> None:
        torch, aten = self.torch, self.torch.ops.aten
        with self.assertRaisesRegex(NotImplementedError, "limited to .* per operation"):
            self.dispatch(aten._foreach_norm.Scalar, [torch.ones(40)], 2)
        self.assertIsNotNone(self.unrouted._recorded)
        with self.cpu_results():
            result = self.dispatch(aten._foreach_norm.Scalar, [torch.full((4,), 3.0) for _ in range(5)], 2)
        self.assertEqual([round(r.item(), 4) for r in result], [6.0] * 5)

    def test_an_embedding_over_the_limit_moves_only_its_rows(self) -> None:
        torch = self.torch
        weight = torch.arange(400, dtype=torch.float32).reshape(100, 4)   # 1600 bytes, over the 64-byte limit
        indices = torch.tensor([[3, 4, 90], [4, 3, 3]])
        moved = []
        to_cpu = self.adapter._cuda_tensor_to_cpu

        def counting(tensor):
            moved.append(tensor.numel())
            return to_cpu(tensor)
        with mock.patch.object(self.adapter, "_cuda_tensor_to_cpu", counting), \
                mock.patch.object(self.adapter, "_cuda_tensors", lambda value: self.adapter._tensor_inputs(value)):
            out = self.adapter._embedding_rows_on_cpu(weight, indices)
        self.assertEqual(out.tolist(), torch.nn.functional.embedding(indices, weight).tolist())
        # Elements moved: row 90 (4), the indices (6), the run of rows 3-4 (8); never the 400-element weight.
        self.assertEqual(sorted(moved), [4, 6, 8])
        self.assertEqual(self.adapter.last_path(), "cpu-fallback-embedding-rows")

    def test_too_many_rows_still_refuse(self) -> None:
        torch = self.torch
        with self.assertRaisesRegex(NotImplementedError, "looked-up rows"):
            self.adapter._embedding_rows_on_cpu(torch.zeros(100, 4), torch.arange(10))

    def test_a_copy_over_the_per_copy_limit_moves_in_pieces_and_keeps_its_gradient(self) -> None:
        torch, adapter = self.torch, self.adapter
        source = torch.arange(40, dtype=torch.float32, requires_grad=True)   # 160 bytes, 64-byte pieces
        pieces = []
        real_copy = torch.Tensor.copy_

        def counting(self_, other, *a, **k):
            pieces.append(other.numel())
            return real_copy(self_, other, *a, **k)
        with mock.patch.object(torch.Tensor, "copy_", counting):
            moved = adapter._move(source, "cpu")
        self.assertEqual((moved.tolist(), pieces), (source.tolist(), [16, 16, 8]))
        (moved * 2).sum().backward()
        self.assertEqual(source.grad.tolist(), [2.0] * 40)

    def test_an_operation_over_the_total_refuses_and_names_the_setting(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "per operation .CUDA_COMPAT_FALLBACK_TOTAL_MIB."):
            self.dispatch(self.torch.ops.aten.zero_.default, self.torch.zeros(40))

    def test_the_total_comes_from_the_environment(self) -> None:
        with mock.patch.dict("os.environ", {"CUDA_COMPAT_FALLBACK_TOTAL_MIB": "2048"}):
            self.assertEqual(self.adapter._total_limit(), 2048 * 1024 * 1024)
        for bad in ("", "0", "-5", "lots"):
            with mock.patch.dict("os.environ", {"CUDA_COMPAT_FALLBACK_TOTAL_MIB": bad}):
                self.assertEqual(self.adapter._total_limit(), 1024 * 1024 * 1024, bad)

    def test_native_add_over_its_size_limit_falls_back(self) -> None:
        seen = []

        def fallback(operation, name):
            return lambda *args, **kwargs: seen.append(name) or "fallback"
        big = self.torch.zeros(1048577)
        with mock.patch.object(self.adapter, "_cuda_tensors", lambda value: [1]), \
                mock.patch.object(self.adapter, "_validate_pair", lambda *a: None), \
                mock.patch.object(self.adapter, "_cpu_fallback", fallback):
            self.assertEqual(self.adapter._add(big, big), "fallback")
        self.assertEqual(seen, ["add"])


if __name__ == "__main__":
    unittest.main()
