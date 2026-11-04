from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class TrainingGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter, unrouted

        self.torch, self.adapter, self.unrouted = torch, torch_adapter, unrouted
        unrouted._recorded = None

    def test_composite_overrides_refuse_when_autograd_needs_them(self) -> None:
        guarded = self.adapter._inference_only(lambda tensor: tensor * 2, "layer_norm")
        weight = self.torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "runs only without autograd"):
            guarded(weight)
        self.assertIsNotNone(self.unrouted.hint_for(self.unrouted._recorded[1]))
        with self.torch.no_grad():
            self.assertEqual(guarded(weight).tolist(), [2.0, 2.0])
        self.assertEqual(guarded(self.torch.ones(2)).tolist(), [2.0, 2.0])

    def test_training_decomposes_and_inference_uses_the_override(self) -> None:
        calls = []

        class Operation:
            def decompose(self, *args, **kwargs):
                calls.append("decompose")
                return "decomposed"

            def __call__(self, *args, **kwargs):
                calls.append("override")
                return "override"

        kernel = self.adapter._composite_autograd(Operation())
        self.assertEqual(kernel(self.torch.ones(2, requires_grad=True)), "decomposed")
        self.assertEqual(kernel(self.torch.ones(2)), "override")
        with self.torch.no_grad():
            self.assertEqual(kernel(self.torch.ones(2, requires_grad=True)), "override")
        self.assertEqual(calls, ["decompose", "override", "override"])

    def test_the_real_decompositions_have_backward_formulas(self) -> None:
        torch = self.torch
        x = torch.randn(2, 3, 4, requires_grad=True)
        for name, args in (("layer_norm", (x, [4])), ("contiguous", (x.transpose(0, 1),)),
                           ("scaled_dot_product_attention", (x, x, x))):
            out = getattr(torch.ops.aten, name).default.decompose(*args)
            self.assertIsNotNone(out.grad_fn, name)

    def test_every_guarded_operator_is_one_autograd_cannot_see_past(self) -> None:
        for name in self.adapter._INFERENCE_ONLY:
            qualified = f"aten::{name}"
            self.assertTrue(self.torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "CompositeImplicitAutograd"), name)
            self.assertFalse(self.torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "Autograd"), name)

    def test_batch_norm_statistics_positions_match_the_schemas(self) -> None:
        aten = self.torch.ops.aten
        for name, (written, training) in self.adapter._UNDECLARED_WRITES.items():
            schema = getattr(aten, name.split("::")[1]).default._schema
            names = [argument.name for argument in schema.arguments]
            self.assertEqual([names[index] for index in written], ["running_mean", "running_var"], name)
            self.assertEqual(names[training], "training", name)


if __name__ == "__main__":
    unittest.main()
