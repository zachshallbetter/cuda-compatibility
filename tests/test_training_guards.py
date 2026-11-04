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
