from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch; titan runs the GPU check")
class FusedRnnCellTests(unittest.TestCase):
    """The CUDA-only fused LSTM and GRU cells, reimplemented for the CPU fallback."""

    def setUp(self) -> None:
        import torch
        from cuda_legacy_compat import torch_adapter

        self.torch, self.adapter = torch, torch_adapter
        torch.manual_seed(0)

    def close(self, a, b) -> None:
        self.assertLess(float((a - b).abs().max()), 1e-6)

    def test_lstm_cell_matches_pytorch_and_its_backward_matches_autograd(self) -> None:
        torch, A = self.torch, self.adapter
        for bias in (True, False):
            cell, x, h, c = torch.nn.LSTMCell(6, 5, bias=bias), torch.randn(3, 6), torch.randn(3, 5), torch.randn(3, 5)
            gates_in, gates_h = x @ cell.weight_ih.t(), h @ cell.weight_hh.t()
            hy, cy, workspace = A._lstm_cell(gates_in, gates_h, c, cell.bias_ih if bias else None,
                                             cell.bias_hh if bias else None)
            ref_h, ref_c = cell(x, (h, c))
            self.close(hy, ref_h), self.close(cy, ref_c)
            gates = (gates_in + gates_h + ((cell.bias_ih + cell.bias_hh) if bias else 0)).detach().requires_grad_(True)
            cx = c.clone().requires_grad_(True)
            out_h, out_c, _ = A._lstm_cell(gates, torch.zeros_like(gates), cx)
            grad_h, grad_c = torch.randn_like(out_h), torch.randn_like(out_c)
            want_gates, want_cx = torch.autograd.grad((out_h * grad_h).sum() + (out_c * grad_c).sum(), (gates, cx))
            got_gates, got_cx, got_bias = A._lstm_cell_backward(grad_h, grad_c, c, out_c.detach(), workspace.detach(), bias)
            self.close(got_gates, want_gates), self.close(got_cx, want_cx)
            self.assertEqual(got_bias is None, not bias)

    def test_gru_cell_matches_pytorch_and_its_backward_matches_autograd(self) -> None:
        torch, A = self.torch, self.adapter
        for bias in (True, False):
            cell, x, h = torch.nn.GRUCell(6, 5, bias=bias), torch.randn(3, 6), torch.randn(3, 5)
            gates_in = (x @ cell.weight_ih.t()).detach().requires_grad_(True)
            gates_h = (h @ cell.weight_hh.t()).detach().requires_grad_(True)
            hx = h.clone().requires_grad_(True)
            biases = (cell.bias_ih.detach().requires_grad_(True), cell.bias_hh.detach().requires_grad_(True)) if bias else (None, None)
            hy, workspace = A._gru_cell(gates_in, gates_h, hx, *biases)
            self.close(hy, cell(x, h))
            grad = torch.randn_like(hy)
            wanted = torch.autograd.grad((hy * grad).sum(), (gates_in, gates_h, hx) + (biases if bias else ()))
            got = A._gru_cell_backward(grad, workspace.detach(), bias)
            for want, have in zip(wanted, got):
                self.close(have, want)

    def test_the_fallback_knows_every_fused_cell(self) -> None:
        for name in self.adapter._CPU_ONLY_HERE:
            self.assertTrue(hasattr(self.torch.ops.aten, name.split("::")[1]), name)


if __name__ == "__main__":
    unittest.main()
