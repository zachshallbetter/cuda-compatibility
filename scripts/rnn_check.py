"""Correctness of the sm_52 fused recurrent cells (#243) against the CPU versions they replace, and LSTM, GRU and RNN
layers trained on the GPU against the CPU. Run on an sm_52 machine after setup: cuda-legacy run scripts/rnn_check.py"""
import copy

import torch
from torch import nn
from cuda_legacy_compat import torch_adapter as ta

torch.manual_seed(0)
fails = []
aten = torch.ops.aten


def report(name, gpu, cpu, prefix, tol=2e-6):
    gpu = gpu if isinstance(gpu, (tuple, list)) else (gpu,)
    cpu = cpu if isinstance(cpu, (tuple, list)) else (cpu,)
    worst = 0.0
    for g, c in zip(gpu, cpu):
        if (g is None) != (c is None):
            worst = float("inf")
        elif g is not None:
            worst = max(worst, float((g.detach().cpu() - c.detach()).abs().max() / (c.detach().abs().max() + 1e-12)))
    ok = worst < tol and ta.last_path().startswith(prefix)
    print(f"{'OK  ' if ok else 'FAIL'} {name:56s} rel={worst:.2e} path={ta.last_path()}")
    if not ok:
        fails.append(name)


rows, hidden = 7, 33


def cuda(t):
    return None if t is None else t.cuda()


for bias in (True, False):
    ig, hg, cx = torch.randn(rows, 4 * hidden), torch.randn(rows, 4 * hidden), torch.randn(rows, hidden)
    ib, hb = (torch.randn(4 * hidden), torch.randn(4 * hidden)) if bias else (None, None)
    out_c = ta._lstm_cell(ig, hg, cx, ib, hb)
    out_g = aten._thnn_fused_lstm_cell(ig.cuda(), hg.cuda(), cx.cuda(), cuda(ib), cuda(hb))
    report(f"LSTM cell forward, bias={bias}", out_g, out_c, "llvm-sm52-lstm-cell-f32")
    ghy, gcy = torch.randn(rows, hidden), torch.randn(rows, hidden)
    for name, a, b in (("both gradients", ghy, gcy), ("no grad_cy", ghy, None), ("no grad_hy", None, gcy)):
        back_c = ta._lstm_cell_backward(a, b, cx, out_c[1], out_c[2], bias)
        back_g = aten._thnn_fused_lstm_cell_backward_impl(cuda(a), cuda(b), cx.cuda(), out_g[1], out_g[2], bias)
        report(f"LSTM cell backward, {name}, bias={bias}", back_g, back_c, "llvm-sm52-lstm-cell-backward-f32")
    ig, hg, hx = torch.randn(rows, 3 * hidden), torch.randn(rows, 3 * hidden), torch.randn(rows, hidden)
    ib, hb = (torch.randn(3 * hidden), torch.randn(3 * hidden)) if bias else (None, None)
    out_c = ta._gru_cell(ig, hg, hx, ib, hb)
    out_g = aten._thnn_fused_gru_cell(ig.cuda(), hg.cuda(), hx.cuda(), cuda(ib), cuda(hb))
    report(f"GRU cell forward, bias={bias}", out_g, out_c, "llvm-sm52-gru-cell-f32")
    g = torch.randn(rows, hidden)
    report(f"GRU cell backward, bias={bias}", aten._thnn_fused_gru_cell_backward(g.cuda(), out_g[1], bias),
           ta._gru_cell_backward(g, out_c[1], bias), "llvm-sm52-gru-cell-backward-f32")

for layer in (nn.LSTM(12, 20, num_layers=2, bidirectional=True, batch_first=True),
              nn.GRU(12, 20, num_layers=2, bidirectional=True, batch_first=True),
              nn.LSTM(12, 20, bias=False), nn.RNN(12, 20, num_layers=2)):
    cpu, gpu = layer, copy.deepcopy(layer).cuda()
    batch_first = layer.batch_first
    x = torch.randn(4, 9, 12) if batch_first else torch.randn(9, 4, 12)
    name = f"{type(layer).__name__} layers={layer.num_layers} bidirectional={layer.bidirectional} bias={layer.bias}"
    for model, dev in ((cpu, "cpu"), (gpu, "cuda")):
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            out, _ = model(x.to(dev))
            (out ** 2).mean().backward()
            opt.step()
    worst = max(float((a.detach().cpu() - b.detach()).abs().max()) for a, b in zip(gpu.parameters(), cpu.parameters()))
    ok = worst < 1e-5
    print(f"{'OK  ' if ok else 'FAIL'} {name + ', three SGD steps':56s} param diff {worst:.1e}")
    if not ok:
        fails.append(name)
print("RESULT", "all passed" if not fails else f"{len(fails)} failed: {fails}")
