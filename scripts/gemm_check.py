"""Correctness and speed of the tiled sm_52 GEMM (#184) under cuda-legacy. Run on an sm_52 machine after setup: cuda-legacy run scripts/gemm_check.py"""
import time

import torch
from cuda_legacy_compat import torch_adapter as ta

torch.manual_seed(0)
fails = []


def rel(a, b):
    return float((a.cpu() - b).abs().max() / (b.abs().max() + 1e-12))


def check(name, gpu, cpu, path, tol=2e-5):
    r = rel(gpu, cpu)
    ok = r < tol and ta.last_path().startswith(path)
    print(f"{'OK  ' if ok else 'FAIL'} {name:46s} rel={r:.2e} path={ta.last_path()}")
    if not ok:
        fails.append(name)


for m, k, n in [(1, 1, 1), (3, 5, 7), (64, 64, 64), (65, 17, 130), (257, 300, 129), (128, 768, 3072), (512, 1, 33)]:
    a, b = torch.randn(m, k), torch.randn(k, n)
    check(f"mm {m}x{k}@{k}x{n}", a.cuda() @ b.cuda(), a @ b, "llvm-sm52-mm")
a, b = torch.randn(40, 70), torch.randn(90, 70)
check("mm transposed right (x @ W.T)", a.cuda() @ b.cuda().t(), a @ b.t(), "llvm-sm52-mm")
A, B = torch.randn(70, 40), torch.randn(70, 90)
check("mm transposed left (A.T @ B)", A.cuda().t() @ B.cuda(), A.t() @ B, "llvm-sm52-mm")
for shape in [(90,), (40, 90), (1, 90), (40, 1)]:
    bias, x, w = torch.randn(*shape), torch.randn(40, 70), torch.randn(70, 90)
    check(f"addmm bias{list(shape)} alpha=0.5 beta=2", torch.addmm(bias.cuda(), x.cuda(), w.cuda(), alpha=0.5, beta=2.0),
          torch.addmm(bias, x, w, alpha=0.5, beta=2.0), "llvm-sm52-addmm")
lin = torch.nn.Linear(768, 3072)
x = torch.randn(2, 128, 768)
check("linear 2x128x768 -> 3072 (addmm)", torch.nn.functional.linear(x.cuda(), lin.weight.cuda(), lin.bias.cuda()),
      torch.nn.functional.linear(x, lin.weight, lin.bias), "llvm-sm52-addmm")
q, kk = torch.randn(24, 128, 64), torch.randn(24, 64, 128)
check("bmm 24x128x64 @ 24x64x128", q.cuda() @ kk.cuda(), q @ kk, "llvm-sm52-bmm")
check("bmm transposed (q @ k.transpose)", q.cuda() @ q.cuda().transpose(1, 2), q @ q.transpose(1, 2), "llvm-sm52-bmm")
e = torch.empty(0, 5).cuda() @ torch.randn(5, 3).cuda()
print("OK  " if e.shape == (0, 3) else "FAIL", "empty mm shape", tuple(e.shape))
# gradients through mm/addmm vs CPU
w = torch.randn(70, 90, requires_grad=True); xg = torch.randn(40, 70, requires_grad=True)
wc, xc = w.detach().cuda().requires_grad_(), xg.detach().cuda().requires_grad_()
(xg @ w).pow(2).sum().backward(); (xc @ wc).pow(2).sum().backward()
check("mm backward grad_w", wc.grad, w.grad, "")
check("mm backward grad_x", xc.grad, xg.grad, "")
# speed
for n in (512, 1024, 2048):
    a, b = torch.randn(n, n).cuda(), torch.randn(n, n).cuda()
    a @ b; torch.cuda.synchronize()
    t = time.time(); reps = 5
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize(); dt = (time.time() - t) / reps
    print(f"SPEED {n}^3 mm: {dt * 1e3:.1f} ms = {2 * n ** 3 / dt / 1e9:.0f} GFLOPS")
print("RESULT", "all passed" if not fails else f"{len(fails)} failed: {fails}")
