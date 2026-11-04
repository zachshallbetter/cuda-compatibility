"""Correctness of the sm_52 elementwise, softmax, layer-norm and sum kernels (#184 C) against the CPU.
Run on an sm_52 machine after setup: cuda-legacy run scripts/elementwise_check.py"""
import torch
import torch.nn.functional as F
from cuda_legacy_compat import torch_adapter as ta

torch.manual_seed(0)
fails = []


def check(name, gpu, cpu, prefix="llvm-sm52", tol=2e-5):
    gpu = gpu if isinstance(gpu, (tuple, list)) else (gpu,)
    cpu = cpu if isinstance(cpu, (tuple, list)) else (cpu,)
    worst = 0.0
    for g, c in zip(gpu, cpu):
        if g is None and c is None:
            continue
        worst = max(worst, float((g.detach().cpu() - c.detach()).abs().max() / (c.detach().abs().max() + 1e-12)))
    ok = worst < tol and ta.last_path().startswith(prefix)
    print(f"{'OK  ' if ok else 'FAIL'} {name:48s} rel={worst:.2e} path={ta.last_path()}")
    if not ok:
        fails.append(name)


a, b = torch.randn(4, 1, 6), torch.randn(5, 1)
for op, f in (("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div)):
    check(f"{op} broadcast [4,1,6]x[5,1]", f(a.cuda(), b.cuda()), f(a, b))
x, y = torch.randn(2_000_000), torch.randn(2_000_000)
check("add alpha=2.5 same shape 2M", torch.add(x.cuda(), y.cuda(), alpha=2.5), torch.add(x, y, alpha=2.5))
check("sub alpha=0.5 transposed", torch.sub(a.cuda().transpose(0, 2), 1.5, alpha=0.5), torch.sub(a.transpose(0, 2), 1.5, alpha=0.5))
for op, f in (("add", lambda t: t + 2.0), ("mul", lambda t: t * 3.0), ("div", lambda t: t / 4.0), ("sub", lambda t: t - 1.0),
              ("rsub", lambda t: 1.0 - t), ("mul 0-d tensor", lambda t: t * torch.tensor(2.0))):
    check(f"scalar {op}", f(a.cuda()), f(a))
x = torch.randn(3, 7, 33)
for approx in ("none", "tanh"):
    check(f"gelu {approx}", F.gelu(x.cuda(), approximate=approx), F.gelu(x, approximate=approx))
    xc, xg = x.clone().requires_grad_(), x.clone().cuda().requires_grad_()
    F.gelu(xc, approximate=approx).pow(2).sum().backward(); F.gelu(xg, approximate=approx).pow(2).sum().backward()
    check(f"gelu {approx} backward", xg.grad, xc.grad, "")
s = torch.randn(6, 300)
s[0, 5:] = float("-inf")
check("softmax last dim (with -inf)", torch.softmax(s.cuda(), -1), torch.softmax(s, -1))
check("log_softmax last dim", torch.log_softmax(s[1:].cuda(), -1), torch.log_softmax(s[1:], -1))
for log in (False, True):
    sc, sg = s[1:].clone().requires_grad_(), s[1:].clone().cuda().requires_grad_()
    f = torch.log_softmax if log else torch.softmax
    (f(sc, -1) * torch.arange(300.0)).sum().backward(); (f(sg, -1) * torch.arange(300.0).cuda()).sum().backward()
    check(f"{'log_' if log else ''}softmax backward", sg.grad, sc.grad, "")
ln = torch.nn.LayerNorm(64)
with torch.no_grad():
    ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
lng = torch.nn.LayerNorm(64).cuda()
lng.load_state_dict(ln.state_dict())
h = torch.randn(2, 9, 64)
hc, hg = h.clone().requires_grad_(), h.clone().cuda().requires_grad_()
(ln(hc) ** 2).sum().backward(); (lng(hg) ** 2).sum().backward()
check("layer_norm training forward+backward input", hg.grad, hc.grad, "")
check("layer_norm weight grad", lng.weight.grad, ln.weight.grad, "")
check("layer_norm bias grad", lng.bias.grad, ln.bias.grad, "")
t = torch.randn(4, 5, 6)
for dims, keep in (((0,), False), ((1, 2), True), ((-1,), False), ((), False), ((0, 2), False)):
    check(f"sum dims={dims} keepdim={keep}", t.cuda().sum(dim=dims, keepdim=keep) if dims else t.cuda().sum(),
          t.sum(dim=dims, keepdim=keep) if dims else t.sum())
big = torch.randn(256, 30522)
check("sum all of 256x30522", big.cuda().sum(), big.sum(), tol=1e-4)
z = torch.randn(5, 7).cuda()
z.fill_(3.5)
check("fill_", z, torch.full((5, 7), 3.5))
z.zero_()
check("zero_", z, torch.zeros(5, 7))
print("RESULT", "all passed" if not fails else f"{len(fails)} failed: {fails}")
