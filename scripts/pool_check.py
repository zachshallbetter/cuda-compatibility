"""Correctness of the sm_52 max pooling (#184) against the CPU: values and indices exactly, across kernel sizes,
strides, padding, dilation and ceil mode, with NaN, -inf and ties; the backward's gradients; and a small CNN's training
step. Run on an sm_52 machine after setup: cuda-legacy run scripts/pool_check.py"""
import copy
import itertools

import torch
import torch.nn.functional as F
from cuda_legacy_compat import torch_adapter as ta

torch.manual_seed(0)
fails = []


def report(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name:58s} {detail} path={ta.last_path()}")
    if not ok:
        fails.append(name)


x = torch.randn(2, 3, 11, 13)
x[0, 0, 2, 3] = float("nan")
x[1, 2, 5, 5] = float("-inf")
x[0, 1, :4, :4] = 1.0                        # ties: the first maximum wins
cases = 0
for k, s, p, d, ceil in itertools.product((1, 2, 3), (1, 2, 3), (0, 1), (1, 2), (False, True)):
    if p > k // 2:
        continue
    try:
        oc, ic = F.max_pool2d(x, k, s, p, d, ceil_mode=ceil, return_indices=True)
    except RuntimeError:
        continue
    og, ig = F.max_pool2d(x.cuda(), k, s, p, d, ceil_mode=ceil, return_indices=True)
    cases += 1
    same = torch.equal(ig.cpu(), ic) and torch.equal(og.cpu().nan_to_num(7.0), oc.nan_to_num(7.0))
    if not same or not ta.last_path().startswith("llvm-sm52"):
        report(f"max_pool2d k={k} s={s} p={p} d={d} ceil={ceil}", False)
report(f"max_pool2d values and indices, exact, {cases} configurations", not fails)
three = torch.randn(4, 9, 9)
report("max_pool2d of a (C, H, W) input", torch.equal(F.max_pool2d(three.cuda(), 2).cpu(), F.max_pool2d(three, 2)))
report("max_pool2d with a non-square kernel and stride",
       torch.equal(F.max_pool2d(x.cuda().nan_to_num(), (3, 2), (2, 1), (1, 0)).cpu(), F.max_pool2d(x.nan_to_num(), (3, 2), (2, 1), (1, 0))))

for k, s in ((2, 2), (3, 1), (3, 2)):           # overlapping windows sum gradients into one position
    xc, xg = x.nan_to_num().clone().requires_grad_(), x.nan_to_num().clone().cuda().requires_grad_()
    up = torch.randn_like(F.max_pool2d(xc, k, s, 1))
    (F.max_pool2d(xc, k, s, 1) * up).sum().backward(); (F.max_pool2d(xg, k, s, 1) * up.cuda()).sum().backward()
    rel = float((xg.grad.cpu() - xc.grad).abs().max() / xc.grad.abs().max())
    report(f"max_pool2d backward k={k} s={s} (overlapping={s < k})", rel < 1e-6, f"rel={rel:.1e}")

xs = x.nan_to_num()
out, idx = torch.ops.aten.max_pool2d_with_indices(xs, [3, 3], [1, 1], [1, 1], [1, 1], False)
up = torch.randn_like(out)
gc = torch.ops.aten.max_pool2d_with_indices_backward(up, xs, [3, 3], [1, 1], [1, 1], [1, 1], False, idx)
gg = torch.ops.aten.max_pool2d_with_indices_backward(up.cuda(), xs.cuda(), [3, 3], [1, 1], [1, 1], [1, 1], False, idx.cuda())
rel = float((gg.cpu() - gc).abs().max() / gc.abs().max())
report("max_pool2d_with_indices_backward itself on the GPU", rel < 1e-6 and ta.last_path() == "llvm-sm52-max-pool2d-backward-f32",
       f"rel={rel:.1e}")

net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                          torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                          torch.nn.Flatten(), torch.nn.Linear(8 * 4 * 4, 5))
gnet = copy.deepcopy(net).cuda()
img, label = torch.randn(4, 3, 16, 16), torch.randint(0, 5, (4,))
for model, dev in ((net, "cpu"), (gnet, "cuda")):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        opt.zero_grad(); F.cross_entropy(model(img.to(dev)), label.to(dev)).backward(); opt.step()
worst = max(float((a.detach().cpu() - b.detach()).abs().max()) for a, b in zip(gnet.parameters(), net.parameters()))
report("small CNN with max pooling, two SGD steps", worst < 1e-5, f"param diff {worst:.1e}")
print("RESULT", "all passed" if not fails else f"{len(fails)} failed: {fails}")
