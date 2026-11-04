"""Correctness of the sm_52 optimizer steps (#184) against the CPU: each optimizer's default foreach update on the GPU
and the CPU with the same gradients set by hand, so only the step's arithmetic is compared (no backward noise for
Adam to amplify). Run on an sm_52 machine after setup: cuda-legacy run scripts/optimizer_check.py"""
import torch
from cuda_legacy_compat import torch_adapter as ta

torch.manual_seed(0)
fails = []
SHAPES = [(2_000_000,), (64, 33), (7,), (3, 4, 5)]


def compare(name, make, steps=5, dtypes=None, prefix="llvm-sm52", tol=1e-6):
    dtypes = dtypes or [torch.float32] * len(SHAPES)
    cpu = [torch.randn(s).to(d) for s, d in zip(SHAPES, dtypes)]
    gpu = [p.clone().cuda() for p in cpu]
    for p in cpu + gpu:
        p.requires_grad_(True)
    oc, og = make(cpu), make(gpu)
    paths = set()
    for _ in range(steps):
        for c, g in zip(cpu, gpu):
            grad = torch.randn(c.shape).to(c.dtype)
            c.grad, g.grad = grad, grad.cuda()
        oc.step(); og.step()
        paths.add(ta.last_path())
    worst = max(float(((g.detach().cpu().float() - c.detach().float()).abs().max()) / (c.detach().float().abs().max() + 1e-12))
                for c, g in zip(cpu, gpu))
    native = any(p.startswith(prefix) for p in paths)
    ok = worst < tol and native
    print(f"{'OK  ' if ok else 'FAIL'} {name:46s} rel={worst:.2e} paths={sorted(paths)}")
    if not ok:
        fails.append(name)


compare("SGD", lambda p: torch.optim.SGD(p, lr=0.1))
compare("SGD momentum + nesterov + weight decay",
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-2))
compare("Adam", lambda p: torch.optim.Adam(p, lr=1e-3))
compare("Adam weight decay", lambda p: torch.optim.Adam(p, lr=1e-3, weight_decay=1e-2))
compare("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.1))
compare("AdamW amsgrad (maximum on the CPU)", lambda p: torch.optim.AdamW(p, lr=1e-3, amsgrad=True), prefix="")
compare("RMSprop (other foreach ops)", lambda p: torch.optim.RMSprop(p, lr=1e-3), prefix="")
compare("Adagrad (other foreach ops)", lambda p: torch.optim.Adagrad(p, lr=1e-2), prefix="")
compare("Adam with a float16 parameter (its dtype group on the CPU)", lambda p: torch.optim.Adam(p, lr=1e-3),
        dtypes=[torch.float32, torch.float32, torch.float16, torch.float32], prefix="", tol=1e-3)
compare("AdamW foreach=False (single-tensor steps)", lambda p: torch.optim.AdamW(p, lr=1e-3, foreach=False), prefix="")
print("RESULT", "all passed" if not fails else f"{len(fails)} failed: {fails}")
