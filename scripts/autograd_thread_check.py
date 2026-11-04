"""Each native kernel as the first CUDA work on PyTorch's autograd thread (#188), in a fresh process per case.

Run on an sm_52 machine after setup: cuda-legacy run scripts/autograd_thread_check.py
A case passes when its gradient matches the CPU. The children inherit the adapter's environment from cuda-legacy run.
"""
import subprocess
import sys

CASES = {
    "linear input grad (mm)": "m = nn.Linear(6, 4); x = torch.randn(3, 6); f = lambda m, x: m(x).sum()",
    "matmul grad": "m = None; x = torch.randn(5, 7); f = lambda m, x: (x @ torch.ones(7, 3, device=x.device)).sum()",
    "bmm grad": "m = None; x = torch.randn(2, 4, 5); f = lambda m, x: (x @ torch.ones(2, 5, 3, device=x.device)).sum()",
    "mul grad": "m = None; x = torch.randn(4, 5); f = lambda m, x: (x * torch.full((4, 5), 3.0, device=x.device)).sum()",
    "gelu grad": "m = None; x = torch.randn(4, 5); f = lambda m, x: F.gelu(x).sum()",
    "softmax grad": "m = None; x = torch.randn(4, 9); f = lambda m, x: (torch.softmax(x, -1) * torch.arange(9.0, device=x.device)).sum()",
    "layer_norm grad": "m = None; x = torch.randn(4, 9); f = lambda m, x: (F.layer_norm(x, (9,)) * torch.arange(9.0, device=x.device)).sum()",
    "accumulated grad (add)": "m = None; x = torch.randn(4, 5); f = lambda m, x: ((x * 2.0) * (x * 3.0)).sum()",
}

CHILD = """
import copy, torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(0)
{case}
grads = []
for dev in ("cpu", "cuda"):
    mm = copy.deepcopy(m).to(dev) if m is not None else None
    xx = x.clone().to(dev).requires_grad_()
    grads.append(torch.autograd.grad(f(mm, xx), xx)[0].cpu())
r = float((grads[1] - grads[0]).abs().max() / (grads[0].abs().max() + 1e-12))
print(f"{{r:.1e}}")
sys_exit = 0 if r < 1e-5 else 1
raise SystemExit(sys_exit)
"""

failed = []
for name, case in CASES.items():
    done = subprocess.run([sys.executable, "-c", CHILD.format(case=case)], capture_output=True, text=True, timeout=300)
    detail = done.stdout.strip() or (done.stderr.strip().splitlines() or ["no output"])[-1][:160]
    ok = done.returncode == 0
    print(f"{'OK  ' if ok else 'FAIL'} {name:28s} {detail}")
    if not ok:
        failed.append(name)
print("RESULT", "all passed" if not failed else f"{len(failed)} failed: {failed}")
raise SystemExit(1 if failed else 0)
