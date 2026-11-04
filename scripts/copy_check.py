"""Correctness of the sm_52 paths that keep tensors on the GPU (#184 B) against the CPU: strided copies (clone,
contiguous), in-place add at any size and SGD's foreach step, embedding with its backward, nll_loss and its backward,
safe softmax, and inference layer norm with parameters that require grad.
Run on an sm_52 machine after setup: cuda-legacy run scripts/copy_check.py"""
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
        g, c = g.detach().cpu(), c.detach()
        if g.shape != c.shape or g.dtype != c.dtype:
            worst = float("inf")
        elif g.dtype.is_floating_point:
            worst = max(worst, float((g - c).abs().max() / (c.abs().max() + 1e-12)) if c.numel() else 0.0)
        else:
            worst = max(worst, 0.0 if torch.equal(g, c) else float("inf"))
    ok = worst < tol and ta.last_path().startswith(prefix)
    print(f"{'OK  ' if ok else 'FAIL'} {name:52s} rel={worst:.2e} path={ta.last_path()}")
    if not ok:
        fails.append(name)


# Strided copies: exact, for every element size.
for dtype in (torch.float32, torch.int64, torch.bool, torch.float16, torch.uint8):
    base = (torch.randn(3, 4, 5) * 10).to(dtype)
    view = base.cuda().permute(2, 0, 1)
    check(f"clone of a permuted {dtype}", view.clone(), base.permute(2, 0, 1).clone(), tol=1e-30)
sliced = torch.randn(8, 10)
check("clone of a stepped slice", sliced.cuda()[::2, 1::3].clone(), sliced[::2, 1::3].clone(), tol=1e-30)
check("clone of a broadcast view", sliced[0].cuda().expand(4, 10).clone(), sliced[0].expand(4, 10).clone(), tol=1e-30)
check("clone of a 0-d tensor", torch.tensor(2.5).cuda().clone(), torch.tensor(2.5), tol=1e-30)
with torch.no_grad():
    check("contiguous of a transpose", sliced.cuda().t().contiguous(), sliced.t().contiguous(), tol=1e-30)
neg = torch.tensor([-0.0, float("nan"), float("inf")])
cloned = neg.cuda()[::1].t().clone().cpu()
same = torch.equal(torch.signbit(cloned), torch.signbit(neg)) and bool(cloned[1].isnan()) and bool(cloned[2].isinf())
print(f"{'OK  ' if same else 'FAIL'} {'a clone keeps -0.0, NaN and inf':52s}")
if not same:
    fails.append("special values")

# The gradient of a selected slice (select_backward): x[:, i] and unbind.
src = torch.randn(4, 3, 5)
sc, sg = src.clone().requires_grad_(), src.clone().cuda().requires_grad_()
(sc[:, 1] * 2).sum().backward(); (sg[:, 1] * 2).sum().backward()
check("gradient of x[:, 1]", sg.grad, sc.grad, tol=1e-30)
sc, sg = src.clone().requires_grad_(), src.clone().cuda().requires_grad_()
w3 = torch.randn(4, 5)
sum((t * w3 * k) for k, t in enumerate(sc.unbind(1))).sum().backward()
sum((t * w3.cuda() * k) for k, t in enumerate(sg.unbind(1))).sum().backward()
check("gradient through unbind", sg.grad, sc.grad, "")

# Concatenation and squaring on the GPU (#243).
for dtype in (torch.float32, torch.int64, torch.bool):
    parts = [(torch.randn(2, k, 5) * 4).to(dtype) for k in (3, 1, 4)]
    check(f"cat of {dtype} along dim 1", torch.cat([t.cuda() for t in parts], 1), torch.cat(parts, 1), "llvm-sm52-cat", 1e-30)
seq = [torch.randn(8, 1, 16) for _ in range(5)]
check("cat along dim -2 with a legacy empty input",
      torch.cat([torch.empty(0).cuda()] + [t.cuda() for t in seq], -2), torch.cat([torch.empty(0)] + seq, -2),
      "llvm-sm52-cat", 1e-30)
check("cat of a transposed view", torch.cat([seq[0].cuda().transpose(0, 2), seq[1].cuda().transpose(0, 2)], 0),
      torch.cat([seq[0].transpose(0, 2), seq[1].transpose(0, 2)], 0), "llvm-sm52-cat", 1e-30)
check("cat of mixed dtypes (the CPU route)", torch.cat([seq[0].cuda(), seq[1].cuda().double()], 0),
      torch.cat([seq[0], seq[1].double()], 0), "", 1e-12)
sq = torch.randn(6, 7)
check("x ** 2 as x * x", sq.cuda() ** 2, sq ** 2, "llvm-sm52-binary-f32", 1e-30)
sc, sg = sq.clone().requires_grad_(), sq.clone().cuda().requires_grad_()
(sc ** 2).sum().backward(); (sg ** 2).sum().backward()
check("gradient of x ** 2", sg.grad, sc.grad, "", 1e-30)

# In-place add at any size, broadcast, a number; SGD's foreach step.
x, y = torch.randn(3_000_000), torch.randn(3_000_000)
xg = x.cuda()
xg.add_(y.cuda(), alpha=-0.1)
check("add_ of 3M values, alpha -0.1", xg, x.clone().add_(y, alpha=-0.1))
m, row = torch.randn(64, 70), torch.randn(70)
mg = m.cuda()
mg.add_(row.cuda())
check("add_ broadcast row", mg, m + row)
mg = m.cuda()
mg.add_(mg)
check("add_ itself", mg, m + m)
mg = m.cuda()
mg[:, 1:].add_(mg[:, :-1])
expect = m.clone()
expect[:, 1:].add_(expect[:, :-1].clone())
check("add_ of an overlapping view (refused to the CPU)", mg, expect, prefix="")
params = [torch.randn(2_000_000), torch.randn(30, 40), torch.randn(7)]
grads = [torch.randn_like(p) for p in params]
pg = [p.cuda() for p in params]
torch._foreach_add_(pg, [g.cuda() for g in grads], alpha=-0.05)
check("foreach add_ (SGD's step)", pg, [p + -0.05 * g for p, g in zip(params, grads)])

# Embedding: forward with a weight that requires grad, and its backward.
weight = torch.randn(1000, 64)
idx = torch.tensor([[1, 5, 5, 999], [0, 3, 5, 7]])
for padding in (None, 5):
    wc, wg = weight.clone().requires_grad_(), weight.clone().cuda().requires_grad_()
    oc, og = F.embedding(idx, wc, padding_idx=padding), F.embedding(idx.cuda(), wg, padding_idx=padding)
    check(f"embedding forward, weight requires grad, padding={padding}", og, oc)
    up = torch.randn(2, 4, 64)
    (oc * up).sum().backward(); (og * up.cuda()).sum().backward()
    check(f"embedding backward, repeated rows, padding={padding}", wg.grad, wc.grad)
big = torch.randn(30522, 768)
ids = torch.randint(0, 30522, (2, 128))
check("embedding of a 30522x768 table", F.embedding(ids.cuda(), big.cuda()), F.embedding(ids, big))

# nll_loss and cross entropy, every reduction, ignore_index and class weights, and the backward.
logits, target = torch.randn(256, 300), torch.randint(0, 300, (256,))
target[3] = -100
cls_w = torch.rand(300) + 0.5
for reduction in ("mean", "sum", "none"):
    for w in (None, cls_w):
        lc, lg = logits.clone().requires_grad_(), logits.clone().cuda().requires_grad_()
        outc = F.nll_loss(F.log_softmax(lc, -1), target, weight=w, reduction=reduction)
        outg = F.nll_loss(F.log_softmax(lg, -1), target.cuda(), weight=None if w is None else w.cuda(),
                          reduction=reduction)
        check(f"nll_loss {reduction} weight={w is not None}", outg, outc, tol=1e-5)
        (outc * 1.7).sum().backward(); (outg * 1.7).sum().backward()
        check(f"nll_loss {reduction} weight={w is not None} backward", lg.grad, lc.grad, "", tol=1e-5)
one = torch.randn(300)
check("nll_loss of one row (1D)", F.nll_loss(one.cuda(), torch.tensor(7).cuda()), F.nll_loss(one, torch.tensor(7)))
everything_ignored = torch.full((4,), -100)
check("nll_loss mean with every row ignored (NaN)",
      torch.nan_to_num(F.nll_loss(logits[:4].cuda(), everything_ignored.cuda()), nan=123.0),
      torch.nan_to_num(F.nll_loss(logits[:4], everything_ignored), nan=123.0), "")
try:
    F.nll_loss(logits[:2].cuda(), torch.tensor([1, 300]).cuda())
    print("FAIL a target out of range raises"); fails.append("out of range target")
except IndexError:
    print(f"OK   {'a target out of range raises IndexError':52s}")
lc, lg = logits.clone().requires_grad_(), logits.clone().cuda().requires_grad_()
F.cross_entropy(lc, target).backward(); F.cross_entropy(lg, target.cuda()).backward()
check("cross_entropy backward end to end", lg.grad, lc.grad, "", tol=1e-5)

# Safe softmax: a fully masked row gives zeros.
s = torch.randn(4, 50)
s[1] = float("-inf")
s[2, 10:] = float("-inf")
check("_safe_softmax with a fully masked row", torch.ops.aten._safe_softmax(s.cuda(), -1),
      torch.ops.aten._safe_softmax(s, -1))
q = torch.randn(2, 4, 16, 32)
mask = torch.ones(16, 16, dtype=torch.bool).tril()
check("attention with a causal mask", F.scaled_dot_product_attention(q.cuda(), q.cuda(), q.cuda(), attn_mask=mask.cuda()),
      F.scaled_dot_product_attention(q, q, q, attn_mask=mask), "", tol=1e-5)

# Inference layer norm with parameters that require grad, past the old 1,048,576-value limit.
ln = torch.nn.LayerNorm(768)
with torch.no_grad():
    ln.weight.uniform_(0.5, 1.5); ln.bias.uniform_(-0.5, 0.5)
lng = torch.nn.LayerNorm(768).cuda()
lng.load_state_dict(ln.state_dict())
h = torch.randn(2, 1024, 768)
with torch.no_grad():
    check("layer_norm inference, 1.5M values, grad-requiring params", lng(h.cuda()), ln(h))
print("RESULT", "all passed" if not fails else f"{len(fails)} failed: {fails}")
