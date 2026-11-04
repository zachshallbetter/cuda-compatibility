"""Small-model sweep for cuda-legacy on sm_52: which everyday architectures run, and which operator stops the rest.

Run on an sm_52 machine after setup: cuda-legacy run scripts/model_sweep.py
Each model is built from torch.nn with seeded random weights (no downloads). For each it runs inference and two
training steps on the GPU and the CPU side by side, compares outputs and parameters, and on failure records the
first operator named in the error. Prints one JSON object with per-model results and a ranked list of blockers.
"""
import copy
import json
import re
import time

import torch
from torch import nn
import torch.nn.functional as F

R = {}


def first_op(msg):
    m = re.search(r"(aten[:.]+[\w.]+)|operator (\S+)|fallback for (\S+)", msg)
    return next((g for g in m.groups() if g), None) if m else None


def run(name, build, batch, loss_fn, opt=lambda p: torch.optim.SGD(p, lr=0.01), train=True, exact=True):
    rec = {}
    try:
        torch.manual_seed(0)
        cpu = build()
        gpu = copy.deepcopy(cpu).cuda()
        x, y = batch()
        cpu.eval(); gpu.eval()
        with torch.no_grad():
            oc, og = cpu(*x), gpu(*[t.cuda() for t in x])
        oc = oc[0] if isinstance(oc, tuple) else oc
        og = og[0] if isinstance(og, tuple) else og
        rec["infer_diff"] = float((og.cpu() - oc).abs().max())
        rec["infer_ok"] = rec["infer_diff"] < 1e-3
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"
        R[name] = {"infer_ok": False, "stage": "inference", "error": msg, "op": first_op(msg)}
        return
    if train:
        try:
            cpu.train(); gpu.train()
            oc_, og_ = opt(cpu.parameters()), opt(gpu.parameters())
            for _ in range(2):
                for m, o, dev in ((cpu, oc_, None), (gpu, og_, "cuda")):
                    o.zero_grad()
                    xs = [t.cuda() for t in x] if dev else x
                    out = m(*xs)
                    out = out[0] if isinstance(out, tuple) else out
                    loss = loss_fn(out, y.cuda() if dev else y)
                    loss.backward()
                    o.step()
            rec["param_diff"] = max(float((a.detach().cpu() - b.detach()).abs().max())
                                    for a, b in zip(gpu.parameters(), cpu.parameters()))
            rec["train_ok"] = rec["param_diff"] < 1e-3 or not exact
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"
            rec.update(train_ok=False, stage="training", error=msg, op=first_op(msg))
    R[name] = rec


def seq_cls(vocab=50, d=16):
    return lambda: (([torch.randint(0, vocab, (4, 10))]), torch.randint(0, 3, (4,)))


class ResBlock(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.a, s.b = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.Conv2d(c, c, 3, padding=1, bias=False)
        s.na, s.nb = nn.BatchNorm2d(c), nn.BatchNorm2d(c)
    def forward(s, x):
        return F.relu(x + s.nb(s.b(F.relu(s.na(s.a(x))))))


class SE(nn.Module):
    def __init__(s, c):
        super().__init__(); s.f = nn.Sequential(nn.Linear(c, c // 2), nn.ReLU(), nn.Linear(c // 2, c), nn.Sigmoid())
    def forward(s, x):
        return x * s.f(x.mean((2, 3)))[:, :, None, None]


class RNNCls(nn.Module):
    def __init__(s, cell):
        super().__init__(); s.e = nn.Embedding(50, 16); s.r = cell(16, 24, batch_first=True); s.h = nn.Linear(24, 3)
    def forward(s, x):
        o, _ = s.r(s.e(x)); return s.h(o[:, -1])


class GPT(nn.Module):
    def __init__(s, v=50, d=32, L=12):
        super().__init__()
        s.tok, s.pos = nn.Embedding(v, d), nn.Embedding(L, d)
        s.blk = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
        s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, v, bias=False); s.head.weight = s.tok.weight
    def forward(s, x):
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))
        return s.head(s.ln(s.blk(h, src_mask=mask)))


class BERT(nn.Module):
    def __init__(s):
        super().__init__()
        s.e = nn.Embedding(50, 32)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True), 2,
                                      enable_nested_tensor=False)
        s.cls = nn.Linear(32, 3)
    def forward(s, x):
        pad = x == 0
        return s.cls(s.enc(s.e(x), src_key_padding_mask=pad)[:, 0])


class UNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.d1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        s.d2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        s.up = nn.ConvTranspose2d(16, 8, 2, stride=2)
        s.o = nn.Conv2d(16, 2, 1)
    def forward(s, x):
        a = s.d1(x); b = s.d2(a)
        u = s.up(b)
        u2 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)[:, :8]
        return s.o(torch.cat([a, u + u2], 1))


class ViT(nn.Module):
    def __init__(s):
        super().__init__()
        s.p = nn.Conv2d(3, 32, 4, stride=4); s.cls = nn.Parameter(torch.zeros(1, 1, 32))
        s.blk = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, activation="gelu")
        s.h = nn.Linear(32, 5)
    def forward(s, x):
        t = s.p(x).flatten(2).transpose(1, 2)
        t = torch.cat([s.cls.expand(x.shape[0], -1, -1), t], 1)
        return s.h(s.blk(t)[:, 0])


ce = nn.functional.cross_entropy
mse = nn.functional.mse_loss
img = lambda c=3, hw=16, k=4: (lambda: ([torch.randn(4, c, hw, hw)], torch.randint(0, k, (4,))))
t0 = time.time()
run("mlp_gelu_adamw", lambda: nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)),
    lambda: ([torch.randn(8, 16)], torch.randn(8, 4)), mse, opt=lambda p: torch.optim.AdamW(p, lr=1e-3))
run("resnet_block_bn", lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), ResBlock(8),
                                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)), img(), ce)
run("mobilenet_dw_se_hswish", lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Hardswish(),
                                                     nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.BatchNorm2d(8), nn.ReLU6(),
                                                     SE(8), nn.Conv2d(8, 16, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                                     nn.Linear(16, 4)), img(), ce)
run("groupnorm_silu_cnn", lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.SiLU(),
                                                 nn.AvgPool2d(2), nn.Flatten(), nn.Linear(8 * 8 * 8, 4)), img(), ce)
run("instancenorm_leaky", lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.InstanceNorm2d(8, affine=True),
                                                 nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(8 * 16 * 16, 4)), img(), ce)
run("lstm_classifier", lambda: RNNCls(nn.LSTM), seq_cls(), ce)
run("gru_classifier", lambda: RNNCls(nn.GRU), seq_cls(), ce)
run("conv1d_audio", lambda: nn.Sequential(nn.Conv1d(1, 8, 9, padding=4), nn.BatchNorm1d(8), nn.GELU(), nn.MaxPool1d(4),
                                          nn.Flatten(), nn.Linear(8 * 32, 4)),
    lambda: ([torch.randn(4, 1, 128)], torch.randint(0, 4, (4,))), ce)
run("gpt_tiny_causal", GPT, lambda: ([torch.randint(0, 50, (2, 12))], torch.randint(0, 50, (2, 12))),
    lambda o, y: ce(o.reshape(-1, o.shape[-1]), y.reshape(-1)), opt=lambda p: torch.optim.AdamW(p, lr=1e-3))
run("bert_tiny_padmask", BERT, lambda: ([torch.cat([torch.randint(1, 50, (4, 7)), torch.zeros(4, 3, dtype=torch.long)], 1)],
                                        torch.randint(0, 3, (4,))), ce, opt=lambda p: torch.optim.AdamW(p, lr=1e-3))
run("vit_tiny_patch", ViT, img(3, 16, 5), lambda o, y: ce(o, y, label_smoothing=0.1),
    opt=lambda p: torch.optim.AdamW(p, lr=1e-3))
run("unet_small", UNet, lambda: ([torch.randn(2, 3, 16, 16)], torch.randn(2, 2, 16, 16)), mse,
    opt=lambda p: torch.optim.Adam(p, lr=1e-3))
class _EB(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.EmbeddingBag(50, 16, mode="mean"); s.h = nn.Linear(16, 3)
    def forward(s, x):
        return s.h(s.e(x))


run("embeddingbag_mean", _EB, lambda: ([torch.randint(0, 50, (4, 6))], torch.randint(0, 3, (4,))), ce)
run("bce_logits_rmsprop", lambda: nn.Sequential(nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 1)),
    lambda: ([torch.randn(8, 16)], torch.randint(0, 2, (8, 1)).float()), nn.functional.binary_cross_entropy_with_logits,
    opt=lambda p: torch.optim.RMSprop(p, lr=1e-3))
run("smoothl1_adagrad", lambda: nn.Sequential(nn.Linear(16, 8), nn.ELU(), nn.Linear(8, 2)),
    lambda: ([torch.randn(8, 16)], torch.randn(8, 2)), nn.functional.smooth_l1_loss, opt=lambda p: torch.optim.Adagrad(p, lr=1e-2))
run("dropout_train_runs", lambda: nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 4)),
    lambda: ([torch.randn(8, 16)], torch.randn(8, 4)), mse, exact=False)

blockers = {}
for n, r in R.items():
    if r.get("op") and not (r.get("infer_ok") and r.get("train_ok", True)):
        blockers.setdefault(r["op"], []).append(n)
ok = [n for n, r in R.items() if r.get("infer_ok") and r.get("train_ok", True)]
print(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "seconds": round(time.time() - t0, 1),
                  "passing": ok, "blockers": dict(sorted(blockers.items(), key=lambda kv: -len(kv[1]))), "models": R},
                 indent=1))
