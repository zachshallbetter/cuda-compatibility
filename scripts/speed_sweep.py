"""Speed of everyday models under cuda-legacy, recorded so every #184 change shows its effect.

Usage on an sm_52 machine after setup:
    cuda-legacy run scripts/speed_sweep.py [--models gpt,bert,...] [--steps 3] [--json out.json] [--compare-cpu]

For each model it times a training step (forward, backward, optimizer) and an inference forward pass,
reports samples or tokens per second, and counts the bytes the CPU fallback copied to the host per
step. With --compare-cpu the same model and batch also run on the CPU, for the speed-up. It prints
one line per model and, with --json, writes the measurements with the environment they came from.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Model:
    name: str
    build: Callable[[], nn.Module]
    batch: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    unit: str  # what one sample is: "tokens" or "samples"
    per_batch: int  # units in one batch


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 8000, dim: int = 256, layers: int = 4, seq: int = 64) -> None:
        super().__init__()
        self.seq = seq
        self.embed, self.position = nn.Embedding(vocab, dim), nn.Parameter(torch.zeros(seq, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.blocks, self.norm, self.head = nn.TransformerEncoder(layer, layers), nn.LayerNorm(dim), nn.Linear(dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(self.seq, device=tokens.device)
        return self.head(self.norm(self.blocks(self.embed(tokens) + self.position, mask=mask, is_causal=True)))


def _gpt_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def _cnn() -> nn.Module:
    def block(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    return nn.Sequential(block(3, 16), block(16, 32), nn.MaxPool2d(2), block(32, 64), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(64, 10))


MODELS = {
    "mlp": Model("mlp", lambda: nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 10)),
                 lambda: (torch.randn(64, 256), torch.randint(0, 10, (64,))), F.cross_entropy, "samples", 64),
    "cnn": Model("cnn", _cnn, lambda: (torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))), F.cross_entropy,
                 "samples", 16),
    "gpt": Model("gpt", TinyGPT, lambda: (torch.randint(0, 8000, (8, 64)), torch.randint(0, 8000, (8, 64))), _gpt_loss,
                 "tokens", 8 * 64),
    "bert": Model("bert", lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.1, batch_first=True), 4),
                  lambda: (torch.randn(8, 64, 256), torch.randn(8, 64, 256)), F.mse_loss, "tokens", 8 * 64),
    "lstm": Model("lstm", lambda: nn.LSTM(64, 128, num_layers=2, batch_first=True),
                  lambda: (torch.randn(8, 32, 64), torch.randn(8, 32, 128)), lambda out, y: F.mse_loss(out[0], y),
                  "tokens", 8 * 32),
}


class CopyCounter:
    """Bytes the adapter's CPU fallback copied from the GPU to the host, if the adapter is loaded."""

    def __init__(self) -> None:
        self.bytes = 0
        try:
            from cuda_legacy_compat import torch_adapter
        except ImportError:
            return
        original = torch_adapter._cuda_tensor_to_cpu

        def counted(tensor: torch.Tensor) -> torch.Tensor:
            self.bytes += tensor.numel() * tensor.element_size()
            return original(tensor)
        torch_adapter._cuda_tensor_to_cpu = counted


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model: Model, device: torch.device, steps: int, counter: CopyCounter | None = None) -> dict:
    """Time `steps` training steps and inference passes of one model on one device, after one warm-up."""
    torch.manual_seed(0)
    net = model.build().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    inputs, target = (tensor.to(device) for tensor in model.batch())

    def train() -> float:
        optimizer.zero_grad()
        loss = model.loss(net(inputs), target)
        loss.backward()
        optimizer.step()
        return float(loss)

    train()
    _sync(device)
    copied = counter.bytes if counter else 0
    start = time.perf_counter()
    for _ in range(steps):
        loss = train()
    _sync(device)
    step = (time.perf_counter() - start) / steps
    copied = ((counter.bytes if counter else 0) - copied) / steps
    net.eval()
    with torch.no_grad():
        net(inputs)
        _sync(device)
        start = time.perf_counter()
        for _ in range(steps):
            net(inputs)
        _sync(device)
    forward = (time.perf_counter() - start) / steps
    return {"model": model.name, "device": str(device), "parameters": sum(p.numel() for p in net.parameters()),
            "step_s": round(step, 4), "forward_s": round(forward, 4),
            f"train_{model.unit}_per_s": round(model.per_batch / step, 1),
            f"infer_{model.unit}_per_s": round(model.per_batch / forward, 1),
            "host_copy_mib_per_step": round(copied / 2**20, 2), "final_loss": round(loss, 4)}


def environment() -> dict:
    gpu = torch.cuda.is_available()
    return {"python": platform.python_version(), "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if gpu else None,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if gpu else None,
            "cuda_legacy": _adapter_version()}


def _adapter_version() -> str | None:
    try:
        from cuda_legacy_compat import __version__
    except ImportError:
        return None
    return __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(MODELS), help="comma-separated: " + ", ".join(MODELS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--json", help="write the measurements to this file")
    parser.add_argument("--compare-cpu", action="store_true", help="also time each model on the CPU")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counter = CopyCounter() if device.type == "cuda" else None
    results = []
    for name in names:
        row = measure(MODELS[name], device, args.steps, counter)
        if args.compare_cpu and device.type == "cuda":
            cpu = measure(MODELS[name], torch.device("cpu"), args.steps)
            row["cpu_step_s"], row["speedup_vs_cpu"] = cpu["step_s"], round(cpu["step_s"] / row["step_s"], 2)
            row["cpu_forward_s"] = cpu["forward_s"]
            row["forward_speedup_vs_cpu"] = round(cpu["forward_s"] / row["forward_s"], 2)
        results.append(row)
        unit = MODELS[name].unit
        extra = (f"; vs CPU: train {row['speedup_vs_cpu']}x, forward {row['forward_speedup_vs_cpu']}x"
                 if "speedup_vs_cpu" in row else "")
        print(f"{name:5} {row['parameters'] / 1e6:5.1f}M params  step {row['step_s']:.3f} s "
              f"({row[f'train_{unit}_per_s']:.0f} {unit}/s)  forward {row['forward_s']:.3f} s  "
              f"host copies {row['host_copy_mib_per_step']:.1f} MiB/step{extra}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"environment": environment(), "steps": args.steps, "results": results}, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
