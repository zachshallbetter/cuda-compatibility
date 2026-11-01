"""Qualify registered PyTorch adapter routes and fail-fast fallback guards."""

from __future__ import annotations

from collections.abc import Callable
import statistics
import time

import torch
import torch.nn.functional as functional

from cuda_legacy_compat import torch_adapter


def _check_fallbacks(device_index: int) -> None:
    device = torch.device(f"cuda:{device_index}")
    vector = torch.linspace(-1.5, 1.5, 24).reshape(2, 3, 4)
    other = torch.linspace(0.25, 1.25, 24).reshape(2, 3, 4)
    query = torch.arange(64, dtype=torch.float32).reshape(1, 2, 4, 8) / 31
    key = query.clone()
    value = query.flip(-1).contiguous()
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 11
    bias = torch.linspace(-0.3, 0.3, 5)
    left_batch = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 13
    right_batch = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5) / 17
    cases: list[tuple[str, Callable, tuple[torch.Tensor, ...], str]] = [
        ("linear", lambda x: functional.linear(*x), (matrix, weight, bias), "cpu-fallback-linear"),
        ("bmm", lambda x: torch.bmm(*x), (left_batch, right_batch), "cpu-fallback-bmm"),
        ("softmax", lambda x: functional.softmax(x[0], dim=-1), (vector,), "cpu-fallback-softmax"),
        (
            "scaled_dot_product_attention",
            lambda x: functional.scaled_dot_product_attention(*x),
            (query, key, value),
            "cpu-fallback-scaled_dot_product_attention",
        ),
        ("gelu", lambda x: functional.gelu(x[0]), (vector,), "cpu-fallback-gelu"),
        ("mul", lambda x: torch.mul(*x), (vector, other), "cpu-fallback-mul"),
        ("div", lambda x: torch.div(*x), (other, vector + 2), "cpu-fallback-div"),
        ("sum", lambda x: torch.sum(x[0], dim=[1]), (vector,), "cpu-fallback-sum"),
        (
            "clamp_min",
            lambda x: torch.clamp_min(x[0], 0.0),
            (vector,),
            "cpu-fallback-clamp_min",
        ),
        (
            "cat",
            lambda x: torch.cat(x, dim=1),
            (vector, other),
            "cpu-fallback-cat",
        ),
        (
            "contiguous",
            lambda x: x[0].contiguous(),
            (torch.arange(24, dtype=torch.float32).reshape(4, 6).T,),
            "cpu-fallback-contiguous",
        ),
    ]
    for name, operation, cpu_inputs, expected_path in cases:
        stream = torch.cuda.Stream(device=device)
        complete = torch.cuda.Event()
        with torch.cuda.stream(stream), torch.inference_mode():
            cuda_inputs = tuple(
                tensor.pin_memory().to(device, non_blocking=True) for tensor in cpu_inputs
            )
            actual = operation(cuda_inputs)
            complete.record(stream)
        del cuda_inputs
        complete.synchronize()
        expected = operation(cpu_inputs)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
        path = torch_adapter.last_path()
        if path != expected_path:
            raise AssertionError(f"{name} used {path!r}, expected {expected_path!r}")
        print(f"PASS device={device_index} registered={name} path={path}", flush=True)


def _check_cross_stream_lifetime(
    device_index: int,
    name: str,
    cpu_inputs: tuple[torch.Tensor, ...],
    operation: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
    expected: torch.Tensor,
    expected_path: str,
) -> None:
    device = torch.device(f"cuda:{device_index}")
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    gate = torch.cuda.Stream(device=device)
    gate_source = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    gate_source.fill_(7)
    gate_buffer = torch.empty_like(gate_source, device=device)
    gate_ready = torch.cuda.Event()
    with torch.cuda.stream(gate):
        for _ in range(64):
            gate_buffer.copy_(gate_source, non_blocking=True)
        gate_ready.record(gate)

    producer_ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        cuda_inputs = tuple(
            tensor.pin_memory().to(device, non_blocking=True) for tensor in cpu_inputs
        )
        producer_ready.record(producer)

    consumer.wait_event(producer_ready)
    consumer.wait_event(gate_ready)
    if gate_ready.query():
        raise AssertionError(f"{name} lifetime gate completed before queuing the consumer")
    complete = torch.cuda.Event()
    with torch.cuda.stream(consumer), torch.inference_mode():
        output = operation(cuda_inputs)
        complete.record(consumer)
    input_pointers = {tensor.data_ptr() for tensor in cuda_inputs}
    input_specs = tuple((tensor.shape, tensor.dtype) for tensor in cpu_inputs)
    del cuda_inputs

    replacements: list[torch.Tensor] = []
    with torch.cuda.stream(producer):
        for index, (shape, dtype) in enumerate(input_specs):
            source = torch.empty(shape, dtype=dtype, pin_memory=True)
            source.fill_(index + 19)
            replacement = torch.empty(shape, dtype=dtype, device=device)
            replacement.copy_(source, non_blocking=True)
            replacements.append(replacement)
    reused = input_pointers.intersection(tensor.data_ptr() for tensor in replacements)
    if reused:
        raise AssertionError(f"{name} input storage was reused before its consumer stream completed")

    complete.synchronize()
    actual = output.cpu()
    absolute_error = float((actual - expected).abs().max()) if actual.numel() else 0.0
    # The larger layer-normalization values expose CPU/GPU reduction-order variance.
    atol = 5e-5 if name == "layernorm-f32" else 1e-6
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=atol)
    path = torch_adapter.last_path()
    if path != expected_path:
        raise AssertionError(f"{name} used {path!r}, expected {expected_path!r}")
    print(
        f"PASS device={device_index} lifetime={name} stream=cross-stream "
        f"premature_reuse=none max_abs_error={absolute_error:.3g} path={path}",
        flush=True,
    )


def _check_native_stream_lifetimes(device_index: int) -> None:
    vector = torch.linspace(-1, 1, 2048, dtype=torch.float32)
    vector_right = torch.linspace(1, -1, 2048, dtype=torch.float32)
    integer = torch.arange(2048, dtype=torch.int32)
    integer_right = torch.full((2048,), 11, dtype=torch.int32)
    left = torch.arange(32 * 24, dtype=torch.float32).reshape(32, 24) / 31
    right = torch.arange(24 * 20, dtype=torch.float32).reshape(24, 20) / 17
    relu_input = torch.linspace(-2, 2, 4096, dtype=torch.float32)
    norm_input = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32) / 101
    norm_weight = torch.linspace(0.5, 1.5, 32)
    norm_bias = torch.linspace(-0.25, 0.25, 32)
    cases = (
        (
            "add-f32",
            (vector, vector_right),
            lambda x: torch.add(*x),
            vector + vector_right,
            "llvm-sm52-f32-add",
        ),
        (
            "add-i32",
            (integer, integer_right),
            lambda x: torch.add(*x),
            integer + integer_right,
            "llvm-sm52-int32",
        ),
        (
            "add-f16",
            (torch.linspace(-1, 1, 2048, dtype=torch.float16),
             torch.linspace(1, -1, 2048, dtype=torch.float16)),
            lambda x: torch.add(*x),
            torch.add(
                torch.linspace(-1, 1, 2048, dtype=torch.float16),
                torch.linspace(1, -1, 2048, dtype=torch.float16),
            ),
            "llvm-sm52-f16-add",
        ),
        (
            "mm-f32",
            (left, right),
            lambda x: torch.mm(*x),
            torch.mm(left, right),
            "llvm-sm52-mm-f32",
        ),
        (
            "relu-f32",
            (relu_input,),
            lambda x: torch.relu(x[0]),
            torch.relu(relu_input),
            "llvm-sm52-relu-f32",
        ),
        (
            "layernorm-f32",
            (norm_input, norm_weight, norm_bias),
            lambda x: functional.layer_norm(x[0], (32,), x[1], x[2]),
            functional.layer_norm(norm_input, (32,), norm_weight, norm_bias),
            "llvm-sm52-layernorm-f32",
        ),
    )
    for name, inputs, operation, expected, path in cases:
        _check_cross_stream_lifetime(device_index, name, inputs, operation, expected, path)

    device = torch.device(f"cuda:{device_index}")
    stream = torch.cuda.Stream(device=device)
    complete = torch.cuda.Event()
    index_cpu = torch.tensor([[1, 7, 3], [3, 1, 7]], dtype=torch.int64)
    weight_cpu = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 13
    with torch.cuda.stream(stream), torch.inference_mode():
        indices = index_cpu.pin_memory().to(device, non_blocking=True)
        weight = weight_cpu.pin_memory().to(device, non_blocking=True)
        embedded = functional.embedding(indices, weight)
        ranged = torch.arange(67, dtype=torch.int64, device=device)
        complete.record(stream)
    del indices, weight
    complete.synchronize()
    torch.testing.assert_close(
        embedded.cpu(), functional.embedding(index_cpu, weight_cpu), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(ranged.cpu(), torch.arange(67, dtype=torch.int64))
    if torch_adapter.last_path() != "llvm-sm52-arange-i64":
        raise AssertionError(f"arange used unexpected path {torch_adapter.last_path()!r}")
    print(
        f"PASS device={device_index} lifetime=embedding+arange stream=non-default "
        "embedding_sync=observed",
        flush=True,
    )


def _check_fail_fast_guards(device_index: int) -> None:
    device = torch.device(f"cuda:{device_index}")
    bridge, _ = torch_adapter._load_bridge()
    copy_calls = 0

    class CopyCounter:
        def __getattr__(self, name: str):
            attribute = getattr(bridge, name)
            if name != "cuda_compat_copy_device_to_host":
                return attribute

            def counted(*args):
                nonlocal copy_calls
                copy_calls += 1
                return attribute(*args)

            return counted

    torch_adapter._bridge = CopyCounter()
    try:
        oversized_left = torch.empty(8_388_609, dtype=torch.float32, device=device)
        oversized_right = torch.empty_like(oversized_left)
        with torch.inference_mode():
            try:
                torch.add(oversized_left, oversized_right, alpha=2)
            except NotImplementedError as exc:
                if "64 MiB" not in str(exc):
                    raise
            else:
                raise AssertionError("aggregate fallback input cap was not enforced")
        if copy_calls:
            raise AssertionError("over-cap fallback copied CUDA inputs before rejecting")
        print(f"PASS device={device_index} fallback=aggregate-input-cap-before-copy", flush=True)

        mixed_gpu = torch.ones((4,), dtype=torch.float32).to(device)
        mixed_cpu = torch.ones((4,), dtype=torch.float32)
        with torch.inference_mode():
            try:
                torch.cat((mixed_gpu, mixed_cpu))
            except NotImplementedError:
                pass
            else:
                raise AssertionError("mixed CPU/CUDA fallback inputs were silently combined")
        if copy_calls:
            raise AssertionError("mixed-device fallback copied CUDA inputs before rejecting")
        print(f"PASS device={device_index} fallback=mixed-device-rejected-before-copy", flush=True)
    finally:
        torch_adapter._bridge = bridge


def _check_training(device_index: int) -> None:
    device = torch.device(f"cuda:{device_index}")
    input_cpu = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    weight_cpu = torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float32)
    bias_cpu = torch.tensor([0.1, -0.2], dtype=torch.float32)
    reference_weight = weight_cpu.clone().requires_grad_()
    reference_bias = bias_cpu.clone().requires_grad_()
    reference_optimizer = torch.optim.SGD(
        [reference_weight, reference_bias], lr=0.05, foreach=False
    )

    weight = weight_cpu.to(device).requires_grad_()
    bias = bias_cpu.to(device).requires_grad_()
    optimizer = torch.optim.SGD([weight, bias], lr=0.05, foreach=False)
    device_input = input_cpu.to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    latencies_ms = []
    for _ in range(7):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = functional.linear(
            input_cpu, reference_weight, reference_bias
        ).relu().sum()
        reference_loss.backward()
        reference_optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = functional.linear(device_input, weight, bias).relu().sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    torch.testing.assert_close(weight.grad.cpu(), reference_weight.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias.grad.cpu(), reference_bias.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight.cpu(), reference_weight, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias.cpu(), reference_bias, rtol=1e-5, atol=1e-6)
    median_latency = statistics.median(latencies_ms)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    print(
        f"PASS device={device_index} training=linear-relu-sgd gradients=cpu-reference "
        f"optimizer_update=sm52-f32-add-inplace steps=7 median_ms={median_latency:.3f} "
        f"peak_allocated_bytes={peak_allocated}",
        flush=True,
    )


def run_registered_operator_probe(device_index: int) -> None:
    torch_adapter.install()
    _check_fallbacks(device_index)
    _check_native_stream_lifetimes(device_index)
    _check_fail_fast_guards(device_index)
    _check_training(device_index)
