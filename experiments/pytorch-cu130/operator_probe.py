"""Exercise the opt-in CUDA 13 PyTorch tensor-add adapter."""

from __future__ import annotations

import ctypes
import os

import torch

from cuda_legacy_compat import torch_adapter as aten_compat
from registered_operator_probe import run_registered_operator_probe


def _check_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    equal_nan: bool = False,
) -> None:
    torch.testing.assert_close(
        actual, expected, rtol=1e-5, atol=1e-6, equal_nan=equal_nan, msg=label
    )


def main() -> int:
    print(f"torch={torch.__version__} built_cuda={torch.version.cuda}", flush=True)
    runtime_version = ctypes.c_int()
    runtime_query = ctypes.CDLL(None).cudaRuntimeGetVersion
    runtime_query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime_query.restype = ctypes.c_int
    runtime_status = runtime_query(ctypes.byref(runtime_version))
    if os.environ.get("CUDA_COMPAT_ENABLE") == "1":
        if runtime_status != 0 or runtime_version.value != 13000:
            raise AssertionError(
                f"runtime facade mismatch status={runtime_status} version={runtime_version.value}"
            )
        print("PASS runtime_facade=13.0.0", flush=True)
    if os.environ.get("CUDA_COMPAT_ENABLE") != "1":
        print("adapter not enabled; baseline probe only")
        return 0
    aten_compat.install()
    for device_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_index)
        print(
            f"device={device_index} name={properties.name} "
            f"compute={properties.major}.{properties.minor}",
            flush=True,
        )
        with torch.cuda.device(device_index):
            arange_cuda = torch.arange(64, dtype=torch.int64, device=f"cuda:{device_index}")
            if not torch.equal(arange_cuda.cpu(), torch.arange(64, dtype=torch.int64)):
                raise AssertionError("sm_52 arange differs from PyTorch CPU reference")
            if aten_compat.last_path() != "llvm-sm52-arange-i64":
                raise AssertionError(f"unexpected arange path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} op=arange dtype=int64 elements=64", flush=True)

            embedding_weight_cpu = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 13
            embedding_indices_cpu = torch.tensor([1, 7, 3], dtype=torch.int64).unsqueeze(0).expand(2, 3)
            with torch.inference_mode():
                embedding_indices = (
                    embedding_indices_cpu[0].to(f"cuda:{device_index}")
                    .unsqueeze(0)
                    .expand(2, 3)
                )
                embedding_output = torch.nn.functional.embedding(
                    embedding_indices,
                    embedding_weight_cpu.to(f"cuda:{device_index}"),
                )
            _check_close(
                embedding_output.cpu(),
                torch.nn.functional.embedding(embedding_indices_cpu, embedding_weight_cpu),
                label="embedding CPU reference",
            )
            if aten_compat.last_path() != "llvm-sm52-embedding-f32":
                raise AssertionError(f"unexpected embedding path: {aten_compat.last_path()}")
            print(
                f"PASS device={device_index} op=embedding dtype=float32 elements=24 "
                f"indices_strides={embedding_indices.stride()}",
                flush=True,
            )

            with torch.inference_mode():
                invalid_indices = torch.tensor([16], dtype=torch.int64).to(
                    f"cuda:{device_index}"
                )
                try:
                    torch.nn.functional.embedding(
                        invalid_indices, embedding_weight_cpu.to(f"cuda:{device_index}")
                    )
                except IndexError:
                    print(f"PASS device={device_index} embedding=out-of-range-rejected", flush=True)
                else:
                    raise AssertionError("out-of-range embedding index did not fail")

            norm_input_cpu = (torch.arange(128, dtype=torch.float32).reshape(4, 32) - 63) / 17
            norm_weight_cpu = torch.linspace(0.5, 1.5, 32)
            norm_bias_cpu = torch.linspace(-0.25, 0.25, 32)
            with torch.inference_mode():
                norm_output = torch.nn.functional.layer_norm(
                    norm_input_cpu.to(f"cuda:{device_index}"), (32,),
                    norm_weight_cpu.to(f"cuda:{device_index}"),
                    norm_bias_cpu.to(f"cuda:{device_index}"),
                )
                norm_reference = torch.nn.functional.layer_norm(
                    norm_input_cpu, (32,), norm_weight_cpu, norm_bias_cpu
                )
            _check_close(norm_output.cpu(), norm_reference, label="layer normalization CPU reference")
            if aten_compat.last_path() != "llvm-sm52-layernorm-f32":
                raise AssertionError(f"unexpected layer norm path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} op=layer_norm dtype=float32 elements=128", flush=True)

            left32_cpu = torch.arange(1024, dtype=torch.int32)
            right32_cpu = torch.full((1024,), 1024, dtype=torch.int32)
            left32 = left32_cpu.to("cuda")
            right32 = right32_cpu.to("cuda")
            result32 = (left32 + right32).cpu()
            expected32 = torch.add(left32_cpu, right32_cpu)
            if not torch.equal(result32, expected32):
                raise AssertionError("int32 GPU kernel result differs from PyTorch CPU reference")
            if aten_compat.last_path() != "llvm-sm52-int32":
                raise AssertionError(f"unexpected int32 path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} dtype=int32 path=llvm-sm52 elements=1024", flush=True)

            cached_contexts = aten_compat.cached_context_count()
            if cached_contexts < 1:
                raise AssertionError("PTX module was not retained in the context cache")
            repeated32 = (left32 + right32).cpu()
            if not torch.equal(repeated32, expected32):
                raise AssertionError("repeated int32 GPU kernel result differs from CPU reference")
            if aten_compat.cached_context_count() != cached_contexts:
                raise AssertionError("repeated addition loaded another PTX module for the same context")
            print(f"PASS device={device_index} module_cache=stable contexts={cached_contexts}", flush=True)

            bridge, ptx_path = aten_compat._load_bridge()
            alternate_ptx_path = os.fsencode(os.environ["CUDA_COMPAT_ALTERNATE_PTX"])
            mismatch = bridge.cuda_compat_launch_vector_add(
                alternate_ptx_path,
                left32.data_ptr(),
                right32.data_ptr(),
                left32.data_ptr(),
                left32.numel(),
                torch.cuda.current_stream(device=left32.device).cuda_stream,
            )
            if mismatch == 0 or aten_compat.cached_context_count() != cached_contexts:
                raise AssertionError("PTX path change did not fail closed for a cached CUDA context")
            print(f"PASS device={device_index} module_cache=reject-path-change", flush=True)

            overflow_count = bridge.cuda_compat_launch_vector_add(
                ptx_path,
                left32.data_ptr(),
                right32.data_ptr(),
                left32.data_ptr(),
                1 << 31,
                torch.cuda.current_stream(device=left32.device).cuda_stream,
            )
            if overflow_count == 0:
                raise AssertionError("bridge accepted an element count outside the kernel ABI")
            print(f"PASS device={device_index} kernel_abi=reject-count-overflow", flush=True)

            empty = torch.empty((0,), dtype=torch.int32, device="cuda")
            empty_result = empty + empty
            if empty_result.numel() != 0 or aten_compat.last_path() != "llvm-sm52-int32-empty":
                raise AssertionError("empty int32 addition did not preserve the empty tensor result")
            print(f"PASS device={device_index} dtype=int32 empty=handled", flush=True)

            stream = torch.cuda.Stream(device=left32.device)
            stream_event = torch.cuda.Event()
            with torch.cuda.stream(stream):
                stream_left = left32_cpu.to(left32.device, non_blocking=True)
                stream_right = right32_cpu.to(right32.device, non_blocking=True)
                stream_result = stream_left + stream_right
                stream_event.record(stream)
            stream_event.synchronize()
            if not torch.equal(stream_result.cpu(), expected32):
                raise AssertionError("non-default-stream int32 result differs from CPU reference")
            if aten_compat.last_path() != "llvm-sm52-int32":
                raise AssertionError(f"unexpected non-default-stream path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} stream=non-default elements=1024", flush=True)

            overflow_left = torch.tensor([2147483647, -1], dtype=torch.int32).to("cuda")
            overflow_right = torch.tensor([1, 1], dtype=torch.int32).to("cuda")
            overflow_result = (overflow_left + overflow_right).cpu()
            expected_overflow = torch.tensor([-2147483648, 0], dtype=torch.int32)
            if not torch.equal(overflow_result, expected_overflow):
                raise AssertionError("int32 overflow behavior differs from PyTorch CPU reference")
            print(f"PASS device={device_index} dtype=int32 overflow=wrap", flush=True)

            left64_cpu = torch.arange(1024, dtype=torch.int64)
            right64_cpu = torch.full((1024,), 1024, dtype=torch.int64)
            left64 = left64_cpu.to("cuda")
            right64 = right64_cpu.to("cuda")
            output64 = left64 + right64
            if not output64.is_cuda:
                raise AssertionError("int64 CPU fallback did not return a CUDA tensor")
            result64 = output64.cpu()
            expected64 = torch.add(left64_cpu, right64_cpu)
            if not torch.equal(result64, expected64):
                raise AssertionError("int64 CPU fallback result differs from PyTorch CPU reference")
            if aten_compat.last_path() != "pytorch-cpu-int64":
                raise AssertionError(f"unexpected int64 path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} dtype=int64 path=cpu-fallback elements=1024", flush=True)

            left32_float = torch.arange(1024, dtype=torch.float32).to("cuda")
            right32_float = torch.ones(1024, dtype=torch.float32).to("cuda")
            float32_sum = (left32_float + right32_float).cpu()
            float32_reference = torch.add(left32_float.cpu(), right32_float.cpu())
            if not torch.equal(float32_sum, float32_reference):
                raise AssertionError("float32 vector addition differs from PyTorch CPU reference")
            if aten_compat.last_path() != "llvm-sm52-f32-add":
                raise AssertionError(f"unexpected float32 add path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} dtype=float32 add=llvm-sm52 elements=1024", flush=True)

            half_left_cpu = torch.tensor(
                [float("nan"), float("-inf"), -1.0, -0.0, 0.0, 0.5, float("inf"), 1.0],
                dtype=torch.float16,
            )
            half_right_cpu = torch.tensor(
                [1.0, 1.0, 1.0, 0.0, -0.0, 0.25, float("-inf"), 0.00048828125],
                dtype=torch.float16,
            )
            with torch.inference_mode():
                half_result = torch.add(
                    half_left_cpu.to(f"cuda:{device_index}"),
                    half_right_cpu.to(f"cuda:{device_index}"),
                ).cpu()
            half_reference = torch.add(half_left_cpu, half_right_cpu)
            torch.testing.assert_close(
                half_result, half_reference, rtol=0, atol=0, equal_nan=True
            )
            if aten_compat.last_path() != "llvm-sm52-f16-add":
                raise AssertionError(f"unexpected float16 add path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} dtype=float16 add=llvm-sm52 edge-values", flush=True)

            broadcast_left_cpu = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
            broadcast_right_cpu = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            with torch.inference_mode():
                broadcast_sum = torch.add(
                    broadcast_left_cpu.to(f"cuda:{device_index}"),
                    broadcast_right_cpu.to(f"cuda:{device_index}"),
                )
            _check_close(
                broadcast_sum.cpu(),
                torch.add(broadcast_left_cpu, broadcast_right_cpu),
                label="broadcast addition CPU fallback",
            )
            if aten_compat.last_path() != "cpu-fallback-add":
                raise AssertionError(f"unexpected broadcast add path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} add=broadcast-cpu-fallback shape=2x3x4", flush=True)

            sigmoid_input_cpu = torch.linspace(-3, 3, 24, dtype=torch.float32).reshape(2, 3, 4)
            with torch.inference_mode():
                sigmoid_output = torch.sigmoid(sigmoid_input_cpu.to(f"cuda:{device_index}"))
            _check_close(
                sigmoid_output.cpu(),
                torch.sigmoid(sigmoid_input_cpu),
                label="automatic unsupported-op CPU fallback",
            )
            if aten_compat.last_path() != "cpu-fallback-aten::sigmoid":
                raise AssertionError(f"unexpected automatic fallback path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} sigmoid=automatic-cpu-fallback", flush=True)

            with torch.inference_mode():
                inplace_value = torch.ones(4, dtype=torch.float32).to(f"cuda:{device_index}")
                try:
                    inplace_value.add_(1)
                except NotImplementedError:
                    print(f"PASS device={device_index} inplace-add=fail-closed", flush=True)
                else:
                    raise AssertionError("scalar in-place add unexpectedly became supported")
                overlap_base = torch.ones(5, dtype=torch.float32).to(f"cuda:{device_index}")
                try:
                    overlap_base[:4].add_(overlap_base[1:])
                except NotImplementedError:
                    if not torch.equal(overlap_base.cpu(), torch.ones(5)):
                        raise AssertionError("rejected overlapping in-place add changed its input")
                else:
                    raise AssertionError("partially overlapping in-place add was accepted")
                exact_alias = torch.ones(4, dtype=torch.float32).to(f"cuda:{device_index}")
                exact_alias.add_(exact_alias, alpha=2)
                if not torch.equal(exact_alias.cpu(), torch.full((4,), 3.0)):
                    raise AssertionError("exact-alias in-place add has incorrect semantics")

            with torch.inference_mode():
                oversized_left = torch.empty(
                    8_388_609, dtype=torch.float32, device=f"cuda:{device_index}"
                )
                oversized_right = torch.empty_like(oversized_left)
                try:
                    torch.add(oversized_left, oversized_right, alpha=2)
                except NotImplementedError as exc:
                    if "64 MiB" not in str(exc):
                        raise
                    print(f"PASS device={device_index} cpu-fallback=input-cap=64MiB", flush=True)
                else:
                    raise AssertionError("oversized CPU fallback input did not fail explicitly")

            matrix_cases = ((3, 5, 4), (1, 7, 1), (2, 0, 3), (0, 5, 2))
            for rows, inner, columns in matrix_cases:
                left_cpu = torch.arange(rows * inner, dtype=torch.float32).reshape(rows, inner)
                right_cpu = torch.arange(inner * columns, dtype=torch.float32).reshape(inner, columns)
                left_device = left_cpu.to("cuda")
                right_device = right_cpu.to("cuda")
                actual = torch.mm(left_device, right_device).cpu()
                expected = torch.mm(left_cpu, right_cpu)
                _check_close(actual, expected, label=f"mm {rows}x{inner} by {inner}x{columns}")
                expected_path = "llvm-sm52-mm-f32-empty" if rows == 0 or columns == 0 else "llvm-sm52-mm-f32"
                if aten_compat.last_path() != expected_path:
                    raise AssertionError(f"unexpected mm path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} dtype=float32 mm=shape-and-k-zero-cases", flush=True)

            relu_cpu = torch.tensor(
                [float("nan"), float("-inf"), -1.0, -0.0, 0.0, 0.5, float("inf")],
                dtype=torch.float32,
            )
            relu_result = torch.relu(relu_cpu.to("cuda")).cpu()
            relu_expected = torch.relu(relu_cpu)
            _check_close(
                relu_result,
                relu_expected,
                label="relu float32 edge values",
                equal_nan=True,
            )
            if not torch.equal(torch.signbit(relu_result), torch.signbit(relu_expected)):
                raise AssertionError("ReLU changed the sign bit of a zero compared with CPU PyTorch")
            if aten_compat.last_path() != "llvm-sm52-relu-f32":
                raise AssertionError(f"unexpected ReLU path: {aten_compat.last_path()}")
            print(f"PASS device={device_index} dtype=float32 relu=edge-values", flush=True)

            mlp_input_cpu = (torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) % 23 - 11) / 11
            first_weight_cpu = torch.linspace(-0.25, 0.25, 16 * 32).reshape(16, 32)
            second_weight_cpu = torch.linspace(-0.3, 0.3, 32 * 8).reshape(32, 8)
            expected_mlp = torch.mm(
                torch.relu(torch.mm(mlp_input_cpu, first_weight_cpu)), second_weight_cpu
            )
            inference_stream = torch.cuda.Stream(device=f"cuda:{device_index}")
            inference_complete = torch.cuda.Event()
            with torch.cuda.stream(inference_stream):
                mlp_input = mlp_input_cpu.to(f"cuda:{device_index}", non_blocking=True)
                first_weight = first_weight_cpu.to(f"cuda:{device_index}", non_blocking=True)
                second_weight = second_weight_cpu.to(f"cuda:{device_index}", non_blocking=True)
                hidden = torch.relu(torch.mm(mlp_input, first_weight))
                actual_mlp = torch.mm(hidden, second_weight)
                inference_complete.record(inference_stream)
            inference_complete.synchronize()
            _check_close(actual_mlp.cpu(), expected_mlp, label="two-layer MLP inference")
            if aten_compat.last_path() != "llvm-sm52-mm-f32":
                raise AssertionError(f"unexpected final MLP path: {aten_compat.last_path()}")
            print(
                f"PASS device={device_index} workload=two-layer-fp32-mlp shape=8x16x32x8 "
                "stream=non-default rtol=1e-5 atol=1e-6",
                flush=True,
            )

            noncontiguous_left = torch.arange(15, dtype=torch.float32).reshape(3, 5).to("cuda").T
            noncontiguous_right = torch.ones((3, 2), dtype=torch.float32).to("cuda")
            noncontiguous_output = torch.mm(noncontiguous_left, noncontiguous_right).cpu()
            noncontiguous_reference = torch.mm(
                torch.arange(15, dtype=torch.float32).reshape(3, 5).T,
                torch.ones((3, 2), dtype=torch.float32),
            )
            _check_close(
                noncontiguous_output,
                noncontiguous_reference,
                label="strided matrix multiplication",
            )
            if aten_compat.last_path() != "llvm-sm52-mm-f32":
                raise AssertionError(f"strided mm used unexpected path {aten_compat.last_path()}")
            print(f"PASS device={device_index} mm=transposed-input-native", flush=True)

            run_registered_operator_probe(device_index)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
