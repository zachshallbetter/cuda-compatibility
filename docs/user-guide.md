# CUDA Compatibility user guide

CUDA Compatibility is an experimental tool for running selected PyTorch CUDA 13 operations on NVIDIA `sm_52` GPUs. The short command is `cuda-legacy`; `cuda-legacy-compat` remains available as an alias. Today, setup installs a Python package in one environment and `cuda-legacy run` adapts one Python process. It does not install a machine-wide shim or make every CUDA 13 application compatible.

## 1. Install the package

From a source checkout, create and activate an environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

For development, use `python -m pip install -e .`. The package has no Python runtime dependencies and bundles no NVIDIA files. Install it into the same environment as the PyTorch CUDA wheel you want to run; the launcher uses that environment's Python interpreter.

### Where the command is installed

- **Recommended — project environment:** install into the active `venv` or Conda environment that already contains the target PyTorch wheel. On Linux, the command is placed in that environment's `bin/` directory (for example, `.venv/bin/cuda-legacy`); activating the environment puts it on `PATH`.
- **User site — only when that Python already has PyTorch:** outside a virtual environment, `python3 -m pip install --user .` puts the command in Python's user scripts directory. Find the base with `python3 -m site --user-base`; on many Linux systems the scripts directory is `~/.local/bin`. Add that directory to `PATH` if needed. Python ignores `--user` installs inside an active virtual environment.
- **System directories:** this project does not install to `/usr/local/bin` or modify system Python. Avoid `sudo pip install`.

`pipx` normally gives a command its own isolated Python environment. That is a good default for independent CLI tools, but this prototype's `run` command uses its own interpreter and therefore must be installed in the same environment as PyTorch. A separate `pipx` environment cannot launch the supported PyTorch workload yet.

The inventory and profile commands work without an NVIDIA GPU. The adapter has been tested only on Linux x86_64 with Python 3.10.12, official `torch 2.14.0+cu130`, CUDA Runtime 13.0.96, LLVM/Clang 23.1.4, CUDA 12.4 development headers, the NVIDIA 580.178.04 driver, and GTX TITAN X GM200 cards at `sm_52`. Other combinations are not yet qualified.

## 2. Check the machine and available targets

```sh
cuda-legacy inspect
cuda-legacy profiles list
cuda-legacy profiles show pytorch-2.14-cu130
cuda-legacy doctor --llvm-root /path/to/LLVM-23.1.4
cuda-legacy audit pytorch-2.14-cu130
```

- `inspect` prints the GPU and `nvcc` inventory it can discover.
- `profiles list` shows the packaged target profiles. `profiles show` prints a profile's source, CUDA package version, and minimum GPU capability.
- `doctor` checks the launcher prerequisites: Linux x86_64, Python, a CUDA-enabled PyTorch build, visible `sm_52` GPU(s), LLVM 23.1.4, CUDA headers, `g++`, and the CUDA Driver API library. It honors numeric `CUDA_VISIBLE_DEVICES` selections so a mixed-GPU host can check the same devices intended for `run`. UUID selectors cannot yet be mapped to the `nvidia-smi` inventory and are reported as unverified. Pass `--cuda-root` if the headers are not under `/usr/local/cuda-12.4`; the equivalent environment variables are `CUDA_COMPAT_LLVM_ROOT` and `CUDA_COMPAT_CUDA_ROOT`.
- `audit` compares a machine inventory with the profile's published package requirements. `compatible` only means the checked profile fields pass; it does not mean the package's operators or models have been tested on that machine.

The JSON forms are available as `inspect --json`, `profiles list --json`, `profiles show <profile> --json`, and `doctor --json`. `inspect` always returns JSON. `audit` always returns JSON.

`doctor` uses exit 0 when every prerequisite check is ready and 1 otherwise. Its `ready` status is not a workload certification. `audit` uses 0 for compatible, 1 for incompatible, 2 for an unreadable/invalid profile, and 3 when a required fact is unknown. `profiles` uses 0 on success and 2 when a profile cannot be found or read.

## 3. Run a Python application

Set paths to the user-installed compiler and CUDA 12.4 development files, then pass the application and its arguments after `--`:

```sh
cuda-legacy run \
  --llvm-root /path/to/LLVM-23.1.4 \
  --cuda-root /usr/local/cuda-12.4 \
  -- path/to/app.py --application-option value
```

The command compiles selected kernels for the real `sm_52` target, builds a driver bridge and a CUDA Runtime version-query facade in a temporary directory, then starts the script as a child Python process. Temporary build files are removed when it exits. The facade changes the runtime-version response in that child process; the GPU name, compute capability, and driver remain genuine. The parent shell, other applications, installed PyTorch wheel, NVIDIA driver, and CUDA toolkit are not modified.

The launcher inherits the parent environment and adds `LD_PRELOAD` and Python startup configuration for the child. Use it only with scripts you trust. The current adapter also requires all visible GPUs to report compute capability 5.2.

`run` returns the application process's exit code. A launcher/preflight error returns 2. This is still a per-process Python entry point; native executables, statically linked CUDA programs, other GPU architectures, and arbitrary libraries are outside the verified scope.

## What the current adapter covers

The adapter has bounded native Maxwell paths for selected addition, arange, embedding, layer normalization, matrix multiplication, and ReLU operations. It also has inference-only CPU fallbacks for selected operators and for recognized missing-image or architecture errors. CPU fallback is limited to one CUDA device and 64 MiB of input and output per operation; mutating and alias-returning operations fail closed. CPU fallback preserves selected operation results by moving data through host memory, so performance can be much lower and memory transfers are part of the cost.

Current end-to-end evidence is narrow:

| Workload | Tested case | GPU execution and fallback |
|---|---|---|
| Owner-supplied transformer checkpoint (experimental) | Six bounded batch/sequence configurations | Broker-selected GPU 2 (`sm_52`); selected operators use native kernels and CPU fallbacks; model files are not included |
| Convolution/pooling classifier | Input `2×3×32×32`; output `(2, 5)` | Broker-selected GPU 2; convolution uses CPU fallback |
| Synthetic two-layer MLP / operator probes | Bounded float32 cases and explicit edge/unsupported checks | All three cards for the original operator slice; not equivalent to full model validation on each card |

All listed model outputs matched CPU references at the tolerances recorded in the [PyTorch CUDA 13 research log](research/pytorch-cu130-maxwell.md). The model workloads have only been validated on GPU 2. Training, autograd, arbitrary models, broad cuDNN/cuBLAS behavior, large fallback transfers, and performance parity are unverified.

## Troubleshooting

| Message or result | Meaning | Next step |
|---|---|---|
| `GPU inventory is not_found` | `nvidia-smi` is unavailable to this environment. | Check the NVIDIA driver installation and whether `nvidia-smi` runs. This package does not install drivers. |
| `PyTorch is not installed` | The Python environment used by the CLI does not contain PyTorch. | Activate the environment that contains the CUDA-enabled PyTorch wheel, then install this package there too. |
| `LLVM clang++ is unavailable` or wrong version | The supplied LLVM root does not contain the tested compiler. | Pass the directory containing `bin/clang++` from LLVM 23.1.4. |
| CUDA headers missing | The given CUDA root does not contain `include/cuda.h` and `include/cuda_runtime.h`. | Install the user-managed development headers and pass their root with `--cuda-root`. |
| GPU architecture unsupported | The current adapter accepts only visible `sm_52` devices. | On a mixed-GPU host, set numeric `CUDA_VISIBLE_DEVICES` indices for only the supported cards before both `doctor` and `run`; the tool does not spoof compute capability. |
| `audit` returns `unknown` | The machine could not supply one or more facts required by the package profile. | Read the JSON `issues` and resolve the missing inventory; do not treat unknown as compatible. |
| PyTorch reports no kernel image / architecture mismatch | The wheel lacks an image for that operation, and the adapter may not implement a compatible path. | Check whether the operation is covered by the evidence; otherwise it remains unsupported. |
| CPU fallback exceeds 64 MiB or needs autograd | The fallback safety/behavior boundary was reached. | Reduce the operation's input size or use a supported native path; training/autograd is not implemented. |

Use `cuda-legacy help`, `cuda-legacy help <command>`, or `<command> --help` for command syntax. Report bugs with the command, environment versions, exact failing operation, and a small reproducer; do not include model weights, secrets, or private input data.
