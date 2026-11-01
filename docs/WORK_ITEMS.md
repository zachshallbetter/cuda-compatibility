# Local Work Items

The project is bound to GitHub repository `zachshallbetter/cuda-compatibility` and board #44. This file retains the detailed work contracts; the bound board tracks active coordination. Neither record grants authority for protected remote effects, which require a fresh ACP decision.

## CLC-001 — Establish the public project and first executable compatibility probe

- **Intent:** `PROJECT_INTENT.md`
- **Status:** Done
- **Objective:** Create the standalone CUDA Compatibility project, preserve the verified GM200 baseline, define evidence-backed application profiles, and implement a read-only host inspection/audit CLI.
- **Scope:** `README.md`, `PROJECT_INTENT.md`, `PROJECT_PROFILE.json`, `FORMAL_RESOURCE_MANIFEST.json`, `bindings/PROJECT_BINDINGS.yaml`, `docs/**`, `experiments/maxwell/**`, `profiles/**`, `src/**`, `tests/**`, `pyproject.toml`, `Makefile`, `.gitignore`, `AGENTS.md`, and `LICENSE-*`.
- **Exclusions:** Driver/toolkit installation or mutation; GPU identity spoofing; exporting fake compute capability; CUDA binary translation; distributing NVIDIA files; remote GitHub/ACP operations; performance claims.
- **Dependencies:** Existing verified CUDA 12.4 `sm_52` vector-add run on all three TITAN X cards; source-backed CUDA 12.9/13 and framework requirements.
- **Acceptance:** Bootstrap and project context validators pass; the Python CLI emits deterministic machine-readable inspection/audit results and gives nonzero exit for unmet profile requirements; unit checks cover success/failure paths; the native CUDA baseline is reproducible from the project; documents distinguish version gates from hardware limits and retain exact experiment evidence.
- **Evidence required:** Actual validator and unit-check output; native baseline output from each GPU; source links and versions for each compatibility profile; `git status` and diff review.
- **Evaluator:** Project owner reviews claims; code checks validate deterministic behavior.
- **Permissions:** E0 reads; E1 edits within scope; local test processes and temporary files only. No host configuration change, network service mutation, commit, push, or release.
- **Target:** `project root`.
- **Budget:** No spend; one bounded implementation pass, stop if an external install or protected effect is required.
- **Stop conditions:** Required behavior depends on undocumented semantics, a proprietary file must be redistributed, or a host/driver/toolchain mutation is required.

The first work item is being executed under the user's explicit request to create and build out this dedicated local project. After the local frontier is connected to an authorized provider, future work will be tracked there.

**Disposition:** Done for the stated first slice. Evidence is recorded in `records/evidence.jsonl`. CUDA 13 compilation/lowering, runtime emulation, application execution, and performance measurements are later work items; this completion does not claim those capabilities.

## CLC-002 — Establish PyTorch Maxwell comparison and LLVM toolchain gate

- **Intent:** `PROJECT_INTENT.md`
- **Status:** Done
- **Objective:** Make the PyTorch CUDA 13 target measurable against the official Maxwell-supported CUDA 12.6.3 build and record whether the current development/target environments can execute the planned LLVM NVPTX experiment.
- **Scope:** `profiles/pytorch-2.14-cu126.json`, `profiles/pytorch-2.14-cu130.json`, `README.md`, `docs/compatibility-targets.md`, `docs/roadmap.md`, and `records/evidence.jsonl`.
- **Exclusions:** Installing compilers or toolkits; changing target host; claiming that the full PyTorch package is operational on Maxwell; runtime interception, API emulation, CUDA 13 binary compatibility, performance claims, remote repository/release actions.
- **Dependencies:** Official PyTorch 2.14 release matrix; current local compiler and reference-host tool inventory.
- **Acceptance:** Both PyTorch profiles distinguish the published CUDA build and architecture gate; the packaged profiles resolve by ID; project checks pass; compiler availability is recorded accurately; roadmap states the evidence needed before a user-space CUDA 13 profile can be claimed.
- **Evidence required:** Source-linked profile facts, profile audit output or equivalent unit evidence, local and reference-host compiler inventory, context/project validators.
- **Evaluator:** Project owner reviews published facts; deterministic CLI and project validators check the implementation.
- **Permissions:** E0 reads; E1 local edits and tests. No install, system configuration change, commit, push, or release.
- **Target:** `project root`.
- **Budget:** No spend; one local bounded pass.
- **Stop conditions:** The matrix does not source the stated architecture coverage or the profile schema cannot represent it without changing a canonical contract.

**Disposition:** Done. The initial tool inventory found no LLVM NVPTX compiler in either environment. CLC-004 later provisioned a verified upstream LLVM build in `/tmp` and passed the first end-to-end vector-add compiler/assembler/driver experiment; see `docs/research/llvm-nvptx.md`.

## CLC-003 — Expand feature-restoration goal and harden packaged profile lookup

- **Intent:** `PROJECT_INTENT.md`; owner clarification recorded in `records/deviations.jsonl`.
- **Status:** Done
- **Objective:** Make software emulation an explicit first-class route for restoring newer CUDA feature semantics and support profile discovery from both standard and `pip --target` installations.
- **Scope:** `PROJECT_INTENT.md`, `docs/architecture.md`, `src/cuda_legacy_compat/cli.py`, `tests/test_cli.py`, `records/deviations.jsonl`, and this work item.
- **Exclusions:** Claims of universal compatibility; compute-capability spoofing; unimplemented runtime/compiler emulation; driver/toolkit installation; external publication.
- **Dependencies:** Active owner instruction; Python standard library; local isolated package install.
- **Acceptance:** Project intent says deprecation alone does not exclude a feature and preserves evidence requirements; installed profile resolution works for standard venv and `pip --target` layouts; project checks and tests pass.
- **Evidence required:** Unit checks for both installation layouts; project context/integrity checks; installed-package profile resolution.
- **Evaluator:** Project owner reviews scope; deterministic tests and validators check behavior.
- **Permissions:** E1 local edits, package builds, and isolated temporary installation only.
- **Target:** `project root`.
- **Budget:** No spend; one local pass.
- **Stop conditions:** Required behavior needs driver/system changes or changes to a canonical profile schema.

**Disposition:** Done. The project goal now treats software emulation as a first-class compatibility route, bounded by correctness evidence and transparent resource costs. The packaged CLI resolves both profile IDs after a normal venv install and a `pip --target` install.

## CLC-004 — Run an isolated LLVM NVPTX Maxwell compile experiment

- **Intent:** `PROJECT_INTENT.md`; compiler-independence milestone M1 in `docs/roadmap.md`.
- **Status:** Done
- **Objective:** Test whether a pinned upstream LLVM release can compile the existing vector-add CUDA source to Maxwell PTX, whether CUDA 12.4 `ptxas` accepts it, and whether the result executes correctly on each GM200 card.
- **Scope:** Temporary verified LLVM 23.1.4 toolchain under `/tmp` on `reference GPU host`; `experiments/llvm-nvptx/**`; `docs/research/llvm-nvptx.md`; `docs/roadmap.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`.
- **Exclusions:** System package install; changes to NVIDIA driver, kernel, or system CUDA; proprietary file redistribution; replacing NVCC globally; CUDA 13 facade/API claims; performance parity claims; remote repository or release operations.
- **Dependencies:** Upstream LLVM 23.1.4 Linux x86_64 release asset and its verification bundle; target host existing CUDA 12.4.131 headers, `ptxas`, and R580 driver; existing Maxwell baseline.
- **Acceptance:** Verify the upstream toolchain artifact; confirm its NVPTX target; compile source for `sm_52`; capture PTX version/target; assemble with CUDA 12.4 `ptxas`; run the kernel and compare all outputs on each GPU; clean temporary downloaded/extracted artifacts.
- **Evidence required:** Artifact verification result, exact tool versions and commands, compiler output, assembler result, per-device correctness results, negative results, cleanup confirmation.
- **Evaluator:** Project owner reviews evidence; numerical comparison and existing per-card execution harness evaluate correctness.
- **Permissions:** E0 reads; E1 network download of the named open-source LLVM release into `/tmp`, bounded execution in `/tmp`, and temporary `/tmp` source/result files on the reference host only.
- **Target:** `project root` and the existing reference host `reference GPU host`.
- **Budget:** No spend; stop if artifact verification fails, available temporary storage is insufficient, or the compiler requires changes to host configuration.
- **Stop conditions:** Unverified artifact, incompatible host runtime, unsupported `sm_52`, unsupported generated PTX in the available assembler, or any need to change host driver/system CUDA.

**Disposition:** Done. LLVM 23.1.4 emitted PTX 8.4 for `sm_52`; CUDA 12.4 `ptxas` assembled it; both PTX JIT and cubin passed all-output checks on each of the three GM200 cards. Temporary compiler archives, extraction, source copies, and binaries were removed from the development and reference hosts.

## CLC-005 — Prototype opt-in CUDA 13 runtime-version spoof

- **Intent:** `PROJECT_INTENT.md`; version/API facade milestone M3 in `docs/roadmap.md`.
- **Status:** Done
- **Objective:** Demonstrate an application-visible CUDA 13.0 runtime-version response while forwarding ordinary CUDA 12.4 runtime and driver behavior to the installed NVIDIA stack, with truthful physical device capability.
- **Scope:** `src/native/**`, `experiments/runtime-facade/**`, `docs/research/runtime-version-facade.md`, `docs/roadmap.md`, `docs/WORK_ITEMS.md`, `README.md`, and `records/evidence.jsonl`.
- **Exclusions:** Spoofing GPU model or compute capability; claiming CUDA 13 APIs or libraries are implemented; intercepting arbitrary applications; system-wide `LD_PRELOAD`; driver/toolkit changes; public release artifacts containing NVIDIA files.
- **Dependencies:** Reference host CUDA 12.4.131 headers/runtime/compiler and R580 driver.
- **Acceptance:** An opt-in process-local preload makes a dynamically linked probe observe runtime version 13000, leaves driver version and `sm_52` properties truthful, and successfully performs a CUDA allocation, launch, synchronization, and copy using the real installed runtime. Without the opt-in environment variable, the probe reports the actual runtime version; unsupported virtual versions fail closed.
- **Evidence required:** Built shim and probe, baseline and opt-in output, per-card actual device identity/capability, kernel correctness output, no system changes.
- **Evaluator:** Project owner reviews the boundary; probe asserts expected version and hardware values and checks arithmetic output.
- **Permissions:** E1 source edits and temporary builds/runs in `/tmp` on the reference host; no persistent system/library path changes.
- **Target:** `project root` and reference host `reference GPU host`.
- **Budget:** No spend; one bounded prototype pass.
- **Stop conditions:** Runtime ABI cannot be safely interposed without modifying installed files, or the spoof changes physical device reporting or causes incorrect results.

**Disposition:** Done for the one version-query prototype. Dynamically linked probes reported runtime 13000 while retaining driver 13000 and physical `sm_52`; preloaded CUDA work passed on each card and unsupported override values failed closed. The reference host's default NVCC output has no dynamic `libcudart` dependency, so `LD_PRELOAD` does not affect statically linked applications. No broader CUDA 13 APIs or libraries are claimed.

## CLC-007 — Restore PyTorch tensor addition on Maxwell

- **Intent:** `PROJECT_INTENT.md`; first application workload target is PyTorch CUDA 13.
- **Status:** Done
- **Objective:** Make bounded `torch.add(Tensor, Tensor)` cases work on the official CUDA 13 PyTorch wheel on GM200 by routing contiguous int32 addition through an LLVM-generated `sm_52` kernel, using a PyTorch CPU fallback for contiguous int64 addition, and failing closed for other cases.
- **Scope:** Temporary verified LLVM 23.1.4 toolchain under `/tmp` on `reference GPU host`; `experiments/pytorch-cu130/**`; `src/native/**`; `docs/research/pytorch-cu130-maxwell.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; `records/negative-results.jsonl`.
- **Exclusions:** Modifying the official wheel; spoofing compute capability; changing PyTorch operators other than `aten::add.Tensor`; claiming CUDA 13 feature or library completeness; performance parity; changing system CUDA or driver.
- **Dependencies:** Verified official LLVM 23.1.4 Linux release; CLC-006 official PyTorch 2.14.0+cu130 wheel in `/tmp/cuda-compat-pytorch-cu130-venv`; target R580 host and physical `sm_52` devices.
- **Acceptance:** An explicitly installed process-local adapter runs contiguous int32 `torch.add` via PTX generated for `sm_52`; contiguous int64 addition uses CPU fallback and returns a CUDA tensor; unsupported dtype/layout/alpha fails explicitly; numerical equivalence passes on all three GPUs; native device identity remains 5.2; all temporary toolchain/package files are removed afterward.
- **Evidence required:** LLVM attestation/checksum; exact versions and commands; operator cases and results; per-device outputs; unsupported scope; cleanup confirmation; project checks.
- **Evaluator:** Project owner reviews the adapter boundary; PyTorch reference results and equality checks validate output.
- **Permissions:** E1 network download and verification of the named open-source LLVM archive, local code edits, and temporary builds/runs under `/tmp`; no system configuration, commit, push, or release.
- **Target:** `project root` and reference host `reference GPU host`.
- **Budget:** No spend; bounded to one operator and one fallback path.
- **Stop conditions:** PyTorch dispatch cannot be overridden process-locally without modifying its installed wheel, or numerical behavior cannot be preserved for the declared paths.

**Disposition:** Done for the declared operator paths. The opt-in process-local adapter passed contiguous int32 GPU addition through LLVM-generated `sm_52` PTX, int32 overflow checks, and contiguous int64 CPU fallback checks on all three physical GPUs, both with the CUDA 13 runtime as shipped and with the version facade enabled. Float32 fails explicitly. Device identity remains CC 5.2. This does not enable PyTorch's built-in CUDA 13 kernels generally, other operators or libraries, autograd, or performance parity; no performance was measured. Temporary package/toolchain artifacts were removed from the reference host.

## CLC-006 — Run the official PyTorch CUDA 13 wheel on Maxwell

- **Intent:** `PROJECT_INTENT.md`; PyTorch CUDA 13 target in `docs/compatibility-targets.md`.
- **Status:** Done
- **Objective:** Install the official PyTorch 2.14.0 CUDA 13.0 wheel into an isolated Python 3.10 environment on `reference GPU host`, observe its unmodified behavior on the three GM200 GPUs, then test the same package with the process-local runtime-version facade and a minimal CUDA tensor operation.
- **Scope:** Temporary Python environment and package files under `/tmp` on `reference GPU host`; `experiments/pytorch-cu130/**`; `docs/research/pytorch-cu130-maxwell.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; `records/negative-results.jsonl`.
- **Exclusions:** Changes to system Python/packages, NVIDIA driver, kernel, or CUDA installation; GPU model/compute-capability spoofing; modifying PyTorch wheel contents; claiming success from package import or version query alone; public redistribution of PyTorch or NVIDIA files.
- **Dependencies:** Official PyTorch 2.14.0 CUDA 13.0 wheel/index; Python 3.10.12; R580 driver and existing three GM200 GPUs; CLC-005 process-local runtime-version facade.
- **Acceptance:** Record install result, `torch.__version__`, runtime/driver versions, device visibility and real capabilities, then execute a small device tensor operation without and with the facade; capture exact failure boundary if either run fails; remove the temporary environment and packages.
- **Evidence required:** Exact official install command, wheel/package versions, stdout/stderr and exit status for import/device/operation probes, physical GPU inventory, cleanup confirmation, project checks.
- **Evaluator:** Project owner reviews the package-level result; tensor result comparison and recorded device metadata validate execution.
- **Permissions:** E0 reads; E1 network package download and temporary isolated environment/processes under `/tmp` only. No system configuration, commit, push, or release.
- **Target:** `project root` and reference host `reference GPU host`.
- **Budget:** No spend; use available storage only; stop if the package requires system-level modifications or the available `/tmp` space is insufficient.
- **Stop conditions:** Official wheel unavailable for Python 3.10/Linux x86_64, installation requires system mutation, or the temporary environment cannot be isolated.

**Disposition:** Done. The official `torch 2.14.0+cu130` wheel imported, discovered each physical GM200 at its truthful CC 5.2, and passed host-to-device copies. Built-in tensor addition failed with “no kernel image is available” on each GPU, with and without the version facade. The scoped CLC-007 adapter then restored only its declared operator cases. All temporary wheel, LLVM, and probe artifacts were removed from the reference host.

## CLC-008 — Preserve asynchronous stream semantics in the PyTorch Maxwell adapter

- **Intent:** `PROJECT_INTENT.md`; PyTorch CUDA 13 adapter in milestone M4 of `docs/roadmap.md`.
- **Parent:** CLC-007.
- **Status:** Done
- **Objective:** Remove per-call PTX module load/unload and forced stream synchronization from the int32 `aten::add.Tensor` bridge while preserving correct execution on PyTorch's current CUDA stream.
- **Scope:** `experiments/pytorch-cu130/driver_add_bridge.cpp`; `experiments/pytorch-cu130/aten_compat.py`; `experiments/pytorch-cu130/operator_probe.py`; `experiments/pytorch-cu130/run_operator_adapter.sh`; `docs/research/pytorch-cu130-maxwell.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; `records/negative-results.jsonl`.
- **Exclusions:** Adding operator/dtype coverage; packaging a public Python distribution; changing the CUDA runtime-version facade; modifying the PyTorch wheel, NVIDIA driver, kernel, or system CUDA; claiming broad CUDA 13 compatibility; commit, push, or release.
- **Dependencies:** Verified LLVM 23.1.4 artifact identity from CLC-004; official PyTorch 2.14.0+cu130 probe environment from CLC-006; reference host `resource` broker and three GM200 `sm_52` devices.
- **Acceptance:** Cache the one PTX module/function per live CUDA context for the process lifetime; reject a different PTX path for an already cached context; launch on PyTorch's current stream without synchronizing or unloading after each call; reject element counts that exceed the kernel ABI's signed 32-bit limit; pass repeated and non-default-stream int32 correctness checks on all three physical cards; preserve the int64 fallback and explicit unsupported-case behavior; keep physical device reporting truthful.
- **Evidence required:** Exact target-host commands and versions; module cache and stream behavior evidence; per-card numeric results and unsupported cases; negative paths; temporary artifact cleanup; project checks.
- **Evaluator:** Project owner reviews the small bridge boundary; the operator probe compares output against CPU PyTorch and checks dispatch-path assertions.
- **Permissions:** E1 local source edits; verified LLVM download if absent; official PyTorch wheel installation and bounded builds/runs in `/tmp` on `reference GPU host`; cleanup of task-created temporary paths only. No system configuration, commit, push, or release.
- **Effect class:** E1.
- **Target:** `project root` and `reference GPU host` through the documented host resource broker.
- **Budget:** No spend; one bounded operator-hardening pass; run sequentially on selected GPUs to limit resource use.
- **Stop conditions:** The current-stream launch cannot be shown correct without an explicit synchronization; safe context-scoped module caching requires unsupported driver assumptions; any test indicates changed numerical results or device identity.

**Disposition:** Done for the declared int32 adapter path. The bridge loads one PTX module/function per CUDA context, rejects a second PTX path for that context, has no per-call stream synchronization or module unload, preserves non-default-stream result ordering, rejects counts above the signed 32-bit kernel limit, and handles empty tensors without a launch. The official PyTorch 2.14.0+cu130 wheel passed the extended operator probe on physical GPUs 0, 1, and 2 in both runtime modes. The int64 CPU fallback, overflow behavior, and explicit float32 rejection remain intact. Module-cache lifetime is process/context lifetime; no performance benchmark or broader operator coverage is claimed. Temporary host artifacts were removed.

## CLC-009 — Run a float32 PyTorch inference slice through Maxwell kernels

- **Intent:** `PROJECT_INTENT.md`; PyTorch CUDA 13 framework adapter in M4 of `docs/roadmap.md`.
- **Parent:** CLC-007; follow-on hardening CLC-008.
- **Status:** Done
- **Objective:** Extend the opt-in `aten` dispatcher adapter with bounded contiguous float32 matrix multiplication and ReLU kernels for `sm_52`, then run a deterministic two-layer MLP inference slice on the official PyTorch CUDA 13 wheel.
- **Scope:** `experiments/pytorch-cu130/compat_kernels.cu`; `experiments/pytorch-cu130/driver_add_bridge.cpp`; `experiments/pytorch-cu130/aten_compat.py`; `experiments/pytorch-cu130/operator_probe.py`; `experiments/pytorch-cu130/run_operator_adapter.sh`; new stock-wheel baseline probe under `experiments/pytorch-cu130/`; `README.md`; `docs/roadmap.md`; `docs/research/pytorch-cu130-maxwell.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; `records/negative-results.jsonl`.
- **Exclusions:** Arbitrary model support; convolution, attention, optimizer, autograd, half precision, non-contiguous matrices, large-matrix performance; expanding runtime API interception; modifying the official wheel, NVIDIA driver, kernel, or system CUDA; commit, push, or release.
- **Dependencies:** CLC-008 cached asynchronous bridge; CLC-006 official `torch 2.14.0+cu130` environment and reference host host; CLC-004 attestation-verified LLVM 23.1.4 artifact identity.
- **Acceptance:** A stock-wheel probe records the exact `sm_52` outcome for built-in `mm` and `relu`; opt-in replacements compile into one `sm_52` PTX module and pass direct operator comparisons against CPU PyTorch; a deterministic `mm → relu → mm` inference slice matches its CPU reference within documented float32 tolerances; non-default-stream execution passes; zero/empty inner/output dimensions have defined correct behavior; unsupported layouts and autograd fail explicitly; all three physical GPUs pass both ordinary and facade runtime modes with real CC 5.2 reporting; task-created remote files are cleaned.
- **Evidence required:** Exact versions and reference host commands; stock failure stage per operator; PTX target; shapes/dtypes/tolerances and CPU-reference results; per-card and per-runtime-mode output; explicit unsupported paths; cleanup; project checks.
- **Evaluator:** Project owner reviews the bounded operator contract; probe asserts output equivalence, dispatch path, and unsupported behavior.
- **Permissions:** E1 code edits and tests; download and verify the previously pinned open-source LLVM release; install the official PyTorch wheel in an isolated `/tmp` venv; bounded GPU execution and cleanup through the host resource broker; no system configuration or protected external effects.
- **Effect class:** E1.
- **Target:** `project root` and `reference GPU host` through the documented host resource broker.
- **Budget:** No spend; one bounded two-layer float32 inference slice; run cards sequentially.
- **Stop conditions:** LLVM cannot emit supported `sm_52` PTX for the selected kernels; output diverges beyond stated tolerance; driver launch/context behavior cannot preserve PyTorch stream semantics; dispatcher registration cannot reach the selected operator; host resources fall below available requirements.

**Disposition:** Done for the declared operator slice. The stock wheel reports `CUBLAS_STATUS_ARCH_MISMATCH` for `mm` and a missing kernel image for `relu` on all three `sm_52` cards. The combined LLVM PTX module passed int32 addition, bounded float32 `mm` and `relu`, edge/empty cases, fail-closed layout/autograd checks, and a deterministic two-layer MLP CPU-reference comparison on each card in both runtime modes. The MLP used a non-default stream and `rtol=1e-5, atol=1e-6`. Physical CC remained 5.2. This does not establish support for real packaged models, larger matrices, or performance. Temporary files were cleaned.

## CLC-010 — Package and launch the opt-in PyTorch compatibility layer

- **Intent:** `PROJECT_INTENT.md`; public CUDA compatibility layer goal and PyTorch CUDA 13 target.
- **Parent:** CLC-007; depends on CLC-008 and CLC-009.
- **Status:** Done
- **Objective:** Turn the tested PyTorch operator adapter into an installable, opt-in `cuda-legacy-compat run` path that builds its `sm_52` PTX and native bridge in a temporary directory, initializes dispatch before the user's Python program, and enables the tested runtime-version facade without modifying the PyTorch wheel or system CUDA installation.
- **Scope:** `src/cuda_legacy_compat/**`; package resource files for PyTorch adapter, kernel source, bridge, and startup bootstrap; `pyproject.toml`; `tests/**`; `experiments/pytorch-cu130/**` only for developer-runner integration; `README.md`; `docs/architecture.md`; `docs/roadmap.md`; `docs/research/pytorch-cu130-maxwell.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; `records/negative-results.jsonl`.
- **Exclusions:** Wider operator support; system-wide `LD_PRELOAD`; modifying PyTorch/NVIDIA files; downloading or installing toolchains into persistent system paths; auto-detecting and masking unknown hardware; arbitrary static binaries; commit, push, or public release.
- **Dependencies:** CLC-009 verified `aten::add.Tensor`, `aten::mm`, and `aten::relu` implementations; package profile/audit CLI; official PyTorch 2.14.0+cu130 wheel; reference host `resource` broker; CUDA 12.4 headers/driver library and verified LLVM 23.1.4.
- **Acceptance:** A built wheel contains the adapter and compile resources without bundling PyTorch/NVIDIA binaries; `cuda-legacy-compat run -- <python-script> [args...]` validates the target and required toolchain, compiles PTX/bridge/facade into a bounded temporary directory, initializes the opt-in dispatcher before the script, sets process-local runtime facade variables, forwards arguments/stdio/exit status, and removes generated artifacts; CPU-only import and CLI checks pass without importing PyTorch; all existing operator and MLP GPU assertions pass through the installed launcher on physical GPUs 0, 1, and 2; unsupported operations still fail explicitly and actual CC remains 5.2.
- **Evidence required:** Wheel contents and install command; exact toolchain/target versions; launcher invocation/result on each card; no system or wheel mutation; cleanup; unit/project checks; limitations and negative outcomes.
- **Evaluator:** Project owner reviews the user-facing launcher and opt-in boundary; isolated package install, CLI tests, and reference host integration probe validate behavior.
- **Permissions:** E1 local source/package edits, wheel build and isolated install, official LLVM/PyTorch downloads under `/tmp`, and bounded reference host runs/cleanup via the host resource broker. No system configuration, commit, push, or release.
- **Effect class:** E1.
- **Target:** `project root` and `reference GPU host` through the host resource broker.
- **Budget:** No spend; one installable launcher for the verified operator set; test GPUs sequentially.
- **Stop conditions:** The wheel cannot load resources portably; bootstrap imports or changes behavior without explicit opt-in; subprocess arguments/exit status are not preserved; hardware identity changes; any required persistent system mutation appears necessary.


**Disposition:** Done for the packaged launcher and the bounded CLC-009 operator subset. The pure-Python wheel contains the adapter, PTX kernel source, driver bridge, runtime facade, and opt-in startup hook; it contains no PyTorch or NVIDIA binaries. An isolated install exposed the CLI without importing PyTorch. The installed wheel's `run` path compiled `sm_52` PTX and native libraries in a temporary directory, returned runtime version 13000 while reporting the actual TITAN X at CC 5.2, and passed the full operator/MLP probe on physical GPUs 0, 1, and 2. A missing local CUDA header during the first build and the probe's alternate-PTX fixture requirement were caught and fixed before the successful runs. The initial reference host environment had no PyTorch venv or LLVM extraction, so both pinned dependencies were recreated under `/tmp`; system CUDA, driver, and wheel contents were not modified. Unsupported operators and dtypes remain outside this adapter. No public release, commit, or push was made.

## CLC-011 — Evaluate one owner-supplied inference checkpoint

- **Intent:** `PROJECT_INTENT.md`; practical PyTorch CUDA 13 use on older NVIDIA GPUs.
- **Parent:** CLC-010 packaged launcher.
- **Status:** In progress
- **Objective:** Evaluate one owner-supplied transformer checkpoint through the CUDA 13 wheel and `cuda-legacy-compat run` on `sm_52`, adding only the operator or library coverage the workload requires. Compare returned tensors with a CPU reference across bounded batch and sequence shapes, retain honest device reporting, and measure peak GPU memory and latency/throughput.
- **Scope:** The adapter and launcher under `src/cuda_legacy_compat/**`; the local-only workload probe (excluded from the public source snapshot); integration checks under `tests/**`; this work item, `docs/roadmap.md`, `docs/research/pytorch-cu130-maxwell.md`, and `records/evidence.jsonl`.
- **Exclusions:** Publishing the checkpoint, model source, tokenizer, project paths, configuration identifiers, or artifact hashes; modifying the external model or its training data; training or fine-tuning; claiming general PyTorch support from one workload; changing the NVIDIA driver, system CUDA installation, or official wheel.
- **Dependencies:** CLC-010 launcher; official PyTorch 2.14.0+cu130 wheel; reference GPU host; verified LLVM 23.1.4 and installed CUDA 12.4 driver headers.
- **Acceptance:** Capture the unmodified wheel's first failing operator/library boundary; run the checkpoint in evaluation/inference mode through the packaged launcher; compare returned tensors with CPU output using documented tolerances; verify truthful device reporting; reject unsupported cases clearly; report workload-level memory and latency methodology; clean task-created temporary dependencies and build files.
- **Evidence required:** Package/toolchain/driver versions and commands; first stock-wheel failure; tested operator/library coverage; output comparisons and unsupported paths; workload metrics; cleanup and project gates. Private input identities and digests stay out of the public record.
- **Evaluator:** CPU PyTorch comparison, evidence-backed workload checks, and project gates.
- **Permissions:** E1 read-only access to owner-supplied inputs on the reference GPU host; temporary Python 3.10 environment, official PyTorch wheel, and verified LLVM under `/tmp`; local adapter edits and bounded GPU tests; cleanup of task-created paths only. No external source/model edits, system configuration, spend, or protected repository effects.
- **Effect class:** E1.
- **Target:** Project root and reference GPU host.
- **Budget:** No spend; one existing checkpoint, bounded batch sizes up to 8 and sequence length up to 128; process GPUs sequentially.
- **Stop conditions:** Input/config mismatch; inability to preserve model semantics within measured tolerance; requirement for an unsupported NVIDIA-only library without a verified fallback; memory/runtime limits; or dependency on external credentials.

**Progress (2026-11-01):** The official `torch 2.14.0+cu130` wheel completed this owner-supplied checkpoint through the packaged launcher on broker-selected GPU 2 (`sm_52`). All 11 outputs matched CPU at `rtol=1e-4, atol=1e-5` for batch 1 at sequences 8, 32, and 128; batch 4 at sequence 32; and batch 8 at sequences 32 and 128. The largest case measured 114.881 ms median latency, 8,914 tokens/s, and 142,024,704 bytes peak allocated GPU memory. Selected operations used native Maxwell kernels; other covered operations used a 64 MiB-capped, inference-only CPU fallback. Per-card validation, training/autograd, and broader workload coverage remain outstanding. The public repository omits the owner-supplied model probe and all identifying paths, configuration, and hashes; this one workload is experimental evidence, not a product feature or general support claim.

## CLC-012 — Validate a convolutional PyTorch CUDA 13 inference path

- **Intent:** `PROJECT_INTENT.md`; practical PyTorch CUDA 13 use on older NVIDIA GPUs.
- **Parent:** CLC-010 packaged launcher; related workload validation CLC-011.
- **Status:** Verified
- **Objective:** Determine whether a small convolutional inference graph can run through the official CUDA 13 wheel on `sm_52`, and add only the narrowly evidenced fallback needed to preserve the graph's output.
- **Scope:** `src/cuda_legacy_compat/torch_adapter.py`; `experiments/pytorch-cu130/cnn_probe.py`; `docs/WORK_ITEMS.md`; `docs/roadmap.md`; `docs/research/pytorch-cu130-maxwell.md`; `README.md`; `records/evidence.jsonl`.
- **Exclusions:** Claiming general cuDNN/convolution compatibility; training, autograd, performance parity, or extra shapes; changing the PyTorch wheel, NVIDIA driver, system CUDA, or reference host workloads; commit, push, or release.
- **Dependencies:** CLC-010 launcher; official PyTorch 2.14.0+cu130 isolated environment; reference host `resource` broker; verified LLVM 23.1.4 and CUDA 12.4 driver headers.
- **Acceptance:** Reproduce the convolutional stock-path failure; add a precisely matched inference-only fallback signal while preserving fail-closed mutation and alias rules; run the packaged launcher via the resource broker and compare final output to CPU at documented tolerance; verify true device capability remains 5.2.
- **Evidence required:** Probe architecture and input shape; exact failure marker; wheel/toolchain/driver versions; broker-selected GPU; CPU comparison result; fallback boundary and limits; project checks.
- **Evaluator:** CPU-reference comparison in `cnn_probe.py`, truthful device output, and project checks.
- **Permissions:** E1 local adapter/probe/document edits; rebuild and install the project wheel only into the existing isolated `/tmp`-scoped reference host target; one bounded host resource-broker GPU run. No system changes or external protected effects.
- **Effect class:** E1.
- **Target:** `project root` and `reference GPU host` through the host resource broker.
- **Budget:** No spend; one deterministic 2×3×32×32 input and one classifier forward pass.
- **Stop conditions:** CPU/GPU output mismatch beyond tolerance; fallback would require relaxing mutation, alias, device, inference, or 64 MiB constraints.

**Disposition (2026-11-01):** The unmodified convolution path failed on broker-selected GPU 2 with `GET was unable to find an engine to execute this computation`. The classifier now recognizes that exact cuDNN no-engine phrase, and the packaged launcher completed `Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Linear` for input `2×3×32×32`; output `(2, 5)` matched CPU at `rtol=1e-4, atol=1e-5`, with true capability reported as 5.2. The convolution work uses CPU fallback and is not evidence of Maxwell convolution kernels or broad cuDNN support. `make check` and `make test` passed; task-created CNN transfer and wheel files were removed from reference host. The recorded limitation remains: one shape/model/card, inference only.

## CLC-013 — Align installation, CLI, and public compatibility claims

- **Intent:** `PROJECT_INTENT.md`; host-level CUDA 13+ compatibility end state and truthful implementation status.
- **Parent:** CLC-010 packaged launcher; public documentation and CLI quality.
- **Status:** Verified
- **Objective:** Make the package installation path, CLI behavior, tested environment, architecture description, and published research claims reproducible and consistent with the current implementation, while clearly distinguishing the process-local prototype from the desired machine-level product.
- **Scope:** `PROJECT_INTENT.md`; `README.md`; `src/cuda_legacy_compat/cli.py`; `tests/test_cli.py`; `docs/architecture.md`; `docs/roadmap.md`; `docs/research/pytorch-cu130-maxwell.md`; `docs/research/runtime-version-facade.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context lock.
- **Exclusions:** A machine-wide installer or interposer; changes to the NVIDIA driver, toolkit, or CUDA wheel; broader operator/library support; public repository creation, package publication, commit, or push.
- **Dependencies:** CLC-010 package and launcher; current PyTorch CUDA 13 evidence; local Python package build tooling.
- **Acceptance:** Document regular and development installs, same-environment requirements, tested GPU launch dependencies, CLI commands and exit semantics, truthful process-only boundaries, and current evidence; provide a CLI version flag; successfully install the built package into a clean virtual environment, resolve an installed profile, and verify unknown hardware reports `unknown`; pass repository checks and tests.
- **Evidence required:** Platform/interpreter; wheel build/install outcome; CLI version/help/inspect/audit outputs and exit status; project check/test results; GPU-specific limitations of the local host.
- **Evaluator:** Installed command behavior, source-linked profile resolution, machine-readable audit result, and documented project gates.
- **Permissions:** E1 local code, test, documentation, and evidence edits; isolated package build/install under `/tmp`; project checks. No system or external protected effects.
- **Effect class:** E1.
- **Target:** `project root`.
- **Budget:** No spend; one package build and isolated local CLI verification.
- **Stop conditions:** Installation changes system packages; profile resolution fails; audit reports compatibility with unavailable hardware; docs require unsupported compatibility claims.

**Disposition (2026-11-01):** Built and installed version 0.1.0 with `python3 -m pip install .` into a fresh virtual environment on macOS arm64 / Python 3.15.2. The installed CLI reported its version, displayed root and `run` help, emitted inventory JSON, resolved the packaged PyTorch CUDA 13 profile by ID, and returned `status=unknown` / exit 3 because this host has no NVIDIA GPU. The source-checkout profile path also resolved and returned unknown. The temporary environment was removed. GPU execution remains untested on this host; the `run` path remains documented only for its tested Linux x86_64 / Python 3.10 / `sm_52` configuration. `make check` and `make test` passed; no system installation, commit, push, or release occurred.

## CLC-014 — Improve CLI guidance and user documentation

- **Intent:** `PROJECT_INTENT.md`; host-level CUDA 13+ compatibility end state with evidence-backed support claims.
- **Parent:** CLC-013 public installation and CLI quality review.
- **Status:** Verified
- **Objective:** Make common user tasks discoverable from the CLI and provide a practical guide to installing, diagnosing, auditing, launching, and understanding the current compatibility tool.
- **Scope:** `src/cuda_legacy_compat/cli.py`; `src/cuda_legacy_compat/launch.py` only if shared read-only prerequisite checks are needed; `tests/test_cli.py`; new `docs/user-guide.md`; `README.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context lock.
- **Exclusions:** Machine-wide installation or loader injection; expanding CUDA operator/library behavior; downloading toolchains automatically; changing the NVIDIA driver, toolkit, or PyTorch wheel; public repository creation, package publication, commit, or push.
- **Dependencies:** CLC-010 packaged CLI; CLC-013 install verification; current profile and PyTorch CUDA 13 research records.
- **Acceptance:** Provide clear root and command help; list and show installed/source profiles in human and JSON forms; add read-only actionable prerequisite diagnostics; preserve truthful compatibility/audit semantics and exit codes; document user workflows, known limits, and troubleshooting; install the package in an isolated environment and exercise documented CLI paths; pass repository checks and tests.
- **Evidence required:** Installed CLI output for help/version/profile/doctor/audit; expected status/exit behavior on a host without NVIDIA GPU; project checks, unit tests, and evidence-record validation.
- **Evaluator:** Clean-environment installation, CLI command output and status assertions, user guide against current implementation, and project gates.
- **Permissions:** E1 local CLI/test/documentation/evidence edits; isolated package build/install under `/tmp`; read-only local host inspection. No system or external protected effects.
- **Effect class:** E1.
- **Target:** `project root`.
- **Budget:** No spend; package build plus bounded local CLI checks.
- **Stop conditions:** A diagnostic needs system mutation or package installation; it cannot distinguish unknown from compatible; its status would imply untested workload support.

**Disposition (2026-11-01):** The installed CLI now provides top-level and command-specific help, `help <command>`, profile list/show in human and JSON forms, and a read-only `doctor` report with next steps. Doctor checks visible numeric `CUDA_VISIBLE_DEVICES` selections, marks UUID selections unknown, and reports ready only when launcher prerequisites are present; it does not certify workloads. The user guide covers install, commands, status codes, environment requirements, verified workload boundaries, and troubleshooting. A fresh Python 3.15.2 virtual environment installed the package and passed version, help, profile list/show, inventory, doctor, and audit commands. On this GPU-less host doctor returned needs_attention/exit 1 and audit returned unknown/exit 3. `make check` passed and `make test` passed all 28 tests. The temporary environment was removed; no system installation, commit, or release occurred.

## CLC-015 — Build a public-facing GitHub Pages project site

- **Intent:** `PROJECT_INTENT.md`; explain and advance practical PyTorch CUDA 13 use on older GPUs with evidence-backed claims.
- **Parent:** CLC-013/014 public documentation and installation quality.
- **Status:** Verified
- **Objective:** Create a comprehensive, accessible, responsive GitHub Pages-ready site that explains the project, its actual state and limitations, validated workloads, installation path, and architecture-matched but unverified GPU candidates; credit Zach Shallbetter with verified personal site and GitHub profile links.
- **Scope:** `site/**`; `.github/workflows/pages.yml`; `README.md`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Creating a GitHub repository, configuring custom domains or repository settings, pushing code, publishing/deploying the site, claiming broad CUDA 13+ support, or expanding GPU/operator compatibility.
- **Dependencies:** Current README, CLI guide, research records, NVIDIA CUDA 13 release notes and legacy compute-capability listing, verified owner links.
- **Acceptance:** The site distinguishes verified, experimental, and unverified states; explains the process-local adapter and CPU fallback honestly; lists same-`sm_52` candidate cards as unverified per-card and per-workload; gives a useful current install/run path; cites primary sources; includes accessible mobile-ready navigation and identity links; deploy workflow packages only `site/` and requires a manual dispatch; static site serves successfully and local project gates pass.
- **Evidence required:** Site HTTP response and rendered preview; local link/markup checks; `make check` and `make test`; reviewed diff/status; source links for GPU family and toolkit claims; record of publication constraints.
- **Evaluator:** Project owner reviews public claims; local static-serving and project checks evaluate integrity.
- **Permissions:** E1 local site, workflow, documentation, and evidence edits; local preview server only. No external repository/settings mutation or site deployment.
- **Effect class:** E1.
- **Target:** `project root`.
- **Budget:** No spend; one static site and bounded local preview/check pass.
- **Stop conditions:** A candidate's architecture cannot be established from NVIDIA sources; a claim implies unsupported workload coverage; publication requires repository access or an external protected effect.

**Disposition (2026-11-01):** Added a responsive, accessible static project site with evidence-based status and workload sections, the current install/launch path, NVIDIA source links, and a per-card list of architecture-matched `sm_52` candidates marked unverified. It credits Zach Shallbetter and links his personal website and GitHub profile. After review, small text was enlarged, mobile navigation remains visible, and all internal reference host-host naming was removed; the tested cards are identified as GM200 Maxwell. The manual-only GitHub Pages workflow uploads only `site/`. Browser preview rendered successfully from the local static server; HTML parsing, all nine in-page anchors, stylesheet, attribution and candidate labels, absence of reference host references, workflow YAML, and HTTP 200 checks passed. `make check` and `make test` passed; all 28 unit tests passed. The first `make check` identified stale generated context after source changes; context was regenerated and the final check passed. No GitHub remote is configured, so no public repository or Pages deployment URL exists yet; nothing was pushed or published.

## CLC-016 — Add a short command and an explained legacy GPU catalog

- **Intent:** `PROJECT_INTENT.md`; public, practical CUDA 13 compatibility for older GPUs with honest hardware and workload claims.
- **Parent:** CLC-015 GitHub Pages project site; CLC-014 CLI guidance.
- **Status:** Done
- **Objective:** Add an intuitive short command alongside the existing CLI name; document correct Python install locations for the current same-environment launcher; and expand the site from the current `sm_52` candidates to an explained catalog of NVIDIA's legacy compute-capability tiers and representative device models.
- **Scope:** `pyproject.toml`; `src/cuda_legacy_compat/cli.py`; `tests/test_cli.py`; `README.md`; `docs/user-guide.md`; `site/index.html`; `site/gpu-catalog.html`; `site/styles.css`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Removing the existing command; changing launcher behavior or adding a system-wide installer; claiming the catalog means CUDA 13 compatibility; expanding GPU support; creating a GitHub repository, pushing, or deploying the site.
- **Dependencies:** Current package entry point and same-environment launcher contract; Python Packaging User Guide install-location documentation; NVIDIA legacy compute-capability listing; CUDA 13 architecture notes.
- **Acceptance:** Both short and existing commands work and print their invoked name in help/version output; common virtual-environment and user-site command locations are explained without recommending isolated `pipx` for the current GPU run path; site covers NVIDIA legacy CC 1.0–7.2 tiers with original explanatory context, model examples, and clear project/PyTorch status; all unverified cards remain clearly labeled; local page links and assets resolve; project checks pass.
- **Evidence required:** Fresh isolated package install exercising both entry points; focused alias/help tests; local site HTTP and link checks; project `make check` and `make test`; source links and reviewed diff/status.
- **Evaluator:** CLI subprocess and unit checks; local static page preview; project gates.
- **Permissions:** E1 local code, test, documentation, and evidence edits; temporary virtual environment and local preview server only. No external repository/settings mutation or site deployment.
- **Effect class:** E1.
- **Target:** `project root`.
- **Budget:** No spend; one bounded alias, documentation, and catalog implementation pass.
- **Stop conditions:** Alias dispatch changes parser behavior for the existing name; catalog architecture mapping cannot be grounded in NVIDIA's listing; install guidance suggests a path whose Python environment lacks the required PyTorch wheel.

- **Disposition (2026-11-01):** Added `cuda-legacy` as the preferred short command while retaining `cuda-legacy-compat`; both display their invoked name in help and version output. README and user guide use the short form and explain installation into the active PyTorch virtual/Conda environment and the Python user scripts directory. The site now links to a dedicated legacy GPU catalog with plain-language explanations for all 18 NVIDIA-listed compute-capability tiers, representative device models, per-tier project status, official source links, and explicit separation between hardware listing and software compatibility. Catalog and homepage copy contain no internal host naming. A clean temporary environment built and installed the package and both commands passed version/help checks. Browser preview rendered; all local page links/fragments resolved. `make check` and all 29 unit tests passed after regenerating generated context. Model names are representative; NVIDIA’s linked source remains authoritative for the complete device inventory. Nothing was published or pushed. Evidence: `CLC-016-command-and-catalog`.


## CLC-017 — Link the site header to the project repository

- **Intent:** Make the primary GitHub navigation open the CUDA Compatibility source repository.
- **Parent:** CLC-015 public project site.
- **Status:** Done
- **Objective:** Replace the top-bar author-profile link with the supplied project repository URL while retaining author-profile links in the attribution section.
- **Scope:** `site/index.html`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Changing repository settings, configuring a Git remote, publishing, or pushing changes.
- **Evidence required:** Confirm the top-bar anchor and label, validate page markup, and pass `make check`.
- **Permissions:** E1 local site, work-item, and evidence updates.
- **Target:** `project root`.

- **Disposition (2026-11-01):** The site header now links to `https://github.com/zachshallbetter/cuda-compatibility` and is labeled “Project repository.” The author attribution links remain unchanged. The source URL was supplied by the project owner; local markup and project checks passed. No repository settings or remote state were changed. Evidence: `CLC-017-repository-header-link`.


## CLC-018 — Present working tool features instead of private workload examples

- **Intent:** Make the public status section explain user-facing tool capabilities and their limits.
- **Parent:** CLC-015 public project site.
- **Status:** Done
- **Objective:** Replace local model/classifier result cards with a feature-oriented view of inventory/profile checks, the process-local CUDA 13 launcher, native `sm_52` operator kernels, and bounded CPU fallback.
- **Scope:** `site/index.html`; `site/styles.css`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Removing reproducibility details from the research log or changing feature implementation/support claims.
- **Evidence required:** Check visible copy for local project/model names, validate capability claims against the user guide/research record, preview desktop and mobile layout, and pass `make check`.
- **Permissions:** E1 local site, work-item, and evidence edits; local preview only.
- **Target:** `project root`.

- **Disposition (2026-11-01):** The “What works today” section now describes four tool features: GPU inventory/profile diagnostics, the CUDA 13 process launcher/version facade, selected native `sm_52` operator kernels, and bounded inference CPU fallback. It no longer features the locally used checkpoint or classifier; limits and experimental labels remain explicit. Responsive layout uses two columns on wider screens and one on narrow screens. Site markup and local links were reviewed; project checks passed. Evidence: `CLC-018-tool-features-status-section`.


## CLC-019 — Publish a complete PyTorch operator and runtime coverage inventory

- **Intent:** Help users evaluating newer CUDA on legacy devices understand exactly which tool paths exist and their execution limits.
- **Parent:** CLC-015 public project site; CLC-018 tool-feature status section.
- **Status:** Done
- **Objective:** List native GPU operators, CPU fallback registrations, version-facade behavior, host/device constraints, and unsupported boundaries on the project site.
- **Scope:** `site/index.html`; `site/styles.css`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Claiming general CUDA 13 API/library compatibility or broadening code support.
- **Dependencies:** `src/cuda_legacy_compat/torch_adapter.py`; `docs/architecture.md`; `docs/research/pytorch-cu130-maxwell.md`; `docs/research/runtime-version-facade.md`.
- **Acceptance:** Public site describes all explicitly registered native and CPU operator paths, exact documented shape/dtype/memory/inference limits, one-query process-local runtime facade, and unsupported general runtime/library/training scope; feature wording does not present private model workloads as product features; responsive layout and local links remain valid.
- **Evidence required:** Compare coverage against code and research; render site preview; validate HTML and links; pass `make check` and `make test`.
- **Permissions:** E1 local site, work-item, evidence, and validation updates. No publication or remote mutation.
- **Target:** `project root`.

- **Disposition (2026-11-01):** Expanded the feature section with a native GPU operator matrix and a complete list of CPU fallback registrations, including case-specific dtype, shape, and size boundaries. Clarified that the launcher changes one CUDA Runtime version query, retains real hardware identity, and relies on the installed PyTorch runtime/libraries rather than replacing the CUDA 13 API surface. Added the CPU transfer/memory, single-device, inference-only, aliasing, and training limits. The desktop/mobile layout and public links were validated; `make check` and `make test` passed. Evidence: `CLC-019-operator-runtime-coverage-inventory`.

## CLC-020 — Apply DPP/RPP and publish user documentation

- **Intent:** Make the public project site useful and discoverable to engineers evaluating CUDA 13-era software on older GPUs, while keeping claims evidence-bound.
- **Parent:** CLC-015 public project site; CLC-019 operator/runtime coverage inventory.
- **Status:** Done
- **Objective:** Apply pinned DPP and RPP guidance to the site, create linked user documentation pages, and keep navigation, crawler metadata, social previews, and agent entry points consistent.
- **Scope:** `site/index.html`; `site/gpu-catalog.html`; `site/docs/`; `site/styles.css`; `site/robots.txt`; `site/sitemap.xml`; `site/llms.txt`; `site/assets/social-card.*`; `docs/FORMAL_RESOURCES.md`; `FORMAL_RESOURCE_MANIFEST.json`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Claiming DPP conformance or empirical discovery; submitting an RPP publication/archive; analytics or tracking; repository settings changes, push, or deployment.
- **Dependencies:** DPP 0.3.0 Working Draft and RPP 1.0.0 Normative Candidate pinned to protocols repository revision `38eb016986731bed3eaa7aecfc6ba14f3fbc8ccd`; actual project implementation, support inventory, and recorded evidence.
- **Acceptance:** Homepage, catalog, and docs share the same primary navigation; documentation provides setup, support matrix, architecture, CLI, and evidence pages with secondary navigation; each page has canonical and social metadata; structured data, sitemap, robots policy, social card, and agent-readable index resolve; public claims match implementation and preserve explicit limits; local links and responsive rendering work; project checks pass. Live crawl/index/retrieval outcomes remain explicitly unverified.
- **Evidence required:** Read-only DPP conformance/release-gate assessment; RPP checker result and applicability; HTML/link/canonical/schema/sitemap/robot validation; local HTTP/browser preview; `make check` and `make test`; final Git status/diff review.
- **Permissions:** E1 local documentation/site/evidence edits and local validation only. No analytics, external communications, publication, or deployment.
- **Target:** `project root`.

- **Disposition (2026-11-01):** Added a six-page documentation section (setup, support matrix, architecture, CLI, research/evidence, and docs index) with a responsive secondary nav; aligned the primary navigation across the homepage, GPU catalog, and docs; and linked all documentation from the homepage. Added per-page canonical/Open Graph/Twitter metadata, shared social preview art, Schema.org article/breadcrumb markup, a sitemap, robots policy, and an agent-readable `llms.txt`. Search and user-initiated retrieval crawlers are allowed while named model-training crawlers are disallowed; the agent index documents its informal status and claims boundary. Public claims were tied to the implementation inventory and RPP-style source/method/limitation records. The DPP review supports local implementation/testing only; live fetch, indexing, social previews, agent retrieval, distribution, and outcome measurement remain gray/unobserved, and analytics were not added. GitHub Pages currently returns HTTP 404, so no live outcome is claimed. RPP checker gates 1 and 5 fail because archival package files (`LICENSE`, `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`) are absent; gates 2–4 pass with warnings, including no manuscript/metadata archive. This site task is not an RPP archival publication and no deposit/readiness claim is made. Local browser preview, 8-page metadata/link/schema/navigation/sitemap validation, 12 HTTP route checks, `make check`, and all 29 unit tests passed. Nothing was published or pushed. Evidence: `CLC-020-dpp-rpp-site-docs`.

## CLC-021 — Add opt-in site analytics and documented search measurement

- **Intent:** Measure whether the project site helps visitors find compatibility, installation, and evidence information while minimizing visitor data collection.
- **Parent:** CLC-020 DPP/RPP site and documentation.
- **Status:** In progress
- **Objective:** Add consent-gated Google Analytics 4 page and named link events, a clear privacy/settings page, and instructions for Google Search Console and sitemap submission. Preserve existing SEO/social/agent metadata and keep analytics inactive until a site owner supplies a valid GA4 measurement ID.
- **Scope:** `site/**`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Creating or modifying Google properties, processing real visitor data, legal-compliance certification, adding advertising/personalization, supplying a measurement ID or Search Console token, configuring repository settings, publishing, or deployment.
- **Dependencies:** DPP 0.3.0 measurement/privacy requirements and current site page inventory.
- **Acceptance:** No Google tag or analytics request loads before affirmative analytics consent; missing measurement ID disables the banner and tracking; rejection persists and can be changed later; withdrawal blocks future manually named project events, updates consent to denied, and clears current-host GA cookies where possible; page paths/referrers exclude query and fragment data; manually named events use a documented allowlist and never carry arbitrary URLs or user input; GA4 automatic events and possible consent signals are documented; privacy, retention, and Search Console setup are documented; every public page links to the policy; existing canonical, sitemap, robots, structured data, social previews, and agent index remain consistent.
- **Evidence required:** Focused browser/network checks for no-ID, reject, accept, event, and revoke paths using a non-production test ID; metadata/link/sitemap review; DPP privacy and measurement review; `git status`, diff stat, and key hunks. No real visitor data may be generated during verification.
- **Evaluator:** Project owner reviews the claims and event scope; local browser and static checks evaluate implementation.
- **Permissions:** E1 local code, documentation, evidence, and generated-context edits; synthetic browser checks only. No external account changes, publication, or deployment.
- **Target:** `project root`.
- **Budget:** No spend; one bounded local implementation and review pass.
- **Stop conditions:** Analytics cannot remain inactive before consent, the implementation would expose personal data, or required behavior needs an external account or credential.

## CLC-022 — Publish the CUDA Compatibility repository and site

- **Intent:** Make the public CUDA Compatibility source repository and its evidence-backed GitHub Pages site available to users.
- **Parent:** CLC-015 public site; CLC-020 DPP/RPP docs; owner request to publish the repository.
- **Status:** In progress
- **Objective:** Review and commit the complete local project candidate, connect it to the owner-confirmed `zachshallbetter/cuda-compatibility` repository, push the intended branch, and publish the site through the existing GitHub Pages workflow when project authorization permits.
- **Scope:** Current project source, documentation, tests, research/evidence, `site/**`, `.github/workflows/pages.yml`, repository-local Git configuration, `docs/WORK_ITEMS.md`, `records/evidence.jsonl`, and generated context.
- **Exclusions:** Creating a different repository, changing project claims, configuring analytics or Search Console credentials, changing GitHub security/branch policies, modifying the NVIDIA driver/toolkit, publishing a software release/package, or promoting unverified GPU/workload claims.
- **Dependencies:** Existing public repository identity confirmed by owner; local candidate review; CUDA Compatibility GitHub Project board #44; an ACP policy registered for this repository; valid ACP authorization for protected push and Pages deployment.
- **Acceptance:** No secrets or private local artifacts enter the public commit; the reviewed project candidate has a root commit on the intended branch; the public repository contains that commit; GitHub Pages deployment succeeds at the canonical URL; the published site and repository are fetched and checked; deployment and search indexing outcomes are reported separately.
- **Evidence required:** Local staged/unstaged/untracked inventory; project checks; secret/path review; commit and remote identities; fresh ACP decision and pre-push/pre-deploy snapshots; GitHub push and Pages workflow receipts; live HTTP and metadata checks. Never claim publication from local files or a successful push alone.
- **Evaluator:** Project owner reviews the public commit contents; Git/ACP/GitHub deployment records and live HTTP checks evaluate completion.
- **Permissions:** E1 local review, remote configuration, and commit preparation; E2/E3 push and production deployment only with a valid exact-action ACP `ALLOW`. No ACP substitute or manual bypass.
- **Target:** Local `project root`; remote `https://github.com/zachshallbetter/cuda-compatibility`; site `https://zachshallbetter.github.io/cuda-compatibility/`.
- **Budget:** No spend; one bounded first-publication pass.
- **Stop conditions:** ACP is unavailable, has no policy registered for this repository, or refuses the exact push/deploy; remote identity/state differs from the confirmed repository; a secret/private artifact is found; or repository policy prevents safe deployment.

**Progress (2026-11-01):** The signed candidate commit `97354d2f04ae752251523b040cb0c433ab57c5f7` is present on the public `codex/cuda-compat-publication` branch. `make check`, all 36 unit tests, a clean wheel build/install, and both CLI entry points passed; the candidate scan found no credential patterns or user home paths. The ACP request for the branch push contained a different SHA than the pushed commit; the mismatch was reported as event `evt_3ec3d4dbab255ced` and ACP returned `WARNING` with authorization state `NORMAL`. The first ACP request to create `main` supplied a `new_sha` that did not match HEAD and returned `DENY` / `RECOVERY_AUTHORIZATION_REQUIRED`. A corrected exact request for commit `03928a86cda12515cfbeef4f29127e4fa20a5325` returned the same denial (request `3eea3d0ff764f02a`); no further `main` request or repair has been attempted. GitHub Pages settings now report workflow mode, but the workflow is absent from the default branch, the latest Pages build endpoint returns 404, and the canonical site returned HTTP 404. The non-protected branch was then advanced to `51ae143e47675c45ea8e7005bd3258c965298743` under exact ACP request `0ed5551370a88939`. Publication is incomplete; analytics/search properties and public announcements remain unconfigured/unpublished.

## CLC-023 — Remove third-party font requests from site page loads

- **Intent:** Keep the static site readable and lightweight without contacting a font host before a visitor chooses to follow an external link or opts into analytics.
- **Parent:** CLC-020 DPP/RPP site and documentation; CLC-021 privacy and measurement.
- **Status:** Done
- **Objective:** Replace externally hosted Google Fonts with system font stacks and accurately disclose the page-load behavior in the privacy documentation.
- **Scope:** `site/*.html`; `site/docs/*.html`; `site/styles.css`; `site/docs/privacy.html`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Adding analytics identifiers, making external account changes, changing page claims, or publishing/deploying the site.
- **Acceptance:** No page references Google Fonts or another third-party font host; system font stacks retain readable hierarchy; the privacy page says font assets are not requested from third parties; existing canonical, social metadata, navigation, and local page links remain intact.
- **Evidence required:** Static external-font scan; local HTTP preview of the homepage; canonical/navigation/link audit; `make check` and `make test`.
- **Evaluator:** Static source checks and local site preview.
- **Permissions:** E1 local site, privacy documentation, evidence, and generated-context edits only.
- **Target:** `project root`.
- **Budget:** No spend; one bounded static-site change.
- **Disposition (2026-11-01):** Removed all 27 Google Fonts stylesheet/preconnect links from nine public HTML pages and switched typography to system sans-serif and monospace stacks. Updated the privacy page to state that page loads do not request font assets from a third-party host. Local HTML/link checks and project gates passed; this does not claim a live performance measurement. Evidence: `CLC-023-system-fonts-privacy`.

## CLC-024 — Implement a scoped DPP presence and discovery system

- **Intent:** Make the project’s public representation, crawl surfaces, claims, measurements, and unknowns auditable against the pinned DPP without overstating conformance or reach.
- **Parent:** CLC-020 DPP/RPP site and documentation; CLC-021 privacy and measurement; CLC-022 publication candidate.
- **Status:** Done
- **Objective:** Maintain a versioned project presence ledger, publish direct discovery guidance, and add deterministic local release checks for canonical pages, metadata, navigation, links, sitemap, robots, and the agent index.
- **Scope:** `docs/DPP_IMPLEMENTATION.md`; `docs/FORMAL_RESOURCES.md`; `site/docs/**`; `site/analytics.js`; `site/llms.txt`; `site/sitemap.xml`; `scripts/site_checks.py`; `tests/test_site_checks.py`; `Makefile`; `docs/WORK_ITEMS.md`; `records/evidence.jsonl`; generated context.
- **Exclusions:** Declaring DPP conformance; submitting or changing search/analytics properties; live crawler probing, outreach, social publication, or AI prompt sampling without an authorized, preregistered run; claiming measured search or AI performance.
- **Dependencies:** DPP 0.3.0 pinned revision; current implementation and evidence records; repository and Pages publication remain subject to CLC-022 ACP authorization.
- **Acceptance:** Stable project/author/site identity and material claims map to sources; owned surfaces and access states distinguish represented/locally tested from publicly observed; seed query intents are labeled as hypotheses rather than trend data; metrics have denominators, source, and missingness rules; distribution and correction authority are recorded; user-facing discovery guidance explains robots/sitemap/agent-index limits; internal directory links use explicit index pages for file previews; automated checks detect canonical, metadata, link, navigation, sitemap, robots, and agent-index drift; all local release checks pass.
- **Evidence required:** Pinned DPP module review; negative and positive checker tests; local route and link audit; `make check` and `make test`; reviewed worktree/diff and updated implementation ledger.
- **Evaluator:** Project owner reviews claim wording and scope; static checker and local test suite verify page structure.
- **Permissions:** E1 local website, documentation, source checks, tests, evidence, and generated-context edits only. No external service mutation or publication.
- **Target:** `project root`.
- **Budget:** No spend; one bounded DPP implementation and release-gate pass.
- **Stop conditions:** A conformance claim would require missing external evidence; any proposed optimization would require unauthorized publication, data collection, crawler circumvention, or outreach; a required canonical schema or external system-of-record change emerges.
- **Disposition (2026-11-01):** Added a scoped DPP implementation record with stable entity IDs, eight qualified claims and source mappings, surface-by-surface state dimensions, an explicitly hypothetical English query-intent seed, metric denominators and missingness, distribution boundaries, module coverage, and update triggers. Added a public search/AI discovery guide and aligned documentation navigation, canonical/schema metadata, sitemap, `llms.txt`, privacy event names, and stable article/author/site IDs. Added `scripts/site_checks.py` to `make check` and seven positive/negative unit checks for metadata, identity, links, external resources, explicit index-page navigation, sitemap, robots, and agent-index consistency. The local checker, route audit, browser preview, `make check`, and all 36 tests passed. This is a partial local DPP implementation, not conformance; public fetch, indexing, AI retrieval, engagement, and outcomes remain `GRAY`. Evidence: `CLC-024-docs-file-preview`.

## CLC-025 — Extend PyTorch CUDA 13 behavior on Maxwell with qualified operator and training coverage

- **Intent:** `PROJECT_INTENT.md`; practical PyTorch CUDA 13 use on older NVIDIA GPUs.
- **Parent:** CLC-008 stream/module lifecycle; CLC-009 native operator slice; CLC-010 packaged launcher; CLC-019 public coverage inventory.
- **Status:** In progress
- **Objective:** Systematically strengthen the PyTorch CUDA 13 adapter on real `sm_52` devices: qualify current stream and allocation-lifetime behavior for every registered route, expand native operation/dtype/layout coverage, implement autograd/training only where gradient semantics can be preserved, map the runtime/library boundaries used by supported workloads, and publish bounded correctness/resource/performance evidence.
- **Scope:** `src/cuda_legacy_compat/torch_adapter.py`; `src/cuda_legacy_compat/compat/**`; `experiments/pytorch-cu130/**`; relevant unit/integration probes; `docs/architecture.md`; `docs/research/pytorch-cu130-maxwell.md`; `docs/roadmap.md`; `docs/WORK_ITEMS.md`; `README.md`; site compatibility/support statements; `records/evidence.jsonl`.
- **Exclusions:** Hardware identity spoofing; changing the NVIDIA driver, kernel, system CUDA installation, or official PyTorch wheel; redistributing proprietary NVIDIA components; claiming CUDA-wide or arbitrary-application compatibility; using private model inputs/configuration as public evidence; commit, push, deployment, or other protected external effects.
- **Dependencies:** Official PyTorch 2.14.0+cu130 environment; verified LLVM/NVPTX and CUDA driver headers; Titan resource broker; actual GTX TITAN X GM200 `sm_52` devices; CPU reference results and current adapter contracts.
- **Acceptance:** (1) For every existing native-kernel and CPU-fallback registration, record supported inputs, output semantics, stream dependencies, tensor lifetime, asynchronous error behavior, and explicit rejection cases; reproduce correctness on all three reference GPUs. (2) Add native Maxwell implementations and dtype/layout cases only when CPU/PyTorch-reference equivalence, edge cases, non-default streams, allocator churn/lifetime, and failure paths pass. (3) Implement forward/backward and optimizer/training paths only for declared operations with verified gradient semantics; unsupported autograd paths fail explicitly. (4) Inventory runtime API and proprietary-library boundaries encountered by selected workloads, then implement replacements only for bounded, identified behaviors. (5) Publish reproducible per-workload correctness, peak memory, latency, throughput, GPU, software-version, and fallback-path evidence with limits.
- **Evidence required:** Frozen candidate revision; exact Titan broker-selected GPU and free-memory state; PyTorch/runtime/driver/compiler/header versions; per-operator positive and negative results; CPU-reference tolerances; stream/event ordering and allocator-lifetime checks; autograd/gradient/optimizer comparisons where implemented; unsupported library/API observations; repeatable memory/latency methodology and measurements; cleanup and project gates.
- **Evaluator:** CPU PyTorch forward/backward references; deterministic operator and workload probes; all three physical reference cards; project checks and review of the published claim matrix.
- **Permissions:** E1 local source, test, probe, and documentation changes; bounded sequential Titan broker GPU runs; temporary isolated build/test artifacts and cleanup; official PyTorch/LLVM downloads into task-scoped `/tmp` paths when absent; transfer of public project source and built test artifacts into task-scoped `/tmp` paths on Titan. No host system mutation, external publication, policy change, or protected repository effect.
- **Effect class:** E1.
- **Target:** Project checkout and Titan reference GPU host through `titan resource`.
- **Budget:** No spend; sequential, small-shape probes under available VRAM; retain the existing 64 MiB fallback bound unless separately evidenced and documented. Stop a run before allocation if broker capacity does not meet its declared floor.
- **Stop conditions:** A route cannot preserve PyTorch's stream, alias, mutation, numerical, or gradient semantics; a proprietary path requires unsupported hardware instructions with no verified replacement; memory/resource limits prevent a safe run; candidate changes invalidate existing evidence; or any change requires protected external effects.
- **Progress (2026-11-01):** On broker-selected physical GPUs 1 and 2 (GTX TITAN X GM200, `sm_52`; driver 580.178.04; GPU 2 had 974 MiB free at the post-run snapshot), the official `torch 2.14.0+cu130` wheel continues to reproduce stock `mm` `CUBLAS_STATUS_ARCH_MISMATCH` and ReLU missing-image failures. All 11 registered CPU fallback routes matched CPU references. Cross-stream allocator-churn checks passed for float16/float32/int32 add, strided float32 mm, ReLU, and layer norm; arange and embedding passed on a non-default stream (embedding retains an observed synchronization). Float16 addition passed exact edge-value comparison; strided mm and float32 in-place scaled add passed. First-order gradients and final weights for a 2×2 float32 linear→ReLU→sum model matched CPU over seven `SGD(foreach=False)` steps. Seven end-to-end measured steps had median wall latency 2.360 ms on GPU 2 and 7.296 ms on GPU 1, with 67,180,032 bytes peak allocated on both; the tiny workload and explicit sync make this diagnostic only. A bounded `/proc/self/maps` probe after matrix/convolution operations observed the wheel's CUDA Runtime, cuBLAS/cuBLASLt, NVRTC, cuDNN, NCCL, `libtorch_cuda`, adapter bridge/facade, and system `libcuda.so.580.178.04`; this inventories mapped libraries, not all API calls. Layer norm's larger positive-offset test had max absolute error `3.34e-5` and uses `atol=5e-5`; its ordinary fixture remains at `atol=1e-6`. New dtype/layout/training probes passed on GPUs 1 and 2; the library-map probe ran on GPU 2 only. GPU 0 was not tested because it was below the 700 MiB free-memory floor. Other optimizers/foreach, higher-order gradients, AMP, broader training, API symbol tracing, and representative performance remain unverified. The 64 MiB per-operation fallback caps are unchanged. Publication work is now tracked on ACP-bound board #44; CLC-025 remains active in this local work ledger.
