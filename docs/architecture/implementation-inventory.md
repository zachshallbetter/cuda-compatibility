# Implementation inventory and roadmap reconciliation

**Status:** audit record for GitHub issue #46
**Reviewed:** 2026-11-03 against `main` at the commit that adds this file
**Sources:** `src/cuda_legacy_compat/`, `tests/`, `experiments/`, `profiles/`, `Formula/`, `README.md`, `docs/architecture.md`, `docs/user-guide.md`, `docs/product/product-contract.md`, `docs/roadmap.md`, `docs/compatibility/`, evidence in `.agents/records/evidence.jsonl`, and open issues #6–#47.

This file says what exists, how far each part is proven, and which open issues duplicate shipped work or each other. It does not change support claims. "Implemented" means code and tests exist; "Experimental" means the code exists and has hardware evidence for the named cases only; "Proposed" means no code exists.

## 1. Implementation matrix

| Capability | State | Code | Tests | Evidence scope | Open issues that build on it |
|---|---|---|---|---|---|
| `inspect`: machine, GPU and compiler inventory (read-only) | Implemented | `inventory.py`, `cli.py` | `test_inventory.py`, `test_cli.py` | Reports `nvidia-smi` and compiler facts; unknown facts stay unknown | #6 (export more fields) |
| `profiles list` / `profiles show` | Implemented | `profiles.py`, `profiles/*.json` | `test_profiles.py` | Three bundled profiles: PyTorch 2.14 cu126, PyTorch 2.14 cu130, TensorRT 11.4.1; each cites its source | #43 |
| `audit PROFILE`: compare machine facts with a profile | Implemented | `profiles.py` (`audit_profile`) | `test_profiles.py`, `test_cli.py` | Field-level comparison; a match is not a workload certification | #8, #12 |
| `doctor`: read-only launcher prerequisites | Implemented | `cli.py` (`_doctor_report`) | `test_cli.py` | Platform, Python, PyTorch wheel, `sm_52` GPUs, LLVM 23.1.4, CUDA 12.4 headers, `g++`; `--python` inspects another interpreter | #42 |
| `run`: process-scoped launcher | Experimental | `launch.py`, `compat/sitecustomize.py` | `test_launch.py` (preflight, interpreter selection) | Linux x86_64, every visible GPU `sm_52`, tested on GM200 with `torch 2.14.0+cu130`; `--python` selects the target interpreter | #7, #42 |
| Kernel compilation | Experimental | `launch.py` (`build_adapter`), `compat/compat_kernels.cu` | `test_launch.py` (missing-compiler path) | Clang/LLVM 23.1.4 to `sm_52` PTX at every launch, in a temporary directory removed on exit; no cache | #9 |
| CUDA Driver API bridge | Experimental | `compat/driver_add_bridge.cpp` | Hardware probes in `experiments/pytorch-cu130/` | Loads and launches the compiled PTX on PyTorch's current stream | #9, #28 |
| CUDA Runtime version facade | Experimental | `compat/cuda_runtime_version_facade.cpp` | `experiments/runtime-facade/` | Answers `cudaRuntimeGetVersion` with 13000 for the child process; forwards everything else | #24 |
| GPU-native PyTorch operators | Experimental | `torch_adapter.py` | Hardware evidence CLC-007, CLC-009, CLC-025 | 6 operation families as 8 support-matrix cases (add, arange, embedding, layer norm, matmul, ReLU); counts are checked against the registrations by `make check` | #10, #27, #28 |
| CPU fallbacks | Experimental | `torch_adapter.py` (`_cpu_fallback`) | CLC-025 | 11 registered routes plus addition's out-of-limit path; 64 MiB input and output caps; selected first-order gradients | #10 |
| Training path | Experimental | `torch_adapter.py` | CLC-025 | Seven first-order SGD steps of a 2×2 linear→ReLU→sum model on GPUs 1 and 2 only | #13, #26 |
| JSON output | Implemented | `cli.py` (`--json` on `inspect`, `profiles`, `doctor`; `audit` always JSON) | `test_cli.py` | Stable field names are not yet versioned | #41, #42 |
| Run reports and a `report` command | Proposed | none | none | `run` returns the child's exit code and prints errors; it writes no report | #42 |
| Workload preflight (`analyze`) | Proposed | none | none | No static or traced operator coverage exists | #12, #42, #43 |
| Install, uninstall, rollback | Delegated, not implemented | `pyproject.toml`, `Formula/cuda-legacy.rb` | Formula: `brew style` only | Installation is pip or Homebrew; `run` makes no system changes and removes its temporary files, so there is nothing to roll back today | #42 |
| Per-device architecture selection | Proposed | `doctor` and `run` reject any non-`sm_52` device | n/a | Hard gate by design until #7 | #7, #14, #35 |
| Correctness, differential and performance harnesses | Proposed (scripts only) | `experiments/maxwell`, `experiments/llvm-nvptx`, `experiments/pytorch-cu130` | Shell and Python probes, not a harness | Each probe proves its named cases | #13, #39, #40 |
| CUDA 12.9→13.0 change register | Partial (v0.1) | `docs/compatibility/cuda-12.9-to-13.0-primary-source-diff.md` | n/a | Exact component versions and verified breaking changes from NVIDIA sources | #32, #37 |
| Evidence ledger | Implemented (JSONL) | `.agents/records/evidence.jsonl`, `docs/DPP_IMPLEMENTATION.md` | `make check` validates the bootstrap records | Claim IDs map to evidence IDs | #41, #50 |

There is one launcher. No issue should add a second one: #42's "process-scoped run wrapper" is `cuda-legacy run`, and new behavior belongs there.

## 2. Overlapping issues and recommended consolidation

These pairs describe the same work or one is a component of the other. The recommendation is to keep the first issue as the owner and track the others as sub-issues or close them as duplicates; closing is an owner decision and has not been done.

| Owner | Overlaps | Why | Recommended shape |
|---|---|---|---|
| #8 capability registry | #28 semantic operation registry | Both define registries of what a device or backend can run; #28 adds operation descriptors and dispatch | One registry: device capabilities (#8) are inputs to operation dispatch (#28). Make #28 depend on #8. |
| #10 operator library and fallback policy | #27 first replacement kernel pack | #27 is the first deliverable of #10 | Make #27 a milestone sub-issue of #10. |
| #13 qualification matrix | #39 correctness harness, #40 performance lab | The matrix needs both harnesses | Keep #39 and #40 as components; #13 owns the matrix output. |
| #7 remove the `sm_52` gate | #35 architecture profiles, #9 compilation cache | Removing the gate needs a backend interface (#35) and per-target builds (#9) | Sequence #35 → #9 → #7; #14 (Pascal) is the first consumer. |
| #42 install/doctor/inspect/run workflow | #12 preflight `analyze`, #43 workload profiles | #12 is the analyze step of #42; #43 supplies its profiles | Build `analyze` under #12 and treat it as part of #42's contract; #43 delivers data for it. |
| #25 kernel compatibility matrix | #47 failure-to-blocker corpus | #47's corpus is #25's input inventory | Make #47 the data source for #25. |
| #36 architecture and decision records | #24 behavioral contract | The contract is one section of the architecture specification | Write #24 as the contract chapter of #36. |
| #32 change diff | #37 artifact extraction tooling | #37 is how #32's register is produced and refreshed | Extend the existing v0.1 register; #37 automates it. |
| #41 evidence ledger design | #50 public examples, `.agents/records/evidence.jsonl` | A ledger already exists | #41 should version and extend the existing JSONL schema, not start a new store. |
| #6 GPU inventory and card swaps | #23 fourth TITAN X, #16 subarchitectures | Each is a card qualification under #6's protocol | Run #23 and #16 under #6's runbook. |
| #45 design system and dashboard | Shipped site redesign (CLC-029) | The public site now has its own design system | Narrow #45 to the optional coverage dashboard. |

`docs/roadmap.md` milestones (M0–M8) predate these issues. M4–M6 map to #10, #13, #26, #27, #28 and #42; M7 to #17 and #22; M8 to #30 and #31.

## 3. CLI naming migration plan

The product contract proposed `cuda-compat install | doctor | inspect -- python app.py | run -- python app.py | report --last`. That conflicts with shipped behavior: `inspect` already reports the machine, not a workload, and the executable is `cuda-legacy`. The plan keeps every existing command and alias:

1. **Executable:** keep `cuda-legacy` and the `cuda-legacy-compat` alias. Do not introduce `cuda-compat`.
2. **`inspect`:** unchanged. It stays the read-only machine and GPU inventory.
3. **Workload inspection:** add `cuda-legacy analyze [--python PATH] -- SCRIPT [args]` (#12). It is static by default; any traced execution is opt-in and isolated.
4. **`run`:** unchanged arguments. Add an optional `--report PATH` that writes a redacted JSON run report; no report is written unless asked.
5. **`report`:** add `cuda-legacy report PATH` to render a saved run report. Only after (4) exists.
6. **Install, uninstall, rollback:** remain the job of pip and Homebrew. `doctor` stays read-only. Add rollback only if a future command changes machine state.
7. **Flags:** `--json` exists. `--no-cpu-fallback` and `--dry-run` are proposed for `run`; `--offline` for any command that would fetch data.

Each step ships with tests that pin the existing commands' output and exit codes so the migration cannot break them.

## 4. Corrections made with this audit

- `docs/product/product-contract.md`: the minimal path now uses the reconciled commands and marks `analyze`, `report` and `--report` as proposed.
- `docs/user-guide.md`: the pipx paragraph now explains `--python` and Homebrew instead of saying a separate environment cannot launch workloads.
- `README.md`: links this inventory.

## 5. Open questions for the owner

- Close or convert the duplicates in section 2 into sub-issues.
- Confirm the `analyze` name for workload inspection, which #12 already uses.
- Decide whether `run --report` should be on by default once it exists. This audit recommends off, so nothing is written without consent.
