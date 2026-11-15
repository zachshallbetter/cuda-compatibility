# Verification Craft & Quality Gates

## 1. Quality Gates That Cannot Fail

- **Prove Failure Capability**: Every gate or test must be proven capable of failing when fed a defect before being trusted in CI.
- **No-Op Protection**: CI workflows must report how many assertions actually executed, ensuring skipped or category-error jobs cannot produce a false green check.
- **Cross-Toolchain Determinism**: Compare build outputs and conformance vectors across distinct engines or runtime environments to prove platform determinism.

## 2. Standard Repository Gates

Before proposing or merging changes, execute the authoritative verification suite:

```bash
python3 .agents/bootstrap/scripts/validate-bootstrap.py --check-context
python3 .agents/bootstrap/scripts/validate-formal-files.py
python3 .agents/bootstrap/scripts/check-protocol-integrity.py
python3 .agents/bootstrap/scripts/check-negative-results.py
```

## 3. Ephemeral Worktree Isolation

- Autonomous verification tasks must run in isolated worktrees (`git worktree add`) or temporary directory environments to avoid dirtying developer working state.
- Automatically clean up temporary worktrees and test harnesses upon completion or interruption.

## 4. Hosted Builds and CI Minutes

Hosted previews, deployments and CI run on paid machines; every push, rebase force-push and merge can start one. The local gate is the check of record.

- **Before the first push**, list each connected service and what starts a build on it: `vercel.json` (and the `vercel.json` in the project's root directory), `.github/workflows/*` triggers, other hosts' configuration.
- **Vercel**: branches other than the production branch do not build (`"git": {"deploymentEnabled": {"main": true, "**": false}}`, or an `ignoreCommand`); a preview is built on request from a `preview/*` branch. Commit-message tokens are not used as triggers.
- **CI**: `pull_request` plus `push` to `main` only; path filters; a `concurrency` group per ref with `cancel-in-progress: true`; `timeout-minutes` on every job; no matrix or schedule unless the owner asked for one.
- **Pushes**: a branch is pushed when it is ready, not after each commit; fixes are batched into one push.
- Run `python3 .agents/bootstrap/scripts/check-build-budget.py` after changing workflows or Vercel configuration; it fails on push-any-branch triggers and jobs without timeouts.

## 5. Local Builds (Rust, Swift, WebAssembly)

- Every compile and test runs through the machine register: `reg build --label <lane/ticket> -- <command>`; combine steps into one `reg build -- sh -c "a && b"`. Dev servers run with `reg serve` from a built binary, never from a shared target folder or `cargo run`.
- Test only the packages a change touches (`cargo test -p <crate>`); the full suite runs in the local gate.
- One integration-test program per package: modules under `tests/<name>/` instead of many files in `tests/`, since each file links its own program against every dependency.
- Expensive targets (wasm, release) build only when their sources changed.
