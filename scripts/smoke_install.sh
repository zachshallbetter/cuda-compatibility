#!/bin/sh
# Install the package into a throwaway venv and drive the first-run path with no GPU and an empty home.
# Usage: sh scripts/smoke_install.sh   (make smoke; CI runs it on every pull request)
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m venv "$work/venv"
"$work/venv/bin/python" -m pip install -q "$root"
bin="$work/venv/bin/cuda-legacy"
export HOME="$work/home" XDG_CONFIG_HOME= XDG_DATA_HOME=
unset CUDA_COMPAT_LLVM_ROOT CUDA_COMPAT_CUDA_ROOT CUDA_COMPAT_PYTHON 2>/dev/null || true
mkdir -p "$HOME"
fail() { echo "smoke: $1" >&2; exit 1; }

"$bin" | grep -q "Not set up yet" || fail "bare command should say the machine is not set up"
"$bin" --version | grep -q "^cuda-legacy " || fail "--version should name the short command"
"$bin" setup --dry-run --json > "$work/plan.json" || fail "setup --dry-run should exit 0"
"$work/venv/bin/python" - "$work/plan.json" <<'PY' || fail "dry-run plan is wrong"
import json, sys
plan = json.load(open(sys.argv[1]))["plan"]
download = plan["llvm_download"]
assert download and len(download["sha256"]) == 64 and download["url"].startswith("https://github.com/llvm/llvm-project/"), download
PY
[ ! -e "$HOME/.config/cuda-legacy" ] || fail "dry run must not save anything"
set +e
"$bin" doctor > "$work/doctor.txt" 2>&1; code=$?
set -e
[ "$code" -eq 1 ] || fail "doctor without setup should exit 1 (got $code)"
grep -q 'Run `cuda-legacy setup`' "$work/doctor.txt" || fail "doctor should point at cuda-legacy setup"
set +e
echo 'print(1)' > "$work/app.py"
"$bin" run "$work/app.py" > "$work/run.txt" 2>&1; code=$?
set -e
[ "$code" -eq 2 ] && grep -q "cuda-legacy setup" "$work/run.txt" || fail "run without setup should exit 2 and name setup"
"$bin" setup --remove | grep -q "Nothing to remove" || fail "remove with nothing saved should say so"
echo "OK install smoke: bare command, dry-run plan, doctor and run guidance, remove"
