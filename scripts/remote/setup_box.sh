#!/usr/bin/env bash
# Runs ON the rented machine, from the copied repository:
#
#   bash scripts/remote/setup_box.sh [--watchdog-minutes N] [--no-watchdog] [--skip-models] [--models "ID@REV ..."]
#
# It builds the Python environment in /workspace/venv, downloads the model snapshots into /workspace/hf and
# writes /workspace/env.sh (which `rent.py run` reads before every command).
#
# How to read its output: every step prints "STEP n: name" and then "STEP n: OK" or "STEP n: FAILED (reason)".
# A failed step that is needed stops the script with exit code 1. A failed optional step (the extra packages
# and the fast kernels) is recorded and the script goes on. The last lines are a table of all steps and one
# line that starts with "RESULT:". The same table is in /workspace/setup_summary.txt, everything that was
# printed is in /workspace/setup.log, and the installed versions are in /workspace/env_freeze.txt.
#
# It can be run again at any time: steps that are already done are quick.
# No secret is used here and "set -x" is not used.
set -u -o pipefail

WORKSPACE="${ACCESSDESK_WORKSPACE:-/workspace}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$WORKSPACE/venv"
PY="$VENV/bin/python"
LOG="$WORKSPACE/setup.log"
SUMMARY="$WORKSPACE/setup_summary.txt"
REQUIREMENTS="$REPO/requirements-gpu.txt"
OPTIONAL_REQUIREMENTS="$REPO/requirements-gpu-optional.txt"
PYTHON_VERSION="3.11"
MIN_FREE_GB=30      # environment about 8 GB, the two models about 14 GB, the package cache; not measured

# The snapshots, as "repository@revision". The revisions are the ones written down in _dev/GPU_SPEC.md
# (read from the Hub on 2026-11-16). The 2B comes first: the first smoke test runs on it.
MODELS="Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc"
MODELS="$MODELS Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

WATCHDOG_MINUTES=""
NO_WATCHDOG=0
SKIP_MODELS=0
STEP=0
REASON=""
OPTIONAL_FAILED=0

usage() {
    sed -n '2,5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ----------------------------------------------------------------------------------------------------------
# Steps: bookkeeping
# ----------------------------------------------------------------------------------------------------------
finish() {  # finish EXIT_CODE: print the table of steps and the RESULT line, then leave
    echo
    echo "Steps of this run:"
    sed 's/^/  /' "$SUMMARY"
    if [ "$1" -ne 0 ]; then
        echo "RESULT: NOT READY (a needed step failed; see above and $LOG)"
    elif [ "$OPTIONAL_FAILED" -ne 0 ]; then
        echo "RESULT: READY, but $OPTIONAL_FAILED optional step(s) failed (see the table)"
    else
        echo "RESULT: READY"
    fi
    exit "$1"
}

step() {  # step needed|optional "name" function [arguments]: run one step and record how it ended
    local kind="$1" name="$2"
    shift 2
    STEP=$((STEP + 1))
    REASON=""
    echo
    echo "STEP $STEP: $name"
    if "$@"; then
        echo "STEP $STEP: OK"
        echo "STEP $STEP: OK      $name" >> "$SUMMARY"
        return 0
    fi
    echo "STEP $STEP: FAILED (${REASON:-no reason recorded; see the lines above})"
    echo "STEP $STEP: FAILED  $name ($kind): ${REASON:-see $LOG}" >> "$SUMMARY"
    if [ "$kind" = "needed" ]; then
        finish 1
    fi
    OPTIONAL_FAILED=$((OPTIONAL_FAILED + 1))
    return 0
}

# ----------------------------------------------------------------------------------------------------------
# The steps
# ----------------------------------------------------------------------------------------------------------
watchdog_alive() {  # true if the process named in the watchdog's pid file runs
    local pid
    pid="$(head -c 20 "$WORKSPACE/watchdog.pid" 2>/dev/null | tr -d '[:space:]')"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

count_lines() {  # count_lines PATTERN: how many lines of the watchdog log contain it ("" counts all lines)
    local count
    count="$(grep -c -- "$1" "$WORKSPACE/watchdog.log" 2>/dev/null)"
    echo "${count:-0}"
}

step_watchdog() {
    if [ -n "$WATCHDOG_MINUTES" ]; then
        local armed_before fatal_before waited=0
        armed_before="$(count_lines ' ARMED: ')"
        fatal_before="$(count_lines ' FATAL: ')"
        $(command -v setsid) nohup bash "$REPO/scripts/remote/watchdog.sh" "$WATCHDOG_MINUTES" \
            >> "$WORKSPACE/watchdog.out" 2>&1 < /dev/null &
        # The watchdog writes one new line when it is ready: "ARMED", or "FATAL" if it refuses to start.
        while [ "$(count_lines ' ARMED: ')" -le "$armed_before" ] \
            && [ "$(count_lines ' FATAL: ')" -le "$fatal_before" ] && [ "$waited" -lt 150 ]; do
            sleep 3
            waited=$((waited + 3))
        done
        if [ "$(count_lines ' ARMED: ')" -le "$armed_before" ]; then
            REASON="the watchdog did not start: $(tail -n 1 "$WORKSPACE/watchdog.log" 2>/dev/null)"
            return 1
        fi
    fi
    if watchdog_alive; then
        grep ' ARMED: ' "$WORKSPACE/watchdog.log" | tail -n 1
        return 0
    fi
    if [ "$NO_WATCHDOG" -eq 1 ]; then
        echo "WARNING: no watchdog runs. This machine will NOT destroy itself."
        return 0
    fi
    REASON="no watchdog runs; pass --watchdog-minutes N (or --no-watchdog to go on without one)"
    return 1
}

step_machine() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        REASON="nvidia-smi not found: this container has no GPU driver"
        return 1
    fi
    if ! nvidia-smi; then
        REASON="nvidia-smi failed"
        return 1
    fi
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    echo "CUDA supported by the driver: $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n 1)"
    echo "system: $(grep -m1 PRETTY_NAME /etc/os-release 2>/dev/null); $(ldd --version 2>/dev/null | head -n 1)"
    echo "CPUs: $(nproc)"
    free -g
    # The host's memory is shared between its containers; this is the share of this one, if a limit is set.
    local limit free_gb
    limit="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
    echo "memory limit of this container: ${limit:-unknown} (bytes, or 'max' for none)"
    df -h "$WORKSPACE"
    free_gb="$(df -BG --output=avail "$WORKSPACE" | tail -n 1 | tr -dc '0-9')"
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
        REASON="only ${free_gb:-0} GB free in $WORKSPACE; at least $MIN_FREE_GB GB are needed"
        return 1
    fi
}

step_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        if ! curl -LsSf https://astral.sh/uv/install.sh -o "$WORKSPACE/uv-install.sh"; then
            REASON="could not download the uv installer from astral.sh"
            return 1
        fi
        if ! sh "$WORKSPACE/uv-install.sh"; then
            REASON="the uv installer failed"
            return 1
        fi
    fi
    if ! command -v uv >/dev/null 2>&1; then
        REASON="uv is not on PATH after the installer ran (looked in $HOME/.local/bin)"
        return 1
    fi
    echo "uv: $(command -v uv), $(uv --version)"
}

step_venv() {
    if ! uv venv --python "$PYTHON_VERSION" --allow-existing "$VENV"; then
        REASON="uv venv failed"
        return 1
    fi
    local version
    version="$("$PY" --version 2>&1)"
    echo "$version at $PY"
    if [[ "$version" != "Python $PYTHON_VERSION."* ]]; then
        REASON="$PY is '$version', not Python $PYTHON_VERSION"
        return 1
    fi
}

step_requirements() {
    if ! uv pip install --python "$PY" -r "$REQUIREMENTS"; then
        REASON="uv pip install -r requirements-gpu.txt failed"
        return 1
    fi
}

install_causal_conv1d() {  # install_causal_conv1d "causal-conv1d==1.7.0"
    # What its setup.py (version 1.7.0, read from the source archive) does:
    #   * it imports torch, packaging and wheel while it runs, so it must be built in OUR environment
    #     (--no-build-isolation); in a separate build environment it would meet a different torch;
    #   * `bdist_wheel` first tries to download a ready-made wheel for this torch, CUDA and Python from the
    #     project's GitHub release page, and builds from source only if that file does not exist;
    #   * building needs nvcc (CUDA_HOME, or nvcc on PATH, or /usr/local/cuda) and g++; without nvcc the
    #     script stops with an error unless CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE is set, and with that
    #     variable only the ready-made wheel can give a working package.
    # So the result is always checked by importing it, and a package that does not import is removed again.
    local requirement="$1" nvcc="" candidate code
    for candidate in "$(command -v nvcc 2>/dev/null)" "${CUDA_HOME:-/nonexistent}/bin/nvcc" \
        /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            nvcc="$candidate"
            break
        fi
    done
    if ! uv pip install --python "$PY" -c "$REQUIREMENTS" setuptools wheel ninja packaging; then
        REASON="could not install the build tools (setuptools, wheel, ninja, packaging)"
        return 1
    fi
    if [ -n "$nvcc" ] && command -v g++ >/dev/null 2>&1; then
        echo "CUDA compiler: $nvcc ($("$nvcc" --version 2>/dev/null | grep -i release)); g++: $(command -v g++)"
        echo "A ready-made wheel is tried first; if there is none, the CUDA code is compiled (some minutes)."
        CUDA_HOME="$(dirname "$(dirname "$nvcc")")" timeout 2400 \
            uv pip install --python "$PY" --no-build-isolation -c "$REQUIREMENTS" "$requirement"
        code=$?
    else
        echo "No nvcc or no g++ in this container (nvcc: '${nvcc:-none}'): only a ready-made wheel can work."
        CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE timeout 900 \
            uv pip install --python "$PY" --no-build-isolation -c "$REQUIREMENTS" "$requirement"
        code=$?
    fi
    if [ "$code" -ne 0 ]; then
        REASON="the install command ended with code $code (124 means: time limit)"
        return 1
    fi
    if ! "$PY" -c "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update"; then
        uv pip uninstall --python "$PY" causal-conv1d
        REASON="it was installed but cannot be imported, so it was removed again"
        return 1
    fi
}

install_optional() {  # install_optional "name==version": one line of requirements-gpu-optional.txt
    local requirement="$1"
    case "$requirement" in
        causal-conv1d*)
            install_causal_conv1d "$requirement"
            ;;
        *)
            if ! uv pip install --python "$PY" -c "$REQUIREMENTS" "$requirement"; then
                REASON="uv pip install $requirement failed"
                return 1
            fi
            ;;
    esac
}

normalise() {  # package lines in lower case with "-" for "_", as uv prints them
    tr 'A-Z_' 'a-z-'
}

step_pins() {  # every "name==version" line of requirements-gpu.txt is installed exactly
    local frozen installed line wanted missing=""
    if ! frozen="$(uv pip freeze --python "$PY")"; then
        REASON="uv pip freeze failed"
        return 1
    fi
    printf '%s\n' "$frozen" > "$WORKSPACE/env_freeze.txt"
    echo "all installed versions: $WORKSPACE/env_freeze.txt ($(wc -l < "$WORKSPACE/env_freeze.txt") packages)"
    installed="$(normalise < "$WORKSPACE/env_freeze.txt")"
    while IFS= read -r line; do
        wanted="$(printf '%s' "$line" | normalise)"
        # grep reads the whole list (no -q): with "pipefail" an early exit could look like a failure
        if ! grep -xF -- "$wanted" <<< "$installed" > /dev/null; then
            missing="$missing $line"
        fi
    done < <(grep -E '^[A-Za-z0-9_.-]+==' "$REQUIREMENTS")
    if [ -n "$missing" ]; then
        REASON="not installed at the pinned version:$missing"
        return 1
    fi
    echo "every pinned version of requirements-gpu.txt is installed"
}

step_torch() {  # versions, and one tiny bfloat16 product on the GPU
    "$PY" - <<'PYTHON'
import importlib.metadata as metadata
import sys

import torch

for name in ("torch", "transformers", "peft", "accelerate", "flash-linear-attention", "fla-core", "numpy",
             "tokenizers", "safetensors", "huggingface-hub", "triton", "causal-conv1d", "kernels", "trl"):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}: not installed")
print("python", sys.version.split()[0], "| torch built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
print("GPU:", torch.cuda.get_device_name(0), "| compute capability", torch.cuda.get_device_capability(0))
ones = torch.ones(4, 4, device="cuda", dtype=torch.bfloat16)
total = float((ones @ ones).sum().item())
print("bfloat16 product on the GPU:", total, "(expected 64.0)")
if total != 64.0:
    sys.exit("the GPU gave a wrong result")
PYTHON
    local code=$?
    if [ "$code" -ne 0 ]; then
        REASON="torch cannot use the GPU (see the lines above)"
        return 1
    fi
}

step_fla() {  # the two functions that transformers takes from flash-linear-attention for Qwen3.5
    # Names from transformers 5.23.0: models/qwen3_5/modeling_qwen3_5.py (the two decorators with "fla") and
    # integrations/hub_kernels.py (_KERNELS_INTERNAL_PATH_MAPPINGS: both live in fla.ops.gated_delta_rule).
    local code
    "$PY" - <<'PYTHON'
import fla
import fla.ops
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

print("fla.ops:", fla.ops.__file__)
print("kernels:", chunk_gated_delta_rule.__module__, fused_recurrent_gated_delta_rule.__module__)
PYTHON
    code=$?
    if [ "$code" -ne 0 ]; then
        REASON="fla.ops cannot be imported: the linear-attention layers would run on the slow PyTorch path"
        return 1
    fi
}

step_causal_conv1d() {
    local code
    "$PY" - <<'PYTHON'
import causal_conv1d
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

print("causal_conv1d:", causal_conv1d.__file__)
PYTHON
    code=$?
    if [ "$code" -ne 0 ]; then
        REASON="causal_conv1d cannot be imported: its convolution runs on the plain PyTorch path (slower)"
        return 1
    fi
}

step_model() {  # step_model "repository@revision": download one snapshot into the Hugging Face cache
    local repository="${1%@*}" revision="${1#*@}" started=$SECONDS
    if [ "$repository" = "$1" ] || [ -z "$revision" ]; then
        REASON="'$1' is not of the form repository@revision"
        return 1
    fi
    # Text files and weights only. HF_XET_HIGH_PERFORMANCE is read by huggingface_hub (constants.py).
    HF_HOME="$WORKSPACE/hf" HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_TELEMETRY=1 \
        "$PY" - "$repository" "$revision" <<'PYTHON'
import os
import sys

from huggingface_hub import snapshot_download

IGNORE = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.mp4", "*.bin", "*.pt", "*.onnx", "*.gguf", "*.h5",
          "*.msgpack"]
path = snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=IGNORE)
files = sorted(os.listdir(path))
print("snapshot:", path)
print("files:", ", ".join(files))
if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
    sys.exit("the snapshot has no config.json or no .safetensors file")
PYTHON
    local code=$?
    echo "took $((SECONDS - started)) s; the cache now holds $(du -sh "$WORKSPACE/hf" 2>/dev/null | cut -f1)"
    if [ "$code" -ne 0 ]; then
        REASON="the download of $1 failed"
        return 1
    fi
}

step_env_file() {
    cat > "$WORKSPACE/env.sh" <<EOF
# Written by scripts/remote/setup_box.sh. Read it with:  . $WORKSPACE/env.sh
export HF_HOME="$WORKSPACE/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Measured on 2026-11-18 (RTX 4090, 2B, 16 x 8 attempts, token cap 260): without this setting torch kept
# up to 21.8 GiB of the 24 GiB card during a rollout while its tensors needed 10.3 GiB. With it the same
# chunks stayed at 11.1 GiB and sampled the same tokens (tested on two chunks; the speed was not measured).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$VENV/bin:\$HOME/.local/bin:\$PATH"
EOF
    cat "$WORKSPACE/env.sh"
}

# ----------------------------------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------------------------------
run_all() {
    local requirement model
    : > "$SUMMARY"
    echo "==== setup_box.sh, $(date -u +%Y-%m-%dT%H:%M:%SZ), repository $REPO ===="
    export PATH="$HOME/.local/bin:$PATH"
    export UV_CACHE_DIR="$WORKSPACE/.uv-cache"      # on the same disk as the environment
    # Packages must come from pypi.org as requirements-gpu.txt says, whatever the image has set.
    unset UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_INDEX UV_TORCH_BACKEND

    step needed "the watchdog runs" step_watchdog
    step needed "machine: GPU, driver, memory, disk" step_machine
    step needed "uv is installed" step_uv
    step needed "Python $PYTHON_VERSION environment in $VENV" step_venv
    step needed "packages of requirements-gpu.txt" step_requirements
    while IFS= read -r requirement <&3; do
        step optional "optional package $requirement" install_optional "$requirement"
    done 3< <(grep -Ev '^[[:space:]]*(#|$)' "$OPTIONAL_REQUIREMENTS")
    step needed "the pinned versions are the installed ones" step_pins
    step needed "torch sees the GPU" step_torch
    step optional "fast kernel: flash-linear-attention (fla.ops)" step_fla
    step optional "fast kernel: causal-conv1d" step_causal_conv1d
    if [ "$SKIP_MODELS" -eq 1 ]; then
        echo
        echo "model download skipped (--skip-models)"
    else
        for model in $MODELS; do
            step needed "model snapshot $model" step_model "$model"
        done
    fi
    step needed "write $WORKSPACE/env.sh" step_env_file
    finish 0
}

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --watchdog-minutes|--models)
                if [ "$#" -lt 2 ]; then
                    echo "$1 needs a value" >&2
                    exit 2
                fi
                if [ "$1" = "--models" ]; then MODELS="$2"; else WATCHDOG_MINUTES="$2"; fi
                shift 2
                ;;
            --no-watchdog) NO_WATCHDOG=1; shift ;;
            --skip-models) SKIP_MODELS=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
        esac
    done
    if ! mkdir -p "$WORKSPACE" || ! : >> "$LOG"; then
        echo "cannot write to $WORKSPACE" >&2
        exit 1
    fi
    run_all 2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
}

# Run only when this file is executed. When it is read with "source", the functions can be tried one by one.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
