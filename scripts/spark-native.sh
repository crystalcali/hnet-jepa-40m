#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root="$PWD"
env_dir="$root/.spark/venv-cu130"
py="$env_dir/bin/python"
torch_version='2.9.1+cu130'

usage() {
    cat <<'HELP'
bash scripts/spark-native.sh smoke                  Install and verify CUDA PyTorch and Mamba-2
bash scripts/spark-native.sh dashboard [--port N]   Serve live metrics on the Spark loopback interface
bash scripts/spark-native.sh run COMMAND [ARGS...]  Run with the project environment active
HELP
}

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

toolkit() {
    local candidate compiler
    local candidates=()
    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("$CUDA_HOME")
    fi
    candidates+=(/usr/local/cuda-13.0 /usr/local/cuda)
    if compiler="$(command -v nvcc)"; then
        candidates+=("$(dirname -- "$(dirname -- "$(readlink -f -- "$compiler")")")")
    fi
    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate/bin/nvcc" ]] || continue
        if "$candidate/bin/nvcc" --version | grep 'release 13\.0,' >/dev/null; then
            export CUDA_HOME="$candidate"
            return
        fi
    done
    fail "CUDA Toolkit 13.0 (nvcc) was not found. Check /usr/local/cuda-13.0/bin/nvcc or set CUDA_HOME to the CUDA 13.0 toolkit directory."
}

setup() {
    [[ "$(uname -m)" == aarch64 ]] || fail "This installer targets ARM64 DGX Spark."
    if ! command -v python3 >/dev/null; then
        fail "Install Python 3.12 and python3.12-venv on DGX OS."
    fi
    if ! python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
        fail "The pinned Spark environment uses Python 3.12."
    fi
    if ! command -v g++ >/dev/null; then
        fail "A C++ compiler is required to build the CUDA extensions. Install build-essential on DGX OS."
    fi
    toolkit
    if [[ ! -x "$py" ]] || ! "$py" -m pip --version >/dev/null 2>&1; then
        python3 -m venv "$env_dir" || fail "Could not create the venv. On DGX OS install python3.12-venv and retry."
    fi
    "$py" -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==$torch_version"
    "$py" -c 'import torch; print("PyTorch", torch.__version__, "GPU", torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()'
    "$py" -m pip install -r requirements-data.txt -r requirements-spark-build.txt
    MAX_JOBS="${MAX_JOBS:-4}" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
        "$py" -m pip install --no-build-isolation --no-deps -r requirements-spark-cuda.txt
}

action="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$action" in
    smoke)
        setup
        exec "$py" -m scripts.check_env --cuda-smoke "$@"
        ;;
    dashboard)
        [[ -x "$py" ]] || fail "Run bash scripts/spark-native.sh smoke first."
        exec "$py" -m dashboard.server --runs runs "$@"
        ;;
    run)
        [[ $# -gt 0 ]] || { usage >&2; exit 2; }
        [[ -x "$py" ]] || fail "Run bash scripts/spark-native.sh smoke first."
        export PATH="$env_dir/bin:$PATH"
        export VIRTUAL_ENV="$env_dir"
        exec "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
