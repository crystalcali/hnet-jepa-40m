#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="hnet-jepa-spark:25.11-v1"

usage() {
    cat <<'HELP'
bash scripts/spark.sh build                  Build the pinned NVIDIA PyTorch environment
bash scripts/spark.sh smoke                  Build if needed and verify CUDA and Mamba-2
bash scripts/spark.sh shell                  Open a project shell inside the environment
bash scripts/spark.sh dashboard [--port N]   Serve live metrics on the Spark loopback interface
bash scripts/spark.sh run COMMAND [ARGS...]  Run a command in the project environment
HELP
}

action="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$action" in
    help|-h|--help) usage; exit 0 ;;
    build|smoke|shell|dashboard|run) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ "$action" == run && $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
if ! command -v docker >/dev/null; then
    printf '%s\n' "Docker with NVIDIA GPU support is required on DGX Spark." >&2
    exit 1
fi
if ! docker info --format '{{.ServerVersion}}' >/dev/null; then
    printf '%s\n' "Cannot reach Docker. Use the account configured for Docker on Spark." >&2
    exit 1
fi

build() {
    docker build --tag "$image" --file "$root/docker/spark.Dockerfile" "$root"
}
if [[ "$action" == build ]]; then
    build
    exit
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    build
fi

case "$action" in
    smoke) set -- python3 -m scripts.check_env --cuda-smoke "$@" ;;
    shell) set -- bash "$@" ;;
    dashboard) set -- python3 -m dashboard.server --runs runs "$@" ;;
esac

mkdir -p "$root/.spark/home"
docker_args=(run --rm --init --network host --shm-size 2g
    --user "$(id -u):$(id -g)"
    --env HOME=/workspace/.spark/home
    --mount "type=bind,source=$root,target=/workspace"
    --workdir /workspace)
if [[ "$action" != dashboard ]]; then
    docker_args+=(--gpus all)
fi
if [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
fi

exec docker "${docker_args[@]}" "$image" "$@"
