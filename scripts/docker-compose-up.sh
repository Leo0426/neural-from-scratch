#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-neural-from-scratch-visualizations}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
VIS_PORT="${VIS_PORT:-8080}"
ARCH="${ARCH:-native}"
DETACH="${DETACH:-0}"
BUILD="${BUILD:-1}"

compose_files=(-f docker-compose.yml)

case "${ARCH}" in
  native)
    IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
    ;;
  amd64|amd)
    IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}-amd64"
    compose_files+=(-f docker-compose.amd64.yml)
    ;;
  arm64|arm)
    IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}-arm64"
    compose_files+=(-f docker-compose.arm64.yml)
    ;;
  *)
    echo "Unknown ARCH: ${ARCH}" >&2
    echo "Supported values: native, amd64, arm64" >&2
    exit 1
    ;;
esac

export IMAGE_REF
export VIS_PORT

up_args=(up)
if [[ "${BUILD}" == "1" ]]; then
  up_args+=(--build)
fi
if [[ "${DETACH}" == "1" ]]; then
  up_args+=(-d)
fi

echo "Visualizations are available at http://localhost:${VIS_PORT}/"
docker compose "${compose_files[@]}" "${up_args[@]}"
