#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-neural-from-scratch-visualizations}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-neural-from-scratch-visualizations}"
VIS_PORT="${VIS_PORT:-8080}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Visualizations are available at http://localhost:${VIS_PORT}/"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  -p "${VIS_PORT}:80" \
  "${IMAGE_NAME}:${IMAGE_TAG}"
