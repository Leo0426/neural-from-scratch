#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-neural-from-scratch-visualizations}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
ARCH="${ARCH:-native}"
PUSH="${PUSH:-0}"

case "${ARCH}" in
  native)
    BUILT_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    docker build \
      -f "${DOCKERFILE}" \
      -t "${BUILT_IMAGE}" \
      "${BUILD_CONTEXT}"
    ;;
  amd64|amd)
    BUILT_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}-amd64"
    docker buildx build \
      --platform linux/amd64 \
      --load \
      -f "${DOCKERFILE}" \
      -t "${BUILT_IMAGE}" \
      "${BUILD_CONTEXT}"
    ;;
  arm64|arm)
    BUILT_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}-arm64"
    docker buildx build \
      --platform linux/arm64 \
      --load \
      -f "${DOCKERFILE}" \
      -t "${BUILT_IMAGE}" \
      "${BUILD_CONTEXT}"
    ;;
  all)
    if [[ "${PUSH}" != "1" ]]; then
      echo "ARCH=all builds a multi-arch manifest and requires PUSH=1." >&2
      echo "Example: ARCH=all PUSH=1 IMAGE_NAME=your-registry/neural-from-scratch-visualizations ./scripts/docker-build.sh" >&2
      exit 1
    fi

    docker buildx build \
      --platform linux/amd64,linux/arm64 \
      --push \
      -f "${DOCKERFILE}" \
      -t "${IMAGE_NAME}:${IMAGE_TAG}" \
      "${BUILD_CONTEXT}"
    BUILT_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    ;;
  *)
    echo "Unknown ARCH: ${ARCH}" >&2
    echo "Supported values: native, amd64, arm64, all" >&2
    exit 1
    ;;
esac

echo "Built ${BUILT_IMAGE} for ARCH=${ARCH}"
