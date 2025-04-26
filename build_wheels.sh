#!/usr/bin/env bash

set -eux

cd "$(dirname "$0")"

DIST_DIR="${PWD}/dist"

echo "Building jaxlib..."
bazel run  @jax_repo//jaxlib/tools:build_wheel --  \
    --output_path="${DIST_DIR}" \
    --cpu=x86_64 \
    --jaxlib_git_hash=

echo "Building jax-cuda-plugin..."
bazel run  @jax_repo//jaxlib/tools:build_gpu_kernels_wheel -- \
    --output_path="${DIST_DIR}" \
    --cpu=x86_64 \
    --enable-cuda=True \
    --platform_version=12 \
    --jaxlib_git_hash=

echo "Building jax-cuda-pjrt..."
bazel run  @jax_repo//jaxlib/tools:build_gpu_plugin_wheel -- \
    --output_path="${DIST_DIR}" \
    --cpu=x86_64 \
    --enable-cuda=True \
    --platform_version=12 \
    --jaxlib_git_hash=
