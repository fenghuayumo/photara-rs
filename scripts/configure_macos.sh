#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Use Blender precompiled dependencies in lib/macos_arm64, or Homebrew packages.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$(uname -s)" != Darwin ]; then
  echo "This build requires macOS." >&2
  exit 1
fi
if command -v brew >/dev/null 2>&1; then
  brew_prefix=$(brew --prefix)
  export CMAKE_PREFIX_PATH="$brew_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi
cmake -S "$repo_dir" -B "${PHOTARA_BUILD_DIR:-$repo_dir/build-macos}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DWITH_CYCLES_DEVICE_METAL=ON \
  -DWITH_CYCLES_DEVICE_CUDA=OFF \
  -DWITH_CYCLES_DEVICE_OPTIX=OFF \
  -DWITH_CYCLES_OSL=OFF \
  -DWITH_CYCLES_USD=OFF \
  -DWITH_CYCLES_HYDRA_RENDER_DELEGATE=OFF \
  -DWITH_CYCLES_ALEMBIC=OFF \
  -DWITH_CYCLES_EMBREE=OFF \
  -DWITH_CYCLES_OPENSUBDIV=OFF \
  -DWITH_CYCLES_OPENVDB=OFF \
  -DWITH_CYCLES_NANOVDB=OFF \
  -DPHOTARA_BUILD_C_API=ON \
  "$@"
