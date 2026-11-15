#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${PHOTARA_BUILD_DIR:-$repo_dir/build-macos}
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" "$@"
cmake --install "$build_dir"
