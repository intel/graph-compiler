#!/bin/sh
################################################################################
# Copyright (C) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################

if [ "$(basename "$0")" = "ov-build.sh" ]; then
  for arg in "$@"; do
    case $arg in
      -c|--clean)
        CLEANUP=1
        ;;
      -r|--reset)
        RESET=1
        ;;
      *)
        echo "Unknown option: $arg"
        exit 1
        ;;
    esac
  done
fi

: ${PROJECT_DIR:="$(cd "$(dirname "$0")/.." && pwd)"}
. "$PROJECT_DIR/scripts/build.sh"

: ${OV_URL:='https://github.com/openvinotoolkit/openvino.git'}
: ${OV_TAG:='e357b2c0ac8a9ee461fbd09655c16beebd21abed'}
: ${OV_DIR:="$GC_EXTERNALS_DIR/openvino"}
: ${OV_BUILD_TYPE:=$GC_BUILD_TYPE}
: ${OV_BUILD_SFX:="/$OV_BUILD_TYPE"}
: ${OV_BUILD_DIR:="$OV_DIR/build${OV_BUILD_SFX}"}
: ${OV_INSTALL_DIR:="$OV_BUILD_DIR/install"}
OV_BUILD_DIR=$(realpath -m "$OV_BUILD_DIR")
OV_INSTALL_DIR=$(realpath -m "$OV_INSTALL_DIR")

build_ov() {
    build_gc

    export GIT_LFS_SKIP_SMUDGE=1
    if ! [ -d "$OV_DIR/.git" ]; then
        git init "$OV_DIR"
        git -C "$OV_DIR" fetch --depth 1 "$OV_URL" "$OV_TAG"
        git -C "$OV_DIR" checkout FETCH_HEAD
    elif [ -n "$RESET" ]; then
        local ref=HEAD
        if [ "$(git -C "$OV_DIR" rev-parse HEAD)" != "$OV_TAG" ]; then
            git -C "$OV_DIR" fetch --depth 1 "$OV_URL" "$OV_TAG"
            ref=FETCH_HEAD
        fi
        git -C "$OV_DIR" reset --hard "$ref"
        [ -z "$CLEANUP" ] || git -C "$OV_DIR" clean -xffd
    fi
    git -C "$OV_DIR" submodule update --init --recursive --depth 1 --jobs "$MAX_JOBS"
    [ -z "$CLEANUP" ] || rm -rf "$OV_BUILD_DIR" "$OV_INSTALL_DIR"

    local llvm_dir=$(realpath -m "$MLIR_DIR/../../../lib/cmake/llvm")
    : "${LLVM_HASH:=$(get_llvm_hash "$llvm_dir")}"
    : "${GC_HASH:=$(get_gc_hash "$GC_INSTALL_DIR")}"
    OV_HASH=$(repo_hash "$OV_DIR" "$LLVM_HASH" "$GC_HASH" \
        "$(sha1sum "$PROJECT_DIR/scripts/ov-build.sh")")
    local stamp_file="$OV_BUILD_DIR/ov_gc_build.stamp"
    [ "$(cat "$stamp_file" 2>/dev/null)" = "$OV_HASH" ] && echo "OpenVINO build is up-to-date" && return 0

    CCACHE_DISABLE=1 cmake -G Ninja -S "$OV_DIR" -B "$OV_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=$OV_BUILD_TYPE \
        -DMLIR_DIR="$MLIR_DIR" \
        -DLLVM_DIR="$llvm_dir" \
        -DENABLE_MLIR_FOR_GPU=ON \
        -DENABLE_DEBUG_CAPS=ON \
        -DLLVM_DYLINK=$GC_DYLINK \
        -DENABLE_INTEL_GPU=ON \
        -DENABLE_TESTS=ON \
        -DENABLE_ONEDNN_FOR_GPU=ON \
        -DENABLE_INTEL_CPU=OFF \
        -DENABLE_INTEL_NPU=OFF \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
        -DENABLE_NCC_STYLE=OFF \
        -DENABLE_STRICT_DEPENDENCIES=OFF \
        -DENABLE_SYSTEM_OPENCL=ON \
        -DCPACK_GENERATOR=TGZ \
        -DCMAKE_CXX_FLAGS="-DOV_GPU_OPENCL_HPP_HAS_UUID -DOV_GPU_OPENCL_HPP_HAS_BUS_INFO" \
        -DGraphCompiler_DIR="$GC_INSTALL_DIR/lib/cmake/GraphCompiler" \
        -DOpenCL_HPP_INCLUDE_DIR="$OV_DIR/thirdparty/ocl/clhpp_headers/include" \
        -DOpenCL_HPP="$OV_DIR/thirdparty/ocl/clhpp_headers/include/CL/opencl.hpp" \
        -DGPU_RT_TYPE=ZE
    cmake --build "$OV_BUILD_DIR" --parallel "$MAX_JOBS"

    local bin_src="$OV_DIR/bin/intel64/$OV_BUILD_TYPE"
    local bin_dir="$OV_INSTALL_DIR/bin/intel64/$OV_BUILD_TYPE"
    local ovc=tools/ovc/openvino/tools/ovc
    mkdir -p "$bin_dir" "$OV_INSTALL_DIR/$ovc/moc_frontend" \
        "$OV_INSTALL_DIR/src/plugins/intel_gpu/tests/functional/mlir_op"
    rm -f "$bin_dir"/libze_loader.so*
    cp -r "$PROJECT_DIR/scripts" "$PROJECT_DIR/python/bench" "$OV_INSTALL_DIR"
    cp -r "$bin_src/ov_gpu_func_tests" "$bin_src/python" "$bin_dir/"
    find "$bin_src" -maxdepth 1 -name '*.so*' ! -name 'libze_loader.so*' -exec cp -t "$bin_dir/" {} +
    cp "$OV_DIR/$ovc/"*.py "$OV_INSTALL_DIR/$ovc"
    cp "$OV_DIR/$ovc/moc_frontend/"*.py "$OV_INSTALL_DIR/$ovc/moc_frontend"
    echo "$OV_HASH" > "$stamp_file"
}

echo "OV_BUILD_TYPE=$OV_BUILD_TYPE"
[ "$(basename "$0")" != "ov-build.sh" ] || build_ov
