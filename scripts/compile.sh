#!/bin/sh
################################################################################
# Copyright (C) 2025 Intel Corporation
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

set -e

# Default values
: ${GC_BUILD_TYPE:=RelWithDebInfo}
: ${GC_DYLINK:=OFF}

print_usage() {
    cat <<EOF
Usage:
$(basename "$0")
    [ -d | --dev     ] Development build
    [ -r | --release ] Release build (default: RelWithDebInfo)
    [ -l | --dyn     ] Dynamical linking, requires rebuild of LLVM, activates 'dev' option
    [ -c | --clean   ] Delete the build artifacts from the previous build
    [ -s | --suffix  ] Build dir suffix
    [ -h | --help    ] Print this message
EOF
}

for arg in "$@"; do
  case $arg in
    -d|--dev)
      GC_BUILD_TYPE="Debug"
      ;;
    -r|--release)
      GC_BUILD_TYPE="Release"
      ;;
    -c|--clean)
      CLEANUP=1
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    -l | --dyn)
      GC_DYLINK=ON
      ;;
    *)
      echo "Unknown option: $arg"
      print_usage
      exit 1
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
: ${EXTERNALS_DIR:="$PROJECT_DIR/externals"}
: ${MAX_JOBS:=$(($(nproc) - 2))}
[ $MAX_JOBS -gt 0 ] || MAX_JOBS=1

build_llvm() {
    local llvm_hash=$(cat "$PROJECT_DIR/cmake/llvm-version.txt")
    local llvm_dir="$EXTERNALS_DIR/llvm-project"
    LLVM_BUILD_DIR="$llvm_dir/build"
    MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"

    if ! [ -d "$llvm_dir" ]; then
        mkdir -p "$EXTERNALS_DIR"
        cd "$EXTERNALS_DIR"
        git clone https://github.com/llvm/llvm-project.git
        cd "$llvm_dir"
    else
        cd "$llvm_dir"
        git fetch --all
        git checkout -- .
    fi

    git checkout ${llvm_hash}
    
    [ -z "$CLEANUP" ] || rm -rf "$LLVM_BUILD_DIR"
    mkdir -p "$LLVM_BUILD_DIR"

    echo "Configuring LLVM..."
    cmake -G Ninja llvm -B "$LLVM_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=$GC_BUILD_TYPE \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
        -DLLVM_BUILD_LLVM_DYLIB=$GC_DYLINK \
        -DLLVM_LINK_LLVM_DYLIB=$GC_DYLINK \
        -DLLVM_INCLUDE_RUNTIMES=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=ON \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_INSTALL_GTEST=ON \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=$(which python3) \
        -DMLIR_ENABLE_LEVELZERO_RUNNER=ON
    cmake --build "$LLVM_BUILD_DIR" --parallel $MAX_JOBS
}

echo "GC_BUILD_TYPE=$GC_BUILD_TYPE"
echo "GC_DYLINK=$GC_DYLINK"

build_llvm

cd "$PROJECT_DIR"
[ -z "$CLEANUP" ] || rm -rf "$BUILD_DIR"

cmake -S . --preset gc \
    -DCMAKE_BUILD_TYPE=$GC_BUILD_TYPE \
    -DGC_DYLINK=$GC_DYLINK
cmake --build "$BUILD_DIR" --parallel $MAX_JOBS
