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
set -e

if [ "$(basename "$0")" = "build.sh" ]; then
  print_usage() {
    cat <<EOF
Usage:
$(basename "$0")
    [ -d  | --dev      ] Development build
    [ -l  | --dyn      ] Dynamic linking
    [ -c  | --clean    ] Delete build artifacts
    [ -h  | --help     ] Print this message
    [ -v  | --llvm     ] Build LLVM only
    [ -np | --no-patch ] Do not reset and apply patches to LLVM
EOF
  }

  for arg in "$@"; do
    case $arg in
      -d|--dev)
        GC_BUILD_TYPE="Debug"
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
      -np | --no-patch)
        LLVM_NO_PATCH=1
        ;;
      -v | --llvm)
        GC_BUILD_SKIP=1
        ;;
      *)
        echo "Unknown option: $arg"
        print_usage
        exit 1
        ;;
    esac
  done
fi

: ${PROJECT_DIR:="$(cd "$(dirname "$0")/.." && pwd)"}
. "$PROJECT_DIR/scripts/utils.sh"

# Default values
: ${GC_BUILD_TYPE:=Release}
: ${GC_DYLINK:=OFF}
: ${GC_EXTERNALS_DIR:="$PROJECT_DIR/externals"}
GC_EXTERNALS_DIR=$(realpath -m "$GC_EXTERNALS_DIR")

: ${LLVM_URL:='https://github.com/llvm/llvm-project.git'}
: ${LLVM_TAG:=$(cat "$PROJECT_DIR/cmake/llvm-version.txt")}
: ${LLVM_DIR:="$GC_EXTERNALS_DIR/llvm-project"}
: ${LLVM_BUILD_TYPE:=$GC_BUILD_TYPE}
: ${LLVM_BUILD_SFX:="/$LLVM_BUILD_TYPE"}
: ${LLVM_BUILD_DIR:="$LLVM_DIR/build${LLVM_BUILD_SFX}"}
: ${LLVM_INSTALL_DIR:="$GC_EXTERNALS_DIR/llvm${LLVM_BUILD_SFX}"}
: ${MLIR_DIR:=$([ "$LLVM_BUILD_TARGET" = install ] && echo "$LLVM_INSTALL_DIR" || echo "$LLVM_BUILD_DIR")/lib/cmake/mlir}
LLVM_DIR=$(realpath -m "$LLVM_DIR")
LLVM_BUILD_DIR=$(realpath -m "$LLVM_BUILD_DIR")
LLVM_INSTALL_DIR=$(realpath -m "$LLVM_INSTALL_DIR")
MLIR_DIR=$(realpath -m "$MLIR_DIR")

: ${GC_BUILD_SFX:="/$GC_BUILD_TYPE"}
: ${GC_BUILD_DIR:="$PROJECT_DIR/build${GC_BUILD_SFX}"}
: ${GC_INSTALL_DIR:="$GC_BUILD_DIR/install"}
GC_BUILD_DIR=$(realpath -m "$GC_BUILD_DIR")
GC_INSTALL_DIR=$(realpath -m "$GC_INSTALL_DIR")

: ${MAX_JOBS:=$(nproc)}

repo_hash() {
  [ -d "$1" ] || { head -c 1000 /dev/random | sha1sum; return 0; }
  cd "$1"
  shift 1
  local rev=$(git rev-parse --short HEAD)
  local hash=$( (echo "$PROJECT_DIR/scripts/build.sh" && \
    git ls-files --modified --others --exclude-standard && \
    git diff --name-only --cached) \
    | sort -u | xargs sha1sum 2>&1 || true)
  echo "${rev}-$(echo "$hash" "$@" | sha1sum | cut -d' ' -f1)"
}

get_cmake_var() {
  [ -f "$1" ] && grep -oP "set *\(${2} \K[^)]+" "$1" || true
}

set_cmake_var() {
  if [ -f "$1" ] && grep -q "set *(${2} " "$1"; then
    sed -r -i "s/set *\(${2} .+/set(${2} ${3})/g" "$1"
  else
    echo "set(${2} ${3})" >> "$1"
  fi
}

get_llvm_hash() {
  get_cmake_var "$1/lib/cmake/llvm/LLVMConfig.cmake" LLVM_VERSION_SUFFIX
}

set_llvm_hash() {
  set_cmake_var "$1/lib/cmake/llvm/LLVMConfig.cmake" LLVM_VERSION_SUFFIX "$2"
}

get_gc_hash() {
  get_cmake_var "$1/lib/cmake/GraphCompiler/GraphCompilerConfig.cmake" GC_BUILD_STAMP
}

set_gc_hash() {
  set_cmake_var "$1/lib/cmake/GraphCompiler/GraphCompilerConfig.cmake" GC_BUILD_STAMP "$2"
}

build_llvm() {
    [ -z "$LLVM_BUILD_SKIP" ] || return 0
    local builds="$LLVM_BUILD_DIR"

    if [ "$GITHUB_ACTIONS" = 'true' ] && [ "$LLVM_BUILD_TARGET" = install ]; then
      if [ -f "$MLIR_DIR/MLIRConfig.cmake" ]; then
          echo "Using LLVM from CI cache $LLVM_INSTALL_DIR"
          return 0
      fi
      builds="$builds:$LLVM_INSTALL_DIR"
    fi

    local build_target
    [ "$LLVM_BUILD_TARGET" = install ] && build_target='--target install' || build_target=''

    if ! [ -d "$LLVM_DIR" ]; then
        mkdir -p "$LLVM_DIR"
        git init $LLVM_DIR
        cd "$LLVM_DIR"
        git remote add origin "$LLVM_URL"
        git fetch --no-tags --depth=1 origin ${LLVM_TAG}
        git checkout FETCH_HEAD
    elif ! is_true "$LLVM_NO_PATCH"; then
        cd "$LLVM_DIR"
        [ $(git -C "$LLVM_DIR" rev-parse HEAD) = "$LLVM_TAG" ] || git fetch --no-tags --depth=1 origin ${LLVM_TAG}
        git reset --hard FETCH_HEAD
        [ -z "$CLEANUP" ] || git clean -xffd;
    else
      cd "$LLVM_DIR"
    fi

    if ! is_true "$LLVM_NO_PATCH"; then
      for patch in "$PROJECT_DIR/patches/"*.patch; do
        if [ -f "$patch" ]; then
          echo "Applying patch: $patch"
          git apply --whitespace=fix "$patch"
        fi
      done
    fi

    local enable_assert=ON
    [ "$LLVM_BUILD_TYPE" = "Release" ] && enable_assert=OFF
    export LLVM_HASH=$(repo_hash "$LLVM_DIR" "$LLVM_INSTALL_DIR" $LLVM_BUILD_TYPE $GC_DYLINK)
    local hash_not_match=$(echo "$builds" | tr ':' '\n' | while read b; do
      [ "$(get_llvm_hash "$b")" = "$LLVM_HASH" ] || echo 1; done)
    [ -z "$hash_not_match" ] && echo "LLVM build is up-to-date" && return 0

    [ -z "$CLEANUP" ] || rm -rf "$LLVM_BUILD_DIR"
    mkdir -p "$LLVM_BUILD_DIR"

    echo "Configuring LLVM..."
    # Configure checks compile in random TryCompile-* dirs and always miss ccache.
    CCACHE_DISABLE=1 cmake -G Ninja -S llvm -B "$LLVM_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_ENABLE_ASSERTIONS=$enable_assert \
        -DLLVM_ENABLE_RTTI=ON \
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
        -DLLVM_USE_PERF=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=$(which python3) \
        -DMLIR_ENABLE_LEVELZERO_RUNNER=OFF \
        -DLLVM_FORCE_VC_REPOSITORY="$LLVM_URL" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
        $LLVM_CMAKE_ARGS
    cmake --build "$LLVM_BUILD_DIR" --parallel $MAX_JOBS $build_target
    
    echo "$builds" | tr ':' '\n' | while read b; do
      set_llvm_hash "$b" "$LLVM_HASH"
    done
}

build_gc() {
  build_llvm
  [ -z "$GC_BUILD_SKIP" ] || return 0

  cd "$PROJECT_DIR"
  [ -z "$CLEANUP" ] || rm -rf "$GC_BUILD_DIR"

  mkdir -p "$GC_BUILD_DIR"
  local lit_path="$LLVM_BUILD_DIR/bin/llvm-lit"
  [ -f "$lit_path" ] || lit_path=$(which lit)
  [ -z "$LLVM_HASH" ] && LLVM_HASH=$(get_llvm_hash "$MLIR_DIR/../../..")
  GC_HASH=$(repo_hash . "$MLIR_DIR" "$lit_path" $LLVM_HASH $GC_BUILD_TYPE $GC_DYLINK)
  [ "$(get_gc_hash "$GC_INSTALL_DIR")" = "$GC_HASH" ] && echo "GC build is up-to-date" && return 0
  
  CCACHE_DISABLE=1 cmake -G Ninja -S . -B "$GC_BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=$GC_BUILD_TYPE \
      -DGC_DYLINK=$GC_DYLINK \
      -DMLIR_DIR="$MLIR_DIR" \
      -DGC_EXTERNALS_DIR="$GC_EXTERNALS_DIR" \
      -DLLVM_EXTERNAL_LIT="$lit_path" \
      -DCMAKE_INSTALL_PREFIX="$GC_INSTALL_DIR" \
      $GC_CMAKE_ARGS
  cmake --build "$GC_BUILD_DIR" --parallel $MAX_JOBS --target install
  set_gc_hash "$GC_INSTALL_DIR" "$GC_HASH"
}

echo "GC_DYLINK=$GC_DYLINK"
echo "GC_BUILD_TYPE=$GC_BUILD_TYPE"
echo "LLVM_BUILD_TYPE=$LLVM_BUILD_TYPE"
[ "$(basename "$0")" != "build.sh" ] || build_gc
