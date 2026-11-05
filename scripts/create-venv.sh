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

: ${PROJECT_DIR:="$(cd "$(dirname "$0")/.." && pwd)"}
. "$PROJECT_DIR/scripts/utils.sh"
: ${LLVM_TAG:="$(head -n1 "$PROJECT_DIR/cmake/llvm-version.txt")"}
: ${NANOBIND:="$(curl -fsSL "https://raw.githubusercontent.com/llvm/llvm-project/${LLVM_TAG}/mlir/python/requirements.txt" | tr -d ' ' | grep -E '^nanobind')"}
: "${NANOBIND:?Failed to get nanobind version for LLVM $LLVM_TAG}"

create() {
    # Minimal required packages
    local packages="cmake setuptools $NANOBIND"
    [ "${1:-}" = '-m' ] || packages="$packages lit pre-commit pytest pytest-xdist ruff torch-mlir"
    local dest_dir="$PROJECT_DIR/.venv"

    "$PYTHON" -m venv "$dest_dir" --prompt gc
    . "$dest_dir/$VENV_BIN/activate"
    # XPU index only: with --extra-index-url pip silently falls back to the PyPI CPU torch if the index is unreachable.
    [ "${1:-}" = '-m' ] || pip install torch --index-url https://download.pytorch.org/whl/xpu
    pip install $packages \
        --find-links https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
    [ "${1:-}" = '-m' ] || [ -n "${CI:-}" ] || (cd "$PROJECT_DIR" && pre-commit install --install-hooks)
}

[ "$(basename "$0")" != "create-venv.sh" ] || create "$@"
