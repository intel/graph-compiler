#!/bin/sh
set -e

DEST_DIR="${1:-"$(dirname "$0")/../.venv"}"

python3 -m venv "$DEST_DIR" --prompt gc
. "$DEST_DIR"/bin/activate
pip install nanobind pytest pytest-xdist ruff torch torch-mlir \
    --extra-index-url https://download.pytorch.org/whl/xpu \
    --find-links https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
