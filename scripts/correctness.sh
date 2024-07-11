#! /bin/bash

pip uninstall -y benchgc || true
pip install -y build/test/benchgc/dist/benchgc-*.whl

# need to import tools as a package 
export PYTHONPATH=$(pwd)
export MLIR_RUNNER_UTILS=${MLIR_DIR}/../../libmlir_runner_utils.so
export MLIR_C_RUNNER_UTILS=${MLIR_DIR}/../../libmlir_c_runner_utils.so
export LD_PRELOAD=/lib/x86_64-linux-gnu/libomp.so.5

python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b -i 1024x512xf32:D -i 1024x512xf32:D -o 1024x1024xf32:D --cast cast_signed