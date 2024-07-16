#! /bin/bash


# need to import tools as a package 
export PYTHONPATH=$(pwd)
export MLIR_RUNNER_UTILS=../install/llvm/lib/libmlir_runner_utils.so
export MLIR_C_RUNNER_UTILS=../install/llvm/lib/libmlir_c_runner_utils.so
export LD_PRELOAD=/lib/x86_64-linux-gnu/libomp5.so

python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b -i 1024x512xf32:D -i 1024x512xf32:D -o 1024x1024xf32:D --cast cast_signed