#! /bin/bash

CONDA_ENV=/tmp/benchgc

# remove old temporary conda if exists
rm -rf $CONDA_ENV

conda create --prefix $CONDA_ENV -c conda-forge python=3.11 llvm-openmp
conda activate $CONDA_ENV

pip install build/test/benchgc/dist/benchgc-*.whl

# need to import tools as a package 
export PYTHONPATH=$(pwd)
export MLIR_RUNNER_UTILS=${MLIR_DIR}/../../libmlir_runner_utils.so
export MLIR_C_RUNNER_UTILS=${MLIR_DIR}/../../libmlir_c_runner_utils.so
export LD_PRELOAD=${CONDA_ENV}/lib/libiomp5.so

python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b -i 1024x512xf32:D -i 1024x512xf32:D -o 1024x1024xf32:D --cast cast_signed