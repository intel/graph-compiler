#! /bin/bash

# need to import tools as a package 
export PYTHONPATH=$(pwd)
export CASE_DIR=$(pwd)/test/benchgc/cases


FAIL=0
set -e

# misc
python3 -m benchgc --verbose 0 --driver linalg --case fill -i f32:U:1:128 -o 32x4096xf32:P:0:0 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case copy -i 1024x1024xf32:U:-128:127 -o 1024x1024xbf16:P:0.0078125:30.0 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case broadcast -i 1024xf32:U:-128:127 -o 2x32x1024xf32:P:0:30.0 --dimensions=0 --dimensions=1 || FAIL=1

# matmul
python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b -i 1024x512xf32:D -i 1024x512xf32:D -o 1024x1024xf32:D --cast cast_signed || FAIL=1

# binary
python3 -m benchgc --verbose 0 --driver linalg --case add -i 1x32x4096xf32:D -i 1x32x4096xf32:D -o 1x32x4096xf32:D || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case mul -i 1x32x4096xf32:D -i 1x32x4096xf32:D -o 1x32x4096xf32:D || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case div -i 1x32x4096xf32:D -i 1x32x4096xf32:D -o 1x32x4096xf32:D || FAIL=1

# element wise
python3 -m benchgc --verbose 0 --driver linalg --case negf -i 32x4096xf32:D -o 32x4096xf32:D || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case exp -i 32x4096xf32:D -o 32x4096xf32:D || FAIL=1

# mlir
# python3 -m benchgc --verbose 0 --driver mlir --case ${CASE_DIR}/llama2.mlir \
#     -i 1x32x4096xbf16:N:0:1 \
#     -i 4096x4096xbf16:N:0:1 \
#     -i 1x32x4096xbf16:N:0:1 \
#     -i 1xf32:N:0:1 \
#     -i 4096xbf16:N:0:1 \
#     -i 11008x4096xbf16:N:0:1 \
#     -i 11008x4096xbf16:N:0:1 \
#     -i 4096x11008xbf16:N:0:1 \
#     -i 1xf32:N:0:1 \
#     -i 4096xbf16:N:0:1 \
#     -o 1x32x4096xbf16:P:0.0078125:30.0 \
#     -o 1x32x4096xbf16:P:0.0078125:30.0 || FAIL=1

set +e
exit $FAIL