#! /bin/bash

export CASE_DIR=$(pwd)/test/benchgc/cases

FAIL=0
set -e

# misc
python3 -m benchgc --verbose 0 --driver linalg --case fill --md 0:f32 --md 1:32x4096xf32 --cmp 1:P:0:0 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case copy --md 0:1024x1024xf32 --md 1:1024x1024xbf16 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case broadcast --md 0:1024xf32 --md 1:2x32x1024xf32 --dimensions=0 --dimensions=1 || FAIL=1

# matmul
python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b --md 0:1024x512xf32 --md 1:1024x512xf32 --md 2:1024x1024xf32 --cast cast_signed || FAIL=1

# binary
python3 -m benchgc --verbose 0 --driver linalg --case add --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case mul --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case div --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1

# element wise
python3 -m benchgc --verbose 0 --driver linalg --case negf --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case exp --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1

# mlir
# python3 -m benchgc --verbose 0 --driver mlir --case ${CASE_DIR}/llama2.mlir || FAIL=1

set +e
exit $FAIL