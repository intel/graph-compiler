#! /bin/bash

export CASE_DIR=$(pwd)/test/benchgc/cases

FAIL=0
set -e

# misc
python3 -m benchgc --verbose 0 --driver linalg --case fill --md 0:f32 --md 1:32x4096xf32 --cmp 1:P:0:0 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case copy --md 0:1024x1024xf32 --md 1:1024x1024xbf16 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case broadcast --md 0:1024xf32 --md 1:2x32x1024xf32 --dimensions=0 --dimensions=1 || FAIL=1

# matmul
python3 -m benchgc --verbose 0 --driver linalg --case batch_matmul --md 0:16x512x64xf32 --md 1:16x64x32xf32 --md 2:16x512x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_matmul_transpose_a --md 0:16x512x64xf32 --md 1:16x512x32xf32 --md 2:16x64x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_matmul_transpose_b --md 0:16x512x64xf32 --md 1:16x128x64xf32 --md 2:16x512x128xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_matvec --md 0:16x512x64xf32 --md 1:16x64xf32 --md 2:16x512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_mmt4d --md 0:4x4x8x4x2xf32 --md 1:4x8x8x4x2xf32 --md 2:4x4x8x4x4xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_reduce_matmul --md 0:16x512x64xf32 --md 1:16x64x32xf32 --md 2:512x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_vecmat --md 0:16x64xf32 --md 1:16x64x512xf32 --md 2:16x512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case dot --md 0:4096xf32 --md 1:4096xf32 --md 2:0xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matmul --md 0:1024x512xf32 --md 1:512x512xf32 --md 2:1024x512xf32 --cast cast_signed || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_a --md 0:1024x512xf32 --md 1:1024x512xf32 --md 2:512x512xf32 --cast cast_signed || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b --md 0:1024x512xf32 --md 1:1024x512xf32 --md 2:1024x1024xf32 --cast cast_signed || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matvec --md 0:512x64xf32 --md 1:64xf32 --md 2:512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case mmt4d --md 0:4x8x4x2xf32 --md 1:8x8x4x2xf32 --md 2:4x8x4x4xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case vecmat --md 0:512xf32 --md 1:512x64xf32 --md 2:64xf32 || FAIL=1

# binary
python3 -m benchgc --verbose 0 --driver linalg --case add --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case mul --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case div --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1

# element wise
python3 -m benchgc --verbose 0 --driver linalg --case abs --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case ceil --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case erf --md 0:1024x512f32 --md 1:1024x512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case floor --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case log --md 0:4096x32xf32 --md 1:4096x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case negf --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case exp --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1

# softmax
# python3 -m benchgc --verbose 0 --driver linalg --case softmax --md 0:32x4096xf32 --md 1:32x4096xf32 --dimension 1 || FAIL=1

# mlir
# python3 -m benchgc --verbose 0 --driver mlir --case ${CASE_DIR}/llama2.mlir || FAIL=1

set +e
exit $FAIL