# benchgc - benchmark tool for graph compiler

## Description

Benchgc is a tool used to verify the correctness and performance of graph compiler. Benchgc accepts MLIR files based on the OneDNN graph dialect as test cases and prepares test data for them. For correctness verification, Benchgc will use PyTorch as a reference for comparison.

## Prerequisite
* python >= 3.11
* torch >= 2.2
* pybind11

## Build and install
```
# Please execute at the top level of the project

mkdir -p build
cd build

cmake .. -DMLIR_DIR=$MLIR_PATH -DGC_TEST_ENABLE=ON -DGC_ENABLE_BINDINGS_PYTHON=ON -DGC_BENCH_ENABLE=ON
make -j benchgc

python -m pip install test/benchgc/dist/benchgc-*.whl

```

## Synopsis
```
python -m benchgc [OPTIONS] [--mlir [FILE] --entry [FUNCTION] | --json [FILE]]
```
## Flags
```
--mlir [FILE]
    Required if --json is not provided. A mlir file describing the case
--entry [FUNCTION]
    Required if --mlir is provided. A function name in the mlir file describing the entry
--json [FILE]
    Required if --mlir is not provided. A json file describing the case.
--seed [INT]
    Optional and default is 0. A random seed value to generate data filling. It is also used in reproducing the issue.
--verbose [INT]
    Optional, default is 0 with no verbose. An integer value describes the verbose level.
```
