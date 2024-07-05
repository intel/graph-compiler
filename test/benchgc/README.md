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
python -m benchgc [OPTIONS] --driver [DRIVER] --case [CASE]
```
## Flags
###  --driver [str]
* linalg: test the single op in linalg dialect
* tensor: test the single op in tensor dialect
* mlir: upload a mlir file and run
* pattern: predefined pattern test such as mlp

### --case [str]
* if driver=mlir, please provide a mlir file here to test
* if driver=pattern, please provide the pre-defined pattern name, such as mlp here
* if driver is a dialect name, please provide the detail op name to start a single op test

### --seed [int]
* set the seed to generate the test data and reprodce the test

### --verbose [int]
* set the verbose level

### -i name:dtype:shape:fill_type[:fill_parameter]*
* set or rewrite the variable {shape, dtype} in mlir or single op test
* set the data filling strategy
* use the variable name defined in your mlir case if driver = mlir

* fill_type & fill_parameter setting
    | description | fill_type | fill_parameter |
    |-------------|-----------|-----------|
    | Normal | N | mean, std |
    | Poisson | P | lambda |
    | Binomial | B | n, p |
    | Uniform | U | a, b |
    | Pytorch tensor dump | F | dump filename |
    | Benchdnn driver | D | no parameter; only available for single op test; generate the parameter automatically |
    | Benchdnn driver | D | driver_name[:driver filling parameter] |

* Benchdnn drvier filling parameter
    | driver_name | drvier filling parameter |
    |-------------|--------------------------|
    | binary | src0 dtype, src1 dtype, dst dtype |
    | matmul | src dtype, wei dtype, dst dtype, amplifier |
    | conv | src dtype, wei dtype, dst dtype, amplifier |

### -o name:dtype:shape:check_type[:check_parameter]*
* set or rewrite the variable {shape, dtype} in mlir or single op test
* set the data compare & check strategy


## Example
```
python3 -m benchgc --verbose 4 --driver linalg --case add -i %arg0:f32:4x5x6:N:0:1 -i %arg1:f32:4x5x6:N:5:2 -o %1:f32:4x5x6::
```