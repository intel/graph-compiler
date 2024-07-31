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

### --md index:SHAPExTYPE
* Describe the shape and data type for argument
* Not available when driver=mlir
* index means the order of argument, including both inputs and outs
* use prefix `0x` (e.g. `0xbf16`) to represent 0d memref or tensor input
* use data type directly (e.g.`f32`) to represent a normal scalar

```
# %arg0             -> index = 0
# tensor<2x2x2xf32> -> index = 1

module {
  func.func @entry(%arg0: f32) -> tensor<2x2x2xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<2x2x2xf32>
    %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    return %1 : tensor<2x2x2xf32>
  }
}
```

### --fill index:fill_type:[:fill_parameter]*
* If not set, benchgc will assign a default method for the argument

| description | fill_type | fill_parameter |
|-------------|-----------|-----------|
| Zero | Z | |
| Normal | N | mean, std |
| Poisson | P | lambda |
| Binomial | B | n, p |
| Uniform | U | a, b |
| Pytorch tensor dump | F | dump filename |
| Benchdnn driver | D | driver_name[:driver filling parameter] |

#### Benchdnn driver filling

| driver_name | drvier filling parameter |
|-------------|--------------------------|
| binary | src0 dtype, src1 dtype, dst dtype |
| matmul | src dtype, wei dtype, dst dtype, amplifier |
| eltwise | algorithm, alpha, beta (please check https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html)  |

### --cmp index:cmp_type:[:cmp_parameter]i*
* If not set, benchgc will assign a default method for the argument

| description | cmp_type | cmp_parameter |
|-------------|-----------|-----------|
| P2P check | P | threshold, zero_percent(mistrust check) |
| Norm check | N | threshold |
| Benchdnn driver | D | driver_name |


### -i shapexdtype:fill_type[:fill_parameter]*
* this flag is order sensitive
* first setting will be the arg0, the second setting will be the arg1, ..
* set the variable shape and dtype in single op test
* set the data filling strategy
* use prefix `0x` (e.g. `0xbf16`) to represent 0d memref or tensor input
* use `f32` to represent a normal scalar

## Example
```
python3 -m benchgc --verbose 0 --driver linalg --case add --md 0:4x5x6xf32 --md 1:4x5x6xf32 --md 2:4x5x6xf32
```