# benchgc - benchmark tool for graph compiler

## Description

Benchgc is a tool used to verify the correctness and performance of graph compiler. Benchgc accepts MLIR files as test cases and prepares test data for them. For correctness verification, Benchgc will use PyTorch as a reference for comparison.

## Prerequisite
* python >= 3.10
* torch >= 2.2
* Enable mlir python binding, Refer to [`python/README.md`](../../python/README.md) for detail

## Build 
There are two ways for using benchgc

* Build `.whl` and install benchgc
```
# Please execute at the top level of the project

mkdir build && cd build
cmake .. -DMLIR_DIR=$MLIR_PATH -DGC_TEST_ENABLE=ON -DGC_ENABLE_BINDINGS_PYTHON=ON -DGC_BENCH_ENABLE=ON
make -j benchgc
python -m pip install test/benchgc/dist/benchgc-*.whl

```

* Run benchgc from source code

```
# Please execute at the top level of the project

mkdir build && cd build
cmake .. -DMLIR_DIR=$MLIR_PATH -DGC_TEST_ENABLE=ON -DGC_ENABLE_BINDINGS_PYTHON=ON -DGC_BENCH_ENABLE=ON
make -j GcPythonModules
export PYTHONPATH=$(pwd)/python_packages/gc_mlir_core/:$(pwd)/../test/benchgc/src/
```

## Synopsis
```
python -m benchgc [OPTIONS] --mode [MODE] --driver [DRIVER] --case [CASE]
```
## Common Options
### --mode [str]
* C : correctness testing (by default)
* P : performance testing

###  --driver [str]
* linalg: test the single op in linalg dialect
* mlir: upload a mlir file and run
* pattern: predefined pattern test such as mlp

### --case [str]
* if driver=mlir, please provide a mlir file here to test
* if driver=pattern, please provide the pre-defined pattern name, such as mlp here
* if driver is a dialect name, please provide the detail op name to start a single op test

### --entry [str]
* default : "entry"
* the entry name of the kernel of input mlir or generated mlir

### --seed [int]
* set the seed to generate the test data and reprodce the test

### --verbose [int]
* set the verbose level, default : 0
* 0 : NO_VERBOSE
* 1 : MODULE_VERBOSE, print the module will be executed
* 2 : ARG_VERBOSE, + print arg information
* 3 : COMPARE_VERBOSE, + print threshold for comparison
* 4 : ERROR_OUTPUT_VERBOSE, + print all error data points if failed
* 5 : OUTPUT_VERBOSE, + print all result including passed tensor
* 6 : INPUT_VERBOSE, + print input torch tensors

### --ir_printing (action=store_true)
* Print the ir during the pass-pipeline 

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
| Integer | I | a, b |
| Pytorch tensor dump | F | dump filename |
| Benchdnn driver | D | driver_name[:driver filling parameter]* |

#### Benchdnn driver filling

| driver_name | driver filling parameter |
|-------------|--------------------------|
| binary | src0/src1:src0 dtype:src1 dtype:dst dtype |
| conv | src/wei:src dtype:wei dtype:dst dtype:amplifier |
| eltwise | algorithm: alpha: beta (please check https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html)  |
| matmul | src/wei:src dtype:wei dtype:dst dtype:amplifier |
| pool | not required |

### --cmp index:cmp_type:[:cmp_parameter]*
* If not set, benchgc will assign a default method for the argument

| description | cmp_type | cmp_parameter |
|-------------|-----------|-----------|
| P2P check | P | threshold, zero_percent(mistrust check) |
| Norm check | N | threshold |
| Benchdnn driver | D | driver_name:dtype:case |

## Bench Options
### --bench_kind [str]
* py : use the MLIR Python API to invoke the kernel and use Python to calculate the time cost
* wrapper : modify MLIR by wrapping the kernel into a new method and calling the `nanoTime()` method before and after calling the kernel. Finally, calculate the difference as the time cost

### --warm_up [int]
* warm-up times of the execution

### --repeat [int]
* repeat times of the execution

## Pattern Options
Each pattern has its own unique options.
### mlp
* `--batch_size`: the input
* `--hidden_size_list`: hidden_sizes of mlp, example: 32x16x64
* `--has_bias`: if the matmul op has bias, example: 1x0
* `--act_type`: choices=["noop", "relu"]
* `--dtype`: choices=["bf16", "f32"]

## Example
### Correctness testing example
```
# single add op test 
# using the same data filling / compare strategy as the benchdnn primitive driver if not set
python3 -m benchgc --verbose 6 --driver linalg --case add --md 0:4x5xf32 --md 1:4x5xf32 --md 2:4x5xf32

arg0 shape: [4, 5] dtype: f32 fill_type: D fill_param: ['binary', 'src0', 'f32', 'f32', 'f32'] cmp_type: D cmp_param: ['binary', 'f32', 'add']
arg1 shape: [4, 5] dtype: f32 fill_type: D fill_param: ['binary', 'src1', 'f32', 'f32', 'f32'] cmp_type: D cmp_param: ['binary', 'f32', 'add']
arg2 shape: [4, 5] dtype: f32 fill_type: Z fill_param: [] cmp_type: D cmp_param: ['binary', 'f32', 'add']
module {
  func.func @entry(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %2 = linalg.add ins(%arg0, %arg1 : tensor<4x5xf32>, tensor<4x5xf32>) outs(%1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    return %2 : tensor<4x5xf32>
  }
}

fill arg0: 
tensor([[ -5.0000,  10.0000,   3.7500,  -2.5000,  -8.7500],
        [  6.2500,   0.0000,  -6.2500,   8.7500,   2.5000],
        [ -3.7500, -10.0000,   5.0000,  -1.2500,  -7.5000],
        [  7.5000,   1.2500,  -5.0000,  10.0000,   3.7500]])
fill arg1: 
tensor([[  1.2500,  -5.0000,  10.0000,   3.7500,  -2.5000],
        [ -8.7500,   6.2500,   1.0000,  -6.2500,   8.7500],
        [  2.5000,  -3.7500, -10.0000,   5.0000,  -1.2500],
        [ -7.5000,   7.5000,   1.2500,  -5.0000,  10.0000]])
fill arg2: 
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
p2p check: threshold: 0.0000001
              (0, 0): ref:   -5.0000000 res:   -5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 1): ref:   10.0000000 res:   10.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 2): ref:    3.7500000 res:    3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 3): ref:   -2.5000000 res:   -2.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 4): ref:   -8.7500000 res:   -8.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 0): ref:    6.2500000 res:    6.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 1): ref:    0.0000000 res:    0.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 2): ref:   -6.2500000 res:   -6.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 3): ref:    8.7500000 res:    8.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 4): ref:    2.5000000 res:    2.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 0): ref:   -3.7500000 res:   -3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 1): ref:  -10.0000000 res:  -10.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 2): ref:    5.0000000 res:    5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 3): ref:   -1.2500000 res:   -1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 4): ref:   -7.5000000 res:   -7.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 0): ref:    7.5000000 res:    7.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 1): ref:    1.2500000 res:    1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 2): ref:   -5.0000000 res:   -5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 3): ref:   10.0000000 res:   10.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 4): ref:    3.7500000 res:    3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
p2p check: threshold: 0.0000001
              (0, 0): ref:    1.2500000 res:    1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 1): ref:   -5.0000000 res:   -5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 2): ref:   10.0000000 res:   10.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 3): ref:    3.7500000 res:    3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 4): ref:   -2.5000000 res:   -2.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 0): ref:   -8.7500000 res:   -8.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 1): ref:    6.2500000 res:    6.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 2): ref:    1.0000000 res:    1.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 3): ref:   -6.2500000 res:   -6.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 4): ref:    8.7500000 res:    8.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 0): ref:    2.5000000 res:    2.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 1): ref:   -3.7500000 res:   -3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 2): ref:  -10.0000000 res:  -10.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 3): ref:    5.0000000 res:    5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 4): ref:   -1.2500000 res:   -1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 0): ref:   -7.5000000 res:   -7.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 1): ref:    7.5000000 res:    7.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 2): ref:    1.2500000 res:    1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 3): ref:   -5.0000000 res:   -5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 4): ref:   10.0000000 res:   10.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
p2p check: threshold: 0.0000001
              (0, 0): ref:   -3.7500000 res:   -3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 1): ref:    5.0000000 res:    5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 2): ref:   13.7500000 res:   13.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 3): ref:    1.2500000 res:    1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 4): ref:  -11.2500000 res:  -11.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 0): ref:   -2.5000000 res:   -2.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 1): ref:    6.2500000 res:    6.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 2): ref:   -5.2500000 res:   -5.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 3): ref:    2.5000000 res:    2.5000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 4): ref:   11.2500000 res:   11.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 0): ref:   -1.2500000 res:   -1.2500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 1): ref:  -13.7500000 res:  -13.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 2): ref:   -5.0000000 res:   -5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 3): ref:    3.7500000 res:    3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (2, 4): ref:   -8.7500000 res:   -8.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 0): ref:    0.0000000 res:    0.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 1): ref:    8.7500000 res:    8.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 2): ref:   -3.7500000 res:   -3.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 3): ref:    5.0000000 res:    5.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (3, 4): ref:   13.7500000 res:   13.7500000 abs_diff:    0.0000000 rel_diff:    0.0000000
PASSED: linalg.add
```

```
# set the arg0 filling follows a distribution N(0, 5)
# set the arg1 filling follows a uniform integer filling [-3, 3]
# use P2P compare strategy on arg2 with threshold = 0 & mistrust rate = 100.0% 
# zero threshold will fail the case here

python3 -m benchgc --verbose 6 --driver linalg --case matmul_transpose_b --md 0:2x5xf32 --md 1:2x5xf32 --md 2:2x2xf32 --fill 0:N:0:5 --fill 1:I:-3:3 --cmp 2:P:0:100
arg0 shape: [2, 5] dtype: f32 fill_type: N fill_param: ['0', '5'] cmp_type: D cmp_param: ['matmul', 'f32', 'matmul_transpose_b']
arg1 shape: [2, 5] dtype: f32 fill_type: I fill_param: ['-3', '3'] cmp_type: D cmp_param: ['matmul', 'f32', 'matmul_transpose_b']
arg2 shape: [2, 2] dtype: f32 fill_type: Z fill_param: [] cmp_type: P cmp_param: ['0', '100']
module {
  func.func @entry(%arg0: tensor<2x5xf32>, %arg1: tensor<2x5xf32>) -> tensor<2x2xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<2x5xf32>, tensor<2x5xf32>) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
  }
}

fill arg0: 
tensor([[  7.7050,  -1.4671, -10.8939,   2.8422,  -5.4226],
        [ -6.9930,   2.0167,   4.1901,  -3.5963,  -2.0167]])
fill arg1: 
tensor([[-3.,  0.,  1.,  0.,  0.],
        [ 3., -3.,  2., -3.,  0.]])
fill arg2: 
tensor([[0., 0.],
        [0., 0.]])
p2p check: threshold: 0.0000010
              (0, 0): ref:    7.7049804 res:    7.7049804 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 1): ref:   -1.4671445 res:   -1.4671445 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 2): ref:  -10.8939466 res:  -10.8939466 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 3): ref:    2.8421564 res:    2.8421564 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 4): ref:   -5.4226117 res:   -5.4226117 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 0): ref:   -6.9929771 res:   -6.9929771 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 1): ref:    2.0167341 res:    2.0167341 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 2): ref:    4.1901317 res:    4.1901317 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 3): ref:   -3.5962880 res:   -3.5962880 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 4): ref:   -2.0167177 res:   -2.0167177 abs_diff:    0.0000000 rel_diff:    0.0000000
p2p check: threshold: 0.0000010
              (0, 0): ref:   -3.0000000 res:   -3.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 1): ref:    0.0000000 res:    0.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 2): ref:    1.0000000 res:    1.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 3): ref:    0.0000000 res:    0.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 4): ref:    0.0000000 res:    0.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 0): ref:    3.0000000 res:    3.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 1): ref:   -3.0000000 res:   -3.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 2): ref:    2.0000000 res:    2.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 3): ref:   -3.0000000 res:   -3.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 4): ref:    0.0000000 res:    0.0000000 abs_diff:    0.0000000 rel_diff:    0.0000000
p2p check: threshold: 0.0000000
              (0, 0): ref:  -34.0088882 res:  -34.0088882 abs_diff:    0.0000000 rel_diff:    0.0000000
              (0, 1): ref:   -2.7979884 res:   -2.7979879 abs_diff:    0.0000005 rel_diff:    0.0000002
              (1, 0): ref:   25.1690636 res:   25.1690636 abs_diff:    0.0000000 rel_diff:    0.0000000
              (1, 1): ref:   -7.8600063 res:   -7.8600044 abs_diff:    0.0000019 rel_diff:    0.0000002
FAIL: linalg.matmul_transpose_b
```

### Perf testing example
* single op example
```
python3 -m benchgc --verbose 1  --mode P  --driver linalg --case add --md 0:4x5xf32 --md 1:4x5xf32 --md 2:4x5xf32

module {
  func.func @entry(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %2 = linalg.add ins(%arg0, %arg1 : tensor<4x5xf32>, tensor<4x5xf32>) outs(%1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    return %2 : tensor<4x5xf32>
  }
}

===========bench result===========
{
    "args": {
        "mode": "P",
        "driver": "linalg",
        "case": "add",
        "md": [
            "0:4x5xf32",
            "1:4x5xf32",
            "2:4x5xf32"
        ],
        "fill": [],
        "cmp": [],
        "seed": 0,
        "verbose": 1,
        "entry": "entry",
        "ir_printing": false,
        "cast": "cast_signed",
        "dimension": null,
        "dimensions": null,
        "dilations": null,
        "strides": null,
        "bench_kind": "py",
        "warm_up": 100,
        "repeat": 100
    },
    "compile_cost(ms)": 37.72595152258873,
    "execute_cost(ms)": 0.00022314488887786865
}
```

* mlir example
```
python3 -m benchgc --mode P --verbose 1  --driver mlir --case=./test.mlir  --bench_kind wrapper --warm_up 50 --repeat 200 
\module {
  func.func @entry(%arg0: tensor<512x128xf32>) -> tensor<512x128xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<512x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x128xf32>) -> tensor<512x128xf32>
    %2 = linalg.abs ins(%arg0 : tensor<512x128xf32>) outs(%1 : tensor<512x128xf32>) -> tensor<512x128xf32>
    return %2 : tensor<512x128xf32>
  }
}

===========bench result===========
{
    "args": {
        "mode": "P",
        "driver": "mlir",
        "case": "/home/xurui/gc_v2/test.mlir",
        "md": [],
        "fill": [],
        "cmp": [],
        "seed": 0,
        "verbose": 1,
        "entry": "entry",
        "ir_printing": false,
        "bench_kind": "wrapper",
        "warm_up": 50,
        "repeat": 200
    },
    "compile_cost(ms)": 70.6995539367199,
    "execute_cost(ms)": 0.029325044999999984
}
```
* mlp example
```
python3 -m benchgc --verbose 1 --mode P --driver pattern --case mlp --batch_size=32 --hidden_size_list=32x16x64 --has_bias=0x0 --act_type=noop --dtype=f32 

module {
  func.func @entry(%arg0: tensor<32x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<16x64xf32>) -> tensor<32x64xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<32x16xf32>
    %1 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x16xf32>) outs(%0 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %arg2 : tensor<32x16xf32>, tensor<16x64xf32>) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %3 : tensor<32x64xf32>
  }
}

===========bench result===========
{
    "args": {
        "mode": "P",
        "driver": "pattern",
        "case": "mlp",
        "md": [],
        "fill": [],
        "cmp": [],
        "seed": 0,
        "verbose": 1,
        "entry": "entry",
        "ir_printing": false,
        "bench_kind": "py",
        "warm_up": 100,
        "repeat": 100,
        "batch_size": 32,
        "hidden_size_list": "32x16x64",
        "has_bias": "0x0",
        "act_type": "noop",
        "dtype": "f32"
    },
    "compile_cost(ms)": 109.86808314919472,
    "execute_cost(ms)": 0.02944003790616989
}

```