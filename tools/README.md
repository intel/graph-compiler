# Python Tools
## Pre-requisites
### Enable python binding
* Enable MLIR python binding
* Install `/tools/requirements.txt`
### Set env
* **PYTHONPATH**=${BUILD_DIR}/python_packages/gc_mlir_core
* **LD_PRELOAD**=path/to/libiomp5.so
* **MLIR_C_RUNNER_UTILS**=${LLVM_INSTALL_DIR}/lib/libmlir_c_runner_utils.so
* **MLIR_RUNNER_UTILS**=${LLVM_INSTALL_DIR}/lib/libmlir_runner_utils.so


## Bench
### Examples:
```
# simple version
python3 ./tools/main.py --driver=load_mlir --path=./tools/workloads/test.mlir

# complex version
python3 ./tools/main.py --type=bench --bench_alg=py --driver=load_mlir --path=./tools/workloads/test.mlir --warm_up=200 --repeat=200 --print_ir --entry=main_entry
```

```
# result example
===========bench result===========
{
    "args": {
        "type": "bench",
        "driver": "load_mlir",
        "path": "./tools/workloads/test.mlir",
        "entry": "main_entry",
        "bench_alg": "py",
        "print_ir": false,
        "warm_up": 20,
        "repeat": 100
    },
    "compile_cost(ms)": 25.58841183781624,
    "execute_cost(ms)": 1.7501823976635933
}
```

### Common Options
*  `--driver`: the pattern to bench, currently support `mlp` and `load_mlir`
*  `--bench_alg`: `py` or `wrapper`, different evaluation implementation of the benchmark
*  `--warm_up`: warm-up times of the execution
*  `--repeat`: repeat times of the execution
*  `--print_ir`: print the ir before execution
*  `--disable_results_to_params`: please do not add this when using the default pipeline (gc-cpu-pipeline)

### Driver Specific Options
* load_mlir
  * `--path`: the mlir file path
  * `--entry`: the name of entry func
```
python3 ./tools/main.py --driver=load_mlir --path=./tools/workloads/test.mlir
```


* mlp  
  * `--batch_size`: the input
  * `--hidden_size_list`: hidden_sizes of mlp, example: 32x16x64
  * `--has_bias`: if the matmul op has bias, example: 1x0
  * `--act_type`: choices=["noop", "relu", "sigmoid"]
  * `--dtype`: choices=["bf16", "f32"]
```
python3 ./tools/main.py --driver=mlp --batch_size=32 --hidden_size_list=32x16x64 --has_bias=0x0 --act_type=noop --dtype=f32

===========bench func name:  main_entry ===========
module {
  func.func @main_entry(%arg0: tensor<32x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<16x64xf32>) -> tensor<32x64xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<32x16xf32>
    %1 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x16xf32>) outs(%0 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %arg2 : tensor<32x16xf32>, tensor<16x64xf32>) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %3 : tensor<32x64xf32>
  }
}
```

## Tuning
The logic of tuner is consistent with that of graph compiler v1 version, which can generate different config based on user set candidates and constraints
### Examples
```
python3 ./tools/main.py  --driver=load_mlir --type=tune --path=./tools/workloads/test.mlir --bench_alg=py  --search_alg=grid
```

### Options