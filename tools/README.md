# Python Tools
## Pre-requisites
### Enable python binding
* Enable MLIR python binding, [README](https://github.com/intel/graph-compiler/blob/main/python/README.md)
* Install `/tools/requirements.txt`
### Set env
* **PYTHONPATH**=*${BUILD_DIR}*/python_packages/gc_mlir_core
* **LD_PRELOAD**=path/to/libiomp5.so
* **MLIR_C_RUNNER_UTILS**=*${LLVM_INSTALL_DIR}*/lib/libmlir_c_runner_utils.so
* **MLIR_RUNNER_UTILS**=*${LLVM_INSTALL_DIR}*/lib/libmlir_runner_utils.so


## Bench
The tool has two different ways to calculate the time cost, and more experiments are needed to test which one is more stable and accurate. Currently, users can choose which way to use through options
* Use the MLIR Python API to invoke the kernel and use Python to calculate the time cost
* Modify MLIR by wrapping the kernel into a new method and calling the `nanoTime()` method before and after calling the kernel. Finally, calculate the difference as the time cost
```
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func public @wrapped_main(%arg0: memref<1xi64>, %arg1: tensor<128x512xbf16>, %arg2: tensor<512x256xbf16>) -> tensor<128x256xbf16> attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = call @main_entry(%arg1, %arg2) : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    %c0 = arith.constant 0 : index
    memref.store %3, %arg0[%c0] : memref<1xi64>
    return %1 : tensor<128x256xbf16>
  }
}
```

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
*  `--disable_results_to_params`: do not use this when using the default pipeline (gc-cpu-pipeline)

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
python3 /home/xurui/gc_v2/graph-compiler/tools/main.py  --driver=load_mlir --type=tune --path=./tools/workloads/test.mlir --bench_alg=wrapper  --search_alg=grid --batch_size=50
```

```
# result
[ 50 / 125 ] skipped: 0 best: 0.8382204799999994 ms
[ 100 / 125 ] skipped: 0 best: 0.83801604 ms
[ 125 / 125 ] skipped: 0 best: 0.83801604 ms
Tuning ends in 28.885201930999756 s
Best cost: 0.83801604 ms
Best config: [{
    "MatMulConfig": {
        "M_threads": 1,
        "K_threads": 1,
        "N_threads": 1,
        "M_block": 128,
        "K_block": 128,
        "N_block": 32,
        "innermostM_block": 1,
        "innermostK_block": 1,
        "innermostN_block": 1
    }
}]
mlir:
 module {
  func.func @main_entry(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x256xbf16>) -> tensor<128x256xbf16> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x256xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %2 = linalg.matmul {KBlock = 128 : i32, Kthreads = 1 : i32, MBlock = 128 : i32, Mthreads = 1 : i32, NBlock = 32 : i32, Nthreads = 1 : i32, innermostKBlock = 1 : i32, innermostMBlock = 1 : i32, innermostNBlock = 1 : i32} ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x256xbf16>) outs(%1 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    return %2 : tensor<128x256xbf16>
  }
}
```

### Options
The tuner will share the option of the bench tool and has the following unique tuner options
* `--tuning_batch`: The tuner will execute after generating a batch number of config, default: 50
* `--timeout`: The time limit setting for the tuner, default: -1
* `--max_tuning_iters`: The maximum number of iterations for tuning can also be considered as the maximum number of configs executed, default: sys.maxsize
* `--space_percent`：The percentage of tuning space is used, default: 1.0
* `--checkpoint_path`: The path to save the tuner status so that you can continue the tuning next time
* `--search_alg`: The algorithm for searching configs, choices=["grid", "ga"]
* `--early_stop`: If the tuner does not find a better config after iterating the early stop value, it will stop, default: -1, never stop

- When using the `ga`
  * `--random_seed`: Seed value for random function in GA tuner
  * `--elite_num`: the concept of elite number in genetic algorithm, default: 9
  * `--mutation_prob`：concept of mutation rate in genetic algorithm, default: 0.1
  * `--expected_tune_num`: The approximate quantity of generated config, If the value is 0, the ga tuner will use HashSet to record the tuned configs, otherwise use bloom filter, default: 0
    