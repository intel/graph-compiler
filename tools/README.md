# Python Tools
## Pre-requisites
### Enable python binding
* Install `/tools/requirements.txt`
### Set env
* PYTHONPATH=${BUILD_DIR}/python_packages/gc_mlir_core
* LD_PRELOAD=path/to/libiomp5.so
* MLIR_C_RUNNER_UTILS=${LLVM_INSTALL_DIR}/lib/libmlir_c_runner_utils.so
* MLIR_RUNNER_UTILS=${LLVM_INSTALL_DIR}/lib/libmlir_runner_utils.so


## Bench
### Examples:
```
# simple version
python3 ./tools/main.py --driver=load_mlir --path=./tools/workloads/test.mlir

# complex version
python3 ./tools/main.py --type=bench --bench_alg=py --driver=load_mlir --path=./tools/workloads/test.mlir --warm_up=200 --repeat=200 --print_ir
```

### Common Options
*  `--driver`: the pattern to bench, currently support `mlp` and `load_mlir`
*  `--bench_alg`: `py` or `wrapper`, different evaluation implementation of benchmark
*  `--warm_up`: warm up times of the execution
*  `--repeat`: repeat times of the execution
*  `--print_ir`: print the ir before execution
*  `--disable_results_to_params`: please do not add this when using with default pipeline (gc-cpu-pipeline)

### Driver Specific Options
* load_mlir
** `--path`:

* mlp
** `--batch_size`



## Tuning
### Examples
### Options