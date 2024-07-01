# Python Tools
## Pre-requisites
* Enable python binding
* Install `/tools/requirements.txt`
* Set env
** PYTHONPATH=${BUILD_DIR}/python_packages/gc_mlir_core
** LD_PRELOAD=path/to/libiomp5.so
** MLIR_C_RUNNER_UTILS=${LLVM_INSTALL_DIR}/lib/libmlir_c_runner_utils.so
** MLIR_RUNNER_UTILS=${LLVM_INSTALL_DIR}/lib/libmlir_runner_utils.so


##Bench
##Tuning
TODO