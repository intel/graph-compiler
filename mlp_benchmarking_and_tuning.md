# MLP Benchmarking and Tuning
## Environment Setup

### Prerequisite

- Python >= 3.10
- GCC >= 11
- jemalloc: See [install-jemalloc](https://github.com/intel-innersource/frameworks.ai.models.intel-models/blob/c7f8f4a93ea426b84e264a59a6bf1d31c4c50889/docs/general/pytorch/BareMetalSetup.md#install-jemalloc)
  
### Setup

Build and install LLVM, GC, benchgc:

```
cd $WORKDIR
git clone https://github.com/intel/graph-compiler.git
cd graph-compiler
git checkout origin/yifei/mlp_benching_new
bash scripts/compile.sh --dev
python -m pip install --force-reinstall build/test/benchgc/dist/benchgc-*.whl
```

### MLP Benchmarking and Tuning

For simple MLP, we can use the following sample command to perform benchmarking or tuning:

```
# benchmarking
python -m benchgc --mode=P --driver=pattern --case mlp --batch_size=128 --hidden_size_list=512x1024x1024x512x256 --has_bias=1x1x1x1 --act_type=relu --warm_up 200 --repeat 500

# tuning
python -m benchgc --mode=T --driver=pattern --case mlp --batch_size=128 --hidden_size_list=512x1024x1024x512x256 --has_bias=1x1x1x1 --act_type=relu --warm_up 200 --repeat 500
```

Current sample MLP workloads can be benchmarked or tuned with the following script `bench_mlp.sh`.

e.g.

```
export PATH_TO_JEMALLOC=$WORKDIR/jemalloc/install/bin/jemalloc.so
export NUM_THREADS=56
bash bench_mlp.sh --bench
```

As for more details of benchgc, see [benchgc README](https://github.com/xurui1995/graph-compiler/blob/xurui/benchgc_tuner/test/benchgc/README.md) for reference.

### Pruning tuning space

The tuning space could be large. We can prune tuning space by updating `init_constraints` function in `test/benchgc/src/benchgc/tuner/op_config.py`. 

### Appendix

If any stage in `compile.sh` script failed, please consider build LLVM/GC separately to analyze the detailed issue.

#### Build LLVM

```
cd $WORKDIR
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
git checkout $WORKDIR/graph-compiler/cmake/llvm-version.txt
python -m pip install -r mlir/python/requirements.txt
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON -DCMAKE_INSTALL_PREFIX=${PATH_TO_LLVM_INSTALL_DIR} -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=${PATH_TO_PYTHON_EXECUTABLE} -DLLVM_INSTALL_GTEST=ON
ninja install -j
```

#### Build GC

```
cd $WORKDIR/graph-compiler
mkdir build
cd build
cmake  .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR=${PATH_TO_INSTALLED_MLIR} -DLLVM_EXTERNAL_LIT=${PATH_TO_LLVM_LIT} -DGC_ENABLE_BINDINGS_PYTHON=ON -DGC_BENCH_ENABLE=ON
cmake --build . --target benchgc
python -m pip install test/benchgc/dist/benchgc-*.whl
```

Note that usually `MLIR_DIR` should be `${CMAKE_INSTALL_PREFIX}/lib/cmake/mlir` and `LLVM_EXTERNAL_LIT` should be `${PATH_TO_LLVM_PROJECT}/build/bin/llvm-lit`.
