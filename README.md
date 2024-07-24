[<img src="https://scan.coverity.com/projects/30281/badge.svg">](https://scan.coverity.com/projects/intel-graph-compiler)

# Graph Compiler
Graph Compiler is an end-to-end, MLIR-based compiler designed to enhance the performance of deep learning workloads. It accepts computation graphs from the frontend, applies domain-specific optimizations and transformations, generates code, and manages runtime execution.

The current frontend for Graph Compiler is [oneDNN Graph API](https://oneapi-src.github.io/oneDNN/graph_extension.html).

## Build instructions

### All-in-one compile script

It is recommended for the users to use the all-in-one compile script at `scripts/compile.sh`. It downloads the LLVM dependency and builds the project.

### Step-by-step build intructions

To build this project step by step, first you need to find the LLVM commit-id we are using at `cmake/llvm-version.txt`. Then clone specific version of LLVM:

```bash
export LLVM_COMMIT_HASH=$(< cmake/llvm-version.txt)
git clone https://github.com/llvm/llvm-project
cd llvm-project
git checkout $LLVM_COMMIT_HASH
```

Build LLVM with the command lines given in `.github/workflows/build-llvm.yml`:

```bash
mkdir llvm-install
cmake -G Ninja llvm -B build -DCMAKE_INSTALL_PREFIX=llvm-install \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_INSTALL_UTILS=true -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_INSTALL_GTEST=ON
cmake --build build --target install
```

Notes
 * It is recommended to add optional options `-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON` to the command `cmake -G Ninja llvm ...` above **if you are building for CPU only**. These will enable the build of LLVM/MLIR dynamic libraries and let MLIR/LLVM tools link to them, to reduce the installed binary size of LLVM/MLIR. These options also enable the `GC_DEV_LINK_LLVM_DYLIB` option of graph-compiler repo (see below).
 * The option `-DLLVM_INSTALL_GTEST=ON` is optional, if the tests of graph-compiler are disabled (see `GC_ENABLE_TEST` below).
 * If you would like to enable GPU components of Graph Compiler, please make sure to statically link Graph Compiler and LLVM(MLIR). It is a known issue that LLVM shared library cannot be linked together with IGC (Intel's low level GPU compiler). Make sure `LLVM_BUILD_LLVM_DYLIB` and `LLVM_LINK_LLVM_DYLIB` are `OFF` (they are off by default). Also make sure Graph Compiler's cmake option `GC_DEV_LINK_LLVM_DYLIB` is `OFF` when configuring Graph Compiler (see below).

We have now installed LLVM at `llvm-project/llvm-install`.

Change working directory to graph-compiler repo and prepare the build directory:

```bash
cd /PATH/TO/graph-compiler
mkdir build && cd build
```

Build and run tests:

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_DIR=/PATH/TO/llvm-project/llvm-install/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build . --target gc-check
```

Notes:
 * `/PATH/TO/llvm-project/llvm-install` should be the install path of LLVM. If you installed LLVM elsewhere by `-DCMAKE_INSTALL_PREFIX` option when building LLVM, you need to change the path in `-DMLIR_DIR` accordingly.
 *  The cmake option `-DLLVM_EXTERNAL_LIT` is for the tests of this project. It requires the `lit` tool to be installed in the system. You can install it via `pip install lit`. If you don't need to run the tests of this repo, you can omit this option in the command line.

More notes if GPU components are on (`-DGC_ENABLE_GPU=ON`):
 * make sure the OpenCL runtime is installed in your system. You can either
  install using OS-provided package (Ubuntu 22.04)
```sh
sudo apt install -y intel-opencl-icd opencl-c-headers
```
  Or, download and install package from: https://github.com/intel/compute-runtime/releases
 * the LLVM codebase needs to be patched to support XeGPU lowering (from IMEX). Please follow instructions of [IMEX](https://github.com/intel/mlir-extensions) on patching LLVM.

Graph Compiler supports the following build-time options.

| CMake Option               | Supported values (defaults in bold)    | Description                                                     |
|:---------------------------|:---------------------------------------|:----------------------------------------------------------------|
| GC_ENABLE_LEGACY           | **ON**, OFF                            | Controls building the legacy graph-compiler component           |
| GC_ENABLE_TEST             | **ON**, OFF                            | Controls building the tests                                     |
| GC_DEV_LINK_LLVM_DYLIB     | ON, **OFF**                            | Controls dynamic link LLVM/MLIR libraries, mainly for developer |
| GC_ENABLE_BINDINGS_PYTHON  | **ON**, OFF                            | Controls building the Python API                                |
| GC_ENABLE_GPU              | ON, **OFF**                            | Whether to enable the GPU components                            |

